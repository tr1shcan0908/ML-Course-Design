"""Train and evaluate a five-fold dual-domain high-resolution network.

Run: python -m vessel_lab.dual_domain_study
Resuming uses the same command and verifies code/config/data fingerprints.
"""

from __future__ import annotations

from .resilient_io import install
install()
from .runtime import capture_rng, restore_rng, now, save_checkpoint, seed_everything, write_json

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

from filelock import FileLock
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.nn import functional as F

from vessel.data import get_fold, verify_dataset
from vessel.metrics import binary_metrics
from .dual_domain import DualDomainNet, dual_loss
from .inference import transform, invert
from .submission_domain import dump, export_masks, script_mask, sha256
from .train import learning_rate, Events


def load_grid(root, ids, split="train"):
    images, targets, native_targets = [], [], []
    for image_id in ids:
        with Image.open(root / split / "image" / f"{image_id}.jpg") as source:
            rgb = np.asarray(source.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        green = rgb[:, :, 1]
        fov = rgb.max(axis=2) > .05
        normalized = np.clip(.5 + .15 * (green - green[fov].mean()) / max(float(green[fov].std()), .01), 0, 1)
        smooth = np.asarray(Image.fromarray(np.round(green * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(8)), dtype=np.float32) / 255
        local = np.clip(.5 + 3 * (green - smooth), 0, 1)
        normalized[~fov] = local[~fov] = 0
        images.append(np.concatenate((rgb, normalized[:, :, None], local[:, :, None]), axis=2).transpose(2, 0, 1))
        if split == "train":
            with Image.open(root / "train/label" / f"{image_id}.jpg") as source:
                label = np.asarray(source.convert("L")) > 0
            thin = np.asarray(Image.fromarray(label).resize((512, 512), Image.Resampling.NEAREST))
            thick = script_mask(label)
            if np.any(thin & ~thick):
                raise ValueError("The ordered native/submission target assumption is false")
            targets.append(np.stack((thin, thick)))
            native_targets.append(label)
    return (torch.from_numpy(np.stack(images).copy()).float(),
            torch.from_numpy(np.stack(targets)).float() if targets else None,
            native_targets)


def sample_batch(images, targets, positive, config):
    n = config["train"]["batch_size"]
    patch = config["data"]["patch"]
    aug = config["augmentation"]
    indices, matrices = [], []
    for _ in range(n):
        index = random.randrange(len(images))
        indices.append(index)
        if random.random() < aug["vessel_center_fraction"]:
            cy, cx = random.choice(positive[index])
        else:
            cy, cx = random.uniform(64, 448), random.uniform(64, 448)
        angle = math.radians(random.uniform(*aug["angle_degrees"]))
        scale = patch / 512 * random.uniform(*aug["scale"])
        flip = random.choice((-1, 1))
        a, b = math.cos(angle) * scale, math.sin(angle) * scale
        matrices.append([[a * flip, -b, 2 * (cx + .5) / 512 - 1],
                         [b * flip, a, 2 * (cy + .5) / 512 - 1]])
    theta = torch.tensor(matrices, dtype=torch.float32, device=images.device)
    grid = F.affine_grid(theta, (n, 5, patch, patch), align_corners=False)
    x = F.grid_sample(images[indices], grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    y = F.grid_sample(targets[indices], grid, mode="nearest", padding_mode="reflection", align_corners=False)
    rgb = x[:, :3]
    def factor(name):
        return torch.empty((n, 1, 1, 1), device=x.device).uniform_(*aug[name])
    mean = rgb.mean((1, 2, 3), keepdim=True)
    rgb = (((rgb - mean) * factor("contrast") + mean) * factor("brightness")).clamp(0, 1)
    rgb = rgb.pow(factor("gamma")) + torch.randn_like(rgb) * factor("noise_std")
    x[:, :3] = rgb.clamp(0, 1)
    return x.contiguous(memory_format=torch.channels_last), y


@torch.no_grad()
def update_ema(ema, model, decay):
    for averaged, current in zip(ema.parameters(), model.parameters()):
        averaged.lerp_(current, 1 - decay)
    for averaged, current in zip(ema.buffers(), model.buffers()):
        averaged.copy_(current)


@torch.inference_mode()
def predict(model, images, device, d4=False):
    model.eval()
    variants = [(False, ())]
    if d4:
        variants = [(transposed, flips) for transposed in (False, True) for flips in ((), (2,), (3,), (2, 3))]
    output = []
    for image in images:
        image = image[None].to(device, memory_format=torch.channels_last)
        total = None
        for transposed, flips in variants:
            mapped = transform(image, transposed, flips)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                probability = model(mapped).float().sigmoid()
            probability = invert(probability, transposed, flips)
            total = probability if total is None else total + probability
        output.append((total[0] / len(variants)).cpu().numpy())
    result = np.stack(output)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite validation predictions")
    return result


def metrics(probabilities, targets, ids):
    results = {}
    for channel, name in enumerate(("native_512", "submission_512")):
        rows = [dict(id=i, **binary_metrics(p[channel] >= .5, y[channel]))
                for i, p, y in zip(ids, probabilities, targets)]
        results[name] = dict(mean_dice=float(np.mean([r["dice"] for r in rows])), per_image=rows)
    return results


def fit_fold(root, run, config, fold, provenance, device):
    directory = run / f"fold_{fold['fold']}"
    directory.mkdir(exist_ok=True)
    seed_everything(config["seed"] + fold["fold"], True)
    x, y, _ = load_grid(root, fold["train_ids"])
    positive = [np.argwhere(t[0].numpy() > 0).tolist() for t in y]
    validation, validation_y, _ = load_grid(root, fold["validation_ids"])
    x, y = x.to(device), y.to(device)
    model = DualDomainNet(**config["model"]).to(device, memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    stage = config["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage["lr"], weight_decay=stage["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    step, best, best_step, elapsed_before = 0, -1., 0, 0.
    last_path = directory / "last.pt"
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance or saved["fold"] != fold:
            raise ValueError(f"Checkpoint lineage differs: {last_path}")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        step, best, best_step = saved["step"], saved["best"], saved["best_step"]
        elapsed_before = saved["elapsed_seconds"]
        del saved
    start = time.perf_counter()
    write_json(directory / "fold.json", fold)
    events = Events(directory / "train.jsonl")
    def status(state, **extra):
        data = dict(state=state, updated_at=now(), pid=os.getpid(), fold=fold["fold"], step=step,
                    target_steps=stage["steps"], best_submission_dice=best, best_step=best_step,
                    elapsed_seconds=elapsed_before + time.perf_counter() - start, **extra)
        write_json(directory / "status.json", data)
        write_json(run / "status.json", data)
    def save():
        save_checkpoint(last_path, dict(model=model.state_dict(), ema=ema.state_dict(),
                                       optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=capture_rng(),
                                       provenance=provenance, fold=fold, step=step, best=best, best_step=best_step,
                                       elapsed_seconds=elapsed_before + time.perf_counter() - start))
    try:
        events.emit("resumed" if step else "started", fold=fold["fold"], step=step,
                    target_steps=stage["steps"], parameters=sum(p.numel() for p in model.parameters()),
                    training_ids=fold["train_ids"], validation_ids=fold["validation_ids"])
        model.train()
        skipped = 0
        losses = []
        while step < stage["steps"]:
            if (run / "STOP").exists():
                save()
                status("paused")
                return False
            batch, target = sample_batch(x, y, positive, config)
            optimizer.param_groups[0]["lr"] = learning_rate(stage, step + 1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = dual_loss(model(batch, auxiliary=True), target)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            finite = math.isfinite(float(norm))
            skipped = 0 if finite else skipped + 1
            if skipped >= 10:
                raise FloatingPointError("Ten consecutive non-finite gradients")
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if finite:
                update_ema(ema, model, min(stage["ema_decay"], (1 + step) / (10 + step)))
            losses.append(value)
            if step == 1 or step % 100 == 0:
                status("training")
                events.emit("train", fold=fold["fold"], step=step, loss=float(np.mean(losses)),
                            elapsed_seconds=time.perf_counter() - start,
                            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
                losses.clear()
            if step % stage["validate_every"] == 0 or step == stage["steps"]:
                p = predict(ema, validation, device)
                result = metrics(p, validation_y.numpy(), fold["validation_ids"])
                score = result["submission_512"]["mean_dice"]
                if score > best:
                    best, best_step = score, step
                    save_checkpoint(directory / "best.pt", dict(model=ema.state_dict(), config=config, fold=fold,
                                    provenance=provenance, step=step, metrics=result, model_state="ema"))
                    write_json(directory / "best_metrics.json", dict(step=step, **result))
                save()
                events.emit("validation", fold=fold["fold"], step=step, best_step=best_step,
                            submission_dice=score, native_512_dice=result["native_512"]["mean_dice"])
        status("completed")
        return True
    finally:
        events.close()


def evaluate_all(root, run, config, folds, provenance, device):
    ids, values, targets, checkpoints = [], [], [], []
    test_ids = [str(i) for i in range(1, 21)]
    test_images, _, _ = load_grid(root, test_ids, "test")
    test_total = None
    for fold in folds:
        path = run / f"fold_{fold['fold']}/best.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["fold"] != fold or saved["provenance"] != provenance:
            raise ValueError("Best checkpoint provenance mismatch")
        model = DualDomainNet(**config["model"]).to(device, memory_format=torch.channels_last)
        model.load_state_dict(saved["model"])
        images, y, _ = load_grid(root, fold["validation_ids"])
        identity = predict(model, images, device)
        replay = metrics(identity, y.numpy(), fold["validation_ids"])
        if replay != saved["metrics"]:
            raise ValueError("Best checkpoint validation did not replay exactly")
        p = predict(model, images, device, d4=True)
        ids.extend(fold["validation_ids"])
        values.append(p)
        targets.append(y.numpy())
        predicted_test = predict(model, test_images, device, d4=True)
        test_total = predicted_test if test_total is None else test_total + predicted_test
        checkpoints.append(dict(fold=fold["fold"], path=str(path.relative_to(root)).replace("\\", "/"),
                                sha256=sha256(path), best_step=saved["step"]))
        write_json(run / "status.json", dict(state="predicting", fold=fold["fold"], updated_at=now(), pid=os.getpid()))
    p, y = np.concatenate(values), np.concatenate(targets)
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError("OOF validation must cover the twenty originals exactly once")
    results = metrics(p, y, ids)
    with np.load(root / "submission_v3/oof_predictions.npz", allow_pickle=False) as archive:
        positions = [archive["ids"].tolist().index(i) for i in ids]
        baseline = archive["probabilities"][positions]
    baseline_masks = np.stack([script_mask(a >= .54) for a in baseline])
    old = [dict(id=i, **binary_metrics(a, b[1])) for i, a, b in zip(ids, baseline_masks, y)]
    baseline_score = float(np.mean([row["dice"] for row in old]))
    results["v3_script_compatible"] = dict(mean_dice=baseline_score, per_image=old)
    paired = {a["id"]: a["dice"] - b["dice"] for a, b in zip(results["submission_512"]["per_image"], old)}
    gains = {str(f["fold"]): float(np.mean([paired[i] for i in f["validation_ids"]])) for f in folds}
    results["comparison"] = dict(gain_over_script_compatible=results["submission_512"]["mean_dice"] - baseline_score,
                                 gain_by_fold=gains, platform_retest_required=True,
                                 note="Checkpoint selection reused these folds; this is a local validation comparison.")
    np.savez_compressed(run / "oof_predictions.npz", ids=np.asarray(ids), probabilities=p)
    test_p = test_total / len(folds)
    np.savez_compressed(run / "test_predictions.npz", ids=np.asarray(test_ids), probabilities=test_p)
    output = root / "submission_v8_dual_domain"
    checks = export_masks(root, output, test_ids, test_p[:, 1] >= .5)
    (root / "submission_v8_dual_domain.csv").write_bytes((output / "submission.csv").read_bytes())
    write_json(output / "inference_recipe.json", dict(config=config, checkpoints=checkpoints,
                output_head="submission", threshold=.5, probability_average="five_folds_then_threshold", tta="d4",
                dataset_sha256=provenance["dataset_sha256"], csv_sha256=checks["csv_sha256"]))
    write_json(run / "results.json", dict(completed_at=now(), scores=results, checkpoints=checkpoints,
                                          source_sha256=provenance["source_sha256"]))
    write_json(run / "status.json", dict(state="completed", updated_at=now(), pid=os.getpid(), scores={
        k: v["mean_dice"] for k, v in results.items() if "mean_dice" in v}, comparison=results["comparison"]))
    print(json.dumps(results["comparison"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/dual_domain_v8.json")
    parser.add_argument("--output", default="runs/dual_domain_v8")
    parser.add_argument("--folds", help="Optional comma-separated fold subset for a pilot")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / args.output
    run.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    source_names = ["vessel_lab/dual_domain.py", "vessel_lab/dual_domain_study.py",
                    "vessel_lab/submission_domain.py", "vessel_lab/inference.py", "vessel_lab/train.py",
                    "vessel_lab/runtime.py", "vessel_lab/resilient_io.py", "vessel/data.py",
                    "vessel/metrics.py", "vessel/runtime.py", "vessel/rle.py", "tools/prepare_submission.py"]
    source = {name: sha256(root / name) for name in source_names}
    data_hash = verify_dataset(root)
    provenance = dict(config=config, source_sha256=source, dataset_sha256=data_hash,
                      fold_sha256=sha256(root / "experiments/folds_v1.json"),
                      torch=torch.__version__, gpu=torch.cuda.get_device_name(), python=sys.version,
                      baseline_oof_sha256=sha256(root / "submission_v3/oof_predictions.npz"))
    indices = [int(i) for i in args.folds.split(",")] if args.folds else config["folds"]
    if len(indices) != len(set(indices)) or any(i not in range(5) for i in indices):
        raise ValueError("Invalid fold selection")
    folds = [get_fold(root, i, data_hash) for i in indices]
    torch.set_num_threads(config["train"]["cpu_threads"])
    seed_everything(config["seed"], True)
    device = torch.device("cuda")
    with FileLock(run / ".training.lock", timeout=0):
        path = run / "provenance.json"
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != provenance:
            raise ValueError("Run code, configuration, data, or environment changed")
        write_json(path, provenance)
        for name in source_names + [args.config, "experiments/folds_v1.json", "experiments/dataset_manifest.json"]:
            target = run / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
        try:
            if not args.evaluate_only:
                for fold in folds:
                    if not fit_fold(root, run, config, fold, provenance, device):
                        return
            if len(folds) == 5:
                evaluate_all(root, run, config, folds, provenance, device)
            else:
                write_json(run / "status.json", dict(state="completed_selected_folds", folds=indices, updated_at=now()))
        except BaseException as error:
            write_json(run / "status.json", dict(state="failed", updated_at=now(), error=str(error), traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
