"""Compare directional supervision with a matched ResNet18 U-Net control."""

from __future__ import annotations

from .runtime import now, seed_everything, source_digest, write_json

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from tools.prepare_submission import resize_probabilities, sha256
from vessel.data import get_fold, verify_dataset
from vessel.metrics import binary_metrics, segmentation_loss
from .data import load_samples, photometric, sample_patches
from .directional import DirectionalHead, directional_loss, interior, narrow_pixels, neighbors
from .export_study import targets_for
from .models import build_model, optimizer_groups
from .train import learning_rate, validate


def fit(root, config, fold, device, stop_step, direction_weight):
    train = config["train"]
    seed_everything(config["seed"] + fold["fold"], train["deterministic"])
    training = load_samples(root, fold["train_ids"], config["data"])
    held_out = load_samples(root, fold["validation_ids"], config["data"])
    model = build_model(config["model"], root, initialize=True)
    if direction_weight:
        # Keep the baseline initialization and augmentation random streams aligned.
        with torch.random.fork_rng(devices=[]):
            model.head = DirectionalHead(model.head)
    model = model.to(device, memory_format=torch.channels_last).train()
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, train["encoder_lr_multiplier"]),
        lr=train["lr"], weight_decay=train["weight_decay"],
    )
    amp = train["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start = time.perf_counter()
    skipped = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, stop_step + 1):
        images, masks = sample_patches(training, train["batch_size"],
                                       config["data"]["patch_hw"], config["augmentation"])
        images = photometric(images.to(device, memory_format=torch.channels_last), config["augmentation"])
        masks = masks.to(device)
        lr = learning_rate(train, step)
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_multiplier"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = segmentation_loss(logits[:, :1], masks, config["loss"])
            if direction_weight:
                loss = loss + direction_weight * directional_loss(logits[:, 1:], masks)
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip_norm"])
        if math.isfinite(float(norm)):
            skipped = 0
        else:
            skipped += 1
            if not amp or skipped >= 10:
                raise FloatingPointError(f"Non-finite gradients at step {step}")
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 100 == 0 or step == stop_step:
            print(f"fold={fold['fold']} direction_weight={direction_weight} step={step}/{stop_step}", flush=True)
    _, probabilities = validate(model, held_out, device, amp)
    return np.stack(probabilities), {
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }


def frozen_oof(root, recipe, ids, targets):
    paths, components = [], []
    total = np.zeros(targets.shape, dtype=np.float32)
    for member in recipe["members"]:
        path = root / f"optimization_v3/{member['name']}_oof_selected.npz"
        with np.load(path, allow_pickle=False) as archive:
            probabilities = archive["probabilities"]
            if (archive["ids"].tolist() != ids or probabilities.shape != targets.shape
                    or not np.isfinite(probabilities).all()
                    or probabilities.min() < 0 or probabilities.max() > 1):
                raise ValueError(f"Invalid v3 OOF archive: {path}")
        components.append(probabilities)
        total += probabilities * member["weight"]
        paths.append(path)
    return total, components, paths


def score(probabilities, threshold, targets, narrow, narrow_edges, ids):
    prediction = probabilities >= threshold
    per_image = [dict(id=i, **binary_metrics(p, y)) for i, p, y in zip(ids, prediction, targets)]
    mask = torch.from_numpy(prediction[:, None])
    predicted_edges = interior(mask) & neighbors(mask)
    return {
        "threshold": threshold,
        "mean_dice": float(np.mean([row["dice"] for row in per_image])),
        "narrow_recall": float((prediction & narrow).sum() / narrow.sum()),
        "narrow_edge_recall": float((predicted_edges & narrow_edges).sum() / narrow_edges.sum()),
        "fp": sum(row["fp"] for row in per_image),
        "fn": sum(row["fn"] for row in per_image),
        "per_image": per_image,
    }


def compare(frozen, control, directional, targets, narrow, ids, threshold, weight):
    y = torch.from_numpy(targets[:, None])
    n = torch.from_numpy(narrow[:, None])
    narrow_edges = interior(y) & neighbors(y) & (interior(n) | neighbors(n))
    probabilities = {
        "v3": frozen,
        "control": control,
        "directional": directional,
        "v3_plus_control": (1 - weight) * frozen + weight * control,
        "v3_plus_directional": (1 - weight) * frozen + weight * directional,
    }
    target_512 = [np.asarray(Image.fromarray(mask).resize((512, 512), Image.Resampling.NEAREST))
                  for mask in targets]
    result = {}
    for name, values in probabilities.items():
        cutoff = threshold if name.startswith("v3") else 0.5
        metrics = score(values, cutoff, targets, narrow, narrow_edges, ids)
        metrics["mean_dice_512_proxy"] = float(np.mean([
            binary_metrics(p >= cutoff, y)["dice"]
            for p, y in zip(resize_probabilities(values, (512, 512)), target_512)
        ]))
        result[name] = metrics
    return result, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/directional_pilot_v4.json"))
    parser.add_argument("--output", type=Path, default=Path("optimization_v4"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = root / args.plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config_path = root / plan["backbone_config"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stop = plan["stop_step"]
    if not 1 <= stop <= config["train"]["steps"]:
        raise ValueError("Pilot stop_step must be within the original training schedule")
    if len(plan["folds"]) != len(set(plan["folds"])) or not plan["folds"]:
        raise ValueError("Pilot needs a nonempty list of distinct folds")
    if not 0 < plan["candidate_weight"] < 1 or not plan["direction_loss_weight"] > 0:
        raise ValueError("Pilot requires a positive direction loss and a valid mixture weight")
    torch.set_num_threads(config["train"]["cpu_threads"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python environment")
    data_hash = verify_dataset(root)
    fold_path = root / "experiments/folds_v1.json"
    ids = [row["id"] for row in json.loads((root / "optimization_v3/per_image_selected.json").read_text())]
    recipe_path = root / plan["baseline_recipe"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    threshold = recipe["threshold"]
    targets = np.stack(targets_for(root, ids))
    narrow = narrow_pixels(torch.from_numpy(targets[:, None].astype(np.float32)))[:, 0].numpy().astype(bool)
    frozen, components, archives = frozen_oof(root, recipe, ids, targets)
    frozen_mask = frozen >= threshold
    missed = targets & ~frozen_mask
    shared_missed = missed & np.logical_and.reduce([p < threshold for p in components])
    baseline = {
        "image_count": len(ids),
        "threshold": threshold,
        "mean_dice": float(np.mean([binary_metrics(p, y)["dice"] for p, y in zip(frozen_mask, targets)])),
        "foreground_pixels": int(targets.sum()),
        "narrow_pixels": int(narrow.sum()),
        "fp": int((frozen_mask & ~targets).sum()),
        "fn": int(missed.sum()),
        "narrow_fn": int((narrow & missed).sum()),
        "shared_fn_at_v3_threshold": int(shared_missed.sum()),
    }
    folds = [get_fold(root, index, data_hash) for index in plan["folds"]]
    held_out_ids = [i for fold in folds for i in fold["validation_ids"]]
    if len(set(held_out_ids)) != len(held_out_ids):
        raise ValueError("Repeated image in pilot validation folds")
    positions = [ids.index(i) for i in held_out_ids]
    output = root / args.output
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "baseline_diagnostics.json", baseline)
    outputs = {"control": [], "directional": []}
    training_runs = []
    for fold in folds:
        for name, weight in (("control", 0.0), ("directional", plan["direction_loss_weight"])):
            values, stats = fit(root, config, fold, device, stop, weight)
            outputs[name].append(values)
            training_runs.append(dict(variant=name, fold=fold["fold"], **stats))
    scores, probabilities = compare(
        frozen[positions], np.concatenate(outputs["control"]), np.concatenate(outputs["directional"]),
        targets[positions], narrow[positions], held_out_ids, threshold, plan["candidate_weight"],
    )
    candidate = scores["v3_plus_directional"]
    reference = scores["v3"]
    delta = np.array([a["dice"] - b["dice"] for a, b in zip(candidate["per_image"], reference["per_image"])])
    by_fold = {fold["fold"]: float(np.mean([delta[held_out_ids.index(i)] for i in fold["validation_ids"]]))
               for fold in folds}
    gain_over_v3 = candidate["mean_dice"] - reference["mean_dice"]
    gain_over_control = candidate["mean_dice"] - scores["v3_plus_control"]["mean_dice"]
    promoted = (
        gain_over_v3 >= plan["minimum_gain_over_v3"]
        and gain_over_control > 0 and all(value >= 0 for value in by_fold.values())
        and candidate["narrow_recall"] > reference["narrow_recall"]
        and candidate["mean_dice_512_proxy"] >= reference["mean_dice_512_proxy"]
    )
    recovered = shared_missed[positions] & (probabilities["v3_plus_directional"] >= threshold)
    result = {
        "completed_at": now(),
        "data_sha256": data_hash,
        "base_source_sha256": source_digest(),
        "inputs_sha256": {path.relative_to(root).as_posix(): sha256(path) for path in [
            plan_path, config_path, fold_path, recipe_path, *archives,
            Path(__file__), Path(__file__).with_name("directional.py"),
        ]},
        "runtime": {"torch": torch.__version__, "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "folds": folds,
        "training_runs": training_runs,
        "scores": scores,
        "gain_over_v3": gain_over_v3,
        "gain_over_matched_control_fusion": gain_over_control,
        "gain_over_v3_by_fold": by_fold,
        "shared_fn_recovered_by_candidate": int(recovered.sum()),
        "promote_to_full_study": promoted,
    }
    np.savez_compressed(output / "pilot_oof.npz", ids=np.asarray(held_out_ids),
                        control=probabilities["control"], directional=probabilities["directional"])
    write_json(output / "pilot_results.json", result)
    print(json.dumps(dict(gain_over_v3=gain_over_v3, gain_over_control=gain_over_control,
                          promote_to_full_study=promoted)), flush=True)


if __name__ == "__main__":
    main()
