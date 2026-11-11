"""Train matched green detail models and compare them with frozen v3 OOF."""

from __future__ import annotations

from .resilient_io import install

install()
from .runtime import (capture_rng, now, restore_rng, save_checkpoint, seed_everything,
                      source_digest, write_json)

import argparse
import json
import math
import os
from pathlib import Path

from filelock import FileLock
import numpy as np
from PIL import Image
import torch
import torchvision

from tools.prepare_submission import resize_probabilities, sha256
from vessel.data import get_fold, verify_dataset
from vessel.metrics import binary_metrics, segmentation_loss
from .data import load_samples, photometric, sample_patches
from .directional import interior, narrow_pixels, neighbors
from .directional_pilot import frozen_oof, score
from .export_study import targets_for
from .green_detail import GreenDetailUNet
from .models import optimizer_groups
from .train import Events, learning_rate, validate


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def train_fold(root, output, mode, fold, config, plan, provenance, device):
    train = config["train"]
    directory = output / mode / f"fold_{fold['fold']}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "last.pt"
    seed_everything(config["seed"] + fold["fold"], train["deterministic"])
    training = load_samples(root, fold["train_ids"], config["data"])
    held_out = load_samples(root, fold["validation_ids"], config["data"])
    model = GreenDetailUNet(config["model"], root, initialize=True, mode=mode)
    model = model.to(device, memory_format=torch.channels_last).train()
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, train["encoder_lr_multiplier"]),
        lr=train["lr"], weight_decay=train["weight_decay"],
    )
    amp = train["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    target = plan["stop_step"]
    step = 0
    skipped = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (saved["mode"], saved["fold"], saved["provenance"]) != (mode, fold, provenance):
            raise ValueError(f"Checkpoint does not match the source, config, data, and fold: {checkpoint_path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        step, skipped = saved["step"], saved["skipped"]
    checkpoint_step = step
    status = dict(state="training", pid=os.getpid(), mode=mode, fold=fold["fold"], target_steps=target)

    def publish(state, **values):
        status.update(state=state, step=step, updated_at=now(), **values)
        write_json(output / "status.json", status)
        write_json(directory / "status.json", status)

    def save():
        save_checkpoint(checkpoint_path, {
            "mode": mode, "fold": fold, "provenance": provenance, "step": step, "skipped": skipped,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rng": capture_rng(),
        })

    events = Events(directory / "train.jsonl")
    try:
        events.emit("started", mode=mode, fold=fold["fold"], step=step, target_steps=target)
        publish("training")
        while step < target and not (output / "STOP").exists():
            images, masks = sample_patches(training, train["batch_size"],
                                           config["data"]["patch_hw"], config["augmentation"])
            images = photometric(images.to(device, memory_format=torch.channels_last), config["augmentation"])
            masks = masks.to(device)
            lr = learning_rate(train, step + 1)
            for group in optimizer.param_groups:
                group["lr"] = lr * group["lr_multiplier"]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = segmentation_loss(model(images), masks, config["loss"])
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip_norm"])
            skipped = 0 if math.isfinite(float(grad_norm)) else skipped + 1
            if skipped and (not amp or skipped >= 10):
                raise FloatingPointError(f"Non-finite gradients at step {step + 1}")
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % plan["checkpoint_every"] == 0 or step == target:
                save()
                checkpoint_step = step
            if step == 1 or step % 100 == 0 or step == target:
                events.emit("train", mode=mode, fold=fold["fold"], step=step, loss=loss_value, lr=lr)
                publish("training", last_loss=loss_value, checkpoint_step=checkpoint_step)
        if step != target:
            if step > checkpoint_step:
                save()
            publish("paused", checkpoint_step=step)
            events.emit("paused", mode=mode, fold=fold["fold"], step=step)
            return None
        metrics, probabilities = validate(model, held_out, device, amp)
        publish("fold_completed", checkpoint_step=step, dice_at_0_5=metrics["mean_dice_at_0_5"])
        events.emit("validation", mode=mode, fold=fold["fold"], step=step,
                    dice_at_0_5=metrics["mean_dice_at_0_5"])
        return np.stack(probabilities)
    except Exception as error:
        publish("failed", error=str(error), checkpoint_step=checkpoint_step)
        raise
    finally:
        events.close()


def compare(root, plan, folds, predictions, recipe, ids):
    targets = np.stack(targets_for(root, ids))
    baseline, _, _ = frozen_oof(root, recipe, ids, targets)
    held_ids = [i for fold in folds for i in fold["validation_ids"]]
    positions = [ids.index(i) for i in held_ids]
    targets, baseline = targets[positions], baseline[positions]
    y = torch.from_numpy(targets[:, None])
    narrow = narrow_pixels(y.float()).bool()
    narrow_edges = interior(y) & neighbors(y) & (interior(narrow) | neighbors(narrow))
    target_512 = [np.asarray(Image.fromarray(mask).resize((512, 512), Image.Resampling.NEAREST))
                  for mask in targets]
    probabilities = {"v3": baseline, **predictions}
    weight = plan["candidate_weight"]
    probabilities.update({f"v3_plus_{mode}": (1 - weight) * baseline + weight * prediction
                          for mode, prediction in predictions.items()})
    scores = {}
    for name, values in probabilities.items():
        threshold = recipe["threshold"] if name.startswith("v3") else 0.5
        metrics = score(values, threshold, targets, narrow[:, 0].numpy(), narrow_edges, held_ids)
        metrics["mean_dice_512_proxy"] = float(np.mean([
            binary_metrics(p >= threshold, t)["dice"]
            for p, t in zip(resize_probabilities(values, (512, 512)), target_512)
        ]))
        scores[name] = metrics
    candidate = scores["v3_plus_local_contrast"]
    reference = scores["v3"]
    matched = scores["v3_plus_raw_green"]
    delta_by_id = {a["id"]: a["dice"] - b["dice"]
                   for a, b in zip(candidate["per_image"], reference["per_image"])}
    by_fold = {str(fold["fold"]): float(np.mean([delta_by_id[i] for i in fold["validation_ids"]]))
               for fold in folds}
    gain = candidate["mean_dice"] - reference["mean_dice"]
    over_matched = candidate["mean_dice"] - matched["mean_dice"]
    promoted = (
        gain >= plan["minimum_gain_over_v3"] and over_matched > 0
        and all(value >= 0 for value in by_fold.values())
        and candidate["narrow_recall"] > reference["narrow_recall"]
        and candidate["mean_dice_512_proxy"] >= reference["mean_dice_512_proxy"]
    )
    return held_ids, dict(scores=scores, gain_over_v3=gain, gain_over_raw_green_fusion=over_matched,
                          gain_over_v3_by_fold=by_fold, promote_to_full_study=promoted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/green_detail_v5.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/green_detail_v5"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = root / args.plan
    plan = read(plan_path)
    config_path = root / plan["backbone_config"]
    config = read(config_path)
    if not 1 <= plan["stop_step"] <= config["train"]["steps"] or plan["checkpoint_every"] < 1:
        raise ValueError("Checkpoints and stop step must fall within the training schedule")
    if not 0 < plan["candidate_weight"] < 1 or plan["minimum_gain_over_v3"] <= 0:
        raise ValueError("Expected positive gain and a mixture weight between zero and one")
    if sorted(plan["variants"]) != ["local_contrast", "raw_green"]:
        raise ValueError("The matched study requires one run of each green detail variant per fold")
    torch.set_num_threads(config["train"]["cpu_threads"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data_hash = verify_dataset(root)
    folds = [get_fold(root, index, data_hash) for index in plan["folds"]]
    held_ids = [i for fold in folds for i in fold["validation_ids"]]
    if not held_ids or len(held_ids) != len(set(held_ids)):
        raise ValueError("Need nonempty, disjoint held-out folds")
    recipe_path = root / plan["baseline_recipe"]
    recipe = read(recipe_path)
    order_path = root / "optimization_v3/per_image_selected.json"
    ids = [row["id"] for row in read(order_path)]
    fixed_inputs = [config_path, recipe_path, order_path, root / "experiments/folds_v1.json"]
    fixed_inputs += [root / f"optimization_v3/{member['name']}_oof_selected.npz" for member in recipe["members"]]
    fixed_inputs += [Path(__file__).with_name(name) for name in
                     ("green_detail_study.py", "green_detail.py", "directional_pilot.py", "directional.py",
                      "resilient_io.py", "export_study.py")]
    provenance = {
        "plan": plan, "config": config, "folds": folds, "data_sha256": data_hash,
        "base_source_sha256": source_digest(), "plan_sha256": sha256(plan_path),
        "inputs_sha256": {path.relative_to(root).as_posix(): sha256(path) for path in fixed_inputs},
        "torch": torch.__version__, "torchvision": torchvision.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    output = root / args.output
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    with FileLock(output / ".training.lock", timeout=0):
        provenance_path = output / "provenance.json"
        if args.resume:
            if read(provenance_path) != provenance:
                raise ValueError("Resume requires identical data, code, environment, and plan")
        else:
            write_json(provenance_path, provenance)
        collected = {mode: [] for mode in plan["variants"]}
        for fold in folds:
            for mode in plan["variants"]:
                if (output / "STOP").exists():
                    write_json(output / "status.json", dict(state="paused", updated_at=now()))
                    return
                values = train_fold(root, output, mode, fold, config, plan, provenance, device)
                if values is None:
                    return
                collected[mode].append(values)
        predictions = {mode: np.concatenate(values) for mode, values in collected.items()}
        ids, result = compare(root, plan, folds, predictions, recipe, ids)
        np.savez_compressed(output / "pilot_oof.npz", ids=np.asarray(ids), **predictions)
        write_json(output / "pilot_results.json", dict(completed_at=now(), **result))
        write_json(output / "status.json", dict(state="pilot_completed", updated_at=now(),
                                               promote_to_full_study=result["promote_to_full_study"]))
        print(json.dumps({key: value for key, value in result.items() if key != "scores"}), flush=True)


if __name__ == "__main__":
    main()
