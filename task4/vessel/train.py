"""Train an image-disjoint fold, or all five folds sequentially."""

from __future__ import annotations

# Configure deterministic CUDA before importing/initializing torch.
from .runtime import (capture_rng, now, record_environment, restore_rng,
                      save_checkpoint, seed_everything, source_digest, write_json)

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .data import get_fold, load_samples, photometric, sample_patches, verify_dataset
from .metrics import evaluate_probabilities, segmentation_loss
from .model import VesselUNet


def learning_rate(config: dict, step: int) -> float:
    warmup = config["warmup_steps"]
    if step <= warmup:
        return config["lr"] * step / max(1, warmup)
    fraction = min(1.0, (step - warmup) / max(1, config["steps"] - warmup))
    return config["min_lr"] + 0.5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * fraction))


@torch.inference_mode()
def validate(model, samples, device, amp: bool) -> tuple[dict, list[np.ndarray]]:
    was_training = model.training
    model.eval()
    probabilities, targets, ids = [], [], []
    for sample in samples:
        image = sample.image[None].to(device, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(image)
        h, w = sample.original_hw
        probability = torch.sigmoid(logits.float())[0, 0, :h, :w].cpu().numpy()
        if not np.isfinite(probability).all():
            raise FloatingPointError("Non-finite validation probabilities")
        probabilities.append(probability)
        targets.append(sample.mask[0, :h, :w].numpy().astype(bool))
        ids.append(sample.image_id)
    model.train(was_training)
    return evaluate_probabilities(probabilities, targets, ids), probabilities


class Events:
    def __init__(self, path: Path):
        self.stream = path.open("a", encoding="utf-8", buffering=1)

    def emit(self, event: str, **values):
        record = dict(time=now(), event=event, **values)
        line = json.dumps(record, ensure_ascii=True, allow_nan=False)
        self.stream.write(line + "\n")
        self.stream.flush()
        print(line, flush=True)

    def close(self):
        self.stream.close()


def train_fold(root: Path, run: Path, config: dict, fold_index: int, dataset_hash: str,
               device: torch.device, resume: bool, stop_after: int | None) -> bool:
    directory = run / f"fold_{fold_index}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "last.pt").exists() and not resume:
        raise FileExistsError(f"Existing fold checkpoint; use --resume: {directory}")
    fold = get_fold(root, fold_index, dataset_hash)
    cfg = copy.deepcopy(config)
    cfg["fold"] = fold_index
    cfg["seed"] = config["seed"] + fold_index
    current_source = source_digest()
    seed_everything(cfg["seed"], cfg["train"]["deterministic"])
    training = load_samples(root, fold["train_ids"], cfg["data"]["pad_hw"])
    validation = load_samples(root, fold["validation_ids"], cfg["data"]["pad_hw"])
    model = VesselUNet(cfg["model"]).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_step, best_score, best_step, previous_elapsed = 0, -1.0, 0, 0.0
    last_metrics = None
    if resume and (directory / "last.pt").exists():
        checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["config"] != cfg or checkpoint["dataset_sha256"] != dataset_hash or checkpoint["fold"] != fold:
            raise ValueError("Resume requires identical configuration, data and fold IDs")
        if checkpoint["source_sha256"] != current_source:
            raise ValueError("Training source changed; keep the original snapshot or start a new run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step, best_score = checkpoint["step"], checkpoint["best_score"]
        best_step, previous_elapsed = checkpoint["best_step"], checkpoint["elapsed_seconds"]
        last_metrics = checkpoint["validation"]
        restore_rng(checkpoint["rng"])
    else:
        write_json(directory / "config.json", cfg)
        write_json(directory / "fold.json", fold)
    logger = Events(directory / "train.jsonl")
    pid = os.getpid()
    target_steps = cfg["train"]["steps"]
    start_time = time.perf_counter()
    last_log_time, last_log_step = start_time, start_step
    step = start_step
    validation_step = -1
    skipped_updates = 0
    loss_window = []
    parameter_count = sum(p.numel() for p in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def status(state: str, **extra):
        record = dict(state=state, updated_at=now(), pid=pid, fold=fold_index, step=step,
                      target_steps=target_steps, best_dice_at_0_5=best_score if best_score >= 0 else None,
                      best_step=best_step, elapsed_seconds=previous_elapsed + time.perf_counter() - start_time,
                      parameter_count=parameter_count, **extra)
        write_json(directory / "status.json", record)
        write_json(run / "status.json", record)

    def checkpoint_payload():
        return dict(schema_version=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), rng=capture_rng(), config=cfg, fold=fold,
                    dataset_sha256=dataset_hash, source_sha256=current_source, step=step,
                    best_score=best_score, best_step=best_step, validation=last_metrics,
                    elapsed_seconds=previous_elapsed + time.perf_counter() - start_time,
                    lr_schedule="saved configuration + completed step", saved_at=now())

    def validation_and_save():
        nonlocal best_score, best_step, last_metrics, validation_step
        last_metrics, predictions = validate(model, validation, device, amp)
        validation_step = step
        improved = last_metrics["mean_dice_at_0_5"] > best_score
        if improved:
            best_score, best_step = last_metrics["mean_dice_at_0_5"], step
            save_checkpoint(directory / "best.pt", checkpoint_payload())
            np.savez_compressed(directory / "best_predictions.npz",
                                ids=np.asarray(fold["validation_ids"]), probabilities=np.stack(predictions))
            write_json(directory / "best_metrics.json", dict(step=step, **last_metrics))
        save_checkpoint(directory / "last.pt", checkpoint_payload())
        write_json(directory / "last_metrics.json", dict(step=step, **last_metrics))
        logger.emit("validation", fold=fold_index, step=step,
                    dice_at_0_5=last_metrics["mean_dice_at_0_5"],
                    calibrated_dice=last_metrics["calibrated_mean_dice"],
                    calibrated_threshold=last_metrics["calibrated_threshold"],
                    best_dice_at_0_5=best_score, best_step=best_step,
                    per_image=last_metrics["per_image"])
        status("running")

    try:
        logger.emit("resumed" if start_step else "started", fold=fold_index, seed=cfg["seed"],
                    step=start_step, target_steps=target_steps, parameters=parameter_count,
                    device=str(device), amp=amp, training_ids=fold["train_ids"], validation_ids=fold["validation_ids"],
                    dataset_sha256=dataset_hash, source_sha256=current_source)
        status("running")
        if start_step >= target_steps:
            status("completed")
            return True
        model.train()
        for step in range(start_step + 1, target_steps + 1):
            if (run / "STOP").exists():
                step -= 1
                break
            images, masks = sample_patches(training, cfg["train"]["batch_size"], cfg["data"]["patch_hw"], cfg["augmentation"])
            images = photometric(images.to(device, memory_format=torch.channels_last), cfg["augmentation"])
            masks = masks.to(device)
            lr = learning_rate(cfg["train"], step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = segmentation_loss(logits, masks, cfg["loss"])
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip_norm"])
            norm_value = float(norm)
            skipped_updates = skipped_updates + 1 if not math.isfinite(norm_value) else 0
            if skipped_updates >= 10:
                raise FloatingPointError("Ten consecutive non-finite gradients")
            scaler.step(optimizer)
            scaler.update()
            loss_window.append(loss_value)
            if step == 1 or step % cfg["train"]["log_every"] == 0:
                elapsed = time.perf_counter() - last_log_time
                seconds_per_step = elapsed / max(1, step - last_log_step)
                logger.emit("train", fold=fold_index, step=step, loss=float(np.mean(loss_window)), lr=lr,
                            grad_norm=norm_value if math.isfinite(norm_value) else None,
                            seconds_per_step=seconds_per_step, eta_seconds=(target_steps - step) * seconds_per_step,
                            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0)
                last_log_time, last_log_step = time.perf_counter(), step
                loss_window.clear()
                status("running")
            if step % cfg["train"]["validate_every"] == 0 or step == target_steps:
                validation_and_save()
            if stop_after is not None and step >= stop_after:
                break
        if step > 0 and validation_step != step:
            validation_and_save()
        complete = step >= target_steps
        status("completed" if complete else "stopped")
        logger.emit("completed" if complete else "stopped", fold=fold_index, step=step, best_step=best_step,
                    best_dice_at_0_5=best_score, elapsed_seconds=previous_elapsed + time.perf_counter() - start_time)
        return complete
    except BaseException as error:
        status("failed", error=str(error), traceback=traceback.format_exc())
        logger.emit("failed", fold=fold_index, step=step, error=str(error))
        raise
    finally:
        logger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_csa.json"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", choices=["all", "0", "1", "2", "3", "4"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--stop-after", type=int, help="Stop after this absolute step; retain original LR schedule for resume")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for attribute, key in (("steps", "steps"), ("batch_size", "batch_size"), ("validate_every", "validate_every"), ("log_every", "log_every")):
        value = getattr(args, attribute)
        if value is not None:
            config["train"][key] = value
    if min(config["train"][k] for k in ("steps", "batch_size", "validate_every", "log_every")) < 1:
        raise ValueError("Training intervals, steps, and batch size must be positive")
    torch.set_num_threads(config["train"]["cpu_threads"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select a working GPU environment")
    run = (root / args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "configuration.json").exists():
        recorded = json.loads((run / "configuration.json").read_text(encoding="utf-8"))
        if recorded != config:
            raise ValueError("Run directory belongs to a different configuration")
    else:
        write_json(run / "configuration.json", config)
        record_environment(run, root)
    dataset_hash = verify_dataset(root)
    folds = range(5) if args.fold == "all" else [int(args.fold)]
    for fold in folds:
        if (run / "STOP").exists():
            break
        complete = train_fold(root, run, config, fold, dataset_hash, device, args.resume, args.stop_after)
        if not complete:
            break
    if args.fold == "all" and all((run / f"fold_{f}/status.json").exists() for f in range(5)):
        fold_states = [json.loads((run / f"fold_{f}/status.json").read_text(encoding="utf-8")) for f in range(5)]
        if all(s["state"] == "completed" for s in fold_states):
            from .evaluate import summarize
            metrics = summarize(root, run)
            print(json.dumps(dict(event="oof_complete", mean_dice_at_0_5=metrics["mean_dice_at_0_5"],
                                  calibrated_mean_dice=metrics["calibrated_mean_dice"],
                                  calibrated_threshold=metrics["calibrated_threshold"])), flush=True)
            write_json(run / "status.json", dict(state="completed_all_folds", updated_at=now(), pid=os.getpid(), folds=fold_states))


if __name__ == "__main__":
    main()
