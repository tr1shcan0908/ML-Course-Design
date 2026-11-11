"""Short, label-supervised fine-tuning of the five ConvNeXt fold checkpoints."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .convnext_full import digest, verify_pilot
from .convnext_study import checkpoint_for, compare, read
from .runtime import capture_rng, now, restore_rng, save_checkpoint, seed_everything, write_json

from filelock import FileLock
import numpy as np
import torch

from tools.prepare_submission import predict_variants
from vessel.data import get_fold, load_samples as load_v1_samples
from vessel.metrics import segmentation_loss
from .convnext import ConvNeXtUNet
from .data import load_samples, photometric, sample_patches
from .export_study import load_checkpoint_model
from .inference import predict_member
from .train import Events, learning_rate


def supervised_loss(logits, targets, anchor, pixel_weights, candidate_weight, ensemble_fraction):
    single = segmentation_loss(logits, targets, pixel_weights)
    if ensemble_fraction == 0:
        return single
    blended = anchor + candidate_weight * logits.float().sigmoid()
    ensemble = segmentation_loss(torch.logit(blended, eps=1e-6), targets, pixel_weights)
    return (1 - ensemble_fraction) * single + ensemble_fraction * ensemble


def training_samples(root, fold, config, anchors, data_hash, device):
    ids = fold["train_ids"]
    samples = load_samples(root, ids, config["data"])
    priors = [torch.zeros_like(sample.mask) for sample in samples]
    for member in anchors:
        entry = next(row for row in member["checkpoints"] if row["fold"] == fold["fold"])
        path = root / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Frozen anchor checkpoint changed: {path}")
        model, checkpoint = load_checkpoint_model(root, path, member["implementation"], device)
        if checkpoint["fold"] != fold or checkpoint["dataset_sha256"] != data_hash:
            raise ValueError(f"Anchor must come from the same held-out fold: {path}")
        data = checkpoint["config"]["data"]
        inputs = (load_v1_samples(root, ids, data["pad_hw"]) if member["implementation"] == "v1"
                  else load_samples(root, ids, data))
        for sample, prior in zip(inputs, priors):
            height, width = sample.original_hw
            p = predict_member(model, sample, device, member)
            if p.shape != (height, width) or not np.isfinite(p).all():
                raise ValueError(f"Invalid anchor probability for image {sample.image_id}")
            prior[0, :height, :width] += member["weight"] * torch.from_numpy(p)
        del model, checkpoint
    for sample, prior in zip(samples, priors):
        sample.mask = torch.cat((sample.mask, prior), dim=0)
    return samples


def fit(root, run, fold, base, base_sha, samples, mode, fraction, candidate_weight, plan, device):
    index = fold["fold"]
    config = base["config"]
    stage = plan["train"]
    train = config["train"]
    seed_everything(config["seed"] + plan["seed_offset"], train["deterministic"])
    model = ConvNeXtUNet(config["model"], root, initialize=False)
    model.load_state_dict(base["model"])
    model.encoder.requires_grad_(False)
    model = model.to(device, memory_format=torch.channels_last).train()
    model.encoder.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=stage["lr"], weight_decay=train["weight_decay"])
    amp = train["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output = run / mode / f"fold_{index}"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "last.pt"
    step = skipped = 0
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved["fold"], saved["mode"], saved["base_sha256"], saved["plan"]) != (fold, mode, base_sha, plan):
            raise ValueError(f"Fine-tuning checkpoint lineage mismatch: {path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        step, skipped = saved["step"], saved["skipped"]
        del saved
    checkpoint_step = step

    def save():
        save_checkpoint(path, dict(fold=fold, mode=mode, base_sha256=base_sha, plan=plan,
                                   config=config, step=step, skipped=skipped, model=model.state_dict(),
                                   optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=capture_rng()))

    events = Events(output / "train.jsonl")
    try:
        events.emit("started", mode=mode, fold=index, step=step, target_steps=stage["steps"])
        while step < stage["steps"] and not (run / "STOP").exists():
            images, supervision = sample_patches(samples, train["batch_size"], config["data"]["patch_hw"],
                                                 config["augmentation"])
            images = photometric(images.to(device, memory_format=torch.channels_last), config["augmentation"])
            supervision = supervision.to(device)
            lr = learning_rate(stage, step + 1)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = supervised_loss(model(images), supervision[:, :1], supervision[:, 1:],
                                       config["loss"], candidate_weight, fraction)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite fine-tuning loss at step {step + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, train["gradient_clip_norm"])
            skipped = 0 if math.isfinite(float(norm)) else skipped + 1
            if skipped and (not amp or skipped >= 10):
                raise FloatingPointError(f"Non-finite fine-tuning gradients at step {step + 1}")
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % stage["checkpoint_every"] == 0 or step == stage["steps"]:
                save()
                checkpoint_step = step
            if step == 1 or step % 100 == 0 or step == stage["steps"]:
                events.emit("train", mode=mode, fold=index, step=step, loss=value, lr=lr,
                            checkpoint_step=checkpoint_step)
                write_json(run / "status.json", dict(state="training", pid=os.getpid(), fold=index,
                                                    mode=mode, step=step, updated_at=now()))
        if step != stage["steps"]:
            if step > checkpoint_step:
                save()
            return None
        model.eval()
        held_out = load_samples(root, fold["validation_ids"], config["data"])
        probabilities = np.stack([predict_variants(model, sample, device, True)[1] for sample in held_out])
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite fine-tuning validation probabilities")
        events.emit("fold_completed", mode=mode, fold=index, step=step)
        return probabilities
    finally:
        events.close()


def evaluate(root, plan, recipe, folds, archived, predictions):
    replacement = plan["replace_member"]
    name = f"replace_{replacement}"
    scoring = {"replacements": [replacement], "minimum_gain_over_v3": plan["minimum_gain_over_v3"]}
    ids, reference = compare(root, scoring, recipe, folds, archived)
    scores = {"v3": reference["scores"]["v3"], "v6": reference["scores"][name]}
    single_scores = {"v6": reference["scores"]["convnext"]}
    comparisons = {}
    for mode, probabilities in predictions.items():
        _, result = compare(root, scoring, recipe, folds, probabilities)
        scores[mode] = result["scores"][name]
        single_scores[mode] = result["scores"]["convnext"]
        comparison = result["comparisons"][replacement]
        gain = scores[mode]["mean_dice"] - scores["v6"]["mean_dice"]
        comparisons[mode] = dict(gain_over_v6=gain, gain_over_v3=comparison["gain_over_v3"],
                                 gain_vs_v3_by_fold=comparison["gain_by_fold"],
                                 passes_improvement_gate=comparison["passes_screening"] and gain > 0)
    return ids, dict(scores=scores, single_model_scores=single_scores, comparisons=comparisons)


def main():
    root = Path(__file__).resolve().parents[1]
    plan_path = root / "experiments/convnext_sft_v7.json"
    plan = read(plan_path)
    base_run = root / plan["base_run"]
    run = root / "runs/convnext_sft_v7"
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run / ".training.lock", timeout=0), FileLock(base_run / ".training.lock", timeout=0):
        try:
            write_json(run / "status.json", dict(state="checking_inputs", pid=os.getpid(), updated_at=now()))
            pilot_path = root / "runs/convnext_tiny_unet_v6/provenance.json"
            pilot = read(pilot_path)
            device, data_hash = verify_pilot(root, pilot)
            config = pilot["config"]
            torch.set_num_threads(config["train"]["cpu_threads"])
            base_meta = read(base_run / "provenance.json")
            if (base_meta["pilot_provenance_sha256"] != digest(pilot_path)
                    or base_meta["training_source_sha256"] != pilot["source_sha256"]
                    or base_meta["config"] != config or base_meta["candidate_tta"] != "flip4"
                    or read(base_run / "workflow_status.json")["state"] != "completed"):
                raise ValueError("A completed and verified ConvNeXt five-fold run is required")
            folds = [get_fold(root, fold["fold"], data_hash) for fold in base_meta["folds"]]
            held_ids = [i for fold in folds for i in fold["validation_ids"]]
            if len(folds) != 5 or folds != base_meta["folds"] or len(held_ids) != 20 or len(set(held_ids)) != 20:
                raise ValueError("Fine-tuning requires the frozen five held-out folds")
            recipe_path = root / "submission_v3/inference_recipe.json"
            recipe = read(recipe_path)
            anchors = [member for member in recipe["members"] if member["name"] != plan["replace_member"]]
            candidate_weight, = [member["weight"] for member in recipe["members"]
                                 if member["name"] == plan["replace_member"]]
            base_metrics = base_run / "full_results.json"
            base_folds = {row["fold"]: row for row in read(base_metrics)["folds"]}
            base_oof = base_run / "full_oof.npz"
            with np.load(base_oof, allow_pickle=False) as archived:
                if archived["ids"].tolist() != held_ids:
                    raise ValueError("Base ConvNeXt OOF image ordering differs")
                reference = archived["convnext"].copy()
            meta = dict(plan=plan, dataset_sha256=data_hash, folds=folds, config=config,
                        base_results_sha256=digest(base_metrics), base_oof_sha256=digest(base_oof),
                        recipe_sha256=digest(recipe_path), plan_sha256=digest(plan_path),
                        code_sha256=digest(Path(__file__)), pilot_source_sha256=pilot["source_sha256"])
            metadata = run / "provenance.json"
            if metadata.is_file() and read(metadata) != meta:
                raise ValueError("Fine-tuning inputs or source differ from the existing run")
            write_json(metadata, meta)

            predictions = {mode: [] for mode in plan["ensemble_loss_fractions"]}
            for fold in folds:
                index = fold["fold"]
                path = base_run / f"fold_{index}/best.pt"
                expected = base_folds[index]
                base_sha = digest(path)
                if base_sha != expected["best_checkpoint_sha256"]:
                    raise ValueError(f"Base fine-tuning checkpoint changed: {path}")
                base = checkpoint_for(path, config, fold, data_hash, pilot["source_sha256"])
                if base["step"] != expected["best_step"]:
                    raise ValueError(f"Unexpected base fine-tuning checkpoint step: {path}")
                seed_everything(base["config"]["seed"] + plan["seed_offset"], config["train"]["deterministic"])
                write_json(run / "status.json", dict(state="preparing_anchors", pid=os.getpid(),
                                                    fold=index, updated_at=now()))
                samples = training_samples(root, fold, config, anchors, data_hash, device)
                for mode, fraction in plan["ensemble_loss_fractions"].items():
                    probabilities = fit(root, run, fold, base, base_sha, samples, mode, fraction,
                                        candidate_weight, plan, device)
                    if probabilities is None:
                        write_json(run / "status.json", dict(state="paused", fold=index, mode=mode, updated_at=now()))
                        return
                    predictions[mode].append(probabilities)
                del base, samples

            predictions = {mode: np.concatenate(values) for mode, values in predictions.items()}
            ids, result = evaluate(root, plan, recipe, folds, reference, predictions)
            np.savez_compressed(run / "sft_oof.npz", ids=np.asarray(ids), **predictions)
            write_json(run / "results.json", dict(completed_at=now(), **result))
            write_json(run / "status.json", dict(state="completed", updated_at=now()))
            print(json.dumps(result["comparisons"]), flush=True)
        except Exception as error:
            write_json(run / "status.json", dict(state="failed", updated_at=now(), error=str(error)))
            raise


if __name__ == "__main__":
    main()
