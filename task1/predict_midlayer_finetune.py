"""Reload five fine-tuned DINOv2-B midlayer models to classify new images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from evaluate_frozen_patch_probe import gram
from midlayer_finetune_framework import load_model
from train_classifier import discover_test_records, make_loader
from train_dino_token import normalize_probs, sha256
from train_midlayer_finetune import extract, transforms_for_run


def predict_records(records, run_dir: Path, batch_size: int, device: torch.device):
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    classes = config["class_names"]
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vitb14_reg4_pretrain.pth"
    if sha256(weights_path) != config["pretrained_sha256"]:
        raise ValueError("Pretrained DINOv2 checkpoint does not match the experiment")
    _, eval_transform = transforms_for_run(int(config["image_size"]))
    loader = make_loader(records, eval_transform, batch_size, 0, False, int(config["seed"]))
    result = np.zeros((len(records), len(classes)), dtype=np.float64)
    expected_names = [record.path.name for record in records]
    for fold in range(5):
        checkpoint = torch.load(run_dir / f"fold_{fold}_final.pt", map_location="cpu",
                                weights_only=False)
        if checkpoint["fold"] != fold or checkpoint["class_names"] != classes or \
                checkpoint["config_sha256"] != sha256(config_path):
            raise ValueError(f"Fold {fold} checkpoint metadata mismatch")
        model = load_model(len(classes))
        missing, unexpected = model.load_state_dict(checkpoint["trained_state"], strict=False)
        train_prefixes = ("backbone.blocks.6.", "backbone.blocks.7.",
                          "backbone.norm.", "head.")
        if unexpected or any(name.startswith(train_prefixes) for name in missing):
            raise RuntimeError(f"Fold {fold} checkpoint is incomplete")
        model.eval().to(device)
        features, _, _, names = extract(model, loader, device)
        if names != expected_names:
            raise RuntimeError("Prediction IDs differ from input order")
        artifact = joblib.load(run_dir / f"fold_{fold}_svc.joblib")
        if artifact["class_names"] != classes or \
                not np.array_equal(artifact["model"].classes_, np.arange(len(classes))):
            raise ValueError("SVC class order mismatch")
        kind, gamma, _ = artifact["params"]
        _, kernel = gram(np.asarray(artifact["train_features"], dtype=np.float64),
                         features, kind, gamma)
        result += normalize_probs(artifact["model"].predict_proba(kernel)) / 5
        print(f"predicted fold={fold} images={len(records)}", flush=True)
        del model
        torch.cuda.empty_cache()
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1, atol=1e-5):
        raise RuntimeError("Invalid ensemble probabilities")
    return normalize_probs(result), classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path(
        "outputs/experiments/dinov2b_mid_finetune_v1"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    records = discover_test_records(args.input_dir.resolve(),
                                    args.template.resolve() if args.template else None)
    probs, classes = predict_records(records, args.run_dir.resolve(), args.batch_size, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "Category", *[f"p_{name}" for name in classes]])
        for record, row in zip(records, probs, strict=True):
            writer.writerow([record.path.name, classes[int(row.argmax())], *map(float, row)])
    print(f"Saved {len(records)} predictions to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
