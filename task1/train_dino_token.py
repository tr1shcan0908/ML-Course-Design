"""Five-fold DINOv2 token-pooling experiment with a fixed training schedule.

The outer validation fold is never used to pick a checkpoint or hyperparameter.
It is scored only after the prescribed warmup and partial fine-tuning epochs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dino_token_framework import DinoTokenClassifier
from train_classifier import (
    build_transforms,
    classification_metrics,
    discover_test_records,
    discover_training_records,
    make_loader,
    predict,
    save_prediction_table,
    set_seed,
    write_fold_assignments,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_assignments(path: Path, records, class_names: list[str]) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(records) or len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Fold assignments are incomplete or contain duplicate filenames")
    lookup = {row["file"]: row for row in rows}
    if set(lookup) != {record.path.name for record in records}:
        raise ValueError("Current training files differ from the reference five-fold split")
    result = []
    for record in records:
        row = lookup[record.path.name]
        if row["class"] != record.class_name or int(row["label"]) != record.label:
            raise ValueError(f"Label mismatch in reference split: {record.path.name}")
        result.append(int(row["fold"]))
    assignments = np.asarray(result, dtype=np.int64)
    if set(assignments) != set(range(5)):
        raise ValueError("Expected the original five folds")
    for fold in range(5):
        indices = np.flatnonzero(assignments == fold)
        labels = [records[index].label for index in indices]
        if len(indices) != 100 or any(labels.count(label) != 20 for label in range(len(class_names))):
            raise ValueError(f"Reference fold {fold} is not the expected balanced 100-image split")
    return assignments


def load_model(classes: int, trainable_blocks: int) -> DinoTokenClassifier:
    hub_dir = Path(torch.hub.get_dir())
    repo = hub_dir / "facebookresearch_dinov2_main"
    weights = hub_dir / "checkpoints" / "dinov2_vits14_pretrain.pth"
    if not repo.is_dir() or not weights.is_file():
        raise FileNotFoundError("Cached DINOv2 ViT-S/14 repo and weights are required")
    backbone = torch.hub.load(str(repo), "dinov2_vits14", source="local", pretrained=True)
    return DinoTokenClassifier(backbone, classes, trainable_blocks)


def normalize_probs(probabilities: np.ndarray) -> np.ndarray:
    output = np.asarray(probabilities, dtype=np.float64)
    output /= output.sum(axis=1, keepdims=True)
    return output


def make_optimizer(model: DinoTokenClassifier, stage: str, args) -> torch.optim.Optimizer:
    if stage == "warmup":
        return torch.optim.AdamW(model.head.parameters(), lr=args.lr_warmup, weight_decay=args.weight_decay)
    groups = []
    for position, block in enumerate(model.backbone.blocks[-args.trainable_blocks:]):
        multiplier = 0.5 ** (args.trainable_blocks - position - 1)
        groups.append({"params": list(block.parameters()), "lr": args.lr_backbone * multiplier})
    groups.append({"params": list(model.backbone.norm.parameters()), "lr": args.lr_backbone})
    groups.append({"params": list(model.head.parameters()), "lr": args.lr_head})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def train_epoch(model, loader, optimizer, criterion, scaler, device, accumulation: int, stage: str) -> dict:
    model.train()
    if stage == "warmup":
        model.backbone.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_correct = total_count = 0
    optimizer_steps = 0
    for batch_index, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        remaining = len(loader) - (batch_index // accumulation) * accumulation
        group_size = min(accumulation, remaining)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss / group_size).backward()
        total_loss += float(loss.detach()) * len(labels)
        total_correct += int((logits.argmax(1) == labels).sum())
        total_count += len(labels)
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    return {"loss": total_loss / total_count, "train_accuracy": total_correct / total_count,
            "optimizer_steps": optimizer_steps}


def save_test_table(path: Path, records, class_names: list[str], probs: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "Category", *[f"p_{name}" for name in class_names]])
        for record, row in zip(records, probs, strict=True):
            writer.writerow([record.path.name, class_names[int(row.argmax())], *map(float, row)])


def save_submission(path: Path, records, class_names: list[str], probs: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "Category"])
        for record, row in zip(records, probs, strict=True):
            writer.writerow([record.path.name, class_names[int(row.argmax())]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset-for-task2/dataset-for-task2"))
    parser.add_argument("--submission-template", type=Path, default=Path("submission-for-task2.csv"))
    parser.add_argument("--reference-run", type=Path, default=Path("outputs/experiments/convnext_tiny_stage2_320"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/dino_token_v1"))
    parser.add_argument("--image-size", type=int, default=280)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--trainable-blocks", type=int, default=4)
    parser.add_argument("--lr-warmup", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-index", type=int, action="append")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size % 14 or args.image_size < 224 or args.batch_size < 1 or args.accumulation < 1:
        raise ValueError("Image size must be a multiple of 14 and at least 224; batches must be positive")
    if args.warmup_epochs < 1 or args.finetune_epochs < 1:
        raise ValueError("Both phases need at least one epoch")
    if not torch.cuda.is_available():
        raise RuntimeError("This run requires CUDA")
    selected_folds = sorted(set(args.fold_index or list(range(5))))
    if not set(selected_folds).issubset(set(range(5))):
        raise ValueError("Fold indices must be 0 to 4")
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    reference = args.reference_run.resolve()
    records, class_names = discover_training_records(data_root / "train")
    test_records = discover_test_records(data_root / "test", args.submission_template.resolve())
    split_path = reference / "fold_assignments.csv"
    assignments = read_assignments(split_path, records, class_names)
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vits14_pretrain.pth"
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in ("fold_index", "resume")}
    config.update({"class_names": class_names, "split_sha256": sha256(split_path),
                   "pretrained_sha256": sha256(weights_path), "torch_version": torch.__version__,
                   "validation_policy": "Fixed epochs; outer fold labels never select weights, schedule, or fusion."})
    config_path = output_dir / "config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing run has another config; use --resume or a new output directory")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("Output directory is nonempty and has no matching run config")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)
    write_fold_assignments(output_dir / "fold_assignments.csv", records, assignments)
    train_transform, eval_transform = build_transforms(args.image_size, "full")
    device = torch.device("cuda")
    for fold in selected_folds:
        fold_start = time.perf_counter()
        prediction_path = output_dir / f"fold_{fold}_predictions.npz"
        checkpoint_path = output_dir / f"fold_{fold}_final.pt"
        metrics_path = output_dir / f"fold_{fold}_metrics.json"
        if args.resume and all(path.exists() for path in (prediction_path, checkpoint_path, metrics_path)):
            print(f"fold={fold} reused completed artifacts", flush=True)
            continue
        set_seed(args.seed + fold)
        train_records = [records[index] for index in np.flatnonzero(assignments != fold)]
        val_records = [records[index] for index in np.flatnonzero(assignments == fold)]
        train_loader = make_loader(train_records, train_transform, args.batch_size, args.workers, True, args.seed + fold)
        val_loader = make_loader(val_records, eval_transform, args.batch_size, args.workers, False, args.seed + fold)
        test_loader = make_loader(test_records, eval_transform, args.batch_size, args.workers, False, args.seed + fold)
        model = load_model(len(class_names), args.trainable_blocks).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        scaler = torch.amp.GradScaler("cuda")
        history = []
        for stage, epochs in (("warmup", args.warmup_epochs), ("finetune", args.finetune_epochs)):
            model.set_backbone_trainable(stage == "finetune")
            optimizer = make_optimizer(model, stage, args)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
            for epoch in range(1, epochs + 1):
                result = train_epoch(model, train_loader, optimizer, criterion, scaler, device, args.accumulation, stage)
                history.append({"stage": stage, "epoch": epoch, **result})
                scheduler.step()
                print(f"fold={fold} stage={stage} epoch={epoch}/{epochs} "
                      f"loss={result['loss']:.4f} train_acc={result['train_accuracy']:.4f}", flush=True)
                write_json(output_dir / f"fold_{fold}_progress.json", {"history": history})
            del optimizer, scheduler
        val_probs, val_labels, val_names = predict(model, val_loader, device, True)
        test_probs, _, test_names = predict(model, test_loader, device, True)
        if val_names != [record.path.name for record in val_records] or test_names != [record.path.name for record in test_records]:
            raise ValueError("Prediction order does not match the input records")
        val_probs, test_probs = normalize_probs(val_probs), normalize_probs(test_probs)
        np.savez_compressed(prediction_path, val_probs=val_probs, test_probs=test_probs,
                            val_files=np.asarray(val_names), test_files=np.asarray(test_names))
        torch.save({"model": {key: value.cpu() for key, value in model.state_dict().items()},
                    "architecture": "dinov2_vits14_token_pool", "fold": fold,
                    "class_names": class_names, "image_size": args.image_size,
                    "trainable_blocks": args.trainable_blocks, "config_sha256": sha256(config_path)}, checkpoint_path)
        result = classification_metrics(val_labels, val_probs.argmax(1), class_names)
        write_json(metrics_path, {"fold": fold, "metrics": result, "history": history,
                                  "validation_files": val_names,
                                  "elapsed_seconds": time.perf_counter() - fold_start})
        print(f"fold={fold} final_accuracy={result['accuracy']:.4f} "
              f"final_macro_f1={result['macro_f1']:.4f}", flush=True)
        del model, scaler
        torch.cuda.empty_cache()
    completed = [fold for fold in range(5) if all((output_dir / f"fold_{fold}_{suffix}").exists()
                 for suffix in ("predictions.npz", "final.pt", "metrics.json"))]
    oof = np.full((len(records), len(class_names)), np.nan, dtype=np.float64)
    test_arrays = []
    fold_results = []
    for fold in completed:
        with np.load(output_dir / f"fold_{fold}_predictions.npz", allow_pickle=False) as archive:
            indices = np.flatnonzero(assignments == fold)
            if archive["val_files"].astype(str).tolist() != [records[index].path.name for index in indices]:
                raise ValueError(f"Fold {fold} saved validation files differ from reference split")
            if archive["test_files"].astype(str).tolist() != [record.path.name for record in test_records]:
                raise ValueError(f"Fold {fold} saved test order differs from template")
            oof[indices] = normalize_probs(archive["val_probs"])
            test_arrays.append(normalize_probs(archive["test_probs"]))
        fold_results.append(json.loads((output_dir / f"fold_{fold}_metrics.json").read_text(encoding="utf-8"))["metrics"])
    available = np.flatnonzero(np.isfinite(oof).all(axis=1))
    labels = np.asarray([records[index].label for index in available])
    metric = classification_metrics(labels, oof[available].argmax(1), class_names)
    metric.update({"completed_folds": completed, "evaluated_images": len(available),
                   "fold_accuracy": [item["accuracy"] for item in fold_results],
                   "fold_macro_f1": [item["macro_f1"] for item in fold_results],
                   "validation_policy": config["validation_policy"]})
    write_json(output_dir / "metrics.json", metric)
    save_prediction_table(output_dir / "oof_predictions.csv",
                          [records[index].path.name for index in available], labels,
                          oof[available], class_names)
    if len(completed) == 5:
        test_probs = normalize_probs(np.mean(test_arrays, axis=0))
        save_test_table(output_dir / "test_probabilities.csv", test_records, class_names, test_probs)
        save_submission(output_dir / "submission.csv", test_records, class_names, test_probs)
    print(json.dumps({"completed_folds": completed, "accuracy": metric["accuracy"],
                      "macro_f1": metric["macro_f1"]}), flush=True)


if __name__ == "__main__":
    main()
