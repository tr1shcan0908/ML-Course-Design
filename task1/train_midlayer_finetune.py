"""Five-fold DINOv2-B layer-8 partial fine-tuning on plant seedlings.

The same fixed schedule is used for every fold.  Each outer validation image
is untouched by training and is scored once after the final epoch.  A second
classifier refits an SVC on the fine-tuned features of the 400 training images.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from torch import nn
from torchvision import transforms

from evaluate_frozen_patch_probe import choose_params, gram
from midlayer_finetune_framework import load_model
from train_classifier import (IMAGENET_MEAN, IMAGENET_STD, classification_metrics,
                              discover_test_records, discover_training_records,
                              make_loader, save_prediction_table, set_seed)
from train_dino_token import (normalize_probs, read_assignments, save_submission,
                              save_test_table, sha256, write_json)


def transforms_for_run(size: int):
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.84, 1.0), ratio=(0.9, 1.1),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=.08, contrast=.08, saturation=.08, hue=.015),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    evaluation = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, evaluation


def optimizer_for(model, args, stage):
    if stage == "warmup":
        groups = [{"params": model.head.parameters(), "lr": args.lr_warmup}]
    else:
        groups = [
            {"params": model.backbone.blocks[6].parameters(), "lr": args.lr_backbone / 2},
            {"params": model.backbone.blocks[7].parameters(), "lr": args.lr_backbone},
            {"params": model.backbone.norm.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(), "lr": args.lr_head},
        ]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def train_epoch(model, loader, optimizer, scaler, criterion, device, accumulation):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count = correct = 0
    loss_sum = 0.0
    for i, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        group_size = min(accumulation, len(loader) - (i // accumulation) * accumulation)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss / group_size).backward()
        count += len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        loss_sum += float(loss.detach()) * len(labels)
        if (i + 1) % accumulation == 0 or i + 1 == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return {"loss": loss_sum / count, "train_accuracy": correct / count}


@torch.inference_mode()
def extract(model, loader, device):
    model.eval()
    features, probabilities, labels, names = [], [], [], []
    for images, batch_labels, batch_names in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            value = model.features(images)
            logits = model.head(value * 10.0)
        features.append(value.float().cpu().numpy())
        probabilities.append(logits.float().softmax(1).cpu().numpy())
        labels.extend(batch_labels.tolist())
        names.extend(batch_names)
    output = np.concatenate(features).astype(np.float64)
    if not np.isfinite(output).all() or not np.allclose(np.linalg.norm(output, axis=1), 1, atol=1e-4):
        raise RuntimeError("Extracted features are not finite, unit-length vectors")
    return output, normalize_probs(np.concatenate(probabilities)), np.asarray(labels), names


def save_checkpoint(path, model, fold, classes, config_path):
    kept = ("backbone.blocks.6.", "backbone.blocks.7.", "backbone.norm.", "head.")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()
             if key.startswith(kept)}
    torch.save({"trained_state": state, "fold": fold, "class_names": classes,
                "architecture": "DINOv2-B/14-register, layer 8, train blocks 7-8",
                "config_sha256": sha256(config_path)}, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset-for-task2/dataset-for-task2"))
    parser.add_argument("--submission-template", type=Path, default=Path("submission-for-task2.csv"))
    parser.add_argument("--reference-run", type=Path, default=Path("outputs/experiments/convnext_tiny_stage2_320"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/dinov2b_mid_finetune_v1"))
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--lr-warmup", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=.05)
    parser.add_argument("--c-grid", default="0.1,1,10,100,1000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-index", type=int, action="append")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is needed for this fine-tuning run")
    if args.image_size % 14 or min(args.batch_size, args.accumulation, args.warmup_epochs,
                                   args.finetune_epochs) < 1:
        raise ValueError("Invalid image size or training schedule")
    selected_folds = sorted(set(args.fold_index or range(5)))
    if not set(selected_folds).issubset(set(range(5))):
        raise ValueError("Fold indices must be 0-4")
    c_values = [float(value) for value in args.c_grid.split(",")]
    if any(value <= 0 or not np.isfinite(value) for value in c_values):
        raise ValueError("C grid must contain positive finite values")
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    records, classes = discover_training_records(data_root / "train")
    test_records = discover_test_records(data_root / "test", args.submission_template.resolve())
    split_path = args.reference_run.resolve() / "fold_assignments.csv"
    assignments = read_assignments(split_path, records, classes)
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vitb14_reg4_pretrain.pth"
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in ("fold_index", "resume")}
    config.update({"class_names": classes, "split_sha256": sha256(split_path),
                   "pretrained_sha256": sha256(weights_path), "torch_version": torch.__version__,
                   "validation_policy": "Fixed schedule and inner SVC CV; outer labels never select checkpoints or hyperparameters."})
    config_path = output_dir / "config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing run has another config; use --resume or a new output directory")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("Output directory is nonempty")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)
        shutil.copyfile(split_path, output_dir / "fold_assignments.csv")
    train_transform, eval_transform = transforms_for_run(args.image_size)
    device = torch.device("cuda")
    labels_all = np.asarray([record.label for record in records])
    for fold in selected_folds:
        start = time.perf_counter()
        pred_path = output_dir / f"fold_{fold}_predictions.npz"
        checkpoint_path = output_dir / f"fold_{fold}_final.pt"
        metrics_path = output_dir / f"fold_{fold}_metrics.json"
        svc_path = output_dir / f"fold_{fold}_svc.joblib"
        if args.resume and all(path.exists() for path in (pred_path, checkpoint_path, metrics_path, svc_path)):
            print(f"fold={fold} reused completed artifacts", flush=True)
            continue
        set_seed(args.seed + fold)
        train_indices = np.flatnonzero(assignments != fold)
        val_indices = np.flatnonzero(assignments == fold)
        train_records = [records[index] for index in train_indices]
        val_records = [records[index] for index in val_indices]
        train_loader = make_loader(train_records, train_transform, args.batch_size,
                                   args.workers, True, args.seed + fold)
        train_eval_loader = make_loader(train_records, eval_transform, args.batch_size,
                                        args.workers, False, args.seed + fold)
        val_loader = make_loader(val_records, eval_transform, args.batch_size,
                                 args.workers, False, args.seed + fold)
        test_loader = make_loader(test_records, eval_transform, args.batch_size,
                                  args.workers, False, args.seed + fold)
        model = load_model(len(classes)).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        scaler = torch.amp.GradScaler("cuda")
        history = []
        for stage, epochs in (("warmup", args.warmup_epochs), ("finetune", args.finetune_epochs)):
            model.set_backbone_trainable(stage == "finetune")
            optimizer = optimizer_for(model, args, stage)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            for epoch in range(1, epochs + 1):
                result = train_epoch(model, train_loader, optimizer, scaler, criterion,
                                     device, args.accumulation)
                history.append({"stage": stage, "epoch": epoch, **result})
                scheduler.step()
                write_json(output_dir / f"fold_{fold}_progress.json", {"history": history})
                print(f"fold={fold} {stage} {epoch}/{epochs} loss={result['loss']:.4f} "
                      f"train_accuracy={result['train_accuracy']:.4f}", flush=True)
            del optimizer, scheduler
        train_features, train_head, train_labels, train_names = extract(model, train_eval_loader, device)
        val_features, val_head, val_labels, val_names = extract(model, val_loader, device)
        test_features, test_head, _, test_names = extract(model, test_loader, device)
        if train_names != [record.path.name for record in train_records] or \
                val_names != [record.path.name for record in val_records] or \
                test_names != [record.path.name for record in test_records]:
            raise RuntimeError("Extracted feature order differs from record order")
        if not np.array_equal(train_labels, labels_all[train_indices]) or \
                not np.array_equal(val_labels, labels_all[val_indices]):
            raise RuntimeError("Labels differ from reference data")
        all_features = np.empty((len(records), train_features.shape[1]), dtype=np.float64)
        all_features[train_indices], all_features[val_indices] = train_features, val_features
        params, search = choose_params(all_features, labels_all, train_indices,
                                       4200 + fold, c_values)
        kind, gamma, c = params
        train_kernel, val_kernel = gram(train_features, val_features, kind, gamma)
        _, test_kernel = gram(train_features, test_features, kind, gamma)
        svc = SVC(kernel="precomputed", C=c, probability=True, random_state=4200 + fold)
        svc.fit(train_kernel, train_labels)
        if not np.array_equal(svc.classes_, np.arange(len(classes))):
            raise RuntimeError("Unexpected SVC class ordering")
        val_svc = normalize_probs(svc.predict_proba(val_kernel))
        test_svc = normalize_probs(svc.predict_proba(test_kernel))
        np.savez_compressed(pred_path, val_head=val_head, test_head=test_head,
                            val_svc=val_svc, test_svc=test_svc,
                            val_files=np.asarray(val_names), test_files=np.asarray(test_names))
        joblib.dump({"model": svc, "train_features": train_features,
                     "train_files": np.asarray(train_names), "params": params,
                     "class_names": classes}, svc_path)
        save_checkpoint(checkpoint_path, model, fold, classes, config_path)
        metrics = {"head": classification_metrics(val_labels, val_head.argmax(1), classes),
                   "svc": classification_metrics(val_labels, val_svc.argmax(1), classes),
                   "mean": classification_metrics(val_labels,
                                                  (val_head + val_svc).argmax(1), classes)}
        write_json(metrics_path, {"fold": fold, "metrics": metrics,
                                  "svc_params": params, "svc_inner_search": search,
                                  "history": history, "validation_files": val_names,
                                  "elapsed_seconds": time.perf_counter() - start})
        print(f"fold={fold} head={metrics['head']['accuracy']:.3f} "
              f"svc={metrics['svc']['accuracy']:.3f} mean={metrics['mean']['accuracy']:.3f} "
              f"params={params}", flush=True)
        del model, scaler
        torch.cuda.empty_cache()
    completed = [fold for fold in range(5) if all((output_dir / f"fold_{fold}_{suffix}").exists()
                 for suffix in ("predictions.npz", "final.pt", "metrics.json", "svc.joblib"))]
    summary = {"completed_folds": completed, "evaluated_images": 100 * len(completed),
               "validation_policy": config["validation_policy"], "methods": {}}
    for method in ("head", "svc", "mean"):
        oof = np.full((len(records), len(classes)), np.nan)
        test_arrays = []
        fold_accuracies = []
        for fold in completed:
            with np.load(output_dir / f"fold_{fold}_predictions.npz", allow_pickle=False) as archive:
                indices = np.flatnonzero(assignments == fold)
                if archive["val_files"].astype(str).tolist() != [records[i].path.name for i in indices]:
                    raise RuntimeError("Saved validation IDs differ from fold")
                if archive["test_files"].astype(str).tolist() != [r.path.name for r in test_records]:
                    raise RuntimeError("Saved test IDs differ from template")
                val = (archive["val_head"] + archive["val_svc"]) / 2 if method == "mean" else archive[f"val_{method}"]
                test = (archive["test_head"] + archive["test_svc"]) / 2 if method == "mean" else archive[f"test_{method}"]
                oof[indices] = normalize_probs(val)
                test_arrays.append(normalize_probs(test))
                fold_accuracies.append(float(accuracy_score(labels_all[indices], val.argmax(1))))
        available = np.flatnonzero(np.isfinite(oof).all(1))
        result = classification_metrics(labels_all[available], oof[available].argmax(1), classes)
        result["fold_accuracy"] = fold_accuracies
        summary["methods"][method] = result
        save_prediction_table(output_dir / f"oof_{method}.csv",
                              [records[i].path.name for i in available],
                              labels_all[available], oof[available], classes)
        if len(completed) == 5:
            test_probs = normalize_probs(np.mean(test_arrays, axis=0))
            save_test_table(output_dir / f"test_probabilities_{method}.csv",
                            test_records, classes, test_probs)
            save_submission(output_dir / f"submission_{method}.csv", test_records,
                            classes, test_probs)
    write_json(output_dir / "metrics.json", summary)
    if len(completed) == 5:
        # The SVC is the experiment's deployable classifier; expose standard names.
        for source, target in (("oof_svc.csv", "oof_predictions.csv"),
                               ("test_probabilities_svc.csv", "test_probabilities.csv"),
                               ("submission_svc.csv", "submission.csv")):
            shutil.copyfile(output_dir / source, output_dir / target)
    print(json.dumps({"completed_folds": completed,
                      "accuracy": {k: v["accuracy"] for k, v in summary["methods"].items()}},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
