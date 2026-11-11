from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    class_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validated plant seedling classifier.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--submission-template", type=Path)
    parser.add_argument(
        "--architecture",
        choices=["resnet18", "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny"],
        default="resnet18",
    )
    parser.add_argument(
        "--initialization",
        choices=["plantdoctor", "imagenet", "random"],
        default="plantdoctor",
    )
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument(
        "--strategy", choices=["linear-probe", "fine-tune"], default="fine-tune"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--reference-folds", type=Path,
                        help="Use an existing filename/label/fold CSV while changing the training seed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-mode", choices=["full", "foreground"], default="full")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--lr-backbone", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fold-index", type=int, action="append")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")
    return image.convert("RGB")


class GreenForegroundCrop:
    def __init__(self, training: bool, padding_ratio: float = 0.25):
        self.training = training
        self.padding_ratio = padding_ratio

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb_image = image_to_rgb(image)
        pixels = np.asarray(rgb_image, dtype=np.int16)
        red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        mask = (green > 45) & (green > red + 8) & (green > blue + 5)
        height, width = mask.shape
        if int(mask.sum()) < max(12, int(height * width * 0.0005)):
            return rgb_image

        row_threshold = max(2, int(width * 0.005))
        column_threshold = max(2, int(height * 0.005))
        valid_rows = np.flatnonzero(mask.sum(axis=1) >= row_threshold)
        valid_columns = np.flatnonzero(mask.sum(axis=0) >= column_threshold)
        if len(valid_rows) == 0 or len(valid_columns) == 0:
            return rgb_image

        top, bottom = int(valid_rows[0]), int(valid_rows[-1]) + 1
        left, right = int(valid_columns[0]), int(valid_columns[-1]) + 1
        box_height = bottom - top
        box_width = right - left
        if box_height <= 2 or box_width <= 2:
            return rgb_image

        padding_ratio = self.padding_ratio
        if self.training:
            padding_ratio = random.uniform(0.18, 0.38)
        vertical_padding = max(4, int(box_height * padding_ratio))
        horizontal_padding = max(4, int(box_width * padding_ratio))
        top = max(0, top - vertical_padding)
        bottom = min(height, bottom + vertical_padding)
        left = max(0, left - horizontal_padding)
        right = min(width, right + horizontal_padding)

        if self.training:
            vertical_jitter = int(0.04 * max(1, bottom - top))
            horizontal_jitter = int(0.04 * max(1, right - left))
            shift_y = random.randint(-vertical_jitter, vertical_jitter) if vertical_jitter else 0
            shift_x = random.randint(-horizontal_jitter, horizontal_jitter) if horizontal_jitter else 0
            top = max(0, min(height - 2, top + shift_y))
            bottom = max(top + 2, min(height, bottom + shift_y))
            left = max(0, min(width - 2, left + shift_x))
            right = max(left + 2, min(width, right + shift_x))
        return rgb_image.crop((left, top, right, bottom))


class PlantDataset(Dataset):
    def __init__(self, records: Sequence[ImageRecord], transform: transforms.Compose):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image_to_rgb(image))
        return tensor, record.label, record.path.name


def discover_training_records(train_root: Path) -> tuple[list[ImageRecord], list[str]]:
    class_names = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if not class_names:
        raise RuntimeError(f"No class folders found below {train_root}")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    records: list[ImageRecord] = []
    for class_name in class_names:
        class_dir = train_root / class_name
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                records.append(ImageRecord(path, class_to_index[class_name], class_name))
    return records, class_names


def discover_test_records(
    test_root: Path, submission_template: Path | None
) -> list[ImageRecord]:
    if submission_template is not None:
        with submission_template.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "ID" not in rows[0]:
            raise RuntimeError("Submission template must contain an ID column.")
        names = [row["ID"] for row in rows]
    else:
        names = sorted(
            path.name
            for path in test_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    records = []
    for name in names:
        path = test_root / name
        if not path.exists():
            raise FileNotFoundError(f"Submission ID does not exist in test folder: {path}")
        records.append(ImageRecord(path, -1, ""))
    return records


def make_folds(records: Sequence[ImageRecord], folds: int, seed: int) -> np.ndarray:
    if folds < 2:
        raise ValueError("At least two folds are required.")
    assignments = np.full(len(records), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_label[record.label].append(index)
    for indices in by_label.values():
        shuffled = np.asarray(indices, dtype=np.int64)
        rng.shuffle(shuffled)
        for position, index in enumerate(shuffled):
            assignments[index] = position % folds
    if np.any(assignments < 0):
        raise RuntimeError("Fold assignment failed.")
    return assignments


def load_reference_folds(path: Path, records: Sequence[ImageRecord], folds: int) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(records) or len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Reference fold CSV is incomplete or has duplicate filenames")
    lookup = {row["file"]: row for row in rows}
    if set(lookup) != {record.path.name for record in records}:
        raise ValueError("Reference folds do not match the current dataset")
    assignments = []
    for record in records:
        row = lookup[record.path.name]
        if row["class"] != record.class_name or int(row["label"]) != record.label:
            raise ValueError(f"Reference label mismatch: {record.path.name}")
        assignments.append(int(row["fold"]))
    result = np.asarray(assignments, dtype=np.int64)
    if set(result.tolist()) != set(range(folds)):
        raise ValueError("Reference folds are not the requested contiguous fold IDs")
    return result


def build_transforms(
    image_size: int, crop_mode: str
) -> tuple[transforms.Compose, transforms.Compose]:
    train_prefix = [GreenForegroundCrop(training=True)] if crop_mode == "foreground" else []
    eval_prefix = [GreenForegroundCrop(training=False)] if crop_mode == "foreground" else []
    random_crop_scale = (0.85, 1.0) if crop_mode == "foreground" else (0.75, 1.0)
    train_transform = transforms.Compose(
        train_prefix
        + [
            transforms.RandomResizedCrop(
                image_size,
                scale=random_crop_scale,
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        eval_prefix
        + [
            transforms.Resize(
                (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(
    architecture: str,
    initialization: str,
    num_classes: int,
    source_checkpoint: Path | None,
) -> tuple[nn.Module, nn.Module, dict]:
    if initialization == "plantdoctor" and architecture != "resnet18":
        raise ValueError("PlantDoctor initialization is only available for resnet18.")
    if initialization == "plantdoctor" and source_checkpoint is None:
        raise ValueError("--source-checkpoint is required for PlantDoctor initialization.")

    weights = None
    if initialization == "imagenet":
        weights = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
            "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
            "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        }[architecture]

    constructors = {
        "resnet18": models.resnet18,
        "mobilenet_v3_large": models.mobilenet_v3_large,
        "efficientnet_b0": models.efficientnet_b0,
        "convnext_tiny": models.convnext_tiny,
    }
    model = constructors[architecture](weights=weights)
    source_metadata: dict = {"initialization": initialization}

    if initialization == "plantdoctor":
        checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        state_dict = dict(checkpoint["model"])
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        source_metadata.update(
            {
                "source_checkpoint": str(source_checkpoint.resolve()),
                "source_classes": checkpoint.get("class_names", []),
                "source_val_acc": checkpoint.get("val_acc"),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
            }
        )

    if architecture == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head = model.fc
    elif architecture == "mobilenet_v3_large":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head = model.classifier
    elif architecture == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head = model.classifier
    else:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head = model.classifier
    return model, head, source_metadata


def set_backbone_trainable(model: nn.Module, head: nn.Module, trainable: bool) -> None:
    head_parameter_ids = {id(parameter) for parameter in head.parameters()}
    for parameter in model.parameters():
        if id(parameter) not in head_parameter_ids:
            parameter.requires_grad = trainable
    for parameter in head.parameters():
        parameter.requires_grad = True


def split_parameter_groups(
    model: nn.Module, head: nn.Module
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    head_parameter_ids = {id(parameter) for parameter in head.parameters()}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_parameter_ids
    ]
    head_parameters = list(head.parameters())
    return backbone_parameters, head_parameters


def make_loader(
    records: Sequence[ImageRecord],
    transform: transforms.Compose,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        PlantDataset(records, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    backbone_frozen: bool,
) -> tuple[float, float]:
    model.train()
    if backbone_frozen:
        model.eval()
        head.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    probability_batches: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    for images, batch_labels, batch_names in loader:
        images = images.to(device, non_blocking=True)
        views = [images]
        if tta:
            views.extend(
                [
                    torch.flip(images, dims=(-1,)),
                    torch.flip(images, dims=(-2,)),
                    torch.rot90(images, 1, dims=(-2, -1)),
                ]
            )
        logits_sum = None
        for view in views:
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(view)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        probabilities = torch.softmax(logits_sum / len(views), dim=1)
        probability_batches.append(probabilities.float().cpu().numpy())
        labels.append(batch_labels.numpy())
        names.extend(batch_names)
    return np.concatenate(probability_batches), np.concatenate(labels), names


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, class_names: Sequence[str]
) -> dict:
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(labels, predictions, strict=True):
        confusion[int(target), int(prediction)] += 1

    per_class = {}
    f1_scores = []
    for index, class_name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def write_fold_assignments(
    output_path: Path, records: Sequence[ImageRecord], assignments: np.ndarray
) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "class", "label", "fold"])
        for record, fold in zip(records, assignments, strict=True):
            writer.writerow([record.path.name, record.class_name, record.label, int(fold)])


def save_prediction_table(
    output_path: Path,
    names: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "true", "predicted", *[f"p_{name}" for name in class_names]])
        predictions = probabilities.argmax(axis=1)
        for name, label, prediction, row in zip(
            names, labels, predictions, probabilities, strict=True
        ):
            writer.writerow(
                [name, class_names[int(label)], class_names[int(prediction)], *[float(x) for x in row]]
            )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_root = args.data_root.resolve()
    train_root = data_root / "train"
    test_root = data_root / "test"
    output_root = args.output_dir.resolve()
    crop_suffix = "" if args.crop_mode == "full" else f"_{args.crop_mode}"
    run_name = (
        f"{args.architecture}_{args.initialization}_{args.strategy.replace('-', '_')}"
        f"{crop_suffix}"
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_records, class_names = discover_training_records(train_root)
    test_records = discover_test_records(test_root, args.submission_template)
    assignments = (load_reference_folds(args.reference_folds.resolve(), train_records, args.folds)
                   if args.reference_folds else make_folds(train_records, args.folds, args.seed))
    write_fold_assignments(run_dir / "fold_assignments.csv", train_records, assignments)
    train_transform, eval_transform = build_transforms(args.image_size, args.crop_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    selected_folds = args.fold_index if args.fold_index else list(range(args.folds))
    if any(fold < 0 or fold >= args.folds for fold in selected_folds):
        raise ValueError(f"Fold indices must be in [0, {args.folds - 1}].")

    run_config = vars(args).copy()
    for key, value in list(run_config.items()):
        if isinstance(value, Path):
            run_config[key] = str(value.resolve())
    run_config.update(
        {
            "class_names": class_names,
            "train_count": len(train_records),
            "test_count": len(test_records),
            "device": str(device),
            "torch_version": torch.__version__,
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    oof_probabilities = np.full((len(train_records), len(class_names)), np.nan, dtype=np.float32)
    fold_metrics = []
    test_probability_sum = np.zeros((len(test_records), len(class_names)), dtype=np.float64)

    print(
        json.dumps(
            {
                "run": run_name,
                "device": str(device),
                "classes": class_names,
                "selected_folds": selected_folds,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for fold in selected_folds:
        fold_start = time.time()
        set_seed(args.seed + fold)
        train_indices = np.flatnonzero(assignments != fold)
        validation_indices = np.flatnonzero(assignments == fold)
        fold_train_records = [train_records[index] for index in train_indices]
        fold_validation_records = [train_records[index] for index in validation_indices]

        train_loader = make_loader(
            fold_train_records,
            train_transform,
            args.batch_size,
            args.workers,
            True,
            args.seed + fold,
        )
        validation_loader = make_loader(
            fold_validation_records,
            eval_transform,
            args.batch_size,
            args.workers,
            False,
            args.seed + fold,
        )
        test_loader = make_loader(
            test_records,
            eval_transform,
            args.batch_size,
            args.workers,
            False,
            args.seed + fold,
        )

        model, head, source_metadata = build_model(
            args.architecture,
            args.initialization,
            len(class_names),
            args.source_checkpoint,
        )
        model.to(device)

        always_frozen = args.strategy == "linear-probe"
        set_backbone_trainable(model, head, trainable=False)
        backbone_parameters, head_parameters = split_parameter_groups(model, head)
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": args.lr_backbone},
                {"params": head_parameters, "lr": args.lr_head},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        best_macro_f1 = -math.inf
        best_epoch = -1
        epochs_without_improvement = 0
        history = []
        fold_checkpoint = run_dir / f"fold_{fold}_best.pt"

        for epoch in range(args.epochs):
            if not always_frozen and epoch == args.freeze_epochs:
                set_backbone_trainable(model, head, trainable=True)
            backbone_frozen = always_frozen or epoch < args.freeze_epochs
            train_loss, train_accuracy = train_one_epoch(
                model,
                head,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                amp_enabled,
                backbone_frozen,
            )
            validation_probabilities, validation_labels, _ = predict(
                model, validation_loader, device, amp_enabled, tta=False
            )
            validation_predictions = validation_probabilities.argmax(axis=1)
            metrics = classification_metrics(
                validation_labels, validation_predictions, class_names
            )
            scheduler.step()
            epoch_log = {
                "epoch": epoch + 1,
                "backbone_frozen": backbone_frozen,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "lr_backbone": optimizer.param_groups[0]["lr"],
                "lr_head": optimizer.param_groups[1]["lr"],
            }
            history.append(epoch_log)
            print(
                f"fold={fold} epoch={epoch + 1:02d} "
                f"loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_acc={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f} "
                f"frozen={backbone_frozen}",
                flush=True,
            )

            if metrics["macro_f1"] > best_macro_f1 + 1e-6:
                best_macro_f1 = metrics["macro_f1"]
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "class_names": class_names,
                        "architecture": args.architecture,
                        "fold": fold,
                        "best_epoch": best_epoch,
                        "best_macro_f1": best_macro_f1,
                        "source": source_metadata,
                    },
                    fold_checkpoint,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"fold={fold} early_stop epoch={epoch + 1}", flush=True)
                    break

        checkpoint = torch.load(fold_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        validation_probabilities, validation_labels, validation_names = predict(
            model, validation_loader, device, amp_enabled, tta=False
        )
        validation_predictions = validation_probabilities.argmax(axis=1)
        metrics = classification_metrics(validation_labels, validation_predictions, class_names)
        oof_probabilities[validation_indices] = validation_probabilities
        test_probabilities, _, _ = predict(
            model, test_loader, device, amp_enabled, tta=args.tta
        )
        test_probability_sum += test_probabilities
        metrics.update(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "elapsed_seconds": time.time() - fold_start,
                "history": history,
                "validation_files": validation_names,
            }
        )
        fold_metrics.append(metrics)
        (run_dir / f"fold_{fold}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"fold={fold} complete best_epoch={best_epoch} "
            f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        del model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    completed_mask = ~np.isnan(oof_probabilities).any(axis=1)
    completed_indices = np.flatnonzero(completed_mask)
    completed_labels = np.asarray([train_records[index].label for index in completed_indices])
    completed_probabilities = oof_probabilities[completed_indices]
    completed_predictions = completed_probabilities.argmax(axis=1)
    overall_metrics = classification_metrics(
        completed_labels, completed_predictions, class_names
    )
    overall_metrics.update(
        {
            "run": run_name,
            "completed_folds": selected_folds,
            "fold_accuracy_mean": float(np.mean([item["accuracy"] for item in fold_metrics])),
            "fold_accuracy_std": float(np.std([item["accuracy"] for item in fold_metrics])),
            "fold_macro_f1_mean": float(np.mean([item["macro_f1"] for item in fold_metrics])),
            "fold_macro_f1_std": float(np.std([item["macro_f1"] for item in fold_metrics])),
            "fold_metrics": [
                {
                    "fold": item["fold"],
                    "accuracy": item["accuracy"],
                    "macro_f1": item["macro_f1"],
                    "best_epoch": item["best_epoch"],
                    "elapsed_seconds": item["elapsed_seconds"],
                }
                for item in fold_metrics
            ],
        }
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(overall_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_prediction_table(
        run_dir / "oof_predictions.csv",
        [train_records[index].path.name for index in completed_indices],
        completed_labels,
        completed_probabilities,
        class_names,
    )

    averaged_test_probabilities = test_probability_sum / len(selected_folds)
    test_predictions = averaged_test_probabilities.argmax(axis=1)
    submission_path = run_dir / "submission.csv"
    with submission_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Category"])
        for record, prediction in zip(test_records, test_predictions, strict=True):
            writer.writerow([record.path.name, class_names[int(prediction)]])

    with (run_dir / "test_probabilities.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Category", *[f"p_{name}" for name in class_names]])
        for record, prediction, row in zip(
            test_records, test_predictions, averaged_test_probabilities, strict=True
        ):
            writer.writerow(
                [record.path.name, class_names[int(prediction)], *[float(x) for x in row]]
            )

    print(json.dumps(overall_metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"submission={submission_path}", flush=True)


if __name__ == "__main__":
    main()
