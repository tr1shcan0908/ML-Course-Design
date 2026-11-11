"""Original-image folds, native-resolution sampling, paired augmentation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


@dataclass
class Sample:
    image_id: str
    image: torch.Tensor
    mask: torch.Tensor | None
    original_hw: tuple[int, int]


def verify_dataset(root: Path) -> str:
    manifest = json.loads((root / "experiments/dataset_manifest.json").read_text(encoding="utf-8"))
    payload = {"records": manifest["records"], "original_inputs": manifest["original_inputs"]}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest["dataset_sha256"]:
        raise ValueError("Dataset manifest content does not match its fingerprint")
    for row in manifest["records"]:
        actual = hashlib.sha256((root / row["path"]).read_bytes()).hexdigest()
        if actual != row["file_sha256"]:
            raise ValueError(f"Dataset file changed: {row['path']}")
    for name, expected in manifest["original_inputs"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Original competition file changed: {name}")
    return digest


def get_fold(root: Path, index: int, dataset_hash: str) -> dict:
    document = json.loads((root / "experiments/folds_v1.json").read_text(encoding="utf-8"))
    if document["dataset_sha256"] != dataset_hash:
        raise ValueError("Folds refer to a different dataset")
    fold = next(f for f in document["folds"] if f["fold"] == index)
    if set(fold["train_ids"]) & set(fold["validation_ids"]):
        raise ValueError("Training/validation original images overlap")
    return fold


def load_samples(root: Path, image_ids: list[str], pad_hw: list[int], split: str = "train") -> list[Sample]:
    samples = []
    for image_id in image_ids:
        with Image.open(root / split / "image" / f"{image_id}.jpg") as source:
            rgb = np.array(source.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        if h > pad_hw[0] or w > pad_hw[1]:
            raise ValueError("Padding target cannot be smaller than the original image")
        padding = (0, pad_hw[1] - w, 0, pad_hw[0] - h)
        image = F.pad(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255, padding)
        mask = None
        if split == "train":
            with Image.open(root / "train/label" / f"{image_id}.jpg") as source:
                label = np.array(source.convert("L"), dtype=np.uint8)
            if label.shape != (h, w) or not set(np.unique(label).tolist()).issubset({0, 255}):
                raise ValueError(f"Invalid paired binary label: {image_id}")
            mask = F.pad(torch.from_numpy((label == 255).astype(np.float32))[None], padding)
        samples.append(Sample(image_id, image.contiguous(), mask, (h, w)))
    return samples


def sample_patches(samples: list[Sample], batch_size: int, patch_hw: list[int], augmentation: dict) -> tuple[torch.Tensor, torch.Tensor]:
    images, masks = [], []
    ph, pw = patch_hw
    if ph != pw and augmentation["rotate90"]:
        raise ValueError("Random 90-degree rotation requires square patches")
    for _ in range(batch_size):
        sample = random.choice(samples)
        if sample.mask is None:
            raise ValueError("A training sample must have a mask")
        h, w = sample.image.shape[-2:]
        top, left = random.randint(0, h - ph), random.randint(0, w - pw)
        image = sample.image[:, top : top + ph, left : left + pw]
        mask = sample.mask[:, top : top + ph, left : left + pw]
        if random.random() < augmentation["hflip"]:
            image, mask = image.flip(2), mask.flip(2)
        if random.random() < augmentation["vflip"]:
            image, mask = image.flip(1), mask.flip(1)
        if augmentation["rotate90"]:
            turns = random.randrange(4)
            image, mask = torch.rot90(image, turns, (1, 2)), torch.rot90(mask, turns, (1, 2))
        images.append(image)
        masks.append(mask)
    return torch.stack(images), torch.stack(masks)


def photometric(images: torch.Tensor, config: dict) -> torch.Tensor:
    def coefficient(name: str) -> torch.Tensor:
        low, high = config[name]
        return torch.empty((len(images), 1, 1, 1), device=images.device).uniform_(low, high)
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    transformed = ((images - mean) * coefficient("contrast") + mean) * coefficient("brightness")
    transformed = transformed.clamp(0, 1).pow(coefficient("gamma"))
    transformed = transformed + torch.randn_like(transformed) * coefficient("noise_std")
    return transformed.clamp(0, 1)
