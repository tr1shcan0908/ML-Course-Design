"""Geometry, fixed grouped folds, and low-memory image datasets."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

cv2.setNumThreads(0)
ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_rgb(path: Path) -> np.ndarray:
    buffer = np.frombuffer(path.read_bytes(), np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 97])
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def homogeneous(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([matrix, [0, 0, 1]]) if matrix.shape == (2, 3) else matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact cv2.resize half-pixel transform followed by integer padding."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    rw, rh = round(width * scale), round(height * scale)
    sx, sy = rw / width, rh / height
    px, py = (size - rw) // 2, (size - rh) // 2
    small = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA)
    result = np.zeros((size, size, 3), np.uint8)
    result[py:py + rh, px:px + rw] = small
    matrix = np.array([[sx, 0, px + (sx - 1) / 2],
                       [0, sy, py + (sy - 1) / 2], [0, 0, 1]], np.float64)
    return result, matrix


def crop_matrix(center: np.ndarray, extent: float, size: int) -> np.ndarray:
    scale = size / extent
    middle = (size - 1) / 2
    return np.array([[scale, 0, middle - scale * center[0]],
                     [0, scale, middle - scale * center[1]], [0, 0, 1]], np.float64)


def warp(image: np.ndarray, matrix: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpAffine(image, matrix[:2], (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def training_augmentation(size: int, fine: bool) -> np.ndarray:
    center = ((size - 1) / 2, (size - 1) / 2)
    angle = np.random.uniform(-15, 15)
    scale = 1.0 if fine else np.random.uniform(0.9, 1.1)
    matrix = homogeneous(cv2.getRotationMatrix2D(center, angle, scale))
    if not fine:
        matrix[:2, 2] += np.random.uniform(-0.035, 0.035, 2) * size
    if np.random.random() < 0.5:
        flip = np.array([[-1, 0, size - 1], [0, 1, 0], [0, 0, 1]], float)
        matrix = flip @ matrix
    return matrix


def color_augmentation(image: np.ndarray) -> np.ndarray:
    # Mild image-wide changes preserve the location and avoid artificial landmarks.
    image = image.astype(np.float32) / 255
    gamma = np.random.uniform(0.85, 1.15)
    contrast = np.random.uniform(0.85, 1.15)
    channel = np.random.uniform(0.94, 1.06, (1, 1, 3))
    image = np.clip((image ** gamma) * contrast * channel, 0, 1)
    return image


def prepare(size: int = 512, roi: int = 768, seed: int = 2026) -> dict:
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    with (ROOT / "fovea_localization_train_GT.csv").open(encoding="utf-8-sig") as f:
        labels = {int(r["data"]): [float(r["Fovea_X"]), float(r["Fovea_Y"])]
                  for r in csv.DictReader(f)}
    records = []
    for split in ("train", "test"):
        for path in sorted((ROOT / split).glob("*.jpg")):
            image = read_rgb(path)
            height, width = image.shape[:2]
            record = dict(id=int(path.stem), path=path.relative_to(ROOT).as_posix(),
                          split=split, width=width, height=height,
                          group=hashlib.sha256(path.read_bytes()).hexdigest(), fold=-1)
            small, matrix = letterbox(image, size)
            cached = output / f"coarse_{size}" / f"{split}_{path.stem}.jpg"
            if not cached.exists():
                save_rgb(cached, small)
            record.update(cache=cached.relative_to(ROOT).as_posix(), transform=matrix.tolist())
            if split == "train":
                point = np.array(labels[record["id"]], float)
                if np.all(point == 0):
                    raise ValueError("This training experiment expects visible-only input. Add an explicit visibility model before including zero labels.")
                xml = ET.parse(ROOT / "train_location" / f"{path.stem}.xml").getroot()
                box = [float(xml.findtext(f"object/bndbox/{key}"))
                       for key in ("xmin", "ymin", "xmax", "ymax")]
                extent = 2 * roi
                fine_matrix = crop_matrix(point, extent, extent)
                fine = output / f"fine_context_{roi}" / f"{path.stem}.jpg"
                if not fine.exists():
                    save_rgb(fine, warp(image, fine_matrix, extent))
                record.update(point=point.tolist(), box=box,
                              fine_cache=fine.relative_to(ROOT).as_posix(),
                              fine_transform=fine_matrix.tolist())
            records.append(record)
    train = [r for r in records if r["split"] == "train"]
    unique = list({r["group"]: r for r in train}.values())
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    strata = [f'{r["width"]}x{r["height"]}' for r in unique]
    for fold, (_, indices) in enumerate(splitter.split(unique, strata)):
        groups = {unique[i]["group"] for i in indices}
        for record in train:
            if record["group"] in groups:
                record["fold"] = fold
    # Match the already established audit folds when present, to preserve comparisons.
    existing = ROOT / "analysis" / "baseline_oof.csv"
    if existing.exists():
        with existing.open(encoding="utf-8-sig") as f:
            established = {int(r["id"]): int(r["fold"]) for r in csv.DictReader(f)}
        if set(established) != {r["id"] for r in train}:
            raise ValueError("Audit fold IDs do not match supplied training data")
        for record in train:
            record["fold"] = established[record["id"]]
    for fold in range(5):
        fit_groups = {r["group"] for r in train if r["fold"] != fold}
        val_groups = {r["group"] for r in train if r["fold"] == fold}
        if fit_groups & val_groups:
            raise AssertionError("Duplicate image crossed folds")
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    manifest = dict(size=size, roi=roi, seed=seed, dataset_hash=digest, records=records)
    manifest_path = output / f"manifest_{size}_{roi}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(manifest=str(manifest_path), train=len(train),
                          unique=len(unique), test=len(records)-len(train),
                          folds={f: sum(r["fold"] == f for r in train) for f in range(5)})), flush=True)
    return manifest


class FoveaDataset(Dataset):
    def __init__(self, rows: list[dict], size: int, stage: str = "coarse",
                 training: bool = False, roi: int = 768,
                 anchors: dict[int, np.ndarray] | None = None, auxiliary: bool = False):
        self.rows, self.size, self.stage = rows, size, stage
        self.training, self.roi, self.auxiliary = training, roi, auxiliary
        self.anchors = anchors
        self.fixed = {}
        if stage == "fine" and not training:
            if anchors is None:
                raise ValueError("Validation/test fine crops require predicted coarse anchors")
            for row in rows:
                center = np.array(anchors[row["id"]], float)
                matrix = crop_matrix(center, roi, size)
                self.fixed[row["id"]] = (warp(read_rgb(ROOT / row["path"]), matrix, size), matrix)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        point = np.array(row.get("point", [0, 0]), dtype=float)
        if self.stage == "coarse":
            image = read_rgb(ROOT / row["cache"])
            matrix = np.array(row["transform"], float)
            if image.shape[0] != self.size:
                raise ValueError("Coarse cache resolution differs from model input")
        elif self.training:
            image = read_rgb(ROOT / row["fine_cache"])
            base = np.array(row["fine_transform"], float)
            center = transform_points(point, base)
            center += np.random.uniform(-0.18, 0.18, 2) * self.roi
            extent = self.roi * np.random.uniform(0.85, 1.15)
            local = crop_matrix(center, extent, self.size)
            image = warp(image, local, self.size)
            matrix = local @ base
        else:
            image, matrix = self.fixed[row["id"]]
        if self.training:
            augmentation = training_augmentation(self.size, self.stage == "fine")
            image = warp(image, augmentation, self.size)
            matrix = augmentation @ matrix
            image = color_augmentation(image)
        else:
            image = image.astype(np.float32) / 255
        image = np.ascontiguousarray(((image - MEAN) / STD).transpose(2, 0, 1), dtype=np.float32)
        result = dict(image=torch.from_numpy(image),
                      point=torch.tensor(transform_points(point, matrix), dtype=torch.float32),
                      original=torch.tensor(point, dtype=torch.float32),
                      inverse=torch.tensor(np.linalg.inv(matrix)[:2], dtype=torch.float32),
                      id=row["id"], width=row["width"], height=row["height"])
        if self.auxiliary:
            a, b, c, d = row["box"]
            corners = transform_points(np.array([[a,b],[c,b],[c,d],[a,d]]), matrix)
            hm_size = self.size // 2
            polygon = np.rint((corners + 0.5) / 2 - 0.5).astype(np.int32)
            mask = np.zeros((hm_size, hm_size), np.float32)
            cv2.fillConvexPoly(mask, polygon, 1)
            result["region"] = torch.from_numpy(mask[None])
        return result

