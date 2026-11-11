"""Explicit submission-grid transforms matching the supplied competition script.

The platform implementation is not visible. Keep this convention separate from
nearest-neighbor/native metrics, and record the user's observed platform score.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.prepare_submission import read_official_encoder, resize_probabilities, validate_csv
from vessel.metrics import binary_metrics
from vessel.rle import encode


def script_mask(mask: np.ndarray, size=(512, 512)) -> np.ndarray:
    """Pillow uint8 bilinear resize, then >0, exactly as turn_to_str does."""
    pixels = np.asarray(mask, dtype=np.uint8) * 255
    return np.asarray(Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR)) > 0


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export_masks(root, directory, ids, masks, native_masks=None):
    """Write, decode, and independently verify against the supplied encoder."""
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / "masks"
    png.mkdir(exist_ok=True)
    ids = [str(i) for i in ids]
    if len(ids) != len(masks) or len(ids) != len(set(ids)):
        raise ValueError("Submission image IDs must be unique and match mask count")
    with (root / "sample_submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        expected = [row["Id"] for row in csv.DictReader(stream)]
    if ids != expected:
        raise ValueError("Submission IDs differ from the sample order")
    rows = []
    for image_id, mask in zip(ids, masks):
        if np.asarray(mask).shape != (512, 512):
            raise ValueError("Expected a 512 by 512 mask")
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(png / f"{image_id}.png")
        rows.append({"Id": image_id, "Predicted": encode(mask)})
    csv_path = directory / "submission.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Id", "Predicted"])
        writer.writeheader()
        writer.writerows(rows)
    report = validate_csv(csv_path, png, ids, (512, 512))
    paths = [png / f"{image_id}.png" for image_id in ids]
    if native_masks is not None:
        if len(native_masks) != len(ids):
            raise ValueError("Native mask count mismatch")
        source = directory / "native_masks"
        source.mkdir(exist_ok=True)
        paths = []
        for image_id, mask in zip(ids, native_masks):
            path = source / f"{image_id}.png"
            Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)
            paths.append(path)
    official = read_official_encoder(root / "segmentation_to_csv.py")(paths)
    if official != [row["Predicted"] for row in rows]:
        raise ValueError("The supplied competition encoder produces different RLE")
    report["exact_supplied_encoder_match"] = True
    report["encoder_input"] = "native_binary_masks" if native_masks is not None else "512_binary_masks"
    dump(directory / "submission_checks.json", report)
    return report


def audit_v3(root):
    out = root / "submission_v3_script_compatible"
    out.mkdir(exist_ok=True)
    with np.load(root / "submission_v3/oof_predictions.npz", allow_pickle=False) as saved:
        ids, probabilities = saved["ids"].tolist(), saved["probabilities"].copy()
    labels = [np.asarray(Image.open(root / "train/label" / f"{i}.jpg").convert("L")) > 0 for i in ids]
    target_script = np.stack([script_mask(y) for y in labels])
    target_nearest = np.stack([np.asarray(Image.fromarray(y).resize((512, 512), Image.Resampling.NEAREST)) for y in labels])
    current = resize_probabilities(probabilities, (512, 512)) >= .54
    compatible = np.stack([script_mask(p >= .54) for p in probabilities])
    results = {}
    for name, predictions in (("current", current), ("script_compatible", compatible)):
        results[name] = {}
        for convention, targets in (("nearest", target_nearest), ("supplied_script", target_script)):
            rows = [dict(id=i, **binary_metrics(p, y)) for i, p, y in zip(ids, predictions, targets)]
            results[name][convention] = dict(mean_dice=float(np.mean([r["dice"] for r in rows])), per_image=rows)
    with np.load(root / "submission_v3/test_probabilities_native.npz", allow_pickle=False) as saved:
        test_ids, test_p = saved["ids"].tolist(), saved["probabilities"].copy()
    native = test_p >= .54
    masks = np.stack([script_mask(p) for p in native])
    checks = export_masks(root, out, test_ids, masks, native)
    recorded_sha = json.loads((root / "submission_v3/submission_checks.json").read_text(encoding="utf-8"))["csv_sha256"]
    if sha256(root / "submission_v3.csv") != recorded_sha:
        raise ValueError("Root submission_v3.csv differs from its archived export")
    result = dict(user_reported_platform_score=.75625, user_reported_file="submission_v3.csv",
                  platform_label_transform_confirmed=False,
                  interpretation="Strong local evidence for a label-transform mismatch; requires platform retest.",
                  oof_metrics=results, threshold=.54, model_retrained=False,
                  original_csv_sha256=recorded_sha, new_csv_sha256=checks["csv_sha256"],
                  input_sha256={str(p): sha256(root / p) for p in (
                      "submission_v3/oof_predictions.npz", "submission_v3/test_probabilities_native.npz",
                      "submission_v3/inference_recipe.json", "segmentation_to_csv.py")})
    dump(out / "audit.json", result)
    (root / "submission_v3_script_compatible.csv").write_bytes((out / "submission.csv").read_bytes())
    print(json.dumps({k: {m: v["mean_dice"] for m, v in values.items()} for k, values in results.items()}), flush=True)
    return result


if __name__ == "__main__":
    audit_v3(Path(__file__).resolve().parents[1])
