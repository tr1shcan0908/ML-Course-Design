"""Foreground-only segmentation metrics, with explicit empty-mask behavior."""

import numpy as np
import torch
from torch.nn import functional as F


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, weights: dict) -> torch.Tensor:
    logits, target = logits.float(), target.float()
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return weights["bce"] * F.binary_cross_entropy_with_logits(logits, target) + weights["dice"] * (1 - dice)


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("Metric mask dimensions differ")
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_probabilities(probabilities: list[np.ndarray], targets: list[np.ndarray], image_ids: list[str]) -> dict:
    fixed = [dict(id=image_id, **binary_metrics(p >= 0.5, y)) for image_id, p, y in zip(image_ids, probabilities, targets)]
    grid = []
    for threshold in np.round(np.arange(0.2, 0.801, 0.02), 2):
        score = float(np.mean([binary_metrics(p >= threshold, y)["dice"] for p, y in zip(probabilities, targets)]))
        grid.append({"threshold": float(threshold), "mean_dice": score})
    best = max(grid, key=lambda r: (r["mean_dice"], -abs(r["threshold"] - 0.5)))
    tp, fp, fn = (sum(r[key] for r in fixed) for key in ("tp", "fp", "fn"))
    return {
        "mean_dice_at_0_5": float(np.mean([r["dice"] for r in fixed])),
        "std_dice_at_0_5": float(np.std([r["dice"] for r in fixed])),
        "global_dice_at_0_5": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "per_image": fixed,
        "calibrated_threshold": best["threshold"],
        "calibrated_mean_dice": best["mean_dice"],
        "calibration_note": "Selected on these validation images; not an independent test score",
        "threshold_grid": grid,
    }
