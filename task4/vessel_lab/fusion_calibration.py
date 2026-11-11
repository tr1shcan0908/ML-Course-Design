"""Calibrate the public-tested v8 fusion on a small, frozen OOF grid."""

from pathlib import Path
import json

import numpy as np
from PIL import Image

from vessel.metrics import binary_metrics
from .submission_domain import dump, export_masks, script_mask, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/dual_domain_v8"
    out = root / "optimization_v9"
    out.mkdir(exist_ok=True)
    plan = json.loads((root / "experiments/fusion_calibration_v9.json").read_text(encoding="utf-8"))
    with np.load(run / "v3_transport_oof.npz", allow_pickle=False) as saved:
        ids, old = saved["ids"].tolist(), saved["probabilities"].copy()
    with np.load(run / "oof_predictions.npz", allow_pickle=False) as saved:
        order = [saved["ids"].tolist().index(i) for i in ids]
        new = saved["probabilities"][order, 1]
    labels = [script_mask(np.asarray(Image.open(root / "train/label" / f"{i}.jpg")) > 0) for i in ids]
    folds = json.loads((root / "experiments/folds_v1.json").read_text(encoding="utf-8"))["folds"]
    rows = []
    for weight in plan["candidate_weights"]:
        probabilities = (1 - weight) * old + weight * new
        for threshold in plan["thresholds"]:
            per_image = [dict(id=i, **binary_metrics(p >= threshold, y)) for i, p, y in zip(ids, probabilities, labels)]
            rows.append(dict(candidate_weight=weight, threshold=threshold, per_image=per_image,
                             mean_dice=float(np.mean([r["dice"] for r in per_image]))))
    baseline, = [r for r in rows if r["candidate_weight"] == plan["baseline_weight"] and r["threshold"] == plan["baseline_threshold"]]
    original = json.loads((run / "fusion_results.json").read_text(encoding="utf-8"))
    reference, = [r for r in original["results"] if r["candidate_weight"] == plan["baseline_weight"]]
    if baseline["mean_dice"] != reference["mean_dice"]:
        raise ValueError("Baseline fusion did not replay exactly")
    for row in rows:
        paired = {a["id"]: a["dice"] - b["dice"] for a, b in zip(row["per_image"], baseline["per_image"])}
        row["gain"] = row["mean_dice"] - baseline["mean_dice"]
        row["gain_by_fold"] = {str(f["fold"]): float(np.mean([paired[i] for i in f["validation_ids"]])) for f in folds}
    crossfit, selections = [], []
    for fold in folds:
        held = set(fold["validation_ids"])
        selected = max(rows, key=lambda row: (
            np.mean([p["dice"] for p in row["per_image"] if p["id"] not in held]),
            -abs(row["threshold"] - plan["baseline_threshold"]),
            -abs(row["candidate_weight"] - plan["baseline_weight"])))
        crossfit.extend(p for p in selected["per_image"] if p["id"] in held)
        selections.append(dict(fold=fold["fold"], weight=selected["candidate_weight"], threshold=selected["threshold"]))
    crossfit_dice = float(np.mean([r["dice"] for r in crossfit]))
    best = max(rows, key=lambda row: row["mean_dice"])
    promote = (best["gain"] >= plan["minimum_gain"] and
               crossfit_dice - baseline["mean_dice"] >= plan["minimum_crossfit_gain"] and
               sum(g >= 0 for g in best["gain_by_fold"].values()) >= plan["minimum_nonnegative_folds"])
    result = dict(plan=plan, results=rows, selected=best, promoted=promote,
                  crossfit=dict(mean_dice=crossfit_dice, gain=crossfit_dice-baseline["mean_dice"], choices=selections, per_image=crossfit),
                  code_sha256=sha256(Path(__file__)), plan_sha256=sha256(root / "experiments/fusion_calibration_v9.json"),
                  inputs={name:sha256(run/name) for name in ("v3_transport_oof.npz", "oof_predictions.npz", "v3_transport_test.npz", "test_predictions.npz")})
    dump(out / "results.json", result)
    if promote:
        with np.load(run / "v3_transport_test.npz", allow_pickle=False) as saved:
            test_ids, a = saved["ids"].tolist(), saved["probabilities"].copy()
        with np.load(run / "test_predictions.npz", allow_pickle=False) as saved:
            order = [saved["ids"].tolist().index(i) for i in test_ids]
            b = saved["probabilities"][order, 1]
        weight = best["candidate_weight"]
        directory = root / "submission_v9_calibrated"
        checks = export_masks(root, directory, test_ids, ((1-weight)*a+weight*b) >= best["threshold"])
        recipe = json.loads((root / "submission_v8_fusion/inference_recipe.json").read_text(encoding="utf-8"))
        recipe.update(candidate_weight=weight, threshold=best["threshold"], csv_sha256=checks["csv_sha256"],
                      calibration_result_sha256=sha256(out / "results.json"))
        dump(directory / "inference_recipe.json", recipe)
        (root / "submission_v9_calibrated.csv").write_bytes((directory / "submission.csv").read_bytes())
    print(json.dumps(dict(grid=[{k:r[k] for k in ("candidate_weight","threshold","mean_dice","gain")} for r in rows],
                          crossfit_dice=crossfit_dice, choices=selections, promoted=promote)), flush=True)


if __name__ == "__main__":
    main()
