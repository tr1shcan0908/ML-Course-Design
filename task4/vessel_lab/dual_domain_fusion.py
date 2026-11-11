"""Map v3 probabilities through the monotone submission transform before fusion.

The transform of hard masks is monotone in the threshold. Its generalized
inverse supplies a submission-domain score whose .5 mask exactly reproduces
the public-tested script-compatible v3. No labels enter this transformation.
"""

from pathlib import Path
import json

import numpy as np

from vessel.metrics import binary_metrics
from .submission_domain import script_mask, dump, export_masks, sha256


def transport(probabilities, threshold=.54):
    threshold = float(threshold)
    values = []
    for source in probabilities:
        score = np.zeros((512, 512), dtype=np.float32)
        for t in np.round(np.arange(.02, 1., .02), 2):
            score[script_mask(source >= float(t))] = float(t)
        score = (score + .01).clip(.001, .999)
        # Shift only the logit intercept so .54 in v3 becomes .5 here.
        odds = score / (1 - score) * ((1 - threshold) / threshold)
        calibrated = odds / (1 + odds)
        exact = script_mask(source >= threshold)
        if not np.array_equal(calibrated >= .5, exact):
            raise ValueError("Transport must preserve the deployed baseline mask exactly")
        values.append(calibrated)
    return np.stack(values)


def cached_transport(root, source_path, output):
    digest = sha256(source_path)
    if output.exists():
        with np.load(output, allow_pickle=False) as saved:
            if saved["source_sha256"].item() != digest:
                raise ValueError("Probability transport cache input changed")
            return saved["ids"].tolist(), saved["probabilities"].copy()
    with np.load(source_path, allow_pickle=False) as saved:
        ids = saved["ids"].tolist()
        values = transport(saved["probabilities"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.asarray(ids), probabilities=values, source_sha256=np.asarray(digest))
    return ids, values


def main():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "experiments/dual_domain_fusion_v8.json").read_text(encoding="utf-8"))
    run = root / "runs/dual_domain_v8"
    ids, base = cached_transport(root, root / "submission_v3/oof_predictions.npz", run / "v3_transport_oof.npz")
    with np.load(run / "oof_predictions.npz", allow_pickle=False) as archive:
        order = [archive["ids"].tolist().index(i) for i in ids]
        candidate = archive["probabilities"][order, 1]
    from PIL import Image
    targets = np.stack([script_mask(np.asarray(Image.open(root / "train/label" / f"{i}.jpg")) > 0) for i in ids])
    folds = json.loads((root / "experiments/folds_v1.json").read_text(encoding="utf-8"))["folds"]
    results = []
    for weight in plan["candidate_weights"]:
        p = (1 - weight) * base + weight * candidate
        rows = [dict(id=i, **binary_metrics(a >= .5, b)) for i, a, b in zip(ids, p, targets)]
        results.append(dict(candidate_weight=weight, mean_dice=float(np.mean([r["dice"] for r in rows])),
                            per_image=rows))
    baseline = results[0]
    for row in results:
        delta = {a["id"]: a["dice"] - b["dice"] for a, b in zip(row["per_image"], baseline["per_image"])}
        row["gain_by_fold"] = {str(f["fold"]): float(np.mean([delta[i] for i in f["validation_ids"]])) for f in folds}
        row["gain_over_baseline"] = row["mean_dice"] - baseline["mean_dice"]
        row["passes_gate"] = (row["gain_over_baseline"] >= plan["promotion_minimum_mean_gain"] and
                              sum(d >= 0 for d in row["gain_by_fold"].values()) >= plan["promotion_minimum_nonnegative_folds"])
    # Select weights on the other four validation folds, then inspect this fold.
    crossfit, selections = [], []
    for fold in folds:
        held = set(fold["validation_ids"])
        selected = max(results, key=lambda row: (np.mean([p["dice"] for p in row["per_image"] if p["id"] not in held]),
                                                -row["candidate_weight"]))
        crossfit.extend([p for p in selected["per_image"] if p["id"] in held])
        selections.append(dict(fold=fold["fold"], candidate_weight=selected["candidate_weight"]))
    cross_score = float(np.mean([p["dice"] for p in crossfit]))
    eligible = [r for r in results if r["passes_gate"]]
    winner = max(eligible, key=lambda row: row["mean_dice"]) if eligible and cross_score > baseline["mean_dice"] else baseline
    output = dict(plan=plan, results=results, selected_weight=winner["candidate_weight"],
                  crossfit_weight_selection=dict(mean_dice=cross_score, choices=selections, per_image=crossfit),
                  source_sha256=sha256(Path(__file__)), plan_sha256=sha256(root / "experiments/dual_domain_fusion_v8.json"))
    dump(run / "fusion_results.json", output)
    if winner["candidate_weight"]:
        test_ids, old = cached_transport(root, root / "submission_v3/test_probabilities_native.npz", run / "v3_transport_test.npz")
        with np.load(run / "test_predictions.npz", allow_pickle=False) as saved:
            order = [saved["ids"].tolist().index(i) for i in test_ids]
            new = saved["probabilities"][order, 1]
        weight = winner["candidate_weight"]
        destination = root / "submission_v8_fusion"
        checks = export_masks(root, destination, test_ids, ((1 - weight) * old + weight * new) >= .5)
        dump(destination / "inference_recipe.json", dict(candidate_weight=weight, threshold=.5,
             transport="vessel_lab.dual_domain_fusion.transport", script=output["source_sha256"],
             baseline_recipe_sha256=sha256(root / "submission_v3/inference_recipe.json"),
             candidate_recipe_sha256=sha256(root / "submission_v8_dual_domain/inference_recipe.json"),
             csv_sha256=checks["csv_sha256"]))
        (root / "submission_v8_fusion.csv").write_bytes((destination / "submission.csv").read_bytes())
    print(json.dumps(dict(scores=[{k:r[k] for k in ("candidate_weight", "mean_dice", "gain_over_baseline", "passes_gate")} for r in results],
                          crossfit_dice=cross_score, selected_weight=winner["candidate_weight"])), flush=True)


if __name__ == "__main__":
    main()
