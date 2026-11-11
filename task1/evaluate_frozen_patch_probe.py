"""Five-fold, training-fold-tuned BioCLIP 2 plant classification probes.

All C/kernel choices use inner folds from each outer training fold.  The outer
fold remains untouched until one final prediction per image is produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


# Weights sum to one so each variant has comparable cosine-kernel scale.
VARIANTS = {
    "center_global": {"center_global": 1.0},
    "center_patch": {"center_global": .5, "center_patch_mean": .5},
    "center_plant": {"center_global": .5, "center_patch_mean": .25, "center_patch_plant": .25},
    "letterbox_plant": {"letterbox_global": .5, "letterbox_patch_mean": .25,
                        "letterbox_patch_plant": .25},
    "dual_plant": {"center_global": .25, "center_patch_mean": .125,
                   "center_patch_plant": .125, "letterbox_global": .25,
                   "letterbox_patch_mean": .125, "letterbox_patch_plant": .125},
    "mid_plant": {"center_mid_global": .5, "center_mid_patch_mean": .25,
                  "center_mid_patch_plant": .25},
    "letterbox_mid_plant": {"letterbox_mid_global": .5,
                            "letterbox_mid_patch_mean": .25,
                            "letterbox_mid_patch_plant": .25},
    "dual_mid_plant": {"center_mid_global": .25, "center_mid_patch_mean": .125,
                       "center_mid_patch_plant": .125, "letterbox_mid_global": .25,
                       "letterbox_mid_patch_mean": .125,
                       "letterbox_mid_patch_plant": .125},
    "final_cls_mid_plant": {"center_global": .5, "center_mid_patch_mean": .25,
                            "center_mid_patch_plant": .25},
    "center_multilevel": {"center_global": .25, "center_patch_mean": .125,
                          "center_patch_plant": .125, "center_mid_global": .25,
                          "center_mid_patch_mean": .125, "center_mid_patch_plant": .125},
    "dual_multilevel": {"center_global": .125, "center_patch_mean": .0625,
                        "center_patch_plant": .0625, "center_mid_global": .125,
                        "center_mid_patch_mean": .0625, "center_mid_patch_plant": .0625,
                        "letterbox_global": .125, "letterbox_patch_mean": .0625,
                        "letterbox_patch_plant": .0625, "letterbox_mid_global": .125,
                        "letterbox_mid_patch_mean": .0625, "letterbox_mid_patch_plant": .0625},
}
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalized_blocks(archive, prefix: str, variant: str) -> np.ndarray:
    blocks = []
    for name, weight in VARIANTS[variant].items():
        features = np.asarray(archive[f"{prefix}_{name}"], dtype=np.float64)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if not np.isfinite(features).all() or (norms <= 0).any():
            raise ValueError(f"Invalid {prefix}_{name} features")
        blocks.append(math.sqrt(weight) * features / norms)
    result = np.concatenate(blocks, axis=1)
    if not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6):
        raise AssertionError("Weighted block normalization failed")
    return result


def gram(train_features: np.ndarray, test_features: np.ndarray, kind: str,
         gamma: float) -> tuple[np.ndarray, np.ndarray]:
    similarity = np.clip(train_features @ train_features.T, -1, 1)
    test_similarity = np.clip(test_features @ train_features.T, -1, 1)
    if kind == "linear":
        return similarity, test_similarity
    if kind == "rbf":
        return np.exp(-2 * gamma * (1 - similarity)), np.exp(-2 * gamma * (1 - test_similarity))
    raise ValueError(kind)


def choose_params(features: np.ndarray, labels: np.ndarray, outer_train: np.ndarray,
                  seed: int, c_values: list[float]) -> tuple[tuple[str, float, float], list[dict]]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = [(outer_train[a], outer_train[b]) for a, b in inner.split(features[outer_train], labels[outer_train])]
    candidates = []
    # Cache both kernels for each choice of geometry.
    for kind, gamma in (("linear", 0.0), ("rbf", 1.0)):
        full_kernel, _ = gram(features, features, kind, gamma)
        for c in c_values:
            predictions = np.full(len(labels), -1, dtype=np.int64)
            for inner_train, inner_val in splits:
                model = SVC(kernel="precomputed", C=c, probability=True, random_state=seed)
                model.fit(full_kernel[np.ix_(inner_train, inner_train)], labels[inner_train])
                probs = model.predict_proba(full_kernel[np.ix_(inner_val, inner_train)])
                predictions[inner_val] = model.classes_[probs.argmax(1)]
            inner_indices = np.concatenate([b for _, b in splits])
            accuracy = accuracy_score(labels[inner_indices], predictions[inner_indices])
            macro_f1 = f1_score(labels[inner_indices], predictions[inner_indices], average="macro")
            candidates.append({"kind": kind, "gamma": gamma, "C": c,
                               "accuracy": float(accuracy), "macro_f1": float(macro_f1)})
    # Prefer the simpler, more regularized model on ties.
    best = max(candidates, key=lambda row: (row["accuracy"], row["macro_f1"],
                                             row["kind"] == "linear", -row["C"]))
    return (best["kind"], best["gamma"], best["C"]), candidates


def save_table(path: Path, files: np.ndarray, labels: np.ndarray | None,
               probabilities: np.ndarray, classes: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "true", "predicted", *[f"p_{name}" for name in classes]]
                        if labels is not None else ["ID", "Category", *[f"p_{name}" for name in classes]])
        for i, (name, row) in enumerate(zip(files, probabilities, strict=True)):
            pred = classes[int(np.argmax(row))]
            writer.writerow([name, classes[int(labels[i])], pred, *map(float, row)]
                            if labels is not None else [name, pred, *map(float, row)])


def evaluate_variant(archive, variant: str, labels: np.ndarray, folds: np.ndarray,
                     train_files: np.ndarray, test_files: np.ndarray, classes: list[str],
                     output_dir: Path, template: list[str], c_values: list[float]) -> dict:
    train_features = normalized_blocks(archive, "train", variant)
    test_features = normalized_blocks(archive, "test", variant)
    oof = np.zeros((len(labels), len(classes)), dtype=np.float64)
    test_probs = np.zeros((len(test_files), len(classes)), dtype=np.float64)
    selected = []
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_dir / "fold_assignments.csv", variant_dir / "fold_assignments.csv")
    for fold in range(5):
        outer_train = np.flatnonzero(folds != fold)
        outer_val = np.flatnonzero(folds == fold)
        params, search = choose_params(train_features, labels, outer_train, 4200 + fold, c_values)
        kind, gamma, c = params
        train_kernel, test_kernel = gram(train_features, test_features, kind, gamma)
        model = SVC(kernel="precomputed", C=c, probability=True, random_state=4200 + fold)
        model.fit(train_kernel[np.ix_(outer_train, outer_train)], labels[outer_train])
        val_prob = model.predict_proba(train_kernel[np.ix_(outer_val, outer_train)])
        test_prob = model.predict_proba(test_kernel[:, outer_train])
        if not np.array_equal(model.classes_, np.arange(len(classes))):
            raise RuntimeError("Unexpected class ordering in SVC")
        oof[outer_val] = val_prob
        test_probs += test_prob / 5
        selected.append({"fold": fold, "kind": kind, "gamma": gamma, "C": c,
                         "inner_search": search,
                         "outer_accuracy": float(accuracy_score(labels[outer_val], val_prob.argmax(1)))})
        joblib.dump({"model": model, "features": train_features[outer_train],
                     "train_files": train_files[outer_train], "classes": classes,
                     "variant": variant, "params": params}, variant_dir / f"fold_{fold}.joblib")
        print(f"{variant} fold={fold} params={params} accuracy={selected[-1]['outer_accuracy']:.3f}", flush=True)

    if not np.allclose(oof.sum(axis=1), 1, atol=1e-5) or not np.allclose(test_probs.sum(axis=1), 1, atol=1e-5):
        raise AssertionError("Missing or invalid probabilities")
    result = {"variant": variant, "accuracy": float(accuracy_score(labels, oof.argmax(1))),
              "macro_f1": float(f1_score(labels, oof.argmax(1), average="macro")),
              "confusion_matrix": confusion_matrix(labels, oof.argmax(1), labels=np.arange(len(classes))).tolist(),
              "folds": selected, "class_names": classes,
              "selection_policy": "3-fold inner CV on outer training images; accuracy, then macro-F1",
              "evaluation_policy": "One outer-fold prediction per labeled image; 5-fold test probability average"}
    (variant_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_table(variant_dir / "oof_predictions.csv", train_files, labels, oof, classes)
    test_lookup = {name: row for name, row in zip(test_files, test_probs, strict=True)}
    if set(test_lookup) != set(template):
        raise ValueError("Test filenames do not match submission template")
    ordered = np.stack([test_lookup[name] for name in template])
    save_table(variant_dir / "test_probabilities.csv", np.asarray(template), None, ordered, classes)
    with (variant_dir / "submission.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "Category"])
        writer.writerows((name, classes[int(row.argmax())]) for name, row in zip(template, ordered, strict=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("outputs/features/bioclip2_patch_views.npz"))
    parser.add_argument("--reference-run", type=Path, default=Path("outputs/experiments/convnext_tiny_stage2_320"))
    parser.add_argument("--template", type=Path, default=Path("submission-for-task2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/bioclip2_patch_probe_v1"))
    parser.add_argument("--variant", choices=list(VARIANTS), action="append")
    parser.add_argument("--c-grid", default="0.1,1,10")
    args = parser.parse_args()
    c_values = [float(value) for value in args.c_grid.split(",")]
    if not c_values or any(not np.isfinite(value) or value <= 0 for value in c_values):
        raise ValueError("c-grid must contain positive finite values")
    with np.load(args.features.resolve(), allow_pickle=False) as archive:
        labels = archive["train_labels"].astype(np.int64)
        train_files = archive["train_files"].astype(str)
        test_files = archive["test_files"].astype(str)
        classes = archive["class_names"].astype(str).tolist()
        assignment = {row["file"]: row for row in read_csv(args.reference_run / "fold_assignments.csv")}
        if len(assignment) != len(train_files) or set(assignment) != set(train_files):
            raise ValueError("Feature files do not match reference folds")
        folds = np.asarray([int(assignment[name]["fold"]) for name in train_files])
        if any(int(assignment[name]["label"]) != label for name, label in zip(train_files, labels, strict=True)):
            raise ValueError("Reference fold labels differ from extracted labels")
        if len(classes) != 5 or any(np.sum((folds == fold) & (labels == cls)) != 20
                                   for fold in range(5) for cls in range(5)):
            raise ValueError("Expected the saved balanced 5-fold split")
        template_rows = read_csv(args.template)
        template = [row["ID"] for row in template_rows]
        if len(set(template)) != len(template):
            raise ValueError("Duplicate submission template IDs")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.reference_run / "fold_assignments.csv", args.output_dir / "fold_assignments.csv")
        (args.output_dir / "config.json").write_text(json.dumps({
            "feature_archive": str(args.features.resolve()),
            "feature_metadata": json.loads(str(archive["metadata"].item())),
            "reference_run": str(args.reference_run.resolve()),
            "class_names": classes, "variants": VARIANTS,
            "search": [(kind, gamma, c) for kind, gamma in (("linear", 0.0), ("rbf", 1.0)) for c in c_values],
            "inner_folds": 3, "selection_metric": "accuracy, then macro-F1",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        variants = args.variant or [name for name, spec in VARIANTS.items()
                                    if all(f"train_{block}" in archive.files for block in spec)]
        results = [evaluate_variant(archive, name, labels, folds, train_files, test_files,
                                    classes, args.output_dir, template, c_values) for name in variants]
    summary = [{key: result[key] for key in ("variant", "accuracy", "macro_f1", "confusion_matrix")}
               for result in results]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
