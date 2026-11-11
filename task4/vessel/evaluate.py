"""Combine only held-out predictions, with each original image appearing once."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .metrics import evaluate_probabilities
from .runtime import now, write_json


def summarize(root: Path, run: Path) -> dict:
    specification = json.loads((root / "experiments/folds_v1.json").read_text(encoding="utf-8"))
    by_id = {}
    for fold in specification["folds"]:
        directory = run / f"fold_{fold['fold']}"
        state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        if state["state"] != "completed":
            raise ValueError("OOF summary requires all folds to have completed")
        provenance = json.loads((directory / "fold.json").read_text(encoding="utf-8"))
        if provenance != fold:
            raise ValueError("Fold provenance mismatch")
        with np.load(directory / "best_predictions.npz", allow_pickle=False) as data:
            ids = data["ids"].tolist()
            probabilities = data["probabilities"]
            if ids != fold["validation_ids"]:
                raise ValueError("Predictions are not from the specified held-out images")
            for image_id, probability in zip(ids, probabilities):
                if image_id in by_id:
                    raise ValueError("An original image appears in more than one OOF fold")
                by_id[image_id] = probability.copy()
    ids = sorted(by_id, key=int)
    expected = sorted({i for f in specification["folds"] for i in f["validation_ids"]}, key=int)
    if ids != expected:
        raise ValueError("Missing OOF images")
    targets = []
    for image_id in ids:
        with Image.open(root / "train/label" / f"{image_id}.jpg") as image:
            targets.append(np.array(image.convert("L")) == 255)
    metrics = evaluate_probabilities([by_id[i] for i in ids], targets, ids)
    write_json(run / "oof_metrics.json", dict(created_at=now(), dataset_sha256=specification["dataset_sha256"], **metrics))
    np.savez_compressed(run / "oof_predictions.npz", ids=np.asarray(ids), probabilities=np.stack([by_id[i] for i in ids]))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root.resolve(), args.run_dir.resolve())
    print(json.dumps(result, indent=2))
