"""Rebuild the fixed final blend from archived predictions; no model selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reproduced/submission.csv"))
    args = parser.parse_args()
    selection = json.loads((ROOT / "outputs/round2_fov/selection.json").read_text(encoding="utf-8"))
    assert selection["selected"] == "heatmap+fov512"
    predictions = {}
    for run, weight in selection["weights"].items():
        path = ROOT / "runs" / run / "test_predictions.csv"
        expected = selection["run_files"][run]["test_predictions.csv"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Prediction fingerprint changed: " + run)
        for row in rows(path):
            key = int(row["id"])
            output = predictions.setdefault(key, {"x": 0., "y": 0.})
            for axis in ("x", "y"):
                output[axis] += weight * float(row[axis])
    template = rows(ROOT / "sample_submission.csv")
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ImageID", "value"])
        writer.writeheader()
        for row in template:
            image_id, _, axis = row["ImageID"].split("_")
            writer.writerow(dict(ImageID=row["ImageID"], value=f"{predictions[int(image_id)][axis.lower()]:.6f}"))
    actual = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual != selection["submission_sha256"]:
        raise ValueError("Final CSV fingerprint differs: " + actual)
    print(json.dumps(dict(rows=len(template), images=len(predictions), sha256=actual, verified=True)))


if __name__ == "__main__":
    main()
