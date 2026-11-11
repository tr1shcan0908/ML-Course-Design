"""Audit completed round-two artifacts without rerunning model inference."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import zipfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    torch.set_num_threads(2)
    manifest = read_json(ROOT / "artifacts/manifest_512_768.json")
    records = manifest["records"]
    assert hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest() == manifest["dataset_hash"]
    assert manifest["dataset_hash"] == "5be9cec862a5c0cd3fa50a7ac8b2c3bda95195cd1380c0de3707b6f2e70b2bd6"
    assert sha(ROOT / "outputs/submission.csv") == "9b97a6a8cfa14470c4745287f55cfcfaaf14fff69ad027ae61e09fe3c7e07e3a"
    for row in records:
        assert sha(ROOT / row["path"]) == row["group"]
    folds = {r["id"]: r["fold"] for r in records if r["split"] == "train"}
    assert folds == {int(r["id"]): int(r["fold"]) for r in read_csv(ROOT / "analysis/baseline_oof.csv")}
    checkpoints = []
    run_names = ("fine_context768", "marginal_ms512", "fov_heatmap512")
    for name in run_names:
        run = ROOT / "runs" / name
        config = read_json(run / "config.json")
        assert read_json(run / "metrics.json")["complete_five_fold"]
        for fold in range(5):
            folder = run / f"fold_{fold}"
            path = folder / "best.pth"
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None, path
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert checkpoint["fold"] == fold
            assert checkpoint["config"]["signature"] == config["signature"]
            assert checkpoint["config"]["dataset_hash"] == config["dataset_hash"]
            assert all(bool(torch.isfinite(value).all()) for value in checkpoint["state_dict"].values())
            complete = read_json(folder / "complete.json")
            assert checkpoint["epoch"] == complete["best_epoch"]
            history = [json.loads(line) for line in (folder / "epochs.jsonl").read_text().splitlines() if line]
            assert [r["epoch"] for r in history] == list(range(1, len(history) + 1))
            assert history[-1]["best_epoch"] == complete["best_epoch"]
            values = read_csv(folder / "val_predictions.csv")
            assert {int(r["id"]) for r in values} == {i for i, f in folds.items() if f == fold}
            prediction = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in values])
            target = np.array([[float(r["x"]), float(r["y"])] for r in values])
            rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
            np.testing.assert_allclose(rmse, complete["metrics"]["pred"]["rmse"], atol=1e-6)
            checkpoints.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=sha(path), fold=fold, best_epoch=complete["best_epoch"], epochs=len(history),
                signature=config["signature"], training_source_sha256=checkpoint["config"].get("source_sha256", {})))
            del checkpoint
    directories = [ROOT / "runs" / name for name in (*run_names, "fine_shift5")]
    multicrop = read_json(ROOT / "runs/fine_shift5/config.json")
    for fold in range(5):
        assert sha(ROOT / multicrop["source_run"] / f"fold_{fold}/best.pth") == multicrop["source_checkpoint_sha256"][str(fold)]
    directories += [ROOT / "outputs/round2", ROOT / "outputs/round2_fov"]
    text_files = []
    for folder in directories:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".csv"):
                assert b"\0" not in path.read_bytes(), path
                if path.suffix == ".json":
                    read_json(path)
                elif path.suffix == ".jsonl":
                    for line in path.read_text().splitlines():
                        if line:
                            json.loads(line)
                else:
                    values = read_csv(path)
                    assert values and all(None not in row and None not in row.values() for row in values), path
                text_files.append(path.relative_to(ROOT).as_posix())
    template = read_csv(ROOT / "sample_submission.csv")
    submissions = []
    for folder in (ROOT / "outputs/round2", ROOT / "outputs/round2_fov"):
        selection = read_json(folder / "selection.json")
        values = read_csv(folder / "submission.csv")
        predictions = {int(r["id"]): r for r in read_csv(folder / "test_predictions.csv")}
        assert len(values) == 40
        assert [r["ImageID"] for r in values] == [r["ImageID"] for r in template]
        for row in values:
            image_id, _, axis = row["ImageID"].split("_")
            record = predictions[int(image_id)]
            value = float(row["value"])
            assert np.isfinite(value)
            assert 0 <= value <= float(record["width" if axis.lower() == "x" else "height"]) - 1
            assert abs(value - float(record[axis.lower()])) <= 0.00000051
        assert sha(folder / "submission.csv") == selection["submission_sha256"]
        submissions.append(dict(path=(folder / "submission.csv").relative_to(ROOT).as_posix(),
            sha256=sha(folder / "submission.csv"), rows=40, selected=selection["selected"]))
    result = dict(checked_at=dt.datetime.now().astimezone().isoformat(), checkpoints_validated=len(checkpoints),
        checkpoint_crc_pass=True, checkpoint_parameters_finite=True, checkpoint_fold_and_signature_pass=True,
        checked_text_files=len(text_files), no_null_bytes=True, structured_text_parse_pass=True,
        original_100_image_hashes_unchanged=True, original_submission_unchanged=True,
        original_fold_assignments_unchanged=True, multicrop_source_weights_unchanged=True,
        checkpoints=checkpoints, text_files=text_files, submissions=submissions,
        interruption_note="FOV fold 1 was interrupted after 12 epochs. Its partial files were archived in analysis/round2/interrupted_20261022_0125; fold 0 was retained and fold 1 restarted with the same configuration and seed. CUDA benchmark mode is not bitwise deterministic.")
    destination = ROOT / "outputs/round2_fov/artifact_integrity.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("checkpoints", "text_files")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
