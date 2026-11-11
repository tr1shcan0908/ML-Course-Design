"""Bundle the new model and its reproducible source after all folds finish."""

from pathlib import Path
import importlib.metadata
import json
import zipfile

from vessel_lab.submission_domain import dump, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/dual_domain_v8"
    if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "completed":
        raise ValueError("Finish all folds and evaluation before packaging")
    destination = root / "submission_v9_calibrated"
    paths = []
    for directory in ("vessel", "vessel_lab"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file() and p.suffix == ".py")
    paths.extend(root / name for name in (
        "tools/__init__.py", "tools/prepare_submission.py", "tools/reproduce_v8.py",
        "tools/package_dual_domain.py", "experiments/dual_domain_v8.json",
        "experiments/dual_domain_fusion_v8.json", "experiments/folds_v1.json",
        "experiments/fusion_calibration_v9.json", "optimization_v9/results.json",
        "experiments/dataset_manifest.json", "segmentation_to_csv.py", "sample_submission.csv",
        "vessel_lab/vendor/LICENSE", "vessel_lab/vendor/README.md",
        "submission_v8_dual_domain.csv", "submission_v8_dual_domain/inference_recipe.json",
        "submission_v8_dual_domain/submission_checks.json", "submission_v8_dual_domain/README.md",
        "runs/dual_domain_v8/provenance.json", "runs/dual_domain_v8/results.json",
        "runs/dual_domain_v8/fusion_results.json", "research/双域模型与提交尺度修正_v8.md"))
    paths.extend(root / name for name in (
        "submission_v3/oof_predictions.npz", "submission_v3/test_probabilities_native.npz",
        "submission_v3/inference_recipe.json", "submission_v8_fusion.csv",
        "submission_v8_fusion/inference_recipe.json", "submission_v8_fusion/platform_result.json",
        "submission_v9_calibrated.csv", "submission_v9_calibrated/inference_recipe.json",
        "submission_v9_calibrated/submission_checks.json", "submission_v9_calibrated/platform_result.json",
        "submission_v9_calibrated/README.md", "review/platform_scores_20261106.json",
        "runs/dual_domain_v8/environment.json", "runs/dual_domain_v8/test_checks.json"))
    for fold in range(5):
        paths.extend(run / f"fold_{fold}" / name for name in ("best.pt", "best_metrics.json", "fold.json", "status.json", "train.jsonl"))
    paths = sorted(set(paths))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    dependencies = (root / "requirements.lock.txt").read_text(encoding="utf-8")
    dependencies += f"torchvision=={importlib.metadata.version('torchvision')}\n"
    readme = """# Public-tested v9: Dice 0.86017

This bundle contains the five new model checkpoints, their source and configuration,
and frozen v3 validation/test probabilities. It reproduces the final v9 CSV.
See submission_v9_calibrated/README.md for measured comparison and recommendation.

Use the recorded Python 3.11 / PyTorch CUDA environment. Place the original `test`
folder beside this README, without renaming or re-encoding its images. Then run:

    python -m tools.reproduce_v8 --kind dual --output reproduced_model
    python -m tools.reproduce_v8 --kind fusion --fusion-recipe submission_v9_calibrated/inference_recipe.json --candidate-probabilities reproduced_model/test_predictions.npz --output reproduced_final

The commands load all five verified checkpoints, run D4 inference, fuse the frozen
v3 probabilities, execute the supplied encoder, and compare the final CSV SHA-256.

For a new-model training replay, add both original `train` and `test` folders:

    python -m vessel_lab.dual_domain_study
    python -m vessel_lab.dual_domain_fusion
    python -m vessel_lab.fusion_calibration

Input data and the 15 older v3 checkpoints are not duplicated here. The older
checkpoints remain in submission_v3/optimized_reproduce.zip in the original
workspace. To rebuild their probabilities as well, follow the final README.
Resumable new-model last.pt files remain in the original runs/dual_domain_v8 folder.
"""
    archive = destination / "v9_reproduce.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(root).as_posix())
        bundle.writestr("requirements.lock.txt", dependencies)
        bundle.writestr("REPRODUCE.md", readme)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Archive integrity check failed")
    manifest = dict(zip_sha256=sha256(archive), bytes=archive.stat().st_size,
                    files={p.relative_to(root).as_posix(): sha256(p) for p in paths},
                    note="Includes five new checkpoints and frozen v3 OOF/test probabilities; input images and the 15 old v3 checkpoints are not duplicated.")
    dump(destination / "package_manifest.json", manifest)
    print(json.dumps(dict(path=str(archive), bytes=manifest["bytes"], sha256=manifest["zip_sha256"])), flush=True)


if __name__ == "__main__":
    main()
