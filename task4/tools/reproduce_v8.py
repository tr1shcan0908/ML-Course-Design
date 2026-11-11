"""Reproduce the script-compatible v3 or trained dual-domain submission."""

from pathlib import Path
import argparse
import json

import numpy as np
import torch

from vessel.runtime import seed_everything
from vessel_lab.dual_domain import DualDomainNet
from vessel_lab.dual_domain_study import load_grid, predict
from vessel_lab.dual_domain_fusion import transport
from vessel_lab.submission_domain import dump, export_masks, script_mask, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("compatible", "dual", "fusion"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, default=Path("submission_v3/test_probabilities_native.npz"))
    parser.add_argument("--recipe", type=Path, default=Path("submission_v8_dual_domain/inference_recipe.json"))
    parser.add_argument("--candidate-probabilities", type=Path, default=Path("runs/dual_domain_v8/test_predictions.npz"))
    parser.add_argument("--fusion-recipe", type=Path, default=Path("submission_v8_fusion/inference_recipe.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    torch.set_num_threads(4)
    if args.kind == "compatible":
        with np.load(root / args.probabilities, allow_pickle=False) as saved:
            ids, p = saved["ids"].tolist(), saved["probabilities"].copy()
        native = p >= .54
        checks = export_masks(root, output, ids, [script_mask(mask) for mask in native], native)
        expected_path = root / "submission_v3_script_compatible.csv"
    elif args.kind == "fusion":
        recipe = json.loads((root / args.fusion_recipe).read_text(encoding="utf-8"))
        if sha256(root / "submission_v3/inference_recipe.json") != recipe["baseline_recipe_sha256"]:
            raise ValueError("Baseline inference recipe changed")
        if sha256(root / args.recipe) != recipe["candidate_recipe_sha256"]:
            raise ValueError("Candidate inference recipe changed")
        if sha256(root / "vessel_lab/dual_domain_fusion.py") != recipe["script"]:
            raise ValueError("Fusion implementation changed")
        with np.load(root / args.probabilities, allow_pickle=False) as saved:
            ids, base = saved["ids"].tolist(), transport(saved["probabilities"])
        with np.load(root / args.candidate_probabilities, allow_pickle=False) as saved:
            order = [saved["ids"].tolist().index(i) for i in ids]
            new = saved["probabilities"][order, 1]
        weight = recipe["candidate_weight"]
        checks = export_masks(root, output, ids, ((1 - weight) * base + weight * new) >= recipe["threshold"])
        expected_path = root / f"{args.fusion_recipe.parent.name}.csv"
    else:
        recipe = json.loads((root / args.recipe).read_text(encoding="utf-8"))
        config = recipe["config"]
        seed_everything(config["seed"], True)
        ids = [str(i) for i in range(1, 21)]
        images, _, _ = load_grid(root, ids, "test")
        total = None
        for entry in recipe["checkpoints"]:
            path = root / entry["path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Checkpoint fingerprint mismatch: {path}")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["config"] != config or saved["fold"]["fold"] != entry["fold"]:
                raise ValueError("Checkpoint config/fold mismatch")
            device = torch.device("cuda")
            model = DualDomainNet(**config["model"]).to(device, memory_format=torch.channels_last)
            model.load_state_dict(saved["model"])
            p = predict(model, images, device, d4=True)
            total = p if total is None else total + p
        probabilities = total / len(recipe["checkpoints"])
        checks = export_masks(root, output, ids, probabilities[:, 1] >= recipe["threshold"])
        np.savez_compressed(output / "test_predictions.npz", ids=np.asarray(ids), probabilities=probabilities)
        expected_path = root / "submission_v8_dual_domain.csv"
    expected = sha256(expected_path)
    if checks["csv_sha256"] != expected:
        raise ValueError("Reproduced CSV differs from the delivered CSV")
    dump(output / "reproduction_check.json", dict(passed=True, kind=args.kind,
         actual_sha256=checks["csv_sha256"], expected_sha256=expected))
    print(json.dumps(dict(passed=True, kind=args.kind, sha256=expected)), flush=True)


if __name__ == "__main__":
    main()
