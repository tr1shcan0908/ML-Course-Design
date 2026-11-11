"""Screen a ConvNeXt member and replay two fixed replacements of the v3 ensemble."""

from __future__ import annotations

from .resilient_io import install

install()
from . import train as trainer
from .runtime import now, seed_everything, source_digest, write_json

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from filelock import FileLock
import numpy as np
from PIL import Image
import torch
import torchvision

from tools.prepare_submission import predict_variants, resize_probabilities, sha256
from vessel.data import get_fold, verify_dataset
from vessel.metrics import binary_metrics
from .convnext import ConvNeXtUNet, optimizer_groups
from .data import load_samples
from .directional import interior, narrow_pixels, neighbors
from .directional_pilot import frozen_oof, score
from .export_study import targets_for


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_for(path, config, fold, data_hash, source_hash):
    if not path.is_file():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected_config = dict(config, fold=fold["fold"], seed=config["seed"] + fold["fold"])
    if (saved["config"], saved["fold"], saved["dataset_sha256"], saved["source_sha256"]) != (
        expected_config, fold, data_hash, source_hash
    ):
        raise ValueError(f"Checkpoint source, config, data or fold differs: {path}")
    return saved


def held_out_probabilities(root, saved, device, flip4):
    model = ConvNeXtUNet(saved["config"]["model"], root, initialize=False)
    model.load_state_dict(saved["model"])
    model = model.to(device, memory_format=torch.channels_last).eval()
    samples = load_samples(root, saved["fold"]["validation_ids"], saved["config"]["data"])
    identity_scores, probabilities = [], []
    for sample in samples:
        identity, augmented = predict_variants(model, sample, device, flip4)
        height, width = sample.original_hw
        target = sample.mask[0, :height, :width].numpy()
        identity_scores.append(binary_metrics(identity >= 0.5, target)["dice"])
        probabilities.append(augmented)
    if abs(float(np.mean(identity_scores)) - saved["validation"]["mean_dice_at_0_5"]) > 1e-12:
        raise ValueError("Identity inference failed to replay the endpoint checkpoint Dice")
    return np.stack(probabilities)


def compare(root, plan, recipe, folds, candidate):
    all_ids = [row["id"] for row in read(root / "optimization_v3/per_image_selected.json")]
    all_targets = np.stack(targets_for(root, all_ids))
    baseline, members, _ = frozen_oof(root, recipe, all_ids, all_targets)
    ids = [i for fold in folds for i in fold["validation_ids"]]
    positions = [all_ids.index(i) for i in ids]
    targets, baseline = all_targets[positions], baseline[positions]
    mask = torch.from_numpy(targets[:, None])
    narrow = narrow_pixels(mask.float()).bool()
    narrow_edges = interior(mask) & neighbors(mask) & (interior(narrow) | neighbors(narrow))
    target_512 = [np.asarray(Image.fromarray(y).resize((512, 512), Image.Resampling.NEAREST)) for y in targets]
    values = {"v3": baseline, "convnext": candidate}
    for name in plan["replacements"]:
        replacement = np.zeros_like(baseline)
        for member, probabilities in zip(recipe["members"], members):
            selected = candidate if member["name"] == name else probabilities[positions]
            replacement += member["weight"] * selected
        values[f"replace_{name}"] = replacement
    scores = {}
    for name, probabilities in values.items():
        threshold = 0.5 if name == "convnext" else recipe["threshold"]
        metrics = score(probabilities, threshold, targets, narrow[:, 0].numpy(), narrow_edges, ids)
        metrics["mean_dice_512_proxy"] = float(np.mean([
            binary_metrics(p >= threshold, y)["dice"]
            for p, y in zip(resize_probabilities(probabilities, (512, 512)), target_512)
        ]))
        scores[name] = metrics
    reference = scores["v3"]
    comparisons = {}
    for name in plan["replacements"]:
        metrics = scores[f"replace_{name}"]
        gain = metrics["mean_dice"] - reference["mean_dice"]
        paired = {a["id"]: a["dice"] - b["dice"]
                  for a, b in zip(metrics["per_image"], reference["per_image"])}
        by_fold = {str(fold["fold"]): float(np.mean([paired[i] for i in fold["validation_ids"]]))
                   for fold in folds}
        passes = (
            gain >= plan["minimum_gain_over_v3"] and all(delta >= 0 for delta in by_fold.values())
            and metrics["mean_dice_512_proxy"] >= reference["mean_dice_512_proxy"]
        )
        comparisons[name] = dict(gain_over_v3=gain, gain_by_fold=by_fold, passes_screening=passes)
    return ids, dict(scores=scores, comparisons=comparisons,
                     promote_to_full_study=any(row["passes_screening"] for row in comparisons.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/convnext_tiny_unet_v6"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = root / "experiments/model_swap_v6.json"
    plan = read(plan_path)
    config = read(root / plan["training_config"])
    config.update(name=plan["name"], model=plan["model"], references=[
        "https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html",
        "https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.convnext_tiny.html",
    ])
    config["train"]["validate_every"] = plan["checkpoint_every"]
    config["train"]["log_every"] = 100
    stop = plan["stop_step"]
    if not 1 <= stop < config["train"]["steps"] or plan["checkpoint_every"] < 1:
        raise ValueError("Screening steps must fall inside the original learning rate schedule")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(config["train"]["cpu_threads"])
    data_hash = verify_dataset(root)
    folds = [get_fold(root, fold, data_hash) for fold in plan["folds"]]
    ids = [i for fold in folds for i in fold["validation_ids"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Screening requires distinct, nonempty validation folds")
    recipe_path = root / "submission_v3/inference_recipe.json"
    recipe = read(recipe_path)
    if not set(plan["replacements"]) <= {member["name"] for member in recipe["members"]}:
        raise ValueError("All replacements must be members of the frozen v3 ensemble")
    code_hashes = {name: sha256(Path(__file__).with_name(name))
                   for name in ("convnext.py", "convnext_study.py", "directional.py", "directional_pilot.py",
                                "resilient_io.py")}
    source = {"base": source_digest(), "files": code_hashes}
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True).encode("utf-8")).hexdigest()
    provenance = dict(plan=plan, plan_sha256=sha256(plan_path), config=config,
                      dataset_sha256=data_hash, folds=folds, source=source, source_sha256=source_hash,
                      recipe_sha256=sha256(recipe_path), torch=torch.__version__,
                      torchvision=torchvision.__version__, device=str(device),
                      gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      oof_sha256={member["name"]: sha256(root / f"optimization_v3/{member['name']}_oof_selected.npz")
                                  for member in recipe["members"]})
    output = root / args.output
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    with FileLock(output / ".training.lock", timeout=0):
        metadata = output / "provenance.json"
        if args.resume:
            if read(metadata) != provenance:
                raise ValueError("Resume requires identical code, data, plan, and runtime")
        else:
            write_json(metadata, provenance)
        probabilities = []
        for fold in folds:
            if (output / "STOP").exists():
                return
            path = output / f"fold_{fold['fold']}/last.pt"
            saved = checkpoint_for(path, config, fold, data_hash, source_hash)
            if saved is None or saved["step"] < stop:
                del saved
                # Route only this standalone study through the new model while
                # keeping the archived trainer and its checkpoint digest intact.
                with patch.multiple(trainer, build_model=ConvNeXtUNet, optimizer_groups=optimizer_groups,
                                    source_digest=lambda: source_hash):
                    trainer.train_fold(root, output, config, fold["fold"], data_hash, device,
                                       resume=args.resume, stop_after=stop)
                saved = checkpoint_for(path, config, fold, data_hash, source_hash)
            if saved is None or saved["step"] < stop:
                return
            if saved["step"] != stop:
                raise ValueError("Screening must use the fixed endpoint checkpoint")
            seed_everything(config["seed"] + fold["fold"], config["train"]["deterministic"])
            probabilities.append(held_out_probabilities(root, saved, device, plan["candidate_tta"] == "flip4"))
            del saved
        candidate = np.concatenate(probabilities)
        ids, result = compare(root, plan, recipe, folds, candidate)
        np.savez_compressed(output / "pilot_oof.npz", ids=np.asarray(ids), convnext=candidate)
        write_json(output / "pilot_results.json", dict(completed_at=now(), **result))
        write_json(output / "status.json", dict(state="pilot_completed", updated_at=now(),
                                               promote_to_full_study=result["promote_to_full_study"]))
        print(json.dumps(result["comparisons"]), flush=True)


if __name__ == "__main__":
    main()
