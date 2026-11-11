"""Continue the archived ConvNeXt screening checkpoints through five folds."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch

from .convnext_study import checkpoint_for, compare, held_out_probabilities, read, trainer
from .convnext import ConvNeXtUNet, optimizer_groups
from .resilient_io import retry_file_operation
from .runtime import now, seed_everything, source_digest, write_json

from filelock import FileLock
import numpy as np
import torch
import torchvision

from vessel.data import get_fold, verify_dataset


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_pilot(root, pilot):
    source = pilot["source"]
    if (source_digest() != source["base"]
            or any(digest(root / "vessel_lab" / name) != sha for name, sha in source["files"].items())
            or hashlib.sha256(json.dumps(source, sort_keys=True).encode("utf-8")).hexdigest()
            != pilot["source_sha256"]):
        raise ValueError("The pilot training sources differ from the checkpoint lineage")
    inputs = {
        "experiments/model_swap_v6.json": pilot["plan_sha256"],
        "submission_v3/inference_recipe.json": pilot["recipe_sha256"],
        pilot["config"]["model"]["pretrained_path"]: pilot["config"]["model"]["pretrained_sha256"],
        **{f"optimization_v3/{name}_oof_selected.npz": sha for name, sha in pilot["oof_sha256"].items()},
    }
    for name, expected in inputs.items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen pilot input differs: {name}")
    data_hash = verify_dataset(root)
    if data_hash != pilot["dataset_sha256"]:
        raise ValueError("The dataset changed after screening")
    device = torch.device(pilot["device"])
    if (device.type != "cuda" or not torch.cuda.is_available()
            or (torch.__version__, torchvision.__version__, torch.cuda.get_device_name(device))
            != (pilot["torch"], pilot["torchvision"], pilot["gpu"])):
        raise RuntimeError("Use the pilot's PyTorch CUDA environment and GPU")
    return device, data_hash


def seed_fold(pilot_dir, run, fold, pilot, data_hash):
    index = fold["fold"]
    source = pilot_dir / f"fold_{index}"
    destination = run / f"fold_{index}"
    if (destination / "last.pt").is_file():
        return
    config, source_hash = pilot["config"], pilot["source_sha256"]
    last = checkpoint_for(source / "last.pt", config, fold, data_hash, source_hash)
    if last is None or last["step"] != pilot["plan"]["stop_step"]:
        raise ValueError(f"Pilot fold {index} is missing its endpoint checkpoint")
    expected_best = last["best_step"], last["best_score"]
    del last
    best = checkpoint_for(source / "best.pt", config, fold, data_hash, source_hash)
    if best is None or (best["step"], best["best_score"]) != expected_best:
        raise ValueError(f"Pilot fold {index} has mismatched best and last checkpoints")
    del best
    destination.mkdir(exist_ok=True)
    replace = retry_file_operation(os.replace)
    # The presence of last.pt commits the copy. A partial copy can be retried.
    for name in ("fold.json", "config.json", "train.jsonl", "status.json",
                 "best_metrics.json", "best_predictions.npz", "best.pt", "last_metrics.json", "last.pt"):
        temporary = destination / f"{name}.seed"
        shutil.copy2(source / name, temporary)
        replace(temporary, destination / name)


def publish(run, state, **fields):
    record = dict(state=state, pid=os.getpid(), updated_at=now(), **fields)
    write_json(run / "workflow_status.json", record)
    print(json.dumps(record, ensure_ascii=True), flush=True)


def main():
    root = Path(__file__).resolve().parents[1]
    pilot_dir = root / "runs/convnext_tiny_unet_v6"
    run = root / "runs/convnext_tiny_unet_v6_full"
    run.mkdir(parents=True, exist_ok=True)
    pilot_path = pilot_dir / "provenance.json"
    with FileLock(run / ".training.lock", timeout=0):
        try:
            publish(run, "checking_inputs")
            with FileLock(pilot_dir / ".training.lock", timeout=0):
                pilot = read(pilot_path)
                device, data_hash = verify_pilot(root, pilot)
                config, plan = pilot["config"], pilot["plan"]
                torch.set_num_threads(config["train"]["cpu_threads"])
                target = config["train"]["steps"]
                indices = plan["folds"] + [index for index in range(5) if index not in plan["folds"]]
                folds = [get_fold(root, index, data_hash) for index in indices]
                held_ids = [i for fold in folds for i in fold["validation_ids"]]
                if len(held_ids) != 20 or len(set(held_ids)) != 20:
                    raise ValueError("The five folds must cover all 20 validation images exactly once")
                if folds[:len(pilot["folds"])] != pilot["folds"]:
                    raise ValueError("The pilot fold membership changed")
                metadata = {
                    "pilot_provenance_sha256": digest(pilot_path),
                    "continuation_source_sha256": digest(Path(__file__)),
                    "training_source_sha256": pilot["source_sha256"],
                    "dataset_sha256": data_hash,
                    "config": config, "folds": folds, "pilot_folds": plan["folds"],
                    "checkpoint_selection": "best native mean Dice at 0.5",
                    "candidate_tta": plan["candidate_tta"],
                    "replacements": plan["replacements"],
                }
                metadata_path = run / "provenance.json"
                if metadata_path.is_file():
                    if read(metadata_path) != metadata:
                        raise ValueError("Continuation provenance differs from the existing run")
                else:
                    write_json(metadata_path, metadata)
                for fold in pilot["folds"]:
                    seed_fold(pilot_dir, run, fold, pilot, data_hash)

            source_hash = pilot["source_sha256"]
            completed = []
            for fold in folds:
                index = fold["fold"]
                path = run / f"fold_{index}/last.pt"
                saved = checkpoint_for(path, config, fold, data_hash, source_hash)
                step = saved["step"] if saved is not None else 0
                del saved
                if step < target:
                    if (run / "STOP").exists():
                        publish(run, "paused", fold=index, completed_folds=completed)
                        return
                    publish(run, "training", fold=index, resume_step=step,
                            target_steps=target, completed_folds=completed)
                    # The trainer and model factory are the same as in screening.
                    with patch.multiple(trainer, build_model=ConvNeXtUNet, optimizer_groups=optimizer_groups,
                                        source_digest=lambda: source_hash):
                        done = trainer.train_fold(root, run, config, index, data_hash, device,
                                                  resume=bool(step), stop_after=None)
                    if not done:
                        publish(run, "paused", fold=index, completed_folds=completed)
                        return
                elif step != target:
                    raise ValueError(f"Fold {index} passed its configured endpoint")
                completed.append(index)

            publish(run, "evaluating", completed_folds=completed)
            probabilities, fold_results = [], []
            for fold in folds:
                index = fold["fold"]
                directory = run / f"fold_{index}"
                last = checkpoint_for(directory / "last.pt", config, fold, data_hash, source_hash)
                if last is None or last["step"] != target:
                    raise ValueError(f"Fold {index} is incomplete")
                best_step, best_score = last["best_step"], last["best_score"]
                del last
                path = directory / "best.pt"
                best = checkpoint_for(path, config, fold, data_hash, source_hash)
                if best is None or (best["step"], best["best_score"]) != (best_step, best_score):
                    raise ValueError(f"Fold {index} has mismatched best and last checkpoints")
                seed_everything(config["seed"] + index, config["train"]["deterministic"])
                probabilities.append(held_out_probabilities(root, best, device, plan["candidate_tta"] == "flip4"))
                del best
                fold_results.append(dict(fold=index, steps=target, best_step=best_step,
                                         best_dice_at_0_5=best_score, best_checkpoint_sha256=digest(path)))

            candidate = np.concatenate(probabilities)
            recipe = read(root / "submission_v3/inference_recipe.json")
            ids, result = compare(root, plan, recipe, folds, candidate)
            for entry in result["comparisons"].values():
                entry["passes_improvement_gate"] = entry.pop("passes_screening")
            passed = result.pop("promote_to_full_study")
            np.savez_compressed(run / "full_oof.npz", ids=np.asarray(ids), convnext=candidate)
            write_json(run / "full_results.json", dict(
                completed_at=now(), folds=fold_results, passes_improvement_gate=passed, **result,
            ))
            publish(run, "completed", completed_folds=completed, passes_improvement_gate=passed)
        except Exception as error:
            publish(run, "failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
