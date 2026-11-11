"""Evaluate predeclared D4 and coarse fusion weights after complete five-fold runs."""

from .resilient_io import install
install()
from .runtime import now, seed_everything, write_json

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from vessel.data import get_fold, verify_dataset
from vessel.data import load_samples as load_v1_samples
from vessel.metrics import binary_metrics, evaluate_probabilities
from .data import load_samples as load_lab_samples
from .export_study import generate_submission, load_checkpoint_model, sha256, targets_for
from .inference import predict_d4, predict_member


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_combinations(names, plan):
    choices = [([name], [1.0]) for name in names]
    for pair in itertools.combinations(names, 2):
        for weight in plan["pair_first_member_weights"]:
            choices.append((list(pair), [weight, 1 - weight]))
    if len(names) == 3:
        choices += [(list(names), weights) for weights in plan["triple_weights"]]
    return choices


def weighted_probabilities(sets, names, weights):
    result = np.zeros_like(sets[names[0]])
    for name, weight in zip(names, weights):
        result += sets[name] * weight
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("optimization_v2"))
    parser.add_argument("--output", type=Path, default=Path("optimization_v3"))
    parser.add_argument("--plan", type=Path, default=Path("experiments/inference_refinement_v3.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study, output = root / args.study, root / args.output
    plan = read(root / args.plan)
    candidates = read(study / "completed_models.json")
    promoted = read(study / "promoted.json")
    if {c["name"] for c in candidates} != {c["name"] for c in promoted}:
        raise ValueError("Both promoted models must finish all five folds first")
    if any(read(root / c["run_dir"] / "status.json")["state"] != "completed_all_folds" for c in candidates):
        raise ValueError("An incomplete model cannot enter final inference refinement")
    output.mkdir(exist_ok=True)
    write_json(output / "plan.json", plan)
    status = dict(state="evaluating_d4", started_at=now(), updated_at=now())
    write_json(output / "status.json", status)
    torch.set_num_threads(4)
    seed_everything(20261103, True)
    device = torch.device("cuda")
    data_hash = verify_dataset(root)
    ids = list(map(str, range(1, 21)))
    targets = targets_for(root, ids)
    summaries, probability_sets, records = {}, {}, {}
    evaluation_v2 = read(study / "model_evaluations.json")
    for name in plan["models"]:
        implementation = "v1" if name == "csa_v1" else "lab"
        run_dir = "runs/" + name
        standard_path = root / "submission/oof_selected.npz" if implementation == "v1" else study / f"{name}_oof_selected.npz"
        with np.load(standard_path, allow_pickle=False) as archive:
            if archive["ids"].tolist() != ids:
                raise ValueError("Selected standard OOF IDs differ")
            standard = archive["probabilities"].copy()
        standard_tta = True if implementation == "v1" else evaluation_v2[name]["tta_selected"]
        d4, tiled, checkpoints = {}, {}, []
        status.update(current_model=name, updated_at=now())
        write_json(output / "status.json", status)
        for fold_index in range(5):
            path = root / run_dir / f"fold_{fold_index}/best.pt"
            model, checkpoint = load_checkpoint_model(root, path, implementation, device)
            fold = get_fold(root, fold_index, data_hash)
            if checkpoint["dataset_sha256"] != data_hash or checkpoint["fold"] != fold:
                raise ValueError("Checkpoint fold/dataset provenance mismatch")
            samples = (load_v1_samples(root, fold["validation_ids"], checkpoint["config"]["data"]["pad_hw"])
                       if implementation == "v1" else load_lab_samples(root, fold["validation_ids"], checkpoint["config"]["data"]))
            with np.load(root / run_dir / f"fold_{fold_index}/best_predictions.npz", allow_pickle=False) as saved:
                if saved["ids"].tolist() != fold["validation_ids"]:
                    raise ValueError("Archived OOF IDs differ from held-out fold")
                archived = saved["probabilities"].copy()
            for sample, reference in zip(samples, archived):
                identity, augmented = predict_d4(model, sample, device)
                if sample.image_id in fold["train_ids"] or sample.image_id in d4:
                    raise ValueError("Invalid held-out prediction")
                if float(np.max(np.abs(identity - reference))) > 1e-6:
                    raise ValueError("D4 identity inference does not replay the saved validation")
                d4[sample.image_id] = augmented
                if name == "csa_v1" and plan.get("csa_tiled_inference"):
                    tiled[sample.image_id] = predict_member(model, sample, device,
                        dict(tta_mode="flip4", spatial_mode="tiled", **plan["csa_tiled_inference"]))
            checkpoints.append(dict(fold=fold_index, path=path.relative_to(root).as_posix(), sha256=sha256(path)))
            print(json.dumps(dict(event="d4_fold_evaluated", name=name, fold=fold_index)), flush=True)
            del model
        if set(d4) != set(ids):
            raise ValueError("D4 OOF does not cover every original image exactly once")
        d4_array = np.stack([d4[i] for i in ids])
        standard_score = evaluate_probabilities(list(standard), targets, ids)
        d4_score = evaluate_probabilities(list(d4_array), targets, ids)
        variants = {"standard": (standard, standard_score, "flip4" if standard_tta else "identity", "full"),
                    "d4": (d4_array, d4_score, "d4", "full")}
        if tiled:
            tiled_array = np.stack([tiled[i] for i in ids])
            blend = (standard + tiled_array) / 2
            variants["tiled_flip4"] = (tiled_array, evaluate_probabilities(list(tiled_array), targets, ids), "flip4", "tiled")
            variants["blend_full_tiled"] = (blend, evaluate_probabilities(list(blend), targets, ids), "flip4", "blend_full_tiled")
        best_variant = max(variants, key=lambda key: variants[key][1]["calibrated_mean_dice"])
        if variants[best_variant][1]["calibrated_mean_dice"] < standard_score["calibrated_mean_dice"] + plan["d4_minimum_gain_per_model"]:
            best_variant = "standard"
        chosen_values, _, tta_mode, spatial_mode = variants[best_variant]
        probability_sets[name] = chosen_values
        records[name] = dict(name=name, implementation=implementation, run_dir=run_dir, tta=standard_tta,
                             tta_mode=tta_mode, spatial_mode=spatial_mode, checkpoints=checkpoints)
        if spatial_mode != "full":
            records[name].update(plan["csa_tiled_inference"])
        summaries[name] = dict(standard=standard_score, d4=d4_score, d4_selected=best_variant == "d4",
                               selected_variant=best_variant,
                               variants={key: value[1] for key, value in variants.items()})
        np.savez_compressed(output / f"{name}_oof_selected.npz", ids=np.asarray(ids), probabilities=probability_sets[name])
        write_json(output / "model_evaluations.json", summaries)
    rows = []
    status.update(state="evaluating_fixed_fusions", updated_at=now())
    write_json(output / "status.json", status)
    for names, weights in fixed_combinations(plan["models"], plan):
        values = weighted_probabilities(probability_sets, names, weights)
        metrics = evaluate_probabilities(list(values), targets, ids)
        rows.append(dict(members=names, weights=weights, metrics=metrics))
    winner = max(rows, key=lambda row: row["metrics"]["calibrated_mean_dice"])
    previous = read(study / "result.json")
    gain = winner["metrics"]["calibrated_mean_dice"] - previous["best_calibrated_oof_dice"]
    export = gain >= plan["minimum_export_gain_over_v2"]
    result = dict(completed_at=now(), previous_v2_oof_dice=previous["best_calibrated_oof_dice"],
                  best_calibrated_oof_dice=winner["metrics"]["calibrated_mean_dice"], gain_over_v2=gain,
                  winner_members=winner["members"], winner_weights=winner["weights"],
                  winner_threshold=winner["metrics"]["calibrated_threshold"],
                  new_submission_exported=False, candidate_passes_improvement_gate=export,
                  note="Checkpoint/model/TTA/threshold/weights selected using the same image-level OOF; not independent platform performance")
    write_json(output / "fusion_results.json", rows)
    write_json(output / "result.json", result)
    values = weighted_probabilities(probability_sets, winner["members"], winner["weights"])
    threshold = winner["metrics"]["calibrated_threshold"]
    selected_scores = [binary_metrics(p >= threshold, target)["dice"] for p, target in zip(values, targets)]
    write_json(output / "per_image_selected.json", [dict(id=i, dice=s) for i, s in zip(ids, selected_scores)])
    if export:
        target = root / plan["final_export_dir"]
        target.mkdir(exist_ok=True)
        members = [dict(records[name], weight=weight) for name, weight in zip(winner["members"], winner["weights"])]
        recipe = dict(members=members, threshold=threshold, target_hw=[512, 512], foreground="vessel",
                      order="row_major_C", seed=20261103, dataset_sha256=data_hash, torch=torch.__version__)
        write_json(target / "inference_recipe.json", recipe)
        np.savez_compressed(target / "oof_predictions.npz", ids=np.asarray(ids), probabilities=values)
        write_json(target / "oof_metrics.json", winner["metrics"])
        checks = generate_submission(root, target, members, threshold, data_hash, device)
        torch.cuda.empty_cache()
        with (target / "independent_rebuild.log").open("w", encoding="utf-8") as log:
            command = [sys.executable, "-u", "-m", "vessel_lab.export_study", "--recipe",
                       str((target / "inference_recipe.json").relative_to(root)), "--output",
                       str((target / "rebuild_check").relative_to(root))]
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        if sha256(target / "submission.csv") != sha256(target / "rebuild_check/submission.csv"):
            raise ValueError("Refined CSV failed independent frozen-recipe reproduction")
        checks["independent_rebuild_exact_csv_match"] = True
        write_json(target / "submission_checks.json", checks)
        (root / "submission_v3.csv").write_bytes((target / "submission.csv").read_bytes())
        result.update(new_submission_exported=True, submission_csv="submission_v3.csv",
                      submission_sha256=checks["csv_sha256"], independent_rebuild_passed=True)
        write_json(target / "result.json", result)
        write_json(output / "result.json", result)
    status.update(state="completed", updated_at=now(), result=result)
    write_json(output / "status.json", status)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
