"""Compare only genuinely held-out predictions and export an improved recipe."""

from .runtime import now, write_json, seed_everything

import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from vessel.data import load_samples as load_v1_samples, verify_dataset
from vessel.metrics import evaluate_probabilities
from vessel.model import VesselUNet
from vessel.rle import encode, decode
from .data import load_samples as load_lab_samples
from .models import build_model
from tools.prepare_submission import (predict_variants, resize_probabilities, read_official_encoder,
                                      sha256, validate_csv, write_table)


def load_checkpoint_model(root, path, implementation, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if implementation == "v1":
        model = VesselUNet(checkpoint["config"]["model"])
    else:
        model = build_model(checkpoint["config"]["model"], root, initialize=False)
    model = model.to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def targets_for(root, ids):
    targets = []
    for image_id in ids:
        with Image.open(root / "train/label" / f"{image_id}.jpg") as image:
            targets.append(np.asarray(image.convert("L")) == 255)
    return targets


def generate_submission(root: Path, output: Path, members: list[dict], threshold: float, data_hash: str, device):
    from .inference import predict_member, validate_weights
    weights = validate_weights(members)
    output.mkdir(parents=True, exist_ok=True)
    with (root / "sample_submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        test_ids = [r["Id"] for r in csv.DictReader(stream)]
    total = np.zeros((len(test_ids), 584, 565), dtype=np.float32)
    for member_index, member in enumerate(members):
        member_total = np.zeros_like(total)
        for index in range(5):
            path = root / member["run_dir"] / f"fold_{index}/best.pt"
            if member.get("checkpoints") and sha256(path) != member["checkpoints"][index]["sha256"]:
                raise ValueError("Frozen inference checkpoint hash mismatch")
            model, checkpoint = load_checkpoint_model(root, path, member["implementation"], device)
            if checkpoint["dataset_sha256"] != data_hash:
                raise ValueError("Frozen checkpoint data fingerprint differs")
            if member["implementation"] == "v1":
                samples = load_v1_samples(root, test_ids, checkpoint["config"]["data"]["pad_hw"], split="test")
            else:
                samples = load_lab_samples(root, test_ids, checkpoint["config"]["data"], split="test")
            for j, sample in enumerate(samples):
                member_total[j] += predict_member(model, sample, device, member)
            print(json.dumps(dict(event="predicted_test_fold", model=member["name"], fold=index)), flush=True)
            del model
        if weights is None:
            total += (member_total / 5) / len(members)
        else:
            total += (member_total / 5) * weights[member_index]
    np.savez_compressed(output / "test_probabilities_native.npz", ids=np.asarray(test_ids), probabilities=total)
    resized = resize_probabilities(total, (512, 512))
    image_dir = output / "image"
    image_dir.mkdir(exist_ok=True)
    rows = []
    for image_id, probability in zip(test_ids, resized):
        mask = probability >= threshold
        Image.fromarray(mask.astype(np.uint8) * 255).save(image_dir / f"{image_id}.png")
        encoded = encode(mask)
        if not np.array_equal(decode(encoded, (512, 512)), mask):
            raise ValueError("New RLE failed roundtrip")
        rows.append(dict(Id=image_id, Predicted=encoded))
    write_table(output / "submission.csv", rows, ["Id", "Predicted"])
    original = read_official_encoder(root / "segmentation_to_csv.py")([str(image_dir / f"{i}.png") for i in test_ids])
    if original != [row["Predicted"] for row in rows]:
        raise ValueError("New CSV differs from the provided encoding function")
    checks = validate_csv(output / "submission.csv", image_dir, test_ids, (512, 512))
    checks["provided_encoder_function_exact_match"] = True
    write_json(output / "submission_checks.json", checks)
    return checks


def compare_and_export(root: Path, study: Path, candidates: list[dict], output: Path) -> dict:
    torch.set_num_threads(4)
    seed_everything(20261103, True)
    device = torch.device("cuda")
    data_hash = verify_dataset(root)
    ids = list(map(str, range(1, 21)))
    targets = targets_for(root, ids)
    with np.load(root / "submission/oof_selected.npz", allow_pickle=False) as archive:
        if archive["ids"].tolist() != ids:
            raise ValueError("Archived v1 OOF IDs differ")
        base = archive["probabilities"].copy()
    probability_sets = {"csa_v1": base}
    base_checkpoints = [dict(fold=i, path=f"runs/csa_v1/fold_{i}/best.pt", sha256=sha256(root / f"runs/csa_v1/fold_{i}/best.pt")) for i in range(5)]
    model_records = {"csa_v1": dict(name="csa_v1", implementation="v1", run_dir="runs/csa_v1", tta=True, checkpoints=base_checkpoints)}
    summaries = {}
    for candidate in candidates:
        run = root / candidate["run_dir"]
        if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "completed_all_folds":
            continue
        plain, tta = {}, {}
        checkpoints = []
        for index in range(5):
            path = run / f"fold_{index}/best.pt"
            model, checkpoint = load_checkpoint_model(root, path, "lab", device)
            if checkpoint["dataset_sha256"] != data_hash:
                raise ValueError("Candidate uses a different dataset")
            fold = checkpoint["fold"]
            samples = load_lab_samples(root, fold["validation_ids"], checkpoint["config"]["data"])
            with np.load(run / f"fold_{index}/best_predictions.npz", allow_pickle=False) as saved:
                saved_ids, saved_probabilities = saved["ids"].tolist(), saved["probabilities"].copy()
            if saved_ids != fold["validation_ids"]:
                raise ValueError("OOF archive/fold mismatch")
            for sample, reference in zip(samples, saved_probabilities):
                p, augmented = predict_variants(model, sample, device, True)
                if sample.image_id in fold["train_ids"] or sample.image_id in plain:
                    raise ValueError("Invalid held-out prediction")
                if np.max(np.abs(p - reference)) > 1e-6:
                    raise ValueError("Checkpoint inference does not replay archived OOF predictions")
                plain[sample.image_id], tta[sample.image_id] = p, augmented
            checkpoints.append(dict(fold=index, path=path.relative_to(root).as_posix(), sha256=sha256(path), best_step=checkpoint["step"]))
            del model
        plain_array, tta_array = np.stack([plain[i] for i in ids]), np.stack([tta[i] for i in ids])
        plain_score = evaluate_probabilities(list(plain_array), targets, ids)
        tta_score = evaluate_probabilities(list(tta_array), targets, ids)
        use_tta = tta_score["calibrated_mean_dice"] >= plain_score["calibrated_mean_dice"] + 0.001
        probability_sets[candidate["name"]] = tta_array if use_tta else plain_array
        model_records[candidate["name"]] = dict(name=candidate["name"], implementation="lab",
                                                 run_dir=candidate["run_dir"], tta=use_tta, checkpoints=checkpoints)
        summaries[candidate["name"]] = dict(plain=plain_score, flip_tta=tta_score, tta_selected=use_tta)
        np.savez_compressed(study / f"{candidate['name']}_oof_selected.npz", ids=np.asarray(ids), probabilities=probability_sets[candidate["name"]])
        print(json.dumps(dict(event="model_evaluated", name=candidate["name"],
                              dice_at_0_5=plain_score["mean_dice_at_0_5"],
                              selected_calibrated_dice=(tta_score if use_tta else plain_score)["calibrated_mean_dice"])), flush=True)
    names = list(probability_sets)
    combinations = [(name,) for name in names] + list(itertools.combinations(names, 2))
    if len(names) > 2:
        combinations.append(tuple(names))
    fusion_results = []
    for members in combinations:
        probabilities = np.mean(np.stack([probability_sets[name] for name in members]), axis=0, dtype=np.float32)
        metrics = evaluate_probabilities(list(probabilities), targets, ids)
        fusion_results.append(dict(members=list(members), weights=[1 / len(members)] * len(members), metrics=metrics))
    winner = max(fusion_results, key=lambda r: r["metrics"]["calibrated_mean_dice"])
    baseline_score = evaluate_probabilities(list(base), targets, ids)["calibrated_mean_dice"]
    best_score = winner["metrics"]["calibrated_mean_dice"]
    improved = best_score >= baseline_score + 0.001
    result = dict(completed_at=now(), baseline_v1_calibrated_oof_dice=baseline_score,
                  best_calibrated_oof_dice=best_score, gain=best_score - baseline_score,
                  winner_members=winner["members"], winner_threshold=winner["metrics"]["calibrated_threshold"],
                  improvement_requirement=0.001, candidate_passes_improvement_gate=improved,
                  new_submission_exported=False,
                  note="Image-level OOF used for model/checkpoint/TTA/threshold/fusion selection; not an independent platform score")
    write_json(study / "model_evaluations.json", summaries)
    write_json(study / "fusion_results.json", fusion_results)
    write_json(study / "result.json", result)
    lines = ["# Additional-model search results", "", "All results use the original five image-level folds.", "",
             "| Equal-weight members | Dice @ 0.5 | Calibrated Dice | Threshold |", "| --- | ---: | ---: | ---: |"]
    for row in sorted(fusion_results, key=lambda r: r["metrics"]["calibrated_mean_dice"], reverse=True):
        m = row["metrics"]
        lines.append(f"| {', '.join(row['members'])} | {m['mean_dice_at_0_5']:.6f} | {m['calibrated_mean_dice']:.6f} | {m['calibrated_threshold']:.2f} |")
    lines += ["", result["note"], "", "Original v1 submission remains preserved. A new CSV is generated only for a gain of at least 0.001."]
    (study / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not improved:
        return result
    output.mkdir(exist_ok=True)
    members = [model_records[name] for name in winner["members"]]
    threshold = winner["metrics"]["calibrated_threshold"]
    winner_oof = np.mean(np.stack([probability_sets[name] for name in winner["members"]]), axis=0, dtype=np.float32)
    np.savez_compressed(output / "oof_predictions.npz", ids=np.asarray(ids), probabilities=winner_oof)
    write_json(output / "oof_metrics.json", winner["metrics"])
    checks = generate_submission(root, output, members, threshold, data_hash, device)
    write_json(output / "inference_recipe.json", dict(members=members, threshold=threshold, target_hw=[512, 512],
                                                       foreground="vessel", order="row_major_C", seed=20261103,
                                                       dataset_sha256=data_hash, torch=torch.__version__))
    torch.cuda.empty_cache()
    with (output / "independent_rebuild.log").open("w", encoding="utf-8") as log:
        command = [sys.executable, "-u", "-m", "vessel_lab.export_study", "--recipe",
                   str((output / "inference_recipe.json").relative_to(root)), "--output",
                   str((output / "rebuild_check").relative_to(root))]
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    if sha256(output / "submission.csv") != sha256(output / "rebuild_check/submission.csv"):
        raise ValueError("Improved submission failed independent frozen-recipe reproduction")
    checks["independent_rebuild_exact_csv_match"] = True
    write_json(output / "submission_checks.json", checks)
    (root / "submission_v2.csv").write_bytes((output / "submission.csv").read_bytes())
    result["submission_csv"] = "submission_v2.csv"
    result["submission_sha256"] = checks["csv_sha256"]
    result["new_submission_exported"] = True
    result["independent_rebuild_passed"] = True
    write_json(output / "result.json", result)
    write_json(study / "result.json", result)
    instructions = ["# 第二轮优化提交", "",
                    f"原v1校准OOF Dice：{baseline_score:.6f}；本轮选择结果：{best_score:.6f}。",
                    "这是参与模型、checkpoint、TTA、阈值和融合选择的本地OOF结果，不是独立平台分数。", "",
                    f"采用模型：{', '.join(winner['members'])}；各模型等权、各模型内部五折等权，阈值{threshold:.2f}。",
                    "", "上传文件：项目根目录 submission_v2.csv；原submission.csv及v1复现包保持不变。",
                    "512×512血管前景RLE、行优先、从1开始；与题目编码函数一致且所有PNG/RLE回解检查通过。",
                    "独立Python进程使用冻结配置重建CSV，SHA-256完全一致。未上传平台。", "",
                    "冻结配置：inference_recipe.json；检查结果：submission_checks.json；逐图验证：oof_metrics.json。",
                    "", "复现命令（项目根目录，使用各run记录的PyTorch/torchvision环境）：", "",
                    "```powershell", "python -m vessel_lab.export_study --recipe submission_v2/inference_recipe.json --output submission_v2/reproduced", "```", "",
                    "模型权重、原图数据、vessel及vessel_lab源码均需保留；分割checkpoint已包含全部编码器参数，重新推理无需再下载ImageNet权重。"]
    (output / "README.md").write_text("\n".join(instructions) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("optimization_v2"))
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submission_v2/rebuilt"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.recipe:
        recipe = json.loads((root / args.recipe).read_text(encoding="utf-8"))
        torch.set_num_threads(4)
        seed_everything(20261103, True)
        data_hash = verify_dataset(root)
        if recipe["dataset_sha256"] != data_hash or recipe["target_hw"] != [512, 512]:
            raise ValueError("Frozen inference data/geometry mismatch")
        result = generate_submission(root, root / args.output, recipe["members"], float(recipe["threshold"]), data_hash, torch.device("cuda"))
        print(json.dumps(result, indent=2))
    else:
        study = root / args.study
        candidates = json.loads((study / "completed_models.json").read_text(encoding="utf-8"))
        print(json.dumps(compare_and_export(root, study, candidates, root / "submission_v2"), indent=2))
