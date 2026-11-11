"""Audit five held-out folds, compare TTA, and export a verified submission.

Usage: python -m tools.prepare_submission --output submission
       python -m tools.prepare_submission --output rebuilt --recipe submission/inference_recipe.json
"""

from __future__ import annotations

from vessel.runtime import now, seed_everything, source_digest, write_json

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import torchvision

from vessel.data import get_fold, load_samples, verify_dataset
from vessel.metrics import evaluate_probabilities
from vessel.model import VesselUNet
from vessel.rle import decode, encode


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resize_probabilities(probabilities: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = torch.from_numpy(np.asarray(probabilities, dtype=np.float32))
    return F.interpolate(array[:, None], size=shape, mode="bilinear", align_corners=False)[:, 0].numpy()


@torch.inference_mode()
def predict_variants(model: VesselUNet, sample, device: torch.device, use_tta: bool) -> tuple[np.ndarray, np.ndarray]:
    image = sample.image[None].to(device, memory_format=torch.channels_last)
    transforms = [(), (2,), (3,), (2, 3)] if use_tta else [()]
    total, identity = None, None
    for dimensions in transforms:
        transformed = image.flip(dimensions) if dimensions else image
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            probability = torch.sigmoid(model(transformed).float())
        if dimensions:
            probability = probability.flip(dimensions)
        if identity is None:
            identity = probability
            total = probability.clone()
        else:
            total.add_(probability)
    h, w = sample.original_hw
    return (identity[0, 0, :h, :w].cpu().numpy(),
            (total / len(transforms))[0, 0, :h, :w].cpu().numpy())


def read_official_encoder(path: Path):
    # Execute the provided function body unchanged, avoiding its unrelated
    # top-level pandas import. The actual torchvision ToTensor is used.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "turn_to_str")
    namespace = {"np": np, "Image": Image, "torchvision": torchvision}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["turn_to_str"]


def validate_csv(path: Path, png_directory: Path, expected_ids: list[str], shape: tuple[int, int]) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Id", "Predicted"]:
            raise ValueError("Submission columns differ from sample_submission.csv")
        rows = list(reader)
    if [row["Id"] for row in rows] != expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Submission IDs are missing, duplicated, or out of sample order")
    details = []
    for row in rows:
        decoded = decode(row["Predicted"], shape)
        with Image.open(png_directory / f"{row['Id']}.png") as image:
            if image.size != (shape[1], shape[0]) or image.mode != "L":
                raise ValueError("Exported mask geometry or mode is wrong")
            pixels = np.array(image)
        if not set(np.unique(pixels).tolist()).issubset({0, 255}):
            raise ValueError("Exported PNG is not strictly binary")
        if not np.array_equal(decoded, pixels == 255):
            raise ValueError("CSV does not decode to the saved vessel mask")
        fraction = float(decoded.mean())
        if not 0 < fraction < 0.5:
            raise ValueError("Unexpected vessel coverage; check foreground polarity")
        details.append(dict(id=row["Id"], vessel_pixels=int(decoded.sum()), vessel_fraction=fraction,
                            rle_runs=len(row["Predicted"].split()) // 2))
    return dict(passed=True, columns=["Id", "Predicted"], rows=len(rows),
                ids_in_sample_order=True, target_hw=list(shape), rle_start_index=1,
                rle_order="row_major_C", rle_foreground="vessel", masks_binary=True,
                all_rle_runs_sorted_positive_nonoverlapping_in_bounds=True,
                all_decode_roundtrips_match_png=True, csv_sha256=sha256(path), per_image=details)


def write_table(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, default=Path("runs/csa_v1"))
    parser.add_argument("--output", type=Path, default=Path("submission"))
    parser.add_argument("--recipe", type=Path, help="Use the frozen TTA/threshold recipe; do not reselect on OOF")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    run, output = (root / args.run_dir).resolve(), (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    seed_everything(20261103, True)
    device = torch.device(args.device)
    dataset_hash = verify_dataset(root)
    state = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if state["state"] != "completed_all_folds":
        raise ValueError("The five-fold run is not complete")
    checkpoints, checkpoint_records = [], []
    for index in range(5):
        path = run / f"fold_{index}/best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["dataset_sha256"] != dataset_hash or checkpoint["fold"] != get_fold(root, index, dataset_hash):
            raise ValueError("Checkpoint data/fold provenance mismatch")
        if checkpoint["source_sha256"] != source_digest():
            raise ValueError("Training package differs from the frozen checkpoint source")
        checkpoints.append(checkpoint)
        checkpoint_records.append(dict(fold=index, path=path.relative_to(root).as_posix(), sha256=sha256(path),
                                       best_step=checkpoint["step"], native_dice_at_0_5=checkpoint["best_score"]))
    write_json(output / "checkpoints.json", checkpoint_records)
    pad_hw = checkpoints[0]["config"]["data"]["pad_hw"]
    native_hw = (584, 565)
    target_hw = (512, 512)
    fold_documents = json.loads((root / "experiments/folds_v1.json").read_text(encoding="utf-8"))
    frozen = None
    validation_report = None
    if args.recipe:
        frozen = json.loads((root / args.recipe).read_text(encoding="utf-8"))
        if frozen["dataset_sha256"] != dataset_hash or frozen["checkpoints"] != checkpoint_records:
            raise ValueError("Frozen recipe does not identify these exact inputs and weights")
        threshold, tta = float(frozen["threshold"]), bool(frozen["tta"])
        if frozen["target_hw"] != list(target_hw):
            raise ValueError("Frozen recipe uses different submission geometry")
    else:
        plain, augmented = {}, {}
        maximum_replay_error = 0.0
        for index, checkpoint in enumerate(checkpoints):
            model = VesselUNet(checkpoint["config"]["model"]).to(device, memory_format=torch.channels_last)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            samples = load_samples(root, checkpoint["fold"]["validation_ids"], pad_hw)
            with np.load(run / f"fold_{index}/best_predictions.npz", allow_pickle=False) as archived:
                archived_ids = archived["ids"].tolist()
                archived_probabilities = archived["probabilities"].copy()
            if archived_ids != checkpoint["fold"]["validation_ids"]:
                raise ValueError("Archived predictions are not this fold's held-out images")
            for sample, saved_probability in zip(samples, archived_probabilities):
                identity, tta_probability = predict_variants(model, sample, device, True)
                error = float(np.max(np.abs(identity - saved_probability)))
                maximum_replay_error = max(maximum_replay_error, error)
                if error > 1e-6:
                    raise ValueError(f"Checkpoint inference replay differs: fold {index}, ID {sample.image_id}, {error}")
                if sample.image_id in plain or sample.image_id in checkpoint["fold"]["train_ids"]:
                    raise ValueError("Invalid OOF provenance")
                plain[sample.image_id], augmented[sample.image_id] = identity, tta_probability
            print(f"verified and evaluated held-out fold {index}", flush=True)
        ids = sorted(plain, key=int)
        if ids != [str(i) for i in range(1, 21)]:
            raise ValueError("OOF predictions must cover all 20 original training images exactly once")
        targets = []
        for image_id in ids:
            with Image.open(root / "train/label" / f"{image_id}.jpg") as image:
                targets.append(np.array(image.convert("L")) == 255)
        predictions = {"no_tta": np.stack([plain[i] for i in ids]), "flip_tta": np.stack([augmented[i] for i in ids])}
        native_results = {name: evaluate_probabilities(list(values), targets, ids) for name, values in predictions.items()}
        # Predeclared operational criterion: retain TTA if its calibrated native
        # OOF Dice improves by at least 0.001 (not a significance claim).
        gain = native_results["flip_tta"]["calibrated_mean_dice"] - native_results["no_tta"]["calibrated_mean_dice"]
        tta = gain >= 0.001
        chosen = "flip_tta" if tta else "no_tta"
        threshold = native_results[chosen]["calibrated_threshold"]
        resized_targets = [np.array(Image.fromarray(y.astype(np.uint8) * 255).resize((512, 512), Image.Resampling.NEAREST)) == 255 for y in targets]
        resized_results = {name: evaluate_probabilities(list(resize_probabilities(values, target_hw)), resized_targets, ids)
                           for name, values in predictions.items()}
        validation_report = dict(created_at=now(), dataset_sha256=dataset_hash,
                                 evaluation_unit="original_image; exactly one held-out model per image",
                                 native_hw=list(native_hw), selected_variant=chosen, selected_threshold=threshold,
                                 tta_calibrated_native_gain=gain,
                                 tta_selection_rule="calibrated native OOF mean Dice gain >= 0.001",
                                 native=native_results, diagnostic_512=resized_results,
                                 diagnostic_512_note="Labels resized by nearest neighbor for a local geometry check; platform labels are unavailable",
                                 maximum_checkpoint_replay_probability_error=maximum_replay_error,
                                 selection_note="Checkpoint, TTA and threshold selection used these validation folds; no independent test score is available")
        write_json(output / "validation_report.json", validation_report)
        np.savez_compressed(output / "oof_selected.npz", ids=np.asarray(ids), probabilities=predictions[chosen])
        rows = []
        for index, image_id in enumerate(ids):
            fold_index = next(f["fold"] for f in fold_documents["folds"] if image_id in f["validation_ids"])
            rows.append(dict(id=image_id, fold=fold_index,
                             no_tta_dice_at_0_5=native_results["no_tta"]["per_image"][index]["dice"],
                             flip_tta_dice_at_0_5=native_results["flip_tta"]["per_image"][index]["dice"]))
        write_table(output / "oof_per_image.csv", rows, list(rows[0]))
        write_table(output / "fold_results.csv", [dict(fold=r["fold"], best_step=r["best_step"],
                                                       native_dice_at_0_5=r["native_dice_at_0_5"],
                                                       elapsed_seconds=state["folds"][r["fold"]]["elapsed_seconds"])
                                                  for r in checkpoint_records],
                    ["fold", "best_step", "native_dice_at_0_5", "elapsed_seconds"])
        print(json.dumps(dict(selected_variant=chosen, threshold=threshold,
                              no_tta_native_dice=native_results["no_tta"]["mean_dice_at_0_5"],
                              tta_native_dice=native_results["flip_tta"]["mean_dice_at_0_5"],
                              selected_calibrated_native_dice=native_results[chosen]["calibrated_mean_dice"])), flush=True)

    recipe = dict(schema_version=1, created_at=now(), dataset_sha256=dataset_hash,
                  model="independently implemented CSA U-Net, five best fold models",
                  checkpoints=checkpoint_records, seed=20261103, deterministic=True,
                  input_padding_hw=pad_hw, native_hw=list(native_hw), target_hw=list(target_hw),
                  tta=tta, tta_transforms=["identity", "vertical_flip", "horizontal_flip", "both_flips"] if tta else ["identity"],
                  fold_fusion="equal-weight mean of vessel probabilities", threshold=threshold,
                  resize="bilinear probability resizing, align_corners=False, before thresholding",
                  csv_columns=["Id", "Predicted"], rle_order="row_major_C", rle_start_index=1,
                  rle_foreground="vessel; converter PNG uses 255 for vessels and 0 for background",
                  submission_geometry_basis="provided segmentation_to_csv.py; no platform endpoint available",
                  screenshot_polarity_conflict="Screenshot says black vessels; actual labels and converter encode white vessels. CSV encodes vessel regions; black display PNGs are separate.",
                  torch=torch.__version__, torchvision=importlib.metadata.version("torchvision"))
    write_json(output / "inference_recipe.json", recipe)
    with (root / "sample_submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        test_ids = [row["Id"] for row in csv.DictReader(stream)]
    test_samples = load_samples(root, test_ids, pad_hw, split="test")
    models = []
    for checkpoint in checkpoints:
        model = VesselUNet(checkpoint["config"]["model"]).to(device, memory_format=torch.channels_last)
        model.load_state_dict(checkpoint["model"])
        models.append(model.eval())
    test_probabilities = []
    for sample in test_samples:
        fused = np.zeros(native_hw, dtype=np.float32)
        for model in models:
            identity, transformed = predict_variants(model, sample, device, tta)
            fused += transformed if tta else identity
        fused /= len(models)
        if not np.isfinite(fused).all() or fused.min() < 0 or fused.max() > 1:
            raise ValueError("Test probability map is invalid")
        test_probabilities.append(fused)
        print(f"predicted test ID {sample.image_id} with {len(models)} folds; TTA={tta}", flush=True)
    probabilities = np.stack(test_probabilities)
    np.savez_compressed(output / "test_probabilities_native.npz", ids=np.asarray(test_ids), probabilities=probabilities)
    target_probabilities = resize_probabilities(probabilities, target_hw)
    converter_images = output / "image"
    display_images = output / "black_vessel_display"
    decoded_images = output / "decoded_submission"
    for directory in (converter_images, display_images, decoded_images):
        directory.mkdir(exist_ok=True)
    rows = []
    for image_id, probability in zip(test_ids, target_probabilities):
        mask = probability >= threshold
        Image.fromarray(mask.astype(np.uint8) * 255).save(converter_images / f"{image_id}.png")
        Image.fromarray((~mask).astype(np.uint8) * 255).save(display_images / f"{image_id}.png")
        encoded = encode(mask)
        Image.fromarray(decode(encoded, target_hw).astype(np.uint8) * 255).save(decoded_images / f"{image_id}.png")
        rows.append(dict(Id=image_id, Predicted=encoded))
    write_table(output / "submission.csv", rows, ["Id", "Predicted"])
    official_function = read_official_encoder(root / "segmentation_to_csv.py")
    official_strings = official_function([str(converter_images / f"{i}.png") for i in test_ids])
    if official_strings != [row["Predicted"] for row in rows]:
        raise ValueError("RLE differs from the actual provided encoder function")
    checks = validate_csv(output / "submission.csv", converter_images, test_ids, target_hw)
    checks.update(provided_turn_to_str_function_exact_match=True,
                  original_script_sha256=sha256(root / "segmentation_to_csv.py"),
                  same_size_512_input_avoids_implicit_mask_resizing=True,
                  test_probabilities_sha256=sha256(output / "test_probabilities_native.npz"),
                  completed_at=now())
    write_json(output / "submission_checks.json", checks)
    lock = (run / "requirements.lock.txt").read_text(encoding="utf-8")
    if "torchvision==" not in lock:
        lock += f"torchvision=={importlib.metadata.version('torchvision')}\n"
    (output / "requirements_submission.lock.txt").write_text(lock, encoding="utf-8")
    print(json.dumps(dict(output=str(output), rows=checks["rows"], csv_sha256=checks["csv_sha256"],
                          threshold=threshold, tta=tta, all_checks_passed=True)), flush=True)


if __name__ == "__main__":
    main()
