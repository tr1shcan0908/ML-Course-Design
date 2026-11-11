"""Rebuild validation/test predictions from this project's saved checkpoints."""

from .runtime import now, write_json

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .data import load_samples, verify_dataset
from .metrics import evaluate_probabilities
from .model import VesselUNet
from .rle import decode, encode


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", action="store_true", help="Four flip variants, inverse-transformed before averaging")
    parser.add_argument("--csv", action="store_true", help="Write a candidate CSV; requires explicit target height and width")
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--target-width", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must lie in [0,1]")
    if args.csv and (not args.target_height or not args.target_width):
        raise ValueError("Candidate CSV requires explicit target dimensions; platform convention is unresolved")
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    dataset_hash = verify_dataset(root)
    checkpoints = [torch.load(p, map_location="cpu", weights_only=False) for p in args.checkpoints]
    if any(c["dataset_sha256"] != dataset_hash for c in checkpoints):
        raise ValueError("Checkpoint belongs to a different dataset")
    pad_hw = checkpoints[0]["config"]["data"]["pad_hw"]
    if any(c["config"]["data"]["pad_hw"] != pad_hw for c in checkpoints):
        raise ValueError("Ensemble members use different input geometry")
    if args.split == "validation":
        ids = checkpoints[0]["fold"]["validation_ids"]
        if any(c["fold"]["validation_ids"] != ids for c in checkpoints):
            raise ValueError("Validation ensemble members must share the same held-out fold")
        if any(set(ids) & set(c["fold"]["train_ids"]) for c in checkpoints):
            raise ValueError("Validation images were used by an ensemble member for training")
        samples = load_samples(root, ids, pad_hw)
    else:
        with (root / "sample_submission.csv").open(encoding="utf-8-sig", newline="") as stream:
            ids = [row["Id"] for row in csv.DictReader(stream)]
        samples = load_samples(root, ids, pad_hw, split="test")
    models = []
    for checkpoint in checkpoints:
        model = VesselUNet(checkpoint["config"]["model"]).to(device, memory_format=torch.channels_last)
        model.load_state_dict(checkpoint["model"])
        models.append(model.eval())
    probabilities, csv_rows = [], []
    for sample in samples:
        image = sample.image[None].to(device, memory_format=torch.channels_last)
        total = torch.zeros_like(image[:, :1], dtype=torch.float32)
        flips = [(), (2,), (3,), (2, 3)] if args.tta else [()]
        for model in models:
            for dimensions in flips:
                transformed = image.flip(dimensions) if dimensions else image
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    probability = torch.sigmoid(model(transformed).float())
                if dimensions:
                    probability = probability.flip(dimensions)
                total += probability
        h, w = sample.original_hw
        native = (total / (len(models) * len(flips)))[0, 0, :h, :w].cpu().numpy()
        probabilities.append(native)
        mask = native >= args.threshold
        for subdir, pixels in (("native_white_vessels", mask.astype(np.uint8) * 255),
                               ("native_black_vessels", (~mask).astype(np.uint8) * 255)):
            destination = output / subdir
            destination.mkdir(exist_ok=True)
            Image.fromarray(pixels).save(destination / f"{sample.image_id}.png")
        if args.csv:
            resized = F.interpolate(torch.from_numpy(native.copy())[None, None],
                                    size=(args.target_height, args.target_width), mode="bilinear", align_corners=False)[0, 0].numpy()
            submission_mask = resized >= args.threshold
            encoded = encode(submission_mask)
            if not np.array_equal(decode(encoded, submission_mask.shape), submission_mask):
                raise AssertionError("RLE roundtrip failed")
            csv_rows.append({"Id": sample.image_id, "Predicted": encoded})
        print(f"predicted {args.split} image {sample.image_id}", flush=True)
    np.savez_compressed(output / "probabilities.npz", ids=np.asarray(ids), probabilities=np.stack(probabilities))
    if args.csv:
        with (output / "candidate_submission.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["Id", "Predicted"])
            writer.writeheader()
            writer.writerows(csv_rows)
    if args.split == "validation":
        targets = [s.mask[0, :s.original_hw[0], :s.original_hw[1]].numpy().astype(bool) for s in samples]
        write_json(output / "validation_metrics.json", evaluate_probabilities(probabilities, targets, ids))
    provenance = dict(created_at=now(), dataset_sha256=dataset_hash, split=args.split, image_ids=ids,
                      threshold=args.threshold, tta=args.tta, rle_foreground="vessel", rle_order="row_major_C",
                      target_hw=[args.target_height, args.target_width] if args.csv else None,
                      platform_submission_conventions_verified=False,
                      checkpoints=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), step=c["step"], fold=c["fold"]["fold"])
                                   for p, c in zip(args.checkpoints, checkpoints)])
    if args.csv:
        provenance["csv_sha256"] = hashlib.sha256((output / "candidate_submission.csv").read_bytes()).hexdigest()
    write_json(output / "provenance.json", provenance)


if __name__ == "__main__":
    main()
