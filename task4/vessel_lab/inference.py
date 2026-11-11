"""Optional eight-direction inference, separate from frozen training sources."""

import numpy as np
import torch

from tools.prepare_submission import predict_variants


def transform(image, transpose, flips):
    value = image.transpose(-2, -1) if transpose else image
    return value.flip(flips) if flips else value


def invert(image, transpose, flips):
    value = image.flip(flips) if flips else image
    return value.transpose(-2, -1) if transpose else value


def tile_positions(length, patch, stride):
    if not 0 < stride <= patch <= length:
        raise ValueError("Tiles require 0 < stride <= patch <= image size")
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


@torch.inference_mode()
def predict_tiles_one(model, image, device, patch, stride, batch_size):
    h, w = image.shape[-2:]
    positions = [(top, left) for top in tile_positions(h, patch, stride) for left in tile_positions(w, patch, stride)]
    window = torch.hann_window(patch, periodic=False, device=device)
    window = (window[:, None] * window[None, :]).clamp_min(.001)[None, None]
    total = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)
    weights = torch.zeros_like(total)
    for start in range(0, len(positions), batch_size):
        group = positions[start:start + batch_size]
        patches = torch.cat([image[:, :, top:top+patch, left:left+patch] for top, left in group], dim=0)
        patches = patches.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            probabilities = torch.sigmoid(model(patches).float())
        for probability, (top, left) in zip(probabilities, group):
            total[:, :, top:top+patch, left:left+patch].add_(probability[None] * window)
            weights[:, :, top:top+patch, left:left+patch].add_(window)
    if not torch.all(weights > 0):
        raise ValueError("Tiled inference left uncovered pixels")
    return total / weights


@torch.inference_mode()
def predict_tiled(model, sample, device, member):
    mode = member.get("tta_mode", "flip4" if member.get("tta", False) else "identity")
    if mode not in ("identity", "flip4", "d4"):
        raise ValueError(f"Unknown TTA mode: {mode}")
    image = sample.image[None].to(device, memory_format=torch.channels_last)
    transforms = [(False, ())]
    if mode in ("flip4", "d4"):
        transforms = [(False, flips) for flips in ((), (2,), (3,), (2,3))]
    if mode == "d4":
        transforms += [(True, flips) for flips in ((), (2,), (3,), (2,3))]
    total = None
    for transpose, flips in transforms:
        mapped = transform(image, transpose, flips)
        probability = predict_tiles_one(model, mapped, device, member.get("tile_patch", 288),
                                        member.get("tile_stride", 144), member.get("tile_batch_size", 4))
        probability = invert(probability, transpose, flips)
        total = probability if total is None else total + probability
    h, w = sample.original_hw
    return (total / len(transforms))[0, 0, :h, :w].cpu().numpy()


@torch.inference_mode()
def predict_d4(model, sample, device):
    image = sample.image[None].to(device, memory_format=torch.channels_last)
    identity, total = None, None
    for transpose in (False, True):
        for flips in ((), (2,), (3,), (2, 3)):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                probability = torch.sigmoid(model(transform(image, transpose, flips)).float())
            probability = invert(probability, transpose, flips)
            if identity is None:
                identity, total = probability, probability.clone()
            else:
                total.add_(probability)
    h, w = sample.original_hw
    return identity[0, 0, :h, :w].cpu().numpy(), (total / 8)[0, 0, :h, :w].cpu().numpy()


def predict_member(model, sample, device, member):
    spatial = member.get("spatial_mode", "full")
    if spatial == "tiled":
        return predict_tiled(model, sample, device, member)
    if spatial == "blend_full_tiled":
        full = predict_member(model, sample, device, dict(member, spatial_mode="full"))
        tiled = predict_tiled(model, sample, device, member)
        return (full + tiled) / 2
    if spatial != "full":
        raise ValueError(f"Unknown spatial inference mode: {spatial}")
    mode = member.get("tta_mode", "flip4" if member.get("tta", False) else "identity")
    if mode == "d4":
        return predict_d4(model, sample, device)[1]
    if mode not in ("identity", "flip4"):
        raise ValueError(f"Unknown TTA mode: {mode}")
    identity, augmented = predict_variants(model, sample, device, mode == "flip4")
    return augmented if mode == "flip4" else identity


def validate_weights(members):
    if not members:
        raise ValueError("No ensemble members")
    if not any("weight" in member for member in members):
        return None
    if not all("weight" in member for member in members):
        raise ValueError("All members must specify weights")
    weights = [float(member["weight"]) for member in members]
    if not all(np.isfinite(w) and w > 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError("Weights must be finite, positive and sum to one")
    return weights
