"""Image-only illumination preprocessing; original v1 data remain untouched."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.nn import functional as F

from vessel.data import load_samples as load_original_samples
from vessel.data import sample_patches, photometric


def load_samples(root: Path, image_ids: list[str], config: dict, split: str = "train"):
    samples = load_original_samples(root, image_ids, config["pad_hw"], split)
    mode = config.get("preprocessing", "rgb")
    if mode == "rgb":
        return samples
    if mode != "green_local":
        raise ValueError(f"Unknown image preprocessing: {mode}")
    for sample in samples:
        h, w = sample.original_hw
        rgb = sample.image[:, :h, :w]
        green = (rgb[1].numpy() * 255).round().astype(np.uint8)
        smooth = np.asarray(Image.fromarray(green).filter(ImageFilter.GaussianBlur(radius=8)), dtype=np.float32) / 255
        channel = np.clip(0.5 + 3.0 * (green.astype(np.float32) / 255 - smooth), 0, 1)
        channel[rgb.amax(dim=0).numpy() < 5 / 255] = 0
        image = torch.from_numpy(channel).repeat(3, 1, 1)
        ph, pw = config["pad_hw"]
        sample.image = F.pad(image, (0, pw - w, 0, ph - h)).contiguous()
    return samples
