"""Frozen global features aligned to a trainable, high-resolution local ROI.

This is a small concatenation baseline inspired by ZIAN, not a ZIAN reproduction.
The coarse model must belong to the same held-out fold as the fine model.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import FoveaDataset, homogeneous
from .model import ConvBlock, FoveaNet, spatial_grid


def fuse(block, value, skip):
    value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    return block(torch.cat([value, skip], dim=1))


def decoder_context(model, image):
    """Return the 1/8 decoder representation and the shallow skip connections."""
    enc = model.encoder
    x0 = enc.relu(enc.bn1(enc.conv1(image)))
    x1 = enc.layer1(enc.maxpool(x0))
    x2 = enc.layer2(x1)
    x3 = enc.layer3(x2)
    x4 = enc.layer4(x3)
    value = fuse(model.up3, x4, x3)
    return fuse(model.up2, value, x2), x1, x0


def align_context(features, local_to_global, local_size, global_size, output_shape):
    """Sample global features at the physical locations of local feature centers.

    Matrices map local input pixel centers to global input pixel centers. The
    half-pixel convention matches cv2.resize and grid_sample(align_corners=False).
    """
    height, width = output_shape
    xx, yy = spatial_grid(height, width, local_size, features.device)
    points = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    original_grid = torch.einsum("bij,hwj->bhwi", local_to_global.float(), points)
    grid = 2 * (original_grid + 0.5) / global_size - 1
    return F.grid_sample(features.float(), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)


class ContextFoveaNet(FoveaNet):
    def __init__(self, context_enabled=True, global_size=512):
        super().__init__("heatmap", pretrained=False)
        self.context_enabled = context_enabled
        self.global_size = global_size
        # Both controls have exactly the same trainable modules and FLOPs.
        self.fusion = nn.Sequential(ConvBlock(128, 64), nn.Conv2d(64, 64, 1))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, image, context, local_to_global):
        value, x1, x0 = decoder_context(self, image)
        aligned = align_context(context, local_to_global, image.shape[-1],
                                self.global_size, value.shape[-2:])
        if not self.context_enabled:
            aligned = aligned * 0
        value = value + self.fusion(torch.cat([value, aligned], dim=1))
        value = fuse(self.up1, value, x1)
        value = fuse(self.up0, value, x0)
        return {"heatmap": self.head(value)}


class ContextDataset(FoveaDataset):
    def __init__(self, rows, size, training, roi, features, anchors=None):
        super().__init__(rows, size, "fine", training, roi, anchors)
        self.features = features
        expected = {r["id"] for r in rows}
        if not expected.issubset(features):
            raise ValueError("Missing frozen context for dataset IDs")

    def __getitem__(self, index):
        result = super().__getitem__(index)
        row = self.rows[index]
        global_matrix = np.asarray(row["transform"], np.float64)
        inverse = homogeneous(result["inverse"].numpy().astype(np.float64))
        result["local_to_global"] = torch.tensor((global_matrix @ inverse)[:2], dtype=torch.float32)
        result["context"] = self.features[row["id"]]
        return result
