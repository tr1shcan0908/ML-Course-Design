"""Trainable ConvNeXt last-stage + foreground SE residual module.

The normalized-color prior below is an experimental heuristic, not a trained
segmentation model or a claim of improved segmentation accuracy.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny

from plant_fusion_modules import ResidualFusionAdapter, foreground_prior, weighted_pool


MODES = {
    "control": "原模型同预算继续训练",
    "old_prior_se": "旧前景 + SE 联合微调",
    "normalized_prior_se": "归一化颜色前景 + SE 联合微调",
}
PRIOR_VERSION = "relative_rgb_seeded_growth4_dilate3_smooth5_floor005_v2"


def shape_descriptors(rgb, binary):
    area = binary.mean((2, 3))
    missing = area < 0.0005
    total = binary.sum((2, 3)).clamp_min(1)
    height, width = rgb.shape[-2:]
    yy = torch.linspace(-1, 1, height, device=rgb.device)[None, None, :, None]
    xx = torch.linspace(-1, 1, width, device=rgb.device)[None, None, None, :]
    mx = (binary * xx).sum((2, 3)) / total
    my = (binary * yy).sum((2, 3)) / total
    dx, dy = xx - mx[:, :, None, None], yy - my[:, :, None, None]
    vx = (binary * dx.square()).sum((2, 3)) / total
    vy = (binary * dy.square()).sum((2, 3)) / total
    cov = (binary * dx * dy).sum((2, 3)) / total
    eccentricity = ((vx - vy).square() + 4 * cov.square()).sqrt() / (vx + vy).clamp_min(1e-6)
    colors = (rgb * binary).sum((2, 3)) / total
    return torch.cat((area, mx.abs(), my.abs(), vx, vy, eccentricity, colors, missing.float()), 1)


def normalized_foreground_prior(rgb):
    rgb = rgb.float().clamp(0, 1)
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    total = (r + g + b).clamp_min(1e-6)
    fraction, rg, gb = g / total, (g - r) / total, (g - b) / total
    visible = total > 0.04
    seed = (fraction > 0.37) & (rg > 0.025) & (gb > 0.025) & visible
    candidate = (fraction > 0.355) & (rg > 0.012) & (gb > 0.015) & visible
    binary = seed.float()
    for _ in range(4):
        binary = torch.maximum(binary, F.max_pool2d(binary, 3, 1, 1) * candidate)
    shape = shape_descriptors(rgb, binary)
    weights = 0.05 + 0.95 * F.avg_pool2d(F.max_pool2d(binary, 3, 1, 1), 5, 1, 2)
    weights = torch.where(shape[:, -1, None, None, None].bool(), torch.ones_like(weights), weights)
    return weights, shape, binary


class JointForegroundConvNeXt(nn.Module):
    def __init__(self, class_count, mode="control", width=32, dropout=0.15):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.backbone = convnext_tiny(weights=None, num_classes=class_count)
        self.adapter = None if mode == "control" else ResidualFusionAdapter(
            {"cn_global": 768, "cn_fg": 768, "cn_mid_fg": 384, "shape": 10},
            class_count, width=width, se=True, dropout=dropout)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for index in (6, 7):
            for parameter in self.backbone.features[index].parameters():
                parameter.requires_grad_(True)
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        # Frozen early stages also retain deterministic inference behavior.
        for block in list(self.backbone.features)[:6]:
            block.eval()
        return self

    def forward(self, images, prior_rgb):
        with torch.no_grad():
            value = images
            for block in list(self.backbone.features)[:6]:
                value = block(value)
            middle = value
        value = self.backbone.features[7](self.backbone.features[6](value))
        pooled = self.backbone.avgpool(value)
        logits = self.backbone.classifier(pooled)
        if self.adapter is None:
            return logits, torch.zeros_like(logits)
        with torch.no_grad():
            prior = foreground_prior if self.mode == "old_prior_se" else normalized_foreground_prior
            mask, shape, _ = prior(prior_rgb.float())
        blocks = {"cn_global": self.backbone.classifier[:2](pooled).float(),
                  "cn_fg": self.backbone.classifier[:2](weighted_pool(value, mask)[:, :, None, None]).float(),
                  "cn_mid_fg": weighted_pool(middle, mask), "shape": shape}
        return self.adapter(blocks, logits)

    def parameter_groups(self, lr_stage, lr_classifier, lr_adapter):
        groups = [{"params": list(self.backbone.features[6].parameters()) + list(self.backbone.features[7].parameters()),
                   "lr": lr_stage},
                  {"params": self.backbone.classifier.parameters(), "lr": lr_classifier}]
        if self.adapter is not None:
            groups.append({"params": self.adapter.parameters(), "lr": lr_adapter})
        return groups
