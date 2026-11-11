"""Foreground-guided feature modules and a small residual fusion adapter.

Foreground masks are RGB heuristics, not supervised segmentation predictions.
Auxiliary classifiers are discarded; only their image representations are used.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import SqueezeExcitation


MASK_VERSION = "green_prior_dilate3_smooth5_floor005_v1"


def foreground_prior(rgb: torch.Tensor):
    """Return aligned soft weights and class-independent shape descriptors."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    binary = ((g > 45 / 255) & (g > r + 8 / 255) & (g > b + 5 / 255)).float()
    area = binary.mean((2, 3))
    missing = area < 0.0005
    smooth = F.avg_pool2d(F.max_pool2d(binary, 3, 1, 1), 5, 1, 2)
    weights = 0.05 + 0.95 * smooth
    weights = torch.where(missing[:, :, None, None], torch.ones_like(weights), weights)
    height, width = rgb.shape[-2:]
    yy = torch.linspace(-1, 1, height, device=rgb.device)[None, None, :, None]
    xx = torch.linspace(-1, 1, width, device=rgb.device)[None, None, None, :]
    total = binary.sum((2, 3)).clamp_min(1)
    mx = (binary * xx).sum((2, 3)) / total
    my = (binary * yy).sum((2, 3)) / total
    dx, dy = xx - mx[:, :, None, None], yy - my[:, :, None, None]
    vx = (binary * dx.square()).sum((2, 3)) / total
    vy = (binary * dy.square()).sum((2, 3)) / total
    cov = (binary * dx * dy).sum((2, 3)) / total
    discriminant = ((vx - vy).square() + 4 * cov.square()).sqrt()
    eccentricity = discriminant / (vx + vy).clamp_min(1e-6)
    colors = (rgb * binary).sum((2, 3)) / total
    shape = torch.cat((area, mx.abs(), my.abs(), vx, vy, eccentricity, colors, missing.float()), 1)
    return weights, shape, binary


def weighted_pool(feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Area downsampling preserves thin foreground occupancy at coarse resolutions.
    weights = F.adaptive_avg_pool2d(mask.float(), feature_map.shape[-2:])
    return (feature_map.float() * weights).sum((2, 3)) / weights.sum((2, 3)).clamp_min(1e-8)


def feature_blocks(kind: str, model: nn.Module, images: torch.Tensor, mask: torch.Tensor):
    if kind == "convnext":
        value = images
        middle = None
        for index, block in enumerate(model.features):
            value = block(value)
            if index == 5:
                middle = value
        assert middle is not None
        pooled = model.avgpool(value)
        result = {
            "cn_global": model.classifier[:2](pooled).float(),
            "cn_fg": model.classifier[:2](weighted_pool(value, mask)[:, :, None, None]).float(),
            "cn_mid_fg": weighted_pool(middle, mask),
            "cn_logits": model.classifier(pooled).float(),
        }
    elif kind == "mobilenet":
        value = model.features(images)
        result = {"mobile_global": value.float().mean((2, 3)), "mobile_fg": weighted_pool(value, mask)}
    elif kind == "resnet":
        value = model.maxpool(model.relu(model.bn1(model.conv1(images))))
        value = model.layer4(model.layer3(model.layer2(model.layer1(value))))
        result = {"resnet_global": value.float().mean((2, 3)), "resnet_fg": weighted_pool(value, mask)}
    elif kind == "dino":
        value = model.forward_features(images)
        patches = value["x_norm_patchtokens"]
        side = int(patches.shape[1] ** 0.5)
        if side * side != patches.shape[1]:
            raise ValueError("DINO patch features must form a square grid.")
        maps = patches.transpose(1, 2).reshape(len(images), -1, side, side)
        result = {"dino_global": value["x_norm_clstoken"].float(), "dino_fg": weighted_pool(maps, mask)}
    elif kind == "bioclip":
        # Normalized zero corresponds to the encoder's RGB normalization mean.
        # Keep nonzero background weights; do not crop away the global image.
        result = {"bio_global": model.encode_image(images, normalize=False).float(),
                  "bio_fg": model.encode_image(images * mask, normalize=False).float()}
    else:
        raise ValueError(kind)
    return result


BASE = ["cn_global"]
FOREGROUND = BASE + ["cn_fg", "cn_mid_fg", "shape"]
VARIANTS = {
    "global_adapter": {"blocks": BASE, "se": False, "label": "ConvNeXt + 同型融合头（控制）"},
    "foreground": {"blocks": FOREGROUND, "se": False, "label": "ConvNeXt + 前景多尺度池化"},
    "foreground_se": {"blocks": FOREGROUND, "se": True, "label": "前景多尺度 + SE"},
    "foreground_mobile": {"blocks": FOREGROUND + ["mobile_global", "mobile_fg"], "se": True, "label": "前景 + SE + MobileNet 特征"},
    "foreground_resnet": {"blocks": FOREGROUND + ["resnet_global", "resnet_fg"], "se": True, "label": "前景 + SE + ResNet 特征"},
    "foreground_dino": {"blocks": FOREGROUND + ["dino_global", "dino_fg"], "se": True, "label": "前景 + SE + DINOv2 特征"},
    "foreground_bio": {"blocks": FOREGROUND + ["bio_global", "bio_fg"], "se": True, "label": "前景 + SE + BioCLIP 特征"},
    "foreground_dino_bio": {"blocks": FOREGROUND + ["dino_global", "dino_fg", "bio_global", "bio_fg"], "se": True, "label": "前景 + SE + DINOv2 + BioCLIP"},
    "global_dino_bio": {"blocks": BASE + ["dino_global", "bio_global"], "se": True, "label": "无前景：ConvNeXt + SE + DINOv2 + BioCLIP"},
    "foreground_all": {"blocks": FOREGROUND + ["mobile_global", "mobile_fg", "resnet_global", "resnet_fg", "dino_global", "dino_fg", "bio_global", "bio_fg"], "se": True, "label": "前景 + SE + 全部辅助骨干"},
}


class ResidualFusionAdapter(nn.Module):
    """Trainable projections + optional SE + zero-initialized residual classifier."""
    def __init__(self, dimensions: dict[str, int], classes: int, width=32, se=False, dropout=0.3):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(size), nn.Linear(size, width), nn.GELU())
            for name, size in dimensions.items()
        })
        total = width * len(dimensions)
        # Reuse the SE architecture, NOT incompatible channel-specific MobileNet SE weights.
        self.attention = SqueezeExcitation(total, max(8, total // 4)) if se else None
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(total, classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, blocks: dict[str, torch.Tensor], baseline_logits: torch.Tensor):
        value = torch.cat([layer(blocks[name]) for name, layer in self.projections.items()], dim=1)
        if self.attention is not None:
            value = (2 * self.attention(value[:, :, None, None])).flatten(1)
        residual = self.output(self.dropout(value))
        residual = residual - residual.mean(1, keepdim=True)
        return baseline_logits + residual, residual
