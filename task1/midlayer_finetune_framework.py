"""Partially train DINOv2-B layer-8 plant features and a lightweight head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from joint_foreground_modules import normalized_foreground_prior
from train_classifier import IMAGENET_MEAN, IMAGENET_STD


class MidlayerPlantClassifier(nn.Module):
    """Freeze blocks 0-5, optionally train blocks 6-7, and stop at layer 8."""

    def __init__(self, backbone: nn.Module, classes: int, head_dropout: float = 0.1):
        super().__init__()
        if backbone.embed_dim != 768 or backbone.num_register_tokens != 4:
            raise ValueError("Expected DINOv2 ViT-B/14 with four register tokens")
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(3 * 768, classes))
        self.register_buffer("rgb_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.set_backbone_trainable(False)

    def set_backbone_trainable(self, enabled: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if enabled:
            for block in self.backbone.blocks[6:8]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen prefix must keep deterministic evaluation behavior.
        for block in self.backbone.blocks[:6]:
            block.eval()
        return self

    def features(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2] != images.shape[-1] or images.shape[-1] % 14:
            raise ValueError("DINOv2 input must be square and divisible by 14")
        with torch.no_grad():
            tokens = self.backbone.prepare_tokens_with_masks(images)
            for block in self.backbone.blocks[:6]:
                tokens = block(tokens)
        for block in self.backbone.blocks[6:8]:
            tokens = block(tokens)
        tokens = self.backbone.norm(tokens)
        cls = tokens[:, 0].float()
        patches = tokens[:, 1 + self.backbone.num_register_tokens:].float()
        grid = images.shape[-1] // 14
        if patches.shape[1] != grid * grid:
            raise RuntimeError("Unexpected patch grid")
        with torch.no_grad():
            rgb = (images.float() * self.rgb_std + self.rgb_mean).clamp(0, 1)
            mask, _, _ = normalized_foreground_prior(rgb)
            weights = F.adaptive_avg_pool2d(mask, (grid, grid)).flatten(1)
        plant = (patches * weights[:, :, None]).sum(1) / weights.sum(1).clamp_min(1e-6)[:, None]
        pieces = (F.normalize(cls, dim=1), F.normalize(patches.mean(1), dim=1),
                  F.normalize(plant, dim=1))
        return torch.cat((pieces[0] * (0.5 ** 0.5), pieces[1] * 0.5, pieces[2] * 0.5), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images) * 10.0)


def load_model(classes: int) -> MidlayerPlantClassifier:
    repo = torch.hub.get_dir() + "/facebookresearch_dinov2_main"
    backbone = torch.hub.load(repo, "dinov2_vitb14_reg", source="local", pretrained=True)
    return MidlayerPlantClassifier(backbone, classes)
