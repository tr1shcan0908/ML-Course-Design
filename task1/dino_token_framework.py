"""DINOv2 token pooling classifier for small plant seedling datasets."""

from __future__ import annotations

import torch
from torch import nn


class TokenPoolHead(nn.Module):
    """Combine global and spatial token evidence with a learned patch selector."""

    def __init__(self, width: int, classes: int, dropout: float = 0.2):
        super().__init__()
        self.attention = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.attention.weight)
        self.norm = nn.LayerNorm(width * 3)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width * 3, classes)

    def forward(self, cls_token: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(patches).squeeze(-1), dim=1)
        attended = torch.sum(weights.unsqueeze(-1) * patches, dim=1)
        pooled = torch.cat((cls_token, patches.mean(dim=1), attended), dim=-1)
        return self.classifier(self.dropout(self.norm(pooled)))


class DinoTokenClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, classes: int, trainable_blocks: int = 4):
        super().__init__()
        if not 1 <= trainable_blocks <= len(backbone.blocks):
            raise ValueError("trainable_blocks must be within the backbone depth")
        self.backbone = backbone
        self.head = TokenPoolHead(backbone.embed_dim, classes)
        self.trainable_blocks = trainable_blocks
        self.set_backbone_trainable(False)

    def set_backbone_trainable(self, enabled: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if enabled:
            for block in self.backbone.blocks[-self.trainable_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad_(True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        return self.head(features["x_norm_clstoken"], features["x_norm_patchtokens"])
