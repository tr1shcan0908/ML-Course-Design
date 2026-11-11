"""Compact U-Net with optional cross-scale spatial attention.

Architecture reference: Guo et al., SA-UNetv2, arXiv:2509.11774.
This implementation was written for this project, not copied from its notebook.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DropBlock2d(nn.Module):
    """Drop contiguous, per-channel spatial blocks during training."""

    def __init__(self, probability: float, block_size: int):
        super().__init__()
        if not 0 <= probability < 1 or block_size < 1 or block_size % 2 != 1:
            raise ValueError("DropBlock needs probability in [0,1) and an odd block size")
        self.probability = probability
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return x
        height, width = x.shape[-2:]
        block = min(self.block_size, height, width)
        if block % 2 == 0:
            block -= 1
        if block == 1:
            return F.dropout(x, self.probability, True)
        radius = block // 2
        gamma = self.probability * height * width / (
            block**2 * (height - block + 1) * (width - block + 1)
        )
        centers = (torch.rand(x.shape, device=x.device) < gamma).to(x.dtype)
        centers[..., :radius, :] = 0
        centers[..., -radius:, :] = 0
        centers[..., :, :radius] = 0
        centers[..., :, -radius:] = 0
        keep = 1 - F.max_pool2d(centers, block, stride=1, padding=radius)
        # Float32 reductions avoid fp16 overflow on full-resolution features.
        denominator = keep.float().sum(dim=(2, 3), keepdim=True).clamp_min(1)
        scale = (height * width / denominator).to(x.dtype)
        return x * keep * scale


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, groups: int, drop: float, block: int):
        layers = []
        for input_channels in (in_channels, out_channels):
            layers.extend([
                nn.Conv2d(input_channels, out_channels, 3, padding=1),
                DropBlock2d(drop, block),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(),
            ])
        super().__init__(*layers)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.filter(pooled))


class CrossScaleAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((encoder.mean(1, keepdim=True), decoder.mean(1, keepdim=True)), dim=1)
        return encoder * torch.sigmoid(self.filter(pooled))


class VesselUNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        channels = config["channels"]
        if len(channels) != 4 or any(c % config["groups"] for c in channels):
            raise ValueError("Expected four channel counts divisible by GroupNorm groups")
        groups, drop, block = config["groups"], config["drop_prob"], config["drop_block_size"]
        self.use_csa = config["skip_attention"]
        self.encoders = nn.ModuleList([
            ConvBlock(i, o, groups, drop, block)
            for i, o in zip([3, *channels[:-1]], channels)
        ])
        self.bottleneck_attention = SpatialAttention() if config["bottleneck_attention"] else nn.Identity()
        reversed_channels = list(reversed(channels))
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(i, o, 3, stride=2, padding=1, output_padding=1)
            for i, o in zip(reversed_channels[:-1], reversed_channels[1:])
        ])
        self.gates = nn.ModuleList([CrossScaleAttention() for _ in range(3)]) if self.use_csa else nn.ModuleList()
        self.decoders = nn.ModuleList([
            ConvBlock(2 * c, c, groups, drop, block) for c in reversed_channels[1:]
        ])
        self.head = nn.Conv2d(channels[0], 1, 1)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError("Input spatial dimensions must be divisible by eight")
        skips = []
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck_attention(self.encoders[-1](x))
        for index, (up, decoder, skip) in enumerate(zip(self.ups, self.decoders, reversed(skips))):
            x = up(x)
            if self.use_csa:
                skip = self.gates[index](skip, x)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.head(x)
