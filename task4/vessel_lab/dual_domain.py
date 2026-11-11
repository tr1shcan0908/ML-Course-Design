"""High-resolution multiscale vessel network with native and submission heads.

Independently implemented for this project. Full-resolution fusion is inspired
by FRD-Net (doi:10.1364/BOE.522482); this is not a reproduction of that model.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvNorm(nn.Sequential):
    def __init__(self, inputs, outputs, kernel=3, dilation=1):
        super().__init__(nn.Conv2d(inputs, outputs, kernel, padding=dilation * (kernel // 2),
                                  dilation=dilation, bias=False),
                         nn.BatchNorm2d(outputs, momentum=.03), nn.SiLU(inplace=True))


class Residual(nn.Module):
    def __init__(self, inputs, outputs, drop=0.):
        super().__init__()
        self.body = nn.Sequential(ConvNorm(inputs, outputs), nn.Dropout2d(drop),
                                  nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(outputs, momentum=.03))
        self.skip = nn.Conv2d(inputs, outputs, 1, bias=False) if inputs != outputs else nn.Identity()

    def forward(self, x):
        return F.silu(self.body(x) + self.skip(x))


class DualDomainNet(nn.Module):
    def __init__(self, channels=(24, 40, 64, 96), drop=.1):
        super().__init__()
        self.encoders = nn.ModuleList([Residual(i, o, drop if n else 0.)
                                      for n, (i, o) in enumerate(zip((5, *channels[:-1]), channels))])
        self.context = nn.ModuleList([ConvNorm(channels[-1], 32, dilation=d) for d in (1, 2, 3)])
        self.context_mix = ConvNorm(96, channels[-1], 1)
        self.decoders = nn.ModuleList([Residual(i + o, o, drop)
                                      for i, o in zip(reversed(channels[1:]), reversed(channels[:-1]))])
        self.projections = nn.ModuleList([ConvNorm(c, 8, 1) for c in reversed(channels)])
        self.fuse = Residual(32, 24)
        self.native_head = nn.Conv2d(24, 1, 1)
        self.expansion_head = nn.Conv2d(24, 1, 1)
        self.aux_head = nn.Conv2d(channels[1], 1, 1)
        nn.init.constant_(self.expansion_head.bias, -1.5)

    def forward(self, image, auxiliary=False):
        if image.shape[-2] % 8 or image.shape[-1] % 8:
            raise ValueError("DualDomainNet requires dimensions divisible by eight")
        features = []
        x = image
        for index, encoder in enumerate(self.encoders):
            x = encoder(x if index == 0 else F.avg_pool2d(x, 2))
            features.append(x)
        x = x + self.context_mix(torch.cat([branch(x) for branch in self.context], dim=1))
        decoded = [x]
        for stage, skip in zip(self.decoders, reversed(features[:-1])):
            x = stage(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="nearest"), skip], dim=1))
            decoded.append(x)
        fused = self.fuse(torch.cat([F.interpolate(project(feature), size=image.shape[-2:], mode="nearest")
                                    for project, feature in zip(self.projections, decoded)], dim=1))
        native = self.native_head(fused)
        # The provided resize/>0 transform expands the nearest-neighbor mask.
        # A nonnegative logit correction represents this ordered pair of targets.
        submitted = native + F.softplus(self.expansion_head(fused))
        result = torch.cat([native, submitted], dim=1)
        if auxiliary:
            return result, self.aux_head(decoded[-2])
        return result


def binary_loss(logits, target):
    logits, target = logits.float(), target.float()
    p = logits.sigmoid()
    intersection = (p * target).sum((2, 3))
    dice = ((2 * intersection + 1) / (p.sum((2, 3)) + target.sum((2, 3)) + 1)).mean()
    return .5 * F.binary_cross_entropy_with_logits(logits, target) + .5 * (1 - dice)


def dual_loss(outputs, targets):
    logits, auxiliary = outputs
    native = binary_loss(logits[:, :1], targets[:, :1])
    submission = binary_loss(logits[:, 1:2], targets[:, 1:2])
    # Area targets preserve the occupancy of thin vessels in the coarse head.
    coarse = F.interpolate(targets[:, :1], size=auxiliary.shape[-2:], mode="area")
    return .4 * native + .6 * submission + .1 * binary_loss(auxiliary, coarse)
