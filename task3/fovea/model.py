"""Pretrained residual encoder, small U-Net decoder, and floating-point targets."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ConvBlock(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outgoing), nn.SiLU(inplace=True),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outgoing), nn.SiLU(inplace=True))


class FoveaNet(nn.Module):
    def __init__(self, architecture="heatmap", pretrained=True, auxiliary=False):
        super().__init__()
        self.architecture, self.auxiliary = architecture, auxiliary
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = nn.Identity()
        if architecture == "regressor":
            self.head = nn.Sequential(nn.Linear(512, 128), nn.SiLU(),
                                      nn.Dropout(0.15), nn.Linear(128, 2))
        else:
            self.up3 = ConvBlock(512+256, 128)
            self.up2 = ConvBlock(128+128, 64)
            self.up1 = ConvBlock(64+64, 48)
            self.up0 = ConvBlock(48+64, 32)
            self.head = nn.Conv2d(32, 1, 1)
            if auxiliary:
                self.region = nn.Conv2d(32, 1, 1)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            # Small batches should not corrupt the ImageNet running statistics.
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, image):
        enc = self.encoder
        x0 = enc.relu(enc.bn1(enc.conv1(image)))
        x1 = enc.layer1(enc.maxpool(x0))
        x2 = enc.layer2(x1)
        x3 = enc.layer3(x2)
        x4 = enc.layer4(x3)
        if self.architecture == "regressor":
            relative = self.head(F.adaptive_avg_pool2d(x4, 1).flatten(1)).sigmoid()
            return {"coordinate": relative * image.shape[-1] - 0.5}
        def fuse(block, x, skip):
            return block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                                  align_corners=False), skip], dim=1))
        x = fuse(self.up3, x4, x3)
        x = fuse(self.up2, x, x2)
        x = fuse(self.up1, x, x1)
        x = fuse(self.up0, x, x0)
        result = {"heatmap": self.head(x)}
        if self.auxiliary:
            result["region"] = self.region(x)
        return result


def spatial_grid(height, width, input_size, device):
    ys = (torch.arange(height, device=device, dtype=torch.float32)+0.5)*input_size/height-0.5
    xs = (torch.arange(width, device=device, dtype=torch.float32)+0.5)*input_size/width-0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def decode(logits, size, local=False):
    logits = logits.float().squeeze(1)
    batch, height, width = logits.shape
    xx, yy = spatial_grid(height, width, size, logits.device)
    if local:
        index = logits.flatten(1).argmax(1)
        cy, cx = index // width, index % width
        gy = torch.arange(height, device=logits.device)[None, :, None]
        gx = torch.arange(width, device=logits.device)[None, None, :]
        mask = ((gy-cy[:,None,None]).abs() <= 5) & ((gx-cx[:,None,None]).abs() <= 5)
        logits = logits.masked_fill(~mask, -1e9)
    probability = logits.flatten(1).softmax(-1).reshape(batch, height, width)
    return torch.stack([(probability*xx).sum((1,2)), (probability*yy).sum((1,2))], -1)


def to_original(coordinates, inverse):
    return torch.bmm(inverse[:, :, :2], coordinates.float().unsqueeze(-1)).squeeze(-1) + inverse[:, :, 2]


def marginal_multiscale_loss(logits, target, factors=(1, 2, 4, 8)):
    """Cross entropy on X/Y marginal probability masses at fixed dyadic scales.

    This is inspired by coordinate classification, not a reproduction of MSCE's
    max-pooled logits or SimCC's independent coordinate heads. logsumexp preserves
    probability mass and gradients when grouping neighboring coordinate bins.
    """
    batch, height, width = logits.shape
    log_probability = logits.flatten(1).log_softmax(-1).reshape(batch, height, width)
    target = target.reshape(batch, height, width)
    terms = []
    for axis in (1, 2):
        log_marginal = log_probability.logsumexp(dim=axis)
        marginal_target = target.sum(dim=axis)
        bins = log_marginal.shape[-1]
        for factor in factors:
            if bins % factor:
                raise ValueError("Coordinate bins must be divisible by every pooling factor")
            grouped_log = log_marginal.reshape(batch, bins//factor, factor).logsumexp(-1)
            grouped_target = marginal_target.reshape(batch, bins//factor, factor).sum(-1)
            terms.append(-(grouped_target*grouped_log).sum(-1).mean())
    # Sum X/Y losses, then average scales, to retain the 2-D CE's rough magnitude.
    return torch.stack(terms).sum()/len(factors)


def localization_loss(output, batch, size, sigma=2.0, coordinate_weight=1000., region_weight=0.,
                      heatmap_loss_type="gaussian"):
    if "coordinate" in output:
        coordinates = output["coordinate"].float()
        heatmap_loss = coordinates.new_zeros(())
    else:
        logits = output["heatmap"].float().squeeze(1)
        _, height, width = logits.shape
        xx, yy = spatial_grid(height, width, size, logits.device)
        sigma_input = sigma * size / width
        target = -((xx-batch["point"][:,0,None,None])**2 +
                   (yy-batch["point"][:,1,None,None])**2) / (2*sigma_input**2)
        target = target.flatten(1).softmax(-1)
        # Gaussian cross entropy avoids the vanishing gradients of a tiny blob in a large map.
        if heatmap_loss_type == "gaussian":
            heatmap_loss = -(target*logits.flatten(1).log_softmax(-1)).sum(1).mean()
        elif heatmap_loss_type == "marginal_multiscale":
            heatmap_loss = marginal_multiscale_loss(logits, target)
        else:
            raise ValueError(f"Unknown heatmap loss: {heatmap_loss_type}")
        coordinates = decode(output["heatmap"], size)
    original = to_original(coordinates, batch["inverse"])
    coordinate_loss = ((original-batch["original"])/2992.).square().mean()
    loss = heatmap_loss + coordinate_weight * coordinate_loss
    if region_weight and "region" in output:
        region = batch["region"].float()
        positives = region.sum().clamp_min(1)
        weight = ((region.numel()-positives)/positives).clamp(1, 100)
        auxiliary = F.binary_cross_entropy_with_logits(output["region"].float(), region, pos_weight=weight)
        loss = loss + region_weight*auxiliary
    return loss, dict(heatmap=float(heatmap_loss.detach()), coordinate=float(coordinate_loss.detach()))
