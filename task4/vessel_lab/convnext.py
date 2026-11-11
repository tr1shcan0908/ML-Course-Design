"""ImageNet ConvNeXt-Tiny encoder with full-resolution RGB skip features."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import convnext_tiny

from .models import DecoderStage, ResidualBlock


class ConvNeXtUNet(nn.Module):
    def __init__(self, config: dict, root: Path, initialize: bool):
        super().__init__()
        if config["family"] != "convnext_tiny_unet":
            raise ValueError(f"Unsupported encoder family: {config['family']}")
        backbone = convnext_tiny(weights=None)
        if initialize:
            path = root / config["pretrained_path"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != config["pretrained_sha256"]:
                raise ValueError("ConvNeXt ImageNet weight hash mismatch")
            backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.encoder = backbone.features
        self.encoder[:2].requires_grad_(False)
        self.detail = ResidualBlock(3, 16)
        self.half_detail = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
        )
        self.decoder = nn.ModuleList([
            DecoderStage(768, 384, 160),
            DecoderStage(160, 192, 112),
            DecoderStage(112, 96, 64),
            DecoderStage(64, 16, 32),
            DecoderStage(32, 16, 24),
        ])
        self.head = nn.Conv2d(24, 1, 1)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder[:2].eval()
        return self

    def forward(self, image):
        height, width = image.shape[-2:]
        if height % 32 or width % 32:
            raise ValueError("ConvNeXt U-Net needs dimensions divisible by 32")
        x = (image - self.image_mean) / self.image_std
        pyramid = []
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index % 2 == 1:
                pyramid.append(x)
        detail = self.detail(image)
        skips = [*reversed(pyramid[:-1]), self.half_detail(detail), detail]
        for decoder, skip in zip(self.decoder, skips):
            x = decoder(x, skip)
        return self.head(x)


def optimizer_groups(model: ConvNeXtUNet, multiplier: float) -> list[dict]:
    encoder = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    encoder_ids = {id(parameter) for parameter in encoder}
    decoder = [parameter for parameter in model.parameters()
               if parameter.requires_grad and id(parameter) not in encoder_ids]
    return [{"params": encoder, "lr_multiplier": multiplier},
            {"params": decoder, "lr_multiplier": 1.0}]
