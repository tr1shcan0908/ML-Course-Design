"""Model factory for controlled, image-disjoint architecture comparisons."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
import torchvision.models as tvm

from vessel.model import VesselUNet
from .vendor.fr_unet import FR_UNet


class ResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, outputs), nn.SiLU(),
                                  nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, outputs))
        self.shortcut = nn.Conv2d(inputs, outputs, 1, bias=False) if inputs != outputs else nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.body(x) + self.shortcut(x))


class DecoderStage(nn.Module):
    def __init__(self, inputs: int, skip: int, outputs: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(inputs, outputs, 2, stride=2)
        self.refine = ResidualBlock(outputs + skip, outputs)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError("Decoder geometry requires input dimensions divisible by 32")
        return self.refine(torch.cat((x, skip), dim=1))


class PretrainedUNet(nn.Module):
    def __init__(self, config: dict, root: Path, initialize: bool):
        super().__init__()
        self.family = config["family"]
        self.freeze_bn_stats = config.get("freeze_encoder_bn", True)
        if self.family == "resnet18_unet":
            encoder = tvm.resnet18(weights=None)
            feature_channels = [64, 64, 128, 256, 512]
        elif self.family == "efficientnet_b0_unet":
            encoder = tvm.efficientnet_b0(weights=None)
            feature_channels = [16, 24, 40, 112, 320]
        else:
            raise ValueError(self.family)
        if initialize:
            path = root / config["pretrained_path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != config["pretrained_sha256"]:
                raise ValueError("ImageNet initialization weight hash mismatch")
            encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        if self.family == "resnet18_unet":
            self.encoder = nn.ModuleDict({"stem": nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu),
                                          "pool": encoder.maxpool, "stage1": encoder.layer1,
                                          "stage2": encoder.layer2, "stage3": encoder.layer3,
                                          "stage4": encoder.layer4})
        else:
            # Features 1/2/3/5/7 are at /2,/4,/8,/16,/32. Exclude the
            # classification-only 1280-channel expansion and classifier.
            self.encoder = nn.ModuleList(list(encoder.features.children())[:8])
        self.detail = ResidualBlock(3, 16)
        decoder_channels = config.get("decoder_channels", [192, 128, 64, 32, 24])
        skip_channels = list(reversed(feature_channels[:-1])) + [16]
        inputs = feature_channels[-1]
        stages = []
        for skip, outputs in zip(skip_channels, decoder_channels):
            stages.append(DecoderStage(inputs, skip, outputs))
            inputs = outputs
        self.decoder = nn.ModuleList(stages)
        self.head = nn.Conv2d(inputs, 1, 1)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn_stats:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, image):
        if image.shape[-2] % 32 or image.shape[-1] % 32:
            raise ValueError("Pretrained U-Net needs dimensions divisible by 32")
        x = (image - self.image_mean) / self.image_std
        features = []
        if self.family == "resnet18_unet":
            x = self.encoder["stem"](x)
            features.append(x)
            x = self.encoder["pool"](x)
            for stage in ("stage1", "stage2", "stage3", "stage4"):
                x = self.encoder[stage](x)
                features.append(x)
        else:
            for index, module in enumerate(self.encoder):
                x = module(x)
                if index in (1, 2, 3, 5, 7):
                    features.append(x)
        x = features[-1]
        skips = list(reversed(features[:-1])) + [self.detail(image)]
        for decoder, skip in zip(self.decoder, skips):
            x = decoder(x, skip)
        return self.head(x)


class GrayFRUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = FR_UNet(num_classes=1, num_channels=1,
                               feature_scale=config.get("feature_scale", 2),
                               dropout=config.get("drop_prob", 0.1), out_ave=True)

    def forward(self, x):
        return self.network(x[:, 1:2])


def build_model(config: dict, root: Path, initialize: bool = False) -> nn.Module:
    family = config["family"]
    if family in ("resnet18_unet", "efficientnet_b0_unet"):
        return PretrainedUNet(config, root, initialize)
    if family == "fr_unet":
        return GrayFRUNet(config)
    if family == "csa_unet":
        return VesselUNet(config)
    raise ValueError(f"Unknown model family: {family}")


def optimizer_groups(model: nn.Module, multiplier: float) -> list[dict]:
    if not isinstance(model, PretrainedUNet):
        return [{"params": list(model.parameters()), "lr_multiplier": 1.0}]
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    decoder = [p for p in model.parameters() if id(p) not in encoder_ids]
    return [{"params": encoder, "lr_multiplier": multiplier}, {"params": decoder, "lr_multiplier": 1.0}]
