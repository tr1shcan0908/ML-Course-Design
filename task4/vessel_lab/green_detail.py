"""RGB U-Net with a gated, full-resolution green detail branch."""

from torch import nn
from torch.nn import functional as F

from .models import PretrainedUNet


def green_detail_input(image, mode):
    green = image[:, 1:2]
    if mode == "raw_green":
        return green
    if mode != "local_contrast":
        raise ValueError(f"Unknown green detail mode: {mode}")
    # Separable 17×17 local mean. Exclude pixels beyond each crop's boundary.
    smooth = F.avg_pool2d(green.float(), (1, 17), stride=1, padding=(0, 8), count_include_pad=False)
    smooth = F.avg_pool2d(smooth, (17, 1), stride=1, padding=(8, 0), count_include_pad=False)
    return (0.5 + 3 * (green - smooth)).clamp(0, 1)


class GreenDetailUNet(PretrainedUNet):
    def __init__(self, config, root, initialize, mode):
        super().__init__(config, root, initialize)
        self.mode = mode
        # Reuse the same encoder, full-resolution decoder, and mask head.
        self.mask_head = self.head
        self.head = nn.Identity()
        self.detail_branch = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
        )
        self.detail_logit = nn.Conv2d(8, 1, 1)
        self.detail_gate = nn.Conv2d(self.mask_head.in_channels, 1, 1)
        nn.init.zeros_(self.detail_logit.weight)
        nn.init.zeros_(self.detail_logit.bias)
        nn.init.zeros_(self.detail_gate.weight)
        nn.init.zeros_(self.detail_gate.bias)

    def forward(self, image):
        features = super().forward(image)
        detail = self.detail_branch(green_detail_input(image, self.mode))
        return self.mask_head(features) + self.detail_gate(features).sigmoid() * self.detail_logit(detail)
