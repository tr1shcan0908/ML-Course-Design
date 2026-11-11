"""Local vessel affinity as auxiliary supervision for the full-resolution decoder."""

import torch
from torch import nn
from torch.nn import functional as F


# One orientation for each undirected edge on the 8-neighbor grid.
DIRECTIONS = ((0, 1), (1, -1), (1, 0), (1, 1))
MARGIN = 3  # Every endpoint needs two pixels of context for the 3×3 opening.


class DirectionalHead(nn.Module):
    def __init__(self, mask_head: nn.Conv2d):
        super().__init__()
        self.mask_head = mask_head
        self.edge_head = nn.Conv2d(mask_head.in_channels, len(DIRECTIONS), 1)

    def forward(self, features):
        logits = self.mask_head(features)
        if self.training:
            return torch.cat((logits, self.edge_head(features)), dim=1)
        return logits


def narrow_pixels(mask: torch.Tensor) -> torch.Tensor:
    """Foreground outside a square 3×3 opening, for float binary masks."""
    eroded = -F.max_pool2d(-F.pad(mask, (1, 1, 1, 1)), 3, stride=1)
    opened = F.max_pool2d(eroded, 3, stride=1, padding=1)
    return mask * (1 - opened)


def interior(values: torch.Tensor) -> torch.Tensor:
    return values[..., MARGIN:-MARGIN, MARGIN:-MARGIN]


def neighbors(values: torch.Tensor) -> torch.Tensor:
    height, width = values.shape[-2:]
    return torch.cat([
        values[..., MARGIN + dy:height - MARGIN + dy, MARGIN + dx:width - MARGIN + dx]
        for dy, dx in DIRECTIONS
    ], dim=1)


def directional_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    edges = interior(mask) * neighbors(mask)
    narrow = narrow_pixels(mask)
    touches_narrow = torch.maximum(interior(narrow), neighbors(narrow))
    # Background edges weigh 1, other vessel edges 3, and narrow vessel edges 6.
    weights = 1 + edges * (2 + 3 * touches_narrow)
    losses = F.binary_cross_entropy_with_logits(interior(logits.float()), edges, reduction="none")
    return (weights * losses).sum() / weights.sum()
