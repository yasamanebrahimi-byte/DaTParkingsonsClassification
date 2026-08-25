"""Small quantitative-feature baseline."""

from __future__ import annotations

import torch
from torch import nn


class FeatureMLP(nn.Module):
    def __init__(self, input_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_features, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)

