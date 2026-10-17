from __future__ import annotations

from typing import Literal

import torch
from torch import nn


ActivationName = Literal["relu", "tanh", "gelu"]


def activation_layer(name: ActivationName) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


class TransitionMLP(nn.Module):
    """Small, explainable feed-forward network for binary Switch/Stay prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 64,
        hidden_layers: int = 2,
        activation: ActivationName = "relu",
        dropout: float = 0.10,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(current, hidden_units),
                    activation_layer(activation),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden_units
        layers.append(nn.Linear(current, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

