from __future__ import annotations

import torch
from torch import nn

from bafuse.hybrid.p5_benchmark_models import (
    CNNVoltageEncoder,
    architecture_config as p5_architecture_config,
)


class CNN1D2MSoH(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discharge_encoder = CNNVoltageEncoder()
        self.charge_encoder = CNNVoltageEncoder()
        self.fusion_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        discharge: torch.Tensor,
        charge: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(discharge, charge)
        fused = torch.cat(
            (self.discharge_encoder(discharge), self.charge_encoder(charge)),
            dim=1,
        )
        return self.fusion_head(fused)


def build_cnn1d_2m_model() -> CNN1D2MSoH:
    return CNN1D2MSoH()


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def architecture_details() -> dict[str, object]:
    return {
        "name": "CNN1D-2M",
        "voltage_encoder": p5_architecture_config("BaFuse_CNN1D_P5")[
            "discharge_and_charge_encoders"
        ],
        "encoders_are_independent": True,
        "physics_encoder": None,
        "fusion_input_dimension": 64,
        "fusion_head": [
            "Linear(64,64)",
            "ReLU",
            "Linear(64,32)",
            "ReLU",
            "Linear(32,1)",
        ],
        "input_shapes": {
            "discharge": ["B", 128],
            "charge": ["B", 128],
        },
        "output_shape": ["B", 1],
        "output_space": "normalized SoH",
    }


def _validate_inputs(discharge: torch.Tensor, charge: torch.Tensor) -> None:
    if discharge.ndim != 2 or discharge.shape[1] != 128:
        raise ValueError(
            f"Expected discharge shape (batch, 128), got {tuple(discharge.shape)}."
        )
    if charge.ndim != 2 or charge.shape[1] != 128:
        raise ValueError(
            f"Expected charge shape (batch, 128), got {tuple(charge.shape)}."
        )
    if discharge.shape[0] != charge.shape[0]:
        raise ValueError("Discharge and Charge batch sizes must match.")
