from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import torch
from torch import nn

from bafuse.hybrid.architecture_models import TCNVoltageEncoder
from bafuse.hybrid.soh_fusion import (
    PHYSICS_FEATURE_NAMES,
    CurveEncoder,
)


ENCODER_NAMES = ("cnn1d", "tcn", "resnet1d")
APPROACH_NAMES = ("multitask", "two_stage")
CURVE_POINTS = 128
EMBEDDING_SIZE = 32
FUSED_SIZE = 64

EncoderName = Literal["cnn1d", "tcn", "resnet1d"]
ApproachName = Literal["multitask", "two_stage"]
EncoderFactory = Callable[[], nn.Module]


class DegradationPredictions(NamedTuple):
    soh: torch.Tensor
    lli: torch.Tensor
    lam_anode: torch.Tensor
    lam_cathode: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.cat(tuple(self), dim=1)


class _Residual1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.projection = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(4, out_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.projection(values)
        hidden = self.activation(self.norm1(self.conv1(values)))
        hidden = self.norm2(self.conv2(hidden))
        return self.activation(hidden + residual)


class ResNet1DVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            _Residual1DBlock(16, 16),
            _Residual1DBlock(16, EMBEDDING_SIZE, stride=2),
            _Residual1DBlock(EMBEDDING_SIZE, EMBEDDING_SIZE),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve, "voltage curve")
        return self.layers(curve.unsqueeze(1))


_ENCODER_FACTORIES: dict[str, EncoderFactory] = {
    "cnn1d": CurveEncoder,
    "tcn": TCNVoltageEncoder,
    "resnet1d": ResNet1DVoltageEncoder,
}


class _SharedVoltageEncoder(nn.Module):
    def __init__(self, curve_encoder: EncoderFactory) -> None:
        super().__init__()
        self.discharge_encoder = curve_encoder()
        self.charge_encoder = curve_encoder()
        self.physics_encoder = nn.Sequential(
            nn.Linear(len(PHYSICS_FEATURE_NAMES), EMBEDDING_SIZE),
            nn.GELU(),
            nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * EMBEDDING_SIZE, FUSED_SIZE),
            nn.GELU(),
        )

    def forward(
        self,
        discharge: torch.Tensor,
        charge: torch.Tensor,
        physics: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(discharge, charge, physics)
        encoded = torch.cat(
            (
                self.discharge_encoder(discharge),
                self.charge_encoder(charge),
                self.physics_encoder(physics),
            ),
            dim=1,
        )
        return self.fusion(encoded)


def _regression_head(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, 32),
        nn.GELU(),
        nn.Linear(32, 1),
    )


class MultiTaskDegradationModel(nn.Module):
    """Shared V2 encoder with independent heads for all four targets."""

    approach: ApproachName = "multitask"

    def __init__(self, curve_encoder: EncoderFactory) -> None:
        super().__init__()
        self.encoder = _SharedVoltageEncoder(curve_encoder)
        self.heads = nn.ModuleDict(
            {
                "soh": _regression_head(FUSED_SIZE),
                "lli": _regression_head(FUSED_SIZE),
                "lam_anode": _regression_head(FUSED_SIZE),
                "lam_cathode": _regression_head(FUSED_SIZE),
            }
        )

    def forward(
        self,
        discharge: torch.Tensor,
        charge: torch.Tensor,
        physics: torch.Tensor,
    ) -> DegradationPredictions:
        features = self.encoder(discharge, charge, physics)
        return DegradationPredictions(
            self.heads["soh"](features),
            self.heads["lli"](features),
            self.heads["lam_anode"](features),
            self.heads["lam_cathode"](features),
        )


class TwoStageDegradationModel(nn.Module):
    """Predict SoH first, then condition diagnostics on that prediction."""

    approach: ApproachName = "two_stage"

    def __init__(self, curve_encoder: EncoderFactory) -> None:
        super().__init__()
        self.encoder = _SharedVoltageEncoder(curve_encoder)
        self.soh_head = _regression_head(FUSED_SIZE)
        self.diagnostic_heads = nn.ModuleDict(
            {
                "lli": _regression_head(FUSED_SIZE + 1),
                "lam_anode": _regression_head(FUSED_SIZE + 1),
                "lam_cathode": _regression_head(FUSED_SIZE + 1),
            }
        )

    def freeze_stage1(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.soh_head.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        discharge: torch.Tensor,
        charge: torch.Tensor,
        physics: torch.Tensor,
    ) -> DegradationPredictions:
        features = self.encoder(discharge, charge, physics)
        predicted_soh = self.soh_head(features)
        stage2_features = torch.cat((features, predicted_soh), dim=1)
        return DegradationPredictions(
            predicted_soh,
            self.diagnostic_heads["lli"](stage2_features),
            self.diagnostic_heads["lam_anode"](stage2_features),
            self.diagnostic_heads["lam_cathode"](stage2_features),
        )


def build_degradation_model(
    approach: str,
    encoder: str,
) -> MultiTaskDegradationModel | TwoStageDegradationModel:
    if approach not in APPROACH_NAMES:
        raise ValueError(f"Unknown degradation approach: {approach}")
    if encoder not in ENCODER_NAMES:
        raise ValueError(f"Unknown degradation encoder: {encoder}")
    factory = _ENCODER_FACTORIES[encoder]
    if approach == "multitask":
        return MultiTaskDegradationModel(factory)
    return TwoStageDegradationModel(factory)


def _validate_curve(curve: torch.Tensor, name: str) -> None:
    if curve.ndim != 2 or curve.shape[1] != CURVE_POINTS:
        raise ValueError(
            f"Expected {name} with shape (batch, {CURVE_POINTS}), "
            f"got {tuple(curve.shape)}."
        )
    if curve.shape[0] == 0 or not torch.isfinite(curve).all():
        raise ValueError(f"{name} must have a non-empty finite batch.")


def _validate_inputs(
    discharge: torch.Tensor,
    charge: torch.Tensor,
    physics: torch.Tensor,
) -> None:
    _validate_curve(discharge, "discharge voltage curve")
    _validate_curve(charge, "charge voltage curve")
    if physics.ndim != 2 or physics.shape[1] != len(PHYSICS_FEATURE_NAMES):
        raise ValueError(
            f"Expected physics input with shape "
            f"(batch, {len(PHYSICS_FEATURE_NAMES)}), got {tuple(physics.shape)}."
        )
    if not torch.isfinite(physics).all():
        raise ValueError("Physics inputs must be finite.")
    if not (discharge.shape[0] == charge.shape[0] == physics.shape[0]):
        raise ValueError("All V2 modalities must have the same batch size.")
