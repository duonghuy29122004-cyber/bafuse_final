from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from bafuse.hybrid.architecture_models import build_v2_model, trainable_parameter_count
from bafuse.hybrid.p5_benchmark_models import build_model as build_p5_model


from bafuse.models.cnn_2m import build_cnn1d_2m_model
from bafuse.models.tcn_2m import build_tcn_2m_model


CURVE_POINTS = 128
PHYSICS_FEATURE_NAMES = (
    "mean_abs_discharge_current",
    "discharge_current_variability",
    "discharge_voltage_slope_20_80",
    "discharge_voltage_derivative_mad_10_90",
    "discharge_voltage_segment_contrast_10_30_vs_70_90",
)


@dataclass(frozen=True)
class Final2ModelSpec:
    name: str
    modalities: tuple[str, ...]
    output: str = "single normalized same-RPT SoH value"


FINAL2_MODEL_SPECS = (
    Final2ModelSpec("CNN-3M", ("discharge_voltage", "charge_voltage", "p5_physics")),
    Final2ModelSpec("CNN-2M", ("discharge_voltage", "charge_voltage")),
    Final2ModelSpec("TCN-2M", ("discharge_voltage", "charge_voltage")),
    Final2ModelSpec("ResNet-2M", ("discharge_voltage", "charge_voltage")),
    Final2ModelSpec("DenseNet-2M", ("discharge_voltage", "charge_voltage")),
    Final2ModelSpec("ConvNeXt-2M", ("discharge_voltage", "charge_voltage")),
)
FINAL2_MODEL_NAMES = tuple(spec.name for spec in FINAL2_MODEL_SPECS)


def _validate_curve(curve: torch.Tensor) -> None:
    if curve.ndim != 2 or curve.shape[1] != CURVE_POINTS:
        raise ValueError(
            f"Expected voltage curve shape (batch, {CURVE_POINTS}), "
            f"got {tuple(curve.shape)}."
        )


def _validate_pair(discharge: torch.Tensor, charge: torch.Tensor) -> None:
    _validate_curve(discharge)
    _validate_curve(charge)
    if discharge.shape[0] != charge.shape[0]:
        raise ValueError("Discharge and charge curves must have matching batch sizes.")


class DenseLayer1D(nn.Module):
    def __init__(
        self, in_channels: int, growth_rate: int, dropout: float = 0.05
    ) -> None:
        super().__init__()
        bottleneck_channels = 4 * growth_rate
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                bottleneck_channels,
                growth_rate,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.Dropout1d(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat((values, self.layers(values)), dim=1)


class DenseBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        layer_count: int,
        growth_rate: int,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(layer_count):
            layers.append(DenseLayer1D(channels, growth_rate, dropout))
            channels += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class TransitionBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class DenseNetVoltageEncoder1D(nn.Module):
    embedding_size = 32

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=False)
        self.block1 = DenseBlock1D(16, layer_count=3, growth_rate=8)
        self.transition1 = TransitionBlock1D(self.block1.out_channels, 24)
        self.block2 = DenseBlock1D(24, layer_count=3, growth_rate=8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.block2.out_channels, self.embedding_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        values = self.stem(curve.unsqueeze(1))
        values = self.block1(values)
        values = self.transition1(values)
        values = self.block2(values)
        return self.projection(self.pool(values))


class DenseNet2MSoH(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discharge_encoder = DenseNetVoltageEncoder1D()
        self.charge_encoder = DenseNetVoltageEncoder1D()
        self.fusion_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, discharge: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
        _validate_pair(discharge, charge)
        fused = torch.cat(
            (self.discharge_encoder(discharge), self.charge_encoder(charge)), dim=1
        )
        return self.fusion_head(fused)


class LayerNorm1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise1 = nn.Linear(channels, expansion * channels)
        self.activation = nn.GELU()
        self.pointwise2 = nn.Linear(expansion * channels, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        hidden = self.depthwise(values).transpose(1, 2)
        hidden = self.norm(hidden)
        hidden = self.pointwise1(hidden)
        hidden = self.activation(hidden)
        hidden = self.pointwise2(hidden) * self.layer_scale
        return residual + hidden.transpose(1, 2)


class ConvNeXtVoltageEncoder1D(nn.Module):
    embedding_size = 32

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            LayerNorm1D(16),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(ConvNeXtBlock1D(16), ConvNeXtBlock1D(16))
        self.downsample = nn.Sequential(
            LayerNorm1D(16),
            nn.Conv1d(16, 32, kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(ConvNeXtBlock1D(32))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, self.embedding_size),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        values = self.stem(curve.unsqueeze(1))
        values = self.stage1(values)
        values = self.downsample(values)
        values = self.stage2(values)
        return self.projection(self.pool(values))


class ConvNeXt2MSoH(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discharge_encoder = ConvNeXtVoltageEncoder1D()
        self.charge_encoder = ConvNeXtVoltageEncoder1D()
        self.fusion_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, discharge: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
        _validate_pair(discharge, charge)
        fused = torch.cat(
            (self.discharge_encoder(discharge), self.charge_encoder(charge)), dim=1
        )
        return self.fusion_head(fused)


def _build_resnet_2m() -> nn.Module:
    return build_v2_model("ResNet1D-2M")


_BUILDERS: dict[str, Callable[[], nn.Module]] = {
    "CNN-3M": lambda: build_p5_model("BaFuse_CNN1D_P5"),
    "CNN-2M": build_cnn1d_2m_model,
    "TCN-2M": build_tcn_2m_model,
    "ResNet-2M": _build_resnet_2m,
    "DenseNet-2M": DenseNet2MSoH,
    "ConvNeXt-2M": ConvNeXt2MSoH,
}


def build_final2_model(name: str) -> nn.Module:
    try:
        return _BUILDERS[name]()
    except KeyError as error:
        raise ValueError(f"Unknown Final2 architecture: {name}") from error


def final2_parameter_counts() -> dict[str, int]:
    return {
        name: trainable_parameter_count(build_final2_model(name))
        for name in FINAL2_MODEL_NAMES
    }


def forward_final2_model(
    model: nn.Module,
    name: str,
    discharge: torch.Tensor,
    charge: torch.Tensor,
    physics: torch.Tensor | None = None,
) -> torch.Tensor:
    if name == "CNN-3M":
        if physics is None:
            raise ValueError("CNN-3M requires the five P5 physics features.")
        if physics.ndim != 2 or physics.shape[1] != len(PHYSICS_FEATURE_NAMES):
            raise ValueError(
                f"CNN-3M expects {len(PHYSICS_FEATURE_NAMES)} P5 physics features."
            )
        if physics.shape[0] != discharge.shape[0] or physics.shape[0] != charge.shape[0]:
            raise ValueError("All CNN-3M modalities must have matching batch sizes.")
        return model(discharge, charge, physics)
    if physics is not None:
        raise ValueError(f"{name} accepts only discharge and charge curves.")
    return model(discharge, charge)
