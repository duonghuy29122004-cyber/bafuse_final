from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from bafuse.hybrid.p5_benchmark_models import (
    CNNVoltageEncoder as _CNNVoltageEncoder,
    TCNVoltageEncoder as _HistoricalTCNVoltageEncoder,
)
from bafuse.hybrid.soh_fusion import PHYSICS_FEATURE_NAMES, ThreeEncoderSoH


ARCHITECTURE_NAMES = (
    "cnn1d_cnn1d_mlp",
    "lstm_lstm_mlp",
    "tcn_tcn_mlp",
    "cnn1d_attention_cnn1d_attention_mlp",
)
EMBEDDING_SIZE = 32
CURVE_POINTS = 128


class LSTMVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=EMBEDDING_SIZE,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        sequence, _ = self.encoder(curve.unsqueeze(-1))
        return sequence[:, -1, :]


class _CausalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, dilation=dilation
        )
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.projection(values)
        hidden = self.conv1(F.pad(values, (self.padding, 0)))
        hidden = self.activation(hidden)
        hidden = self.conv2(F.pad(hidden, (self.padding, 0)))
        return self.activation(hidden + residual)


class TCNVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _CausalResidualBlock(1, 16, dilation=1),
            _CausalResidualBlock(16, EMBEDDING_SIZE, dilation=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        return self.layers(curve.unsqueeze(1))


class AttentionVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, EMBEDDING_SIZE, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=EMBEDDING_SIZE,
                nhead=4,
                dim_feedforward=64,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
                norm_first=False,
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        sequence = self.convolutions(curve.unsqueeze(1)).transpose(1, 2)
        return self.attention(sequence).mean(dim=1)


class FixedConcatenationSoH(nn.Module):
    def __init__(self, curve_encoder: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.discharge_encoder = curve_encoder()
        self.charge_encoder = curve_encoder()
        self.physics_encoder = nn.Sequential(
            nn.Linear(len(PHYSICS_FEATURE_NAMES), EMBEDDING_SIZE),
            nn.GELU(),
            nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE),
            nn.GELU(),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(3 * EMBEDDING_SIZE, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        discharge: torch.Tensor,
        charge: torch.Tensor,
        physics: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(discharge, charge, physics)
        fused = torch.cat(
            (
                self.discharge_encoder(discharge),
                self.charge_encoder(charge),
                self.physics_encoder(physics),
            ),
            dim=1,
        )
        return self.regression_head(fused)


def _validate_curve(curve: torch.Tensor) -> None:
    if curve.ndim != 2 or curve.shape[1] != CURVE_POINTS:
        raise ValueError(
            f"Expected voltage curve shape (batch, {CURVE_POINTS}), "
            f"got {tuple(curve.shape)}."
        )


def _validate_inputs(
    discharge: torch.Tensor, charge: torch.Tensor, physics: torch.Tensor
) -> None:
    _validate_curve(discharge)
    _validate_curve(charge)
    if physics.ndim != 2 or physics.shape[1] != len(PHYSICS_FEATURE_NAMES):
        raise ValueError(
            f"Expected physics input with shape (batch, {len(PHYSICS_FEATURE_NAMES)}), "
            f"got {tuple(physics.shape)}."
        )
    if not (discharge.shape[0] == charge.shape[0] == physics.shape[0]):
        raise ValueError("All three modalities must have the same batch size.")


def build_architecture(name: str) -> nn.Module:
    if name == "cnn1d_cnn1d_mlp":
        return ThreeEncoderSoH()
    if name == "lstm_lstm_mlp":
        return FixedConcatenationSoH(LSTMVoltageEncoder)
    if name == "tcn_tcn_mlp":
        return FixedConcatenationSoH(TCNVoltageEncoder)
    if name == "cnn1d_attention_cnn1d_attention_mlp":
        return FixedConcatenationSoH(AttentionVoltageEncoder)
    raise ValueError(f"Unknown BaFuse architecture: {name}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def architecture_parameter_counts() -> dict[str, int]:
    return {
        name: trainable_parameter_count(build_architecture(name))
        for name in ARCHITECTURE_NAMES
    }


class ResNetBasicBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(values)
        hidden = F.relu(self.norm1(self.conv1(values)))
        hidden = self.norm2(self.conv2(hidden))
        return F.relu(hidden + residual)


class ResNet1DVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResNetBasicBlock1D(16, 16),
            ResNetBasicBlock1D(16, 32, stride=2),
            ResNetBasicBlock1D(32, 32),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        return self.pool(self.blocks(self.stem(curve.unsqueeze(1))))


class InceptionModule1D(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        bottleneck_channels = 16
        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.branches = nn.ModuleList(
            nn.Conv1d(
                bottleneck_channels,
                8,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=False,
            )
            for kernel in (3, 7, 15)
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, 8, kernel_size=1, bias=False),
        )
        self.norm = nn.BatchNorm1d(32)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(values)
        branches = [branch(bottleneck) for branch in self.branches]
        branches.append(self.pool_branch(values))
        return F.gelu(self.norm(torch.cat(branches, dim=1)))


class InceptionResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.inception_modules = nn.Sequential(
            InceptionModule1D(in_channels),
            InceptionModule1D(32),
            InceptionModule1D(32),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == 32
            else nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=1, bias=False),
                nn.BatchNorm1d(32),
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.inception_modules(values) + self.shortcut(values))


class InceptionTimeVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            InceptionResidualBlock1D(1),
            InceptionResidualBlock1D(32),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        return self.pool(self.blocks(curve.unsqueeze(1)))


class CausalResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, dilation=dilation
        )
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.projection(values)
        hidden = F.relu(self.conv1(F.pad(values, (self.padding, 0))))
        hidden = self.conv2(F.pad(hidden, (self.padding, 0)))
        return F.relu(hidden + residual)


class TCNAttentionVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tcn = nn.Sequential(
            CausalResidualBlock1D(1, 16, dilation=1),
            CausalResidualBlock1D(16, EMBEDDING_SIZE, dilation=2),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, CURVE_POINTS, EMBEDDING_SIZE)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.attention = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_SIZE,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        sequence = self.tcn(curve.unsqueeze(1)).transpose(1, 2)
        sequence = sequence + self.position_embedding
        return self.attention(sequence).mean(dim=1)


class LateConcatenationFusion(nn.Module):
    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        if not embeddings:
            raise ValueError("Late concatenation requires at least one embedding.")
        if any(embedding.ndim != 2 for embedding in embeddings):
            raise ValueError("Fusion embeddings must each have shape (batch, features).")
        batch_sizes = {embedding.shape[0] for embedding in embeddings}
        if len(batch_sizes) != 1:
            raise ValueError("All fusion embeddings must have the same batch size.")
        return torch.cat(embeddings, dim=1)


class PhysicsFeatureEncoder(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Linear(len(PHYSICS_FEATURE_NAMES), EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE),
            nn.ReLU(),
        )


class TwoModalitySoH(nn.Module):
    def __init__(self, encoder: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.discharge_encoder = encoder()
        self.charge_encoder = encoder()
        self.fusion = LateConcatenationFusion()
        self.fusion_head = nn.Sequential(
            nn.Linear(2 * EMBEDDING_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, discharge: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
        _validate_voltage_inputs(discharge, charge)
        fused = self.fusion(
            self.discharge_encoder(discharge),
            self.charge_encoder(charge),
        )
        return self.fusion_head(fused)


class ThreeModalitySoH(nn.Module):
    def __init__(self, encoder: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.discharge_encoder = encoder()
        self.charge_encoder = encoder()
        self.physics_encoder = PhysicsFeatureEncoder()
        self.fusion = LateConcatenationFusion()
        self.fusion_head = nn.Sequential(
            nn.Linear(3 * EMBEDDING_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        discharge: torch.Tensor,
        charge: torch.Tensor,
        physics: torch.Tensor,
    ) -> torch.Tensor:
        _validate_voltage_inputs(discharge, charge)
        _validate_physics(physics, discharge.shape[0])
        fused = self.fusion(
            self.discharge_encoder(discharge),
            self.charge_encoder(charge),
            self.physics_encoder(physics),
        )
        return self.fusion_head(fused)


@dataclass(frozen=True)
class V2ModelDefinition:
    family: str
    modality: str
    role: str
    encoder: Callable[[], nn.Module]


V2_MODEL_REGISTRY: dict[str, V2ModelDefinition] = {
    "CNN1D-3M": V2ModelDefinition(
        "CNN1D", "3M", "official_historical_baseline", _CNNVoltageEncoder
    ),
    "CNN1D-2M": V2ModelDefinition(
        "CNN1D", "2M", "historical_comparison", _CNNVoltageEncoder
    ),
    "TCN-2M": V2ModelDefinition(
        "TCN", "2M", "historical_comparison", _HistoricalTCNVoltageEncoder
    ),
    "TCN-3M": V2ModelDefinition(
        "TCN", "3M", "historical_comparison", _HistoricalTCNVoltageEncoder
    ),
    "ResNet1D-2M": V2ModelDefinition(
        "ResNet1D", "2M", "new_architecture", ResNet1DVoltageEncoder
    ),
    "ResNet1D-3M": V2ModelDefinition(
        "ResNet1D", "3M", "new_architecture", ResNet1DVoltageEncoder
    ),
    "InceptionTime-2M": V2ModelDefinition(
        "InceptionTime", "2M", "new_architecture", InceptionTimeVoltageEncoder
    ),
    "InceptionTime-3M": V2ModelDefinition(
        "InceptionTime", "3M", "new_architecture", InceptionTimeVoltageEncoder
    ),
    "TCN-Attention-2M": V2ModelDefinition(
        "TCN-Attention", "2M", "new_architecture", TCNAttentionVoltageEncoder
    ),
    "TCN-Attention-3M": V2ModelDefinition(
        "TCN-Attention", "3M", "new_architecture", TCNAttentionVoltageEncoder
    ),
}
V2_MODEL_NAMES = tuple(V2_MODEL_REGISTRY)
V2_NEW_MODEL_NAMES = tuple(
    name
    for name, definition in V2_MODEL_REGISTRY.items()
    if definition.role == "new_architecture"
)


def build_v2_model(name: str) -> nn.Module:
    try:
        definition = V2_MODEL_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unknown BaFuse V2 model: {name}") from error
    if definition.modality == "2M":
        return TwoModalitySoH(definition.encoder)
    if definition.modality == "3M":
        return ThreeModalitySoH(definition.encoder)
    raise ValueError(f"Unsupported modality for {name}: {definition.modality}")


def v2_model_config(name: str) -> dict[str, Any]:
    try:
        definition = V2_MODEL_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unknown BaFuse V2 model: {name}") from error
    if definition.family == "CNN1D":
        encoder_details: dict[str, Any] = {
            "type": "independent Conv1D encoders",
            "layers": [
                "Conv1D(1,16,kernel=9,padding=4)",
                "ReLU",
                "MaxPool1D(kernel=2)",
                "Conv1D(16,32,kernel=5,padding=2)",
                "ReLU",
                "AdaptiveAvgPool1D(1)",
            ],
        }
    elif definition.family == "TCN":
        encoder_details = {
            "type": "independent causal TCN encoders",
            "blocks": [
                {"channels": [1, 16, 16], "kernel_size": 3, "dilation": 1},
                {"channels": [16, 32, 32], "kernel_size": 3, "dilation": 2},
            ],
            "residual_connections": True,
            "pooling": "global average",
        }
    elif definition.family == "ResNet1D":
        encoder_details = {
            "stem": ["Conv1D(1,16,kernel=7,padding=3)", "BatchNorm1D", "ReLU"],
            "residual_blocks": [
                "BasicBlock(16,16,kernel=3,stride=1)",
                "BasicBlock(16,32,kernel=3,stride=2)",
                "BasicBlock(32,32,kernel=3,stride=1)",
            ],
            "pooling": "adaptive average pool to one position",
        }
    elif definition.family == "InceptionTime":
        encoder_details = {
            "residual_blocks": 2,
            "inception_modules_per_block": 3,
            "parallel_kernels": [3, 7, 15],
            "channels_per_convolution_branch": 8,
            "pool_branch": "MaxPool1D(kernel=3,stride=1) then Conv1D(.,8,kernel=1)",
            "bottleneck_channels": 16,
            "activation": "GELU",
            "pooling": "adaptive average pool to one position",
        }
    else:
        encoder_details = {
            "tcn_blocks": [
                {"channels": [1, 16, 16], "kernel_size": 3, "dilation": 1},
                {"channels": [16, 32, 32], "kernel_size": 3, "dilation": 2},
            ],
            "residual_connections": True,
            "tcn_padding": "causal left padding",
            "attention": {
                "type": "one TransformerEncoderLayer over 128 temporal positions",
                "model_dimension": EMBEDDING_SIZE,
                "heads": 4,
                "feedforward_dimension": 64,
                "dropout": 0.1,
                "activation": "GELU",
                "positional_embedding": "learned",
                "temporal_attention": "noncausal within the observed RPT curve",
            },
            "pooling": "mean over temporal positions",
        }
    config: dict[str, Any] = {
        "name": name,
        "family": definition.family,
        "modality": definition.modality,
        "role": definition.role,
        "encoder": encoder_details,
        "input_shapes": {
            "discharge": ["B", CURVE_POINTS],
            "charge": ["B", CURVE_POINTS],
        },
        "encoder_output_dimension": EMBEDDING_SIZE,
        "encoders_are_independent": True,
        "encoder_weights_shared_between_modalities": False,
        "fusion": "late_concatenation",
        "fusion_head_input_dimension": (
            2 if definition.modality == "2M" else 3
        )
        * EMBEDDING_SIZE,
        "physics_encoder": (
            ["Linear(5,32)", "ReLU", "Linear(32,32)", "ReLU"]
            if definition.modality == "3M"
            else None
        ),
        "fusion_head": (
            ["Linear(96,64)", "ReLU", "Linear(64,32)", "ReLU", "Linear(32,1)"]
            if definition.modality == "3M"
            else ["Linear(64,64)", "ReLU", "Linear(64,32)", "ReLU", "Linear(32,1)"]
        ),
        "output_shape": ["B", 1],
        "output_space": "same-RPT SoH percentage points",
        "output_activation": None,
    }
    if definition.modality == "3M":
        config["input_shapes"]["physics"] = ["B", len(PHYSICS_FEATURE_NAMES)]
        config["physics_features"] = list(PHYSICS_FEATURE_NAMES)
    return config


def load_v2_architecture_config(
    path: Path | str | None = None,
) -> dict[str, Any]:
    config_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3]
        / "configs"
        / "bafuse_v2_architectures.json"
    )
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"Unsupported BaFuse V2 architecture config: {config_path}")
    contract = config.get("input_contract")
    if not isinstance(contract, dict):
        raise ValueError("V2 architecture config must define input_contract.")
    if contract.get("discharge_voltage_shape") != ["B", CURVE_POINTS]:
        raise ValueError("V2 discharge voltage shape must be [B, 128].")
    if contract.get("charge_voltage_shape") != ["B", CURVE_POINTS]:
        raise ValueError("V2 charge voltage shape must be [B, 128].")
    if contract.get("physics_feature_names") != list(PHYSICS_FEATURE_NAMES):
        raise ValueError("V2 physics feature names do not match the shared contract.")
    if contract.get("output_shape") != ["B", 1]:
        raise ValueError("V2 output shape must be [B, 1].")
    if contract.get("output_space") != "same-RPT SoH percentage points":
        raise ValueError("V2 output must be same-RPT SoH percentage points.")
    if contract.get("prohibited_predictors") != [
        "capacity",
        "cell_id",
        "group_id",
        "external_labels",
        "target_derived_features",
    ]:
        raise ValueError("V2 prohibited-predictor contract was changed.")

    entries = config.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("V2 architecture config must list at least one model.")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each V2 model config entry must be an object.")
        name = entry.get("name")
        if not isinstance(name, str) or name in seen:
            raise ValueError(f"Invalid or duplicate V2 model name: {name!r}.")
        seen.add(name)
        try:
            expected = V2_MODEL_REGISTRY[name]
        except KeyError as error:
            raise ValueError(f"Unregistered BaFuse V2 model: {name}") from error
        if (
            entry.get("family") != expected.family
            or entry.get("modality") != expected.modality
            or entry.get("role") != expected.role
        ):
            raise ValueError(f"V2 model config disagrees with registry for {name}.")
    return config


def build_v2_models(
    config_path: Path | str | None = None,
) -> dict[str, nn.Module]:
    config = load_v2_architecture_config(config_path)
    return {entry["name"]: build_v2_model(entry["name"]) for entry in config["models"]}


def v2_architecture_parameter_counts() -> dict[str, int]:
    return {
        name: trainable_parameter_count(build_v2_model(name))
        for name in V2_MODEL_NAMES
    }


def _validate_voltage_inputs(discharge: torch.Tensor, charge: torch.Tensor) -> None:
    _validate_curve(discharge)
    _validate_curve(charge)
    if discharge.shape[0] != charge.shape[0]:
        raise ValueError("Discharge and charge batch sizes must match.")


def _validate_physics(physics: torch.Tensor, batch_size: int) -> None:
    expected_features = len(PHYSICS_FEATURE_NAMES)
    if physics.ndim != 2 or physics.shape[1] != expected_features:
        raise ValueError(
            f"Expected physics input with shape (batch, {expected_features}), "
            f"got {tuple(physics.shape)}."
        )
    if physics.shape[0] != batch_size:
        raise ValueError("All three modality batch sizes must match.")
