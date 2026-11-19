from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


CURVE_POINTS = 128
PHYSICS_FEATURE_COUNT = 5
EMBEDDING_SIZE = 32
ATTENTION_TOKENS = 64

ARCHITECTURE_NAMES = (
    "BaFuse_CNN1D_P5",
    "BaFuse_LSTM_P5",
    "BaFuse_TCN_P5",
    "BaFuse_CNN1D_Attention_P5",
)


class CNNVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, EMBEDDING_SIZE, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        return self.layers(curve.unsqueeze(1))


class LSTMVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=EMBEDDING_SIZE,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        _, (hidden, _) = self.lstm(curve.unsqueeze(-1))
        return hidden[-1]


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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.projection(values)
        hidden = F.relu(self.conv1(F.pad(values, (self.padding, 0))))
        hidden = self.conv2(F.pad(hidden, (self.padding, 0)))
        return F.relu(hidden + residual)


class TCNVoltageEncoder(nn.Module):
    receptive_field = 13

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


class CNN1DAttentionVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, EMBEDDING_SIZE, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, ATTENTION_TOKENS, EMBEDDING_SIZE)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.attention = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_SIZE,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        sequence = self.stem(curve.unsqueeze(1)).transpose(1, 2)
        if sequence.shape[1] != ATTENTION_TOKENS:
            raise ValueError(
                f"Expected {ATTENTION_TOKENS} attention tokens, got {sequence.shape[1]}."
            )
        sequence = sequence + self.position_embedding
        return self.attention(sequence).mean(dim=1)


class P5MultimodalSoH(nn.Module):
    def __init__(self, curve_encoder: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.discharge_encoder = curve_encoder()
        self.charge_encoder = curve_encoder()
        self.physics_encoder = nn.Sequential(
            nn.Linear(PHYSICS_FEATURE_COUNT, EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE),
            nn.ReLU(),
        )
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
        _validate_inputs(discharge, charge, physics)
        fused = torch.cat(
            (
                self.discharge_encoder(discharge),
                self.charge_encoder(charge),
                self.physics_encoder(physics),
            ),
            dim=1,
        )
        return self.fusion_head(fused)


def _validate_curve(curve: torch.Tensor) -> None:
    if curve.ndim != 2 or curve.shape[1] != CURVE_POINTS:
        raise ValueError(
            f"Expected voltage curve with shape (batch, {CURVE_POINTS}), "
            f"got {tuple(curve.shape)}."
        )


def _validate_inputs(
    discharge: torch.Tensor, charge: torch.Tensor, physics: torch.Tensor
) -> None:
    _validate_curve(discharge)
    _validate_curve(charge)
    if physics.ndim != 2 or physics.shape[1] != PHYSICS_FEATURE_COUNT:
        raise ValueError(
            f"Expected physics input with shape (batch, {PHYSICS_FEATURE_COUNT}), "
            f"got {tuple(physics.shape)}."
        )
    if not (discharge.shape[0] == charge.shape[0] == physics.shape[0]):
        raise ValueError("All three modalities must have the same batch size.")


def build_model(name: str) -> P5MultimodalSoH:
    encoders: dict[str, Callable[[], nn.Module]] = {
        ARCHITECTURE_NAMES[0]: CNNVoltageEncoder,
        ARCHITECTURE_NAMES[1]: LSTMVoltageEncoder,
        ARCHITECTURE_NAMES[2]: TCNVoltageEncoder,
        ARCHITECTURE_NAMES[3]: CNN1DAttentionVoltageEncoder,
    }
    try:
        encoder = encoders[name]
    except KeyError as error:
        raise ValueError(f"Unknown P5 architecture: {name}") from error
    return P5MultimodalSoH(encoder)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def architecture_config(name: str) -> dict[str, object]:
    curve_encoders: dict[str, dict[str, object]] = {
        ARCHITECTURE_NAMES[0]: {
            "type": "independent Conv1D encoders",
            "layers": [
                "Conv1D(1,16,kernel=9,padding=4)",
                "ReLU",
                "MaxPool1D(kernel=2)",
                "Conv1D(16,32,kernel=5,padding=2)",
                "ReLU",
                "AdaptiveAvgPool1D(1)",
            ],
        },
        ARCHITECTURE_NAMES[1]: {
            "type": "independent unidirectional LSTM encoders",
            "input_size": 1,
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.0,
            "bidirectional": False,
            "representation": "final hidden state",
        },
        ARCHITECTURE_NAMES[2]: {
            "type": "independent causal TCN encoders",
            "blocks": [
                {"channels": [1, 16, 16], "kernel_size": 3, "dilation": 1},
                {"channels": [16, 32, 32], "kernel_size": 3, "dilation": 2},
            ],
            "residual_connections": True,
            "pooling": "global average",
            "receptive_field_points": TCNVoltageEncoder.receptive_field,
        },
        ARCHITECTURE_NAMES[3]: {
            "type": "independent CNN1D plus self-attention encoders",
            "stem": [
                "Conv1D(1,16,kernel=9,padding=4)",
                "ReLU",
                "MaxPool1D(kernel=2)",
                "Conv1D(16,32,kernel=5,padding=2)",
                "ReLU",
            ],
            "tokens": ATTENTION_TOKENS,
            "positional_information": "learned position embedding",
            "attention_heads": 4,
            "feedforward_dimension": 64,
            "attention_layers": 1,
            "attention_dropout": 0.1,
            "pooling": "mean over attended tokens",
        },
    }
    try:
        curve_encoder = curve_encoders[name]
    except KeyError as error:
        raise ValueError(f"Unknown P5 architecture: {name}") from error
    return {
        "name": name,
        "discharge_and_charge_encoders": curve_encoder,
        "encoder_weights_shared_between_modalities": False,
        "physics_encoder": ["Linear(5,32)", "ReLU", "Linear(32,32)", "ReLU"],
        "fusion": "concatenate discharge, charge, and physics embeddings (96)",
        "fusion_head": [
            "Linear(96,64)",
            "ReLU",
            "Linear(64,32)",
            "ReLU",
            "Linear(32,1)",
        ],
        "output_activation": None,
    }
