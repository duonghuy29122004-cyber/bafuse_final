from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from bafuse.hybrid.p5_benchmark_models import (
    P5MultimodalSoH,
    architecture_config as p5_architecture_config,
    build_model as build_p5_model,
)


ARCHITECTURES = ("CNN1D", "LSTM", "GRU", "TCN", "CNN-LSTM")
_EXISTING_P5_NAMES = {
    "CNN1D": "BaFuse_CNN1D_P5",
    "LSTM": "BaFuse_LSTM_P5",
    "TCN": "BaFuse_TCN_P5",
}
_CURVE_POINTS = 128
_EMBEDDING_SIZE = 32


class GRUVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=_EMBEDDING_SIZE,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        _, hidden = self.gru(curve.unsqueeze(-1))
        return hidden[-1]


class CNNLSTMVoltageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, _EMBEDDING_SIZE, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=_EMBEDDING_SIZE,
            hidden_size=_EMBEDDING_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        _validate_curve(curve)
        sequence = self.convolution(curve.unsqueeze(1)).transpose(1, 2)
        _, (hidden, _) = self.lstm(sequence)
        return hidden[-1]


def build_architecture(name: str) -> nn.Module:
    if name in _EXISTING_P5_NAMES:
        return build_p5_model(_EXISTING_P5_NAMES[name])
    encoders: dict[str, Callable[[], nn.Module]] = {
        "GRU": GRUVoltageEncoder,
        "CNN-LSTM": CNNLSTMVoltageEncoder,
    }
    try:
        encoder = encoders[name]
    except KeyError as error:
        raise ValueError(f"Unknown normalized P5 architecture: {name}") from error
    return P5MultimodalSoH(encoder)


def architecture_details(name: str) -> dict[str, object]:
    if name in _EXISTING_P5_NAMES:
        return {
            "benchmark_name": name,
            "implementation": "existing P5 architecture, unchanged",
            "p5_model": p5_architecture_config(_EXISTING_P5_NAMES[name]),
            "input_shapes": {
                "X_discharge": ["B", _CURVE_POINTS],
                "X_charge": ["B", _CURVE_POINTS],
                "X_physics": ["B", 5],
            },
            "output_shape": ["B", 1],
        }
    if name == "GRU":
        encoder = {
            "type": "separate two-layer GRU per voltage modality",
            "layers": [
                "GRU(input_size=1, hidden_size=32, num_layers=2, batch_first=True)",
                "unidirectional; dropout=0; use final hidden state",
            ],
        }
    elif name == "CNN-LSTM":
        encoder = {
            "type": "separate CNN-LSTM per voltage modality",
            "layers": [
                "Conv1D(1,16,kernel=9,padding=4)",
                "ReLU",
                "MaxPool1D(kernel=2)",
                "Conv1D(16,32,kernel=5,padding=2)",
                "ReLU",
                "LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)",
                "unidirectional; dropout=0; use final hidden state",
            ],
        }
    else:
        raise ValueError(f"Unknown normalized P5 architecture: {name}")
    return {
        "benchmark_name": name,
        "encoder": encoder,
        "physics_encoder": [
            "Linear(5,32)",
            "ReLU",
            "Linear(32,32)",
            "ReLU",
        ],
        "fusion": [
            "concatenate three 32-dimensional embeddings",
            "Linear(96,64)",
            "ReLU",
            "Linear(64,32)",
            "ReLU",
            "Linear(32,1)",
        ],
        "input_shapes": {
            "X_discharge": ["B", _CURVE_POINTS],
            "X_charge": ["B", _CURVE_POINTS],
            "X_physics": ["B", 5],
        },
        "output_shape": ["B", 1],
        "encoder_weights_shared_between_modalities": False,
        "dropout": 0.0,
    }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _validate_curve(curve: torch.Tensor) -> None:
    if curve.ndim != 2 or curve.shape[1] != _CURVE_POINTS:
        raise ValueError(
            f"Expected voltage curve with shape (batch, {_CURVE_POINTS}), "
            f"got {tuple(curve.shape)}."
        )
