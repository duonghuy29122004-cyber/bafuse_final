from __future__ import annotations

from torch import nn

from bafuse.hybrid.architecture_models import (
    CausalResidualBlock1D,
    InceptionModule1D,
    InceptionResidualBlock1D,
    InceptionTimeVoltageEncoder,
    ResNet1DVoltageEncoder,
    ResNetBasicBlock1D,
    TCNAttentionVoltageEncoder,
    ThreeModalitySoH,
    TwoModalitySoH,
    build_v2_model,
    trainable_parameter_count,
    v2_model_config,
)


CURVE_POINTS = 128
EMBEDDING_SIZE = 32
PHYSICS_FEATURES = 5
ARCHITECTURES = ("ResNet1D", "InceptionTime", "TCN-Attention")

ExtendedBaFuse2M = TwoModalitySoH
ExtendedBaFuse3M = ThreeModalitySoH


def build_model(architecture: str, modality: str) -> nn.Module:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown extended architecture: {architecture}")
    if modality not in {"2M", "3M"}:
        raise ValueError(f"Unknown modality configuration: {modality}")
    return build_v2_model(f"{architecture}-{modality}")


def architecture_details(architecture: str, modality: str) -> dict[str, object]:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown extended architecture: {architecture}")
    if modality not in {"2M", "3M"}:
        raise ValueError(f"Unknown modality configuration: {modality}")
    details = v2_model_config(f"{architecture}-{modality}")
    details["output_space"] = "normalized SoH"
    return details
