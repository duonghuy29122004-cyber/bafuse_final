from __future__ import annotations

import pytest
import torch

from bafuse.hybrid.architecture_models import (
    ARCHITECTURE_NAMES,
    AttentionVoltageEncoder,
    LSTMVoltageEncoder,
    architecture_parameter_counts,
    build_architecture,
)
from bafuse.hybrid.soh_fusion import PHYSICS_FEATURE_NAMES


@pytest.mark.parametrize("architecture", ARCHITECTURE_NAMES)
def test_architecture_contract_is_one_prediction_per_rpt(architecture: str):
    model = build_architecture(architecture).eval()
    with torch.inference_mode():
        prediction = model(
            torch.zeros(2, 128),
            torch.zeros(2, 128),
            torch.zeros(2, len(PHYSICS_FEATURE_NAMES)),
        )

    assert prediction.shape == (2, 1)


@pytest.mark.parametrize("architecture", ARCHITECTURE_NAMES[1:])
def test_non_cnn_encoders_reject_wrong_curve_shape(architecture: str):
    model = build_architecture(architecture)

    with pytest.raises(ValueError, match="voltage curve"):
        model(
            torch.zeros(2, 127),
            torch.zeros(2, 128),
            torch.zeros(2, len(PHYSICS_FEATURE_NAMES)),
        )


def test_architecture_parameter_counts_are_distinct_and_positive():
    counts = architecture_parameter_counts()

    assert tuple(counts) == ARCHITECTURE_NAMES
    assert all(value > 0 for value in counts.values())
    assert len(set(counts.values())) > 1
    assert counts == {
        "cnn1d_cnn1d_mlp": 15073,
        "lstm_lstm_mlp": 35425,
        "tcn_tcn_mlp": 21761,
        "cnn1d_attention_cnn1d_attention_mlp": 32161,
    }


def test_lstm_and_attention_encoders_match_the_documented_sequence_shapes():
    lstm = LSTMVoltageEncoder().eval()
    assert lstm.encoder.num_layers == 2
    assert lstm.encoder.hidden_size == 32
    assert lstm.encoder.bidirectional is False
    assert lstm.encoder.dropout == 0.0
    assert lstm(torch.zeros(2, 128)).shape == (2, 32)

    attention = AttentionVoltageEncoder().eval()
    layer = attention.attention.layers[0]
    assert layer.self_attn.num_heads == 4
    assert layer.linear1.out_features == 64
    assert layer.dropout.p == pytest.approx(0.1)
    assert layer.activation.__name__ == "gelu"
    assert layer.norm_first is False
    assert attention.attention.enable_nested_tensor is False
    attention_inputs = []
    hook = attention.attention.register_forward_pre_hook(
        lambda _module, args: attention_inputs.append(args[0].shape)
    )
    try:
        assert attention(torch.zeros(2, 128)).shape == (2, 32)
    finally:
        hook.remove()

    assert attention_inputs == [torch.Size((2, 64, 32))]
