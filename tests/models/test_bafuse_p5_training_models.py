from __future__ import annotations

import pytest
import torch
from torch import nn

from bafuse.hybrid.p5_benchmark import _metric_values
from bafuse.hybrid.p5_benchmark_models import (
    ARCHITECTURE_NAMES,
    CNN1DAttentionVoltageEncoder,
    LSTMVoltageEncoder,
    TCNVoltageEncoder,
    build_model,
    trainable_parameter_count,
)


@pytest.mark.parametrize("architecture", ARCHITECTURE_NAMES)
def test_p5_model_contract_has_independent_curve_encoders_and_one_output(
    architecture: str,
):
    model = build_model(architecture).eval()
    with torch.inference_mode():
        prediction = model(
            torch.zeros(2, 128),
            torch.zeros(2, 128),
            torch.zeros(2, 5),
        )

    assert prediction.shape == (2, 1)
    assert model.discharge_encoder is not model.charge_encoder
    assert trainable_parameter_count(model) > 0


def test_all_models_use_the_same_unbounded_relu_fusion_head_and_physics_encoder():
    models = [build_model(name) for name in ARCHITECTURE_NAMES]
    expected_head = [
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
    ]
    for model in models:
        assert [type(layer) for layer in model.fusion_head] == expected_head
        assert model.fusion_head[0].in_features == 96
        assert model.fusion_head[-1].out_features == 1
        assert [type(layer) for layer in model.physics_encoder] == [
            nn.Linear,
            nn.ReLU,
            nn.Linear,
            nn.ReLU,
        ]
    assert all(
        [layer.weight.shape for layer in model.physics_encoder if isinstance(layer, nn.Linear)]
        == [layer.weight.shape for layer in models[0].physics_encoder if isinstance(layer, nn.Linear)]
        for model in models
    )


def test_lstm_tcn_and_attention_architecture_details_are_explicit():
    lstm = LSTMVoltageEncoder()
    assert lstm.lstm.num_layers == 2
    assert lstm.lstm.hidden_size == 32
    assert not lstm.lstm.bidirectional
    assert lstm.lstm.dropout == 0.0

    tcn = TCNVoltageEncoder()
    assert tcn.receptive_field == 13
    assert tcn.layers[0].padding == 2
    assert tcn.layers[1].padding == 4

    attention = CNN1DAttentionVoltageEncoder()
    assert attention.position_embedding.shape == (1, 64, 32)
    assert attention.attention.self_attn.num_heads == 4


def test_metrics_are_observation_level_and_use_percentage_point_errors():
    metrics = _metric_values(
        torch.tensor([90.0, 100.0]),
        torch.tensor([92.0, 97.0]),
    )

    assert metrics["MAE"] == pytest.approx(2.5)
    assert metrics["RMSE"] == pytest.approx((6.5) ** 0.5)
    assert metrics["R2"] == pytest.approx(0.74)
