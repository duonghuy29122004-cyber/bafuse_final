import numpy as np
import pytest
import torch

from bafuse.hybrid.p5_normalization import PopulationZScore
from experiments.soh_forecast.bafuse_p5_architecture_comparison_v1.evaluate_external import (
    _metrics,
    _predict_external,
)
from experiments.soh_forecast.bafuse_p5_architecture_comparison_v1.architecture_models import (
    ARCHITECTURES,
    architecture_details,
    build_architecture,
    trainable_parameter_count,
)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_architecture_contract_and_separate_voltage_encoders(architecture: str):
    model = build_architecture(architecture).eval()

    with torch.inference_mode():
        prediction = model(
            torch.zeros(2, 128),
            torch.zeros(2, 128),
            torch.zeros(2, 5),
        )

    assert prediction.shape == (2, 1)
    assert model.discharge_encoder is not model.charge_encoder
    assert trainable_parameter_count(model) > 0
    details = architecture_details(architecture)
    assert details["input_shapes"] == {
        "X_discharge": ["B", 128],
        "X_charge": ["B", 128],
        "X_physics": ["B", 5],
    }
    assert details["output_shape"] == ["B", 1]


def test_gru_and_cnn_lstm_use_expected_recurrent_dimensions_and_no_dropout():
    gru = build_architecture("GRU")
    assert gru.discharge_encoder.gru.num_layers == 2
    assert gru.discharge_encoder.gru.hidden_size == 32
    assert gru.discharge_encoder.gru.dropout == 0.0
    assert not gru.discharge_encoder.gru.bidirectional

    hybrid = build_architecture("CNN-LSTM")
    assert hybrid.discharge_encoder.lstm.num_layers == 1
    assert hybrid.discharge_encoder.lstm.input_size == 32
    assert hybrid.discharge_encoder.lstm.hidden_size == 32
    assert hybrid.discharge_encoder.lstm.dropout == 0.0


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unknown normalized"):
        build_architecture("transformer")


def test_external_metrics_use_percentage_point_errors():
    result = _metrics(
        np.array([70.0, 80.0], dtype=np.float32),
        np.array([72.0, 77.0], dtype=np.float32),
    )

    assert result["MAE"] == pytest.approx(2.5)
    assert result["RMSE"] == pytest.approx(np.sqrt(6.5))
    assert result["mean_signed_error_pp"] == pytest.approx(-0.5)


def test_external_prediction_uses_frozen_target_inverse_transform():
    class FixedNormalizedPredictor(torch.nn.Module):
        def forward(self, discharge, charge, physics):
            assert discharge.shape == (2, 128)
            assert charge.shape == (2, 128)
            assert torch.equal(physics, torch.full((2, 5), 9.0))
            return torch.full((2, 1), 0.5)

    voltage_scaler = PopulationZScore(
        mean=np.zeros(128),
        observed_std=np.ones(128),
        scale=np.ones(128),
        fit_row_count=102,
        minimum_std=1e-8,
        small_std_indices=(),
    )
    target_scaler = PopulationZScore(
        mean=np.array([75.0]),
        observed_std=np.array([10.0]),
        scale=np.array([10.0]),
        fit_row_count=102,
        minimum_std=1e-8,
        small_std_indices=(),
    )
    arrays = {
        "X_discharge": np.zeros((2, 128), dtype=np.float32),
        "X_charge": np.ones((2, 128), dtype=np.float32),
        "X_physics": np.full((2, 5), 9.0, dtype=np.float32),
    }

    prediction = _predict_external(
        FixedNormalizedPredictor(),
        arrays,
        (voltage_scaler, voltage_scaler, target_scaler),
        torch.device("cpu"),
    )

    np.testing.assert_allclose(prediction, [80.0, 80.0])
