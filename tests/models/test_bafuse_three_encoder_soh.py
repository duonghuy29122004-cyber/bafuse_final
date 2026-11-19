import pytest
import torch

from bafuse.hybrid.soh_fusion import (
    EXCLUDED_FEATURES,
    PHYSICS_FEATURE_NAMES,
    ThreeEncoderSoH,
    build_physics_features,
    resample_curve,
)


def test_physics_feature_builder_uses_only_nonleaking_schema():
    voltage = [4.1 - index * 0.01 for index in range(129)]
    features = build_physics_features(
        voltage,
        [-1.4, -1.5, -1.6],
        [24.0, 25.0, 27.0],
    )

    assert tuple(features) == PHYSICS_FEATURE_NAMES
    assert features["mean_discharge_temperature_native"] == pytest.approx(
        25.3333333333
    )
    assert features["discharge_temperature_rise_native"] == pytest.approx(3.0)
    assert features["mean_absolute_discharge_current_native"] == pytest.approx(1.5)
    assert features["discharge_plateau_slope_v_per_progress"] < 0
    assert features["discharge_slope_variability_v_per_progress"] == pytest.approx(
        0.0, abs=1e-5
    )
    assert "step4_discharge_duration" in EXCLUDED_FEATURES


def test_resample_curve_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="non-finite"):
        resample_curve([4.1, float("nan"), 2.5])


def test_three_encoder_outputs_soh_and_freezes_only_discharge_when_requested():
    model = ThreeEncoderSoH(freeze_discharge=True)
    output = model(
        torch.zeros(3, 128),
        torch.zeros(3, 128),
        torch.zeros(3, len(PHYSICS_FEATURE_NAMES)),
    )

    assert output.shape == (3, 1)
    assert all(
        not parameter.requires_grad
        for parameter in model.discharge_encoder.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in model.charge_encoder.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in model.physics_encoder.parameters()
    )


def test_discharge_initialization_copies_only_compatible_convolutions():
    model = ThreeEncoderSoH()
    state = {
        "layers.0.weight": torch.full((16, 1, 9), 0.25),
        "layers.0.bias": torch.full((16,), 0.5),
        "layers.3.weight": torch.full((32, 16, 5), 0.75),
        "layers.3.bias": torch.full((32,), 1.0),
        "layers.7.weight": torch.full((1, 32), 9.0),
        "layers.7.bias": torch.full((1,), 9.0),
    }

    loaded = model.initialize_discharge_encoder(state)

    assert loaded == (
        "layers.0.weight",
        "layers.0.bias",
        "layers.3.weight",
        "layers.3.bias",
    )
    assert torch.all(model.discharge_encoder.layers[0].weight == 0.25)
    assert torch.all(model.discharge_encoder.layers[3].weight == 0.75)
