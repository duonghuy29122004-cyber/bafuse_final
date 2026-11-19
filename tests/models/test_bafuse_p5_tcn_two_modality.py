import pytest
import torch

from bafuse.hybrid.p5_benchmark_models import TCNVoltageEncoder
from experiments.soh_forecast.bafuse_p5_tcn_two_modality_ablation_v1.tcn_2m_model import (
    TCN2MSoH,
    architecture_details,
    build_tcn_2m_model,
    trainable_parameter_count,
)


def test_tcn_2m_has_two_independent_existing_tcn_encoders_and_no_physics():
    model = build_tcn_2m_model()

    assert type(model.discharge_encoder) is TCNVoltageEncoder
    assert type(model.charge_encoder) is TCNVoltageEncoder
    assert model.discharge_encoder is not model.charge_encoder
    assert not hasattr(model, "physics_encoder")
    assert model.fusion_head[0].in_features == 64
    assert model.fusion_head[0].out_features == 64
    assert model.fusion_head[2].in_features == 64
    assert model.fusion_head[2].out_features == 32
    assert model.fusion_head[4].in_features == 32
    assert model.fusion_head[4].out_features == 1
    assert trainable_parameter_count(model) > 0

    reference = TCNVoltageEncoder()
    assert {
        name: tuple(value.shape)
        for name, value in model.discharge_encoder.state_dict().items()
    } == {
        name: tuple(value.shape)
        for name, value in reference.state_dict().items()
    }


def test_tcn_2m_forward_contract_excludes_physics():
    model = build_tcn_2m_model().eval()

    with torch.inference_mode():
        prediction = model(torch.zeros(3, 128), torch.ones(3, 128))

    assert prediction.shape == (3, 1)
    with pytest.raises(TypeError):
        model(torch.zeros(3, 128), torch.ones(3, 128), torch.zeros(3, 5))


def test_tcn_2m_rejects_bad_voltage_shapes_and_batch_mismatch():
    model = build_tcn_2m_model()

    with pytest.raises(ValueError, match="discharge shape"):
        model(torch.zeros(2, 127), torch.zeros(2, 128))
    with pytest.raises(ValueError, match="batch sizes"):
        model(torch.zeros(2, 128), torch.zeros(3, 128))

    details = architecture_details()
    assert details["fusion_input_dimension"] == 64
    assert details["physics_encoder"] is None
