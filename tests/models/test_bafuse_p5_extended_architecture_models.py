import pytest
import torch

from experiments.soh_forecast.bafuse_p5_extended_architecture_benchmark_v1.extended_models import (
    ARCHITECTURES,
    InceptionTimeVoltageEncoder,
    ResNet1DVoltageEncoder,
    TCNAttentionVoltageEncoder,
    architecture_details,
    build_model,
    trainable_parameter_count,
)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("modality", ("2M", "3M"))
def test_extended_model_input_output_contract(architecture: str, modality: str):
    model = build_model(architecture, modality).eval()
    discharge = torch.zeros(2, 128)
    charge = torch.ones(2, 128)

    with torch.inference_mode():
        if modality == "3M":
            prediction = model(discharge, charge, torch.zeros(2, 5))
            assert hasattr(model, "physics_encoder")
            assert model.physics_encoder[0].in_features == 5
            assert model.fusion_head[0].in_features == 96
        else:
            prediction = model(discharge, charge)
            assert not hasattr(model, "physics_encoder")
            assert model.fusion_head[0].in_features == 64

    assert prediction.shape == (2, 1)
    assert model.discharge_encoder is not model.charge_encoder
    assert trainable_parameter_count(model) > 0
    assert architecture_details(architecture, modality)["encoder_output_dimension"] == 32


def test_each_new_encoder_is_independently_constructed():
    assert isinstance(build_model("ResNet1D", "2M").discharge_encoder, ResNet1DVoltageEncoder)
    assert isinstance(build_model("InceptionTime", "2M").charge_encoder, InceptionTimeVoltageEncoder)
    assert isinstance(build_model("TCN-Attention", "2M").discharge_encoder, TCNAttentionVoltageEncoder)


def test_two_modality_models_reject_physics_and_invalid_inputs():
    model = build_model("ResNet1D", "2M")
    with pytest.raises(TypeError):
        model(torch.zeros(2, 128), torch.zeros(2, 128), torch.zeros(2, 5))
    with pytest.raises(ValueError, match="curve shape"):
        model(torch.zeros(2, 127), torch.zeros(2, 128))
    with pytest.raises(ValueError, match="batch sizes"):
        model(torch.zeros(2, 128), torch.zeros(3, 128))


def test_tcn_attention_documents_temporal_attention_scope():
    details = architecture_details("TCN-Attention", "3M")
    assert details["encoder"]["tcn_padding"] == "causal left padding"
    assert (
        details["encoder"]["attention"]["temporal_attention"]
        == "noncausal within the observed RPT curve"
    )
