from __future__ import annotations

import pytest
import torch

from bafuse.hybrid.architecture_models import (
    V2_MODEL_NAMES,
    V2_NEW_MODEL_NAMES,
    build_v2_model,
    build_v2_models,
    load_v2_architecture_config,
    trainable_parameter_count,
    v2_architecture_parameter_counts,
    v2_model_config,
)


@pytest.mark.parametrize("name", V2_MODEL_NAMES)
def test_v2_models_follow_the_same_rpt_input_output_contract(name: str):
    model = build_v2_model(name).eval()
    discharge = torch.zeros(2, 128)
    charge = torch.ones(2, 128)

    with torch.inference_mode():
        if name.endswith("-3M"):
            prediction = model(discharge, charge, torch.zeros(2, 5))
            assert model.physics_encoder[0].in_features == 5
            assert model.fusion_head[0].in_features == 96
        else:
            prediction = model(discharge, charge)
            assert not hasattr(model, "physics_encoder")
            assert model.fusion_head[0].in_features == 64

    assert prediction.shape == (2, 1)
    assert model.discharge_encoder is not model.charge_encoder
    assert trainable_parameter_count(model) > 0
    assert v2_model_config(name)["output_space"] == "same-RPT SoH percentage points"


@pytest.mark.parametrize("name", V2_MODEL_NAMES)
def test_v2_models_reject_invalid_curve_and_batch_shapes(name: str):
    model = build_v2_model(name).eval()
    with pytest.raises(ValueError, match="voltage curve"):
        if name.endswith("-3M"):
            model(torch.zeros(2, 127), torch.zeros(2, 128), torch.zeros(2, 5))
        else:
            model(torch.zeros(2, 127), torch.zeros(2, 128))
    with pytest.raises(ValueError, match="batch sizes"):
        if name.endswith("-3M"):
            model(torch.zeros(2, 128), torch.zeros(3, 128), torch.zeros(2, 5))
        else:
            model(torch.zeros(2, 128), torch.zeros(3, 128))
    if name.endswith("-3M"):
        with pytest.raises(ValueError, match="physics input"):
            model(torch.zeros(2, 128), torch.zeros(2, 128), torch.zeros(2, 4))
        with pytest.raises(ValueError, match="batch sizes"):
            model(torch.zeros(2, 128), torch.zeros(2, 128), torch.zeros(3, 5))


def test_v2_model_registry_preserves_comparison_roles_and_six_new_models():
    config = load_v2_architecture_config()
    assert tuple(entry["name"] for entry in config["models"]) == V2_MODEL_NAMES
    assert V2_NEW_MODEL_NAMES == (
        "ResNet1D-2M",
        "ResNet1D-3M",
        "InceptionTime-2M",
        "InceptionTime-3M",
        "TCN-Attention-2M",
        "TCN-Attention-3M",
    )
    assert config["models"][0]["role"] == "official_historical_baseline"
    assert {model["name"] for model in config["models"] if model["role"] == "historical_comparison"} == {
        "CNN1D-2M",
        "TCN-2M",
        "TCN-3M",
    }
    assert set(build_v2_models()) == set(V2_MODEL_NAMES)
    assert config["input_contract"]["prohibited_predictors"] == [
        "capacity",
        "cell_id",
        "group_id",
        "external_labels",
        "target_derived_features",
    ]


def test_v2_parameter_counts_are_reported_for_the_full_registry():
    counts = v2_architecture_parameter_counts()
    assert tuple(counts) == V2_MODEL_NAMES
    assert counts == {
        "CNN1D-3M": 15073,
        "CNN1D-2M": 11777,
        "TCN-2M": 18465,
        "TCN-3M": 21761,
        "ResNet1D-2M": 32929,
        "ResNet1D-3M": 36225,
        "InceptionTime-2M": 53361,
        "InceptionTime-3M": 56657,
        "TCN-Attention-2M": 43745,
        "TCN-Attention-3M": 47041,
    }
    assert all(
        value == trainable_parameter_count(build_v2_model(name))
        for name, value in counts.items()
    )
