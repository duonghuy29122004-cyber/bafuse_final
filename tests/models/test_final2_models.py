from __future__ import annotations

import json
from pathlib import Path

import torch

from bafuse.hybrid.final2_models import (
    FINAL2_MODEL_NAMES,
    DenseNet2MSoH,
    ConvNeXt2MSoH,
    build_final2_model,
    final2_parameter_counts,
    forward_final2_model,
)
from bafuse.hybrid.final2_runtime import autocast_context, resolve_device, seed_everything


def _inputs(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch_size, 128),
        torch.randn(batch_size, 128),
        torch.randn(batch_size, 5),
    )


def test_final2_registry_has_exact_six_models_and_nonempty_parameter_counts():
    counts = final2_parameter_counts()
    assert tuple(counts) == FINAL2_MODEL_NAMES
    assert all(value > 0 for value in counts.values())
    registry_path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "final2"
        / "v1"
        / "model_registry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert tuple(model["name"] for model in registry["models"]) == FINAL2_MODEL_NAMES
    assert {
        model["name"]: model["trainable_parameters"] for model in registry["models"]
    } == counts


def test_all_final2_models_support_batch_one_and_two():
    for name in FINAL2_MODEL_NAMES:
        model = build_final2_model(name).eval()
        assert model.discharge_encoder is not model.charge_encoder
        for batch_size in (1, 2):
            discharge, charge, physics = _inputs(batch_size)
            with torch.inference_mode():
                if name == "CNN-3M":
                    prediction = forward_final2_model(
                        model, name, discharge, charge, physics
                    )
                else:
                    prediction = forward_final2_model(model, name, discharge, charge)
            assert prediction.shape == (batch_size, 1)
            assert torch.isfinite(prediction).all()


def test_new_architecture_state_dict_checkpoint_round_trip(tmp_path):
    for model_type in (DenseNet2MSoH, ConvNeXt2MSoH):
        source = model_type().eval()
        checkpoint = tmp_path / f"{model_type.__name__}.pt"
        torch.save(source.state_dict(), checkpoint)
        restored = model_type().eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored.load_state_dict(state)
        discharge, charge, _ = _inputs(2)
        with torch.inference_mode():
            expected = source(discharge, charge)
            actual = restored(discharge, charge)
        torch.testing.assert_close(actual, expected)


def test_model_input_contract_rejects_wrong_shapes_and_modalities():
    model = build_final2_model("CNN-3M").eval()
    discharge = torch.zeros(2, 128)
    charge = torch.zeros(2, 128)
    with torch.inference_mode():
        try:
            forward_final2_model(model, "CNN-3M", discharge, charge)
        except ValueError as error:
            assert "requires" in str(error)
        else:
            raise AssertionError("CNN-3M accepted a missing physics modality.")
        try:
            forward_final2_model(model, "CNN-3M", discharge, charge, torch.zeros(2, 4))
        except ValueError as error:
            assert "physics features" in str(error)
        else:
            raise AssertionError("CNN-3M accepted an invalid physics feature count.")

    two_modality = build_final2_model("DenseNet-2M").eval()
    try:
        forward_final2_model(
            two_modality, "DenseNet-2M", discharge, charge, torch.zeros(2, 5)
        )
    except ValueError as error:
        assert "only discharge and charge" in str(error)
    else:
        raise AssertionError("DenseNet-2M accepted an unsupported third modality.")


def test_unknown_final2_model_is_reported():
    try:
        build_final2_model("unknown")
    except ValueError as error:
        assert "Unknown Final2 architecture" in str(error)
    else:
        raise AssertionError("Unknown architecture unexpectedly built.")


def test_runtime_supports_cpu_seed_reproducibility_and_optional_autocast(monkeypatch):
    assert resolve_device("cpu") == torch.device("cpu")
    with autocast_context(torch.device("cpu"), enabled=True):
        assert torch.isfinite(torch.ones(2) + torch.ones(2)).all()

    seed_everything(123)
    first = torch.rand(4)
    seed_everything(123)
    torch.testing.assert_close(torch.rand(4), first)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device() == torch.device("cpu")
    try:
        resolve_device("cuda")
    except RuntimeError as error:
        assert "not available" in str(error)
    else:
        raise AssertionError("Unavailable CUDA device was not rejected.")
    try:
        seed_everything(-1)
    except ValueError as error:
        assert "between 0 and" in str(error)
    else:
        raise AssertionError("Negative seed was not rejected.")
