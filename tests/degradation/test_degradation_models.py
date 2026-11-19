from __future__ import annotations

import numpy as np
import pytest
import torch

from bafuse.degradation.models import (
    APPROACH_NAMES,
    ENCODER_NAMES,
    DegradationPredictions,
    TwoStageDegradationModel,
    build_degradation_model,
)
from bafuse.degradation.targets import (
    TARGET_NAMES,
    LabelProvenance,
    build_target_masks,
    fit_target_standardizer,
    masked_training_mse,
)
from bafuse.hybrid.soh_fusion import PHYSICS_FEATURE_NAMES


@pytest.mark.parametrize("approach", APPROACH_NAMES)
@pytest.mark.parametrize("encoder", ENCODER_NAMES)
def test_model_family_has_shared_v2_prediction_contract(approach: str, encoder: str):
    model = build_degradation_model(approach, encoder).eval()
    inputs = (
        torch.zeros(3, 128),
        torch.zeros(3, 128),
        torch.zeros(3, len(PHYSICS_FEATURE_NAMES)),
    )

    with torch.inference_mode():
        predictions = model(*inputs)

    assert isinstance(predictions, DegradationPredictions)
    assert predictions.as_tensor().shape == (3, len(TARGET_NAMES))
    assert torch.isfinite(predictions.as_tensor()).all()


@pytest.mark.parametrize("approach", APPROACH_NAMES)
def test_models_reject_incompatible_v2_inputs(approach: str):
    model = build_degradation_model(approach, "cnn1d")

    with pytest.raises(ValueError, match="discharge voltage curve"):
        model(
            torch.zeros(2, 127),
            torch.zeros(2, 128),
            torch.zeros(2, len(PHYSICS_FEATURE_NAMES)),
        )


def test_two_stage_inference_has_no_true_soh_input_and_can_freeze_stage1():
    model = build_degradation_model("two_stage", "cnn1d")
    assert isinstance(model, TwoStageDegradationModel)
    model.freeze_stage1()

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.soh_head.parameters())
    assert any(
        parameter.requires_grad
        for parameter in model.diagnostic_heads.parameters()
    )
    stage2_inputs: list[torch.Tensor] = []
    hook = model.diagnostic_heads["lli"].register_forward_pre_hook(
        lambda _module, args: stage2_inputs.append(args[0][:, -1:].detach())
    )
    inputs = (
        torch.zeros(2, 128),
        torch.zeros(2, 128),
        torch.zeros(2, len(PHYSICS_FEATURE_NAMES)),
    )
    with torch.inference_mode():
        predictions = model(*inputs)
    hook.remove()

    torch.testing.assert_close(stage2_inputs[0], predictions.soh)
    with pytest.raises(TypeError):
        model(
            *inputs,
            true_soh=torch.ones(2, 1),
        )


def test_unknown_model_components_fail_explicitly():
    with pytest.raises(ValueError, match="approach"):
        build_degradation_model("unknown", "cnn1d")
    with pytest.raises(ValueError, match="encoder"):
        build_degradation_model("multitask", "unknown")


def _provenance(**overrides) -> LabelProvenance:
    values = {
        "split": "train",
        "dataset_part": "1",
        "group_id": "Group 1",
        "physical_cell_id": "G1-Cell1",
        "rpt": "RPT2",
        "source_artifact": "saved_results.csv",
        "source_row_id": "saved_results.csv:12",
        "procedure": "TPG1.6",
        "source_file": "TPG1.6-Cell1.csv",
        "mapping_status": "verified",
        "curve_status": "verified",
        "soh_label_status": "ACCEPTED",
        "diagnostic_label_status": "diagnostic_estimate_accepted",
        "direction": "discharge",
        "direction_policy_id": "reviewed-direction-policy-v1",
        "reference_source": "part1-bol.csv",
        "reference_status": "verified",
        "fit_rmse_mV": 14.9,
        "accepted_runs": 3,
        "fit_is_accepted": True,
    }
    values.update(overrides)
    return LabelProvenance(**values)


def test_target_masks_require_explicit_mapping_reference_direction_and_fit_qc():
    targets = np.asarray([[90.0, -2.0, 3.0, 4.0]])
    accepted = _provenance()
    audit = build_target_masks(
        targets,
        [accepted],
        approved_direction_policy_ids={"reviewed-direction-policy-v1"},
    )
    np.testing.assert_array_equal(audit.masks, [[True, True, True, True]])

    blocked = build_target_masks(
        targets,
        [
            _provenance(
                fit_rmse_mV=15.0,
                mapping_status="needs_curve_audit",
                direction_policy_id=None,
            )
        ],
        approved_direction_policy_ids={"reviewed-direction-policy-v1"},
    )
    assert not blocked.masks.any()
    assert {
        "mapping_not_verified",
        "direction_policy_not_approved",
        "pydma_fit_gate_failed",
    }.issubset(blocked.exclusion_reasons[0])
    assert targets[0, 1] == -2.0


@pytest.mark.parametrize("group_id", ["5", "Group 6"])
def test_part2_groups_5_and_6_are_excluded_from_all_target_masks(group_id: str):
    audit = build_target_masks(
        np.asarray([[90.0, 1.0, 2.0, 3.0]]),
        [_provenance(dataset_part="2", group_id=group_id)],
        approved_direction_policy_ids={"reviewed-direction-policy-v1"},
    )

    assert not audit.masks.any()
    assert "part2_group5_6_excluded" in audit.exclusion_reasons[0]


def test_target_normalization_uses_only_train_observed_values_and_round_trips():
    targets = np.asarray(
        [
            [80.0, -5.0, 2.0, 3.0],
            [100.0, 5.0, 4.0, 7.0],
            [9999.0, 9999.0, 9999.0, 9999.0],
        ]
    )
    masks = np.ones_like(targets, dtype=bool)
    scaler = fit_target_standardizer(
        targets,
        build_target_masks(
            targets,
            [
                _provenance(split="train"),
                _provenance(split="train", rpt="RPT3"),
                _provenance(
                    split="external",
                    dataset_part="3",
                    group_id="Group 7",
                    physical_cell_id="G7-Cell1",
                    rpt="RPT1",
                ),
            ],
            approved_direction_policy_ids={"reviewed-direction-policy-v1"},
        ),
    )

    np.testing.assert_allclose(scaler.mean, [90.0, 0.0, 3.0, 5.0])
    assert scaler.fit_counts.tolist() == [2, 2, 2, 2]
    normalized = scaler.transform(targets[:2], masks[:2])
    np.testing.assert_allclose(scaler.inverse_transform(normalized), targets[:2])


def test_target_normalizer_requires_each_target_in_train():
    targets = np.asarray([[90.0, np.nan, 2.0, 3.0]])
    masks = np.asarray([[True, False, True, True]])

    with pytest.raises(ValueError, match="no accepted observations.*lli"):
        fit_target_standardizer(
            targets,
            build_target_masks(
                targets,
                [_provenance()],
                approved_direction_policy_ids={"reviewed-direction-policy-v1"},
            ),
        )


def test_masked_training_loss_skips_missing_targets_and_blocks_external_rows():
    predictions = DegradationPredictions(
        torch.tensor([[1.0], [1.0]], requires_grad=True),
        torch.tensor([[2.0], [2.0]], requires_grad=True),
        torch.tensor([[3.0], [3.0]], requires_grad=True),
        torch.tensor([[4.0], [4.0]], requires_grad=True),
    )
    targets = torch.tensor([[0.0, 0.0, 0.0, 0.0], [float("nan")] * 4])
    masks = torch.tensor(
        [[True, True, True, True], [False, False, False, False]],
        dtype=torch.bool,
    )
    loss = masked_training_mse(
        predictions,
        targets,
        masks,
        [_provenance(), _provenance(rpt="RPT3")],
    )

    assert loss.item() == pytest.approx(7.5)
    loss.backward()
    assert predictions.soh.grad is not None
    with pytest.raises(ValueError, match="external labels are blocked"):
        masked_training_mse(
            predictions,
            targets,
            masks,
            [_provenance(), _provenance(split="external", rpt="RPT3")],
        )
    with pytest.raises(ValueError, match="no observed targets"):
        masked_training_mse(
            predictions,
            targets,
            torch.zeros_like(masks),
            [_provenance(), _provenance(rpt="RPT3")],
        )
