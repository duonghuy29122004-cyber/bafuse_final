import sys

import pytest
import train
from benchmark import MODELS


def test_updated_physics_training_requires_verified_features(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train, "__file__", str(tmp_path / "train.py"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--model", "CNN-3M"])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 1
    assert "unverified updated Physics" in capsys.readouterr().err


@pytest.mark.parametrize("model", MODELS[1:])
def test_two_modality_cli_dispatches_one_model_without_a_physics_audit(
    model, tmp_path, monkeypatch
):
    from bafuse.training import updated_cnn

    destination = tmp_path / "candidate"
    calls = []

    def stage(root, coverage, *, model_name, output):
        calls.append((root, model_name, output, coverage.exists()))
        return {}

    monkeypatch.setattr(train, "__file__", str(tmp_path / "train.py"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--model", model, "--output-dir", "candidate"])
    monkeypatch.setattr(updated_cnn, "train_candidate", stage)
    monkeypatch.setattr(train, "print_results", lambda _: None)
    train.main()
    assert calls == [(tmp_path, model, destination, False)]


def test_removed_cnn_2m_is_rejected_by_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--model", "CNN-2M"])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
