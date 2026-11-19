from benchmark import MODELS, catalog


def test_official_benchmark_separates_baseline_and_two_modality_models():
    rows = [entry for entry, _ in catalog()]
    assert MODELS == ("CNN-3M", "TCN-2M", "ResNet-2M", "DenseNet-2M", "ConvNeXt-2M")
    assert tuple(entry["model"] for entry in rows) == MODELS
    assert rows[0]["modalities"] == ["Discharge", "Charge", "Physics"]
    assert rows[0]["physics_revision"] == "updated Physics"
    assert rows[0]["promoted"] is False
    assert all(row["modalities"] == ["Discharge", "Charge"] for row in rows[1:])
