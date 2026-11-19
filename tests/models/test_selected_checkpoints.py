import torch

from app.model_runtime import Final2ModelRuntime
from benchmark import MODELS


def test_selected_checkpoints_load_strictly_on_cpu():
    runtime = Final2ModelRuntime()
    runtime.device = torch.device("cpu")
    for name in MODELS:
        assert runtime._load_model(name) is not None, runtime.load_errors
