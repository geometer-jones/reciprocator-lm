import argparse
import importlib.util
from pathlib import Path

import pytest
import torch


def _load_runtime_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "reciprocator_lm" / "runtime.py"
    spec = importlib.util.spec_from_file_location("reciprocator_runtime", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_add_device_argument_defaults_to_cpu() -> None:
    runtime = _load_runtime_module()
    parser = argparse.ArgumentParser()
    runtime.add_device_argument(parser)

    args = parser.parse_args([])

    assert args.device == "cpu"


def test_add_device_argument_help_uses_configured_default() -> None:
    runtime = _load_runtime_module()
    parser = argparse.ArgumentParser()
    runtime.add_device_argument(parser, default="auto")

    help_text = parser.format_help()

    assert "Defaults to auto" in help_text


def test_resolve_torch_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _load_runtime_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    device = runtime.resolve_torch_device("auto")

    assert device.type == "cuda"


def test_resolve_torch_device_auto_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _load_runtime_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    device = runtime.resolve_torch_device("auto")

    assert device.type == "mps"


def test_resolve_torch_device_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _load_runtime_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA requested"):
        runtime.resolve_torch_device("cuda")


def test_resolve_torch_device_rejects_unavailable_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _load_runtime_module()
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(ValueError, match="MPS requested"):
        runtime.resolve_torch_device("mps")
