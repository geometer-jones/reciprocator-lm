import argparse

import torch


DEVICE_CHOICES = ("cpu", "cuda", "mps", "auto")


def add_device_argument(parser: argparse.ArgumentParser, *, default: str = "cpu") -> None:
    if default not in DEVICE_CHOICES:
        raise ValueError(f"unsupported device default: {default}")
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default=default,
        help=f"Execution device. Defaults to {default}; use auto to prefer cuda or mps when available.",
    )


def _mps_is_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def resolve_torch_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if normalized == "mps":
        if not _mps_is_available():
            raise ValueError("MPS requested but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    raise ValueError(f"unsupported device selection: {requested}")
