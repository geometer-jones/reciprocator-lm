from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations
from typing import Callable, Iterator, Optional

import torch
from torch import Tensor, nn


_MODE_PARAMETER_TRANSFORMS: dict[str, Callable[[Tensor], Tensor]] = {
    "input_gain": torch.sigmoid,
    "recurrent_gain": torch.tanh,
    "carry_gain": torch.tanh,
    "decay": torch.sigmoid,
}


def _active_slice(active_sizes: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(0, int(size)) for size in active_sizes)


def _iter_mode_parameters(
    model: nn.Module,
) -> Iterator[tuple[str, nn.Parameter, Callable[[Tensor], Tensor], tuple[int, ...]]]:
    for layer_index, block in enumerate(getattr(model, "blocks", [])):
        mixer = getattr(block, "mixer", None)
        engines = getattr(mixer, "cube_engines", [])
        diagnostics = mixer.diagnostics() if mixer is not None and hasattr(mixer, "diagnostics") else {}
        mixer_active_sizes = tuple(int(size) for size in diagnostics.get("active_sizes", getattr(mixer, "init_mode_sizes", ())))
        for engine_index, engine in enumerate(engines):
            for parameter_name, transform in _MODE_PARAMETER_TRANSFORMS.items():
                parameter = getattr(engine, parameter_name, None)
                if not isinstance(parameter, nn.Parameter):
                    continue
                active_sizes = (
                    mixer_active_sizes if len(mixer_active_sizes) == parameter.ndim else tuple(parameter.shape)
                )
                yield (
                    f"layer_{layer_index}.engine_{engine_index}.{parameter_name}",
                    parameter,
                    transform,
                    active_sizes,
                )


def _effective_tensor(
    parameter: nn.Parameter,
    transform: Callable[[Tensor], Tensor],
    active_sizes: tuple[int, ...],
) -> Tensor:
    tensor = transform(parameter.detach()).float()
    return tensor[_active_slice(active_sizes)]


def _pair_supported(
    parameters: list[tuple[str, nn.Parameter, Callable[[Tensor], Tensor], tuple[int, ...]]],
    axis_a: int,
    axis_b: int,
) -> bool:
    return all(
        parameter.shape[axis_a] == parameter.shape[axis_b] and active_sizes[axis_a] == active_sizes[axis_b]
        for _, parameter, _, active_sizes in parameters
    )


@contextmanager
def _temporarily_swapped_mode_parameters(
    parameters: list[tuple[str, nn.Parameter, Callable[[Tensor], Tensor], tuple[int, ...]]],
    axis_a: int,
    axis_b: int,
) -> Iterator[None]:
    originals = [(parameter, parameter.detach().clone()) for _, parameter, _, _ in parameters]
    permutation: list[int] | None = None
    try:
        with torch.no_grad():
            for parameter, original in originals:
                if permutation is None:
                    permutation = list(range(parameter.ndim))
                    permutation[axis_a], permutation[axis_b] = permutation[axis_b], permutation[axis_a]
                parameter.copy_(original.permute(*permutation).contiguous())
        yield
    finally:
        with torch.no_grad():
            for parameter, original in originals:
                parameter.copy_(original)


def _compute_parameter_deltas(
    parameters: list[tuple[str, nn.Parameter, Callable[[Tensor], Tensor], tuple[int, ...]]],
    axis_a: int,
    axis_b: int,
) -> tuple[float, float]:
    """Compute mean and max relative parameter deltas for a mode-axis pair."""
    deltas = []
    for _, parameter, transform, active_sizes in parameters:
        tensor = _effective_tensor(parameter, transform, active_sizes)
        permutation = list(range(tensor.ndim))
        permutation[axis_a], permutation[axis_b] = permutation[axis_b], permutation[axis_a]
        swapped = tensor.permute(*permutation)
        denominator = float(torch.linalg.vector_norm(tensor).item())
        delta = 0.0 if denominator <= 1e-8 else float(
            (torch.linalg.vector_norm(tensor - swapped) / denominator).item()
        )
        deltas.append(delta)
    mean_delta = sum(deltas) / len(deltas)
    return mean_delta, max(deltas)


_SYMMETRY_THRESHOLD = 1e-4


def mode_axis_permutation_probe(
    model: nn.Module,
    *,
    evaluate_fn: Optional[Callable[[], float]] = None,
    metric_name: str = "loss",
    baseline_metric: Optional[float] = None,
) -> dict[str, object]:
    parameters = list(_iter_mode_parameters(model))
    if not parameters:
        return {
            "supported_pair_count": 0,
            "parameter_count": 0,
            "pairs": {},
            "reason": "model does not expose cube-engine mode parameters",
        }

    rank = parameters[0][1].ndim
    diagnostics: dict[str, object] = {}
    first_mixer = getattr(getattr(model, "blocks", [None])[0], "mixer", None) if getattr(model, "blocks", None) else None
    if first_mixer is not None and hasattr(first_mixer, "diagnostics"):
        diagnostics = first_mixer.diagnostics()
    results: dict[str, object] = {
        "state_rank": rank,
        "active_rank": diagnostics.get("active_rank", rank),
        "parameter_count": len(parameters),
        "pairs": {},
    }

    baseline_value: Optional[float] = None
    if evaluate_fn is not None:
        baseline_value = float(baseline_metric) if baseline_metric is not None else float(evaluate_fn())
        results[f"baseline_{metric_name}"] = baseline_value

    supported_pair_count = 0
    max_parameter_delta = 0.0
    max_metric_relative_delta = 0.0
    for axis_a, axis_b in combinations(range(rank), 2):
        pair_key = f"{axis_a}-{axis_b}"
        supported = _pair_supported(parameters, axis_a, axis_b)
        pair_result: dict[str, object] = {"supported": supported}
        if not supported:
            pair_result["reason"] = "mode sizes differ in the active or full tensor shape; swap is not comparable"
            results["pairs"][pair_key] = pair_result
            continue

        supported_pair_count += 1
        mean_delta, max_delta = _compute_parameter_deltas(parameters, axis_a, axis_b)
        pair_result["parameter_relative_delta_mean"] = mean_delta
        pair_result["parameter_relative_delta_max"] = max_delta
        max_parameter_delta = max(max_parameter_delta, mean_delta)

        if evaluate_fn is not None and baseline_value is not None:
            if mean_delta < _SYMMETRY_THRESHOLD:
                pair_result[metric_name] = baseline_value
                pair_result[f"{metric_name}_delta"] = 0.0
                pair_result[f"{metric_name}_relative_delta"] = 0.0
                pair_result["metric_skipped_reason"] = (
                    f"parameter delta ({mean_delta:.2e}) below threshold ({_SYMMETRY_THRESHOLD:.0e}); "
                    "parameters are symmetric, metric evaluation skipped"
                )
            else:
                with _temporarily_swapped_mode_parameters(parameters, axis_a, axis_b):
                    permuted_metric = float(evaluate_fn())
                relative_metric_delta = abs(permuted_metric - baseline_value) / max(abs(baseline_value), 1e-8)
                pair_result[metric_name] = permuted_metric
                pair_result[f"{metric_name}_delta"] = permuted_metric - baseline_value
                pair_result[f"{metric_name}_relative_delta"] = relative_metric_delta
                max_metric_relative_delta = max(max_metric_relative_delta, relative_metric_delta)

        results["pairs"][pair_key] = pair_result

    results["supported_pair_count"] = supported_pair_count
    results["max_parameter_relative_delta_mean"] = max_parameter_delta
    if evaluate_fn is not None:
        results[f"max_{metric_name}_relative_delta"] = max_metric_relative_delta
    return results
