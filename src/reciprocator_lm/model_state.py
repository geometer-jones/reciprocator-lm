from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

def _parallel_scan_linear(decay: Tensor, inputs: Tensor) -> Tensor:
    """Parallel prefix scan for y[t] = decay * y[t-1] + inputs[t], y[-1] = 0.

    Kogge-Stone inclusive scan with semiring (a2,b2)⊗(a1,b1) = (a2*a1, a2*b1+b2).
    decay: [D] or [B, T, D] - per-element multiplicative factor
    inputs: [B, T, D] - per-timestep additive terms
    Returns: [B, T, D]
    """
    n = inputs.shape[1]
    if decay.ndim == 1:
        a = decay.unsqueeze(0).unsqueeze(0).expand_as(inputs).clone()
    elif decay.shape == inputs.shape:
        a = decay.clone()
    else:
        raise ValueError("decay must have shape [D] or match inputs with shape [B, T, D]")
    b = inputs.clone()
    stride = 1
    while stride < n:
        right_a = a[:, stride:]
        left_a = a[:, :-stride]
        left_b = b[:, :-stride]
        right_b = b[:, stride:]
        new_a = torch.cat([a[:, :stride], right_a * left_a], dim=1)
        new_b = torch.cat([b[:, :stride], right_a * left_b + right_b], dim=1)
        a = new_a
        b = new_b
        stride *= 2
    return b


def _inverse_softplus(value: float, *, eps: float = 1e-6) -> float:
    clamped = max(float(value), eps)
    return math.log(math.expm1(clamped))


def _inverse_sigmoid(value: float, *, eps: float = 1e-6) -> float:
    clamped = min(max(float(value), eps), 1.0 - eps)
    return math.log(clamped / (1.0 - clamped))


def _normalize_complex(real: Tensor, imag: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    norm = torch.sqrt((real.square() + imag.square()).sum(dim=-1, keepdim=True).clamp_min(eps))
    return real / norm, imag / norm


def _active_mode_dims(tensor_ndim: int, state_rank: int, active_rank: Optional[int] = None) -> tuple[int, ...]:
    effective_rank = state_rank if active_rank is None else int(active_rank)
    if effective_rank < 0 or effective_rank > state_rank:
        raise ValueError("active_rank must be in [0, state_rank]")
    start = tensor_ndim - state_rank
    return tuple(range(start, start + effective_rank))


def _normalize_complex_frobenius(
    real: Tensor,
    imag: Tensor,
    state_rank: int,
    active_rank: Optional[int] = None,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    dims = _active_mode_dims(real.ndim, state_rank, active_rank)
    if not dims:
        return real, imag
    norm = torch.sqrt((real.square() + imag.square()).sum(dim=dims, keepdim=True).clamp_min(eps))
    return real / norm, imag / norm


def _normalize_complex_per_mode_unrolled(
    real: Tensor,
    imag: Tensor,
    state_rank: int,
    active_rank: Optional[int] = None,
    eps: float = 1e-6,
    max_iters: int = 3,
    step_sizes: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    dims = _active_mode_dims(real.ndim, state_rank, active_rank)
    if not dims:
        return real, imag
    if step_sizes is None:
        resolved_step_sizes = real.new_ones(len(dims))
    else:
        resolved_step_sizes = step_sizes.to(dtype=real.dtype, device=real.device).reshape(-1)
        if resolved_step_sizes.numel() == 1:
            resolved_step_sizes = resolved_step_sizes.expand(len(dims))
        elif resolved_step_sizes.numel() < len(dims):
            raise ValueError("step_sizes must provide at least one value per active mode")
        else:
            resolved_step_sizes = resolved_step_sizes[: len(dims)]
    for _ in range(max_iters):
        magnitude_sq = real.square() + imag.square()
        # Fixed sweeps keep the normalization compile-friendly and avoid host syncs.
        for mode_idx, axis in enumerate(dims):
            axis_norm = torch.sqrt(magnitude_sq.sum(dim=axis, keepdim=True).clamp_min(eps))
            axis_scale = axis_norm.pow(resolved_step_sizes[mode_idx])
            real = real / axis_scale
            imag = imag / axis_scale
            magnitude_sq = real.square() + imag.square()
    return real, imag


def _normalize_complex_tensor(
    real: Tensor,
    imag: Tensor,
    mode: str,
    state_rank: int,
    active_rank: Optional[int] = None,
    eps: float = 1e-6,
    max_iter: int = 3,
    step_sizes: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    if mode == "frobenius":
        return _normalize_complex_frobenius(
            real,
            imag,
            state_rank=state_rank,
            active_rank=active_rank,
            eps=eps,
        )
    if mode == "per_mode":
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        return _normalize_complex_per_mode_unrolled(
            real,
            imag,
            state_rank=state_rank,
            active_rank=active_rank,
            eps=eps,
            max_iters=max_iter,
            step_sizes=step_sizes,
        )
    raise ValueError(f"unsupported normalization mode: {mode}")


def _prepare_per_mode_step_sizes_state_dict(
    per_mode_step_sizes: Optional[Tensor],
    state_dict: dict[str, Tensor],
    prefix: str,
) -> None:
    if per_mode_step_sizes is None:
        return
    state_dict.setdefault(prefix + "per_mode_step_sizes", per_mode_step_sizes.detach().clone())


def _prepare_optional_parameter_state_dict(
    parameter: Optional[Tensor],
    state_dict: dict[str, Tensor],
    prefix: str,
    name: str,
) -> None:
    if parameter is None:
        return
    state_dict.setdefault(prefix + name, parameter.detach().clone())


def _prepare_optional_module_state_dict(
    module: Optional[nn.Module],
    state_dict: dict[str, Tensor],
    prefix: str,
) -> None:
    if module is None:
        return
    for name, value in module.state_dict().items():
        state_dict.setdefault(prefix + name, value.detach().clone())


def _complex_mul(ar: Tensor, ai: Tensor, br: Tensor, bi: Tensor) -> tuple[Tensor, Tensor]:
    return ar * br - ai * bi, ar * bi + ai * br


def _complex_dtype_for(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def _phase_preserving_coupling(
    logits_real: Tensor,
    logits_imag: Tensor,
    *,
    scale: float,
    temperature: Union[float, Tensor],
) -> Tensor:
    magnitude_sq = logits_real.square() + logits_imag.square()
    scaled_magnitude = torch.sqrt(magnitude_sq + 1e-8) / max(scale, 1e-8)
    weights = F.softmax(scaled_magnitude / temperature, dim=-1)

    # Reuse the normalized complex score as the unit phasor instead of calling
    # angle()+exp(); this is algebraically equivalent and avoids extra trig.
    unit_denom = torch.sqrt(magnitude_sq + 1e-8)
    unit_real = logits_real / unit_denom
    unit_imag = logits_imag / unit_denom
    zero_mask = magnitude_sq <= 1e-12
    unit_real = torch.where(zero_mask, torch.ones_like(unit_real), unit_real)
    unit_imag = torch.where(zero_mask, torch.zeros_like(unit_imag), unit_imag)
    return torch.complex(weights * unit_real, weights * unit_imag)


def _partial_trace_couplings(
    state_real: Tensor,
    state_imag: Tensor,
    state_rank: int,
    active_rank: Optional[int] = None,
    *,
    phase_aware_coupling: bool = True,
    coupling_temperature: Union[float, Tensor] = 1.0,
) -> list[Tensor]:
    magnitude = torch.sqrt(state_real.square() + state_imag.square() + 1e-8)
    complex_state = torch.complex(state_real, state_imag)
    batch_dims = magnitude.ndim - state_rank
    batch_shape = magnitude.shape[:batch_dims]
    effective_active_rank = state_rank if active_rank is None else int(active_rank)
    couplings = []
    for mode_idx in range(state_rank):
        axis = batch_dims + mode_idx
        mode_size = magnitude.shape[axis]
        if mode_idx >= effective_active_rank:
            if phase_aware_coupling:
                eye = torch.eye(
                    mode_size,
                    dtype=_complex_dtype_for(state_real.dtype),
                    device=state_real.device,
                )
            else:
                eye = torch.eye(mode_size, dtype=state_real.dtype, device=state_real.device)
            couplings.append(eye.expand(*batch_shape, mode_size, mode_size))
            continue
        if not phase_aware_coupling and state_rank == 3:
            coupling = magnitude.sum(dim=axis)
        elif not phase_aware_coupling:
            moved = torch.movedim(magnitude, axis, batch_dims)
            flattened = moved.reshape(*magnitude.shape[:batch_dims], mode_size, -1)
            flattened = F.normalize(flattened, dim=-1, eps=1e-6)
            coupling = torch.matmul(flattened, flattened.transpose(-1, -2))
        elif state_rank == 3:
            traced = complex_state.sum(dim=axis)
            if traced.shape[-2:] == (mode_size, mode_size):
                coupling = _phase_preserving_coupling(
                    traced.real,
                    traced.imag,
                    scale=1.0,
                    temperature=coupling_temperature,
                )
            else:
                moved = torch.movedim(complex_state, axis, batch_dims)
                flattened = moved.reshape(*batch_shape, mode_size, -1)
                flattened = flattened / flattened.abs().square().sum(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
                logits = torch.matmul(flattened, flattened.conj().transpose(-1, -2))
                coupling = _phase_preserving_coupling(
                    logits.real,
                    logits.imag,
                    scale=math.sqrt(flattened.shape[-1]),
                    temperature=coupling_temperature,
                )
        else:
            mode_size = complex_state.shape[axis]
            moved = torch.movedim(complex_state, axis, batch_dims)
            flattened = moved.reshape(*batch_shape, mode_size, -1)
            flattened = flattened / flattened.abs().square().sum(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
            logits = torch.matmul(flattened, flattened.conj().transpose(-1, -2))
            coupling = _phase_preserving_coupling(
                logits.real,
                logits.imag,
                scale=math.sqrt(flattened.shape[-1]),
                temperature=coupling_temperature,
            )
        if not phase_aware_coupling:
            coupling = F.softmax(coupling, dim=-1)
        couplings.append(coupling)
    return couplings


def _apply_single_mode_coupling(tensor: Tensor, coupling: Tensor, *, state_rank: int, mode_idx: int) -> Tensor:
    mixed = tensor
    batch_dims = mixed.ndim - state_rank
    axis = batch_dims + mode_idx
    mixed = torch.movedim(mixed, axis, -1)
    mode_size = mixed.shape[-1]
    if coupling.shape[-2:] != (mode_size, mode_size):
        raise ValueError(
            f"mode coupling shape {tuple(coupling.shape)} is incompatible with mode size {mode_size}"
        )
    if coupling.shape[:-2] != mixed.shape[:batch_dims]:
        raise ValueError("mode couplings could not be broadcast across the active tensor slices")
    vector = mixed.unsqueeze(-2)
    if torch.is_complex(coupling) and not torch.is_complex(vector):
        vector = vector.to(dtype=_complex_dtype_for(mixed.dtype))
    elif torch.is_complex(vector) and not torch.is_complex(coupling):
        coupling = coupling.to(dtype=vector.dtype)
    expanded_coupling = coupling.reshape(
        *coupling.shape[:-2],
        *([1] * (vector.ndim - coupling.ndim)),
        mode_size,
        mode_size,
    )
    mixed = torch.matmul(vector, expanded_coupling).squeeze(-2)
    mixed = torch.movedim(mixed, -1, axis)
    return mixed


def _apply_mode_couplings(tensor: Tensor, couplings: list[Tensor], state_rank: int) -> Tensor:
    mixed = tensor
    for mode_idx, coupling in enumerate(couplings):
        mixed = _apply_single_mode_coupling(mixed, coupling, state_rank=state_rank, mode_idx=mode_idx)
    return mixed


def _apply_mode_couplings_pair(
    real: Tensor,
    imag: Tensor,
    couplings: list[Tensor],
    state_rank: int,
) -> tuple[Tensor, Tensor]:
    if real.shape != imag.shape:
        raise ValueError("real and imag tensors must have matching shapes")
    mixed = _apply_mode_couplings(torch.complex(real, imag), couplings, state_rank)
    return mixed.real, mixed.imag


def _project_mode_coupling_layout(coupling: Tensor, *, layout: str) -> Tensor:
    if layout == "full":
        return coupling
    if layout != "diagonal":
        raise ValueError(f"unsupported mode coupling layout: {layout}")
    diagonal = torch.diagonal(coupling, dim1=-2, dim2=-1)
    return torch.diag_embed(diagonal)


def _active_slice(active_sizes: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(0, int(size)) for size in active_sizes)


def _mask_to_active(tensor: Tensor, active_sizes: tuple[int, ...], state_rank: int) -> Tensor:
    masked = tensor
    batch_dims = tensor.ndim - state_rank
    mask_dtype = tensor.real.dtype if torch.is_complex(tensor) else tensor.dtype
    for mode_idx, active in enumerate(active_sizes):
        axis = batch_dims + mode_idx
        mask = torch.zeros(tensor.shape[axis], dtype=mask_dtype, device=tensor.device)
        mask[: int(active)] = 1.0
        shape = [1] * tensor.ndim
        shape[axis] = tensor.shape[axis]
        masked = masked * mask.view(*shape)
    return masked

def _relational_product(
    signal_real: Tensor,
    signal_imag: Tensor,
    state_real: Tensor,
    state_imag: Tensor,
    active_sizes: tuple[int, ...],
    state_rank: int,
) -> tuple[Tensor, Tensor]:
    product_real, product_imag = _complex_mul(signal_real, signal_imag, state_real, state_imag)
    product_real = _mask_to_active(product_real, active_sizes, state_rank)
    product_imag = _mask_to_active(product_imag, active_sizes, state_rank)
    return product_real, product_imag


def _relational_gain_statistics(
    product_real: Tensor,
    product_imag: Tensor,
    active_sizes: tuple[int, ...],
    state_rank: int,
    active_rank: Optional[int] = None,
) -> Tensor:
    if product_real.shape != product_imag.shape:
        raise ValueError("product_real and product_imag must have matching shapes")

    batch_dims = product_real.ndim - state_rank
    reduce_dims = tuple(range(batch_dims, product_real.ndim))
    active_count = float(math.prod(int(size) for size in active_sizes))
    effective_active_rank = state_rank if active_rank is None else int(active_rank)

    magnitude = torch.sqrt(product_real.square() + product_imag.square() + 1e-8)
    unit_real = product_real / magnitude.clamp_min(1e-6)
    unit_imag = product_imag / magnitude.clamp_min(1e-6)

    # Use circular mean components for phase so alignment remains continuous at
    # the wrap boundary instead of jumping at +/-pi.
    mean_phase_real = unit_real.sum(dim=reduce_dims) / active_count
    mean_phase_imag = unit_imag.sum(dim=reduce_dims) / active_count
    mean_magnitude = magnitude.sum(dim=reduce_dims) / active_count

    magnitude_sq = magnitude.square()
    mode_energy_stats = []
    for mode_idx, active_size in enumerate(active_sizes):
        if mode_idx >= effective_active_rank:
            mode_energy_stats.append(mean_magnitude.new_zeros(mean_magnitude.shape))
            continue
        axis = batch_dims + mode_idx
        other_dims = tuple(dim for dim in reduce_dims if dim != axis)
        other_count = float(active_count / max(1, int(active_size)))
        if other_dims:
            fiber_energy = magnitude_sq.sum(dim=other_dims) / other_count
        else:
            fiber_energy = magnitude_sq
        mode_energy_stats.append(torch.sqrt(fiber_energy.square().mean(dim=-1) + 1e-8))

    return torch.cat(
        (
            mean_phase_real.unsqueeze(-1),
            mean_phase_imag.unsqueeze(-1),
            mean_magnitude.unsqueeze(-1),
            torch.stack(mode_energy_stats, dim=-1),
        ),
        dim=-1,
    )


def _summarize_complex_tensor(
    real: Tensor,
    imag: Tensor,
    *,
    active_sizes: tuple[int, ...],
    state_rank: int,
    active_rank: Optional[int] = None,
) -> Tensor:
    del active_rank  # Summary should collapse the entire masked state tensor.
    dims = tuple(range(real.ndim - state_rank, real.ndim))
    if not dims:
        leading_shape = real.shape[: real.ndim - state_rank]
        return torch.zeros(*leading_shape, 3, dtype=real.dtype, device=real.device)
    masked_real = _mask_to_active(real, active_sizes, state_rank)
    masked_imag = _mask_to_active(imag, active_sizes, state_rank)
    magnitude = torch.sqrt(masked_real.square() + masked_imag.square() + 1e-6)
    active_count = float(max(1, math.prod(int(size) for size in active_sizes)))
    return torch.cat(
        (
            (masked_real.sum(dim=dims, keepdim=False) / active_count).unsqueeze(-1),
            (masked_imag.sum(dim=dims, keepdim=False) / active_count).unsqueeze(-1),
            (magnitude.sum(dim=dims, keepdim=False) / active_count).unsqueeze(-1),
        ),
        dim=-1,
    )


def _mode_energy_summary(
    magnitude: Tensor,
    *,
    active_sizes: tuple[int, ...],
    state_rank: int,
    active_rank: Optional[int] = None,
) -> Tensor:
    batch_dims = magnitude.ndim - state_rank
    reduce_dims = tuple(range(batch_dims, magnitude.ndim))
    active_count = float(max(1, math.prod(int(size) for size in active_sizes)))
    effective_active_rank = state_rank if active_rank is None else int(active_rank)
    magnitude_sq = magnitude.square()
    mode_energy_stats = []
    for mode_idx, active_size in enumerate(active_sizes):
        if mode_idx >= effective_active_rank:
            leading_shape = magnitude.shape[:batch_dims]
            mode_energy_stats.append(torch.zeros(*leading_shape, dtype=magnitude.dtype, device=magnitude.device))
            continue
        axis = batch_dims + mode_idx
        other_dims = tuple(dim for dim in reduce_dims if dim != axis)
        other_count = float(active_count / max(1, int(active_size)))
        if other_dims:
            fiber_energy = magnitude_sq.sum(dim=other_dims) / other_count
        else:
            fiber_energy = magnitude_sq
        mode_energy_stats.append(torch.sqrt(fiber_energy.square().mean(dim=-1) + 1e-8))
    return torch.stack(mode_energy_stats, dim=-1)


def _engine_state_readout_features(
    state_real: Tensor,
    state_imag: Tensor,
    magnitude_accumulator: Tensor,
    *,
    active_sizes: tuple[int, ...],
    state_rank: int,
    active_rank: Optional[int] = None,
) -> Tensor:
    accumulator_imag = torch.zeros_like(magnitude_accumulator)
    state_summary = _summarize_complex_tensor(
        state_real,
        state_imag,
        active_sizes=active_sizes,
        state_rank=state_rank,
        active_rank=active_rank,
    )
    accumulator_summary = _summarize_complex_tensor(
        magnitude_accumulator,
        accumulator_imag,
        active_sizes=active_sizes,
        state_rank=state_rank,
        active_rank=active_rank,
    )
    state_magnitude = _mask_to_active(
        torch.sqrt(state_real.square() + state_imag.square() + 1e-6),
        active_sizes,
        state_rank,
    )
    accumulator_magnitude = _mask_to_active(magnitude_accumulator.abs(), active_sizes, state_rank)
    state_mode_energy = _mode_energy_summary(
        state_magnitude,
        active_sizes=active_sizes,
        state_rank=state_rank,
        active_rank=active_rank,
    )
    accumulator_mode_energy = _mode_energy_summary(
        accumulator_magnitude,
        active_sizes=active_sizes,
        state_rank=state_rank,
        active_rank=active_rank,
    )
    return torch.cat(
        (
            state_summary,
            accumulator_summary,
            state_mode_energy,
            accumulator_mode_energy,
        ),
        dim=-1,
    )

__all__ = [name for name in globals() if name.startswith("_") and name not in {"__builtins__", "__all__"}]
