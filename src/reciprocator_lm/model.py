import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .complex import (
    ComplexFeedForward,
    ComplexLayerNorm,
    ComplexLinear,
    apply_complex_rope,
    complex_dropout,
    complex_modulation_factor,
    complex_rope_frequencies,
    complex_readout_features,
)
from .config import ModelConfig


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
    tol: float = 1e-6,
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
        del tol  # Retained for helper compatibility; per-mode normalization now uses a fixed unroll depth.
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


class _NormalizationBlendPredictor(nn.Module):
    """Small gate that interpolates between Frobenius and per-mode normalization."""

    def __init__(
        self,
        *,
        state_rank: int,
        prefer_per_mode: bool,
        hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        self.state_rank = state_rank
        self.summary_proj = nn.Linear(3, hidden_dim)
        self.summary_out = nn.Linear(hidden_dim, 1)
        self.prefer_per_mode = bool(prefer_per_mode)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.summary_proj.weight)
        nn.init.zeros_(self.summary_proj.bias)
        nn.init.zeros_(self.summary_out.weight)
        nn.init.constant_(self.summary_out.bias, 2.0 if self.prefer_per_mode else -2.0)

    def forward(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        active_sizes: tuple[int, ...],
        active_rank: Optional[int] = None,
    ) -> Tensor:
        summary = _summarize_complex_tensor(
            real,
            imag,
            active_sizes=active_sizes,
            state_rank=self.state_rank,
            active_rank=active_rank,
        )
        return torch.sigmoid(self.summary_out(F.silu(self.summary_proj(summary))))


def _blend_normalized_complex_tensor(
    real: Tensor,
    imag: Tensor,
    *,
    state_rank: int,
    active_sizes: tuple[int, ...],
    active_rank: Optional[int] = None,
    eps: float = 1e-6,
    max_iter: int = 3,
    step_sizes: Optional[Tensor] = None,
    blend_predictor: _NormalizationBlendPredictor,
) -> tuple[Tensor, Tensor]:
    frobenius_real, frobenius_imag = _normalize_complex_frobenius(
        real,
        imag,
        state_rank=state_rank,
        active_rank=active_rank,
        eps=eps,
    )
    per_mode_real, per_mode_imag = _normalize_complex_per_mode_unrolled(
        real,
        imag,
        state_rank=state_rank,
        active_rank=active_rank,
        eps=eps,
        max_iters=max_iter,
        step_sizes=step_sizes,
    )
    blend = blend_predictor(
        real,
        imag,
        active_sizes=active_sizes,
        active_rank=active_rank,
    )
    blend_shape = (*blend.shape[:-1], *([1] * state_rank))
    blend = blend.view(*blend_shape)
    # The blended path intentionally relaxes the hard family choice into a
    # differentiable interpolation. Exact family constraints are recovered when
    # the gate saturates near 0 or 1.
    return (
        (1.0 - blend) * frobenius_real + blend * per_mode_real,
        (1.0 - blend) * frobenius_imag + blend * per_mode_imag,
    )


class _InputDependentGainPredictor(nn.Module):
    """Lightweight additive gain predictor shared across mixer paths.

    The predictor keeps the legacy small relational-statistics projection, but
    makes the per-cell normalized signal the primary feature source via an
    elementwise MLP. Outputs live in the gain logit / pre-tanh space so
    `_CubeEngineCell.step()` remains the canonical location that applies the
    nonlinear parameterization.
    """

    def __init__(
        self,
        *,
        state_rank: int,
        state_mode_sizes: tuple[int, ...],
        hidden_dim: int = 12,
        selective_gains: bool = False,
        selection_hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        self.state_rank = state_rank
        self.state_mode_sizes = state_mode_sizes
        self.state_dim = math.prod(state_mode_sizes)
        self.selective_gains = bool(selective_gains)
        self.signal_proj = nn.Linear(3, hidden_dim)
        self.signal_out = nn.Linear(hidden_dim, 4)
        self.context_proj = nn.Linear(state_rank + 3, 4 * self.state_dim, bias=False)
        self.selection_proj = nn.Linear(state_rank + 6, selection_hidden_dim)
        self.selection_out = nn.Linear(selection_hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.signal_proj.weight)
        nn.init.zeros_(self.signal_proj.bias)
        nn.init.normal_(self.signal_out.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.signal_out.bias)
        nn.init.zeros_(self.context_proj.weight)
        nn.init.xavier_uniform_(self.selection_proj.weight)
        nn.init.zeros_(self.selection_proj.bias)
        nn.init.zeros_(self.selection_out.weight)
        nn.init.constant_(self.selection_out.bias, -2.0)

    def zero_dynamic_response(self) -> None:
        with torch.no_grad():
            self.signal_out.weight.zero_()
            self.signal_out.bias.zero_()
            self.context_proj.weight.zero_()
            self.selection_out.weight.zero_()
            self.selection_out.bias.fill_(-20.0)

    def _signal_summary(self, signal_real: Tensor, signal_imag: Tensor) -> Tensor:
        gain_axis = signal_real.ndim - self.state_rank
        reduce_dims = tuple(range(gain_axis, signal_real.ndim))
        signal_magnitude = torch.sqrt(signal_real.square() + signal_imag.square() + 1e-6)
        return torch.cat(
            (
                signal_real.mean(dim=reduce_dims, keepdim=False).unsqueeze(-1),
                signal_imag.mean(dim=reduce_dims, keepdim=False).unsqueeze(-1),
                signal_magnitude.mean(dim=reduce_dims, keepdim=False).unsqueeze(-1),
            ),
            dim=-1,
        )

    def forward(
        self,
        signal_real: Tensor,
        signal_imag: Tensor,
        relational_stats: Optional[Tensor] = None,
        *,
        return_selection_strength: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        signal_magnitude = torch.sqrt(signal_real.square() + signal_imag.square() + 1e-6)
        signal_features = torch.stack((signal_real, signal_imag, signal_magnitude), dim=-1)
        signal_delta = self.signal_out(F.silu(self.signal_proj(signal_features)))
        gain_axis = signal_real.ndim - self.state_rank
        signal_delta = signal_delta.movedim(-1, gain_axis)

        if relational_stats is None:
            return signal_delta

        context_delta = self.context_proj(relational_stats).view(
            *signal_real.shape[:gain_axis],
            4,
            *self.state_mode_sizes,
        )
        gain_delta = signal_delta + context_delta
        if not self.selective_gains:
            if return_selection_strength:
                selection_strength = gain_delta.new_ones(*gain_delta.shape[:gain_axis], 1)
                return gain_delta, selection_strength
            return gain_delta

        signal_summary = self._signal_summary(signal_real, signal_imag)
        if relational_stats is None:
            relational_stats = signal_summary.new_zeros(*signal_summary.shape[:-1], self.state_rank + 3)
        selection_features = torch.cat((signal_summary, relational_stats), dim=-1)
        selection_strength = torch.sigmoid(
            self.selection_out(F.silu(self.selection_proj(selection_features)))
        )
        selection_shape = (*gain_delta.shape[:gain_axis], 1, *([1] * self.state_rank))
        modulated_delta = gain_delta * selection_strength.view(*selection_shape)
        if return_selection_strength:
            return modulated_delta, selection_strength
        return modulated_delta

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        missing_keys[:] = [key for key in missing_keys if not key.startswith(prefix)]


def _predict_gain_bias_tensor(
    gain_predictor: Optional[_InputDependentGainPredictor],
    *,
    signal_real: Tensor,
    signal_imag: Tensor,
    state_real: Tensor,
    state_imag: Tensor,
    active_sizes: tuple[int, ...],
    state_rank: int,
    active_rank: Optional[int] = None,
    return_selection_strength: bool = False,
) -> Optional[Union[Tensor, tuple[Tensor, Tensor]]]:
    if gain_predictor is None:
        return None

    relational_real, relational_imag = _relational_product(
        signal_real,
        signal_imag,
        state_real,
        state_imag,
        active_sizes,
        state_rank,
    )
    relational_stats = _relational_gain_statistics(
        relational_real,
        relational_imag,
        active_sizes,
        state_rank,
        active_rank=active_rank,
    )
    gain_biases = gain_predictor(
        signal_real,
        signal_imag,
        relational_stats,
        return_selection_strength=return_selection_strength,
    )
    if return_selection_strength:
        modulated_biases, selection_strength = gain_biases
        return _mask_to_active(modulated_biases, active_sizes, state_rank), selection_strength
    return _mask_to_active(gain_biases, active_sizes, state_rank)


def _prepare_gain_predictor_state_dict(
    gain_predictor: Optional[_InputDependentGainPredictor],
    state_dict: dict[str, Tensor],
    prefix: str,
) -> None:
    legacy_gain_key = prefix + "gain_proj.weight"
    predictor_prefix = prefix + "gain_predictor."

    if gain_predictor is None:
        state_dict.pop(legacy_gain_key, None)
        for key in list(state_dict.keys()):
            if key.startswith(predictor_prefix):
                state_dict.pop(key)
        return

    context_key = predictor_prefix + "context_proj.weight"
    predictor_keys = [predictor_prefix + name for name in gain_predictor.state_dict()]
    has_new_predictor_payload = any(key in state_dict for key in predictor_keys)

    if legacy_gain_key in state_dict and context_key not in state_dict:
        legacy_weight = state_dict.pop(legacy_gain_key)
        if legacy_weight.shape == gain_predictor.context_proj.weight.shape:
            state_dict[context_key] = legacy_weight
            gain_predictor.zero_dynamic_response()
        elif not has_new_predictor_payload:
            gain_predictor.zero_dynamic_response()
    elif not has_new_predictor_payload:
        gain_predictor.zero_dynamic_response()
    else:
        state_dict.pop(legacy_gain_key, None)


def _filter_gain_predictor_load_keys(
    gain_predictor: Optional[_InputDependentGainPredictor],
    *,
    prefix: str,
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    legacy_gain_key = prefix + "gain_proj.weight"
    predictor_prefix = prefix + "gain_predictor."
    if gain_predictor is not None:
        missing_keys[:] = [key for key in missing_keys if not key.startswith(predictor_prefix)]
    unexpected_keys[:] = [
        key for key in unexpected_keys if key != legacy_gain_key and not (gain_predictor is None and key.startswith(predictor_prefix))
    ]


def _gain_predictor_post_load_hook(module: nn.Module, incompatible_keys: nn.modules.module._IncompatibleKeys) -> None:
    gain_predictor = getattr(module, "gain_predictor", None)
    if gain_predictor is not None:
        incompatible_keys.missing_keys[:] = [
            key for key in incompatible_keys.missing_keys if ".gain_predictor." not in key
        ]
        incompatible_keys.unexpected_keys[:] = [
            key for key in incompatible_keys.unexpected_keys if not key.endswith(".gain_proj.weight")
        ]
        return

    incompatible_keys.unexpected_keys[:] = [
        key
        for key in incompatible_keys.unexpected_keys
        if ".gain_predictor." not in key and not key.endswith(".gain_proj.weight")
    ]


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.resid_dropout(self.out_proj(y))


class _CubeEngineCell(nn.Module):
    def __init__(
        self,
        *,
        state_rank: int,
        initial_state_rank: Optional[int] = None,
        max_mode_sizes: tuple[int, ...],
        normalization: str,
        impression_rate: float,
        growth_threshold: float,
        growth_interval: int,
        prune_floor: float,
        prune_horizon: int,
        dynamic_rank: bool = False,
        prediction_eta: float = 0.1,
        learnable_prediction_eta: bool = False,
        accumulator_modulates_gains: bool = True,
        phase_aware_coupling: bool = True,
        coupling_temperature: float = 1.0,
        learnable_coupling_temperature: bool = False,
        adaptive_growth_controls: bool = False,
    ) -> None:
        super().__init__()
        self.state_rank = state_rank
        self.max_state_rank = state_rank
        self.initial_state_rank = state_rank if initial_state_rank is None else int(initial_state_rank)
        self.state_mode_sizes = max_mode_sizes
        self.max_mode_sizes = max_mode_sizes
        self.normalization = normalization
        self.state_dim = math.prod(max_mode_sizes)
        self.dynamic_rank = bool(dynamic_rank)
        self.learnable_prediction_eta = bool(learnable_prediction_eta)
        self.prediction_eta_raw = (
            nn.Parameter(torch.tensor(_inverse_softplus(prediction_eta), dtype=torch.float32))
            if self.learnable_prediction_eta
            else None
        )
        self._fixed_prediction_eta = float(prediction_eta)
        self.phase_aware_coupling = bool(phase_aware_coupling)
        self.learnable_coupling_temperature = bool(learnable_coupling_temperature)
        self.coupling_temperature_raw = (
            nn.Parameter(torch.tensor(_inverse_softplus(coupling_temperature), dtype=torch.float32))
            if self.learnable_coupling_temperature
            else None
        )
        self._fixed_coupling_temperature = float(coupling_temperature)
        self.adaptive_growth_controls = bool(adaptive_growth_controls)
        self._nominal_growth_threshold = float(growth_threshold)
        self.growth_interval = growth_interval
        self._nominal_prune_floor = float(prune_floor)
        self._nominal_prune_horizon = int(prune_horizon)
        self.accumulator_modulates_gains = bool(accumulator_modulates_gains)
        self.input_gain = nn.Parameter(torch.full(max_mode_sizes, impression_rate))
        self.recurrent_gain = nn.Parameter(torch.full(max_mode_sizes, impression_rate * 0.5))
        self.carry_gain = nn.Parameter(torch.full(max_mode_sizes, impression_rate * 0.25))
        self.decay = nn.Parameter(torch.full(max_mode_sizes, 0.5))
        # Preserve deposited magnitude separately from the normalized directional state.
        self.magnitude_decay = 0.9
        # Keep the new default modulation mild at initialization so the
        # accumulator immediately affects dynamics without destabilizing
        # training. Scales are constrained positive via softplus at use time.
        self.accumulator_input_gain_scale = nn.Parameter(torch.tensor(-3.0))
        self.accumulator_recurrent_gain_scale = nn.Parameter(torch.tensor(-3.0))
        self.accumulator_carry_gain_scale = nn.Parameter(torch.tensor(-3.0))
        self.accumulator_coupling_scale = nn.Parameter(torch.tensor(-3.0))
        # Ablation switch: retain the convex softmax routing path, but default to
        # an expressive complex projection so coupling is not row-stochastic.
        self.use_expressive_mode_couplings = True
        self.use_static_mode_couplings = False
        self.mode_couplings = nn.ParameterList(
            [nn.Parameter(torch.eye(mode_size) * 0.05) for mode_size in max_mode_sizes]
        )
        self.cpl_proj_real = nn.ParameterList(
            [nn.Parameter(torch.eye(mode_size) * 0.1) for mode_size in max_mode_sizes]
        )
        self.cpl_proj_imag = nn.ParameterList(
            [nn.Parameter(torch.zeros(mode_size, mode_size)) for mode_size in max_mode_sizes]
        )
        self.prediction_proj = ComplexLinear(self.state_dim, self.state_dim)
        nn.init.zeros_(self.prediction_proj.weight_real)
        nn.init.zeros_(self.prediction_proj.weight_imag)
        # Keep anticipation in the same geometric language as the update:
        # self-state continuation plus phase-aware self-transport under the
        # learned mode couplings. Zero init preserves legacy behavior.
        self.prediction_state_mix = nn.Parameter(torch.tensor(0.0))
        self.prediction_coupling_mix = nn.Parameter(torch.tensor(0.0))
        self.prediction_accumulator_scale = nn.Parameter(torch.tensor(-3.0))
        max_mode_size = max(max_mode_sizes)
        self.register_buffer("_usage_running", torch.zeros(state_rank, max_mode_size))
        self.register_buffer("_underused_steps", torch.zeros(state_rank, max_mode_size, dtype=torch.long))
        self.register_buffer("_last_novelty", torch.tensor(0.0))
        self.register_buffer("_last_prediction_error", torch.tensor(0.0))
        self.register_buffer("_novelty_ema", torch.tensor(float(growth_threshold)))
        self.register_buffer("_usage_ema", torch.tensor(float(prune_floor)))
        self.register_buffer("_last_growth_mode", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("_growth_event_count", torch.tensor(0, dtype=torch.long), persistent=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for parameter_name in (
            "accumulator_input_gain_scale",
            "accumulator_recurrent_gain_scale",
            "accumulator_carry_gain_scale",
            "accumulator_coupling_scale",
            "prediction_state_mix",
            "prediction_coupling_mix",
            "prediction_accumulator_scale",
        ):
            state_dict.setdefault(prefix + parameter_name, getattr(self, parameter_name).detach().clone())
        _prepare_optional_parameter_state_dict(
            self.prediction_eta_raw,
            state_dict,
            prefix,
            "prediction_eta_raw",
        )
        _prepare_optional_parameter_state_dict(
            self.coupling_temperature_raw,
            state_dict,
            prefix,
            "coupling_temperature_raw",
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        prediction_eta_key = prefix + "prediction_eta_raw"
        coupling_temperature_key = prefix + "coupling_temperature_raw"
        if self.prediction_eta_raw is None:
            unexpected_keys[:] = [key for key in unexpected_keys if key != prediction_eta_key]
        if self.coupling_temperature_raw is None:
            unexpected_keys[:] = [key for key in unexpected_keys if key != coupling_temperature_key]

    @property
    def prediction_eta(self) -> float:
        if self.prediction_eta_raw is None:
            return self._fixed_prediction_eta
        return float(F.softplus(self.prediction_eta_raw.detach()).item())

    def _prediction_eta_tensor(self, reference: Tensor) -> Tensor:
        if self.prediction_eta_raw is None:
            return reference.new_tensor(self._fixed_prediction_eta)
        return F.softplus(self.prediction_eta_raw).to(dtype=reference.dtype, device=reference.device)

    @property
    def coupling_temperature(self) -> float:
        if self.coupling_temperature_raw is None:
            return self._fixed_coupling_temperature
        return float(F.softplus(self.coupling_temperature_raw.detach()).item())

    def _coupling_temperature_tensor(self, reference: Tensor) -> Tensor:
        if self.coupling_temperature_raw is None:
            return reference.new_tensor(self._fixed_coupling_temperature)
        return F.softplus(self.coupling_temperature_raw).to(dtype=reference.dtype, device=reference.device)

    @property
    def growth_threshold(self) -> float:
        if not self.adaptive_growth_controls or self._nominal_growth_threshold <= 0.0:
            return self._nominal_growth_threshold
        scale = float(self._novelty_ema.item()) / max(self._nominal_growth_threshold, 1e-6)
        return self._nominal_growth_threshold * min(max(scale, 0.5), 2.0)

    @growth_threshold.setter
    def growth_threshold(self, value: float) -> None:
        self._nominal_growth_threshold = float(value)
        self._novelty_ema.fill_(float(value))

    @property
    def prune_floor(self) -> float:
        if not self.adaptive_growth_controls or self._nominal_prune_floor <= 0.0:
            return self._nominal_prune_floor
        scale = float(self._usage_ema.item()) / max(self._nominal_prune_floor, 1e-8)
        return self._nominal_prune_floor * min(max(scale, 0.5), 2.0)

    @prune_floor.setter
    def prune_floor(self, value: float) -> None:
        self._nominal_prune_floor = float(value)
        self._usage_ema.fill_(float(value))

    @property
    def prune_horizon(self) -> int:
        if not self.adaptive_growth_controls or self._nominal_prune_floor <= 0.0:
            return self._nominal_prune_horizon
        scale = float(self._usage_ema.item()) / max(self._nominal_prune_floor, 1e-8)
        return max(1, int(round(self._nominal_prune_horizon / min(max(scale, 0.5), 2.0))))

    @prune_horizon.setter
    def prune_horizon(self, value: int) -> None:
        self._nominal_prune_horizon = int(value)

    def _active_parameter(
        self,
        parameter: Tensor,
        active_sizes: tuple[int, ...],
        transform,
    ) -> Tensor:
        active = parameter.new_zeros(parameter.shape)
        active[_active_slice(active_sizes)] = transform(parameter[_active_slice(active_sizes)])
        return active

    def _accumulator_multiplier(self, magnitude_accumulator: Tensor, raw_scale: Tensor) -> Tensor:
        if not self.accumulator_modulates_gains:
            return torch.ones_like(magnitude_accumulator)
        scale = F.softplus(raw_scale).to(
            dtype=magnitude_accumulator.dtype,
            device=magnitude_accumulator.device,
        )
        return 1.0 + scale * magnitude_accumulator

    def _modulate_gain(self, gain: Tensor, magnitude_accumulator: Tensor, raw_scale: Tensor) -> Tensor:
        if not self.accumulator_modulates_gains:
            return gain
        return gain * self._accumulator_multiplier(magnitude_accumulator, raw_scale)

    def _scale_coupling_drive(
        self,
        local_real: Tensor,
        local_imag: Tensor,
        magnitude_accumulator: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self.accumulator_modulates_gains:
            return local_real, local_imag
        coupling_strength = self._accumulator_multiplier(
            magnitude_accumulator,
            self.accumulator_coupling_scale,
        )
        return local_real * coupling_strength, local_imag * coupling_strength

    def _scan_magnitude_accumulator(self, magnitude: Tensor) -> Tensor:
        accumulator_decay = magnitude.new_full((magnitude.shape[-1],), self.magnitude_decay)
        accumulator_inputs = (1.0 - self.magnitude_decay) * magnitude
        return _parallel_scan_linear(accumulator_decay, accumulator_inputs)

    def _previous_magnitude_accumulator(self, magnitude: Tensor) -> Tensor:
        accumulator = self._scan_magnitude_accumulator(magnitude)
        previous = torch.zeros_like(accumulator)
        previous[:, 1:] = accumulator[:, :-1]
        return previous

    def _grow_mode(
        self,
        *,
        error_real: Tensor,
        error_imag: Tensor,
        proposal_real: Tensor,
        proposal_imag: Tensor,
        active_sizes: tuple[int, ...],
        active_rank: int,
        mode_scores: list[float],
    ) -> tuple[Tensor, Tensor, tuple[int, ...]]:
        sorted_modes = sorted(range(active_rank), key=lambda idx: mode_scores[idx], reverse=True)
        growth_mode = next(
            (
                mode_idx
                for mode_idx in sorted_modes
                if int(active_sizes[mode_idx]) < int(self.max_mode_sizes[mode_idx])
            ),
            None,
        )
        if growth_mode is None:
            return proposal_real, proposal_imag, active_sizes

        grown_sizes = list(active_sizes)
        new_index = int(grown_sizes[growth_mode])
        grown_sizes[growth_mode] += 1
        mode_axis = proposal_real.ndim - self.state_rank + growth_mode
        seed_real = error_real.mean(dim=mode_axis)
        seed_imag = error_imag.mean(dim=mode_axis)

        target = [slice(None)] * proposal_real.ndim
        seed_view = [slice(None)] * seed_real.ndim
        seed_axis = 1
        for mode_idx in range(self.state_rank):
            axis = proposal_real.ndim - self.state_rank + mode_idx
            if mode_idx == growth_mode:
                target[axis] = new_index
                continue
            target[axis] = slice(0, int(active_sizes[mode_idx]))
            seed_view[seed_axis] = slice(0, int(active_sizes[mode_idx]))
            seed_axis += 1

        proposal_real[tuple(target)] = seed_real[tuple(seed_view)]
        proposal_imag[tuple(target)] = seed_imag[tuple(seed_view)]
        self._usage_running[growth_mode, new_index] = 0.0
        self._underused_steps[growth_mode, new_index] = 0
        self._last_growth_mode.fill_(int(growth_mode))
        self._growth_event_count.add_(1)
        return proposal_real, proposal_imag, tuple(grown_sizes)

    def _grow_rank(
        self,
        *,
        error_real: Tensor,
        error_imag: Tensor,
        proposal_real: Tensor,
        proposal_imag: Tensor,
        active_sizes: tuple[int, ...],
        active_rank: int,
    ) -> tuple[Tensor, Tensor, tuple[int, ...], int]:
        if not self.dynamic_rank or active_rank >= self.max_state_rank:
            return proposal_real, proposal_imag, active_sizes, active_rank

        growth_mode = active_rank
        grown_sizes = list(active_sizes)
        target_mode_size = 2 if self.max_mode_sizes[growth_mode] >= 2 else 1
        grown_sizes[growth_mode] = min(
            max(target_mode_size, grown_sizes[growth_mode]),
            self.max_mode_sizes[growth_mode],
        )
        next_active_rank = active_rank + 1

        batch_dims = proposal_real.ndim - self.state_rank
        target_slice = (slice(None),) * batch_dims + _active_slice(tuple(grown_sizes))
        active_slice = (slice(None),) * batch_dims + _active_slice(active_sizes)
        active_error_real = error_real[active_slice]
        active_error_imag = error_imag[active_slice]
        reduce_dims = tuple(range(batch_dims, active_error_real.ndim))
        seed_real = active_error_real.mean(dim=reduce_dims, keepdim=True)
        seed_imag = active_error_imag.mean(dim=reduce_dims, keepdim=True)

        proposal_real[target_slice] = proposal_real[target_slice] + seed_real.expand_as(proposal_real[target_slice])
        proposal_imag[target_slice] = proposal_imag[target_slice] + seed_imag.expand_as(proposal_imag[target_slice])
        self._usage_running[growth_mode, : int(self.max_mode_sizes[growth_mode])] = 0.0
        self._underused_steps[growth_mode, : int(self.max_mode_sizes[growth_mode])] = 0
        self._last_growth_mode.fill_(int(growth_mode))
        self._growth_event_count.add_(1)
        return proposal_real, proposal_imag, tuple(grown_sizes), next_active_rank

    def _update_usage_and_prune(
        self,
        *,
        next_real: Tensor,
        next_imag: Tensor,
        active_sizes: tuple[int, ...],
        active_rank: int,
        allow_resize: bool,
    ) -> tuple[int, ...]:
        magnitude = torch.sqrt(next_real.square() + next_imag.square() + 1e-6)
        pruned_sizes = list(active_sizes)
        mean_usage_values: list[Tensor] = []
        for mode_idx in range(active_rank):
            max_size = self.max_mode_sizes[mode_idx]
            axis = magnitude.ndim - self.state_rank + mode_idx
            reduce_dims = tuple(dim for dim in range(magnitude.ndim) if dim != axis)
            usage = magnitude.mean(dim=reduce_dims).detach()
            running = self._usage_running[mode_idx, :max_size]
            running.mul_(0.95).add_(0.05 * usage)
            active_mode_size = int(active_sizes[mode_idx])
            if active_mode_size > 0:
                mean_usage_values.append(running[:active_mode_size].mean())
            self._underused_steps[mode_idx, :max_size] = torch.where(
                running < self.prune_floor,
                self._underused_steps[mode_idx, :max_size] + 1,
                torch.zeros_like(self._underused_steps[mode_idx, :max_size]),
            )
            if not allow_resize:
                continue
            while (
                pruned_sizes[mode_idx] > 1
                and int(self._underused_steps[mode_idx, pruned_sizes[mode_idx] - 1].item()) >= self.prune_horizon
            ):
                pruned_sizes[mode_idx] -= 1
        if mean_usage_values:
            mean_usage = torch.stack(mean_usage_values).mean()
            self._usage_ema.mul_(0.95).add_(0.05 * mean_usage)
        return tuple(pruned_sizes)

    def diagnostics(self, active_sizes: tuple[int, ...], active_rank: Optional[int] = None) -> dict[str, object]:
        resolved_active_rank = self.initial_state_rank if active_rank is None else int(active_rank)
        return {
            "active_sizes": tuple(int(size) for size in active_sizes),
            "active_rank": resolved_active_rank,
            "initial_state_rank": self.initial_state_rank,
            "max_state_rank": self.max_state_rank,
            "dynamic_rank": self.dynamic_rank,
            "accumulator_modulates_gains": self.accumulator_modulates_gains,
            "prediction_eta": self.prediction_eta,
            "coupling_temperature": self.coupling_temperature,
            "growth_threshold": self.growth_threshold,
            "prune_floor": self.prune_floor,
            "prune_horizon": self.prune_horizon,
            "last_novelty": float(self._last_novelty.item()),
            "last_prediction_error": float(self._last_prediction_error.item()),
            "growth_event_count": int(self._growth_event_count.item()),
            "last_growth_mode": None
            if int(self._last_growth_mode.item()) < 0
            else int(self._last_growth_mode.item()),
        }

    def _predict_signal(
        self,
        state_real: Tensor,
        state_imag: Tensor,
        magnitude_accumulator: Tensor,
        active_sizes: tuple[int, ...],
        active_rank: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        resolved_active_rank = self.initial_state_rank if active_rank is None else int(active_rank)
        state_real = _mask_to_active(state_real, active_sizes, self.state_rank)
        state_imag = _mask_to_active(state_imag, active_sizes, self.state_rank)
        magnitude_accumulator = _mask_to_active(magnitude_accumulator, active_sizes, self.state_rank)
        leading_shape = state_real.shape[: state_real.ndim - self.state_rank]
        flat_real = state_real.reshape(-1, self.state_dim)
        flat_imag = state_imag.reshape(-1, self.state_dim)
        predicted = self.prediction_proj(torch.complex(flat_real, flat_imag))
        predicted_real = predicted.real.view(*leading_shape, *self.state_mode_sizes)
        predicted_imag = predicted.imag.view(*leading_shape, *self.state_mode_sizes)
        if self.use_expressive_mode_couplings:
            coupled_state_real, coupled_state_imag = self._apply_expressive_mode_couplings(
                state_real,
                state_imag,
                active_sizes,
                resolved_active_rank,
            )
        else:
            mode_couplings = self._phase_aware_mode_couplings(
                state_real,
                state_imag,
                active_sizes,
                resolved_active_rank,
            )
            coupled_state_real, coupled_state_imag = _apply_mode_couplings_pair(
                state_real,
                state_imag,
                mode_couplings,
                self.state_rank,
            )
        state_mix = torch.tanh(self.prediction_state_mix).to(dtype=state_real.dtype, device=state_real.device)
        coupling_mix = torch.tanh(self.prediction_coupling_mix).to(
            dtype=state_real.dtype,
            device=state_real.device,
        )
        prediction_multiplier = self._accumulator_multiplier(
            magnitude_accumulator,
            self.prediction_accumulator_scale,
        )
        predicted_real = predicted_real + prediction_multiplier * (
            state_mix * state_real + coupling_mix * coupled_state_real
        )
        predicted_imag = predicted_imag + prediction_multiplier * (
            state_mix * state_imag + coupling_mix * coupled_state_imag
        )
        predicted_real = _mask_to_active(predicted_real, active_sizes, self.state_rank)
        predicted_imag = _mask_to_active(predicted_imag, active_sizes, self.state_rank)
        return predicted_real, predicted_imag

    def _empty_mode_coupling(self, local_real: Tensor, batch_shape: tuple[int, ...], mode_size: int) -> Tensor:
        if self.phase_aware_coupling:
            return torch.zeros(
                *batch_shape,
                mode_size,
                mode_size,
                dtype=_complex_dtype_for(local_real.dtype),
                device=local_real.device,
            )
        return local_real.new_zeros(*batch_shape, mode_size, mode_size)

    def _identity_mode_coupling(self, local_real: Tensor, batch_shape: tuple[int, ...], mode_size: int) -> Tensor:
        if self.phase_aware_coupling:
            eye = torch.eye(
                mode_size,
                dtype=_complex_dtype_for(local_real.dtype),
                device=local_real.device,
            )
        else:
            eye = torch.eye(mode_size, dtype=local_real.dtype, device=local_real.device)
        return eye.expand(*batch_shape, mode_size, mode_size)

    def _coupling_from_scores(
        self,
        logits_real: Tensor,
        logits_imag: Tensor,
        *,
        scale: float,
    ) -> Tensor:
        if not self.phase_aware_coupling:
            logits = (logits_real + logits_imag) / max(scale, 1e-8)
            return F.softmax(logits, dim=-1)
        return _phase_preserving_coupling(
            logits_real,
            logits_imag,
            scale=scale,
            temperature=self._coupling_temperature_tensor(logits_real),
        )

    def _mode_score_logits(
        self,
        source_real: Tensor,
        source_imag: Tensor,
        *,
        mode_idx: int,
    ) -> tuple[Tensor, Tensor, float]:
        key_real = source_real.transpose(-1, -2)
        key_imag = source_imag.transpose(-1, -2)
        active_mode = source_real.shape[-2]
        weight_real = self.cpl_proj_real[mode_idx][:active_mode, :active_mode]
        weight_imag = self.cpl_proj_imag[mode_idx][:active_mode, :active_mode]

        proj_real = torch.matmul(key_real, weight_real.transpose(0, 1)) - torch.matmul(
            key_imag,
            weight_imag.transpose(0, 1),
        )
        proj_imag = torch.matmul(key_real, weight_imag.transpose(0, 1)) + torch.matmul(
            key_imag,
            weight_real.transpose(0, 1),
        )
        proj_real = proj_real.transpose(-1, -2)
        proj_imag = proj_imag.transpose(-1, -2)

        logits_real = torch.matmul(proj_real, source_real.transpose(-1, -2)) - torch.matmul(
            proj_imag,
            source_imag.transpose(-1, -2),
        )
        logits_imag = torch.matmul(proj_real, source_imag.transpose(-1, -2)) + torch.matmul(
            proj_imag,
            source_real.transpose(-1, -2),
        )
        return logits_real, logits_imag, math.sqrt(source_real.shape[-1])

    def _phase_aware_mode_couplings(
        self,
        local_real: Tensor,
        local_imag: Tensor,
        active_sizes: tuple[int, ...],
        active_rank: Optional[int] = None,
    ) -> list[Tensor]:
        couplings = []
        batch_dims = local_real.ndim - self.state_rank
        batch_shape = local_real.shape[:batch_dims]
        resolved_active_rank = self.initial_state_rank if active_rank is None else int(active_rank)
        for mode_idx, active_mode_size in enumerate(active_sizes):
            axis = batch_dims + mode_idx
            moved_real = torch.movedim(local_real, axis, batch_dims)
            moved_imag = torch.movedim(local_imag, axis, batch_dims)
            mode_size = moved_real.shape[batch_dims]
            if mode_idx >= resolved_active_rank:
                couplings.append(self._identity_mode_coupling(local_real, batch_shape, mode_size))
                continue
            flat_real = moved_real.reshape(*batch_shape, mode_size, -1)
            flat_imag = moved_imag.reshape(*batch_shape, mode_size, -1)
            active_mode = int(active_mode_size)
            coupling = self._empty_mode_coupling(local_real, batch_shape, mode_size)
            if active_mode == 0:
                couplings.append(coupling)
                continue

            if self.use_static_mode_couplings:
                static_logits = self.mode_couplings[mode_idx][:active_mode, :active_mode]
                static_coupling = F.softmax(static_logits, dim=-1)
                coupling[..., :active_mode, :active_mode] = static_coupling.expand(
                    *batch_shape,
                    active_mode,
                    active_mode,
                )
                couplings.append(coupling)
                continue

            source_real = flat_real[..., :active_mode, :]
            source_imag = flat_imag[..., :active_mode, :]
            logits_real, logits_imag, scale = self._mode_score_logits(
                source_real,
                source_imag,
                mode_idx=mode_idx,
            )
            coupling[..., :active_mode, :active_mode] = self._coupling_from_scores(
                logits_real,
                logits_imag,
                scale=scale,
            )
            couplings.append(coupling)
        return couplings

    def _expressive_mode_couplings(
        self,
        local_real: Tensor,
        local_imag: Tensor,
        active_sizes: tuple[int, ...],
        active_rank: Optional[int] = None,
    ) -> tuple[list[Tensor], tuple[Tensor, Tensor]]:
        resolved_active_rank = self.initial_state_rank if active_rank is None else int(active_rank)
        if not self.phase_aware_coupling:
            couplings = self._phase_aware_mode_couplings(
                local_real,
                local_imag,
                active_sizes,
                resolved_active_rank,
            )
            return couplings, _apply_mode_couplings_pair(local_real, local_imag, couplings, self.state_rank)

        batch_dims = local_real.ndim - self.state_rank
        batch_shape = local_real.shape[:batch_dims]
        couplings: list[Tensor] = []
        coupled_real = local_real
        coupled_imag = local_imag
        for mode_idx, active_mode_size in enumerate(active_sizes):
            axis = batch_dims + mode_idx
            moved_real = torch.movedim(coupled_real, axis, batch_dims)
            moved_imag = torch.movedim(coupled_imag, axis, batch_dims)
            mode_size = moved_real.shape[batch_dims]
            if mode_idx >= resolved_active_rank:
                couplings.append(self._identity_mode_coupling(local_real, batch_shape, mode_size))
                continue
            flat_real = moved_real.reshape(*batch_shape, mode_size, -1)
            flat_imag = moved_imag.reshape(*batch_shape, mode_size, -1)

            active_mode = int(active_mode_size)
            coupling = self._empty_mode_coupling(local_real, batch_shape, mode_size)
            if active_mode == 0:
                couplings.append(coupling)
                continue

            source_real = flat_real[..., :active_mode, :]
            source_imag = flat_imag[..., :active_mode, :]
            logits_real, logits_imag, scale = self._mode_score_logits(
                source_real,
                source_imag,
                mode_idx=mode_idx,
            )
            coupling[..., :active_mode, :active_mode] = self._coupling_from_scores(
                logits_real,
                logits_imag,
                scale=scale,
            )
            couplings.append(coupling)

            coupled = _apply_single_mode_coupling(
                torch.complex(coupled_real, coupled_imag),
                coupling,
                state_rank=self.state_rank,
                mode_idx=mode_idx,
            )
            coupled_real = coupled.real
            coupled_imag = coupled.imag
        return couplings, (coupled_real, coupled_imag)

    def _apply_expressive_mode_couplings(
        self,
        local_real: Tensor,
        local_imag: Tensor,
        active_sizes: tuple[int, ...],
        active_rank: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        couplings, coupled = self._expressive_mode_couplings(
            local_real,
            local_imag,
            active_sizes,
            active_rank,
        )
        if not couplings:
            return local_real, local_imag
        return coupled

    def step(
        self,
        *,
        signal_real: Tensor,
        signal_imag: Tensor,
        state_real: Tensor,
        state_imag: Tensor,
        magnitude_accumulator: Tensor,
        carry_real: Tensor,
        carry_imag: Tensor,
        active_sizes: tuple[int, ...],
        step_index: int,
        active_rank: Optional[int] = None,
        decay_bias: Optional[Tensor] = None,
        input_gain_bias: Optional[Tensor] = None,
        recurrent_gain_bias: Optional[Tensor] = None,
        carry_gain_bias: Optional[Tensor] = None,
        normalization_step_sizes: Optional[Tensor] = None,
        normalization_blend_predictor: Optional[_NormalizationBlendPredictor] = None,
        allow_growth: bool = True,
        return_active_sizes: bool = False,
    ) -> Union[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, tuple[int, ...], int],
    ]:
        # This is the canonical single-step update rule.
        resolved_active_rank = self.initial_state_rank if active_rank is None else int(active_rank)
        signal_real = _mask_to_active(signal_real, active_sizes, self.state_rank)
        signal_imag = _mask_to_active(signal_imag, active_sizes, self.state_rank)
        state_real = _mask_to_active(state_real, active_sizes, self.state_rank)
        state_imag = _mask_to_active(state_imag, active_sizes, self.state_rank)
        magnitude_accumulator = _mask_to_active(magnitude_accumulator, active_sizes, self.state_rank)
        carry_real = _mask_to_active(carry_real, active_sizes, self.state_rank)
        carry_imag = _mask_to_active(carry_imag, active_sizes, self.state_rank)

        local_real, local_imag = _relational_product(
            signal_real,
            signal_imag,
            state_real,
            state_imag,
            active_sizes,
            self.state_rank,
        )
        # The magnitude accumulator is now a causal gating signal, not just a
        # downstream feature: stronger retained traces can push harder through
        # the mode-coupling path on the next update.
        local_real, local_imag = self._scale_coupling_drive(
            local_real,
            local_imag,
            magnitude_accumulator,
        )
        if self.use_expressive_mode_couplings:
            coupled_real, coupled_imag = self._apply_expressive_mode_couplings(
                local_real,
                local_imag,
                active_sizes,
                resolved_active_rank,
            )
        else:
            # Legacy convex path retained for ablations.
            mode_couplings = self._phase_aware_mode_couplings(
                local_real,
                local_imag,
                active_sizes,
                resolved_active_rank,
            )
            coupled_real, coupled_imag = _apply_mode_couplings_pair(
                local_real, local_imag, mode_couplings, self.state_rank
            )
        predicted_signal_real, predicted_signal_imag = self._predict_signal(
            state_real,
            state_imag,
            magnitude_accumulator,
            active_sizes,
            resolved_active_rank,
        )
        prediction_error_real = signal_real - predicted_signal_real
        prediction_error_imag = signal_imag - predicted_signal_imag

        decay_logits = self._active_parameter(self.decay, active_sizes, lambda value: value).unsqueeze(0)
        input_gain_logits = self._active_parameter(self.input_gain, active_sizes, lambda value: value).unsqueeze(0)
        recurrent_gain_logits = self._active_parameter(self.recurrent_gain, active_sizes, lambda value: value).unsqueeze(0)
        carry_gain_logits = self._active_parameter(self.carry_gain, active_sizes, lambda value: value).unsqueeze(0)
        if decay_bias is not None:
            decay_logits = decay_logits + _mask_to_active(decay_bias, active_sizes, self.state_rank)
        if input_gain_bias is not None:
            input_gain_logits = input_gain_logits + _mask_to_active(input_gain_bias, active_sizes, self.state_rank)
        if recurrent_gain_bias is not None:
            recurrent_gain_logits = recurrent_gain_logits + _mask_to_active(
                recurrent_gain_bias,
                active_sizes,
                self.state_rank,
            )
        if carry_gain_bias is not None:
            carry_gain_logits = carry_gain_logits + _mask_to_active(carry_gain_bias, active_sizes, self.state_rank)
        decay = torch.sigmoid(decay_logits)
        input_gain = torch.sigmoid(input_gain_logits)
        recurrent_gain = torch.tanh(recurrent_gain_logits)
        carry_gain = torch.tanh(carry_gain_logits)
        # Keep input-dependent and static gains fully functional, then let the
        # retained magnitude scale their effective strength elementwise.
        input_gain = self._modulate_gain(
            input_gain,
            magnitude_accumulator,
            self.accumulator_input_gain_scale,
        )
        recurrent_gain = self._modulate_gain(
            recurrent_gain,
            magnitude_accumulator,
            self.accumulator_recurrent_gain_scale,
        )
        carry_gain = self._modulate_gain(
            carry_gain,
            magnitude_accumulator,
            self.accumulator_carry_gain_scale,
        )
        prediction_eta = self._prediction_eta_tensor(signal_real)

        proposal_real = (
            decay * state_real
            + input_gain * signal_real
            + recurrent_gain * coupled_real
            + carry_gain * carry_real
            + prediction_eta * prediction_error_real
        )
        proposal_imag = (
            decay * state_imag
            + input_gain * signal_imag
            + recurrent_gain * coupled_imag
            + carry_gain * carry_imag
            + prediction_eta * prediction_error_imag
        )
        proposal_real = _mask_to_active(proposal_real, active_sizes, self.state_rank)
        proposal_imag = _mask_to_active(proposal_imag, active_sizes, self.state_rank)
        self._last_growth_mode.fill_(-1)
        next_active_sizes = active_sizes
        next_active_rank = resolved_active_rank
        detached_prediction_error_real = signal_real - predicted_signal_real.detach()
        detached_prediction_error_imag = signal_imag - predicted_signal_imag.detach()
        prediction_error_energy = (
            detached_prediction_error_real.square() + detached_prediction_error_imag.square()
        )
        signal_energy = signal_real.square() + signal_imag.square()
        mode_scores: list[float] = []
        for mode_idx in range(resolved_active_rank):
            axis = prediction_error_energy.ndim - self.state_rank + mode_idx
            reduce_dims = tuple(dim for dim in range(prediction_error_energy.ndim) if dim != axis)
            mode_scores.append(float(prediction_error_energy.sum(dim=reduce_dims).max().item()))
        novelty = sum(mode_scores) / float(signal_energy.sum().item() + 1e-8) if mode_scores else 0.0
        self._last_prediction_error.fill_(float(novelty))
        self._last_novelty.fill_(float(novelty))
        self._novelty_ema.mul_(0.95).add_(0.05 * novelty)
        if allow_growth:
            if novelty > self.growth_threshold and step_index % self.growth_interval == 0:
                if self.dynamic_rank and resolved_active_rank < self.max_state_rank:
                    proposal_real, proposal_imag, next_active_sizes, next_active_rank = self._grow_rank(
                        error_real=detached_prediction_error_real,
                        error_imag=detached_prediction_error_imag,
                        proposal_real=proposal_real,
                        proposal_imag=proposal_imag,
                        active_sizes=active_sizes,
                        active_rank=resolved_active_rank,
                    )
                else:
                    proposal_real, proposal_imag, next_active_sizes = self._grow_mode(
                        error_real=detached_prediction_error_real,
                        error_imag=detached_prediction_error_imag,
                        proposal_real=proposal_real,
                        proposal_imag=proposal_imag,
                        active_sizes=active_sizes,
                        active_rank=resolved_active_rank,
                        mode_scores=mode_scores,
                    )

        proposal_real = _mask_to_active(proposal_real, next_active_sizes, self.state_rank)
        proposal_imag = _mask_to_active(proposal_imag, next_active_sizes, self.state_rank)
        proposal_magnitude = _mask_to_active(
            torch.sqrt(proposal_real.square() + proposal_imag.square() + 1e-6),
            next_active_sizes,
            self.state_rank,
        )
        # This accumulator now serves two roles:
        # 1. it remains a readout feature for the hidden-space return map
        # 2. on the next step it will modulate gains/coupling strength
        next_magnitude_accumulator = _mask_to_active(
            self.magnitude_decay * magnitude_accumulator + (1.0 - self.magnitude_decay) * proposal_magnitude,
            next_active_sizes,
            self.state_rank,
        )
        if normalization_blend_predictor is None:
            next_real, next_imag = _normalize_complex_tensor(
                proposal_real,
                proposal_imag,
                self.normalization,
                state_rank=self.state_rank,
                active_rank=next_active_rank,
                step_sizes=normalization_step_sizes,
            )
        else:
            next_real, next_imag = _blend_normalized_complex_tensor(
                proposal_real,
                proposal_imag,
                state_rank=self.state_rank,
                active_sizes=next_active_sizes,
                active_rank=next_active_rank,
                step_sizes=normalization_step_sizes,
                blend_predictor=normalization_blend_predictor,
            )
        next_real = _mask_to_active(next_real, next_active_sizes, self.state_rank)
        next_imag = _mask_to_active(next_imag, next_active_sizes, self.state_rank)
        if allow_growth:
            next_active_sizes = self._update_usage_and_prune(
                next_real=next_real,
                next_imag=next_imag,
                active_sizes=next_active_sizes,
                active_rank=next_active_rank,
                allow_resize=allow_growth,
            )
            next_real = _mask_to_active(next_real, next_active_sizes, self.state_rank)
            next_imag = _mask_to_active(next_imag, next_active_sizes, self.state_rank)
            next_magnitude_accumulator = _mask_to_active(
                next_magnitude_accumulator,
                next_active_sizes,
                self.state_rank,
            )
        else:
            next_active_sizes = active_sizes
            next_active_rank = resolved_active_rank

        new_carry_real = next_real
        new_carry_imag = next_imag
        if return_active_sizes:
            return (
                next_real,
                next_imag,
                next_magnitude_accumulator,
                new_carry_real,
                new_carry_imag,
                next_active_sizes,
                next_active_rank,
            )
        return next_real, next_imag, next_magnitude_accumulator, new_carry_real, new_carry_imag


class ReciprocatorMixer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        state_rank: int,
        state_mode_sizes: tuple[int, ...],
        init_mode_sizes: tuple[int, ...],
        num_cube_engines: int,
        normalization: str,
        impression_rate: float,
        phase_scale: float,
        magnitude_floor: float,
        dropout: float,
        growth_threshold: float,
        growth_interval: int,
        prune_floor: float,
        prune_horizon: int,
        initial_state_rank: Optional[int] = None,
        dynamic_rank: bool = False,
        input_dependent_gains: bool = False,
        selective_gains: bool = False,
        prediction_eta: float = 0.1,
        learnable_prediction_eta: bool = False,
        accumulator_modulates_gains: bool = True,
        phase_aware_coupling: bool = True,
        coupling_temperature: float = 1.0,
        learnable_coupling_temperature: bool = False,
        learned_per_mode_scaling: bool = False,
        learned_normalization_blend: bool = False,
        adaptive_growth_controls: bool = False,
    ) -> None:
        super().__init__()
        self.phase_scale = phase_scale
        del magnitude_floor  # Deprecated no-op retained for config compatibility.
        self.state_rank = state_rank
        self.initial_state_rank = state_rank if initial_state_rank is None else int(initial_state_rank)
        self.state_mode_sizes = state_mode_sizes
        self.init_mode_sizes = init_mode_sizes
        self.num_cube_engines = num_cube_engines
        self.state_dim = state_dim
        self.normalization = normalization
        self.accumulator_modulates_gains = accumulator_modulates_gains
        self.dynamic_rank = bool(dynamic_rank)
        self.input_dependent_gains = bool(input_dependent_gains)
        self.selective_gains = bool(selective_gains)
        self.learned_normalization_blend = bool(learned_normalization_blend)
        # Optional relaxation: when enabled, share one learned exponent vector
        # across the mixer's per-mode normalization calls.
        self.per_mode_step_sizes = (
            nn.Parameter(torch.ones(state_rank))
            if (normalization == "per_mode" or self.learned_normalization_blend) and learned_per_mode_scaling
            else None
        )

        self.mag_proj = nn.Linear(hidden_dim, state_dim)
        self.phase_proj = nn.Linear(hidden_dim, state_dim)
        self.cube_engines = nn.ModuleList(
            [
                _CubeEngineCell(
                    state_rank=state_rank,
                    initial_state_rank=initial_state_rank,
                    max_mode_sizes=state_mode_sizes,
                    normalization=normalization,
                    impression_rate=impression_rate,
                    prediction_eta=prediction_eta,
                    learnable_prediction_eta=learnable_prediction_eta,
                    growth_threshold=growth_threshold,
                    growth_interval=growth_interval,
                    prune_floor=prune_floor,
                    prune_horizon=prune_horizon,
                    dynamic_rank=dynamic_rank,
                    accumulator_modulates_gains=accumulator_modulates_gains,
                    phase_aware_coupling=phase_aware_coupling,
                    coupling_temperature=coupling_temperature,
                    learnable_coupling_temperature=learnable_coupling_temperature,
                    adaptive_growth_controls=adaptive_growth_controls,
                )
                for _ in range(num_cube_engines)
            ]
        )
        self.engine_state_to_hidden = nn.ModuleList(
            [nn.Linear(state_dim * 4, hidden_dim) for _ in range(num_cube_engines)]
        )
        self.engine_fusion = nn.Linear(hidden_dim * num_cube_engines, hidden_dim)
        self.gain_predictor = (
            _InputDependentGainPredictor(
                state_rank=state_rank,
                state_mode_sizes=state_mode_sizes,
                selective_gains=selective_gains,
            )
            if input_dependent_gains
            else None
        )
        self.normalization_blend_predictor = (
            _NormalizationBlendPredictor(
                state_rank=state_rank,
                prefer_per_mode=normalization == "per_mode",
            )
            if self.learned_normalization_blend
            else None
        )

        self.dropout = nn.Dropout(dropout)
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.register_load_state_dict_post_hook(_gain_predictor_post_load_hook)

    @property
    def gain_proj(self) -> Optional[nn.Linear]:
        return None if self.gain_predictor is None else self.gain_predictor.context_proj

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        _prepare_per_mode_step_sizes_state_dict(self.per_mode_step_sizes, state_dict, prefix)
        _prepare_gain_predictor_state_dict(self.gain_predictor, state_dict, prefix)
        _prepare_optional_module_state_dict(
            self.normalization_blend_predictor,
            state_dict,
            prefix + "normalization_blend_predictor.",
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        _filter_gain_predictor_load_keys(
            self.gain_predictor,
            prefix=prefix,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
        normalization_prefix = prefix + "normalization_blend_predictor."
        if self.normalization_blend_predictor is not None:
            missing_keys[:] = [key for key in missing_keys if not key.startswith(normalization_prefix)]
        else:
            unexpected_keys[:] = [key for key in unexpected_keys if not key.startswith(normalization_prefix)]

    def _normalize_tensor(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        active_sizes: tuple[int, ...],
        active_rank: Optional[int],
    ) -> tuple[Tensor, Tensor]:
        if self.normalization_blend_predictor is not None:
            return _blend_normalized_complex_tensor(
                real,
                imag,
                state_rank=self.state_rank,
                active_sizes=active_sizes,
                active_rank=active_rank,
                step_sizes=self.per_mode_step_sizes,
                blend_predictor=self.normalization_blend_predictor,
            )
        return _normalize_complex_tensor(
            real,
            imag,
            self.normalization,
            state_rank=self.state_rank,
            active_rank=active_rank,
            step_sizes=self.per_mode_step_sizes,
        )

    def forward(self, x: Tensor) -> Tensor:
        magnitude = F.softplus(self.mag_proj(x))
        phase = torch.tanh(self.phase_proj(x)) * self.phase_scale

        signal_shape = (-1, x.size(1), *self.state_mode_sizes)
        signal_real = (magnitude * torch.cos(phase)).view(*signal_shape)
        signal_imag = (magnitude * torch.sin(phase)).view(*signal_shape)
        signal_real = _mask_to_active(signal_real, self.init_mode_sizes, self.state_rank)
        signal_imag = _mask_to_active(signal_imag, self.init_mode_sizes, self.state_rank)
        signal_real, signal_imag = self._normalize_tensor(
            signal_real,
            signal_imag,
            active_sizes=self.init_mode_sizes,
            active_rank=self.initial_state_rank,
        )

        batch, steps, _ = x.shape
        state_reals = [x.new_zeros(batch, *self.state_mode_sizes) for _ in range(self.num_cube_engines)]
        state_imags = [x.new_zeros(batch, *self.state_mode_sizes) for _ in range(self.num_cube_engines)]
        state_accumulators = [x.new_zeros(batch, *self.state_mode_sizes) for _ in range(self.num_cube_engines)]
        outputs = []
        active_sizes = self.init_mode_sizes
        active_rank = self.initial_state_rank

        for step in range(steps):
            carry_real = torch.zeros(batch, *self.state_mode_sizes, dtype=x.dtype, device=x.device)
            carry_imag = torch.zeros_like(carry_real)
            engine_deltas = []
            allow_resize = self.dynamic_rank and self.training

            # Stack a small bank of cube engines so later engines can refine the
            # current token with the structured state emitted by earlier ones.
            for engine_index, engine in enumerate(self.cube_engines):
                sr = _mask_to_active(signal_real[:, step, :], active_sizes, self.state_rank)
                si = _mask_to_active(signal_imag[:, step, :], active_sizes, self.state_rank)
                sr, si = self._normalize_tensor(
                    sr,
                    si,
                    active_sizes=active_sizes,
                    active_rank=active_rank,
                )
                decay_bias = None
                input_gain_bias = None
                recurrent_gain_bias = None
                carry_gain_bias = None
                if self.gain_predictor is not None:
                    gain_biases = _predict_gain_bias_tensor(
                        self.gain_predictor,
                        signal_real=sr,
                        signal_imag=si,
                        state_real=state_reals[engine_index],
                        state_imag=state_imags[engine_index],
                        active_sizes=active_sizes,
                        state_rank=self.state_rank,
                        active_rank=active_rank,
                    )
                    assert gain_biases is not None
                    gain_axis = sr.ndim - self.state_rank
                    decay_bias, input_gain_bias, recurrent_gain_bias, carry_gain_bias = gain_biases.unbind(
                        dim=gain_axis
                    )
                next_real, next_imag, next_accumulator, carry_real, carry_imag, active_sizes, active_rank = engine.step(
                    signal_real=sr,
                    signal_imag=si,
                    state_real=state_reals[engine_index],
                    state_imag=state_imags[engine_index],
                    magnitude_accumulator=state_accumulators[engine_index],
                    carry_real=carry_real,
                    carry_imag=carry_imag,
                    active_sizes=active_sizes,
                    active_rank=active_rank,
                    step_index=step,
                    decay_bias=decay_bias,
                    input_gain_bias=input_gain_bias,
                    recurrent_gain_bias=recurrent_gain_bias,
                    carry_gain_bias=carry_gain_bias,
                    normalization_step_sizes=self.per_mode_step_sizes,
                    normalization_blend_predictor=self.normalization_blend_predictor,
                    allow_growth=allow_resize,
                    return_active_sizes=True,
                )
                state_reals[engine_index] = next_real
                state_imags[engine_index] = next_imag
                state_accumulators[engine_index] = next_accumulator

                state_mag = _mask_to_active(
                    torch.sqrt(next_real.square() + next_imag.square() + 1e-6),
                    active_sizes,
                    self.state_rank,
                )
                state_features = torch.cat(
                    (
                        next_real.reshape(batch, -1),
                        next_imag.reshape(batch, -1),
                        state_mag.reshape(batch, -1),
                        next_accumulator.reshape(batch, -1),
                    ),
                    dim=-1,
                )
                # Feed forward the same accumulator that shaped the update so
                # the hidden-space return map can read out memory strength.
                engine_deltas.append(self.engine_state_to_hidden[engine_index](state_features))

            delta = self.engine_fusion(torch.cat(engine_deltas, dim=-1))
            gate = torch.sigmoid(self.gate_proj(torch.cat((x[:, step, :], delta), dim=-1)))
            outputs.append(self.dropout(gate * delta))

        return torch.stack(outputs, dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = int(config.dim * config.mlp_ratio)
        self.attn_norm = nn.LayerNorm(config.dim)
        self.attn = CausalSelfAttention(config.dim, config.n_heads, config.dropout)
        self.mixer_norm = nn.LayerNorm(config.dim)
        self.mixer = ReciprocatorMixer(
            hidden_dim=config.dim,
            state_dim=config.state_dim,
            state_rank=config.max_state_rank,
            initial_state_rank=config.state_rank,
            state_mode_sizes=config.state_mode_sizes,
            init_mode_sizes=config.init_mode_sizes,
            num_cube_engines=config.num_cube_engines,
            normalization=config.normalization,
            impression_rate=config.impression_rate,
            prediction_eta=config.prediction_eta,
            learnable_prediction_eta=config.learnable_prediction_eta,
            phase_scale=config.phase_scale,
            magnitude_floor=config.magnitude_floor,
            dropout=config.dropout,
            growth_threshold=config.growth_threshold,
            growth_interval=config.growth_interval,
            prune_floor=config.prune_floor,
            prune_horizon=config.prune_horizon,
            dynamic_rank=config.dynamic_rank,
            input_dependent_gains=config.input_dependent_gains,
            selective_gains=config.selective_gains,
            accumulator_modulates_gains=config.accumulator_modulates_gains,
            phase_aware_coupling=config.phase_aware_coupling,
            coupling_temperature=config.coupling_temperature,
            learnable_coupling_temperature=config.learnable_coupling_temperature,
            learned_per_mode_scaling=config.learned_per_mode_scaling,
            learned_normalization_blend=config.learned_normalization_blend,
            adaptive_growth_controls=config.adaptive_growth_controls,
        )
        self.ffn_norm = nn.LayerNorm(config.dim)
        self.ffn = FeedForward(config.dim, hidden_dim, config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mixer(self.mixer_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ModifiedTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor, targets: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        batch, steps = input_ids.shape
        if steps > self.config.max_seq_len:
            raise ValueError(f"sequence length {steps} exceeds max_seq_len={self.config.max_seq_len}")

        positions = torch.arange(steps, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated


class ComplexReciprocatorMixer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        state_rank: int,
        state_mode_sizes: tuple[int, ...],
        init_mode_sizes: tuple[int, ...],
        num_cube_engines: int,
        normalization: str,
        impression_rate: float,
        magnitude_floor: float,
        dropout: float,
        growth_threshold: float,
        growth_interval: int,
        prune_floor: float,
        prune_horizon: int,
        persist_state: bool,
        initial_state_rank: Optional[int] = None,
        dynamic_rank: bool = False,
        input_dependent_gains: bool = False,
        selective_gains: bool = False,
        prediction_eta: float = 0.1,
        learnable_prediction_eta: bool = False,
        accumulator_modulates_gains: bool = True,
        phase_aware_coupling: bool = True,
        coupling_temperature: float = 1.0,
        learnable_coupling_temperature: bool = False,
        learned_per_mode_scaling: bool = False,
        learned_normalization_blend: bool = False,
        adaptive_growth_controls: bool = False,
    ) -> None:
        super().__init__()
        del magnitude_floor  # Deprecated no-op retained for config compatibility.
        self.state_rank = state_rank
        self.initial_state_rank = state_rank if initial_state_rank is None else int(initial_state_rank)
        self.state_mode_sizes = state_mode_sizes
        self.init_mode_sizes = init_mode_sizes
        self.num_cube_engines = num_cube_engines
        self.normalization = normalization
        self.dropout = dropout
        self.state_dim = state_dim
        self.persist_state = persist_state
        self.supports_persistent_state = True
        self.dynamic_rank = bool(dynamic_rank)
        self.input_dependent_gains = input_dependent_gains
        self.selective_gains = bool(selective_gains)
        self.accumulator_modulates_gains = accumulator_modulates_gains
        self.learned_normalization_blend = bool(learned_normalization_blend)
        self.per_mode_step_sizes = (
            nn.Parameter(torch.ones(state_rank))
            if (normalization == "per_mode" or self.learned_normalization_blend) and learned_per_mode_scaling
            else None
        )

        self.signal_proj = ComplexLinear(hidden_dim, state_dim)
        self.cube_engines = nn.ModuleList(
            [
                _CubeEngineCell(
                    state_rank=state_rank,
                    initial_state_rank=initial_state_rank,
                    max_mode_sizes=state_mode_sizes,
                    normalization=normalization,
                    impression_rate=impression_rate,
                    prediction_eta=prediction_eta,
                    learnable_prediction_eta=learnable_prediction_eta,
                    growth_threshold=growth_threshold,
                    growth_interval=growth_interval,
                    prune_floor=prune_floor,
                    prune_horizon=prune_horizon,
                    dynamic_rank=dynamic_rank,
                    accumulator_modulates_gains=accumulator_modulates_gains,
                    phase_aware_coupling=phase_aware_coupling,
                    coupling_temperature=coupling_temperature,
                    learnable_coupling_temperature=learnable_coupling_temperature,
                    adaptive_growth_controls=adaptive_growth_controls,
                )
                for _ in range(num_cube_engines)
            ]
        )
        self.engine_state_to_hidden = nn.ModuleList(
            [ComplexLinear(state_dim * 4, hidden_dim) for _ in range(num_cube_engines)]
        )
        self.engine_fusion = ComplexLinear(hidden_dim * num_cube_engines, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.gain_predictor = (
            _InputDependentGainPredictor(
                state_rank=state_rank,
                state_mode_sizes=state_mode_sizes,
                selective_gains=selective_gains,
            )
            if input_dependent_gains
            else None
        )
        self.normalization_blend_predictor = (
            _NormalizationBlendPredictor(
                state_rank=state_rank,
                prefer_per_mode=normalization == "per_mode",
            )
            if self.learned_normalization_blend
            else None
        )
        self._track_persistent_state_gradients = False
        self.register_buffer("_persistent_state_reals", None, persistent=False)
        self.register_buffer("_persistent_state_imags", None, persistent=False)
        self.register_buffer("_persistent_state_accumulators", None, persistent=False)
        self.register_buffer("_persistent_active_sizes", None, persistent=False)
        self.register_buffer("_persistent_active_rank", None, persistent=False)
        self.register_buffer(
            "_last_active_sizes",
            torch.tensor(init_mode_sizes, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_last_active_rank",
            torch.tensor(self.initial_state_rank, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("_step_counter", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_load_state_dict_post_hook(_gain_predictor_post_load_hook)

    def _reset_persistent_state(self) -> None:
        self._persistent_state_reals = None
        self._persistent_state_imags = None
        self._persistent_state_accumulators = None
        self._persistent_active_sizes = None
        self._persistent_active_rank = None
        self._last_active_sizes = torch.tensor(
            self.init_mode_sizes,
            dtype=torch.long,
            device=self._step_counter.device,
        )
        self._last_active_rank = torch.tensor(
            self.initial_state_rank,
            dtype=torch.long,
            device=self._step_counter.device,
        )
        self._step_counter.zero_()

    def _detach_persistent_state(self) -> None:
        if self._persistent_state_reals is not None:
            self._persistent_state_reals = self._persistent_state_reals.detach()
        if self._persistent_state_imags is not None:
            self._persistent_state_imags = self._persistent_state_imags.detach()
        if self._persistent_state_accumulators is not None:
            self._persistent_state_accumulators = self._persistent_state_accumulators.detach()

    def _set_track_persistent_state_gradients(self, enabled: bool) -> None:
        self._track_persistent_state_gradients = bool(enabled)
        if not enabled:
            self._detach_persistent_state()

    @property
    def gain_proj(self) -> Optional[nn.Linear]:
        return None if self.gain_predictor is None else self.gain_predictor.context_proj

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        _prepare_per_mode_step_sizes_state_dict(self.per_mode_step_sizes, state_dict, prefix)
        _prepare_gain_predictor_state_dict(self.gain_predictor, state_dict, prefix)
        _prepare_optional_module_state_dict(
            self.normalization_blend_predictor,
            state_dict,
            prefix + "normalization_blend_predictor.",
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        _filter_gain_predictor_load_keys(
            self.gain_predictor,
            prefix=prefix,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
        normalization_prefix = prefix + "normalization_blend_predictor."
        if self.normalization_blend_predictor is not None:
            missing_keys[:] = [key for key in missing_keys if not key.startswith(normalization_prefix)]
        else:
            unexpected_keys[:] = [key for key in unexpected_keys if not key.startswith(normalization_prefix)]

    def _normalize_tensor(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        active_sizes: tuple[int, ...],
        active_rank: Optional[int],
    ) -> tuple[Tensor, Tensor]:
        if self.normalization_blend_predictor is not None:
            return _blend_normalized_complex_tensor(
                real,
                imag,
                state_rank=self.state_rank,
                active_sizes=active_sizes,
                active_rank=active_rank,
                step_sizes=self.per_mode_step_sizes,
                blend_predictor=self.normalization_blend_predictor,
            )
        return _normalize_complex_tensor(
            real,
            imag,
            self.normalization,
            state_rank=self.state_rank,
            active_rank=active_rank,
            step_sizes=self.per_mode_step_sizes,
        )

    def get_extra_state(self) -> dict[str, object]:
        return {
            "persist_state": bool(self.persist_state),
            "persistent_state_reals": None
            if self._persistent_state_reals is None
            else self._persistent_state_reals.detach().cpu().clone(),
            "persistent_state_imags": None
            if self._persistent_state_imags is None
            else self._persistent_state_imags.detach().cpu().clone(),
            "persistent_state_accumulators": None
            if self._persistent_state_accumulators is None
            else self._persistent_state_accumulators.detach().cpu().clone(),
            "persistent_active_sizes": None
            if self._persistent_active_sizes is None
            else self._persistent_active_sizes.detach().cpu().clone(),
            "persistent_active_rank": None
            if self._persistent_active_rank is None
            else int(self._persistent_active_rank.item()),
            "step_counter": int(self._step_counter.item()),
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, dict):
            self._reset_persistent_state()
            return

        self.persist_state = bool(state.get("persist_state", self.persist_state))
        device = self.signal_proj.weight_real.device
        dtype = self.signal_proj.weight_real.dtype

        persistent_state_reals = state.get("persistent_state_reals")
        self._persistent_state_reals = (
            None
            if persistent_state_reals is None
            else persistent_state_reals.to(device=device, dtype=dtype).clone()
        )

        persistent_state_imags = state.get("persistent_state_imags")
        self._persistent_state_imags = (
            None
            if persistent_state_imags is None
            else persistent_state_imags.to(device=device, dtype=dtype).clone()
        )
        persistent_state_accumulators = state.get("persistent_state_accumulators")
        if persistent_state_accumulators is None:
            self._persistent_state_accumulators = (
                None if self._persistent_state_reals is None else torch.zeros_like(self._persistent_state_reals)
            )
        else:
            self._persistent_state_accumulators = persistent_state_accumulators.to(
                device=device,
                dtype=dtype,
            ).clone()

        persistent_active_sizes = state.get("persistent_active_sizes")
        self._persistent_active_sizes = (
            None
            if persistent_active_sizes is None
            else persistent_active_sizes.to(device=device, dtype=torch.long).clone()
        )
        persistent_active_rank = state.get("persistent_active_rank")
        self._persistent_active_rank = (
            None
            if persistent_active_rank is None
            else torch.tensor(int(persistent_active_rank), dtype=torch.long, device=device)
        )
        if self._persistent_active_rank is None and self._persistent_active_sizes is not None:
            self._persistent_active_rank = torch.tensor(
                self.initial_state_rank,
                dtype=torch.long,
                device=device,
            )

        step_counter = int(state.get("step_counter", 0))
        self._step_counter = torch.tensor(step_counter, dtype=torch.long, device=device)

    def _ensure_persistent_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> None:
        needs_reset = (
            self._persistent_state_reals is None
            or self._persistent_state_imags is None
            or self._persistent_state_accumulators is None
            or self._persistent_active_sizes is None
            or self._persistent_active_rank is None
            or self._persistent_state_reals.shape[1] != batch
            or self._persistent_state_reals.device != device
        )
        if needs_reset:
            self._persistent_state_reals = torch.zeros(
                self.num_cube_engines,
                batch,
                *self.state_mode_sizes,
                dtype=dtype,
                device=device,
            )
            self._persistent_state_imags = torch.zeros_like(self._persistent_state_reals)
            self._persistent_state_accumulators = torch.zeros_like(self._persistent_state_reals)
            self._persistent_active_sizes = torch.tensor(
                self.init_mode_sizes,
                dtype=torch.long,
                device=device,
            )
            self._persistent_active_rank = torch.tensor(
                self.initial_state_rank,
                dtype=torch.long,
                device=device,
            )
            self._step_counter.zero_()

    def diagnostics(self) -> dict[str, object]:
        if self._persistent_active_sizes is not None:
            active_sizes = tuple(int(size.item()) for size in self._persistent_active_sizes)
        elif self._last_active_sizes is not None:
            active_sizes = tuple(int(size.item()) for size in self._last_active_sizes)
        else:
            active_sizes = self.init_mode_sizes
        if self._persistent_active_rank is not None:
            active_rank = int(self._persistent_active_rank.item())
        elif self._last_active_rank is not None:
            active_rank = int(self._last_active_rank.item())
        else:
            active_rank = self.initial_state_rank
        return {
            "active_sizes": active_sizes,
            "active_rank": active_rank,
            "initial_state_rank": self.initial_state_rank,
            "max_state_rank": self.state_rank,
            "dynamic_rank": self.dynamic_rank,
            "persist_state": self.persist_state,
            "input_dependent_gains": self.input_dependent_gains,
            "selective_gains": self.selective_gains,
            "accumulator_modulates_gains": self.accumulator_modulates_gains,
            "step_counter": int(self._step_counter.item()),
            "track_persistent_state_gradients": self._track_persistent_state_gradients,
            "engines": [engine.diagnostics(active_sizes, active_rank) for engine in self.cube_engines],
        }

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, _ = x.shape
        if self.persist_state:
            self._ensure_persistent_state(batch, x.device, x.real.dtype)
            assert self._persistent_state_reals is not None
            assert self._persistent_state_imags is not None
            assert self._persistent_state_accumulators is not None
            assert self._persistent_active_sizes is not None
            assert self._persistent_active_rank is not None
            state_reals = [self._persistent_state_reals[index] for index in range(self.num_cube_engines)]
            state_imags = [self._persistent_state_imags[index] for index in range(self.num_cube_engines)]
            state_accumulators = [
                self._persistent_state_accumulators[index] for index in range(self.num_cube_engines)
            ]
            active_sizes = tuple(int(size.item()) for size in self._persistent_active_sizes)
            active_rank = int(self._persistent_active_rank.item())
            step_base = int(self._step_counter.item())
        else:
            state_reals = [
                torch.zeros(batch, *self.state_mode_sizes, dtype=x.real.dtype, device=x.device)
                for _ in range(self.num_cube_engines)
            ]
            state_imags = [torch.zeros_like(state_reals[0]) for _ in range(self.num_cube_engines)]
            state_accumulators = [torch.zeros_like(state_reals[0]) for _ in range(self.num_cube_engines)]
            active_sizes = self.init_mode_sizes
            active_rank = self.initial_state_rank
            step_base = 0
        outputs = []
        signal = self.signal_proj(x)
        signal_shape = (-1, x.size(1), *self.state_mode_sizes)
        signal_real = signal.real.view(*signal_shape)
        signal_imag = signal.imag.view(*signal_shape)

        for step in range(steps):
            carry_real = torch.zeros(batch, *self.state_mode_sizes, dtype=x.real.dtype, device=x.device)
            carry_imag = torch.zeros_like(carry_real)
            engine_deltas = []
            allow_resize = self.dynamic_rank and (self.training or self.persist_state)
            if self.persist_state and self.training and self._track_persistent_state_gradients:
                allow_resize = False

            # Serial loop is currently the reference implementation.
            for engine_index, engine in enumerate(self.cube_engines):
                sr = _mask_to_active(signal_real[:, step, :], active_sizes, self.state_rank)
                si = _mask_to_active(signal_imag[:, step, :], active_sizes, self.state_rank)
                sr, si = self._normalize_tensor(
                    sr,
                    si,
                    active_sizes=active_sizes,
                    active_rank=active_rank,
                )
                decay_bias = None
                input_gain_bias = None
                recurrent_gain_bias = None
                carry_gain_bias = None
                if self.gain_predictor is not None:
                    gain_biases = _predict_gain_bias_tensor(
                        self.gain_predictor,
                        signal_real=sr,
                        signal_imag=si,
                        state_real=state_reals[engine_index],
                        state_imag=state_imags[engine_index],
                        active_sizes=active_sizes,
                        state_rank=self.state_rank,
                        active_rank=active_rank,
                    )
                    assert gain_biases is not None
                    gain_axis = sr.ndim - self.state_rank
                    decay_bias, input_gain_bias, recurrent_gain_bias, carry_gain_bias = gain_biases.unbind(
                        dim=gain_axis
                    )
                (
                    next_real,
                    next_imag,
                    next_accumulator,
                    carry_real,
                    carry_imag,
                    active_sizes,
                    active_rank,
                ) = engine.step(
                    signal_real=sr,
                    signal_imag=si,
                    state_real=state_reals[engine_index],
                    state_imag=state_imags[engine_index],
                    magnitude_accumulator=state_accumulators[engine_index],
                    carry_real=carry_real,
                    carry_imag=carry_imag,
                    active_sizes=active_sizes,
                    active_rank=active_rank,
                    step_index=step_base + step + 1,
                    decay_bias=decay_bias,
                    input_gain_bias=input_gain_bias,
                    recurrent_gain_bias=recurrent_gain_bias,
                    carry_gain_bias=carry_gain_bias,
                    normalization_step_sizes=self.per_mode_step_sizes,
                    normalization_blend_predictor=self.normalization_blend_predictor,
                    allow_growth=allow_resize,
                    return_active_sizes=True,
                )
                state_reals[engine_index] = next_real
                state_imags[engine_index] = next_imag
                state_accumulators[engine_index] = next_accumulator

                state_mag = _mask_to_active(
                    torch.sqrt(next_real.square() + next_imag.square() + 1e-6),
                    active_sizes,
                    self.state_rank,
                )
                state_features = torch.cat(
                    (
                        next_real.reshape(batch, -1),
                        next_imag.reshape(batch, -1),
                        state_mag.reshape(batch, -1),
                        next_accumulator.reshape(batch, -1),
                    ),
                    dim=-1,
                )
                # Preserve accumulator visibility in the hidden projection even
                # though it now actively modulates the recurrent dynamics.
                engine_deltas.append(self.engine_state_to_hidden[engine_index](state_features))

            delta = self.engine_fusion(torch.cat(engine_deltas, dim=-1))

            x_step = x[:, step, :]
            gate_input = torch.cat([x_step.real, x_step.imag, delta.real, delta.imag], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))
            outputs.append(complex_dropout(gate * delta, self.dropout, self.training))

        if self.persist_state:
            stored_reals = torch.stack(state_reals, dim=0)
            stored_imags = torch.stack(state_imags, dim=0)
            stored_accumulators = torch.stack(state_accumulators, dim=0)
            if not (self.training and self._track_persistent_state_gradients):
                stored_reals = stored_reals.detach()
                stored_imags = stored_imags.detach()
                stored_accumulators = stored_accumulators.detach()
            self._persistent_state_reals = stored_reals
            self._persistent_state_imags = stored_imags
            self._persistent_state_accumulators = stored_accumulators
            self._persistent_active_sizes = torch.tensor(active_sizes, dtype=torch.long, device=x.device)
            self._persistent_active_rank = torch.tensor(active_rank, dtype=torch.long, device=x.device)
            self._step_counter.add_(steps)
        self._last_active_sizes = torch.tensor(active_sizes, dtype=torch.long, device=x.device)
        self._last_active_rank = torch.tensor(active_rank, dtype=torch.long, device=x.device)

        return torch.stack(outputs, dim=1)

    def forward_parallel(self, x: Tensor) -> Tensor:
        """Parallel version (to be implemented later).
        MUST produce mathematically identical results to the serial forward()."""
        raise NotImplementedError(
            "Parallel path must be bit-exact with the serial reference implementation."
        )


class ParallelComplexReciprocatorMixer(nn.Module):
    """Parallelized mixer: linear recurrence via prefix scan + nonlinear correction."""

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        state_rank: int,
        state_mode_sizes: tuple[int, ...],
        init_mode_sizes: tuple[int, ...],
        num_cube_engines: int,
        normalization: str,
        impression_rate: float,
        magnitude_floor: float,
        dropout: float,
        growth_threshold: float,
        growth_interval: int,
        prune_floor: float,
        prune_horizon: int,
        persist_state: bool,
        initial_state_rank: Optional[int] = None,
        dynamic_rank: bool = False,
        input_dependent_gains: bool = False,
        selective_gains: bool = False,
        prediction_eta: float = 0.1,
        learnable_prediction_eta: bool = False,
        accumulator_modulates_gains: bool = True,
        phase_aware_coupling: bool = True,
        coupling_temperature: float = 1.0,
        learnable_coupling_temperature: bool = False,
        learned_per_mode_scaling: bool = False,
        learned_normalization_blend: bool = False,
        adaptive_growth_controls: bool = False,
    ) -> None:
        super().__init__()
        if persist_state:
            raise ValueError("parallel_mixer does not support persist_state")
        del magnitude_floor  # Deprecated no-op retained for config compatibility.
        self.state_rank = state_rank
        self.initial_state_rank = state_rank if initial_state_rank is None else int(initial_state_rank)
        self.state_mode_sizes = state_mode_sizes
        self.init_mode_sizes = init_mode_sizes
        self.num_cube_engines = num_cube_engines
        self.normalization = normalization
        self.dropout = dropout
        self.persist_state = False
        self.supports_persistent_state = False
        self.state_dim = state_dim
        self.dynamic_rank = bool(dynamic_rank)
        self.input_dependent_gains = input_dependent_gains
        self.selective_gains = bool(selective_gains)
        self.accumulator_modulates_gains = accumulator_modulates_gains
        self.learned_normalization_blend = bool(learned_normalization_blend)
        self.per_mode_step_sizes = (
            nn.Parameter(torch.ones(state_rank))
            if (normalization == "per_mode" or self.learned_normalization_blend) and learned_per_mode_scaling
            else None
        )

        self.signal_proj = ComplexLinear(hidden_dim, state_dim)
        self.cube_engines = nn.ModuleList(
            [
                _CubeEngineCell(
                    state_rank=state_rank,
                    initial_state_rank=initial_state_rank,
                    max_mode_sizes=state_mode_sizes,
                    normalization=normalization,
                    impression_rate=impression_rate,
                    prediction_eta=prediction_eta,
                    learnable_prediction_eta=learnable_prediction_eta,
                    growth_threshold=growth_threshold,
                    growth_interval=growth_interval,
                    prune_floor=prune_floor,
                    prune_horizon=prune_horizon,
                    dynamic_rank=dynamic_rank,
                    accumulator_modulates_gains=accumulator_modulates_gains,
                    phase_aware_coupling=phase_aware_coupling,
                    coupling_temperature=coupling_temperature,
                    learnable_coupling_temperature=learnable_coupling_temperature,
                    adaptive_growth_controls=adaptive_growth_controls,
                )
                for _ in range(num_cube_engines)
            ]
        )
        self.engine_state_to_hidden = nn.ModuleList(
            [ComplexLinear(state_dim * 4, hidden_dim) for _ in range(num_cube_engines)]
        )
        self.engine_fusion = ComplexLinear(hidden_dim * num_cube_engines, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.gain_predictor = (
            _InputDependentGainPredictor(
                state_rank=state_rank,
                state_mode_sizes=state_mode_sizes,
                selective_gains=selective_gains,
            )
            if input_dependent_gains
            else None
        )
        self.normalization_blend_predictor = (
            _NormalizationBlendPredictor(
                state_rank=state_rank,
                prefer_per_mode=normalization == "per_mode",
            )
            if self.learned_normalization_blend
            else None
        )
        self.register_buffer(
            "_last_active_sizes",
            torch.tensor(init_mode_sizes, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_last_active_rank",
            torch.tensor(self.initial_state_rank, dtype=torch.long),
            persistent=False,
        )
        self.register_load_state_dict_post_hook(_gain_predictor_post_load_hook)

    def _reset_persistent_state(self) -> None:
        return None

    @property
    def gain_proj(self) -> Optional[nn.Linear]:
        return None if self.gain_predictor is None else self.gain_predictor.context_proj

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        _prepare_per_mode_step_sizes_state_dict(self.per_mode_step_sizes, state_dict, prefix)
        _prepare_gain_predictor_state_dict(self.gain_predictor, state_dict, prefix)
        _prepare_optional_module_state_dict(
            self.normalization_blend_predictor,
            state_dict,
            prefix + "normalization_blend_predictor.",
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        _filter_gain_predictor_load_keys(
            self.gain_predictor,
            prefix=prefix,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
        normalization_prefix = prefix + "normalization_blend_predictor."
        if self.normalization_blend_predictor is not None:
            missing_keys[:] = [key for key in missing_keys if not key.startswith(normalization_prefix)]
        else:
            unexpected_keys[:] = [key for key in unexpected_keys if not key.startswith(normalization_prefix)]

    def _normalize_tensor(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        active_sizes: tuple[int, ...],
        active_rank: Optional[int],
    ) -> tuple[Tensor, Tensor]:
        if self.normalization_blend_predictor is not None:
            return _blend_normalized_complex_tensor(
                real,
                imag,
                state_rank=self.state_rank,
                active_sizes=active_sizes,
                active_rank=active_rank,
                step_sizes=self.per_mode_step_sizes,
                blend_predictor=self.normalization_blend_predictor,
            )
        return _normalize_complex_tensor(
            real,
            imag,
            self.normalization,
            state_rank=self.state_rank,
            active_rank=active_rank,
            step_sizes=self.per_mode_step_sizes,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "active_sizes": tuple(int(size.item()) for size in self._last_active_sizes),
            "active_rank": int(self._last_active_rank.item()),
            "initial_state_rank": self.initial_state_rank,
            "max_state_rank": self.state_rank,
            "dynamic_rank": self.dynamic_rank,
            "persist_state": self.persist_state,
            "input_dependent_gains": self.input_dependent_gains,
            "selective_gains": self.selective_gains,
            "accumulator_modulates_gains": self.accumulator_modulates_gains,
            "step_counter": 0,
            "engines": [
                engine.diagnostics(
                    tuple(int(size.item()) for size in self._last_active_sizes),
                    int(self._last_active_rank.item()),
                )
                for engine in self.cube_engines
            ],
        }

    def _forward_dynamic_rank(self, x: Tensor) -> Tensor:
        batch, steps, _ = x.shape
        signal = self.signal_proj(x)
        signal_shape = (-1, x.size(1), *self.state_mode_sizes)
        signal_real = signal.real.view(*signal_shape)
        signal_imag = signal.imag.view(*signal_shape)

        state_reals = [
            torch.zeros(batch, *self.state_mode_sizes, dtype=x.real.dtype, device=x.device)
            for _ in range(self.num_cube_engines)
        ]
        state_imags = [torch.zeros_like(state_reals[0]) for _ in range(self.num_cube_engines)]
        state_accumulators = [torch.zeros_like(state_reals[0]) for _ in range(self.num_cube_engines)]
        active_sizes = self.init_mode_sizes
        active_rank = self.initial_state_rank
        outputs = []

        for step in range(steps):
            carry_real = torch.zeros(batch, *self.state_mode_sizes, dtype=x.real.dtype, device=x.device)
            carry_imag = torch.zeros_like(carry_real)
            engine_deltas = []
            for engine_index, engine in enumerate(self.cube_engines):
                sr = _mask_to_active(signal_real[:, step, :], active_sizes, self.state_rank)
                si = _mask_to_active(signal_imag[:, step, :], active_sizes, self.state_rank)
                sr, si = self._normalize_tensor(
                    sr,
                    si,
                    active_sizes=active_sizes,
                    active_rank=active_rank,
                )
                decay_bias = None
                input_gain_bias = None
                recurrent_gain_bias = None
                carry_gain_bias = None
                if self.gain_predictor is not None:
                    gain_biases = _predict_gain_bias_tensor(
                        self.gain_predictor,
                        signal_real=sr,
                        signal_imag=si,
                        state_real=state_reals[engine_index],
                        state_imag=state_imags[engine_index],
                        active_sizes=active_sizes,
                        state_rank=self.state_rank,
                        active_rank=active_rank,
                    )
                    assert gain_biases is not None
                    gain_axis = sr.ndim - self.state_rank
                    decay_bias, input_gain_bias, recurrent_gain_bias, carry_gain_bias = gain_biases.unbind(
                        dim=gain_axis
                    )
                (
                    next_real,
                    next_imag,
                    next_accumulator,
                    carry_real,
                    carry_imag,
                    active_sizes,
                    active_rank,
                ) = engine.step(
                    signal_real=sr,
                    signal_imag=si,
                    state_real=state_reals[engine_index],
                    state_imag=state_imags[engine_index],
                    magnitude_accumulator=state_accumulators[engine_index],
                    carry_real=carry_real,
                    carry_imag=carry_imag,
                    active_sizes=active_sizes,
                    active_rank=active_rank,
                    step_index=step + 1,
                    decay_bias=decay_bias,
                    input_gain_bias=input_gain_bias,
                    recurrent_gain_bias=recurrent_gain_bias,
                    carry_gain_bias=carry_gain_bias,
                    normalization_step_sizes=self.per_mode_step_sizes,
                    normalization_blend_predictor=self.normalization_blend_predictor,
                    allow_growth=self.training,
                    return_active_sizes=True,
                )
                state_reals[engine_index] = next_real
                state_imags[engine_index] = next_imag
                state_accumulators[engine_index] = next_accumulator

                state_mag = _mask_to_active(
                    torch.sqrt(next_real.square() + next_imag.square() + 1e-6),
                    active_sizes,
                    self.state_rank,
                )
                state_features = torch.cat(
                    (
                        next_real.reshape(batch, -1),
                        next_imag.reshape(batch, -1),
                        state_mag.reshape(batch, -1),
                        next_accumulator.reshape(batch, -1),
                    ),
                    dim=-1,
                )
                engine_deltas.append(self.engine_state_to_hidden[engine_index](state_features))

            delta = self.engine_fusion(torch.cat(engine_deltas, dim=-1))
            gate_input = torch.cat([x[:, step, :].real, x[:, step, :].imag, delta.real, delta.imag], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))
            outputs.append(complex_dropout(gate * delta, self.dropout, self.training))

        self._last_active_sizes = torch.tensor(active_sizes, dtype=torch.long, device=x.device)
        self._last_active_rank = torch.tensor(active_rank, dtype=torch.long, device=x.device)
        return torch.stack(outputs, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        if self.dynamic_rank:
            return self._forward_dynamic_rank(x)

        batch, steps, _ = x.shape
        signal = self.signal_proj(x)
        signal_shape = (-1, x.size(1), *self.state_mode_sizes)
        signal_real = signal.real.view(*signal_shape)
        signal_imag = signal.imag.view(*signal_shape)
        signal_real = _mask_to_active(signal_real, self.init_mode_sizes, self.state_rank)
        signal_imag = _mask_to_active(signal_imag, self.init_mode_sizes, self.state_rank)
        active_rank = self.initial_state_rank
        signal_real, signal_imag = self._normalize_tensor(
            signal_real,
            signal_imag,
            active_sizes=self.init_mode_sizes,
            active_rank=active_rank,
        )
        sig_flat_real = signal_real.reshape(batch, steps, -1)
        sig_flat_imag = signal_imag.reshape(batch, steps, -1)

        carry_real = torch.zeros(batch, *self.state_mode_sizes, dtype=torch.float32, device=x.device)
        carry_imag = torch.zeros(batch, *self.state_mode_sizes, dtype=torch.float32, device=x.device)

        all_states_real = []
        all_states_imag = []
        all_state_accumulators = []

        for engine_index, engine in enumerate(self.cube_engines):
            carry_flat_real = carry_real.reshape(batch, -1).unsqueeze(1).expand(batch, steps, -1)
            carry_flat_imag = carry_imag.reshape(batch, -1).unsqueeze(1).expand(batch, steps, -1)

            if not engine.accumulator_modulates_gains:
                decay = torch.sigmoid(engine.decay).reshape(-1)
                input_gain = torch.sigmoid(engine.input_gain).reshape(-1)
                recurrent_gain_state = torch.tanh(engine.recurrent_gain)
                carry_gain = torch.tanh(engine.carry_gain).reshape(-1)

                if self.gain_predictor is not None:
                    # The disabled path intentionally matches the historical
                    # behavior exactly so legacy checkpoints and ablations stay
                    # numerically aligned.
                    provisional_input_real = input_gain * sig_flat_real + carry_gain * carry_flat_real
                    provisional_input_imag = input_gain * sig_flat_imag + carry_gain * carry_flat_imag
                    provisional_sl_flat_real = _parallel_scan_linear(decay, provisional_input_real)
                    provisional_sl_flat_imag = _parallel_scan_linear(decay, provisional_input_imag)
                    provisional_sl_real = provisional_sl_flat_real.reshape(batch, steps, *self.state_mode_sizes)
                    provisional_sl_imag = provisional_sl_flat_imag.reshape(batch, steps, *self.state_mode_sizes)

                    provisional_prev_real = torch.zeros_like(provisional_sl_real)
                    provisional_prev_imag = torch.zeros_like(provisional_sl_imag)
                    provisional_prev_real[:, 1:] = provisional_sl_real[:, :-1]
                    provisional_prev_imag[:, 1:] = provisional_sl_imag[:, :-1]

                    gain_biases = _predict_gain_bias_tensor(
                        self.gain_predictor,
                        signal_real=signal_real,
                        signal_imag=signal_imag,
                        state_real=provisional_prev_real,
                        state_imag=provisional_prev_imag,
                        active_sizes=self.init_mode_sizes,
                        state_rank=self.state_rank,
                        active_rank=active_rank,
                    )
                    assert gain_biases is not None
                    gain_axis = signal_real.ndim - self.state_rank
                    decay = torch.sigmoid(
                        engine.decay.reshape(1, 1, -1)
                        + gain_biases.select(dim=gain_axis, index=0).reshape(batch, steps, -1)
                    )
                    input_gain = torch.sigmoid(
                        engine.input_gain.reshape(1, 1, -1)
                        + gain_biases.select(dim=gain_axis, index=1).reshape(batch, steps, -1)
                    )
                    recurrent_gain_state = torch.tanh(
                        engine.recurrent_gain.reshape(1, 1, *self.state_mode_sizes)
                        + gain_biases.select(dim=gain_axis, index=2).view(batch, steps, *self.state_mode_sizes)
                    )
                    carry_gain = torch.tanh(
                        engine.carry_gain.reshape(1, 1, -1)
                        + gain_biases.select(dim=gain_axis, index=3).reshape(batch, steps, -1)
                    )

                linear_input_real = input_gain * sig_flat_real + carry_gain * carry_flat_real
                linear_input_imag = input_gain * sig_flat_imag + carry_gain * carry_flat_imag

                # Parallel prefix scan over all timesteps
                sl_flat_real = _parallel_scan_linear(decay, linear_input_real)
                sl_flat_imag = _parallel_scan_linear(decay, linear_input_imag)
                sl_real = sl_flat_real.reshape(batch, steps, *self.state_mode_sizes)
                sl_imag = sl_flat_imag.reshape(batch, steps, *self.state_mode_sizes)

                # Nonlinear correction: MC(signal[t] ⊙ sl[t-1])
                sl_prev_real = torch.zeros_like(sl_real)
                sl_prev_imag = torch.zeros_like(sl_imag)
                sl_prev_real[:, 1:] = sl_real[:, :-1]
                sl_prev_imag[:, 1:] = sl_imag[:, :-1]

                local_real, local_imag = _relational_product(
                    signal_real,
                    signal_imag,
                    sl_prev_real,
                    sl_prev_imag,
                    self.init_mode_sizes,
                    self.state_rank,
                )
                mode_couplings = _partial_trace_couplings(
                    sl_prev_real,
                    sl_prev_imag,
                    self.state_rank,
                    active_rank=active_rank,
                    phase_aware_coupling=engine.phase_aware_coupling,
                    coupling_temperature=engine._coupling_temperature_tensor(sl_prev_real),
                )
                coupled_real, coupled_imag = _apply_mode_couplings_pair(
                    local_real, local_imag, mode_couplings, self.state_rank
                )
                predicted_signal_real, predicted_signal_imag = engine._predict_signal(
                    sl_prev_real,
                    sl_prev_imag,
                    torch.zeros_like(sl_prev_real),
                    self.init_mode_sizes,
                    active_rank,
                )

                prediction_eta = engine._prediction_eta_tensor(signal_real)
                combined_real = (
                    sl_real
                    + recurrent_gain_state * coupled_real
                    + prediction_eta * (signal_real - predicted_signal_real)
                )
                combined_imag = (
                    sl_imag
                    + recurrent_gain_state * coupled_imag
                    + prediction_eta * (signal_imag - predicted_signal_imag)
                )
                norm_real, norm_imag = self._normalize_tensor(
                    combined_real,
                    combined_imag,
                    active_sizes=self.init_mode_sizes,
                    active_rank=active_rank,
                )
                combined_mag = _mask_to_active(
                    torch.sqrt(combined_real.square() + combined_imag.square() + 1e-6),
                    self.init_mode_sizes,
                    self.state_rank,
                )
                accumulator = engine._scan_magnitude_accumulator(
                    combined_mag.reshape(batch, steps, -1)
                ).reshape(
                    batch,
                    steps,
                    *self.state_mode_sizes,
                )
                accumulator = _mask_to_active(accumulator, self.init_mode_sizes, self.state_rank)
            else:
                base_decay = torch.sigmoid(engine.decay).reshape(-1)
                base_input_gain = torch.sigmoid(engine.input_gain).reshape(1, 1, -1)
                base_recurrent_gain_state = torch.tanh(engine.recurrent_gain).reshape(
                    1,
                    1,
                    *self.state_mode_sizes,
                )
                base_carry_gain = torch.tanh(engine.carry_gain).reshape(1, 1, -1)

                # The serial cell uses accumulator[t-1] as a causal modulation
                # signal. In parallel we recover the same kind of signal by
                # scanning a provisional linear state once, shifting the running
                # magnitude accumulator by one timestep, then using that trace to
                # scale the real update.
                provisional_input_real = base_input_gain * sig_flat_real + base_carry_gain * carry_flat_real
                provisional_input_imag = base_input_gain * sig_flat_imag + base_carry_gain * carry_flat_imag
                provisional_sl_flat_real = _parallel_scan_linear(base_decay, provisional_input_real)
                provisional_sl_flat_imag = _parallel_scan_linear(base_decay, provisional_input_imag)
                provisional_sl_real = provisional_sl_flat_real.reshape(batch, steps, *self.state_mode_sizes)
                provisional_sl_imag = provisional_sl_flat_imag.reshape(batch, steps, *self.state_mode_sizes)

                provisional_prev_real = torch.zeros_like(provisional_sl_real)
                provisional_prev_imag = torch.zeros_like(provisional_sl_imag)
                provisional_prev_real[:, 1:] = provisional_sl_real[:, :-1]
                provisional_prev_imag[:, 1:] = provisional_sl_imag[:, :-1]

                decay = base_decay
                input_gain = base_input_gain
                recurrent_gain_state = base_recurrent_gain_state
                carry_gain = base_carry_gain
                if self.gain_predictor is not None:
                    gain_biases = _predict_gain_bias_tensor(
                        self.gain_predictor,
                        signal_real=signal_real,
                        signal_imag=signal_imag,
                        state_real=provisional_prev_real,
                        state_imag=provisional_prev_imag,
                        active_sizes=self.init_mode_sizes,
                        state_rank=self.state_rank,
                        active_rank=active_rank,
                    )
                    assert gain_biases is not None
                    gain_axis = signal_real.ndim - self.state_rank
                    decay = torch.sigmoid(
                        engine.decay.reshape(1, 1, -1)
                        + gain_biases.select(dim=gain_axis, index=0).reshape(batch, steps, -1)
                    )
                    input_gain = torch.sigmoid(
                        engine.input_gain.reshape(1, 1, -1)
                        + gain_biases.select(dim=gain_axis, index=1).reshape(batch, steps, -1)
                    )
                    recurrent_gain_state = torch.tanh(
                        engine.recurrent_gain.reshape(1, 1, *self.state_mode_sizes)
                        + gain_biases.select(dim=gain_axis, index=2).view(batch, steps, *self.state_mode_sizes)
                    )
                    carry_gain = torch.tanh(
                        engine.carry_gain.reshape(1, 1, -1)
                        + gain_biases.select(dim=gain_axis, index=3).reshape(batch, steps, -1)
                    )

                provisional_magnitude = torch.sqrt(
                    provisional_sl_flat_real.square() + provisional_sl_flat_imag.square() + 1e-6
                )
                accumulator_prev_flat = engine._previous_magnitude_accumulator(provisional_magnitude)
                accumulator_prev = accumulator_prev_flat.reshape(batch, steps, *self.state_mode_sizes)
                accumulator_prev = _mask_to_active(accumulator_prev, self.init_mode_sizes, self.state_rank)

                input_gain = engine._modulate_gain(
                    input_gain,
                    accumulator_prev_flat,
                    engine.accumulator_input_gain_scale,
                )
                recurrent_gain_state = engine._modulate_gain(
                    recurrent_gain_state,
                    accumulator_prev,
                    engine.accumulator_recurrent_gain_scale,
                )
                carry_gain = engine._modulate_gain(
                    carry_gain,
                    accumulator_prev_flat,
                    engine.accumulator_carry_gain_scale,
                )

                linear_input_real = input_gain * sig_flat_real + carry_gain * carry_flat_real
                linear_input_imag = input_gain * sig_flat_imag + carry_gain * carry_flat_imag

                sl_flat_real = _parallel_scan_linear(decay, linear_input_real)
                sl_flat_imag = _parallel_scan_linear(decay, linear_input_imag)
                sl_real = sl_flat_real.reshape(batch, steps, *self.state_mode_sizes)
                sl_imag = sl_flat_imag.reshape(batch, steps, *self.state_mode_sizes)

                sl_prev_real = torch.zeros_like(sl_real)
                sl_prev_imag = torch.zeros_like(sl_imag)
                sl_prev_real[:, 1:] = sl_real[:, :-1]
                sl_prev_imag[:, 1:] = sl_imag[:, :-1]

                local_real, local_imag = _relational_product(
                    signal_real,
                    signal_imag,
                    sl_prev_real,
                    sl_prev_imag,
                    self.init_mode_sizes,
                    self.state_rank,
                )
                local_real, local_imag = engine._scale_coupling_drive(
                    local_real,
                    local_imag,
                    accumulator_prev,
                )
                mode_couplings = _partial_trace_couplings(
                    sl_prev_real,
                    sl_prev_imag,
                    self.state_rank,
                    active_rank=active_rank,
                    phase_aware_coupling=engine.phase_aware_coupling,
                    coupling_temperature=engine._coupling_temperature_tensor(sl_prev_real),
                )
                coupled_real, coupled_imag = _apply_mode_couplings_pair(
                    local_real, local_imag, mode_couplings, self.state_rank
                )
                predicted_signal_real, predicted_signal_imag = engine._predict_signal(
                    sl_prev_real,
                    sl_prev_imag,
                    accumulator_prev,
                    self.init_mode_sizes,
                    active_rank,
                )

                prediction_eta = engine._prediction_eta_tensor(signal_real)
                combined_real = (
                    sl_real
                    + recurrent_gain_state * coupled_real
                    + prediction_eta * (signal_real - predicted_signal_real)
                )
                combined_imag = (
                    sl_imag
                    + recurrent_gain_state * coupled_imag
                    + prediction_eta * (signal_imag - predicted_signal_imag)
                )
                norm_real, norm_imag = self._normalize_tensor(
                    combined_real,
                    combined_imag,
                    active_sizes=self.init_mode_sizes,
                    active_rank=active_rank,
                )
                combined_mag = _mask_to_active(
                    torch.sqrt(combined_real.square() + combined_imag.square() + 1e-6),
                    self.init_mode_sizes,
                    self.state_rank,
                )
                accumulator = engine._scan_magnitude_accumulator(
                    combined_mag.reshape(batch, steps, -1)
                ).reshape(
                    batch,
                    steps,
                    *self.state_mode_sizes,
                )
                accumulator = _mask_to_active(accumulator, self.init_mode_sizes, self.state_rank)

            # Carry for next engine = normalized state at each timestep
            carry_real = norm_real[:, -1].reshape(batch, *self.state_mode_sizes)
            carry_imag = norm_imag[:, -1].reshape(batch, *self.state_mode_sizes)

            all_states_real.append(norm_real)
            all_states_imag.append(norm_imag)
            all_state_accumulators.append(accumulator)

        # Extract features from all engines (vectorized across timesteps)
        engine_deltas = []
        for engine_index in range(self.num_cube_engines):
            sr = all_states_real[engine_index]
            si = all_states_imag[engine_index]
            se = all_state_accumulators[engine_index]
            sm = _mask_to_active(
                torch.sqrt(sr.square() + si.square() + 1e-6),
                self.init_mode_sizes,
                self.state_rank,
            )
            features = torch.cat(
                (
                    sr.reshape(batch, steps, -1),
                    si.reshape(batch, steps, -1),
                    sm.reshape(batch, steps, -1),
                    se.reshape(batch, steps, -1),
                ),
                dim=-1,
            )
            # Keep the accumulator in the readout features even though it now
            # also causally modulates gains and coupling earlier in the engine.
            engine_deltas.append(self.engine_state_to_hidden[engine_index](features))

        delta = self.engine_fusion(torch.cat(engine_deltas, dim=-1))
        gate_input = torch.cat([x.real, x.imag, delta.real, delta.imag], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        self._last_active_sizes = torch.tensor(self.init_mode_sizes, dtype=torch.long, device=x.device)
        self._last_active_rank = torch.tensor(self.initial_state_rank, dtype=torch.long, device=x.device)
        return complex_dropout(gate * delta, self.dropout, self.training)


class ReciprocatorOnlyBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        ffw_dim = int(config.dim * config.mlp_ratio)
        self.ffn_on_difference = config.ffn_on_difference

        mixer_cls = ParallelComplexReciprocatorMixer if config.parallel_mixer else ComplexReciprocatorMixer
        self.mixer_norm = ComplexLayerNorm(config.dim)
        self.mixer = mixer_cls(
            hidden_dim=config.dim,
            state_dim=config.state_dim,
            state_rank=config.max_state_rank,
            initial_state_rank=config.state_rank,
            state_mode_sizes=config.state_mode_sizes,
            init_mode_sizes=config.init_mode_sizes,
            num_cube_engines=config.num_cube_engines,
            normalization=config.normalization,
            impression_rate=config.impression_rate,
            prediction_eta=config.prediction_eta,
            learnable_prediction_eta=config.learnable_prediction_eta,
            magnitude_floor=config.magnitude_floor,
            dropout=config.dropout,
            growth_threshold=config.growth_threshold,
            growth_interval=config.growth_interval,
            prune_floor=config.prune_floor,
            prune_horizon=config.prune_horizon,
            persist_state=config.persist_state,
            dynamic_rank=config.dynamic_rank,
            input_dependent_gains=config.input_dependent_gains,
            selective_gains=config.selective_gains,
            accumulator_modulates_gains=config.accumulator_modulates_gains,
            phase_aware_coupling=config.phase_aware_coupling,
            coupling_temperature=config.coupling_temperature,
            learnable_coupling_temperature=config.learnable_coupling_temperature,
            learned_per_mode_scaling=config.learned_per_mode_scaling,
            learned_normalization_blend=config.learned_normalization_blend,
            adaptive_growth_controls=config.adaptive_growth_controls,
        )
        self.ffn_norm = ComplexLayerNorm(config.dim)
        self.ffn = ComplexFeedForward(
            input_dim=config.dim,
            hidden_dim=ffw_dim,
            output_dim=config.dim,
            dropout=config.dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        # The mixer produces the pure token-induced update. We always add that
        # update into the hidden state, but the cFFN can optionally operate on
        # the relational difference itself instead of the full post-mixer state.
        delta = self.mixer(self.mixer_norm(x))
        x = x + delta
        ffn_source = delta if self.ffn_on_difference else x
        x = x + self.ffn(self.ffn_norm(ffn_source))
        return x


class ReciprocatorOnlyLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self._inside_online_generate = False
        self._online_generation_prompt: Optional[Tensor] = None
        self._online_next_token_logits: Optional[Tensor] = None

        self.token_log_scale = nn.Embedding(config.vocab_size, config.dim)
        self.register_buffer(
            "rope_inverse_frequencies",
            complex_rope_frequencies(config.dim),
            persistent=False,
        )

        self.blocks = nn.ModuleList(
            [ReciprocatorOnlyBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = ComplexLayerNorm(config.dim)
        # Keep the historical wide head for phase-aware checkpoints, but feed it
        # continuous U(1)-invariant bilinear features instead of gauge-fixed channels.
        readout_dim = config.dim * 3 if config.phase_aware_readout else config.dim
        self.lm_head = nn.Linear(readout_dim, config.vocab_size, bias=False)
        self.phase_aware_readout = config.phase_aware_readout

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, ComplexLinear):
            nn.init.normal_(module.weight_real, mean=0.0, std=0.02)
            nn.init.normal_(module.weight_imag, mean=0.0, std=0.02)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def get_extra_state(self) -> dict[str, object]:
        return {
            "online_generation_prompt": None
            if self._online_generation_prompt is None
            else self._online_generation_prompt.detach().cpu().clone(),
            "online_next_token_logits": None
            if self._online_next_token_logits is None
            else self._online_next_token_logits.detach().cpu().clone(),
        }

    def set_extra_state(self, state: object) -> None:
        self._inside_online_generate = False
        if not isinstance(state, dict):
            self._online_generation_prompt = None
            self._online_next_token_logits = None
            return

        prompt = state.get("online_generation_prompt")
        logits = state.get("online_next_token_logits")
        self._online_generation_prompt = None if prompt is None else prompt.detach().cpu().clone()
        self._online_next_token_logits = None if logits is None else logits.detach().cpu().clone()

    def _load_from_state_dict(self, state_dict: dict[str, Tensor], prefix: str, *args: object, **kwargs: object) -> None:
        # Backward compatibility for checkpoints saved before rotary position encoding.
        for legacy_key in (
            "token_phase.weight",
            "position_log_scale.weight",
            "position_phase.weight",
        ):
            state_dict.pop(prefix + legacy_key, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _position_offset(self) -> int:
        if not self._online_mode_enabled():
            return 0
        offsets = []
        for block in self.blocks:
            step_counter = getattr(block.mixer, "_step_counter", None)
            if isinstance(step_counter, Tensor):
                offsets.append(int(step_counter.item()))
        if not offsets:
            return 0
        offset = offsets[0]
        if any(candidate != offset for candidate in offsets[1:]):
            raise RuntimeError("online mixer step counters diverged across layers")
        return offset

    def forward(self, input_ids: Tensor, targets: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        batch, steps = input_ids.shape
        if steps > self.config.max_seq_len:
            raise ValueError(f"sequence length {steps} exceeds max_seq_len={self.config.max_seq_len}")

        position_offset = self._position_offset()
        positions = torch.arange(position_offset, position_offset + steps, device=input_ids.device)
        reference = torch.ones(
            batch, steps, self.config.dim, dtype=torch.complex64, device=input_ids.device
        )
        x = complex_modulation_factor(
            log_scale=0.1 * torch.tanh(self.token_log_scale(input_ids)),
            phase=torch.zeros(batch, steps, self.config.dim, dtype=reference.real.dtype, device=input_ids.device),
            reference=reference,
        )
        x = apply_complex_rope(
            x,
            positions,
            inverse_frequencies=self.rope_inverse_frequencies,
        )

        for block in self.blocks:
            x = block(x)

        hidden = self.final_norm(x)
        if self.phase_aware_readout:
            # Phase-aware mode now stays globally phase-invariant all the way into
            # the real lm_head by using invariant bilinear readout features.
            readout = complex_readout_features(hidden, mode="phase_aware")
        else:
            readout = hidden.abs()
        logits = self.lm_head(readout)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if self._online_mode_enabled() and not self._inside_online_generate:
            self._online_generation_prompt = None
            self._online_next_token_logits = None
        return logits, loss

    def enter_online_mode(self) -> None:
        for block in self.blocks:
            if not bool(getattr(block.mixer, "supports_persistent_state", False)):
                raise RuntimeError("Online mode is not supported when parallel_mixer=True.")
            if hasattr(block.mixer, "persist_state"):
                block.mixer.persist_state = True

    def reset_online_state(self) -> None:
        for block in self.blocks:
            reset = getattr(block.mixer, "_reset_persistent_state", None)
            if callable(reset):
                reset()
        self._online_generation_prompt = None
        self._online_next_token_logits = None

    def detach_online_state(self) -> None:
        for block in self.blocks:
            detach = getattr(block.mixer, "_detach_persistent_state", None)
            if callable(detach):
                detach()

    def set_online_state_gradient_tracking(self, enabled: bool) -> None:
        for block in self.blocks:
            configure = getattr(block.mixer, "_set_track_persistent_state_gradients", None)
            if callable(configure):
                configure(enabled)

    def online_diagnostics(self) -> dict[str, object]:
        return {
            "layers": [
                block.mixer.diagnostics()
                for block in self.blocks
                if hasattr(block.mixer, "diagnostics")
            ]
        }

    def _online_mode_enabled(self) -> bool:
        return any(
            bool(getattr(block.mixer, "supports_persistent_state", False))
            and bool(getattr(block.mixer, "persist_state", False))
            for block in self.blocks
        )

    def _common_prefix_len(self, left: Tensor, right: Tensor) -> int:
        if left.shape[0] != right.shape[0]:
            return 0
        width = min(left.size(1), right.size(1))
        if width == 0:
            return 0
        matches = (left[:, :width] == right[:, :width]).all(dim=0)
        prefix = 0
        for index in range(width):
            if not bool(matches[index].item()):
                break
            prefix += 1
        return prefix

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
        generated = input_ids
        online_mode = self._online_mode_enabled()

        if max_new_tokens <= 0:
            return generated

        if online_mode:
            prompt_cpu = input_ids.detach().cpu()
            next_token_logits: Optional[Tensor] = None
            cached_prompt = self._online_generation_prompt

            if cached_prompt is None:
                if input_ids.size(1) == 0:
                    raise RuntimeError("Online generation needs a prompt after reset.")
                self._inside_online_generate = True
                try:
                    for start in range(0, generated.size(1), self.config.max_seq_len):
                        logits, _ = self(generated[:, start : start + self.config.max_seq_len])
                    next_token_logits = logits[:, -1, :]
                finally:
                    self._inside_online_generate = False
            else:
                prefix_len = self._common_prefix_len(prompt_cpu, cached_prompt)
                if prefix_len != prompt_cpu.size(1):
                    raise RuntimeError(
                        "Online generation cannot accept a new or extended prompt after a generation stream starts; "
                        "call reset_online_state() first."
                    )
                if self._online_next_token_logits is None:
                    raise RuntimeError("Online generation cache is missing; call reset_online_state() and retry.")
                next_token_logits = self._online_next_token_logits.to(device=input_ids.device)

            assert next_token_logits is not None
            self._inside_online_generate = True
            try:
                for _ in range(max_new_tokens):
                    if temperature <= 0:
                        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                    else:
                        probs = F.softmax(next_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat((generated, next_token), dim=1)
                    logits, _ = self(next_token)
                    next_token_logits = logits[:, -1, :]
            finally:
                self._inside_online_generate = False

            self._online_generation_prompt = prompt_cpu.clone()
            self._online_next_token_logits = next_token_logits.detach().cpu().clone()
            return generated

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated
