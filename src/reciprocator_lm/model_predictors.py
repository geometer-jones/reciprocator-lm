from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model_state import (
    _apply_mode_couplings,
    _complex_dtype_for,
    _mask_to_active,
    _normalize_complex_frobenius,
    _normalize_complex_per_mode_unrolled,
    _relational_gain_statistics,
    _relational_product,
    _summarize_complex_tensor,
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


class _ModewisePredictionProjector(nn.Module):
    """Complex separable prediction transport over tensor modes."""

    def __init__(self, state_mode_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.state_mode_sizes = state_mode_sizes
        self.weight_real = nn.ParameterList(
            [nn.Parameter(torch.zeros(mode_size, mode_size)) for mode_size in state_mode_sizes]
        )
        self.weight_imag = nn.ParameterList(
            [nn.Parameter(torch.zeros(mode_size, mode_size)) for mode_size in state_mode_sizes]
        )

    def zero_(self) -> None:
        with torch.no_grad():
            for weight_real, weight_imag in zip(self.weight_real, self.weight_imag):
                weight_real.zero_()
                weight_imag.zero_()

    def set_identity_(self) -> None:
        self.zero_()
        with torch.no_grad():
            for weight_real in self.weight_real:
                size = weight_real.shape[0]
                weight_real.copy_(torch.eye(size, dtype=weight_real.dtype, device=weight_real.device))

    def _complex_identity(
        self,
        *,
        mode_size: int,
        batch_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        eye = torch.eye(mode_size, dtype=_complex_dtype_for(dtype), device=device)
        return eye.expand(*batch_shape, mode_size, mode_size)

    def couplings(
        self,
        reference_real: Tensor,
        *,
        active_sizes: tuple[int, ...],
        state_rank: int,
        active_rank: Optional[int] = None,
    ) -> list[Tensor]:
        batch_dims = reference_real.ndim - state_rank
        batch_shape = reference_real.shape[:batch_dims]
        resolved_active_rank = state_rank if active_rank is None else int(active_rank)
        couplings: list[Tensor] = []
        for mode_idx, mode_size in enumerate(self.state_mode_sizes):
            if mode_idx >= resolved_active_rank:
                couplings.append(
                    self._complex_identity(
                        mode_size=mode_size,
                        batch_shape=batch_shape,
                        device=reference_real.device,
                        dtype=reference_real.dtype,
                    )
                )
                continue
            active_mode = int(active_sizes[mode_idx])
            coupling = torch.zeros(
                *batch_shape,
                mode_size,
                mode_size,
                dtype=_complex_dtype_for(reference_real.dtype),
                device=reference_real.device,
            )
            weight_real = self.weight_real[mode_idx][:active_mode, :active_mode].to(
                dtype=reference_real.dtype,
                device=reference_real.device,
            )
            weight_imag = self.weight_imag[mode_idx][:active_mode, :active_mode].to(
                dtype=reference_real.dtype,
                device=reference_real.device,
            )
            coupling[..., :active_mode, :active_mode] = torch.complex(weight_real, weight_imag)
            couplings.append(coupling)
        return couplings

    def forward(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        active_sizes: tuple[int, ...],
        state_rank: int,
        active_rank: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        couplings = self.couplings(
            real,
            active_sizes=active_sizes,
            state_rank=state_rank,
            active_rank=active_rank,
        )
        projected = _apply_mode_couplings(torch.complex(real, imag), couplings, state_rank)
        return projected.real, projected.imag

__all__ = [name for name in globals() if name.startswith("_") and name not in {"__builtins__", "__all__"}]
