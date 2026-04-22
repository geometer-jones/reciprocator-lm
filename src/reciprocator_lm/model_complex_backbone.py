from __future__ import annotations

from typing import Optional

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
from .model_engine import _CubeEngineCell
from .model_predictors import *
from .model_spectral import *
from .model_state import *

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
        mode_coupling_layout: str = "full",
        mode_coupling_schedule: str = "sequential",
        coupling_temperature: float = 1.0,
        learnable_coupling_temperature: bool = False,
        learned_per_mode_scaling: bool = False,
        learned_normalization_blend: bool = False,
        use_spectral_reciprocation: bool = False,
        learnable_spectral_reciprocation: bool = False,
        spectral_mode: str = "wavelet_packet_max_ultimate",
        joint_spectral_mode: Optional[bool] = None,
        spectral_low_frequency_gain: float = 0.15,
        spectral_low_frequency_sigma: float = 0.2,
        spectral_high_frequency_gain: float = 0.85,
        spectral_high_frequency_cutoff: float = 0.25,
        wavelet_name: str = "haar",
        wavelet_levels: int = 3,
        wavelet_packet_best_basis: bool = True,
        wavelet_packet_prune_ratio: float = 1e-3,
        wavelet_packet_spectral_subtraction: bool = True,
        wavelet_packet_stationary: bool = True,
        wavelet_packet_cycle_spins: int = 2,
        adaptive_growth_controls: bool = False,
    ) -> None:
        super().__init__()
        self.state_rank = state_rank
        self.initial_state_rank = state_rank if initial_state_rank is None else int(initial_state_rank)
        self.state_mode_sizes = state_mode_sizes
        self.init_mode_sizes = init_mode_sizes
        self.num_cube_engines = num_cube_engines
        self.normalization = normalization
        self.dropout = dropout
        self.state_dim = state_dim
        self.engine_state_feature_dim = 6 + 2 * state_rank
        self.persist_state = persist_state
        self.supports_persistent_state = True
        self.dynamic_rank = bool(dynamic_rank)
        self.input_dependent_gains = input_dependent_gains
        self.selective_gains = bool(selective_gains)
        self.accumulator_modulates_gains = accumulator_modulates_gains
        self.learned_normalization_blend = bool(learned_normalization_blend)
        self.use_spectral_reciprocation = bool(use_spectral_reciprocation)
        resolved_joint_spectral_mode = num_cube_engines > 1 if joint_spectral_mode is None else bool(joint_spectral_mode)
        self.joint_spectral_mode = bool(
            self.use_spectral_reciprocation and resolved_joint_spectral_mode and num_cube_engines > 1
        )
        self.spectral_mode = spectral_mode
        self.per_mode_step_sizes = (
            nn.Parameter(torch.ones(state_rank))
            if (normalization == "per_mode" or self.learned_normalization_blend) and learned_per_mode_scaling
            else None
        )
        self.joint_spectral_reciprocator = (
            SpectralReciprocator(
                state_rank=1,
                spectral_mode=spectral_mode,
                wavelet_name=wavelet_name,
                wavelet_levels=wavelet_levels,
                wavelet_packet_best_basis=wavelet_packet_best_basis,
                wavelet_packet_prune_ratio=wavelet_packet_prune_ratio,
                wavelet_packet_spectral_subtraction=wavelet_packet_spectral_subtraction,
                wavelet_packet_stationary=wavelet_packet_stationary,
                wavelet_packet_cycle_spins=wavelet_packet_cycle_spins,
            )
            if self.joint_spectral_mode
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
                    mode_coupling_layout=mode_coupling_layout,
                    mode_coupling_schedule=mode_coupling_schedule,
                    coupling_temperature=coupling_temperature,
                    learnable_coupling_temperature=learnable_coupling_temperature,
                    use_spectral_reciprocation=self.use_spectral_reciprocation and not self.joint_spectral_mode,
                    learnable_spectral_reciprocation=learnable_spectral_reciprocation,
                    spectral_mode=spectral_mode,
                    spectral_low_frequency_gain=spectral_low_frequency_gain,
                    spectral_low_frequency_sigma=spectral_low_frequency_sigma,
                    spectral_high_frequency_gain=spectral_high_frequency_gain,
                    spectral_high_frequency_cutoff=spectral_high_frequency_cutoff,
                    wavelet_name=wavelet_name,
                    wavelet_levels=wavelet_levels,
                    wavelet_packet_best_basis=wavelet_packet_best_basis,
                    wavelet_packet_prune_ratio=wavelet_packet_prune_ratio,
                    wavelet_packet_spectral_subtraction=wavelet_packet_spectral_subtraction,
                    wavelet_packet_stationary=wavelet_packet_stationary,
                    wavelet_packet_cycle_spins=wavelet_packet_cycle_spins,
                    adaptive_growth_controls=adaptive_growth_controls,
                )
                for _ in range(num_cube_engines)
            ]
        )
        self.engine_state_to_hidden = nn.ModuleList(
            [ComplexLinear(self.engine_state_feature_dim, hidden_dim) for _ in range(num_cube_engines)]
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

    def _maybe_apply_joint_engine_spectral_reciprocation(
        self,
        state_reals: list[Tensor],
        state_imags: list[Tensor],
        *,
        active_sizes: tuple[int, ...],
    ) -> tuple[list[Tensor], list[Tensor]]:
        if not self.joint_spectral_mode or self.joint_spectral_reciprocator is None:
            return state_reals, state_imags
        (
            low_frequency_gain,
            sigma,
            high_frequency_gain,
            cutoff,
        ) = _mean_engine_spectral_parameters(self.cube_engines, reference=state_reals[0])
        return _apply_joint_engine_spectral_reciprocation(
            self.joint_spectral_reciprocator,
            state_reals=state_reals,
            state_imags=state_imags,
            state_rank=self.state_rank,
            active_sizes=active_sizes,
            low_frequency_gain=low_frequency_gain,
            sigma=sigma,
            high_frequency_gain=high_frequency_gain,
            cutoff=cutoff,
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
            "joint_spectral_mode": self.joint_spectral_mode,
            "spectral_mode": self.spectral_mode,
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

            state_reals, state_imags = self._maybe_apply_joint_engine_spectral_reciprocation(
                state_reals,
                state_imags,
                active_sizes=active_sizes,
            )
            engine_deltas = []
            for engine_index in range(self.num_cube_engines):
                state_features_real = _engine_state_readout_features(
                    state_reals[engine_index],
                    state_imags[engine_index],
                    state_accumulators[engine_index],
                    active_sizes=active_sizes,
                    state_rank=self.state_rank,
                    active_rank=active_rank,
                )
                state_features = torch.complex(
                    state_features_real,
                    torch.zeros_like(state_features_real),
                )
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
        mode_coupling_layout: str = "full",
        mode_coupling_schedule: str = "sequential",
        coupling_temperature: float = 1.0,
        learnable_coupling_temperature: bool = False,
        learned_per_mode_scaling: bool = False,
        learned_normalization_blend: bool = False,
        use_spectral_reciprocation: bool = False,
        learnable_spectral_reciprocation: bool = False,
        spectral_mode: str = "wavelet_packet_max_ultimate",
        joint_spectral_mode: Optional[bool] = None,
        spectral_low_frequency_gain: float = 0.15,
        spectral_low_frequency_sigma: float = 0.2,
        spectral_high_frequency_gain: float = 0.85,
        spectral_high_frequency_cutoff: float = 0.25,
        wavelet_name: str = "haar",
        wavelet_levels: int = 3,
        wavelet_packet_best_basis: bool = True,
        wavelet_packet_prune_ratio: float = 1e-3,
        wavelet_packet_spectral_subtraction: bool = True,
        wavelet_packet_stationary: bool = True,
        wavelet_packet_cycle_spins: int = 2,
        adaptive_growth_controls: bool = False,
    ) -> None:
        super().__init__()
        if persist_state:
            raise ValueError("parallel_mixer does not support persist_state")
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
        self.engine_state_feature_dim = 6 + 2 * state_rank
        self.dynamic_rank = bool(dynamic_rank)
        self.input_dependent_gains = input_dependent_gains
        self.selective_gains = bool(selective_gains)
        self.accumulator_modulates_gains = accumulator_modulates_gains
        self.learned_normalization_blend = bool(learned_normalization_blend)
        self.use_spectral_reciprocation = bool(use_spectral_reciprocation)
        resolved_joint_spectral_mode = num_cube_engines > 1 if joint_spectral_mode is None else bool(joint_spectral_mode)
        self.joint_spectral_mode = bool(
            self.use_spectral_reciprocation and resolved_joint_spectral_mode and num_cube_engines > 1
        )
        self.spectral_mode = spectral_mode
        self.per_mode_step_sizes = (
            nn.Parameter(torch.ones(state_rank))
            if (normalization == "per_mode" or self.learned_normalization_blend) and learned_per_mode_scaling
            else None
        )
        self.joint_spectral_reciprocator = (
            SpectralReciprocator(
                state_rank=1,
                spectral_mode=spectral_mode,
                wavelet_name=wavelet_name,
                wavelet_levels=wavelet_levels,
                wavelet_packet_best_basis=wavelet_packet_best_basis,
                wavelet_packet_prune_ratio=wavelet_packet_prune_ratio,
                wavelet_packet_spectral_subtraction=wavelet_packet_spectral_subtraction,
                wavelet_packet_stationary=wavelet_packet_stationary,
                wavelet_packet_cycle_spins=wavelet_packet_cycle_spins,
            )
            if self.joint_spectral_mode
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
                    mode_coupling_layout=mode_coupling_layout,
                    mode_coupling_schedule=mode_coupling_schedule,
                    coupling_temperature=coupling_temperature,
                    learnable_coupling_temperature=learnable_coupling_temperature,
                    use_spectral_reciprocation=self.use_spectral_reciprocation and not self.joint_spectral_mode,
                    learnable_spectral_reciprocation=learnable_spectral_reciprocation,
                    spectral_mode=spectral_mode,
                    spectral_low_frequency_gain=spectral_low_frequency_gain,
                    spectral_low_frequency_sigma=spectral_low_frequency_sigma,
                    spectral_high_frequency_gain=spectral_high_frequency_gain,
                    spectral_high_frequency_cutoff=spectral_high_frequency_cutoff,
                    wavelet_name=wavelet_name,
                    wavelet_levels=wavelet_levels,
                    wavelet_packet_best_basis=wavelet_packet_best_basis,
                    wavelet_packet_prune_ratio=wavelet_packet_prune_ratio,
                    wavelet_packet_spectral_subtraction=wavelet_packet_spectral_subtraction,
                    wavelet_packet_stationary=wavelet_packet_stationary,
                    wavelet_packet_cycle_spins=wavelet_packet_cycle_spins,
                    adaptive_growth_controls=adaptive_growth_controls,
                )
                for _ in range(num_cube_engines)
            ]
        )
        self.engine_state_to_hidden = nn.ModuleList(
            [ComplexLinear(self.engine_state_feature_dim, hidden_dim) for _ in range(num_cube_engines)]
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

    def _maybe_apply_joint_engine_spectral_reciprocation(
        self,
        state_reals: list[Tensor],
        state_imags: list[Tensor],
        *,
        active_sizes: tuple[int, ...],
    ) -> tuple[list[Tensor], list[Tensor]]:
        if not self.joint_spectral_mode or self.joint_spectral_reciprocator is None:
            return state_reals, state_imags
        (
            low_frequency_gain,
            sigma,
            high_frequency_gain,
            cutoff,
        ) = _mean_engine_spectral_parameters(self.cube_engines, reference=state_reals[0])
        return _apply_joint_engine_spectral_reciprocation(
            self.joint_spectral_reciprocator,
            state_reals=state_reals,
            state_imags=state_imags,
            state_rank=self.state_rank,
            active_sizes=active_sizes,
            low_frequency_gain=low_frequency_gain,
            sigma=sigma,
            high_frequency_gain=high_frequency_gain,
            cutoff=cutoff,
        )

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

            state_reals, state_imags = self._maybe_apply_joint_engine_spectral_reciprocation(
                state_reals,
                state_imags,
                active_sizes=active_sizes,
            )
            engine_deltas = []
            for engine_index in range(self.num_cube_engines):
                state_features_real = _engine_state_readout_features(
                    state_reals[engine_index],
                    state_imags[engine_index],
                    state_accumulators[engine_index],
                    active_sizes=active_sizes,
                    state_rank=self.state_rank,
                    active_rank=active_rank,
                )
                state_features = torch.complex(
                    state_features_real,
                    torch.zeros_like(state_features_real),
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
                norm_real, norm_imag = engine._apply_spectral_reciprocation(
                    norm_real,
                    norm_imag,
                    active_sizes=self.init_mode_sizes,
                    active_rank=active_rank,
                    normalization_step_sizes=self.per_mode_step_sizes,
                    normalization_blend_predictor=self.normalization_blend_predictor,
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
                norm_real, norm_imag = engine._apply_spectral_reciprocation(
                    norm_real,
                    norm_imag,
                    active_sizes=self.init_mode_sizes,
                    active_rank=active_rank,
                    normalization_step_sizes=self.per_mode_step_sizes,
                    normalization_blend_predictor=self.normalization_blend_predictor,
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

        all_states_real, all_states_imag = self._maybe_apply_joint_engine_spectral_reciprocation(
            all_states_real,
            all_states_imag,
            active_sizes=self.init_mode_sizes,
        )

        # Extract features from all engines (vectorized across timesteps)
        engine_deltas = []
        for engine_index in range(self.num_cube_engines):
            sr = all_states_real[engine_index]
            si = all_states_imag[engine_index]
            se = all_state_accumulators[engine_index]
            features_real = _engine_state_readout_features(
                sr,
                si,
                se,
                active_sizes=self.init_mode_sizes,
                state_rank=self.state_rank,
                active_rank=active_rank,
            )
            features = torch.complex(features_real, torch.zeros_like(features_real))
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
            mode_coupling_layout=config.mode_coupling_layout,
            mode_coupling_schedule=config.mode_coupling_schedule,
            coupling_temperature=config.coupling_temperature,
            learnable_coupling_temperature=config.learnable_coupling_temperature,
            learned_per_mode_scaling=config.learned_per_mode_scaling,
            learned_normalization_blend=config.learned_normalization_blend,
            use_spectral_reciprocation=config.use_spectral_reciprocation,
            learnable_spectral_reciprocation=config.learnable_spectral_reciprocation,
            spectral_mode=config.spectral_mode,
            joint_spectral_mode=config.joint_spectral_mode,
            spectral_low_frequency_gain=config.spectral_low_frequency_gain,
            spectral_low_frequency_sigma=config.spectral_low_frequency_sigma,
            spectral_high_frequency_gain=config.spectral_high_frequency_gain,
            spectral_high_frequency_cutoff=config.spectral_high_frequency_cutoff,
            wavelet_name=config.wavelet_name,
            wavelet_levels=config.wavelet_levels,
            wavelet_packet_best_basis=config.wavelet_packet_best_basis,
            wavelet_packet_prune_ratio=config.wavelet_packet_prune_ratio,
            wavelet_packet_spectral_subtraction=config.wavelet_packet_spectral_subtraction,
            wavelet_packet_stationary=config.wavelet_packet_stationary,
            wavelet_packet_cycle_spins=config.wavelet_packet_cycle_spins,
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

__all__ = [
    "ComplexReciprocatorMixer",
    "ParallelComplexReciprocatorMixer",
    "ReciprocatorOnlyBlock",
    "ReciprocatorOnlyLM",
]
