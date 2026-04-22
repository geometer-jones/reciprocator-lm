from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model_predictors import *
from .model_spectral import *
from .model_state import *

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
        mode_coupling_layout: str = "full",
        mode_coupling_schedule: str = "sequential",
        coupling_temperature: float = 1.0,
        learnable_coupling_temperature: bool = False,
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
        self.mode_coupling_layout = mode_coupling_layout
        self.mode_coupling_schedule = mode_coupling_schedule
        self.learnable_coupling_temperature = bool(learnable_coupling_temperature)
        self.coupling_temperature_raw = (
            nn.Parameter(torch.tensor(_inverse_softplus(coupling_temperature), dtype=torch.float32))
            if self.learnable_coupling_temperature
            else None
        )
        self._fixed_coupling_temperature = float(coupling_temperature)
        self.use_spectral_reciprocation = bool(use_spectral_reciprocation)
        self.learnable_spectral_reciprocation = bool(learnable_spectral_reciprocation)
        self.spectral_mode = spectral_mode
        self.spectral_low_frequency_gain_raw = (
            nn.Parameter(torch.tensor(_inverse_softplus(spectral_low_frequency_gain), dtype=torch.float32))
            if self.learnable_spectral_reciprocation
            else None
        )
        self._fixed_spectral_low_frequency_gain = float(spectral_low_frequency_gain)
        self.spectral_low_frequency_sigma_raw = (
            nn.Parameter(torch.tensor(_inverse_softplus(spectral_low_frequency_sigma), dtype=torch.float32))
            if self.learnable_spectral_reciprocation
            else None
        )
        self._fixed_spectral_low_frequency_sigma = float(spectral_low_frequency_sigma)
        self.spectral_high_frequency_gain_raw = (
            nn.Parameter(torch.tensor(_inverse_sigmoid(spectral_high_frequency_gain), dtype=torch.float32))
            if self.learnable_spectral_reciprocation
            else None
        )
        self._fixed_spectral_high_frequency_gain = float(spectral_high_frequency_gain)
        self.spectral_high_frequency_cutoff_raw = (
            nn.Parameter(torch.tensor(_inverse_softplus(spectral_high_frequency_cutoff), dtype=torch.float32))
            if self.learnable_spectral_reciprocation
            else None
        )
        self._fixed_spectral_high_frequency_cutoff = float(spectral_high_frequency_cutoff)
        self.wavelet_name = wavelet_name
        self.wavelet_levels = int(wavelet_levels)
        self.wavelet_packet_best_basis = bool(wavelet_packet_best_basis)
        self.wavelet_packet_prune_ratio = float(wavelet_packet_prune_ratio)
        self.wavelet_packet_spectral_subtraction = bool(wavelet_packet_spectral_subtraction)
        self.wavelet_packet_stationary = bool(wavelet_packet_stationary)
        self.wavelet_packet_cycle_spins = int(wavelet_packet_cycle_spins)
        self.spectral_reciprocator = SpectralReciprocator(
            state_rank=self.state_rank,
            spectral_mode=self.spectral_mode,
            wavelet_name=self.wavelet_name,
            wavelet_levels=self.wavelet_levels,
            wavelet_packet_best_basis=self.wavelet_packet_best_basis,
            wavelet_packet_prune_ratio=self.wavelet_packet_prune_ratio,
            wavelet_packet_spectral_subtraction=self.wavelet_packet_spectral_subtraction,
            wavelet_packet_stationary=self.wavelet_packet_stationary,
            wavelet_packet_cycle_spins=self.wavelet_packet_cycle_spins,
        )
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
        # Preserve the current default while allowing explicit schedule
        # ablations to switch to independently-derived couplings.
        self.use_expressive_mode_couplings = self.mode_coupling_schedule == "sequential"
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
        self.prediction_proj = _ModewisePredictionProjector(self.state_mode_sizes)
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
        _prepare_optional_parameter_state_dict(
            self.spectral_low_frequency_gain_raw,
            state_dict,
            prefix,
            "spectral_low_frequency_gain_raw",
        )
        _prepare_optional_parameter_state_dict(
            self.spectral_low_frequency_sigma_raw,
            state_dict,
            prefix,
            "spectral_low_frequency_sigma_raw",
        )
        _prepare_optional_parameter_state_dict(
            self.spectral_high_frequency_gain_raw,
            state_dict,
            prefix,
            "spectral_high_frequency_gain_raw",
        )
        _prepare_optional_parameter_state_dict(
            self.spectral_high_frequency_cutoff_raw,
            state_dict,
            prefix,
            "spectral_high_frequency_cutoff_raw",
        )
        legacy_prediction_weight_real_key = prefix + "prediction_proj.weight_real"
        legacy_prediction_weight_imag_key = prefix + "prediction_proj.weight_imag"
        had_legacy_prediction_proj = (
            legacy_prediction_weight_real_key in state_dict or legacy_prediction_weight_imag_key in state_dict
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
        spectral_low_frequency_gain_key = prefix + "spectral_low_frequency_gain_raw"
        spectral_low_frequency_sigma_key = prefix + "spectral_low_frequency_sigma_raw"
        spectral_high_frequency_gain_key = prefix + "spectral_high_frequency_gain_raw"
        spectral_high_frequency_cutoff_key = prefix + "spectral_high_frequency_cutoff_raw"
        if self.prediction_eta_raw is None:
            unexpected_keys[:] = [key for key in unexpected_keys if key != prediction_eta_key]
        if self.coupling_temperature_raw is None:
            unexpected_keys[:] = [key for key in unexpected_keys if key != coupling_temperature_key]
        if self.spectral_low_frequency_gain_raw is None:
            unexpected_keys[:] = [key for key in unexpected_keys if key != spectral_low_frequency_gain_key]
        if self.spectral_low_frequency_sigma_raw is None:
            unexpected_keys[:] = [key for key in unexpected_keys if key != spectral_low_frequency_sigma_key]
        if self.spectral_high_frequency_gain_raw is None:
            unexpected_keys[:] = [key for key in unexpected_keys if key != spectral_high_frequency_gain_key]
        if self.spectral_high_frequency_cutoff_raw is None:
            unexpected_keys[:] = [key for key in unexpected_keys if key != spectral_high_frequency_cutoff_key]
        unexpected_keys[:] = [
            key
            for key in unexpected_keys
            if key not in {legacy_prediction_weight_real_key, legacy_prediction_weight_imag_key}
        ]
        if had_legacy_prediction_proj:
            missing_keys[:] = [
                key
                for key in missing_keys
                if not key.startswith(prefix + "prediction_proj.weight_real.")
                and not key.startswith(prefix + "prediction_proj.weight_imag.")
            ]

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
    def spectral_low_frequency_gain(self) -> float:
        if self.spectral_low_frequency_gain_raw is None:
            return self._fixed_spectral_low_frequency_gain
        return float(F.softplus(self.spectral_low_frequency_gain_raw.detach()).item())

    def _spectral_low_frequency_gain_tensor(self, reference: Tensor) -> Tensor:
        if self.spectral_low_frequency_gain_raw is None:
            return reference.new_tensor(self._fixed_spectral_low_frequency_gain)
        return F.softplus(self.spectral_low_frequency_gain_raw).to(dtype=reference.dtype, device=reference.device)

    @property
    def spectral_low_frequency_sigma(self) -> float:
        if self.spectral_low_frequency_sigma_raw is None:
            return self._fixed_spectral_low_frequency_sigma
        return float(F.softplus(self.spectral_low_frequency_sigma_raw.detach()).item())

    def _spectral_low_frequency_sigma_tensor(self, reference: Tensor) -> Tensor:
        if self.spectral_low_frequency_sigma_raw is None:
            return reference.new_tensor(self._fixed_spectral_low_frequency_sigma)
        return F.softplus(self.spectral_low_frequency_sigma_raw).to(dtype=reference.dtype, device=reference.device)

    @property
    def spectral_high_frequency_gain(self) -> float:
        if self.spectral_high_frequency_gain_raw is None:
            return self._fixed_spectral_high_frequency_gain
        return float(torch.sigmoid(self.spectral_high_frequency_gain_raw.detach()).item())

    def _spectral_high_frequency_gain_tensor(self, reference: Tensor) -> Tensor:
        if self.spectral_high_frequency_gain_raw is None:
            return reference.new_tensor(self._fixed_spectral_high_frequency_gain)
        return torch.sigmoid(self.spectral_high_frequency_gain_raw).to(dtype=reference.dtype, device=reference.device)

    @property
    def spectral_high_frequency_cutoff(self) -> float:
        if self.spectral_high_frequency_cutoff_raw is None:
            return self._fixed_spectral_high_frequency_cutoff
        return float(F.softplus(self.spectral_high_frequency_cutoff_raw.detach()).item())

    def _spectral_high_frequency_cutoff_tensor(self, reference: Tensor) -> Tensor:
        if self.spectral_high_frequency_cutoff_raw is None:
            return reference.new_tensor(self._fixed_spectral_high_frequency_cutoff)
        return F.softplus(self.spectral_high_frequency_cutoff_raw).to(dtype=reference.dtype, device=reference.device)

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
            "mode_coupling_layout": self.mode_coupling_layout,
            "mode_coupling_schedule": self.mode_coupling_schedule,
            "coupling_temperature": self.coupling_temperature,
            "use_spectral_reciprocation": self.use_spectral_reciprocation,
            "learnable_spectral_reciprocation": self.learnable_spectral_reciprocation,
            "spectral_mode": self.spectral_mode,
            "spectral_low_frequency_gain": self.spectral_low_frequency_gain,
            "spectral_low_frequency_sigma": self.spectral_low_frequency_sigma,
            "spectral_high_frequency_gain": self.spectral_high_frequency_gain,
            "spectral_high_frequency_cutoff": self.spectral_high_frequency_cutoff,
            "wavelet_name": self.wavelet_name,
            "wavelet_levels": self.wavelet_levels,
            "wavelet_packet_best_basis": self.wavelet_packet_best_basis,
            "wavelet_packet_prune_ratio": self.wavelet_packet_prune_ratio,
            "wavelet_packet_spectral_subtraction": self.wavelet_packet_spectral_subtraction,
            "wavelet_packet_stationary": self.wavelet_packet_stationary,
            "wavelet_packet_cycle_spins": self.wavelet_packet_cycle_spins,
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
        predicted_real, predicted_imag = self.prediction_proj(
            state_real,
            state_imag,
            active_sizes=active_sizes,
            state_rank=self.state_rank,
            active_rank=resolved_active_rank,
        )
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

    def _apply_spectral_reciprocation(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        active_sizes: tuple[int, ...],
        active_rank: Optional[int] = None,
        normalization_step_sizes: Optional[Tensor] = None,
        normalization_blend_predictor: Optional[_NormalizationBlendPredictor] = None,
    ) -> tuple[Tensor, Tensor]:
        if not self.use_spectral_reciprocation:
            return real, imag

        resolved_active_rank = self.initial_state_rank if active_rank is None else int(active_rank)
        state = torch.complex(real, imag)
        batch_dims = state.ndim - self.state_rank
        active_slice = (slice(None),) * batch_dims + _active_slice(active_sizes)
        active_state = state[active_slice]
        if resolved_active_rank <= 0:
            return real, imag

        reciprocated = self.spectral_reciprocator(
            active_state,
            active_sizes=active_sizes,
            active_rank=resolved_active_rank,
            low_frequency_gain=self._spectral_low_frequency_gain_tensor(real),
            sigma=self._spectral_low_frequency_sigma_tensor(real).clamp_min(1e-6),
            high_frequency_gain=self._spectral_high_frequency_gain_tensor(real),
            cutoff=self._spectral_high_frequency_cutoff_tensor(real),
        )

        state = state.clone()
        state[active_slice] = reciprocated
        state = _mask_to_active(state, active_sizes, self.state_rank)
        spectral_real = state.real
        spectral_imag = state.imag
        if normalization_blend_predictor is None:
            spectral_real, spectral_imag = _normalize_complex_tensor(
                spectral_real,
                spectral_imag,
                self.normalization,
                state_rank=self.state_rank,
                active_rank=active_rank,
                step_sizes=normalization_step_sizes,
            )
        else:
            spectral_real, spectral_imag = _blend_normalized_complex_tensor(
                spectral_real,
                spectral_imag,
                state_rank=self.state_rank,
                active_sizes=active_sizes,
                active_rank=active_rank,
                step_sizes=normalization_step_sizes,
                blend_predictor=normalization_blend_predictor,
            )
        spectral_real = _mask_to_active(spectral_real, active_sizes, self.state_rank)
        spectral_imag = _mask_to_active(spectral_imag, active_sizes, self.state_rank)
        return spectral_real, spectral_imag

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
            active_coupling = self._coupling_from_scores(
                logits_real,
                logits_imag,
                scale=scale,
            )
            coupling[..., :active_mode, :active_mode] = _project_mode_coupling_layout(
                active_coupling,
                layout=self.mode_coupling_layout,
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
            active_coupling = self._coupling_from_scores(
                logits_real,
                logits_imag,
                scale=scale,
            )
            coupling[..., :active_mode, :active_mode] = _project_mode_coupling_layout(
                active_coupling,
                layout=self.mode_coupling_layout,
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
        next_real, next_imag = self._apply_spectral_reciprocation(
            next_real,
            next_imag,
            active_sizes=next_active_sizes,
            active_rank=next_active_rank,
            normalization_step_sizes=normalization_step_sizes,
            normalization_blend_predictor=normalization_blend_predictor,
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

__all__ = ["_CubeEngineCell"]
