from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model_state import _active_slice, _normalize_complex_tensor

def _spectral_frequency_radius_squared(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if not shape:
        return torch.zeros((), dtype=dtype, device=device)
    axes = [torch.fft.fftfreq(size, device=device).to(dtype=dtype) for size in shape]
    radius_squared = torch.zeros(shape, dtype=dtype, device=device)
    for axis in torch.meshgrid(*axes, indexing="ij"):
        radius_squared = radius_squared + axis.square()
    return radius_squared


def _haar_wavelet_split(coefficients: Tensor) -> tuple[Tensor, Tensor, int]:
    original_size = coefficients.size(-1)
    if original_size <= 1:
        raise ValueError("haar wavelet split requires at least two coefficients")
    if original_size % 2 != 0:
        pad_shape = (*coefficients.shape[:-1], 1)
        coefficients = torch.cat((coefficients, coefficients.new_zeros(*pad_shape)), dim=-1)
    even = coefficients[..., 0::2]
    odd = coefficients[..., 1::2]
    scale = math.sqrt(2.0)
    low = (even + odd) / scale
    high = (even - odd) / scale
    return low, high, original_size


def _haar_wavelet_merge(low: Tensor, high: Tensor, original_size: int) -> Tensor:
    if low.shape != high.shape:
        raise ValueError("haar wavelet merge requires matching low/high shapes")
    scale = math.sqrt(2.0)
    even = (low + high) / scale
    odd = (low - high) / scale
    restored = low.new_empty(*low.shape[:-1], low.size(-1) * 2)
    restored[..., 0::2] = even
    restored[..., 1::2] = odd
    return restored[..., :original_size]


def _spectral_mode_is_gauge_aware(mode: str) -> bool:
    return mode in {"wavelet_packet_max_gauge", "wavelet_packet_max_ultimate"}


class SpectralReciprocator(nn.Module):
    """Multi-resolution spectral reciprocation over the active oscillator state."""

    def __init__(
        self,
        *,
        state_rank: int,
        spectral_mode: str,
        wavelet_name: str,
        wavelet_levels: int,
        wavelet_packet_best_basis: bool,
        wavelet_packet_prune_ratio: float,
        wavelet_packet_spectral_subtraction: bool,
        wavelet_packet_stationary: bool,
        wavelet_packet_cycle_spins: int,
        packet_gain_hidden_dim: int = 12,
    ) -> None:
        super().__init__()
        self.state_rank = state_rank
        self.spectral_mode = spectral_mode
        self.wavelet_name = wavelet_name
        self.wavelet_levels = int(wavelet_levels)
        self.wavelet_packet_best_basis = bool(wavelet_packet_best_basis)
        self.wavelet_packet_prune_ratio = float(wavelet_packet_prune_ratio)
        self.wavelet_packet_spectral_subtraction = bool(wavelet_packet_spectral_subtraction)
        self.wavelet_packet_stationary = bool(wavelet_packet_stationary)
        self.wavelet_packet_cycle_spins = int(wavelet_packet_cycle_spins)
        self.packet_gain_proj = nn.Linear(7, packet_gain_hidden_dim)
        self.packet_gain_out = nn.Linear(packet_gain_hidden_dim, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.packet_gain_proj.weight)
        nn.init.zeros_(self.packet_gain_proj.bias)
        nn.init.zeros_(self.packet_gain_out.weight)
        nn.init.zeros_(self.packet_gain_out.bias)

    def _fft_filter(
        self,
        reference: Tensor,
        active_sizes: tuple[int, ...],
        *,
        low_frequency_gain: Tensor,
        sigma: Tensor,
        high_frequency_gain: Tensor,
        cutoff: Tensor,
    ) -> Tensor:
        radius_squared = _spectral_frequency_radius_squared(
            tuple(int(size) for size in active_sizes),
            device=reference.device,
            dtype=reference.dtype,
        )
        low_frequency_boost = 1.0 + low_frequency_gain * torch.exp(-radius_squared / sigma.square())
        radius = torch.sqrt(radius_squared + 1e-12)
        smoothing_width = sigma.clamp_min(5e-2)
        low_band_gate = torch.sigmoid((cutoff - radius) / smoothing_width)
        high_frequency_damping = high_frequency_gain + (1.0 - high_frequency_gain) * low_band_gate
        return low_frequency_boost * high_frequency_damping

    def _phase_coherence(self, coefficients: Tensor, global_phase_unit: Tensor) -> Tensor:
        unit = coefficients / coefficients.abs().clamp_min(1e-6)
        coherence = (unit * global_phase_unit.conj()).mean(dim=-1, keepdim=True).abs()
        return coherence.clamp(0.0, 1.0)

    def _packet_cost(self, coefficients: Tensor, global_phase_unit: Tensor) -> float:
        if coefficients.size(-1) <= 1:
            return 0.0
        energy = coefficients.abs().square()
        total = energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        probabilities = (energy / total).clamp_min(1e-6)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        if not _spectral_mode_is_gauge_aware(self.spectral_mode):
            return float(entropy.mean().detach().item())
        coherence = self._phase_coherence(coefficients, global_phase_unit)
        gauge_cost = entropy + (1.0 - coherence.squeeze(-1))
        return float(gauge_cost.mean().detach().item())

    def _packet_band_center(self, path: tuple[int, ...]) -> float:
        if not path:
            return 0.0
        band_index = 0
        for bit in path:
            band_index = (band_index << 1) | int(bit)
        return (band_index + 0.5) / (2 ** len(path))

    def _complex_leaf_gain(
        self,
        coefficients: Tensor,
        *,
        path: tuple[int, ...],
        depth: int,
        global_phase_unit: Tensor,
        low_frequency_gain: Tensor,
        sigma: Tensor,
        high_frequency_gain: Tensor,
        cutoff: Tensor,
    ) -> Tensor:
        magnitude = coefficients.abs().clamp_min(1e-6)
        phase = torch.atan2(coefficients.imag, coefficients.real) / math.pi
        node_log_energy = torch.log(coefficients.abs().square().mean(dim=-1, keepdim=True).clamp_min(1e-6))
        node_log_energy = node_log_energy.expand_as(magnitude)
        depth_norm = magnitude.new_full(magnitude.shape, depth / max(1, self.wavelet_levels))
        band_center = magnitude.new_full(magnitude.shape, self._packet_band_center(path))
        coherence = self._phase_coherence(coefficients, global_phase_unit).expand_as(magnitude)
        position = torch.linspace(
            0.0,
            1.0,
            steps=coefficients.size(-1),
            dtype=magnitude.dtype,
            device=magnitude.device,
        ).view(*([1] * (coefficients.ndim - 1)), coefficients.size(-1))
        position = position.expand_as(magnitude)
        features = torch.stack(
            (
                torch.log(magnitude),
                phase,
                node_log_energy,
                depth_norm,
                band_center,
                coherence,
                position,
            ),
            dim=-1,
        )
        gain_delta = self.packet_gain_out(F.silu(self.packet_gain_proj(features)))
        delta_magnitude = 0.25 * torch.tanh(gain_delta[..., 0])
        delta_phase = 0.5 * math.pi * torch.tanh(gain_delta[..., 1])

        low_band_boost = 1.0 + low_frequency_gain * torch.exp(-(band_center.square()) / sigma.square())
        high_band_damping = high_frequency_gain + (1.0 - high_frequency_gain) * torch.sigmoid(
            (cutoff - band_center) / sigma
        )
        coherence_boost = 1.0 + low_frequency_gain * coherence
        coherence_damping = high_frequency_gain + (1.0 - high_frequency_gain) * coherence
        gain_magnitude = low_band_boost * high_band_damping * coherence_boost * coherence_damping * torch.exp(
            delta_magnitude
        )
        return torch.polar(gain_magnitude, delta_phase.to(dtype=magnitude.dtype))

    def _filter_leaf(
        self,
        coefficients: Tensor,
        *,
        path: tuple[int, ...],
        depth: int,
        total_energy: float,
        global_phase_unit: Tensor,
        low_frequency_gain: Tensor,
        sigma: Tensor,
        high_frequency_gain: Tensor,
        cutoff: Tensor,
    ) -> Tensor:
        if coefficients.numel() == 0:
            return coefficients
        node_energy = float(coefficients.abs().square().sum().detach().item())
        energy_ratio = 0.0 if total_energy <= 0.0 else node_energy / total_energy
        filtered = coefficients * self._complex_leaf_gain(
            coefficients,
            path=path,
            depth=depth,
            global_phase_unit=global_phase_unit,
            low_frequency_gain=low_frequency_gain,
            sigma=sigma,
            high_frequency_gain=high_frequency_gain,
            cutoff=cutoff,
        )
        coherence_value = float(self._phase_coherence(coefficients, global_phase_unit).mean().detach().item())
        if self.wavelet_packet_spectral_subtraction:
            filtered_magnitude = filtered.abs()
            noise_floor = filtered_magnitude.median(dim=-1, keepdim=True).values
            damping = high_frequency_gain + (1.0 - high_frequency_gain) * torch.sigmoid(
                (cutoff - filtered_magnitude.new_tensor(self._packet_band_center(path))) / sigma
            )
            denoised_magnitude = torch.relu(filtered_magnitude - (1.0 - damping) * noise_floor)
            unit = filtered / filtered_magnitude.clamp_min(1e-6)
            filtered = denoised_magnitude * unit
        if energy_ratio < self.wavelet_packet_prune_ratio or coherence_value < self.wavelet_packet_prune_ratio:
            filtered = filtered * high_frequency_gain.square()
        return filtered

    def _apply_packet_node(
        self,
        coefficients: Tensor,
        *,
        depth: int,
        path: tuple[int, ...],
        max_depth: int,
        total_energy: float,
        global_phase_unit: Tensor,
        low_frequency_gain: Tensor,
        sigma: Tensor,
        high_frequency_gain: Tensor,
        cutoff: Tensor,
    ) -> tuple[Tensor, float]:
        leaf_cost = self._packet_cost(coefficients, global_phase_unit)
        leaf = self._filter_leaf(
            coefficients,
            path=path,
            depth=depth,
            total_energy=total_energy,
            global_phase_unit=global_phase_unit,
            low_frequency_gain=low_frequency_gain,
            sigma=sigma,
            high_frequency_gain=high_frequency_gain,
            cutoff=cutoff,
        )
        if depth >= max_depth or coefficients.size(-1) <= 1:
            return leaf, leaf_cost

        low, high, original_size = _haar_wavelet_split(coefficients)
        low_reconstructed, low_cost = self._apply_packet_node(
            low,
            depth=depth + 1,
            path=path + (0,),
            max_depth=max_depth,
            total_energy=total_energy,
            global_phase_unit=global_phase_unit,
            low_frequency_gain=low_frequency_gain,
            sigma=sigma,
            high_frequency_gain=high_frequency_gain,
            cutoff=cutoff,
        )
        high_reconstructed, high_cost = self._apply_packet_node(
            high,
            depth=depth + 1,
            path=path + (1,),
            max_depth=max_depth,
            total_energy=total_energy,
            global_phase_unit=global_phase_unit,
            low_frequency_gain=low_frequency_gain,
            sigma=sigma,
            high_frequency_gain=high_frequency_gain,
            cutoff=cutoff,
        )
        split_cost = low_cost + high_cost
        if self.wavelet_packet_best_basis and split_cost >= leaf_cost:
            return leaf, leaf_cost
        return _haar_wavelet_merge(low_reconstructed, high_reconstructed, original_size), split_cost

    def _apply_dwt_node(
        self,
        coefficients: Tensor,
        *,
        depth: int,
        path: tuple[int, ...],
        max_depth: int,
        total_energy: float,
        global_phase_unit: Tensor,
        low_frequency_gain: Tensor,
        sigma: Tensor,
        high_frequency_gain: Tensor,
        cutoff: Tensor,
    ) -> Tensor:
        if depth >= max_depth or coefficients.size(-1) <= 1:
            return self._filter_leaf(
                coefficients,
                path=path,
                depth=depth,
                total_energy=total_energy,
                global_phase_unit=global_phase_unit,
                low_frequency_gain=low_frequency_gain,
                sigma=sigma,
                high_frequency_gain=high_frequency_gain,
                cutoff=cutoff,
            )
        low, high, original_size = _haar_wavelet_split(coefficients)
        filtered_low = self._apply_dwt_node(
            low,
            depth=depth + 1,
            path=path + (0,),
            max_depth=max_depth,
            total_energy=total_energy,
            global_phase_unit=global_phase_unit,
            low_frequency_gain=low_frequency_gain,
            sigma=sigma,
            high_frequency_gain=high_frequency_gain,
            cutoff=cutoff,
        )
        filtered_high = self._filter_leaf(
            high,
            path=path + (1,),
            depth=depth + 1,
            total_energy=total_energy,
            global_phase_unit=global_phase_unit,
            low_frequency_gain=low_frequency_gain,
            sigma=sigma,
            high_frequency_gain=high_frequency_gain,
            cutoff=cutoff,
        )
        return _haar_wavelet_merge(filtered_low, filtered_high, original_size)

    def _apply_wavelet_once(
        self,
        flat_state: Tensor,
        *,
        low_frequency_gain: Tensor,
        sigma: Tensor,
        high_frequency_gain: Tensor,
        cutoff: Tensor,
    ) -> Tensor:
        if flat_state.size(-1) <= 1:
            return flat_state
        total_energy = float(flat_state.abs().square().sum().detach().item())
        global_phase = flat_state.mean(dim=-1, keepdim=True)
        global_phase_unit = global_phase / global_phase.abs().clamp_min(1e-6)
        max_depth = min(self.wavelet_levels, max(1, int(math.ceil(math.log2(flat_state.size(-1))))))
        if self.spectral_mode == "dwt":
            return self._apply_dwt_node(
                flat_state,
                depth=0,
                path=(),
                max_depth=max_depth,
                total_energy=total_energy,
                global_phase_unit=global_phase_unit,
                low_frequency_gain=low_frequency_gain,
                sigma=sigma,
                high_frequency_gain=high_frequency_gain,
                cutoff=cutoff,
            )
        filtered, _ = self._apply_packet_node(
            flat_state,
            depth=0,
            path=(),
            max_depth=max_depth,
            total_energy=total_energy,
            global_phase_unit=global_phase_unit,
            low_frequency_gain=low_frequency_gain,
            sigma=sigma,
            high_frequency_gain=high_frequency_gain,
            cutoff=cutoff,
        )
        return filtered

    def _apply_wavelet(
        self,
        state: Tensor,
        *,
        active_rank: int,
        low_frequency_gain: Tensor,
        sigma: Tensor,
        high_frequency_gain: Tensor,
        cutoff: Tensor,
    ) -> Tensor:
        batch_dims = state.ndim - self.state_rank
        flat_state = state.reshape(*state.shape[:batch_dims], -1)
        if not self.wavelet_packet_stationary:
            reciprocated = self._apply_wavelet_once(
                flat_state,
                low_frequency_gain=low_frequency_gain,
                sigma=sigma,
                high_frequency_gain=high_frequency_gain,
                cutoff=cutoff,
            )
            return reciprocated.reshape_as(state)

        cycle_spins = min(self.wavelet_packet_cycle_spins, max(1, flat_state.size(-1)))
        reciprocated = flat_state.new_zeros(flat_state.shape)
        for shift in range(cycle_spins):
            shifted = torch.roll(flat_state, shifts=shift, dims=-1)
            filtered = self._apply_wavelet_once(
                shifted,
                low_frequency_gain=low_frequency_gain,
                sigma=sigma,
                high_frequency_gain=high_frequency_gain,
                cutoff=cutoff,
            )
            reciprocated = reciprocated + torch.roll(filtered, shifts=-shift, dims=-1)
        reciprocated = reciprocated / cycle_spins
        return reciprocated.reshape_as(state)

    def forward(
        self,
        state: Tensor,
        *,
        active_sizes: tuple[int, ...],
        active_rank: int,
        low_frequency_gain: Tensor,
        sigma: Tensor,
        high_frequency_gain: Tensor,
        cutoff: Tensor,
    ) -> Tensor:
        if self.spectral_mode == "fft":
            batch_dims = state.ndim - self.state_rank
            fft_dims = tuple(range(batch_dims, batch_dims + active_rank))
            if not fft_dims:
                return state
            spectrum = torch.fft.fftn(state, dim=fft_dims)
            spectral_filter = self._fft_filter(
                state.real,
                active_sizes[:active_rank],
                low_frequency_gain=low_frequency_gain,
                sigma=sigma,
                high_frequency_gain=high_frequency_gain,
                cutoff=cutoff,
            ).view(*([1] * batch_dims), *active_sizes[:active_rank])
            for _ in range(self.state_rank - active_rank):
                spectral_filter = spectral_filter.unsqueeze(-1)
            return torch.fft.ifftn(spectrum * spectral_filter, dim=fft_dims)
        return self._apply_wavelet(
            state,
            active_rank=active_rank,
            low_frequency_gain=low_frequency_gain,
            sigma=sigma,
            high_frequency_gain=high_frequency_gain,
            cutoff=cutoff,
        )


def _flatten_active_complex_state(
    state: Tensor,
    *,
    active_sizes: tuple[int, ...],
    state_rank: int,
) -> Tensor:
    batch_dims = state.ndim - state_rank
    active_slice = (slice(None),) * batch_dims + _active_slice(active_sizes)
    return state[active_slice].reshape(*state.shape[:batch_dims], -1)


def _restore_active_complex_state(
    flat_state: Tensor,
    *,
    template: Tensor,
    active_sizes: tuple[int, ...],
    state_rank: int,
) -> Tensor:
    restored = template.new_zeros(template.shape)
    batch_dims = template.ndim - state_rank
    active_slice = (slice(None),) * batch_dims + _active_slice(active_sizes)
    restored[active_slice] = flat_state.reshape(*template.shape[:batch_dims], *active_sizes)
    return restored


def _mean_engine_spectral_parameters(
    cube_engines,
    *,
    reference: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    low_frequency_gain = torch.stack(
        [engine._spectral_low_frequency_gain_tensor(reference) for engine in cube_engines],
        dim=0,
    ).mean(dim=0)
    sigma = torch.stack(
        [engine._spectral_low_frequency_sigma_tensor(reference).clamp_min(1e-6) for engine in cube_engines],
        dim=0,
    ).mean(dim=0)
    high_frequency_gain = torch.stack(
        [engine._spectral_high_frequency_gain_tensor(reference) for engine in cube_engines],
        dim=0,
    ).mean(dim=0)
    cutoff = torch.stack(
        [engine._spectral_high_frequency_cutoff_tensor(reference) for engine in cube_engines],
        dim=0,
    ).mean(dim=0)
    return low_frequency_gain, sigma, high_frequency_gain, cutoff


def _apply_joint_engine_spectral_reciprocation(
    reciprocator: SpectralReciprocator,
    *,
    state_reals: list[Tensor],
    state_imags: list[Tensor],
    state_rank: int,
    active_sizes: tuple[int, ...],
    low_frequency_gain: Tensor,
    sigma: Tensor,
    high_frequency_gain: Tensor,
    cutoff: Tensor,
) -> tuple[list[Tensor], list[Tensor]]:
    if len(state_reals) != len(state_imags):
        raise ValueError("state_reals and state_imags must contain the same number of engines")
    if not state_reals:
        return state_reals, state_imags

    joint_chunks = []
    for state_real, state_imag in zip(state_reals, state_imags):
        joint_chunks.append(
            _flatten_active_complex_state(
                torch.complex(state_real, state_imag),
                active_sizes=active_sizes,
                state_rank=state_rank,
            )
        )
    joint_state = torch.cat(joint_chunks, dim=-1)
    joint_state = reciprocator(
        joint_state,
        active_sizes=(joint_state.size(-1),),
        active_rank=1,
        low_frequency_gain=low_frequency_gain,
        sigma=sigma,
        high_frequency_gain=high_frequency_gain,
        cutoff=cutoff,
    )
    joint_real, joint_imag = _normalize_complex_tensor(
        joint_state.real,
        joint_state.imag,
        "frobenius",
        state_rank=1,
        active_rank=1,
    )
    joint_state = torch.complex(joint_real, joint_imag)

    chunk_width = joint_chunks[0].size(-1)
    reciprocated_chunks = joint_state.split(chunk_width, dim=-1)
    next_reals: list[Tensor] = []
    next_imags: list[Tensor] = []
    for state_real, state_imag, chunk in zip(state_reals, state_imags, reciprocated_chunks):
        restored = _restore_active_complex_state(
            chunk,
            template=torch.complex(state_real, state_imag),
            active_sizes=active_sizes,
            state_rank=state_rank,
        )
        next_reals.append(restored.real)
        next_imags.append(restored.imag)
    return next_reals, next_imags

__all__ = [
    name
    for name in globals()
    if (name.startswith("_") or name == "SpectralReciprocator") and name not in {"__builtins__", "__all__"}
]
