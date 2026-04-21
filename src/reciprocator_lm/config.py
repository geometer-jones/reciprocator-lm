import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .ablation import select_mode_size_pair


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 256
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    state_dim: int = 4096
    state_rank: int = 3
    max_state_rank: int = 0
    dynamic_rank: bool = False
    state_mode_sizes: Optional[Tuple[int, ...]] = None
    max_mode_sizes: Optional[Tuple[int, ...]] = None
    init_mode_sizes: Optional[Tuple[int, ...]] = None
    num_cube_engines: int = 4
    normalization: str = "frobenius"
    learned_per_mode_scaling: bool = False
    impression_rate: float = 0.35
    prediction_eta: float = 0.1
    learnable_prediction_eta: bool = False
    phase_scale: float = 3.141592653589793
    # Deprecated no-op retained so older configs/checkpoints still deserialize.
    magnitude_floor: float = 1e-8
    growth_threshold: float = 0.15
    growth_interval: int = 4
    prune_floor: float = 1e-6
    prune_horizon: int = 128
    adaptive_growth_controls: bool = False
    persist_state: bool = False
    complex_backbone: bool = False
    parallel_mixer: bool = False
    # Default True preserves the new relational FFN routing.
    ffn_on_difference: bool = True
    input_dependent_gains: bool = True
    selective_gains: bool = False
    accumulator_modulates_gains: bool = True
    phase_aware_readout: bool = True
    phase_aware_coupling: bool = True
    coupling_temperature: float = 1.0
    learnable_coupling_temperature: bool = False
    learned_normalization_blend: bool = False
    use_spectral_reciprocation: bool = True
    learnable_spectral_reciprocation: bool = True
    spectral_mode: str = "wavelet_packet_max_ultimate"
    joint_spectral_mode: Optional[bool] = None
    spectral_low_frequency_gain: float = 0.15
    spectral_low_frequency_sigma: float = 0.2
    spectral_high_frequency_gain: float = 0.85
    spectral_high_frequency_cutoff: float = 0.25
    wavelet_name: str = "haar"
    wavelet_levels: int = 3
    wavelet_packet_best_basis: bool = True
    wavelet_packet_prune_ratio: float = 1e-3
    wavelet_packet_spectral_subtraction: bool = True
    wavelet_packet_stationary: bool = True
    wavelet_packet_cycle_spins: int = 2
    # Deprecated no-op, retained so older checkpoints/config payloads still load.
    training_growth_enabled: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0 or self.dim % self.n_heads != 0:
            raise ValueError("n_heads must evenly divide dim")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.state_rank <= 0:
            raise ValueError("state_rank must be positive")
        if self.max_state_rank <= 0:
            self.max_state_rank = self.state_rank
        if self.max_state_rank < self.state_rank:
            raise ValueError("max_state_rank must be greater than or equal to state_rank")
        if self.num_cube_engines <= 0:
            raise ValueError("num_cube_engines must be positive")

        def _validate_mode_sizes(
            name: str,
            value: Optional[Tuple[int, ...]],
            *,
            allow_initial_rank: bool = False,
            pad_future_ranks: bool = False,
        ) -> Optional[Tuple[int, ...]]:
            if value is None:
                return None
            normalized = tuple(int(size) for size in value)
            if any(size <= 0 for size in normalized):
                raise ValueError(f"{name} must contain positive integers")
            if len(normalized) == self.max_state_rank:
                return normalized
            if pad_future_ranks and len(normalized) == self.state_rank and self.max_state_rank > self.state_rank:
                return normalized + (2,) * (self.max_state_rank - self.state_rank)
            if allow_initial_rank and len(normalized) == self.state_rank:
                return normalized + (1,) * (self.max_state_rank - self.state_rank)
            target = "max_state_rank" if not allow_initial_rank else "state_rank or max_state_rank"
            raise ValueError(f"{name} length must match {target}")

        explicit_mode_sizes = _validate_mode_sizes("state_mode_sizes", self.state_mode_sizes)
        self.max_mode_sizes = _validate_mode_sizes(
            "max_mode_sizes",
            self.max_mode_sizes,
            pad_future_ranks=True,
        )
        self.init_mode_sizes = _validate_mode_sizes(
            "init_mode_sizes",
            self.init_mode_sizes,
            allow_initial_rank=True,
        )

        if explicit_mode_sizes is None:
            explicit_mode_sizes = self.max_mode_sizes or self.init_mode_sizes

        if explicit_mode_sizes is None:
            # Default to an exact but non-symmetric factorization when the
            # capacity allows it so modes start with distinct identities.
            _, explicit_mode_sizes = select_mode_size_pair(
                state_rank=self.max_state_rank,
                max_capacity=self.state_dim,
            )
        assert explicit_mode_sizes is not None

        if self.max_mode_sizes is None:
            self.max_mode_sizes = explicit_mode_sizes
        if self.init_mode_sizes is None:
            if self.state_rank == self.max_state_rank:
                self.init_mode_sizes = explicit_mode_sizes
            else:
                self.init_mode_sizes = (
                    explicit_mode_sizes[: self.state_rank]
                    + (1,) * (self.max_state_rank - self.state_rank)
                )

        if any(init > max_size for init, max_size in zip(self.init_mode_sizes, self.max_mode_sizes)):
            raise ValueError("init_mode_sizes must be less than or equal to max_mode_sizes")
        if self.dynamic_rank and self.max_state_rank > self.state_rank:
            future_mode_caps = self.max_mode_sizes[self.state_rank :]
            if any(max_size < 2 for max_size in future_mode_caps):
                raise ValueError(
                    "dynamic_rank requires max_mode_sizes for future ranks to provide at least 2 slots"
                )

        self.state_mode_sizes = self.max_mode_sizes
        self.state_dim = math.prod(self.state_mode_sizes)

        if self.normalization not in {"frobenius", "per_mode"}:
            raise ValueError("normalization must be 'frobenius' or 'per_mode'")
        if self.learned_per_mode_scaling and self.normalization != "per_mode" and not self.learned_normalization_blend:
            raise ValueError("learned_per_mode_scaling requires normalization='per_mode'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.prediction_eta < 0.0:
            raise ValueError("prediction_eta must be non-negative")
        if self.coupling_temperature <= 0.0:
            raise ValueError("coupling_temperature must be positive")
        if self.learnable_spectral_reciprocation and not self.use_spectral_reciprocation:
            raise ValueError("learnable_spectral_reciprocation requires use_spectral_reciprocation=True")
        if self.joint_spectral_mode is None:
            self.joint_spectral_mode = self.use_spectral_reciprocation and self.num_cube_engines > 1
        if self.spectral_mode not in {
            "wavelet_packet_max_ultimate",
            "wavelet_packet_max_gauge",
            "wavelet_packet",
            "dwt",
            "fft",
        }:
            raise ValueError(
                "spectral_mode must be 'wavelet_packet_max_ultimate', "
                "'wavelet_packet_max_gauge', 'wavelet_packet', 'dwt', or 'fft'"
            )
        if self.spectral_low_frequency_gain < 0.0:
            raise ValueError("spectral_low_frequency_gain must be non-negative")
        if self.spectral_low_frequency_sigma <= 0.0:
            raise ValueError("spectral_low_frequency_sigma must be positive")
        if not 0.0 < self.spectral_high_frequency_gain <= 1.0:
            raise ValueError("spectral_high_frequency_gain must be in (0, 1]")
        if self.spectral_high_frequency_cutoff < 0.0:
            raise ValueError("spectral_high_frequency_cutoff must be non-negative")
        if self.wavelet_name not in {"haar", "db1"}:
            raise ValueError("wavelet_name must be 'haar' or 'db1'")
        if self.wavelet_levels <= 0:
            raise ValueError("wavelet_levels must be positive")
        if self.wavelet_packet_prune_ratio < 0.0:
            raise ValueError("wavelet_packet_prune_ratio must be non-negative")
        if self.wavelet_packet_cycle_spins <= 0:
            raise ValueError("wavelet_packet_cycle_spins must be positive")
        if self.growth_threshold < 0.0:
            raise ValueError("growth_threshold must be non-negative")
        if self.growth_interval <= 0:
            raise ValueError("growth_interval must be positive")
        if self.prune_floor < 0.0:
            raise ValueError("prune_floor must be non-negative")
        if self.prune_horizon <= 0:
            raise ValueError("prune_horizon must be positive")
