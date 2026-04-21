import math
from dataclasses import asdict
from typing import Optional

import pytest
import torch

from reciprocator_lm import ModelConfig, ModifiedTransformerLM, ReciprocatorOnlyLM, apply_complex_rope, complex_rope_frequencies
from reciprocator_lm.experiments import _annealed_growth_threshold, _reset_optimizer_moments
from reciprocator_lm.model import (
    ComplexReciprocatorMixer,
    ParallelComplexReciprocatorMixer,
    ReciprocatorOnlyBlock,
    _CubeEngineCell,
    _apply_mode_couplings,
    _apply_mode_couplings_pair,
    _mask_to_active,
    _normalize_complex_tensor,
    _parallel_scan_linear,
    _partial_trace_couplings,
    _relational_product,
    _relational_gain_statistics,
)


def make_model() -> ModifiedTransformerLM:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        num_cube_engines=3,
        normalization="per_mode",
        dropout=0.0,
    )
    return ModifiedTransformerLM(config)


def _assert_prediction_projector_has_grad(projector: torch.nn.Module) -> None:
    assert any(weight.grad is not None for weight in projector.weight_real)
    assert any(weight.grad is not None for weight in projector.weight_imag)


def test_config_derives_rank3_state_shape() -> None:
    config = ModelConfig(
        vocab_size=32,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        num_cube_engines=4,
    )

    assert config.state_rank == 3
    assert config.state_dim == 27
    assert config.state_mode_sizes == (3, 3, 3)
    assert config.num_cube_engines == 4
    assert config.normalization == "frobenius"
    assert config.learned_per_mode_scaling is False
    assert config.prediction_eta == pytest.approx(0.1)
    assert config.learnable_prediction_eta is False
    assert config.accumulator_modulates_gains is True
    assert config.input_dependent_gains is True
    assert config.selective_gains is False
    assert config.learnable_coupling_temperature is False
    assert config.learned_normalization_blend is False
    assert config.use_spectral_reciprocation is True
    assert config.learnable_spectral_reciprocation is True
    assert config.spectral_mode == "wavelet_packet_max_ultimate"
    assert config.joint_spectral_mode is True
    assert config.wavelet_name == "haar"
    assert config.wavelet_levels == 3
    assert config.wavelet_packet_best_basis is True
    assert config.wavelet_packet_stationary is True
    assert config.adaptive_growth_controls is False


def test_config_derives_asymmetric_mode_sizes_from_state_dim() -> None:
    config = ModelConfig(vocab_size=32, state_rank=3, state_dim=27)

    assert config.state_mode_sizes == (1, 3, 9)
    assert config.state_dim == 27


def test_config_can_derive_nonperfect_power_mode_sizes() -> None:
    config = ModelConfig(vocab_size=32, state_rank=3, state_dim=12)

    assert config.state_mode_sizes == (1, 3, 4)
    assert config.state_dim == 12


def test_config_tracks_max_and_init_mode_sizes() -> None:
    config = ModelConfig(
        vocab_size=32,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        max_mode_sizes=(3, 3, 3),
    )

    assert config.init_mode_sizes == (2, 2, 2)
    assert config.state_mode_sizes == (3, 3, 3)
    assert config.max_mode_sizes == (3, 3, 3)
    assert config.state_dim == 27


def test_config_pads_init_mode_sizes_for_dynamic_rank_growth() -> None:
    config = ModelConfig(
        vocab_size=32,
        state_rank=1,
        max_state_rank=3,
        dynamic_rank=True,
        max_mode_sizes=(2, 3, 4),
    )

    assert config.state_rank == 1
    assert config.max_state_rank == 3
    assert config.dynamic_rank is True
    assert config.init_mode_sizes == (2, 1, 1)
    assert config.max_mode_sizes == (2, 3, 4)
    assert config.state_mode_sizes == (2, 3, 4)
    assert config.state_dim == 24


def test_config_pads_rank8_max_mode_sizes_from_initial_rank_shorthand() -> None:
    config = ModelConfig(
        vocab_size=32,
        state_rank=4,
        max_state_rank=8,
        dynamic_rank=True,
        init_mode_sizes=(4, 4, 2, 2),
        max_mode_sizes=(8, 8, 4, 4),
    )

    assert config.init_mode_sizes == (4, 4, 2, 2, 1, 1, 1, 1)
    assert config.max_mode_sizes == (8, 8, 4, 4, 2, 2, 2, 2)
    assert config.state_mode_sizes == (8, 8, 4, 4, 2, 2, 2, 2)
    assert config.state_dim == 16384


def test_config_rejects_dynamic_rank_growth_into_singleton_future_mode() -> None:
    with pytest.raises(ValueError, match="at least 2 slots"):
        ModelConfig(
            vocab_size=32,
            state_rank=1,
            max_state_rank=2,
            dynamic_rank=True,
            max_mode_sizes=(4, 1),
        )


def test_config_rejects_learned_per_mode_scaling_without_per_mode_normalization() -> None:
    with pytest.raises(ValueError, match="learned_per_mode_scaling requires normalization='per_mode'"):
        ModelConfig(
            vocab_size=32,
            state_rank=3,
            state_mode_sizes=(3, 3, 3),
            normalization="frobenius",
            learned_per_mode_scaling=True,
        )


def test_config_allows_learned_per_mode_scaling_when_normalization_family_is_learned() -> None:
    config = ModelConfig(
        vocab_size=32,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        normalization="frobenius",
        learned_per_mode_scaling=True,
        learned_normalization_blend=True,
    )

    assert config.normalization == "frobenius"
    assert config.learned_per_mode_scaling is True
    assert config.learned_normalization_blend is True


def test_config_rejects_learnable_spectral_reciprocation_without_spectral_block() -> None:
    with pytest.raises(ValueError, match="learnable_spectral_reciprocation requires use_spectral_reciprocation=True"):
        ModelConfig(
            vocab_size=32,
            state_rank=3,
            state_mode_sizes=(3, 3, 3),
            use_spectral_reciprocation=False,
            learnable_spectral_reciprocation=True,
        )


def test_model_config_defaults_ffn_on_difference_when_loading_legacy_payload() -> None:
    config = ModelConfig(
        vocab_size=32,
        state_rank=2,
        state_mode_sizes=(2, 2),
    )
    payload = asdict(config)
    payload.pop("ffn_on_difference")

    restored = ModelConfig(**payload)

    assert restored.ffn_on_difference is True


@pytest.mark.parametrize(
    ("ffn_on_difference", "expected_source"),
    [
        (True, "delta"),
        (False, "post_mixer"),
    ],
)
def test_reciprocator_block_routes_ffn_input_from_config(
    ffn_on_difference: bool,
    expected_source: str,
) -> None:
    class IdentityModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class ConstantMixer(torch.nn.Module):
        def __init__(self, delta: torch.Tensor) -> None:
            super().__init__()
            self.delta = delta

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.delta

    class CaptureModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_input: Optional[torch.Tensor] = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.last_input = x.detach().clone()
            return x

    class ZeroModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        dim=8,
        n_layers=1,
        n_heads=4,
        state_rank=2,
        state_mode_sizes=(2, 2),
        num_cube_engines=1,
        dropout=0.0,
        ffn_on_difference=ffn_on_difference,
    )
    block = ReciprocatorOnlyBlock(config)
    x = torch.complex(torch.randn(2, 3, config.dim), torch.randn(2, 3, config.dim))
    delta = torch.complex(
        torch.full((2, 3, config.dim), 0.5),
        torch.full((2, 3, config.dim), -0.25),
    )
    capture_norm = CaptureModule()

    block.mixer_norm = IdentityModule()
    block.mixer = ConstantMixer(delta)
    block.ffn_norm = capture_norm
    block.ffn = ZeroModule()

    output = block(x)
    expected_hidden = x + delta
    expected_ffn_source = delta if expected_source == "delta" else expected_hidden

    assert capture_norm.last_input is not None
    torch.testing.assert_close(output, expected_hidden)
    torch.testing.assert_close(capture_norm.last_input, expected_ffn_source)


def test_forward_shapes_and_loss() -> None:
    torch.manual_seed(0)
    model = make_model()
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 8, 32)
    assert loss is not None
    assert loss.ndim == 0


def test_backward_pass_produces_gradients() -> None:
    torch.manual_seed(0)
    model = make_model()
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert model.blocks[0].mixer.engine_state_to_hidden[0].weight.grad is not None
    assert model.blocks[0].mixer.engine_fusion.weight.grad is not None
    assert model.blocks[0].mixer.gain_proj is not None
    assert model.blocks[0].mixer.gain_proj.weight.grad is not None
    assert model.blocks[0].mixer.gain_predictor is not None
    assert model.blocks[0].mixer.gain_predictor.signal_out.weight.grad is not None
    assert len(model.blocks[0].mixer.cube_engines) == 3
    for engine in model.blocks[0].mixer.cube_engines:
        assert engine.input_gain.grad is not None
        _assert_prediction_projector_has_grad(engine.prediction_proj)


def test_model_is_causal() -> None:
    torch.manual_seed(0)
    model = make_model().eval()
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
    altered = torch.tensor([[1, 2, 3, 9, 9, 9]])

    prefix_logits, _ = model(prefix)
    altered_logits, _ = model(altered)

    torch.testing.assert_close(prefix_logits[:, :3], altered_logits[:, :3])


def test_per_mode_normalization_reduces_fiber_norm_error_on_cube() -> None:
    torch.manual_seed(0)
    real = torch.randn(2, 3, 3, 3)
    imag = torch.randn(2, 3, 3, 3)

    def max_fiber_error(tensor_real: torch.Tensor, tensor_imag: torch.Tensor) -> torch.Tensor:
        magnitude_sq = tensor_real.square() + tensor_imag.square()
        errors = []
        for axis in range(1, 4):
            fiber_norms = torch.sqrt(magnitude_sq.sum(dim=axis, keepdim=True))
            errors.append((fiber_norms - 1.0).abs().amax())
        return torch.stack(errors).max()

    baseline_error = max_fiber_error(real, imag)
    norm_real, norm_imag = _normalize_complex_tensor(real, imag, mode="per_mode", state_rank=3)
    normalized_error = max_fiber_error(norm_real, norm_imag)

    assert normalized_error.item() < baseline_error.item()


def test_per_mode_normalization_zero_step_sizes_preserve_tensor() -> None:
    torch.manual_seed(0)
    real = torch.randn(1, 4, 3, 2)
    imag = torch.randn(1, 4, 3, 2)

    actual_real, actual_imag = _normalize_complex_tensor(
        real,
        imag,
        mode="per_mode",
        state_rank=3,
        step_sizes=torch.zeros(3),
    )

    torch.testing.assert_close(actual_real, real)
    torch.testing.assert_close(actual_imag, imag)


def test_per_mode_normalization_max_iter_one_matches_single_sweep() -> None:
    torch.manual_seed(0)
    real = torch.randn(1, 4, 3, 2)
    imag = torch.randn(1, 4, 3, 2)

    expected_real = real.clone()
    expected_imag = imag.clone()
    magnitude_sq = expected_real.square() + expected_imag.square()
    for axis in range(1, 4):
        axis_norm = torch.sqrt(magnitude_sq.sum(dim=axis, keepdim=True).clamp_min(1e-6))
        expected_real = expected_real / axis_norm
        expected_imag = expected_imag / axis_norm
        magnitude_sq = expected_real.square() + expected_imag.square()

    actual_real, actual_imag = _normalize_complex_tensor(
        real,
        imag,
        mode="per_mode",
        state_rank=3,
        max_iter=1,
        step_sizes=torch.ones(3),
    )

    torch.testing.assert_close(actual_real, expected_real)
    torch.testing.assert_close(actual_imag, expected_imag)


def make_reciprocator_only_model() -> ReciprocatorOnlyLM:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        num_cube_engines=3,
        normalization="per_mode",
        dropout=0.0,
    )
    return ReciprocatorOnlyLM(config)


def test_per_mode_learned_scaling_is_opt_in() -> None:
    default_model = make_reciprocator_only_model()
    assert default_model.blocks[0].mixer.per_mode_step_sizes is None

    learned_model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            dim=32,
            n_layers=2,
            n_heads=4,
            state_rank=3,
            state_mode_sizes=(3, 3, 3),
            num_cube_engines=3,
            normalization="per_mode",
            learned_per_mode_scaling=True,
            dropout=0.0,
        )
    )
    assert learned_model.blocks[0].mixer.per_mode_step_sizes is not None

    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))
    _, loss = learned_model(inputs, targets)
    assert loss is not None
    loss.backward()
    assert learned_model.blocks[0].mixer.per_mode_step_sizes.grad is not None


def test_learned_normalization_blend_is_opt_in() -> None:
    default_model = make_reciprocator_only_model()
    assert default_model.blocks[0].mixer.normalization_blend_predictor is None

    learned_model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            dim=32,
            n_layers=2,
            n_heads=4,
            state_rank=3,
            state_mode_sizes=(3, 3, 3),
            num_cube_engines=3,
            normalization="frobenius",
            learned_normalization_blend=True,
            dropout=0.0,
        )
    )
    mixer = learned_model.blocks[0].mixer
    assert mixer.normalization_blend_predictor is not None

    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))
    _, loss = learned_model(inputs, targets)
    assert loss is not None
    loss.backward()
    assert mixer.normalization_blend_predictor.summary_out.weight.grad is not None


def test_learned_normalization_blend_supports_dynamic_rank_with_inactive_future_modes() -> None:
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=64,
            max_seq_len=16,
            dim=8,
            n_layers=1,
            n_heads=1,
            mlp_ratio=2.0,
            state_rank=1,
            max_state_rank=6,
            dynamic_rank=True,
            init_mode_sizes=(2, 1, 1, 1, 1, 1),
            max_mode_sizes=(2, 2, 4, 4, 4, 4),
            num_cube_engines=1,
            normalization="per_mode",
            learned_normalization_blend=True,
            dropout=0.0,
        )
    )

    inputs = torch.randint(0, 64, (1, 16))
    targets = torch.randint(0, 64, (1, 16))
    logits, loss = model(inputs, targets)

    assert logits.shape == (1, 16, 64)
    assert loss is not None


def test_cube_engine_learnable_eta_and_temperature_receive_gradients() -> None:
    torch.manual_seed(0)
    active_sizes = (2, 2)
    cell = _CubeEngineCell(
        state_rank=2,
        max_mode_sizes=active_sizes,
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.1,
        prune_horizon=8,
        learnable_prediction_eta=True,
        learnable_coupling_temperature=True,
    )

    signal_real = torch.randn(2, *active_sizes)
    signal_imag = torch.randn(2, *active_sizes)
    state_real = torch.randn(2, *active_sizes)
    state_imag = torch.randn(2, *active_sizes)
    magnitude_accumulator = torch.rand(2, *active_sizes)
    carry_real = torch.randn(2, *active_sizes)
    carry_imag = torch.randn(2, *active_sizes)

    next_real, next_imag, _, _, _ = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=state_real,
        state_imag=state_imag,
        magnitude_accumulator=magnitude_accumulator,
        carry_real=carry_real,
        carry_imag=carry_imag,
        active_sizes=active_sizes,
        step_index=1,
        allow_growth=False,
    )
    loss = next_real.square().mean() + next_imag.square().mean()
    loss.backward()

    assert cell.prediction_eta_raw is not None
    assert cell.prediction_eta_raw.grad is not None
    assert cell.coupling_temperature_raw is not None
    assert cell.coupling_temperature_raw.grad is not None


def test_cube_engine_adaptive_growth_controls_adjust_effective_thresholds() -> None:
    cell = _CubeEngineCell(
        state_rank=2,
        max_mode_sizes=(2, 2),
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=0.2,
        growth_interval=1,
        prune_floor=0.1,
        prune_horizon=10,
        adaptive_growth_controls=True,
    )

    assert cell.growth_threshold == pytest.approx(0.2)
    assert cell.prune_floor == pytest.approx(0.1)
    assert cell.prune_horizon == 10

    cell._novelty_ema.fill_(0.4)
    cell._usage_ema.fill_(0.2)
    assert cell.growth_threshold == pytest.approx(0.4)
    assert cell.prune_floor == pytest.approx(0.2)
    assert cell.prune_horizon == 5

    cell._usage_ema.fill_(0.01)
    assert cell.prune_floor == pytest.approx(0.05)
    assert cell.prune_horizon == 20


def make_growth_sensitive_reciprocator_only_model() -> ReciprocatorOnlyLM:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        init_mode_sizes=(2, 2, 2),
        max_mode_sizes=(3, 3, 3),
        num_cube_engines=2,
        normalization="per_mode",
        growth_threshold=0.0,
        growth_interval=1,
        dropout=0.0,
        training_growth_enabled=True,
    )
    return ReciprocatorOnlyLM(config)


def test_reciprocator_only_forward_shapes_and_loss() -> None:
    torch.manual_seed(0)
    model = make_reciprocator_only_model()
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 8, 32)
    assert loss is not None
    assert loss.ndim == 0


@pytest.mark.parametrize(
    ("state_rank", "state_mode_sizes"),
    [
        (1, (8,)),
        (2, (2, 4)),
        (3, (2, 2, 2)),
    ],
)
def test_reciprocator_only_supports_multiple_state_ranks(
    state_rank: int,
    state_mode_sizes: tuple[int, ...],
) -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=state_rank,
        state_mode_sizes=state_mode_sizes,
        num_cube_engines=1,
        normalization="per_mode",
        dropout=0.0,
    )
    model = ReciprocatorOnlyLM(config)
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 8, 32)
    assert loss is not None
    assert loss.ndim == 0


def test_reciprocator_only_backward_pass() -> None:
    torch.manual_seed(0)
    model = make_reciprocator_only_model()
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()

    assert model.token_log_scale.weight.grad is not None
    assert model.blocks[0].mixer.signal_proj.weight_real.grad is not None
    for engine in model.blocks[0].mixer.cube_engines:
        assert engine.input_gain.grad is not None
        _assert_prediction_projector_has_grad(engine.prediction_proj)


def test_apply_complex_rope_matches_expected_complex_rotation() -> None:
    hidden = torch.ones(2, 4, 6, dtype=torch.complex64)
    positions = torch.arange(4)
    inverse_frequencies = complex_rope_frequencies(6)

    rotated = apply_complex_rope(hidden, positions, inverse_frequencies=inverse_frequencies)
    expected_angles = positions.to(dtype=torch.float32).unsqueeze(-1) * inverse_frequencies.unsqueeze(0)
    expected = torch.polar(torch.ones_like(expected_angles), expected_angles).unsqueeze(0).expand_as(rotated)

    torch.testing.assert_close(rotated, expected)
    torch.testing.assert_close(rotated.abs(), hidden.abs())


def test_reciprocator_only_parallel_selection_backward_pass() -> None:
    torch.manual_seed(0)
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            dim=32,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            state_mode_sizes=(2, 2, 2),
            num_cube_engines=2,
            normalization="frobenius",
            dropout=0.0,
            parallel_mixer=True,
            input_dependent_gains=True,
        )
    )
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()

    assert model.blocks[0].mixer.gain_proj is not None
    assert model.blocks[0].mixer.gain_predictor is not None
    assert model.blocks[0].mixer.gain_proj.weight.shape == (4 * 8, 3 + 3)
    assert model.blocks[0].mixer.gain_proj.weight.grad is not None
    assert model.blocks[0].mixer.gain_predictor.signal_out.weight.grad is not None
    for engine in model.blocks[0].mixer.cube_engines:
        _assert_prediction_projector_has_grad(engine.prediction_proj)


def test_reciprocator_only_serial_relational_selection_backward_pass() -> None:
    torch.manual_seed(0)
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            dim=32,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            state_mode_sizes=(2, 2, 2),
            num_cube_engines=2,
            normalization="frobenius",
            dropout=0.0,
            parallel_mixer=False,
            input_dependent_gains=True,
        )
    )
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()

    assert model.blocks[0].mixer.gain_proj is not None
    assert model.blocks[0].mixer.gain_proj.weight.grad is not None
    assert model.blocks[0].mixer.gain_predictor is not None
    assert model.blocks[0].mixer.gain_predictor.signal_out.weight.grad is not None


def test_input_dependent_gain_predictor_varies_with_signal() -> None:
    torch.manual_seed(0)
    mixer = ComplexReciprocatorMixer(
        hidden_dim=8,
        state_dim=8,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        init_mode_sizes=(2, 2, 2),
        num_cube_engines=1,
        normalization="frobenius",
        impression_rate=0.35,
        magnitude_floor=1e-3,
        dropout=0.0,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        persist_state=False,
        input_dependent_gains=True,
    )
    assert mixer.gain_predictor is not None

    signal_a_real = torch.zeros(1, 2, 2, 2)
    signal_a_imag = torch.zeros_like(signal_a_real)
    signal_b_real = torch.zeros_like(signal_a_real)
    signal_b_imag = torch.zeros_like(signal_a_real)
    signal_a_real[0, 0, 0, 0] = 1.0
    signal_b_imag[0, 0, 0, 0] = 1.0
    signal_a_real, signal_a_imag = _normalize_complex_tensor(
        signal_a_real,
        signal_a_imag,
        mode="frobenius",
        state_rank=3,
    )
    signal_b_real, signal_b_imag = _normalize_complex_tensor(
        signal_b_real,
        signal_b_imag,
        mode="frobenius",
        state_rank=3,
    )
    relational_stats = torch.zeros(1, mixer.state_rank + 3)

    gain_a = mixer.gain_predictor(signal_a_real, signal_a_imag, relational_stats)
    gain_b = mixer.gain_predictor(signal_b_real, signal_b_imag, relational_stats)

    assert (gain_a - gain_b).abs().amax().item() > 1e-8


def test_selective_gain_predictor_can_suppress_dynamic_modulation() -> None:
    torch.manual_seed(0)
    mixer = ComplexReciprocatorMixer(
        hidden_dim=8,
        state_dim=8,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        init_mode_sizes=(2, 2, 2),
        num_cube_engines=1,
        normalization="frobenius",
        impression_rate=0.35,
        magnitude_floor=1e-3,
        dropout=0.0,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        persist_state=False,
        input_dependent_gains=True,
        selective_gains=True,
    )
    assert mixer.gain_predictor is not None
    predictor = mixer.gain_predictor

    with torch.no_grad():
        predictor.signal_proj.weight.zero_()
        predictor.signal_proj.bias.fill_(1.0)
        predictor.signal_out.weight.fill_(0.5)
        predictor.signal_out.bias.zero_()
        predictor.context_proj.weight.zero_()
        predictor.selection_proj.weight.zero_()
        predictor.selection_proj.bias.zero_()
        predictor.selection_out.weight.zero_()
        predictor.selection_out.bias.fill_(-20.0)

    signal_real = torch.zeros(1, 2, 2, 2)
    signal_imag = torch.zeros_like(signal_real)
    signal_real[0, 0, 0, 0] = 1.0
    signal_real, signal_imag = _normalize_complex_tensor(
        signal_real,
        signal_imag,
        mode="frobenius",
        state_rank=3,
    )
    relational_stats = torch.zeros(1, mixer.state_rank + 3)

    suppressed_biases, suppressed_strength = predictor(
        signal_real,
        signal_imag,
        relational_stats,
        return_selection_strength=True,
    )
    assert suppressed_strength.item() < 1e-6
    assert suppressed_biases.abs().amax().item() < 1e-6

    with torch.no_grad():
        predictor.selection_out.bias.fill_(20.0)
    active_biases, active_strength = predictor(
        signal_real,
        signal_imag,
        relational_stats,
        return_selection_strength=True,
    )
    assert active_strength.item() > 1.0 - 1e-6
    assert active_biases.abs().amax().item() > 1e-2


@pytest.mark.parametrize("parallel_mixer", [False, True])
def test_selective_gains_backward_pass_produces_selection_gradients(parallel_mixer: bool) -> None:
    torch.manual_seed(0)
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            dim=32,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            state_mode_sizes=(2, 2, 2),
            num_cube_engines=2,
            normalization="frobenius",
            dropout=0.0,
            parallel_mixer=parallel_mixer,
            input_dependent_gains=True,
            selective_gains=True,
        )
    )
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()

    assert model.blocks[0].mixer.gain_predictor is not None
    assert model.blocks[0].mixer.gain_predictor.selection_out.weight.grad is not None
    assert model.blocks[0].mixer.gain_predictor.selection_proj.weight.grad is not None


def test_new_default_model_loads_static_checkpoint_without_gain_predictor_weights() -> None:
    base_config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        normalization="frobenius",
        dropout=0.0,
        input_dependent_gains=False,
    )
    static_model = ReciprocatorOnlyLM(base_config)
    static_state = static_model.state_dict()

    restored = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            dim=32,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            state_mode_sizes=(2, 2, 2),
            num_cube_engines=2,
            normalization="frobenius",
            dropout=0.0,
        )
    )
    restored.load_state_dict(static_state)

    assert restored.blocks[0].mixer.gain_predictor is not None
    signal_real = torch.zeros(1, 2, 2, 2)
    signal_imag = torch.zeros_like(signal_real)
    relational_stats = torch.zeros(1, restored.config.state_rank + 3)
    with torch.no_grad():
        gain_biases = restored.blocks[0].mixer.gain_predictor(signal_real, signal_imag, relational_stats)
    assert torch.count_nonzero(gain_biases) == 0


def test_per_mode_real_mixer_loads_legacy_checkpoint_without_step_sizes() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        normalization="per_mode",
        learned_per_mode_scaling=True,
        dropout=0.0,
    )
    model = ModifiedTransformerLM(config)
    legacy_state = model.state_dict()
    legacy_state.pop("blocks.0.mixer.per_mode_step_sizes")

    restored = ModifiedTransformerLM(config)
    restored.load_state_dict(legacy_state)

    assert restored.blocks[0].mixer.per_mode_step_sizes is not None
    torch.testing.assert_close(
        restored.blocks[0].mixer.per_mode_step_sizes,
        torch.ones_like(restored.blocks[0].mixer.per_mode_step_sizes),
    )


def test_new_default_real_model_loads_static_checkpoint_without_gain_predictor_weights() -> None:
    base_config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        normalization="frobenius",
        dropout=0.0,
        input_dependent_gains=False,
    )
    static_model = ModifiedTransformerLM(base_config)
    static_state = static_model.state_dict()

    restored = ModifiedTransformerLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            dim=32,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            state_mode_sizes=(2, 2, 2),
            num_cube_engines=2,
            normalization="frobenius",
            dropout=0.0,
        )
    )
    restored.load_state_dict(static_state)

    assert restored.blocks[0].mixer.gain_predictor is not None
    signal_real = torch.zeros(1, 2, 2, 2)
    signal_imag = torch.zeros_like(signal_real)
    relational_stats = torch.zeros(1, restored.config.state_rank + 3)
    with torch.no_grad():
        gain_biases = restored.blocks[0].mixer.gain_predictor(signal_real, signal_imag, relational_stats)
    assert torch.count_nonzero(gain_biases) == 0


def test_per_mode_complex_mixer_loads_legacy_checkpoint_without_step_sizes() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        normalization="per_mode",
        learned_per_mode_scaling=True,
        dropout=0.0,
    )
    model = ReciprocatorOnlyLM(config)
    legacy_state = model.state_dict()
    legacy_state.pop("blocks.0.mixer.per_mode_step_sizes")

    restored = ReciprocatorOnlyLM(config)
    restored.load_state_dict(legacy_state)

    assert restored.blocks[0].mixer.per_mode_step_sizes is not None
    torch.testing.assert_close(
        restored.blocks[0].mixer.per_mode_step_sizes,
        torch.ones_like(restored.blocks[0].mixer.per_mode_step_sizes),
    )


def test_legacy_gain_proj_state_dict_loads_into_context_projection() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        normalization="frobenius",
        dropout=0.0,
        input_dependent_gains=True,
    )
    model = ReciprocatorOnlyLM(config)
    legacy_state = model.state_dict()
    legacy_key = "blocks.0.mixer.gain_predictor.context_proj.weight"
    legacy_weight = legacy_state.pop(legacy_key).clone()
    legacy_state.pop("blocks.0.mixer.gain_predictor.signal_proj.weight")
    legacy_state.pop("blocks.0.mixer.gain_predictor.signal_proj.bias")
    legacy_state.pop("blocks.0.mixer.gain_predictor.signal_out.weight")
    legacy_state.pop("blocks.0.mixer.gain_predictor.signal_out.bias")
    legacy_state["blocks.0.mixer.gain_proj.weight"] = legacy_weight

    restored = ReciprocatorOnlyLM(config)
    restored.load_state_dict(legacy_state)

    assert restored.blocks[0].mixer.gain_proj is not None
    torch.testing.assert_close(restored.blocks[0].mixer.gain_proj.weight, legacy_weight)


def test_relational_gain_statistics_capture_phase_and_coupling_strength() -> None:
    active_sizes = (2, 2)
    aligned_real = torch.ones(1, 2, 2)
    aligned_imag = torch.zeros_like(aligned_real)
    quarter_turn_real = torch.zeros_like(aligned_real)
    quarter_turn_imag = torch.ones_like(aligned_real)
    weak_real = 0.25 * aligned_real
    weak_imag = torch.zeros_like(aligned_real)

    aligned_stats = _relational_gain_statistics(aligned_real, aligned_imag, active_sizes, state_rank=2)
    quarter_turn_stats = _relational_gain_statistics(quarter_turn_real, quarter_turn_imag, active_sizes, state_rank=2)
    weak_stats = _relational_gain_statistics(weak_real, weak_imag, active_sizes, state_rank=2)

    assert aligned_stats[0, 0].item() > quarter_turn_stats[0, 0].item()
    assert quarter_turn_stats[0, 1].item() > aligned_stats[0, 1].item()
    assert aligned_stats[0, 2].item() > weak_stats[0, 2].item()
    assert torch.all(aligned_stats[0, 3:] > weak_stats[0, 3:])


def test_cube_engine_masking() -> None:
    cell = _CubeEngineCell(
        state_rank=3,
        max_mode_sizes=(3, 3, 3),
        normalization="per_mode",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    active_sizes = (2, 2, 2)
    signal_real = torch.ones(1, 3, 3, 3)
    signal_imag = torch.ones(1, 3, 3, 3)
    state_real = torch.ones(1, 3, 3, 3)
    state_imag = torch.ones(1, 3, 3, 3)
    magnitude_accumulator = torch.zeros(1, 3, 3, 3)
    carry_real = torch.ones(1, 3, 3, 3)
    carry_imag = torch.ones(1, 3, 3, 3)

    next_real, next_imag, next_accumulator, new_carry_real, new_carry_imag = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=state_real,
        state_imag=state_imag,
        magnitude_accumulator=magnitude_accumulator,
        carry_real=carry_real,
        carry_imag=carry_imag,
        active_sizes=active_sizes,
        step_index=1,
        allow_growth=False,
    )

    active_mask = _mask_to_active(torch.ones_like(next_real), active_sizes, state_rank=3)
    inactive_mask = active_mask == 0
    torch.testing.assert_close(new_carry_real, next_real)
    torch.testing.assert_close(new_carry_imag, next_imag)
    assert torch.count_nonzero(next_real[inactive_mask]) == 0
    assert torch.count_nonzero(next_imag[inactive_mask]) == 0
    assert torch.count_nonzero(next_accumulator[inactive_mask]) == 0


def test_cube_engine_decay_bias_modulates_retention_mix() -> None:
    cell = _CubeEngineCell(
        state_rank=1,
        max_mode_sizes=(2,),
        normalization="per_mode",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    with torch.no_grad():
        cell.recurrent_gain.zero_()
        cell.carry_gain.zero_()

    active_sizes = (2,)
    signal_real = torch.zeros(1, 2)
    signal_imag = torch.ones(1, 2)
    state_real = torch.ones(1, 2)
    state_imag = torch.zeros(1, 2)
    zeros = torch.zeros_like(state_real)

    high_decay_real, high_decay_imag, _, _, _ = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=state_real,
        state_imag=state_imag,
        magnitude_accumulator=zeros,
        carry_real=zeros,
        carry_imag=zeros,
        active_sizes=active_sizes,
        step_index=1,
        decay_bias=torch.full_like(state_real, 4.0),
        allow_growth=False,
    )
    low_decay_real, low_decay_imag, _, _, _ = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=state_real,
        state_imag=state_imag,
        magnitude_accumulator=zeros,
        carry_real=zeros,
        carry_imag=zeros,
        active_sizes=active_sizes,
        step_index=1,
        decay_bias=torch.full_like(state_real, -4.0),
        allow_growth=False,
    )

    assert high_decay_real.mean().item() > low_decay_real.mean().item()
    assert high_decay_imag.mean().item() < low_decay_imag.mean().item()


def test_cube_engine_magnitude_accumulator_tracks_pre_normalized_strength() -> None:
    cell = _CubeEngineCell(
        state_rank=3,
        max_mode_sizes=(3, 3, 3),
        normalization="per_mode",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    active_sizes = (2, 2, 2)
    signal_real = torch.ones(1, 3, 3, 3)
    signal_imag = torch.zeros_like(signal_real)
    zeros = torch.zeros_like(signal_real)

    _, _, next_accumulator, _, _ = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=zeros,
        state_imag=zeros,
        magnitude_accumulator=zeros,
        carry_real=zeros,
        carry_imag=zeros,
        active_sizes=active_sizes,
        step_index=1,
        allow_growth=False,
    )

    expected_proposal_magnitude = _mask_to_active(
        (torch.sigmoid(cell.input_gain) + cell.prediction_eta).unsqueeze(0),
        active_sizes,
        state_rank=3,
    )
    expected_accumulator = (1.0 - cell.magnitude_decay) * expected_proposal_magnitude

    torch.testing.assert_close(next_accumulator, expected_accumulator)


def test_cube_engine_growth() -> None:
    torch.manual_seed(0)
    cell = _CubeEngineCell(
        state_rank=3,
        max_mode_sizes=(3, 3, 3),
        normalization="per_mode",
        impression_rate=0.35,
        growth_threshold=0.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    active_sizes = (1, 1, 1)
    state_real = torch.zeros(1, 3, 3, 3)
    state_imag = torch.zeros(1, 3, 3, 3)

    for step in range(1, 6):
        signal_real = torch.randn(1, 3, 3, 3)
        signal_imag = torch.randn(1, 3, 3, 3)
        state_real, state_imag, _, _, _, active_sizes, _ = cell.step(
            signal_real=signal_real,
            signal_imag=signal_imag,
            state_real=state_real,
            state_imag=state_imag,
            magnitude_accumulator=torch.zeros_like(state_real),
            carry_real=torch.zeros_like(state_real),
            carry_imag=torch.zeros_like(state_imag),
            active_sizes=active_sizes,
            step_index=step,
            allow_growth=True,
            return_active_sizes=True,
        )

    assert sum(active_sizes) > 3


def test_cube_engine_dynamic_rank_growth_increases_active_rank() -> None:
    torch.manual_seed(0)
    cell = _CubeEngineCell(
        state_rank=3,
        initial_state_rank=1,
        max_mode_sizes=(2, 2, 2),
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=0.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        dynamic_rank=True,
    )
    active_sizes = (2, 1, 1)
    zeros = torch.zeros(1, 2, 2, 2)

    _, _, _, _, _, grown_sizes, grown_rank = cell.step(
        signal_real=torch.randn(1, 2, 2, 2),
        signal_imag=torch.randn(1, 2, 2, 2),
        state_real=zeros,
        state_imag=zeros,
        magnitude_accumulator=zeros,
        carry_real=zeros,
        carry_imag=zeros,
        active_sizes=active_sizes,
        active_rank=1,
        step_index=1,
        allow_growth=True,
        return_active_sizes=True,
    )

    diagnostics = cell.diagnostics(grown_sizes, grown_rank)
    assert grown_rank == 2
    assert grown_sizes == (2, 2, 1)
    assert diagnostics["active_rank"] == 2
    assert diagnostics["last_growth_mode"] == 1
    assert diagnostics["growth_event_count"] == 1


def test_cube_engine_prediction_error_term_updates_state() -> None:
    cell = _CubeEngineCell(
        state_rank=1,
        max_mode_sizes=(2,),
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        prediction_eta=1.0,
    )
    active_sizes = (2,)

    with torch.no_grad():
        cell.input_gain.fill_(-30.0)
        cell.recurrent_gain.zero_()
        cell.carry_gain.zero_()
        cell.decay.fill_(-30.0)
        cell.prediction_proj.set_identity_()

    signal_real = torch.tensor([[1.0, 0.0]])
    signal_imag = torch.zeros_like(signal_real)
    state_real = torch.tensor([[0.0, 1.0]])
    state_imag = torch.zeros_like(state_real)
    zeros = torch.zeros_like(state_real)

    next_real, next_imag, next_accumulator, _, _ = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=state_real,
        state_imag=state_imag,
        magnitude_accumulator=zeros,
        carry_real=zeros,
        carry_imag=zeros,
        active_sizes=active_sizes,
        step_index=1,
        allow_growth=False,
    )

    expected = torch.tensor([[1.0, -1.0]]) / math.sqrt(2.0)
    expected_accumulator = (1.0 - cell.magnitude_decay) * torch.ones_like(next_accumulator)

    torch.testing.assert_close(next_real, expected, atol=1e-4, rtol=0.0)
    torch.testing.assert_close(next_imag, torch.zeros_like(next_imag), atol=1e-4, rtol=0.0)
    torch.testing.assert_close(next_accumulator, expected_accumulator, atol=1e-4, rtol=0.0)


def test_cube_engine_growth_uses_prediction_error_signal() -> None:
    cell = _CubeEngineCell(
        state_rank=1,
        max_mode_sizes=(2,),
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=0.5,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        prediction_eta=1.0,
    )
    active_sizes = (1,)

    with torch.no_grad():
        cell.input_gain.fill_(-30.0)
        cell.recurrent_gain.zero_()
        cell.carry_gain.zero_()
        cell.decay.fill_(-30.0)
        cell.prediction_proj.set_identity_()

    state_real = torch.tensor([[1.0, 0.0]])
    state_imag = torch.zeros_like(state_real)
    signal_real = torch.tensor([[1.0, 0.0]])
    signal_imag = torch.zeros_like(signal_real)
    zeros = torch.zeros_like(state_real)

    _, _, _, _, _, unchanged_sizes, unchanged_rank = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=state_real,
        state_imag=state_imag,
        magnitude_accumulator=zeros,
        carry_real=zeros,
        carry_imag=zeros,
        active_sizes=active_sizes,
        step_index=1,
        allow_growth=True,
        return_active_sizes=True,
    )
    assert unchanged_sizes == (1,)
    assert unchanged_rank == 1
    assert cell.diagnostics(unchanged_sizes, unchanged_rank)["last_prediction_error"] == pytest.approx(0.0, abs=1e-6)

    signal_real = torch.tensor([[-1.0, 0.0]])
    _, _, _, _, _, grown_sizes, grown_rank = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=state_real,
        state_imag=state_imag,
        magnitude_accumulator=zeros,
        carry_real=zeros,
        carry_imag=zeros,
        active_sizes=active_sizes,
        step_index=2,
        allow_growth=True,
        return_active_sizes=True,
    )

    assert grown_sizes == (2,)
    assert grown_rank == 1
    assert cell.diagnostics(grown_sizes, grown_rank)["last_prediction_error"] > cell.growth_threshold


def test_cube_engine_gradient_flows_through_active_region() -> None:
    torch.manual_seed(0)
    cell = _CubeEngineCell(
        state_rank=3,
        max_mode_sizes=(3, 3, 3),
        normalization="per_mode",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    active_sizes = (2, 2, 2)
    signal_real = torch.randn(1, 3, 3, 3)
    signal_imag = torch.randn(1, 3, 3, 3)
    state_real = torch.randn(1, 3, 3, 3)
    state_imag = torch.randn(1, 3, 3, 3)
    with torch.no_grad():
        cell.prediction_state_mix.fill_(1.0)

    next_real, next_imag, _, _, _ = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=state_real,
        state_imag=state_imag,
        magnitude_accumulator=torch.zeros_like(state_real),
        carry_real=torch.zeros_like(state_real),
        carry_imag=torch.zeros_like(state_imag),
        active_sizes=active_sizes,
        step_index=1,
        allow_growth=False,
    )
    (next_real.sum() + next_imag.sum()).backward()

    active_mask = _mask_to_active(torch.ones_like(cell.input_gain), active_sizes, state_rank=3).bool()
    grad = cell.input_gain.grad
    assert grad is not None
    assert grad[active_mask].abs().sum().item() > 0.0
    assert torch.count_nonzero(grad[~active_mask]) == 0
    coupling_real_grad = cell.cpl_proj_real[0].grad
    coupling_imag_grad = cell.cpl_proj_imag[0].grad
    assert coupling_real_grad is not None
    assert coupling_imag_grad is not None
    active_mode = active_sizes[0]
    assert coupling_real_grad[:active_mode, :active_mode].abs().sum().item() > 0.0
    assert coupling_imag_grad[:active_mode, :active_mode].abs().sum().item() > 0.0
    assert torch.count_nonzero(coupling_real_grad[active_mode:, :]) == 0
    assert torch.count_nonzero(coupling_real_grad[:, active_mode:]) == 0
    assert torch.count_nonzero(coupling_imag_grad[active_mode:, :]) == 0
    assert torch.count_nonzero(coupling_imag_grad[:, active_mode:]) == 0
    for mode_idx, active_mode in enumerate(active_sizes):
        prediction_real_grad = cell.prediction_proj.weight_real[mode_idx].grad
        prediction_imag_grad = cell.prediction_proj.weight_imag[mode_idx].grad
        assert prediction_real_grad is not None
        assert prediction_imag_grad is not None
        assert torch.count_nonzero(prediction_real_grad[active_mode:, :]) == 0
        assert torch.count_nonzero(prediction_real_grad[:, active_mode:]) == 0
        assert torch.count_nonzero(prediction_imag_grad[active_mode:, :]) == 0
        assert torch.count_nonzero(prediction_imag_grad[:, active_mode:]) == 0


def test_cube_engine_phase_aware_couplings_change_with_phase() -> None:
    cell = _CubeEngineCell(
        state_rank=2,
        max_mode_sizes=(2, 2),
        normalization="per_mode",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    active_sizes = (2, 2)
    in_phase_real = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    in_phase_imag = torch.zeros_like(in_phase_real)
    quarter_turn_real = torch.zeros_like(in_phase_real)
    quarter_turn_imag = in_phase_real.clone()

    with torch.no_grad():
        for weight in cell.cpl_proj_real:
            weight.zero_()
        for weight in cell.cpl_proj_imag:
            weight.zero_()
        cell.cpl_proj_imag[0][0, 0] = 4.0
        cell.cpl_proj_imag[0][0, 1] = -4.0

    in_phase_couplings = cell._phase_aware_mode_couplings(in_phase_real, in_phase_imag, active_sizes)
    quarter_turn_couplings = cell._phase_aware_mode_couplings(
        quarter_turn_real,
        quarter_turn_imag,
        active_sizes,
    )

    assert not torch.allclose(in_phase_couplings[0], quarter_turn_couplings[0])
    assert torch.is_complex(in_phase_couplings[0])
    torch.testing.assert_close(
        in_phase_couplings[0].abs().sum(dim=-1),
        torch.ones_like(in_phase_couplings[0].abs().sum(dim=-1)),
    )
    torch.testing.assert_close(
        quarter_turn_couplings[0].abs().sum(dim=-1),
        torch.ones_like(quarter_turn_couplings[0].abs().sum(dim=-1)),
    )

    mixed_in_phase_real, mixed_in_phase_imag = _apply_mode_couplings_pair(
        in_phase_real,
        torch.zeros_like(in_phase_real),
        in_phase_couplings,
        state_rank=2,
    )
    mixed_quarter_turn_real, mixed_quarter_turn_imag = _apply_mode_couplings_pair(
        in_phase_real,
        torch.zeros_like(in_phase_real),
        quarter_turn_couplings,
        state_rank=2,
    )
    assert not torch.allclose(
        torch.complex(mixed_in_phase_real, mixed_in_phase_imag),
        torch.complex(mixed_quarter_turn_real, mixed_quarter_turn_imag),
    )


def test_phase_aware_coupling_false_matches_legacy_real_softmax() -> None:
    cell = _CubeEngineCell(
        state_rank=2,
        max_mode_sizes=(2, 2),
        normalization="per_mode",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        phase_aware_coupling=False,
    )
    active_sizes = (2, 2)
    local_real = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    local_imag = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])

    couplings = cell._phase_aware_mode_couplings(local_real, local_imag, active_sizes)

    moved_real = torch.movedim(local_real, 1, 1)
    moved_imag = torch.movedim(local_imag, 1, 1)
    flat_real = moved_real.reshape(1, 2, -1)
    flat_imag = moved_imag.reshape(1, 2, -1)
    logits_real, logits_imag, scale = cell._mode_score_logits(flat_real, flat_imag, mode_idx=0)
    expected = torch.softmax((logits_real + logits_imag) / scale, dim=-1)

    assert not torch.is_complex(couplings[0])
    torch.testing.assert_close(couplings[0], expected)


def test_cube_engine_expressive_couplings_are_nonconvex_and_phase_aware() -> None:
    cell = _CubeEngineCell(
        state_rank=2,
        max_mode_sizes=(2, 2),
        normalization="per_mode",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    active_sizes = (2, 2)
    in_phase_real = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    in_phase_imag = torch.zeros_like(in_phase_real)
    quarter_turn_real = torch.zeros_like(in_phase_real)
    quarter_turn_imag = in_phase_real.clone()

    with torch.no_grad():
        for weight in cell.cpl_proj_real:
            weight.zero_()
        for weight in cell.cpl_proj_imag:
            weight.zero_()
        cell.cpl_proj_real[0][0, 0] = -2.0
        cell.cpl_proj_real[1].copy_(torch.eye(2))
        cell.cpl_proj_imag[0][0, 0] = 1.5

    couplings, _ = cell._expressive_mode_couplings(in_phase_real, in_phase_imag, active_sizes)

    mixed_in_phase_real, mixed_in_phase_imag = cell._apply_expressive_mode_couplings(
        in_phase_real,
        in_phase_imag,
        active_sizes,
    )
    mixed_quarter_turn_real, mixed_quarter_turn_imag = cell._apply_expressive_mode_couplings(
        quarter_turn_real,
        quarter_turn_imag,
        active_sizes,
    )

    assert torch.is_complex(couplings[0])
    torch.testing.assert_close(
        couplings[0].abs().sum(dim=-1),
        torch.ones_like(couplings[0].abs().sum(dim=-1)),
    )
    assert not torch.allclose(mixed_in_phase_real, mixed_quarter_turn_real)
    assert not torch.allclose(mixed_in_phase_imag, mixed_quarter_turn_imag)


@pytest.mark.parametrize(
    ("state_rank", "state_mode_sizes"),
    [
        (1, (4,)),
        (2, (2, 3)),
        (4, (2, 2, 2, 2)),
    ],
)
def test_phase_aware_mode_couplings_support_multiple_state_ranks(
    state_rank: int,
    state_mode_sizes: tuple[int, ...],
) -> None:
    torch.manual_seed(0)
    cell = _CubeEngineCell(
        state_rank=state_rank,
        max_mode_sizes=state_mode_sizes,
        normalization="per_mode",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    local_real = torch.randn(1, *state_mode_sizes)
    local_imag = torch.randn(1, *state_mode_sizes)

    couplings = cell._phase_aware_mode_couplings(local_real, local_imag, state_mode_sizes)

    assert len(couplings) == state_rank
    for coupling, mode_size in zip(couplings, state_mode_sizes):
        assert coupling.shape == (1, mode_size, mode_size)
        assert torch.isfinite(coupling).all()


@pytest.mark.parametrize(
    ("state_rank", "state_mode_sizes"),
    [
        (1, (4,)),
        (2, (2, 3)),
        (4, (2, 2, 2, 2)),
    ],
)
def test_expressive_mode_couplings_support_multiple_state_ranks(
    state_rank: int,
    state_mode_sizes: tuple[int, ...],
) -> None:
    torch.manual_seed(0)
    cell = _CubeEngineCell(
        state_rank=state_rank,
        max_mode_sizes=state_mode_sizes,
        normalization="per_mode",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    local_real = torch.randn(1, *state_mode_sizes)
    local_imag = torch.randn(1, *state_mode_sizes)

    couplings, (coupled_real, coupled_imag) = cell._expressive_mode_couplings(
        local_real,
        local_imag,
        state_mode_sizes,
    )

    assert len(couplings) == state_rank
    for coupling, mode_size in zip(couplings, state_mode_sizes):
        assert coupling.shape == (1, mode_size, mode_size)
        assert torch.isfinite(coupling).all()
    assert coupled_real.shape == local_real.shape
    assert coupled_imag.shape == local_imag.shape
    assert torch.isfinite(coupled_real).all()
    assert torch.isfinite(coupled_imag).all()


def test_cube_engine_matches_legacy_update_when_accumulator_modulation_disabled() -> None:
    torch.manual_seed(0)
    active_sizes = (2, 2)
    cell = _CubeEngineCell(
        state_rank=2,
        max_mode_sizes=active_sizes,
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        prediction_eta=0.2,
        accumulator_modulates_gains=False,
    )

    signal_real = torch.randn(2, *active_sizes)
    signal_imag = torch.randn(2, *active_sizes)
    state_real = torch.randn(2, *active_sizes)
    state_imag = torch.randn(2, *active_sizes)
    magnitude_accumulator = torch.rand(2, *active_sizes)
    carry_real = torch.randn(2, *active_sizes)
    carry_imag = torch.randn(2, *active_sizes)
    decay_bias = torch.randn(2, *active_sizes)
    input_gain_bias = torch.randn(2, *active_sizes)
    recurrent_gain_bias = torch.randn(2, *active_sizes)
    carry_gain_bias = torch.randn(2, *active_sizes)

    actual = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=state_real,
        state_imag=state_imag,
        magnitude_accumulator=magnitude_accumulator,
        carry_real=carry_real,
        carry_imag=carry_imag,
        active_sizes=active_sizes,
        step_index=1,
        decay_bias=decay_bias,
        input_gain_bias=input_gain_bias,
        recurrent_gain_bias=recurrent_gain_bias,
        carry_gain_bias=carry_gain_bias,
        allow_growth=False,
    )

    signal_real = _mask_to_active(signal_real, active_sizes, cell.state_rank)
    signal_imag = _mask_to_active(signal_imag, active_sizes, cell.state_rank)
    state_real = _mask_to_active(state_real, active_sizes, cell.state_rank)
    state_imag = _mask_to_active(state_imag, active_sizes, cell.state_rank)
    magnitude_accumulator = _mask_to_active(magnitude_accumulator, active_sizes, cell.state_rank)
    carry_real = _mask_to_active(carry_real, active_sizes, cell.state_rank)
    carry_imag = _mask_to_active(carry_imag, active_sizes, cell.state_rank)

    local_real, local_imag = _relational_product(
        signal_real,
        signal_imag,
        state_real,
        state_imag,
        active_sizes,
        cell.state_rank,
    )
    if cell.use_expressive_mode_couplings:
        coupled_real, coupled_imag = cell._apply_expressive_mode_couplings(
            local_real,
            local_imag,
            active_sizes,
        )
    else:
        mode_couplings = cell._phase_aware_mode_couplings(local_real, local_imag, active_sizes)
        coupled_real, coupled_imag = _apply_mode_couplings_pair(
            local_real,
            local_imag,
            mode_couplings,
            cell.state_rank,
        )
    predicted_signal_real, predicted_signal_imag = cell._predict_signal(
        state_real,
        state_imag,
        magnitude_accumulator,
        active_sizes,
        cell.state_rank,
    )
    prediction_error_real = signal_real - predicted_signal_real
    prediction_error_imag = signal_imag - predicted_signal_imag

    decay_logits = cell._active_parameter(cell.decay, active_sizes, lambda value: value).unsqueeze(0)
    input_gain_logits = cell._active_parameter(cell.input_gain, active_sizes, lambda value: value).unsqueeze(0)
    recurrent_gain_logits = cell._active_parameter(cell.recurrent_gain, active_sizes, lambda value: value).unsqueeze(0)
    carry_gain_logits = cell._active_parameter(cell.carry_gain, active_sizes, lambda value: value).unsqueeze(0)
    decay_logits = decay_logits + _mask_to_active(decay_bias, active_sizes, cell.state_rank)
    input_gain_logits = input_gain_logits + _mask_to_active(input_gain_bias, active_sizes, cell.state_rank)
    recurrent_gain_logits = recurrent_gain_logits + _mask_to_active(
        recurrent_gain_bias,
        active_sizes,
        cell.state_rank,
    )
    carry_gain_logits = carry_gain_logits + _mask_to_active(carry_gain_bias, active_sizes, cell.state_rank)

    decay = torch.sigmoid(decay_logits)
    input_gain = torch.sigmoid(input_gain_logits)
    recurrent_gain = torch.tanh(recurrent_gain_logits)
    carry_gain = torch.tanh(carry_gain_logits)
    proposal_real = (
        decay * state_real
        + input_gain * signal_real
        + recurrent_gain * coupled_real
        + carry_gain * carry_real
        + cell.prediction_eta * prediction_error_real
    )
    proposal_imag = (
        decay * state_imag
        + input_gain * signal_imag
        + recurrent_gain * coupled_imag
        + carry_gain * carry_imag
        + cell.prediction_eta * prediction_error_imag
    )
    proposal_real = _mask_to_active(proposal_real, active_sizes, cell.state_rank)
    proposal_imag = _mask_to_active(proposal_imag, active_sizes, cell.state_rank)
    proposal_magnitude = _mask_to_active(
        torch.sqrt(proposal_real.square() + proposal_imag.square() + 1e-6),
        active_sizes,
        cell.state_rank,
    )
    expected_accumulator = _mask_to_active(
        cell.magnitude_decay * magnitude_accumulator + (1.0 - cell.magnitude_decay) * proposal_magnitude,
        active_sizes,
        cell.state_rank,
    )
    expected_real, expected_imag = _normalize_complex_tensor(
        proposal_real,
        proposal_imag,
        cell.normalization,
        state_rank=cell.state_rank,
    )
    expected_real = _mask_to_active(expected_real, active_sizes, cell.state_rank)
    expected_imag = _mask_to_active(expected_imag, active_sizes, cell.state_rank)

    expected = (
        expected_real,
        expected_imag,
        expected_accumulator,
        expected_real,
        expected_imag,
    )
    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(actual_part, expected_part)


def test_cube_engine_spectral_reciprocation_matches_manual_fft_filter() -> None:
    torch.manual_seed(0)
    active_sizes = (2, 2)
    cell = _CubeEngineCell(
        state_rank=2,
        max_mode_sizes=active_sizes,
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        prediction_eta=0.2,
        accumulator_modulates_gains=False,
        use_spectral_reciprocation=True,
        spectral_mode="fft",
        spectral_low_frequency_gain=0.4,
        spectral_low_frequency_sigma=0.15,
        spectral_high_frequency_gain=0.3,
        spectral_high_frequency_cutoff=0.1,
    )

    signal_real = torch.randn(2, *active_sizes)
    signal_imag = torch.randn(2, *active_sizes)
    state_real = torch.randn(2, *active_sizes)
    state_imag = torch.randn(2, *active_sizes)
    magnitude_accumulator = torch.rand(2, *active_sizes)
    carry_real = torch.randn(2, *active_sizes)
    carry_imag = torch.randn(2, *active_sizes)

    actual = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=state_real,
        state_imag=state_imag,
        magnitude_accumulator=magnitude_accumulator,
        carry_real=carry_real,
        carry_imag=carry_imag,
        active_sizes=active_sizes,
        step_index=1,
        allow_growth=False,
    )

    signal_real = _mask_to_active(signal_real, active_sizes, cell.state_rank)
    signal_imag = _mask_to_active(signal_imag, active_sizes, cell.state_rank)
    state_real = _mask_to_active(state_real, active_sizes, cell.state_rank)
    state_imag = _mask_to_active(state_imag, active_sizes, cell.state_rank)
    magnitude_accumulator = _mask_to_active(magnitude_accumulator, active_sizes, cell.state_rank)
    carry_real = _mask_to_active(carry_real, active_sizes, cell.state_rank)
    carry_imag = _mask_to_active(carry_imag, active_sizes, cell.state_rank)

    local_real, local_imag = _relational_product(
        signal_real,
        signal_imag,
        state_real,
        state_imag,
        active_sizes,
        cell.state_rank,
    )
    if cell.use_expressive_mode_couplings:
        coupled_real, coupled_imag = cell._apply_expressive_mode_couplings(
            local_real,
            local_imag,
            active_sizes,
        )
    else:
        mode_couplings = cell._phase_aware_mode_couplings(local_real, local_imag, active_sizes)
        coupled_real, coupled_imag = _apply_mode_couplings_pair(
            local_real,
            local_imag,
            mode_couplings,
            cell.state_rank,
        )
    predicted_signal_real, predicted_signal_imag = cell._predict_signal(
        state_real,
        state_imag,
        magnitude_accumulator,
        active_sizes,
        cell.state_rank,
    )
    prediction_error_real = signal_real - predicted_signal_real
    prediction_error_imag = signal_imag - predicted_signal_imag

    proposal_real = (
        torch.sigmoid(cell.decay).unsqueeze(0) * state_real
        + torch.sigmoid(cell.input_gain).unsqueeze(0) * signal_real
        + torch.tanh(cell.recurrent_gain).unsqueeze(0) * coupled_real
        + torch.tanh(cell.carry_gain).unsqueeze(0) * carry_real
        + cell.prediction_eta * prediction_error_real
    )
    proposal_imag = (
        torch.sigmoid(cell.decay).unsqueeze(0) * state_imag
        + torch.sigmoid(cell.input_gain).unsqueeze(0) * signal_imag
        + torch.tanh(cell.recurrent_gain).unsqueeze(0) * coupled_imag
        + torch.tanh(cell.carry_gain).unsqueeze(0) * carry_imag
        + cell.prediction_eta * prediction_error_imag
    )
    proposal_real = _mask_to_active(proposal_real, active_sizes, cell.state_rank)
    proposal_imag = _mask_to_active(proposal_imag, active_sizes, cell.state_rank)
    proposal_magnitude = _mask_to_active(
        torch.sqrt(proposal_real.square() + proposal_imag.square() + 1e-6),
        active_sizes,
        cell.state_rank,
    )
    expected_accumulator = _mask_to_active(
        cell.magnitude_decay * magnitude_accumulator + (1.0 - cell.magnitude_decay) * proposal_magnitude,
        active_sizes,
        cell.state_rank,
    )
    expected_real, expected_imag = _normalize_complex_tensor(
        proposal_real,
        proposal_imag,
        cell.normalization,
        state_rank=cell.state_rank,
    )
    expected_state = torch.complex(expected_real, expected_imag)
    freq = torch.fft.fftn(expected_state, dim=(1, 2))
    freq_axes = [torch.fft.fftfreq(size).to(dtype=expected_real.dtype) for size in active_sizes]
    radius_squared = torch.zeros(active_sizes, dtype=expected_real.dtype)
    for axis in torch.meshgrid(*freq_axes, indexing="ij"):
        radius_squared = radius_squared + axis.square()
    spectral_filter = 1.0 + cell.spectral_low_frequency_gain * torch.exp(
        -radius_squared / (cell.spectral_low_frequency_sigma**2)
    )
    radius = torch.sqrt(radius_squared + 1e-12)
    smoothing_width = expected_real.new_tensor(cell.spectral_low_frequency_sigma).clamp_min(5e-2)
    low_band_gate = torch.sigmoid((cell.spectral_high_frequency_cutoff - radius) / smoothing_width)
    spectral_filter = spectral_filter * (
        cell.spectral_high_frequency_gain + (1.0 - cell.spectral_high_frequency_gain) * low_band_gate
    )
    expected_state = torch.fft.ifftn(freq * spectral_filter.unsqueeze(0), dim=(1, 2))
    expected_real, expected_imag = _normalize_complex_tensor(
        expected_state.real,
        expected_state.imag,
        cell.normalization,
        state_rank=cell.state_rank,
    )
    expected_real = _mask_to_active(expected_real, active_sizes, cell.state_rank)
    expected_imag = _mask_to_active(expected_imag, active_sizes, cell.state_rank)

    expected = (
        expected_real,
        expected_imag,
        expected_accumulator,
        expected_real,
        expected_imag,
    )
    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(actual_part, expected_part)
    frobenius_norm = torch.sqrt((actual[0].square() + actual[1].square()).sum(dim=(1, 2)))
    torch.testing.assert_close(frobenius_norm, torch.ones_like(frobenius_norm))


def test_cube_engine_learnable_spectral_params_receive_gradients() -> None:
    torch.manual_seed(0)
    active_sizes = (2, 2)
    cell = _CubeEngineCell(
        state_rank=2,
        max_mode_sizes=active_sizes,
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        use_spectral_reciprocation=True,
        learnable_spectral_reciprocation=True,
        spectral_mode="wavelet_packet_max_gauge",
    )

    signal_real = torch.randn(2, *active_sizes)
    signal_imag = torch.randn(2, *active_sizes)
    state_real = torch.randn(2, *active_sizes)
    state_imag = torch.randn(2, *active_sizes)
    magnitude_accumulator = torch.rand(2, *active_sizes)
    carry_real = torch.randn(2, *active_sizes)
    carry_imag = torch.randn(2, *active_sizes)

    next_real, next_imag, _, _, _ = cell.step(
        signal_real=signal_real,
        signal_imag=signal_imag,
        state_real=state_real,
        state_imag=state_imag,
        magnitude_accumulator=magnitude_accumulator,
        carry_real=carry_real,
        carry_imag=carry_imag,
        active_sizes=active_sizes,
        step_index=1,
        allow_growth=False,
    )
    loss = next_real.square().mean() + next_imag.square().mean()
    loss.backward()

    assert cell.spectral_low_frequency_gain_raw is not None
    assert cell.spectral_low_frequency_gain_raw.grad is not None
    assert cell.spectral_low_frequency_sigma_raw is not None
    assert cell.spectral_low_frequency_sigma_raw.grad is not None
    assert cell.spectral_high_frequency_gain_raw is not None
    assert cell.spectral_high_frequency_gain_raw.grad is not None
    assert cell.spectral_high_frequency_cutoff_raw is not None
    assert cell.spectral_high_frequency_cutoff_raw.grad is not None


def test_cube_engine_relational_prediction_branch_uses_phase_aware_self_transport() -> None:
    cell = _CubeEngineCell(
        state_rank=2,
        max_mode_sizes=(2, 2),
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    active_sizes = (2, 2)
    state_real = torch.tensor([[[1.0, 0.0], [0.5, -0.5]]])
    state_imag = torch.tensor([[[0.0, 1.0], [-0.5, 0.25]]])
    magnitude_accumulator = torch.full_like(state_real, 0.25)

    with torch.no_grad():
        cell.prediction_proj.zero_()
        cell.prediction_state_mix.zero_()
        cell.prediction_coupling_mix.fill_(1.25)

    predicted_signal_real, predicted_signal_imag = cell._predict_signal(
        state_real,
        state_imag,
        magnitude_accumulator,
        active_sizes,
        active_rank=2,
    )
    coupled_real, coupled_imag = cell._apply_expressive_mode_couplings(
        state_real,
        state_imag,
        active_sizes,
        active_rank=2,
    )
    prediction_multiplier = cell._accumulator_multiplier(
        magnitude_accumulator,
        cell.prediction_accumulator_scale,
    )
    expected_scale = torch.tanh(cell.prediction_coupling_mix)
    expected_real = prediction_multiplier * expected_scale * coupled_real
    expected_imag = prediction_multiplier * expected_scale * coupled_imag

    torch.testing.assert_close(predicted_signal_real, expected_real, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(predicted_signal_imag, expected_imag, atol=1e-5, rtol=0.0)
    assert predicted_signal_real.abs().sum().item() + predicted_signal_imag.abs().sum().item() > 0.0


def test_cube_engine_loads_legacy_state_without_relational_prediction_weights() -> None:
    cell = _CubeEngineCell(
        state_rank=2,
        max_mode_sizes=(2, 2),
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    legacy_state = cell.state_dict()
    legacy_state.pop("prediction_state_mix")
    legacy_state.pop("prediction_coupling_mix")
    legacy_state.pop("prediction_accumulator_scale")

    restored = _CubeEngineCell(
        state_rank=2,
        max_mode_sizes=(2, 2),
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
    )
    restored.load_state_dict(legacy_state, strict=True)

    assert restored.prediction_state_mix.item() == pytest.approx(0.0)
    assert restored.prediction_coupling_mix.item() == pytest.approx(0.0)
    assert restored.prediction_accumulator_scale.item() == pytest.approx(-3.0)


def test_cube_engine_accumulator_strengthens_recurrent_update() -> None:
    cell = _CubeEngineCell(
        state_rank=1,
        max_mode_sizes=(1,),
        normalization="frobenius",
        impression_rate=0.35,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        prediction_eta=0.0,
        accumulator_modulates_gains=True,
    )
    cell.use_expressive_mode_couplings = False
    cell.use_static_mode_couplings = True
    with torch.no_grad():
        cell.input_gain.fill_(-20.0)
        cell.recurrent_gain.fill_(0.5)
        cell.carry_gain.zero_()
        cell.decay.fill_(-20.0)
        cell.accumulator_input_gain_scale.fill_(-20.0)
        cell.accumulator_carry_gain_scale.fill_(-20.0)
        cell.accumulator_recurrent_gain_scale.fill_(0.0)
        cell.accumulator_coupling_scale.fill_(0.0)
        cell.prediction_proj.zero_()
        cell.mode_couplings[0].zero_()

    base_kwargs = dict(
        signal_real=torch.ones(1, 1),
        signal_imag=torch.zeros(1, 1),
        state_real=torch.ones(1, 1),
        state_imag=torch.zeros(1, 1),
        carry_real=torch.zeros(1, 1),
        carry_imag=torch.zeros(1, 1),
        active_sizes=(1,),
        step_index=1,
        allow_growth=False,
    )
    low_accumulator = torch.zeros(1, 1)
    high_accumulator = torch.full((1, 1), 2.0)

    low = cell.step(magnitude_accumulator=low_accumulator, **base_kwargs)
    high = cell.step(magnitude_accumulator=high_accumulator, **base_kwargs)

    low_proposal_mag = (low[2] - cell.magnitude_decay * low_accumulator) / (1.0 - cell.magnitude_decay)
    high_proposal_mag = (high[2] - cell.magnitude_decay * high_accumulator) / (1.0 - cell.magnitude_decay)

    assert high_proposal_mag.item() > low_proposal_mag.item()


def test_parallel_mixer_accumulator_modulation_amplifies_late_steps() -> None:
    enabled = ParallelComplexReciprocatorMixer(
        hidden_dim=1,
        state_dim=1,
        state_rank=1,
        state_mode_sizes=(1,),
        init_mode_sizes=(1,),
        num_cube_engines=1,
        normalization="frobenius",
        impression_rate=0.35,
        magnitude_floor=1e-3,
        dropout=0.0,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        persist_state=False,
        accumulator_modulates_gains=True,
    ).eval()
    disabled = ParallelComplexReciprocatorMixer(
        hidden_dim=1,
        state_dim=1,
        state_rank=1,
        state_mode_sizes=(1,),
        init_mode_sizes=(1,),
        num_cube_engines=1,
        normalization="frobenius",
        impression_rate=0.35,
        magnitude_floor=1e-3,
        dropout=0.0,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        persist_state=False,
        accumulator_modulates_gains=False,
    ).eval()
    disabled.load_state_dict(enabled.state_dict())

    with torch.no_grad():
        for mixer in (enabled, disabled):
            engine = mixer.cube_engines[0]
            mixer.signal_proj.weight_real.fill_(1.0)
            mixer.signal_proj.weight_imag.zero_()
            engine.input_gain.zero_()
            engine.recurrent_gain.fill_(0.5)
            engine.carry_gain.fill_(-20.0)
            engine.decay.fill_(-3.0)
            engine.accumulator_input_gain_scale.fill_(-20.0)
            engine.accumulator_carry_gain_scale.fill_(-20.0)
            engine.accumulator_recurrent_gain_scale.fill_(0.0)
            engine.accumulator_coupling_scale.fill_(0.0)
            engine.prediction_proj.zero_()
            mixer.engine_state_to_hidden[0].weight_real.zero_()
            mixer.engine_state_to_hidden[0].weight_imag.zero_()
            mixer.engine_state_to_hidden[0].weight_real[0, 3] = 1.0
            mixer.engine_fusion.weight_real.fill_(1.0)
            mixer.engine_fusion.weight_imag.zero_()
            mixer.gate_proj.weight.zero_()
            mixer.gate_proj.bias.fill_(10.0)

    hidden = torch.ones(1, 4, 1, dtype=torch.complex64)
    with torch.no_grad():
        enabled_output = enabled(hidden)
        disabled_output = disabled(hidden)

    torch.testing.assert_close(enabled_output[:, :1], disabled_output[:, :1], atol=1e-6, rtol=0.0)
    assert enabled_output.real[0, -1, 0].item() > disabled_output.real[0, -1, 0].item()


def test_mixer_preallocated_projections() -> None:
    torch.manual_seed(0)
    mixer = ComplexReciprocatorMixer(
        hidden_dim=8,
        state_dim=27,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        init_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        normalization="per_mode",
        impression_rate=0.35,
        magnitude_floor=1e-3,
        dropout=0.0,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        persist_state=False,
    )
    hidden = torch.complex(torch.randn(2, 4, 8), torch.randn(2, 4, 8))

    output = mixer(hidden)
    output.abs().sum().backward()

    active_mask = _mask_to_active(torch.ones(3, 3, 3), (2, 2, 2), state_rank=3).reshape(-1).bool()
    signal_grad = mixer.signal_proj.weight_real.grad
    hidden_grad = mixer.engine_state_to_hidden[0].weight_real.grad
    assert signal_grad is not None
    assert hidden_grad is not None
    assert signal_grad.shape[0] == 27
    assert signal_grad.abs().sum(dim=1)[active_mask].sum().item() > 0.0
    assert torch.count_nonzero(signal_grad.abs().sum(dim=1)[~active_mask]) == 0
    assert hidden_grad.shape[1] == mixer.engine_state_feature_dim
    assert hidden_grad.abs().sum().item() > 0.0


def test_online_mixer_preallocated_projections_respect_active_mask() -> None:
    torch.manual_seed(0)
    mixer = ComplexReciprocatorMixer(
        hidden_dim=8,
        state_dim=27,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        init_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        normalization="per_mode",
        impression_rate=0.35,
        magnitude_floor=1e-3,
        dropout=0.0,
        growth_threshold=1e9,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        persist_state=True,
    )
    hidden = torch.complex(torch.randn(2, 4, 8), torch.randn(2, 4, 8))

    output = mixer(hidden)
    output.abs().sum().backward()

    active_mask = _mask_to_active(torch.ones(3, 3, 3), (2, 2, 2), state_rank=3).reshape(-1).bool()
    signal_grad = mixer.signal_proj.weight_real.grad
    assert signal_grad is not None
    assert signal_grad.abs().sum(dim=1)[active_mask].sum().item() > 0.0
    assert torch.count_nonzero(signal_grad.abs().sum(dim=1)[~active_mask]) == 0


def test_complex_mixer_zero_signal_stays_zero_without_magnitude_floor() -> None:
    mixer = ComplexReciprocatorMixer(
        hidden_dim=8,
        state_dim=27,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        init_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        normalization="per_mode",
        impression_rate=0.35,
        magnitude_floor=1e-3,
        dropout=0.0,
        growth_threshold=1.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        persist_state=False,
    ).eval()
    hidden = torch.zeros(2, 4, 8, dtype=torch.complex64)

    with torch.no_grad():
        for projection in mixer.engine_state_to_hidden:
            projection.weight_real.zero_()
            projection.weight_imag.zero_()
        mixer.engine_fusion.weight_real.zero_()
        mixer.engine_fusion.weight_imag.zero_()
        output = mixer(hidden)

    assert torch.isfinite(output.real).all()
    assert torch.isfinite(output.imag).all()
    assert torch.count_nonzero(output) == 0


def _reference_apply_mode_couplings(
    tensor: torch.Tensor,
    couplings: list[torch.Tensor],
    state_rank: int,
) -> torch.Tensor:
    mixed = tensor
    batch_dims = mixed.ndim - state_rank
    for mode_idx, coupling in enumerate(couplings):
        axis = batch_dims + mode_idx
        mixed = torch.movedim(mixed, axis, -1)
        mode_size = mixed.shape[-1]
        orig_shape = mixed.shape
        flat_mixed = mixed.reshape(-1, mode_size)
        flat_coupling = coupling.reshape(-1, mode_size, mode_size)
        if flat_mixed.shape[0] % flat_coupling.shape[0] != 0:
            raise ValueError("mode couplings could not be broadcast across the active tensor slices")
        repeat_factor = flat_mixed.shape[0] // flat_coupling.shape[0]
        if repeat_factor != 1:
            flat_coupling = flat_coupling.repeat_interleave(repeat_factor, dim=0)
        if torch.is_complex(flat_coupling) and not torch.is_complex(flat_mixed):
            flat_mixed = flat_mixed.to(dtype=flat_coupling.dtype)
        mixed = torch.bmm(flat_mixed.unsqueeze(1), flat_coupling).squeeze(1).reshape(*orig_shape)
        mixed = torch.movedim(mixed, -1, axis)
    return mixed


def test_apply_mode_couplings_pair_matches_reference() -> None:
    torch.manual_seed(0)
    state_real = torch.randn(2, 3, 4, 4, 4)
    state_imag = torch.randn(2, 3, 4, 4, 4)
    local_real = torch.randn(2, 3, 4, 4, 4)
    local_imag = torch.randn(2, 3, 4, 4, 4)
    couplings = _partial_trace_couplings(state_real, state_imag, state_rank=3)

    expected = _reference_apply_mode_couplings(torch.complex(local_real, local_imag), couplings, state_rank=3)
    actual = _apply_mode_couplings(torch.complex(local_real, local_imag), couplings, state_rank=3)
    actual_pair_real, actual_pair_imag = _apply_mode_couplings_pair(
        local_real, local_imag, couplings, state_rank=3
    )

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual_pair_real, expected.real)
    assert torch.allclose(actual_pair_imag, expected.imag)


def test_partial_trace_couplings_use_phase_when_enabled() -> None:
    in_phase_real = torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]], [[0.5, 0.0], [0.0, 0.0]]]]
    )
    in_phase_imag = torch.zeros_like(in_phase_real)
    quarter_turn_real = torch.zeros_like(in_phase_real)
    quarter_turn_imag = in_phase_real.clone()

    phase_couplings = _partial_trace_couplings(
        in_phase_real,
        in_phase_imag,
        state_rank=3,
        phase_aware_coupling=True,
    )
    rotated_couplings = _partial_trace_couplings(
        quarter_turn_real,
        quarter_turn_imag,
        state_rank=3,
        phase_aware_coupling=True,
    )
    legacy_in_phase = _partial_trace_couplings(
        in_phase_real,
        in_phase_imag,
        state_rank=3,
        phase_aware_coupling=False,
    )
    legacy_rotated = _partial_trace_couplings(
        quarter_turn_real,
        quarter_turn_imag,
        state_rank=3,
        phase_aware_coupling=False,
    )

    assert torch.is_complex(phase_couplings[0])
    assert not torch.allclose(phase_couplings[0], rotated_couplings[0])
    torch.testing.assert_close(
        phase_couplings[0].abs().sum(dim=-1),
        torch.ones_like(phase_couplings[0].abs().sum(dim=-1)),
    )
    torch.testing.assert_close(legacy_in_phase[0], legacy_rotated[0])


def test_nonpersistent_training_mixer_keeps_fixed_support_even_with_growth_recipe_enabled() -> None:
    torch.manual_seed(0)
    mixer = ComplexReciprocatorMixer(
        hidden_dim=8,
        state_dim=27,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        init_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        normalization="per_mode",
        impression_rate=0.35,
        magnitude_floor=1e-3,
        dropout=0.0,
        growth_threshold=0.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        persist_state=False,
    ).train()
    hidden = torch.complex(torch.randn(2, 4, 8), torch.randn(2, 4, 8))

    growth_before = sum(int(engine._growth_event_count.item()) for engine in mixer.cube_engines)
    output = mixer(hidden)
    growth_after = sum(int(engine._growth_event_count.item()) for engine in mixer.cube_engines)

    assert output.shape == hidden.shape
    assert growth_after == growth_before


def test_nonpersistent_eval_mixer_does_not_grow_by_default() -> None:
    torch.manual_seed(0)
    mixer = ComplexReciprocatorMixer(
        hidden_dim=8,
        state_dim=27,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        init_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        normalization="per_mode",
        impression_rate=0.35,
        magnitude_floor=1e-3,
        dropout=0.0,
        growth_threshold=0.0,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=32,
        persist_state=False,
    ).eval()
    hidden = torch.complex(torch.randn(2, 4, 8), torch.randn(2, 4, 8))

    growth_before = sum(int(engine._growth_event_count.item()) for engine in mixer.cube_engines)
    _ = mixer(hidden)
    growth_after = sum(int(engine._growth_event_count.item()) for engine in mixer.cube_engines)

    assert growth_after == growth_before


def test_growth_threshold_anneals_from_ten_x_to_nominal() -> None:
    nominal = 0.02

    assert _annealed_growth_threshold(nominal, step=1, total_steps=100) == pytest.approx(0.2)
    assert _annealed_growth_threshold(nominal, step=10, total_steps=100) == pytest.approx(nominal)
    assert _annealed_growth_threshold(nominal, step=100, total_steps=100) == pytest.approx(nominal)


def test_reset_optimizer_moments_clears_adam_state() -> None:
    layer = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    inputs = torch.randn(2, 4)

    loss = layer(inputs).square().mean()
    loss.backward()
    optimizer.step()

    assert optimizer.state
    _reset_optimizer_moments(optimizer)
    assert not optimizer.state


def test_reciprocator_only_training_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = make_reciprocator_only_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randint(0, 32, (4, 4))
    targets = torch.randint(0, 32, (4, 4))

    model.train()
    _, loss_before = model(inputs, targets)
    for _ in range(20):
        _, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _, loss_after = model(inputs, targets)

    assert loss_after.item() < loss_before.item()


def test_reciprocator_only_forward_is_train_eval_invariant() -> None:
    torch.manual_seed(0)
    model = make_growth_sensitive_reciprocator_only_model()
    inputs = torch.randint(0, 32, (2, 8))

    model.train()
    with torch.no_grad():
        train_logits, _ = model(inputs)
    model.eval()
    with torch.no_grad():
        eval_logits, _ = model(inputs)

    torch.testing.assert_close(train_logits, eval_logits)


def test_reciprocator_only_forward_is_grad_mode_invariant() -> None:
    torch.manual_seed(0)
    model = make_growth_sensitive_reciprocator_only_model().train()
    inputs = torch.randint(0, 32, (2, 8))

    with torch.no_grad():
        no_grad_logits, _ = model(inputs)
    grad_logits, _ = model(inputs)

    torch.testing.assert_close(no_grad_logits, grad_logits)


def test_online_reciprocator_only_forward_is_train_eval_invariant_when_state_resets() -> None:
    torch.manual_seed(0)
    model = make_growth_sensitive_reciprocator_only_model()
    model.enter_online_mode()
    inputs = torch.randint(0, 32, (1, 4))

    model.train()
    model.reset_online_state()
    with torch.no_grad():
        train_logits, _ = model(inputs)

    model.eval()
    model.reset_online_state()
    with torch.no_grad():
        eval_logits, _ = model(inputs)

    torch.testing.assert_close(train_logits, eval_logits)


def test_online_reciprocator_lm_persists_state() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        max_mode_sizes=(3, 3, 3),
        num_cube_engines=2,
        normalization="per_mode",
        dropout=0.0,
    )
    model = ReciprocatorOnlyLM(config).eval()
    model.enter_online_mode()
    model.reset_online_state()
    inputs = torch.randint(0, 32, (1, 4))

    with torch.no_grad():
        model(inputs)
        first_state = model.blocks[0].mixer._persistent_state_reals.clone()
        first_accumulator = model.blocks[0].mixer._persistent_state_accumulators.clone()
        model(inputs)
        second_state = model.blocks[0].mixer._persistent_state_reals.clone()
        second_accumulator = model.blocks[0].mixer._persistent_state_accumulators.clone()

    assert first_state is not None
    assert second_state is not None
    assert first_accumulator is not None
    assert second_accumulator is not None
    assert not torch.allclose(first_state, second_state)
    assert not torch.allclose(first_accumulator, second_accumulator)


def test_online_reciprocator_lm_can_backprop_across_multiple_chunks() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        max_mode_sizes=(3, 3, 3),
        num_cube_engines=2,
        normalization="per_mode",
        dropout=0.0,
    )
    model = ReciprocatorOnlyLM(config).train()
    model.enter_online_mode()
    model.reset_online_state()
    model.set_online_state_gradient_tracking(True)

    first_inputs = torch.randint(0, 32, (1, 4))
    first_targets = torch.randint(0, 32, (1, 4))
    second_inputs = torch.randint(0, 32, (1, 4))
    second_targets = torch.randint(0, 32, (1, 4))

    _, first_loss = model(first_inputs, first_targets)
    first_state = model.blocks[0].mixer._persistent_state_reals
    assert first_state is not None
    assert first_state.requires_grad
    assert first_state.grad_fn is not None

    _, second_loss = model(second_inputs, second_targets)
    total_loss = first_loss + second_loss
    total_loss.backward()

    signal_grad = model.blocks[0].mixer.signal_proj.weight_real.grad
    assert signal_grad is not None
    assert signal_grad.abs().sum().item() > 0.0

    model.detach_online_state()
    detached_state = model.blocks[0].mixer._persistent_state_reals
    assert detached_state is not None
    assert detached_state.grad_fn is None


def test_online_reciprocator_lm_keeps_fixed_active_sizes_across_sequences() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        max_mode_sizes=(3, 3, 3),
        num_cube_engines=2,
        normalization="per_mode",
        dropout=0.0,
        growth_threshold=0.0,
        growth_interval=1,
    )
    model = ReciprocatorOnlyLM(config).eval()
    model.enter_online_mode()
    model.reset_online_state()

    with torch.no_grad():
        for _ in range(4):
            model(torch.randint(0, 32, (1, 4)))

    diagnostics = model.online_diagnostics()
    active_sizes = diagnostics["layers"][0]["active_sizes"]
    assert active_sizes == config.init_mode_sizes
    assert diagnostics["layers"][0]["active_rank"] == config.state_rank


def test_online_reciprocator_lm_dynamic_rank_growth_updates_active_rank() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=1,
        max_state_rank=2,
        dynamic_rank=True,
        init_mode_sizes=(2, 1),
        max_mode_sizes=(2, 2),
        num_cube_engines=1,
        normalization="per_mode",
        dropout=0.0,
        growth_threshold=0.0,
        growth_interval=1,
    )
    model = ReciprocatorOnlyLM(config).eval()
    model.enter_online_mode()
    model.reset_online_state()

    with torch.no_grad():
        model(torch.randint(0, 32, (1, 4)))

    diagnostics = model.online_diagnostics()
    assert diagnostics["layers"][0]["active_rank"] == 2
    assert diagnostics["layers"][0]["active_sizes"] == (2, 2)


def test_parallel_mixer_dynamic_rank_growth_updates_last_active_rank() -> None:
    torch.manual_seed(0)
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            dim=32,
            n_layers=1,
            n_heads=4,
            state_rank=1,
            max_state_rank=2,
            dynamic_rank=True,
            init_mode_sizes=(2, 1),
            max_mode_sizes=(2, 2),
            num_cube_engines=1,
            normalization="frobenius",
            dropout=0.0,
            growth_threshold=0.0,
            growth_interval=1,
            parallel_mixer=True,
        )
    ).train()
    inputs = torch.randint(0, 32, (2, 4))

    model(inputs)

    diagnostics = model.blocks[0].mixer.diagnostics()
    assert diagnostics["active_rank"] == 2
    assert diagnostics["active_sizes"] == (2, 2)


def test_enter_online_mode_rejects_parallel_mixer() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        num_cube_engines=2,
        normalization="per_mode",
        dropout=0.0,
        parallel_mixer=True,
    )
    model = ReciprocatorOnlyLM(config)

    with pytest.raises(RuntimeError, match="parallel_mixer"):
        model.enter_online_mode()


def test_online_generate_same_prompt_does_not_replay_prompt_state() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        max_mode_sizes=(3, 3, 3),
        num_cube_engines=2,
        normalization="per_mode",
        dropout=0.0,
    )
    model = ReciprocatorOnlyLM(config).eval()
    model.enter_online_mode()
    model.reset_online_state()
    prompt = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        model.generate(prompt, max_new_tokens=2, temperature=0.0)
        first_steps = int(model.blocks[0].mixer._step_counter.item())
        model.generate(prompt, max_new_tokens=1, temperature=0.0)
        second_steps = int(model.blocks[0].mixer._step_counter.item())

    assert first_steps == prompt.size(1) + 2
    assert second_steps == first_steps + 1


def test_state_dict_round_trip_restores_online_runtime_state() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        max_mode_sizes=(3, 3, 3),
        num_cube_engines=2,
        normalization="per_mode",
        dropout=0.0,
    )
    model = ReciprocatorOnlyLM(config).eval()
    model.enter_online_mode()
    model.reset_online_state()
    prompt = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        model(torch.randint(0, 32, (1, 4)))
        model.generate(prompt, max_new_tokens=2, temperature=0.0)

    saved_state = model.state_dict()
    original_step_counter = int(model.blocks[0].mixer._step_counter.item())
    original_state_real = model.blocks[0].mixer._persistent_state_reals.clone()
    original_state_accumulator = model.blocks[0].mixer._persistent_state_accumulators.clone()
    original_active_sizes = model.blocks[0].mixer._persistent_active_sizes.clone()
    original_next_logits = model._online_next_token_logits.clone()

    restored = ReciprocatorOnlyLM(config).eval()
    restored.load_state_dict(saved_state)

    assert restored.blocks[0].mixer.persist_state is True
    assert int(restored.blocks[0].mixer._step_counter.item()) == original_step_counter
    torch.testing.assert_close(restored.blocks[0].mixer._persistent_state_reals, original_state_real)
    torch.testing.assert_close(restored.blocks[0].mixer._persistent_state_accumulators, original_state_accumulator)
    torch.testing.assert_close(restored.blocks[0].mixer._persistent_active_sizes, original_active_sizes)
    torch.testing.assert_close(restored._online_next_token_logits, original_next_logits)

    with torch.no_grad():
        restored.generate(prompt, max_new_tokens=1, temperature=0.0)

    assert int(restored.blocks[0].mixer._step_counter.item()) == original_step_counter + 1


def test_reciprocator_only_with_context_2() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=2,
        dim=32,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        num_cube_engines=3,
        normalization="per_mode",
        dropout=0.0,
    )
    model = ReciprocatorOnlyLM(config)
    inputs = torch.randint(0, 32, (2, 2))
    targets = torch.randint(0, 32, (2, 2))

    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 2, 32)
    assert loss is not None

    loss.backward()
    assert model.blocks[0].mixer.signal_proj.weight_real.grad is not None
    for engine in model.blocks[0].mixer.cube_engines:
        _assert_prediction_projector_has_grad(engine.prediction_proj)


def test_parallel_mixer_forward_and_backward() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        num_cube_engines=3,
        normalization="per_mode",
        dropout=0.0,
        parallel_mixer=True,
    )
    model = ReciprocatorOnlyLM(config)
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 8, 32)
    assert loss is not None
    loss.backward()
    assert model.blocks[0].mixer.signal_proj.weight_real.grad is not None
    for engine in model.blocks[0].mixer.cube_engines:
        _assert_prediction_projector_has_grad(engine.prediction_proj)


def test_parallel_mixer_forward_and_backward_with_spectral_reciprocation() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        num_cube_engines=2,
        normalization="frobenius",
        dropout=0.0,
        parallel_mixer=True,
        use_spectral_reciprocation=True,
        learnable_spectral_reciprocation=True,
        spectral_mode="wavelet_packet_max_ultimate",
    )
    model = ReciprocatorOnlyLM(config)
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 8, 32)
    assert loss is not None
    loss.backward()
    assert model.blocks[0].mixer.signal_proj.weight_real.grad is not None
    assert model.blocks[0].mixer.joint_spectral_mode is True
    assert model.blocks[0].mixer.joint_spectral_reciprocator is not None
    for engine in model.blocks[0].mixer.cube_engines:
        assert engine.use_spectral_reciprocation is False
        assert engine.learnable_spectral_reciprocation is True
        assert engine.spectral_mode == "wavelet_packet_max_ultimate"
        _assert_prediction_projector_has_grad(engine.prediction_proj)
        assert engine.spectral_low_frequency_gain_raw is not None
        assert engine.spectral_low_frequency_gain_raw.grad is not None


def test_rank8_dynamic_profile_is_viable() -> None:
    config = ModelConfig(
        vocab_size=512,
        max_seq_len=512,
        dim=256,
        n_layers=4,
        n_heads=8,
        state_rank=4,
        max_state_rank=8,
        dynamic_rank=True,
        init_mode_sizes=(4, 4, 2, 2),
        max_mode_sizes=(8, 8, 4, 4),
        num_cube_engines=4,
        normalization="frobenius",
        dropout=0.05,
    )
    model = ReciprocatorOnlyLM(config)
    params = sum(parameter.numel() for parameter in model.parameters())
    inputs = torch.randint(0, config.vocab_size, (1, 2))
    targets = torch.randint(0, config.vocab_size, (1, 2))

    logits, loss = model(inputs, targets)
    assert logits.shape == (1, 2, config.vocab_size)
    assert loss is not None
    loss.backward()

    assert config.state_mode_sizes == (8, 8, 4, 4, 2, 2, 2, 2)
    assert params < 60_000_000
    assert model.blocks[0].mixer.signal_proj.weight_real.grad is not None


def test_parallel_scan_linear_supports_time_varying_decay() -> None:
    decay = torch.tensor(
        [
            [[0.5, 0.1], [0.2, 0.3], [0.4, 0.7]],
        ]
    )
    inputs = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
        ]
    )

    actual = _parallel_scan_linear(decay, inputs)
    expected = torch.zeros_like(inputs)
    for step in range(inputs.shape[1]):
        if step == 0:
            expected[:, step] = inputs[:, step]
        else:
            expected[:, step] = decay[:, step] * expected[:, step - 1] + inputs[:, step]

    torch.testing.assert_close(actual, expected)


def test_parallel_mixer_supports_frobenius_normalization() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(3, 3, 3),
        num_cube_engines=2,
        normalization="frobenius",
        dropout=0.0,
        parallel_mixer=True,
    )
    model = ReciprocatorOnlyLM(config)
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 8, 32)
    assert loss is not None
