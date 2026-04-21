import pytest
import torch

from reciprocator_lm import ModelConfig, ReciprocatorOnlyLM, mode_axis_permutation_probe


def _make_model(
    state_mode_sizes: tuple[int, ...] = (3, 3, 3),
    *,
    phase_aware_coupling: bool = True,
) -> ReciprocatorOnlyLM:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        dim=16,
        n_layers=1,
        n_heads=4,
        state_rank=len(state_mode_sizes),
        state_mode_sizes=state_mode_sizes,
        num_cube_engines=1,
        normalization="per_mode",
        dropout=0.0,
        phase_aware_coupling=phase_aware_coupling,
    )
    return ReciprocatorOnlyLM(config)


def _assign_engine_tensor(model: ReciprocatorOnlyLM, tensor: torch.Tensor) -> None:
    engine = model.blocks[0].mixer.cube_engines[0]
    with torch.no_grad():
        for name in ("input_gain", "recurrent_gain", "carry_gain", "decay"):
            getattr(engine, name).copy_(tensor.to(dtype=getattr(engine, name).dtype))


def _parameter_score(model: ReciprocatorOnlyLM) -> float:
    engine = model.blocks[0].mixer.cube_engines[0]
    device = engine.input_gain.device
    dtype = engine.input_gain.dtype
    axes = [torch.arange(size, device=device, dtype=dtype) for size in engine.input_gain.shape]
    coords = torch.meshgrid(*axes, indexing="ij")
    weight = sum((index + 1) * axis for index, axis in enumerate(coords))
    total = torch.zeros((), device=device, dtype=dtype)
    for scale, name in enumerate(("input_gain", "recurrent_gain", "carry_gain", "decay"), start=1):
        total = total + scale * (getattr(engine, name) * weight).sum()
    return float(total.item())


def _snapshot_engine_parameters(model: ReciprocatorOnlyLM) -> dict[str, torch.Tensor]:
    engine = model.blocks[0].mixer.cube_engines[0]
    return {
        name: getattr(engine, name).detach().clone()
        for name in (
            "input_gain",
            "recurrent_gain",
            "carry_gain",
            "decay",
            "accumulator_input_gain_scale",
            "accumulator_recurrent_gain_scale",
            "accumulator_carry_gain_scale",
            "accumulator_coupling_scale",
        )
    }


def _configure_phase_sensitive_routing(model: ReciprocatorOnlyLM) -> None:
    engine = model.blocks[0].mixer.cube_engines[0]
    with torch.no_grad():
        engine.input_gain.fill_(2.0)
        engine.recurrent_gain.fill_(3.0)
        engine.carry_gain.zero_()
        engine.decay.fill_(-4.0)
        for weight in engine.cpl_proj_real:
            weight.zero_()
        for weight in engine.cpl_proj_imag:
            weight.zero_()
        engine.cpl_proj_imag[0][0, 0] = 4.0
        engine.cpl_proj_imag[0][0, 1] = -4.0


def test_mode_axis_permutation_probe_reports_zero_for_symmetric_tensors() -> None:
    model = _make_model()
    coords = torch.meshgrid(*(torch.arange(3, dtype=torch.float32) for _ in range(3)), indexing="ij")
    symmetric = coords[0] + coords[1] + coords[2]
    _assign_engine_tensor(model, symmetric)

    result = mode_axis_permutation_probe(model, evaluate_fn=lambda: _parameter_score(model), metric_name="score")

    assert result["supported_pair_count"] == 3
    for pair_result in result["pairs"].values():
        assert pair_result["supported"] is True
        assert float(pair_result["parameter_relative_delta_mean"]) == pytest.approx(0.0, abs=1e-7)
        assert float(pair_result["score_delta"]) == pytest.approx(0.0, abs=1e-7)


def test_mode_axis_permutation_probe_detects_asymmetry_and_restores_parameters() -> None:
    model = _make_model()
    coords = torch.meshgrid(*(torch.arange(3, dtype=torch.float32) for _ in range(3)), indexing="ij")
    asymmetric = 5.0 * coords[0] + 2.0 * coords[1] + 0.25 * coords[2]
    _assign_engine_tensor(model, asymmetric)
    original = _snapshot_engine_parameters(model)

    result = mode_axis_permutation_probe(model, evaluate_fn=lambda: _parameter_score(model), metric_name="score")

    assert any(
        float(pair_result["parameter_relative_delta_mean"]) > 0.1
        for pair_result in result["pairs"].values()
        if pair_result["supported"]
    )
    assert any(
        abs(float(pair_result["score_delta"])) > 1e-3
        for pair_result in result["pairs"].values()
        if pair_result["supported"]
    )

    restored = _snapshot_engine_parameters(model)
    for name in original:
        torch.testing.assert_close(restored[name], original[name])


def test_mode_axis_permutation_probe_skips_unequal_mode_sizes() -> None:
    model = _make_model((2, 3, 4))

    result = mode_axis_permutation_probe(model)

    assert result["supported_pair_count"] == 0
    for pair_result in result["pairs"].values():
        assert pair_result["supported"] is False
        assert "reason" in pair_result


def test_mode_axis_permutation_probe_skips_unequal_active_mode_sizes() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        dim=16,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        init_mode_sizes=(2, 3, 4),
        max_mode_sizes=(4, 4, 4),
        num_cube_engines=1,
        normalization="per_mode",
        dropout=0.0,
    )
    model = ReciprocatorOnlyLM(config)

    result = mode_axis_permutation_probe(model)

    assert result["supported_pair_count"] == 0
    for pair_result in result["pairs"].values():
        assert pair_result["supported"] is False
        assert "reason" in pair_result


def test_phase_aware_coupling_changes_probe_model_logits() -> None:
    torch.manual_seed(0)
    phase_model = _make_model(phase_aware_coupling=True).eval()
    _configure_phase_sensitive_routing(phase_model)
    fallback_model = _make_model(phase_aware_coupling=False).eval()
    fallback_model.load_state_dict(phase_model.state_dict())
    inputs = torch.tensor([[5, 5, 5, 5]])

    with torch.no_grad():
        phase_logits, _ = phase_model(inputs)
        fallback_logits, _ = fallback_model(inputs)

    assert not torch.allclose(phase_logits[:, 1:], fallback_logits[:, 1:])
    result = mode_axis_permutation_probe(
        phase_model,
        evaluate_fn=lambda: float(phase_model(inputs)[0].sum().item()),
        metric_name="logit_sum",
    )
    assert result["supported_pair_count"] == 3
