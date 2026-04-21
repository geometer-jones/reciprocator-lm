import torch

from reciprocator_lm import (
    ModelConfig,
    ReciprocatorOnlyLM,
    SyntheticSequenceBatch,
    build_default_benchmark_suite,
    make_compositional_binding_batch,
    make_controlled_novelty_batch,
    make_hierarchical_conditioning_batch,
    make_induction_batch,
    make_long_range_retrieval_batch,
    make_role_rebinding_batch,
    sequence_accuracy,
)


def _oracle_logits(batch: SyntheticSequenceBatch, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(batch.input_ids.shape[0], batch.input_ids.shape[1], vocab_size)
    for position in batch.prediction_positions.tolist():
        target = batch.labels[:, position + 1]
        logits[torch.arange(batch.input_ids.shape[0]), position, target] = 10.0
    return logits


def _make_benchmark_model(*, phase_aware_coupling: bool) -> ReciprocatorOnlyLM:
    config = ModelConfig(
        vocab_size=64,
        max_seq_len=16,
        dim=16,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        num_cube_engines=1,
        normalization="per_mode",
        dropout=0.0,
        phase_aware_coupling=phase_aware_coupling,
    )
    return ReciprocatorOnlyLM(config).eval()


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


def test_default_benchmark_suite_contains_requested_tasks() -> None:
    suite = build_default_benchmark_suite(vocab_size=64)

    assert [benchmark.name for benchmark in suite] == [
        "long_range_retrieval",
        "hierarchical_conditioning",
        "compositional_binding",
        "role_rebinding",
        "induction",
        "controlled_novelty",
    ]


def test_synthetic_benchmarks_return_supervised_positions() -> None:
    batches = [
        make_long_range_retrieval_batch(num_examples=2, vocab_size=64),
        make_hierarchical_conditioning_batch(num_examples=2, vocab_size=64),
        make_compositional_binding_batch(num_examples=2, vocab_size=64),
        make_role_rebinding_batch(num_examples=2, vocab_size=64),
        make_induction_batch(num_examples=2, vocab_size=64),
        make_controlled_novelty_batch(num_examples=2, vocab_size=64),
    ]

    for batch in batches:
        assert batch.input_ids.ndim == 2
        assert batch.attention_mask.shape == batch.input_ids.shape
        assert batch.labels.shape == batch.input_ids.shape
        assert batch.prediction_positions.ndim == 1
        assert batch.prediction_positions.numel() > 0
        assert (batch.labels != -100).any().item() is True


def test_sequence_accuracy_is_one_for_oracle_logits() -> None:
    batch = make_controlled_novelty_batch(num_examples=3, vocab_size=64)
    logits = _oracle_logits(batch, vocab_size=64)

    assert sequence_accuracy(logits, batch.labels, batch.prediction_positions) == 1.0


def test_phase_aware_coupling_changes_benchmark_logits() -> None:
    torch.manual_seed(0)
    batch = make_induction_batch(num_examples=2, vocab_size=64)
    phase_model = _make_benchmark_model(phase_aware_coupling=True)
    _configure_phase_sensitive_routing(phase_model)
    fallback_model = _make_benchmark_model(phase_aware_coupling=False)
    fallback_model.load_state_dict(phase_model.state_dict())

    with torch.no_grad():
        phase_logits, _ = phase_model(batch.input_ids)
        fallback_logits, _ = fallback_model(batch.input_ids)

    target_positions = batch.prediction_positions
    assert torch.isfinite(phase_logits).all()
    assert torch.isfinite(fallback_logits).all()
    assert not torch.allclose(
        phase_logits.index_select(1, target_positions),
        fallback_logits.index_select(1, target_positions),
    )
