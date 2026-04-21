import torch
from torch import nn

from reciprocator_lm import (
    BaselineTransformerConfig,
    BenchmarkTrainingConfig,
    BudgetMatch,
    ModelConfig,
    PlainTransformerLM,
    ReciprocatorOnlyLM,
    SmallMambaConfig,
    TrainingRunConfig,
    count_trainable_parameters,
    evaluate_benchmark_task,
    estimate_plain_transformer_train_flops,
    estimate_reciprocator_only_train_flops,
    estimate_small_mamba_train_flops,
    evaluate_benchmark_suite_generic,
    evaluate_causal_lm,
    make_controlled_novelty_batch,
    match_parameter_and_flop_budget,
    match_parameter_budget,
    split_train_val_tokens,
    train_benchmark_task,
    train_causal_language_model,
)


def test_split_train_val_tokens_produces_validation_tail() -> None:
    train_tokens, val_tokens = split_train_val_tokens(list(range(100)), seq_len=8, val_fraction=0.2)

    assert train_tokens.ndim == 1
    assert val_tokens is not None
    assert train_tokens.numel() + val_tokens.numel() == 100
    assert torch.equal(train_tokens[-3:], torch.tensor([77, 78, 79]))
    assert torch.equal(val_tokens[:3], torch.tensor([80, 81, 82]))


def test_evaluate_causal_lm_runs_for_transformer() -> None:
    model = PlainTransformerLM(
        BaselineTransformerConfig(
            vocab_size=32,
            model_dim=24,
            num_heads=4,
            num_layers=2,
            max_seq_len=12,
            dropout=0.0,
        )
    )
    metrics = evaluate_causal_lm(
        model,
        torch.randint(0, 32, (48,)),
        seq_len=8,
        device=torch.device("cpu"),
        max_batches=2,
    )

    assert metrics["loss"] > 0.0
    assert metrics["tokens"] > 0.0


def test_evaluate_benchmark_suite_generic_runs_for_reciprocator() -> None:
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=64,
            max_seq_len=32,
            dim=24,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            state_mode_sizes=(2, 2, 2),
            num_cube_engines=2,
            dropout=0.0,
        )
    )
    results = evaluate_benchmark_suite_generic(
        model,
        vocab_size=64,
        num_examples=4,
        device=torch.device("cpu"),
        seed=0,
    )

    assert set(results) == {
        "long_range_retrieval",
        "hierarchical_conditioning",
        "compositional_binding",
        "role_rebinding",
        "induction",
        "controlled_novelty",
    }


def test_evaluate_benchmark_task_returns_loss_and_accuracy() -> None:
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=64,
            max_seq_len=32,
            dim=24,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            state_mode_sizes=(2, 2, 2),
            num_cube_engines=2,
            dropout=0.0,
        )
    )

    metrics = evaluate_benchmark_task(
        model,
        benchmark_name="controlled_novelty",
        vocab_size=64,
        num_examples=4,
        device=torch.device("cpu"),
        seed=0,
    )

    assert metrics["loss"] > 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["examples"] == 4.0


def test_evaluate_benchmark_task_respects_causal_shift() -> None:
    batch = make_controlled_novelty_batch(num_examples=3, vocab_size=64, seed=0)

    class OracleBenchmarkModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids: torch.Tensor, attention_mask=None):
            del attention_mask
            assert torch.equal(input_ids.cpu(), batch.input_ids)
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 64, device=input_ids.device)
            for position in batch.prediction_positions.tolist():
                target = batch.labels[:, position + 1]
                logits[torch.arange(input_ids.shape[0]), position, target.to(device=input_ids.device)] = 10.0
            return (logits,)

    metrics = evaluate_benchmark_task(
        OracleBenchmarkModel(),
        benchmark_name="controlled_novelty",
        vocab_size=64,
        num_examples=3,
        device=torch.device("cpu"),
        seed=0,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["loss"] < 0.01


def test_train_benchmark_task_runs_and_evaluates_requested_tasks() -> None:
    torch.manual_seed(0)
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=64,
            max_seq_len=32,
            dim=24,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            state_mode_sizes=(2, 2, 2),
            num_cube_engines=1,
            dropout=0.0,
            dynamic_rank=False,
        )
    )

    summary = train_benchmark_task(
        model,
        benchmark_name="controlled_novelty",
        vocab_size=64,
        device=torch.device("cpu"),
        config=BenchmarkTrainingConfig(
            steps=2,
            batch_size=4,
            lr=1e-3,
            eval_every=1,
            eval_examples=4,
            log_every=1,
            seed=0,
        ),
        eval_benchmark_names=("controlled_novelty", "role_rebinding"),
    )

    assert summary["final_train_loss"] > 0.0
    assert summary["eval_metrics"] is not None
    assert set(summary["eval_metrics"]) == {"controlled_novelty", "role_rebinding"}
    assert len(summary["loss_history"]) == 2


def test_train_benchmark_task_supports_streaming_persistent_state() -> None:
    torch.manual_seed(0)
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=64,
            max_seq_len=32,
            dim=24,
            n_layers=1,
            n_heads=4,
            state_rank=2,
            init_mode_sizes=(4, 4),
            max_mode_sizes=(4, 4),
            num_cube_engines=1,
            dropout=0.0,
            dynamic_rank=False,
            persist_state=True,
        )
    )

    summary = train_benchmark_task(
        model,
        benchmark_name="controlled_novelty",
        vocab_size=64,
        device=torch.device("cpu"),
        config=BenchmarkTrainingConfig(
            steps=2,
            batch_size=4,
            lr=1e-3,
            eval_every=1,
            eval_examples=4,
            log_every=1,
            seed=0,
            streaming=True,
        ),
        eval_benchmark_names=("controlled_novelty",),
    )

    diagnostics = model.online_diagnostics()
    assert summary["final_train_loss"] > 0.0
    assert diagnostics["layers"][0]["persist_state"] is True
    assert diagnostics["layers"][0]["step_counter"] > 0


def test_match_parameter_budget_returns_closest_candidate() -> None:
    reference_model = PlainTransformerLM(
        BaselineTransformerConfig(
            vocab_size=32,
            model_dim=32,
            num_heads=4,
            num_layers=2,
            max_seq_len=16,
            dropout=0.0,
        )
    )
    target_parameter_count = count_trainable_parameters(reference_model)

    match = match_parameter_budget(
        target_parameter_count=target_parameter_count,
        candidate_values=[16, 24, 32, 40],
        build_model=lambda model_dim: PlainTransformerLM(
            BaselineTransformerConfig(
                vocab_size=32,
                model_dim=model_dim,
                num_heads=4,
                num_layers=2,
                max_seq_len=16,
                dropout=0.0,
            )
        ),
        parameter_name="model_dim",
    )

    assert match.parameter_value == 32
    assert match.parameter_count == target_parameter_count


def test_estimate_plain_transformer_train_flops_increases_with_seq_len() -> None:
    config = BaselineTransformerConfig(
        vocab_size=64,
        model_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=32,
        dropout=0.0,
    )

    short = estimate_plain_transformer_train_flops(config, batch_size=2, seq_len=8)
    long = estimate_plain_transformer_train_flops(config, batch_size=2, seq_len=16)

    assert short > 0.0
    assert long > short


def test_estimate_reciprocator_only_train_flops_increases_with_state_dim() -> None:
    small = ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        dim=24,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        init_mode_sizes=(2, 2, 2),
        max_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        dropout=0.0,
        parallel_mixer=True,
    )
    large = ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        dim=24,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        init_mode_sizes=(3, 3, 3),
        max_mode_sizes=(3, 3, 3),
        num_cube_engines=2,
        dropout=0.0,
        parallel_mixer=True,
    )

    small_flops = estimate_reciprocator_only_train_flops(small, batch_size=2, seq_len=8)
    large_flops = estimate_reciprocator_only_train_flops(large, batch_size=2, seq_len=8)

    assert small_flops > 0.0
    assert large_flops > small_flops


def test_estimate_reciprocator_only_train_flops_counts_input_dependent_gains() -> None:
    baseline = ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        dim=24,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        init_mode_sizes=(2, 2, 2),
        max_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        dropout=0.0,
        parallel_mixer=True,
        input_dependent_gains=False,
    )
    selected = ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        dim=24,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        init_mode_sizes=(2, 2, 2),
        max_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        dropout=0.0,
        parallel_mixer=True,
        input_dependent_gains=True,
    )

    baseline_flops = estimate_reciprocator_only_train_flops(baseline, batch_size=2, seq_len=8)
    selected_flops = estimate_reciprocator_only_train_flops(selected, batch_size=2, seq_len=8)

    assert selected_flops > baseline_flops


def test_estimate_reciprocator_only_train_flops_counts_selective_gains() -> None:
    baseline = ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        dim=24,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        init_mode_sizes=(2, 2, 2),
        max_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        dropout=0.0,
        parallel_mixer=True,
        input_dependent_gains=True,
        selective_gains=False,
    )
    selected = ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        dim=24,
        n_layers=2,
        n_heads=4,
        state_rank=3,
        init_mode_sizes=(2, 2, 2),
        max_mode_sizes=(2, 2, 2),
        num_cube_engines=2,
        dropout=0.0,
        parallel_mixer=True,
        input_dependent_gains=True,
        selective_gains=True,
    )

    baseline_flops = estimate_reciprocator_only_train_flops(baseline, batch_size=2, seq_len=8)
    selected_flops = estimate_reciprocator_only_train_flops(selected, batch_size=2, seq_len=8)

    assert selected_flops > baseline_flops


def test_estimate_small_mamba_train_flops_increases_with_state_size() -> None:
    small = SmallMambaConfig(
        vocab_size=64,
        model_dim=32,
        num_layers=2,
        state_size=8,
        expand=2,
        conv_kernel=4,
        max_seq_len=32,
        dropout=0.0,
    )
    large = SmallMambaConfig(
        vocab_size=64,
        model_dim=32,
        num_layers=2,
        state_size=16,
        expand=2,
        conv_kernel=4,
        max_seq_len=32,
        dropout=0.0,
    )

    small_flops = estimate_small_mamba_train_flops(small, batch_size=2, seq_len=8)
    large_flops = estimate_small_mamba_train_flops(large, batch_size=2, seq_len=8)

    assert small_flops > 0.0
    assert large_flops > small_flops


def test_match_parameter_and_flop_budget_returns_exact_match_when_available() -> None:
    target_config = BaselineTransformerConfig(
        vocab_size=64,
        model_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=32,
        dropout=0.0,
    )
    target_model = PlainTransformerLM(target_config)
    target_params = count_trainable_parameters(target_model)
    target_flops = estimate_plain_transformer_train_flops(target_config, batch_size=2, seq_len=8)

    match = match_parameter_and_flop_budget(
        target_parameter_count=target_params,
        target_train_flops_per_step=target_flops,
        candidate_values=[24, 32, 40],
        build_model=lambda model_dim: PlainTransformerLM(
            BaselineTransformerConfig(
                vocab_size=64,
                model_dim=model_dim,
                num_heads=4,
                num_layers=2,
                max_seq_len=32,
                dropout=0.0,
            )
        ),
        estimate_train_flops=lambda model: estimate_plain_transformer_train_flops(
            model.config,
            batch_size=2,
            seq_len=8,
        ),
        parameter_name="model_dim",
    )

    assert isinstance(match, BudgetMatch)
    assert match.parameter_value == 32
    assert match.parameter_relative_gap == 0.0
    assert match.train_flops_relative_gap == 0.0


def test_training_run_config_validates_positive_fields() -> None:
    config = TrainingRunConfig(
        steps=4,
        batch_size=2,
        seq_len=8,
        lr=1e-3,
        eval_every=2,
        eval_batches=1,
        log_every=2,
        seed=0,
    )

    assert config.steps == 4


def test_train_causal_language_model_freezes_reciprocator_growth_by_default() -> None:
    torch.manual_seed(0)
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=12,
            dim=24,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            init_mode_sizes=(2, 2, 2),
            max_mode_sizes=(3, 3, 3),
            num_cube_engines=1,
            growth_threshold=0.0,
            growth_interval=1,
            dropout=0.0,
        )
    )
    results = train_causal_language_model(
        model,
        train_tokens=torch.randint(0, 32, (64,)),
        device=torch.device("cpu"),
        config=TrainingRunConfig(
            steps=2,
            batch_size=2,
            seq_len=8,
            lr=1e-3,
            eval_every=1,
            eval_batches=1,
            log_every=1,
            seed=0,
        ),
        val_tokens=None,
    )

    growth_events = sum(int(engine._growth_event_count.item()) for engine in model.blocks[0].mixer.cube_engines)

    assert results["final_train_loss"] > 0.0
    assert growth_events == 0


def test_train_causal_language_model_growth_recipe_does_not_change_forward_support() -> None:
    torch.manual_seed(0)
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=12,
            dim=24,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            init_mode_sizes=(2, 2, 2),
            max_mode_sizes=(3, 3, 3),
            num_cube_engines=1,
            growth_threshold=0.0,
            growth_interval=1,
            dropout=0.0,
            training_growth_enabled=True,
        )
    )
    results = train_causal_language_model(
        model,
        train_tokens=torch.randint(0, 32, (64,)),
        device=torch.device("cpu"),
        config=TrainingRunConfig(
            steps=2,
            batch_size=2,
            seq_len=8,
            lr=1e-3,
            eval_every=1,
            eval_batches=1,
            log_every=1,
            seed=0,
        ),
        val_tokens=None,
    )

    growth_events = sum(int(engine._growth_event_count.item()) for engine in model.blocks[0].mixer.cube_engines)

    assert results["final_train_loss"] > 0.0
    assert growth_events == 0


def test_train_causal_language_model_runs_parallel_reciprocator() -> None:
    torch.manual_seed(0)
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=12,
            dim=24,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            init_mode_sizes=(2, 2, 2),
            max_mode_sizes=(3, 3, 3),
            num_cube_engines=1,
            normalization="frobenius",
            dropout=0.0,
            parallel_mixer=True,
        )
    )
    results = train_causal_language_model(
        model,
        train_tokens=torch.randint(0, 32, (64,)),
        val_tokens=None,
        device=torch.device("cpu"),
        config=TrainingRunConfig(
            steps=2,
            batch_size=2,
            seq_len=8,
            lr=1e-3,
            eval_every=1,
            eval_batches=1,
            log_every=1,
            seed=0,
        ),
    )

    assert results["final_train_loss"] > 0.0
    assert model.config.parallel_mixer is True
    assert model.config.normalization == "frobenius"


def test_train_causal_language_model_saves_latest_and_best_checkpoints(tmp_path) -> None:
    model = PlainTransformerLM(
        BaselineTransformerConfig(
            vocab_size=32,
            model_dim=24,
            num_heads=4,
            num_layers=1,
            max_seq_len=12,
            dropout=0.0,
        )
    )
    latest_checkpoint = tmp_path / "latest.pt"
    best_checkpoint = tmp_path / "best.pt"

    summary = train_causal_language_model(
        model,
        train_tokens=torch.randint(0, 32, (64,)),
        val_tokens=torch.randint(0, 32, (32,)),
        device=torch.device("cpu"),
        config=TrainingRunConfig(
            steps=2,
            batch_size=2,
            seq_len=8,
            lr=1e-3,
            save_every=1,
            eval_every=1,
            eval_batches=1,
            log_every=2,
            seed=0,
        ),
        latest_checkpoint_path=latest_checkpoint,
        best_checkpoint_path=best_checkpoint,
        checkpoint_metadata={"model_name": "transformer"},
    )

    latest_payload = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)

    assert latest_checkpoint.exists()
    assert best_checkpoint.exists()
    assert summary["latest_checkpoint"] == str(latest_checkpoint)
    assert summary["best_checkpoint"] == str(best_checkpoint)
    assert int(latest_payload["step"]) == 2
    assert "optimizer_state_dict" in latest_payload
    assert best_payload["metadata"]["model_name"] == "transformer"
    assert summary["token_budget"] == 32
    assert len(summary["loss_history"]) == 2
    assert summary["loss_history"][0]["tokens_seen"] == 16.0
    assert summary["loss_history"][1]["tokens_seen"] == 32.0
    assert len(latest_payload["loss_history"]) == 2


def test_train_causal_language_model_resume_matches_uninterrupted_run(tmp_path) -> None:
    model_config = BaselineTransformerConfig(
        vocab_size=32,
        model_dim=24,
        num_heads=4,
        num_layers=1,
        max_seq_len=12,
        dropout=0.0,
    )
    train_tokens = torch.randint(0, 32, (96,))
    full_config = TrainingRunConfig(
        steps=2,
        batch_size=2,
        seq_len=8,
        lr=1e-3,
        save_every=1,
        eval_every=2,
        eval_batches=1,
        log_every=2,
        seed=7,
    )

    torch.manual_seed(123)
    uninterrupted = PlainTransformerLM(model_config)
    train_causal_language_model(
        uninterrupted,
        train_tokens=train_tokens,
        val_tokens=None,
        device=torch.device("cpu"),
        config=full_config,
    )
    uninterrupted_state = {name: value.detach().clone() for name, value in uninterrupted.state_dict().items()}

    latest_checkpoint = tmp_path / "latest.pt"
    torch.manual_seed(123)
    first_leg = PlainTransformerLM(model_config)
    train_causal_language_model(
        first_leg,
        train_tokens=train_tokens,
        val_tokens=None,
        device=torch.device("cpu"),
        config=TrainingRunConfig(
            steps=1,
            batch_size=2,
            seq_len=8,
            lr=1e-3,
            save_every=1,
            eval_every=1,
            eval_batches=1,
            log_every=1,
            seed=7,
        ),
        latest_checkpoint_path=latest_checkpoint,
    )

    resumed = PlainTransformerLM(model_config)
    summary = train_causal_language_model(
        resumed,
        train_tokens=train_tokens,
        val_tokens=None,
        device=torch.device("cpu"),
        config=full_config,
        latest_checkpoint_path=latest_checkpoint,
        resume_from_checkpoint=True,
    )

    for name, value in uninterrupted_state.items():
        torch.testing.assert_close(resumed.state_dict()[name], value)
    assert summary["resumed_from_checkpoint"] is True
