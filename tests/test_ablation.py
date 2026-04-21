import json
import importlib.util
from pathlib import Path

import pytest
import torch

from reciprocator_lm import ModelConfig, ReciprocatorOnlyLM
from reciprocator_lm.ablation import factor_tuples, select_mode_size_pair


def _load_script_module(script_name: str):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_factor_tuples_enumerate_exact_non_decreasing_products() -> None:
    tuples = factor_tuples(64, 3)

    assert (4, 4, 4) in tuples
    assert (2, 4, 8) in tuples
    assert all(left <= right for mode_sizes in tuples for left, right in zip(mode_sizes, mode_sizes[1:]))
    assert all(mode_sizes[0] * mode_sizes[1] * mode_sizes[2] == 64 for mode_sizes in tuples)


@pytest.mark.parametrize(
    ("rank", "expected_init", "expected_max"),
    [
        (1, (27,), (64,)),
        (2, (3, 9), (4, 16)),
        (3, (1, 3, 9), (1, 4, 16)),
    ],
)
def test_select_mode_size_pair_matches_capacity_across_ranks(
    rank: int,
    expected_init: tuple[int, ...],
    expected_max: tuple[int, ...],
) -> None:
    init_mode_sizes, max_mode_sizes = select_mode_size_pair(
        state_rank=rank,
        init_capacity=27,
        max_capacity=64,
    )

    assert init_mode_sizes == expected_init
    assert max_mode_sizes == expected_max


def test_train_reciprocator_only_resolves_rank_aware_default_mode_sizes() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")

    init_mode_sizes, max_mode_sizes = train_script._resolve_mode_sizes(
        state_rank=2,
        init_mode_sizes=None,
        max_mode_sizes=None,
        init_state_capacity=None,
        state_capacity=None,
    )

    assert init_mode_sizes == (4, 16)
    assert max_mode_sizes == (4, 16)


def test_train_reciprocator_only_resolves_asymmetric_growth_recipe() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")

    init_mode_sizes, max_mode_sizes = train_script._resolve_mode_sizes(
        state_rank=3,
        init_mode_sizes=None,
        max_mode_sizes=None,
        init_state_capacity=27,
        state_capacity=64,
    )

    assert init_mode_sizes == (1, 3, 9)
    assert max_mode_sizes == (1, 4, 16)


def test_train_reciprocator_only_pads_rank8_max_mode_size_shorthand() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")

    init_mode_sizes, max_mode_sizes = train_script._resolve_mode_sizes(
        state_rank=4,
        max_state_rank=8,
        init_mode_sizes=(4, 4, 2, 2),
        max_mode_sizes=(8, 8, 4, 4),
        init_state_capacity=None,
        state_capacity=None,
    )

    assert init_mode_sizes == (4, 4, 2, 2, 1, 1, 1, 1)
    assert max_mode_sizes == (8, 8, 4, 4, 2, 2, 2, 2)


def test_train_reciprocator_only_defaults_to_frobenius_normalization() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")

    args = train_script._build_arg_parser().parse_args([])

    assert args.dynamic_rank is True
    assert args.normalization == "frobenius"
    assert args.learned_per_mode_scaling is True
    assert args.learnable_prediction_eta is True
    assert args.learnable_coupling_temperature is True
    assert args.learned_normalization_blend is True
    assert args.device == "auto"
    assert args.steps == 5000
    assert args.batch_size == 2
    assert args.seq_len == 256
    assert args.dim == 256
    assert args.layers == 4
    assert args.heads == 8
    assert args.state_rank == 4
    assert args.max_state_rank == 8
    assert args.init_mode_sizes == (4, 4, 2, 2)
    assert args.max_mode_sizes == (8, 8, 4, 4)
    assert args.dropout == pytest.approx(0.05)
    assert args.use_spectral_reciprocation is True
    assert args.learnable_spectral_reciprocation is True
    assert args.spectral_mode == "wavelet_packet_max_ultimate"
    assert args.joint_spectral_mode is None
    assert args.spectral_low_frequency_gain == pytest.approx(0.15)
    assert args.spectral_low_frequency_sigma == pytest.approx(0.2)
    assert args.spectral_high_frequency_gain == pytest.approx(0.85)
    assert args.spectral_high_frequency_cutoff == pytest.approx(0.25)
    assert args.wavelet_name == "haar"
    assert args.wavelet_levels == 3
    assert args.wavelet_packet_best_basis is True
    assert args.wavelet_packet_prune_ratio == pytest.approx(1e-3)
    assert args.wavelet_packet_spectral_subtraction is True
    assert args.wavelet_packet_stationary is True
    assert args.wavelet_packet_cycle_spins == 2
    assert args.log_every == 100
    assert args.save_every == 100
    assert args.eval_every == 100
    assert args.eval_batches == 8
    assert args.benchmark_examples == 128
    assert args.benchmark_every == 200
    assert args.skip_online_demo is True


def test_train_reciprocator_only_fresh_run_defaults_to_streaming_runtime_mode() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")

    args = train_script._build_arg_parser().parse_args([])

    training_mode = args.training_mode or None or train_script.DEFAULT_FRESH_TRAINING_MODE
    stream_reset_policy = args.stream_reset_policy or None or train_script.DEFAULT_FRESH_STREAM_RESET_POLICY
    lr_schedule = args.lr_schedule or None or train_script.DEFAULT_FRESH_LR_SCHEDULE
    warmup_fraction = (
        train_script.DEFAULT_FRESH_WARMUP_FRACTION if args.warmup_fraction is None else args.warmup_fraction
    )
    min_lr_ratio = train_script.DEFAULT_FRESH_MIN_LR_RATIO if args.min_lr_ratio is None else args.min_lr_ratio

    assert args.training_mode is None
    assert args.stream_reset_policy is None
    assert args.lr_schedule is None
    assert args.warmup_fraction is None
    assert args.min_lr_ratio is None
    assert training_mode == "streaming"
    assert stream_reset_policy == "wrap"
    assert lr_schedule == "cosine"
    assert warmup_fraction == pytest.approx(0.02)
    assert min_lr_ratio == pytest.approx(0.1)


def test_train_reciprocator_only_can_opt_back_into_online_demo() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")

    args = train_script._build_arg_parser().parse_args(["--run-online-demo"])

    assert args.skip_online_demo is False


def test_train_reciprocator_only_can_disable_optional_growth_and_mixer_flags() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")

    args = train_script._build_arg_parser().parse_args(
        [
            "--no-dynamic-rank",
            "--normalization",
            "per_mode",
            "--no-learned-per-mode-scaling",
            "--no-learnable-prediction-eta",
            "--no-learnable-coupling-temperature",
            "--no-learned-normalization-blend",
        ]
    )

    assert args.normalization == "per_mode"
    assert args.dynamic_rank is False
    assert args.learned_per_mode_scaling is False
    assert args.learnable_prediction_eta is False
    assert args.learnable_coupling_temperature is False
    assert args.learned_normalization_blend is False


def test_train_reciprocator_only_can_override_spectral_flags() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")

    args = train_script._build_arg_parser().parse_args(
        [
            "--no-use-spectral-reciprocation",
            "--no-learnable-spectral-reciprocation",
            "--spectral-mode",
            "fft",
            "--joint-spectral-mode",
            "--spectral-low-frequency-gain",
            "0.3",
            "--spectral-low-frequency-sigma",
            "0.4",
            "--spectral-high-frequency-gain",
            "0.7",
            "--spectral-high-frequency-cutoff",
            "0.2",
            "--wavelet-name",
            "db1",
            "--wavelet-levels",
            "4",
            "--no-wavelet-packet-best-basis",
            "--wavelet-packet-prune-ratio",
            "0.02",
            "--no-wavelet-packet-spectral-subtraction",
            "--no-wavelet-packet-stationary",
            "--wavelet-packet-cycle-spins",
            "3",
        ]
    )

    assert args.use_spectral_reciprocation is False
    assert args.learnable_spectral_reciprocation is False
    assert args.spectral_mode == "fft"
    assert args.joint_spectral_mode is True
    assert args.spectral_low_frequency_gain == pytest.approx(0.3)
    assert args.spectral_low_frequency_sigma == pytest.approx(0.4)
    assert args.spectral_high_frequency_gain == pytest.approx(0.7)
    assert args.spectral_high_frequency_cutoff == pytest.approx(0.2)
    assert args.wavelet_name == "db1"
    assert args.wavelet_levels == 4
    assert args.wavelet_packet_best_basis is False
    assert args.wavelet_packet_prune_ratio == pytest.approx(0.02)
    assert args.wavelet_packet_spectral_subtraction is False
    assert args.wavelet_packet_stationary is False
    assert args.wavelet_packet_cycle_spins == 3


def test_train_reciprocator_only_all_learnable_mixer_params_enables_optional_controls() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")

    args = train_script._build_arg_parser().parse_args(["--all-learnable-mixer-params"])
    resolved = train_script._resolve_optional_learnable_mixer_args(args)

    assert resolved.learnable_prediction_eta is True
    assert resolved.learnable_coupling_temperature is True
    assert resolved.learned_per_mode_scaling is True
    assert resolved.learned_normalization_blend is True


def test_run_settling_experiment_records_raw_args_and_persist_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settling_script = _load_script_module("run_settling_experiment.py")

    class _DummyProcessor:
        def serialized_model_proto(self) -> bytes:
            return b"dummy-model"

    class _DummyTokenizer:
        vocab_size = 32
        processor = _DummyProcessor()

        def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
            del text, add_bos, add_eos
            return list(range(32))

    class _DummyLM(torch.nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.config = config
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids: torch.Tensor, targets=None):
            batch, steps = input_ids.shape
            vocab_size = int(getattr(self.config, "vocab_size", 32))
            logits = self.weight.view(1, 1, 1).expand(batch, steps, vocab_size)
            loss = logits.sum() * 0.0
            return logits, loss

    metadata_seen: list[dict[str, object]] = []

    def _fake_train_causal_language_model(*args, **kwargs):
        del args
        metadata_seen.append(kwargs["checkpoint_metadata"])
        return {
            "val_metrics": {"loss": 1.0, "perplexity": 2.0},
            "loss_history": [{"tokens_seen": 4, "val_loss": 1.0}],
        }

    monkeypatch.setattr(settling_script, "read_corpus_text", lambda corpus: "alpha beta gamma")
    monkeypatch.setattr(settling_script, "_load_or_train_tokenizer", lambda args, text: _DummyTokenizer())
    monkeypatch.setattr(
        settling_script,
        "split_train_val_tokens",
        lambda token_ids, seq_len, val_fraction: (
            torch.arange(12, dtype=torch.long) % 8,
            torch.arange(12, dtype=torch.long) % 8,
        ),
    )
    monkeypatch.setattr(settling_script, "ReciprocatorOnlyLM", _DummyLM)
    monkeypatch.setattr(settling_script, "SmallMambaLM", _DummyLM)
    monkeypatch.setattr(settling_script, "count_trainable_parameters", lambda model: 1)
    monkeypatch.setattr(settling_script, "estimate_reciprocator_only_train_flops", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(settling_script, "estimate_small_mamba_train_flops", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        settling_script,
        "_match_mamba_budget",
        lambda **kwargs: {
            "model_dim": 32,
            "num_layers": 1,
            "parameter_count": 1,
            "target_parameter_count": 1,
            "parameter_relative_gap": 0.0,
            "train_flops_per_step": 1.0,
            "target_train_flops_per_step": 1.0,
            "train_flops_relative_gap": 0.0,
        },
    )
    monkeypatch.setattr(settling_script, "load_scan_length_split", lambda cache_dir: ([("x", "y")], [("x", "y")]))
    monkeypatch.setattr(settling_script, "build_scan_symbol_table", lambda *args, **kwargs: {})
    monkeypatch.setattr(settling_script, "encode_scan_examples", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(settling_script, "train_causal_language_model", _fake_train_causal_language_model)
    monkeypatch.setattr(
        settling_script,
        "evaluate_benchmark_suite_generic",
        lambda *args, **kwargs: {
            "compositional_binding": 0.1,
            "role_rebinding": 0.2,
            "controlled_novelty": 0.3,
        },
    )
    monkeypatch.setattr(settling_script, "train_scan", lambda *args, **kwargs: {"steps": 1})
    monkeypatch.setattr(
        settling_script,
        "evaluate_scan",
        lambda *args, **kwargs: {"loss": 1.0, "token_accuracy": 0.5, "exact_match": 0.25},
    )
    monkeypatch.setattr(settling_script, "_write_loss_curve_svg", lambda *args, **kwargs: None)
    monkeypatch.setattr(settling_script, "_write_summary_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(settling_script, "_interpret_results", lambda *args, **kwargs: ["ok"])

    report_path = tmp_path / "settling_report.json"
    argv = [
        "--device",
        "cpu",
        "--steps",
        "1",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--eval-batches",
        "1",
        "--benchmark-examples",
        "1",
        "--scan-steps",
        "1",
        "--scan-batch-size",
        "1",
        "--scan-eval-batch-size",
        "1",
        "--persist-state",
        "--report-file",
        str(report_path),
    ]

    settling_script.main(argv)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["raw_args"] == argv
    assert report["models"]["reciprocator_rank3"]["config"]["persist_state"] is True
    assert report["models"]["reciprocator_rank3_selected"]["config"]["persist_state"] is True
    assert report["models"]["reciprocator_rank1_control"]["config"]["persist_state"] is True
    assert metadata_seen
    assert all(metadata["raw_args"] == argv for metadata in metadata_seen)
    reciprocator_metadata = [metadata for metadata in metadata_seen if metadata["model_name"] != "mamba"]
    assert reciprocator_metadata
    assert all(metadata["config"]["persist_state"] is True for metadata in reciprocator_metadata)


def test_capacity_matched_sweep_defaults_to_full_support() -> None:
    sweep_script = _load_script_module("capacity_matched_sweep.py")

    init_mode_sizes, max_mode_sizes = sweep_script._resolve_mode_sizes(
        state_rank=3,
        init_mode_sizes=None,
        max_mode_sizes=None,
    )

    assert init_mode_sizes == (2, 4, 8)
    assert max_mode_sizes == (2, 4, 8)


def test_train_reciprocator_only_snapshot_skips_base_extra_state_hooks() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")
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

    snapshot = train_script._snapshot_runtime_state(model)

    assert snapshot["model"] is not None
    assert snapshot["mixers"] == [None]
    train_script._restore_runtime_state(model, snapshot)


def test_train_reciprocator_only_lr_multiplier_handles_constant_and_cosine() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")

    assert train_script._lr_multiplier(
        step=5,
        total_steps=100,
        schedule="constant",
        warmup_fraction=0.1,
        min_lr_ratio=0.2,
    ) == pytest.approx(1.0)
    assert train_script._lr_multiplier(
        step=1,
        total_steps=100,
        schedule="cosine",
        warmup_fraction=0.1,
        min_lr_ratio=0.2,
    ) == pytest.approx(0.1)
    assert train_script._lr_multiplier(
        step=10,
        total_steps=100,
        schedule="cosine",
        warmup_fraction=0.1,
        min_lr_ratio=0.2,
    ) == pytest.approx(1.0)
    assert train_script._lr_multiplier(
        step=100,
        total_steps=100,
        schedule="cosine",
        warmup_fraction=0.1,
        min_lr_ratio=0.2,
    ) == pytest.approx(0.2)
    taper_start = train_script._lr_multiplier(
        step=101,
        total_steps=150,
        schedule="cosine",
        warmup_fraction=0.0,
        min_lr_ratio=0.0,
        step_offset=100,
    )
    assert taper_start < 1.0
    assert taper_start > 0.99
    assert train_script._lr_multiplier(
        step=150,
        total_steps=150,
        schedule="cosine",
        warmup_fraction=0.0,
        min_lr_ratio=0.0,
        step_offset=100,
    ) == pytest.approx(0.0)


def test_train_reciprocator_only_set_optimizer_lr_updates_param_groups() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")
    layer = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)

    train_script._set_optimizer_lr(optimizer, 2.5e-4)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)


def _make_streaming_test_model() -> ReciprocatorOnlyLM:
    model = ReciprocatorOnlyLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=12,
            dim=24,
            n_layers=1,
            n_heads=4,
            state_rank=3,
            init_mode_sizes=(2, 2, 2),
            max_mode_sizes=(2, 2, 2),
            num_cube_engines=1,
            normalization="frobenius",
            dropout=0.0,
        )
    )
    return model.train()


def test_train_reciprocator_only_streaming_step_uses_tbptt_windows() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")
    model = _make_streaming_test_model()
    train_script._set_persistent_training_mode(model, enabled=True, reset_state=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    stream = train_script.ContiguousTokenStream(torch.arange(32, dtype=torch.long) % 32, seq_len=4)

    train_loss, backward_passes = train_script._streaming_train_step(
        model=model,
        train_stream=stream,
        device=torch.device("cpu"),
        chunks_per_step=5,
        stream_reset_policy="never",
        tbptt_horizon=2,
    )

    assert train_loss > 0.0
    assert backward_passes == 3
    assert model.blocks[0].mixer._track_persistent_state_gradients is True
    assert model.blocks[0].mixer._persistent_state_reals is not None
    assert model.blocks[0].mixer._persistent_state_reals.grad_fn is None
    grad = model.blocks[0].mixer.signal_proj.weight_real.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0


def test_train_reciprocator_only_streaming_step_defaults_to_single_backward_without_tbptt() -> None:
    train_script = _load_script_module("train_reciprocator_only.py")
    model = _make_streaming_test_model()
    train_script._set_persistent_training_mode(model, enabled=True, reset_state=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    stream = train_script.ContiguousTokenStream(torch.arange(32, dtype=torch.long) % 32, seq_len=4)

    train_loss, backward_passes = train_script._streaming_train_step(
        model=model,
        train_stream=stream,
        device=torch.device("cpu"),
        chunks_per_step=5,
        stream_reset_policy="never",
        tbptt_horizon=0,
    )

    assert train_loss > 0.0
    assert backward_passes == 1
    assert model.blocks[0].mixer._track_persistent_state_gradients is False
    grad = model.blocks[0].mixer.signal_proj.weight_real.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0


def test_run_rank_ablation_builds_distinct_rank_commands() -> None:
    ablation_script = _load_script_module("run_rank_ablation.py")
    output_dir = Path("/tmp/rank_ablation")

    specs = ablation_script.build_run_specs(
        ranks=(1, 2, 3),
        init_state_capacity=27,
        state_capacity=64,
        run_prefix="tensor_lift",
        output_dir=output_dir,
        train_args=["--steps", "5", "--seed", "7"],
    )

    assert [spec.rank for spec in specs] == [1, 2, 3]
    assert specs[0].init_mode_sizes == (27,)
    assert specs[1].max_mode_sizes == (4, 16)
    assert specs[2].best_checkpoint == output_dir / "tensor_lift_r3_best.pt"
    assert "--steps" in specs[1].command
    assert "--state-rank" in specs[2].command


def test_run_rank_ablation_rejects_reserved_train_args() -> None:
    ablation_script = _load_script_module("run_rank_ablation.py")

    with pytest.raises(ValueError, match="override"):
        ablation_script.build_run_specs(
            ranks=(1, 2, 3),
            init_state_capacity=27,
            state_capacity=64,
            run_prefix="tensor_lift",
            output_dir=Path("/tmp/rank_ablation"),
            train_args=["--state-rank", "2"],
        )
