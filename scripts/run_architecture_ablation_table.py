"""Plan or run a single Reciprocator ablation table plus matched baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reciprocator_lm import (  # noqa: E402
    BaselineTransformerConfig,
    ModelConfig,
    OnlineReplayBuffer,
    PlainTransformerLM,
    ReciprocatorOnlyLM,
    SentencePieceTokenizer,
    SmallMambaConfig,
    SmallMambaLM,
    TrainingRunConfig,
    available_corpora,
    build_default_benchmark_suite,
    collect_online_episode,
    compute_sleep_loss,
    count_trainable_parameters,
    evaluate_benchmark_suite_generic,
    estimate_plain_transformer_train_flops,
    estimate_reciprocator_only_train_flops,
    estimate_small_mamba_train_flops,
    read_corpus_text,
    sample_causal_lm_batch,
    sample_replay_batch,
    select_mode_size_pair,
    split_train_val_tokens,
    train_causal_language_model,
    train_sentencepiece_tokenizer,
)
from reciprocator_lm.runtime import add_device_argument, resolve_torch_device  # noqa: E402


DEFAULT_CORPUS = "greek_philosophy_classics"
DEFAULT_VOCAB_SIZE = 2048
DEFAULT_STATE_CAPACITY = 64
DEFAULT_TOKENIZER_PREFIX = ROOT / "runs" / "architecture_ablation_tokenizer"


@dataclass(frozen=True)
class GridBudgetMatch:
    family: str
    model_dim: int
    num_layers: int
    parameter_count: int
    target_parameter_count: int
    parameter_relative_gap: float
    train_flops_per_step: float
    target_train_flops_per_step: float
    train_flops_relative_gap: float


@dataclass(frozen=True)
class ExperimentRow:
    name: str
    family: str
    ablation: str
    description: str
    sleep_enabled: bool
    parameter_count: int
    train_flops_per_step: float
    target_parameter_count: Optional[int]
    parameter_relative_gap: Optional[float]
    target_train_flops_per_step: Optional[float]
    train_flops_relative_gap: Optional[float]
    config_payload: dict[str, Any]


def _parse_size_tuple(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("mode sizes must be a comma-separated list of positive integers")
    try:
        sizes = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mode sizes must be integers") from exc
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("mode sizes must be positive")
    return sizes


def _candidate_dims(min_dim: int, max_dim: int, step: int, *, multiple_of: int = 1) -> list[int]:
    if min_dim <= 0 or max_dim < min_dim or step <= 0:
        raise ValueError("invalid search range for candidate dims")
    dims = [dim for dim in range(min_dim, max_dim + 1, step) if dim % multiple_of == 0]
    if not dims:
        raise ValueError("candidate dim search produced no valid values")
    return dims


def _candidate_layers(anchor_layers: int) -> list[int]:
    if anchor_layers <= 0:
        raise ValueError("anchor_layers must be positive")
    return list(range(1, max(anchor_layers + 4, 8) + 1))


def _default_output_prefix() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return ROOT / "runs" / f"architecture_ablation_{timestamp}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _probe_macro_average(metrics: dict[str, dict[str, float]] | dict[str, float]) -> float:
    if not metrics:
        return 0.0
    values = []
    for item in metrics.values():
        if isinstance(item, dict):
            values.append(float(item["accuracy"]))
        else:
            values.append(float(item))
    return sum(values) / len(values)


def _normalize_benchmark_metrics(metrics: dict[str, float] | dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for name, value in metrics.items():
        if isinstance(value, dict):
            normalized[name] = {key: float(item) for key, item in value.items()}
        else:
            normalized[name] = {
                "accuracy": float(value),
                "loss": float("nan"),
                "tokens": 0.0,
                "examples": 0.0,
            }
    return normalized


def _resolve_mode_sizes(
    *,
    state_rank: int,
    max_state_rank: int,
    init_mode_sizes: Optional[tuple[int, ...]],
    max_mode_sizes: Optional[tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    _, resolved_max_mode_sizes = select_mode_size_pair(
        state_rank=max_state_rank,
        init_mode_sizes=None,
        max_mode_sizes=max_mode_sizes,
        init_capacity=None,
        max_capacity=DEFAULT_STATE_CAPACITY if max_mode_sizes is None else None,
    )
    if init_mode_sizes is None:
        if max_state_rank == state_rank:
            return resolved_max_mode_sizes, resolved_max_mode_sizes
        return (
            resolved_max_mode_sizes[:state_rank] + (1,) * (max_state_rank - state_rank),
            resolved_max_mode_sizes,
        )
    if len(init_mode_sizes) == max_state_rank:
        return init_mode_sizes, resolved_max_mode_sizes
    if len(init_mode_sizes) == state_rank:
        return init_mode_sizes + (1,) * (max_state_rank - state_rank), resolved_max_mode_sizes
    raise ValueError("init_mode_sizes length must match state_rank or max_state_rank")


def _available_corpus_names() -> list[str]:
    return [corpus.name for corpus in available_corpora()]


def _benchmark_names(vocab_size: int) -> tuple[str, ...]:
    return tuple(benchmark.name for benchmark in build_default_benchmark_suite(vocab_size))


def _build_anchor_config(args: argparse.Namespace, *, vocab_size: int) -> ModelConfig:
    max_state_rank = args.max_state_rank if args.max_state_rank is not None else args.state_rank
    init_mode_sizes, max_mode_sizes = _resolve_mode_sizes(
        state_rank=args.state_rank,
        max_state_rank=max_state_rank,
        init_mode_sizes=args.init_mode_sizes,
        max_mode_sizes=args.max_mode_sizes,
    )
    return ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        state_rank=args.state_rank,
        max_state_rank=max_state_rank,
        dynamic_rank=True,
        init_mode_sizes=init_mode_sizes,
        max_mode_sizes=max_mode_sizes,
        num_cube_engines=args.num_cube_engines,
        normalization=args.normalization,
        growth_threshold=args.growth_threshold,
        growth_interval=args.growth_interval,
        persist_state=False,
        parallel_mixer=False,
        input_dependent_gains=True,
        accumulator_modulates_gains=True,
        phase_aware_readout=True,
        phase_aware_coupling=True,
        mode_coupling_layout="full",
        mode_coupling_schedule="sequential",
        coupling_temperature=1.0,
        use_spectral_reciprocation=True,
        learnable_spectral_reciprocation=True,
    )


def _match_transformer_budget(
    *,
    vocab_size: int,
    max_seq_len: int,
    dropout: float,
    batch_size: int,
    seq_len: int,
    target_parameter_count: int,
    target_train_flops_per_step: float,
    transformer_heads: int,
    transformer_ffw_multiplier: int,
    candidate_dims: Sequence[int],
    candidate_layers: Sequence[int],
) -> tuple[BaselineTransformerConfig, GridBudgetMatch]:
    best: Optional[tuple[BaselineTransformerConfig, GridBudgetMatch]] = None
    for num_layers in candidate_layers:
        for model_dim in candidate_dims:
            config = BaselineTransformerConfig(
                vocab_size=vocab_size,
                model_dim=model_dim,
                num_heads=transformer_heads,
                num_layers=num_layers,
                ffw_multiplier=transformer_ffw_multiplier,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            model = PlainTransformerLM(config)
            parameter_count = count_trainable_parameters(model)
            train_flops_per_step = estimate_plain_transformer_train_flops(
                config,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            match = GridBudgetMatch(
                family="transformer",
                model_dim=model_dim,
                num_layers=num_layers,
                parameter_count=parameter_count,
                target_parameter_count=target_parameter_count,
                parameter_relative_gap=abs(parameter_count - target_parameter_count) / target_parameter_count,
                train_flops_per_step=train_flops_per_step,
                target_train_flops_per_step=target_train_flops_per_step,
                train_flops_relative_gap=abs(train_flops_per_step - target_train_flops_per_step)
                / target_train_flops_per_step,
            )
            if best is None or (
                max(match.parameter_relative_gap, match.train_flops_relative_gap),
                match.parameter_relative_gap,
                match.train_flops_relative_gap,
                abs(match.parameter_count - match.target_parameter_count),
                abs(match.train_flops_per_step - match.target_train_flops_per_step),
                match.num_layers,
                match.model_dim,
            ) < (
                max(best[1].parameter_relative_gap, best[1].train_flops_relative_gap),
                best[1].parameter_relative_gap,
                best[1].train_flops_relative_gap,
                abs(best[1].parameter_count - best[1].target_parameter_count),
                abs(best[1].train_flops_per_step - best[1].target_train_flops_per_step),
                best[1].num_layers,
                best[1].model_dim,
            ):
                best = (config, match)
    if best is None:
        raise ValueError("failed to match transformer budget")
    return best


def _match_mamba_budget(
    *,
    vocab_size: int,
    max_seq_len: int,
    dropout: float,
    batch_size: int,
    seq_len: int,
    target_parameter_count: int,
    target_train_flops_per_step: float,
    mamba_state_size: int,
    mamba_expand: int,
    mamba_conv_kernel: int,
    candidate_dims: Sequence[int],
    candidate_layers: Sequence[int],
) -> tuple[SmallMambaConfig, GridBudgetMatch]:
    best: Optional[tuple[SmallMambaConfig, GridBudgetMatch]] = None
    for num_layers in candidate_layers:
        for model_dim in candidate_dims:
            config = SmallMambaConfig(
                vocab_size=vocab_size,
                model_dim=model_dim,
                num_layers=num_layers,
                state_size=mamba_state_size,
                expand=mamba_expand,
                conv_kernel=mamba_conv_kernel,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            model = SmallMambaLM(config)
            parameter_count = count_trainable_parameters(model)
            train_flops_per_step = estimate_small_mamba_train_flops(
                config,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            match = GridBudgetMatch(
                family="mamba",
                model_dim=model_dim,
                num_layers=num_layers,
                parameter_count=parameter_count,
                target_parameter_count=target_parameter_count,
                parameter_relative_gap=abs(parameter_count - target_parameter_count) / target_parameter_count,
                train_flops_per_step=train_flops_per_step,
                target_train_flops_per_step=target_train_flops_per_step,
                train_flops_relative_gap=abs(train_flops_per_step - target_train_flops_per_step)
                / target_train_flops_per_step,
            )
            if best is None or (
                max(match.parameter_relative_gap, match.train_flops_relative_gap),
                match.parameter_relative_gap,
                match.train_flops_relative_gap,
                abs(match.parameter_count - match.target_parameter_count),
                abs(match.train_flops_per_step - match.target_train_flops_per_step),
                match.num_layers,
                match.model_dim,
            ) < (
                max(best[1].parameter_relative_gap, best[1].train_flops_relative_gap),
                best[1].parameter_relative_gap,
                best[1].train_flops_relative_gap,
                abs(best[1].parameter_count - best[1].target_parameter_count),
                abs(best[1].train_flops_per_step - best[1].target_train_flops_per_step),
                best[1].num_layers,
                best[1].model_dim,
            ):
                best = (config, match)
    if best is None:
        raise ValueError("failed to match mamba budget")
    return best


def build_experiment_rows(args: argparse.Namespace, *, vocab_size: int) -> tuple[ExperimentRow, ...]:
    anchor_config = _build_anchor_config(args, vocab_size=vocab_size)
    anchor_model = ReciprocatorOnlyLM(anchor_config)
    anchor_parameter_count = count_trainable_parameters(anchor_model)
    anchor_train_flops = estimate_reciprocator_only_train_flops(
        anchor_config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    rows: list[ExperimentRow] = []

    def add_reciprocator_row(
        *,
        name: str,
        ablation: str,
        description: str,
        sleep_enabled: bool = False,
        **overrides: Any,
    ) -> None:
        config = replace(anchor_config, **overrides)
        model = ReciprocatorOnlyLM(config)
        rows.append(
            ExperimentRow(
                name=name,
                family="reciprocator",
                ablation=ablation,
                description=description,
                sleep_enabled=sleep_enabled,
                parameter_count=count_trainable_parameters(model),
                train_flops_per_step=estimate_reciprocator_only_train_flops(
                    config,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                ),
                target_parameter_count=None,
                parameter_relative_gap=None,
                target_train_flops_per_step=None,
                train_flops_relative_gap=None,
                config_payload=asdict(config),
            )
        )

    add_reciprocator_row(
        name="reciprocator_full",
        ablation="baseline",
        description="Full Reciprocator reference row.",
    )
    add_reciprocator_row(
        name="reciprocator_no_spectral",
        ablation="spectral_off",
        description="Disable spectral reciprocation and its learned spectral gains.",
        use_spectral_reciprocation=False,
        learnable_spectral_reciprocation=False,
    )
    add_reciprocator_row(
        name="reciprocator_diagonal_coupling",
        ablation="diagonal_coupling",
        description="Restrict mode couplings to diagonal self-routing only.",
        mode_coupling_layout="diagonal",
    )
    add_reciprocator_row(
        name="reciprocator_independent_couplings",
        ablation="independent_couplings",
        description="Derive each mode coupling independently instead of sequentially composing them.",
        mode_coupling_schedule="independent",
    )
    add_reciprocator_row(
        name="reciprocator_no_dynamic_rank",
        ablation="dynamic_rank_off",
        description="Disable novelty-driven rank growth while keeping the same max manifold.",
        dynamic_rank=False,
    )
    add_reciprocator_row(
        name="reciprocator_no_input_gains",
        ablation="input_dependent_gains_off",
        description="Disable input-dependent gain prediction.",
        input_dependent_gains=False,
    )
    add_reciprocator_row(
        name="reciprocator_sleep_on",
        ablation="sleep_on",
        description="Run the full Reciprocator plus post-wake sleep consolidation.",
        sleep_enabled=True,
    )

    transformer_config, transformer_match = _match_transformer_budget(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        target_parameter_count=anchor_parameter_count,
        target_train_flops_per_step=anchor_train_flops,
        transformer_heads=args.transformer_heads,
        transformer_ffw_multiplier=args.transformer_ffw_multiplier,
        candidate_dims=_candidate_dims(
            args.search_min_dim,
            args.search_max_dim,
            args.search_step,
            multiple_of=args.transformer_heads,
        ),
        candidate_layers=_candidate_layers(args.layers),
    )
    rows.append(
        ExperimentRow(
            name="transformer_matched",
            family="transformer",
            ablation="matched_baseline",
            description="Parameter-and-FLOP matched plain Transformer baseline.",
            sleep_enabled=False,
            parameter_count=transformer_match.parameter_count,
            train_flops_per_step=transformer_match.train_flops_per_step,
            target_parameter_count=transformer_match.target_parameter_count,
            parameter_relative_gap=transformer_match.parameter_relative_gap,
            target_train_flops_per_step=transformer_match.target_train_flops_per_step,
            train_flops_relative_gap=transformer_match.train_flops_relative_gap,
            config_payload=asdict(transformer_config),
        )
    )

    mamba_config, mamba_match = _match_mamba_budget(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        target_parameter_count=anchor_parameter_count,
        target_train_flops_per_step=anchor_train_flops,
        mamba_state_size=args.mamba_state_size,
        mamba_expand=args.mamba_expand,
        mamba_conv_kernel=args.mamba_conv_kernel,
        candidate_dims=_candidate_dims(args.search_min_dim, args.search_max_dim, args.search_step),
        candidate_layers=_candidate_layers(args.layers),
    )
    rows.append(
        ExperimentRow(
            name="mamba_matched",
            family="mamba",
            ablation="matched_baseline",
            description="Parameter-and-FLOP matched small Mamba baseline.",
            sleep_enabled=False,
            parameter_count=mamba_match.parameter_count,
            train_flops_per_step=mamba_match.train_flops_per_step,
            target_parameter_count=mamba_match.target_parameter_count,
            parameter_relative_gap=mamba_match.parameter_relative_gap,
            target_train_flops_per_step=mamba_match.target_train_flops_per_step,
            train_flops_relative_gap=mamba_match.train_flops_relative_gap,
            config_payload=asdict(mamba_config),
        )
    )

    return tuple(rows)


def _instantiate_model(row: ExperimentRow) -> torch.nn.Module:
    if row.family == "reciprocator":
        return ReciprocatorOnlyLM(ModelConfig(**row.config_payload))
    if row.family == "transformer":
        return PlainTransformerLM(BaselineTransformerConfig(**row.config_payload))
    if row.family == "mamba":
        return SmallMambaLM(SmallMambaConfig(**row.config_payload))
    raise ValueError(f"unsupported family: {row.family}")


def _load_or_train_tokenizer(args: argparse.Namespace, text: str) -> SentencePieceTokenizer:
    if args.tokenizer_model is not None:
        return SentencePieceTokenizer.from_model_file(args.tokenizer_model)
    cached_model_path = args.tokenizer_prefix.with_suffix(".model")
    if cached_model_path.is_file():
        return SentencePieceTokenizer.from_model_file(cached_model_path)
    return train_sentencepiece_tokenizer(
        text=text,
        vocab_size=args.vocab_size,
        model_prefix=args.tokenizer_prefix,
    )


def _disable_online_mode(model: ReciprocatorOnlyLM) -> None:
    model.reset_online_state()
    for block in model.blocks:
        if hasattr(block.mixer, "persist_state"):
            block.mixer.persist_state = False


def _run_sleep_phase(
    *,
    model: ReciprocatorOnlyLM,
    token_ids: Sequence[int],
    device: torch.device,
    benchmark_examples: int,
    seed: int,
    sleep_steps: int,
    sleep_batch_size: int,
    sleep_rehearsal_batch_size: int,
    sleep_lr: float,
    sleep_log_every: int,
    sleep_distill_weight: float,
    sleep_rehearsal_weight: float,
    sleep_temperature: float,
    sleep_seq_len: int,
    sleep_stride: Optional[int],
    sleep_max_chunks: Optional[int],
) -> dict[str, Any]:
    episode = collect_online_episode(
        model.eval(),
        token_ids,
        seq_len=sleep_seq_len,
        episode_id="wake_corpus",
        stride=sleep_stride,
        max_chunks=sleep_max_chunks,
        device=device,
    )
    _disable_online_mode(model)
    buffer = OnlineReplayBuffer(
        episodes=(episode,),
        tokenizer_vocab_size=model.config.vocab_size,
        metadata={"seq_len": int(sleep_seq_len)},
    )
    before_metrics = evaluate_benchmark_suite_generic(
        model,
        vocab_size=model.config.vocab_size,
        num_examples=benchmark_examples,
        device=device,
        seed=seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=sleep_lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for step in range(1, sleep_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        replay_batch = sample_replay_batch(
            buffer,
            sleep_batch_size,
            device=device,
            generator=generator,
        )
        replay_logits, _ = model(replay_batch.input_ids)
        rehearsal_loss = None
        if sleep_rehearsal_weight > 0.0 and sleep_rehearsal_batch_size > 0:
            rehearsal_inputs, rehearsal_targets = sample_causal_lm_batch(
                token_ids,
                sleep_seq_len,
                sleep_rehearsal_batch_size,
                device=device,
                generator=generator,
            )
            _, rehearsal_loss = model(rehearsal_inputs, rehearsal_targets)
        losses = compute_sleep_loss(
            replay_logits,
            replay_batch.target_ids,
            replay_batch.teacher_logits,
            base_loss=rehearsal_loss,
            distillation_weight=sleep_distill_weight,
            base_loss_weight=sleep_rehearsal_weight,
            temperature=sleep_temperature,
        )
        losses.total.backward()
        optimizer.step()
        if step == 1 or step % sleep_log_every == 0 or step == sleep_steps:
            print(
                f"[sleep:{step:04d}] "
                f"total={losses.total.item():.4f} "
                f"wake_ce={losses.wake_ce.item():.4f} "
                f"distill={losses.distillation.item():.4f} "
                f"rehearsal={losses.base_ce.item():.4f}"
            )
    model.eval()
    after_metrics = evaluate_benchmark_suite_generic(
        model,
        vocab_size=model.config.vocab_size,
        num_examples=benchmark_examples,
        device=device,
        seed=seed,
    )
    return {
        "benchmark_before": before_metrics,
        "benchmark_after": after_metrics,
        "benchmark_macro_before": _probe_macro_average(before_metrics),
        "benchmark_macro_after": _probe_macro_average(after_metrics),
        "replay_chunks": len(episode.chunks),
    }


def _write_table_csv(
    *,
    output_path: Path,
    rows: Sequence[ExperimentRow],
    benchmark_names: Sequence[str],
    results: Optional[dict[str, dict[str, Any]]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "family",
        "ablation",
        "description",
        "sleep_enabled",
        "parameter_count",
        "train_flops_per_step",
        "target_parameter_count",
        "parameter_relative_gap",
        "target_train_flops_per_step",
        "train_flops_relative_gap",
        "status",
        "val_loss",
        "val_perplexity",
        "val_token_accuracy",
        "benchmark_macro_average",
    ] + [f"benchmark_{name}" for name in benchmark_names]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            result = None if results is None else results.get(row.name)
            training = None if result is None else result.get("training")
            benchmarks = None if result is None else result.get("benchmarks")
            writer.writerow(
                {
                    "name": row.name,
                    "family": row.family,
                    "ablation": row.ablation,
                    "description": row.description,
                    "sleep_enabled": int(row.sleep_enabled),
                    "parameter_count": row.parameter_count,
                    "train_flops_per_step": row.train_flops_per_step,
                    "target_parameter_count": row.target_parameter_count,
                    "parameter_relative_gap": row.parameter_relative_gap,
                    "target_train_flops_per_step": row.target_train_flops_per_step,
                    "train_flops_relative_gap": row.train_flops_relative_gap,
                    "status": "completed" if result is not None else "planned",
                    "val_loss": None if training is None else training["val_metrics"]["loss"],
                    "val_perplexity": None if training is None else training["val_metrics"]["perplexity"],
                    "val_token_accuracy": None if training is None else training["val_metrics"]["token_accuracy"],
                    "benchmark_macro_average": None if benchmarks is None else _probe_macro_average(benchmarks),
                    **{
                        f"benchmark_{name}": None if benchmarks is None else benchmarks[name]["accuracy"]
                        for name in benchmark_names
                    },
                }
            )


def _select_rows(rows: Sequence[ExperimentRow], selected_names: Sequence[str]) -> tuple[ExperimentRow, ...]:
    if not selected_names:
        return tuple(rows)
    by_name = {row.name: row for row in rows}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"unknown row selection: {joined}")
    return tuple(by_name[name] for name in selected_names)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run a single Reciprocator ablation table.")
    add_device_argument(parser, default="auto")
    parser.add_argument("--execute", action="store_true", help="Run the planned rows instead of only writing the plan.")
    parser.add_argument("--row", action="append", default=None, help="Optional row name to run/plan. Can be passed multiple times.")
    parser.add_argument("--output-prefix", type=Path, default=None, help="Prefix for the plan, table, and results files.")
    parser.add_argument(
        "--corpus",
        action="append",
        choices=_available_corpus_names(),
        help=f"Bundled training corpus. Can be passed multiple times; defaults to {DEFAULT_CORPUS}.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--benchmark-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--tokenizer-model", type=Path, default=None)
    parser.add_argument("--tokenizer-prefix", type=Path, default=DEFAULT_TOKENIZER_PREFIX)
    parser.add_argument("--search-min-dim", type=int, default=32)
    parser.add_argument("--search-max-dim", type=int, default=512)
    parser.add_argument("--search-step", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--state-rank", type=int, default=4)
    parser.add_argument("--max-state-rank", type=int, default=8)
    parser.add_argument("--init-mode-sizes", type=_parse_size_tuple, default=(4, 4, 2, 2))
    parser.add_argument("--max-mode-sizes", type=_parse_size_tuple, default=(8, 8, 4, 4))
    parser.add_argument("--num-cube-engines", type=int, default=4)
    parser.add_argument("--normalization", choices=("frobenius", "per_mode"), default="frobenius")
    parser.add_argument("--growth-threshold", type=float, default=0.02)
    parser.add_argument("--growth-interval", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-ffw-multiplier", type=int, default=4)
    parser.add_argument("--mamba-state-size", type=int, default=16)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-conv-kernel", type=int, default=4)
    parser.add_argument("--sleep-steps", type=int, default=200)
    parser.add_argument("--sleep-batch-size", type=int, default=8)
    parser.add_argument("--sleep-rehearsal-batch-size", type=int, default=8)
    parser.add_argument("--sleep-lr", type=float, default=3e-4)
    parser.add_argument("--sleep-log-every", type=int, default=25)
    parser.add_argument("--sleep-distill-weight", type=float, default=1.0)
    parser.add_argument("--sleep-rehearsal-weight", type=float, default=1.0)
    parser.add_argument("--sleep-temperature", type=float, default=1.0)
    parser.add_argument("--sleep-seq-len", type=int, default=64)
    parser.add_argument("--sleep-stride", type=int, default=None)
    parser.add_argument("--sleep-max-chunks", type=int, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1)")
    if args.benchmark_examples <= 0:
        raise ValueError("--benchmark-examples must be positive")
    if args.sleep_steps <= 0:
        raise ValueError("--sleep-steps must be positive")
    if args.sleep_batch_size <= 0:
        raise ValueError("--sleep-batch-size must be positive")
    if args.sleep_rehearsal_batch_size < 0:
        raise ValueError("--sleep-rehearsal-batch-size must be non-negative")
    if args.sleep_seq_len <= 0:
        raise ValueError("--sleep-seq-len must be positive")

    output_prefix = args.output_prefix or _default_output_prefix()
    plan_path = Path(f"{output_prefix}_plan.json")
    table_path = Path(f"{output_prefix}_table.csv")
    results_path = Path(f"{output_prefix}_results.json")

    planning_vocab_size = args.vocab_size
    benchmark_names = _benchmark_names(planning_vocab_size)
    planned_rows = _select_rows(build_experiment_rows(args, vocab_size=planning_vocab_size), args.row or [])
    plan_payload = {
        "script": "run_architecture_ablation_table.py",
        "mode": "execute" if args.execute else "plan_only",
        "args": _json_safe(vars(args)),
        "plan_files": {
            "plan_json": str(plan_path),
            "table_csv": str(table_path),
            "results_json": str(results_path),
        },
        "benchmark_names": benchmark_names,
        "rows": [_json_safe(asdict(row)) for row in planned_rows],
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan_payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_table_csv(output_path=table_path, rows=planned_rows, benchmark_names=benchmark_names)
    print(f"wrote plan to {plan_path}")
    print(f"wrote table scaffold to {table_path}")

    if not args.execute:
        return

    corpora = args.corpus or [DEFAULT_CORPUS]
    text = "\n".join(read_corpus_text(name) for name in corpora)
    tokenizer = _load_or_train_tokenizer(args, text)
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    train_tokens, val_tokens = split_train_val_tokens(token_ids, args.seq_len, args.val_fraction)
    if val_tokens is None:
        raise ValueError("validation split is unavailable for the requested corpus/seq-len combination")

    device = resolve_torch_device(args.device)
    benchmark_names = _benchmark_names(tokenizer.vocab_size)
    rows = _select_rows(build_experiment_rows(args, vocab_size=tokenizer.vocab_size), args.row or [])

    training_config = TrainingRunConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        log_every=args.log_every,
        seed=args.seed,
    )

    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        torch.manual_seed(args.seed)
        model = _instantiate_model(row).to(device)
        training_summary = train_causal_language_model(
            model,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            device=device,
            config=training_config,
            log_prefix=row.name,
        )
        benchmark_metrics = _normalize_benchmark_metrics(
            evaluate_benchmark_suite_generic(
                model,
                vocab_size=tokenizer.vocab_size,
                num_examples=args.benchmark_examples,
                device=device,
                seed=args.seed,
            )
        )
        sleep_summary = None
        if row.sleep_enabled:
            if not isinstance(model, ReciprocatorOnlyLM):
                raise ValueError("sleep is only supported for Reciprocator rows")
            sleep_summary = _run_sleep_phase(
                model=model,
                token_ids=token_ids,
                device=device,
                benchmark_examples=args.benchmark_examples,
                seed=args.seed,
                sleep_steps=args.sleep_steps,
                sleep_batch_size=args.sleep_batch_size,
                sleep_rehearsal_batch_size=args.sleep_rehearsal_batch_size,
                sleep_lr=args.sleep_lr,
                sleep_log_every=args.sleep_log_every,
                sleep_distill_weight=args.sleep_distill_weight,
                sleep_rehearsal_weight=args.sleep_rehearsal_weight,
                sleep_temperature=args.sleep_temperature,
                sleep_seq_len=args.sleep_seq_len,
                sleep_stride=args.sleep_stride,
                sleep_max_chunks=args.sleep_max_chunks,
            )
            benchmark_metrics = _normalize_benchmark_metrics(sleep_summary["benchmark_after"])
        results[row.name] = {
            "row": _json_safe(asdict(row)),
            "training": _json_safe(training_summary),
            "benchmarks": _json_safe(benchmark_metrics),
            "benchmark_macro_average": _probe_macro_average(benchmark_metrics),
            "sleep": None if sleep_summary is None else _json_safe(sleep_summary),
        }
        results_path.write_text(json.dumps(_json_safe(results), indent=2, sort_keys=True), encoding="utf-8")
        _write_table_csv(
            output_path=table_path,
            rows=rows,
            benchmark_names=benchmark_names,
            results=results,
        )
        print(f"completed {row.name}")

    print(f"wrote results to {results_path}")


if __name__ == "__main__":
    main()
