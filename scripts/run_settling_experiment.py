from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Any, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reciprocator_lm import (
    ModelConfig,
    ReciprocatorOnlyLM,
    SCAN_LENGTH_TEST_URL,
    SCAN_LENGTH_TRAIN_URL,
    ScanTransferConfig,
    SentencePieceTokenizer,
    SmallMambaConfig,
    SmallMambaLM,
    TrainingRunConfig,
    build_scan_symbol_table,
    count_trainable_parameters,
    encode_scan_examples,
    estimate_reciprocator_only_train_flops,
    estimate_small_mamba_train_flops,
    evaluate_benchmark_suite_generic,
    evaluate_scan,
    load_scan_length_split,
    read_corpus_text,
    select_mode_size_pair,
    split_train_val_tokens,
    train_causal_language_model,
    train_scan,
    train_sentencepiece_tokenizer,
)
from reciprocator_lm.runtime import add_device_argument, resolve_torch_device


DEFAULT_CORPUS = "greek_philosophy_classics"
DEFAULT_CONTEXT_LENGTH = 128
DEFAULT_VOCAB_SIZE = 2048
DEFAULT_TOKENIZER_PREFIX = ROOT / "runs" / "frozen_threeway_tokenizer"
DEFAULT_SCAN_CACHE_DIR = ROOT / "runs" / "scan_length_cache"
DEFAULT_STATE_CAPACITY = 64
BINDING_TASKS = ("compositional_binding", "role_rebinding", "controlled_novelty")
MODEL_ORDER = (
    "mamba",
    "reciprocator_rank3",
    "reciprocator_rank3_selected",
    "reciprocator_rank1_control",
)
PLOT_COLORS = {
    "mamba": "#ff7f0e",
    "reciprocator_rank3": "#2ca02c",
    "reciprocator_rank3_selected": "#d62728",
    "reciprocator_rank1_control": "#1f77b4",
}
PLOT_LABELS = {
    "mamba": "Mamba baseline",
    "reciprocator_rank3": "Reciprocator rank-3",
    "reciprocator_rank3_selected": "Reciprocator rank-3 + selection",
    "reciprocator_rank1_control": "Reciprocator rank-1 control",
}


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


def _candidate_dims(min_dim: int, max_dim: int, step: int) -> list[int]:
    if min_dim <= 0 or max_dim < min_dim or step <= 0:
        raise ValueError("invalid search range for candidate dims")
    dims = list(range(min_dim, max_dim + 1, step))
    if not dims:
        raise ValueError("candidate dim search produced no valid values")
    return dims


def _candidate_layers(anchor_layers: int) -> list[int]:
    if anchor_layers <= 0:
        raise ValueError("anchor_layers must be positive")
    return list(range(1, max(anchor_layers + 4, 8) + 1))


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


def _probe_macro_average(probes: dict[str, float]) -> float:
    return sum(probes.values()) / len(probes) if probes else 0.0


def _binding_macro_average(probes: dict[str, float]) -> float:
    return sum(probes[task] for task in BINDING_TASKS) / len(BINDING_TASKS)


def _resolve_mode_sizes(
    *,
    state_rank: int,
    max_state_rank: Optional[int] = None,
    init_mode_sizes: Optional[tuple[int, ...]],
    max_mode_sizes: Optional[tuple[int, ...]],
) -> tuple[Optional[tuple[int, ...]], tuple[int, ...]]:
    resolved_max_state_rank = state_rank if max_state_rank is None else max_state_rank
    _, resolved_max_mode_sizes = select_mode_size_pair(
        state_rank=resolved_max_state_rank,
        init_mode_sizes=None,
        max_mode_sizes=max_mode_sizes,
        init_capacity=None,
        max_capacity=DEFAULT_STATE_CAPACITY if max_mode_sizes is None else None,
    )
    if init_mode_sizes is None:
        if resolved_max_state_rank == state_rank:
            return resolved_max_mode_sizes, resolved_max_mode_sizes
        return None, resolved_max_mode_sizes
    if len(init_mode_sizes) == resolved_max_state_rank:
        return init_mode_sizes, resolved_max_mode_sizes
    if len(init_mode_sizes) == state_rank:
        return (
            init_mode_sizes + (1,) * (resolved_max_state_rank - state_rank),
            resolved_max_mode_sizes,
        )
    raise ValueError("init_mode_sizes length must match state_rank or max_state_rank")


def _default_report_path() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return ROOT / "runs" / f"settling_experiment_{timestamp}.json"


def _default_loss_plot_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}_val_loss_vs_tokens.svg")


def _default_summary_csv_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}_summary.csv")


def _write_loss_curve_svg(model_reports: dict[str, dict[str, Any]], output_path: Path) -> None:
    width = 960
    height = 540
    margin_left = 90
    margin_right = 40
    margin_top = 50
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    histories = {name: model_reports[name]["training"]["loss_history"] for name in MODEL_ORDER}
    all_points = [point for history in histories.values() for point in history]
    if not all_points:
        raise ValueError("loss history is empty; cannot render plot.")

    min_x = min(point["tokens_seen"] for point in all_points)
    max_x = max(point["tokens_seen"] for point in all_points)
    min_y = min(point["val_loss"] for point in all_points)
    max_y = max(point["val_loss"] for point in all_points)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0
    y_pad = 0.05 * (max_y - min_y)
    min_y -= y_pad
    max_y += y_pad

    def sx(value: float) -> float:
        return margin_left + ((value - min_x) / (max_x - min_x)) * plot_width

    def sy(value: float) -> float:
        return margin_top + (1.0 - (value - min_y) / (max_y - min_y)) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="22" font-family="Arial, sans-serif">Validation loss vs. tokens seen</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#444" stroke-width="1.5" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#444" stroke-width="1.5" />',
    ]

    for index in range(6):
        frac = index / 5.0
        x_value = min_x + frac * (max_x - min_x)
        x_pos = sx(x_value)
        y_axis = margin_top + plot_height
        elements.append(
            f'<line x1="{x_pos:.2f}" y1="{margin_top}" x2="{x_pos:.2f}" y2="{y_axis}" stroke="#ddd" stroke-width="1" />'
        )
        elements.append(
            f'<text x="{x_pos:.2f}" y="{y_axis + 24}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif">{int(round(x_value)):,}</text>'
        )

        y_value = min_y + frac * (max_y - min_y)
        y_pos = sy(y_value)
        elements.append(
            f'<line x1="{margin_left}" y1="{y_pos:.2f}" x2="{margin_left + plot_width}" y2="{y_pos:.2f}" stroke="#eee" stroke-width="1" />'
        )
        elements.append(
            f'<text x="{margin_left - 12}" y="{y_pos + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial, sans-serif">{y_value:.2f}</text>'
        )

    elements.append(
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif">Tokens seen</text>'
    )
    elements.append(
        f'<text x="24" y="{height / 2}" transform="rotate(-90 24 {height / 2})" text-anchor="middle" font-size="14" font-family="Arial, sans-serif">Validation loss (nats)</text>'
    )

    legend_x = margin_left + plot_width - 250
    legend_y = margin_top + 10
    for index, name in enumerate(MODEL_ORDER):
        y = legend_y + index * 24
        elements.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{PLOT_COLORS[name]}" stroke-width="3" />'
        )
        elements.append(
            f'<text x="{legend_x + 32}" y="{y + 4}" font-size="13" font-family="Arial, sans-serif">{escape(PLOT_LABELS[name])}</text>'
        )

    for name in MODEL_ORDER:
        history = histories[name]
        points = " ".join(f"{sx(point['tokens_seen']):.2f},{sy(point['val_loss']):.2f}" for point in history)
        elements.append(
            f'<polyline fill="none" stroke="{PLOT_COLORS[name]}" stroke-width="3" points="{points}" />'
        )
        for point in history:
            elements.append(
                f'<circle cx="{sx(point["tokens_seen"]):.2f}" cy="{sy(point["val_loss"]):.2f}" r="3.5" fill="{PLOT_COLORS[name]}" />'
            )

    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def _write_summary_csv(model_reports: dict[str, dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "parameter_count",
                "train_flops_per_step",
                "lm_val_loss",
                "lm_val_perplexity",
                "binding_macro_average",
                "suite_macro_average",
                "scan_loss",
                "scan_token_accuracy",
                "scan_exact_match",
            ),
        )
        writer.writeheader()
        for name in MODEL_ORDER:
            report = model_reports[name]
            writer.writerow(
                {
                    "model": name,
                    "parameter_count": report["parameter_count"],
                    "train_flops_per_step": report["train_flops_per_step"],
                    "lm_val_loss": report["training"]["val_metrics"]["loss"],
                    "lm_val_perplexity": report["training"]["val_metrics"]["perplexity"],
                    "binding_macro_average": report["binding_macro_average"],
                    "suite_macro_average": report["probe_macro_average"],
                    "scan_loss": report["scan"]["eval"]["loss"],
                    "scan_token_accuracy": report["scan"]["eval"]["token_accuracy"],
                    "scan_exact_match": report["scan"]["eval"]["exact_match"],
                }
            )


def _match_mamba_budget(
    *,
    target_parameter_count: int,
    target_train_flops_per_step: float,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    max_seq_len: int,
    state_size: int,
    expand: int,
    conv_kernel: int,
    dropout: float,
    candidate_dims: list[int],
    candidate_layers: list[int],
) -> dict[str, float | int]:
    best: Optional[dict[str, float | int]] = None
    for num_layers in candidate_layers:
        for model_dim in candidate_dims:
            config = SmallMambaConfig(
                vocab_size=vocab_size,
                model_dim=model_dim,
                num_layers=num_layers,
                state_size=state_size,
                expand=expand,
                conv_kernel=conv_kernel,
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
            candidate = {
                "model_dim": model_dim,
                "num_layers": num_layers,
                "parameter_count": parameter_count,
                "target_parameter_count": target_parameter_count,
                "parameter_relative_gap": abs(parameter_count - target_parameter_count) / target_parameter_count,
                "train_flops_per_step": train_flops_per_step,
                "target_train_flops_per_step": target_train_flops_per_step,
                "train_flops_relative_gap": abs(train_flops_per_step - target_train_flops_per_step)
                / target_train_flops_per_step,
            }
            if best is None or (
                max(candidate["parameter_relative_gap"], candidate["train_flops_relative_gap"]),
                candidate["parameter_relative_gap"],
                candidate["train_flops_relative_gap"],
                candidate["num_layers"],
                candidate["model_dim"],
            ) < (
                max(best["parameter_relative_gap"], best["train_flops_relative_gap"]),
                best["parameter_relative_gap"],
                best["train_flops_relative_gap"],
                best["num_layers"],
                best["model_dim"],
            ):
                best = candidate
    if best is None:
        raise ValueError("failed to match a Mamba baseline budget")
    return best


def _interpret_results(model_reports: dict[str, dict[str, Any]]) -> list[str]:
    messages: list[str] = []
    rank3 = model_reports["reciprocator_rank3"]
    mamba = model_reports["mamba"]
    selected = model_reports["reciprocator_rank3_selected"]
    rank1 = model_reports["reciprocator_rank1_control"]

    rank3_wins_binding = rank3["binding_macro_average"] > mamba["binding_macro_average"]
    rank3_wins_scan = rank3["scan"]["eval"]["exact_match"] > mamba["scan"]["eval"]["exact_match"]
    rank3_wins_lm = rank3["training"]["val_metrics"]["perplexity"] < mamba["training"]["val_metrics"]["perplexity"]

    if rank3_wins_binding and (not rank3_wins_lm) and (not rank3_wins_scan):
        messages.append("Rank-3 wins the in-house binding suite but loses LM perplexity and SCAN. The suite is likely leaking the architectural prior.")
    elif rank3_wins_binding and rank3_wins_scan:
        messages.append("Rank-3 beats Mamba on the binding suite and SCAN. The binding thesis is materially supported.")

    selected_lm_gap = selected["training"]["val_metrics"]["perplexity"] - mamba["training"]["val_metrics"]["perplexity"]
    rank3_lm_gap = rank3["training"]["val_metrics"]["perplexity"] - mamba["training"]["val_metrics"]["perplexity"]
    selection_helps_lm = selected_lm_gap < rank3_lm_gap
    selection_preserves_binding = selected["binding_macro_average"] + 1e-6 >= rank3["binding_macro_average"] - 0.01
    if selection_helps_lm and selection_preserves_binding:
        messages.append("Input-dependent gains improve LM perplexity relative to rank-3 without materially hurting the binding suite. Selection looks like the missing architectural piece.")

    rank1_binding_gap = abs(rank1["binding_macro_average"] - rank3["binding_macro_average"])
    rank1_scan_gap = abs(rank1["scan"]["eval"]["exact_match"] - rank3["scan"]["eval"]["exact_match"])
    rank1_lm_gap = abs(
        rank1["training"]["val_metrics"]["perplexity"] - rank3["training"]["val_metrics"]["perplexity"]
    ) / rank3["training"]["val_metrics"]["perplexity"]
    if rank1_binding_gap <= 0.02 and rank1_scan_gap <= 0.02 and rank1_lm_gap <= 0.05:
        messages.append("Rank-1 matches rank-3 within a small tolerance across LM, binding, and SCAN. Tensor geometry does not appear to be paying rent.")

    if not messages:
        messages.append("The raw metric gaps do not trigger a single clean diagnosis. Inspect the full report for tradeoffs across LM, binding, and SCAN.")
    return messages


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the settling experiment across four architectures.")
    add_device_argument(parser, default="auto")
    parser.add_argument("--corpus", choices=("plato_jowett", "greek_philosophy_classics"), default=DEFAULT_CORPUS)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--benchmark-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--tokenizer-model", type=Path, default=None)
    parser.add_argument("--tokenizer-prefix", type=Path, default=DEFAULT_TOKENIZER_PREFIX)
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--search-min-dim", type=int, default=32)
    parser.add_argument("--search-max-dim", type=int, default=256)
    parser.add_argument("--search-step", type=int, default=8)

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--state-rank", type=int, default=3)
    parser.add_argument(
        "--max-state-rank",
        type=int,
        default=None,
        help="Maximum supported tensor rank. Defaults to --state-rank for backward-compatible fixed-rank runs.",
    )
    parser.add_argument(
        "--dynamic-rank",
        action="store_true",
        help="Enable novelty-driven rank growth up to --max-state-rank during training.",
    )
    parser.add_argument("--init-mode-sizes", type=_parse_size_tuple, default=None)
    parser.add_argument("--max-mode-sizes", type=_parse_size_tuple, default=None)
    parser.add_argument("--num-cube-engines", type=int, default=4)
    parser.add_argument("--normalization", choices=("frobenius", "per_mode"), default="per_mode")
    parser.add_argument(
        "--learned-per-mode-scaling",
        action="store_true",
        help="Relax per-mode normalization with learned per-mode exponents. Only applies with --normalization=per_mode.",
    )
    parser.add_argument("--serial-mixer", action="store_true", help="Use serial (phase-aware) mixer instead of parallel.")
    parser.add_argument(
        "--persist-state",
        action="store_true",
        help="Keep reciprocator mixer state across training/eval calls instead of resetting every sequence.",
    )
    parser.add_argument("--growth-threshold", type=float, default=0.02)
    parser.add_argument("--growth-interval", type=int, default=1)

    parser.add_argument("--mamba-state-size", type=int, default=16)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-conv-kernel", type=int, default=4)

    parser.add_argument("--scan-cache-dir", type=Path, default=DEFAULT_SCAN_CACHE_DIR)
    parser.add_argument("--scan-steps", type=int, default=100)
    parser.add_argument("--scan-batch-size", type=int, default=32)
    parser.add_argument("--scan-lr", type=float, default=3e-4)
    parser.add_argument("--scan-log-every", type=int, default=25)
    parser.add_argument("--scan-eval-batch-size", type=int, default=128)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1).")
    raw_args = list(sys.argv[1:] if argv is None else argv)

    device = resolve_torch_device(args.device)
    torch.manual_seed(args.seed)

    text = read_corpus_text(args.corpus)
    tokenizer = _load_or_train_tokenizer(args, text)
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    train_tokens, val_tokens = split_train_val_tokens(token_ids, args.seq_len, args.val_fraction)
    if val_tokens is None:
        raise ValueError("validation split is unavailable for the requested corpus/seq-len combination.")

    train_config = TrainingRunConfig(
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
    scan_config = ScanTransferConfig(
        steps=args.scan_steps,
        batch_size=args.scan_batch_size,
        lr=args.scan_lr,
        log_every=args.scan_log_every,
        seed=args.seed,
    )
    report_path = args.report_file or _default_report_path()
    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else report_path.with_suffix("").with_name(f"{report_path.stem}_artifacts")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resolved_init_mode_sizes, resolved_max_mode_sizes = _resolve_mode_sizes(
        state_rank=args.state_rank,
        max_state_rank=args.max_state_rank or args.state_rank,
        init_mode_sizes=args.init_mode_sizes,
        max_mode_sizes=args.max_mode_sizes,
    )
    rank3_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        state_rank=args.state_rank,
        max_state_rank=args.max_state_rank or args.state_rank,
        dynamic_rank=args.dynamic_rank,
        init_mode_sizes=resolved_init_mode_sizes,
        max_mode_sizes=resolved_max_mode_sizes,
        num_cube_engines=args.num_cube_engines,
        normalization=args.normalization,
        learned_per_mode_scaling=args.learned_per_mode_scaling,
        dropout=args.dropout,
        growth_threshold=args.growth_threshold,
        growth_interval=args.growth_interval,
        persist_state=args.persist_state,
        parallel_mixer=not args.serial_mixer,
        input_dependent_gains=False,
        accumulator_modulates_gains=True,
        phase_aware_readout=True,
        phase_aware_coupling=True,
        coupling_temperature=1.0,
    )
    rank3_model = ReciprocatorOnlyLM(rank3_config)
    target_parameters = count_trainable_parameters(rank3_model)
    target_train_flops_per_step = estimate_reciprocator_only_train_flops(
        rank3_config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    mamba_match = _match_mamba_budget(
        target_parameter_count=target_parameters,
        target_train_flops_per_step=target_train_flops_per_step,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        state_size=args.mamba_state_size,
        expand=args.mamba_expand,
        conv_kernel=args.mamba_conv_kernel,
        dropout=args.dropout,
        candidate_dims=_candidate_dims(args.search_min_dim, args.search_max_dim, args.search_step),
        candidate_layers=_candidate_layers(args.layers),
    )
    mamba_config = SmallMambaConfig(
        vocab_size=tokenizer.vocab_size,
        model_dim=int(mamba_match["model_dim"]),
        num_layers=int(mamba_match["num_layers"]),
        state_size=args.mamba_state_size,
        expand=args.mamba_expand,
        conv_kernel=args.mamba_conv_kernel,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    )
    rank3_selected_config = ModelConfig(
        vocab_size=rank3_config.vocab_size,
        max_seq_len=rank3_config.max_seq_len,
        dim=rank3_config.dim,
        n_layers=rank3_config.n_layers,
        n_heads=rank3_config.n_heads,
        mlp_ratio=rank3_config.mlp_ratio,
        state_rank=rank3_config.state_rank,
        max_state_rank=rank3_config.max_state_rank,
        dynamic_rank=rank3_config.dynamic_rank,
        init_mode_sizes=rank3_config.init_mode_sizes,
        max_mode_sizes=rank3_config.max_mode_sizes,
        num_cube_engines=rank3_config.num_cube_engines,
        normalization=rank3_config.normalization,
        impression_rate=rank3_config.impression_rate,
        phase_scale=rank3_config.phase_scale,
        growth_threshold=rank3_config.growth_threshold,
        growth_interval=rank3_config.growth_interval,
        prune_floor=rank3_config.prune_floor,
        prune_horizon=rank3_config.prune_horizon,
        dropout=rank3_config.dropout,
        persist_state=rank3_config.persist_state,
        complex_backbone=rank3_config.complex_backbone,
        parallel_mixer=rank3_config.parallel_mixer,
        input_dependent_gains=True,
        accumulator_modulates_gains=rank3_config.accumulator_modulates_gains,
        phase_aware_readout=rank3_config.phase_aware_readout,
        phase_aware_coupling=rank3_config.phase_aware_coupling,
        coupling_temperature=rank3_config.coupling_temperature,
        training_growth_enabled=rank3_config.training_growth_enabled,
    )
    rank1_control_config = ModelConfig(
        vocab_size=rank3_config.vocab_size,
        max_seq_len=rank3_config.max_seq_len,
        dim=rank3_config.dim,
        n_layers=rank3_config.n_layers,
        n_heads=rank3_config.n_heads,
        mlp_ratio=rank3_config.mlp_ratio,
        state_rank=1,
        max_state_rank=1,
        dynamic_rank=False,
        init_mode_sizes=(rank3_config.state_dim,),
        max_mode_sizes=(rank3_config.state_dim,),
        num_cube_engines=rank3_config.num_cube_engines,
        normalization=rank3_config.normalization,
        impression_rate=rank3_config.impression_rate,
        phase_scale=rank3_config.phase_scale,
        growth_threshold=rank3_config.growth_threshold,
        growth_interval=rank3_config.growth_interval,
        prune_floor=rank3_config.prune_floor,
        prune_horizon=rank3_config.prune_horizon,
        dropout=rank3_config.dropout,
        persist_state=rank3_config.persist_state,
        complex_backbone=rank3_config.complex_backbone,
        parallel_mixer=rank3_config.parallel_mixer,
        input_dependent_gains=False,
        accumulator_modulates_gains=rank3_config.accumulator_modulates_gains,
        phase_aware_readout=rank3_config.phase_aware_readout,
        phase_aware_coupling=rank3_config.phase_aware_coupling,
        coupling_temperature=rank3_config.coupling_temperature,
        training_growth_enabled=rank3_config.training_growth_enabled,
    )

    scan_train_raw, scan_test_raw = load_scan_length_split(args.scan_cache_dir)
    scan_symbol_table = build_scan_symbol_table(
        [*scan_train_raw, *scan_test_raw],
        vocab_size=tokenizer.vocab_size,
    )
    scan_train = encode_scan_examples(scan_train_raw, symbol_table=scan_symbol_table, max_seq_len=args.seq_len)
    scan_test = encode_scan_examples(scan_test_raw, symbol_table=scan_symbol_table, max_seq_len=args.seq_len)

    print(f"Device: {device}")
    print(f"Frozen corpus: {args.corpus}")
    print(f"Tokenized corpus: {len(token_ids):,} tokens (vocab_size={tokenizer.vocab_size})")
    print(f"Train tokens: {train_tokens.numel():,}")
    print(f"Validation tokens: {val_tokens.numel():,}")
    print(f"Frozen context length: {args.seq_len}")
    print(f"Persist state: {args.persist_state}")
    print(f"Target parameters (rank-3 reciprocator): {target_parameters:,}")
    print(f"Target train FLOPs/step (rank-3 reciprocator): {target_train_flops_per_step:,.0f}")
    print(
        f"Matched Mamba dim={mamba_match['model_dim']} layers={mamba_match['num_layers']} "
        f"param_gap={100.0 * mamba_match['parameter_relative_gap']:.2f}% "
        f"flop_gap={100.0 * mamba_match['train_flops_relative_gap']:.2f}%"
    )
    print(f"SCAN length split: train={len(scan_train):,} test={len(scan_test):,}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    model_specs = [
        ("mamba", mamba_config, SmallMambaLM),
        ("reciprocator_rank3", rank3_config, ReciprocatorOnlyLM),
        ("reciprocator_rank3_selected", rank3_selected_config, ReciprocatorOnlyLM),
        ("reciprocator_rank1_control", rank1_control_config, ReciprocatorOnlyLM),
    ]

    model_reports: dict[str, dict[str, Any]] = {}
    for name, config, model_cls in model_specs:
        torch.manual_seed(args.seed)
        model = model_cls(config)
        model_checkpoint_dir = checkpoint_dir / name
        latest_checkpoint_path = model_checkpoint_dir / "latest.pt"
        best_checkpoint_path = model_checkpoint_dir / "best.pt"
        print(f"\n=== Training {name} ===")
        training_summary = train_causal_language_model(
            model,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            device=device,
            config=train_config,
            log_prefix=f"[{name}]",
            latest_checkpoint_path=latest_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            resume_from_checkpoint=args.resume,
            checkpoint_metadata={
                "script": "run_settling_experiment.py",
                "model_name": name,
                "model_class": model_cls.__name__,
                "config": asdict(config),
                "training_config": asdict(train_config),
                "corpus": args.corpus,
                "raw_args": raw_args,
                "tokenizer_model_path": str(args.tokenizer_model or args.tokenizer_prefix.with_suffix(".model")),
                "tokenizer_model_proto": tokenizer.processor.serialized_model_proto(),
            },
        )
        probes = evaluate_benchmark_suite_generic(
            model,
            vocab_size=tokenizer.vocab_size,
            num_examples=args.benchmark_examples,
            device=device,
            seed=args.seed,
        )

        scan_model = model_cls(config)
        scan_model.load_state_dict(model.state_dict())
        print(f"=== SCAN transfer {name} ===")
        scan_train_summary = train_scan(
            scan_model,
            scan_train,
            device=device,
            config=scan_config,
            log_prefix=f"[{name}/scan]",
        )
        scan_eval = evaluate_scan(
            scan_model,
            scan_test,
            device=device,
            batch_size=args.scan_eval_batch_size,
        )

        if isinstance(config, SmallMambaConfig):
            train_flops_per_step = estimate_small_mamba_train_flops(
                config,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )
        else:
            train_flops_per_step = estimate_reciprocator_only_train_flops(
                config,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )

        model_reports[name] = {
            "parameter_count": count_trainable_parameters(model),
            "train_flops_per_step": train_flops_per_step,
            "config": asdict(config),
            "training": training_summary,
            "probes": probes,
            "probe_macro_average": _probe_macro_average(probes),
            "binding_macro_average": _binding_macro_average(probes),
            "scan": {
                "train": scan_train_summary,
                "eval": scan_eval,
            },
        }

    loss_plot_path = _default_loss_plot_path(report_path)
    summary_csv_path = _default_summary_csv_path(report_path)
    _write_loss_curve_svg(model_reports, loss_plot_path)
    _write_summary_csv(model_reports, summary_csv_path)
    interpretation = _interpret_results(model_reports)

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "raw_args": raw_args,
        "frozen_setup": {
            "corpus": args.corpus,
            "tokenizer_kind": "sentencepiece_unigram",
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "context_length": args.seq_len,
            "lm_token_budget_per_model": train_config.steps * train_config.batch_size * train_config.seq_len,
            "scan_transfer_steps": scan_config.steps,
            "scan_split": "length",
        },
        "device": str(device),
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "model_path": str(args.tokenizer_model or args.tokenizer_prefix.with_suffix(".model")),
        },
        "scan_sources": {
            "train_url": SCAN_LENGTH_TRAIN_URL,
            "test_url": SCAN_LENGTH_TEST_URL,
            "cache_dir": str(args.scan_cache_dir),
        },
        "training_config": asdict(train_config),
        "scan_transfer_config": asdict(scan_config),
        "target_parameters": target_parameters,
        "target_train_flops_per_step": target_train_flops_per_step,
        "mamba_match": mamba_match,
        "artifacts": {
            "loss_plot": str(loss_plot_path),
            "summary_csv": str(summary_csv_path),
        },
        "models": model_reports,
        "interpretation": interpretation,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for name in MODEL_ORDER:
        model_report = model_reports[name]
        print(
            f"{name:27s} "
            f"ppl={model_report['training']['val_metrics']['perplexity']:.2f} "
            f"binding={model_report['binding_macro_average']:.4f} "
            f"scan_em={model_report['scan']['eval']['exact_match']:.4f} "
            f"scan_tok={model_report['scan']['eval']['token_accuracy']:.4f}"
        )
    for message in interpretation:
        print(f"Interpretation: {message}")
    print(f"Report: {report_path}")
    print(f"Loss plot: {loss_plot_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
