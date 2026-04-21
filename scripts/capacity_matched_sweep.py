"""Train a frozen 3-way Reciprocator / Mamba / Transformer comparison."""

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
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reciprocator_lm import (
    BaselineTransformerConfig,
    ModelConfig,
    PlainTransformerLM,
    ReciprocatorOnlyLM,
    SmallMambaConfig,
    SmallMambaLM,
    TrainingRunConfig,
    count_trainable_parameters,
    estimate_plain_transformer_train_flops,
    estimate_reciprocator_only_train_flops,
    estimate_small_mamba_train_flops,
    evaluate_benchmark_suite_generic,
    read_corpus_text,
    select_mode_size_pair,
    split_train_val_tokens,
    train_causal_language_model,
    train_sentencepiece_tokenizer,
    SentencePieceTokenizer,
)
from reciprocator_lm.runtime import add_device_argument, resolve_torch_device


DEFAULT_CORPUS = "greek_philosophy_classics"
DEFAULT_CONTEXT_LENGTH = 128
DEFAULT_VOCAB_SIZE = 2048
DEFAULT_TOKENIZER_PREFIX = ROOT / "runs" / "frozen_threeway_tokenizer"
DEFAULT_STATE_CAPACITY = 64
LOSS_GATE_NATS = 0.2
PLOT_COLORS = {
    "transformer": "#1f77b4",
    "mamba": "#ff7f0e",
    "reciprocator": "#2ca02c",
}
PLOT_LABELS = {
    "transformer": "Tiny Transformer",
    "mamba": "Minimal Mamba-1",
    "reciprocator": "ReciprocatorOnlyLM",
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
    upper = max(anchor_layers + 4, 8)
    return list(range(1, upper + 1))


def _probe_macro_average(probes: dict[str, float]) -> float:
    return sum(probes.values()) / len(probes) if probes else 0.0


def _json_safe(value):
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
    return ROOT / "runs" / f"capacity_matched_sweep_{timestamp}.json"


def _default_loss_plot_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}_val_loss_vs_tokens.svg")


def _default_loss_points_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}_val_loss_vs_tokens.csv")


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


def _estimate_plain_transformer_parameter_count(
    *,
    vocab_size: int,
    model_dim: int,
    num_layers: int,
    ffw_multiplier: int,
    max_seq_len: int,
) -> int:
    ffw_dim = model_dim * ffw_multiplier
    per_block = (4 * model_dim * model_dim) + (2 * model_dim * ffw_dim) + ffw_dim + (9 * model_dim)
    return int((vocab_size * model_dim) + (max_seq_len * model_dim) + (num_layers * per_block) + (2 * model_dim))


def _estimate_small_mamba_parameter_count(
    *,
    vocab_size: int,
    model_dim: int,
    num_layers: int,
    state_size: int,
    expand: int,
    conv_kernel: int,
) -> int:
    inner_dim = model_dim * expand
    per_block = (
        (2 * model_dim)
        + (3 * model_dim * inner_dim)
        + (inner_dim * inner_dim)
        + (2 * inner_dim * inner_dim * state_size)
        + (inner_dim * state_size)
        + (inner_dim * conv_kernel)
        + (3 * inner_dim)
    )
    return int((vocab_size * model_dim) + (num_layers * per_block) + (2 * model_dim))


def _match_budget_grid(
    *,
    target_parameter_count: int,
    target_train_flops_per_step: float,
    candidate_dims: list[int],
    candidate_layers: list[int],
    estimate_parameter_count,
    estimate_train_flops,
) -> dict[str, float | int]:
    best_match: Optional[dict[str, float | int]] = None
    for num_layers in candidate_layers:
        for model_dim in candidate_dims:
            parameter_count = int(estimate_parameter_count(model_dim=model_dim, num_layers=num_layers))
            train_flops_per_step = float(estimate_train_flops(model_dim=model_dim, num_layers=num_layers))
            parameter_relative_gap = abs(parameter_count - target_parameter_count) / target_parameter_count
            train_flops_relative_gap = abs(train_flops_per_step - target_train_flops_per_step) / target_train_flops_per_step
            candidate = {
                "model_dim": int(model_dim),
                "num_layers": int(num_layers),
                "parameter_count": parameter_count,
                "target_parameter_count": int(target_parameter_count),
                "parameter_relative_gap": float(parameter_relative_gap),
                "train_flops_per_step": train_flops_per_step,
                "target_train_flops_per_step": float(target_train_flops_per_step),
                "train_flops_relative_gap": float(train_flops_relative_gap),
            }
            if best_match is None or (
                max(candidate["parameter_relative_gap"], candidate["train_flops_relative_gap"]),
                candidate["parameter_relative_gap"],
                candidate["train_flops_relative_gap"],
                abs(candidate["parameter_count"] - candidate["target_parameter_count"]),
                abs(candidate["train_flops_per_step"] - candidate["target_train_flops_per_step"]),
                candidate["num_layers"],
                candidate["model_dim"],
            ) < (
                max(best_match["parameter_relative_gap"], best_match["train_flops_relative_gap"]),
                best_match["parameter_relative_gap"],
                best_match["train_flops_relative_gap"],
                abs(best_match["parameter_count"] - best_match["target_parameter_count"]),
                abs(best_match["train_flops_per_step"] - best_match["target_train_flops_per_step"]),
                best_match["num_layers"],
                best_match["model_dim"],
            ):
                best_match = candidate
    if best_match is None:
        raise ValueError("budget matching failed to produce a candidate")
    return best_match


def _write_loss_points_csv(model_reports: dict[str, dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "step", "tokens_seen", "train_loss", "val_loss", "val_perplexity", "val_token_accuracy"),
        )
        writer.writeheader()
        for name in ("transformer", "mamba", "reciprocator"):
            history = model_reports[name]["training"]["loss_history"]
            for point in history:
                writer.writerow({"model": name, **point})


def _write_loss_curve_svg_fallback(model_reports: dict[str, dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 960
    height = 540
    margin_left = 90
    margin_right = 40
    margin_top = 50
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    histories = {
        name: model_reports[name]["training"]["loss_history"]
        for name in ("transformer", "mamba", "reciprocator")
    }
    all_points = [point for history in histories.values() for point in history]
    if not all_points:
        raise ValueError("loss history is empty; cannot render validation-loss plot.")

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

    legend_x = margin_left + plot_width - 210
    legend_y = margin_top + 10
    for index, name in enumerate(("transformer", "mamba", "reciprocator")):
        y = legend_y + index * 24
        elements.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{PLOT_COLORS[name]}" stroke-width="3" />'
        )
        elements.append(
            f'<text x="{legend_x + 32}" y="{y + 4}" font-size="13" font-family="Arial, sans-serif">{escape(PLOT_LABELS[name])}</text>'
        )

    for name, history in histories.items():
        points = " ".join(f"{sx(point['tokens_seen']):.2f},{sy(point['val_loss']):.2f}" for point in history)
        elements.append(
            f'<polyline fill="none" stroke="{PLOT_COLORS[name]}" stroke-width="3" points="{points}" />'
        )
        for point in history:
            elements.append(
                f'<circle cx="{sx(point["tokens_seen"]):.2f}" cy="{sy(point["val_loss"]):.2f}" r="3.5" fill="{PLOT_COLORS[name]}" />'
            )

    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def _write_loss_curve_plot(model_reports: dict[str, dict[str, object]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        _write_loss_curve_svg_fallback(model_reports, output_path)
        return

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    for name in ("transformer", "mamba", "reciprocator"):
        history = model_reports[name]["training"]["loss_history"]
        axis.plot(
            [point["tokens_seen"] for point in history],
            [point["val_loss"] for point in history],
            label=PLOT_LABELS[name],
            color=PLOT_COLORS[name],
            linewidth=2.2,
            marker="o",
            markersize=4,
        )
    axis.set_title("Validation loss vs. tokens seen")
    axis.set_xlabel("Tokens seen")
    axis.set_ylabel("Validation loss (nats)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen matched sweep for ReciprocatorOnlyLM.")
    add_device_argument(parser, default="auto")
    parser.add_argument(
        "--corpus",
        action="append",
        choices=("plato_jowett", "greek_philosophy_classics"),
        help=f"Frozen bundled corpus. Pass exactly once; defaults to {DEFAULT_CORPUS}.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONTEXT_LENGTH)
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
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for per-model latest/best training checkpoints. Defaults to a sibling of --report-file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each model from its latest checkpoint in --checkpoint-dir when available.",
    )
    parser.add_argument("--search-min-dim", type=int, default=32)
    parser.add_argument("--search-max-dim", type=int, default=512)
    parser.add_argument("--search-step", type=int, default=8)

    parser.add_argument("--dim", type=int, default=128, help="Reciprocator anchor width.")
    parser.add_argument("--layers", type=int, default=4, help="Reciprocator anchor depth.")
    parser.add_argument("--heads", type=int, default=4, help="Reciprocator anchor head count.")
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
    parser.add_argument(
        "--init-mode-sizes",
        type=_parse_size_tuple,
        default=None,
        help=(
            "Optional comma-separated init mode sizes. If omitted, defaults to the same support "
            "as --max-mode-sizes so the full manifold trains from step 1."
        ),
    )
    parser.add_argument(
        "--max-mode-sizes",
        type=_parse_size_tuple,
        default=None,
        help=(
            "Optional comma-separated max mode sizes. If omitted, derived from a rank-aware "
            f"default capacity of {DEFAULT_STATE_CAPACITY}."
        ),
    )
    parser.add_argument("--num-cube-engines", type=int, default=4)
    parser.add_argument(
        "--normalization",
        choices=("frobenius", "per_mode"),
        default="per_mode",
        help="Reciprocator state normalization. Defaults to per-mode normalization.",
    )
    parser.add_argument(
        "--learned-per-mode-scaling",
        action="store_true",
        help="Relax per-mode normalization with learned per-mode exponents. Only applies with --normalization=per_mode.",
    )
    parser.add_argument("--growth-threshold", type=float, default=0.02)
    parser.add_argument("--growth-interval", type=int, default=1)
    parser.add_argument(
        "--parallel-mixer",
        dest="parallel_mixer",
        action="store_true",
        default=True,
        help="Use the parallel Reciprocator mixer. Enabled by default for the frozen sweep.",
    )
    parser.add_argument(
        "--sequential-mixer",
        dest="parallel_mixer",
        action="store_false",
        help="Force the sequential Reciprocator mixer.",
    )

    parser.add_argument("--transformer-layers", type=int, default=None)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffw-multiplier", type=int, default=4)

    parser.add_argument("--mamba-layers", type=int, default=None)
    parser.add_argument("--mamba-state-size", type=int, default=16)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-conv-kernel", type=int, default=4)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1) for a validation report.")

    corpora = args.corpus or [DEFAULT_CORPUS]
    if len(corpora) != 1:
        raise ValueError("This sweep freezes exactly one corpus. Pass --corpus once or accept the default.")
    corpus = corpora[0]
    transformer_layers = args.transformer_layers or args.layers
    mamba_layers = args.mamba_layers or args.layers
    device = resolve_torch_device(args.device)
    torch.manual_seed(args.seed)

    text = read_corpus_text(corpus)
    tokenizer = _load_or_train_tokenizer(args, text)
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    train_tokens, val_tokens = split_train_val_tokens(token_ids, args.seq_len, args.val_fraction)
    if val_tokens is None:
        raise ValueError("validation split is unavailable for the requested corpus/seq-len combination.")
    model_max_seq_len = args.seq_len

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

    reciprocator_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=model_max_seq_len,
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
        persist_state=False,
        parallel_mixer=args.parallel_mixer,
        input_dependent_gains=True,
        accumulator_modulates_gains=True,
        phase_aware_readout=True,
        phase_aware_coupling=True,
        coupling_temperature=1.0,
    )
    reciprocator_model = ReciprocatorOnlyLM(reciprocator_config)
    target_parameters = count_trainable_parameters(reciprocator_model)
    target_train_flops = estimate_reciprocator_only_train_flops(
        reciprocator_config,
        batch_size=train_config.batch_size,
        seq_len=train_config.seq_len,
    )

    transformer_dim_match = _match_budget_grid(
        target_parameter_count=target_parameters,
        target_train_flops_per_step=target_train_flops,
        candidate_dims=_candidate_dims(
            args.search_min_dim,
            args.search_max_dim,
            args.search_step,
            multiple_of=args.transformer_heads,
        ),
        candidate_layers=_candidate_layers(transformer_layers),
        estimate_parameter_count=lambda *, model_dim, num_layers: _estimate_plain_transformer_parameter_count(
            vocab_size=tokenizer.vocab_size,
            model_dim=model_dim,
            num_layers=num_layers,
            ffw_multiplier=args.transformer_ffw_multiplier,
            max_seq_len=model_max_seq_len,
        ),
        estimate_train_flops=lambda *, model_dim, num_layers: estimate_plain_transformer_train_flops(
            BaselineTransformerConfig(
                vocab_size=tokenizer.vocab_size,
                model_dim=model_dim,
                num_heads=args.transformer_heads,
                num_layers=num_layers,
                ffw_multiplier=args.transformer_ffw_multiplier,
                max_seq_len=model_max_seq_len,
                dropout=args.dropout,
            ),
            batch_size=train_config.batch_size,
            seq_len=train_config.seq_len,
        ),
    )
    mamba_dim_match = _match_budget_grid(
        target_parameter_count=target_parameters,
        target_train_flops_per_step=target_train_flops,
        candidate_dims=_candidate_dims(args.search_min_dim, args.search_max_dim, args.search_step),
        candidate_layers=_candidate_layers(mamba_layers),
        estimate_parameter_count=lambda *, model_dim, num_layers: _estimate_small_mamba_parameter_count(
            vocab_size=tokenizer.vocab_size,
            model_dim=model_dim,
            num_layers=num_layers,
            state_size=args.mamba_state_size,
            expand=args.mamba_expand,
            conv_kernel=args.mamba_conv_kernel,
        ),
        estimate_train_flops=lambda *, model_dim, num_layers: estimate_small_mamba_train_flops(
            SmallMambaConfig(
                vocab_size=tokenizer.vocab_size,
                model_dim=model_dim,
                num_layers=num_layers,
                state_size=args.mamba_state_size,
                expand=args.mamba_expand,
                conv_kernel=args.mamba_conv_kernel,
                max_seq_len=model_max_seq_len,
                dropout=args.dropout,
            ),
            batch_size=train_config.batch_size,
            seq_len=train_config.seq_len,
        ),
    )

    transformer_config = BaselineTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        model_dim=int(transformer_dim_match["model_dim"]),
        num_heads=args.transformer_heads,
        num_layers=int(transformer_dim_match["num_layers"]),
        ffw_multiplier=args.transformer_ffw_multiplier,
        max_seq_len=model_max_seq_len,
        dropout=args.dropout,
    )
    mamba_config = SmallMambaConfig(
        vocab_size=tokenizer.vocab_size,
        model_dim=int(mamba_dim_match["model_dim"]),
        num_layers=int(mamba_dim_match["num_layers"]),
        state_size=args.mamba_state_size,
        expand=args.mamba_expand,
        conv_kernel=args.mamba_conv_kernel,
        max_seq_len=model_max_seq_len,
        dropout=args.dropout,
    )

    flop_estimates = {
        "reciprocator": target_train_flops,
        "transformer": estimate_plain_transformer_train_flops(
            transformer_config,
            batch_size=train_config.batch_size,
            seq_len=train_config.seq_len,
        ),
        "mamba": estimate_small_mamba_train_flops(
            mamba_config,
            batch_size=train_config.batch_size,
            seq_len=train_config.seq_len,
        ),
    }

    print(f"Device: {device}")
    print(f"Frozen corpus: {corpus}")
    print(f"Tokenized corpus: {len(token_ids):,} tokens (vocab_size={tokenizer.vocab_size})")
    print(f"Train tokens: {train_tokens.numel():,}")
    print(f"Validation tokens: {val_tokens.numel():,}")
    print(f"Frozen context length: {model_max_seq_len}")
    print(f"Tokenizer: SentencePiece unigram (vocab_size={tokenizer.vocab_size})")
    print(f"Token budget per model: {train_config.steps * train_config.batch_size * train_config.seq_len:,}")
    print(f"Target parameters (Reciprocator): {target_parameters:,}")
    print(f"Target train FLOPs/step (Reciprocator): {target_train_flops:,.0f}")
    print(
        f"Matched Transformer dim={transformer_dim_match['model_dim']} "
        f"layers={transformer_dim_match['num_layers']} "
        f"params={transformer_dim_match['parameter_count']:,} "
        f"param_gap={100.0 * transformer_dim_match['parameter_relative_gap']:.2f}% "
        f"flop_gap={100.0 * transformer_dim_match['train_flops_relative_gap']:.2f}%"
    )
    print(
        f"Matched Mamba dim={mamba_dim_match['model_dim']} "
        f"layers={mamba_dim_match['num_layers']} "
        f"params={mamba_dim_match['parameter_count']:,} "
        f"param_gap={100.0 * mamba_dim_match['parameter_relative_gap']:.2f}% "
        f"flop_gap={100.0 * mamba_dim_match['train_flops_relative_gap']:.2f}%"
    )
    print(f"Checkpoint dir: {checkpoint_dir}")

    model_specs = [
        ("reciprocator", reciprocator_config, ReciprocatorOnlyLM, args.seed),
        ("mamba", mamba_config, SmallMambaLM, args.seed),
        ("transformer", transformer_config, PlainTransformerLM, args.seed),
    ]

    model_reports: dict[str, dict[str, object]] = {}
    for name, config, model_cls, init_seed in model_specs:
        torch.manual_seed(init_seed)
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
                "script": "capacity_matched_sweep.py",
                "model_name": name,
                "model_class": model_cls.__name__,
                "config": asdict(config),
                "training_config": asdict(train_config),
                "tokenizer_model_path": None if args.tokenizer_model is None else str(args.tokenizer_model),
                "tokenizer_model_proto": tokenizer.processor.serialized_model_proto(),
                "corpus": corpus,
                "tokenizer_kind": "sentencepiece_unigram",
                "context_length": model_max_seq_len,
            },
        )
        probes = {}
        if args.benchmark_examples > 0:
            probes = evaluate_benchmark_suite_generic(
                model,
                vocab_size=tokenizer.vocab_size,
                num_examples=args.benchmark_examples,
                device=device,
                seed=args.seed,
            )
        parameter_count = count_trainable_parameters(model)
        model_reports[name] = {
            "parameter_count": parameter_count,
            "train_flops_per_step": flop_estimates[name],
            "train_flop_budget": flop_estimates[name] * train_config.steps,
            "config": asdict(config),
            "training": training_summary,
            "probes": probes,
            "probe_macro_average": _probe_macro_average(probes),
        }

    reciprocator_val_loss = float(model_reports["reciprocator"]["training"]["val_metrics"]["loss"])
    mamba_val_loss = float(model_reports["mamba"]["training"]["val_metrics"]["loss"])
    loss_gap = reciprocator_val_loss - mamba_val_loss
    if loss_gap <= LOSS_GATE_NATS:
        decision = "proceed"
        if loss_gap <= 0.0:
            reason = (
                f"Reciprocator is {-loss_gap:.4f} nats better than Mamba, so it clears the {LOSS_GATE_NATS:.1f}-nat gate."
            )
        else:
            reason = (
                f"Reciprocator is {loss_gap:.4f} nats worse than Mamba, which is within the {LOSS_GATE_NATS:.1f}-nat gate."
            )
    else:
        decision = "architecture_changes_blocking"
        reason = (
            f"Reciprocator is {loss_gap:.4f} nats worse than Mamba, outside the {LOSS_GATE_NATS:.1f}-nat gate. "
            "Land the architecture changes below before further work."
        )

    loss_plot_path = _default_loss_plot_path(report_path)
    loss_points_path = _default_loss_points_path(report_path)
    _write_loss_points_csv(model_reports, loss_points_path)
    _write_loss_curve_plot(model_reports, loss_plot_path)

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frozen_setup": {
            "corpus": corpus,
            "tokenizer_kind": "sentencepiece_unigram",
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "context_length": model_max_seq_len,
            "token_budget_per_model": train_config.steps * train_config.batch_size * train_config.seq_len,
        },
        "device": str(device),
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "model_path": str(args.tokenizer_model or args.tokenizer_prefix.with_suffix(".model")),
        },
        "training_config": asdict(train_config),
        "target_parameters": target_parameters,
        "target_train_flops_per_step": target_train_flops,
        "artifacts": {
            "loss_plot": str(loss_plot_path),
            "loss_points": str(loss_points_path),
        },
        "parameter_matching": {
            "reciprocator": {
                "parameter_name": "model_dim",
                "parameter_value": args.dim,
                "num_layers": args.layers,
                "parameter_count": target_parameters,
                "target_parameter_count": target_parameters,
                "parameter_relative_gap": 0.0,
                "train_flops_per_step": target_train_flops,
                "target_train_flops_per_step": target_train_flops,
                "train_flops_relative_gap": 0.0,
            },
            "mamba": mamba_dim_match,
            "transformer": transformer_dim_match,
        },
        "models": model_reports,
        "comparison": {
            "reciprocator_minus_mamba_val_loss_nats": loss_gap,
            "gate_threshold_nats": LOSS_GATE_NATS,
            "gate_passed": loss_gap <= LOSS_GATE_NATS,
            "decision": decision,
            "reason": reason,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for name in ("reciprocator", "mamba", "transformer"):
        model_report = model_reports[name]
        val_metrics = model_report["training"]["val_metrics"]
        print(
            f"{name:13s} params={model_report['parameter_count']:>9,} "
            f"flops/step={model_report['train_flops_per_step']:>12,.0f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_ppl={val_metrics['perplexity']:.2f} "
            f"probe_avg={model_report['probe_macro_average']:.4f}"
        )
    print(f"Loss gap vs Mamba: {loss_gap:.4f} nats")
    print(f"Decision: {decision} ({reason})")
    print(f"Report: {report_path}")
    print(f"Loss plot: {loss_plot_path}")


if __name__ == "__main__":
    main()
