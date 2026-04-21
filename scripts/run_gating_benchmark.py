"""Run the Step-0 gating benchmark: ReciprocatorOnlyLM vs PlainTransformerLM."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reciprocator_lm import (
    BaselineTransformerConfig,
    ModelConfig,
    PlainTransformerLM,
    ReciprocatorOnlyLM,
    SentencePieceTokenizer,
    TrainingRunConfig,
    count_trainable_parameters,
    estimate_plain_transformer_train_flops,
    estimate_reciprocator_only_train_flops,
    read_corpus_text,
    split_train_val_tokens,
    train_causal_language_model,
    train_sentencepiece_tokenizer,
)
from reciprocator_lm.runtime import add_device_argument, resolve_torch_device


DEFAULT_TOKENIZER_PREFIX = ROOT / "runs" / "gating_benchmark_tokenizer"


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


def _default_report_path() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return ROOT / "runs" / f"gating_benchmark_{timestamp}.json"


def _transformer_parameter_count(config: BaselineTransformerConfig) -> int:
    model_dim = config.model_dim
    ffw_dim = model_dim * config.ffw_multiplier
    block_parameters = (
        (4 * model_dim * model_dim)
        + (2 * model_dim * ffw_dim)
        + (9 * model_dim)
        + ffw_dim
    )
    total = (
        (config.vocab_size * model_dim)
        + (config.max_seq_len * model_dim)
        + (config.num_layers * block_parameters)
        + (2 * model_dim)
    )
    return int(total)


def _match_transformer_budget(
    *,
    vocab_size: int,
    max_seq_len: int,
    target_parameter_count: int,
    target_train_flops_per_step: float,
    batch_size: int,
    seq_len: int,
    candidate_dims: list[int],
    candidate_layers: list[int],
    num_heads: int,
    ffw_multiplier: int,
    dropout: float,
) -> dict[str, float | int]:
    best_match: dict[str, float | int] | None = None
    for num_layers in candidate_layers:
        for model_dim in candidate_dims:
            config = BaselineTransformerConfig(
                vocab_size=vocab_size,
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ffw_multiplier=ffw_multiplier,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            parameter_count = _transformer_parameter_count(config)
            train_flops_per_step = estimate_plain_transformer_train_flops(
                config,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            parameter_relative_gap = abs(parameter_count - target_parameter_count) / target_parameter_count
            train_flops_relative_gap = abs(train_flops_per_step - target_train_flops_per_step) / target_train_flops_per_step
            candidate = {
                "model_dim": model_dim,
                "num_layers": num_layers,
                "parameter_count": parameter_count,
                "parameter_relative_gap": parameter_relative_gap,
                "train_flops_per_step": train_flops_per_step,
                "target_train_flops_per_step": target_train_flops_per_step,
                "train_flops_relative_gap": train_flops_relative_gap,
            }
            if best_match is None or (
                max(candidate["parameter_relative_gap"], candidate["train_flops_relative_gap"]),
                candidate["parameter_relative_gap"],
                candidate["train_flops_relative_gap"],
                abs(candidate["parameter_count"] - target_parameter_count),
                abs(candidate["train_flops_per_step"] - target_train_flops_per_step),
                candidate["num_layers"],
                candidate["model_dim"],
            ) < (
                max(best_match["parameter_relative_gap"], best_match["train_flops_relative_gap"]),
                best_match["parameter_relative_gap"],
                best_match["train_flops_relative_gap"],
                abs(best_match["parameter_count"] - target_parameter_count),
                abs(best_match["train_flops_per_step"] - target_train_flops_per_step),
                best_match["num_layers"],
                best_match["model_dim"],
            ):
                best_match = candidate
    assert best_match is not None
    return best_match


def _load_or_train_tokenizer(args: argparse.Namespace, text: str) -> SentencePieceTokenizer:
    if args.tokenizer_model is not None:
        return SentencePieceTokenizer.from_model_file(args.tokenizer_model)
    return train_sentencepiece_tokenizer(
        text=text,
        vocab_size=args.vocab_size,
        model_prefix=args.tokenizer_prefix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Step-0 Reciprocator vs Transformer gating benchmark.")
    add_device_argument(parser, default="auto")
    parser.add_argument(
        "--corpus",
        action="append",
        choices=("plato_jowett", "greek_philosophy_classics"),
        help="Bundled corpus to include. Repeat to include more than one corpus.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=512)
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
    parser.add_argument("--search-min-dim", type=int, default=128)
    parser.add_argument("--search-max-dim", type=int, default=1024)
    parser.add_argument("--search-step", type=int, default=8)

    parser.add_argument("--dim", type=int, default=352, help="Reciprocator anchor width.")
    parser.add_argument("--layers", type=int, default=7, help="Reciprocator anchor depth.")
    parser.add_argument("--heads", type=int, default=8, help="Reciprocator anchor head count.")
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
    parser.add_argument("--init-mode-sizes", type=_parse_size_tuple, default=(8, 8, 8))
    parser.add_argument("--max-mode-sizes", type=_parse_size_tuple, default=(8, 8, 8))
    parser.add_argument("--num-cube-engines", type=int, default=3)
    parser.add_argument(
        "--normalization",
        choices=("frobenius", "per_mode"),
        default="per_mode",
        help="Reciprocator state normalization.",
    )
    parser.add_argument(
        "--learned-per-mode-scaling",
        action="store_true",
        help="Relax per-mode normalization with learned per-mode exponents. Only applies with --normalization=per_mode.",
    )
    parser.add_argument("--growth-threshold", type=float, default=0.02)
    parser.add_argument("--growth-interval", type=int, default=1)

    parser.add_argument("--transformer-layers", type=int, default=None)
    parser.add_argument("--transformer-min-layers", type=int, default=None)
    parser.add_argument("--transformer-max-layers", type=int, default=None)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-ffw-multiplier", type=int, default=4)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1) for a validation report.")

    corpora = args.corpus or ["plato_jowett", "greek_philosophy_classics"]
    device = resolve_torch_device(args.device)
    torch.manual_seed(args.seed)

    text = "\n".join(read_corpus_text(name) for name in corpora)
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
    report_path = args.report_file or _default_report_path()
    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else report_path.with_suffix("").with_name(f"{report_path.stem}_artifacts")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    reciprocator_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        state_rank=args.state_rank,
        max_state_rank=args.max_state_rank or args.state_rank,
        dynamic_rank=args.dynamic_rank,
        init_mode_sizes=args.init_mode_sizes,
        max_mode_sizes=args.max_mode_sizes,
        num_cube_engines=args.num_cube_engines,
        normalization=args.normalization,
        learned_per_mode_scaling=args.learned_per_mode_scaling,
        dropout=args.dropout,
        growth_threshold=args.growth_threshold,
        growth_interval=args.growth_interval,
        persist_state=False,
        parallel_mixer=True,
        accumulator_modulates_gains=True,
        phase_aware_readout=True,
        phase_aware_coupling=True,
        coupling_temperature=1.0,
    )
    reciprocator_model = ReciprocatorOnlyLM(reciprocator_config)
    target_parameters = count_trainable_parameters(reciprocator_model)
    target_train_flops_per_step = estimate_reciprocator_only_train_flops(
        reciprocator_config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    if args.transformer_layers is not None:
        candidate_layers = [args.transformer_layers]
    else:
        min_layers = args.transformer_min_layers or max(1, args.layers - 3)
        max_layers = args.transformer_max_layers or (args.layers + 3)
        candidate_layers = list(range(min_layers, max_layers + 1))

    transformer_match = _match_transformer_budget(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        target_parameter_count=target_parameters,
        target_train_flops_per_step=target_train_flops_per_step,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        candidate_dims=_candidate_dims(
            args.search_min_dim,
            args.search_max_dim,
            args.search_step,
            multiple_of=args.transformer_heads,
        ),
        candidate_layers=candidate_layers,
        num_heads=args.transformer_heads,
        ffw_multiplier=args.transformer_ffw_multiplier,
        dropout=args.dropout,
    )

    transformer_config = BaselineTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        model_dim=int(transformer_match["model_dim"]),
        num_heads=args.transformer_heads,
        num_layers=int(transformer_match["num_layers"]),
        ffw_multiplier=args.transformer_ffw_multiplier,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    )

    print(f"Device: {device}")
    print(f"Corpora: {', '.join(corpora)}")
    print(f"Tokenized corpus: {len(token_ids):,} tokens (vocab_size={tokenizer.vocab_size})")
    print(f"Train tokens: {train_tokens.numel():,}")
    print(f"Validation tokens: {val_tokens.numel():,}")
    print(f"Training tokens seen per model: {args.steps * args.batch_size * args.seq_len:,}")
    print(f"Target parameters (Reciprocator): {target_parameters:,}")
    print(f"Target train FLOPs/step (Reciprocator est.): {target_train_flops_per_step:,.0f}")
    print(
        f"Matched Transformer layers={int(transformer_match['num_layers'])} "
        f"dim={int(transformer_match['model_dim'])} "
        f"params={int(transformer_match['parameter_count']):,} "
        f"param_gap={100.0 * float(transformer_match['parameter_relative_gap']):.2f}% "
        f"flop_gap={100.0 * float(transformer_match['train_flops_relative_gap']):.2f}%"
    )
    print(f"Checkpoint dir: {checkpoint_dir}")

    model_specs = [
        ("reciprocator", reciprocator_config, ReciprocatorOnlyLM, args.seed),
        ("transformer", transformer_config, PlainTransformerLM, args.seed + 1),
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
                "script": "run_gating_benchmark.py",
                "model_name": name,
                "model_class": model_cls.__name__,
                "config": asdict(config),
                "training_config": asdict(train_config),
                "tokenizer_model_path": None if args.tokenizer_model is None else str(args.tokenizer_model),
                "tokenizer_model_proto": tokenizer.processor.serialized_model_proto(),
                "corpora": list(corpora),
            },
        )
        parameter_count = count_trainable_parameters(model)
        train_flops_per_step = (
            estimate_reciprocator_only_train_flops(config, batch_size=args.batch_size, seq_len=args.seq_len)
            if name == "reciprocator"
            else estimate_plain_transformer_train_flops(config, batch_size=args.batch_size, seq_len=args.seq_len)
        )
        model_reports[name] = {
            "parameter_count": parameter_count,
            "config": asdict(config),
            "estimated_train_flops_per_step": train_flops_per_step,
            "estimated_total_train_flops": train_flops_per_step * args.steps,
            "training": training_summary,
        }

    reciprocator_ppl = float(model_reports["reciprocator"]["training"]["val_metrics"]["perplexity"])
    transformer_ppl = float(model_reports["transformer"]["training"]["val_metrics"]["perplexity"])
    perplexity_ratio = reciprocator_ppl / transformer_ppl
    perplexity_gap_pct = 100.0 * (perplexity_ratio - 1.0)

    if perplexity_gap_pct <= 5.0:
        decision = "keep_going"
        reason = "Reciprocator is within the 5% perplexity gate against the matched transformer."
    elif perplexity_gap_pct >= 20.0:
        decision = "reconsider"
        reason = "Reciprocator is at least 20% worse in perplexity than the matched transformer."
    else:
        decision = "borderline"
        reason = "Reciprocator is between the 5% keep-going gate and the 20% reconsider gate."

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpora": corpora,
        "device": str(device),
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "model_path": None if args.tokenizer_model is None else str(args.tokenizer_model),
        },
        "training_config": asdict(train_config),
        "training_tokens_seen_per_model": args.steps * args.batch_size * args.seq_len,
        "parameter_matching": {
            "reciprocator": {
                "parameter_name": "model_dim",
                "parameter_value": args.dim,
                "parameter_count": target_parameters,
                "target_parameter_count": target_parameters,
                "parameter_relative_gap": 0.0,
                "train_flops_per_step": target_train_flops_per_step,
                "target_train_flops_per_step": target_train_flops_per_step,
                "train_flops_relative_gap": 0.0,
            },
            "transformer": dict(transformer_match),
        },
        "models": model_reports,
        "comparison": {
            "reciprocator_perplexity": reciprocator_ppl,
            "transformer_perplexity": transformer_ppl,
            "perplexity_ratio": perplexity_ratio,
            "perplexity_gap_pct": perplexity_gap_pct,
            "decision": decision,
            "reason": reason,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for name in ("reciprocator", "transformer"):
        model_report = model_reports[name]
        val_metrics = model_report["training"]["val_metrics"]
        print(
            f"{name:13s} params={model_report['parameter_count']:>10,} "
            f"train_flops/step={model_report['estimated_train_flops_per_step']:>14,.0f} "
            f"val_ppl={val_metrics['perplexity']:.2f}"
        )
    print(
        f"Perplexity gap: {perplexity_gap_pct:.2f}% "
        f"(reciprocator={reciprocator_ppl:.2f}, transformer={transformer_ppl:.2f})"
    )
    print(f"Decision: {decision} ({reason})")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
