"""Prime Reciprocator online state by streaming text and save a resumable checkpoint."""

import argparse
import copy
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reciprocator_lm.ablation import select_mode_size_pair
from reciprocator_lm import (
    ModelConfig,
    ReciprocatorOnlyLM,
    SentencePieceTokenizer,
    available_corpora,
    read_corpus_text,
    train_sentencepiece_tokenizer,
)
from reciprocator_lm.runtime import add_device_argument, resolve_torch_device


DEFAULT_TOKENIZER_PREFIX = ROOT / "scripts" / "reciprocator_only_tokenizer"
DEFAULT_OUTPUT_CHECKPOINT = ROOT / "runs" / "reciprocator_only_primed.pt"
DEFAULT_STATE_CAPACITY = 64
DEFAULT_FRESH_STREAM_RESET_POLICY = "wrap"
DEFAULT_FRESH_LR_SCHEDULE = "cosine"
DEFAULT_FRESH_WARMUP_FRACTION = 0.02
DEFAULT_FRESH_MIN_LR_RATIO = 0.1
DEFAULT_TRAINING_MODE = "streaming"
DEFAULT_CORPORA = ("plato_jowett", "greek_philosophy_classics")


@dataclass(frozen=True)
class LoadedBundle:
    config: ModelConfig
    tokenizer: SentencePieceTokenizer
    model: ReciprocatorOnlyLM
    payload: dict[str, object]
    start_step: int


@dataclass(frozen=True)
class PrimingSummary:
    chunks: int
    average_loss: float
    stream_position: int
    stream_wrap_count: int
    completed_passes: int
    tokens: int


def _make_priming_summary(
    *,
    chunks: int,
    total_loss: float,
    stream_position: int,
    stream_wrap_count: int,
    tokens: int,
) -> PrimingSummary:
    average_loss = total_loss / chunks if chunks > 0 else math.nan
    return PrimingSummary(
        chunks=chunks,
        average_loss=average_loss,
        stream_position=stream_position,
        stream_wrap_count=stream_wrap_count,
        completed_passes=stream_wrap_count,
        tokens=tokens,
    )


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


def _resolve_mode_sizes(
    *,
    state_rank: int,
    max_state_rank: Optional[int] = None,
    init_mode_sizes: Optional[tuple[int, ...]],
    max_mode_sizes: Optional[tuple[int, ...]],
    init_state_capacity: Optional[int],
    state_capacity: Optional[int],
) -> tuple[Optional[tuple[int, ...]], tuple[int, ...]]:
    resolved_max_state_rank = state_rank if max_state_rank is None else max_state_rank
    normalized_max_mode_sizes = max_mode_sizes
    if normalized_max_mode_sizes is not None:
        if len(normalized_max_mode_sizes) == state_rank and resolved_max_state_rank > state_rank:
            normalized_max_mode_sizes = normalized_max_mode_sizes + (2,) * (resolved_max_state_rank - state_rank)
        elif len(normalized_max_mode_sizes) != resolved_max_state_rank:
            raise ValueError("max_mode_sizes length must match state_rank or max_state_rank")
        normalized_max_mode_sizes = tuple(int(size) for size in normalized_max_mode_sizes)
        if any(size <= 0 for size in normalized_max_mode_sizes):
            raise ValueError("max_mode_sizes must contain positive integers")
    effective_state_capacity = (
        DEFAULT_STATE_CAPACITY
        if normalized_max_mode_sizes is None and state_capacity is None
        else state_capacity
    )
    if normalized_max_mode_sizes is None:
        _, resolved_max_mode_sizes = select_mode_size_pair(
            state_rank=resolved_max_state_rank,
            init_mode_sizes=None,
            max_mode_sizes=None,
            init_capacity=None,
            max_capacity=effective_state_capacity,
        )
    else:
        resolved_max_mode_sizes = normalized_max_mode_sizes

    if init_mode_sizes is not None:
        if len(init_mode_sizes) == resolved_max_state_rank:
            return init_mode_sizes, resolved_max_mode_sizes
        if len(init_mode_sizes) == state_rank:
            return (
                init_mode_sizes + (1,) * (resolved_max_state_rank - state_rank),
                resolved_max_mode_sizes,
            )
        raise ValueError("init_mode_sizes length must match state_rank or max_state_rank")

    if init_state_capacity is not None:
        if resolved_max_state_rank == state_rank:
            resolved_init_mode_sizes, resolved_max_mode_sizes = select_mode_size_pair(
                state_rank=state_rank,
                init_mode_sizes=None,
                max_mode_sizes=normalized_max_mode_sizes if normalized_max_mode_sizes is not None else None,
                init_capacity=init_state_capacity,
                max_capacity=effective_state_capacity if normalized_max_mode_sizes is None else None,
            )
            return resolved_init_mode_sizes, resolved_max_mode_sizes
        resolved_init_mode_sizes, _ = select_mode_size_pair(
            state_rank=state_rank,
            init_mode_sizes=None,
            max_mode_sizes=None,
            init_capacity=init_state_capacity,
            max_capacity=init_state_capacity,
        )
        return (
            resolved_init_mode_sizes + (1,) * (resolved_max_state_rank - state_rank),
            resolved_max_mode_sizes,
        )

    if resolved_max_state_rank == state_rank:
        return resolved_max_mode_sizes, resolved_max_mode_sizes
    return None, resolved_max_mode_sizes


def _recursive_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _recursive_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_recursive_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_recursive_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def _resolve_optional_learnable_mixer_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.all_learnable_mixer_params:
        args.learnable_prediction_eta = True
        args.learnable_coupling_temperature = True
        args.learned_per_mode_scaling = True
        args.learned_normalization_blend = True
    return args


def _read_prime_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.text_file is not None:
        return args.text_file.stem, args.text_file.read_text(encoding="utf-8")

    corpus_names = tuple(args.corpus) if args.corpus else DEFAULT_CORPORA
    combined = "\n".join(read_corpus_text(name) for name in corpus_names)
    return "+".join(corpus_names), combined


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prime Reciprocator online state and save a resumable checkpoint.")
    add_device_argument(parser, default="auto")
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        default=None,
        help="Optional source checkpoint. Accepts training checkpoints or exported model checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CHECKPOINT,
        help="Output training-style checkpoint path containing the primed online state.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file to prime on instead of bundled corpora.",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        choices=[corpus.name for corpus in available_corpora()],
        help="Bundled corpus name to prime on. Can be passed multiple times; defaults to the full training corpus order.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Number of ordered corpus passes to stream through the online state.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional cap on streamed chunks for smoke tests or partial priming.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print progress every N streamed chunks. Set 0 to disable intermediate logs.",
    )
    parser.add_argument(
        "--save-every-chunks",
        type=int,
        default=1000,
        help=(
            "Atomically overwrite --output every N completed chunks so interrupted priming can resume. "
            "Set 0 to disable intermediate checkpoint writes."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate stored in fresh optimizer state.")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Priming chunk length. Fresh models also use it as max_seq_len; checkpoints default to their saved max_seq_len.",
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--state-rank", type=int, default=4)
    parser.add_argument("--max-state-rank", type=int, default=8)
    parser.add_argument(
        "--dynamic-rank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable novelty-driven rank growth for freshly built models.",
    )
    parser.add_argument("--init-mode-sizes", type=_parse_size_tuple, default=(4, 4, 2, 2))
    parser.add_argument("--max-mode-sizes", type=_parse_size_tuple, default=(8, 8, 4, 4))
    parser.add_argument("--init-state-capacity", type=int, default=None)
    parser.add_argument("--state-capacity", type=int, default=None)
    parser.add_argument("--num-cube-engines", type=int, default=4)
    parser.add_argument(
        "--normalization",
        choices=("frobenius", "per_mode"),
        default="frobenius",
        help="Reciprocator state normalization for freshly built models.",
    )
    parser.add_argument(
        "--learned-per-mode-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Relax per-mode normalization with learned exponents.",
    )
    parser.add_argument(
        "--learnable-prediction-eta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn the anticipation gain eta in freshly built models.",
    )
    parser.add_argument(
        "--learnable-coupling-temperature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn the phase-aware coupling temperature in freshly built models.",
    )
    parser.add_argument(
        "--learned-normalization-blend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn a blend between normalization families in freshly built models.",
    )
    parser.add_argument(
        "--all-learnable-mixer-params",
        action="store_true",
        help="Enable all optional learnable mixer controls for freshly built models.",
    )
    parser.add_argument("--growth-threshold", type=float, default=0.02)
    parser.add_argument(
        "--growth-warmup-steps",
        type=int,
        default=800,
        help="Steps to suppress growth so engine can stabilize first",
    )
    parser.add_argument(
        "--growth-warmup-multiplier",
        type=float,
        default=10.0,
        help="Multiplier on growth_threshold during warmup (default 10x)",
    )
    parser.add_argument("--growth-interval", type=int, default=1)
    parser.add_argument(
        "--parallel-mixer",
        action="store_true",
        help="Use the parallel Reciprocator mixer for freshly built models. Online priming is incompatible with this.",
    )
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--use-spectral-reciprocation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the spectral reciprocation block.",
    )
    parser.add_argument(
        "--learnable-spectral-reciprocation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn the spectral reciprocation filter parameters.",
    )
    parser.add_argument(
        "--spectral-mode",
        choices=("wavelet_packet_max_ultimate", "wavelet_packet_max_gauge", "wavelet_packet", "dwt", "fft"),
        default="wavelet_packet_max_ultimate",
        help="Spectral reciprocation backend.",
    )
    parser.add_argument(
        "--joint-spectral-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether spectral reciprocation is applied jointly across cube engines.",
    )
    parser.add_argument("--spectral-low-frequency-gain", type=float, default=0.15)
    parser.add_argument("--spectral-low-frequency-sigma", type=float, default=0.2)
    parser.add_argument("--spectral-high-frequency-gain", type=float, default=0.85)
    parser.add_argument("--spectral-high-frequency-cutoff", type=float, default=0.25)
    parser.add_argument("--wavelet-name", choices=("haar", "db1"), default="haar")
    parser.add_argument("--wavelet-levels", type=int, default=3)
    parser.add_argument(
        "--wavelet-packet-best-basis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable best-basis selection inside wavelet packet spectral reciprocation.",
    )
    parser.add_argument("--wavelet-packet-prune-ratio", type=float, default=1e-3)
    parser.add_argument(
        "--wavelet-packet-spectral-subtraction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable wavelet packet spectral subtraction.",
    )
    parser.add_argument(
        "--wavelet-packet-stationary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average over cycle-spun stationary wavelet packet passes.",
    )
    parser.add_argument("--wavelet-packet-cycle-spins", type=int, default=2)
    parser.add_argument(
        "--phase-aware-readout",
        dest="phase_aware_readout",
        action="store_true",
        default=True,
        help="Use phase-aware complex readout features before lm_head.",
    )
    parser.add_argument(
        "--magnitude-readout",
        dest="phase_aware_readout",
        action="store_false",
        help="Disable phase-aware readout and use magnitude-only readout.",
    )
    parser.add_argument(
        "--phase-aware-coupling",
        dest="phase_aware_coupling",
        action="store_true",
        default=True,
        help="Preserve complex-score phase in the reciprocator routing matrices.",
    )
    parser.add_argument(
        "--real-coupling-fallback",
        dest="phase_aware_coupling",
        action="store_false",
        help="Use the legacy real-valued routing collapse.",
    )
    parser.add_argument("--coupling-temperature", type=float, default=1.0)
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=None,
        help="Optional SentencePiece .model used when the checkpoint does not embed one or for fresh builds.",
    )
    parser.add_argument(
        "--tokenizer-prefix",
        type=Path,
        default=DEFAULT_TOKENIZER_PREFIX,
        help="Output prefix used when training a SentencePiece model for a fresh build.",
    )
    parser.add_argument(
        "--stream-reset-policy",
        choices=("wrap", "never"),
        default=DEFAULT_FRESH_STREAM_RESET_POLICY,
        help="Reset policy stored in the output checkpoint for later streaming training.",
    )
    parser.add_argument(
        "--tbptt-horizon",
        type=int,
        default=0,
        help="TBPTT horizon stored in the output checkpoint metadata for later training.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Validation fraction stored in the output checkpoint metadata for later training.",
    )
    parser.add_argument(
        "--benchmark-examples",
        type=int,
        default=128,
        help="Benchmark probe count stored in the output checkpoint metadata for later training.",
    )
    parser.add_argument(
        "--benchmark-every",
        type=int,
        default=200,
        help="Benchmark probe cadence stored in the output checkpoint metadata for later training.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default=DEFAULT_FRESH_LR_SCHEDULE,
        help="Learning-rate schedule stored in the output checkpoint metadata for later training.",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=DEFAULT_FRESH_WARMUP_FRACTION,
        help="Warmup fraction stored in the output checkpoint metadata for later training.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=DEFAULT_FRESH_MIN_LR_RATIO,
        help="Final LR ratio stored in the output checkpoint metadata for later training.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Gradient clipping value stored in the output checkpoint metadata for later training.",
    )
    parser.add_argument(
        "--lr-step-offset",
        type=int,
        default=0,
        help="LR schedule offset stored in the output checkpoint metadata for later training.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.passes <= 0:
        raise ValueError("--passes must be positive")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("--max-chunks must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every must be non-negative")
    if args.save_every_chunks < 0:
        raise ValueError("--save-every-chunks must be non-negative")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.seq_len is not None and args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.tbptt_horizon < 0:
        raise ValueError("--tbptt-horizon must be non-negative")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    if args.benchmark_examples < 0:
        raise ValueError("--benchmark-examples must be non-negative")
    if args.benchmark_every < 0:
        raise ValueError("--benchmark-every must be non-negative")
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("--warmup-fraction must be in [0, 1)")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if args.grad_clip < 0.0:
        raise ValueError("--grad-clip must be non-negative")
    if args.lr_step_offset < 0:
        raise ValueError("--lr-step-offset must be non-negative")
    if args.coupling_temperature <= 0.0:
        raise ValueError("--coupling-temperature must be positive")


def _load_checkpoint_bundle(
    path: Path,
    *,
    device: torch.device,
    tokenizer_model: Optional[Path],
) -> LoadedBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config_payload = payload.get("config")
    model_state_dict = payload.get("model_state_dict")
    if config_payload is None or model_state_dict is None:
        raise ValueError("checkpoint does not contain config/model_state_dict")
    config = config_payload if isinstance(config_payload, ModelConfig) else ModelConfig(**config_payload)
    tokenizer_proto = payload.get("tokenizer_model_proto")
    if tokenizer_proto is not None:
        tokenizer = SentencePieceTokenizer.from_serialized_proto(tokenizer_proto)
    elif tokenizer_model is not None:
        tokenizer = SentencePieceTokenizer.from_model_file(tokenizer_model)
    else:
        raise ValueError("checkpoint does not embed a tokenizer; pass --tokenizer-model.")
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer vocab size does not match checkpoint config.")
    model = ReciprocatorOnlyLM(config).to(device)
    model.load_state_dict(model_state_dict)
    return LoadedBundle(
        config=config,
        tokenizer=tokenizer,
        model=model,
        payload=payload,
        start_step=int(payload.get("step", 0)),
    )


def _build_fresh_bundle(
    args: argparse.Namespace,
    *,
    text: str,
    device: torch.device,
) -> LoadedBundle:
    if args.tokenizer_model is not None:
        print(f"Loading SentencePiece tokenizer from {args.tokenizer_model}...")
        tokenizer = SentencePieceTokenizer.from_model_file(args.tokenizer_model)
    else:
        print("Training SentencePiece tokenizer for fresh priming build...")
        tokenizer = train_sentencepiece_tokenizer(
            text=text,
            vocab_size=args.vocab_size,
            model_prefix=args.tokenizer_prefix,
        )

    seq_len = 256 if args.seq_len is None else args.seq_len
    resolved_init_mode_sizes, resolved_max_mode_sizes = _resolve_mode_sizes(
        state_rank=args.state_rank,
        max_state_rank=args.max_state_rank or args.state_rank,
        init_mode_sizes=args.init_mode_sizes,
        max_mode_sizes=args.max_mode_sizes,
        init_state_capacity=args.init_state_capacity,
        state_capacity=args.state_capacity,
    )

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
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
        learnable_prediction_eta=args.learnable_prediction_eta,
        learnable_coupling_temperature=args.learnable_coupling_temperature,
        learned_normalization_blend=args.learned_normalization_blend,
        dropout=args.dropout,
        use_spectral_reciprocation=args.use_spectral_reciprocation,
        learnable_spectral_reciprocation=args.learnable_spectral_reciprocation,
        spectral_mode=args.spectral_mode,
        joint_spectral_mode=args.joint_spectral_mode,
        spectral_low_frequency_gain=args.spectral_low_frequency_gain,
        spectral_low_frequency_sigma=args.spectral_low_frequency_sigma,
        spectral_high_frequency_gain=args.spectral_high_frequency_gain,
        spectral_high_frequency_cutoff=args.spectral_high_frequency_cutoff,
        wavelet_name=args.wavelet_name,
        wavelet_levels=args.wavelet_levels,
        wavelet_packet_best_basis=args.wavelet_packet_best_basis,
        wavelet_packet_prune_ratio=args.wavelet_packet_prune_ratio,
        wavelet_packet_spectral_subtraction=args.wavelet_packet_spectral_subtraction,
        wavelet_packet_stationary=args.wavelet_packet_stationary,
        wavelet_packet_cycle_spins=args.wavelet_packet_cycle_spins,
        growth_threshold=args.growth_threshold,
        growth_warmup_steps=args.growth_warmup_steps,
        growth_warmup_multiplier=args.growth_warmup_multiplier,
        growth_interval=args.growth_interval,
        persist_state=True,
        parallel_mixer=args.parallel_mixer,
        input_dependent_gains=True,
        accumulator_modulates_gains=True,
        phase_aware_readout=args.phase_aware_readout,
        phase_aware_coupling=args.phase_aware_coupling,
        coupling_temperature=args.coupling_temperature,
    )
    if config.parallel_mixer:
        raise ValueError("--parallel-mixer is incompatible with online priming")
    model = ReciprocatorOnlyLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    payload: dict[str, object] = {
        "optimizer_state_dict": _recursive_to_cpu(optimizer.state_dict()),
        "metadata": {},
    }
    return LoadedBundle(
        config=config,
        tokenizer=tokenizer,
        model=model,
        payload=payload,
        start_step=0,
    )


def _resolve_bundle(
    args: argparse.Namespace,
    *,
    text: str,
    device: torch.device,
) -> LoadedBundle:
    if args.checkpoint_in is not None:
        print(f"Loading checkpoint: {args.checkpoint_in}")
        bundle = _load_checkpoint_bundle(args.checkpoint_in, device=device, tokenizer_model=args.tokenizer_model)
        if args.seq_len is not None and args.seq_len > bundle.config.max_seq_len:
            raise ValueError(
                f"--seq-len {args.seq_len} exceeds checkpoint max_seq_len={bundle.config.max_seq_len}"
            )
        bundle.config.persist_state = True
        return bundle
    return _build_fresh_bundle(args, text=text, device=device)


def _prime_model(
    *,
    model: ReciprocatorOnlyLM,
    tokenizer: SentencePieceTokenizer,
    text: str,
    seq_len: int,
    device: torch.device,
    passes: int,
    max_chunks: Optional[int],
    log_every: int,
    save_every_chunks: int,
    on_checkpoint: Optional[Callable[[PrimingSummary, str], None]] = None,
) -> PrimingSummary:
    token_ids = torch.tensor(tokenizer.encode(text, add_bos=True, add_eos=True), dtype=torch.long)
    if token_ids.numel() < seq_len + 1:
        raise ValueError("priming text is too short for the requested seq_len")

    max_start = token_ids.numel() - seq_len - 1
    starts = list(range(0, max_start + 1, seq_len))
    if not starts:
        raise ValueError("priming text does not contain any full chunks")

    model.eval()
    model.enter_online_mode()
    # Priming is meant to seed a fresh working memory snapshot from the text
    # stream itself, so we clear any loaded online state exactly once here.
    model.reset_online_state()
    model.set_online_state_gradient_tracking(False)

    chunks = 0
    total_loss = 0.0
    stream_position = 0
    stream_wrap_count = 0
    tokens = int(token_ids.numel())
    t0 = time.time()
    stop = False
    last_checkpoint_chunk = 0

    with torch.no_grad():
        for pass_index in range(passes):
            for index, start in enumerate(starts):
                chunk = token_ids[start : start + seq_len + 1]
                inputs = chunk[:-1].unsqueeze(0).to(device=device)
                targets = chunk[1:].unsqueeze(0).to(device=device)
                _, loss = model(inputs, targets)
                assert loss is not None
                chunks += 1
                total_loss += float(loss.item())
                stream_position = start + seq_len
                is_last_chunk_in_pass = index == len(starts) - 1
                if log_every > 0 and (chunks == 1 or chunks % log_every == 0):
                    elapsed = time.time() - t0
                    print(
                        f"pass={pass_index + 1:3d}  chunk={chunks:6d}  "
                        f"avg_loss={total_loss / chunks:.4f}  "
                        f"stream_position={stream_position:8d}  elapsed={elapsed:.1f}s"
                    )
                if on_checkpoint is not None and save_every_chunks > 0 and chunks % save_every_chunks == 0:
                    on_checkpoint(
                        _make_priming_summary(
                            chunks=chunks,
                            total_loss=total_loss,
                            stream_position=stream_position,
                            stream_wrap_count=stream_wrap_count,
                            tokens=tokens,
                        ),
                        "interval",
                    )
                    last_checkpoint_chunk = chunks
                if max_chunks is not None and chunks >= max_chunks and not is_last_chunk_in_pass:
                    stop = True
                    break
            if stop:
                break
            # After a completed pass we record the next chunk position exactly as
            # streaming training would see it at the beginning of the next pass:
            # token 0 with a completed-wrap count, while preserving the primed
            # numerical state so a resumed run can keep using it immediately.
            stream_position = 0
            stream_wrap_count += 1
            if on_checkpoint is not None and chunks > 0 and chunks != last_checkpoint_chunk:
                on_checkpoint(
                    _make_priming_summary(
                        chunks=chunks,
                        total_loss=total_loss,
                        stream_position=stream_position,
                        stream_wrap_count=stream_wrap_count,
                        tokens=tokens,
                    ),
                    "pass_end",
                )
                last_checkpoint_chunk = chunks
            if max_chunks is not None and chunks >= max_chunks:
                break

    return _make_priming_summary(
        chunks=chunks,
        total_loss=total_loss,
        stream_position=stream_position,
        stream_wrap_count=stream_wrap_count,
        tokens=tokens,
    )


def _build_output_payload(
    *,
    args: argparse.Namespace,
    bundle: LoadedBundle,
    summary: PrimingSummary,
    source_name: str,
) -> dict[str, object]:
    source_payload = bundle.payload
    source_metadata = dict(source_payload.get("metadata", {}))
    optimizer_state = source_payload.get("optimizer_state_dict")
    if optimizer_state is None:
        optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=args.lr)
        optimizer_state = optimizer.state_dict()

    metadata = dict(source_metadata)
    metadata.update(
        {
            "script": "prime_reciprocator_state.py",
            "seed": int(args.seed),
            "val_fraction": float(args.val_fraction),
            "training_mode": DEFAULT_TRAINING_MODE,
            "stream_reset_policy": args.stream_reset_policy,
            "tbptt_horizon": int(args.tbptt_horizon),
            "benchmark_examples": int(args.benchmark_examples),
            "benchmark_every": int(args.benchmark_every),
            "lr_schedule": args.lr_schedule,
            "warmup_fraction": float(args.warmup_fraction),
            "min_lr_ratio": float(args.min_lr_ratio),
            "grad_clip": float(args.grad_clip),
            "lr_step_offset": int(args.lr_step_offset),
            "stream_position": int(summary.stream_position),
            "stream_wrap_count": int(summary.stream_wrap_count),
            "priming_source": source_name,
            "priming_requested_passes": int(args.passes),
            "priming_completed_passes": int(summary.completed_passes),
            "priming_save_every_chunks": int(args.save_every_chunks),
            "primed_chunks": int(summary.chunks),
            "primed_average_loss": float(summary.average_loss),
        }
    )
    if args.checkpoint_in is not None:
        metadata["primed_from_checkpoint"] = str(args.checkpoint_in)

    best_metric_name = str(
        source_payload.get("best_metric_name", "val_loss" if args.val_fraction > 0.0 else "train_loss")
    )
    return {
        "config": asdict(bundle.config),
        "model_state_dict": _recursive_to_cpu(bundle.model.state_dict()),
        "optimizer_state_dict": _recursive_to_cpu(optimizer_state),
        "tokenizer_model_proto": bundle.tokenizer.processor.serialized_model_proto(),
        "step": int(source_payload.get("step", bundle.start_step)),
        "best_metric": float(source_payload.get("best_metric", float("inf"))),
        "best_metric_name": best_metric_name,
        "last_train_loss": float(summary.average_loss),
        "last_val_metrics": copy.deepcopy(source_payload.get("last_val_metrics")),
        "last_benchmark_metrics": copy.deepcopy(source_payload.get("last_benchmark_metrics")),
        "benchmark_history": copy.deepcopy(source_payload.get("benchmark_history", [])),
        "metadata": metadata,
    }


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    args = _resolve_optional_learnable_mixer_args(args)
    _validate_args(args)
    torch.manual_seed(args.seed)

    source_name, text = _read_prime_text(args)
    print(f"Priming source: {source_name}")
    print(f"Corpus size: {len(text):,} characters")

    device = resolve_torch_device(args.device)
    print(f"Device: {device}")
    bundle = _resolve_bundle(args, text=text, device=device)

    seq_len = bundle.config.max_seq_len if args.seq_len is None else args.seq_len
    if seq_len <= 0:
        raise ValueError("resolved seq_len must be positive")
    if seq_len > bundle.config.max_seq_len:
        raise ValueError(f"resolved seq_len {seq_len} exceeds max_seq_len={bundle.config.max_seq_len}")
    if getattr(bundle.config, "parallel_mixer", False):
        raise ValueError("parallel_mixer checkpoints cannot be primed in online mode")

    bundle.config.persist_state = True
    n_params = sum(parameter.numel() for parameter in bundle.model.parameters())
    print(f"Tokenized vocab size: {bundle.tokenizer.vocab_size}")
    print(
        "Config: "
        f"{bundle.config.n_layers} layers, dim={bundle.config.dim}, "
        f"seq_len={bundle.config.max_seq_len}, "
        f"state_rank={bundle.config.state_rank}, max_state_rank={bundle.config.max_state_rank}, "
        f"init_mode_sizes={bundle.config.init_mode_sizes}, max_mode_sizes={bundle.config.max_mode_sizes}, "
        f"num_cube_engines={bundle.config.num_cube_engines}, dynamic_rank={bundle.config.dynamic_rank}"
    )
    print(f"Model parameters: {n_params:,}")
    print(
        f"Priming chunks: seq_len={seq_len}, passes={args.passes}, "
        f"max_chunks={'full' if args.max_chunks is None else args.max_chunks}"
    )
    if args.save_every_chunks > 0:
        print(f"Intermediate checkpoint cadence: every {args.save_every_chunks} chunks -> {args.output}")

    def save_intermediate_checkpoint(summary: PrimingSummary, reason: str) -> None:
        payload = _build_output_payload(
            args=args,
            bundle=bundle,
            summary=summary,
            source_name=source_name,
        )
        _atomic_torch_save(args.output, payload)
        print(
            f"checkpoint={reason:>8s}  chunk={summary.chunks:6d}  "
            f"avg_loss={summary.average_loss:.4f}  "
            f"stream_position={summary.stream_position:8d}  "
            f"saved={args.output}"
        )

    try:
        summary = _prime_model(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            text=text,
            seq_len=seq_len,
            device=device,
            passes=args.passes,
            max_chunks=args.max_chunks,
            log_every=args.log_every,
            save_every_chunks=args.save_every_chunks,
            on_checkpoint=save_intermediate_checkpoint if args.save_every_chunks > 0 else None,
        )
    except KeyboardInterrupt:
        print()
        if args.output.exists():
            print(f"Priming interrupted. Resume from the last checkpoint written to {args.output}.")
        else:
            print("Priming interrupted before any checkpoint was written.")
        return
    payload = _build_output_payload(
        args=args,
        bundle=bundle,
        summary=summary,
        source_name=source_name,
    )
    _atomic_torch_save(args.output, payload)

    diagnostics = bundle.model.online_diagnostics()
    active_rank = None
    active_sizes = None
    if diagnostics.get("layers"):
        active_rank = diagnostics["layers"][0].get("active_rank")
        active_sizes = diagnostics["layers"][0].get("active_sizes")

    print(f"Primed chunks: {summary.chunks}")
    print(f"Average priming loss: {summary.average_loss:.4f}")
    print(
        "Saved primed checkpoint: "
        f"{args.output} "
        f"(stream_position={summary.stream_position}, stream_wrap_count={summary.stream_wrap_count})"
    )
    if active_rank is not None:
        print(f"First-layer active_rank={active_rank} active_sizes={active_sizes}")


if __name__ == "__main__":
    main()
