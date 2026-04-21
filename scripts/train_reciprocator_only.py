"""Train ReciprocatorOnlyLM (no attention) on bundled corpora."""

import argparse
import copy
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reciprocator_lm.ablation import select_mode_size_pair
from reciprocator_lm import (
    ModelConfig,
    ReciprocatorOnlyLM,
    SentencePieceTokenizer,
    build_default_benchmark_suite,
    evaluate_named_benchmark_tasks,
    mode_axis_permutation_probe,
    save_reciprocator_checkpoint,
    read_corpus_text,
    train_sentencepiece_tokenizer,
)
from reciprocator_lm.experiments import (
    _annealed_growth_threshold,
    _count_growth_events,
    _reset_optimizer_moments,
    _set_growth_threshold,
)
from reciprocator_lm.runtime import add_device_argument, resolve_torch_device


DEFAULT_TOKENIZER_PREFIX = ROOT / "scripts" / "reciprocator_only_tokenizer"
DEFAULT_LATEST_CHECKPOINT = ROOT / "runs" / "reciprocator_only_latest.pt"
DEFAULT_BEST_CHECKPOINT = ROOT / "runs" / "reciprocator_only_best.pt"
DEFAULT_STATE_CAPACITY = 64


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
    effective_state_capacity = (
        DEFAULT_STATE_CAPACITY
        if max_mode_sizes is None and state_capacity is None
        else state_capacity
    )
    _, resolved_max_mode_sizes = select_mode_size_pair(
        state_rank=resolved_max_state_rank,
        init_mode_sizes=None,
        max_mode_sizes=max_mode_sizes,
        init_capacity=None,
        max_capacity=effective_state_capacity,
    )

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
                max_mode_sizes=max_mode_sizes if max_mode_sizes is not None else None,
                init_capacity=init_state_capacity,
                max_capacity=effective_state_capacity if max_mode_sizes is None else None,
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
    return value


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def build_dataset(
    token_ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = token_ids
    if data.numel() < seq_len + 1:
        repeats = ((seq_len + 1) // max(1, data.numel())) + 1
        data = data.repeat(repeats)
    max_start = max(1, data.numel() - seq_len - 1)
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    inputs = torch.stack([data[start : start + seq_len] for start in starts.tolist()])
    targets = torch.stack([data[start + 1 : start + seq_len + 1] for start in starts.tolist()])
    return inputs.to(device=device), targets.to(device=device)


class ContiguousTokenStream:
    def __init__(
        self,
        token_ids: torch.Tensor,
        seq_len: int,
        *,
        position: int = 0,
        wrap_count: int = 0,
    ) -> None:
        data = token_ids
        if data.numel() < seq_len + 1:
            repeats = ((seq_len + 1) // max(1, data.numel())) + 1
            data = data.repeat(repeats)
        self.data = data
        self.seq_len = seq_len
        self.max_start = max(1, self.data.numel() - self.seq_len - 1)
        self.position = int(position) % max(1, self.data.numel())
        self.wrap_count = int(wrap_count)

    def state_dict(self) -> dict[str, int]:
        return {
            "position": int(self.position),
            "wrap_count": int(self.wrap_count),
        }

    def next_chunk(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        wrapped = False
        if self.position + self.seq_len + 1 > self.data.numel():
            self.position = 0
            self.wrap_count += 1
            wrapped = True
        start = self.position
        stop = start + self.seq_len + 1
        chunk = self.data[start:stop]
        if chunk.numel() != self.seq_len + 1:
            raise RuntimeError("streamed training chunk did not match seq_len")
        self.position = start + self.seq_len
        return chunk[:-1], chunk[1:], wrapped


def _should_reset_stream_state(reset_policy: str, *, wrapped: bool) -> bool:
    if reset_policy == "wrap":
        return wrapped
    if reset_policy == "never":
        return False
    raise ValueError(f"Unsupported stream reset policy: {reset_policy}")


def _lr_multiplier(
    *,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_fraction: float,
    min_lr_ratio: float,
    step_offset: int = 0,
) -> float:
    if step_offset < 0:
        raise ValueError("step_offset must be non-negative")
    effective_total_steps = total_steps - step_offset
    effective_step = step - step_offset
    if effective_total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if effective_step <= 0:
        raise ValueError("step must be positive")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"Unsupported learning-rate schedule: {schedule}")

    warmup_steps = min(effective_total_steps, int(math.ceil(effective_total_steps * warmup_fraction)))
    if warmup_steps > 0 and effective_step <= warmup_steps:
        return effective_step / float(warmup_steps)
    if effective_total_steps <= warmup_steps:
        return 1.0
    decay_progress = (effective_step - warmup_steps) / float(effective_total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _streaming_train_step(
    *,
    model: ReciprocatorOnlyLM,
    train_stream: ContiguousTokenStream,
    device: torch.device,
    chunks_per_step: int,
    stream_reset_policy: str,
    tbptt_horizon: int,
) -> tuple[float, int]:
    if chunks_per_step <= 0:
        raise ValueError("chunks_per_step must be positive")
    if tbptt_horizon < 0:
        raise ValueError("tbptt_horizon must be non-negative")

    tbptt_enabled = tbptt_horizon > 0
    model.reset_online_state()
    model.set_online_state_gradient_tracking(tbptt_enabled)

    chunk_losses: list[float] = []
    window_chunk_count = 0
    window_loss: Optional[torch.Tensor] = None
    backward_passes = 0

    def flush_window() -> None:
        nonlocal backward_passes, window_chunk_count, window_loss
        if window_loss is None:
            return
        window_loss.backward()
        backward_passes += 1
        window_loss = None
        window_chunk_count = 0
        if tbptt_enabled:
            model.detach_online_state()

    for _ in range(chunks_per_step):
        chunk_inputs, chunk_targets, wrapped = train_stream.next_chunk()
        if _should_reset_stream_state(stream_reset_policy, wrapped=wrapped):
            if tbptt_enabled:
                flush_window()
            model.reset_online_state()
        inputs = chunk_inputs.unsqueeze(0).to(device=device)
        targets = chunk_targets.unsqueeze(0).to(device=device)
        _, loss = model(inputs, targets)
        assert loss is not None
        scaled_loss = loss / chunks_per_step
        window_loss = scaled_loss if window_loss is None else window_loss + scaled_loss
        window_chunk_count += 1
        chunk_losses.append(float(loss.item()))
        if tbptt_enabled and window_chunk_count >= tbptt_horizon:
            flush_window()

    flush_window()
    return sum(chunk_losses) / len(chunk_losses), backward_passes


def _overrides_extra_state_hook(module: Any, hook_name: str) -> bool:
    module_type = type(module)
    hook = getattr(module_type, hook_name, None)
    base_hook = getattr(torch.nn.Module, hook_name, None)
    return callable(hook) and hook is not base_hook


def _snapshot_runtime_state(model: ReciprocatorOnlyLM) -> dict[str, object]:
    model_state: Any = None
    get_model_state = getattr(model, "get_extra_state", None)
    if _overrides_extra_state_hook(model, "get_extra_state"):
        model_state = _recursive_to_cpu(get_model_state())

    mixer_states: list[Any] = []
    for block in getattr(model, "blocks", []):
        mixer = getattr(block, "mixer", None)
        get_mixer_state = getattr(mixer, "get_extra_state", None)
        mixer_states.append(
            None
            if not _overrides_extra_state_hook(mixer, "get_extra_state")
            else _recursive_to_cpu(get_mixer_state())
        )

    return {
        "model": model_state,
        "mixers": mixer_states,
    }


def _restore_runtime_state(model: ReciprocatorOnlyLM, snapshot: dict[str, object]) -> None:
    set_model_state = getattr(model, "set_extra_state", None)
    model_state = snapshot.get("model")
    if _overrides_extra_state_hook(model, "set_extra_state") and model_state is not None:
        set_model_state(model_state)

    mixer_states = snapshot.get("mixers", [])
    for block, mixer_state in zip(getattr(model, "blocks", []), mixer_states):
        mixer = getattr(block, "mixer", None)
        set_mixer_state = getattr(mixer, "set_extra_state", None)
        if _overrides_extra_state_hook(mixer, "set_extra_state"):
            set_mixer_state(mixer_state)


def _set_persistent_training_mode(
    model: ReciprocatorOnlyLM,
    *,
    enabled: bool,
    reset_state: bool,
) -> None:
    if enabled:
        model.enter_online_mode()
        if reset_state:
            model.reset_online_state()
        return

    model.reset_online_state()
    for block in getattr(model, "blocks", []):
        mixer = getattr(block, "mixer", None)
        if hasattr(mixer, "persist_state"):
            mixer.persist_state = False


def evaluate_model(
    model: ReciprocatorOnlyLM,
    token_ids: torch.Tensor,
    seq_len: int,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    was_training = model.training
    runtime_snapshot = _snapshot_runtime_state(model)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    batches = 0

    try:
        model.reset_online_state()
        with torch.no_grad():
            max_start = token_ids.numel() - seq_len - 1
            for start in range(0, max_start + 1, seq_len):
                if max_batches is not None and batches >= max_batches:
                    break
                chunk = token_ids[start : start + seq_len + 1]
                if chunk.numel() != seq_len + 1:
                    continue
                inputs = chunk[:-1].unsqueeze(0).to(device=device)
                targets = chunk[1:].unsqueeze(0).to(device=device)
                logits, _ = model(inputs)
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)
                total_loss += float(F.cross_entropy(flat_logits, flat_targets, reduction="sum").item())
                total_tokens += int(flat_targets.numel())
                correct += int((flat_logits.argmax(dim=-1) == flat_targets).sum().item())
                batches += 1
    finally:
        _restore_runtime_state(model, runtime_snapshot)
        if was_training:
            model.train()

    if total_tokens == 0:
        raise ValueError("validation split is too small for the requested seq_len")

    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss) if avg_loss < 20 else float("inf"),
        "token_accuracy": correct / total_tokens,
        "batches": float(batches),
        "tokens": float(total_tokens),
    }


def run_mode_degeneracy_probe(
    model: ReciprocatorOnlyLM,
    token_ids: torch.Tensor,
    seq_len: int,
    device: torch.device,
    *,
    max_batches: Optional[int],
    baseline_loss: Optional[float] = None,
) -> dict[str, object]:
    def evaluate_loss() -> float:
        metrics = evaluate_model(
            model,
            token_ids,
            seq_len,
            device,
            max_batches=max_batches,
        )
        return float(metrics["loss"])

    return mode_axis_permutation_probe(
        model,
        evaluate_fn=evaluate_loss,
        metric_name="loss",
        baseline_metric=baseline_loss,
    )


def _split_tokens(
    token_ids: list[int],
    seq_len: int,
    val_fraction: float,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    all_tokens = torch.tensor(token_ids, dtype=torch.long)
    if val_fraction <= 0.0:
        return all_tokens, None

    min_tokens = seq_len + 1
    if all_tokens.numel() < min_tokens * 2:
        return all_tokens, None

    val_tokens = max(min_tokens, int(all_tokens.numel() * val_fraction))
    if val_tokens >= all_tokens.numel() - min_tokens:
        return all_tokens, None

    split_index = all_tokens.numel() - val_tokens
    return all_tokens[:split_index].clone(), all_tokens[split_index:].clone()


def _training_checkpoint_payload(
    *,
    model: ReciprocatorOnlyLM,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    tokenizer: SentencePieceTokenizer,
    step: int,
    best_metric: float,
    best_metric_name: str,
    last_train_loss: float,
    last_val_metrics: Optional[dict[str, float]],
    last_benchmark_metrics: Optional[dict[str, dict[str, float]]],
    benchmark_history: list[dict[str, object]],
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "config": asdict(config),
        "model_state_dict": _recursive_to_cpu(model.state_dict()),
        "optimizer_state_dict": _recursive_to_cpu(optimizer.state_dict()),
        "tokenizer_model_proto": tokenizer.processor.serialized_model_proto(),
        "step": int(step),
        "best_metric": float(best_metric),
        "best_metric_name": best_metric_name,
        "last_train_loss": float(last_train_loss),
        "last_val_metrics": None if last_val_metrics is None else dict(last_val_metrics),
        "last_benchmark_metrics": None if last_benchmark_metrics is None else copy.deepcopy(last_benchmark_metrics),
        "benchmark_history": copy.deepcopy(benchmark_history),
        "metadata": dict(metadata),
    }


def _load_training_state(
    path: Path,
    device: torch.device,
    lr: float,
) -> tuple[ModelConfig, SentencePieceTokenizer, ReciprocatorOnlyLM, torch.optim.Optimizer, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config_payload = payload["config"]
    config = config_payload if isinstance(config_payload, ModelConfig) else ModelConfig(**config_payload)
    tokenizer_proto = payload.get("tokenizer_model_proto")
    if tokenizer_proto is None:
        raise ValueError("resume checkpoint does not contain a tokenizer model proto")
    tokenizer = SentencePieceTokenizer.from_serialized_proto(tokenizer_proto)

    model = ReciprocatorOnlyLM(config).to(device)
    model.load_state_dict(payload["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device=device)

    return config, tokenizer, model, optimizer, payload


def _derive_checkpoint_paths(
    latest_path: Path,
    best_path: Path,
    resume_path: Optional[Path],
) -> tuple[Path, Path]:
    if resume_path is None:
        return latest_path, best_path

    stem = resume_path.stem
    if stem.endswith("_latest"):
        prefix = stem[: -len("_latest")]
    elif stem.endswith("_best"):
        prefix = stem[: -len("_best")]
    else:
        prefix = stem

    resolved_latest = latest_path
    resolved_best = best_path
    if latest_path == DEFAULT_LATEST_CHECKPOINT:
        resolved_latest = resume_path.parent / f"{prefix}_latest.pt"
    if best_path == DEFAULT_BEST_CHECKPOINT:
        resolved_best = resume_path.parent / f"{prefix}_best.pt"
    return resolved_latest, resolved_best


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ReciprocatorOnlyLM on bundled corpora.")
    add_device_argument(parser, default="auto")
    parser.add_argument("--steps", type=int, default=1000, help="Total optimization steps to run.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=512)
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
        help="Enable novelty-driven rank growth up to --max-state-rank during training or online mode.",
    )
    parser.add_argument(
        "--init-mode-sizes",
        type=_parse_size_tuple,
        default=None,
        help=(
            "Optional comma-separated init mode sizes. "
            "If omitted, derived from --init-state-capacity under the requested rank."
        ),
    )
    parser.add_argument(
        "--max-mode-sizes",
        type=_parse_size_tuple,
        default=None,
        help=(
            "Optional comma-separated max mode sizes. "
            "If omitted, derived from --state-capacity under the requested rank."
        ),
    )
    parser.add_argument(
        "--init-state-capacity",
        type=int,
        default=None,
        help=(
            "Initial active state capacity used to derive init mode sizes when "
            "--init-mode-sizes is omitted. Defaults to the same capacity as --state-capacity "
            "for fresh runs, so the full manifold trains from step 1."
        ),
    )
    parser.add_argument(
        "--state-capacity",
        type=int,
        default=None,
        help=(
            "Maximum state capacity used to derive max mode sizes when "
            "--max-mode-sizes is omitted. Defaults to 64 for fresh runs."
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
    parser.add_argument(
        "--learnable-prediction-eta",
        action="store_true",
        help="Learn the anticipation gain eta instead of keeping it fixed at --prediction-eta.",
    )
    parser.add_argument(
        "--learnable-coupling-temperature",
        action="store_true",
        help="Learn the phase-aware coupling temperature instead of keeping it fixed.",
    )
    parser.add_argument(
        "--learned-normalization-blend",
        action="store_true",
        help="Learn a blend between normalization families instead of using a fixed normalization path.",
    )
    parser.add_argument(
        "--all-learnable-mixer-params",
        action="store_true",
        help=(
            "Enable all optional learnable reciprocator mixer controls: "
            "learnable eta, learnable coupling temperature, learned per-mode scaling, "
            "and learned normalization blend."
        ),
    )
    parser.add_argument("--growth-threshold", type=float, default=0.02)
    parser.add_argument("--growth-interval", type=int, default=1)
    parser.add_argument(
        "--parallel-mixer",
        action="store_true",
        help="Use the parallel Reciprocator mixer. Only supported for non-streaming training.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default=None,
        help="Learning-rate schedule. Fresh runs default to constant; resumes default to the saved value.",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=None,
        help="Warmup fraction for cosine LR. Fresh runs default to 0.0; resumes default to the saved value.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=None,
        help="Final LR / base LR for cosine decay. Fresh runs default to 0.0; resumes default to the saved value.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Clip gradient norm to this value. Fresh runs default to 0.0 (disabled); resumes default to the saved value.",
    )
    parser.add_argument(
        "--lr-step-offset",
        type=int,
        default=None,
        help=(
            "Shift the learning-rate schedule origin by this many global steps. "
            "Useful when resuming into a dedicated taper phase so cosine decay is "
            "measured relative to the resumed segment instead of the original run."
        ),
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument(
        "--benchmark-examples",
        type=int,
        default=0,
        help="Run periodic synthetic benchmark-suite probes with this many examples per task. Disabled when 0.",
    )
    parser.add_argument(
        "--benchmark-every",
        type=int,
        default=0,
        help="Probe the synthetic benchmark suite every N steps. Defaults to --eval-every when probes are enabled.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Validation fraction for fresh runs; resumes default to the saved checkpoint value.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=None,
        help="Optional existing SentencePiece .model file. Ignored when --resume is used.",
    )
    parser.add_argument(
        "--tokenizer-prefix",
        type=Path,
        default=DEFAULT_TOKENIZER_PREFIX,
        help="Output prefix used when training a SentencePiece model for a fresh run.",
    )
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
        help="Use the legacy real-valued routing collapse for coupling ablations.",
    )
    parser.add_argument(
        "--coupling-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for phase-aware routing magnitudes.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a training checkpoint produced by this script.",
    )
    parser.add_argument(
        "--latest-checkpoint",
        type=Path,
        default=DEFAULT_LATEST_CHECKPOINT,
        help="Path for periodic latest training checkpoints.",
    )
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=DEFAULT_BEST_CHECKPOINT,
        help="Path for best training checkpoint, selected by validation loss when available.",
    )
    parser.add_argument(
        "--skip-online-demo",
        action="store_true",
        help="Skip the online adaptation demo after training finishes.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Optional exported model checkpoint with embedded tokenizer.",
    )
    parser.add_argument(
        "--training-mode",
        choices=("random", "streaming"),
        default=None,
        help=(
            "Fresh runs default to random-window training. "
            "Streaming mode uses contiguous chunks with online reciprocator state inside "
            "each optimizer step and treats --batch-size as chunks-per-step."
        ),
    )
    parser.add_argument(
        "--stream-reset-policy",
        choices=("wrap", "never"),
        default=None,
        help=(
            "Streaming mode only. 'wrap' resets persistent reciprocator state when the "
            "training stream wraps to the corpus start; 'never' keeps state across wraps. "
            "Document boundaries are not tracked separately."
        ),
    )
    parser.add_argument(
        "--tbptt-horizon",
        type=int,
        default=None,
        help=(
            "Streaming mode only. Number of contiguous chunks to backprop through before "
            "detaching persistent reciprocator state. Fresh runs default to disabled (0); "
            "resumes default to the checkpoint's saved value."
        ),
    )
    return parser


def _resolve_optional_learnable_mixer_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.all_learnable_mixer_params:
        args.learnable_prediction_eta = True
        args.learnable_coupling_temperature = True
        args.learned_per_mode_scaling = True
        args.learned_normalization_blend = True
    return args


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    args = _resolve_optional_learnable_mixer_args(args)

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be positive")
    if args.benchmark_examples < 0:
        raise ValueError("--benchmark-examples must be non-negative")
    if args.benchmark_every < 0:
        raise ValueError("--benchmark-every must be non-negative")
    if args.val_fraction is not None and not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    if args.warmup_fraction is not None and not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("--warmup-fraction must be in [0, 1)")
    if args.min_lr_ratio is not None and not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if args.grad_clip is not None and args.grad_clip < 0.0:
        raise ValueError("--grad-clip must be non-negative")
    if args.coupling_temperature <= 0.0:
        raise ValueError("--coupling-temperature must be positive")
    if args.tbptt_horizon is not None and args.tbptt_horizon < 0:
        raise ValueError("--tbptt-horizon must be non-negative")
    if args.parallel_mixer and args.training_mode == "streaming":
        raise ValueError("--parallel-mixer is incompatible with --training-mode streaming")

    torch.manual_seed(args.seed)

    print("Loading corpora...")
    text = read_corpus_text("plato_jowett") + "\n" + read_corpus_text("greek_philosophy_classics")
    print(f"Corpus size: {len(text):,} characters")

    device = resolve_torch_device(args.device)
    print(f"Device: {device}")

    latest_checkpoint, best_checkpoint = _derive_checkpoint_paths(
        args.latest_checkpoint,
        args.best_checkpoint,
        args.resume,
    )

    start_step = 0
    best_metric = float("inf")
    best_metric_name = "val_loss"
    last_val_metrics: Optional[dict[str, float]] = None
    last_benchmark_metrics: Optional[dict[str, dict[str, float]]] = None
    benchmark_history: list[dict[str, object]] = []
    metadata: dict[str, object] = {}
    loaded_training_mode: Optional[str] = None
    loaded_stream_reset_policy: Optional[str] = None
    loaded_tbptt_horizon: Optional[int] = None
    loaded_lr_schedule: Optional[str] = None
    loaded_warmup_fraction: Optional[float] = None
    loaded_min_lr_ratio: Optional[float] = None
    loaded_grad_clip: Optional[float] = None
    loaded_lr_step_offset: Optional[int] = None

    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        config, tokenizer, model, optimizer, payload = _load_training_state(args.resume, device, args.lr)
        start_step = int(payload.get("step", 0))
        best_metric = float(payload.get("best_metric", float("inf")))
        best_metric_name = str(payload.get("best_metric_name", "val_loss"))
        loaded_val_metrics = payload.get("last_val_metrics")
        last_val_metrics = None if loaded_val_metrics is None else dict(loaded_val_metrics)
        loaded_benchmark_metrics = payload.get("last_benchmark_metrics")
        last_benchmark_metrics = None if loaded_benchmark_metrics is None else copy.deepcopy(loaded_benchmark_metrics)
        benchmark_history = [dict(entry) for entry in payload.get("benchmark_history", [])]
        metadata = dict(payload.get("metadata", {}))
        loaded_training_mode = str(metadata.get("training_mode", "random"))
        loaded_stream_reset_policy = str(metadata.get("stream_reset_policy", "wrap"))
        loaded_tbptt_horizon = int(metadata.get("tbptt_horizon", 0))
        loaded_lr_schedule = str(metadata.get("lr_schedule", "constant"))
        loaded_warmup_fraction = float(metadata.get("warmup_fraction", 0.0))
        loaded_min_lr_ratio = float(metadata.get("min_lr_ratio", 0.0))
        loaded_grad_clip = float(metadata.get("grad_clip", 0.0))
        loaded_lr_step_offset = int(metadata.get("lr_step_offset", 0))
        val_fraction = float(metadata.get("val_fraction", 0.05)) if args.val_fraction is None else args.val_fraction
    else:
        if args.tokenizer_model is not None:
            print(f"Loading SentencePiece tokenizer from {args.tokenizer_model}...")
            tokenizer = SentencePieceTokenizer.from_model_file(args.tokenizer_model)
        else:
            print("Training SentencePiece tokenizer...")
            tokenizer = train_sentencepiece_tokenizer(
                text=text,
                vocab_size=args.vocab_size,
                model_prefix=args.tokenizer_prefix,
            )

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
            learnable_prediction_eta=args.learnable_prediction_eta,
            learnable_coupling_temperature=args.learnable_coupling_temperature,
            learned_normalization_blend=args.learned_normalization_blend,
            dropout=args.dropout,
            growth_threshold=args.growth_threshold,
            growth_interval=args.growth_interval,
            persist_state=args.training_mode == "streaming",
            parallel_mixer=args.parallel_mixer,
            input_dependent_gains=True,
            accumulator_modulates_gains=True,
            phase_aware_readout=args.phase_aware_readout,
            phase_aware_coupling=args.phase_aware_coupling,
            coupling_temperature=args.coupling_temperature,
        )
        model = ReciprocatorOnlyLM(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        val_fraction = 0.05 if args.val_fraction is None else args.val_fraction

    training_mode = args.training_mode or loaded_training_mode or "random"
    if training_mode not in {"random", "streaming"}:
        raise ValueError(f"Unsupported training mode: {training_mode}")
    stream_reset_policy = args.stream_reset_policy or loaded_stream_reset_policy or "wrap"
    if stream_reset_policy not in {"wrap", "never"}:
        raise ValueError(f"Unsupported stream reset policy: {stream_reset_policy}")
    tbptt_horizon = int(loaded_tbptt_horizon or 0) if args.tbptt_horizon is None else int(args.tbptt_horizon)
    if training_mode != "streaming" and tbptt_horizon != 0:
        raise ValueError("--tbptt-horizon is only supported with --training-mode streaming")
    lr_schedule = args.lr_schedule or loaded_lr_schedule or "constant"
    if lr_schedule not in {"constant", "cosine"}:
        raise ValueError(f"Unsupported learning-rate schedule: {lr_schedule}")
    warmup_fraction = loaded_warmup_fraction if args.warmup_fraction is None else args.warmup_fraction
    if warmup_fraction is None:
        warmup_fraction = 0.0
    min_lr_ratio = loaded_min_lr_ratio if args.min_lr_ratio is None else args.min_lr_ratio
    if min_lr_ratio is None:
        min_lr_ratio = 0.0
    grad_clip = loaded_grad_clip if args.grad_clip is None else args.grad_clip
    if grad_clip is None:
        grad_clip = 0.0
    lr_step_offset = loaded_lr_step_offset if args.lr_step_offset is None else args.lr_step_offset
    if lr_step_offset is None:
        lr_step_offset = 0
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if grad_clip < 0.0:
        raise ValueError("grad_clip must be non-negative")
    if lr_step_offset < 0:
        raise ValueError("lr_step_offset must be non-negative")
    if lr_step_offset >= args.steps:
        raise ValueError("lr_step_offset must be smaller than --steps")

    config.persist_state = training_mode == "streaming"
    if config.parallel_mixer and config.persist_state:
        raise ValueError("parallel_mixer is incompatible with streaming/persistent state")
    _set_persistent_training_mode(
        model,
        enabled=config.persist_state,
        reset_state=args.resume is None or loaded_training_mode != training_mode,
    )
    model.set_online_state_gradient_tracking(False)

    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    print(f"Tokenized: {len(token_ids):,} tokens (vocab_size={tokenizer.vocab_size})")
    benchmark_names = tuple(benchmark.name for benchmark in build_default_benchmark_suite(tokenizer.vocab_size))
    benchmark_every = args.benchmark_every if args.benchmark_every > 0 else args.eval_every

    train_tokens, val_tokens = _split_tokens(token_ids, config.max_seq_len, val_fraction)
    print(f"Training tokens: {train_tokens.numel():,}")
    if val_tokens is None:
        print("Validation split: disabled")
        best_metric_name = "train_loss"
    else:
        print(f"Validation tokens: {val_tokens.numel():,}")
        best_metric_name = "val_loss"

    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(
        "Config: "
        f"{config.n_layers} layers, dim={config.dim}, seq_len={config.max_seq_len}, "
        f"state_rank={config.state_rank}, max_state_rank={config.max_state_rank}, "
        f"init_mode_sizes={config.init_mode_sizes}, max_mode_sizes={config.max_mode_sizes}, "
        f"dynamic_rank={config.dynamic_rank}, "
        f"accumulator_modulates_gains={config.accumulator_modulates_gains}, "
        f"input_dependent_gains={config.input_dependent_gains}, "
        f"selective_gains={config.selective_gains}, "
        f"learned_per_mode_scaling={config.learned_per_mode_scaling}, "
        f"learnable_prediction_eta={config.learnable_prediction_eta}, "
        f"learnable_coupling_temperature={config.learnable_coupling_temperature}, "
        f"learned_normalization_blend={config.learned_normalization_blend}, "
        f"phase_aware_readout={config.phase_aware_readout}, "
        f"phase_aware_coupling={config.phase_aware_coupling}, "
        f"coupling_temperature={config.coupling_temperature}, training_mode={training_mode}"
    )
    optimizer_summary = (
        f"Optimizer: lr={args.lr:g}, lr_schedule={lr_schedule}, "
        f"warmup_fraction={warmup_fraction:.3f}, min_lr_ratio={min_lr_ratio:.3f}, "
        f"grad_clip={grad_clip:.3f}"
    )
    if lr_step_offset > 0:
        optimizer_summary += f", lr_step_offset={lr_step_offset}"
    print(optimizer_summary)
    if training_mode == "streaming":
        print(
            f"Streaming chunks per optimizer step: {args.batch_size} "
            f"(tbptt_horizon={tbptt_horizon if tbptt_horizon > 0 else 'disabled'}) "
            f"(stream_reset_policy={stream_reset_policy})"
        )
        if tbptt_horizon > 0:
            print("TBPTT note: persistent-state gradient tracking disables online growth/pruning during training.")
    if args.benchmark_examples > 0:
        print(
            "Benchmark probes: "
            f"{len(benchmark_names)} tasks, {args.benchmark_examples} examples/task, every {benchmark_every} steps"
        )
    print(f"Latest checkpoint: {latest_checkpoint}")
    print(f"Best checkpoint: {best_checkpoint}")
    print()

    train_stream: Optional[ContiguousTokenStream] = None
    if training_mode == "streaming":
        train_stream = ContiguousTokenStream(
            train_tokens,
            config.max_seq_len,
            position=int(metadata.get("stream_position", 0)),
            wrap_count=int(metadata.get("stream_wrap_count", 0)),
        )

    model.train()
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        current_lr = args.lr * _lr_multiplier(
            step=step,
            total_steps=args.steps,
            schedule=lr_schedule,
            warmup_fraction=warmup_fraction,
            min_lr_ratio=min_lr_ratio,
            step_offset=lr_step_offset,
        )
        _set_optimizer_lr(optimizer, current_lr)
        current_growth_threshold = _annealed_growth_threshold(config.growth_threshold, step, args.steps)
        _set_growth_threshold(model, current_growth_threshold)
        growth_events_before = _count_growth_events(model)
        optimizer.zero_grad(set_to_none=True)
        if training_mode == "streaming":
            assert train_stream is not None
            train_loss, _ = _streaming_train_step(
                model=model,
                train_stream=train_stream,
                device=device,
                chunks_per_step=args.batch_size,
                stream_reset_policy=stream_reset_policy,
                tbptt_horizon=tbptt_horizon,
            )
        else:
            inputs, targets = build_dataset(train_tokens, config.max_seq_len, args.batch_size, device)
            _, loss = model(inputs, targets)
            assert loss is not None
            loss.backward()
            train_loss = float(loss.item())
        grad_norm = None
        if grad_clip > 0.0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
        growth_events = _count_growth_events(model) - growth_events_before
        if growth_events > 0:
            _reset_optimizer_moments(optimizer)
        optimizer.step()
        if training_mode == "streaming":
            model.detach_online_state()

        metric_value = train_loss
        ran_eval = False
        if val_tokens is not None and (
            step == start_step + 1 or step % args.eval_every == 0 or step == args.steps
        ):
            ran_eval = True
            last_val_metrics = evaluate_model(
                model,
                val_tokens,
                config.max_seq_len,
                device,
                max_batches=args.eval_batches,
            )
            metric_value = float(last_val_metrics["loss"])
        if args.benchmark_examples > 0 and (
            step == start_step + 1 or step % benchmark_every == 0 or step == args.steps
        ):
            last_benchmark_metrics = evaluate_named_benchmark_tasks(
                model,
                benchmark_names=benchmark_names,
                vocab_size=config.vocab_size,
                num_examples=args.benchmark_examples,
                device=device,
                seed=args.seed,
            )
            benchmark_history.append(
                {
                    "step": float(step),
                    "mean_accuracy": float(
                        sum(metrics["accuracy"] for metrics in last_benchmark_metrics.values())
                        / len(last_benchmark_metrics)
                    ),
                    "tasks": copy.deepcopy(last_benchmark_metrics),
                }
            )

        if ((val_tokens is None) or ran_eval) and metric_value < best_metric:
            best_metric = metric_value
            payload = _training_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                step=step,
                best_metric=best_metric,
                best_metric_name=best_metric_name,
                last_train_loss=train_loss,
                last_val_metrics=last_val_metrics,
                last_benchmark_metrics=last_benchmark_metrics,
                benchmark_history=benchmark_history,
                metadata={
                    "script": "train_reciprocator_only.py",
                    "seed": int(args.seed),
                    "val_fraction": float(val_fraction),
                    "training_mode": training_mode,
                    "stream_reset_policy": stream_reset_policy,
                    "tbptt_horizon": int(tbptt_horizon),
                    "benchmark_examples": int(args.benchmark_examples),
                    "benchmark_every": int(benchmark_every),
                    "lr_schedule": lr_schedule,
                    "warmup_fraction": float(warmup_fraction),
                    "min_lr_ratio": float(min_lr_ratio),
                    "grad_clip": float(grad_clip),
                    "lr_step_offset": int(lr_step_offset),
                    "stream_position": None if train_stream is None else int(train_stream.position),
                    "stream_wrap_count": None if train_stream is None else int(train_stream.wrap_count),
                },
            )
            _atomic_torch_save(best_checkpoint, payload)

        if step % args.save_every == 0 or step == args.steps:
            payload = _training_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                step=step,
                best_metric=best_metric,
                best_metric_name=best_metric_name,
                last_train_loss=train_loss,
                last_val_metrics=last_val_metrics,
                last_benchmark_metrics=last_benchmark_metrics,
                benchmark_history=benchmark_history,
                metadata={
                    "script": "train_reciprocator_only.py",
                    "seed": int(args.seed),
                    "val_fraction": float(val_fraction),
                    "training_mode": training_mode,
                    "stream_reset_policy": stream_reset_policy,
                    "tbptt_horizon": int(tbptt_horizon),
                    "benchmark_examples": int(args.benchmark_examples),
                    "benchmark_every": int(benchmark_every),
                    "lr_schedule": lr_schedule,
                    "warmup_fraction": float(warmup_fraction),
                    "min_lr_ratio": float(min_lr_ratio),
                    "grad_clip": float(grad_clip),
                    "lr_step_offset": int(lr_step_offset),
                    "stream_position": None if train_stream is None else int(train_stream.position),
                    "stream_wrap_count": None if train_stream is None else int(train_stream.wrap_count),
                },
            )
            _atomic_torch_save(latest_checkpoint, payload)

        if step == start_step + 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.time() - t0
            if last_val_metrics is None:
                print(
                    f"step={step:5d}  train_loss={train_loss:.4f}  "
                    f"lr={current_lr:.6f}  "
                    f"growth_threshold={current_growth_threshold:.4f}  "
                    f"growth_events={growth_events}  "
                    f"best_{best_metric_name}={best_metric:.4f}  elapsed={elapsed:.1f}s"
                )
            else:
                print(
                    f"step={step:5d}  train_loss={train_loss:.4f}  "
                    f"lr={current_lr:.6f}  "
                    f"growth_threshold={current_growth_threshold:.4f}  "
                    f"growth_events={growth_events}  "
                    f"val_loss={last_val_metrics['loss']:.4f}  "
                    f"val_ppl={last_val_metrics['perplexity']:.2f}  "
                    f"val_acc={last_val_metrics['token_accuracy']:.4f}  "
                    f"best_{best_metric_name}={best_metric:.4f}  elapsed={elapsed:.1f}s"
                )
            if last_benchmark_metrics is not None:
                benchmark_mean = sum(metrics["accuracy"] for metrics in last_benchmark_metrics.values()) / len(last_benchmark_metrics)
                print(
                    "           "
                    f"bench_mean={benchmark_mean:.4f}  "
                    f"bench_cn={last_benchmark_metrics['controlled_novelty']['accuracy']:.4f}  "
                    f"bench_rr={last_benchmark_metrics['role_rebinding']['accuracy']:.4f}"
                )
            if grad_norm is not None:
                print(f"           grad_norm={grad_norm:.4f}")

    probe_tokens = val_tokens if val_tokens is not None else train_tokens
    probe_split = "validation" if val_tokens is not None else "train"
    probe_baseline_loss = (
        None if val_tokens is None else None if last_val_metrics is None else float(last_val_metrics["loss"])
    )
    probe_batches = min(args.eval_batches or 8, 3)

    # Use a single fixed batch as a fast proxy for the degeneracy probe.
    probe_batch_size = 2
    probe_generator = torch.Generator(device="cpu").manual_seed(args.seed + 999)
    probe_input_ids, probe_target_ids = build_dataset(
        probe_tokens, config.max_seq_len, probe_batch_size, device, generator=probe_generator,
    )
    model.eval()
    with torch.no_grad():
        baseline_logits, baseline_probe_loss = model(probe_input_ids, probe_target_ids)
    if probe_baseline_loss is None and baseline_probe_loss is not None:
        probe_baseline_loss = float(baseline_probe_loss.item())

    def _probe_loss() -> float:
        with torch.no_grad():
            logits, loss = model(probe_input_ids, probe_target_ids)
        return float(loss.item()) if loss is not None else float(logits.sum().item())

    mode_probe = mode_axis_permutation_probe(
        model,
        evaluate_fn=_probe_loss,
        metric_name="loss",
        baseline_metric=probe_baseline_loss,
    )
    print("\n--- Mode degeneracy probe ---")
    print(
        f"probe_split={probe_split} "
        f"baseline_loss={float(mode_probe['baseline_loss']):.4f} "
        f"probe_batches=1(fixed) probe_batch_size={probe_batch_size}"
    )
    for pair_key, pair_result in mode_probe["pairs"].items():
        if not pair_result["supported"]:
            print(f"swap({pair_key})  skipped  reason={pair_result['reason']}")
            continue
        skip_reason = pair_result.get("metric_skipped_reason")
        if skip_reason is not None:
            print(
                f"swap({pair_key})  "
                f"param_delta_mean={float(pair_result['parameter_relative_delta_mean']):.4f}  "
                f"param_delta_max={float(pair_result['parameter_relative_delta_max']):.4f}  "
                f"[metric skipped: {skip_reason}]"
            )
        else:
            print(
                f"swap({pair_key})  "
                f"param_delta_mean={float(pair_result['parameter_relative_delta_mean']):.4f}  "
                f"param_delta_max={float(pair_result['parameter_relative_delta_max']):.4f}  "
                f"loss={float(pair_result['loss']):.4f}  "
                f"loss_delta={float(pair_result['loss_delta']):+.4f}  "
                f"loss_rel_delta={float(pair_result['loss_relative_delta']):.4f}"
            )
    if (
        int(mode_probe.get("supported_pair_count", 0)) > 0
        and float(mode_probe.get("max_loss_relative_delta", 0.0)) < 0.01
    ):
        print(
            "Probe warning: mode-axis swaps leave loss nearly unchanged on this split; "
            "treat the subject/entity/feature interpretation as unsupported unless you add an asymmetric prior."
        )

    print("\n--- Generation sample ---")
    model.eval()
    prompts = ["Socrates", "The soul", "Justice is", "Knowledge"]
    for prompt_text in prompts:
        model.reset_online_state()
        prompt_ids = tokenizer.encode(prompt_text, add_bos=True)
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(prompt, max_new_tokens=80, temperature=0.8)[0].tolist()
        print(f"\n[{prompt_text}] -> {tokenizer.decode(generated)}")

    if not args.skip_online_demo:
        print("\n--- Online adaptation demo ---")
        model.enter_online_mode()
        model.reset_online_state()
        online_windows = min(4, max(1, len(token_ids) // max(1, config.max_seq_len)))
        with torch.no_grad():
            for window_index in range(online_windows):
                start = window_index * config.max_seq_len
                chunk = token_ids[start : start + config.max_seq_len]
                if len(chunk) < 2:
                    break
                online_input = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
                model(online_input)
                diagnostics = model.online_diagnostics()
                first_layer = diagnostics["layers"][0]
                first_engine = first_layer["engines"][0]
                print(
                    f"online_window={window_index + 1} "
                    f"active_rank={first_layer['active_rank']} "
                    f"active_sizes={first_layer['active_sizes']} "
                    f"last_novelty={first_engine['last_novelty']:.4f}"
                )

        prompt_ids = tokenizer.encode("Socrates", add_bos=True)
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(prompt, max_new_tokens=80, temperature=0.8)[0].tolist()
        print(f"\n[online memory] -> {tokenizer.decode(generated)}")

    if args.checkpoint_out is not None:
        save_reciprocator_checkpoint(
            args.checkpoint_out,
            model=model,
            config=config,
            tokenizer=tokenizer,
            metadata={
                "script": "train_reciprocator_only.py",
                "steps": int(args.steps),
                "batch_size": int(args.batch_size),
                "seq_len": int(config.max_seq_len),
                "dim": int(config.dim),
                "layers": int(config.n_layers),
                "best_metric_name": best_metric_name,
                "best_metric": float(best_metric),
                "resume": None if args.resume is None else str(args.resume),
                "training_mode": training_mode,
                "stream_reset_policy": stream_reset_policy,
                "tbptt_horizon": int(tbptt_horizon),
                "benchmark_examples": int(args.benchmark_examples),
                "benchmark_every": int(benchmark_every),
                "last_benchmark_metrics": None if last_benchmark_metrics is None else copy.deepcopy(last_benchmark_metrics),
                "benchmark_history": copy.deepcopy(benchmark_history),
                "lr_schedule": lr_schedule,
                "warmup_fraction": float(warmup_fraction),
                "min_lr_ratio": float(min_lr_ratio),
                "grad_clip": float(grad_clip),
                "lr_step_offset": int(lr_step_offset),
                "mode_degeneracy_probe": mode_probe,
            },
        )
        print(f"\nSaved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
