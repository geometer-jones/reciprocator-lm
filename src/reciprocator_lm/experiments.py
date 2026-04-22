from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .baselines import BaselineTransformerConfig, SmallMambaConfig
from .benchmarks import BENCHMARK_FACTORIES, build_default_benchmark_suite, sequence_accuracy
from .config import ModelConfig
from .sleep import sample_causal_lm_batch


@dataclass(frozen=True)
class TrainingRunConfig:
    steps: int = 1000
    batch_size: int = 16
    seq_len: int = 64
    lr: float = 3e-4
    save_every: int = 100
    eval_every: int = 100
    eval_batches: int = 16
    log_every: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.save_every <= 0:
            raise ValueError("save_every must be positive.")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive.")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive.")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")


@dataclass(frozen=True)
class BenchmarkTrainingConfig:
    steps: int = 1000
    batch_size: int = 64
    lr: float = 3e-4
    eval_every: int = 100
    eval_examples: int = 256
    log_every: int = 50
    seed: int = 0
    streaming: bool = False

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive.")
        if self.eval_examples <= 0:
            raise ValueError("eval_examples must be positive.")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")


@dataclass(frozen=True)
class CapacityMatch:
    parameter_name: str
    parameter_value: int
    parameter_count: int
    target_parameter_count: int
    relative_gap: float


@dataclass(frozen=True)
class BudgetMatch:
    parameter_name: str
    parameter_value: int
    parameter_count: int
    target_parameter_count: int
    parameter_relative_gap: float
    train_flops_per_step: float
    target_train_flops_per_step: float
    train_flops_relative_gap: float


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _real_linear_forward_flops(*, in_features: int, out_features: int, tokens: int) -> float:
    return float(2 * tokens * in_features * out_features)


def _complex_linear_forward_flops(*, in_features: int, out_features: int, tokens: int) -> float:
    return float(8 * tokens * in_features * out_features)


def estimate_plain_transformer_train_flops(
    config: BaselineTransformerConfig,
    *,
    batch_size: int,
    seq_len: int,
) -> float:
    """Approximate per-step training FLOPs for PlainTransformerLM.

    The estimate intentionally tracks only the dominant tensor ops so different
    architectures can be compared on the same scale. A single 3x multiplier is
    used for the backward/update cost across all models.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")

    tokens = batch_size * seq_len
    ffw_dim = config.model_dim * config.ffw_multiplier

    layer_forward = 0.0
    layer_forward += 4.0 * _real_linear_forward_flops(
        in_features=config.model_dim,
        out_features=config.model_dim,
        tokens=tokens,
    )
    layer_forward += float(4 * batch_size * seq_len * seq_len * config.model_dim)
    layer_forward += _real_linear_forward_flops(
        in_features=config.model_dim,
        out_features=ffw_dim,
        tokens=tokens,
    )
    layer_forward += _real_linear_forward_flops(
        in_features=ffw_dim,
        out_features=config.model_dim,
        tokens=tokens,
    )

    lm_head_forward = _real_linear_forward_flops(
        in_features=config.model_dim,
        out_features=config.vocab_size,
        tokens=tokens,
    )
    return 3.0 * (config.num_layers * layer_forward + lm_head_forward)


def estimate_reciprocator_only_train_flops(
    config: ModelConfig,
    *,
    batch_size: int,
    seq_len: int,
) -> float:
    """Approximate per-step training FLOPs for ReciprocatorOnlyLM."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")

    tokens = batch_size * seq_len
    ffw_dim = int(config.dim * config.mlp_ratio)
    readout_dim = config.dim * 3 if config.phase_aware_readout else config.dim
    effective_state_rank = config.max_state_rank if config.max_state_rank > 0 else config.state_rank
    gain_context_dim = effective_state_rank + 3
    gain_signal_hidden_dim = 12
    gain_selection_hidden_dim = 8
    scan_rounds = max(1, math.ceil(math.log2(seq_len)))
    coupling_cost = sum(4.0 * tokens * config.state_dim * mode_size for mode_size in config.state_mode_sizes)

    engine_forward = 0.0
    engine_forward += float(6 * tokens * config.state_dim * scan_rounds)
    engine_forward += float(6 * tokens * config.state_dim)
    engine_forward += float(6 * effective_state_rank * tokens * config.state_dim)
    engine_forward += coupling_cost
    engine_forward += float(8 * tokens * config.state_dim)
    engine_forward += _complex_linear_forward_flops(
        in_features=3 * config.state_dim,
        out_features=config.dim,
        tokens=tokens,
    )

    layer_forward = 0.0
    layer_forward += _complex_linear_forward_flops(
        in_features=config.dim,
        out_features=config.state_dim,
        tokens=tokens,
    )
    layer_forward += float(config.num_cube_engines) * engine_forward
    layer_forward += _complex_linear_forward_flops(
        in_features=config.dim * config.num_cube_engines,
        out_features=config.dim,
        tokens=tokens,
    )
    if config.input_dependent_gains:
        layer_forward += float(config.num_cube_engines) * _real_linear_forward_flops(
            in_features=3,
            out_features=gain_signal_hidden_dim,
            tokens=tokens * config.state_dim,
        )
        layer_forward += float(config.num_cube_engines) * _real_linear_forward_flops(
            in_features=gain_signal_hidden_dim,
            out_features=4,
            tokens=tokens * config.state_dim,
        )
        if config.parallel_mixer:
            layer_forward += float(config.num_cube_engines) * float(6 * tokens * config.state_dim * scan_rounds)
            layer_forward += float(config.num_cube_engines) * _real_linear_forward_flops(
                in_features=gain_context_dim,
                out_features=4 * config.state_dim,
                tokens=tokens,
            )
        else:
            layer_forward += float(config.num_cube_engines) * _real_linear_forward_flops(
                in_features=gain_context_dim,
                out_features=4 * config.state_dim,
                tokens=tokens,
            )
        if config.selective_gains:
            layer_forward += float(config.num_cube_engines) * _real_linear_forward_flops(
                in_features=gain_context_dim + 3,
                out_features=gain_selection_hidden_dim,
                tokens=tokens,
            )
            layer_forward += float(config.num_cube_engines) * _real_linear_forward_flops(
                in_features=gain_selection_hidden_dim,
                out_features=1,
                tokens=tokens,
            )
    layer_forward += _real_linear_forward_flops(
        in_features=4 * config.dim,
        out_features=config.dim,
        tokens=tokens,
    )
    layer_forward += _complex_linear_forward_flops(
        in_features=config.dim,
        out_features=ffw_dim,
        tokens=tokens,
    )
    layer_forward += _complex_linear_forward_flops(
        in_features=ffw_dim,
        out_features=config.dim,
        tokens=tokens,
    )

    lm_head_forward = _real_linear_forward_flops(
        in_features=readout_dim,
        out_features=config.vocab_size,
        tokens=tokens,
    )
    return 3.0 * (config.n_layers * layer_forward + lm_head_forward)


def estimate_small_mamba_train_flops(
    config: SmallMambaConfig,
    *,
    batch_size: int,
    seq_len: int,
) -> float:
    """Approximate per-step training FLOPs for SmallMambaLM."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")

    tokens = batch_size * seq_len
    inner_dim = config.model_dim * config.expand

    layer_forward = 0.0
    layer_forward += _real_linear_forward_flops(
        in_features=config.model_dim,
        out_features=inner_dim * 2,
        tokens=tokens,
    )
    layer_forward += float(2 * tokens * inner_dim * config.conv_kernel)
    layer_forward += _real_linear_forward_flops(
        in_features=inner_dim,
        out_features=inner_dim,
        tokens=tokens,
    )
    layer_forward += _real_linear_forward_flops(
        in_features=inner_dim,
        out_features=inner_dim * config.state_size,
        tokens=tokens,
    )
    layer_forward += _real_linear_forward_flops(
        in_features=inner_dim,
        out_features=inner_dim * config.state_size,
        tokens=tokens,
    )
    layer_forward += float(12 * tokens * inner_dim * config.state_size)
    layer_forward += float(2 * tokens * inner_dim)
    layer_forward += _real_linear_forward_flops(
        in_features=inner_dim,
        out_features=config.model_dim,
        tokens=tokens,
    )

    lm_head_forward = _real_linear_forward_flops(
        in_features=config.model_dim,
        out_features=config.vocab_size,
        tokens=tokens,
    )
    return 3.0 * (config.num_layers * layer_forward + lm_head_forward)


def split_train_val_tokens(
    token_ids: Sequence[int] | Tensor,
    seq_len: int,
    val_fraction: float,
) -> tuple[Tensor, Optional[Tensor]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")

    all_tokens = torch.as_tensor(token_ids, dtype=torch.long).clone()
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


def _forward_logits(
    model: nn.Module,
    input_ids: Tensor,
    *,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError:
        outputs = model(input_ids)

    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("model output dict must include 'logits'.")
        return logits
    if isinstance(outputs, tuple):
        if not outputs:
            raise ValueError("model output tuple must not be empty.")
        return outputs[0]
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def _causal_label_cross_entropy(logits: Tensor, labels: Tensor) -> tuple[Tensor, int]:
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels != -100
    total_tokens = int(valid_mask.sum().item())
    if total_tokens <= 0:
        raise ValueError("benchmark batch produced no supervised causal positions.")
    loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )
    return loss, total_tokens


def _recursive_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _recursive_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_recursive_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_recursive_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _cpu_state_dict(model: nn.Module) -> Dict[str, Any]:
    return {name: _recursive_to_cpu(value) for name, value in model.state_dict().items()}


def _overrides_extra_state_hook(module: Any, hook_name: str) -> bool:
    module_type = type(module)
    hook = getattr(module_type, hook_name, None)
    base_hook = getattr(nn.Module, hook_name, None)
    return callable(hook) and hook is not base_hook


def _snapshot_runtime_state(model: nn.Module) -> dict[str, object]:
    model_state: Any = None
    get_model_state = getattr(model, "get_extra_state", None)
    if _overrides_extra_state_hook(model, "get_extra_state") and callable(get_model_state):
        model_state = _recursive_to_cpu(get_model_state())

    mixer_states: list[Any] = []
    for block in getattr(model, "blocks", []):
        mixer = getattr(block, "mixer", None)
        get_mixer_state = getattr(mixer, "get_extra_state", None)
        mixer_states.append(
            None
            if not _overrides_extra_state_hook(mixer, "get_extra_state") or not callable(get_mixer_state)
            else _recursive_to_cpu(get_mixer_state())
        )

    return {
        "model": model_state,
        "mixers": mixer_states,
    }


def _restore_runtime_state(model: nn.Module, snapshot: dict[str, object]) -> None:
    set_model_state = getattr(model, "set_extra_state", None)
    model_state = snapshot.get("model")
    if _overrides_extra_state_hook(model, "set_extra_state") and callable(set_model_state) and model_state is not None:
        set_model_state(model_state)

    mixer_states = snapshot.get("mixers", [])
    for block, mixer_state in zip(getattr(model, "blocks", []), mixer_states):
        mixer = getattr(block, "mixer", None)
        set_mixer_state = getattr(mixer, "set_extra_state", None)
        if _overrides_extra_state_hook(mixer, "set_extra_state") and callable(set_mixer_state):
            set_mixer_state(mixer_state)


def _atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _iter_cube_engines(model: nn.Module):
    for block in getattr(model, "blocks", []):
        mixer = getattr(block, "mixer", None)
        for engine in getattr(mixer, "cube_engines", ()):
            yield engine


def _set_growth_threshold(model: nn.Module, threshold: float) -> None:
    for engine in _iter_cube_engines(model):
        engine.growth_threshold = float(threshold)


def _count_growth_events(model: nn.Module) -> int:
    return sum(int(engine._growth_event_count.item()) for engine in _iter_cube_engines(model))


def _nominal_growth_threshold(model: nn.Module) -> Optional[float]:
    first_engine = next(_iter_cube_engines(model), None)
    if first_engine is None:
        return None
    if hasattr(first_engine, "_nominal_growth_threshold"):
        return float(first_engine._nominal_growth_threshold)
    return float(first_engine.growth_threshold)


def _effective_growth_threshold(
    nominal_threshold: float,
    step: int,
    total_steps: int,
    warmup_steps: int = 0,
    warmup_multiplier: float = 10.0,
) -> float:
    """Schedule-independent growth threshold with explicit warmup period."""
    del total_steps
    if nominal_threshold <= 0.0:
        return nominal_threshold

    if step <= warmup_steps:
        # Strong suppression during warmup so the engine can learn patterns first.
        return nominal_threshold * warmup_multiplier

    return nominal_threshold


def _reset_optimizer_moments(optimizer: torch.optim.Optimizer) -> None:
    optimizer.state.clear()


def _training_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_step: int,
    best_val_loss: float,
    last_train_loss: float,
    last_val_metrics: Optional[Dict[str, float]],
    best_val_metrics: Optional[Dict[str, float]],
    batch_generator_state: Optional[Tensor],
    metadata: Optional[Dict[str, Any]],
    loss_history: Optional[list[Dict[str, float]]] = None,
    best_state_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": _recursive_to_cpu(optimizer.state_dict()),
        "step": int(step),
        "best_step": int(best_step),
        "best_val_loss": None if not math.isfinite(best_val_loss) else float(best_val_loss),
        "last_train_loss": float(last_train_loss),
        "last_val_metrics": None if last_val_metrics is None else dict(last_val_metrics),
        "best_val_metrics": None if best_val_metrics is None else dict(best_val_metrics),
        "batch_generator_state": None if batch_generator_state is None else batch_generator_state.detach().cpu().clone(),
        "best_state_dict": None if best_state_dict is None else _recursive_to_cpu(best_state_dict),
        "loss_history": [] if loss_history is None else copy.deepcopy(loss_history),
        "metadata": {} if metadata is None else copy.deepcopy(metadata),
    }


def _load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(optimizer, device)
    return payload


def evaluate_causal_lm(
    model: nn.Module,
    token_ids: Sequence[int] | Tensor,
    seq_len: int,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")

    tokens = torch.as_tensor(token_ids, dtype=torch.long)
    if tokens.numel() < seq_len + 1:
        raise ValueError("validation split is too small for the requested seq_len")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    batches = 0

    try:
        with torch.no_grad():
            max_start = tokens.numel() - seq_len - 1
            for start in range(0, max_start + 1, seq_len):
                if max_batches is not None and batches >= max_batches:
                    break
                chunk = tokens[start : start + seq_len + 1]
                if chunk.numel() != seq_len + 1:
                    continue
                input_ids = chunk[:-1].unsqueeze(0).to(device=device)
                target_ids = chunk[1:].unsqueeze(0).to(device=device)
                logits = _forward_logits(model, input_ids)
                total_loss += float(
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        target_ids.reshape(-1),
                        reduction="sum",
                    ).item()
                )
                total_tokens += int(target_ids.numel())
                correct += int((logits.argmax(dim=-1) == target_ids).sum().item())
                batches += 1
    finally:
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


def evaluate_benchmark_suite_generic(
    model: nn.Module,
    *,
    vocab_size: int,
    num_examples: int,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, float]:
    if num_examples <= 0:
        raise ValueError("num_examples must be positive.")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")

    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()
    results: Dict[str, float] = {}
    try:
        with torch.no_grad():
            for benchmark_index, benchmark in enumerate(build_default_benchmark_suite(vocab_size)):
                batch = benchmark.make_batch(num_examples, seed + benchmark_index, device)
                logits = _forward_logits(model, batch.input_ids, attention_mask=batch.attention_mask)
                results[benchmark.name] = sequence_accuracy(logits, batch.labels, batch.prediction_positions)
    finally:
        if was_training:
            model.train()
    return results


def evaluate_benchmark_task(
    model: nn.Module,
    *,
    benchmark_name: str,
    vocab_size: int,
    num_examples: int,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, float]:
    if num_examples <= 0:
        raise ValueError("num_examples must be positive.")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if benchmark_name not in BENCHMARK_FACTORIES:
        raise ValueError(f"unknown benchmark: {benchmark_name}")

    if device is None:
        device = next(model.parameters()).device

    batch = BENCHMARK_FACTORIES[benchmark_name](
        num_examples=num_examples,
        vocab_size=vocab_size,
        device=device,
        seed=seed,
    )
    was_training = model.training
    runtime_snapshot = _snapshot_runtime_state(model)
    model.eval()
    try:
        reset_online_state = getattr(model, "reset_online_state", None)
        if callable(reset_online_state):
            reset_online_state()
        with torch.no_grad():
            logits = _forward_logits(model, batch.input_ids, attention_mask=batch.attention_mask)
            shifted_logits = logits[:, :-1]
            shifted_labels = batch.labels[:, 1:]
            valid_mask = shifted_labels != -100
            total_loss = float(
                F.cross_entropy(
                    shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                    shifted_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ).item()
            )
            total_tokens = int(valid_mask.sum().item())
            return {
                "loss": total_loss / total_tokens,
                "accuracy": sequence_accuracy(logits, batch.labels, batch.prediction_positions),
                "tokens": float(total_tokens),
                "examples": float(batch.input_ids.shape[0]),
            }
    finally:
        _restore_runtime_state(model, runtime_snapshot)
        if was_training:
            model.train()


def evaluate_named_benchmark_tasks(
    model: nn.Module,
    *,
    benchmark_names: Sequence[str],
    vocab_size: int,
    num_examples: int,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    if not benchmark_names:
        raise ValueError("benchmark_names must be non-empty.")
    results: Dict[str, Dict[str, float]] = {}
    for benchmark_index, benchmark_name in enumerate(benchmark_names):
        results[benchmark_name] = evaluate_benchmark_task(
            model,
            benchmark_name=benchmark_name,
            vocab_size=vocab_size,
            num_examples=num_examples,
            device=device,
            seed=seed + benchmark_index,
        )
    return results


def train_benchmark_task(
    model: nn.Module,
    *,
    benchmark_name: str,
    vocab_size: int,
    device: torch.device,
    config: BenchmarkTrainingConfig,
    eval_benchmark_names: Optional[Sequence[str]] = None,
    log_prefix: str = "",
) -> Dict[str, Any]:
    if benchmark_name not in BENCHMARK_FACTORIES:
        raise ValueError(f"unknown benchmark: {benchmark_name}")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model_config = getattr(model, "config", None)
    nominal_growth_threshold = (
        float(model_config.growth_threshold)
        if model_config is not None and hasattr(model_config, "growth_threshold")
        else _nominal_growth_threshold(model)
    )
    last_train_loss = float("nan")
    last_eval_metrics: Optional[Dict[str, Dict[str, float]]] = None
    loss_history: list[Dict[str, float]] = []
    if config.streaming:
        enter_online_mode = getattr(model, "enter_online_mode", None)
        if callable(enter_online_mode):
            enter_online_mode()
        reset_online_state = getattr(model, "reset_online_state", None)
        if callable(reset_online_state):
            reset_online_state()
        set_online_state_gradient_tracking = getattr(model, "set_online_state_gradient_tracking", None)
        if callable(set_online_state_gradient_tracking):
            set_online_state_gradient_tracking(False)

    for step in range(1, config.steps + 1):
        if nominal_growth_threshold is not None:
            effective_threshold = _effective_growth_threshold(
                nominal_threshold=nominal_growth_threshold,
                step=step,
                total_steps=config.steps,
                warmup_steps=getattr(model_config, "growth_warmup_steps", 0),
                warmup_multiplier=getattr(model_config, "growth_warmup_multiplier", 10.0),
            )
            _set_growth_threshold(model, effective_threshold)
        model.train()
        batch = BENCHMARK_FACTORIES[benchmark_name](
            num_examples=config.batch_size,
            vocab_size=vocab_size,
            device=device,
            seed=config.seed + step,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_logits(model, batch.input_ids, attention_mask=batch.attention_mask)
        loss, _ = _causal_label_cross_entropy(logits, batch.labels)
        loss.backward()
        optimizer.step()
        if config.streaming:
            detach_online_state = getattr(model, "detach_online_state", None)
            if callable(detach_online_state):
                detach_online_state()
        last_train_loss = float(loss.item())

        if eval_benchmark_names is not None and (step == 1 or step % config.eval_every == 0 or step == config.steps):
            last_eval_metrics = evaluate_named_benchmark_tasks(
                model,
                benchmark_names=eval_benchmark_names,
                vocab_size=vocab_size,
                num_examples=config.eval_examples,
                device=device,
                seed=config.seed,
            )
            primary_metrics = last_eval_metrics[benchmark_name]
            loss_history.append(
                {
                    "step": float(step),
                    "train_loss": last_train_loss,
                    "eval_loss": float(primary_metrics["loss"]),
                    "eval_accuracy": float(primary_metrics["accuracy"]),
                }
            )

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            prefix = f"{log_prefix} " if log_prefix else ""
            if last_eval_metrics is None:
                print(f"{prefix}step={step:5d} train_loss={last_train_loss:.4f}")
            else:
                primary_metrics = last_eval_metrics[benchmark_name]
                print(
                    f"{prefix}step={step:5d} train_loss={last_train_loss:.4f} "
                    f"eval_{benchmark_name}_loss={primary_metrics['loss']:.4f} "
                    f"eval_{benchmark_name}_acc={primary_metrics['accuracy']:.4f}"
                )

    return {
        "final_train_loss": last_train_loss,
        "eval_metrics": last_eval_metrics,
        "token_budget": int(config.steps * config.batch_size),
        "loss_history": loss_history,
    }


def train_causal_language_model(
    model: nn.Module,
    *,
    train_tokens: Sequence[int] | Tensor,
    device: torch.device,
    config: TrainingRunConfig,
    val_tokens: Optional[Sequence[int] | Tensor] = None,
    log_prefix: str = "",
    latest_checkpoint_path: Optional[Path] = None,
    best_checkpoint_path: Optional[Path] = None,
    resume_from_checkpoint: bool = False,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    train_tokens = torch.as_tensor(train_tokens, dtype=torch.long)
    if train_tokens.numel() < config.seq_len + 1:
        raise ValueError("train_tokens must contain at least seq_len + 1 tokens.")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model_config = getattr(model, "config", None)
    nominal_growth_threshold = (
        float(model_config.growth_threshold)
        if model_config is not None and hasattr(model_config, "growth_threshold")
        else _nominal_growth_threshold(model)
    )
    batch_generator = torch.Generator(device="cpu").manual_seed(config.seed)

    best_val_loss = float("inf")
    best_step = 0
    start_step = 0
    resumed_from_checkpoint = False
    best_state_dict: Optional[Dict[str, Tensor]] = None
    best_val_metrics: Optional[Dict[str, float]] = None
    last_val_metrics: Optional[Dict[str, float]] = None
    last_train_loss = float("nan")
    loss_history: list[Dict[str, float]] = []

    if resume_from_checkpoint and latest_checkpoint_path is not None and latest_checkpoint_path.exists():
        payload = _load_training_checkpoint(
            latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_step = int(payload.get("step", 0))
        best_step = int(payload.get("best_step", 0))
        best_val_loss_payload = payload.get("best_val_loss")
        best_val_loss = float("inf") if best_val_loss_payload is None else float(best_val_loss_payload)
        last_train_loss = float(payload.get("last_train_loss", float("nan")))
        loaded_val_metrics = payload.get("last_val_metrics")
        last_val_metrics = None if loaded_val_metrics is None else dict(loaded_val_metrics)
        loaded_best_val_metrics = payload.get("best_val_metrics")
        best_val_metrics = None if loaded_best_val_metrics is None else dict(loaded_best_val_metrics)
        loaded_best_state_dict = payload.get("best_state_dict")
        if loaded_best_state_dict is not None:
            best_state_dict = dict(loaded_best_state_dict)
        loaded_loss_history = payload.get("loss_history", [])
        loss_history = [dict(entry) for entry in loaded_loss_history]
        batch_generator_state = payload.get("batch_generator_state")
        if batch_generator_state is not None:
            batch_generator.set_state(batch_generator_state)
        resumed_from_checkpoint = True

    for step in range(start_step + 1, config.steps + 1):
        if nominal_growth_threshold is not None:
            effective_threshold = _effective_growth_threshold(
                nominal_threshold=nominal_growth_threshold,
                step=step,
                total_steps=config.steps,
                warmup_steps=getattr(model_config, "growth_warmup_steps", 0),
                warmup_multiplier=getattr(model_config, "growth_warmup_multiplier", 10.0),
            )
            _set_growth_threshold(model, effective_threshold)
        model.train()
        input_ids, target_ids = sample_causal_lm_batch(
            train_tokens,
            config.seq_len,
            config.batch_size,
            device=device,
            generator=batch_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_logits(model, input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
        loss.backward()
        optimizer.step()
        last_train_loss = float(loss.item())

        if val_tokens is not None and (step == 1 or step % config.eval_every == 0 or step == config.steps):
            last_val_metrics = evaluate_causal_lm(
                model,
                val_tokens,
                config.seq_len,
                device,
                max_batches=config.eval_batches,
            )
            loss_history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(step * config.batch_size * config.seq_len),
                    "train_loss": float(last_train_loss),
                    "val_loss": float(last_val_metrics["loss"]),
                    "val_perplexity": float(last_val_metrics["perplexity"]),
                    "val_token_accuracy": float(last_val_metrics["token_accuracy"]),
                }
            )
            if last_val_metrics["loss"] < best_val_loss:
                best_val_loss = float(last_val_metrics["loss"])
                best_step = step
                best_val_metrics = copy.deepcopy(last_val_metrics)
                if best_checkpoint_path is None:
                    best_state_dict = _cpu_state_dict(model)
                best_payload = _training_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_step=best_step,
                    best_val_loss=best_val_loss,
                    last_train_loss=last_train_loss,
                    last_val_metrics=last_val_metrics,
                    best_val_metrics=best_val_metrics,
                    batch_generator_state=batch_generator.get_state(),
                    metadata=checkpoint_metadata,
                    loss_history=loss_history,
                    best_state_dict=None if best_checkpoint_path is not None else best_state_dict,
                )
                if best_checkpoint_path is not None:
                    _atomic_torch_save(best_checkpoint_path, best_payload)

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            prefix = f"{log_prefix} " if log_prefix else ""
            if last_val_metrics is None:
                message = f"{prefix}step={step:5d} train_loss={last_train_loss:.4f}"
            else:
                message = (
                    f"{prefix}step={step:5d} train_loss={last_train_loss:.4f} "
                    f"val_loss={last_val_metrics['loss']:.4f} "
                    f"val_ppl={last_val_metrics['perplexity']:.2f} "
                    f"val_acc={last_val_metrics['token_accuracy']:.4f}"
                )
            print(message)

        if latest_checkpoint_path is not None and (
            step == start_step + 1 or step % config.save_every == 0 or step == config.steps
        ):
            latest_payload = _training_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                step=step,
                best_step=best_step,
                best_val_loss=best_val_loss,
                last_train_loss=last_train_loss,
                last_val_metrics=last_val_metrics,
                best_val_metrics=best_val_metrics,
                batch_generator_state=batch_generator.get_state(),
                metadata=checkpoint_metadata,
                loss_history=loss_history,
                best_state_dict=None if best_checkpoint_path is not None else best_state_dict,
            )
            _atomic_torch_save(latest_checkpoint_path, latest_payload)

    if best_checkpoint_path is not None and best_checkpoint_path.exists():
        best_payload = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_payload["model_state_dict"])
        loaded_val_metrics = best_payload.get("last_val_metrics")
        final_val_metrics = None if loaded_val_metrics is None else dict(loaded_val_metrics)
        best_step = int(best_payload.get("step", best_step))
        best_val_loss_payload = best_payload.get("best_val_loss")
        if best_val_loss_payload is not None:
            best_val_loss = float(best_val_loss_payload)
    elif best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        final_val_metrics = copy.deepcopy(best_val_metrics)
    elif val_tokens is not None:
        final_val_metrics = evaluate_causal_lm(
            model,
            val_tokens,
            config.seq_len,
            device,
            max_batches=config.eval_batches,
        )
        best_val_loss = float(final_val_metrics["loss"])
        best_step = config.steps
    else:
        final_val_metrics = None

    return {
        "final_train_loss": last_train_loss,
        "best_step": int(best_step),
        "best_val_loss": None if final_val_metrics is None else float(best_val_loss),
        "val_metrics": final_val_metrics,
        "latest_checkpoint": None if latest_checkpoint_path is None else str(latest_checkpoint_path),
        "best_checkpoint": None if best_checkpoint_path is None else str(best_checkpoint_path),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "token_budget": int(config.steps * config.batch_size * config.seq_len),
        "loss_history": loss_history,
    }


def match_parameter_budget(
    *,
    target_parameter_count: int,
    candidate_values: Sequence[int],
    build_model: Callable[[int], nn.Module],
    parameter_name: str,
) -> CapacityMatch:
    if target_parameter_count <= 0:
        raise ValueError("target_parameter_count must be positive.")
    if not candidate_values:
        raise ValueError("candidate_values must be non-empty.")

    best_match: Optional[CapacityMatch] = None
    for parameter_value in candidate_values:
        model = build_model(parameter_value)
        parameter_count = count_trainable_parameters(model)
        relative_gap = abs(parameter_count - target_parameter_count) / target_parameter_count
        candidate = CapacityMatch(
            parameter_name=parameter_name,
            parameter_value=int(parameter_value),
            parameter_count=int(parameter_count),
            target_parameter_count=int(target_parameter_count),
            relative_gap=float(relative_gap),
        )
        if best_match is None or (
            candidate.relative_gap,
            abs(candidate.parameter_count - candidate.target_parameter_count),
            candidate.parameter_value,
        ) < (
            best_match.relative_gap,
            abs(best_match.parameter_count - best_match.target_parameter_count),
            best_match.parameter_value,
        ):
            best_match = candidate
        del model

    assert best_match is not None
    return best_match


def match_parameter_and_flop_budget(
    *,
    target_parameter_count: int,
    target_train_flops_per_step: float,
    candidate_values: Sequence[int],
    build_model: Callable[[int], nn.Module],
    estimate_train_flops: Callable[[nn.Module], float],
    parameter_name: str,
) -> BudgetMatch:
    if target_parameter_count <= 0:
        raise ValueError("target_parameter_count must be positive.")
    if target_train_flops_per_step <= 0.0:
        raise ValueError("target_train_flops_per_step must be positive.")
    if not candidate_values:
        raise ValueError("candidate_values must be non-empty.")

    best_match: Optional[BudgetMatch] = None
    for parameter_value in candidate_values:
        model = build_model(parameter_value)
        parameter_count = count_trainable_parameters(model)
        train_flops_per_step = float(estimate_train_flops(model))
        parameter_relative_gap = abs(parameter_count - target_parameter_count) / target_parameter_count
        train_flops_relative_gap = abs(train_flops_per_step - target_train_flops_per_step) / target_train_flops_per_step
        candidate = BudgetMatch(
            parameter_name=parameter_name,
            parameter_value=int(parameter_value),
            parameter_count=int(parameter_count),
            target_parameter_count=int(target_parameter_count),
            parameter_relative_gap=float(parameter_relative_gap),
            train_flops_per_step=float(train_flops_per_step),
            target_train_flops_per_step=float(target_train_flops_per_step),
            train_flops_relative_gap=float(train_flops_relative_gap),
        )
        if best_match is None or (
            max(candidate.parameter_relative_gap, candidate.train_flops_relative_gap),
            candidate.parameter_relative_gap,
            candidate.train_flops_relative_gap,
            abs(candidate.parameter_count - candidate.target_parameter_count),
            abs(candidate.train_flops_per_step - candidate.target_train_flops_per_step),
            candidate.parameter_value,
        ) < (
            max(best_match.parameter_relative_gap, best_match.train_flops_relative_gap),
            best_match.parameter_relative_gap,
            best_match.train_flops_relative_gap,
            abs(best_match.parameter_count - best_match.target_parameter_count),
            abs(best_match.train_flops_per_step - best_match.target_train_flops_per_step),
            best_match.parameter_value,
        ):
            best_match = candidate
        del model

    assert best_match is not None
    return best_match


__all__ = [
    "BenchmarkTrainingConfig",
    "BudgetMatch",
    "CapacityMatch",
    "TrainingRunConfig",
    "count_trainable_parameters",
    "estimate_plain_transformer_train_flops",
    "estimate_reciprocator_only_train_flops",
    "estimate_small_mamba_train_flops",
    "evaluate_benchmark_task",
    "evaluate_benchmark_suite_generic",
    "evaluate_causal_lm",
    "evaluate_named_benchmark_tasks",
    "match_parameter_and_flop_budget",
    "match_parameter_budget",
    "split_train_val_tokens",
    "train_benchmark_task",
    "train_causal_language_model",
]
