from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .benchmarks import build_default_benchmark_suite, sequence_accuracy
from .config import ModelConfig
from .model import ReciprocatorOnlyLM
from .tokenization import SentencePieceTokenizer


@dataclass(frozen=True)
class OnlineReplayChunk:
    input_ids: Tensor
    target_ids: Tensor
    teacher_logits: Tensor
    chunk_index: int
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1:
            raise ValueError("input_ids must be rank-1 [seq].")
        if self.target_ids.shape != self.input_ids.shape:
            raise ValueError("target_ids must match input_ids shape.")
        if self.teacher_logits.ndim != 2:
            raise ValueError("teacher_logits must have shape [seq, vocab].")
        if self.teacher_logits.shape[0] != self.input_ids.shape[0]:
            raise ValueError("teacher_logits sequence dimension must match input_ids.")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative.")

    def to_payload(self) -> Dict[str, object]:
        return {
            "input_ids": self.input_ids.detach().cpu(),
            "target_ids": self.target_ids.detach().cpu(),
            "teacher_logits": self.teacher_logits.detach().cpu(),
            "chunk_index": int(self.chunk_index),
            "diagnostics": copy.deepcopy(self.diagnostics),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "OnlineReplayChunk":
        return cls(
            input_ids=torch.as_tensor(payload["input_ids"], dtype=torch.long),
            target_ids=torch.as_tensor(payload["target_ids"], dtype=torch.long),
            teacher_logits=torch.as_tensor(payload["teacher_logits"]),
            chunk_index=int(payload["chunk_index"]),
            diagnostics=dict(payload.get("diagnostics", {})),
        )


@dataclass(frozen=True)
class OnlineReplayEpisode:
    episode_id: str
    chunks: Tuple[OnlineReplayChunk, ...]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty.")
        if not self.chunks:
            raise ValueError("episodes must contain at least one chunk.")

    def to_payload(self) -> Dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "chunks": [chunk.to_payload() for chunk in self.chunks],
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "OnlineReplayEpisode":
        return cls(
            episode_id=str(payload["episode_id"]),
            chunks=tuple(OnlineReplayChunk.from_payload(chunk) for chunk in payload["chunks"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class OnlineReplayBuffer:
    episodes: Tuple[OnlineReplayEpisode, ...]
    tokenizer_vocab_size: int
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("replay buffer must contain at least one episode.")
        if self.tokenizer_vocab_size <= 0:
            raise ValueError("tokenizer_vocab_size must be positive.")

    def to_payload(self) -> Dict[str, object]:
        return {
            "episodes": [episode.to_payload() for episode in self.episodes],
            "tokenizer_vocab_size": int(self.tokenizer_vocab_size),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "OnlineReplayBuffer":
        return cls(
            episodes=tuple(OnlineReplayEpisode.from_payload(episode) for episode in payload["episodes"]),
            tokenizer_vocab_size=int(payload["tokenizer_vocab_size"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ReplayBatch:
    input_ids: Tensor
    target_ids: Tensor
    teacher_logits: Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq].")
        if self.target_ids.shape != self.input_ids.shape:
            raise ValueError("target_ids must match input_ids.")
        if self.teacher_logits.ndim != 3:
            raise ValueError("teacher_logits must have shape [batch, seq, vocab].")
        if self.teacher_logits.shape[:2] != self.input_ids.shape:
            raise ValueError("teacher_logits batch and sequence dims must match input_ids.")


@dataclass(frozen=True)
class SleepLossBreakdown:
    total: Tensor
    wake_ce: Tensor
    distillation: Tensor
    base_ce: Tensor


@dataclass(frozen=True)
class ReciprocatorCheckpoint:
    config: ModelConfig
    model_state_dict: Dict[str, Tensor]
    tokenizer_model_proto: Optional[bytes] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def build_model(self) -> ReciprocatorOnlyLM:
        model = ReciprocatorOnlyLM(self.config)
        model.load_state_dict(self.model_state_dict)
        return model

    def build_tokenizer(self) -> Optional[SentencePieceTokenizer]:
        if self.tokenizer_model_proto is None:
            return None
        return SentencePieceTokenizer.from_serialized_proto(self.tokenizer_model_proto)


def iter_causal_training_chunks(
    token_ids: Sequence[int],
    seq_len: int,
    *,
    stride: Optional[int] = None,
) -> Iterable[tuple[Tensor, Tensor]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if stride is None:
        stride = seq_len
    if stride <= 0:
        raise ValueError("stride must be positive.")

    window = seq_len + 1
    if len(token_ids) < window:
        return

    max_start = len(token_ids) - window
    for start in range(0, max_start + 1, stride):
        chunk = token_ids[start : start + window]
        if len(chunk) != window:
            continue
        yield (
            torch.tensor(chunk[:-1], dtype=torch.long),
            torch.tensor(chunk[1:], dtype=torch.long),
        )


def sample_causal_lm_batch(
    token_ids: Sequence[int],
    seq_len: int,
    batch_size: int,
    *,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")

    data = torch.as_tensor(token_ids, dtype=torch.long)
    if data.numel() < seq_len + 1:
        repeats = ((seq_len + 1) // max(1, data.numel())) + 1
        data = data.repeat(repeats)

    max_start = max(1, data.numel() - seq_len - 1)
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    input_ids = torch.stack([data[start : start + seq_len] for start in starts.tolist()])
    target_ids = torch.stack([data[start + 1 : start + seq_len + 1] for start in starts.tolist()])
    if device is not None:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
    return input_ids, target_ids


def collect_online_episode(
    model: ReciprocatorOnlyLM,
    token_ids: Sequence[int],
    *,
    seq_len: int,
    episode_id: str,
    stride: Optional[int] = None,
    max_chunks: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> OnlineReplayEpisode:
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive when provided.")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    model.enter_online_mode()
    model.reset_online_state()

    chunks = []
    with torch.no_grad():
        for chunk_index, (input_ids, target_ids) in enumerate(
            iter_causal_training_chunks(token_ids, seq_len, stride=stride)
        ):
            if max_chunks is not None and chunk_index >= max_chunks:
                break
            logits, _ = model(input_ids.unsqueeze(0).to(device))
            diagnostics = copy.deepcopy(model.online_diagnostics())
            chunks.append(
                OnlineReplayChunk(
                    input_ids=input_ids,
                    target_ids=target_ids,
                    teacher_logits=logits.squeeze(0).detach().cpu(),
                    chunk_index=chunk_index,
                    diagnostics=diagnostics,
                )
            )

    if not chunks:
        raise ValueError("token_ids did not produce any full causal training chunks.")

    return OnlineReplayEpisode(
        episode_id=episode_id,
        chunks=tuple(chunks),
        metadata={
            "seq_len": int(seq_len),
            "stride": int(seq_len if stride is None else stride),
            "num_tokens": int(len(token_ids)),
            "num_chunks": int(len(chunks)),
        },
    )


def flatten_replay_chunks(buffer: OnlineReplayBuffer) -> Tuple[OnlineReplayChunk, ...]:
    return tuple(chunk for episode in buffer.episodes for chunk in episode.chunks)


def collate_replay_chunks(
    chunks: Sequence[OnlineReplayChunk],
    *,
    device: Optional[torch.device] = None,
) -> ReplayBatch:
    if not chunks:
        raise ValueError("chunks must be non-empty.")

    seq_lens = {chunk.input_ids.numel() for chunk in chunks}
    vocab_sizes = {chunk.teacher_logits.shape[-1] for chunk in chunks}
    if len(seq_lens) != 1:
        raise ValueError("all replay chunks must share a sequence length.")
    if len(vocab_sizes) != 1:
        raise ValueError("all replay chunks must share a vocabulary size.")

    input_ids = torch.stack([chunk.input_ids for chunk in chunks])
    target_ids = torch.stack([chunk.target_ids for chunk in chunks])
    teacher_logits = torch.stack([chunk.teacher_logits for chunk in chunks])
    if device is not None:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        teacher_logits = teacher_logits.to(device)
    return ReplayBatch(input_ids=input_ids, target_ids=target_ids, teacher_logits=teacher_logits)


def sample_replay_batch(
    buffer: OnlineReplayBuffer,
    batch_size: int,
    *,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> ReplayBatch:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    all_chunks = flatten_replay_chunks(buffer)
    if not all_chunks:
        raise ValueError("replay buffer contains no chunks.")

    indices = torch.randint(0, len(all_chunks), (batch_size,), generator=generator)
    return collate_replay_chunks([all_chunks[index] for index in indices.tolist()], device=device)


def distillation_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must match.")

    vocab_size = student_logits.shape[-1]
    student_flat = student_logits.reshape(-1, vocab_size)
    teacher_flat = teacher_logits.reshape(-1, vocab_size)
    return F.kl_div(
        F.log_softmax(student_flat / temperature, dim=-1),
        F.softmax(teacher_flat / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)


def compute_sleep_loss(
    student_logits: Tensor,
    target_ids: Tensor,
    teacher_logits: Tensor,
    *,
    base_loss: Optional[Tensor] = None,
    distillation_weight: float = 1.0,
    base_loss_weight: float = 1.0,
    temperature: float = 1.0,
) -> SleepLossBreakdown:
    if distillation_weight < 0.0:
        raise ValueError("distillation_weight must be non-negative.")
    if base_loss_weight < 0.0:
        raise ValueError("base_loss_weight must be non-negative.")

    wake_ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        target_ids.reshape(-1),
    )
    distillation = distillation_kl_loss(
        student_logits,
        teacher_logits,
        temperature=temperature,
    )
    base_ce = (
        torch.zeros((), dtype=wake_ce.dtype, device=wake_ce.device)
        if base_loss is None
        else base_loss
    )
    total = wake_ce + distillation_weight * distillation + base_loss_weight * base_ce
    return SleepLossBreakdown(
        total=total,
        wake_ce=wake_ce,
        distillation=distillation,
        base_ce=base_ce,
    )


def evaluate_benchmark_suite(
    model: ReciprocatorOnlyLM,
    *,
    num_examples: int,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, float]:
    if num_examples <= 0:
        raise ValueError("num_examples must be positive.")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results: Dict[str, float] = {}
    with torch.no_grad():
        for benchmark_index, benchmark in enumerate(build_default_benchmark_suite(model.config.vocab_size)):
            batch = benchmark.make_batch(num_examples, seed + benchmark_index, device)
            logits, _ = model(batch.input_ids)
            results[benchmark.name] = sequence_accuracy(logits, batch.labels, batch.prediction_positions)
    return results


def save_online_replay_buffer(path: Path | str, buffer: OnlineReplayBuffer) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(buffer.to_payload(), output_path)


def load_online_replay_buffer(path: Path | str) -> OnlineReplayBuffer:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return OnlineReplayBuffer.from_payload(payload)


def save_reciprocator_checkpoint(
    path: Path | str,
    *,
    model: ReciprocatorOnlyLM,
    config: ModelConfig,
    tokenizer: Optional[SentencePieceTokenizer] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": {
                name: value.detach().cpu() if isinstance(value, Tensor) else copy.deepcopy(value)
                for name, value in model.state_dict().items()
            },
            "tokenizer_model_proto": None
            if tokenizer is None
            else tokenizer.processor.serialized_model_proto(),
            "metadata": {} if metadata is None else copy.deepcopy(metadata),
        },
        output_path,
    )


def load_reciprocator_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> ReciprocatorCheckpoint:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return ReciprocatorCheckpoint(
        config=ModelConfig(**payload["config"]),
        model_state_dict=dict(payload["model_state_dict"]),
        tokenizer_model_proto=payload.get("tokenizer_model_proto"),
        metadata=dict(payload.get("metadata", {})),
    )


__all__ = [
    "OnlineReplayBuffer",
    "OnlineReplayChunk",
    "OnlineReplayEpisode",
    "ReciprocatorCheckpoint",
    "ReplayBatch",
    "SleepLossBreakdown",
    "collect_online_episode",
    "collate_replay_chunks",
    "compute_sleep_loss",
    "distillation_kl_loss",
    "evaluate_benchmark_suite",
    "flatten_replay_chunks",
    "iter_causal_training_chunks",
    "load_online_replay_buffer",
    "load_reciprocator_checkpoint",
    "sample_causal_lm_batch",
    "sample_replay_batch",
    "save_online_replay_buffer",
    "save_reciprocator_checkpoint",
]
