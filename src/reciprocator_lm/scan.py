from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from urllib.request import Request, urlopen

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .benchmarks import BOS_ID, EOS_ID, PAD_ID, SEP_ID, SPECIAL_TOKENS


SCAN_LENGTH_TRAIN_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_train_length.txt"
SCAN_LENGTH_TEST_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_test_length.txt"


@dataclass(frozen=True)
class ScanExample:
    command_tokens: tuple[str, ...]
    action_tokens: tuple[str, ...]


@dataclass(frozen=True)
class EncodedScanExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class ScanTransferConfig:
    steps: int = 100
    batch_size: int = 32
    lr: float = 3e-4
    log_every: int = 25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")


def _download_text(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "reciprocator-llm/0.1"})
    with urlopen(request) as response:
        text = response.read().decode("utf-8")
    destination.write_text(text, encoding="utf-8")
    return text


def _load_or_download_text(path: Path, url: str) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return _download_text(url, path)


def _parse_scan_lines(text: str) -> list[ScanExample]:
    examples: list[ScanExample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("IN: ") or " OUT: " not in line:
            raise ValueError(f"Malformed SCAN example: {line!r}")
        command_part, action_part = line[4:].split(" OUT: ", maxsplit=1)
        examples.append(
            ScanExample(
                command_tokens=tuple(command_part.split()),
                action_tokens=tuple(action_part.split()),
            )
        )
    if not examples:
        raise ValueError("SCAN text produced no examples.")
    return examples


def load_scan_length_split(cache_dir: Path) -> tuple[list[ScanExample], list[ScanExample]]:
    train_text = _load_or_download_text(cache_dir / "tasks_train_length.txt", SCAN_LENGTH_TRAIN_URL)
    test_text = _load_or_download_text(cache_dir / "tasks_test_length.txt", SCAN_LENGTH_TEST_URL)
    return _parse_scan_lines(train_text), _parse_scan_lines(test_text)


def build_scan_symbol_table(examples: Sequence[ScanExample], *, vocab_size: int) -> dict[str, int]:
    symbols = sorted(
        {
            token
            for example in examples
            for token in (*example.command_tokens, *example.action_tokens)
        }
    )
    first_symbol_id = SPECIAL_TOKENS
    required_vocab = first_symbol_id + len(symbols)
    if required_vocab > vocab_size:
        raise ValueError(f"SCAN requires vocab_size >= {required_vocab}, got {vocab_size}.")
    return {symbol: first_symbol_id + index for index, symbol in enumerate(symbols)}


def encode_scan_examples(
    examples: Sequence[ScanExample],
    *,
    symbol_table: dict[str, int],
    max_seq_len: Optional[int] = None,
) -> list[EncodedScanExample]:
    encoded: list[EncodedScanExample] = []
    for example in examples:
        command_ids = [symbol_table[token] for token in example.command_tokens]
        action_ids = [symbol_table[token] for token in example.action_tokens]
        full_sequence = [BOS_ID, *command_ids, SEP_ID, *action_ids, EOS_ID]
        input_ids = tuple(full_sequence[:-1])
        labels = tuple([-100] * (1 + len(command_ids)) + action_ids + [EOS_ID])
        if len(input_ids) != len(labels):
            raise ValueError("encoded SCAN example produced misaligned input_ids and labels")
        if max_seq_len is not None and len(input_ids) > max_seq_len:
            raise ValueError(
                f"encoded SCAN example length {len(input_ids)} exceeds max_seq_len {max_seq_len}"
            )
        encoded.append(EncodedScanExample(input_ids=input_ids, labels=labels))
    return encoded


def _collate_scan_batch(
    examples: Sequence[EncodedScanExample],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if not examples:
        raise ValueError("examples must be non-empty")
    batch_size = len(examples)
    max_len = max(len(example.input_ids) for example in examples)
    input_ids = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for index, example in enumerate(examples):
        example_len = len(example.input_ids)
        input_ids[index, :example_len] = torch.tensor(example.input_ids, dtype=torch.long)
        labels[index, :example_len] = torch.tensor(example.labels, dtype=torch.long)
        attention_mask[index, :example_len] = True
    return (
        input_ids.to(device=device),
        labels.to(device=device),
        attention_mask.to(device=device),
    )


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


def evaluate_scan(
    model: nn.Module,
    examples: Sequence[EncodedScanExample],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not examples:
        raise ValueError("examples must be non-empty.")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_examples = 0
    exact_matches = 0
    correct = 0
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                batch_examples = examples[start : start + batch_size]
                input_ids, labels, attention_mask = _collate_scan_batch(batch_examples, device=device)
                logits = _forward_logits(model, input_ids, attention_mask=attention_mask)
                valid_mask = labels != -100
                total_loss += float(
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    ).item()
                )
                predictions = logits.argmax(dim=-1)
                correct += int(predictions.eq(labels).masked_select(valid_mask).sum().item())
                total_tokens += int(valid_mask.sum().item())
                exact_matches += int(((predictions == labels) | ~valid_mask).all(dim=1).sum().item())
                total_examples += int(input_ids.shape[0])
    finally:
        if was_training:
            model.train()

    return {
        "loss": total_loss / total_tokens,
        "token_accuracy": correct / total_tokens,
        "exact_match": exact_matches / total_examples,
        "tokens": float(total_tokens),
        "examples": float(total_examples),
    }


def train_scan(
    model: nn.Module,
    train_examples: Sequence[EncodedScanExample],
    *,
    device: torch.device,
    config: ScanTransferConfig,
    log_prefix: str = "",
) -> Dict[str, Any]:
    if not train_examples:
        raise ValueError("train_examples must be non-empty.")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    final_train_loss = float("nan")

    for step in range(1, config.steps + 1):
        indices = torch.randint(
            low=0,
            high=len(train_examples),
            size=(config.batch_size,),
            generator=generator,
        ).tolist()
        batch_examples = [train_examples[index] for index in indices]
        input_ids, labels, attention_mask = _collate_scan_batch(batch_examples, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_logits(model, input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()
        final_train_loss = float(loss.item())

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            prefix = f"{log_prefix} " if log_prefix else ""
            print(f"{prefix}step={step:5d} scan_train_loss={final_train_loss:.4f}")

    return {
        "final_train_loss": final_train_loss,
        "token_budget": int(config.steps * config.batch_size),
    }


__all__ = [
    "EncodedScanExample",
    "ScanExample",
    "ScanTransferConfig",
    "SCAN_LENGTH_TEST_URL",
    "SCAN_LENGTH_TRAIN_URL",
    "build_scan_symbol_table",
    "encode_scan_examples",
    "evaluate_scan",
    "load_scan_length_split",
    "train_scan",
]
