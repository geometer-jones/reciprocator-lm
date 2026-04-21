from __future__ import annotations

import torch
from torch import nn

from reciprocator_lm import (
    EncodedScanExample,
    ScanExample,
    build_scan_symbol_table,
    encode_scan_examples,
    evaluate_scan,
)


class _NextTokenOracle(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), fill_value=-10.0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        lengths = attention_mask.sum(dim=1)
        for batch_index in range(batch_size):
            length = int(lengths[batch_index].item())
            if length <= 0:
                continue
            if length > 1:
                next_tokens = input_ids[batch_index, 1:length]
                positions = torch.arange(length - 1)
                logits[batch_index, positions, next_tokens] = 10.0
            logits[batch_index, length - 1, 3] = 10.0
        return {"logits": logits}


def test_encode_scan_examples_marks_only_action_side_as_supervised() -> None:
    examples = [
        ScanExample(command_tokens=("jump",), action_tokens=("I_JUMP",)),
        ScanExample(command_tokens=("walk", "twice"), action_tokens=("I_WALK", "I_WALK")),
    ]
    symbol_table = build_scan_symbol_table(examples, vocab_size=64)
    encoded = encode_scan_examples(examples, symbol_table=symbol_table, max_seq_len=16)

    assert len(encoded) == 2
    assert encoded[0].labels[:2] == (-100, -100)
    assert encoded[0].labels[2:] == (symbol_table["I_JUMP"], 3)
    assert encoded[1].labels[:3] == (-100, -100, -100)
    assert encoded[1].labels[3:] == (symbol_table["I_WALK"], symbol_table["I_WALK"], 3)


def test_evaluate_scan_reports_perfect_scores_for_oracle() -> None:
    examples = [
        EncodedScanExample(input_ids=(1, 4, 2, 5), labels=(-100, -100, 5, 3)),
        EncodedScanExample(input_ids=(1, 6, 7, 2, 8, 8), labels=(-100, -100, -100, 8, 8, 3)),
    ]
    model = _NextTokenOracle(vocab_size=16)

    metrics = evaluate_scan(
        model,
        examples,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert metrics["loss"] < 1e-3
    assert metrics["token_accuracy"] == 1.0
    assert metrics["exact_match"] == 1.0
