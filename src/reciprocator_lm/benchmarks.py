from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch


Tensor = torch.Tensor
BatchFactory = Callable[[int, int, Optional[torch.device]], "SyntheticSequenceBatch"]


PAD_ID = 0
BOS_ID = 1
SEP_ID = 2
EOS_ID = 3
SPECIAL_TOKENS = 4


@dataclass(frozen=True)
class SyntheticSequenceBatch:
    task_name: str
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    prediction_positions: Tensor


@dataclass(frozen=True)
class BenchmarkDefinition:
    name: str
    description: str
    make_batch: BatchFactory


def _cpu_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _move(tensor: Tensor, device: Optional[torch.device]) -> Tensor:
    return tensor if device is None else tensor.to(device)


def _labels_for_targets(input_ids: Tensor, target_positions: Tensor) -> Tensor:
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[:, target_positions] = input_ids[:, target_positions]
    return labels


def _batch_from_targets(
    *,
    task_name: str,
    input_ids: Tensor,
    target_positions: Tensor,
    device: Optional[torch.device],
) -> SyntheticSequenceBatch:
    if target_positions.numel() == 0:
        raise ValueError("target_positions must contain at least one position.")
    if int(target_positions.min().item()) <= 0:
        raise ValueError("target_positions must be > 0 so they can be predicted causally.")

    input_ids = _move(input_ids, device)
    labels = _move(_labels_for_targets(input_ids.cpu(), target_positions.cpu()), device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
    prediction_positions = _move(target_positions - 1, device)
    return SyntheticSequenceBatch(
        task_name=task_name,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prediction_positions=prediction_positions,
    )


def sequence_accuracy(logits: Tensor, labels: Tensor, prediction_positions: Tensor) -> float:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq, vocab].")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, seq].")
    shifted_predictions = logits[:, :-1].argmax(dim=-1)
    shifted_labels = labels[:, 1:]
    target_predictions = shifted_predictions.index_select(1, prediction_positions)
    target_labels = shifted_labels.index_select(1, prediction_positions)
    valid_mask = target_labels != -100
    if not bool(valid_mask.any().item()):
        return 0.0
    return float(target_predictions.eq(target_labels).masked_select(valid_mask).float().mean().item())


def make_long_range_retrieval_batch(
    num_examples: int,
    vocab_size: int,
    device: Optional[torch.device] = None,
    *,
    content_length: int = 8,
    reverse_k: int = 4,
    seed: int = 0,
) -> SyntheticSequenceBatch:
    if vocab_size <= SPECIAL_TOKENS + 4:
        raise ValueError("vocab_size is too small for long-range retrieval.")
    if reverse_k <= 0 or reverse_k > content_length:
        raise ValueError("reverse_k must be positive and <= content_length.")

    generator = _cpu_generator(seed)
    content = torch.randint(
        low=SPECIAL_TOKENS,
        high=vocab_size,
        size=(num_examples, content_length),
        generator=generator,
        dtype=torch.long,
    )
    reversed_tail = torch.flip(content[:, -reverse_k:], dims=[1])
    bos = torch.full((num_examples, 1), BOS_ID, dtype=torch.long)
    sep = torch.full((num_examples, 1), SEP_ID, dtype=torch.long)
    eos = torch.full((num_examples, 1), EOS_ID, dtype=torch.long)
    input_ids = torch.cat([bos, content, sep, reversed_tail, eos], dim=1)
    target_positions = torch.arange(content_length + 2, content_length + 2 + reverse_k, dtype=torch.long)
    return _batch_from_targets(
        task_name="long_range_retrieval",
        input_ids=input_ids,
        target_positions=target_positions,
        device=device,
    )


def make_hierarchical_conditioning_batch(
    num_examples: int,
    vocab_size: int,
    device: Optional[torch.device] = None,
    *,
    content_length: int = 6,
    seed: int = 0,
) -> SyntheticSequenceBatch:
    if content_length < 2 or content_length % 2 != 0:
        raise ValueError("content_length must be an even integer >= 2.")
    if vocab_size <= SPECIAL_TOKENS + 8:
        raise ValueError("vocab_size is too small for hierarchical conditioning.")

    generator = _cpu_generator(seed)
    content = torch.randint(
        low=SPECIAL_TOKENS + 2,
        high=vocab_size,
        size=(num_examples, content_length),
        generator=generator,
        dtype=torch.long,
    )
    mode = torch.randint(
        low=SPECIAL_TOKENS,
        high=SPECIAL_TOKENS + 2,
        size=(num_examples, 1),
        generator=generator,
        dtype=torch.long,
    )
    half = content_length // 2
    target = torch.where(mode.eq(SPECIAL_TOKENS), content[:, :half], content[:, half:])
    bos = torch.full((num_examples, 1), BOS_ID, dtype=torch.long)
    sep = torch.full((num_examples, 1), SEP_ID, dtype=torch.long)
    eos = torch.full((num_examples, 1), EOS_ID, dtype=torch.long)
    input_ids = torch.cat([bos, mode, content, sep, target, eos], dim=1)
    target_start = 2 + content_length + 1
    target_positions = torch.arange(target_start, target_start + half, dtype=torch.long)
    return _batch_from_targets(
        task_name="hierarchical_conditioning",
        input_ids=input_ids,
        target_positions=target_positions,
        device=device,
    )


def make_compositional_binding_batch(
    num_examples: int,
    vocab_size: int,
    device: Optional[torch.device] = None,
    *,
    num_pairs: int = 3,
    seed: int = 0,
) -> SyntheticSequenceBatch:
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")
    min_vocab = SPECIAL_TOKENS + (2 * num_pairs) + 8
    if vocab_size < min_vocab:
        raise ValueError(f"vocab_size must be >= {min_vocab} for compositional binding.")

    generator = _cpu_generator(seed)
    role_pool = torch.arange(SPECIAL_TOKENS, SPECIAL_TOKENS + (2 * num_pairs) + 2, dtype=torch.long)
    filler_pool = torch.arange(role_pool[-1].item() + 1, vocab_size, dtype=torch.long)

    role_indices = torch.stack(
        [role_pool[torch.randperm(role_pool.numel(), generator=generator)[:num_pairs]] for _ in range(num_examples)],
        dim=0,
    )
    filler_indices = torch.stack(
        [filler_pool[torch.randperm(filler_pool.numel(), generator=generator)[:num_pairs]] for _ in range(num_examples)],
        dim=0,
    )
    query_choice = torch.randint(0, num_pairs, (num_examples,), generator=generator)
    query_role = role_indices[torch.arange(num_examples), query_choice].unsqueeze(1)
    target_value = filler_indices[torch.arange(num_examples), query_choice].unsqueeze(1)

    bos = torch.full((num_examples, 1), BOS_ID, dtype=torch.long)
    sep = torch.full((num_examples, 1), SEP_ID, dtype=torch.long)
    eos = torch.full((num_examples, 1), EOS_ID, dtype=torch.long)
    support = torch.stack([role_indices, filler_indices], dim=2).reshape(num_examples, 2 * num_pairs)
    input_ids = torch.cat([bos, support, sep, query_role, target_value, eos], dim=1)
    target_positions = torch.tensor([2 * num_pairs + 3], dtype=torch.long)
    return _batch_from_targets(
        task_name="compositional_binding",
        input_ids=input_ids,
        target_positions=target_positions,
        device=device,
    )


def make_role_rebinding_batch(
    num_examples: int,
    vocab_size: int,
    device: Optional[torch.device] = None,
    *,
    num_pairs: int = 3,
    seed: int = 0,
) -> SyntheticSequenceBatch:
    if num_pairs < 2:
        raise ValueError("num_pairs must be >= 2 for role rebinding.")
    min_vocab = SPECIAL_TOKENS + (3 * num_pairs) + 12
    if vocab_size < min_vocab:
        raise ValueError(f"vocab_size must be >= {min_vocab} for role rebinding.")

    generator = _cpu_generator(seed)
    role_pool = torch.arange(SPECIAL_TOKENS, SPECIAL_TOKENS + (2 * num_pairs) + 4, dtype=torch.long)
    filler_pool = torch.arange(role_pool[-1].item() + 1, vocab_size, dtype=torch.long)

    roles = torch.stack(
        [role_pool[torch.randperm(role_pool.numel(), generator=generator)[:num_pairs]] for _ in range(num_examples)],
        dim=0,
    )
    fillers = torch.stack(
        [filler_pool[torch.randperm(filler_pool.numel(), generator=generator)[:num_pairs]] for _ in range(num_examples)],
        dim=0,
    )
    rebound_slot = torch.randint(0, num_pairs, (num_examples,), generator=generator)
    control_slot = (rebound_slot + 1) % num_pairs
    new_fillers = torch.stack(
        [filler_pool[torch.randperm(filler_pool.numel(), generator=generator)[num_pairs : num_pairs + 1]] for _ in range(num_examples)],
        dim=0,
    ).view(num_examples)

    rebound_role = roles[torch.arange(num_examples), rebound_slot].unsqueeze(1)
    rebound_answer = new_fillers.unsqueeze(1)
    control_role = roles[torch.arange(num_examples), control_slot].unsqueeze(1)
    control_answer = fillers[torch.arange(num_examples), control_slot].unsqueeze(1)

    bos = torch.full((num_examples, 1), BOS_ID, dtype=torch.long)
    sep = torch.full((num_examples, 1), SEP_ID, dtype=torch.long)
    eos = torch.full((num_examples, 1), EOS_ID, dtype=torch.long)
    support = torch.stack([roles, fillers], dim=2).reshape(num_examples, 2 * num_pairs)
    overwrite = torch.cat([rebound_role, rebound_answer], dim=1)
    queries = torch.cat([rebound_role, rebound_answer, control_role, control_answer], dim=1)
    input_ids = torch.cat([bos, support, sep, overwrite, sep, queries, eos], dim=1)
    target_start = 2 * num_pairs + 6
    target_positions = torch.arange(target_start, target_start + 4, step=2, dtype=torch.long)
    return _batch_from_targets(
        task_name="role_rebinding",
        input_ids=input_ids,
        target_positions=target_positions,
        device=device,
    )


def make_induction_batch(
    num_examples: int,
    vocab_size: int,
    device: Optional[torch.device] = None,
    *,
    prefix_length: int = 3,
    bridge_length: int = 3,
    seed: int = 0,
) -> SyntheticSequenceBatch:
    if vocab_size <= SPECIAL_TOKENS + 8:
        raise ValueError("vocab_size is too small for induction.")

    generator = _cpu_generator(seed)
    prefix = torch.randint(
        low=SPECIAL_TOKENS,
        high=vocab_size,
        size=(num_examples, prefix_length),
        generator=generator,
        dtype=torch.long,
    )
    bridge = torch.randint(
        low=SPECIAL_TOKENS,
        high=vocab_size,
        size=(num_examples, bridge_length),
        generator=generator,
        dtype=torch.long,
    )
    anchor = torch.randint(
        low=SPECIAL_TOKENS,
        high=vocab_size,
        size=(num_examples, 1),
        generator=generator,
        dtype=torch.long,
    )
    response = torch.randint(
        low=SPECIAL_TOKENS,
        high=vocab_size,
        size=(num_examples, 1),
        generator=generator,
        dtype=torch.long,
    )
    bos = torch.full((num_examples, 1), BOS_ID, dtype=torch.long)
    eos = torch.full((num_examples, 1), EOS_ID, dtype=torch.long)
    input_ids = torch.cat([bos, prefix, anchor, response, bridge, anchor, response, eos], dim=1)
    target_positions = torch.tensor([prefix_length + bridge_length + 4], dtype=torch.long)
    return _batch_from_targets(
        task_name="induction",
        input_ids=input_ids,
        target_positions=target_positions,
        device=device,
    )


def make_controlled_novelty_batch(
    num_examples: int,
    vocab_size: int,
    device: Optional[torch.device] = None,
    *,
    num_pairs: int = 6,
    num_queries: int = 3,
    seed: int = 0,
) -> SyntheticSequenceBatch:
    if num_pairs <= 0 or num_queries <= 0 or num_queries > num_pairs:
        raise ValueError("num_queries must be in [1, num_pairs].")
    min_vocab = SPECIAL_TOKENS + (3 * num_pairs) + 8
    if vocab_size < min_vocab:
        raise ValueError(f"vocab_size must be >= {min_vocab} for controlled novelty.")

    generator = _cpu_generator(seed)
    anchor_pool = torch.arange(SPECIAL_TOKENS, SPECIAL_TOKENS + (2 * num_pairs) + 4, dtype=torch.long)
    value_pool = torch.arange(anchor_pool[-1].item() + 1, vocab_size, dtype=torch.long)

    anchors = torch.stack(
        [anchor_pool[torch.randperm(anchor_pool.numel(), generator=generator)[:num_pairs]] for _ in range(num_examples)],
        dim=0,
    )
    values = torch.stack(
        [value_pool[torch.randperm(value_pool.numel(), generator=generator)[:num_pairs]] for _ in range(num_examples)],
        dim=0,
    )
    query_slots = torch.stack(
        [torch.randperm(num_pairs, generator=generator)[:num_queries] for _ in range(num_examples)],
        dim=0,
    )
    queries = torch.gather(anchors, 1, query_slots)
    answers = torch.gather(values, 1, query_slots)

    bos = torch.full((num_examples, 1), BOS_ID, dtype=torch.long)
    sep = torch.full((num_examples, 1), SEP_ID, dtype=torch.long)
    eos = torch.full((num_examples, 1), EOS_ID, dtype=torch.long)
    support = torch.stack([anchors, values], dim=2).reshape(num_examples, 2 * num_pairs)
    qa = torch.stack([queries, answers], dim=2).reshape(num_examples, 2 * num_queries)
    input_ids = torch.cat([bos, support, sep, qa, eos], dim=1)
    target_start = 2 * num_pairs + 3
    target_positions = torch.arange(target_start, target_start + (2 * num_queries), step=2, dtype=torch.long)
    return _batch_from_targets(
        task_name="controlled_novelty",
        input_ids=input_ids,
        target_positions=target_positions,
        device=device,
    )


def build_default_benchmark_suite(vocab_size: int) -> Tuple[BenchmarkDefinition, ...]:
    def wrap(factory: Callable[..., SyntheticSequenceBatch]) -> BatchFactory:
        def make_batch(num_examples: int, seed: int, device: Optional[torch.device]) -> SyntheticSequenceBatch:
            return factory(num_examples=num_examples, vocab_size=vocab_size, device=device, seed=seed)

        return make_batch

    return (
        BenchmarkDefinition(
            name="long_range_retrieval",
            description="Reverse a distant suffix after a separator.",
            make_batch=wrap(make_long_range_retrieval_batch),
        ),
        BenchmarkDefinition(
            name="hierarchical_conditioning",
            description="Use a document-level control token to choose a later continuation.",
            make_batch=wrap(make_hierarchical_conditioning_batch),
        ),
        BenchmarkDefinition(
            name="compositional_binding",
            description="Bind roles to fillers and answer a later role query.",
            make_batch=wrap(make_compositional_binding_batch),
        ),
        BenchmarkDefinition(
            name="role_rebinding",
            description="Overwrite one role/filler binding and later recover both rebound and control fillers.",
            make_batch=wrap(make_role_rebinding_batch),
        ),
        BenchmarkDefinition(
            name="induction",
            description="Infer a repeated A -> B transition and reuse it later.",
            make_batch=wrap(make_induction_batch),
        ),
        BenchmarkDefinition(
            name="controlled_novelty",
            description="Store many fresh within-sequence bindings and answer later queries.",
            make_batch=wrap(make_controlled_novelty_batch),
        ),
    )


BENCHMARK_FACTORIES: Dict[str, Callable[..., SyntheticSequenceBatch]] = {
    "long_range_retrieval": make_long_range_retrieval_batch,
    "hierarchical_conditioning": make_hierarchical_conditioning_batch,
    "compositional_binding": make_compositional_binding_batch,
    "role_rebinding": make_role_rebinding_batch,
    "induction": make_induction_batch,
    "controlled_novelty": make_controlled_novelty_batch,
}


__all__ = [
    "BENCHMARK_FACTORIES",
    "BenchmarkDefinition",
    "SyntheticSequenceBatch",
    "build_default_benchmark_suite",
    "make_compositional_binding_batch",
    "make_controlled_novelty_batch",
    "make_hierarchical_conditioning_batch",
    "make_induction_batch",
    "make_long_range_retrieval_batch",
    "make_role_rebinding_batch",
    "sequence_accuracy",
]
