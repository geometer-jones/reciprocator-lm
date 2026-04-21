"""Consolidate wake-phase replay into model weights."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reciprocator_lm import (  # noqa: E402
    available_corpora,
    compute_sleep_loss,
    evaluate_benchmark_suite,
    load_online_replay_buffer,
    load_reciprocator_checkpoint,
    read_corpus_text,
    sample_causal_lm_batch,
    sample_replay_batch,
    save_reciprocator_checkpoint,
)
from reciprocator_lm.runtime import add_device_argument, resolve_torch_device  # noqa: E402


def _load_rehearsal_text(args: argparse.Namespace) -> str:
    if args.rehearsal_text_file is not None:
        return args.rehearsal_text_file.read_text(encoding="utf-8")

    corpus_names = args.rehearsal_corpus or [corpus.name for corpus in available_corpora()]
    return "\n".join(read_corpus_text(name) for name in corpus_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sleep-phase consolidation pass on replayed wake traces.")
    add_device_argument(parser)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Starting checkpoint.")
    parser.add_argument("--replay-buffer", type=Path, required=True, help="Replay buffer collected during wake mode.")
    parser.add_argument("--output-checkpoint", type=Path, required=True, help="Path for the consolidated checkpoint.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rehearsal-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--rehearsal-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-examples", type=int, default=64)
    parser.add_argument(
        "--rehearsal-corpus",
        action="append",
        choices=[corpus.name for corpus in available_corpora()],
        help="Bundled corpus used during sleep rehearsal. Can be passed multiple times.",
    )
    parser.add_argument(
        "--rehearsal-text-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file used instead of bundled corpora for rehearsal.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    device = resolve_torch_device(args.device)
    print(f"Device: {device}")

    checkpoint = load_reciprocator_checkpoint(args.checkpoint)
    replay_buffer = load_online_replay_buffer(args.replay_buffer)
    if replay_buffer.tokenizer_vocab_size != checkpoint.config.vocab_size:
        raise ValueError("replay buffer vocab size does not match checkpoint config.")

    tokenizer = checkpoint.build_tokenizer()
    model = checkpoint.build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    first_chunk = replay_buffer.episodes[0].chunks[0]
    seq_len = int(first_chunk.input_ids.numel())

    rehearsal_token_ids = None
    if args.rehearsal_weight > 0.0 and args.rehearsal_batch_size > 0:
        if tokenizer is None:
            raise ValueError("checkpoint does not embed a tokenizer, so rehearsal text cannot be tokenized.")
        rehearsal_text = _load_rehearsal_text(args)
        rehearsal_token_ids = tokenizer.encode(rehearsal_text, add_bos=True, add_eos=True)

    before_metrics = evaluate_benchmark_suite(
        model,
        num_examples=args.benchmark_examples,
        device=device,
        seed=args.seed,
    )
    print(f"benchmark_before={json.dumps(before_metrics, sort_keys=True)}")

    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        replay_batch = sample_replay_batch(
            replay_buffer,
            args.batch_size,
            device=device,
            generator=generator,
        )
        replay_logits, _ = model(replay_batch.input_ids)

        rehearsal_loss = None
        if rehearsal_token_ids is not None:
            rehearsal_inputs, rehearsal_targets = sample_causal_lm_batch(
                rehearsal_token_ids,
                seq_len,
                args.rehearsal_batch_size,
                device=device,
                generator=generator,
            )
            _, rehearsal_loss = model(rehearsal_inputs, rehearsal_targets)

        losses = compute_sleep_loss(
            replay_logits,
            replay_batch.target_ids,
            replay_batch.teacher_logits,
            base_loss=rehearsal_loss,
            distillation_weight=args.distill_weight,
            base_loss_weight=args.rehearsal_weight,
            temperature=args.temperature,
        )
        losses.total.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            print(
                "step="
                f"{step:5d} "
                f"total={losses.total.item():.4f} "
                f"wake_ce={losses.wake_ce.item():.4f} "
                f"distill={losses.distillation.item():.4f} "
                f"rehearsal={losses.base_ce.item():.4f}"
            )

    after_metrics = evaluate_benchmark_suite(
        model.eval(),
        num_examples=args.benchmark_examples,
        device=device,
        seed=args.seed,
    )
    print(f"benchmark_after={json.dumps(after_metrics, sort_keys=True)}")

    save_reciprocator_checkpoint(
        args.output_checkpoint,
        model=model,
        config=checkpoint.config,
        tokenizer=tokenizer,
        metadata={
            "script": "sleep_train.py",
            "steps": int(args.steps),
            "distill_weight": float(args.distill_weight),
            "rehearsal_weight": float(args.rehearsal_weight),
            "temperature": float(args.temperature),
            "benchmark_before": before_metrics,
            "benchmark_after": after_metrics,
        },
    )
    print(f"saved checkpoint to {args.output_checkpoint}")


if __name__ == "__main__":
    main()
