"""Collect online adaptation traces into a replay buffer."""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reciprocator_lm import (  # noqa: E402
    OnlineReplayBuffer,
    SentencePieceTokenizer,
    available_corpora,
    collect_online_episode,
    load_reciprocator_checkpoint,
    read_corpus_text,
    save_online_replay_buffer,
)
from reciprocator_lm.runtime import add_device_argument, resolve_torch_device  # noqa: E402


def _episode_sources(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.text_file is not None:
        return [(args.text_file.stem, args.text_file.read_text(encoding="utf-8"))]

    corpus_names = args.corpus or [corpus.name for corpus in available_corpora()]
    return [(name, read_corpus_text(name)) for name in corpus_names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect online wake-phase experience into a replay buffer.")
    add_device_argument(parser)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint created by train_reciprocator_only.")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt replay buffer path.")
    parser.add_argument("--text-file", type=Path, default=None, help="Optional UTF-8 text file for one episode.")
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=None,
        help="Optional SentencePiece model if the checkpoint does not embed a tokenizer.",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        choices=[corpus.name for corpus in available_corpora()],
        help="Bundled corpus name to collect from. Can be passed multiple times.",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()

    checkpoint = load_reciprocator_checkpoint(args.checkpoint)
    tokenizer = checkpoint.build_tokenizer()
    if tokenizer is None:
        if args.tokenizer_model is None:
            raise ValueError("checkpoint does not embed a tokenizer; pass --tokenizer-model.")
        tokenizer = SentencePieceTokenizer.from_model_file(args.tokenizer_model)
    if tokenizer.vocab_size != checkpoint.config.vocab_size:
        raise ValueError("tokenizer vocab size does not match checkpoint config.")

    device = resolve_torch_device(args.device)
    print(f"Device: {device}")
    model = checkpoint.build_model().to(device).eval()

    episodes = []
    for episode_id, text in _episode_sources(args):
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        episode = collect_online_episode(
            model,
            token_ids,
            seq_len=args.seq_len,
            episode_id=episode_id,
            stride=args.stride,
            max_chunks=args.max_chunks,
            device=device,
        )
        episodes.append(episode)
        print(f"episode={episode_id} chunks={len(episode.chunks)} tokens={len(token_ids)}")

    buffer = OnlineReplayBuffer(
        episodes=tuple(episodes),
        tokenizer_vocab_size=tokenizer.vocab_size,
        metadata={
            "checkpoint": str(args.checkpoint),
            "seq_len": int(args.seq_len),
            "stride": None if args.stride is None else int(args.stride),
        },
    )
    save_online_replay_buffer(args.output, buffer)
    print(f"saved replay buffer to {args.output}")


if __name__ == "__main__":
    main()
