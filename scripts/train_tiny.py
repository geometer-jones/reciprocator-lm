import argparse
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reciprocator_lm import ModelConfig, ModifiedTransformerLM, SentencePieceTokenizer, train_sentencepiece_tokenizer
from reciprocator_lm.runtime import add_device_argument, resolve_torch_device


def build_dataset(token_ids: list[int], seq_len: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(token_ids, dtype=torch.long, device=device)
    if data.numel() < seq_len + 1:
        repeats = ((seq_len + 1) // max(1, data.numel())) + 1
        data = data.repeat(repeats)
    max_start = max(1, data.numel() - seq_len - 1)
    starts = torch.randint(0, max_start, (batch_size,), device=device)
    x = torch.stack([data[start : start + seq_len] for start in starts])
    y = torch.stack([data[start + 1 : start + seq_len + 1] for start in starts])
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny Reciprocator-style LM on SentencePiece token IDs.")
    add_device_argument(parser)
    parser.add_argument("--text-file", type=Path, default=None, help="Optional UTF-8 text file for training data.")
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=None,
        help="Optional existing SentencePiece .model file. If omitted, a tokenizer is trained for this run.",
    )
    parser.add_argument(
        "--tokenizer-prefix",
        type=Path,
        default=None,
        help="Optional output prefix used when training a new SentencePiece model.",
    )
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    sample_text = (
        "Reciprocator language models combine causal attention with a recurrent memory mixer. "
        "This script trains a tiny SentencePiece-tokenized prototype for smoke testing.\n"
    )
    text = args.text_file.read_text(encoding="utf-8") if args.text_file else sample_text * 128

    if args.tokenizer_model is not None:
        tokenizer = SentencePieceTokenizer.from_model_file(args.tokenizer_model)
    elif args.tokenizer_prefix is not None:
        tokenizer = train_sentencepiece_tokenizer(
            text=text,
            vocab_size=args.vocab_size,
            model_prefix=args.tokenizer_prefix,
        )
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer = train_sentencepiece_tokenizer(
                text=text,
                vocab_size=args.vocab_size,
                model_prefix=Path(temp_dir) / "sentencepiece",
            )
            run_training(args, text, tokenizer)
        return

    run_training(args, text, tokenizer)


def run_training(args: argparse.Namespace, text: str, tokenizer: SentencePieceTokenizer) -> None:
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

    device = resolve_torch_device(args.device)
    print(f"Device: {device}")
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        dim=128,
        n_layers=3,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(4, 4, 4),
        num_cube_engines=4,
        normalization="per_mode",
        dropout=0.0,
        accumulator_modulates_gains=True,
        phase_aware_coupling=True,
        coupling_temperature=1.0,
    )
    model = ModifiedTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = build_dataset(token_ids, args.seq_len, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    prompt_ids = tokenizer.encode("Reciprocator", add_bos=True)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    generated = model.generate(prompt, max_new_tokens=80, temperature=0.8)[0].tolist()
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
