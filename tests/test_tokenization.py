from pathlib import Path
import importlib.util

import torch

from reciprocator_lm import SentencePieceTokenizer, train_sentencepiece_tokenizer


def test_train_sentencepiece_tokenizer_round_trips_text(tmp_path: Path) -> None:
    training_text = ("Plato studies reason and dialogue.\n" * 32).strip()
    model_prefix = tmp_path / "toy_spm"

    tokenizer = train_sentencepiece_tokenizer(
        text=training_text,
        vocab_size=32,
        model_prefix=model_prefix,
    )

    token_ids = tokenizer.encode("Plato studies reason.", add_bos=True, add_eos=True)

    assert model_prefix.with_suffix(".model").is_file()
    assert tokenizer.vocab_size >= 4
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.pad_id == 3
    assert tokenizer.decode(token_ids) == "Plato studies reason."


def test_sentencepiece_tokenizer_can_load_saved_model(tmp_path: Path) -> None:
    training_text = ("Socrates asks careful questions.\n" * 32).strip()
    model_prefix = tmp_path / "saved_spm"
    train_sentencepiece_tokenizer(text=training_text, vocab_size=24, model_prefix=model_prefix)

    tokenizer = SentencePieceTokenizer.from_model_file(model_prefix.with_suffix(".model"))
    token_ids = tokenizer.encode("Socrates asks careful questions.")

    assert tokenizer.decode(token_ids) == "Socrates asks careful questions."


def test_build_dataset_returns_shifted_sentencepiece_targets() -> None:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train_tiny.py"
    spec = importlib.util.spec_from_file_location("train_tiny", module_path)
    assert spec is not None
    assert spec.loader is not None
    train_tiny = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_tiny)

    token_ids = list(range(12))
    inputs, targets = train_tiny.build_dataset(token_ids, seq_len=5, batch_size=3, device=torch.device("cpu"))

    assert inputs.shape == (3, 5)
    assert targets.shape == (3, 5)
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
