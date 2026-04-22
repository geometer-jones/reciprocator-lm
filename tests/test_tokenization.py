from pathlib import Path

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
