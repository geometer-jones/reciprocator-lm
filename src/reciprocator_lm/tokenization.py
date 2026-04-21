from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Sequence

import sentencepiece as spm


def _normalize_model_prefix(path: Path) -> Path:
    if path.suffix in {".model", ".vocab"}:
        return path.with_suffix("")
    return path


def _minimum_vocab_size(text: str) -> int:
    return len(set(text)) + 4


@dataclass(frozen=True)
class SentencePieceTokenizer:
    processor: spm.SentencePieceProcessor

    @classmethod
    def from_model_file(cls, model_file: Path | str) -> "SentencePieceTokenizer":
        return cls(spm.SentencePieceProcessor(model_file=str(model_file)))

    @classmethod
    def from_serialized_proto(cls, model_proto: bytes) -> "SentencePieceTokenizer":
        processor = spm.SentencePieceProcessor()
        processor.LoadFromSerializedProto(model_proto)
        return cls(processor)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    @property
    def bos_id(self) -> int:
        return int(self.processor.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self.processor.eos_id())

    @property
    def pad_id(self) -> int:
        return int(self.processor.pad_id())

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids = list(self.processor.encode(text, out_type=int))
        if add_bos and self.bos_id >= 0:
            token_ids.insert(0, self.bos_id)
        if add_eos and self.eos_id >= 0:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        ignored = {token_id for token_id in (self.bos_id, self.eos_id, self.pad_id) if token_id >= 0}
        filtered_ids = [int(token_id) for token_id in token_ids if int(token_id) not in ignored]
        return self.processor.decode(filtered_ids)


def _train_sentencepiece_model(
    *,
    text: str,
    model_prefix: Path,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
) -> Path:
    if not text.strip():
        raise ValueError("text must be non-empty to train a SentencePiece model.")

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    effective_vocab_size = max(vocab_size, _minimum_vocab_size(text))
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "training.txt"
        input_path.write_text(text, encoding="utf-8")
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(model_prefix),
            vocab_size=effective_vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            hard_vocab_limit=False,
            minloglevel=1,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            pad_piece="<pad>",
        )
    return model_prefix.with_suffix(".model")


def train_sentencepiece_tokenizer(
    *,
    text: str,
    vocab_size: int,
    model_prefix: Path | str | None = None,
    model_type: str = "unigram",
    character_coverage: float = 1.0,
) -> SentencePieceTokenizer:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")

    if model_prefix is None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_prefix = Path(temp_dir) / "sentencepiece"
            model_path = _train_sentencepiece_model(
                text=text,
                model_prefix=temp_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
            )
            model_proto = model_path.read_bytes()
        return SentencePieceTokenizer.from_serialized_proto(model_proto)

    normalized_prefix = _normalize_model_prefix(Path(model_prefix))
    model_path = _train_sentencepiece_model(
        text=text,
        model_prefix=normalized_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
    )
    return SentencePieceTokenizer.from_model_file(model_path)


__all__ = [
    "SentencePieceTokenizer",
    "train_sentencepiece_tokenizer",
]
