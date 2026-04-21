"""Training corpora stored under the repo-root ``corpora/`` directory."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple


@dataclass(frozen=True)
class BundledCorpus:
    """Metadata for one bundled corpus."""

    name: str
    combined_filename: str
    readme_filename: str = "README.md"
    sources_filename: str = "sources.tsv"

    @property
    def path(self) -> Path:
        return _corpora_root() / self.name

    @property
    def combined_path(self) -> Path:
        return self.path / self.combined_filename

    @property
    def readme_path(self) -> Path:
        return self.path / self.readme_filename

    @property
    def sources_path(self) -> Path:
        return self.path / self.sources_filename


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repository root from reciprocator_lm corpora module.")


def _corpora_root() -> Path:
    return _find_repo_root() / "corpora"


_CORPORA: Dict[str, BundledCorpus] = {
    "greek_philosophy_classics": BundledCorpus(
        name="greek_philosophy_classics",
        combined_filename="greek_philosophy_classics_combined.txt",
    ),
    "plato_jowett": BundledCorpus(
        name="plato_jowett",
        combined_filename="plato_jowett_combined.txt",
    ),
}


def available_corpora() -> Tuple[BundledCorpus, ...]:
    """Return the bundled corpora available in this package."""

    return tuple(_CORPORA[name] for name in sorted(_CORPORA))


def get_corpus(name: str) -> BundledCorpus:
    """Return metadata for one bundled corpus."""

    try:
        return _CORPORA[name]
    except KeyError as exc:
        available = ", ".join(sorted(_CORPORA))
        raise KeyError(f"Unknown corpus '{name}'. Available corpora: {available}.") from exc


@contextmanager
def corpus_path(name: str) -> Iterator[Path]:
    """Yield a filesystem path for a repo-root corpus directory."""

    path = get_corpus(name).path
    if not path.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {path}")
    yield path


def read_corpus_text(name: str) -> str:
    """Read the combined training text for one bundled corpus."""

    return get_corpus(name).combined_path.read_text(encoding="utf-8")


def read_corpus_readme(name: str) -> str:
    """Read the README for one bundled corpus."""

    return get_corpus(name).readme_path.read_text(encoding="utf-8")


def read_corpus_sources(name: str) -> str:
    """Read the tab-separated provenance file for one bundled corpus."""

    return get_corpus(name).sources_path.read_text(encoding="utf-8")


__all__ = [
    "BundledCorpus",
    "available_corpora",
    "corpus_path",
    "get_corpus",
    "read_corpus_readme",
    "read_corpus_sources",
    "read_corpus_text",
]
