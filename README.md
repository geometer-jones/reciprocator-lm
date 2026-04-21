# reciprocator-lm

`reciprocator-lm` is a Python research prototype for the Reciprocator, a modified transformer-style language model built around complex-valued tensor memory and "memory engine" dynamics.

The repository includes:

- `src/reciprocator_lm/`: core model code, baselines, benchmark utilities, corpora helpers, tokenization, SCAN tasks, and sleep/replay training logic
- `scripts/`: experiment and training entry-point scripts
- `tests/`: `pytest` coverage for the main package modules
- `corpora/`: bundled text corpora plus source metadata
- `reciprocator.md`: mathematical architecture notes
- `memory-engines.md`: higher-level geometric intuition for the approach

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

Experiment outputs and checkpoints are written under `runs/`, which is intentionally ignored by Git.
