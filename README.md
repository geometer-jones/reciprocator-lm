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

## Current Training Run

To resume the current `full_corpus_dynamic_rank_streaming_r1to6_alllearn` run from its latest checkpoint, run this from the repository root inside your active Python environment:

```bash
python -u scripts/train_reciprocator_only.py \
  --device cpu \
  --steps 5000 \
  --batch-size 8 \
  --resume runs/full_corpus_dynamic_rank_streaming_r1to6_alllearn/latest.pt \
  --training-mode streaming \
  --stream-reset-policy wrap \
  --log-every 100 \
  --save-every 100 \
  --eval-every 100 \
  --eval-batches 8 \
  --benchmark-examples 128 \
  --benchmark-every 200 \
  --latest-checkpoint runs/full_corpus_dynamic_rank_streaming_r1to6_alllearn/latest.pt \
  --best-checkpoint runs/full_corpus_dynamic_rank_streaming_r1to6_alllearn/best.pt \
  --skip-online-demo
```

## `train_reciprocator_only.py` Flags

Defaults below are for fresh runs unless noted otherwise. For resume-aware flags, the script uses the saved checkpoint value when the CLI flag is omitted. Paths are shown repo-relative for readability even though the parser resolves them from the repository root.

### Runtime and Optimization

| Flag | Default | Meaning |
| --- | --- | --- |
| `--device {cpu,cuda,mps,auto}` | `auto` | Execution device. `auto` prefers CUDA, then MPS, then CPU. |
| `--steps` | `5000` | Total optimizer steps. |
| `--batch-size` | `8` | Batch size for random mode, or chunks-per-step in streaming mode. |
| `--seq-len` | `64` | Sequence length. |
| `--lr` | `3e-4` | Base learning rate. |
| `--lr-schedule {constant,cosine}` | fresh run: `constant`; resume: checkpoint value | Learning-rate schedule. |
| `--warmup-fraction` | fresh run: `0.0`; resume: checkpoint value | Warmup fraction used with cosine LR. |
| `--min-lr-ratio` | fresh run: `0.0`; resume: checkpoint value | Final LR divided by base LR for cosine decay. |
| `--grad-clip` | fresh run: `0.0`; resume: checkpoint value | Gradient clipping threshold. |
| `--lr-step-offset` | fresh run: `0`; resume: checkpoint value | LR schedule origin shift for resumed taper phases. |
| `--log-every` | `100` | Print training metrics every N steps. |
| `--save-every` | `100` | Save latest checkpoint every N steps. |
| `--eval-every` | `100` | Run validation every N steps. |
| `--eval-batches` | `8` | Validation batches per evaluation pass. |
| `--seed` | `0` | Random seed. |

### Model Shape

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dim` | `128` | Model width. |
| `--layers` | `4` | Number of layers. |
| `--heads` | `4` | Number of attention-style heads used by the model config. |
| `--mlp-ratio` | `4.0` | Feed-forward expansion ratio. |
| `--dropout` | `0.0` | Dropout probability. |
| `--vocab-size` | `512` | SentencePiece vocabulary size for fresh tokenizer training. |
| `--num-cube-engines` | `4` | Number of Reciprocator engines per layer. |

### Tensor State and Growth

| Flag | Default | Meaning |
| --- | --- | --- |
| `--state-rank` | `3` | Active tensor rank. |
| `--max-state-rank` | `--state-rank` | Maximum supported rank for dynamic growth. |
| `--dynamic-rank` / `--no-dynamic-rank` | `true` | Enable novelty-driven rank growth. |
| `--init-mode-sizes` | derived | Explicit comma-separated initial mode sizes. |
| `--max-mode-sizes` | derived | Explicit comma-separated maximum mode sizes. |
| `--init-state-capacity` | matches `--state-capacity` if omitted | Initial active state capacity used to derive init mode sizes. |
| `--state-capacity` | `64` if omitted | Maximum state capacity used to derive max mode sizes. |
| `--growth-threshold` | `0.02` | Novelty threshold for growth events. |
| `--growth-interval` | `1` | Check growth every N steps. |

### Normalization and Learnable Mixer Controls

| Flag | Default | Meaning |
| --- | --- | --- |
| `--normalization {frobenius,per_mode}` | `per_mode` | State normalization family. |
| `--learned-per-mode-scaling` / `--no-learned-per-mode-scaling` | `true` | Learn per-mode normalization exponents. |
| `--learnable-prediction-eta` / `--no-learnable-prediction-eta` | `true` | Learn anticipation gain `eta`. |
| `--learnable-coupling-temperature` / `--no-learnable-coupling-temperature` | `true` | Learn phase-aware coupling temperature. |
| `--learned-normalization-blend` / `--no-learned-normalization-blend` | `true` | Learn the blend between normalization families. |
| `--all-learnable-mixer-params` | `false` | Force all optional learnable mixer controls on. |
| `--parallel-mixer` | `false` | Use the parallel Reciprocator mixer. Not supported with streaming/persistent state. |

### Streaming and Persistent State

| Flag | Default | Meaning |
| --- | --- | --- |
| `--training-mode {random,streaming}` | fresh run resolves to `streaming`; resume uses checkpoint value | Training data mode. In streaming mode, persistent state is enabled inside the model. |
| `--stream-reset-policy {wrap,never}` | fresh run resolves to `wrap`; resume uses checkpoint value | Whether streaming state resets when the corpus stream wraps. |
| `--tbptt-horizon` | fresh run: `0`; resume: checkpoint value | Truncated BPTT horizon in streaming mode. |

### Tokenization, Validation, and Benchmarks

| Flag | Default | Meaning |
| --- | --- | --- |
| `--val-fraction` | fresh run: `0.05`; resume: checkpoint value | Validation split fraction. |
| `--benchmark-examples` | `128` | Benchmark examples per synthetic task. Set `0` to disable probes. |
| `--benchmark-every` | `200` | Benchmark probe cadence. If `0`, falls back to `--eval-every`. |
| `--tokenizer-model` | unset | Existing SentencePiece model to load for fresh runs. Ignored on resume. |
| `--tokenizer-prefix` | `scripts/reciprocator_only_tokenizer` | Output prefix for freshly trained SentencePiece models. |

### Readout and Coupling Variants

| Flag | Default | Meaning |
| --- | --- | --- |
| `--phase-aware-readout` | `true` | Use phase-aware complex readout features. |
| `--magnitude-readout` | `false` | Disable phase-aware readout and use magnitude-only readout. |
| `--phase-aware-coupling` | `true` | Preserve complex-score phase in routing matrices. |
| `--real-coupling-fallback` | `false` | Use the legacy real-valued routing collapse. |
| `--coupling-temperature` | `1.0` | Softmax temperature for phase-aware routing magnitudes. |

### Checkpoints and Output

| Flag | Default | Meaning |
| --- | --- | --- |
| `--resume` | unset | Resume from a training checkpoint. |
| `--latest-checkpoint` | `runs/reciprocator_only_latest.pt` | Periodic latest checkpoint path. |
| `--best-checkpoint` | `runs/reciprocator_only_best.pt` | Best checkpoint path, selected by validation loss when available. |
| `--checkpoint-out` | unset | Optional exported model checkpoint with embedded tokenizer. |
| `--skip-online-demo` | `true` | Skip the post-training online adaptation demo. |
| `--run-online-demo` | `false` | Re-enable the post-training online adaptation demo. |
