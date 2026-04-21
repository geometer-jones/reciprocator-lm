"""Launch matched-capacity rank ablations against train_reciprocator_only.py."""

import argparse
from dataclasses import dataclass
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reciprocator_lm.ablation import select_mode_size_pair  # noqa: E402


def _parse_ranks(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("ranks must be a comma-separated list of positive integers")
    try:
        ranks = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ranks must be integers") from exc
    if any(rank <= 0 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must be positive")
    return ranks


def _format_mode_sizes(mode_sizes: tuple[int, ...]) -> str:
    return ",".join(str(size) for size in mode_sizes)


def _strip_train_separator(train_args: list[str]) -> list[str]:
    if train_args and train_args[0] == "--":
        return train_args[1:]
    return train_args


def _assert_no_reserved_train_args(train_args: list[str]) -> None:
    reserved = {
        "--state-rank",
        "--init-mode-sizes",
        "--max-mode-sizes",
        "--init-state-capacity",
        "--state-capacity",
        "--latest-checkpoint",
        "--best-checkpoint",
        "--checkpoint-out",
        "--resume",
    }
    present = {
        token.split("=", 1)[0]
        for token in train_args
        if token.startswith("--")
    }
    overlaps = sorted(present & reserved)
    if overlaps:
        joined = ", ".join(overlaps)
        raise ValueError(f"train args must not override ablation-controlled options: {joined}")


@dataclass(frozen=True)
class RankRunSpec:
    rank: int
    init_mode_sizes: tuple[int, ...]
    max_mode_sizes: tuple[int, ...]
    latest_checkpoint: Path
    best_checkpoint: Path
    command: tuple[str, ...]


def build_run_specs(
    *,
    ranks: tuple[int, ...],
    init_state_capacity: int,
    state_capacity: int,
    run_prefix: str,
    output_dir: Path,
    train_args: list[str],
) -> list[RankRunSpec]:
    _assert_no_reserved_train_args(train_args)
    train_script = ROOT / "scripts" / "train_reciprocator_only.py"

    specs: list[RankRunSpec] = []
    for rank in ranks:
        init_mode_sizes, max_mode_sizes = select_mode_size_pair(
            state_rank=rank,
            init_capacity=init_state_capacity,
            max_capacity=state_capacity,
        )
        latest_checkpoint = output_dir / f"{run_prefix}_r{rank}_latest.pt"
        best_checkpoint = output_dir / f"{run_prefix}_r{rank}_best.pt"
        command = (
            sys.executable,
            str(train_script),
            "--state-rank",
            str(rank),
            "--init-mode-sizes",
            _format_mode_sizes(init_mode_sizes),
            "--max-mode-sizes",
            _format_mode_sizes(max_mode_sizes),
            "--latest-checkpoint",
            str(latest_checkpoint),
            "--best-checkpoint",
            str(best_checkpoint),
            *train_args,
        )
        specs.append(
            RankRunSpec(
                rank=rank,
                init_mode_sizes=init_mode_sizes,
                max_mode_sizes=max_mode_sizes,
                latest_checkpoint=latest_checkpoint,
                best_checkpoint=best_checkpoint,
                command=command,
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run rank-1/2/3 ablations with matched state capacity using the existing trainer."
    )
    parser.add_argument("--ranks", type=_parse_ranks, default=(1, 2, 3))
    parser.add_argument("--init-state-capacity", type=int, default=27)
    parser.add_argument("--state-capacity", type=int, default=64)
    parser.add_argument("--run-prefix", type=str, default="rank_ablation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--dry-run", action="store_true", help="Print the derived commands without running them.")
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to train_reciprocator_only.py. Prefix them with '--'.",
    )
    args = parser.parse_args()

    train_args = _strip_train_separator(list(args.train_args))
    specs = build_run_specs(
        ranks=args.ranks,
        init_state_capacity=args.init_state_capacity,
        state_capacity=args.state_capacity,
        run_prefix=args.run_prefix,
        output_dir=args.output_dir,
        train_args=train_args,
    )

    for spec in specs:
        print(
            f"rank={spec.rank} "
            f"init_mode_sizes={spec.init_mode_sizes} "
            f"max_mode_sizes={spec.max_mode_sizes}"
        )
        print(shlex.join(spec.command))

    if args.dry_run:
        return

    for spec in specs:
        subprocess.run(spec.command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
