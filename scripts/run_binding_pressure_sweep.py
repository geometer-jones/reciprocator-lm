"""Run a fixed full-support binding-pressure sweep for Reciprocator ranks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reciprocator_lm import (  # noqa: E402
    BenchmarkTrainingConfig,
    ModelConfig,
    ReciprocatorOnlyLM,
    ScanTransferConfig,
    build_scan_symbol_table,
    count_trainable_parameters,
    encode_scan_examples,
    evaluate_named_benchmark_tasks,
    evaluate_scan,
    load_scan_length_split,
    save_reciprocator_checkpoint,
    train_benchmark_task,
    train_scan,
)
from reciprocator_lm.runtime import add_device_argument, resolve_torch_device  # noqa: E402


DEFAULT_REPORT_PATH = ROOT / "runs" / "binding_pressure_sweep_report.json"
DEFAULT_ARTIFACT_DIR = ROOT / "runs" / "binding_pressure_sweep"
DEFAULT_SCAN_CACHE_DIR = ROOT / "runs" / "scan_length_cache"
PRIMARY_TASK = "controlled_novelty"
SECONDARY_TASK = "role_rebinding"
OPTIONAL_TASK = "compositional_binding"


@dataclass(frozen=True)
class SweepSpec:
    name: str
    state_rank: int
    mode_sizes: tuple[int, ...]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full-support binding-pressure rank sweep.")
    add_device_argument(parser, default="auto")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-cube-engines", type=int, default=2)
    parser.add_argument("--streaming", action="store_true", help="Train the binding task in online streaming mode.")
    parser.add_argument(
        "--persist-state",
        action="store_true",
        help="Construct the model with persistent state enabled for online streaming.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--scan-cache-dir", type=Path, default=DEFAULT_SCAN_CACHE_DIR)
    parser.add_argument("--scan-steps", type=int, default=100)
    parser.add_argument("--scan-batch-size", type=int, default=32)
    parser.add_argument("--scan-lr", type=float, default=3e-4)
    parser.add_argument("--scan-log-every", type=int, default=25)
    parser.add_argument("--scan-eval-batch-size", type=int, default=128)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional comma-separated subset of configs to run, e.g. 'r1,r3'.",
    )
    return parser


def _build_sweep_specs() -> list[SweepSpec]:
    return [
        SweepSpec(name="r1", state_rank=1, mode_sizes=(64,)),
        SweepSpec(name="r2", state_rank=2, mode_sizes=(4, 16)),
        SweepSpec(name="r3", state_rank=3, mode_sizes=(2, 4, 8)),
    ]


def _make_model_config(args: argparse.Namespace, spec: SweepSpec) -> ModelConfig:
    return ModelConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        state_rank=spec.state_rank,
        max_state_rank=spec.state_rank,
        dynamic_rank=False,
        init_mode_sizes=spec.mode_sizes,
        max_mode_sizes=spec.mode_sizes,
        num_cube_engines=args.num_cube_engines,
        normalization="per_mode",
        growth_threshold=1e9,
        growth_interval=1,
        prune_floor=0.0,
        prune_horizon=args.steps + 1,
        persist_state=args.persist_state,
        parallel_mixer=False,
        input_dependent_gains=True,
        accumulator_modulates_gains=True,
        phase_aware_readout=True,
        phase_aware_coupling=True,
        coupling_temperature=1.0,
    )


def _checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "binding_final.pt"


def _scan_transfer(
    *,
    model: ReciprocatorOnlyLM,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    scan_device = torch.device("cpu")
    scan_train_raw, scan_test_raw = load_scan_length_split(args.scan_cache_dir)
    scan_symbol_table = build_scan_symbol_table([*scan_train_raw, *scan_test_raw], vocab_size=model.config.vocab_size)
    scan_train = encode_scan_examples(scan_train_raw, symbol_table=scan_symbol_table, max_seq_len=model.config.max_seq_len)
    scan_test = encode_scan_examples(scan_test_raw, symbol_table=scan_symbol_table, max_seq_len=model.config.max_seq_len)
    scan_model = ReciprocatorOnlyLM(model.config).to(scan_device)
    scan_model.load_state_dict(model.state_dict())
    scan_train_summary = train_scan(
        scan_model,
        scan_train,
        device=scan_device,
        config=ScanTransferConfig(
            steps=args.scan_steps,
            batch_size=args.scan_batch_size,
            lr=args.scan_lr,
            log_every=args.scan_log_every,
            seed=args.seed,
        ),
        log_prefix=f"[scan/{model.config.state_rank}]",
    )
    scan_eval = evaluate_scan(
        scan_model,
        scan_test,
        device=scan_device,
        batch_size=args.scan_eval_batch_size,
    )
    return {
        "train": scan_train_summary,
        "eval": scan_eval,
    }


def _evaluate_tasks(
    *,
    model: ReciprocatorOnlyLM,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    return evaluate_named_benchmark_tasks(
        model,
        benchmark_names=(PRIMARY_TASK, SECONDARY_TASK, OPTIONAL_TASK),
        vocab_size=args.vocab_size,
        num_examples=args.eval_examples,
        device=device,
        seed=args.seed,
    )


def _success_call(results: list[dict[str, Any]]) -> dict[str, Any]:
    if {result["name"] for result in results} != {"r1", "r2", "r3"}:
        return {
            "verdict": "partial",
            "reason": "Only a subset of ranks was run in this report; aggregate across r1/r2/r3 before applying the success criteria.",
        }
    by_name = {result["name"]: result for result in results}
    r1_primary = float(by_name["r1"]["benchmarks"][PRIMARY_TASK]["accuracy"])
    r2_primary = float(by_name["r2"]["benchmarks"][PRIMARY_TASK]["accuracy"])
    r3_primary = float(by_name["r3"]["benchmarks"][PRIMARY_TASK]["accuracy"])
    r1_secondary = float(by_name["r1"]["benchmarks"][SECONDARY_TASK]["accuracy"])
    r2_secondary = float(by_name["r2"]["benchmarks"][SECONDARY_TASK]["accuracy"])
    r3_secondary = float(by_name["r3"]["benchmarks"][SECONDARY_TASK]["accuracy"])

    best_primary_name = max(results, key=lambda item: item["benchmarks"][PRIMARY_TASK]["accuracy"])["name"]
    primary_margin_vs_r1 = max(r2_primary - r1_primary, r3_primary - r1_primary)
    secondary_order = sorted(
        ((result["name"], float(result["benchmarks"][SECONDARY_TASK]["accuracy"])) for result in results),
        key=lambda item: item[1],
        reverse=True,
    )
    primary_order = sorted(
        ((result["name"], float(result["benchmarks"][PRIMARY_TASK]["accuracy"])) for result in results),
        key=lambda item: item[1],
        reverse=True,
    )
    same_order = [name for name, _ in primary_order] == [name for name, _ in secondary_order]
    r2_r3_tie_primary = abs(r2_primary - r3_primary) < 0.02
    verdict = "weak"
    reason = "No rank beat R1 by the requested 10 absolute points on controlled_novelty."
    if primary_margin_vs_r1 >= 0.10 and same_order:
        verdict = "rank_matters"
        reason = "A higher-rank model cleared the 10-point controlled_novelty margin and the ordering repeated on role_rebinding."
    elif r2_r3_tie_primary and best_primary_name in {"r2", "r3"}:
        verdict = "third_mode_not_paying_rent"
        reason = "R2 and R3 remained effectively tied on controlled_novelty, so the third mode did not pay rent."
    elif max(abs(r2_primary - r1_primary), abs(r3_primary - r1_primary)) < 0.02:
        verdict = "rank_blind"
        reason = "All three ranks stayed effectively tied on the primary task."

    return {
        "verdict": verdict,
        "reason": reason,
        "primary_order": primary_order,
        "secondary_order": secondary_order,
        "same_order_primary_secondary": same_order,
        "primary_margin_best_vs_r1": primary_margin_vs_r1,
        "r2_r3_primary_gap": abs(r2_primary - r3_primary),
        "r2_secondary": r2_secondary,
        "r3_secondary": r3_secondary,
        "r1_secondary": r1_secondary,
    }


def main() -> None:
    args = _build_arg_parser().parse_args()
    device = resolve_torch_device(args.device)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Binding-pressure sweep: task={PRIMARY_TASK} steps={args.steps} batch_size={args.batch_size} "
        f"eval_every={args.eval_every} seed={args.seed}"
    )

    specs = _build_sweep_specs()
    if args.only.strip():
        allowed = {name.strip() for name in args.only.split(",") if name.strip()}
        specs = [spec for spec in specs if spec.name in allowed]
        if not specs:
            raise ValueError("--only did not match any known config names.")
    results: list[dict[str, Any]] = []
    for spec in specs:
        torch.manual_seed(args.seed)
        run_dir = args.artifact_dir / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)
        binding_ckpt = _checkpoint_path(run_dir)
        model_config = _make_model_config(args, spec)
        model = ReciprocatorOnlyLM(model_config).to(device)

        print(f"\n=== Training {spec.name} on {PRIMARY_TASK} ===")
        training_summary = train_benchmark_task(
            model,
            benchmark_name=PRIMARY_TASK,
            vocab_size=args.vocab_size,
            device=device,
            config=BenchmarkTrainingConfig(
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_every=args.eval_every,
                eval_examples=args.eval_examples,
                log_every=args.log_every,
                seed=args.seed,
                streaming=args.streaming,
            ),
            eval_benchmark_names=(PRIMARY_TASK, SECONDARY_TASK, OPTIONAL_TASK),
            log_prefix=f"[{spec.name}]",
        )
        save_reciprocator_checkpoint(
            binding_ckpt,
            model=model,
            config=model_config,
            tokenizer=None,
            metadata={
                "script": "run_binding_pressure_sweep.py",
                "task": PRIMARY_TASK,
                "training_summary": training_summary,
                "seed": args.seed,
            },
        )

        benchmark_metrics = _evaluate_tasks(model=model, args=args, device=device)
        scan_metrics = _scan_transfer(model=model, args=args, device=device)
        results.append(
            {
                "name": spec.name,
                "rank": spec.state_rank,
                "mode_sizes": list(spec.mode_sizes),
                "parameter_count": count_trainable_parameters(model),
                "config": asdict(model_config),
                "training": training_summary,
                "benchmarks": benchmark_metrics,
                "scan": scan_metrics,
                "binding_checkpoint": str(binding_ckpt),
                "growth_events": 0,
                "active_sizes_fixed": list(spec.mode_sizes),
            }
        )

    verdict = _success_call(results)
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "raw_args": sys.argv[1:],
        "device": str(device),
        "experiment": {
            "training_task": PRIMARY_TASK,
            "primary_eval": PRIMARY_TASK,
            "secondary_eval": SECONDARY_TASK,
            "optional_eval": OPTIONAL_TASK,
            "tertiary_eval": "scan_length_split_transfer",
        },
        "matched_small_model": {
            "dim": args.dim,
            "n_layers": args.layers,
            "num_cube_engines": args.num_cube_engines,
            "vocab_size": args.vocab_size,
            "lr": args.lr,
            "dropout": args.dropout,
            "input_dependent_gains": True,
            "phase_aware_coupling": True,
            "phase_aware_readout": True,
            "dynamic_rank": False,
            "full_support_fixed_rank": True,
            "prune_floor": 0.0,
            "streaming": bool(args.streaming),
            "persist_state": bool(args.persist_state),
        },
        "results": results,
        "success_criteria_call": verdict,
    }

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    print(f"\nReport written to {args.report_file}")
    for result in results:
        primary_acc = result["benchmarks"][PRIMARY_TASK]["accuracy"]
        secondary_acc = result["benchmarks"][SECONDARY_TASK]["accuracy"]
        scan_em = result["scan"]["eval"]["exact_match"]
        print(
            f"{result['name']:2s} primary={primary_acc:.4f} "
            f"secondary={secondary_acc:.4f} scan_em={scan_em:.4f}"
        )
    print(f"Verdict: {verdict['verdict']} | {verdict['reason']}")


if __name__ == "__main__":
    main()
