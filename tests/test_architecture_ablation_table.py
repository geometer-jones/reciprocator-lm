import importlib.util
import json
import sys
from pathlib import Path


def _load_script_module(script_name: str):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    module_name = script_name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_run_architecture_ablation_table_builds_requested_rows() -> None:
    module = _load_script_module("run_architecture_ablation_table.py")
    args = module._build_arg_parser().parse_args(
        [
            "--dim",
            "32",
            "--layers",
            "1",
            "--heads",
            "4",
            "--seq-len",
            "16",
            "--batch-size",
            "2",
            "--vocab-size",
            "64",
            "--state-rank",
            "2",
            "--max-state-rank",
            "2",
            "--init-mode-sizes",
            "4,4",
            "--max-mode-sizes",
            "4,4",
            "--num-cube-engines",
            "2",
            "--search-min-dim",
            "16",
            "--search-max-dim",
            "32",
            "--search-step",
            "8",
            "--transformer-heads",
            "4",
            "--mamba-state-size",
            "8",
        ]
    )

    rows = module.build_experiment_rows(args, vocab_size=64)
    by_name = {row.name: row for row in rows}

    assert set(by_name) == {
        "reciprocator_full",
        "reciprocator_no_spectral",
        "reciprocator_diagonal_coupling",
        "reciprocator_independent_couplings",
        "reciprocator_no_dynamic_rank",
        "reciprocator_no_input_gains",
        "reciprocator_sleep_on",
        "transformer_matched",
        "mamba_matched",
    }
    assert by_name["reciprocator_full"].config_payload["mode_coupling_layout"] == "full"
    assert by_name["reciprocator_diagonal_coupling"].config_payload["mode_coupling_layout"] == "diagonal"
    assert by_name["reciprocator_independent_couplings"].config_payload["mode_coupling_schedule"] == "independent"
    assert by_name["reciprocator_sleep_on"].sleep_enabled is True
    assert by_name["transformer_matched"].target_parameter_count == by_name["reciprocator_full"].parameter_count
    assert by_name["mamba_matched"].target_train_flops_per_step == by_name["reciprocator_full"].train_flops_per_step


def test_run_architecture_ablation_table_plan_only_writes_files(tmp_path, monkeypatch) -> None:
    module = _load_script_module("run_architecture_ablation_table.py")
    output_prefix = tmp_path / "ablation"
    argv = [
        "run_architecture_ablation_table.py",
        "--output-prefix",
        str(output_prefix),
        "--dim",
        "32",
        "--layers",
        "1",
        "--heads",
        "4",
        "--seq-len",
        "16",
        "--batch-size",
        "2",
        "--vocab-size",
        "64",
        "--state-rank",
        "2",
        "--max-state-rank",
        "2",
        "--init-mode-sizes",
        "4,4",
        "--max-mode-sizes",
        "4,4",
        "--num-cube-engines",
        "2",
        "--search-min-dim",
        "16",
        "--search-max-dim",
        "32",
        "--search-step",
        "8",
        "--transformer-heads",
        "4",
        "--mamba-state-size",
        "8",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    plan_path = Path(f"{output_prefix}_plan.json")
    table_path = Path(f"{output_prefix}_table.csv")
    assert plan_path.is_file()
    assert table_path.is_file()

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["mode"] == "plan_only"
    assert len(plan["rows"]) == 9
    assert "benchmark_long_range_retrieval" in table_path.read_text(encoding="utf-8")
