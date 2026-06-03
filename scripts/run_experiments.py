#!/usr/bin/env python3
"""Run evaluation experiments 1–5 and emit CSVs, plots, and full_report.json."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from evaluation.experiment_runner import ExperimentRunner  # noqa: E402
from evaluation.metrics import MetricsCalculator  # noqa: E402
from evaluation.result_reporter import ResultReporter  # noqa: E402
from models.llama_model import LlamaModel  # noqa: E402
from models.mistral_model import MistralModel  # noqa: E402
from models.qwen_model import QwenModel  # noqa: E402
from models.resource_monitor import ResourceMonitor  # noqa: E402


import argparse

def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation experiments")
    parser.add_argument("--model", type=str, default="all", choices=["llama", "mistral", "qwen", "all"], help="Model to evaluate (default: all)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("run_experiments")

    stmt_path = ROOT / "data" / "processed" / "monthly_statements.json"
    if not stmt_path.is_file():
        log.error("Missing %s — run notebooks/01_preprocessing.ipynb first.", stmt_path)
        return 1

    with open(stmt_path, encoding="utf-8") as f:
        statements = json.load(f)
    if not statements:
        log.error("monthly_statements.json is empty.")
        return 1

    patterns_path = ROOT / "data" / "processed" / "patterns.json"
    if patterns_path.is_file():
        log.info("Found patterns.json (informational).")

    models = {}
    if args.model in ("llama", "all"):
        models["llama"] = LlamaModel()
    if args.model in ("mistral", "all"):
        models["mistral"] = MistralModel()
    if args.model in ("qwen", "all"):
        models["qwen"] = QwenModel()

    metrics = MetricsCalculator()
    monitor = ResourceMonitor()
    results_dir = ROOT / "evaluation" / "results"
    runner = ExperimentRunner(
        statements=statements,
        models=models,
        metrics=metrics,
        monitor=monitor,
        results_dir=results_dir,
    )

    log.info("Running experiments 1–5 (many LLM generations — can take a long time).")
    dfs = runner.run_all()

    reporter = ResultReporter(results_dir=results_dir, viz_dir=ROOT / "visualizations")

    reporter.plot_rouge_comparison(dfs["experiment_1"])
    reporter.plot_resource_efficiency(dfs["experiment_5"])
    reporter.plot_prompt_strategy_comparison(dfs["experiment_3"])

    for n, df in dfs.items():
        num = int(n.split("_")[1])
        reporter.print_experiment_summary(num, df)

    print("\n--- Experiment 4 comparison (markdown) ---\n")
    print(reporter.generate_comparison_table(dfs["experiment_4"]))

    first_month = sorted(statements.keys())[0]
    reporter.plot_spending_breakdown(statements[first_month], first_month)

    bundle = reporter.generate_full_report_data()
    print("\nFull report metadata:", json.dumps(bundle.get("report_json_path"), indent=2))

    print("\nFinal resource comparison (experiment 5):")
    print(dfs["experiment_5"].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
