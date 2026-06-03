#!/usr/bin/env python3
"""Run the full FinSight-AI pipeline end-to-end (setup → preprocess → embeddings → model check → experiments)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_step(description: str, argv: list[str]) -> int:
    print("\n" + "=" * 60)
    print(description)
    print("=" * 60)
    cmd = [sys.executable, *argv]
    print(" ", " ".join(cmd), "\n", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def print_artifact_summary() -> None:
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — artifacts")
    print("=" * 60)
    checks: list[tuple[str, Path]] = [
        ("Raw PaySim CSV", ROOT / "data" / "raw" / "paysim1.csv"),
        ("Cleaned transactions (parquet)", ROOT / "data" / "processed" / "transactions_clean.parquet"),
        ("Monthly statements (JSON)", ROOT / "data" / "processed" / "monthly_statements.json"),
        ("Clustered sample (parquet)", ROOT / "data" / "processed" / "transactions_clustered.parquet"),
        ("Cluster metadata (JSON)", ROOT / "data" / "processed" / "cluster_labels.json"),
        ("Elbow plot", ROOT / "visualizations" / "elbow_curve.png"),
    ]
    for i in range(1, 6):
        checks.append((f"Experiment {i} CSV", ROOT / "evaluation" / "results" / f"experiment_{i}.csv"))
    checks.append(("Full report bundle", ROOT / "evaluation" / "results" / "full_report.json"))

    for label, path in checks:
        status = "present" if path.is_file() else "missing"
        print(f"  [{status}] {label}: {path}")

    report_path = ROOT / "evaluation" / "results" / "full_report.json"
    if report_path.is_file():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            tables = data.get("tables") or {}
            if tables:
                print(f"\n  full_report.json lists {len(tables)} experiment table(s); open file for column stats.")
        except (json.JSONDecodeError, OSError):
            print("\n  (Could not read full_report.json.)")

    print("\n" + "=" * 60)
    print("Launch UI (Next.js + FastAPI — see SETUP_AND_PREPROCESSING_GUIDE.md § Section 5)")
    print("=" * 60)
    print("  Terminal 1 (API):  python scripts/run_api.py")
    print("  Terminal 2 (web):  cd frontend && npm install && npm run dev")
    print("  Open:              http://localhost:3000")
    print(
        "\n  (Legacy course text may mention Streamlit; this repo implements the UI in frontend/ + backend/.)\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full FinSight-AI pipeline.")
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing and embeddings (steps 2–3); require existing parquet/statements.",
    )
    args = parser.parse_args()

    print("FinSight-AI — full pipeline")
    print("Root:", ROOT)

    code = run_step("Step 1: Project setup checks", [str(ROOT / "scripts" / "setup_project.py")])
    if code != 0:
        return code

    if not args.skip_preprocessing:
        code = run_step("Step 2: Preprocessing", [str(ROOT / "scripts" / "run_preprocessing.py")])
        if code != 0:
            return code
        code = run_step("Step 3: Embeddings & clustering", [str(ROOT / "scripts" / "run_embeddings.py")])
        if code != 0:
            return code
    else:
        print("\n" + "=" * 60)
        print("Skipping steps 2–3 (--skip-preprocessing)")
        print("=" * 60)
        need = ROOT / "data" / "processed" / "monthly_statements.json"
        if not need.is_file():
            print(f"ERROR: {need} missing — run without --skip-preprocessing first.")
            return 1

    code = run_step(
        "Step 4: Model smoke test (aborts if vLLM chat models unreachable)",
        [str(ROOT / "scripts" / "test_models.py")],
    )
    if code != 0:
        return code

    code = run_step("Step 5: Experiments 1–5", [str(ROOT / "scripts" / "run_experiments.py")])
    if code != 0:
        return code

    print_artifact_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
