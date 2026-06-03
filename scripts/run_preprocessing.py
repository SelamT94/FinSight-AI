#!/usr/bin/env python3
"""Legacy CLI preprocessor (modular package).

For the course project, prefer ``notebooks/01_preprocessing.ipynb`` — simpler and easier to show in reports.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from preprocessing.cleaner import TransactionCleaner  # noqa: E402
from preprocessing.loader import PaySimLoader  # noqa: E402
from preprocessing.statement_builder import StatementBuilder  # noqa: E402


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    load_dotenv(ROOT / ".env")
    configure_logging()
    log = logging.getLogger("run_preprocessing")

    csv_path = os.getenv("PAYSIM_CSV_PATH", str(ROOT / "data" / "raw" / "paysim1.csv"))
    row_limit_raw = os.getenv("DEV_ROW_LIMIT", "500000")
    try:
        row_limit = int(row_limit_raw) if row_limit_raw.strip() else None
    except ValueError:
        row_limit = 500_000
        log.warning("Invalid DEV_ROW_LIMIT=%r; using %s", row_limit_raw, row_limit)

    out_parquet = ROOT / "data" / "processed" / "transactions_clean.parquet"
    out_json = ROOT / "data" / "processed" / "monthly_statements.json"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    log.info("CSV path: %s", csv_path)
    log.info("Row limit (dev): %s", row_limit)

    loader = PaySimLoader(csv_path, row_limit=row_limit, chunksize=50_000)
    try:
        df = loader.load_raw()
    except FileNotFoundError as e:
        log.error("%s", e)
        log.error("Download PaySim from Kaggle and set PAYSIM_CSV_PATH, or run: python3 scripts/setup_project.py")
        return 1

    if not loader.validate(df):
        log.warning("Validation reported missing columns; continuing with available data.")

    stats_before = loader.get_summary_stats(df)
    log.info("Raw summary: %s", stats_before)

    cleaner = TransactionCleaner()
    df = cleaner.clean(df)
    df = cleaner.categorize(df)

    builder = StatementBuilder()
    statements = builder.build_all_statements(df)

    df.to_parquet(out_parquet, index=False)
    log.info("Wrote cleaned transactions: %s", out_parquet)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(statements, f, indent=2, ensure_ascii=False)
    log.info("Wrote monthly statements: %s", out_json)

    print()
    print("Processing summary")
    print("-" * 50)
    print(f"Cleaned rows:           {len(df):,}")
    print(f"Unique months:          {len(statements)}")
    print(f"Parquet output:         {out_parquet}")
    print(f"Monthly statements JSON: {out_json}")
    if statements:
        first_month = min(statements.keys())
        sample = statements[first_month]
        print(f"Example month ({first_month}): transactions={sample['transaction_count']}, "
              f"net_flow={sample['net_flow']:.2f}, anomalies={sample['anomaly_count']}")
    print("-" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
