"""Chunked PaySim CSV loading with validation and summary statistics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


REQUIRED_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "isFraud",
)


class PaySimLoader:
    """Load large PaySim CSV files in chunks with optional row cap (dev mode)."""

    def __init__(
        self,
        csv_path: str | Path,
        row_limit: int | None = None,
        chunksize: int = 50_000,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.row_limit = row_limit
        self.chunksize = chunksize
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

    def load_raw(self) -> pd.DataFrame:
        """Load the CSV using chunked reads; optionally truncate to ``row_limit``."""
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"PaySim CSV not found: {self.csv_path.resolve()}")

        chunks: list[pd.DataFrame] = []
        rows_read = 0

        # Estimate total lines for tqdm (optional, fast line count is expensive on 6M rows)
        # Use chunks-only progress instead of total rows when unknown
        reader = pd.read_csv(self.csv_path, chunksize=self.chunksize)
        pbar = tqdm(desc="Loading PaySim chunks", unit="chunk", dynamic_ncols=True)

        try:
            for chunk in reader:
                pbar.update(1)
                if self.row_limit is not None:
                    remaining = self.row_limit - rows_read
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk.iloc[:remaining].copy()
                chunks.append(chunk)
                rows_read += len(chunk)
                if self.row_limit is not None and rows_read >= self.row_limit:
                    break
        finally:
            pbar.close()

        if not chunks:
            self.logger.warning("No rows loaded from %s", self.csv_path)
            return pd.DataFrame()

        df = pd.concat(chunks, ignore_index=True)
        self.logger.info("Loaded shape: %s", df.shape)
        self.logger.info("Columns: %s", list(df.columns))
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Return True if all required PaySim columns are present."""
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            self.logger.warning("Missing required columns: %s", missing)
            return False
        return True

    def get_summary_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        """Aggregate stats for quick inspection and downstream reporting."""
        if df.empty or "step" not in df.columns:
            return {
                "total_rows": int(len(df)),
                "date_range": None,
                "transaction_types": {},
                "fraud_count": 0,
                "amount_stats": {"mean": None, "std": None, "max": None},
            }

        type_counts = (
            df["type"].value_counts().to_dict()
            if "type" in df.columns
            else {}
        )
        fraud_count = int(df["isFraud"].sum()) if "isFraud" in df.columns else 0
        amount = df["amount"] if "amount" in df.columns else pd.Series(dtype=float)

        return {
            "total_rows": int(len(df)),
            "date_range": {
                "step_min": int(df["step"].min()),
                "step_max": int(df["step"].max()),
            },
            "transaction_types": {str(k): int(v) for k, v in type_counts.items()},
            "fraud_count": fraud_count,
            "amount_stats": {
                "mean": float(amount.mean()) if len(amount) else None,
                "std": float(amount.std()) if len(amount) else None,
                "max": float(amount.max()) if len(amount) else None,
            },
        }
