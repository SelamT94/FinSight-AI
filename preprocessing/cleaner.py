"""Clean PaySim transactions and derive time-based and risk features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


TYPE_TO_CATEGORY: dict[str, str] = {
    "PAYMENT": "Bill Payment",
    "TRANSFER": "Transfer Out",
    "CASH_OUT": "Cash Withdrawal",
    "DEBIT": "Debit Purchase",
    "CASH_IN": "Deposit",
}


class TransactionCleaner:
    """Drop invalid rows, attach timestamps, categories, and anomaly flags."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate, remove non-positive amounts, and add calendar features."""
        initial = len(df)
        out = df.copy()

        before = len(out)
        out = out.drop_duplicates()
        self.logger.info("Dropped %s duplicate rows", before - len(out))

        before = len(out)
        if "amount" in out.columns:
            out = out[out["amount"] > 0]
        self.logger.info("Dropped %s rows with amount <= 0", before - len(out))

        if "step" not in out.columns:
            self.logger.error("Column 'step' missing; cannot build timestamps")
            return out

        base_date = pd.Timestamp("2024-01-01")
        out["timestamp"] = base_date + pd.to_timedelta(out["step"], unit="h")
        out["date"] = out["timestamp"].dt.date.astype(str)
        out["month"] = out["timestamp"].dt.strftime("%Y-%m")
        iso = out["timestamp"].dt.isocalendar()
        out["week"] = (
            iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
        )
        out["day_of_week"] = out["timestamp"].dt.dayofweek
        out["hour"] = out["timestamp"].dt.hour
        out["is_weekend"] = out["day_of_week"] >= 5

        self.logger.info("Rows after clean: %s (removed %s total)", len(out), initial - len(out))
        return out

    def categorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map transaction type to human-readable category and flag anomalies."""
        out = df.copy()

        if "type" in out.columns:
            out["category"] = out["type"].map(TYPE_TO_CATEGORY).fillna("Other")
        else:
            out["category"] = "Unknown"

        mean_amt = float(out["amount"].mean()) if len(out) and "amount" in out else 0.0
        std_amt = float(out["amount"].std()) if len(out) and "amount" in out else 0.0
        threshold = mean_amt + 3.0 * std_amt if std_amt and not np.isnan(std_amt) else mean_amt

        fraud_mask = out["isFraud"].eq(1) if "isFraud" in out.columns else pd.Series(False, index=out.index)
        amount_mask = out["amount"] > threshold if "amount" in out.columns else pd.Series(False, index=out.index)
        out["is_anomaly"] = fraud_mask | amount_mask

        def _band(a: float) -> str:
            if a < 500:
                return "Small"
            if a <= 5000:
                return "Medium"
            return "Large"

        if "amount" in out.columns:
            out["amount_band"] = out["amount"].map(_band)
        else:
            out["amount_band"] = "Small"

        return out
