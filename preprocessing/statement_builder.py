"""Build structured monthly account statements from cleaned transaction data."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd


DEPOSIT_CATEGORY = "Deposit"


class StatementBuilder:
    """Aggregate per-month metrics for LLM-ready structured summaries."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

    def _row_to_dict(self, row: pd.Series) -> dict[str, Any]:
        """Serialize a transaction row for JSON-friendly output."""
        d: dict[str, Any] = {}
        for k, v in row.items():
            if hasattr(v, "item"):  # numpy scalar
                try:
                    d[k] = v.item()
                except (ValueError, AttributeError):
                    d[k] = v
            elif isinstance(v, (pd.Timestamp,)):
                d[k] = v.isoformat()
            elif pd.isna(v):
                d[k] = None
            else:
                d[k] = v
        return d

    def build_monthly_statement(self, df: pd.DataFrame, month: str) -> dict[str, Any]:
        """Filter to ``month`` (YYYY-MM) and compute aggregates and breakdowns."""
        if "month" not in df.columns:
            raise ValueError("DataFrame must contain a 'month' column (run TransactionCleaner.clean first).")

        mdf = df[df["month"] == month].copy()
        if mdf.empty:
            self.logger.warning("No rows for month %s", month)

        is_deposit = mdf["category"] == DEPOSIT_CATEGORY if "category" in mdf.columns else pd.Series(False, index=mdf.index)
        spending_mask = ~is_deposit

        total_spent = float(mdf.loc[spending_mask, "amount"].sum()) if "amount" in mdf.columns else 0.0
        total_received = float(mdf.loc[is_deposit, "amount"].sum()) if "amount" in mdf.columns else 0.0
        net_flow = total_received - total_spent

        transaction_count = int(len(mdf))
        anomaly_count = int(mdf["is_anomaly"].sum()) if "is_anomaly" in mdf.columns else 0

        weekend_mask = mdf["is_weekend"] if "is_weekend" in mdf.columns else pd.Series(False, index=mdf.index)
        # Per spec: sum of amount on weekend vs weekday rows (all transaction types).
        weekend_spending = float(mdf.loc[weekend_mask, "amount"].sum()) if "amount" in mdf.columns else 0.0
        weekday_spending = float(mdf.loc[~weekend_mask, "amount"].sum()) if "amount" in mdf.columns else 0.0

        category_breakdown: list[dict[str, Any]] = []
        if "category" in mdf.columns and "amount" in mdf.columns:
            grp = mdf.groupby("category", dropna=False)
            total_cat_amount = float(mdf["amount"].sum()) or 1.0
            for name, g in grp:
                amt = float(g["amount"].sum())
                cnt = int(len(g))
                category_breakdown.append(
                    {
                        "name": str(name),
                        "total": amt,
                        "count": cnt,
                        "percentage": round(100.0 * amt / total_cat_amount, 4),
                        "avg_transaction": round(amt / cnt, 6) if cnt else 0.0,
                    }
                )
            category_breakdown.sort(key=lambda x: x["total"], reverse=True)

        top_cols = [
            c
            for c in (
                "timestamp",
                "amount",
                "category",
                "type",
                "nameOrig",
                "nameDest",
                "is_anomaly",
                "amount_band",
            )
            if c in mdf.columns
        ]
        top_transactions: list[dict[str, Any]] = []
        if "amount" in mdf.columns and top_cols:
            top5 = mdf.nlargest(5, "amount")[top_cols]
            for _, r in top5.iterrows():
                top_transactions.append(self._row_to_dict(r))

        anomalies: list[dict[str, Any]] = []
        if "is_anomaly" in mdf.columns:
            anomaly_df = mdf[mdf["is_anomaly"]]
            anomaly_cols = [
                c
                for c in (
                    "timestamp",
                    "amount",
                    "category",
                    "type",
                    "nameOrig",
                    "nameDest",
                    "isFraud",
                    "amount_band",
                )
                if c in anomaly_df.columns
            ]
            for _, r in anomaly_df[anomaly_cols].iterrows():
                anomalies.append(self._row_to_dict(r))

        return {
            "month": month,
            "total_spent": total_spent,
            "total_received": total_received,
            "net_flow": net_flow,
            "transaction_count": transaction_count,
            "anomaly_count": anomaly_count,
            "weekend_spending": weekend_spending,
            "weekday_spending": weekday_spending,
            "category_breakdown": category_breakdown,
            "top_transactions": top_transactions,
            "anomalies": anomalies,
        }

    def build_all_statements(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Build a statement dict for every distinct ``month`` in the data."""
        if "month" not in df.columns:
            raise ValueError("DataFrame must contain a 'month' column.")
        months = sorted(df["month"].dropna().unique())
        return {str(m): self.build_monthly_statement(df, str(m)) for m in months}
