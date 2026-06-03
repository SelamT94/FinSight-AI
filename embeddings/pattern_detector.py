"""Rule-based patterns combined with dataframe signals (recurring, trends, behavior)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _ensure_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        return out
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    if "is_weekend" not in out.columns:
        out["is_weekend"] = ts.dt.dayofweek >= 5
    if "day" not in out.columns:
        out["day"] = ts.dt.day
    iso = ts.dt.isocalendar()
    if "year_week" not in out.columns:
        out["year_week"] = (
            iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
        )
    return out


def _amount_band(amount: float) -> str:
    if amount < 500:
        return "Small"
    if amount <= 5000:
        return "Medium"
    return "Large"


class PatternDetector:
    """Recurring payments, week-over-week trends, and short behavioral observations."""

    def detect_recurring_payments(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Recurring-like groups: same category and amount_band with 3+ occurrences;
        ``approx_amount`` is the median amount in the group.
        """
        if df.empty:
            return []
        d = df.copy()
        if "amount_band" not in d.columns:
            d["amount_band"] = d["amount"].map(_amount_band)
        if "date" not in d.columns and "timestamp" in d.columns:
            d["date"] = pd.to_datetime(d["timestamp"]).dt.date.astype(str)

        results: list[dict[str, Any]] = []
        for (cat, band), g in d.groupby(["category", "amount_band"], dropna=False):
            if len(g) < 3:
                continue
            med = float(g["amount"].median())
            dates = g["date"].tolist() if "date" in g.columns else []
            results.append(
                {
                    "category": str(cat),
                    "approx_amount": med,
                    "occurrences": int(len(g)),
                    "dates": dates[:50],
                    "pattern_label": f"Recurring {cat} ~${med:,.0f} ({len(g)}×, {band})",
                }
            )
        results.sort(key=lambda x: x["occurrences"], reverse=True)
        return results

    def detect_spending_trends(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Week-over-week change in total amount per category (all rows)."""
        d = _ensure_calendar(df)
        if "year_week" not in d.columns or "category" not in d.columns:
            return {}
        weekly = (
            d.groupby(["year_week", "category"], observed=True)["amount"]
            .sum()
            .reset_index()
        )
        out: dict[str, dict[str, Any]] = {}
        for cat, g in weekly.groupby("category"):
            g = g.sort_values("year_week")
            if len(g) < 2:
                out[str(cat)] = {"trend": "stable", "pct_change": 0.0}
                continue
            prev = float(g["amount"].iloc[-2])
            cur = float(g["amount"].iloc[-1])
            if prev == 0:
                pct = 0.0 if cur == 0 else 100.0
            else:
                pct = (cur - prev) / abs(prev) * 100.0
            if pct > 5:
                trend = "increasing"
            elif pct < -5:
                trend = "decreasing"
            else:
                trend = "stable"
            out[str(cat)] = {"trend": trend, "pct_change": round(pct, 4)}
        return out

    def detect_behavioral_patterns(self, df: pd.DataFrame) -> list[str]:
        """Short human-readable observations for reports / LLM context."""
        obs: list[str] = []
        if df.empty:
            return obs
        d = _ensure_calendar(df)
        if "amount" in d.columns and "is_weekend" in d.columns:
            w_end = float(d.loc[d["is_weekend"], "amount"].sum())
            w_day = float(d.loc[~d["is_weekend"], "amount"].sum())
            if w_end > w_day * 1.05:
                obs.append(
                    "Weekend transaction volume (sum of amounts) is higher than weekday volume."
                )
            elif w_day > w_end * 1.05:
                obs.append(
                    "Weekday transaction volume exceeds weekend volume."
                )
        if "timestamp" in d.columns:
            ts = pd.to_datetime(d["timestamp"], errors="coerce")
            d = d.assign(_ts=ts)
            small = d["amount"] < 100 if "amount" in d.columns else pd.Series(False, index=d.index)
            day_counts = d.loc[small].groupby(d.loc[small, "_ts"].dt.date).size()
            if len(day_counts) and day_counts.max() >= 10:
                obs.append(
                    "Several days show many small transactions (< $100), suggesting rapid micro-activity."
                )
            d["_dom"] = d["_ts"].dt.day
            last3 = d["_dom"] >= 28
            if last3.any() and (~last3).any() and "amount" in d.columns:
                end_m = float(d.loc[last3, "amount"].sum()) / max(int(last3.sum()), 1)
                rest = float(d.loc[~last3, "amount"].sum()) / max(int((~last3).sum()), 1)
                if end_m > rest * 1.2:
                    obs.append(
                        "Higher average transaction size in the last few days of the month vs earlier days."
                    )
        if not obs:
            obs.append("No strong behavioral flags in the current slice; patterns may be subtle.")
        return obs
