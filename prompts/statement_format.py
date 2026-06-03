"""Normalize monthly statement dicts from Section 1 for prompt placeholders."""

from __future__ import annotations

import json
from typing import Any


def category_breakdown_lines(statement: dict[str, Any]) -> str:
    """Render category rows as readable lines for prompts."""
    rows = statement.get("category_breakdown") or statement.get("by_category") or []
    if not rows:
        return "(no category breakdown provided)"
    lines: list[str] = []
    for r in rows:
        if isinstance(r, dict):
            name = r.get("name", r.get("category", "?"))
            total = float(r.get("total", 0))
            pct = r.get("percentage", r.get("pct"))
            cnt = r.get("count", "")
            if pct is not None:
                lines.append(f"- {name}: ${total:,.2f} ({pct}% of flow), count={cnt}")
            else:
                lines.append(f"- {name}: ${total:,.2f}, count={cnt}")
        else:
            lines.append(str(r))
    return "\n".join(lines)


def statement_placeholders(statement: dict[str, Any]) -> dict[str, Any]:
    """Build kwargs for `.format()` / f-string templates."""
    period = str(statement.get("month") or statement.get("period") or "unknown")

    total_received = float(statement.get("total_received", 0))
    ts_spent = statement.get("total_spent")
    ts_non_dep = statement.get("total_spent_non_deposit")
    if ts_spent is not None:
        total_spent = float(ts_spent)
    elif ts_non_dep is not None:
        total_spent = float(ts_non_dep)
    else:
        total_spent = 0.0

    net = statement.get("net_flow")
    if net is None:
        net_flow = total_received - total_spent
    else:
        net_flow = float(net)

    tx_count = int(statement.get("transaction_count", 0))
    anomaly_count = int(statement.get("anomaly_count", 0))

    cat_text = category_breakdown_lines(statement)

    return {
        "period": period,
        "total_spent": total_spent,
        "total_received": total_received,
        "net_flow": float(net_flow),
        "transaction_count": tx_count,
        "anomaly_count": anomaly_count,
        "category_breakdown_text": cat_text,
        "statement_json": json.dumps(statement, indent=2, default=str),
    }
