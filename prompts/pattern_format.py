"""Compact Section 2 pattern payloads so behavioral prompts fit vLLM context (8192 tokens)."""

from __future__ import annotations

import json
from typing import Any


def _compact_item(item: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in item.items() if k != "dates"}
    dates = item.get("dates")
    if isinstance(dates, list) and dates:
        out["dates_count"] = len(dates)
        if len(dates) == 1:
            out["date_range"] = str(dates[0])
        else:
            out["date_range"] = f"{dates[0]} … {dates[-1]}"
    return out


def compact_patterns_for_prompt(
    patterns: list[dict[str, Any]],
    *,
    max_items: int = 25,
    max_json_chars: int = 12_000,
) -> str:
    """Serialize patterns for LLM prompts, dropping bulky date lists."""
    trimmed = [_compact_item(p) if isinstance(p, dict) else p for p in patterns[:max_items]]
    text = json.dumps(trimmed, indent=2, default=str)
    if len(text) <= max_json_chars:
        return text
    note = (
        f"\n\n[Truncated: {len(patterns)} pattern(s) total; "
        f"showing first {max_items} compact rows, JSON capped at {max_json_chars} chars.]"
    )
    while len(text) + len(note) > max_json_chars and len(trimmed) > 1:
        trimmed = trimmed[:-1]
        text = json.dumps(trimmed, indent=2, default=str)
    if len(text) + len(note) > max_json_chars:
        text = text[: max_json_chars - len(note)]
    return text + note
