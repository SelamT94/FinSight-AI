"""Primary chat track (experiment “llama” column) via Ollama or OpenAI-compatible API (vLLM)."""

from __future__ import annotations

import json
import re
from typing import Any

from prompts.pattern_format import compact_patterns_for_prompt
from prompts.statement_format import statement_placeholders
from prompts import templates as T

from .chat_backend import create_chat_client


class LlamaModel:
    """High-level helpers for financial prompts; backs the legacy “LlamaModel” experiment path."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._client = create_chat_client(
            model_env="LLAMA_MODEL",
            default_ollama="llama3:8b",
            model_name=model_name,
            base_url=base_url,
        )

    @property
    def model_name(self) -> str:
        return self._client.model_name

    @property
    def base_url(self) -> str:
        return self._client.base_url

    def is_available(self) -> bool:
        return self._client.is_available()

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _build_summary_user_prompt(self, strategy: str, ph: dict[str, Any]) -> str:
        if strategy == "zero_shot":
            return T.ZERO_SHOT_SUMMARY_PROMPT.format(**ph)
        if strategy == "few_shot":
            return T.FEW_SHOT_SUMMARY_PROMPT.format(**ph)
        if strategy == "chain_of_thought":
            return T.CHAIN_OF_THOUGHT_SUMMARY_PROMPT.format(**ph)
        raise ValueError(f"Unknown prompt_strategy: {strategy!r}")

    def generate_summary(
        self,
        statement: dict[str, Any],
        prompt_strategy: str = "zero_shot",
    ) -> dict[str, Any]:
        ph = statement_placeholders(statement)
        user_prompt = self._build_summary_user_prompt(prompt_strategy, ph)
        meta = self.generate(
            user_prompt,
            system_prompt=T.SYSTEM_PROMPT_FINANCIAL_ANALYST,
            temperature=0.3,
            max_tokens=1024,
        )
        return {
            "summary": meta["response"],
            "prompt_strategy": prompt_strategy,
            "metadata": {k: v for k, v in meta.items() if k != "raw"},
        }

    def evaluate_summary(
        self,
        statement: dict[str, Any],
        llama_summary: str,
    ) -> dict[str, Any]:
        ph = statement_placeholders(statement)
        user_prompt = T.COMPARATIVE_EVALUATION_PROMPT.format(
            statement_json=ph["statement_json"],
            llama_summary=llama_summary,
        )
        meta = self.generate(
            user_prompt,
            system_prompt=T.SYSTEM_PROMPT_FINANCIAL_ANALYST,
            temperature=0.2,
            max_tokens=1200,
        )
        text = meta["response"].strip()
        parsed = self._parse_evaluation_json(text)
        parsed["metadata"] = {k: v for k, v in meta.items() if k != "raw"}
        return parsed

    def _parse_evaluation_json(self, text: str) -> dict[str, Any]:
        base: dict[str, Any] = {
            "accuracy_score": 0,
            "completeness_score": 0,
            "clarity_score": 0,
            "alternative_summary": "",
            "factual_errors": "",
        }

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                base["accuracy_score"] = int(obj.get("accuracy_score", 0))
                base["completeness_score"] = int(obj.get("completeness_score", 0))
                base["clarity_score"] = int(obj.get("clarity_score", 0))
                base["alternative_summary"] = str(obj.get("alternative_summary", ""))
                base["factual_errors"] = str(obj.get("factual_errors", ""))
                return base
        except json.JSONDecodeError:
            pass

        def grab_int(pattern: str) -> int:
            m = re.search(pattern, text, re.I)
            return int(m.group(1)) if m else 0

        base["accuracy_score"] = grab_int(r"accuracy_score[\"']?\s*:\s*(\d+)")
        if base["accuracy_score"] == 0:
            base["accuracy_score"] = grab_int(r"accuracy[^\d]*(\d+)")
        base["completeness_score"] = grab_int(r"completeness_score[\"']?\s*:\s*(\d+)")
        if base["completeness_score"] == 0:
            base["completeness_score"] = grab_int(r"completeness[^\d]*(\d+)")
        base["clarity_score"] = grab_int(r"clarity_score[\"']?\s*:\s*(\d+)")
        if base["clarity_score"] == 0:
            base["clarity_score"] = grab_int(r"clarity[^\d]*(\d+)")

        alt = re.search(
            r"alternative_summary[\"']?\s*:\s*\"([^\"]+)\"",
            text,
            re.DOTALL,
        )
        if alt:
            base["alternative_summary"] = alt.group(1).strip()

        err = re.search(r"factual_errors[\"']?\s*:\s*\"([^\"]*)\"", text, re.DOTALL)
        if err:
            base["factual_errors"] = err.group(1).strip()

        return base

    def generate_insights(self, statement: dict[str, Any], summary: str) -> dict[str, Any]:
        ph = statement_placeholders(statement)
        user_prompt = T.INSIGHT_GENERATION_PROMPT.format(
            statement_json=ph["statement_json"],
            summary=summary,
        )
        meta = self.generate(
            user_prompt,
            system_prompt=T.SYSTEM_PROMPT_FINANCIAL_ANALYST,
            temperature=0.35,
            max_tokens=1200,
        )
        insights = self._parse_numbered_list(meta["response"])
        return {
            "insights": insights,
            "metadata": {k: v for k, v in meta.items() if k != "raw"},
        }

    @staticmethod
    def _parse_numbered_list(text: str) -> list[str]:
        lines = []
        for line in text.splitlines():
            m = re.match(r"^\s*\d+\.\s*(.+)$", line.strip())
            if m:
                lines.append(m.group(1).strip())
        if not lines:
            return [ln.strip("- •").strip() for ln in text.splitlines() if ln.strip()]
        return lines

    def generate_behavioral_analysis(self, patterns: list[dict[str, Any]]) -> str:
        patterns_text = compact_patterns_for_prompt(patterns)
        user_prompt = T.BEHAVIORAL_ANALYSIS_PROMPT.format(patterns_text=patterns_text)
        meta = self.generate(
            user_prompt,
            system_prompt=T.SYSTEM_PROMPT_FINANCIAL_ANALYST,
            temperature=0.35,
            max_tokens=1024,
        )
        return meta["response"]
