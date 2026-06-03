"""Five-way experiment pipeline: dual chat tracks via OpenAI API, prompts, metrics, resources.

CSV columns keep labels ``llama`` / ``mistral`` / ``qwen`` for ``LlamaModel`` vs ``MistralModel`` paths.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from models.llama_model import LlamaModel
from models.mistral_model import MistralModel
from models.qwen_model import QwenModel
from models.resource_monitor import ResourceMonitor

from .metrics import MetricsCalculator


logger = logging.getLogger(__name__)


def _pick_months(keys: list[str], months: list[str] | None, n: int, seed: int = 42) -> list[str]:
    if months:
        return [m for m in months if m in keys][:n]
    rng = random.Random(seed)
    k = list(keys)
    rng.shuffle(k)
    return sorted(k[: min(n, len(k))])


def _insight_has_number(insight: str) -> bool:
    return bool(re.search(r"\d", insight))


def _parse_rating_array(text: str, n: int) -> list[float]:
    m = re.search(r"\[[\s\d,.]+\]", text)
    if not m:
        return [float("nan")] * n
    try:
        arr = json.loads(m.group(0))
        out = [float(x) for x in arr]
        while len(out) < n:
            out.append(float("nan"))
        return out[:n]
    except json.JSONDecodeError:
        return [float("nan")] * n


class ExperimentRunner:
    def __init__(
        self,
        statements: dict[str, dict[str, Any]],
        models: dict[str, Any],
        metrics: MetricsCalculator,
        monitor: ResourceMonitor,
        results_dir: Path | None = None,
    ) -> None:
        self.statements = statements
        self.models = models
        self.metrics = metrics
        self.monitor = monitor
        self.results_dir = results_dir or Path(__file__).resolve().parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._resource_events: list[dict[str, Any]] = []

    def _get_judge(self, default_model: Any) -> Any:
        return self.models.get("mistral", default_model)

    def _log_resource(
        self,
        experiment: str,
        month: str | None,
        model: str,
        meta: dict[str, Any],
        delta: dict[str, Any],
    ) -> None:
        self._resource_events.append(
            {
                "experiment": experiment,
                "month": month or "",
                "model": model,
                "inference_time_sec": float(meta.get("inference_time_sec", 0)),
                "tokens_per_sec": float(meta.get("tokens_per_sec", 0)),
                "total_tokens": int(meta.get("total_tokens", 0)),
                "peak_ram_gb": float(delta.get("peak_ram_gb", 0)),
            }
        )

    def run_experiment_1(self, months: list[str] | None = None) -> pd.DataFrame:
        """
        Zero-shot summaries; reference text = Mistral ``alternative_summary`` (or self).
        """
        keys = sorted(self.statements.keys())
        picked = _pick_months(keys, months, 3)
        rows: list[dict[str, Any]] = []

        for m in picked:
            stmt = self.statements[m]
            summaries = {}
            metas = {}

            for model_name, model in self.models.items():
                def run_model():
                    return model.generate_summary(stmt, "zero_shot")
                
                out, d = self.monitor.measure(run_model)
                self._log_resource("exp1", m, model_name, out["metadata"], d)
                summaries[model_name] = out["summary"]
                metas[model_name] = out["metadata"]

            for model_name, model in self.models.items():
                judge = self._get_judge(model)
                judge_name = "mistral" if "mistral" in self.models else model_name
                hyp = summaries[model_name]

                def run_ev():
                    return judge.evaluate_summary(stmt, hyp)

                ev, d_e = self.monitor.measure(run_ev)
                self._log_resource("exp1", m, judge_name, ev.get("metadata", {}), d_e)

                alt_ref = str(ev.get("alternative_summary", "") or hyp)
                if not alt_ref.strip():
                    alt_ref = hyp

                met = self.metrics.compute_all(hyp, alt_ref)
                rows.append(
                    {
                        "month": m,
                        "model": model_name,
                        "reference": f"{judge_name}_alternative_summary",
                        "summary_len": len(hyp),
                        **met,
                        "inference_time_sec": metas[model_name]["inference_time_sec"],
                        "tokens_per_sec": metas[model_name]["tokens_per_sec"],
                        "total_tokens": metas[model_name]["total_tokens"],
                    }
                )

        return pd.DataFrame(rows)

    def run_experiment_2(self, months: list[str] | None = None) -> pd.DataFrame:
        """Insights for 3 months; numeric grounding + ratings."""
        keys = sorted(self.statements.keys())
        picked = _pick_months(keys, months, 3)
        rows: list[dict[str, Any]] = []

        for m in picked:
            stmt = self.statements[m]

            for model_name, model in self.models.items():
                def run_s():
                    return model.generate_summary(stmt, "zero_shot")

                so, ds = self.monitor.measure(run_s)
                self._log_resource("exp2", m, model_name, so["metadata"], ds)
                summary_text = so["summary"]

                def run_i():
                    return model.generate_insights(stmt, summary_text)

                ins, di = self.monitor.measure(run_i)
                self._log_resource("exp2", m, model_name, ins["metadata"], di)

                insights_list = ins["insights"]
                n_ins = len(insights_list)
                n_num = sum(1 for x in insights_list if _insight_has_number(x))
                numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(insights_list))
                rate_prompt = (
                    "Rate each insight below from 1 (weak) to 10 (strong) on usefulness and "
                    "whether it is grounded in quantitative evidence.\n"
                    "Reply with ONLY a JSON array of numbers in order, e.g. [7,8,9,6].\n\n"
                    f"{numbered}"
                )

                judge = self._get_judge(model)
                judge_name = "mistral" if "mistral" in self.models else model_name

                def run_rate():
                    return judge.generate(
                        rate_prompt,
                        system_prompt="Reply with JSON only.",
                        temperature=0.15,
                        max_tokens=256,
                    )

                rmeta, dr = self.monitor.measure(run_rate)
                self._log_resource("exp2", m, judge_name, rmeta, dr)
                ratings = _parse_rating_array(rmeta["response"], len(insights_list))
                valid_rat = [x for x in ratings if x == x]
                rows.append(
                    {
                        "month": m,
                        "model": model_name,
                        "n_insights": n_ins,
                        "target_insights": "4-5",
                        "n_with_digit": n_num,
                        "numeric_rate": n_num / n_ins if n_ins else 0.0,
                        f"{judge_name}_rating_mean": float(sum(valid_rat) / len(valid_rat))
                        if valid_rat
                        else float("nan"),
                        f"{judge_name}_ratings": json.dumps(ratings),
                        "insights_inference_sec": ins["metadata"]["inference_time_sec"],
                        "total_tokens_insights": ins["metadata"]["total_tokens"],
                        "peak_ram_gb": di["peak_ram_gb"],
                    }
                )

        return pd.DataFrame(rows)

    def run_experiment_3(self, month: str | None = None) -> pd.DataFrame:
        """Compare prompt strategies; evaluated by judge."""
        keys = sorted(self.statements.keys())
        m = month if month and month in self.statements else keys[0]
        stmt = self.statements[m]

        rows: list[dict[str, Any]] = []

        for model_name, model in self.models.items():
            judge = self._get_judge(model)
            judge_name = "mistral" if "mistral" in self.models else model_name

            def run_baseline():
                return judge.generate_summary(stmt, "zero_shot")

            base, db = self.monitor.measure(run_baseline)
            self._log_resource("exp3", m, judge_name, base["metadata"], db)
            ref_summary = base["summary"]

            for strat in ("zero_shot", "few_shot", "chain_of_thought"):
                def run_strat():
                    return model.generate_summary(stmt, strat)

                out, d = self.monitor.measure(run_strat)
                self._log_resource("exp3", m, model_name, out["metadata"], d)
                summ = out["summary"]

                ev, de = self.monitor.measure(
                    lambda s=summ: judge.evaluate_summary(stmt, s)
                )
                self._log_resource("exp3", m, judge_name, ev.get("metadata", {}), de)

                rouge_met = self.metrics.compute_all(summ, ref_summary)
                rows.append(
                    {
                        "month": m,
                        "model": model_name,
                        "strategy": strat,
                        "accuracy_score": int(ev.get("accuracy_score", 0)),
                        "completeness_score": int(ev.get("completeness_score", 0)),
                        "clarity_score": int(ev.get("clarity_score", 0)),
                        "rouge1_f": rouge_met["rouge1_f"],
                        "rouge2_f": rouge_met["rouge2_f"],
                        "rougeL_f": rouge_met["rougeL_f"],
                        "bertscore_f1": rouge_met.get("bertscore_f1"),
                        "bleu": rouge_met.get("bleu"),
                        "inference_time_sec": out["metadata"]["inference_time_sec"],
                        "total_tokens": out["metadata"]["total_tokens"],
                    }
                )

        return pd.DataFrame(rows)

    def run_experiment_4(self, months: list[str] | None = None) -> pd.DataFrame:
        """Side-by-side zero-shot; symmetric NLP metrics + resources."""
        keys = sorted(self.statements.keys())
        picked = _pick_months(keys, months, 3)
        rows: list[dict[str, Any]] = []

        model_names = list(self.models.keys())

        for m in picked:
            stmt = self.statements[m]
            summaries = {}
            metas = {}
            peaks = {}

            for model_name, model in self.models.items():
                def run_model():
                    return model.generate_summary(stmt, "zero_shot")

                out, d = self.monitor.measure(run_model)
                self._log_resource("exp4", m, model_name, out["metadata"], d)
                summaries[model_name] = out["summary"]
                metas[model_name] = out["metadata"]
                peaks[model_name] = d["peak_ram_gb"]

            row_data = {"month": m}
            for name_a in model_names:
                for name_b in model_names:
                    if name_a != name_b:
                        met = self.metrics.compute_all(summaries[name_a], summaries[name_b])
                        row_data[f"rouge1_{name_a}_vs_{name_b}"] = met["rouge1_f"]
                        row_data[f"rougeL_{name_a}_vs_{name_b}"] = met["rougeL_f"]
                        row_data[f"bert_f1_{name_a}_vs_{name_b}"] = met.get("bertscore_f1")
                        row_data[f"bleu_{name_a}_vs_{name_b}"] = met.get("bleu")

            for name in model_names:
                row_data[f"{name}_inference_sec"] = metas[name]["inference_time_sec"]
                row_data[f"{name}_tokens_per_sec"] = metas[name]["tokens_per_sec"]
                row_data[f"{name}_total_tokens"] = metas[name]["total_tokens"]
                row_data[f"{name}_peak_ram_gb"] = peaks[name]

            rows.append(row_data)

        return pd.DataFrame(rows)

    def run_experiment_5(self) -> pd.DataFrame:
        """Aggregate resource metrics from exp1–exp4 runs."""
        if not self._resource_events:
            return pd.DataFrame()
        df = pd.DataFrame(self._resource_events)
        df = df[df["experiment"].isin({"exp1", "exp2", "exp3", "exp4"})]
        if df.empty:
            return pd.DataFrame()
        agg = (
            df.groupby("model", as_index=False)
            .agg(
                mean_inference_time=("inference_time_sec", "mean"),
                mean_tokens_per_sec=("tokens_per_sec", "mean"),
                mean_ram_gb=("peak_ram_gb", "mean"),
                total_tokens_generated=("total_tokens", "sum"),
            )
        )
        return agg

    def run_all(self) -> dict[str, pd.DataFrame]:
        """Run experiments 1–5 in order; save CSVs under ``evaluation/results/``."""
        out: dict[str, pd.DataFrame] = {}
        out["experiment_1"] = self.run_experiment_1()
        out["experiment_1"].to_csv(self.results_dir / "experiment_1.csv", index=False)

        out["experiment_2"] = self.run_experiment_2()
        out["experiment_2"].to_csv(self.results_dir / "experiment_2.csv", index=False)

        out["experiment_3"] = self.run_experiment_3()
        out["experiment_3"].to_csv(self.results_dir / "experiment_3.csv", index=False)

        out["experiment_4"] = self.run_experiment_4()
        out["experiment_4"].to_csv(self.results_dir / "experiment_4.csv", index=False)

        out["experiment_5"] = self.run_experiment_5()
        out["experiment_5"].to_csv(self.results_dir / "experiment_5.csv", index=False)

        logger.info("Saved experiment CSVs to %s", self.results_dir)
        return out
