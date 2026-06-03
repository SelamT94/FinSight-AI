"""Tables, plots, and JSON bundle for experiment reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


class ResultReporter:
    def __init__(
        self,
        results_dir: Path | None = None,
        viz_dir: Path | None = None,
        processed_parquet: Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parent.parent
        self.results_dir = results_dir or (Path(__file__).resolve().parent / "results")
        self.viz_dir = viz_dir or (root / "visualizations")
        self.processed_parquet = processed_parquet or (
            root / "data" / "processed" / "transactions_clean.parquet"
        )
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def print_experiment_summary(self, exp_num: int, df: pd.DataFrame) -> None:
        print(f"\n{'=' * 60}\nExperiment {exp_num}\n{'=' * 60}")
        if df.empty:
            print("(empty DataFrame)")
            return
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(df.to_string(index=False))

    def generate_comparison_table(self, exp4_df: pd.DataFrame) -> str:
        """Markdown-friendly LLaMA vs Mistral comparison for reports."""
        if exp4_df.empty:
            return ""
        cols = [
            c
            for c in exp4_df.columns
            if c
            in (
                "month",
                "rouge1_llama_vs_mistral",
                "rougeL_llama_vs_mistral",
                "bert_f1_llama_vs_mistral",
                "llama_inference_sec",
                "mistral_inference_sec",
                "llama_tokens_per_sec",
                "mistral_tokens_per_sec",
            )
        ]
        sub = exp4_df[cols] if cols else exp4_df
        lines = ["| " + " | ".join(sub.columns) + " |", "| " + " | ".join(["---"] * len(sub.columns)) + " |"]
        for _, row in sub.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in sub.columns) + " |")
        return "\n".join(lines)

    def plot_rouge_comparison(self, exp1_df: pd.DataFrame) -> Path:
        """Grouped bars: mean ROUGE vs reference by model."""
        out = self.viz_dir / "rouge_comparison.png"
        if exp1_df.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No experiment 1 data", ha="center")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return out

        g = exp1_df.groupby("model")[["rouge1_f", "rouge2_f", "rougeL_f"]].mean()
        labels = list(g.index)
        x = range(len(labels))
        w = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, col in enumerate(["rouge1_f", "rouge2_f", "rougeL_f"]):
            ax.bar([xi + i * w for xi in x], g[col].values, width=w, label=col.replace("_f", ""))
        ax.set_xticks([xi + w for xi in x])
        ax.set_xticklabels(labels)
        ax.set_ylabel("ROUGE F1")
        ax.set_title("Mean ROUGE (vs Mistral alternative reference)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_resource_efficiency(self, exp5_df: pd.DataFrame) -> Path:
        out = self.viz_dir / "resource_efficiency.png"
        if exp5_df.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No resource data", ha="center")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return out

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].barh(exp5_df["model"], exp5_df["mean_inference_time"])
        axes[0].set_xlabel("Mean inference time (s)")
        axes[0].set_title("Latency")

        axes[1].barh(exp5_df["model"], exp5_df["mean_ram_gb"])
        axes[1].set_xlabel("Mean peak RAM (GB)")
        axes[1].set_title("Memory")

        fig.suptitle("Resource efficiency (experiments 1–4 aggregate)")
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_prompt_strategy_comparison(self, exp3_df: pd.DataFrame) -> Path:
        out = self.viz_dir / "prompt_strategy_comparison.png"
        if exp3_df.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No experiment 3 data", ha="center")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return out

        df = exp3_df.set_index("strategy")[["accuracy_score", "completeness_score", "clarity_score"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        df.plot(kind="bar", ax=ax, rot=0)
        ax.set_ylabel("Score (1–10)")
        ax.set_title("Mistral evaluation of Llama summaries by prompt strategy")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Criterion")
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_spending_breakdown(self, statement: dict[str, Any], month: str) -> Path:
        """Pie (categories) + weekly bar if transaction parquet exists."""
        out = self.viz_dir / f"spending_{month}.png"
        categories = statement.get("category_breakdown") or statement.get("by_category") or []
        names: list[str] = []
        totals: list[float] = []
        for row in categories:
            if isinstance(row, dict):
                names.append(str(row.get("name", "?")))
                totals.append(float(row.get("total", 0)))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if names and totals:
            axes[0].pie(totals, labels=names, autopct="%1.1f%%")
            axes[0].set_title(f"Category mix — {month}")
        else:
            axes[0].text(0.5, 0.5, "No categories", ha="center")

        weekly_vals: list[float] = []
        weekly_labels: list[str] = []
        try:
            if self.processed_parquet.is_file():
                df = pd.read_parquet(self.processed_parquet)
                if "month" in df.columns and "amount" in df.columns:
                    sub = df[df["month"] == month]
                    if "timestamp" in sub.columns:
                        sub = sub.copy()
                        sub["_w"] = pd.to_datetime(sub["timestamp"]).dt.isocalendar().week
                        sub["_y"] = pd.to_datetime(sub["timestamp"]).dt.year
                        g = sub.groupby(["_y", "_w"], as_index=False)["amount"].sum()
                        weekly_vals = g["amount"].tolist()[:10]
                        weekly_labels = [f"W{int(r['_w'])}" for _, r in g.iterrows()][:10]
        except Exception:
            pass

        if weekly_vals:
            axes[1].bar(weekly_labels, weekly_vals)
            axes[1].set_title("Weekly spending (sum of amounts)")
            axes[1].tick_params(axis="x", rotation=45)
        else:
            axes[1].text(
                0.5,
                0.5,
                "Weekly trend needs timestamp in parquet",
                ha="center",
                va="center",
            )

        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def generate_full_report_data(self) -> dict[str, Any]:
        """Bundle CSV summaries + chart paths into one JSON-serializable dict."""
        bundle: dict[str, Any] = {"charts": {}, "tables": {}}
        for n in range(1, 6):
            p = self.results_dir / f"experiment_{n}.csv"
            if p.is_file():
                df = pd.read_csv(p)
                bundle["tables"][f"experiment_{n}"] = {
                    "path": str(p),
                    "n_rows": len(df),
                    "columns": list(df.columns),
                    "mean_row": df.mean(numeric_only=True).to_dict() if len(df) else {},
                }
        bundle["charts"]["rouge_comparison"] = str(self.viz_dir / "rouge_comparison.png")
        bundle["charts"]["resource_efficiency"] = str(self.viz_dir / "resource_efficiency.png")
        bundle["charts"]["prompt_strategy_comparison"] = str(
            self.viz_dir / "prompt_strategy_comparison.png"
        )
        out_path = self.results_dir / "full_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, default=str)
        bundle["report_json_path"] = str(out_path)
        return bundle
