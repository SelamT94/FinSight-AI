#!/usr/bin/env python3
"""Smoke-test LLaMA + Mistral wrappers against one monthly statement (requires vLLM OpenAI API)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from models.chat_backend import llm_backend  # noqa: E402
from models.llama_model import LlamaModel  # noqa: E402
from models.mistral_model import MistralModel  # noqa: E402
from models.resource_monitor import ResourceMonitor  # noqa: E402


def main() -> int:
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stmt_path = ROOT / "data" / "processed" / "monthly_statements.json"
    if not stmt_path.is_file():
        print(f"Missing {stmt_path} — run notebooks/01_preprocessing.ipynb first.")
        return 1

    with open(stmt_path, encoding="utf-8") as f:
        monthly = json.load(f)
    if not monthly:
        print("monthly_statements.json is empty.")
        return 1

    first_key = sorted(monthly.keys())[0]
    statement = monthly[first_key]
    print(f"Using statement period: {first_key}\n")

    mon = ResourceMonitor()
    llama = LlamaModel()
    mistral = MistralModel()
    qwen = QwenModel()

    missing: list[str] = []
    if not llama.is_available():
        missing.append(llama.model_name)
    if not mistral.is_available():
        missing.append(mistral.model_name)
    if not qwen.is_available():
        missing.append(qwen.model_name)
    if missing:
        backend = llm_backend()
        if backend in ("vllm", "openai", "openai_compat"):
            hint = (
                "Start vLLM (`docker compose --profile infra up -d vllm-chat`) and set "
                "OPENAI_COMPAT_BASE_URL, LLAMA_MODEL, MISTRAL_MODEL to match --model."
            )
        else:
            hint = (
                "Start Ollama on OLLAMA_BASE_URL and run: "
                "ollama pull llama3:8b && ollama pull mistral:7b"
            )
        print(
            "\nABORT: Required chat model(s) not available: "
            + ", ".join(missing)
            + f"\n(LLM_BACKEND={backend}) {hint}\n"
        )
        return 2

    # --- Llama: three strategies ------------------------------------------------
    llama_results: dict[str, dict] = {}
    for strat in ("zero_shot", "few_shot", "chain_of_thought"):
        print(f"\n=== LlamaModel.generate_summary ({strat}) ===")
        try:

            def run():
                return llama.generate_summary(statement, prompt_strategy=strat)

            out, delta = mon.measure(run)
            llama_results[strat] = out
            print(out["summary"][:1200])
            if len(out["summary"]) > 1200:
                print("… [truncated]")
            meta = out["metadata"]
            print(
                f"\n[inference {meta['inference_time_sec']:.2f}s | "
                f"tok/s ~{meta['tokens_per_sec']:.1f} | "
                f"peak_ram {delta['peak_ram_gb']:.2f} GB | "
                f"cpu_avg {delta['cpu_avg_percent']:.1f}%]"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            llama_results[strat] = {}

    # --- Mistral: zero-shot + comparative eval ---------------------------------
    print("\n=== MistralModel.generate_summary (zero_shot) ===")
    mistral_zero: dict = {}
    try:

        def run_m():
            return mistral.generate_summary(statement, prompt_strategy="zero_shot")

        mz, d_m = mon.measure(run_m)
        mistral_zero = mz
        print(mz["summary"][:1200])
        meta = mz["metadata"]
        print(
            f"\n[inference {meta['inference_time_sec']:.2f}s | "
            f"peak_ram {d_m['peak_ram_gb']:.2f} GB]"
        )
    except Exception as e:
        print(f"FAILED: {e}")

    # --- Qwen: zero-shot ---------------------------------
    print("\n=== QwenModel.generate_summary (zero_shot) ===")
    try:

        def run_q():
            return qwen.generate_summary(statement, prompt_strategy="zero_shot")

        qz, d_q = mon.measure(run_q)
        print(qz["summary"][:1200])
        q_meta = qz["metadata"]
        print(
            f"\n[inference {q_meta['inference_time_sec']:.2f}s | "
            f"peak_ram {d_q['peak_ram_gb']:.2f} GB]"
        )
    except Exception as e:
        print(f"FAILED: {e}")

    llama_summary_text = ""
    z = llama_results.get("zero_shot") or {}
    if isinstance(z, dict):
        llama_summary_text = str(z.get("summary", ""))

    if llama_summary_text:
        print("\n=== MistralModel.evaluate_summary (vs Llama zero-shot) ===")
        try:

            def run_e():
                return mistral.evaluate_summary(statement, llama_summary_text)

            ev, d_e = mon.measure(run_e)
            print(
                json.dumps(
                    {k: v for k, v in ev.items() if k != "metadata"},
                    indent=2,
                )
            )
            meta = ev.get("metadata", {})
            print(
                f"\n[inference {meta.get('inference_time_sec', 0):.2f}s | "
                f"peak_ram {d_e['peak_ram_gb']:.2f} GB]"
            )
        except Exception as e:
            print(f"FAILED: {e}")
    else:
        print("\nSkipping comparative eval (no Llama zero-shot summary).")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
