#!/usr/bin/env python3
"""Create project directories, verify vLLM (OpenAI API), TEI embeddings, seed .env from example."""

from __future__ import annotations

import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"
ENV_FILE = ROOT / ".env"



REQUIRED_DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "embeddings" / "cache",
    ROOT / "embeddings",
    ROOT / "prompts",
    ROOT / "models",
    ROOT / "evaluation",
    ROOT / "evaluation" / "results",
    ROOT / "visualizations",
    ROOT / "notebooks",
    ROOT / "report",
    ROOT / "scripts",
]

def ensure_directories() -> None:
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists() and d.name in ("raw", "processed", "cache"):
            gitkeep.touch()


def _read_env_var(key: str, default: str) -> str:
    for src in (ENV_FILE, ENV_EXAMPLE):
        if not src.is_file():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{key}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


def openai_compat_base_url() -> str:
    return _read_env_var(
        "OPENAI_COMPAT_BASE_URL",
        "http://localhost:5000/v1",
    )


def tei_base_url() -> str:
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("EMBEDDING_SERVER_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
            if line.startswith("TEI_BASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    if ENV_EXAMPLE.is_file():
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("EMBEDDING_SERVER_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "http://localhost:8001"


def fetch_tei_openapi(base: str) -> tuple[bool, str, str]:
    """TEI publishes OpenAPI at /openapi.json when healthy."""
    base = base.rstrip("/")
    url = f"{base}/openapi.json"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
            if resp.status != 200:
                return False, url, f"HTTP {resp.status}"
            _ = resp.read(512)
        return True, url, ""
    except urllib.error.URLError as e:
        return False, url, str(e)
    except TimeoutError:
        return False, url, "timed out"


def fetch_openai_compat_models(base: str) -> tuple[bool, str, list[str]]:
    """GET /v1/models on an OpenAI-compatible server (vLLM)."""
    url = base.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        models = []
        for m in data.get("data", []) or []:
            if isinstance(m, dict) and "id" in m:
                models.append(str(m["id"]))
        return True, url, models
    except urllib.error.URLError as e:
        return False, url, [f"URLError: {e}"]
    except TimeoutError:
        return False, url, ["Request timed out"]
    except json.JSONDecodeError as e:
        return False, url, [f"Invalid JSON: {e}"]


def _hf_tail(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def chat_model_registered(api_models: list[str], want: str) -> bool:
    if not api_models:
        return False
    tail = _hf_tail(want)
    for mid in api_models:
        if mid == want or mid.endswith(tail) or tail in mid or mid in want:
            return True
    return False


def ensure_env_file() -> str:
    if ENV_FILE.is_file():
        return f".env already exists at {ENV_FILE}"
    if not ENV_EXAMPLE.is_file():
        return f"Missing {ENV_EXAMPLE}; cannot create .env"
    shutil.copy(ENV_EXAMPLE, ENV_FILE)
    return f"Created .env from .env.example at {ENV_FILE}"


def main() -> int:
    print("FinSight-AI — project setup")
    print("=" * 50)

    ensure_directories()
    print("Directories OK (data/, embeddings/, prompts/, models/, etc.).")

    env_msg = ensure_env_file()
    print(env_msg)

    api_base = openai_compat_base_url()
    llama_id = _read_env_var(
        "LLAMA_MODEL",
        "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    )
    mistral_id = _read_env_var(
        "MISTRAL_MODEL",
        "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    )
    ok, url, models = fetch_openai_compat_models(api_base)
    print()
    print("Chat LLM (OpenAI-compatible API, e.g. vLLM)")
    print("-" * 50)
    print(f"Endpoint checked: {url}")
    if not ok:
        print(
            "Status: NOT REACHABLE (start docker compose profile `infra` "
            "or fix OPENAI_COMPAT_BASE_URL in .env)"
        )
        for line in models:
            print(f"  {line}")
    else:
        print("Status: reachable")
        print(f"  Server model id(s): {', '.join(models) if models else '(none listed)'}")
        for label, mid in ("LLAMA_MODEL", llama_id), ("MISTRAL_MODEL", mistral_id):
            reg = chat_model_registered(models, mid)
            hint = (
                "listed"
                if reg
                else "not matched (fix model name / wait for weights to load)"
            )
            print(f"  - {label} ({mid}): {hint}")

    tei = tei_base_url().rstrip("/")
    ok_tei, tei_url, tei_err = fetch_tei_openapi(tei)
    print()
    print("Text Embeddings Inference (BGE-M3)")
    print("-" * 50)
    print(f"Endpoint checked: {tei_url}")
    if ok_tei:
        print(f"Status: reachable ({tei})")
    else:
        print(f"Status: NOT REACHABLE — start embedding-server ({tei})")
        if tei_err:
            print(f"  Detail: {tei_err}")

    print()
    print("Next steps:")
    print("  1. Place paysim1.csv at data/raw/paysim1.csv (or set PAYSIM_CSV_PATH in .env).")
    print("  2. pip install -r requirements.txt")
    print(
        "  3. Start docker compose profile `infra` (vLLM chat on host :5000, TEI on :8001)."
    )
    print("  4. Open notebooks/01_preprocessing.ipynb and run all cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
