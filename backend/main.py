"""FinSight API — serves data + LLM/Qdrant for the Next.js frontend."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import subprocess  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from fastapi import FastAPI, HTTPException, BackgroundTasks  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend import services  # noqa: E402

app = FastAPI(title="FinSight API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:3001,http://127.0.0.1:3001",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "root": str(ROOT)}


@app.get("/api/months")
def list_months():
    s = services.load_statements()
    return {"months": sorted(s.keys())}


@app.get("/api/statement/{month}")
def get_statement(month: str):
    s = services.load_statements()
    if month not in s:
        raise HTTPException(404, "Month not found")
    return s[month]


@app.get("/api/dashboard/{month}")
def dashboard(month: str):
    payload = services.dashboard_payload(month)
    if "error" in payload:
        raise HTTPException(404, payload["error"])
    return payload


@app.get("/api/clusters/summary")
def clusters_summary():
    return {"cards": services.cluster_cards()}


@app.get("/api/clusters/scatter")
def clusters_scatter(max_points: int = 400):
    return services.clusters_scatter_data(max_points=max_points)


@app.get("/api/patterns")
def patterns():
    return services.load_patterns()


class SummarizeBody(BaseModel):
    month: str
    strategy: Literal["zero_shot", "few_shot", "chain_of_thought"] = "zero_shot"
    model: Literal["llama", "mistral", "qwen"] = "llama"


@app.post("/api/summarize")
def summarize(body: SummarizeBody):
    stmts = services.load_statements()
    if body.month not in stmts:
        raise HTTPException(404, "Month not found")
    stmt = stmts[body.month]
    try:
        if body.model == "llama":
            out = services.get_llama().generate_summary(stmt, body.strategy)
        elif body.model == "mistral":
            out = services.get_mistral().generate_summary(stmt, body.strategy)
        else:
            out = services.get_qwen().generate_summary(stmt, body.strategy)
    except Exception as e:
        raise HTTPException(502, f"Model error: {e}") from e
    return out


class InsightsBody(BaseModel):
    month: str
    summary: str
    model: Literal["llama", "mistral", "qwen"] = "llama"


@app.post("/api/insights")
def insights(body: InsightsBody):
    stmts = services.load_statements()
    if body.month not in stmts:
        raise HTTPException(404, "Month not found")
    try:
        if body.model == "llama":
            out = services.get_llama().generate_insights(stmts[body.month], body.summary)
        elif body.model == "mistral":
            out = services.get_mistral().generate_insights(stmts[body.month], body.summary)
        else:
            out = services.get_qwen().generate_insights(stmts[body.month], body.summary)
    except Exception as e:
        raise HTTPException(502, f"Model error: {e}") from e
    return out


class CompareBody(BaseModel):
    month: str
    model_a: Literal["llama", "mistral", "qwen"] = "llama"
    model_b: Literal["llama", "mistral", "qwen"] = "mistral"


@app.post("/api/compare")
def compare(body: CompareBody):
    stmts = services.load_statements()
    if body.month not in stmts:
        raise HTTPException(404, "Month not found")
    stmt = stmts[body.month]
    
    def _get_api(m_name):
        if m_name == "llama": return services.get_llama()
        if m_name == "mistral": return services.get_mistral()
        return services.get_qwen()

    try:
        api_a = _get_api(body.model_a)
        api_b = _get_api(body.model_b)
        
        out_a = api_a.generate_summary(stmt, "zero_shot")
        out_b = api_b.generate_summary(stmt, "zero_shot")
        eval_out = api_b.evaluate_summary(stmt, out_a["summary"])
    except Exception as e:
        raise HTTPException(502, f"Model error: {e}") from e

    rouge = {}
    try:
        from evaluation.metrics import MetricsCalculator

        m = MetricsCalculator()
        rouge = m.compute_rouge(out_a["summary"], out_b["summary"])
    except Exception:
        rouge = {"rouge1_f": None, "rouge2_f": None, "rougeL_f": None}

    return {
        "model_a_id": body.model_a,
        "model_b_id": body.model_b,
        "a": {
            "summary": out_a["summary"],
            "metadata": out_a["metadata"],
        },
        "b": {
            "summary": out_b["summary"],
            "metadata": out_b["metadata"],
        },
        "evaluation": {k: v for k, v in eval_out.items() if k != "metadata"},
        "evaluation_metadata": eval_out.get("metadata"),
        "rouge_a_vs_b": rouge,
    }


class SimilarBody(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)


@app.post("/api/similar")
def similar(body: SimilarBody):
    parq = ROOT / "data" / "processed" / "transactions_clustered.parquet"
    if not parq.is_file():
        raise HTTPException(
            400,
            "Run notebooks/02_embeddings.ipynb (or scripts/run_embeddings.py) to build Qdrant index + clustered parquet.",
        )
    try:
        from embeddings.embed_generator import create_embedder  # noqa: PLC0415

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "transactions_embeddings")
        client = QdrantClient(url=qdrant_url.rstrip("/"))
        df = pd.read_parquet(parq)
        embedder = create_embedder()
        query_vector = embedder.embed_text(body.query)
        hits = client.search(
            collection_name=qdrant_collection,
            query_vector=query_vector,
            limit=min(body.k, len(df)),
        )
        hit_ids = [int(hit.id) for hit in hits if hit.id is not None]
        rows = df.iloc[hit_ids].replace({np.nan: None}).to_dict("records")
        return {"results": rows}
    except Exception as e:
        raise HTTPException(502, f"Similarity search failed: {e}") from e


@app.get("/api/experiments")
def experiments_list():
    out = []
    for n in range(1, 6):
        p = ROOT / "evaluation" / "results" / f"experiment_{n}.csv"
        out.append({"id": n, "path": str(p), "exists": p.is_file()})
    return {"experiments": out}


@app.get("/api/experiments/{n}")
def experiment_data(n: int):
    if n < 1 or n > 5:
        raise HTTPException(404, "Invalid experiment id")
    rows = services.experiment_csv(n)
    if rows is None:
        raise HTTPException(404, f"experiment_{n}.csv not found — run scripts/run_experiments.py")
    return {"experiment": n, "rows": rows}


@app.get("/api/experiments/{n}/download")
def experiment_download(n: int):
    if n < 1 or n > 5:
        raise HTTPException(404, "Invalid experiment id")
    p = ROOT / "evaluation" / "results" / f"experiment_{n}.csv"
    if not p.is_file():
        raise HTTPException(404, "CSV not found")
    return FileResponse(
        path=str(p),
        media_type="text/csv",
        filename=f"experiment_{n}.csv",
    )


class RunExperimentBody(BaseModel):
    model_id: Literal["llama", "mistral", "qwen", "all"] = "all"


def _run_experiment_task(model_id: str):
    script_path = ROOT / "scripts" / "run_experiments.py"
    try:
        subprocess.run(
            [sys.executable, str(script_path), "--model", model_id],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running experiments: {e.stderr.decode()}")


@app.post("/api/experiments/run")
def run_experiment(body: RunExperimentBody, background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_experiment_task, body.model_id)
    return {"status": "ok", "message": f"Started experiment runner for model: {body.model_id}"}
