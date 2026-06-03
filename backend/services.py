"""Data paths, cached loads, and model singletons for the API."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _statement_spent_received(statement: dict[str, Any]) -> tuple[float, float, float]:
    total_received = float(statement.get("total_received", 0))
    ts = statement.get("total_spent")
    tsn = statement.get("total_spent_non_deposit")
    if ts is not None:
        total_spent = float(ts)
    elif tsn is not None:
        total_spent = float(tsn)
    else:
        total_spent = 0.0
    net = statement.get("net_flow")
    if net is None:
        net_flow = total_received - total_spent
    else:
        net_flow = float(net)
    return total_spent, total_received, net_flow


@lru_cache(maxsize=1)
def load_statements() -> dict[str, Any]:
    p = ROOT / "data" / "processed" / "monthly_statements.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_transactions_df() -> pd.DataFrame | None:
    clustered = ROOT / "data" / "processed" / "transactions_clustered.parquet"
    clean = ROOT / "data" / "processed" / "transactions_clean.parquet"
    if clustered.is_file():
        return pd.read_parquet(clustered)
    if clean.is_file():
        return pd.read_parquet(clean)
    return None


@lru_cache(maxsize=1)
def load_cluster_labels() -> dict[str, Any]:
    p = ROOT / "data" / "processed" / "cluster_labels.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_patterns() -> dict[str, Any]:
    p = ROOT / "data" / "processed" / "patterns.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


_llama = None
_mistral = None
_qwen = None


def get_llama():
    global _llama
    if _llama is None:
        from models.llama_model import LlamaModel

        _llama = LlamaModel()
    return _llama


def get_mistral():
    global _mistral
    if _mistral is None:
        from models.mistral_model import MistralModel

        _mistral = MistralModel()
    return _mistral


def get_qwen():
    global _qwen
    if _qwen is None:
        from models.qwen_model import QwenModel

        _qwen = QwenModel()
    return _qwen


def dashboard_payload(month: str) -> dict[str, Any]:
    stmts = load_statements()
    if month not in stmts:
        return {"error": f"Unknown month: {month}"}
    s = stmts[month]
    spent, received, net = _statement_spent_received(s)
    tx_count = int(s.get("transaction_count", 0))
    anomaly_count = int(s.get("anomaly_count", 0))

    pie_rows = s.get("category_breakdown") or s.get("by_category") or []
    pie = []
    for r in pie_rows:
        if isinstance(r, dict):
            pie.append(
                {
                    "name": str(r.get("name", r.get("category", "?"))),
                    "value": float(r.get("total", 0)),
                }
            )

    top5 = s.get("top_transactions") or []
    if not top5:
        df = load_transactions_df()
        if df is not None and "month" in df.columns and "amount" in df.columns:
            sub = df[df["month"] == month].nlargest(5, "amount")
            top5 = sub.replace({np.nan: None}).to_dict("records")

    table_rows: list[dict] = []
    df = load_transactions_df()
    if df is not None and "month" in df.columns:
        sub = df[df["month"] == month]
        table_rows = sub.head(500).replace({np.nan: None}).to_dict("records")

    return {
        "month": month,
        "total_spent": spent,
        "total_received": received,
        "net_flow": net,
        "transaction_count": tx_count,
        "anomaly_count": anomaly_count,
        "category_pie": pie,
        "top_transactions": top5[:5] if isinstance(top5, list) else top5,
        "transactions_sample": table_rows,
    }


def experiment_csv(n: int) -> list[dict] | None:
    p = ROOT / "evaluation" / "results" / f"experiment_{n}.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p).replace({np.nan: None}).to_dict("records")


def clusters_scatter_data(max_points: int = 400) -> dict[str, Any]:
    """t-SNE on embeddings when available; else bubble chart from cluster summaries."""
    emb_path = ROOT / "embeddings" / "cache" / "transaction_embeddings.npy"
    parq = ROOT / "data" / "processed" / "transactions_clustered.parquet"
    labels_json = load_cluster_labels()
    clusters_meta = labels_json.get("clusters") or {}

    if emb_path.is_file() and parq.is_file():
        X = np.load(emb_path)
        df = pd.read_parquet(parq)
        n = min(len(X), len(df), max_points)
        if n < 5:
            return {"mode": "bubble", "points": [], "bubbles": _bubble_from_meta(clusters_meta)}
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=n, replace=False)
        Xs = X[idx].astype(np.float64)
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1))
        xy = tsne.fit_transform(Xs)
        labs = df.iloc[idx]["cluster"].values if "cluster" in df.columns else [0] * n
        amts = df.iloc[idx]["amount"].values if "amount" in df.columns else [0] * n
        cats = (
            df.iloc[idx]["category"].values.astype(str)
            if "category" in df.columns
            else [""] * n
        )
        points = [
            {
                "x": float(xy[i, 0]),
                "y": float(xy[i, 1]),
                "cluster": int(labs[i]) if pd.notna(labs[i]) else 0,
                "amount": float(amts[i]),
                "category": str(cats[i]),
            }
            for i in range(n)
        ]
        return {"mode": "tsne", "points": points, "bubbles": _bubble_from_meta(clusters_meta)}

    return {"mode": "bubble", "points": [], "bubbles": _bubble_from_meta(clusters_meta)}


def _bubble_from_meta(clusters_meta: dict) -> list[dict]:
    out = []
    for cid, info in clusters_meta.items():
        st = (info or {}).get("stats") or {}
        out.append(
            {
                "id": str(cid),
                "label": (info or {}).get("label", f"Cluster {cid}"),
                "size": int(st.get("size", 0)),
                "mean_amount": float(st.get("mean_amount", 0)),
                "top_category": str(st.get("top_category", "")),
            }
        )
    return out


def cluster_cards() -> list[dict]:
    labels = load_cluster_labels()
    clusters = labels.get("clusters") or {}
    cards = []

    for cid, info in sorted(clusters.items(), key=lambda x: str(x[0])):
        st = (info or {}).get("stats") or {}
        cards.append(
            {
                "id": str(cid),
                "name": (info or {}).get("label", f"Cluster {cid}"),
                "size": int(st.get("size", 0)),
                "top_category": str(st.get("top_category", "")),
                "mean_amount": float(st.get("mean_amount", 0)),
            }
        )
    return cards
