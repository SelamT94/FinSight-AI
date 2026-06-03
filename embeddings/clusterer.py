"""KMeans clustering, elbow heuristic, Qdrant index, and cluster labeling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .embed_generator import TEIEmbedder


class SemanticClusterer:
    """KMeans on embedding vectors; elbow plot; Qdrant search; human-readable labels."""

    def __init__(self, n_clusters: int = 6) -> None:
        self.n_clusters = n_clusters

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit KMeans and return one label per row."""
        X = np.asarray(embeddings, dtype=np.float32)
        km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        return km.fit_predict(X)

    def find_optimal_k(
        self,
        embeddings: np.ndarray,
        k_range: range = range(3, 10),
        plot_path: Path | None = None,
    ) -> int:
        """
        Plot inertia elbow; pick smallest k (>=4) where relative inertia drop vs
        previous k is below 10%. If never triggered, return ``self.n_clusters``.
        """
        X = np.asarray(embeddings, dtype=np.float32)
        ks = list(k_range)
        inertias: list[float] = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(float(km.inertia_))

        if plot_path is None:
            plot_path = (
                Path(__file__).resolve().parent.parent
                / "visualizations"
                / "elbow_curve.png"
            )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(ks, inertias, "o-", linewidth=2)
        plt.xlabel("k (clusters)")
        plt.ylabel("Inertia")
        plt.title("KMeans elbow (inertia vs k)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        optimal = self.n_clusters
        for i in range(1, len(ks)):
            prev_i, cur_i = inertias[i - 1], inertias[i]
            if prev_i <= 0:
                continue
            rel_drop = (prev_i - cur_i) / prev_i
            if rel_drop < 0.10:
                optimal = ks[i]
                break

        return int(optimal)

    def label_clusters(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
    ) -> dict[int, dict[str, Any]]:
        """Summarize each cluster: dominant category, stats, sample rows."""
        out: dict[int, dict[str, Any]] = {}
        df = df.reset_index(drop=True)
        labels = np.asarray(cluster_labels)

        # First pass — collect stats for all clusters
        raw: dict[int, dict[str, Any]] = {}
        for cid in sorted(np.unique(labels)):
            mask = labels == cid
            sub = df.loc[mask]
            top_cat = (
                str(sub["category"].mode().iloc[0])
                if "category" in sub.columns and len(sub["category"].mode())
                else "Unknown"
            )
            mean_amt = float(sub["amount"].mean()) if "amount" in sub.columns else 0.0
            raw[int(cid)] = {
                "sub": sub,
                "top_cat": top_cat,
                "mean_amt": mean_amt,
            }

        # Second pass — deduplicate labels using amount tier
        used_labels: set[str] = set()
        for cid, r in raw.items():
            top_cat = r["top_cat"]
            mean_amt = r["mean_amt"]
            base_label = top_cat

            if base_label in used_labels:
                tier = (
                    "Small"      if mean_amt < 5_000  else
                    "Mid"        if mean_amt < 50_000 else
                    "Large"      if mean_amt < 200_000 else
                    "High-Value"
                )
                base_label = f"{top_cat} ({tier})"

            # If still clashing (e.g. three clusters same category + tier), append cluster id
            if base_label in used_labels:
                base_label = f"{top_cat} (Cluster {cid})"

            used_labels.add(base_label)
            sub = r["sub"]
            out[cid] = {
                "label": base_label,
                "stats": {
                    "size": int(len(sub)),
                    "mean_amount": mean_amt,
                    "top_category": top_cat,
                },
                "sample_transactions": sub.head(3).to_dict("records"),
            }

        return out

    def build_qdrant_index(
        self,
        embeddings: np.ndarray,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "transactions_embeddings",
        upsert_batch_size: int = 64,
    ) -> tuple[Any, str]:
        """Create/recreate a Qdrant collection and upload embeddings by row id."""
        from qdrant_client import QdrantClient  # noqa: PLC0415
        from qdrant_client.models import Distance, VectorParams  # noqa: PLC0415

        X = np.asarray(embeddings, dtype=np.float32)
        n = int(X.shape[0])
        client = QdrantClient(url=qdrant_url.rstrip("/"))
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=int(X.shape[1]), distance=Distance.COSINE),
        )
        env_batch = os.getenv("QDRANT_UPSERT_BATCH_SIZE")
        batch_size = int(env_batch) if env_batch else upsert_batch_size
        batch_size = max(1, min(batch_size, 512))
        client.upload_collection(
            collection_name=collection_name,
            vectors=X,
            payload=[{"row_id": i} for i in range(n)],
            ids=list(range(n)),
            batch_size=batch_size,
            wait=True,
        )
        return client, collection_name

    def find_similar_transactions(
        self,
        query_text: str,
        embedder: TEIEmbedder,
        qdrant_client: Any,
        collection_name: str,
        df: pd.DataFrame,
        k: int = 5,
    ) -> pd.DataFrame:
        """Embed ``query_text`` and return ``k`` nearest rows from Qdrant hits."""
        query_vector = embedder.embed_text(query_text)
        hits = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=min(k, len(df)),
        )
        idx_flat = [int(hit.id) for hit in hits if hit.id is not None]
        return df.iloc[idx_flat].copy()

