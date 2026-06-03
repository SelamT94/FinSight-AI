#!/usr/bin/env python3
"""CLI entry point for embeddings + clustering (same steps as ``02_embeddings.ipynb``)."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from embeddings.clusterer import SemanticClusterer  # noqa: E402
from embeddings.embed_generator import create_embedder  # noqa: E402
from embeddings.pattern_detector import PatternDetector  # noqa: E402


def configure_logging() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    load_dotenv(ROOT / ".env")
    configure_logging()
    log = logging.getLogger("run_embeddings")

    parquet_in = ROOT / "data" / "processed" / "transactions_clean.parquet"
    if not parquet_in.is_file():
        log.error("Missing %s — run notebooks/01_preprocessing.ipynb first.", parquet_in)
        return 1

    embedder = create_embedder()
    log.info("TEI embeddings at %s", embedder.base_url)

    import pandas as pd

    df_full = pd.read_parquet(parquet_in)
    sample_n = min(10_000, len(df_full))
    df_sample = df_full.sample(n=sample_n, random_state=42).reset_index(drop=True)
    log.info("Loaded %s rows; embedding sample n=%s", len(df_full), sample_n)
    embeddings = embedder.embed_transactions(df_sample)

    clusterer = SemanticClusterer(n_clusters=6)
    elbow_path = ROOT / "visualizations" / "elbow_curve.png"
    optimal_k = clusterer.find_optimal_k(embeddings, k_range=range(3, 10), plot_path=elbow_path)
    log.info("find_optimal_k -> k=%s (plot: %s)", optimal_k, elbow_path)
    clusterer.n_clusters = optimal_k
    labels = clusterer.fit_predict(embeddings)
    df_sample = df_sample.copy()
    df_sample["cluster"] = labels

    summary = clusterer.label_clusters(df_sample, labels)
    log.info("Cluster summary keys: %s", list(summary.keys()))

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "transactions_embeddings")
    _, _ = clusterer.build_qdrant_index(
        embeddings,
        qdrant_url=qdrant_url,
        collection_name=qdrant_collection,
    )

    detector = PatternDetector()
    patterns = {
        "recurring_payments": detector.detect_recurring_payments(df_full),
        "spending_trends": detector.detect_spending_trends(df_full),
        "behavioral_patterns": detector.detect_behavioral_patterns(df_full),
    }

    out_parquet = ROOT / "data" / "processed" / "transactions_clustered.parquet"
    out_clusters = ROOT / "data" / "processed" / "cluster_labels.json"
    out_patterns = ROOT / "data" / "processed" / "patterns.json"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df_sample.to_parquet(out_parquet, index=False)
    with open(out_clusters, "w", encoding="utf-8") as f:
        json.dump(
            {
                "optimal_k": optimal_k,
                "sample_size": sample_n,
                "elbow_plot": str(elbow_path.relative_to(ROOT)),
                "clusters": {str(k): v for k, v in summary.items()},
                "vector_store": {
                    "provider": "qdrant",
                    "url": qdrant_url,
                    "collection": qdrant_collection,
                },
            },
            f,
            indent=2,
            default=str,
        )
    with open(out_patterns, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2, default=str)

    log.info("Wrote %s", out_parquet)
    log.info("Wrote %s", out_clusters)
    log.info("Wrote %s", out_patterns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
