"""Embeddings via Hugging Face Text Embeddings Inference (TEI) HTTP API."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def _amount_band(amount: float) -> str:
    if amount < 500:
        return "Small"
    if amount <= 5000:
        return "Medium"
    return "Large"


def transaction_to_text(row: dict[str, Any]) -> str:
    """Build a single natural-language line for embedding."""
    category = str(row.get("category", "Unknown"))
    amount = float(row.get("amount", 0.0))
    date = str(row.get("date", ""))
    band = row.get("amount_band")
    if band is None or (isinstance(band, float) and np.isnan(band)):
        band = _amount_band(amount)

    dow = row.get("day_of_week")
    if dow is None or (isinstance(dow, float) and np.isnan(dow)):
        ts = row.get("timestamp")
        if ts is not None:
            t = pd.Timestamp(ts)
            dow = t.day_name()
        else:
            dow = "Unknown"
    else:
        if isinstance(dow, (int, float)) and not np.isnan(dow):
            names = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            dow = names[int(dow) % 7]
        else:
            dow = str(dow)

    return (
        f"{category} of ${amount:.2f} on {date} ({dow}), "
        f"{band} transaction"
    )


def embedding_backend() -> str:
    return os.getenv("EMBEDDING_BACKEND", "tei").strip().lower()


def _coerce_embedding_vector(seq: Any) -> list[float]:
    """Flatten TEI vectors; reject null slots (broken fp16 / truncated JSON)."""
    if not isinstance(seq, list):
        raise TypeError(f"Expected list of floats, got {type(seq).__name__}")
    out: list[float] = []
    for i, x in enumerate(seq):
        if x is None:
            raise ValueError(
                f"Embedding entry {i} is null — if you use TEI on T4, run the server "
                "with `--dtype float32` (see docker compose for embedding-server)."
            )
        try:
            out.append(float(x))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Non-numeric embedding at index {i}: {x!r}") from e
    return out


def _normalize_batch_response(payload: Any, n_inputs: int) -> list[list[float]]:
    """
    TEI ``POST /embed`` returns either a flat vector (single input) or a list of
    vectors (batched inputs). Optionally some gateways wrap payloads in JSON objects.
    """
    if isinstance(payload, dict):
        inner = payload.get("embeddings")
        if inner is None and n_inputs == 1:
            inner = payload.get("embedding")
        if isinstance(inner, list):
            payload = inner
        else:
            raise ValueError(f"Unexpected embed response keys: {list(payload.keys())}")

    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("TEI embed response is empty or not a JSON array")

    first = payload[0]
    if isinstance(first, (int, float)) or first is None:
        if n_inputs != 1:
            raise ValueError("Got single embedding vector for batched inputs")
        return [_coerce_embedding_vector(payload)]

    if len(payload) != n_inputs:
        raise ValueError(f"Expected {n_inputs} embedding rows, got {len(payload)}")
    return [_coerce_embedding_vector(row) for row in payload]


class TEIEmbedder:
    """
    Transaction text → vectors via TEI ``POST {{base_url}}/embed`` (e.g. BAAI/bge-m3).

    Defaults match a local Docker map ``8001:80`` hosting one model so no model id
    is sent on each request. Point ``TEI_BASE_URL`` / ``EMBEDDING_SERVER_URL`` at that host.

    **Batch size:** Many TEI builds cap ``/embed`` batch size (often **32** on consumer GPUs).
    Set ``TEI_EMBED_BATCH_SIZE`` or ``embed_batch_size=`` to match. If an oversized batch still
    reaches TEI (e.g. Jupyter kept an old module in memory), a **422 batch-limit** response is
    handled by **splitting the batch in half** until requests fit.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float = 600.0,
        embed_batch_size: int | None = None,
    ) -> None:
        env_url = (
            os.getenv("EMBEDDING_SERVER_URL")
            or os.getenv("TEI_BASE_URL")
            or "http://localhost:8001"
        )
        self.base_url = (base_url or env_url).rstrip("/")
        self.timeout_s = timeout_s
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())
        # Fingerprint caches when URL or advertised model identity changes.
        self._backend_id = "|".join(
            (
                self.base_url,
                os.getenv("EMBED_TEI_MODEL_ID", "BAAI/bge-m3"),
                "tei",
            )
        )
        if embed_batch_size is not None:
            self.embed_batch_size = max(1, int(embed_batch_size))
        else:
            self.embed_batch_size = max(1, int(os.getenv("TEI_EMBED_BATCH_SIZE", "32")))

    @staticmethod
    def _parse_tei_server_max_batch(detail: str) -> int | None:
        m = re.search(r"maximum allowed batch size\s*(\d+)", detail, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _embed_raw_batch(self, inputs: list[str]) -> list[list[float]]:
        """Call TEI ``/embed``; split recursively if TEI rejects the batch size (HTTP 422)."""
        if not inputs:
            return []
        url = f"{self.base_url}/embed"
        body = json.dumps({"inputs": inputs}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            low = detail.lower()
            if (
                e.code == 422
                and "maximum allowed batch size" in low
                and len(inputs) > 1
            ):
                cap = self._parse_tei_server_max_batch(detail)
                if cap is not None:
                    self.embed_batch_size = min(self.embed_batch_size, cap)
                mid = len(inputs) // 2 or 1
                self.logger.info(
                    "TEI rejected batch size n=%s; splitting into sizes %s and %s "
                    "(client embed_batch_size now %s)",
                    len(inputs),
                    mid,
                    len(inputs) - mid,
                    self.embed_batch_size,
                )
                return (
                    self._embed_raw_batch(inputs[:mid])
                    + self._embed_raw_batch(inputs[mid:])
                )

            self.logger.exception("HTTPError TEI embed: %s %s", e.code, detail)
            msg = f"TEI embed failed ({e.code}): {detail}"
            if e.code == 422 and "batch size" in low:
                msg += (
                    "\nReduce client batches: set TEI_EMBED_BATCH_SIZE (e.g. 32 or 16), "
                    f"currently embed_batch_size={self.embed_batch_size}, "
                    "or TEIEmbedder(embed_batch_size=...). "
                    "Or restart Jupyter to pick up embed_generator fixes."
                )
            raise RuntimeError(msg) from e
        except urllib.error.URLError as e:
            self.logger.exception("URLError: is TEI running at %s?", self.base_url)
            raise RuntimeError(
                f"TEI unreachable at {self.base_url}: {e}\n"
                "Start embedding-server (e.g. ghcr.io/huggingface/text-embeddings-inference "
                "mapped to host port 8001)."
            ) from e

        elapsed = time.perf_counter() - t0
        vectors = _normalize_batch_response(payload, len(inputs))
        dim = len(vectors[0]) if vectors else 0
        self.logger.debug(
            "embed batch: %.3fs n=%s dim=%s", elapsed, len(inputs), dim
        )
        return vectors

    def transaction_to_text(self, row: dict[str, Any]) -> str:
        return transaction_to_text(row)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string (one round-trip)."""
        return self._embed_raw_batch([text])[0]

    def _batch_cache_path(self, texts: list[str], cache_dir: Path) -> Path:
        h = hashlib.sha256()
        h.update(self._backend_id.encode("utf-8"))
        h.update(b"\0")
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\0")
        return cache_dir / f"{h.hexdigest()}.npy"

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        cache_dir: Path | None = None,
    ) -> np.ndarray:
        """Embed many strings with TEI batching; optional per-text-list cache."""
        bs = self.embed_batch_size if batch_size is None else max(1, batch_size)
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = self._batch_cache_path(texts, cache_dir)
        if cache_file.is_file():
            self.logger.info("Loading cached embeddings: %s", cache_file)
            return np.load(cache_file)

        vectors: list[list[float]] = []
        for i in tqdm(
            range(0, len(texts), bs),
            desc=f"Embedding batches ({bs}x TEI)",
        ):
            chunk = texts[i : i + bs]
            vectors.extend(self._embed_raw_batch(chunk))

        arr = np.asarray(vectors, dtype=np.float32)
        np.save(cache_file, arr)
        self.logger.info("Saved batch cache: %s shape=%s", cache_file, arr.shape)
        return arr

    def embed_transactions(
        self,
        df: pd.DataFrame,
        cache_dir: Path | None = None,
    ) -> np.ndarray:
        """Embed each row; cache matrix under a model-scoped parquet name."""
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        final_path = cache_dir / "transaction_embeddings.npy"

        texts = [self.transaction_to_text(r) for r in df.to_dict("records")]
        arr = self.embed_batch(texts, cache_dir=cache_dir)
        np.save(final_path, arr)
        self.logger.info("Wrote %s shape=%s", final_path, arr.shape)
        return arr


class OllamaEmbedder:
    """Transaction text → vectors via Ollama ``POST /api/embed`` (e.g. nomic-embed-text)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 600.0,
        embed_batch_size: int | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.timeout_s = timeout_s
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())
        self._backend_id = "|".join((self.base_url, self.model, "ollama"))
        if embed_batch_size is not None:
            self.embed_batch_size = max(1, int(embed_batch_size))
        else:
            self.embed_batch_size = max(
                1, int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", "32"))
            )

    def _embed_raw_batch(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        url = f"{self.base_url}/api/embed"
        payload_input: str | list[str] = inputs[0] if len(inputs) == 1 else inputs
        body = json.dumps({"model": self.model, "input": payload_input}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama embed failed ({e.code}): {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama unreachable at {self.base_url}: {e}\n"
                f"Start Ollama and run: ollama pull {self.model}"
            ) from e

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) == 0:
            raise ValueError(f"Unexpected Ollama embed response: {list(payload.keys())}")
        if len(embeddings) == 1 and len(inputs) == 1:
            row = embeddings[0]
            return [_coerce_embedding_vector(row)]
        if len(embeddings) != len(inputs):
            raise ValueError(
                f"Expected {len(inputs)} embedding rows, got {len(embeddings)}"
            )
        return [_coerce_embedding_vector(row) for row in embeddings]

    def transaction_to_text(self, row: dict[str, Any]) -> str:
        return transaction_to_text(row)

    def embed_text(self, text: str) -> list[float]:
        return self._embed_raw_batch([text])[0]

    def _batch_cache_path(self, texts: list[str], cache_dir: Path) -> Path:
        h = hashlib.sha256()
        h.update(self._backend_id.encode("utf-8"))
        h.update(b"\0")
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\0")
        return cache_dir / f"{h.hexdigest()}.npy"

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        cache_dir: Path | None = None,
    ) -> np.ndarray:
        bs = self.embed_batch_size if batch_size is None else max(1, batch_size)
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = self._batch_cache_path(texts, cache_dir)
        if cache_file.is_file():
            self.logger.info("Loading cached embeddings: %s", cache_file)
            return np.load(cache_file)

        vectors: list[list[float]] = []
        for i in tqdm(
            range(0, len(texts), bs),
            desc=f"Embedding batches ({bs}x Ollama)",
        ):
            chunk = texts[i : i + bs]
            vectors.extend(self._embed_raw_batch(chunk))

        arr = np.asarray(vectors, dtype=np.float32)
        np.save(cache_file, arr)
        self.logger.info("Saved batch cache: %s shape=%s", cache_file, arr.shape)
        return arr

    def embed_transactions(
        self,
        df: pd.DataFrame,
        cache_dir: Path | None = None,
    ) -> np.ndarray:
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        final_path = cache_dir / "transaction_embeddings.npy"

        texts = [self.transaction_to_text(r) for r in df.to_dict("records")]
        arr = self.embed_batch(texts, cache_dir=cache_dir)
        np.save(final_path, arr)
        self.logger.info("Wrote %s shape=%s", final_path, arr.shape)
        return arr


def create_embedder(**kwargs: Any) -> TEIEmbedder | OllamaEmbedder:
    """Return TEIEmbedder or OllamaEmbedder based on ``EMBEDDING_BACKEND`` (default: tei)."""
    backend = embedding_backend()
    if backend in ("tei", "hf", "huggingface"):
        return TEIEmbedder(**kwargs)
    return OllamaEmbedder(**kwargs)
