"""NLP and similarity metrics for summarization experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics.pairwise import cosine_similarity


class MetricsCalculator:
    """ROUGE, BERTScore, BLEU, cosine similarity """

    _bert_models_loaded = False

    def __init__(self) -> None:
        self._rouge = RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )

    def compute_rouge(self, hypothesis: str, reference: str) -> dict[str, float]:
        if not hypothesis.strip() or not reference.strip():
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
        # Library convention: target/reference first, prediction second
        scores = self._rouge.score(reference, hypothesis)
        return {
            "rouge1_f": float(scores["rouge1"].fmeasure),
            "rouge2_f": float(scores["rouge2"].fmeasure),
            "rougeL_f": float(scores["rougeL"].fmeasure),
        }

    def compute_bert_score(self, hypothesis: str, reference: str) -> float:
        """Return corpus-level F1 (single pair treated as batch of 1)."""
        import bert_score  # noqa: PLC0415 — heavy deps

        if not hypothesis.strip() or not reference.strip():
            return 0.0
        P, R, F1 = bert_score.score(
            [hypothesis],
            [reference],
            lang="en",
            verbose=False,
            rescale_with_baseline=False,
        )
        return float(F1.mean().item())

    def compute_cosine_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        a = np.asarray(emb1, dtype=np.float64).reshape(1, -1)
        b = np.asarray(emb2, dtype=np.float64).reshape(1, -1)
        sim = cosine_similarity(a, b)[0, 0]
        return float(sim)

    def compute_bleu(self, hypothesis: str, reference: str) -> float:
        """Sentence BLEU with smoothing (handles short texts; whitespace tokenization)."""
        try:
            from nltk.translate.bleu_score import (  # noqa: PLC0415
                SmoothingFunction,
                sentence_bleu,
            )

            ref_toks = reference.lower().split()
            hyp_toks = hypothesis.lower().split()
            if not ref_toks or not hyp_toks:
                return 0.0
            chencherry = SmoothingFunction()
            return float(
                sentence_bleu(
                    [ref_toks],
                    hyp_toks,
                    smoothing_function=chencherry.method1,
                )
            )
        except Exception:
            return 0.0

    def compute_all(
        self,
        hypothesis: str,
        reference: str,
        emb1: list[float] | None = None,
        emb2: list[float] | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out.update(self.compute_rouge(hypothesis, reference))
        try:
            out["bertscore_f1"] = self.compute_bert_score(hypothesis, reference)
        except Exception:
            out["bertscore_f1"] = float("nan")
        out["bleu"] = self.compute_bleu(hypothesis, reference)
        if emb1 is not None and emb2 is not None and len(emb1) and len(emb2):
            out["cosine_similarity"] = self.compute_cosine_similarity(emb1, emb2)
        else:
            out["cosine_similarity"] = None
        return out
