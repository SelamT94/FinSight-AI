# Presentation slide copy — AI account statement summarization (FinSight-AI)

Use this document as the **exact slide text** for PowerPoint, Google Slides, or similar.  
For any metric, table, or chart: **paste real outputs** from your run (paths below). **Do not invent numbers.**

**Primary result sources (after `python scripts/run_all.py`):**

- `evaluation/results/experiment_1.csv` … `experiment_5.csv`
- `evaluation/results/full_report.json`
- `visualizations/rouge_comparison.png`, `prompt_strategy_comparison.png`, `resource_efficiency.png`, `elbow_curve.png`, `spending_*.png`
- Console / notebook tables from `notebooks/04_experiment_analysis.ipynb`

---

## Slide 1 — Title

**Title:** AI-Based Account Statement Summarization and Financial Insight Generation

**Subtitle:** MSc Foundation Models — Semester Project · FinSight-AI

**Content (footer / your name):**  
Your name, institution, date.

---

## Slide 2 — Problem & motivation

**Title:** Problem and motivation

**Content:**

- Banks and users face **long, raw transaction feeds** that are hard to interpret quickly.
- Goal: turn PaySim-style transaction data into **short narratives**, **insights**, and **comparative model evidence**.
- Focus: **local foundation models** (LLaMA 3, Mistral) via Ollama, plus **embeddings** for structure.

---

## Slide 3 — Dataset

**Title:** Dataset — PaySim

**Content:**

- **PaySim** synthetic mobile-money transactions (fraud labels, amounts, types).
- Pipeline: clean → monthly **statement objects** → optional clustering / patterns.
- **Citation / link:** paste the reference you use in the report (e.g. Kaggle / paper URL).

**Result to paste here (optional):**  
`[PASTE: row counts and date range from your preprocessing log or notebook — e.g. output of scripts/run_preprocessing.py or 01_preprocessing.ipynb. Do not guess.]`

---

## Slide 4 — System architecture

**Title:** End-to-end architecture

**Content:**

- **Ingest & clean:** CSV → parquet + `monthly_statements.json`.
- **Embeddings:** **BAAI/bge-m3** via **Text Embeddings Inference** (Docker) → clusters, FAISS index, elbow plot.
- **LLMs:** LLaMA 3 8B & Mistral 7B — summaries, insights, judge-style evaluation.
- **Eval:** five experiments → CSVs, plots, `full_report.json`.
- **UI:** Next.js frontend + FastAPI backend (`scripts/run_api.py`, `frontend/`).

**Diagram:**  
`[PASTE: your architecture figure, or copy the ASCII diagram from README.md System Architecture.]`

---

## Slide 5 — Foundation models & stack

**Title:** Models and tooling

**Content:**

| Model / component | Role | Notes |
|-------------------|------|--------|
| `llama3:8b` (configurable) | Summaries, insights, multi-strategy prompts | Via Ollama |
| `mistral:7b` | Summaries + comparative / judge outputs | Via Ollama |
| `BAAI/bge-m3` | Transaction embeddings | TEI container; **`float32`** recommended on T4 |
| Metrics | ROUGE, BERTScore, BLEU, resource logging | See `evaluation/metrics.py` |

**Result to paste here:**  
[PASTE: chat model names from **ollama list** and TEI model / URL from embedding-server if you show them on this slide.]

---

## Slide 6 — Preprocessing & monthly statements

**Title:** From raw transactions to monthly statements

**Content:**

- Cleaning, categorization, anomaly flags, per-month rollups.
- Output: `data/processed/transactions_clean.parquet`, `monthly_statements.json`.

**Result to paste here:**  
`[PASTE: one example month key + transaction_count / net_flow from monthly_statements.json or preprocessing summary printout. Do not guess.]`

---

## Slide 7 — Embeddings & clustering

**Title:** Semantic clustering

**Content:**

- Sample of transactions embedded; **k** chosen via elbow method.
- Outputs: `transactions_clustered.parquet`, `cluster_labels.json`, `visualizations/elbow_curve.png`.

**Result to paste here:**  
`[PASTE: optimal_k and sample_size from cluster_labels.json or run_embeddings log. Insert elbow_curve.png as the slide image.]`

---

## Slide 8 — Prompting strategies

**Title:** Prompt strategies for summarization

**Content:**

- **Zero-shot, few-shot, chain-of-thought** templates for financial statements.
- Same statement object fed to LLaMA / Mistral for fair comparison where applicable.

*(No numeric results on this slide unless you quote example snippet text from a real run.)*

---

## Slide 9 — Experiment 1 — ROUGE / overlap vs reference

**Title:** Experiment 1 — summary quality (ROUGE family)

**Content:**

- Random months; Llama & Mistral zero-shot; reference strategy per `experiment_runner` design.
- Metrics in `experiment_1.csv`.

**Comparison — result to paste here:**  
`[PASTE: small table or bullet metrics from experiment_1.csv and/or rouge_comparison.png. Do not invent.]`

---

## Slide 10 — Experiment 2 — Insights

**Title:** Experiment 2 — financial insights

**Content:**

- Insight generation and numeric grounding checks; judge ratings where applicable.

**Result to paste here:**  
`[PASTE: summary statistics or one row from experiment_2.csv — actual columns from your file.]`

---

## Slide 11 — Experiment 3 — Prompt strategy comparison

**Title:** Experiment 3 — zero vs few-shot vs CoT (Llama)

**Content:**

- Same month; strategies compared; metrics in `experiment_3.csv`.

**Comparison — result to paste here:**  
`[PASTE: key numbers from experiment_3.csv and/or insert prompt_strategy_comparison.png.]`

---

## Slide 12 — Experiment 4 — LLaMA vs Mistral (headline comparison)

**Title:** Experiment 4 — LLaMA vs Mistral

**Content:**

- Side-by-side summaries and metric columns (ROUGE-style, latency, throughput) as implemented.

**Comparison — result to paste here:**  
`[PASTE: markdown table from run_experiments.py output (“Experiment 4 comparison”) or from experiment_4.csv. Do not guess.]`

---

## Slide 13 — Experiment 5 — Resource efficiency

**Title:** Experiment 5 — time, tokens/s, memory

**Content:**

- Aggregated resource stats from experiments 1–4 runs.

**Comparison — result to paste here:**  
`[PASTE: experiment_5.csv contents or console block from run_experiments.py; insert resource_efficiency.png.]`

---

## Slide 14 — Web demo

**Title:** Demonstration UI

**Content:**

- Dashboard, summaries, model comparison, clusters, experiment tabs.
- **Run:** API `python scripts/run_api.py`; frontend `cd frontend && npm run dev`.

**Result to paste here:**  
`[PASTE: 1–2 screenshots of localhost:3000 pages after a real run.]`

---

## Slide 15 — Limitations & ethics

**Title:** Limitations

**Content:**

- Synthetic data; **pseudo-references** for automatic metrics; judge bias.
- Local inference constraints (RAM, runtime).

---

## Slide 16 — Conclusion

**Title:** Conclusion

**Content:**

- **Takeaway 1:** `[PASTE: one sentence grounded in your actual experiment_4 / experiment_5 findings.]`
- **Takeaway 2:** `[PASTE: optional second sentence — must match your results, not generic claims.]`
- **Future work:** longer context, real bank formats, stronger evaluation.

---

## Slide 17 — References & appendix pointer

**Title:** References

**Content:**

- PaySim, Ollama, model cards, libraries (BERTScore, FAISS, etc.).
- **Appendix:** full tables → `evaluation/results/`, figures → `visualizations/`.
