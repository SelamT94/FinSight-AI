# FinSight-AI: Financial Statement Summarization and Insight Generation with Foundation Models

**MSc Foundation Models – Semester Project**

FinSight-AI is an end-to-end financial analytics system that transforms raw transaction data into structured monthly statements, semantic transaction clusters, AI-generated summaries, behavioral insights, and comparative model evaluations.

The project combines modern NLP and LLM technologies including:

* **Mistral / LLaMA models** served through an OpenAI-compatible API (vLLM)
* **BGE-M3 embeddings** served via Text Embeddings Inference (TEI)
* **Clustering and pattern detection** for transaction behavior analysis
* **Automated evaluation framework** using ROUGE, BERTScore, BLEU, and resource metrics
* **Interactive web application** built with FastAPI and Next.js

The system is designed as a complete financial intelligence pipeline suitable for demonstrating practical applications of foundation models in the financial domain.

---

# Features

### Transaction Processing

* Load and clean PaySim transaction data
* Generate realistic monthly account statement objects
* Detect anomalies and fraudulent activity
* Create category-level spending summaries

### Semantic Analysis

* Generate transaction embeddings using BGE-M3
* Cluster transactions based on semantic similarity
* Detect recurring patterns and behavioral trends
* Enable similarity search using vector representations

### AI-Powered Insights

* Monthly statement summarization
* Spending behavior analysis
* Financial insight generation
* Prompt engineering experiments:

  * Zero-shot
  * Few-shot
  * Chain-of-Thought (CoT)

### Evaluation Framework

* Automated experiment runner
* ROUGE evaluation
* BERTScore evaluation
* BLEU evaluation
* Resource usage monitoring
* Throughput and latency measurements

### Interactive Dashboard

* Financial overview dashboard
* Summary generation interface
* Model comparison tools
* Cluster exploration
* Experiment result visualization

---

# System Architecture

```text
PaySim Dataset
(paysim1.csv)
        │
        ▼
┌───────────────────────┐
│  Preprocessing        │
│  Monthly Statements   │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Transactions Parquet  │
│ Monthly JSON Objects  │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ BGE-M3 Embeddings     │
│ (TEI Server)          │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Clustering & Patterns │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ LLM Summarization     │
│ Insights Generation   │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Experiments 1–5       │
│ Evaluation Pipeline   │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ FastAPI + Next.js UI  │
└───────────────────────┘
```

---

# Foundation Models Used

| Component        | Model                                 |
| ---------------- | ------------------------------------- |
| Chat Models      | Mistral 7B / LLaMA (via vLLM)         |
| Embeddings       | BAAI/bge-m3                           |
| Embedding Server | HuggingFace Text Embeddings Inference |
| LLM API          | OpenAI-Compatible API                 |

Environment variables allow model swapping without changing source code:

```env
OPENAI_COMPAT_BASE_URL=
LLAMA_MODEL=
MISTRAL_MODEL=
EMBEDDING_SERVER_URL=
```

---

# Dataset

This project uses the **PaySim** synthetic financial transaction dataset.

Expected location:

```text
data/raw/paysim1.csv
```

If your downloaded file has a different name:

```bash
mv data/raw/PS_20174392719_1491204439457_log.csv \
   data/raw/paysim1.csv
```

---

# Prerequisites

## Software

* Python 3.11+
* Docker
* Node.js (LTS)
* npm

## Hardware

Recommended:

* NVIDIA GPU for vLLM and TEI
* 8 GB RAM minimum
* 16 GB+ RAM recommended for larger runs

---

# Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd FinSight-AI
```

## 2. Create Virtual Environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Project Setup

```bash
python3 scripts/setup_project.py
```

This script:

* Creates required directories
* Seeds `.env` from `.env.example`
* Verifies OpenAI-compatible API access
* Verifies embedding server connectivity

---

# Infrastructure Setup

## Start vLLM

```bash
docker compose --profile infra up -d vllm-chat
```

The service exposes:

```text
http://localhost:5000/v1
```

## Start Text Embeddings Inference

Recommended configuration:

```bash
docker run \
  --gpus all \
  -p 8001:80 \
  ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
  --model-id BAAI/bge-m3 \
  --dtype float32
```

Set:

```env
EMBEDDING_SERVER_URL=http://localhost:8001
```

For stability:

```env
TEI_EMBED_BATCH_SIZE=32
```

---

# Workflow Overview

The project is organized around five primary notebooks.

| Notebook                     | Purpose                                |
| ---------------------------- | -------------------------------------- |
| 01_preprocessing.ipynb       | Data cleaning and statement generation |
| 02_embeddings.ipynb          | Embeddings, clustering, patterns       |
| 03_llm_summarization.ipynb   | Summaries and insights                 |
| 04_experiment_analysis.ipynb | Evaluation and analysis                |
| 05_nextjs_frontend.ipynb     | Frontend setup and testing             |

---

# Section 1: Preprocessing

Open:

```text
notebooks/01_preprocessing.ipynb
```

Set:

```python
MAX_ROWS = 500_000
```

for quick experimentation.

Or:

```python
MAX_ROWS = None
```

for the full dataset.

### Outputs

| Artifact             | Path                                      |
| -------------------- | ----------------------------------------- |
| Cleaned transactions | data/processed/transactions_clean.parquet |
| Monthly statements   | data/processed/monthly_statements.json    |

### Processing Steps

1. Load PaySim data
2. Keep relevant columns
3. Generate timestamps
4. Map transaction categories
5. Detect anomalies
6. Create monthly statements

---

# Section 2: Embeddings & Clustering

Open:

```text
notebooks/02_embeddings.ipynb
```

### Outputs

| Artifact               | Path                                          |
| ---------------------- | --------------------------------------------- |
| Clustered transactions | data/processed/transactions_clustered.parquet |
| Cluster labels         | data/processed/cluster_labels.json            |
| Behavioral patterns    | data/processed/patterns.json                  |
| Elbow plot             | visualizations/elbow_curve.png                |

### Components

* TEIEmbedder
* Clusterer
* PatternDetector

---

# Section 3: LLM Summarization

Open:

```text
notebooks/03_llm_summarization.ipynb
```

### Supported Prompting Strategies

* Zero-Shot
* Few-Shot
* Chain-of-Thought

### Generated Outputs

* Monthly summaries
* Financial insights
* Behavioral analysis
* Comparative evaluations

---

# Section 4: Experiments & Evaluation

Run:

```bash
python3 scripts/run_experiments.py
```

or analyze through:

```text
notebooks/04_experiment_analysis.ipynb
```

## Experiment 1

Summary quality evaluation.

Metrics:

* ROUGE
* BLEU
* BERTScore

## Experiment 2

Insight quality assessment.

## Experiment 3

Prompt strategy comparison.

* Zero-shot
* Few-shot
* Chain-of-Thought

## Experiment 4

LLaMA vs Mistral comparison.

## Experiment 5

Resource efficiency analysis.

Metrics:

* RAM usage
* Throughput
* Tokens/sec
* Inference latency

### Outputs

```text
evaluation/results/
```

Includes:

```text
experiment_1.csv
experiment_2.csv
experiment_3.csv
experiment_4.csv
experiment_5.csv
full_report.json
```

---

# Section 5: Web Application

## Backend

```bash
python3 scripts/run_api.py
```

Default:

```text
http://localhost:8000
```

## Frontend

```bash
cd frontend

cp .env.local.example .env.local

npm install

npm run dev
```

Open:

```text
http://localhost:3000
```

---

# Web Application Pages

| Route        | Description            |
| ------------ | ---------------------- |
| /            | Dashboard              |
| /summaries   | AI-generated summaries |
| /comparison  | Model comparison       |
| /clusters    | Transaction clusters   |
| /experiments | Evaluation results     |

---

# Project Structure

```text
FinSight-AI/
│
├── backend/
├── frontend/
├── notebooks/
├── preprocessing/
├── embeddings/
├── models/
├── prompts/
├── evaluation/
├── visualizations/
├── scripts/
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
├── README.md
└── .env.example
```

---

# Key Outputs

## Processed Data

```text
transactions_clean.parquet
monthly_statements.json
```

## Embeddings & Clusters

```text
transactions_clustered.parquet
cluster_labels.json
patterns.json
```

## Evaluation

```text
full_report.json
experiment_*.csv
```

## Visualizations

```text
elbow_curve.png
rouge_comparison.png
resource_efficiency.png
prompt_strategy_comparison.png
```

---

# Known Limitations

* PaySim is a synthetic dataset rather than real banking data.
* Summary evaluation relies on pseudo-references.
* Mistral-as-judge may introduce evaluation bias.
* Generated financial insights should not be considered professional financial advice.

---

# Future Work

* Retrieval-Augmented Generation (RAG)
* Real banking statement ingestion
* Multi-modal financial analysis
* Advanced anomaly detection
* Fine-tuned financial foundation models
* Human evaluation studies

---

# Authors

**FinSight-AI**

MSc Foundation Models Semester Project

Financial Statement Summarization and Insight Generation using Foundation Models.
