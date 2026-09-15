# Dual Stopping Iterative RAG on HotpotQA

A complete research implementation of **Dual Stopping Iterative RAG** using **Qwen 3B** on the HotpotQA multi-hop question answering dataset (distractor setting).

## Overview

This system implements iterative multi-hop retrieval-augmented generation with two independent early stopping mechanisms:

1. **Reasoning Early Stopping**: Stops generation within a single hop when the answer converges across consecutive output chunks
2. **Retrieval Early Stopping**: Stops retrieving additional hops when marginal information gain is low

## Features

- ✅ Iterative multi-hop RAG with up to H hops (configurable)
- ✅ Four baseline configurations for ablation studies
- ✅ Open-weight LLM: Qwen 3B (no paid APIs)
- ✅ Simulated retrieval using sentence embeddings (the current ranker uses NumPy cosine scoring)
- ✅ Comprehensive evaluation (EM/F1, avg hops, avg tokens, cost metrics)
- ✅ Detailed JSONL trace logs for analysis
- ✅ Embedding cache for fast experimentation
- ✅ GPU/CPU support with automatic device selection

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- ~6GB VRAM for Qwen 3B inference (or run on CPU)

### Setup

```bash
# Clone or navigate to the repository
cd dual-stoppingRAG

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download Data

```bash
python scripts/download_data.py
```

This downloads HotpotQA distractor dev set to `data/hotpotqa/`.

### 2. (Optional) Pre-build Embedding Cache

```bash
python scripts/build_index.py --max_examples 500
```

This pre-computes embeddings for faster experimentation.

### 3. Run Experiments

#### Baseline 1: Fixed Hops (No Stopping)

```bash
python scripts/run_experiment.py \
  --method fixed_hops \
  --max_examples 200 \
  --hops 3 \
  --top_k 5 \
  --seed 42
```

#### Baseline 2: Fixed Hops + Reasoning Stop

```bash
python scripts/run_experiment.py \
  --method fixed_hops_reasoning_stop \
  --max_examples 200 \
  --hops 3 \
  --top_k 5 \
  --chunk_tokens 64 \
  --convergence_k 2 \
  --seed 42
```

#### Baseline 3: Retrieval Stop Only

```bash
python scripts/run_experiment.py \
  --method retrieval_stop_only \
  --max_examples 200 \
  --hops 4 \
  --top_k 5 \
  --min_hops 2 \
  --overlap_threshold 0.7 \
  --conf_threshold 0.6 \
  --seed 42
```

#### Baseline 4: Dual Stopping (Both Mechanisms)

```bash
python scripts/run_experiment.py \
  --method dual_stopping \
  --max_examples 200 \
  --hops 4 \
  --top_k 5 \
  --chunk_tokens 64 \
  --convergence_k 2 \
  --min_hops 2 \
  --overlap_threshold 0.7 \
  --conf_threshold 0.6 \
  --seed 42
```

### 4. Evaluate Results

```bash
# Compute EM/F1 and aggregate statistics
python scripts/evaluate_runs.py --run_id <run_id>

# Detailed analysis with distributions and CSV export
python scripts/analyze_logs.py --run_id <run_id>
```

Run IDs are automatically generated and printed when you run experiments. Check `outputs/<run_id>/` for all results.

## Directory Structure

```
dual-stoppingRAG/
├── README.md
├── requirements.txt
├── scripts/
│   ├── download_data.py          # Download HotpotQA dataset
│   ├── build_index.py             # Pre-build embedding cache
│   ├── run_experiment.py          # Main experiment runner
│   ├── evaluate_runs.py           # Compute EM/F1 metrics
│   └── analyze_logs.py            # Deep log analysis
├── src/
│   ├── config.py                  # Configuration dataclass
│   ├── data/
│   │   └── hotpot_loader.py       # HotpotQA data loading
│   ├── retrieval/
│   │   ├── embedder.py            # Sentence embedding model
│   │   ├── index.py               # FAISS index builder & cache
│   │   └── ranker.py              # Paragraph ranking
│   ├── llm/
│   │   ├── qwen.py                # Qwen 3B model wrapper
│   │   ├── prompts.py             # All prompt templates
│   │   ├── generation.py          # Text generation utilities
│   │   └── early_stop.py          # Answer convergence detector
│   ├── rag/
│   │   ├── iterative_rag.py       # Main RAG engine
│   │   ├── stopping.py            # Retrieval stopping logic
│   │   └── baselines.py           # Four baseline configs
│   ├── eval/
│   │   ├── hotpot_eval.py         # HotpotQA EM/F1 evaluation
│   │   └── metrics.py             # Metrics aggregation
│   └── utils/
│       ├── logging.py             # Logging utilities
│       └── text.py                # Text normalization & utilities
├── data/                          # (created by download script)
│   └── hotpotqa/
├── runs/                          # (created automatically)
│   └── cache_embeddings/
└── outputs/                       # (created automatically)
    └── <run_id>/
        ├── config.json            # Full run configuration
        ├── predictions.json       # Final predictions
        ├── traces.jsonl           # Detailed hop-by-hop logs
        ├── metrics.json           # EM/F1 and aggregate stats
        └── summary.csv            # Per-example analysis
```

## Methodology

### Iterative RAG Loop

For each question, the system performs up to H hops:

1. **Query Formation**: Use original question (hop 1) or LLM-generated follow-up query
2. **Retrieval**: Rank candidate paragraphs by cosine similarity, select top-K
3. **Generation**: LLM produces answer + follow-up query + confidence
4. **Stopping Check**: Apply reasoning and/or retrieval stopping criteria

### Stopping Mechanisms

#### 1. Reasoning Early Stopping (Answer Convergence)

- Generates output in chunks (e.g., 64 tokens)
- After each chunk, extracts current best answer
- Stops if answer is stable for k consecutive chunks (default k=2)
- **Benefit**: Reduces generation cost when answer is found early

#### 2. Retrieval Early Stopping

Stops retrieving new hops if ALL conditions hold:
- Hop >= minimum threshold (default 2)
- Answer stable across last 2 hops
- Low retrieval novelty (high Jaccard overlap with previous hop)
- Confidence >= threshold (default 0.6)

**Benefit**: Reduces retrieval cost when additional hops provide little new information

### Four Baselines

| Baseline | Reasoning Stop | Retrieval Stop | Description |
|----------|---------------|----------------|-------------|
| `fixed_hops` | ❌ | ❌ | Always H hops, full generation |
| `fixed_hops_reasoning_stop` | ✅ | ❌ | Early stop generation, always H hops |
| `retrieval_stop_only` | ❌ | ✅ | Early stop hops, full generation |
| `dual_stopping` | ✅ | ✅ | Both mechanisms enabled |

## Output Files

### predictions.json
```json
{
  "example_id_1": "predicted answer",
  "example_id_2": "predicted answer"
}
```

### traces.jsonl
One JSON object per line, per example:
```json
{
  "id": "example_id",
  "question": "...",
  "gold_answer": "...",
  "method": "dual_stopping",
  "hops_taken": 3,
  "total_tokens": 245,
  "final_answer": "...",
  "hops": [
    {
      "hop": 1,
      "query": "...",
      "retrieved_paragraphs": [...],
      "intermediate_answer": "...",
      "confidence": 0.7,
      "tokens_generated": 120,
      "reasoning_stopped": true,
      "extracted_answers": ["ans1", "ans2", "ans2"]
    }
  ],
  "retrieval_stopped": true,
  "retrieval_stop_reason": "answer_stable_and_low_novelty"
}
```

### metrics.json
```json
{
  "em": 0.35,
  "f1": 0.42,
  "avg_hops": 2.3,
  "avg_tokens": 180.5,
  "reasoning_early_stop_rate": 0.68,
  "retrieval_early_stop_rate": 0.45,
  "avg_cost": 2480.5
}
```

## Configuration Options

See `python scripts/run_experiment.py --help` for all options. Key parameters:

- `--max_examples N`: Run on first N examples (for quick testing)
- `--hops H`: Maximum hops (default 4)
- `--top_k K`: Paragraphs to retrieve per hop (default 5)
- `--chunk_tokens`: Tokens per chunk for reasoning stop (default 64)
- `--convergence_k`: Consecutive stable chunks needed (default 2)
- `--min_hops`: Minimum hops before retrieval stop (default 2)
- `--overlap_threshold`: Jaccard threshold for low novelty (default 0.7)
- `--conf_threshold`: Confidence threshold for retrieval stop (default 0.6)

## Reproducibility

All runs use fixed seeds and save full configuration to `config.json`. To reproduce:

```bash
python scripts/run_experiment.py \
  --method dual_stopping \
  --max_examples 500 \
  --seed 42 \
  --hops 4 \
  --top_k 5
```

## Performance claims

Results depend on the model, dataset subset, prompt, and stopping thresholds. The
repository does not ship a validated benchmark, so accuracy and token-reduction
claims must be measured from matched runs on the same examples. A one-example GPU
smoke test is only a wiring check, not a performance result.

## Citation

If you use this codebase, please cite:

```bibtex
@misc{dual-stopping-rag-2026,
  title={Dual Stopping Iterative RAG on HotpotQA},
  author={Research Implementation},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.
