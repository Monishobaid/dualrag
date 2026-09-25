# Dual Stopping Iterative RAG on HotpotQA

A research implementation of iterative retrieval-augmented generation on the
HotpotQA distractor dataset. The September 24 study uses
`Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-14B-Instruct`. The historical
`run_experiment.py` runner still defaults to 3B unless a model is supplied;
the new `research_experiment.py` runner defaults to 7B.

## Current research results

The September 24 experiments evaluate Qwen2.5 **7B and 14B Instruct**, a short
cited-answer reader, stronger embeddings, and a separate prompted sufficiency
critic. They use the original exploratory50 and a disjoint frozen100 cohort.
These are instruction models, not Qwen3 thinking models. See:

- [Measured results and paired uncertainty](docs/RESULTS_2026-09-24.md)
- [Research conclusions and next experiments](docs/RESEARCH_CONCLUSIONS_2026-09-24.md)
- [Paper-to-code mapping and frozen protocol](docs/RESEARCH_REVIEW_2026-09-24.md)
- [Manual error review](docs/ERROR_REVIEW_2026-09-24.md)
- [Critic diagnostics](docs/CRITIC_DIAGNOSTICS_2026-09-24.md)
- [Dataset slices](docs/DATASET_SLICES_2026-09-24.md) and [embedding input audit](docs/EMBEDDING_AUDIT_2026-09-24.md)
- [Relation-aware answer verification plan](docs/ANSWER_VERIFICATION_PLAN_2026-09-25.md) and isolated runner
- [Running research log](RESEARCH_LOG.md)

The historical pipeline below remains runnable for comparison. The newer
`scripts/research_experiment.py` entry point is described at the end of this
README. The new reader does not use within-answer reasoning stopping; lower
token counts from its shorter outputs must not be attributed to that mechanism.

## Historical pipeline overview

This system implements iterative multi-hop retrieval-augmented generation with
two optional early stopping mechanisms:

1. **Reasoning Early Stopping**: Stops generation within a single hop when the answer converges across consecutive output chunks
2. **Retrieval Early Stopping**: Stops retrieving additional hops when marginal information gain is low

## Features

- ✅ Iterative multi-hop RAG with up to H hops (configurable)
- ✅ Four stopping ablations plus an oracle-evidence diagnostic
- ✅ Open-weight Qwen generation (no paid API required)
- ✅ Dense sentence retrieval, lexical BM25, or normalized hybrid ranking
- ✅ Optional cross-encoder second-stage reranking over a configurable shortlist
- ✅ Explicit evidence-context budget (no hidden ten-paragraph truncation)
- ✅ Optional model-free evidence sufficiency critic
- ✅ Comprehensive evaluation (EM/F1, avg hops, avg tokens, cost metrics)
- ✅ Detailed JSONL trace logs for analysis
- ✅ Per-example normal-versus-oracle comparison report for bottleneck diagnosis
- ✅ Running paper notes and measurements in `RESEARCH_LOG.md`
- ✅ Embedding cache for fast experimentation
- ✅ GPU/CPU support with automatic device selection

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- GPU memory depends on the selected Qwen checkpoint; Renku GPU is recommended
- CPU execution is supported but substantially slower

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

### Retrieval improvements

Hybrid retrieval interpolates normalized dense and BM25 scores. A cross-encoder
can then rerank a larger first-stage shortlist (the default model below is a
small English MS MARCO reranker):

```bash
python scripts/run_experiment.py \
  --method fixed_hops \
  --retrieval_mode hybrid \
  --dense_weight 0.5 \
  --reranker_model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --reranker_candidate_k 20 \
  --max_evidence_paragraphs 15 \
  --max_examples 50 --seed 42
```

`--max_evidence_paragraphs` is optional. When omitted, all retrieved
paragraphs are passed to the reader; setting it makes the context budget
explicit and auditable.

The oracle-evidence mode supplies gold supporting paragraphs for diagnosis
only. After a matched normal and oracle run, generate the per-example report:

```bash
python scripts/compare_predictions.py \
  --normal_dir outputs/<normal_run> \
  --oracle_dir outputs/<oracle_run> \
  --output outputs/normal_vs_oracle.json
```

See `RESEARCH_LOG.md` for the frozen protocol, baseline table, and dated
experiment entries.

#### Grounded candidate reader

The default `--reader_mode legacy` keeps the original confidence-first answer
selection. The optional `grounded_candidates` reader asks Qwen for short
alternatives at each hop, then scores every candidate for exact evidence
support, coarse question type, and repetition across hops. It is an ablation,
not a learned verifier and does not use gold answers:

```bash
python scripts/run_experiment.py --method fixed_hops \
  --reader_mode grounded_candidates \
  --example_ids_file outputs/example_ids_dev_50_seed42.json \
  --model_name Qwen/Qwen2.5-7B-Instruct --seed 42 --hops 3 \
  --top_k 5 --max_new_tokens 192 --temperature 0 --top_p 1
```

The selected candidate and all verification scores are stored in each trace.
For a matched normal/oracle failure taxonomy, run:

```bash
python scripts/error_analysis.py \
  --normal_dir outputs/<normal_run> --oracle_dir outputs/<oracle_run> \
  --dataset_path data/hotpotqa/hotpot_dev_distractor_v1.json \
  --ids_file outputs/example_ids_dev_50_seed42.json \
  --output outputs/error_analysis.json
```

The categories are trace-based diagnostics: missed first-hop evidence,
wrong answer type, unsupported answer, parser issue, incomplete multi-hop
composition, or other generation error. See `RESEARCH_LOG.md` for the latest
matched reader result.

#### Hybrid retrieval and oracle diagnostic

The ranker supports dense embeddings (`dense`), lexical BM25 (`bm25`), and a
per-query normalized interpolation (`hybrid`). For example:

```bash
python scripts/run_experiment.py --method fixed_hops --retrieval_mode hybrid \
  --dense_weight 0.5 --max_examples 200 --seed 42
```

`oracle_evidence` is an evaluation-only baseline. It gives the generator the
gold HotpotQA supporting-fact paragraphs, measuring answer generation when
retrieval is perfect. Do not use it for deployment:

```bash
python scripts/run_experiment.py --method oracle_evidence --max_examples 200 --seed 42
```

Enable the model-free evidence critic to gate retrieval stopping and log its
per-hop sufficiency score:

```bash
python scripts/run_experiment.py --method dual_stopping \
  --enable_evidence_critic --evidence_critic_threshold 0.65 --max_examples 200 --seed 42
```

For Qwen models, reasoning stopping uses `TextIteratorStreamer` and a stopping
criterion so callbacks can interrupt decoding. Token counts are measured from
the streamed prefix. Azure runs remain API-driven and cannot interrupt a
provider-side completion. The four stopping modes are experimental ablations;
retrieval-only stopping is currently the simplest production-style baseline.

### Relation-aware verification experiment (new, isolated)

The next experiment is implemented in `scripts/run_verification_experiment.py`.
It keeps the legacy runner unchanged and compares a ranked all-context reader,
structured claims, deterministic relation checks, a prompted claim verifier,
one bounded evidence-only revision, and an oracle diagnostic on the same frozen
IDs. The structured schema and checks adapt the atomic-claim and grounded
revision ideas from FActScore, RARR, Re-Ex, RAC, and Provenance; the exact
boundaries are recorded in the [implementation plan](docs/ANSWER_VERIFICATION_PLAN_2026-09-25.md).

```bash
python scripts/run_verification_experiment.py \
  --ids research_results/2026-09-25/verification_ids_100_seed20260925.json \
  --output outputs/verification_2026-09-25 \
  --model Qwen/Qwen2.5-14B-Instruct \
  --arms static structured checked verified revised oracle \
  --seed 42 --max-new-tokens 192

python scripts/analyze_verification.py \
  --experiment outputs/verification_2026-09-25 \
  --base static --candidate revised \
  --output outputs/verification_2026-09-25/normal_vs_revised.json
```

This first implementation uses the prompted verifier as a comparison arm; it
does not claim that verifier is independent or calibrated. The independent NLI
adapter and train-only calibration are later phases in the plan. Every arm
writes raw calls, structured fields, evidence IDs, checks, revision decisions,
costs, and failed IDs to its trace and metrics files.

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
│   │   ├── index.py               # Embedding cache and NumPy index helpers
│   │   └── ranker.py              # Dense, BM25, and hybrid paragraph ranking
│   ├── llm/
│   │   ├── qwen.py                # Configurable Qwen model wrapper
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

### Current approach

The current system is a controlled HotpotQA experiment rather than an
open-corpus production retriever. Each example supplies a question and a
small candidate paragraph pool containing supporting paragraphs and
distractors.

For each hop, the engine:

1. Encodes the current query and ranks candidate paragraphs with dense cosine
   similarity, BM25, or normalized dense/BM25 interpolation.
2. Adds the selected paragraphs to cumulative evidence while avoiding repeated
   titles when enough candidates remain.
3. Prompts Qwen for a short answer, an evidence flag, supporting quotes,
   confidence, and an optional focused follow-up query.
4. Applies the grounding guard and, when enabled, the lexical evidence critic.
5. Selects the highest-confidence non-empty answer as the final answer and
   records all hop-level decisions in `traces.jsonl`.

### Iterative RAG Loop

For each question, the system performs up to H hops:

1. **Query Formation**: Use original question (hop 1) or LLM-generated follow-up query
2. **Retrieval**: Rank candidate paragraphs with the configured dense/BM25/hybrid
   ranker and select top-K
3. **Generation**: LLM produces answer + follow-up query + confidence
4. **Stopping Check**: Apply reasoning and/or retrieval stopping criteria

### Stopping Mechanisms

#### 1. Reasoning Early Stopping (Answer Convergence)

- Generates output in chunks (e.g., 64 tokens)
- After each chunk, extracts current best answer
- Stops if answer is stable for k consecutive chunks (default k=2)
- **Benefit**: Reduces generation cost when answer is found early

#### 2. Retrieval Early Stopping

Stops retrieving new hops after the minimum hop count when the answer is
stable and supported with sufficient confidence, or when the latest evidence
has low novelty while confidence and evidence support remain high. Repeated
documents alone are never sufficient to stop.

The default stopping parameters are:

- Hop >= minimum threshold (default 2)
- Answer stability window of 2 hops
- Novelty overlap threshold 0.7
- Confidence threshold 0.6

**Benefit**: Reduces retrieval cost when additional hops provide little new information

### Ablation modes and diagnostics

| Baseline | Reasoning Stop | Retrieval Stop | Description |
|----------|---------------|----------------|-------------|
| `fixed_hops` | ❌ | ❌ | Always H hops, full generation |
| `fixed_hops_reasoning_stop` | ✅ | ❌ | Early stop generation, always H hops |
| `retrieval_stop_only` | ❌ | ✅ | Early stop hops, full generation |
| `dual_stopping` | ✅ | ✅ | Both mechanisms enabled |
| `oracle_evidence` | ❌ | ❌ | Diagnostic only; supplies gold supporting paragraphs |

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
      "answer": "...",
      "confidence": 0.7,
      "tokens_generated": 120,
      "reasoning_stopped": true,
      "parse_failed": false,
      "retrieval_novelty": null
    }
  ],
  "retrieval_stopped": true,
  "retrieval_stop_reason": "stable_answer_high_conf_hop_2"
}
```

### metrics.json
```json
{
  "em": 0.35,
  "f1": 0.42,
  "avg_hops": 2.3,
  "avg_tokens_generated": 180.5,
  "reasoning_stop_rate": 0.68,
  "retrieval_stop_rate": 0.45,
  "parse_failure_rate": 0.0,
  "supporting_fact_recall_at_k": 0.82,
  "avg_latency_seconds": 4.4,
  "stop_reason_counts": {"max_hops_reached": 50}
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
- `--retrieval_mode`: `dense`, `bm25`, or `hybrid` (default `dense`)
- `--dense_weight`: Dense weight for hybrid ranking (default 0.5)
- `--enable_evidence_critic`: Add the model-free evidence sufficiency signal
- `--example_ids_file`: Reuse an exact JSON list of evaluation IDs

## Reproducibility

All runs use fixed seeds and save the full configuration and evaluation IDs to
`config.json` and `metrics.json`. For a matched benchmark, create one frozen ID
manifest and pass it to every method:

```bash
python scripts/run_experiment.py --method all \
  --dataset_path data/hotpotqa/hotpot_dev_distractor_v1.json \
  --example_ids_file example_ids_dev_50_seed42.json \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --embed_model_name sentence-transformers/all-MiniLM-L6-v2 \
  --seed 42 --hops 3 --top_k 5 --max_new_tokens 192
```

Run this command in the Renku terminal after installing `requirements.txt` and
downloading the dataset. The same repository is also runnable locally when the
PyTorch and Transformers dependencies are installed.

## Performance claims

Results depend on the model, dataset subset, prompt, and stopping thresholds.
The repository includes measured Renku artifacts and paired comparisons under
`research_results/2026-09-24`; these are small HotpotQA distractor evaluations,
not independent replication or evidence of state-of-the-art performance.
Accuracy and token-reduction claims require matched runs on the same examples.
A one-example GPU smoke test is only a wiring check, not a performance result.

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

## September 24 research experiments

The new literature-driven experiment is described in
[the research review](docs/RESEARCH_REVIEW_2026-09-24.md). It keeps the historical
reader available and tests corrected Qwen chat formatting, a short cited-answer
reader, all-context/oracle controls, and a prompted evidence-sufficiency critic.
It is **not** a trained Stop-RAG or Self-RAG reproduction. Current historical
candidate-reader metrics were corrected in [RESEARCH_LOG.md](RESEARCH_LOG.md).

On Renku, activate `.venv`, use branch `v-2`, and run:

```bash
python scripts/research_experiment.py \
  --ids outputs/example_ids_dev_50_seed42.json \
  --output outputs/my_new_research_run
```

For the prespecified sequential comparison, `bash scripts/run_research_suite.sh`
requires the existing frozen50 manifest and the recorded train5 smoke subset.
It refuses to overwrite research runs. Full prompts and raw generations are
saved under each run's `traces.jsonl`; compact shareable artifacts are exported
to `research_results/2026-09-24`. The original50 is exploratory. A separately
frozen disjoint100 sample is used for confirmation; see the protocol for limits.

To reproduce the confirmation comparison from the committed manifest without
depending on the train smoke subset, use a new output directory:

```bash
python scripts/research_experiment.py \
  --ids research_results/2026-09-24/research_confirm100_ids.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --arms top5 all_context adaptive \
  --output outputs/confirmation100_repeat
```

The default dataset path is `data/hotpotqa/hotpot_dev_distractor_v1.json`.
Use `--model Qwen/Qwen2.5-14B-Instruct` for the larger generator; use `--arms top9
ranked_all` for the static budget controls. Published runs keep MiniLM fixed for
these comparisons. Model revisions and library versions are recorded in each
run's protocol; pin those revisions/packages when reproducing the exact run.

After exporting completed runs, regenerate the analysis with
`python scripts/report_research_results.py` and
`python scripts/analyze_research_critic.py`. The latter also needs the original
dev dataset. `scripts/plot_research_results.py` optionally requires Matplotlib
and produces PNG/PDF figures only when all ten comparison arms are present.
The 14B top9/ranked-all arms are explicitly marked as post-hoc controls.
