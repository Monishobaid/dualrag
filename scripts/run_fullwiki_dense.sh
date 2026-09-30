#!/usr/bin/env bash
# Matched fullwiki continuation. Start after the reader releases the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/runs/hf-cache}"
export TOKENIZERS_PARALLELISM=false
test -f data/corpus/fullwiki.sqlite
# Pin the same model snapshots used in the completed pooled-corpus experiment.
python scripts/run_corpus_retrieval.py --corpus data/corpus/fullwiki.sqlite \
  --output outputs/fullwiki_20260930_all --device cuda --batch-size 128 \
  --embedder-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
python scripts/analyze_corpus_retrieval.py --run outputs/fullwiki_20260930_all \
  --output research_results/2026-09-30/fullwiki_all
python scripts/read_corpus_evidence.py --retrieval-run outputs/fullwiki_20260930_all \
  --output outputs/fullwiki_20260930_reader --critic \
  --expected-model-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --arms plain_bm25 plain_dense plain_hybrid plain_reranked contextual_bm25 contextual_dense contextual_hybrid contextual_reranked
python scripts/analyze_corpus_reader.py --run outputs/fullwiki_20260930_reader \
  --output research_results/2026-09-30/fullwiki_reader
