#!/usr/bin/env bash
# Pinned evaluation order: no dev-driven parameter changes between arms.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/runs/hf-cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p outputs data/corpus
run_name="${1:-evidence_20260930_renku}"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required for this run"; print(torch.cuda.get_device_properties(0))'
PYTHONPATH=. python -m pytest -q tests/test_corpus.py tests/test_sufficiency.py tests/test_reader_budget.py tests/test_paired_research.py
if [ ! -f data/hotpotqa/hotpot_dev_distractor_v1.json ]; then
  python scripts/restore_hotpot_dev.py
fi
if [ ! -f data/corpus/dev-pool.sqlite ]; then
  python scripts/build_corpus.py --contexts data/hotpotqa/hotpot_dev_distractor_v1.json --output data/corpus/dev-pool.sqlite
fi
python scripts/run_corpus_retrieval.py --corpus data/corpus/dev-pool.sqlite --output "outputs/${run_name}_retrieval" --device cuda --batch-size 128
python scripts/analyze_corpus_retrieval.py --run "outputs/${run_name}_retrieval" --output "research_results/2026-09-30/${run_name}_retrieval"
python scripts/read_corpus_evidence.py --retrieval-run "outputs/${run_name}_retrieval" --output "outputs/${run_name}_reader" --critic --arms plain_bm25 plain_dense plain_hybrid plain_reranked contextual_bm25 contextual_dense contextual_hybrid contextual_reranked
