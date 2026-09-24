#!/usr/bin/env bash
# Launch after the main suite to keep model latency free from GPU contention.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python scripts/run_experiment.py --method fixed_hops --model_name Qwen/Qwen2.5-7B-Instruct --example_ids_file outputs/example_ids_dev_50_seed42.json --hops 3 --top_k 5 --max_new_tokens 192 --temperature 0 --top_p 1 --max_evidence_paragraphs 15 --run_id research_raw_legacy50 > outputs/research_raw_legacy50.log 2>&1
python scripts/research_experiment.py --ids outputs/research_confirm100_ids.json --output outputs/research7b_budget_confirm100 --arms top9 ranked_all
python scripts/export_research_results.py
