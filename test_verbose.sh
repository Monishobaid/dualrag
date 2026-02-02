#!/bin/bash
# Test script with verbose logging

python scripts/run_experiment.py \
  --method dual_stopping \
  --max_examples 2 \
  --hops 3 \
  --top_k 3 \
  --chunk_tokens 64 \
  --convergence_k 2 \
  --min_hops 2 \
  --overlap_threshold 0.7 \
  --conf_threshold 0.6 \
  --seed 42 \
  --model_name "Qwen/Qwen2.5-0.5B-Instruct"
