#!/usr/bin/env python3
"""Evaluate experiment results."""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.hotpot_eval import evaluate_predictions
from src.eval.metrics import aggregate_trace_metrics, compute_efficiency_metrics
from src.utils.logging import read_jsonl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate experiment results')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID to evaluate')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Load results
    run_path = Path(args.output_dir) / args.run_id
    
    if not run_path.exists():
        logger.error(f"Run directory not found: {run_path}")
        return
    
    logger.info(f"Evaluating run: {args.run_id}")
    
    # Load predictions
    predictions_file = run_path / 'predictions.json'
    if not predictions_file.exists():
        logger.error("predictions.json not found")
        return
    
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Load traces
    traces_file = run_path / 'traces.jsonl'
    if not traces_file.exists():
        logger.error("traces.jsonl not found")
        return
    
    traces = read_jsonl(str(traces_file))
    
    # Extract gold answers from traces
    gold_answers = {t['id']: t['gold_answer'] for t in traces}
    
    # Evaluate
    logger.info("Computing metrics...")
    
    eval_metrics = evaluate_predictions(predictions, gold_answers)
    trace_metrics = aggregate_trace_metrics(traces)
    
    # Compute efficiency metrics
    em_scores = [t['em'] for t in traces]
    efficiency_metrics = compute_efficiency_metrics(traces, em_scores)
    
    # Combine all metrics
    all_metrics = {
        **eval_metrics,
        **trace_metrics,
        **efficiency_metrics
    }
    
    # Save metrics
    metrics_file = run_path / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print summary
    logger.info("="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Examples: {all_metrics['num_examples']}")
    logger.info(f"")
    logger.info(f"Accuracy:")
    logger.info(f"  EM:  {all_metrics['em']:.3f}")
    logger.info(f"  F1:  {all_metrics['f1']:.3f}")
    logger.info(f"")
    logger.info(f"Efficiency:")
    logger.info(f"  Avg hops:   {all_metrics['avg_hops']:.2f}")
    logger.info(f"  Avg tokens: {all_metrics.get('avg_tokens_generated', all_metrics.get('avg_tokens', 0.0)):.1f}")
    logger.info(f"  Avg cost:   {all_metrics['avg_cost']:.1f}")
    logger.info(f"")
    logger.info(f"Stopping Rates:")
    logger.info(f"  Reasoning: {all_metrics.get('reasoning_stop_rate', all_metrics.get('reasoning_early_stop_rate', 0.0)):.1%}")
    logger.info(f"  Retrieval: {all_metrics.get('retrieval_stop_rate', all_metrics.get('retrieval_early_stop_rate', 0.0)):.1%}")
    logger.info(f"")
    logger.info(f"Hop Distribution: {all_metrics['hop_distribution']}")
    logger.info("="*80)
    logger.info(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
