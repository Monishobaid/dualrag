#!/usr/bin/env python3
"""Analyze experiment logs in detail."""

import argparse
import json
import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import read_jsonl
from src.eval.metrics import compute_cost_metric

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_traces(traces, config):
    """Analyze trace logs in detail."""
    
    # Per-example analysis
    analysis_data = []
    
    for trace in traces:
        # Compute cost
        cost = compute_cost_metric(
            trace['hops_taken'],
            trace['total_tokens'],
            alpha=config.get('cost_alpha', 1000.0),
            beta=config.get('cost_beta', 1.0)
        )
        
        # Check if reasoning stopped in any hop
        reasoning_stopped = any(h.get('reasoning_stopped', False) for h in trace['hops'])
        
        analysis_data.append({
            'example_id': trace['id'],
            'question': trace['question'][:100],
            'gold_answer': trace['gold_answer'],
            'predicted_answer': trace['final_answer'],
            'em': trace['em'],
            'f1': trace['f1'],
            'hops': trace['hops_taken'],
            'tokens': trace['total_tokens'],
            'cost': cost,
            'reasoning_stopped': reasoning_stopped,
            'retrieval_stopped': trace.get('retrieval_stopped', False),
            'retrieval_stop_reason': trace.get('retrieval_stop_reason', ''),
        })
    
    return pd.DataFrame(analysis_data)


def print_distributions(df):
    """Print distribution statistics."""
    logger.info("\n" + "="*80)
    logger.info("DISTRIBUTIONS")
    logger.info("="*80)
    
    # Hops distribution
    logger.info("\nHops Distribution:")
    hop_dist = df['hops'].value_counts().sort_index()
    for hops, count in hop_dist.items():
        pct = count / len(df) * 100
        logger.info(f"  {hops} hops: {count:4d} ({pct:5.1f}%)")
    
    # Tokens distribution
    logger.info("\nTokens Statistics:")
    logger.info(f"  Min:    {df['tokens'].min():.0f}")
    logger.info(f"  Max:    {df['tokens'].max():.0f}")
    logger.info(f"  Mean:   {df['tokens'].mean():.1f}")
    logger.info(f"  Median: {df['tokens'].median():.1f}")
    logger.info(f"  Std:    {df['tokens'].std():.1f}")
    
    # Cost distribution
    logger.info("\nCost Statistics:")
    logger.info(f"  Min:    {df['cost'].min():.1f}")
    logger.info(f"  Max:    {df['cost'].max():.1f}")
    logger.info(f"  Mean:   {df['cost'].mean():.1f}")
    logger.info(f"  Median: {df['cost'].median():.1f}")
    logger.info(f"  Total:  {df['cost'].sum():.1f}")


def print_stopping_analysis(df):
    """Print stopping mechanism analysis."""
    logger.info("\n" + "="*80)
    logger.info("STOPPING ANALYSIS")
    logger.info("="*80)
    
    # Reasoning stopping
    reasoning_stopped = df['reasoning_stopped'].sum()
    reasoning_pct = reasoning_stopped / len(df) * 100
    logger.info(f"\nReasoning Early Stopping:")
    logger.info(f"  Stopped: {reasoning_stopped}/{len(df)} ({reasoning_pct:.1f}%)")
    
    if reasoning_stopped > 0:
        stopped_df = df[df['reasoning_stopped']]
        logger.info(f"  Avg tokens (stopped):     {stopped_df['tokens'].mean():.1f}")
        logger.info(f"  Avg tokens (not stopped): {df[~df['reasoning_stopped']]['tokens'].mean():.1f}")
    
    # Retrieval stopping
    retrieval_stopped = df['retrieval_stopped'].sum()
    retrieval_pct = retrieval_stopped / len(df) * 100
    logger.info(f"\nRetrieval Early Stopping:")
    logger.info(f"  Stopped: {retrieval_stopped}/{len(df)} ({retrieval_pct:.1f}%)")
    
    if retrieval_stopped > 0:
        # Stop reasons
        logger.info(f"\n  Stop Reasons:")
        stop_reasons = df[df['retrieval_stopped']]['retrieval_stop_reason'].value_counts()
        for reason, count in stop_reasons.items():
            pct = count / retrieval_stopped * 100
            logger.info(f"    {reason}: {count} ({pct:.1f}%)")
        
        stopped_df = df[df['retrieval_stopped']]
        logger.info(f"\n  Avg hops (stopped):     {stopped_df['hops'].mean():.2f}")
        logger.info(f"  Avg hops (not stopped): {df[~df['retrieval_stopped']]['hops'].mean():.2f}")


def print_accuracy_vs_cost(df):
    """Print accuracy vs cost analysis."""
    logger.info("\n" + "="*80)
    logger.info("ACCURACY vs COST")
    logger.info("="*80)
    
    # Correct vs incorrect
    correct = df[df['em'] > 0.5]
    incorrect = df[df['em'] <= 0.5]
    
    logger.info(f"\nCorrect Answers ({len(correct)} examples):")
    logger.info(f"  Avg hops:   {correct['hops'].mean():.2f}")
    logger.info(f"  Avg tokens: {correct['tokens'].mean():.1f}")
    logger.info(f"  Avg cost:   {correct['cost'].mean():.1f}")
    
    logger.info(f"\nIncorrect Answers ({len(incorrect)} examples):")
    logger.info(f"  Avg hops:   {incorrect['hops'].mean():.2f}")
    logger.info(f"  Avg tokens: {incorrect['tokens'].mean():.1f}")
    logger.info(f"  Avg cost:   {incorrect['cost'].mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment logs')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID to analyze')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Load results
    run_path = Path(args.output_dir) / args.run_id
    
    if not run_path.exists():
        logger.error(f"Run directory not found: {run_path}")
        return
    
    logger.info(f"Analyzing run: {args.run_id}")
    
    # Load config
    config_file = run_path / 'config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load traces
    traces_file = run_path / 'traces.jsonl'
    traces = read_jsonl(str(traces_file))
    
    logger.info(f"Loaded {len(traces)} traces")
    
    # Analyze
    df = analyze_traces(traces, config)
    
    # Print analyses
    print_distributions(df)
    print_stopping_analysis(df)
    print_accuracy_vs_cost(df)
    
    # Save CSV
    csv_file = run_path / 'summary.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"\nSummary CSV saved to: {csv_file}")


if __name__ == "__main__":
    main()
