"""Metrics aggregation utilities."""

from typing import List, Dict, Any
import numpy as np


def aggregate_trace_metrics(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics from trace logs.
    
    Args:
        traces: List of trace dictionaries
    
    Returns:
        Aggregated metrics
    """
    if not traces:
        return {}
    
    hops_taken = [t['hops_taken'] for t in traces]
    total_tokens = [t['total_tokens'] for t in traces]
    
    # Stopping rates
    reasoning_stopped = sum(
        1 for t in traces
        if any(h.get('reasoning_stopped', False) for h in t['hops'])
    )
    retrieval_stopped = sum(1 for t in traces if t.get('retrieval_stopped', False))
    
    metrics = {
        'num_examples': len(traces),
        'avg_hops': np.mean(hops_taken),
        'median_hops': np.median(hops_taken),
        'std_hops': np.std(hops_taken),
        'avg_tokens_generated': np.mean(total_tokens),
        'median_tokens': np.median(total_tokens),
        'std_tokens': np.std(total_tokens),
        'reasoning_stop_rate': reasoning_stopped / len(traces) if traces else 0.0,
        'retrieval_stop_rate': retrieval_stopped / len(traces) if traces else 0.0,
    }

    critic_hops = [
        h.get('evidence_critic')
        for t in traces for h in t.get('hops', [])
        if h.get('evidence_critic') is not None
    ]
    if critic_hops:
        metrics['evidence_critic_hops'] = len(critic_hops)
        metrics['evidence_critic_sufficient_rate'] = sum(
            bool(c.get('sufficient', False)) for c in critic_hops
        ) / len(critic_hops)
        metrics['evidence_critic_avg_score'] = float(np.mean([
            float(c.get('score', 0.0)) for c in critic_hops
        ]))
    
    # Hop distribution
    hop_counts = {}
    for h in hops_taken:
        hop_counts[h] = hop_counts.get(h, 0) + 1
    metrics['hop_distribution'] = hop_counts
    
    return metrics


def compute_cost_metric(
    hops_taken: int,
    total_tokens: int,
    alpha: float = 1000.0,
    beta: float = 1.0
) -> float:
    """
    Compute cost metric.
    
    Cost = alpha * hops + beta * tokens
    
    Args:
        hops_taken: Number of hops
        total_tokens: Total tokens generated
        alpha: Weight for hop cost (retrieval cost)
        beta: Weight for token cost (generation cost)
    
    Returns:
        Total cost
    """
    return alpha * hops_taken + beta * total_tokens


def compute_efficiency_metrics(
    traces: List[Dict[str, Any]],
    em_scores: List[float],
    alpha: float = 1000.0,
    beta: float = 1.0
) -> Dict[str, Any]:
    """
    Compute efficiency metrics.
    
    Args:
        traces: Trace dictionaries
        em_scores: EM scores for each example
        alpha: Hop cost weight
        beta: Token cost weight
    
    Returns:
        Efficiency metrics
    """
    if not traces:
        return {}
    
    costs = [
        compute_cost_metric(t['hops_taken'], t['total_tokens'], alpha, beta)
        for t in traces
    ]
    
    # Cost per correct answer
    correct_indices = [i for i, em in enumerate(em_scores) if em > 0.5]
    costs_correct = [costs[i] for i in correct_indices] if correct_indices else []
    
    metrics = {
        'avg_cost': np.mean(costs),
        'median_cost': np.median(costs),
        'avg_cost_correct': np.mean(costs_correct) if costs_correct else 0.0,
        'total_cost': sum(costs)
    }
    
    return metrics
