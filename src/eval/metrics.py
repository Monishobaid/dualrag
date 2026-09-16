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
        'avg_hops': float(np.mean(hops_taken)),
        'median_hops': float(np.median(hops_taken)),
        'std_hops': float(np.std(hops_taken)),
        'avg_tokens_generated': float(np.mean(total_tokens)),
        'median_tokens': float(np.median(total_tokens)),
        'std_tokens': float(np.std(total_tokens)),
        'reasoning_stop_rate': reasoning_stopped / len(traces) if traces else 0.0,
        'retrieval_stop_rate': retrieval_stopped / len(traces) if traces else 0.0,
    }
    # Per-example wall-clock latency is recorded by run_experiment.  Keep
    # missing values out of the aggregate so older traces remain readable.
    latencies = [float(t['latency_seconds']) for t in traces
                 if t.get('latency_seconds') is not None]
    if latencies:
        metrics.update({
            'avg_latency_seconds': float(np.mean(latencies)),
            'median_latency_seconds': float(np.median(latencies)),
            'std_latency_seconds': float(np.std(latencies)),
        })

    # A parse failure means the model output could not be interpreted as the
    # expected JSON response for at least one hop.  This is reported per
    # example (the unit used for benchmark failure rates), while hop-level
    # counts are also exposed for diagnosis.
    parse_failed_examples = sum(
        1 for t in traces
        if any(bool(h.get('parse_failed', False)) for h in t.get('hops', []))
    )
    parse_failed_hops = sum(
        1 for t in traces for h in t.get('hops', [])
        if bool(h.get('parse_failed', False))
    )
    total_hops = sum(len(t.get('hops', [])) for t in traces)
    metrics.update({
        'num_parse_failed_examples': parse_failed_examples,
        'parse_failure_rate': parse_failed_examples / len(traces) if traces else 0.0,
        'num_parse_failed_hops': parse_failed_hops,
        'parse_failure_hop_rate': parse_failed_hops / total_hops if total_hops else 0.0,
    })
    recalls = [t.get('supporting_title_recall') for t in traces
               if t.get('supporting_title_recall') is not None]
    if recalls:
        # Cumulative recall over all retrieved hops, retained for multihop
        # diagnostics and backwards compatibility.
        metrics['supporting_title_recall'] = float(np.mean(recalls))
    first_hop_recalls = [t.get('supporting_fact_recall_at_k') for t in traces
                         if t.get('supporting_fact_recall_at_k') is not None]
    if first_hop_recalls:
        metrics['supporting_fact_recall_at_k'] = float(np.mean(first_hop_recalls))

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

    # Keep the complete stop-reason distribution for auditability.  A method
    # may stop because of convergence, evidence sufficiency, missing follow-up
    # query, or simply reaching the hop budget.
    stop_reasons = {}
    for trace in traces:
        reason = trace.get('retrieval_stop_reason') or 'unknown'
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    metrics['stop_reason_counts'] = stop_reasons
    
    # Stop reasons are kept as distributions instead of one opaque rate so
    # benchmark reports can distinguish max-hop termination, convergence,
    # missing follow-up queries, and critic/controller decisions.
    retrieval_reasons = {}
    reasoning_reasons = {}
    for trace in traces:
        rreason = trace.get('retrieval_stop_reason') or 'unspecified'
        retrieval_reasons[rreason] = retrieval_reasons.get(rreason, 0) + 1
        for hop in trace.get('hops', []):
            stats = hop.get('reasoning_stop_stats') or {}
            if stats.get('stopped_early'):
                reason = stats.get('stop_reason') or 'unspecified'
                reasoning_reasons[reason] = reasoning_reasons.get(reason, 0) + 1
    metrics['retrieval_stop_reason_counts'] = retrieval_reasons
    metrics['reasoning_stop_reason_counts'] = reasoning_reasons

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
