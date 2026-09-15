"""Baseline configurations for ablation studies."""

from src.config import RAGConfig, get_baseline_config


def get_fixed_hops_baseline(base_config: RAGConfig) -> RAGConfig:
    """
    Baseline 1: Fixed hops with no stopping.
    
    - Always runs H hops
    - No reasoning early stopping
    - No retrieval early stopping
    """
    return get_baseline_config("fixed_hops", base_config)


def get_fixed_hops_reasoning_stop_baseline(base_config: RAGConfig) -> RAGConfig:
    """
    Baseline 2: Fixed hops with reasoning stop.
    
    - Always runs H hops
    - Reasoning early stopping enabled
    - No retrieval early stopping
    """
    return get_baseline_config("fixed_hops_reasoning_stop", base_config)


def get_retrieval_stop_only_baseline(base_config: RAGConfig) -> RAGConfig:
    """
    Baseline 3: Retrieval stop only.
    
    - Retrieval early stopping enabled
    - No reasoning early stopping
    - May stop before H hops
    """
    return get_baseline_config("retrieval_stop_only", base_config)


def get_dual_stopping_baseline(base_config: RAGConfig) -> RAGConfig:
    """
    Baseline 4: Dual stopping (both mechanisms).
    
    - Reasoning early stopping enabled
    - Retrieval early stopping enabled
    - Most aggressive stopping
    """
    return get_baseline_config("dual_stopping", base_config)


BASELINE_METHODS = {
    "fixed_hops": get_fixed_hops_baseline,
    "fixed_hops_reasoning_stop": get_fixed_hops_reasoning_stop_baseline,
    "retrieval_stop_only": get_retrieval_stop_only_baseline,
    "dual_stopping": get_dual_stopping_baseline
}

def get_oracle_evidence_baseline(base_config: RAGConfig) -> RAGConfig:
    """Diagnostic baseline: generation receives gold supporting paragraphs."""
    config = RAGConfig(**base_config.to_dict())
    config.method = "oracle_evidence"
    config.enable_reasoning_stop = False
    config.enable_retrieval_stop = False
    return config


def get_baseline(method: str, base_config: RAGConfig) -> RAGConfig:
    """
    Get configuration for a baseline method.
    
    Args:
        method: Baseline method name
        base_config: Base configuration
    
    Returns:
        Configured RAGConfig for the baseline
    
    Raises:
        ValueError: If method is unknown
    """
    if method == "oracle_evidence":
        return get_oracle_evidence_baseline(base_config)
    if method not in BASELINE_METHODS:
        raise ValueError(f"Unknown baseline method: {method}. Must be one of {list(BASELINE_METHODS.keys())}")
    
    return BASELINE_METHODS[method](base_config)
