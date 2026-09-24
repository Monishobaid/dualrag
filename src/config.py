"""Configuration for reproducible Dual Stopping RAG experiments.

The configuration keeps model, dataset, retrieval, generation, stopping, and
seed choices in one JSON-serializable object. The default model is Qwen 3B;
Renku benchmark commands explicitly select Qwen 7B when comparing methods.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass
class RAGConfig:
    """Configuration for the Dual Stopping RAG system."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # Change this to your preferred model
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Dataset configuration
    dataset_path: str = "data/hotpotqa/hotpot_dev_distractor_v1.json"
    split: str = "dev"
    max_examples: Optional[int] = None
    
    # Retrieval configuration
    top_k: int = 5  # Number of paragraphs to retrieve per hop
    max_hops: int = 4  # Maximum number of retrieval hops
    retrieval_mode: str = "dense"  # dense, bm25, or hybrid
    dense_weight: float = 0.5  # Hybrid dense interpolation weight
    # Optional second-stage reranking.  The first-stage retriever still
    # supplies the candidates; the cross-encoder only reorders that shortlist.
    reranker_model_name: Optional[str] = None
    reranker_candidate_k: int = 20
    max_evidence_paragraphs: Optional[int] = None
    # Reader ablation. ``legacy`` preserves the original confidence-based
    # selection; ``grounded_candidates`` verifies candidates collected across
    # hops before selecting the final answer.
    reader_mode: str = "legacy"
    
    # Generation configuration
    max_new_tokens: int = 256  # Max tokens per hop generation
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Reasoning early stopping configuration
    enable_reasoning_stop: bool = False
    chunk_tokens: int = 64  # Tokens per chunk for convergence checking
    convergence_k: int = 2  # Number of consecutive stable chunks needed
    
    # Retrieval early stopping configuration
    enable_retrieval_stop: bool = False
    min_hops: int = 2  # Minimum hops before considering stopping
    overlap_threshold: float = 0.7  # Jaccard overlap threshold for low novelty
    conf_threshold: float = 0.6  # Confidence threshold for stopping

    # Model-free evidence critic.  Disabled by default for backwards-compatible
    # baselines; enable it for critic-controlled retrieval experiments.
    enable_evidence_critic: bool = False
    evidence_critic_threshold: float = 0.65
    
    # Cost calculation
    cost_alpha: float = 1000.0  # Weight for hop cost
    cost_beta: float = 1.0  # Weight for token cost
    
    # Output configuration
    output_dir: str = "outputs"
    cache_dir: str = "runs/cache_embeddings"
    
    # Reproducibility
    seed: int = 42
    
    # Run metadata
    method: str = "fixed_hops"  # One of: fixed_hops, fixed_hops_reasoning_stop, retrieval_stop_only, dual_stopping
    run_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RAGConfig':
        """Load configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: str) -> 'RAGConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_baseline_config(method: str, base_config: Optional[RAGConfig] = None) -> RAGConfig:
    """
    Get configuration for a specific baseline method.
    
    Args:
        method: One of 'fixed_hops', 'fixed_hops_reasoning_stop', 
                'retrieval_stop_only', 'dual_stopping'
        base_config: Base configuration to modify (creates new if None)
    
    Returns:
        RAGConfig with appropriate flags set
    """
    if base_config is None:
        config = RAGConfig()
    else:
        config = RAGConfig(**base_config.to_dict())
    
    config.method = method
    
    if method == "fixed_hops":
        config.enable_reasoning_stop = False
        config.enable_retrieval_stop = False
    elif method == "fixed_hops_reasoning_stop":
        config.enable_reasoning_stop = True
        config.enable_retrieval_stop = False
    elif method == "retrieval_stop_only":
        config.enable_reasoning_stop = False
        config.enable_retrieval_stop = True
    elif method == "dual_stopping":
        config.enable_reasoning_stop = True
        config.enable_retrieval_stop = True
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return config
