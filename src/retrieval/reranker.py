"""Optional cross-encoder second-stage reranking.

The dense/BM25/hybrid ranker is fast over the full HotpotQA candidate pool.
This module applies a query-document cross-encoder only to its shortlist,
which gives a stronger relevance signal without replacing the reproducible
first-stage retrieval ablations.
"""

from typing import Any, Dict, List, Optional


class CrossEncoderReranker:
    """Score ``(question, paragraph)`` pairs with a sentence-transformers model."""

    def __init__(self, model_name: str, device: str = "auto"):
        # Import lazily so dense/BM25 experiments do not require the optional
        # model at import time and can still run in lightweight environments.
        from sentence_transformers import CrossEncoder

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.model_name = model_name
        self.device = device
        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, paragraphs: List[Dict[str, Any]]) -> List[float]:
        """Return one relevance score for each paragraph in input order."""
        pairs = [
            (query, f"{p.get('title', '')}\n{p.get('text', '')}")
            for p in paragraphs
        ]
        if not pairs:
            return []
        scores = self.model.predict(pairs, show_progress_bar=False)
        # numpy arrays and scalar tensors both support ``tolist``; normalise
        # to plain floats for JSON traces and deterministic sorting.
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not isinstance(scores, list):
            scores = [scores]
        return [float(score) for score in scores]
