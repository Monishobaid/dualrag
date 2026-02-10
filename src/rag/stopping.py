"""Retrieval early stopping logic."""

from typing import Dict, Any, List, Optional
import logging
from src.utils.text import normalize_text


class RetrievalStopper:
    """
    Retrieval early stopping based on multiple criteria.
    
    Stops retrieving new hops when:
    1. Answer is stable across hops
    2. Retrieval novelty is low
    3. Confidence is high enough
    4. Minimum hops threshold met
    """
    
    def __init__(
        self,
        min_hops: int = 2,
        overlap_threshold: float = 0.7,
        conf_threshold: float = 0.6,
        answer_stability_window: int = 2
    ):
        """
        Initialize retrieval stopper.
        
        Args:
            min_hops: Minimum hops before considering stopping
            overlap_threshold: Jaccard overlap threshold for low novelty
            conf_threshold: Confidence threshold
            answer_stability_window: Number of hops to check for answer stability
        """
        self.min_hops = min_hops
        self.overlap_threshold = overlap_threshold
        self.conf_threshold = conf_threshold
        self.answer_stability_window = answer_stability_window
    
    def should_stop(
        self,
        current_hop: int,
        hop_history: List[Dict[str, Any]],
        retrieval_novelty: Optional[Dict[str, float]] = None
    ) -> tuple:
        """
        Determine if retrieval should stop.
        
        Args:
            current_hop: Current hop number (1-indexed)
            hop_history: List of hop data dicts with 'answer' and 'confidence'
            retrieval_novelty: Optional novelty metrics from latest retrieval
        
        Returns:
            (should_stop: bool, reason: str)
        """
        if not hop_history[-1].get('has_enough_evidence', True):
            return False, "insufficient_evidence"

        # Must meet minimum hops
        if current_hop < self.min_hops:
            return False, "below_min_hops"
        
        if len(hop_history) < 2:
            return False, "insufficient_history"
        
        # Check answer stability
        recent_hops = hop_history[-self.answer_stability_window:]
        answers_norm = [normalize_text(h['answer']) for h in recent_hops]
        
        answer_stable = len(set(answers_norm)) == 1 and answers_norm[0]  # All same and not empty
        
        # Check latest confidence
        latest_confidence = hop_history[-1].get('confidence', 0.0)
        high_confidence = latest_confidence >= self.conf_threshold
        
        # Check retrieval novelty
        low_novelty = False
        if retrieval_novelty:
            jaccard_overlap = retrieval_novelty.get('jaccard_overlap', 0.0)
            low_novelty = jaccard_overlap >= self.overlap_threshold
        
        # Decision logic: stop if answer is stable and confident
        # We trust the model's stability more than retrieval novelty
        if answer_stable and high_confidence:
            reason = f"answer_stable_and_high_conf"
            logging.info(f"Retrieval stopping at hop {current_hop}: {reason}")
            return True, reason
            
        # Stopping if very high confidence even if not strictly stable history (e.g. first hop super confident? No, requires history)
        # But if we have low novelty and ANY confidence, maybe stop? No.
        
        # Original strict check:
        # if answer_stable and high_confidence and low_novelty: ...
        
        # Continue
        reasons = []
        if not answer_stable:
            reasons.append("answer_unstable")
        if not high_confidence:
            reasons.append(f"low_conf_{latest_confidence:.2f}")
        if not low_novelty:
            reasons.append("high_novelty")
        
        reason = "_".join(reasons) if reasons else "continue"
        return False, reason
    
    def get_stopping_stats(
        self,
        hop_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get stopping statistics for analysis.
        
        Args:
            hop_history: Full hop history
        
        Returns:
            Statistics dict
        """
        if not hop_history:
            return {
                'final_confidence': 0.0,
                'answer_changes': 0,
                'avg_confidence': 0.0
            }
        
        answers_norm = [normalize_text(h['answer']) for h in hop_history]
        confidences = [h.get('confidence', 0.0) for h in hop_history]
        
        # Count answer changes
        answer_changes = sum(1 for i in range(1, len(answers_norm)) if answers_norm[i] != answers_norm[i-1])
        
        return {
            'final_confidence': confidences[-1] if confidences else 0.0,
            'answer_changes': answer_changes,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'unique_answers': len(set(answers_norm))
        }
