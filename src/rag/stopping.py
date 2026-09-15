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
        retrieval_novelty: Optional[Dict[str, Any]] = None
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
        if current_hop < self.min_hops:
            return False, f"min_hops_not_met_({current_hop}<{self.min_hops})"
            
        # Check explicit empty retrieval
        if retrieval_novelty and retrieval_novelty.get('empty_retrieval', False):
            reason = "empty_retrieval"
            logging.info(f"Retrieval stopping at hop {current_hop}: {reason}")
            return True, reason

        # Determine answer stability
        stable = False
        if len(hop_history) >= self.answer_stability_window:
            recent_answers = [
                normalize_text(h['answer']) for h in hop_history[-self.answer_stability_window:]
            ]
            if len(set(recent_answers)) == 1 and recent_answers[0] != "":
                stable = True

        latest_confidence = hop_history[-1].get('confidence', 0.0)
        has_enough_evidence = hop_history[-1].get('has_enough_evidence', False)
        high_confidence = latest_confidence >= self.conf_threshold

        # Strict combinatorial stopping Check
        if stable and high_confidence and has_enough_evidence:
            reason = f"stable_answer_high_conf_hop_{current_hop}"
            logging.info(f"Retrieval stopping at hop {current_hop}: {reason}")
            return True, reason
        
        # Check novelty if confidence/stability fails but we are repeating the same docs
        if retrieval_novelty and retrieval_novelty.get('jaccard_overlap', 0.0) >= self.overlap_threshold:
             reason = f"low_novelty_overlap_{retrieval_novelty['jaccard_overlap']:.2f}"
             logging.info(f"Retrieval stopping at hop {current_hop}: {reason}")
             return True, reason
        
        # We didn't stop, which means condition not met
        reason = f"continue_stable={stable}_conf={latest_confidence:.2f}_ev={has_enough_evidence}"
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
