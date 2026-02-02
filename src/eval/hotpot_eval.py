"""HotpotQA evaluation metrics."""

from typing import Dict, Any, List
from src.utils.text import compute_exact_match, compute_f1


def evaluate_predictions(
    predictions: Dict[str, str],
    gold_answers: Dict[str, str]
) -> Dict[str, Any]:
    """
    Evaluate predictions against gold answers.
    
    Args:
        predictions: Dict mapping example ID to predicted answer
        gold_answers: Dict mapping example ID to gold answer
    
    Returns:
        Dictionary with EM and F1 scores
    """
    if not predictions:
        return {
            'em': 0.0,
            'f1': 0.0,
            'num_examples': 0
        }
    
    em_scores = []
    f1_scores = []
    
    for example_id, pred_answer in predictions.items():
        gold_answer = gold_answers.get(example_id, "")
        
        em = compute_exact_match(pred_answer, gold_answer)
        f1 = compute_f1(pred_answer, gold_answer)
        
        em_scores.append(em)
        f1_scores.append(f1)
    
    return {
        'em': sum(em_scores) / len(em_scores) if em_scores else 0.0,
        'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        'num_examples': len(em_scores)
    }


def evaluate_single(prediction: str, gold_answer: str) -> Dict[str, float]:
    """
    Evaluate single prediction.
    
    Args:
        prediction: Predicted answer
        gold_answer: Gold answer
    
    Returns:
        Dict with 'em' and 'f1'
    """
    return {
        'em': compute_exact_match(prediction, gold_answer),
        'f1': compute_f1(prediction, gold_answer)
    }
