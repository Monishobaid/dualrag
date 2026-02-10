"""Text processing utilities."""

import re
import string
import json
from typing import Optional, Tuple, List


def normalize_text(text: str) -> str:
    """
    Normalize text for answer matching (HotpotQA style).
    
    Steps:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Collapse whitespace
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # Collapse whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match score.
    
    Args:
        prediction: Predicted answer
        ground_truth: Gold answer
    
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(ground_truth)
    return 1.0 if pred_norm == gold_norm else 0.0


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.
    
    Args:
        prediction: Predicted answer
        ground_truth: Gold answer
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    
    # Common tokens
    common = set(pred_tokens) & set(gold_tokens)
    num_common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract first JSON object from text.
    
    Tries to find {...} block and parse it.
    
    Args:
        text: Text potentially containing JSON
    
    Returns:
        Parsed JSON dict or None if extraction fails
    """
    # Remove known stop tokens and markdown
    text = text.replace('<END_JSON>', '')
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON block
    start_idx = text.find('{')
    if start_idx == -1:
        return None
        
    # Find the last closing brace
    end_idx = text.rfind('}')
    if end_idx == -1 or end_idx < start_idx:
        return None
        
    candidate = text[start_idx:end_idx+1]
    
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try to fix common issues
        try:
            # Remove trailing commas
            candidate = re.sub(r',\s*}', '}', candidate)
            candidate = re.sub(r',\s*]', ']', candidate)
            return json.loads(candidate)
        except json.JSONDecodeError:
             pass
             
    return None


def extract_answer_from_text(text: str) -> str:
    """
    Extract answer from free-form text.
    
    Heuristics:
    - Look for "Answer:" pattern
    - Take first sentence/line if no pattern
    - Strip and clean
    
    Args:
        text: Generated text potentially containing answer
    
    Returns:
        Extracted answer string
    """
    text = text.strip()
    
    # Look for explicit answer marker
    patterns = [
        r"(?:Answer|answer):\s*(.+)",
        r"(?:The answer is|the answer is)\s+(.+)",
        r"(?:Final answer|final answer):\s*(.+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Take first sentence
            answer = re.split(r'[.!?\n]', answer)[0].strip()
            return answer
    
    # Fallback: take first sentence
    sentences = re.split(r'[.!?\n]', text)
    for sent in sentences:
        sent = sent.strip()
        if sent:
            return sent
    
    return text[:100]  # Truncate if very long


def compute_jaccard_similarity(set1: set, set2: set) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
    
    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to maximum length with ellipsis.
    
    Args:
        text: Input text
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."
