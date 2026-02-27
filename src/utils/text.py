"""Text processing utilities."""

import re
import string
import json
import logging
from typing import Optional, Tuple, List


def normalize_text(text: str) -> str:
    """
    Normalize text for answer matching (HotpotQA style).
    """
    if not text:
        return ""
        
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


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
    
    # Prioritize <JSON> tags if present - Robust Regex
    # Matches <JSON>, optionally content, then </JSON>, ignoring case and leading/trailing junk
    json_tag_match = re.search(r'<JSON>\s*(?:```json)?\s*(.*?)\s*(?:```)?\s*</JSON>', text, re.DOTALL | re.IGNORECASE)
    if json_tag_match:
        text = json_tag_match.group(1).strip()
    
    # Try strict parsing first
    extracted = None
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx:end_idx+1]
        try:
            extracted = json.loads(candidate)
        except json.JSONDecodeError:
            # Try to fix common issues
            try:
                # Remove trailing commas
                candidate_fix = re.sub(r',\s*}', '}', candidate)
                candidate_fix = re.sub(r',\s*]', ']', candidate_fix)
                extracted = json.loads(candidate_fix)
            except json.JSONDecodeError as e:
                 logging.warning(f"DEBUG: JSON Parse Failed.\nOriginal: {text[:100]}...\nCandidate: {candidate}\nError: {e}")
                 pass

    if extracted:
        return extracted
             
     # Fallback: Regex extraction for critical fields if JSON parse fails (e.g. truncation or missing braces)
    logging.warning(f"DEBUG: strict JSON parse failed (braces missing or invalid). Attempting regex fallback. Raw text snippet:\n{text[:500]}...")
    
    extracted = {}
    
    # Extract Answer
    ans_match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
    if ans_match:
        extracted['answer'] = ans_match.group(1).strip()
    else:
        # Try single quotes
        ans_match = re.search(r"'answer'\s*:\s*'([^']+)'", text)
        if ans_match:
             extracted['answer'] = ans_match.group(1).strip()

    # Extract Confidence - handle optional quotes
    conf_match = re.search(r'"confidence"\s*:\s*"?([\d\.]+)"?', text)
    if conf_match:
        try:
            val = float(conf_match.group(1))
            extracted['confidence'] = val
        except ValueError:
            extracted['confidence'] = 0.5
            
    # Extract Follow-up
    fup_match = re.search(r'"followup_query"\s*:\s*"([^"]+)"', text)
    if fup_match:
        extracted['followup_query'] = fup_match.group(1).strip()
        
    # Extract Evidence Flag
    ev_match = re.search(r'"has_enough_evidence"\s*:\s*(true|false|True|False)', text)
    if ev_match:
        extracted['has_enough_evidence'] = ev_match.group(1).lower() == 'true'

    if 'answer' in extracted:
        logging.info(f"DEBUG: Recovered partial JSON: {extracted}")
        # Fill missing defaults
        if 'confidence' not in extracted: extracted['confidence'] = 0.5
        if 'reasoning' not in extracted: extracted['reasoning'] = "Parsed from partial JSON"
        if 'followup_query' not in extracted: extracted['followup_query'] = ""
        if 'has_enough_evidence' not in extracted: extracted['has_enough_evidence'] = False
        if 'supporting_quotes' not in extracted: extracted['supporting_quotes'] = []
        return extracted

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
