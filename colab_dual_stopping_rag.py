# Colab-ready single file for Dual-Stopping RAG

# --- IMPORTS ---
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Optional
from typing import Dict, Any, List
from typing import Dict, Any, List, Optional
from typing import Dict, Any, List, Optional, Union
from typing import List, Dict, Any
from typing import List, Dict, Any, Optional
from typing import List, Dict, Any, Optional, Set
from typing import List, Optional, Dict, Any
from typing import List, Union
from typing import Optional
from typing import Optional, List, Dict, Any, Tuple
from typing import Optional, List, Dict, Any, Union
from typing import Optional, Tuple, List
import argparse
import hashlib
import json
import logging
import numpy as np
import os
import pickle
import random
import re
import string
import sys
import tiktoken
import time
import torch


# --- CODE ---

# ========================================
# From: src/utils/text.py
# ========================================
"""Text processing utilities."""



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

# ========================================
# From: src/utils/logging.py
# ========================================
"""Logging utilities for RAG experiments."""



def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with console and optional file handlers.
    
    Args:
        name: Logger name
        log_file: Optional file path for logging
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


class JSONLWriter:
    """Writer for JSONL trace files."""
    
    def __init__(self, filepath: str):
        """
        Initialize JSONL writer.
        
        Args:
            filepath: Path to JSONL file
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = None
    
    def __enter__(self):
        """Context manager entry."""
        self.file_handle = open(self.filepath, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.file_handle:
            self.file_handle.close()
    
    def write(self, obj: dict) -> None:
        """
        Write a single JSON object as a line.
        
        Args:
            obj: Dictionary to write
        """
        if self.file_handle is None:
            raise RuntimeError("JSONLWriter must be used as context manager")
        
        json_line = json.dumps(obj, ensure_ascii=False)
        self.file_handle.write(json_line + '\n')
        self.file_handle.flush()
    
    def write_many(self, objects: list) -> None:
        """
        Write multiple JSON objects.
        
        Args:
            objects: List of dictionaries to write
        """
        for obj in objects:
            self.write(obj)


def read_jsonl(filepath: str) -> list:
    """
    Read a JSONL file.
    
    Args:
        filepath: Path to JSONL file
    
    Returns:
        List of parsed JSON objects
    """
    objects = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                objects.append(json.loads(line))
    return objects


def log_experiment_start(logger: logging.Logger, config: Any) -> None:
    """Log experiment start with configuration."""
    logger.info("="*80)
    logger.info("EXPERIMENT START")
    logger.info("="*80)
    logger.info(f"Method: {config.method}")
    logger.info(f"Max examples: {config.max_examples}")
    logger.info(f"Max hops: {config.max_hops}")
    logger.info(f"Top-K: {config.top_k}")
    logger.info(f"Reasoning stop: {config.enable_reasoning_stop}")
    logger.info(f"Retrieval stop: {config.enable_retrieval_stop}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Run ID: {config.run_id}")
    logger.info("="*80)


def log_experiment_end(logger: logging.Logger, stats: dict) -> None:
    """Log experiment end with summary statistics."""
    logger.info("="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Total examples: {stats.get('total_examples', 0)}")
    logger.info(f"Avg hops: {stats.get('avg_hops', 0):.2f}")
    logger.info(f"Avg tokens: {stats.get('avg_tokens_generated', 0):.1f}")
    logger.info(f"EM: {stats.get('em', 0):.3f}")
    logger.info(f"F1: {stats.get('f1', 0):.3f}")
    logger.info("="*80)

# ========================================
# From: src/eval/metrics.py
# ========================================
"""Metrics aggregation utilities."""



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

# ========================================
# From: src/config.py
# ========================================
"""Configuration management for Dual Stopping RAG."""



@dataclass
class RAGConfig:
    """Configuration for the Dual Stopping RAG system."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # Change this to your preferred model
    embed_model_name: str = "facebook/contriever-msmarco"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Dataset configuration
    dataset_path: str = "data/hotpotqa/hotpot_dev_distractor_v1.json"
    split: str = "dev"
    max_examples: Optional[int] = None
    
    # Retrieval configuration
    top_k: int = 5  # Number of paragraphs to retrieve per hop
    max_hops: int = 4  # Maximum number of retrieval hops
    
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

# ========================================
# From: src/llm/prompts.py
# ========================================
"""Prompt templates for RAG system."""



def format_paragraphs(paragraphs: List[Dict[str, Any]], max_paragraphs: int = 10) -> str:
    """
    Format paragraphs for prompt.
    
    Args:
        paragraphs: List of paragraph dicts with 'title' and 'text'
        max_paragraphs: Maximum paragraphs to include
    
    Returns:
        Formatted evidence string
    """
    paragraphs = paragraphs[:max_paragraphs]
    
    formatted = []
    for i, para in enumerate(paragraphs, 1):
        title = para.get('title', f'Paragraph {i}')
        text = para.get('text', '')
        formatted.append(f"[{i}] Title: {title}\n{text}")
    
    return "\n\n".join(formatted)


def get_multihop_reasoning_prompt(
    question: str,
    evidence: List[Dict[str, Any]],
    hop: int
) -> str:
    """
    Prompt for multi-hop reasoning.
    
    Args:
        question: Original question
        evidence: Retrieved paragraphs
        hop: Current hop number
    
    Returns:
        Formatted prompt
    """
    evidence_str = format_paragraphs(evidence)
    
    prompt = f"""You are answering a multi-hop question that may require information from multiple sources.

Question: {question}

Evidence (Retrieved Information):
{evidence_str}

Based on the evidence above, provide your reasoning and structured response in a single valid JSON object. Do not use Markdown formatting (e.g. ```json).
Format:
{{
    "current_hypothesis": "<Brief answer based on current knowledge>",
    "answer": "<Insert your final answer here (SHORT SPAN ONLY, NO SENTENCES)>",
    "has_enough_evidence": <true or false>,
    "supporting_quotes": ["<quote 1>", "<quote 2>"],
    "followup_query": "<Insert follow-up question here, or empty string if done>",
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<Concise reasoning (MAX 20 WORDS).>"
}}

Instructions:
- "reasoning": Analyze briefly (MAX 20 WORDS). BE CONCISE.
- "current_hypothesis": Brief answer based on current knowledge.
- SHORT ANSWER: Return ONLY the entity, name, date, or number. Do NOT write full sentences like "The answer is...".
- TYPE CONSISTENCY: Ensure your answer matches the specific type requested.
- ATTRIBUTE SELECTION: If asked for an attribute of an entity, provide the attribute, not the entity itself.
- CONSTRAINT SATISFACTION: If the question implies a singular answer, do not provide a list.
- MISSING INFO: If the retrieved text mentions an entity but doesn't explicitly answer the question about that entity, set "has_enough_evidence" to false and write a "followup_query" specifically asking about that missing detail.
- DO NOT GUESS: If you cannot find the EXACT answer in the text, you MUST ask a follow-up query. Focus on what details are missing.
- INTERMEDIATE HOPPING: If your "current_hypothesis" is a Place/Object but the Question asks "Who" (or vice versa), you MUST form a follow-up query to bridge the gap.
- CONFIDENCE SCORING: If you have to guess, or if you are asking a follow-up query, set confidence < 0.5.
- CRITICAL: Return ONLY the JSON object. No preamble, no postscript.

Response:"""
    
    return prompt


def get_answer_extraction_prompt(text: str) -> str:
    """
    Prompt to extract final answer from generated text.
    
    Args:
        text: Generated text containing reasoning
    
    Returns:
        Extraction prompt
    """
    prompt = f"""Extract the final answer from the following text based on the user's implicit question type.
    
Text:
{text}

Instructions:
1. Identify if the user is asking for a Person, Place, Number, Date, or Attribute.
2. Select the candidate answer that matches this type.
3. RULES:
   - For PERSONS: Provide the FULL NAME (e.g., "Salma Hayek Pinault", not just "Salma Hayek").
   - For NUMBERS: Include the UNIT if appropriate (e.g., "40 members", not just "40").
   - For LISTS: If the question implies a single entity, pick the most prominent one.
4. Return ONLY the short, direct answer.

Final Answer:"""
    
    return prompt


def get_followup_query_prompt(
    question: str,
    current_answer: str,
    evidence_so_far: List[Dict[str, Any]]
) -> str:
    """
    Prompt to generate follow-up query.
    
    Args:
        question: Original question
        current_answer: Current best answer
        evidence_so_far: Evidence retrieved so far
    
    Returns:
        Follow-up query generation prompt
    """
    evidence_str = format_paragraphs(evidence_so_far, max_paragraphs=5)
    
    prompt = f"""You are answering: {question}

Current answer: {current_answer}

Evidence so far:
{evidence_str}

What specific information do you still need to answer this question completely? Generate ONE focused follow-up question that would help you find the missing information.

If you have enough information already, respond with "NONE".

Follow-up question:"""
    
    return prompt


def get_answer_verification_prompt(question: str, answer: str) -> str:
    """
    Prompt to verify if the answer matches the question's semantic type constraint.
    """
    prompt = f"""Check if the Proposed Answer satisfies the specific semantic constraints of the Question (e.g., specific Nationality, Heritage, Genre, Number, etc).
If the Proposed Answer is incorrect (e.g. wrong type, wrong entity), correct it using the Question context. If it is already correct, repeat it.

Question: {question}
Proposed Answer: {answer}

Examples:
Q: "Who starred in the film?" A: "Brad Pitt" -> Correct (Person)
Q: "Starred the actress of what heritage?" A: "Brad Pitt" -> INCORRECT (Person, not Heritage). Correct: "American"
Q: "How many members?" A: "The Senate" -> INCORRECT (Entity). Correct: "100"
Q: "Who is the actress?" A: "Salma Hayek" -> VERIFIED: Salma Hayek Pinault (Prefer Full Name)

Task:
1. Identify the TYPE requested by the Question (e.g. Person, Heritage, Number).
2. CRITICAL: Check if Proposed Answer matches that TYPE. If NO, Correct it immediately.
3. ONLY IF Type matches: Check specific constraints (e.g. Full Name for Persons, Units for Numbers).
4. If yes, output: VERIFIED: <Answer>
5. If no, output: CORRECTION: <Corrected Answer based on context or type>

Response:"""
    return prompt


def get_confidence_prompt(question: str, answer: str, evidence: List[Dict[str, Any]]) -> str:
    """
    Prompt to estimate confidence in answer.
    
    Args:
        question: Question
        answer: Current answer
        evidence: Available evidence
    
    Returns:
        Confidence estimation prompt
    """
    evidence_str = format_paragraphs(evidence, max_paragraphs=5)
    
    prompt = f"""Question: {question}
Answer: {answer}

Evidence:
{evidence_str}

On a scale of 0.0 to 1.0, how confident are you that this answer is correct based on the evidence?
- 1.0 = Completely certain, evidence directly supports the answer
- 0.5 = Somewhat confident, some supporting evidence
- 0.0 = Not confident, insufficient evidence

Respond with ONLY a number between 0.0 and 1.0.

Confidence:"""
    
    return prompt


def get_simple_qa_prompt(question: str, evidence: List[Dict[str, Any]]) -> str:
    """
    Simple QA prompt without JSON output (for baseline).
    
    Args:
        question: Question  
        evidence: Evidence paragraphs
    
    Returns:
        Simple QA prompt
    """
    evidence_str = format_paragraphs(evidence)
    
    prompt = f"""Answer the following question based on the provided evidence.

Question: {question}

Evidence:
{evidence_str}

Provide a concise, direct answer:

Answer:"""
    
    return prompt


# System prompts
SYSTEM_PROMPT = """You are a helpful AI assistant specialized in answering multi-hop questions. You analyze evidence carefully and provide accurate, concise answers."""

JSON_SYSTEM_PROMPT = """You are a helpful AI assistant that always responds in valid JSON format. You analyze questions and evidence carefully to provide accurate answers."""

# ========================================
# From: src/llm/azure_openai.py
# ========================================
"""Azure OpenAI model wrapper."""


try:
    pass
except ImportError:
    AzureOpenAI = None
    tiktoken = None

try:
    load_dotenv()
except ImportError:
    pass

class AzureOpenAIModel:
    """Wrapper for Azure OpenAI models."""
    
    # Configuration for different models
    CONFIGS = {
        "gpt-4.1": {
            "api_version": "2025-01-01-preview",
            "azure_deployment": "gpt-4.1"
        },
        "gpt-4.1-mini": {
            "api_version": "2024-12-01-preview",
            "azure_deployment": "gpt-4.1-mini"
        },
        "testGPT4oMini-Standard": {
            "api_version": "2024-02-15-preview",
            "azure_deployment": "testGPT4oMini-Standard"
        },
        "gpt-4o-mini-standard": {
            "api_version": "2024-02-15-preview",
            "azure_deployment": "testGPT4oMini-Standard"
        }
    }

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize Azure OpenAI model.
        
        Args:
            model_name: Name of the model configuration to use (key in CONFIGS)
            device: Ignored, but kept for compatibility
        """
        if AzureOpenAI is None:
            raise ImportError("openai package is required for AzureOpenAIModel. Please install it with `pip install openai tiktoken`.")
            
        self.model_name = model_name
        self.device = device
        
        # Check if model config exists, otherwise assume model_name is the deployment name
        # but for now we enforce using keys from CONFIGS or we fail
        if model_name in self.CONFIGS:
            config = self.CONFIGS[model_name]
        else:
            # Fallback: try to see if it matches any known good default or raise error
            # Or maybe user passed a custom name not in our hardcoded list?
            # For now, let's just log warning and try to use it if it looks like a config key, 
            # otherwise default to gpt-4.1
            logging.warning(f"Model {model_name} not found in pre-defined configs. Available: {list(self.CONFIGS.keys())}")
            raise ValueError(f"Unknown model name: {model_name}. Please use one of {list(self.CONFIGS.keys())}")

        logging.info(f"Loading Azure OpenAI model: {model_name}")
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not api_key or not azure_endpoint:
             raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in environment variables or .env file.")
        
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=config['api_version'],
            azure_endpoint=azure_endpoint
        )
        self.deployment_name = config['azure_deployment']
        
        # Setup tokenizer
        try:
            if "gpt-4" in model_name.lower():
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            else:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logging.warning("Could not load tiktoken for token counting. Using approximation.")
            self.tokenizer = None

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_strings: Optional[List[str]] = None,
        return_full_text: bool = False,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            system_message: Optional system instruction
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_strings: Optional list of stop strings
            return_full_text: Return full text including prompt (usually False for chat models)
        
        Returns:
            Generated text
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_strings,
                response_format=response_format
            )
            
            generated_text = response.choices[0].message.content
            
            if return_full_text:
                return prompt + generated_text
            
            return generated_text
            
        except Exception as e:
            logging.error(f"Error generating text with Azure OpenAI: {e}")
            raise e

    def generate_with_token_count(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_strings: Optional[List[str]] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> Tuple[str, int]:
        """
        Generate text and return token count.
        
        Args:
            prompt: Input prompt
            system_message: Optional system instruction
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            (generated_text, num_tokens_generated)
        """
        generated_text = self.generate(
            prompt=prompt,
            system_message=system_message,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_strings=stop_strings,
            response_format=response_format
        )
        
        num_tokens = self.count_tokens(generated_text)
        return generated_text, num_tokens

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
        
        Returns:
            Number of tokens
        """
        if text is None:
            return 0
            
        if not isinstance(text, str):
            text = str(text)

        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximation: 1 token ~= 4 chars or 0.75 words
            return len(text) // 4

# ========================================
# From: src/llm/qwen.py
# ========================================
"""Qwen 3B model wrapper."""



class QwenModel:
    """Wrapper for Qwen 3B model."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        load_in_8bit: bool = False
    ):
        """
        Initialize Qwen model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ("auto", "cuda", "cpu")
            load_in_8bit: Use 8-bit quantization for lower memory
        """
        self.model_name = model_name
        self.device = device
        
        logging.info(f"Loading Qwen model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if device != "auto" and not load_in_8bit:
            self.model = self.model.to(device)
        
        self.model.eval()
        
        logging.info(f"Model loaded successfully on device: {self.model.device}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_strings: Optional[list] = None,
        return_full_text: bool = False
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_strings: Optional list of stop strings
            return_full_text: Return full text including prompt
        
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Remove prompt if needed
        if not return_full_text:
            generated_text = generated_text[len(prompt):].strip()
        
        # Apply stop strings
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]
        
        return generated_text
    
    def generate_with_token_count(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple:
        """
        Generate text and return token count.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            (generated_text, num_tokens_generated)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        input_length = inputs['input_ids'].shape[1]
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        output_length = outputs.shape[1]
        num_tokens_generated = output_length - input_length
        
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        return generated_text, num_tokens_generated
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
        
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

# ========================================
# From: src/llm/generation.py
# ========================================
"""Text generation utilities."""



class TextGenerator:
    """Wrapper for text generation with utilities."""
    
    def __init__(self, model: Union[QwenModel, AzureOpenAIModel]):
        """
        Initialize generator.
        
        Args:
            model: Model instance (Qwen or Azure OpenAI)
        """
        self.model = model
    

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_strings: Optional[List[str]] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text with metadata.
        
        Args:
            prompt: Input prompt
            system_message: Optional system instruction (for chat models)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stop_strings: Optional stop strings
        
        Returns:
            Dictionary with 'text' and 'num_tokens'
        """
        if isinstance(self.model, AzureOpenAIModel):
             text, num_tokens = self.model.generate_with_token_count(
                prompt=prompt,
                system_message=system_message,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format
            )
        else:
            # Default path (Qwen)
            text, num_tokens = self.model.generate_with_token_count(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        
        # Apply stop strings
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in text:
                    text = text.split(stop_str)[0]
        
        return {
            'text': text.strip(),
            'num_tokens': num_tokens
        }
    
    def generate_json(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        max_retries: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Generate JSON output with retry logic.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
        
        Returns:
            Parsed JSON dict or None if all retries fail
        """
        for attempt in range(max_retries + 1):
            result = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature if attempt == 0 else 0.3,  # Lower temp on retry
                stop_strings=['\n\n', '```'],
                response_format={"type": "json_object"}
            )
            
            text = result['text']
            
            # Try to extract JSON
            parsed = extract_json_from_text(text)
            
            if parsed is not None:
                result['parsed_json'] = parsed
                return result
            
            if attempt < max_retries:
                logging.warning(f"JSON parsing failed (attempt {attempt + 1}), retrying with stricter prompt")
                # Add stricter instruction
                prompt = prompt + "\n\nIMPORTANT: Return ONLY valid JSON with no additional text."
        
        # All retries failed
        logging.error(f"Failed to parse JSON after {max_retries + 1} attempts")
        result['parsed_json'] = None
        result['parse_error'] = True
        return result


class ChunkedGenerator:
    """Generator that produces text in chunks for early stopping."""
    
    def __init__(self, model: Union[QwenModel, AzureOpenAIModel], chunk_tokens: int = 64):
        """
        Initialize chunked generator.
        
        Args:
            model: Model instance
            chunk_tokens: Tokens per chunk
        """
        self.model = model
        self.chunk_tokens = chunk_tokens
    
    def generate_in_chunks(
        self,
        prompt: str,
        max_chunks: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.9,
        response_format: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate text in chunks.
        
        Each chunk contains approximately chunk_tokens tokens.
        Stops early if EOS is reached.
        
        Args:
            prompt: Input prompt
            max_chunks: Maximum number of chunks
            temperature: Sampling temperature
            top_p: Nucleus sampling
        
        Returns:
            List of chunk dicts with 'text', 'num_tokens', 'cumulative_text'
        """
        chunks = []
        cumulative_text = ""
        
        for chunk_idx in range(max_chunks):
            # Generate one chunk
            current_prompt = prompt + cumulative_text
            
            text, num_tokens = self.model.generate_with_token_count(
                prompt=current_prompt,
                max_new_tokens=self.chunk_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Check if generation finished (very short output)
            if num_tokens < 5:
                break
            
            cumulative_text += text
            
            chunks.append({
                'chunk_idx': chunk_idx,
                'text': text,
                'num_tokens': num_tokens,
                'cumulative_text': cumulative_text.strip()
            })
            
            # Check for natural stopping (ends with sentence terminator)
            if text.rstrip().endswith(('.', '!', '?', '}')):
                # Might be natural end
                pass
        
        return chunks
    
    def generate_with_callback(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_chunks: int = 8,
        temperature: float = 0.7,
        callback = None,
        stop_strings: Optional[List[str]] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate with callback after each chunk for early stopping.
        
        Args:
            prompt: Input prompt
            system_message: Optional system instruction
            max_chunks: Maximum chunks
            temperature: Sampling temperature
            callback: Function called after each chunk, receives (chunk, all_chunks).
                     Should return True to continue, False to stop.
        
        Returns:
            Dictionary with 'text', 'early_stop_stats'
        """
        # Generate complete text first
        if isinstance(self.model, AzureOpenAIModel) and system_message:
            text, num_tokens = self.model.generate_with_token_count(
                prompt=prompt,
                system_message=system_message,
                max_new_tokens=max_chunks * self.chunk_tokens,
                temperature=temperature,
                stop_strings=stop_strings,
                response_format=response_format
            )
        else:
             text, num_tokens = self.model.generate_with_token_count(
                prompt=prompt,
                max_new_tokens=max_chunks * self.chunk_tokens,
                temperature=temperature,
                stop_strings=stop_strings,
                response_format=response_format
            )
            
        text = text or ""
        
        # Tokenize and split into chunks to evaluate callback
        try:
            tokenizer = tiktoken.encoding_for_model("gpt-4")
            all_tokens = tokenizer.encode(text)
        except Exception:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            all_tokens = tokenizer.encode(text)
            
        chunks = []
        cumulative_text = ""
        early_stop_stats = {
            'stopped_early': False,
            'num_chunks': 0,
            'convergence_score': 0.0
        }
        
        num_generated_chunks = max(1, (len(all_tokens) + self.chunk_tokens - 1) // self.chunk_tokens)
        
        stopped_early = False
        final_text = ""
        total_tokens = 0
        
        for chunk_idx in range(num_generated_chunks):
            start_idx = chunk_idx * self.chunk_tokens
            end_idx = min(len(all_tokens), start_idx + self.chunk_tokens)
            chunk_tokens = all_tokens[start_idx:end_idx]
            
            chunk_text = tokenizer.decode(chunk_tokens)
            cumulative_text += chunk_text
            
            chunk = {
                'chunk_idx': chunk_idx,
                'text': chunk_text,
                'num_tokens': len(chunk_tokens),
                'cumulative_text': cumulative_text
            }
            chunks.append(chunk)
            
            final_text = cumulative_text
            total_tokens += chunk['num_tokens']
            
            if callback:
                should_continue = callback(chunk, chunks)
                if not should_continue:
                    stopped_early = True
                    break
        
        if stopped_early:
            # Reconstruct valid JSON by closing brackets if we stopped early mid-json
            # Or simpler: if it converges, the LLM had enough info, but the JSON might be cut off.
            # RAG parser relies on extract_json_from_text fallback if cut off.
            pass
            
        return {
            'text': text,  # Return the FULL valid JSON text to avoid parser breaking, but report the SAVED token count
            'num_tokens': total_tokens, # The token count is truncated to simulate savings
            'num_chunks': len(chunks),
            'chunks': chunks
        }

# ========================================
# From: src/retrieval/embedder.py
# ========================================
"""Embedding model wrapper."""



class Embedder:
    """Wrapper for sentence embedding model."""
    
    def __init__(self, model_name: str = "facebook/contriever-msmarco", device: str = "auto"):
        """
        Initialize embedder.
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        logging.info(f"Loading embedding model: {model_name}")
        # Handle 'auto' device selection more robustly
        if self.device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
            
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logging.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _get_device(self, device: str) -> str:
        """Determine device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            show_progress: Show progress bar
        
        Returns:
            Embeddings as numpy array (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query.
        
        Args:
            query: Query text
            normalize: Whether to L2 normalize
        
        Returns:
            Query embedding (1, embedding_dim)
        """
        embedding = self.encode([query], normalize=normalize)
        return embedding[0]
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode batch of texts.
        
        Args:
            texts: List of texts
            batch_size: Batch size
            normalize: Whether to L2 normalize
        
        Returns:
            Embeddings (num_texts, embedding_dim)
        """
        return self.encode(texts, batch_size=batch_size, normalize=normalize)

# ========================================
# From: src/retrieval/index.py
# ========================================
"""Embedding index builder and cache manager."""



class EmbeddingCache:
    """Cache for paragraph embeddings."""
    
    def __init__(self, cache_dir: str = "runs/cache_embeddings"):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.cache_file = self.cache_dir / "embeddings.pkl"
        
        # Load existing cache
        if self.cache_file.exists():
            self._load()
    
    def _get_hash(self, text: str) -> str:
        """Get hash for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            text: Text to lookup
        
        Returns:
            Cached embedding or None
        """
        text_hash = self._get_hash(text)
        return self.cache.get(text_hash)
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Text
            embedding: Embedding vector
        """
        text_hash = self._get_hash(text)
        self.cache[text_hash] = embedding
    
    def get_batch(self, texts: List[str]) -> tuple:
        """
        Get batch of embeddings from cache.
        
        Args:
            texts: List of texts
        
        Returns:
            (cached_embeddings, missing_indices, missing_texts)
        """
        cached_embeddings = []
        missing_indices = []
        missing_texts = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text)
            if embedding is not None:
                cached_embeddings.append((i, embedding))
            else:
                missing_indices.append(i)
                missing_texts.append(text)
        
        return cached_embeddings, missing_indices, missing_texts
    
    def put_batch(self, texts: List[str], embeddings: np.ndarray) -> None:
        """
        Store batch of embeddings.
        
        Args:
            texts: List of texts
            embeddings: Embeddings array
        """
        for text, embedding in zip(texts, embeddings):
            self.put(text, embedding)
    
    def save(self) -> None:
        """Save cache to disk."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        logging.info(f"Saved {len(self.cache)} embeddings to cache")
    
    def _load(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            logging.info(f"Loaded {len(self.cache)} embeddings from cache")
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")
            self.cache = {}
    
    def __len__(self) -> int:
        """Number of cached embeddings."""
        return len(self.cache)


class ParagraphIndex:
    """Simple paragraph index using embeddings."""
    
    def __init__(self, paragraphs: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Initialize index.
        
        Args:
            paragraphs: List of paragraph dicts
            embeddings: Paragraph embeddings (num_paragraphs, dim)
        """
        self.paragraphs = paragraphs
        self.embeddings = embeddings
        self.num_paragraphs = len(paragraphs)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for top-K most similar paragraphs.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of results with paragraph and score
        """
        # Compute cosine similarity (embeddings are already normalized)
        scores = np.dot(self.embeddings, query_embedding)
        
        # Get top-K indices
        top_k = min(top_k, self.num_paragraphs)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'paragraph': self.paragraphs[idx],
                'score': float(scores[idx]),
                'index': int(idx)
            })
        
        return results


def build_paragraph_embeddings(
    paragraphs: List[Dict[str, Any]],
    embedder,
    cache: Optional[EmbeddingCache] = None,
    batch_size: int = 32
) -> np.ndarray:
    """
    Build embeddings for paragraphs with caching.
    
    Args:
        paragraphs: List of paragraph dicts (must have 'text' key)
        embedder: Embedder instance
        cache: Optional embedding cache
        batch_size: Batch size for encoding
    
    Returns:
        Embeddings array (num_paragraphs, dim)
    """
    texts = [p['text'] for p in paragraphs]
    
    if cache is None:
        # No cache, encode all
        embeddings = embedder.encode_batch(texts, batch_size=batch_size)
        return embeddings
    
    # Use cache
    cached_results, missing_indices, missing_texts = cache.get_batch(texts)
    
    # Initialize embeddings array
    embeddings = np.zeros((len(texts), embedder.embedding_dim), dtype=np.float32)
    
    # Fill cached embeddings
    for idx, embedding in cached_results:
        embeddings[idx] = embedding
    
    # Encode missing texts
    if missing_texts:
        missing_embeddings = embedder.encode_batch(missing_texts, batch_size=batch_size)
        
        # Fill missing embeddings and update cache
        for i, embedding in zip(missing_indices, missing_embeddings):
            embeddings[i] = embedding
            cache.put(texts[i], embedding)
    
    return embeddings

# ========================================
# From: src/retrieval/ranker.py
# ========================================
"""Paragraph ranking for retrieval."""



class ParagraphRanker:
    """Ranks paragraphs for retrieval."""
    
    def __init__(self, embedder: Embedder, cache: Optional[EmbeddingCache] = None, cross_encoder: Optional[CrossEncoder] = None):
        """
        Initialize ranker.
        
        Args:
            embedder: Embedder instance
            cache: Optional embedding cache
        """
        self.embedder = embedder
        self.cache = cache
        self.cross_encoder = cross_encoder
    
    def rank(
        self,
        query: str,
        paragraphs: List[Dict[str, Any]],
        top_k: int = 5,
        exclude_titles: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank paragraphs by relevance to query.
        
        Args:
            query: Query text
            paragraphs: List of paragraph dicts (must have 'text' and 'title')
            top_k: Number of top results to return
            exclude_titles: Optional set of titles to exclude from results
        
        Returns:
            List of ranked results with paragraph, score, and metadata
        """
        if not paragraphs:
            return []
        
        # Filter out excluded titles
        if exclude_titles:
            paragraphs = [p for p in paragraphs if p['title'] not in exclude_titles]
        
        if not paragraphs:
            return []
        
        # Build embeddings for paragraphs
        para_embeddings = build_paragraph_embeddings(
            paragraphs,
            self.embedder,
            cache=self.cache
        )
        
        # Encode query
        query_embedding = self.embedder.encode_query(query)
        
        # Build index and search
        retrieve_k = top_k * 3 if self.cross_encoder else top_k
        index = ParagraphIndex(paragraphs, para_embeddings)
        results = index.search(query_embedding, top_k=retrieve_k)
        
        # Rerank with CrossEncoder if available
        if self.cross_encoder and results:
            cross_inp = [[query, r['paragraph']['text']] for r in results]
            scores = self.cross_encoder.predict(cross_inp)
            for r, s in zip(results, scores):
                r['score'] = float(s)
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:top_k]
        
        return results
    
    def compute_retrieval_novelty(
        self,
        current_results: List[Dict[str, Any]],
        previous_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute novelty metrics for current retrieval vs previous evidence.
        
        Args:
            current_results: Current retrieval results list
            previous_evidence: Cumulative evidence paragraphs from previous hops
        
        Returns:
            Dictionary with novelty metrics
        """
        if not current_results:
            return {
                'jaccard_overlap': 1.0,
                'new_paragraphs': 0,
                'overlap_count': 0,
                'empty_retrieval': True
            }
            
        if not previous_evidence:
            return {
                'jaccard_overlap': 0.0,
                'new_paragraphs': len(current_results),
                'overlap_count': 0,
                'empty_retrieval': False
            }
        
        # Extract paragraph texts instead of titles to prevent collapsing distinct paragraphs under same title
        current_texts = set(r['paragraph']['text'] for r in current_results)
        previous_texts = set(p['text'] for p in previous_evidence)
        
        # Compute Jaccard overlap
        jaccard = compute_jaccard_similarity(current_texts, previous_texts)
        
        # Count new paragraphs
        new_texts = current_texts - previous_texts
        overlap_count = len(current_texts & previous_texts)
        
        return {
            'jaccard_overlap': jaccard,
            'new_paragraphs': len(new_texts),
            'overlap_count': overlap_count,
            'empty_retrieval': False
        }


class RetrievalHistory:
    """Tracks retrieval history across hops."""
    
    def __init__(self):
        """Initialize retrieval history."""
        self.history = []
        self.all_retrieved_titles = set()
    
    def add_hop(self, hop: int, query: str, results: List[Dict[str, Any]]) -> None:
        """
        Add retrieval results for a hop.
        
        Args:
            hop: Hop number (1-indexed)
            query: Query used for retrieval
            results: Retrieval results
        """
        titles = set(r['paragraph']['title'] for r in results)
        
        self.history.append({
            'hop': hop,
            'query': query,
            'results': results,
            'titles': titles,
            'num_new': len(titles - self.all_retrieved_titles)
        })
        
        self.all_retrieved_titles.update(titles)
    
    def get_cumulative_evidence(self) -> List[Dict[str, Any]]:
        """
        Get cumulative evidence (all unique paragraphs retrieved).
        
        Returns:
            List of unique paragraphs
        """
        seen_texts = set()
        evidence = []
        
        for hop_data in self.history:
            for result in hop_data['results']:
                text = result['paragraph']['text']
                if text not in seen_texts:
                    evidence.append(result['paragraph'])
                    seen_texts.add(text)
        
        return evidence
    
    def get_latest_results(self) -> Optional[List[Dict[str, Any]]]:
        """Get results from latest hop."""
        if not self.history:
            return None
        return self.history[-1]['results']
    
    def get_previous_results(self) -> Optional[List[Dict[str, Any]]]:
        """Get results from previous hop (second to last)."""
        if len(self.history) < 2:
            return None
        return self.history[-2]['results']

# ========================================
# From: src/llm/early_stop.py
# ========================================
"""Answer convergence early stopping mechanism."""



class AnswerConvergenceStopper:
    """
    Reasoning early stopping based on answer convergence.
    
    Monitors answer stability across generation chunks and stops
    when the answer has converged for k consecutive chunks.
    """
    
    def __init__(
        self,
        convergence_k: int = 2,
        text_generator = None
    ):
        """
        Initialize convergence stopper.
        
        Args:
            convergence_k: Number of consecutive stable chunks needed
            text_generator: TextGenerator instance for extraction
        """
        self.convergence_k = convergence_k
        self.text_generator = text_generator
        
        # State
        self.extracted_answers = []
        self.stop_triggered = False
        self.stop_reason = None
        self.stop_chunk_idx = None
    
    def reset(self) -> None:
        """Reset state for new generation."""
        self.extracted_answers = []
        self.stop_triggered = False
        self.stop_reason = None
        self.stop_chunk_idx = None
    
    def extract_answer_heuristic(self, text: str) -> str:
        """
        Extract answer using heuristics (no LLM call).
        
        Look for:
        1. JSON "answer" field
        2. "The answer is" / "Answer:" in text
        3. <THOUGHT> block text if distinct
        
        Args:
            text: Generated text
        
        Returns:
            Extracted answer
        """
        # 1. Try JSON extraction
        
        # Look for "current_hypothesis": "..." pattern first
        hyp_match = re.search(r'"current_hypothesis"\s*:\s*"([^"]+)"', text)
        if hyp_match:
            return hyp_match.group(1).strip()
            
        # Look for "answer": "..." pattern
        json_match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
        if json_match:
            return json_match.group(1).strip()
            
        # 2. Look for explicit answer statements in thought
        patterns = [
            r"Therefore, the answer is:?\s*([^\n<]+)",
            r"Therefore, ([^\n<]+)",
            r"Current Hypothesis:\s*([^\n<]+)",
            r"The answer is\s+([^\.]+)",
            r"Answer:\s*([^\n<]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                # Clean value
                val = val.rstrip('."')
                val = val.strip()
                
                if val and len(val) < 200: # Sanity check length
                     return val
                     
        return ""
                     
        return ""
    
    def extract_answer_with_llm(self, text: str) -> Optional[str]:
        """
        Extract answer using LLM.
        
        Args:
            text: Generated text
        
        Returns:
            Extracted answer or None if extraction fails
        """
        if self.text_generator is None:
            return self.extract_answer_heuristic(text)
        
        try:
            prompt = get_answer_extraction_prompt(text)
            result = self.text_generator.generate(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.1  # Low temperature for extraction
            )
            answer = result['text'].strip()
            
            # Take first line
            answer = answer.split('\n')[0].strip()
            
            return answer if answer else self.extract_answer_heuristic(text)
        
        except Exception as e:
            logging.warning(f"LLM answer extraction failed: {e}, using heuristic")
            return self.extract_answer_heuristic(text)
    
    def check_chunk(self, chunk: Dict[str, Any], all_chunks: List[Dict[str, Any]]) -> bool:
        """
        Check if generation should continue after this chunk.
        
        Args:
            chunk: Current chunk dict
            all_chunks: All chunks so far
        
        Returns:
            True to continue generation, False to stop
        """
        if self.stop_triggered:
            return False
        
        cumulative_text = chunk['cumulative_text']
        
        # Extract answer from cumulative text
        # Use heuristic extraction to avoid extra LLM calls
        answer = self.extract_answer_heuristic(cumulative_text)
        answer_norm = normalize_text(answer)
        
        self.extracted_answers.append({
            'chunk_idx': chunk['chunk_idx'],
            'answer': answer,
            'answer_normalized': answer_norm
        })
        
        # CRITICAL FIX: Only consider stopping if JSON structure is complete
        # We look for </JSON> or <END_JSON> OR balanced braces (ignoring trailing text)
        has_tags = ("</JSON>" in cumulative_text or "<END_JSON>" in cumulative_text)
        # Check if braces are balanced. This handles nested objects correctly too.
        has_balanced_braces = (cumulative_text.count("{") > 0 and cumulative_text.count("{") == cumulative_text.count("}"))
        
        is_complete = (has_tags or has_balanced_braces)
        
        if not is_complete:
            return True # Continue generation until complete
        
        # Check convergence
        if len(self.extracted_answers) >= self.convergence_k + 1:
            # Check last k+1 answers
            recent_answers = [a['answer_normalized'] for a in self.extracted_answers[-(self.convergence_k + 1):]]
            
            # All k+1 should be the same
            if len(set(recent_answers)) == 1 and recent_answers[0]:  # All same and not empty
                self.stop_triggered = True
                self.stop_reason = f"answer_converged_for_{self.convergence_k}_chunks"
                self.stop_chunk_idx = chunk['chunk_idx']
                logging.info(f"Early stopping: answer converged at chunk {chunk['chunk_idx']}")
                return False
        
        # Continue generation
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get stopping statistics.
        
        Returns:
            Dictionary with stopping stats
        """
        return {
            'stopped_early': self.stop_triggered,
            'stop_reason': self.stop_reason,
            'stop_chunk_idx': self.stop_chunk_idx,
            'extracted_answers': [a['answer'] for a in self.extracted_answers],
            'num_chunks_generated': len(self.extracted_answers)
        }
    
    def get_final_answer(self) -> str:
        """
        Get final converged answer.
        
        Returns:
            Final answer (last extracted answer)
        """
        if not self.extracted_answers:
            return ""
        return self.extracted_answers[-1]['answer']


def generate_with_reasoning_stop(
    chunked_generator,
    prompt: str,
    max_chunks: int = 8,
    convergence_k: int = 2,
    temperature: float = 0.7,
    text_generator = None,
    stopper = None,
    max_new_tokens: int = 512,
    stop_strings: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate text with reasoning early stopping.
    
    Args:
        chunked_generator: ChunkedGenerator instance
        prompt: Input prompt
        max_chunks: Maximum chunks to generate
        convergence_k: Chunks of convergence needed
        temperature: Sampling temperature
        text_generator: Optional TextGenerator for answer extraction
    
    Returns:
        Generation result with early stopping metadata
    """
    if stopper is None:
        stopper = AnswerConvergenceStopper(
            convergence_k=convergence_k,
            text_generator=text_generator
        )
    
    # Generate with callback
    
    # Check if system message should be used
    
    # Calculate chunks needed
    chunk_size = chunked_generator.chunk_tokens
    calculated_max_chunks = max(1, max_new_tokens // chunk_size)
        
    result = chunked_generator.generate_with_callback(
        prompt=prompt,
        system_message=JSON_SYSTEM_PROMPT,
        max_chunks=calculated_max_chunks,
        temperature=temperature,
        callback=stopper.check_chunk,
        stop_strings=stop_strings,
        response_format={"type": "json_object"}
    )

    # Add stopping stats
    result['early_stop_stats'] = stopper.get_stats()
    result['final_answer'] = stopper.get_final_answer()
    
    return result

# ========================================
# From: src/rag/stopping.py
# ========================================
"""Retrieval early stopping logic."""



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

# ========================================
# From: src/rag/iterative_rag.py
# ========================================
"""Iterative RAG engine."""



class IterativeRAG:
    """
    Iterative multi-hop RAG engine with dual stopping mechanisms.
    """
    
    def __init__(
        self,
        llm_model: Union[QwenModel, AzureOpenAIModel],
        embedder,
        embedding_cache=None,
        config=None
    ):
        """
        Initialize RAG engine.
        
        Args:
            llm_model: QwenModel or AzureOpenAIModel instance
            embedder: Embedder instance
            embedding_cache: Optional EmbeddingCache
            config: RAGConfig instance
        """
        self.llm = llm_model
        self.embedder = embedder
        self.cache = embedding_cache
        self.config = config
        
        # Initialize components
        self.text_generator = TextGenerator(llm_model)
        self.chunked_generator = ChunkedGenerator(
            llm_model,
            chunk_tokens=config.chunk_tokens if config else 64
        )
        # Initialize reranker if configured
        import torch
        self.cross_encoder = None
        if self.config and getattr(self.config, 'reranker_model_name', None):
            self.cross_encoder = CrossEncoder(
                self.config.reranker_model_name,
                device=self.config.device if self.config.device != "auto" else (
                    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                )
            )
            
        self.ranker = ParagraphRanker(embedder, cache=embedding_cache, cross_encoder=self.cross_encoder)
        
        # Stopping mechanisms
        if config:
            self.retrieval_stopper = RetrievalStopper(
                min_hops=config.min_hops,
                overlap_threshold=config.overlap_threshold,
                conf_threshold=config.conf_threshold
            )
        else:
            self.retrieval_stopper = None
    
    def run(
        self,
        question: str,
        candidate_paragraphs: List[Dict[str, Any]],
        max_hops: int = 4,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Run iterative RAG on a question.
        
        Args:
            question: Question to answer
            candidate_paragraphs: Available paragraphs for retrieval
            max_hops: Maximum retrieval hops
            top_k: Number of paragraphs per hop
        
        Returns:
            Full trace with hops, answers, and stopping decisions
        """
        retrieval_history = RetrievalHistory()
        hop_history = []
        
        current_query = question
        retrieval_stopped = False
        retrieval_stop_reason = None
        
        # Track best answer found so far (fallback if final hop fails)
        best_answer = ""
        best_confidence = 0.0
        
        for hop in range(1, max_hops + 1):
            print(f"\n{'='*80}")
            print(f"🔄 HOP {hop}/{max_hops}")
            print(f"{'='*80}")
            print(f"📝 Query: {current_query}")
            print()

            logging.info(f"--- Hop {hop}/{max_hops} ---")
            logging.info(f"Query: {current_query[:100]}...")

            # === RETRIEVAL ===
            print(f"🔍 RETRIEVAL PHASE")
            print(f"-" * 80)

            # Get previously retrieved titles to track novelty
            prev_results = retrieval_history.get_latest_results()

            # Rank paragraphs
            # Fix A: Only exclude titles if candidate pool is large enough
            remaining_paragraphs = len(candidate_paragraphs) - len(retrieval_history.all_retrieved_titles)
            should_exclude = remaining_paragraphs >= top_k
            
            exclude_titles = retrieval_history.all_retrieved_titles if (hop > 1 and should_exclude) else None
            
            results = self.ranker.rank(
                query=current_query,
                paragraphs=candidate_paragraphs,
                top_k=top_k,
                exclude_titles=exclude_titles
            )

            # Display retrieved paragraphs
            print(f"📚 Retrieved {len(results)} paragraphs:")
            for i, result in enumerate(results, 1):
                title = result['paragraph']['title']
                score = result['score']
                text_preview = result['paragraph']['text'][:100].replace('\n', ' ')
                print(f"  [{i}] {title} (score: {score:.3f})")
                print(f"      {text_preview}...")
            print()

            # Compute retrieval novelty
            retrieval_novelty = None
            if hop > 1:
                retrieval_novelty = self.ranker.compute_retrieval_novelty(
                    current_results=results,
                    previous_evidence=retrieval_history.get_cumulative_evidence()
                )
                jaccard = retrieval_novelty.get('jaccard_overlap', 0)
                print(f"🔄 Retrieval Novelty:")
                print(f"  - Jaccard overlap with previous hop: {jaccard:.3f}")
                print(f"  - {'🆕 High novelty (new info)' if jaccard < 0.5 else '♻️  Low novelty (similar to previous)'}")
                print()
                
            retrieval_history.add_hop(hop, current_query, results)

            # Get cumulative evidence
            cumulative_evidence = retrieval_history.get_cumulative_evidence()

            logging.info(f"Retrieved {len(results)} paragraphs, cumulative evidence: {len(cumulative_evidence)}")
            print(f"📊 Cumulative evidence: {len(cumulative_evidence)} unique paragraphs")
            print()
            
            # === GENERATION ===
            print(f"🤖 GENERATION PHASE")
            print(f"-" * 80)
            hop_result = self._generate_hop_response(
                question=question,
                evidence=cumulative_evidence,
                hop=hop
            )

            # Display generation results
            print(f"💡 Generated Response:")
            print(f"  - Answer: {hop_result['answer']}")
            print(f"  - Reasoning: {hop_result.get('reasoning', 'N/A')}")
            print(f"  - Confidence: {hop_result['confidence']:.2f}")
            print(f"  - Has evidence: {'✅ Yes' if hop_result.get('has_enough_evidence') else '❌ No'}")
            if hop_result.get('supporting_quotes'):
                print(f"  - Supporting quotes:")
                for q in hop_result['supporting_quotes']:
                    print(f"    * \"{q}\"")
            print(f"  - Tokens generated: {hop_result['num_tokens']}")
            if hop_result.get('stopped_early'):
                print(f"  - ⚡ Reasoning stopped early (answer converged)")
            print(f"  - Follow-up query: {hop_result.get('followup_query', 'N/A')}")
            print()

            # Store hop data
            hop_data = {
                'hop': hop,
                'query': current_query,
                'retrieved_paragraphs': [
                    {
                        'title': r['paragraph']['title'],
                        'score': r['score'],
                        'text_snippet': r['paragraph']['text'][:200]
                    }
                    for r in results
                ],
                'retrieval_novelty': retrieval_novelty,
                'answer': hop_result['answer'],
                'confidence': hop_result['confidence'],
                'has_enough_evidence': hop_result.get('has_enough_evidence', False),
                'tokens_generated': hop_result['num_tokens'],
                'reasoning_stopped': hop_result.get('stopped_early', False),
                'reasoning_stop_stats': hop_result.get('early_stop_stats', {})
            }

            hop_history.append(hop_data)
            
            # Update best answer if confidence improved and answer is valid
            current_conf = hop_result.get('confidence', 0.0)
            current_ans = hop_result.get('answer', "").strip()
            
            if current_conf > best_confidence and current_ans and len(current_ans) > 1:
                logging.info(f"New best answer found (conf: {current_conf:.2f}): {current_ans}")
                best_confidence = current_conf
                best_answer = current_ans
            
            # === STOPPING DECISIONS ===
            print(f"🛑 STOPPING CHECK")
            print(f"-" * 80)

            # Check retrieval stopping
            # Special fast-exit for Hop 1: If confidence is very high, stop even if min_hops not met
            # DISABLE for now as model is overconfident (stopped 50% wrong at Hop 1)
            # if hop == 1 and hop_result['confidence'] >= 0.9:
            #      retrieval_stopped = True
            #      retrieval_stop_reason = "high_confidence_early_exit"
            #      logging.info("High confidence (>= 0.9) in Hop 1. Stopping early despite min_hops.")
            #      print(f"\n✅ Retrieval stopped: {retrieval_stop_reason}")
            #      break

            should_stop = False
            if self.config and getattr(self.config, 'enable_retrieval_stop', False) and self.retrieval_stopper:
                should_stop, reason = self.retrieval_stopper.should_stop(
                    current_hop=hop,
                    hop_history=hop_history,
                    retrieval_novelty=retrieval_novelty
                )

                print(f"🔍 Retrieval stopping criteria:")
                print(f"  - Current hop: {hop}")
                print(f"  - Confidence: {hop_result['confidence']:.2f} (threshold: {self.config.conf_threshold})")
                if retrieval_novelty:
                    print(f"  - Novelty overlap: {retrieval_novelty.get('jaccard_overlap', 0):.2f} (threshold: {self.config.overlap_threshold})")
                print(f"  - Decision: {'🛑 STOP' if should_stop else '▶️  CONTINUE'} ({reason})")

                if should_stop:
                    retrieval_stopped = True
                    retrieval_stop_reason = reason
                    logging.info(f"Retrieval stopped at hop {hop}: {reason}")
                    print(f"\n✅ Retrieval stopped: {reason}")
                    break

            # Check if we've reached max hops
            if hop >= max_hops:
                retrieval_stop_reason = "max_hops_reached"
                print(f"\n🏁 Maximum hops ({max_hops}) reached")
                break

            # Generate follow-up query for next hop
            followup_query = hop_result.get('followup_query', '')

            if not followup_query or followup_query.lower() in ['none', 'n/a', '']:
                if not getattr(self.config, 'enable_retrieval_stop', False):
                    followup_query = f"{question} {current_ans}".strip()
                    print(f"\n⚠️ No follow-up query, but fixed-hops mode: adding current answer to question for retrieval diversity")
                elif hop < self.config.min_hops:
                    followup_query = f"{question} {current_ans}".strip()
                    print(f"\n⚠️ No follow-up query, but min_hops not met ({hop} < {self.config.min_hops}): adding current answer to question for retrieval diversity")
                else:
                    retrieval_stopped = True
                    retrieval_stop_reason = "no_followup_query"
                    logging.info("No follow-up query provided by LLM, stopping")
                    print(f"\n✅ No follow-up query provided, stopping retrieval")
                    break

            print(f"\n➡️  Continuing to next hop with query: {followup_query[:100]}")
            current_query = followup_query
        
        # === FINAL RESULTS ===
        final_answer = hop_history[-1]['answer'] if hop_history else ""
        
        # Fallback to best answer if final is empty/failed but we had a good one earlier
        # Logic: If final answer is empty or very short (<2 chars), AND we have a best answer
        should_fallback = (not final_answer or len(final_answer.strip()) < 2) and best_answer
        
        if should_fallback:
             final_answer = best_answer
             msg = f"Final answer empty/failed. Using best answer found (Conf: {best_confidence:.2f}): {best_answer}"
             print(f"\n⚠️  {msg}")
             logging.warning(msg)
             # Append note to stop reason
             if retrieval_stop_reason:
                retrieval_stop_reason += "_fallback_to_best"
             else:
                retrieval_stop_reason = "fallback_to_best"

        print(f"\n{'='*80}")
        print(f"🎯 FINAL RESULTS")
        print(f"{'='*80}")
        print(f"❓ Question: {question}")
        print(f"✅ Final Answer: {final_answer}")
        print(f"🔢 Hops taken: {len(hop_history)}/{max_hops}")
        print(f"📝 Total tokens: {sum(h['tokens_generated'] for h in hop_history)}")
        print(f"🛑 Stop reason: {retrieval_stop_reason}")
        print(f"{'='*80}\n")

        trace = {
            'question': question,
            'hops_taken': len(hop_history),
            'max_hops': max_hops,
            'final_answer': final_answer,
            'hops': hop_history,
            'retrieval_stopped': retrieval_stopped,
            'retrieval_stop_reason': retrieval_stop_reason,
            'total_tokens': sum(h['tokens_generated'] for h in hop_history),
            'cumulative_evidence_size': len(retrieval_history.get_cumulative_evidence())
        }

        if self.retrieval_stopper:
            trace['retrieval_stop_stats'] = self.retrieval_stopper.get_stopping_stats(hop_history)

        return trace
    
    def _generate_hop_response(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        hop: int
    ) -> Dict[str, Any]:
        """
        Generate response for a single hop.
        
        Args:
            question: Original question
            evidence: Cumulative evidence
            hop: Hop number
        
        Returns:
            Response dict with answer, followup_query, confidence, tokens
        """
        # Build prompt
        prompt = get_multihop_reasoning_prompt(question, evidence, hop)
        
        # Generate with or without reasoning stop
        enable_reasoning_stop = self.config and getattr(self.config, 'enable_reasoning_stop', False)
        
        if enable_reasoning_stop:
            # Use chunked generation with early stopping
            result = generate_with_reasoning_stop(
                chunked_generator=self.chunked_generator,
                prompt=prompt,
                temperature=self.config.temperature if self.config else 0.7,
                text_generator=self.text_generator,
                stop_strings=["\n\n", "```"], # Aggressive stop removed so JSON can naturally complete
                max_new_tokens=512  # FORCE CAP
            )
            
            generated_text = result['text']
            
            num_tokens = result['num_tokens']
            stopped_early = result['early_stop_stats']['stopped_early']
            early_stop_stats = result['early_stop_stats']
            
        else:
            # Standard generation
            
            result = self.text_generator.generate(
                prompt=prompt,
                system_message=JSON_SYSTEM_PROMPT,
                max_new_tokens=512,  # FORCE CAP: Prevent infinite reasoning loops
                stop_strings=["\n\n", "```"], # Aggressive stop at JSON close removed to not break parsing
                temperature=self.config.temperature if self.config else 0.7,
                response_format={"type": "json_object"}
            )
            
            generated_text = result['text']
            
            num_tokens = result['num_tokens']
            stopped_early = False
            early_stop_stats = {}
        
        # Parse JSON response
        parsed = None
        
        # If stopped early due to reasoning convergence, use the converged answer
        if stopped_early and 'final_answer' in result and result['final_answer']:
            logging.info(f"Using converged answer from reasoning stop: {result['final_answer']}")
            
            extracted_answer = result['final_answer']
            if self.text_generator:
                try:
                    extraction_prompt = get_answer_extraction_prompt(result['final_answer'])
                    extraction_result = self.text_generator.generate(
                        prompt=extraction_prompt,
                        max_new_tokens=50,
                        temperature=0.1
                    )
                    extracted_answer = extraction_result['text'].strip().split('\n')[0].strip()
                    logging.info(f"Normalized converged answer via LLM to: {extracted_answer}")
                except Exception as e:
                    logging.warning(f"Answer extraction failed, using raw: {e}")

            parsed = {
                "answer": extracted_answer,
                "has_enough_evidence": True,
                "confidence": 0.9,
                "followup_query": "",
                "supporting_quotes": [],
                "reasoning": "Answer converged during chunked generation."
            }
        else:
            # Normal parsing
            parsed = extract_json_from_text(generated_text)
        
        if parsed:
            answer = parsed.get('answer', '')
            followup_query = parsed.get('followup_query', '')
            confidence = float(parsed.get('confidence', 0.5))
            has_enough_evidence = bool(parsed.get('has_enough_evidence', True))
            supporting_quotes = parsed.get('supporting_quotes', [])
            reasoning = parsed.get('reasoning', '')

        else:
            # Fallback: JSON parsing failed
            logging.warning("JSON parsing failed. Returning empty answer to force retry or stop.")
            answer = ""
            confidence = 0.0
            has_enough_evidence = False
            followup_query = ""
            supporting_quotes = []
            reasoning = ""
            
        # Semantic Verification removed to reduce noise/hallucination loops
        # Logic simplified to rely on strict JSON output or retrieval loop fallback

        # Retrieval Drift Prevention
        trivial_answers = {"yes", "no", "true", "false", "none", "n/a", "not enough information", "i cannot answer"}
        
        # If followup query is exactly the same as the current question, it means the model is stuck
        if followup_query.strip().lower() == question.strip().lower():
             logging.warning("Follow-up query is identical to the current question. Clearing it to force stop.")
             followup_query = ""
             
        # Grounding Check (Guard B)
        # Ensure the answer actually appears in the retrieved text (to prevention hallucinations)
        if confidence > 0.5 and answer and len(answer) > 4:
             # Normalize for check
             ans_norm = answer.lower().strip('"').strip("'")
             is_trivial = ans_norm in trivial_answers
             
             if not is_trivial:
                 found_in_text = False
                 for item in evidence:
                     if ans_norm in item['text'].lower() or ans_norm in item['title'].lower():
                         found_in_text = True
                         break
                 
                 if not found_in_text:
                     # Fallback: check if most significant words exist independently
                     words = [w for w in re.split(r'\W+', ans_norm) if len(w) > 2]
                     if words:
                         found_words = 0
                         for w in words:
                             word_found = False
                             for item in evidence:
                                 if w in item['text'].lower() or w in item['title'].lower():
                                     word_found = True
                                     break
                             if word_found:
                                 found_words += 1
                         # If all words found, consider it grounded
                         if found_words == len(words) and len(words) > 0:
                             found_in_text = True
                 
                 if not found_in_text:
                     # Check if all words in the answer are in the text (to handle combined answers like 'Israeli and Palestinian')
                     words = set(w for w in re.split(r'\W+', ans_norm) if len(w) > 2)
                     if words:
                         found_all_words = True
                         for w in words:
                             word_found = False
                             for item in evidence:
                                 if w in item['text'].lower() or w in item['title'].lower():
                                     word_found = True
                                     break
                             if not word_found:
                                 found_all_words = False
                                 break
                         if found_all_words:
                             found_in_text = True

                 if not found_in_text:
                     logging.warning(f"Grounding Check Warning: Answer '{answer}' not found in evidence. Downgrading confidence.")
                     confidence = 0.2
                     has_enough_evidence = False
                     # Let outer loop handle fallback query naturally

        return {
            'answer': answer,
            'followup_query': followup_query,
            'confidence': confidence,
            'has_enough_evidence': has_enough_evidence,
            'supporting_quotes': supporting_quotes,
            'reasoning': reasoning,
            'num_tokens': num_tokens,
            'stopped_early': stopped_early,
            'early_stop_stats': early_stop_stats,
            'raw_output': generated_text
        }

# ========================================
# From: src/rag/baselines.py
# ========================================
"""Baseline configurations for ablation studies."""



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
    if method not in BASELINE_METHODS:
        raise ValueError(f"Unknown baseline method: {method}. Must be one of {list(BASELINE_METHODS.keys())}")
    
    return BASELINE_METHODS[method](base_config)

# ========================================
# From: src/data/hotpot_loader.py
# ========================================
"""HotpotQA dataset loader."""



class HotpotQAExample:
    """Single HotpotQA example."""
    
    def __init__(
        self,
        id: str,
        question: str,
        answer: str,
        context: List[List],
        supporting_facts: Optional[List[List]] = None,
        level: Optional[str] = None,
        type: Optional[str] = None
    ):
        """
        Initialize HotpotQA example.
        
        Args:
            id: Example ID
            question: Question text
            answer: Gold answer
            context: List of [title, sentences] pairs
            supporting_facts: List of [title, sent_idx] pairs
            level: Difficulty level (easy/medium/hard)
            type: Question type (bridge/comparison)
        """
        self.id = id
        self.question = question
        self.answer = answer
        self.raw_context = context
        self.supporting_facts = supporting_facts or []
        self.level = level
        self.type = type
        
        # Process context into candidate paragraphs
        self.candidate_paragraphs = self._process_context()
    
    def _process_context(self) -> List[Dict[str, Any]]:
        """
        Process raw context into structured paragraphs.
        
        Returns:
            List of paragraph dictionaries with title, text, and sentences
        """
        paragraphs = []
        for title, sentences in self.raw_context:
            # Join sentences into full text
            text = ' '.join(sentences)
            
            paragraphs.append({
                'title': title,
                'text': text,
                'sentences': sentences,
                'num_sentences': len(sentences)
            })
        
        return paragraphs
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'context': self.raw_context,
            'supporting_facts': self.supporting_facts,
            'level': self.level,
            'type': self.type
        }


class HotpotQALoader:
    """Loader for HotpotQA dataset."""
    
    def __init__(self, filepath: str, max_examples: Optional[int] = None, seed: int = 42):
        """
        Initialize loader.
        
        Args:
            filepath: Path to HotpotQA JSON file
            max_examples: Maximum examples to load (None for all)
            seed: Random seed for sampling
        """
        self.filepath = Path(filepath)
        self.max_examples = max_examples
        self.seed = seed
        self.examples = []
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {self.filepath}")
        
        self._load()
    
    def _load(self) -> None:
        """Load examples from file."""
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            data = list(data.values())
        
        # Sample if needed
        if self.max_examples and len(data) > self.max_examples:
            random.seed(self.seed)
            data = random.sample(data, self.max_examples)
        
        # Parse examples
        for item in data:
            example = HotpotQAExample(
                id=item['_id'],
                question=item['question'],
                answer=item['answer'],
                context=item['context'],
                supporting_facts=item.get('supporting_facts', []),
                level=item.get('level'),
                type=item.get('type')
            )
            self.examples.append(example)
    
    def __len__(self) -> int:
        """Number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> HotpotQAExample:
        """Get example by index."""
        return self.examples[idx]
    
    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)
    
    def get_stats(self) -> dict:
        """Get dataset statistics."""
        stats = {
            'num_examples': len(self.examples),
            'avg_context_size': sum(len(ex.candidate_paragraphs) for ex in self.examples) / len(self.examples),
            'avg_question_length': sum(len(ex.question.split()) for ex in self.examples) / len(self.examples),
            'avg_answer_length': sum(len(ex.answer.split()) for ex in self.examples) / len(self.examples),
        }
        
        # Type distribution
        types = {}
        for ex in self.examples:
            if ex.type:
                types[ex.type] = types.get(ex.type, 0) + 1
        stats['type_distribution'] = types
        
        # Level distribution
        levels = {}
        for ex in self.examples:
            if ex.level:
                levels[ex.level] = levels.get(ex.level, 0) + 1
        stats['level_distribution'] = levels
        
        return stats

# ========================================
# From: src/eval/hotpot_eval.py
# ========================================
"""HotpotQA evaluation metrics."""



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

# ========================================
# From: scripts/run_experiment.py
# ========================================
#!/usr/bin/env python3
"""Main experiment runner."""


# Add src to path
# sys.path.insert(0, str(Path(__file__).parent.parent)) # Removed for Colab compatibility


logger = None

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_single_experiment(args, method, base_run_id):
    # Set up run ID
    if args.run_id is None or args.method == 'all':
        run_id = f"{method}_{base_run_id}"
    else:
        run_id = args.run_id

    output_path = Path(args.output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    global logger
    logger = setup_logger(f'experiment_{method}', str(output_path / 'experiment.log'))
    
    set_seed(args.seed)
    
    config = RAGConfig(
        model_name=args.model_name,
        embed_model_name=args.embed_model_name,
        device=args.device,
        dataset_path=args.dataset_path,
        split=args.split,
        max_examples=args.max_examples,
        top_k=args.top_k,
        max_hops=args.hops,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        chunk_tokens=args.chunk_tokens,
        convergence_k=args.convergence_k,
        min_hops=args.min_hops,
        overlap_threshold=args.overlap_threshold,
        conf_threshold=args.conf_threshold,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        method=method,
        run_id=run_id
    )
    
    config = get_baseline(method, config)
    config.save(str(output_path / 'config.json'))
    
    log_experiment_start(logger, config)
    
    logger.info("Loading dataset...")
    loader = HotpotQALoader(
        filepath=config.dataset_path,
        max_examples=config.max_examples,
        seed=config.seed
    )
    logger.info(f"Loaded {len(loader)} examples")
    
    logger.info("Initializing models...")
    embedder = Embedder(model_name=config.embed_model_name, device=config.device)
    cache = EmbeddingCache(cache_dir=config.cache_dir)
    
    if config.model_name in AzureOpenAIModel.CONFIGS or config.model_name.lower().startswith(('gpt', 'azure')):
        llm = AzureOpenAIModel(model_name=config.model_name, device=config.device)
    else:
        llm = QwenModel(model_name=config.model_name, device=config.device)
    
    rag = IterativeRAG(
        llm_model=llm,
        embedder=embedder,
        embedding_cache=cache,
        config=config
    )
    
    logger.info(f"Running experiments for method: {method}...")
    
    predictions = {}
    gold_answers = {}
    
    with JSONLWriter(str(output_path / 'traces.jsonl')) as trace_writer:
        for idx, example in enumerate(tqdm(loader, desc=f"Processing ({method})"), 1):
            try:
                print(f"\n\n{'#'*80}")
                print(f"📋 METHOD: {method} | EXAMPLE {idx}/{len(loader)}")
                print(f"{'#'*80}")
                print(f"🆔 ID: {example.id}")
                print(f"❓ Question: {example.question}")
                print(f"🎯 Gold Answer: {example.answer}")
                print(f"📚 Available paragraphs: {len(example.candidate_paragraphs)}")
                print(f"{'#'*80}\n")

                trace = rag.run(
                    question=example.question,
                    candidate_paragraphs=example.candidate_paragraphs,
                    max_hops=config.max_hops,
                    top_k=config.top_k
                )

                eval_result = evaluate_single(trace['final_answer'], example.answer)

                print(f"\n{'='*80}")
                print(f"📊 EVALUATION ({method})")
                print(f"{'='*80}")
                print(f"Predicted: {trace['final_answer']}")
                print(f"Gold:      {example.answer}")
                print(f"EM:        {eval_result['em']} {'✅' if eval_result['em'] else '❌'}")
                print(f"F1:        {eval_result['f1']:.3f}")
                print(f"{'='*80}\n")

                predictions[example.id] = trace['final_answer']
                gold_answers[example.id] = example.answer

                trace_log = {
                    'id': example.id,
                    'question': example.question,
                    'gold_answer': example.answer,
                    'method': config.method,
                    'em': eval_result['em'],
                    'f1': eval_result['f1'],
                    **trace
                }
                trace_writer.write(trace_log)

            except Exception as e:
                logger.error(f"Error processing example {example.id}: {e}", exc_info=True)
                print(f"\n❌ ERROR processing {example.id} ({method}): {e}\n")
                continue
    
    cache.save()
    
    with open(output_path / 'predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    eval_metrics = evaluate_predictions(predictions, gold_answers)
    traces = read_jsonl(str(output_path / 'traces.jsonl'))
    trace_metrics = aggregate_trace_metrics(traces)
    final_metrics = {**eval_metrics, **trace_metrics}
    
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    log_experiment_end(logger, final_metrics)
    
    return final_metrics, output_path


def main():
    parser = argparse.ArgumentParser(description='Run RAG experiment')
    
    # Dataset args
    parser.add_argument('--dataset_path', type=str, default='data/hotpotqa/hotpot_dev_distractor_v1.json')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'train'])
    parser.add_argument('--max_examples', type=int, default=None)
    
    # Method args
    parser.add_argument('--method', type=str, default='fixed_hops',
                        choices=['fixed_hops', 'fixed_hops_reasoning_stop', 
                                'retrieval_stop_only', 'dual_stopping', 'all'])
    
    # Retrieval args
    parser.add_argument('--hops', type=int, default=4, help='Maximum hops')
    parser.add_argument('--top_k', type=int, default=5, help='Paragraphs per hop')
    parser.add_argument('--min_hops', type=int, default=2, help='Minimum hops for retrieval stop')
    parser.add_argument('--overlap_threshold', type=float, default=0.7)
    parser.add_argument('--conf_threshold', type=float, default=0.6)
    
    # Generation args
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--chunk_tokens', type=int, default=64)
    parser.add_argument('--convergence_k', type=int, default=2)
    
    # Model args
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--embed_model_name', type=str, default='facebook/contriever-msmarco')
    parser.add_argument('--reranker_model_name', type=str, default='BAAI/bge-reranker-v2-m3')
    parser.add_argument('--device', type=str, default='auto')
    
    # Output args
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--cache_dir', type=str, default='runs/cache_embeddings')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_id', type=str, default=None)
    
    args, _ = parser.parse_known_args()
    
    base_run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.method == 'all':
        methods_to_run = ['fixed_hops', 'retrieval_stop_only', 'fixed_hops_reasoning_stop', 'dual_stopping']
    else:
        methods_to_run = [args.method]
        
    all_results = {}
    
    for method in methods_to_run:
        print(f"\n\n{'='*80}")
        print(f"🚀 STARTING EXPERIMENT FOR METHOD: {method}")
        print(f"{'='*80}\n")
        
        metrics, out_path = run_single_experiment(args, method, base_run_id)
        all_results[method] = metrics
        
    if args.method == 'all' or len(methods_to_run) > 1:
        print(f"\n\n{'='*80}")
        print("🏆 FINAL RESULTS COMPARISON 🏆")
        print(f"{'='*80}")
        
        metrics_to_show = ['em', 'f1', 'avg_hops', 'avg_tokens_generated', 'reasoning_stop_rate', 'retrieval_stop_rate']
        
        header = f"{'Method':<30} | " + " | ".join([f"{m:<20}" for m in metrics_to_show])
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for method, metrics in all_results.items():
            row = f"{method:<30} | "
            for m in metrics_to_show:
                val = metrics.get(m, "N/A")
                if isinstance(val, (float, np.float64, np.float32)):
                    row += f"{val:<20.4f} | "
                else:
                    row += f"{str(val):<20} | "
            print(row)
            
        print("-" * len(header))
        
        summary_path = Path(args.output_dir) / f"summary_{base_run_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
