"""Loader for the HotpotQA distractor-format dataset.

The dataset is the standard HotpotQA benchmark; Stop-RAG evaluates on it but
does not define its schema. Each record contains a question, answer, a list of
``[title, sentences]`` contexts, and ``supporting_facts`` identifying gold
titles and sentence indices. The loader turns each context entry into one
retrievable paragraph while retaining the original sentence list for audits.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import random


class HotpotQAExample:
    """One question, its distractor paragraph pool, and gold support labels.

    Example: a context entry ``["Film", ["Sentence one.", "Sentence two."]]``
    becomes a paragraph with title ``Film`` and joined text, while the original
    sentences remain available in the record.
    """
    
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
    
    def __init__(self, filepath: str, max_examples: Optional[int] = None, seed: int = 42,
                 example_ids: Optional[List[str]] = None):
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
        # Optional frozen evaluation IDs.  When supplied, examples are loaded
        # in exactly this order and no fresh random sample is drawn.
        self.example_ids = list(example_ids) if example_ids is not None else None
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
        
        if self.example_ids is not None:
            by_id = {item.get('_id'): item for item in data}
            missing = [item_id for item_id in self.example_ids if item_id not in by_id]
            if missing:
                raise KeyError(f"Frozen example IDs missing from dataset: {missing[:5]}")
            data = [by_id[item_id] for item_id in self.example_ids]
        # Sample if needed (only when no explicit frozen IDs were supplied).
        elif self.max_examples and len(data) > self.max_examples:
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
