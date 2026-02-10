"""Answer convergence early stopping mechanism."""

from typing import List, Optional, Dict, Any
import logging
from src.utils.text import normalize_text, extract_answer_from_text
from src.llm.prompts import get_answer_extraction_prompt


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
        import json
        import re
        
        # Look for "answer": "..." pattern even if JSON is incomplete
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
    text_generator = None
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
    stopper = AnswerConvergenceStopper(
        convergence_k=convergence_k,
        text_generator=text_generator
    )
    
    # Generate with callback
    
    # Check if system message should be used
    from src.llm.prompts import SYSTEM_PROMPT
    
    result = chunked_generator.generate_with_callback(
        prompt=prompt,
        system_message=SYSTEM_PROMPT,
        max_chunks=max_chunks,
        temperature=temperature,
        callback=stopper.check_chunk
    )
    
    # Add stopping stats
    result['early_stop_stats'] = stopper.get_stats()
    result['final_answer'] = stopper.get_final_answer()
    
    return result
