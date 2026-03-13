"""Text generation utilities."""

import torch
from typing import Optional, List, Dict, Any, Union
import logging
from src.llm.qwen import QwenModel
from src.llm.azure_openai import AzureOpenAIModel
from src.utils.text import extract_json_from_text


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
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_strings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text with metadata.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stop_strings: Optional stop strings
        
        Returns:
            Dictionary with 'text' and 'num_tokens'
        """
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_strings: Optional[List[str]] = None
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
                top_p=top_p
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
                stop_strings=['\n\n', '```']
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
        top_p: float = 0.9
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
        callback = None
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
        chunks = []
        cumulative_text = ""
        early_stop_stats = {
            'stopped_early': False,
            'num_chunks': 0,
            'convergence_score': 0.0
        }
        
        for chunk_idx in range(max_chunks):
            # Generate one chunk
            current_prompt = prompt + cumulative_text
            
            # Pass system message if available
            if isinstance(self.model, AzureOpenAIModel) and system_message:
                text, num_tokens = self.model.generate_with_token_count(
                    prompt=current_prompt,
                    system_message=system_message,
                    max_new_tokens=self.chunk_tokens,
                    temperature=temperature
                )
            else:
                 text, num_tokens = self.model.generate_with_token_count(
                    prompt=current_prompt,
                    max_new_tokens=self.chunk_tokens,
                    temperature=temperature
                )
            
            # Ensure text is not None
            text = text or ""
            cumulative_text += text
            
            chunk = {
                'chunk_idx': chunk_idx,
                'text': text,
                'num_tokens': num_tokens,
                'cumulative_text': cumulative_text
            }
            
            chunks.append(chunk)
            
            # Call callback
            if callback:
                should_continue = callback(chunk, chunks)
                if not should_continue:
                    break
        
        # Aggregate results
        total_tokens = sum(c['num_tokens'] for c in chunks)
        final_text = chunks[-1]['cumulative_text'] if chunks else ""
        
        return {
            'text': final_text,
            'num_tokens': total_tokens,
            'num_chunks': len(chunks),
            'chunks': chunks
        }
