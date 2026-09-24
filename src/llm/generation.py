"""Text generation and streaming helpers.

The Qwen streaming path exists so a stopping callback can interrupt decoding.
This is an enabling mechanism for experiments such as HALT-CoT, but this file
does not implement HALT-CoT's entropy signal; the current callback checks answer
convergence instead.
"""

import torch
import threading
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
            text, num_tokens = self.model.generate_with_token_count(
                prompt=prompt, system_message=system_message,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, stop_strings=stop_strings, response_format=response_format)

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
                # A blank line is common inside model preambles and JSON
                # string values; using it as a stop marker truncates valid
                # responses before the closing brace.
                stop_strings=['```'],
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
        max_new_tokens: Optional[int] = None,
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
        top_p: float = 0.9,
        callback = None,
        stop_strings: Optional[List[str]] = None,
        response_format: Optional[Dict[str, str]] = None,
        max_new_tokens: Optional[int] = None,
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
        # Qwen supports true token streaming through Transformers.  This path
        # lets the callback stop the underlying ``generate`` call before the
        # configured maximum, so the reported token count reflects work the
        # model actually performed.  Azure keeps the API based path below.
        if isinstance(self.model, QwenModel):
            return self._generate_qwen_streaming(
                prompt=prompt,
                system_message=system_message,
                max_chunks=max_chunks,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                callback=callback,
                stop_strings=stop_strings,
                response_format=response_format,
            )

        # Generate complete text first
        if isinstance(self.model, AzureOpenAIModel) and system_message:
            text, num_tokens = self.model.generate_with_token_count(
                prompt=prompt,
                system_message=system_message,
                max_new_tokens=max_new_tokens or max_chunks * self.chunk_tokens,
                temperature=temperature,
                stop_strings=stop_strings,
                response_format=response_format
            )
        else:
             text, num_tokens = self.model.generate_with_token_count(
                prompt=prompt,
                max_new_tokens=max_new_tokens or max_chunks * self.chunk_tokens,
                temperature=temperature,
                stop_strings=stop_strings,
                response_format=response_format
            )
            
        text = text or ""
        
        # Tokenize and split into chunks to evaluate callback
        from src.llm.azure_openai import tiktoken
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

    def _generate_qwen_streaming(
        self,
        prompt: str,
        max_chunks: int,
        temperature: float,
        top_p: float,
        max_new_tokens: Optional[int] = None,
        callback=None,
        stop_strings: Optional[List[str]] = None,
        response_format: Optional[Dict[str, str]] = None,
        system_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate Qwen output with a real streamer and interrupt flag."""
        # Imports are local so installations that only use Azure do not need
        # the optional Transformers streamer classes at module import time.
        from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

        class _CallbackStop(StoppingCriteria):
            def __init__(self):
                self.stop = False

            def __call__(self, input_ids, scores, **kwargs):
                return self.stop

        class _CountingStreamer(TextIteratorStreamer):
            """Text streamer that counts generated token ids exactly.

            Counting decoded text is unsafe with BPE tokenizers because
            re-encoding a growing prefix can change token boundaries.  The
            streamer receives the actual token tensors, so count those while
            leaving prompt tokens excluded by ``skip_prompt``.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.generated_token_count = 0
                self._prompt_seen = False

            def put(self, value):
                # GenerationMixin sends one prefill tensor before generated
                # ids.  Track it ourselves instead of depending on the
                # private ``next_tokens_are_prompt`` attribute, which varies
                # across Transformers releases.
                if self._prompt_seen:
                    try:
                        self.generated_token_count += int(value.numel())
                    except AttributeError:
                        self.generated_token_count += len(value)
                else:
                    self._prompt_seen = True
                return super().put(value)

        model = self.model
        # Both paths must format identical messages before we compare stopping.
        inputs = model.prepare_inputs(prompt, system_message)
        stop_flag = _CallbackStop()
        streamer = _CountingStreamer(
            model.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=30.0,
        )
        kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens or max_chunks * self.chunk_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": model.tokenizer.pad_token_id,
            "eos_token_id": model.tokenizer.eos_token_id,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([stop_flag]),
        }
        errors = []

        def _run_generate():
            try:
                with torch.inference_mode():
                    model.model.generate(**kwargs)
            except Exception as exc:  # propagate after streamer drains
                errors.append(exc)

        worker = threading.Thread(target=_run_generate, daemon=True)
        worker.start()

        chunks = []
        cumulative_text = ""
        last_boundary_tokens = 0
        stopped_early = False
        previous_text_len = 0

        for piece in streamer:
            cumulative_text += piece
            current_tokens = streamer.generated_token_count
            # Streamer pieces are normally one or a few tokens.  The loop also
            # handles a larger piece without losing chunk callbacks.
            while current_tokens >= last_boundary_tokens + self.chunk_tokens:
                target_tokens = last_boundary_tokens + self.chunk_tokens
                # Pass only the newly emitted text for this callback.  The
                # cumulative form remains available for convergence checks.
                chunk_text = cumulative_text[previous_text_len:]
                previous_text_len = len(cumulative_text)
                chunk = {
                    "chunk_idx": len(chunks),
                    "text": chunk_text,
                    "num_tokens": target_tokens - last_boundary_tokens,
                    "cumulative_text": cumulative_text.strip(),
                }
                chunks.append(chunk)
                last_boundary_tokens = target_tokens
                if callback is not None and not callback(chunk, chunks):
                    stop_flag.stop = True
                    stopped_early = True
                    break
            if stopped_early:
                break

        # Ensure a short final completion is visible to the callback.  This is
        # also the natural EOS case where fewer than one full chunk was emitted.
        current_tokens = streamer.generated_token_count
        if not stopped_early and current_tokens > last_boundary_tokens:
            chunk = {
                "chunk_idx": len(chunks),
                "text": cumulative_text[previous_text_len:],
                "num_tokens": current_tokens - last_boundary_tokens,
                "cumulative_text": cumulative_text.strip(),
            }
            previous_text_len = len(cumulative_text)
            chunks.append(chunk)
            if callback is not None and not callback(chunk, chunks):
                stop_flag.stop = True
                stopped_early = True

        worker.join(timeout=30.0)
        if errors:
            raise errors[0]

        final_text = cumulative_text
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in final_text:
                    final_text = final_text.split(stop_str)[0]
        # The streamer count is the authoritative number of output ids.  Keep
        # a tokenizer fallback for unusual streamer implementations.
        actual_tokens = streamer.generated_token_count
        if actual_tokens <= 0 and final_text:
            actual_tokens = len(model.tokenizer.encode(final_text, add_special_tokens=False))
        return {
            "text": final_text,
            # Count every token that actually arrived from the streamer.  A
            # stop callback can race with generation by a token or two; using
            # the boundary count here understated compute and produced fake
            # efficiency gains.
            "num_tokens": actual_tokens,
            "num_chunks": len(chunks),
            "chunks": chunks,
            "early_stop_stats": {
                "stopped_early": stopped_early,
                "num_chunks": len(chunks),
                "convergence_score": 0.0,
            },
        }
