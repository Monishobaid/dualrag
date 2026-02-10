"""Azure OpenAI model wrapper."""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
import time

try:
    from openai import AzureOpenAI
    import tiktoken
except ImportError:
    AzureOpenAI = None
    tiktoken = None

try:
    from dotenv import load_dotenv
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
        return_full_text: bool = False
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
                stop=stop_strings
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
            top_p=top_p
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
