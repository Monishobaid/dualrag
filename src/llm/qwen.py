"""Configurable Qwen causal-language-model wrapper.

The default is Qwen2.5-3B-Instruct for local smoke tests; Renku benchmark
commands select Qwen2.5-7B-Instruct explicitly. This wrapper provides normal
generation and token accounting, while ``generation.py`` adds the streamed
callback path used by the answer-convergence ablation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Optional


class QwenModel:
    """Load an instruction-tuned Qwen checkpoint on CPU or GPU."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
        load_in_8bit: bool = False,
        use_chat_template: bool = False,
        max_input_tokens: int = 8192
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
        self.use_chat_template = use_chat_template
        self.max_input_tokens = max_input_tokens
        self.last_generation = {}
        
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
    
    def prepare_inputs(self, prompt, system_message=None):
        """Use Qwen's official chat format; reject oversized corrected inputs.

        Historical control: tokenizer(raw_prompt, truncation=True, max_length=4096).
        That can cut evidence and encourages continuation rather than an assistant
        reply. Keep it only when use_chat_template=False for the old ablation.
        Source: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#quickstart
        """
        if self.use_chat_template:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            if inputs["input_ids"].shape[1] > self.max_input_tokens:
                raise ValueError("Context exceeds explicit input budget; evidence was not silently truncated")
        else:
            if system_message:
                prompt = f"{system_message}\n\n{prompt}"
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True,
                                    truncation=True, max_length=4096)
        return inputs.to(self.model.device)

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
        text, _ = self.generate_with_token_count(
            prompt, max_new_tokens, temperature, top_p, stop_strings)
        return prompt + text if return_full_text else text

    def generate_with_token_count(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_strings: Optional[list] = None,
        response_format: Optional[dict] = None,
        system_message: Optional[str] = None,
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
        inputs = self.prepare_inputs(prompt, system_message)

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

        self.last_generation = {
            "input_tokens": int(input_length), "generated_tokens": int(num_tokens_generated),
            "hit_token_limit": num_tokens_generated >= max_new_tokens,
            "raw_output": generated_text,
        }
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]
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
