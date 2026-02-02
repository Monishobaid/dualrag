"""Embedding model wrapper."""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging


class Embedder:
    """Wrapper for sentence embedding model."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"):
        """
        Initialize embedder.
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        logging.info(f"Loading embedding model: {model_name}")
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
