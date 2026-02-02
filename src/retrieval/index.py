"""Embedding index builder and cache manager."""

import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import logging


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
