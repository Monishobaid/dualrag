#!/usr/bin/env python3
"""Pre-build embedding index cache."""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hotpot_loader import HotpotQALoader
from src.retrieval.embedder import Embedder
from src.retrieval.index import EmbeddingCache, build_paragraph_embeddings
from src.config import RAGConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Pre-build embedding cache')
    parser.add_argument('--dataset_path', type=str, default='data/hotpotqa/hotpot_dev_distractor_v1.json',
                        help='Path to HotpotQA dataset')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum examples to process')
    parser.add_argument('--embed_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Embedding model name')
    parser.add_argument('--cache_dir', type=str, default='runs/cache_embeddings',
                        help='Cache directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    logger.info("Building embedding cache...")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Max examples: {args.max_examples}")
    
    # Load dataset
    logger.info("Loading dataset...")
    loader = HotpotQALoader(
        filepath=args.dataset_path,
        max_examples=args.max_examples,
        seed=args.seed
    )
    logger.info(f"Loaded {len(loader)} examples")
    
    # Initialize embedder
    logger.info(f"Loading embedding model: {args.embed_model_name}")
    embedder = Embedder(model_name=args.embed_model_name)
    
    # Initialize cache
    cache = EmbeddingCache(cache_dir=args.cache_dir)
    logger.info(f"Cache initialized with {len(cache)} existing embeddings")
    
    # Collect all unique paragraphs
    all_paragraphs = []
    seen_texts = set()
    
    for example in loader:
        for para in example.candidate_paragraphs:
            text = para['text']
            if text not in seen_texts:
                all_paragraphs.append(para)
                seen_texts.add(text)
    
    logger.info(f"Found {len(all_paragraphs)} unique paragraphs")
    
    # Build embeddings
    logger.info("Building embeddings...")
    embeddings = build_paragraph_embeddings(
        paragraphs=all_paragraphs,
        embedder=embedder,
        cache=cache,
        batch_size=32
    )
    
    logger.info(f"Built {len(embeddings)} embeddings")
    
    # Save cache
    cache.save()
    logger.info(f"Cache saved to {args.cache_dir}")
    logger.info(f"Total cached embeddings: {len(cache)}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
