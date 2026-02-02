"""Paragraph ranking for retrieval."""

from typing import List, Dict, Any, Optional, Set
import numpy as np
from src.retrieval.embedder import Embedder
from src.retrieval.index import ParagraphIndex, build_paragraph_embeddings, EmbeddingCache
from src.utils.text import compute_jaccard_similarity


class ParagraphRanker:
    """Ranks paragraphs for retrieval."""
    
    def __init__(self, embedder: Embedder, cache: Optional[EmbeddingCache] = None):
        """
        Initialize ranker.
        
        Args:
            embedder: Embedder instance
            cache: Optional embedding cache
        """
        self.embedder = embedder
        self.cache = cache
    
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
        index = ParagraphIndex(paragraphs, para_embeddings)
        results = index.search(query_embedding, top_k=top_k)
        
        return results
    
    def compute_retrieval_novelty(
        self,
        current_results: List[Dict[str, Any]],
        previous_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute novelty metrics for current retrieval vs previous.
        
        Args:
            current_results: Current retrieval results
            previous_results: Previous retrieval results
        
        Returns:
            Dictionary with novelty metrics
        """
        if not previous_results:
            return {
                'jaccard_overlap': 0.0,
                'new_paragraphs': len(current_results),
                'overlap_count': 0
            }
        
        # Extract titles
        current_titles = set(r['paragraph']['title'] for r in current_results)
        previous_titles = set(r['paragraph']['title'] for r in previous_results)
        
        # Compute Jaccard overlap
        jaccard = compute_jaccard_similarity(current_titles, previous_titles)
        
        # Count new paragraphs
        new_titles = current_titles - previous_titles
        overlap_count = len(current_titles & previous_titles)
        
        return {
            'jaccard_overlap': jaccard,
            'new_paragraphs': len(new_titles),
            'overlap_count': overlap_count
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
        seen_titles = set()
        evidence = []
        
        for hop_data in self.history:
            for result in hop_data['results']:
                title = result['paragraph']['title']
                if title not in seen_titles:
                    evidence.append(result['paragraph'])
                    seen_titles.add(title)
        
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
