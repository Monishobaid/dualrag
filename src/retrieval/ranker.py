"""Paragraph ranking for retrieval."""

from typing import List, Dict, Any, Optional, Set
import numpy as np
import re
from src.retrieval.embedder import Embedder
from src.retrieval.index import ParagraphIndex, build_paragraph_embeddings, EmbeddingCache
from src.utils.text import compute_jaccard_similarity


class ParagraphRanker:
    """Ranks paragraphs for retrieval."""
    
    def __init__(self, embedder: Embedder, cache: Optional[EmbeddingCache] = None,
                 retrieval_mode: str = "dense", dense_weight: float = 0.5,
                 bm25_k1: float = 1.5, bm25_b: float = 0.75):
        """
        Initialize ranker.
        
        Args:
            embedder: Embedder instance
            cache: Optional embedding cache
        """
        self.embedder = embedder
        self.cache = cache
        if retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("retrieval_mode must be dense, bm25, or hybrid")
        self.retrieval_mode = retrieval_mode
        self.dense_weight = float(np.clip(dense_weight, 0.0, 1.0))
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

    @staticmethod
    def _tokens(text: str) -> List[str]:
        """Small dependency-free tokenizer suitable for HotpotQA titles/text."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _bm25_scores(self, query: str, paragraphs: List[Dict[str, Any]]) -> np.ndarray:
        docs = [self._tokens(p.get("title", "") + " " + p.get("text", "")) for p in paragraphs]
        q_tokens = set(self._tokens(query))
        n = len(docs)
        if not n or not q_tokens:
            return np.zeros(n, dtype=np.float32)
        lengths = np.asarray([len(d) for d in docs], dtype=np.float32)
        avgdl = float(lengths.mean()) or 1.0
        df = {term: sum(term in d for d in docs) for term in q_tokens}
        scores = np.zeros(n, dtype=np.float32)
        for i, doc in enumerate(docs):
            tf = {}
            for token in doc:
                if token in q_tokens:
                    tf[token] = tf.get(token, 0) + 1
            for term, freq in tf.items():
                # BM25+ style IDF with a non-negative floor for common terms.
                idf = max(0.0, np.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0))
                denom = freq + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * lengths[i] / avgdl)
                scores[i] += idf * (freq * (self.bm25_k1 + 1) / denom)
        return scores
    
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
        
        dense_scores = None
        if self.retrieval_mode in {"dense", "hybrid"}:
            para_embeddings = build_paragraph_embeddings(paragraphs, self.embedder, cache=self.cache)
            query_embedding = self.embedder.encode_query(query)
            dense_scores = np.dot(para_embeddings, query_embedding)
        bm25_scores = None
        if self.retrieval_mode in {"bm25", "hybrid"}:
            bm25_scores = self._bm25_scores(query, paragraphs)

        if self.retrieval_mode == "dense":
            combined = dense_scores
        elif self.retrieval_mode == "bm25":
            combined = bm25_scores
        else:
            # Normalize each ranker to [0, 1] per query before interpolation.
            def norm(x):
                lo, hi = float(np.min(x)), float(np.max(x))
                return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
            combined = self.dense_weight * norm(dense_scores) + (1 - self.dense_weight) * norm(bm25_scores)

        top_k = min(top_k, len(paragraphs))
        top_indices = np.argsort(combined)[::-1][:top_k]
        results = []
        for idx in top_indices:
            item = {'paragraph': paragraphs[int(idx)], 'score': float(combined[idx]), 'index': int(idx),
                    'retrieval_mode': self.retrieval_mode}
            if dense_scores is not None:
                item['dense_score'] = float(dense_scores[idx])
            if bm25_scores is not None:
                item['bm25_score'] = float(bm25_scores[idx])
            results.append(item)
        
        return results
    
    def compute_retrieval_novelty(
        self,
        current_results: List[Dict[str, Any]],
        previous_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute novelty metrics for current retrieval vs previous evidence.
        
        Args:
            current_results: Current retrieval results list
            previous_evidence: Cumulative evidence paragraphs from previous hops
        
        Returns:
            Dictionary with novelty metrics
        """
        if not current_results:
            return {
                'jaccard_overlap': 1.0,
                'new_paragraphs': 0,
                'overlap_count': 0,
                'empty_retrieval': True
            }
            
        if not previous_evidence:
            return {
                'jaccard_overlap': 0.0,
                'new_paragraphs': len(current_results),
                'overlap_count': 0,
                'empty_retrieval': False
            }
        
        # Extract paragraph texts instead of titles to prevent collapsing distinct paragraphs under same title
        current_texts = set(r['paragraph']['text'] for r in current_results)
        previous_texts = set(p['text'] for p in previous_evidence)
        
        # Compute Jaccard overlap
        jaccard = compute_jaccard_similarity(current_texts, previous_texts)
        
        # Count new paragraphs
        new_texts = current_texts - previous_texts
        overlap_count = len(current_texts & previous_texts)
        
        return {
            'jaccard_overlap': jaccard,
            'new_paragraphs': len(new_texts),
            'overlap_count': overlap_count,
            'empty_retrieval': False
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
        seen_texts = set()
        evidence = []
        
        for hop_data in self.history:
            for result in hop_data['results']:
                text = result['paragraph']['text']
                if text not in seen_texts:
                    evidence.append(result['paragraph'])
                    seen_texts.add(text)
        
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
