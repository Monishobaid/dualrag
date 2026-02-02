"""Iterative RAG engine."""

from typing import Dict, Any, List, Optional
import logging
from src.llm.prompts import get_multihop_reasoning_prompt
from src.llm.generation import TextGenerator, ChunkedGenerator
from src.llm.early_stop import generate_with_reasoning_stop
from src.retrieval.ranker import ParagraphRanker, RetrievalHistory
from src.rag.stopping import RetrievalStopper
from src.utils.text import extract_json_from_text, normalize_text


class IterativeRAG:
    """
    Iterative multi-hop RAG engine with dual stopping mechanisms.
    """
    
    def __init__(
        self,
        llm_model,
        embedder,
        embedding_cache=None,
        config=None
    ):
        """
        Initialize RAG engine.
        
        Args:
            llm_model: QwenModel instance
            embedder: Embedder instance
            embedding_cache: Optional EmbeddingCache
            config: RAGConfig instance
        """
        self.llm = llm_model
        self.embedder = embedder
        self.cache = embedding_cache
        self.config = config
        
        # Initialize components
        self.text_generator = TextGenerator(llm_model)
        self.chunked_generator = ChunkedGenerator(
            llm_model,
            chunk_tokens=config.chunk_tokens if config else 64
        )
        self.ranker = ParagraphRanker(embedder, cache=embedding_cache)
        
        # Stopping mechanisms
        if config:
            self.retrieval_stopper = RetrievalStopper(
                min_hops=config.min_hops,
                overlap_threshold=config.overlap_threshold,
                conf_threshold=config.conf_threshold
            )
        else:
            self.retrieval_stopper = None
    
    def run(
        self,
        question: str,
        candidate_paragraphs: List[Dict[str, Any]],
        max_hops: int = 4,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Run iterative RAG on a question.
        
        Args:
            question: Question to answer
            candidate_paragraphs: Available paragraphs for retrieval
            max_hops: Maximum retrieval hops
            top_k: Number of paragraphs per hop
        
        Returns:
            Full trace with hops, answers, and stopping decisions
        """
        retrieval_history = RetrievalHistory()
        hop_history = []
        
        current_query = question
        retrieval_stopped = False
        retrieval_stop_reason = None
        
        for hop in range(1, max_hops + 1):
            print(f"\n{'='*80}")
            print(f"🔄 HOP {hop}/{max_hops}")
            print(f"{'='*80}")
            print(f"📝 Query: {current_query}")
            print()

            logging.info(f"--- Hop {hop}/{max_hops} ---")
            logging.info(f"Query: {current_query[:100]}...")

            # === RETRIEVAL ===
            print(f"🔍 RETRIEVAL PHASE")
            print(f"-" * 80)

            # Get previously retrieved titles to track novelty
            prev_results = retrieval_history.get_latest_results()

            # Rank paragraphs
            results = self.ranker.rank(
                query=current_query,
                paragraphs=candidate_paragraphs,
                top_k=top_k
            )

            retrieval_history.add_hop(hop, current_query, results)

            # Display retrieved paragraphs
            print(f"📚 Retrieved {len(results)} paragraphs:")
            for i, result in enumerate(results, 1):
                title = result['paragraph']['title']
                score = result['score']
                text_preview = result['paragraph']['text'][:100].replace('\n', ' ')
                print(f"  [{i}] {title} (score: {score:.3f})")
                print(f"      {text_preview}...")
            print()

            # Compute retrieval novelty
            retrieval_novelty = None
            if prev_results:
                retrieval_novelty = self.ranker.compute_retrieval_novelty(
                    results,
                    prev_results
                )
                jaccard = retrieval_novelty.get('jaccard_overlap', 0)
                print(f"🔄 Retrieval Novelty:")
                print(f"  - Jaccard overlap with previous hop: {jaccard:.3f}")
                print(f"  - {'🆕 High novelty (new info)' if jaccard < 0.5 else '♻️  Low novelty (similar to previous)'}")
                print()

            # Get cumulative evidence
            cumulative_evidence = retrieval_history.get_cumulative_evidence()

            logging.info(f"Retrieved {len(results)} paragraphs, cumulative evidence: {len(cumulative_evidence)}")
            print(f"📊 Cumulative evidence: {len(cumulative_evidence)} unique paragraphs")
            print()
            
            # === GENERATION ===
            print(f"🤖 GENERATION PHASE")
            print(f"-" * 80)
            hop_result = self._generate_hop_response(
                question=question,
                evidence=cumulative_evidence,
                hop=hop
            )

            # Display generation results
            print(f"💡 Generated Response:")
            print(f"  - Answer: {hop_result['answer']}")
            print(f"  - Confidence: {hop_result['confidence']:.2f}")
            print(f"  - Tokens generated: {hop_result['num_tokens']}")
            if hop_result.get('stopped_early'):
                print(f"  - ⚡ Reasoning stopped early (answer converged)")
            print(f"  - Follow-up query: {hop_result.get('followup_query', 'N/A')}")
            print()

            # Store hop data
            hop_data = {
                'hop': hop,
                'query': current_query,
                'retrieved_paragraphs': [
                    {
                        'title': r['paragraph']['title'],
                        'score': r['score'],
                        'text_snippet': r['paragraph']['text'][:200]
                    }
                    for r in results
                ],
                'retrieval_novelty': retrieval_novelty,
                'answer': hop_result['answer'],
                'confidence': hop_result['confidence'],
                'tokens_generated': hop_result['num_tokens'],
                'reasoning_stopped': hop_result.get('stopped_early', False),
                'reasoning_stop_stats': hop_result.get('early_stop_stats', {})
            }

            hop_history.append(hop_data)
            
            # === STOPPING DECISIONS ===
            print(f"🛑 STOPPING CHECK")
            print(f"-" * 80)

            # Check retrieval stopping
            if self.config and self.config.enable_retrieval_stop and self.retrieval_stopper:
                should_stop, reason = self.retrieval_stopper.should_stop(
                    current_hop=hop,
                    hop_history=hop_history,
                    retrieval_novelty=retrieval_novelty
                )

                print(f"🔍 Retrieval stopping criteria:")
                print(f"  - Current hop: {hop} (min: {self.config.min_hops})")
                print(f"  - Confidence: {hop_result['confidence']:.2f} (threshold: {self.config.conf_threshold})")
                if retrieval_novelty:
                    print(f"  - Novelty overlap: {retrieval_novelty.get('jaccard_overlap', 0):.2f} (threshold: {self.config.overlap_threshold})")
                print(f"  - Decision: {'🛑 STOP' if should_stop else '▶️  CONTINUE'} ({reason})")

                if should_stop:
                    retrieval_stopped = True
                    retrieval_stop_reason = reason
                    logging.info(f"Retrieval stopped at hop {hop}: {reason}")
                    print(f"\n✅ Retrieval stopped: {reason}")
                    break

            # Check if we've reached max hops
            if hop >= max_hops:
                retrieval_stop_reason = "max_hops_reached"
                print(f"\n🏁 Maximum hops ({max_hops}) reached")
                break

            # Generate follow-up query for next hop
            followup_query = hop_result.get('followup_query', '')

            if not followup_query or followup_query.lower() in ['none', 'n/a', '']:
                # Check if we must continue despite no query
                if self.config and hop < self.config.min_hops:
                    logging.info(f"No follow-up query but min_hops ({self.config.min_hops}) not reached. Forcing continuation.")
                    print(f"\n⚠️  No follow-up query, but min_hops ({self.config.min_hops}) not reached.")
                    
                    # Generate fallback query
                    if hop_result.get('answer'):
                        followup_query = f"evidence for {hop_result['answer']}"
                        print(f"➡️  Forcing continuation with verification query: {followup_query}")
                    else:
                        followup_query = question
                        print(f"➡️  Forcing continuation with original question")
                        
                else:
                    # No follow-up needed
                    retrieval_stopped = True
                    retrieval_stop_reason = "no_followup_needed"
                    logging.info("No follow-up query needed, stopping")
                    print(f"\n✅ No follow-up query needed, stopping retrieval")
                    break

            print(f"\n➡️  Continuing to next hop with query: {followup_query[:100]}")
            current_query = followup_query
        
        # === FINAL RESULTS ===
        final_answer = hop_history[-1]['answer'] if hop_history else ""

        print(f"\n{'='*80}")
        print(f"🎯 FINAL RESULTS")
        print(f"{'='*80}")
        print(f"❓ Question: {question}")
        print(f"✅ Final Answer: {final_answer}")
        print(f"🔢 Hops taken: {len(hop_history)}/{max_hops}")
        print(f"📝 Total tokens: {sum(h['tokens_generated'] for h in hop_history)}")
        print(f"🛑 Stop reason: {retrieval_stop_reason}")
        print(f"{'='*80}\n")

        trace = {
            'question': question,
            'hops_taken': len(hop_history),
            'max_hops': max_hops,
            'final_answer': final_answer,
            'hops': hop_history,
            'retrieval_stopped': retrieval_stopped,
            'retrieval_stop_reason': retrieval_stop_reason,
            'total_tokens': sum(h['tokens_generated'] for h in hop_history),
            'cumulative_evidence_size': len(retrieval_history.get_cumulative_evidence())
        }

        if self.retrieval_stopper:
            trace['retrieval_stop_stats'] = self.retrieval_stopper.get_stopping_stats(hop_history)

        return trace
    
    def _generate_hop_response(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        hop: int
    ) -> Dict[str, Any]:
        """
        Generate response for a single hop.
        
        Args:
            question: Original question
            evidence: Cumulative evidence
            hop: Hop number
        
        Returns:
            Response dict with answer, followup_query, confidence, tokens
        """
        # Build prompt
        prompt = get_multihop_reasoning_prompt(question, evidence, hop)
        
        # Generate with or without reasoning stop
        if self.config and self.config.enable_reasoning_stop:
            # Use chunked generation with early stopping
            result = generate_with_reasoning_stop(
                chunked_generator=self.chunked_generator,
                prompt=prompt,
                max_chunks=self.config.max_new_tokens // self.config.chunk_tokens,
                convergence_k=self.config.convergence_k,
                temperature=self.config.temperature,
                text_generator=self.text_generator
            )
            
            generated_text = result['text']
            num_tokens = result['num_tokens']
            stopped_early = result['early_stop_stats']['stopped_early']
            early_stop_stats = result['early_stop_stats']
            
        else:
            # Standard generation
            result = self.text_generator.generate(
                prompt=prompt,
                max_new_tokens=self.config.max_new_tokens if self.config else 256,
                temperature=self.config.temperature if self.config else 0.7
            )
            
            generated_text = result['text']
            num_tokens = result['num_tokens']
            stopped_early = False
            early_stop_stats = {}
        
        # Parse JSON response
        parsed = extract_json_from_text(generated_text)
        
        if parsed:
            answer = parsed.get('answer', '')
            followup_query = parsed.get('followup_query', '')
            confidence = float(parsed.get('confidence', 0.5))
        else:
            # Fallback parsing
            logging.warning("Failed to parse JSON, using fallback")
            answer = generated_text[:200]  # Take first 200 chars as answer
            followup_query = ""
            confidence = 0.3
        
        return {
            'answer': answer,
            'followup_query': followup_query,
            'confidence': confidence,
            'num_tokens': num_tokens,
            'stopped_early': stopped_early,
            'early_stop_stats': early_stop_stats,
            'raw_output': generated_text
        }
