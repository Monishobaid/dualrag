"""Iterative RAG engine."""

from typing import Dict, Any, List, Optional, Union
import logging
from src.llm.prompts import get_multihop_reasoning_prompt, get_answer_extraction_prompt, get_answer_verification_prompt, get_followup_query_prompt
from src.llm.generation import TextGenerator, ChunkedGenerator
from src.llm.early_stop import generate_with_reasoning_stop
from src.retrieval.ranker import ParagraphRanker, RetrievalHistory
from src.rag.stopping import RetrievalStopper
from src.utils.text import extract_json_from_text, normalize_text
from src.llm.qwen import QwenModel
from src.llm.azure_openai import AzureOpenAIModel


class IterativeRAG:
    """
    Iterative multi-hop RAG engine with dual stopping mechanisms.
    """
    
    def __init__(
        self,
        llm_model: Union[QwenModel, AzureOpenAIModel],
        embedder,
        embedding_cache=None,
        config=None
    ):
        """
        Initialize RAG engine.
        
        Args:
            llm_model: QwenModel or AzureOpenAIModel instance
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
        
        # Track best answer found so far (fallback if final hop fails)
        best_answer = ""
        best_confidence = 0.0
        
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
            # Fix A: Only exclude titles if candidate pool is large enough
            should_exclude = True
            if len(candidate_paragraphs) <= top_k: # If pool is small (e.g. distractor has ~10), don't exclude unless we have excess
                 should_exclude = False
            
            exclude_titles = retrieval_history.all_retrieved_titles if (hop > 1 and should_exclude) else None
            
            results = self.ranker.rank(
                query=current_query,
                paragraphs=candidate_paragraphs,
                top_k=top_k,
                exclude_titles=exclude_titles
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
            print(f"  - Has evidence: {'✅ Yes' if hop_result.get('has_enough_evidence') else '❌ No'}")
            if hop_result.get('supporting_quotes'):
                print(f"  - Supporting quotes:")
                for q in hop_result['supporting_quotes']:
                    print(f"    * \"{q}\"")
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
            
            # Update best answer if confidence improved and answer is valid
            current_conf = hop_result.get('confidence', 0.0)
            current_ans = hop_result.get('answer', "").strip()
            
            if current_conf > best_confidence and current_ans and len(current_ans) > 1:
                logging.info(f"New best answer found (conf: {current_conf:.2f}): {current_ans}")
                best_confidence = current_conf
                best_answer = current_ans
            
            # === STOPPING DECISIONS ===
            print(f"🛑 STOPPING CHECK")
            print(f"-" * 80)

            # Check retrieval stopping
            # Special fast-exit for Hop 1: If confidence is very high, stop even if min_hops not met
            # DISABLE for now as model is overconfident (stopped 50% wrong at Hop 1)
            # if hop == 1 and hop_result['confidence'] >= 0.9:
            #      retrieval_stopped = True
            #      retrieval_stop_reason = "high_confidence_early_exit"
            #      logging.info("High confidence (>= 0.9) in Hop 1. Stopping early despite min_hops.")
            #      print(f"\n✅ Retrieval stopped: {retrieval_stop_reason}")
            #      break

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
                    
                    # Fix D: Do not poison query with answer
                    followup_query = question 
                    print(f"➡️  Forcing continuation with original question")
                
                # ALSO force continuation if the stopper said we should continue (i.e. we didn't break above)
                # This handles cases where answer is unstable but model thinks it's done
                elif self.retrieval_stopper and not should_stop:
                     logging.info(f"No follow-up query but retrieval stopper says CONTINUE. Forcing continuation.")
                     print(f"\n⚠️  No follow-up query, but stopping criteria NOT met (e.g. answer unstable).")
                     
                     # Fix D: Do not poison query with answer
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
        
        # Fallback to best answer if final is empty/failed but we had a good one earlier
        # Logic: If final answer is empty or very short (<2 chars), AND we have a best answer
        should_fallback = (not final_answer or len(final_answer.strip()) < 2) and best_answer
        
        if should_fallback:
             final_answer = best_answer
             msg = f"Final answer empty/failed. Using best answer found (Conf: {best_confidence:.2f}): {best_answer}"
             print(f"\n⚠️  {msg}")
             logging.warning(msg)
             # Append note to stop reason
             if retrieval_stop_reason:
                retrieval_stop_reason += "_fallback_to_best"
             else:
                retrieval_stop_reason = "fallback_to_best"

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
        # FORCE DISABLE chunked generation to avoid repetition loops
        enable_reasoning_stop = False # self.config and self.config.enable_reasoning_stop
        
        if enable_reasoning_stop:
            # Use chunked generation with early stopping
            result = generate_with_reasoning_stop(
                chunked_generator=self.chunked_generator,
                prompt=prompt,
                temperature=self.config.temperature if self.config else 0.7,
                text_generator=self.text_generator,
                stop_strings=["\n\n", "```", "}"], # Aggressive stop at JSON close
                max_new_tokens=512  # FORCE CAP
            )
            
            generated_text = result['text']
            # If we aggressive stopped at '}', re-append it
            if generated_text.strip() and not generated_text.strip().endswith('}'):
                 generated_text += "}"
            
            num_tokens = result['num_tokens']
            stopped_early = result['early_stop_stats']['stopped_early']
            early_stop_stats = result['early_stop_stats']
            
        else:
            # Standard generation
            from src.llm.prompts import SYSTEM_PROMPT, JSON_SYSTEM_PROMPT
            
            result = self.text_generator.generate(
                prompt=prompt,
                system_message=JSON_SYSTEM_PROMPT,
                max_new_tokens=512,  # FORCE CAP: Prevent infinite reasoning loops
                stop_strings=["\n\n", "```", "}"], # Aggressive stop at JSON close
                temperature=self.config.temperature if self.config else 0.7,
                response_format={"type": "json_object"}
            )
            
            generated_text = result['text']
            # If we aggressive stopped at '}', re-append it
            if generated_text.strip() and not generated_text.strip().endswith('}'):
                 generated_text += "}"
            
            num_tokens = result['num_tokens']
            stopped_early = False
            early_stop_stats = {}
        
        # Parse JSON response
        parsed = None
        
        # If stopped early due to reasoning convergence, use the converged answer
        if stopped_early and 'final_answer' in result and result['final_answer']:
            logging.info(f"Using converged answer from reasoning stop: {result['final_answer']}")
            parsed = {
                "answer": result['final_answer'],
                "has_enough_evidence": True,
                "confidence": 0.9,
                "followup_query": "",
                "supporting_quotes": []
            }
        else:
            # Normal parsing
            parsed = extract_json_from_text(generated_text)
        
        if parsed:
            answer = parsed.get('answer', '')
            followup_query = parsed.get('followup_query', '')
            confidence = float(parsed.get('confidence', 0.5))
            has_enough_evidence = bool(parsed.get('has_enough_evidence', True))
            supporting_quotes = parsed.get('supporting_quotes', [])
            

        else:
            # Fallback: JSON parsing failed
            logging.warning("JSON parsing failed. Returning empty answer to force retry or stop.")
            answer = ""
            confidence = 0.0
            has_enough_evidence = False
            followup_query = ""
            supporting_quotes = []
            
        # Semantic Verification removed to reduce noise/hallucination loops
        # Logic simplified to rely on strict JSON output or retrieval loop fallback

        # Retrieval Drift Prevention
        trivial_answers = {"yes", "no", "true", "false", "none", "n/a", "not enough information", "i cannot answer"}
        
        # If the model didn't give a followup query, check if we should verify
        if not followup_query:
            # Avoid verifying trivial/short answers which leads to drift
            if answer.lower() in trivial_answers or len(answer) < 4:
                 logging.info(f"Answer '{answer}' is trivial/short. Defaulting followup to original question.")
                 followup_query = question
            else:
                 # Standard verification
                 followup_query = "" # Let the outer loop decide (it usually does 'evidence for X')
        # Grounding Check (Guard B)
        # Ensure the answer actually appears in the retrieved text (to prevention hallucinations)
        if confidence > 0.5 and answer and len(answer) > 4:
             # Normalize for check
             ans_norm = answer.lower().strip('"').strip("'")
             is_trivial = ans_norm in trivial_answers
             
             if not is_trivial:
                 found_in_text = False
                 for item in evidence:
                     if ans_norm in item['text'].lower() or ans_norm in item['title'].lower():
                         found_in_text = True
                         break
                 
                 if not found_in_text:
                     logging.warning(f"Grounding Check Warning: Answer '{answer}' not found in evidence. Downgrading confidence.")
                     confidence = 0.2
                     has_enough_evidence = False
                     # Let outer loop handle fallback query naturally

        return {
            'answer': answer,
            'followup_query': followup_query,
            'confidence': confidence,
            'has_enough_evidence': has_enough_evidence,
            'supporting_quotes': supporting_quotes,
            'num_tokens': num_tokens,
            'stopped_early': stopped_early,
            'early_stop_stats': early_stop_stats,
            'raw_output': generated_text
        }
