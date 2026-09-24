"""Iterative multi-hop RAG engine.

The loop follows the common retrieve -> reason -> formulate-a-follow-up-query
pattern used in multi-round RAG research. It is a custom research pipeline:
it combines dense/BM25 retrieval, optional evidence checks, and two ablation
stoppers, but it is not a faithful implementation of the trained Stop-RAG or
SIM-RAG systems.
"""

from typing import Dict, Any, List, Optional, Union
import logging
from src.llm.prompts import get_multihop_reasoning_prompt, get_answer_extraction_prompt, get_answer_verification_prompt, get_followup_query_prompt
from src.llm.generation import TextGenerator, ChunkedGenerator
from src.llm.early_stop import generate_with_reasoning_stop
from src.retrieval.ranker import ParagraphRanker, RetrievalHistory
from src.rag.stopping import RetrievalStopper
from src.rag.evidence_critic import EvidenceSufficiencyCritic
from src.rag.reader import GroundedCandidateReader
from src.utils.text import extract_json_from_text, normalize_text
from src.llm.qwen import QwenModel
from src.llm.azure_openai import AzureOpenAIModel


class IterativeRAG:
    """
    Iterative multi-hop RAG engine with optional dual stopping.

    Example flow: retrieve paragraphs for a question, ask Qwen for an answer
    and a focused follow-up query, retrieve again using that query, then choose
    the highest-confidence non-empty answer from the hop history.
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
        self.reader = GroundedCandidateReader()
        
        # Initialize components
        self.text_generator = TextGenerator(llm_model)
        self.chunked_generator = ChunkedGenerator(
            llm_model,
            chunk_tokens=config.chunk_tokens if config else 64
        )
        self.ranker = ParagraphRanker(embedder, cache=embedding_cache)
        self.evidence_critic = None
        if config and getattr(config, 'enable_evidence_critic', False):
            self.evidence_critic = EvidenceSufficiencyCritic(
                threshold=getattr(config, 'evidence_critic_threshold', 0.65)
            )
        if config:
            self.ranker = ParagraphRanker(
                embedder, cache=embedding_cache,
                retrieval_mode=getattr(config, 'retrieval_mode', 'dense'),
                dense_weight=getattr(config, 'dense_weight', 0.5),
                reranker_model_name=getattr(config, 'reranker_model_name', None),
                reranker_candidate_k=getattr(config, 'reranker_candidate_k', 20),
                device=getattr(config, 'device', 'auto'))
        
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
        best_selection_score = -1.0
        reader_candidates = []
        
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
            # Avoid repeating titles when there are enough unused candidates.
            # If the pool is too small, filtering would return fewer than
            # top_k paragraphs and could remove useful evidence.
            remaining_paragraphs = len(candidate_paragraphs) - len(retrieval_history.all_retrieved_titles)
            should_exclude = remaining_paragraphs >= top_k
            
            exclude_titles = retrieval_history.all_retrieved_titles if (hop > 1 and should_exclude) else None
            
            results = self.ranker.rank(
                query=current_query,
                paragraphs=candidate_paragraphs,
                top_k=top_k,
                exclude_titles=exclude_titles
            )

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
            if hop > 1:
                retrieval_novelty = self.ranker.compute_retrieval_novelty(
                    current_results=results,
                    previous_evidence=retrieval_history.get_cumulative_evidence()
                )
                jaccard = retrieval_novelty.get('jaccard_overlap', 0)
                print(f"🔄 Retrieval Novelty:")
                print(f"  - Jaccard overlap with previous hop: {jaccard:.3f}")
                print(f"  - {'🆕 High novelty (new info)' if jaccard < 0.5 else '♻️  Low novelty (similar to previous)'}")
                print()
                
            retrieval_history.add_hop(hop, current_query, results)

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
            print(f"  - Reasoning: {hop_result.get('reasoning', 'N/A')}")
            print(f"  - Confidence: {hop_result['confidence']:.2f}")
            print(f"  - Has evidence: {'✅ Yes' if hop_result.get('has_enough_evidence') else '❌ No'}")
            if hop_result.get('evidence_critic') is not None:
                critic = hop_result['evidence_critic']
                print(f"  - Evidence critic: {critic['score']:.2f} ({critic['reason']})")
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
                'has_enough_evidence': hop_result.get('has_enough_evidence', False),
                'tokens_generated': hop_result['num_tokens'],
                'reasoning_stopped': hop_result.get('stopped_early', False),
                'reasoning_stop_stats': hop_result.get('early_stop_stats', {}),
                'parse_failed': bool(hop_result.get('parse_failed', False))
            }
            hop_data['raw_output'] = hop_result.get('raw_output', '')
            hop_data['supporting_quotes'] = hop_result.get('supporting_quotes', [])
            hop_data['generation_metadata'] = dict(getattr(self.text_generator.model, 'last_generation', {}))
            hop_data['grounding_score'] = hop_result.get('grounding_score', 0.0)
            hop_data['answer_selection_score'] = hop_result.get('answer_selection_score', 0.0)
            hop_data['current_hypothesis'] = hop_result.get('current_hypothesis', '')
            hop_data['answer_candidates'] = hop_result.get('answer_candidates', [])
            if 'evidence_critic' in hop_result:
                hop_data['evidence_critic'] = hop_result['evidence_critic']

            if getattr(self.config, 'reader_mode', 'legacy') == 'grounded_candidates':
                reader_candidates.extend(self.reader.candidates_from_hop(
                    question=question,
                    hop_result=hop_result,
                    evidence=cumulative_evidence,
                    hop=hop,
                    prior_candidates=[row.get('candidate', '') for row in reader_candidates],
                ))

            hop_history.append(hop_data)
            
            # Select answers using confidence after the grounding guard has
            # downgraded unsupported answers, with deterministic grounding as
            # a secondary tie-breaker. This keeps selection auditable without
            # blindly trusting a raw model confidence value.
            current_conf = hop_result.get('confidence', 0.0)
            current_ans = hop_result.get('answer', "").strip()
            current_score = hop_result.get('answer_selection_score', 0.0)

            if current_score > best_selection_score and current_ans and len(current_ans) > 1:
                logging.info(
                    "New best answer found (selection %.3f, confidence %.2f): %s",
                    current_score, current_conf, current_ans,
                )
                best_confidence = current_conf
                best_selection_score = current_score
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

            should_stop = False
            if self.config and getattr(self.config, 'enable_retrieval_stop', False) and self.retrieval_stopper:
                should_stop, reason = self.retrieval_stopper.should_stop(
                    current_hop=hop,
                    hop_history=hop_history,
                    retrieval_novelty=retrieval_novelty
                )

                print(f"🔍 Retrieval stopping criteria:")
                print(f"  - Current hop: {hop}")
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
                if not getattr(self.config, 'enable_retrieval_stop', False):
                    followup_query = f"{question} {current_ans}".strip()
                    print(f"\n⚠️ No follow-up query, but fixed-hops mode: adding current answer to question for retrieval diversity")
                elif hop < self.config.min_hops:
                    followup_query = f"{question} {current_ans}".strip()
                    print(f"\n⚠️ No follow-up query, but min_hops not met ({hop} < {self.config.min_hops}): adding current answer to question for retrieval diversity")
                else:
                    retrieval_stopped = True
                    retrieval_stop_reason = "no_followup_query"
                    logging.info("No follow-up query provided by LLM, stopping")
                    print(f"\n✅ No follow-up query provided, stopping retrieval")
                    break

            print(f"\n➡️  Continuing to next hop with query: {followup_query[:100]}")
            current_query = followup_query
        
        # === FINAL RESULTS ===
        # ``best_answer`` is chosen by calibrated confidence with a grounding
        # tie-breaker, rather than an unguarded confidence value. This keeps a
        # supported later-hop answer from being discarded by an unsupported
        # overconfident first hop.
        reader_selection = None
        if getattr(self.config, 'reader_mode', 'legacy') == 'grounded_candidates':
            reader_selection = self.reader.select(reader_candidates)
            final_answer = reader_selection.get('answer') or best_answer or (
                hop_history[-1]['answer'] if hop_history else ""
            )
        else:
            final_answer = best_answer or (hop_history[-1]['answer'] if hop_history else "")
        
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
        if reader_selection is not None:
            trace['reader_mode'] = 'grounded_candidates'
            trace['reader_selection'] = reader_selection

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
        prompt = get_multihop_reasoning_prompt(
            question,
            evidence,
            hop,
            max_evidence_paragraphs=getattr(self.config, 'max_evidence_paragraphs', None),
            generate_candidates=(getattr(self.config, 'reader_mode', 'legacy') == 'grounded_candidates'),
        )
        
        # Generate with or without reasoning stop
        enable_reasoning_stop = self.config and getattr(self.config, 'enable_reasoning_stop', False)
        
        if enable_reasoning_stop:
            # Use chunked generation with early stopping
            result = generate_with_reasoning_stop(
                chunked_generator=self.chunked_generator,
                prompt=prompt,
                temperature=self.config.temperature if self.config else 0.7,
                top_p=self.config.top_p if self.config else 0.9,
                convergence_k=self.config.convergence_k if self.config else 2,
                text_generator=self.text_generator,
                stop_strings=["```"],
                max_new_tokens=self.config.max_new_tokens if self.config else 512
            )
            
            generated_text = result['text']
            
            num_tokens = result['num_tokens']
            stopped_early = result['early_stop_stats']['stopped_early']
            early_stop_stats = result['early_stop_stats']
            
        else:
            # Standard generation
            from src.llm.prompts import SYSTEM_PROMPT, JSON_SYSTEM_PROMPT
            
            result = self.text_generator.generate(
                prompt=prompt,
                system_message=JSON_SYSTEM_PROMPT,
                max_new_tokens=self.config.max_new_tokens if self.config else 512,
                stop_strings=["```"],
                temperature=self.config.temperature if self.config else 0.7,
                response_format={"type": "json_object"}
            )
            
            generated_text = result['text']
            
            num_tokens = result['num_tokens']
            stopped_early = False
            early_stop_stats = {}
        
        # Parse JSON response
        extraction_tokens = 0
        parsed = None
        
        # A stopped response follows exactly the same parsing, grounding, and
        # evidence-critic path as an unstopped response.  In particular, do
        # not manufacture confidence/evidence metadata for a heuristic stop.
        parsed = extract_json_from_text(generated_text)
        if stopped_early:
            from src.utils.text import extract_complete_json_from_text
            parsed = extract_complete_json_from_text(generated_text)
        
        if parsed:
            current_hypothesis = str(parsed.get('current_hypothesis') or '').strip()
            answer_candidates = parsed.get('answer_candidates', [])
            if not isinstance(answer_candidates, list):
                answer_candidates = []
            answer = str(parsed.get('answer') or '').strip()
            followup_query = str(parsed.get('followup_query') or '').strip()
            try:
                confidence = max(0.0, min(1.0, float(parsed.get('confidence', 0.5))))
            except (TypeError, ValueError):
                confidence = 0.0
            # Missing evidence metadata must be conservative: malformed or
            # partial JSON should not be allowed to trigger retrieval stop.
            raw_evidence_flag = parsed.get('has_enough_evidence', False)
            if isinstance(raw_evidence_flag, str):
                has_enough_evidence = raw_evidence_flag.strip().lower() in {'true', '1', 'yes'}
            else:
                has_enough_evidence = bool(raw_evidence_flag)
            supporting_quotes = parsed.get('supporting_quotes', [])
            if not isinstance(supporting_quotes, list):
                supporting_quotes = []
            reasoning = str(parsed.get('reasoning') or '')

        else:
            # Fallback: JSON parsing failed
            logging.warning("JSON parsing failed. Returning empty answer to force retry or stop.")
            answer = ""
            confidence = 0.0
            has_enough_evidence = False
            followup_query = ""
            supporting_quotes = []
            reasoning = ""
            current_hypothesis = ""
            answer_candidates = []
            
        # Semantic Verification removed to reduce noise/hallucination loops
        # Logic simplified to rely on strict JSON output or retrieval loop fallback

        # Retrieval Drift Prevention
        trivial_answers = {"yes", "no", "true", "false", "none", "n/a", "not enough information", "i cannot answer"}
        
        # If followup query is exactly the same as the current question, it means the model is stuck
        if followup_query.strip().lower() == question.strip().lower():
             logging.warning("Follow-up query is identical to the current question. Clearing it to force stop.")
             followup_query = ""
             
        # Grounding score is also used for answer selection.  Exact phrase
        # matches are strongest; otherwise score the fraction of meaningful
        # answer words found in the accumulated evidence.
        grounding_score = self._grounding_score(answer, evidence)

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
                     # Fallback: check if most significant words exist independently
                     import re
                     words = [w for w in re.split(r'\W+', ans_norm) if len(w) > 2]
                     if words:
                         found_words = 0
                         for w in words:
                             word_found = False
                             for item in evidence:
                                 if w in item['text'].lower() or w in item['title'].lower():
                                     word_found = True
                                     break
                             if word_found:
                                 found_words += 1
                         # If all words found, consider it grounded
                         if found_words == len(words) and len(words) > 0:
                             found_in_text = True
                 
                 if not found_in_text:
                     # Check if all words in the answer are in the text (to handle combined answers like 'Israeli and Palestinian')
                     import re
                     words = set(w for w in re.split(r'\W+', ans_norm) if len(w) > 2)
                     if words:
                         found_all_words = True
                         for w in words:
                             word_found = False
                             for item in evidence:
                                 if w in item['text'].lower() or w in item['title'].lower():
                                     word_found = True
                                     break
                             if not word_found:
                                 found_all_words = False
                                 break
                         if found_all_words:
                             found_in_text = True

                 if not found_in_text:
                     logging.warning(f"Grounding Check Warning: Answer '{answer}' not found in evidence. Downgrading confidence.")
                     confidence = 0.2
                     has_enough_evidence = False
                     # Let outer loop handle fallback query naturally

        # Optional model-free evidence critic.  It runs after the existing
        # grounding guard and can only make stopping more conservative.
        evidence_critic = None
        if self.evidence_critic is not None:
            evidence_critic = self.evidence_critic.evaluate(
                question=question,
                answer=answer,
                evidence=evidence,
                supporting_quotes=supporting_quotes,
            )
            if not evidence_critic['sufficient']:
                has_enough_evidence = False
                confidence = min(confidence, evidence_critic['score'])

        # Confidence remains primary because the grounding guard has already
        # downgraded unsupported confident answers.  Grounding is a secondary
        # tie-breaker; treating the model's evidence flag as a large positive
        # weight made a cautious but correct hop lose to a distractor answer
        # on the same traces.
        answer_selection_score = (
            0.75 * confidence + 0.25 * grounding_score
        ) if answer else 0.0

        return {
            'answer': answer,
            'current_hypothesis': current_hypothesis,
            'answer_candidates': answer_candidates,
            'followup_query': followup_query,
            'confidence': confidence,
            'has_enough_evidence': has_enough_evidence,
            'supporting_quotes': supporting_quotes,
            'reasoning': reasoning,
            'num_tokens': num_tokens,
            'extraction_tokens': extraction_tokens,
            'stopped_early': stopped_early,
            'early_stop_stats': early_stop_stats,
            'evidence_critic': evidence_critic,
            'grounding_score': grounding_score,
            'answer_selection_score': answer_selection_score,
            'raw_output': generated_text,
            'parse_failed': parsed is None
        }

    @staticmethod
    def _grounding_score(answer: str, evidence: List[Dict[str, Any]]) -> float:
        """Return a simple auditable answer-to-evidence grounding score."""
        answer = (answer or '').strip().lower().strip('"\'')
        if not answer or answer in {"yes", "no", "true", "false", "none", "n/a"}:
            return 0.0
        corpus = " ".join(
            f"{item.get('title', '')} {item.get('text', '')}".lower()
            for item in evidence
        )
        if answer in corpus:
            return 1.0
        import re
        words = [word for word in re.findall(r"[a-z0-9]+", answer) if len(word) > 2]
        if not words:
            return 0.0
        return sum(word in corpus for word in words) / len(words)
