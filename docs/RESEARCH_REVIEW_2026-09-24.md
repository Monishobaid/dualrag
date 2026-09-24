# Literature review and experiment protocol — 24 September 2026

## Question and scope

Can an explicit check for missing evidence improve the accuracy/cost trade-off
of our multi-hop reader? This is a research hypothesis, not a claim that the
system beats commercial RAG or the state of the art. The current evaluation is
HotpotQA **distractor**, with approximately ten supplied paragraphs per question.
It does not search Wikipedia, the web, or a production vector database.

Review scope: the linked articles and the methods/experimental sections relevant
to our design; selected additional primary papers and practitioner discussions.
We have not independently reproduced the papers' reported benchmark results.

## What the sources actually suggest

| Source | Method and relevance | Decision for this codebase |
|---|---|---|
| [Google: Sufficient Context](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/) / [ICLR paper](https://arxiv.org/abs/2411.06037) | Relevance does not imply enough information to answer. Its selective-generation results trade coverage for accuracy; they are not unconditional EM gains. | Ask a critic whether all required relations are supported. Log its decisions and errors; do not assume Google's autorater accuracy transfers to Qwen. |
| [Anthropic: Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) | Adds document-informed contextual text to chunks before both lexical and dense indexing, then optionally reranks. Its headline improvements concern retrieval failure, not QA EM. | Test title metadata and stronger retrieval separately. A title prefix is an inexpensive metadata baseline, **not a reproduction** of LLM contextualization. Our dataset lacks full parent Wikipedia documents. |
| [DREAM, 2606.24667](https://arxiv.org/html/2606.24667v1) | Trains a dense retriever via next-token loss passing through selected retrieval-focused attention heads of a frozen LLM. | Promising later training direction. Requires a training pipeline and head selection; cannot be implemented honestly by switching a vector database or prompt. |
| [KaLM-Reranker, 2606.22807](https://arxiv.org/html/2606.22807v1) | Reusable passage encoding with Matryoshka compression, followed by query-dependent decoder cross-attention and yes/no scoring. | A potential efficiency comparison for a larger corpus. Not a drop-in generic CrossEncoder checkpoint. Ten candidates do not establish large-scale reranking throughput. |
| [EvoEmbedding, 2606.21649](https://arxiv.org/html/2606.21649v1) | Context-dependent representations use a bounded evolving latent-memory queue; trained with memory-generation and contrastive objectives. | Better aligned with long histories and changing user memory than our static ten-paragraph pools. Defer rather than label static cached vectors as EvoEmbedding. |
| [MCompassRAG, 2606.18508](https://arxiv.org/html/2606.18508v1) | Corpus topic metadata guides query/chunk representations; a teacher distills relevance into a student retriever. | Motivates keeping useful metadata. Its learned topic bank and student are not present in our title-prefix baseline. |
| [Environment-aware IR, 2606.16817](https://arxiv.org/html/2606.16817v1) | Trains query rewriting against the actual retriever; rewrite style and rewards depend on the retrieval environment. | Log rewritten queries and compare against the original. Do not assume longer queries always help. Any learned rewriter must use train-only supervision. |
| [Direct Corpus Interaction, 2605.05242](https://arxiv.org/html/2605.05242v1) | Agents inspect raw files with search/read tools rather than only receiving ranked snippets. | Supports a future bounded find-entity/read-neighbor tool. Its agent scaffolds and model budgets differ from ours; results cannot be treated as an apples-to-apples Qwen comparison. |
| [IRCoT, ACL 2023](https://aclanthology.org/2023.acl-long.557/) | Interleaves evidence-dependent reasoning and retrieval to discover later-hop evidence. | Our critic emits the missing relation and a targeted query. This is inspired by interleaving, not an exact IRCoT prompt or experimental reproduction. |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | Relevant information's position affects reader performance in long context. | Log exactly what the reader receives; include all-context and oracle controls. More retrieved text is not automatically better. |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | Learns retrieval and reflection behavior with specialized tokens. | Our untrained prompted critic must not be called a trained Self-RAG controller. |
| [Corrective RAG](https://arxiv.org/abs/2401.15884) | Evaluates retrieval quality and takes corrective actions when evidence is unreliable. | Motivates separating detection of inadequate evidence from the action to repair it. No external-web recovery is implemented here. |

### Practitioner reports: useful hypotheses, not benchmark evidence

A [Reddit report on a small support corpus](https://www.reddit.com/r/Rag/comments/1v7g3oe/hybrid_search_and_reranking_made_my_rag_worse/)
found hybrid retrieval and reranking worse than its dense baseline. Its corpus,
labels, model and metric differ from ours; the transferable lesson is to measure
components rather than assume a stronger-looking pipeline wins.

Another [evaluation-framework discussion](https://www.reddit.com/r/Rag/comments/1v7xiqg/built_a_framework_to_benchmark_rag_pipelines/)
flags the distinction between finding a document and actually including it in
the reader's context. These are self-reported anecdotes, not verified controlled
studies. We use them to motivate trace logging, not to establish effectiveness.

## Audit findings that matter before new methods

1. **Model interface:** the old Qwen wrapper passes raw text without the model's
   chat template. [Qwen's official example](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
   uses `apply_chat_template(..., add_generation_prompt=True)`. The corrected
   interface is opt-in for old runs and enabled for the new research harness.
   Both normal and streamed Qwen paths use the same formatter.
2. **Reader damage:** intermediate hypotheses are not necessarily final answers.
   Splitting names on punctuation, rejecting one-character answers, and trusting
   repeated guesses can lower EM. New extraction preserves the answer string.
3. **Silent context loss:** the old wrapper truncates at 4096 input tokens.
   Corrected runs use an explicit 8192-token budget and fail visibly if exceeded.
   This is not a claim to have implemented optimal evidence packing.
4. **Cache contamination:** the old cache key includes only passage text, so model
   changes can reuse incompatible vectors. New cache files include a namespace.
5. **Incomplete diagnostics:** recovered regex fields were counted as parse
   success and raw outputs were not retained in the hop trace. New runs record
   complete prompts, raw outputs, strict JSON status and output-budget hits.
6. **Misleading recall name:** the old `supporting_fact_recall_at_k` is title-level
   recall. New reports explicitly distinguish title recall, visible supporting
   sentence recall and recall of the sentences the reader actually cites.
7. **Reporting correction:** the previous candidate-reader run scored 46% EM,
   not 56%. See the correction in `RESEARCH_LOG.md`.

## Implemented experiment

`src/rag/sufficiency.py` and `scripts/research_experiment.py` add:

- **Top-five reader:** retrieve five paragraphs, generate one short answer with
  sentence citations and a brief evidence explanation.
- **All-context reader:** same reader sees every supplied paragraph in dataset
  order. This is a necessary simple baseline for this small candidate pool.
- **Oracle reader:** same reader sees the gold supporting paragraphs. Gold labels
  enter only this explicitly diagnostic arm, never normal prompts or retrieval.
- **Adaptive reader:** start with five paragraphs; ask a separate critic which
  relation remains missing. Stop retrieving only after an explicit sufficient
  verdict with valid citation IDs and no stated missing information. Otherwise
  retrieve up to two fresh paragraphs per step, at most three retrieval steps,
  then run the same final reader. Every critic call is counted.

A valid citation ID is only a structural check. It does not prove entailment or
that a multi-hop relation was correctly established. The critic is uncalibrated,
shares the reader's model, and may be confidently wrong. These are evaluation
questions, not properties established by naming it a verifier.

The old `legacy` and `grounded_candidates` implementations remain executable.
Comments explain which problematic behaviors the new path avoids. We do not
bury the old implementation in a large inactive commented-out duplicate.

## Frozen protocol (specified before new dev results)

1. Use five deterministic **train** examples for execution/schema smoke tests.
2. Use the existing 50 dev IDs only as an **exploratory**, repeatedly inspected
   cohort. Preserve exact IDs and report all ablations, including regressions.
3. Compare old fixed-hop reader with its chat-format correction on that cohort;
   compare the four new reader arms with Qwen2.5-7B-Instruct, MiniLM, seed42,
   greedy decoding, 192 output tokens **per call**. Adaptive has additional calls;
   do not describe this as equal total compute.
4. Test BGE-base-en-v1.5 using its documented query instruction, title metadata,
   and hybrid retrieval. Split these factors where practical, rather than
   attributing a bundled gain to one component.
5. Test Qwen2.5-14B-Instruct on the same controls to separate generator scale from
   method. Model downloads and weights are not part of measured request latency.
6. Freeze a disjoint 100-example dev confirmation manifest, excluding the old50,
   seed20260924. Run the prespecified simple and adaptive controls without tuning
   thresholds or prompts on those results. This is a held-out development sample,
   not the official hidden test or proof against training-data contamination.
7. Report EM, official-style token F1, title recall@5 (retrieved arms), visible
   sentence recall, citation fact recall, parse failures, budget hits, calls,
   generated tokens, mean hops, latency, stop reasons and failed examples. Oracle
   and all-context first-hop recall must not be called recall@5.
8. Use paired example-level comparisons and bootstrap confidence intervals.
   Publish failures and denominator checks. Fifty examples are insufficient for
   small or universal superiority claims.

## What might become a contribution

A testable direction is **missing-relation retrieval under a measured budget**:
judge which link in the evidence is absent, retrieve specifically for that link,
and stop only when the evidence covers the question. The combination is not
automatically novel; Google's sufficiency work, IRCoT, Self-RAG and corrective
retrieval are close prior art. A paper would need a precise difference, trained
and calibrated controls using train-only data, multiple datasets, larger corpus
retrieval, matched compute, and uncertainty estimates.

A practical success criterion here is a reproducible accuracy/cost improvement
over our corrected reader and simple all-context baseline. If the critic loses,
keep the simpler reader and report the negative result instead of changing the
story or tuning until the same 50 examples look good.
