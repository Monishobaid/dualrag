# Evidence-selection experiment — 30 September 2026

## Protocol fixed before results

Retain Qwen2.5-14B short static reader as the generation reference. First isolate
retrieval so errors in the unbenchmarked verifier cannot conceal retrieval gains.
Use the existing, not-yet-evaluated 100 IDs in
`research_results/2026-09-25/verification_ids_100_seed20260925.json`; seed 42.
No threshold tuning on these IDs. This extends the umbrella plan, not a new thesis.

Build a persistent, deduplicated corpus independently of questions/answers/support
labels. Initial integration benchmark pools **all available train and dev context
paragraphs**, searches the entire shared pool for every question, and records its
scope as `pooled-contexts`, never official HotpotQA fullwiki. Also implement a
streaming importer for the official fullwiki introductory-paragraph archive.
Report coverage gaps instead of injecting gold evidence into ranked results.

Arms, fixed before measurement: BM25; dense MiniLM; reciprocal-rank-fusion
(RRF k=60) of each top100; hybrid top100 reranked by
`cross-encoder/ms-marco-MiniLM-L6-v2`; corresponding contextual index arms.
Return top5/10/20 evidence chunks with identical budgets. Reranker scores are
relevance, not multi-hop sufficiency. Count indexing and query costs separately.

Chunks retain original sentence addresses. Plain retrieval indexes chunk text;
contextual retrieval prefixes document title and first sentence to each chunk.
This deterministic document-context baseline is inspired by Anthropic but is
NOT its LLM-generated full-document contextualization. Reader sees only original
sentences, never fabricated summaries. No question or answer enters indexing.

Record title recall, exact supporting-sentence recall, complete gold-support
coverage, corpus-availability ceiling, per-example rankings/scores, evidence,
latency, IDs and hashes. Complete support is a coverage proxy, NOT a sufficiency
label; absence does not prove unanswerability. Optional prompted sufficiency and
reader calls save raw outputs, parse failures, tokens, latency, and EM/F1 with gold
used only after generation. Paired bootstrap deltas compare identical IDs.
Human sufficiency precision/recall remains unavailable without independent labels.

Model checkpoints, tokenizer limits, truncation counts, dependency versions,
corpus/source hashes, prompt text, command and code revision go in protocol.json.
No overwrite of completed experiments. Synthetic tests must cover no gold leakage,
deduplication, sentence-address alignment, fusion, reranker ordering, and metrics.

## Attribution and limits

- HotpotQA (Yang et al., EMNLP 2018): dataset and sentence support supervision.
  https://hotpotqa.github.io/ ; https://hotpotqa.github.io/wiki-readme.html
  Official intro archive: 1,553,565,403 bytes; CC BY-SA 4.0. The pooled corpus is
  a smaller diagnostic, not a substitute for the official fullwiki evaluation.
- Anthropic Contextual Retrieval: contextualizing chunks before lexical/dense
  indexing. https://www.anthropic.com/engineering/contextual-retrieval
- Cormack et al., SIGIR 2009: reciprocal rank fusion (not a learned retriever).
  https://doi.org/10.1145/1571941.1572114
- Sentence Transformers MS MARCO cross encoder: query/passage relevance model.
  https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
- Sufficient Context: coverage and semantic sufficiency are separate.
  https://arxiv.org/abs/2411.06037
- 2WikiMultihopQA (Ho et al., COLING 2020) is a follow-up dataset, using its
  official data and alias-aware scorer. Merely accepting its Hotpot-shaped schema
  does not establish completed evaluation. https://github.com/Alab-NII/2wikimultihop

## Decision

Publish every arm including regressions. A retrieval gain needs higher paired
sentence/complete-support recall at the same k; generator improvement requires a
separate matched EM/F1 result. Do not call more retrieved words an accuracy gain.
Fullwiki and second-dataset runs remain pending until actual artifacts exist.
