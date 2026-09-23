# Dual-Stopping RAG research log

This is the running record for experiments that may be used in the paper. Each
entry should name the code revision, frozen example manifest, model settings,
and output directory before making a performance claim.

## Evaluation protocol

- Dataset: HotpotQA distractor development set.
- Cohort: `example_ids_dev_50_seed42.json` (the same IDs for matched runs).
- Generator: `Qwen/Qwen2.5-7B-Instruct`, greedy decoding (`temperature=0`,
  `top_p=1`, `max_new_tokens=192`).
- Dense embedder: `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieval: five paragraphs per hop, at most three hops unless a method says
  otherwise. Seed 42.
- Metrics: EM, token F1, supporting-title recall, parse failures, hops, output
  tokens, latency, stop reasons, and failed example IDs.

## Baseline (50 examples, before the current retrieval changes)

| Method | EM | F1 | Supporting-title recall | Mean hops | Mean generated tokens |
|---|---:|---:|---:|---:|---:|
| Fixed hops | 0.56 | 0.6622 | 0.82 | 3.0 | 576 |
| Retrieval stopping | 0.56 | 0.6622 | 0.82 | 2.0 | 384 |
| Reasoning stopping | 0.56 | 0.6622 | 0.82 | 3.0 | 576 |
| Dual stopping | 0.56 | 0.6622 | 0.82 | 2.0 | 384 |
| Oracle evidence | 0.60 | 0.7258 | 1.00 | 1.0 | 192 |

The oracle row supplies gold supporting paragraphs and therefore diagnoses the
generation ceiling; it is not a deployable retriever. The small EM gap (two of
50 examples) indicates retrieval matters, while the remaining errors also
involve answer extraction, evidence interpretation, or the model itself.

## Current implementation changes

1. Evidence formatting no longer silently drops paragraphs after ten. The
   runner now has an explicit `--max_evidence_paragraphs` budget; omitted means
   preserve all retrieved evidence.
2. Answer selection now combines deterministic evidence grounding, the model's
   confidence, evidence sufficiency, and evidence coverage. It no longer picks
   solely from the model's highest self-reported confidence.
3. Added a second-stage `sentence-transformers` cross-encoder reranker. It
   reranks a configurable first-stage shortlist, so dense, BM25, and hybrid
   retrieval remain directly comparable.
4. Added `scripts/compare_predictions.py` to classify each matched example as
   retrieval-rescued, oracle-regressed, both-correct, or both-wrong.

## Pending matched experiments

Run the same 50 IDs and frozen generator settings for dense, hybrid, and
hybrid-plus-cross-encoder retrieval. Append the resulting metrics and the
normal/oracle per-example report here only after `metrics.json` and the report
files have been verified on Renku.
