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
2. Answer selection uses model confidence after the deterministic grounding
   guard, with grounding as a secondary tie-breaker. Unsupported confident
   answers are downgraded before selection instead of trusting raw confidence.
3. Added a second-stage `sentence-transformers` cross-encoder reranker. It
   reranks a configurable first-stage shortlist, so dense, BM25, and hybrid
   retrieval remain directly comparable.
4. Added `scripts/compare_predictions.py` to classify each matched example as
   retrieval-rescued, oracle-regressed, both-correct, or both-wrong.

## Matched experiment checklist

Run the same 50 IDs and frozen generator settings for dense, hybrid, and
hybrid-plus-cross-encoder retrieval. Append the resulting metrics and the
normal/oracle per-example report here only after `metrics.json` and the report
files have been verified on Renku. This checklist is now complete for the
23 September run; future cohorts should append a new dated section.

## Matched Renku results (23 September 2026)

Revision: `e9f673f` on branch `v-2`. Cohort: the generated
`outputs/example_ids_dev_50_seed42.json` manifest (50 IDs). All runs used
Qwen 7B, seed 42, greedy decoding, three hops, five paragraphs per hop, 192
new tokens per hop, and an explicit 15-paragraph evidence budget.

| Run | EM | F1 | Supporting-fact recall@k | Mean hops | Mean tokens | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| Dense fixed hops | 0.56 | 0.6622 | 0.82 | 3.0 | 576 | 7.4876 |
| Hybrid dense+BM25 (0.5/0.5) | 0.54 | 0.6709 | 0.93 | 3.0 | 576 | 7.4863 |
| Hybrid + `ms-marco-MiniLM-L-6-v2` cross-encoder | 0.48 | 0.6068 | 0.90 | 3.0 | 576 | 7.5490 |
| Oracle evidence (diagnostic ceiling) | 0.60 | 0.7258 | 1.00 | 1.0 | 192 | 2.3831 |

All four runs had zero failed examples and a reported parse-failure rate of
zero. The console contains occasional “extra data” JSON warnings where the
fallback parser recovered a valid object; this should be counted separately in
future parser diagnostics rather than treated as a clean generation signal.

The hybrid retriever improved first-hop evidence recall by 0.11 and F1 by
0.0087, but exact match fell by two examples. The cross-encoder did not help on
this cohort and reduced both EM and F1, so it should not replace the hybrid
baseline without a larger validation study. The oracle improves dense EM by
four points (two examples), while 17 examples remain wrong in both runs and
three are correct under dense retrieval but wrong with oracle evidence. This
shows that retrieval is a measurable bottleneck, but Qwen answer formulation
and evidence interpretation also limit the ceiling.

Per-example normal/oracle report: `outputs/dense_e9_vs_oracle.json` and
`outputs/dense_e9_vs_oracle.md` on Renku. Categories were 25 both-correct, 17
both-wrong, 5 retrieval-rescued, and 3 oracle-regressed.

## Reader/error-analysis ablation (24 September 2026)

Revisions: `615a16f` plus `41edcc5` on branch `v-2`. The new
`grounded_candidates` reader collects Qwen's answer, intermediate hypothesis,
and up to three alternatives at every hop. A deterministic verifier scores
exact phrase support in retrieved evidence, coarse question-type compatibility,
and repetition across hops. The original confidence-first selector remains
the `legacy` control. `scripts/error_analysis.py` writes a trace-based failure
taxonomy; its labels are diagnostic heuristics rather than manual gold labels.

Both readers used the same 50-ID manifest, Qwen 7B checkpoint, MiniLM
embedder, seed 42, greedy decoding, dense retrieval, three hops, five
paragraphs per hop, 192 new tokens per hop, and a 15-paragraph evidence
budget. The grounded-reader run is `outputs/dense_reader50_615a16f` and the
control is `outputs/dense_fixed50_e9f673f`.

| Reader | EM | F1 | First-hop supporting-fact recall | Parse-failure rate | Mean latency (s) |
|---|---:|---:|---:|---:|---:|
| Legacy control | 0.56 | 0.6622 | 0.82 | 0.00 | 7.4876 |
| Grounded candidates | 0.46 | 0.5835 | 0.82 | 0.00 | 7.5065 |

The grounded reader changed 15 of 50 predictions: one exact-match gain and
six exact-match losses, with the remaining changes still incorrect. It therefore
regressed by five correct answers on this frozen cohort. The result
is evidence against claiming that deterministic post-hoc verification alone
solves the reader bottleneck; a trained verifier, better answer extraction,
or a stronger generator is needed before tuning this component further.

### Correction recorded 24 September

The earlier version of this section incorrectly copied the control EM/F1 into
the candidate-reader row and reported one loss instead of six. Live Renku
`outputs/dense_reader50_615a16f/metrics.json` was re-read on 24 September:
EM=0.46, F1=0.5835238095238096, n=50, latency=7.50651058781892 seconds.
The correction is a reporting fix, not a rerun or a changed scoring rule.

The automated error taxonomy is provisional: first-hop title misses do not
prove evidence was missing at the final answer, because cumulative title recall
was 1.0. The script skips normal-correct cases, so it cannot establish the three
oracle-regression causes. A zero recovered-parse failure rate also does not
establish valid, complete JSON. Earlier stronger interpretations are withdrawn.

## Literature-driven reader/sufficiency experiment (24 September 2026)

Protocol and source attribution: `docs/RESEARCH_REVIEW_2026-09-24.md`.
The earlier heuristic reader remains available; new functions explain which old
behaviors are bypassed and why. Added opt-in Qwen chat templates shared by normal
and streamed generation, explicit input budget, model-isolated embedding cache,
raw hop outputs, a short cited-answer reader, and a separate prompted sufficiency
critic. No model training or learned stopping is claimed.

Train smoke: Renku lacks the train file; the local train download is incomplete.
Five records were sampled with seed20260924 from its first100 completely parsed
records, exported as valid JSON, and used only to test execution/schema. The
initial smoke exposed bracketed citation IDs (`[p6s0]` versus `p6s0`), preventing
valid critic stops. The parser now normalizes that display syntax; nonexistent
IDs are still rejected. This correction precedes new dev evaluation. Smoke
accuracy is not benchmark evidence. No thresholds were fitted on dev.

The current GPU allocation is an H200 NVL MIG 3g.71gb device. Models will run
sequentially to avoid resource contention. The new harness logs every critic
call, actual generated tokens, raw prompts and output, strict JSON health,
model revision, package versions, dataset/manifest hashes, latency, failures and
sentence/title/citation recall separately. The old50 cohort is exploratory;
a disjoint100 manifest is reserved for a frozen confirmation comparison.
