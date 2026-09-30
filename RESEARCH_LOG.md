# Dual-Stopping RAG research log

This is the running record for experiments that may be used in the paper. Each
entry should name the code revision, frozen example manifest, model settings,
and output directory before making a performance claim.

## 2026-09-25 — plan for relation-aware answer verification

**Status: plan documented; implementation and new evaluation have not started.**

The current evidence does not justify abandoning retrieval. It does justify
pausing the prompted adaptive critic, reasoning-token stopping, and dual
stopping as the main idea. On confirmation100, the strongest observed static
reader (Qwen2.5-14B, ranked all ten paragraphs, one call) scored 60% EM /
71.51% F1 with 66.7 output tokens and 1.74 seconds per example. The adaptive
critic scored 59% / 69.95% with 149.8 output tokens and 3.51 seconds. The
adaptive result is retained as a fair historical comparison; it is not being
silently removed.

The next experiment is specified in
[`docs/ANSWER_VERIFICATION_PLAN_2026-09-25.md`](docs/ANSWER_VERIFICATION_PLAN_2026-09-25.md).
It tests a structured answer, deterministic relation/operation checks, an
independent entailment-style verifier, and one bounded evidence-only revision
call. The aim is to address observed comparison-direction, answer-type,
intersection, and answer/explanation contradiction errors. It deliberately
avoids another open-ended stopping loop.

The plan adapts ideas from RARR, FActScore, Re-Ex, RAC, Provenance, Generate
but Verify, grounded claim checking, and FRAMES. The plan links each paper and
states where our bounded HotpotQA implementation differs; their reported
numbers are not imported as expected results. Google sufficient-context,
Anthropic Contextual Retrieval, IRCoT, Self-RAG, CRAG, and Lost in the Middle
remain background evidence for context and retrieval controls in the earlier
research review.

Before code changes, the new test manifest must be frozen disjoint from both
the exploratory50 and confirmation100 cohorts. All arms will share IDs,
dataset hash, seed, model, embedder, evidence order, and generation budgets.
The oracle arm remains an evaluation-only generation diagnostic. No gold answer,
gold title, or dev-derived threshold may enter a deployable prompt or routing
rule.

The new logs will preserve raw outputs, evidence IDs and sentences, structured
claims, operation/type checks, verifier labels/confidences, revision decisions,
per-example EM/F1, support precision/recall, contradiction and unsupported
rates, calls, tokens, latency, parse/schema failures, and failed IDs. Reports
will show paired gains, regressions, unchanged cases, bootstrap intervals, and
the most informative failures. Private chain-of-thought is neither requested
nor stored.

The predefined gate is at least +3 EM points over the static 14B reference on
the untouched cohort, no increase in unsupported or contradictory answers, and
an acceptable measured cost. If this gate is not met, the static reader remains
the reference and the current critic/stopping idea is reported as a negative
result rather than tuned indefinitely. A verifier that improves accuracy but
costs more may remain an optional high-accuracy mode.

Expected additions are isolated structured-answer, relation-check, verifier,
revision, experiment-runner, and analysis modules. The legacy stopping runner
will remain intact for reproducibility. At this point this is a documented
plan; no GPU result is claimed in this entry. The implementation is recorded
below at commit `c67f7ce`, and the frozen cohort at commit `8af743e`; an output
directory and Renku revision are still required before reporting measurements.

### Phase-1 implementation (same date, no benchmark yet)

The plan was written and committed before the implementation. The isolated
phase-1 modules are now in the source tree:

* `src/rag/structured_answer.py` parses the concise answer/type/operation/
  claim schema and records structural failures.
* `src/rag/relation_checks.py` performs explainable type, operation, citation,
  lexical claim-support, numeric consistency, comparison, and intersection
  checks without gold answers.
* `src/rag/claim_verifier.py` parses the prompted verifier contract. This is a
  comparison arm only; it is not being described as independent or calibrated.
* `src/rag/revision.py` builds one evidence-only correction prompt.
* `scripts/run_verification_experiment.py` runs matched `static`, `structured`,
  `checked`, `verified`, `revised`, and `oracle` arms and refuses to overwrite
  output directories.
* `scripts/analyze_verification.py` reports per-example gains, regressions,
  unchanged cases, revision decisions, and failure reasons.

All new files contain module-level paper attribution. The runner passed Python
compile, synthetic parser/checker smoke tests, and the existing focused suite
(`PYTHONPATH=. pytest ...`): 18 tests passed. No new cohort has
been opened, no threshold has been tuned, and no GPU accuracy result exists
yet. A new disjoint 100-example manifest is now frozen at
`research_results/2026-09-25/verification_ids_100_seed20260925.json`; it
excludes the earlier 50-example, confirmation100, and train smoke manifests.
Its ordered ID SHA256 is
`c52904c8119e3b79abd83c9ed6cbf27e2a9569ee24a5de71339eef0d955dc633`. The next
entry must name the implementation revision and exact Renku output directories
before reporting EM/F1.

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

### Implementation and evaluation audit

The research implementation begins at `6de44cd`; `30c564f` adds the static
budget controls and paired reporter. Later result-only commits preserve
completed artifacts. Each new run records its exact code/model revision and
working-diff hash; published non-smoke runs started with a clean tracked diff.
The old source paths remain runnable. Human-readable comments attribute Google
sufficiency/IRCoT inspiration and explain removed selection behavior in the new
path. A prompted critic is not a learned Stop-RAG controller.

Relevant implementation files:

- `src/llm/qwen.py`, `src/llm/generation.py`: shared Qwen chat formatting and
  explicit input handling; legacy raw formatting is retained for comparison.
- `src/retrieval/index.py`, `src/retrieval/embedder.py`: embedding namespaces and
  BGE query instruction; title enrichment is a metadata baseline.
- `src/rag/sufficiency.py`: sentence IDs, cited short-answer prompt, evidence
  critic, structural citation validation and preserved answer strings.
- `scripts/research_experiment.py`: label-isolated oracle, top5/all-context,
  adaptive 5→7→9 retrieval, and static top9/ranked-all controls.
- `scripts/report_research_results.py`, `scripts/paired_research_analysis.py`:
  matched-ID/gold checks, rescoring and paired bootstrap intervals.
- `scripts/analyze_research_critic.py`, `scripts/analyze_research_slices.py`:
  post-run evidence-coverage and question-type diagnostics, with no tuning.

Validation: all 18 targeted tests passed on Renku with PyTorch. Locally 15 passed
and three PyTorch-dependent tests were skipped. Reporting scripts are exercised
against exported real artifacts; compile success alone is not treated as GPU
runtime verification.

The frozen100 manifest excludes all original50 IDs and uses sampling seed
20260924; generation uses seed42. It contains 85 bridge and 15 comparison
questions, all labeled hard. All ten candidate paragraphs fit within the new
reader's explicit 8192-token input limit in these completed runs. No dev-based
prompt or threshold tuning is performed. The train smoke subset is a wiring
check, not controller training.

The legacy raw-versus-chat control differs only in formatting/input handling;
the historical older prompt is reported separately. Supplemental 7B top9 and
ranked-all controls were specified before confirmation results. The equivalent
14B controls were added **after** observing 59% adaptive versus 53% all-context
EM on confirmation100. These are explicitly post-hoc mechanism checks, not a
second independent confirmation. Their prompts/settings are unchanged.

Raw logs remain on Renku under `outputs/research*`; completed compact artifacts
are tracked under `research_results/2026-09-24`. They include model/package
versions, IDs/hashes, predictions, source sentences, raw outputs, per-call input
and output token counts, stop reasons, and parse/schema diagnostics. The analysis
counts empty/wrong answers separately from execution failures. Original gold
answers and official-style EM/F1 are never corrected using the manual review.

A CPU-only audit downloaded tokenizer/config JSON at the recorded MiniLM/BGE
revisions and counted complete paragraph lengths. Plain MiniLM overflowed 17/500
paragraphs on original50 and 41/1000 on confirmation100; none were gold-title
paragraphs. BGE+title original50 overflowed 2/500, also no gold-title paragraphs.
The full tables and affected IDs are in `docs/EMBEDDING_AUDIT_2026-09-24.md` and
`research_results/2026-09-24/embedding_input_audit.json`. This is not an additional
generation run and does not change the frozen pipeline. The title-prefix100
variants in that audit were not benchmarked for QA.

Dataset integrity: local and Renku byte hashes differ because their JSON
serialization differs. Independently computing the canonical JSON SHA256 on
both copies produced `8c45526920c1b11d0f16d59775c8a53b7530a48a6309f423c8f79cecbe9e8b3c`
for all 7,405 records, including list order. Therefore the local manual/length
audits use identical parsed data. Both cohorts contain unique IDs and have zero
overlap. File hashes and verification details are in `dataset_integrity.json`.

### Completed outcomes and interpretation

All 23 new dev evaluation arms completed: 1,650 predictions over 150 unique
questions, with zero execution failures. Five historical arms are retained in
the 28-arm report. Repeated predictions of the same question are not independent
samples. Train smoke checks are excluded from these performance counts.

| Fresh100 configuration | EM | F1 | Mean output tokens | Mean seconds |
|---|---:|---:|---:|---:|
| 7B / top5 | 0.37 | 0.4973 | 65.7 | 1.01 |
| 7B / all10 dataset order | 0.49 | 0.6107 | 69.2 | 1.19 |
| 7B / adaptive | 0.45 | 0.5950 | 206.7 | 2.54 |
| 14B / top5 | 0.48 | 0.5709 | 65.3 | 1.71 |
| 14B / all10 dataset order | 0.53 | 0.6353 | 66.0 | 1.91 |
| 14B / adaptive | 0.59 | 0.6995 | 149.8 | 3.51 |
| 14B / top9, post-hoc control | 0.57 | 0.6941 | 65.5 | 1.91 |
| 14B / all10 ranked, post-hoc control | 0.60 | 0.7151 | 66.7 | 1.74 |

The 7B top9/ranked-all controls score 42%/44% EM. Full tables include initial
title recall, final visible sentence recall, cited supporting facts, calls,
input/output tokens, parse/schema failures, stop reasons and failed predictions.

The original50 adaptive result (62% EM versus 52% all-context) did not replicate
with 7B on fresh100 (45% versus 49%). With 14B, adaptive initially appears better
than dataset-order all-context by 6 points, but its paired 95% interval is −1 to +13.
The stronger static controls remove the apparent practical advantage: ranked-all
scores 60%, compared with 59% adaptive, at about half the observed latency and 44.5%
of its generated tokens. Adaptive minus ranked-all has a paired EM interval of
−7 to +5 points. This does not prove equality; it does not establish an adaptive
gain either. The 14B static controls are explicitly post hoc.

A useful measured efficiency result remains: on original50, the short all-context
reader and chat-formatted legacy reader both have 52% EM, while the short reader
uses about 79% fewer output tokens and 72% less observed time. Its EM interval is
−8 to +8 points, so general accuracy equivalence is not established. New-reader
savings come from the short prompt/one-call design, not reasoning stopping.

BGE+title+hybrid top5 improves original50 EM from 52% with the matched MiniLM
hybrid configuration to 58% (three gains, no losses; interval 0 to +14 points).
This is exploratory and has not been confirmed on fresh100. The oracle reference
scores 60%/75.87% EM/F1 with 7B and 64%/78.93% with 14B on original50; oracle context
is not a mathematical ceiling, and its gold evidence is evaluation-only.

Manual review documents answer-type errors, relation/comparison failures,
unsupported citations, answer–explanation contradictions, aliases and an
apparent question–gold inconsistency. Eleven of the 7B adaptive method's twelve
losses against all-context on fresh100 still have complete annotated supporting
sentences. Retrieval alone cannot explain these losses.

Decision: keep the simpler short reader as a strong baseline and replicate
14B ranked-all on untouched data before selecting it as a final system. Do not
claim that the current critic, dual stopping, or a vector database beats market
RAG systems. The next research hypothesis is grounded relation/operation checking
and calibrated train-only control, with strong static baselines and full-corpus
retrieval. See `docs/RESEARCH_CONCLUSIONS_2026-09-24.md` for the detailed plan.

## 2026-09-30 / 2026-10-01 — evidence-selection implementation and first results

Plan: `docs/EVIDENCE_SELECTION_PLAN_2026-09-30.md`, recorded before measurement.
Implementation commit: `d47e34c`. Source pushed to GitHub; Git commits transferred
as a Git bundle (not an archive deployment) to the freshly started Renku checkout
and pushed there through its existing GitLab authentication to `v-2`.

Renku's launch UI stayed at "Requesting session resources" while its ttyd endpoint
was already reachable. Direct navigation to the terminal endpoint printed in the
service logs worked. This is a UI/session readiness mismatch, not a claim that
Renku networking was down. Fresh checkout path is
`/home/renku/work/dual-stoppingrag-code`; earlier checkout and installed ML packages
were absent. Dependencies are being restored. Historical compact artifacts remain
in the backed-up repository.

### Corpus integrity

Local `hotpot_train_v1.1.json` is truncated at character 103423974 (JSON decode
failure), so it was excluded rather than partially ingested. The first valid
index pools all 7,405 dev records' context paragraphs: 66,635 distinct exact
(title, sentence-list) documents, 81,276 chunks, sentence-boundary budget 120 words.
Same-title differing versions remain separate. This is a **pooled-dev diagnostic**,
not official fullwiki and not a train-derived corpus. Corpus building ignores
questions, answers and support annotations; it includes every available context,
not just selected examples or gold paragraphs. Every frozen example has all gold
support available in this corpus.

The official 1,553,565,403-byte fullwiki introductory archive was downloaded
locally and verified against the published MD5 `01edf64cd120ecc03a2745352779514c`.
Importer reads archive members without extracting paths and preserves sentence
addresses. Fullwiki model runs are pending, not implied by the download.

### Completed local BM25 comparison (100 frozen IDs)

Run: `outputs/evidence_20260930_bm25`; compact source artifacts and paired analysis:
`research_results/2026-09-30/pooled_bm25/`. No query/threshold tuning on this cohort.

| Index text | Sentence recall@5 | Complete gold support@5 | Sentence recall@10 | Complete gold support@10 |
|---|---:|---:|---:|---:|
| Raw chunk | 0.7110 | 0.45 | 0.8100 | 0.64 |
| Title + document lead + chunk | 0.7535 | 0.52 | 0.8508 | 0.70 |

Contextual minus raw sentence R@5: +0.0425, paired bootstrap 95% interval
[+0.0083, +0.0792]. Complete-support@5 improves on eight questions and regresses
on one; net +7 points, interval [+2,+13]. These are exploratory retrieval results,
not EM/F1 improvement or human sufficiency judgments. Context is deterministic
metadata/lead enrichment inspired by Anthropic; it is not their LLM-generated
full-document method. Reader evidence remains original sentences only.

### Implementation and validation

New modules use persistent SQLite FTS5 corpus-wide BM25, batched normalized dense
embeddings with corpus/model/variant-keyed disk caches, reciprocal rank fusion
(k=60), and a top100 cross-encoder reranker. Logs retain every rank, score, original
sentence, missing support, query latency, tokenizer overflow and per-example
coverage at k=5/10/20. Analysis includes paired bootstrap deltas and regressions.
Separate `read_corpus_evidence.py` fixes Qwen's reader prompt and logs EM/F1 and
optional sufficiency decisions without allowing gold labels into model calls.
Semantic sufficiency precision/recall requires independent labels; complete
annotated support is deliberately reported under its own name.

24 targeted tests passed after fixing a Python 3.9 streaming tar/bz2 compatibility
issue. Two dense attempts failed before inference because sentence-transformers
2.3 expected an empty stateless Normalize directory absent from the hub snapshot;
that loader compatibility issue was fixed without changing model weights. Their
protocols are marked failed, not silently removed. Dense/hybrid/reranker and reader
runs remain pending until complete artifacts exist.

### Wiring audit before downstream use

The earlier unbenchmarked verification scaffold did not pass the candidate JSON
to the prompted verifier, and wrapped its parse result as if it were always
strict JSON. Fixed both in `scripts/run_verification_experiment.py` and added a
regression test with an intentionally malformed verifier response. The new corpus
benchmark uses the separate frozen static reader, so its rankings and BM25
results are unaffected. The older deterministic checker remains heuristic; this
fix does not make it an independent, calibrated, or semantic relation verifier.

Full suite before that new regression test: 28 passed. The added wiring test and
10 corpus tests also pass. Exact new GPU runtime and checkpoint revisions will
be captured in each Renku protocol, rather than assuming the old environment.

### Renku pooled-corpus retrieval completed (100 frozen IDs)

All eight arms completed on the same 66,635-document / 81,276-chunk dev-context
pool. Artifacts were pushed to DCU GitLab `v-2` in commit `c35c30c` under
`research_results/2026-09-30/evidence_20260930_renku_retrieval/`.

| Arm | Sentence R@5 | Complete support@5 | Sentence R@10 | Complete support@10 |
|---|---:|---:|---:|---:|
| BM25 | .7110 | .45 | .8100 | .64 |
| Dense MiniLM | .6415 | .36 | .7398 | .52 |
| Hybrid RRF | .7442 | .52 | .8333 | .69 |
| Hybrid + cross encoder | .8160 | .63 | .8800 | .77 |
| Contextual BM25 | .7535 | .52 | .8508 | .70 |
| Contextual dense | .6315 | .33 | .7232 | .49 |
| Contextual hybrid | .7608 | .53 | .8483 | .70 |
| Contextual hybrid + cross encoder | .8143 | .62 | .8742 | .74 |

Plain reranked minus BM25 sentence R@5 = +.1050, paired bootstrap 95% interval
[+.0567, +.1550]. Complete-support difference = +.18, interval [+.08,+.28]:
23 rescued / 5 regressed. Hybrid alone has an interval crossing zero for R@5
[-.0155,+.0825]. Dense alone is weaker on this cohort. Context prefixes help
lexical retrieval but do not improve the reranked arm. This is evidence for
reranking this candidate pool, not evidence of novel or market-leading RAG.

Frozen embedding revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
reranker revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
MiniLM's 256-token limit truncated 23 plain and 669 contextual index inputs
out of 81,276; the 512-token reranker limit affected zero plain / one contextual
query-passage pair. Reader evidence still uses the original complete sentences.
This overflow difference is a limitation, not a proven explanation of the dense
regression. Runtime: torch 2.14.1, numpy 2.5.0, sentence-transformers 6.1.0,
transformers 5.12.1 on Renku's H200 NVL MIG 3g.71gb allocation.

Qwen2.5-14B reader/critic calls and the official fullwiki index are still running
at this entry. Do not treat the table above as fullwiki or answer EM/F1 results.
The reader analyzer preserves compressed raw calls and separates critic cost
from reader cost, paired answer gains/regressions, parser failures, and coverage
strata. Critic agreement with gold-support coverage is not human sufficiency
accuracy. Full local suite after the analyzer and verifier regressions: 31 passed.

### Execution checkpoint (30 September 2026, 23:17 UTC)

The same 31 tests passed on Renku (4.73 seconds). DCU GitLab `v-2` contains the
implementation, completed retrieval artifacts, and analysis scripts; remote merge
commit `13820b3` preserves both the result commit and local source commits. GitHub
contains source commit `05da077`; remote result commits are not yet mirrored there.
No force-push or history replacement was used.

The pooled Qwen reader uses resolved checkpoint
`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, one reader call and one diagnostic
critic call per question/arm, k=5, temperature=0, seed=42, output budget=192.
The fullwiki continuation checks this same reader revision and pins both
retrieval snapshots. No thresholds or prompts were tuned on the reported IDs.

Renku process 929 runs the pooled suite; 1468 builds/evaluates official fullwiki
lexical retrieval; 2735 waits for those jobs, checks completed protocols, archives
raw traces, updates this log with generated reports, pushes result commits to
GitLab, then executes the fullwiki dense/hybrid/reranker and reader comparisons.
Status: `outputs/evidence_completion_status.txt`; logs:
`outputs/evidence_suite_console.log`, `outputs/fullwiki_suite_console.log`, and
`outputs/evidence_completion_console.log`. A failed prerequisite stops the
continuation and records failure. These process handles are launch receipts,
not completed-result evidence. The official index was still building at this
checkpoint. Optional 2Wiki replication has not run.

### Backup and regression audit (23:21 UTC)

The completed retrieval artifact files were downloaded from authenticated GitLab
and their Git tree checked against the original result commit. Local and GitHub
were then fast-forwarded to the exact DCU merge commit `9d5e9db`; the prior
checkpoint's GitHub result-backup gap is resolved. No credentials were copied
between machines and no commit contents were altered during mirroring.

Reviewed all five complete-support regressions from plain reranking. The observed
pattern is loss of a complementary page: Baldwin, Collaborations 3, Tremont,
Totalitarianism, or Kevin Garnett, while closely related pages about the main
entity remain. The detailed case table is in
`docs/EVIDENCE_SELECTION_ERROR_REVIEW_2026-09-30.md`. This motivates a future
joint-evidence selection hypothesis, but does not justify tuning on these IDs
or claiming five answer failures before joining reader predictions.
