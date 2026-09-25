# Relation-aware answer verification: implementation plan

**Status:** plan approved for implementation; phase-1 isolated scaffolding is
now present, but no GPU benchmark result is claimed yet.

**Decision that motivates the plan:** the current strong reference is a static
one-call reader with Qwen2.5-14B and ranked all-context evidence. On the frozen
confirmation100 it scored 60% EM / 71.51% F1, while the adaptive critic scored
59% / 69.95% and used about twice the latency and output tokens. The current
prompted critic, reasoning stop, and dual stop therefore remain useful
ablations, but they are not the next main contribution.

## 1. Research question

Can a bounded, evidence-grounded verification and correction stage improve
answer correctness on multi-hop HotpotQA without adding the cost and
instability of an open-ended stopping loop?

The target errors are relation and operation errors: reversing a comparison,
answering an empire when the question asks for its predecessor, returning a
population when the question asks for an entity, failing an intersection, and
producing an answer that contradicts its own explanation. Retrieval coverage
alone cannot fix these cases; several inspected losses already had complete
annotated support in the retrieved context.

This is an adaptation for our bounded HotpotQA setting. It is not a claim that
we reproduce any cited paper or that the method is state of the art.

## 2. Evidence and paper credit

The design uses the following ideas as hypotheses, with the boundary of the
adaptation stated explicitly:

* [RARR](https://arxiv.org/abs/2210.08726) supports finding evidence for a
  generated claim and revising unsupported text. We will use local retrieved
  paragraphs and bounded answer revision rather than reproducing its complete
  research-and-revise system.
* [FActScore](https://arxiv.org/abs/2305.14251) motivates decomposing an answer
  into checkable atomic facts instead of treating a whole response as one
  factual unit. We will use short answer facts and evidence IDs for diagnostics;
  we will not present HotpotQA EM as FActScore.
* [Re-Ex](https://arxiv.org/abs/2402.17097) and
  [RAC](https://aclanthology.org/2025.findings-emnlp.1370/) motivate a
  retrieve/check/explain/revise sequence. Our revision is limited to supplied
  evidence and one bounded call, so their reported gains are not transferred
  to this project.
* [Provenance](https://aclanthology.org/2024.emnlp-industry.97/) motivates an
  independent compact entailment-style checker that can trace claims to
  chunks. This is important because the current critic shares the generator
  checkpoint and is not calibrated.
* [Generate but Verify](https://aclanthology.org/2025.ijcnlp-long.56/)
  motivates evaluating faithfulness as a first-class output rather than using
  an informal confidence prompt.
* [Teaching Language Models to Check Grounded Claim Factuality](https://aclanthology.org/2026.acl-long.1468/)
  motivates a constrained
  true/false-style verifier with explicit evidence rather than unrestricted
  hidden reasoning. We will log concise structured decisions, not private
  chain-of-thought.
* [FRAMES](https://aclanthology.org/2025.naacl-long.243/) motivates separating
  retrieval, factuality, and reasoning measurements. Its published numbers are
  not used as baselines for our HotpotQA runs.

The earlier retrieval hypotheses remain documented in
[`RESEARCH_REVIEW_2026-09-24.md`](RESEARCH_REVIEW_2026-09-24.md), including
Google's sufficient-context work, Anthropic's Contextual Retrieval, IRCoT,
Self-RAG, CRAG, and Lost in the Middle. Those papers support testing context
coverage and evidence order; they do not validate the current stopping loop.

## 3. Proposed architecture

### Phase 0: freeze the reference

Before implementation, create and commit a new example manifest that is
disjoint from the exploratory50 and confirmation100 cohorts. Record the exact
dataset hash, code revision, model, tokenizer, embedder, retrieval order, seed,
prompt IDs, evidence budget, and generation budget. The reference arm is:

* Qwen2.5-14B-Instruct;
* ranked all ten supplied paragraphs;
* one reader call, greedy decoding, the current short reader prompt.

Retain a matched 7B arm and an oracle-evidence arm as diagnostics. Oracle
evidence is evaluation-only and must never be described as a deployable mode.

### Phase 1: structured reader output

Add an optional schema to the reader. It should return only concise fields:

```json
{
  "answer": "...",
  "answer_type": "entity|date|number|yes_no|list|other",
  "operation": "lookup|comparison|intersection|count|boolean|other",
  "claims": [
    {"text": "...", "evidence_ids": ["title#sentence"]}
  ],
  "explanation": "short evidence-grounded explanation"
}
```

The schema is an observable intermediate representation, not a request for
hidden chain-of-thought. Parsing failures, missing fields, invalid evidence
IDs, and empty answers must be recorded separately.

### Phase 2: deterministic relation checks

Implement checks that do not use the gold answer:

1. Every cited evidence ID exists in the supplied context.
2. The answer type is compatible with the question's requested type.
3. The answer and explanation do not disagree about the entity, date, number,
   or comparison direction.
4. Numeric/date comparisons are executed from extracted values when possible.
5. Intersection questions require evidence for both sides and a non-empty
   intersection; count questions require a countable list.
6. A cited sentence is not automatically considered support merely because its
   title is present; the checker records citation validity separately from
   textual support.

These checks are safeguards and diagnostics, not an oracle. They must not use
gold titles, gold answers, or dev-derived thresholds.

### Phase 3: independent verification baseline

Evaluate two verifiers under the same frozen evidence:

* a compact entailment/NLI checker, following the provenance-style design; and
* a prompted verifier, used only as a comparison to the independent checker.

For each claim, the verifier returns `supported`, `unsupported`, or `uncertain`,
the evidence IDs considered, a calibrated-on-train confidence, and a short
error label (`missing_relation`, `wrong_type`, `contradiction`, `unsupported`,
or `other`). It cannot rewrite the answer directly. The verifier's false
positive and false negative rates must be measured on a train-derived
validation split before it is used to route revisions.

### Phase 4: one bounded correction call

If a claim is unsupported or contradicts another claim, send the original
answer, the flagged claim, and the available evidence to one revision call.
The revision prompt must preserve supported claims and may only replace a
flagged answer using the supplied evidence. If the verifier is uncertain, use a
predefined fallback policy rather than repeatedly calling the model. Record
whether the revision was accepted, rejected, unchanged, or caused a regression.

No open-ended critic/retrieval loop is part of this first experiment. Every
additional call, token, and second is charged to the arm.

### Phase 5: train-only calibration

Only after the untrained structured/checker arms are measured, construct
training examples with sufficient and deliberately incomplete contexts. Train
or calibrate a small verifier on train plus a train-derived validation split.
Thresholds and routing rules are fixed before the new test cohort is opened.
The reported test/dev examples are never used to tune a stop or revision
threshold.

### Phase 6: scale beyond the supplied ten paragraphs

Only a successful local verification result justifies a full-corpus experiment.
Then compare dense, BM25, hybrid, contextualized chunks, and a reranker with
the same reader and verifier. Add another multi-hop dataset, such as FRAMES or
2WikiMultiHopQA, only after checking availability, licensing, and a compatible
evaluation script.

## 4. Frozen evaluation arms

The first implementation experiment will use the same untouched manifest for
all arms:

| Arm | Reader / verifier | Purpose |
|---|---|---|
| A | Current 14B ranked-all static reader | primary reference |
| B | Structured reader only | schema and relation representation cost |
| C | Structured reader + deterministic checks | isolate rule-based safeguards |
| D | Structured reader + independent verifier | measure verification without rewriting |
| E | Structured reader + verifier + one revision | measure bounded correction |
| F | Oracle evidence + current reader | generation diagnostic only |

The 7B reference is a secondary scale check. Existing adaptive and stopping
arms remain historical comparisons, not a threshold-tuning playground.

## 5. Measurements and detailed logs

For every example and arm, save raw output and a JSON trace containing:

* question ID, code/data/model/embedder hashes and prompt IDs;
* retrieved titles, sentence-addressed evidence, and evidence order;
* answer, answer type, operation, claims, citations, and explanation;
* each deterministic check and its reason;
* verifier labels/confidences and revision route;
* final answer and whether it changed;
* EM, F1, answer-type accuracy, relation/operation accuracy;
* claim support precision/recall, unsupported-answer rate, and
  answer/explanation consistency;
* parse/schema failures, calls, input/output tokens, latency, and failed IDs.

The aggregate report must include paired per-example deltas, revision gains,
revision regressions, unchanged cases, uncertainty intervals, and a table of
the 20 most informative failures with the original and revised evidence. Raw
model text is retained for audit; private chain-of-thought is not requested or
stored.

## 6. Predefined analysis and stop/go criteria

Use paired bootstrap intervals and per-example matched comparisons. Do not
select a favorable subset after seeing results. The experiment proceeds to a
larger corpus only if the bounded correction arm:

1. improves EM on the untouched cohort by at least 3 percentage points over
   the static 14B reference, with a paired interval that is not entirely below
   zero;
2. does not increase answer/explanation contradictions or unsupported answers;
3. keeps added latency and output tokens explicitly reported and acceptable for
   the intended use; and
4. has no unexplained schema or verifier failure spike.

These are practical project gates, not claims of statistical significance from
one small cohort. If no arm meets them, retain the static reader, report the
negative result, and stop investing in the current prompted critic and
reasoning-stop mechanism as the main method. If verification improves
correctness but costs too much, keep it as an optional high-accuracy mode and
report the cost curve.

## 7. Risks and controls

* A verifier that shares Qwen can repeat the same error. The independent NLI
  arm and raw claim logs expose this risk.
* NLI models can be poorly calibrated for multi-hop Wikipedia facts. Report
  calibration and uncertain cases rather than forcing binary decisions.
* Deterministic type and comparison rules can be brittle for aliases, units,
  and list ordering. Keep official HotpotQA EM/F1 unchanged and report these
  checks as secondary diagnostics.
* Post-generation correction can damage a correct answer. Every revision is
  paired against the original and counted as a gain, loss, or unchanged case.
* Oracle evidence and gold-title coverage are diagnostics only. They must not
  enter prompts, labels, or thresholds for the deployable arms.

## Expected code and documentation changes

The implementation should add isolated modules rather than rewrite the legacy
runner:

* `src/rag/structured_answer.py` for schema parsing and validation;
* `src/rag/relation_checks.py` for deterministic checks;
* `src/rag/claim_verifier.py` for independent/prompted verifier adapters;
* `src/rag/revision.py` for one bounded correction call;
* `scripts/run_verification_experiment.py` for frozen matched arms;
* `scripts/analyze_verification.py` for paired metrics and failure tables;
* `research_results/<date>/` for manifests, raw traces, metrics, and hashes.

The legacy stopping code remains available for reproducibility and is not
silently deleted. Each new module will include a short module-level comment
that names the relevant paper and states exactly which idea is adapted.

The phase-1 files now present are `structured_answer.py`,
`relation_checks.py`, `claim_verifier.py`, `revision.py`,
`run_verification_experiment.py`, and `analyze_verification.py`. Phase 1
implements the structured reader, deterministic checks, prompted verifier,
one bounded revision, and raw per-example logging. The prompted verifier is
explicitly a comparison arm: it is not yet an independent or train-calibrated
verifier. NLI integration, train-only calibration, and a new frozen cohort are
required before any accuracy claim.
