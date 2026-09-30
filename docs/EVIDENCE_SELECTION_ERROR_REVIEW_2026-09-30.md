# Initial evidence-selection error review

Scope: pooled-dev corpus, 100 frozen questions; plain versus title+lead BM25.
These are retrieval observations, not new gold labels or prompt changes.

## What changed in the completed lexical comparison

- `5ab6be2f55429954757d335c` (Zakk Wylde / Damon Albarn): title+lead brings
  the Zakk Wylde biography into top5. Raw BM25 preferred related albums and
  collaborators. This supplies the missing entity's occupation sentence.
- `5abbe188554299642a094ba0` (highest-paid model / Fair Game role): title+lead
  retrieves **Fair Game (1995 film)** at rank2; raw BM25 retrieved the unrelated
  Scientology sense but missed the movie in top5. Entity disambiguation is a
  plausible reason for the gain, inferred from the saved rankings.
- `5ab6faec5542991d32223708` (literary award comparison): the contextual list
  includes both author biographies; the raw list contains books and related
  writers while missing the García Márquez support. A correct comparison still
  requires the reader to interpret the award evidence.
- `5a80752b5542992bc0c4a720` (Simpsons episode air date): both lists contain
  Matt Selman, but only the contextual top5 contains the needed sentence about
  his episode-writing credit. This is why title recall alone is inadequate.

## Regression retained

`5a8b7d7d5542997f31a41d4b` (Six by Sondheim director): contextual top5 loses
James Lapine's biography and introduces Tony Leung Ka-fai and Ben Wright.
Repeated title/lead terms can also promote distractors. This is an observed
ranking change; it does not prove the exact causal weight of a single token.
The film evidence may still answer the question, so incomplete annotated support
must not automatically be called insufficient context.

The comparison has eight complete-support gains and one regression at k=5.
Every case, including unchanged failures, is retained in
`research_results/2026-09-30/pooled_bm25/per_example.json`. Parameters stay frozen;
these dev cases are not used to tune term weights or stopping thresholds.
