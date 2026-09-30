# Evidence-selection results

Corpus: pooled-contexts; 66,635 documents, 81,276 chunks.

100 frozen questions. Recall is original sentence evidence coverage, not answer accuracy or semantic sufficiency.

| Arm | Sentence R@5 | Complete@5 | Sentence R@10 | Complete@10 | Query seconds |
|---|---:|---:|---:|---:|---:|
| plain_bm25 | 0.711 | 0.450 | 0.810 | 0.640 | 0.159 |
| contextual_bm25 | 0.754 | 0.520 | 0.851 | 0.700 | 0.178 |

Paired bootstrap intervals use 10,000 draws, seed 42; multiple comparisons are exploratory.
Raw rankings, scores, source sentences and missing evidence remain in the original run. All failures are retained.
Corpus availability and tokenizer truncation are recorded in JSON artifacts. EM/F1 requires the separate reader run.
