# Evidence-selection results

Corpus: pooled-contexts; 66,635 documents, 81,276 chunks.

100 frozen questions. Recall is original sentence evidence coverage, not answer accuracy or semantic sufficiency.

| Arm | Sentence R@5 | Complete@5 | Sentence R@10 | Complete@10 | Query seconds |
|---|---:|---:|---:|---:|---:|
| plain_bm25 | 0.711 | 0.450 | 0.810 | 0.640 | 0.063 |
| plain_dense | 0.642 | 0.360 | 0.740 | 0.520 | 0.008 |
| plain_hybrid | 0.744 | 0.520 | 0.833 | 0.690 | 0.071 |
| plain_reranked | 0.816 | 0.630 | 0.880 | 0.770 | 0.155 |
| contextual_bm25 | 0.753 | 0.520 | 0.851 | 0.700 | 0.067 |
| contextual_dense | 0.631 | 0.330 | 0.723 | 0.490 | 0.008 |
| contextual_hybrid | 0.761 | 0.530 | 0.848 | 0.700 | 0.075 |
| contextual_reranked | 0.814 | 0.620 | 0.874 | 0.740 | 0.189 |

Paired bootstrap intervals use 10,000 draws, seed 42; multiple comparisons are exploratory.
Raw rankings, scores, source sentences and missing evidence remain in the original run. All failures are retained.
Corpus availability and tokenizer truncation are recorded in JSON artifacts. EM/F1 requires the separate reader run.
