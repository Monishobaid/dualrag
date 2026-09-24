# Critic decisions versus annotated evidence coverage

Post-run diagnostics only. Annotated support completeness is not a human sufficiency judgment; alternative supporting evidence may exist. Calls from the same example are correlated. These counts do not establish calibrated probabilities or classifier accuracy.

| Run | Critic calls | Accept, complete | Accept, incomplete | Continue, complete | Continue, incomplete | Empty final answers |
|---|---:|---:|---:|---:|---:|---:|
| research14b_50 | 83 | 35 | 5 | 30 | 13 | 3 |
| research14b_confirm100 | 160 | 70 | 10 | 41 | 39 | 7 |
| research7b_50 | 115 | 24 | 1 | 68 | 22 | 2 |
| research7b_bgehybrid50 | 102 | 28 | 0 | 68 | 6 | 3 |
| research7b_confirm100 | 228 | 42 | 5 | 133 | 48 | 11 |

**Accept** means the actual stopping guard accepted the model response: boolean true, valid nonempty citation IDs, and no stated missing information. **Complete** means all annotated supporting sentence addresses were available at that call. A continue-complete case may reflect an overly conservative critic, confusing context, or annotation limitations; it is not automatically a reasoning error.

Every decision and missing-fact query is exported in `research_results/2026-09-24/critic_diagnostics.json`. Full raw outputs and source sentences remain in each run’s `audit.jsonl`.
