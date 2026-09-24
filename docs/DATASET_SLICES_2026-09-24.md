# Confirmation100 dataset slices

Frozen sample question types: {'bridge': 85, 'comparison': 15}. Difficulty labels: {'hard': 100}.

HotpotQA **bridge** questions require linking facts about related entities. **Comparison** questions compare properties of two entities. Each question provides ten candidate paragraphs; this is the distractor setting. These are descriptive, small subgroup results, not corrected hypothesis tests. No prompts or thresholds were changed using them.

| Run / arm | Question type | N | EM | F1 | Empty answers |
|---|---|---:|---:|---:|---:|
| research14b_budget_confirm100/ranked_all | bridge | 85 | 57.6% | 71.18% | 9 |
| research14b_budget_confirm100/ranked_all | comparison | 15 | 73.3% | 73.33% | 0 |
| research14b_budget_confirm100/top9 | bridge | 85 | 54.1% | 68.71% | 9 |
| research14b_budget_confirm100/top9 | comparison | 15 | 73.3% | 73.33% | 0 |
| research14b_confirm100/adaptive | bridge | 85 | 55.3% | 68.18% | 7 |
| research14b_confirm100/adaptive | comparison | 15 | 80.0% | 80.00% | 0 |
| research14b_confirm100/all_context | bridge | 85 | 50.6% | 62.97% | 9 |
| research14b_confirm100/all_context | comparison | 15 | 66.7% | 66.67% | 0 |
| research14b_confirm100/top5 | bridge | 85 | 42.4% | 53.04% | 22 |
| research14b_confirm100/top5 | comparison | 15 | 80.0% | 80.00% | 1 |
| research7b_budget_confirm100/ranked_all | bridge | 85 | 38.8% | 55.34% | 13 |
| research7b_budget_confirm100/ranked_all | comparison | 15 | 73.3% | 73.33% | 1 |
| research7b_budget_confirm100/top9 | bridge | 85 | 37.6% | 55.14% | 11 |
| research7b_budget_confirm100/top9 | comparison | 15 | 66.7% | 71.11% | 1 |
| research7b_confirm100/adaptive | bridge | 85 | 40.0% | 57.05% | 10 |
| research7b_confirm100/adaptive | comparison | 15 | 73.3% | 73.33% | 1 |
| research7b_confirm100/all_context | bridge | 85 | 45.9% | 59.00% | 13 |
| research7b_confirm100/all_context | comparison | 15 | 66.7% | 72.78% | 0 |
| research7b_confirm100/top5 | bridge | 85 | 28.2% | 43.21% | 18 |
| research7b_confirm100/top5 | comparison | 15 | 86.7% | 86.67% | 1 |
