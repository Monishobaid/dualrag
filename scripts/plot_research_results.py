#!/usr/bin/env python3
"""Plot completed confirmation arms; requires matplotlib only for reporting.

These are cohort means, not confidence intervals or repeated timing estimates.
Use the paired bootstrap table in RESULTS for uncertainty about differences.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('research_results/2026-09-24')
ARMS = [
    ('7B · top 5', 'research7b_confirm100/top5'),
    ('7B · top 9', 'research7b_budget_confirm100/top9'),
    ('7B · all 10, ranked', 'research7b_budget_confirm100/ranked_all'),
    ('7B · all 10, dataset order', 'research7b_confirm100/all_context'),
    ('7B · adaptive critic', 'research7b_confirm100/adaptive'),
    ('14B · top 5', 'research14b_confirm100/top5'),
    ('14B · top 9*', 'research14b_budget_confirm100/top9'),
    ('14B · all 10, ranked*', 'research14b_budget_confirm100/ranked_all'),
    ('14B · all 10, dataset order', 'research14b_confirm100/all_context'),
    ('14B · adaptive critic', 'research14b_confirm100/adaptive'),
]


def main():
    # Refuse to render a misleading partial comparison while an arm is running.
    rows = [json.loads((ROOT / path / 'metrics.json').read_text()) for _, path in ARMS]
    assert all(r['num_examples'] == 100 for r in rows)
    cohorts = [set(json.loads((ROOT / path / 'predictions.json').read_text())) for _, path in ARMS]
    assert all(ids == cohorts[0] for ids in cohorts)
    labels = [label for label, _ in ARMS]
    colors = ['#CE6B28' if 'adaptive' in path else '#267B90' if label.startswith('14B')
              else '#4664A6' for label, path in ARMS]
    fig, axes = plt.subplots(1, 4, figsize=(15, 6.7), sharey=True)
    columns = [('em', 'Exact match (%)', 100), ('f1', 'Token F1 (%)', 100),
               ('mean_generated_tokens', 'Output tokens / example', 1),
               ('mean_latency_seconds', 'Seconds / example', 1)]
    for ax, (key, title, scale) in zip(axes, columns):
        values = [row[key] * scale for row in rows]
        bars = ax.barh(range(len(rows)), values, color=colors, height=.64)
        ax.bar_label(bars, labels=[f'{x:.1f}' for x in values], padding=4, fontsize=9)
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_xlim(0, 100 if scale == 100 else max(values) * 1.22)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle('Frozen confirmation cohort: 100 HotpotQA distractor examples', fontsize=15)
    fig.text(.02, .025, 'Qwen2.5-Instruct · MiniLM retrieval · seed 42 · same reader prompt · '
             'orange = adaptive critic\nTokens include critic calls; latency excludes model load. '
             'Means shown; paired uncertainty is reported in RESULTS_2026-09-24.md.\n'
             '*14B static budget controls were added after observing the adaptive result; these are post-hoc checks.', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .93))
    out = Path('docs/figures'); out.mkdir(exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'confirmation100_2026-09-24.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
