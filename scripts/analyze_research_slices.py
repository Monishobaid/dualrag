#!/usr/bin/env python3
"""Descriptive bridge/comparison breakdown; no model or threshold selection.

Small slices are not independently powered tests. Keep the overall paired
comparison primary, and use these numbers only to locate future audit work.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_experiment import score

ROOT = Path('research_results/2026-09-24')


def main():
    data = {r['_id']: r for r in json.loads(Path('data/hotpotqa/hotpot_dev_distractor_v1.json').read_text())}
    manifest = json.loads((ROOT / 'research_confirm100_ids.json').read_text())['ids']
    expected = set(manifest)
    types = Counter(data[key]['type'] for key in manifest)
    levels = Counter(data[key]['level'] for key in manifest)
    lines = ['# Confirmation100 dataset slices', '',
             f'Frozen sample question types: {dict(types)}. Difficulty labels: {dict(levels)}.', '',
             'HotpotQA **bridge** questions require linking facts about related entities. '
             '**Comparison** questions compare properties of two entities. Each question '
             'provides ten candidate paragraphs; this is the distractor setting. These '
             'are descriptive, small subgroup results, not corrected hypothesis tests. '
             'No prompts or thresholds were changed using them.', '',
             '| Run / arm | Question type | N | EM | F1 | Empty answers |',
             '|---|---|---:|---:|---:|---:|']
    report = {}
    for path in sorted(ROOT.glob('*confirm100/*/audit.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert {r['id'] for r in rows} == expected and len(rows) == 100
        groups = defaultdict(list)
        for row in rows:
            groups[data[row['id']]['type']].append(row)
        name = str(path.parent.relative_to(ROOT))
        report[name] = {}
        for kind, group in sorted(groups.items()):
            scores = [score(r['final_answer'], r['gold_answer']) for r in group]
            em = sum(s['em'] for s in scores) / len(group)
            f1 = sum(s['f1'] for s in scores) / len(group)
            empty = sum(not r['final_answer'].strip() for r in group)
            report[name][kind] = {'ids': [r['id'] for r in group], 'n': len(group),
                                  'em': em, 'f1': f1, 'empty_answers': empty}
            lines.append(f'| {name} | {kind} | {len(group)} | {em:.1%} | {f1:.2%} | {empty} |')
    Path('docs/DATASET_SLICES_2026-09-24.md').write_text('\n'.join(lines) + '\n')
    (ROOT / 'dataset_slices.json').write_text(json.dumps(report, indent=2))
    print('Audited question-type slices for', len(report), 'completed arms')


if __name__ == '__main__':
    main()
