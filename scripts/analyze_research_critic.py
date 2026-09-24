#!/usr/bin/env python3
"""Audit frozen critic decisions against annotated evidence coverage.

Coverage is an imperfect proxy for sufficiency: another sentence can provide the
same fact. No labels or predictions are changed, and no thresholds are fitted.
"""
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.sufficiency import sufficient_decision

ROOT = Path('research_results/2026-09-24')


def main():
    data = json.loads(Path('data/hotpotqa/hotpot_dev_distractor_v1.json').read_text())
    gold = {r['_id']: {tuple(fact) for fact in r['supporting_facts']} for r in data}
    reports = {}
    lines = ['# Critic decisions versus annotated evidence coverage', '',
             'Post-run diagnostics only. Annotated support completeness is not a human '
             'sufficiency judgment; alternative supporting evidence may exist. Calls from '
             'the same example are correlated. These counts do not establish calibrated '
             'probabilities or classifier accuracy.', '',
             '| Run | Critic calls | Accept, complete | Accept, incomplete | Continue, complete | Continue, incomplete | Empty final answers |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for path in sorted(ROOT.glob('*/adaptive/audit.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        # The train smoke is intentionally not a performance evaluation.
        if any(r['id'] not in gold for r in rows):
            continue
        counts = Counter(); decisions = []
        for row in rows:
            seen = set()
            critics = [c for c in row['calls'] if c['role'] == 'critic']
            assert len(critics) == len(row['hops'])
            for hop, (call, retrieval) in enumerate(zip(critics, row['hops']), 1):
                seen.update(retrieval['titles'])
                sources = {sid: source for sid, source in row['sources'].items()
                           if source['title'] in seen}
                visible = {(s['title'], s['sentence_index']) for s in sources.values()}
                complete = gold[row['id']] <= visible
                accepted = sufficient_decision(call['parsed'], sources)
                category = ('accept' if accepted else 'continue') + ('_complete' if complete else '_incomplete')
                counts[category] += 1
                obj = call['parsed'] or {}
                decisions.append({'id': row['id'], 'hop': hop, 'category': category,
                                  'query': obj.get('query'), 'missing_information': obj.get('missing_information'),
                                  'raw_sufficient': obj.get('sufficient')})
        name = str(path.parent.parent.relative_to(ROOT))
        empty = sum(not r['final_answer'].strip() for r in rows)
        counts['critic_calls'] = len(decisions)
        counts['empty_final_answers'] = empty
        reports[name] = {'counts': dict(counts), 'decisions': decisions}
        cells = [counts[k] for k in ['critic_calls', 'accept_complete', 'accept_incomplete',
                                    'continue_complete', 'continue_incomplete', 'empty_final_answers']]
        lines.append('| ' + name + ' | ' + ' | '.join(map(str, cells)) + ' |')
    lines += ['', '**Accept** means the actual stopping guard accepted the model response: '
              'boolean true, valid nonempty citation IDs, and no stated missing information. '
              '**Complete** means all annotated supporting sentence addresses were available '
              'at that call. A continue-complete case may reflect an overly conservative '
              'critic, confusing context, or annotation limitations; it is not automatically '
              'a reasoning error.', '',
              'Every decision and missing-fact query is exported in '
              '`research_results/2026-09-24/critic_diagnostics.json`. Full raw outputs and '
              'source sentences remain in each run’s `audit.jsonl`.', '']
    (ROOT / 'critic_diagnostics.json').write_text(json.dumps(reports, indent=2))
    Path('docs/CRITIC_DIAGNOSTICS_2026-09-24.md').write_text('\n'.join(lines))
    print('Audited', len(reports), 'completed adaptive arms')


if __name__ == '__main__':
    main()
