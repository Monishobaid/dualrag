#!/usr/bin/env python3
"""Matched answer analysis with separate reader/critic costs and full raw logs.

Gold-support coverage is an annotation proxy. Critic agreement with that proxy
does not establish semantic sufficiency accuracy; no tuning follows this report.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize(rows):
    """Separate answer quality from the optional critic's extra work.

    For example, a correct answer with missing annotated support stays correct;
    we cannot label it hallucinated from support recall alone.
    """
    calls = [c for row in rows for c in row['calls']]
    result = {key: mean(r[key] for r in rows) for key in ['em', 'f1', 'latency_seconds', 'generated_tokens']}
    result['num_examples'] = len(rows)
    result['roles'] = {}
    for role in ['reader', 'sufficiency_critic']:
        selected = [c for c in calls if c['role'] == role]
        result['roles'][role] = {
            'num_calls': len(selected),
            'mean_generated_tokens': mean(c['generated_tokens'] for c in selected),
            'mean_latency_seconds': mean(c['latency_seconds'] for c in selected),
            'parse_failure_rate': mean(bool(c['parse_failed']) for c in selected),
            'non_strict_json_rate': mean(not c['strict_json'] for c in selected),
            'token_limit_rate': mean(c.get('hit_token_limit', False) for c in selected),
        }
    result['mean_reader_plus_retrieval_seconds'] = mean(
        r['retrieval_seconds'] + sum(c['latency_seconds'] for c in r['calls'] if c['role'] == 'reader') for r in rows)
    result['runtime_error_ids'] = [r['id'] for r in rows if r.get('error')]
    result['wrong_answer_ids'] = [r['id'] for r in rows if not r['em']]
    result['mean_hops'] = mean(r['num_hops'] for r in rows)
    result['stop_reasons'] = {s: sum(r['stop_reason'] == s for r in rows) for s in sorted({r['stop_reason'] for r in rows})}
    result['by_annotation_coverage'] = {}
    for covered in [True, False]:
        subset = [r for r in rows if bool(r['retrieval_metrics']['complete_support']) == covered]
        result['by_annotation_coverage'][str(covered)] = {'n': len(subset), 'em': mean(r['em'] for r in subset), 'f1': mean(r['f1'] for r in subset)}
    judged = [r for r in rows if 'predicted_sufficient' in r]
    result['critic_vs_annotation_proxy'] = {
        'note': 'Coverage is NOT a semantic sufficiency label. Counts include parser-backed decisions; inspect parse failures separately.',
        'num_judged': len(judged),
        'counts': {f'critic_{pred}_coverage_{covered}': sum(bool(r['predicted_sufficient']) == pred and bool(r['retrieval_metrics']['complete_support']) == covered for r in judged)
            for pred in [True, False] for covered in [True, False]},
        'em_when_predicted_sufficient': mean(r['em'] for r in judged if r['predicted_sufficient']),
        'em_when_predicted_insufficient': mean(r['em'] for r in judged if not r['predicted_sufficient']),
    }
    return result


def paired(rows, baseline, samples):
    """Resample the same example pairs, retaining both rescued and harmed IDs."""
    if [r['id'] for r in rows] != [r['id'] for r in baseline]:
        raise ValueError('Different IDs/order')
    result = {}
    for metric in ['em', 'f1']:
        delta = np.asarray([r[metric] - b[metric] for r, b in zip(rows, baseline)])
        result[metric] = {'mean_delta': float(delta.mean()), 'ci95': np.quantile(delta[samples].mean(axis=1), [.025, .975]).tolist(),
            'improved_ids': [r['id'] for r, d in zip(rows, delta) if d > 0],
            'regressed_ids': [r['id'] for r, d in zip(rows, delta) if d < 0]}
    result['evidence_gain_without_em_gain'] = [r['id'] for r, b in zip(rows, baseline)
        if r['retrieval_metrics']['sentence_recall'] > b['retrieval_metrics']['sentence_recall'] and r['em'] <= b['em']]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--base', default='plain_bm25')
    a = p.parse_args()
    root, out = Path(a.run), Path(a.output)
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol['status'] != 'complete':
        raise ValueError('Reader run is incomplete')
    arms = {}
    for arm in protocol['arms']:
        rows = [json.loads(line) for line in (root/(arm+'.jsonl')).read_text().splitlines()]
        if [r['id'] for r in rows] != protocol['ids']:
            raise ValueError('Mismatched IDs/order for ' + arm)
        arms[arm] = rows
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    samples = np.random.default_rng(42).integers(0, len(protocol['ids']), (10000, len(protocol['ids'])))
    summary = {arm: summarize(rows) for arm, rows in arms.items()}
    comparisons = {arm: paired(rows, arms[a.base], samples) for arm, rows in arms.items()}
    compact = {arm: [{k: v for k, v in row.items() if k not in ['calls', 'sources']} for row in rows] for arm, rows in arms.items()}
    # Compressed JSONL preserves exact prompts, outputs and source text for audit.
    manifest = {}
    for arm in arms:
        target = out/(arm+'.jsonl.gz')
        with (root/(arm+'.jsonl')).open('rb') as source, target.open('wb') as dest:
            with gzip.GzipFile(filename='', mode='wb', fileobj=dest, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)
        manifest[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    for name, obj in [('protocol', protocol), ('summary', summary), ('paired', comparisons), ('per_example', compact), ('raw_trace_sha256', manifest)]:
        (out/(name+'.json')).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    lines = ['# Matched corpus reader results', '', f"{len(protocol['ids'])} frozen examples; {protocol['model']}; k={protocol['k']}; one reader call per example.", '',
        '| Arm | EM | F1 | Reader+retrieval s | Critic s | Reader tokens | Critic tokens | Reader parse failure |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, m in summary.items():
        reader, critic = m['roles']['reader'], m['roles']['sufficiency_critic']
        lines.append(f"| {arm} | {m['em']:.3f} | {m['f1']:.3f} | {m['mean_reader_plus_retrieval_seconds']:.2f} | {(critic['mean_latency_seconds'] or 0):.2f} | {(reader['mean_generated_tokens'] or 0):.1f} | {(critic['mean_generated_tokens'] or 0):.1f} | {(reader['parse_failure_rate'] or 0):.3f} |")
    lines += ['', 'Paired intervals: 10,000 example bootstrap draws, seed 42. Multiple comparisons are exploratory.',
        'Critic calls do not change the answers. Coverage/critic agreement is not semantic sufficiency accuracy.',
        'Token counts include generated JSON and EOS as counted by the model wrapper. Raw traces retain exact calls.',
        'Reported latency is observational; concurrent indexing and fixed arm order can confound comparisons.',
        'This cohort has now been examined. Further method development requires a fresh frozen reporting cohort.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
