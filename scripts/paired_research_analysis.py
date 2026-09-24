#!/usr/bin/env python3
"""Paired answer comparison with bootstrap intervals; refuses mismatched IDs.

The interval quantifies sample uncertainty, not multiple-comparison correction,
training contamination, or post-selection bias. Old50 remains exploratory.
"""
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_experiment import score


def read_run(path):
    rows=[json.loads(x) for x in (Path(path)/'traces.jsonl').read_text().splitlines() if x.strip()]
    result={r['id']:r for r in rows}
    if len(result)!=len(rows): raise ValueError('Duplicate trace IDs')
    return result


def compare(a,b):
    if set(a)!=set(b): raise ValueError('Mismatched cohorts; cannot report a paired comparison')
    rows=[]
    for key in a:
        x,y=a[key],b[key]
        if x['gold_answer']!=y['gold_answer']: raise ValueError('Gold answer mismatch')
        sx=score(x['final_answer'],x['gold_answer']);sy=score(y['final_answer'],y['gold_answer'])
        rows.append({'id':key,'question':x['question'],'gold':x['gold_answer'],
            'control':x['final_answer'],'candidate':y['final_answer'],
            'em_delta':sy['em']-sx['em'],'f1_delta':sy['f1']-sx['f1'],
            'category':('gain' if sy['em']>sx['em'] else 'loss' if sy['em']<sx['em'] else
                        'both_correct' if sx['em'] else 'both_wrong')})
    rng=random.Random(42); n=len(rows); intervals={}
    for field in ['em_delta','f1_delta']:
        samples=sorted(sum(rows[rng.randrange(n)][field] for _ in range(n))/n for _ in range(10000))
        intervals[field]={'mean':sum(r[field] for r in rows)/n,'paired_bootstrap_95_ci':[samples[250],samples[9749]]}
    return {'n':n,'gains':sum(r['category']=='gain' for r in rows),
            'losses':sum(r['category']=='loss' for r in rows), 'effects':intervals,'examples':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('control');p.add_argument('candidate');p.add_argument('--output',required=True)
    args=p.parse_args(); report=compare(read_run(args.control),read_run(args.candidate))
    report.update(control=args.control,candidate=args.candidate)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='examples'},indent=2))

if __name__=='__main__':main()
