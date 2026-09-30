#!/usr/bin/env python3
"""Paired example bootstrap, complete rankings, and compact failure summaries."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--base',default='plain_bm25')
    a=p.parse_args()
    root=Path(a.run); out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(out)
    protocol=json.loads((root/'protocol.json').read_text())
    if protocol['status']!='complete': raise ValueError('Incomplete run')
    summary=json.loads((root/'summary.json').read_text())
    ids=protocol['ids']; arms={}
    for arm in summary:
        rows=[json.loads(line) for line in (root/(arm+'.jsonl')).read_text().splitlines()]
        if [r['id'] for r in rows]!=ids: raise ValueError('Mismatched IDs')
        arms[arm]=rows
    rng=np.random.default_rng(42)
    samples=rng.integers(0,len(ids),size=(10000,len(ids)))
    paired={}; compact={}
    for arm,rows in arms.items():
        paired[arm]={}
        for k in protocol['ks']:
            key=str(k); paired[arm][key]={}
            for metric in ['sentence_recall','complete_support']:
                delta=np.array([float(r['metrics'][key][metric])-float(b['metrics'][key][metric])
                    for r,b in zip(rows,arms[a.base])])
                lower,upper=np.quantile(delta[samples].mean(axis=1),[.025,.975])
                paired[arm][key][metric]={'mean_delta':float(delta.mean()),'ci95':[float(lower),float(upper)],
                    'improved_ids':[ids[i] for i,d in enumerate(delta) if d>0],
                    'regressed_ids':[ids[i] for i,d in enumerate(delta) if d<0]}
        compact[arm]=[{k:r[k] for k in ['id','question','metrics','query_seconds','reranker_overflow_pairs']}
            | {'selected_chunks':[{'id':c['id'],'title':c['title'],'start':c['start'],'sentences':c['sentences']} for c in r['chunks']]}
            for r in rows]
    for name,obj in [('protocol',protocol),('summary',summary),('paired',paired),('per_example',compact)]:
        (out/(name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
    text=['# Evidence-selection results','',f"Corpus: {protocol['corpus']['scope']}; {protocol['corpus']['documents']:,} documents, {protocol['corpus']['chunks']:,} chunks.",
        '',f"{len(ids)} frozen questions. Recall is original sentence evidence coverage, not answer accuracy or semantic sufficiency.",
        '', '| Arm | Sentence R@5 | Complete@5 | Sentence R@10 | Complete@10 | Query seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,m in summary.items():
        text.append(f"| {arm} | {m['5']['sentence_recall']:.3f} | {m['5']['complete_support']:.3f} | {m['10']['sentence_recall']:.3f} | {m['10']['complete_support']:.3f} | {m['mean_query_seconds']:.3f} |")
    text += ['', 'Paired bootstrap intervals use 10,000 draws, seed 42; multiple comparisons are exploratory.',
        'Raw rankings, scores, source sentences and missing evidence remain in the original run. All failures are retained.',
        'Corpus availability and tokenizer truncation are recorded in JSON artifacts. EM/F1 requires the separate reader run.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text))


if __name__=='__main__': main()
