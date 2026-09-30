#!/usr/bin/env python3
"""Frozen-reader evaluation of saved corpus rankings, plus sufficiency audit.

Google Sufficient Context motivates an explicit critic decision. Its prediction
is not a gold label and is never used to select or stop retrieval in this test.
Reader uses the same Qwen short-answer prompt for every retrieval arm.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.rag.sufficiency import READER_SYSTEM, CRITIC_SYSTEM, parse_object, short_answer, sufficient_decision
from src.retrieval.corpus import file_hash
from scripts.research_experiment import score


def evidence_prompt(question, chunks):
    sources={}; blocks=[]
    for c in chunks:
        lines=['TITLE: '+c['title']]
        for i,sentence in enumerate(c['sentences'],c['start']):
            sid=f"c{c['id']}s{i}"
            sources[sid]={'title':c['title'],'sentence_index':i,'text':sentence}
            lines.append(f'[{sid}] {sentence.strip()}')
        blocks.append('\n'.join(lines))
    return 'EVIDENCE:\n'+'\n\n'.join(blocks)+'\n\nQUESTION: '+question,sources


def generate(llm,prompt,system,role,budget):
    started=time.perf_counter()
    raw,tokens=llm.generate_with_token_count(prompt,system_message=system,max_new_tokens=budget,temperature=0,top_p=1)
    obj,strict=parse_object(raw)
    return {'role':role,'prompt':prompt,'system':system,'raw_output':raw,'parsed':obj,
        'strict_json':strict,'parse_failed':obj is None,'generated_tokens':tokens,
        'latency_seconds':time.perf_counter()-started,**llm.last_generation}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retrieval-run',required=True)
    p.add_argument('--dataset',default='data/hotpotqa/hotpot_dev_distractor_v1.json')
    p.add_argument('--output',required=True)
    p.add_argument('--arms',nargs='+',required=True)
    p.add_argument('--k',type=int,default=5)
    p.add_argument('--model',default='Qwen/Qwen2.5-14B-Instruct')
    p.add_argument('--expected-model-revision',default=None)
    p.add_argument('--critic',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=192)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    import numpy as np
    import torch
    from src.llm.qwen import QwenModel
    root=Path(a.retrieval_run);out=Path(a.output)
    prior=json.loads((root/'protocol.json').read_text())
    if prior['status']!='complete': raise ValueError('Retrieval not complete')
    if a.k not in prior['ks']: raise ValueError('k was not measured in retrieval run')
    if file_hash(a.dataset)!=prior['dataset_sha256']:raise ValueError('Dataset changed')
    dataset={x['_id']:x for x in json.loads(Path(a.dataset).read_text())}
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError(out)
    def seed():
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    seed()
    llm=QwenModel(a.model,use_chat_template=True,max_input_tokens=8192)
    actual_revision = getattr(llm.model.config, '_commit_hash', None)
    if a.expected_model_revision and actual_revision != a.expected_model_revision:
        raise ValueError('Reader checkpoint changed: ' + str(actual_revision))
    protocol={**vars(a),'retrieval_protocol_sha256':file_hash(root/'protocol.json'),'ids':prior['ids'],
        'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'model_revision':getattr(llm.model.config,'_commit_hash',None),'reader_system':READER_SYSTEM,
        'critic_system':CRITIC_SYSTEM,'status':'running','hardware':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summary={}
    for arm in a.arms:
        seed()
        if arm not in prior['arms_completed']: raise ValueError('Unknown arm '+arm)
        retrieved=[json.loads(line) for line in (root/(arm+'.jsonl')).read_text().splitlines()]
        if [x['id'] for x in retrieved]!=prior['ids']:raise ValueError('IDs changed')
        rows=[]
        with (out/(arm+'.jsonl')).open('w') as f:
            for r in retrieved:
                row={'id':r['id'],'question':r['question'],'arm':arm,'retrieval_metrics':r['metrics'][str(a.k)],
                     'retrieval_seconds':r['query_seconds'],'calls':[],'num_hops':1,'stop_reason':'fixed_evidence_budget'}
                started=time.perf_counter()
                try:
                    prompt,sources=evidence_prompt(r['question'],r['chunks'][:a.k])
                    row['sources']=sources
                    call=generate(llm,prompt,READER_SYSTEM,'reader',a.max_new_tokens)
                    row['calls'].append(call)
                    row['final_answer']=short_answer(call['parsed'])
                    if a.critic:
                        critic=generate(llm,prompt,CRITIC_SYSTEM,'sufficiency_critic',a.max_new_tokens)
                        row['calls'].append(critic)
                        row['predicted_sufficient']=sufficient_decision(critic['parsed'],sources)
                except Exception as exc:
                    row.update(error=repr(exc),final_answer='')
                # Gold is consulted only after all model calls have finished.
                row['gold_answer']=dataset[r['id']]['answer']
                row.update(score(row['final_answer'],row['gold_answer']))
                row['latency_seconds']=time.perf_counter()-started+row['retrieval_seconds']
                row['generated_tokens']=sum(c['generated_tokens'] for c in row['calls'])
                rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                print(arm,len(rows),'/',len(retrieved),row['em'],row.get('error',''),flush=True)
        calls=[c for r in rows for c in r['calls']]
        summary[arm]={m:sum(r[m] for r in rows)/len(rows) for m in ['em','f1','latency_seconds','generated_tokens']}
        summary[arm].update(num_examples=len(rows),parse_failure_rate=sum(c['parse_failed'] for c in calls)/max(1,len(calls)),
            failed_ids=[r['id'] for r in rows if r.get('error')],wrong_answer_ids=[r['id'] for r in rows if not r['em']],
            predicted_sufficient_rate=sum(r.get('predicted_sufficient',False) for r in rows)/len(rows) if a.critic else None,
            num_model_calls=len(calls),mean_hops=1,stop_reason='fixed_evidence_budget')
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    protocol['status']='complete';(out/'protocol.json').write_text(json.dumps(protocol,indent=2))


if __name__=='__main__':main()
