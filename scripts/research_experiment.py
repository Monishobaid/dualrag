#!/usr/bin/env python3
"""Matched, label-isolated reader/retrieval experiments; see research protocol.

A single process reuses a frozen generator, avoiding model reload noise. Each arm
is independently seeded and has its own predictions, complete calls and metrics.
Failed examples stay in the denominator. Completed output is never overwritten.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.hotpot_loader import HotpotQALoader
from src.rag.sufficiency import (READER_SYSTEM, CRITIC_SYSTEM, evidence_view,
    parse_object, valid_citations, sufficient_decision, short_answer, citation_ids)
from src.utils.text import compute_exact_match, compute_f1, normalize_text


def score(pred, gold):
    """HotpotQA answer scoring, including its special yes/no/noanswer mismatch rule."""
    p, g = normalize_text(pred), normalize_text(gold)
    f1 = compute_f1(pred, gold)
    if p != g and ({p, g} & {'yes', 'no', 'noanswer'}):
        f1 = 0.0
    return {'em': compute_exact_match(pred, gold), 'f1': f1}


def load_ids(path):
    obj = json.loads(Path(path).read_text())
    ids = obj.get('ids') if isinstance(obj, dict) else obj
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError('Manifest must contain unique, nonempty IDs')
    return ids


def call_model(llm, question, paragraphs, system, role, budget):
    """Log the actual input and raw output, including truncation and format health."""
    context, sources = evidence_view(paragraphs)
    prompt = f'EVIDENCE:\n{context}\n\nQUESTION: {question}'
    start = time.perf_counter()
    text, tokens = llm.generate_with_token_count(prompt, system_message=system,
        max_new_tokens=budget, temperature=0, top_p=1)
    obj, strict = parse_object(text)
    call = {'role': role, 'prompt': prompt, 'system': system, 'raw_output': text,
        'parsed': obj, 'strict_json': strict, 'parse_failed': obj is None,
        'citation_ids_valid': valid_citations(obj, sources), 'generated_tokens': tokens,
        'latency_seconds': time.perf_counter()-start, **llm.last_generation}
    return obj, sources, call


def run_reader(question, paragraphs, arm, ranker, llm, budget=192):
    """Compare a direct reader with a critic that searches for missing relations.

    IRCoT motivates evidence-dependent queries; Google motivates judging complete
    context. All-context is essential here: Hotpot distractor pools only have
    about ten paragraphs, so this is not an open-domain retrieval benchmark.
    """
    calls, hops, seen = [], [], set()
    if arm in {'all_context', 'oracle'}:
        evidence = list(paragraphs)
        hops.append({'query': question, 'titles': [p['title'] for p in evidence]})
        reason = 'oracle_evidence' if arm == 'oracle' else 'all_context'
    else:
        evidence, query = [], question
        for hop in range(3 if arm == 'adaptive' else 1):
            ranked = ranker.rank(query, paragraphs, top_k=5 if hop == 0 else 2)
            fresh = [x['paragraph'] for x in ranked if x['paragraph']['_pid'] not in seen]
            # Rank the remaining pool when a query repeats earlier hits. This
            # bounds the loop without silently resetting the evidence history.
            if hop and len(fresh) < 2:
                remaining = [p for p in paragraphs if p['_pid'] not in seen]
                fresh = [x['paragraph'] for x in ranker.rank(query, remaining, top_k=2)]
            evidence.extend(fresh)
            seen.update(p['_pid'] for p in fresh)
            hops.append({'query': query, 'titles': [p['title'] for p in fresh]})
            reason = 'fixed_top5'
            if arm != 'adaptive':
                break
            obj, sources, call = call_model(llm, question, evidence, CRITIC_SYSTEM, 'critic', budget)
            calls.append(call)
            if sufficient_decision(obj, sources):
                reason = 'critic_sufficient'
                break
            if len(seen) == len(paragraphs) or not fresh:
                reason = 'pool_exhausted'
                break
            reason = 'hop_budget'
            rewrite = obj.get('query', '') if obj else ''
            query = question + (' ' + rewrite if isinstance(rewrite, str) else '')
    obj, sources, call = call_model(llm, question, evidence, READER_SYSTEM, 'reader', budget)
    calls.append(call)
    return {'final_answer': short_answer(obj), 'calls': calls, 'hops': hops,
            'stop_reason': reason, 'visible_titles': [p['title'] for p in evidence],
            'sources': sources, 'num_hops': len(hops)}


def summarize(rows):
    """Keep failures in EM/F1; expose generation cost instead of assumed savings."""
    n = len(rows)
    good = [r for r in rows if not r.get('error')]
    calls = [c for r in good for c in r['calls']]
    avg = lambda field: sum(r.get(field, 0) for r in rows) / n
    return {'num_examples': n, 'num_successful': len(good), 'em': avg('em'), 'f1': avg('f1'),
        'mean_hops': avg('num_hops'), 'mean_generated_tokens': avg('generated_tokens'),
        'mean_latency_seconds': avg('latency_seconds'), 'first_context_supporting_title_recall': avg('title_recall_at_5'),
        'supporting_title_recall_at_5': avg('title_recall_at_5') if rows[0]['arm'] in {'top5','adaptive'} else None,
        'visible_supporting_sentence_recall': avg('visible_sentence_recall'),
        'citation_supporting_fact_recall': avg('citation_fact_recall'),
        'parse_failure_rate': sum(c['parse_failed'] for c in calls)/max(1,len(calls)),
        'non_strict_json_rate': sum(not c['strict_json'] for c in calls)/max(1,len(calls)),
        'output_budget_hit_rate': sum(c['hit_token_limit'] for c in calls)/max(1,len(calls)),
        'num_model_calls': len(calls), 'failed_examples': [r for r in rows if r.get('error')],
        'stop_reasons': dict(Counter(r.get('stop_reason','error') for r in rows)),
        'latency_note': 'per-example warm process; excludes model load, includes retrieval and every critic call'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='data/hotpotqa/hotpot_dev_distractor_v1.json')
    p.add_argument('--ids', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--embedder', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--retrieval', choices=['dense','bm25','hybrid'], default='dense')
    p.add_argument('--title-prefix', action='store_true')
    p.add_argument('--reranker', default=None)
    p.add_argument('--arms', nargs='+', choices=['top5','all_context','oracle','adaptive'],
                   default=['top5','all_context','oracle','adaptive'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-new-tokens', type=int, default=192)
    args = p.parse_args()
    import numpy as np
    import torch
    from src.llm.qwen import QwenModel
    from src.retrieval.embedder import Embedder
    from src.retrieval.ranker import ParagraphRanker
    from src.retrieval.index import EmbeddingCache
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise FileExistsError('Use a new output directory; refusing to overwrite research')
    ids = load_ids(args.ids)
    loader = HotpotQALoader(args.dataset, example_ids=ids)
    def seed():
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    seed()
    llm = QwenModel(args.model, use_chat_template=True, max_input_tokens=8192)
    embedder = Embedder(args.embedder)
    cache = EmbeddingCache(namespace=args.embedder+'|'+str(args.title_prefix))
    ranker = ParagraphRanker(embedder, cache, args.retrieval, reranker_model_name=args.reranker)
    protocol = {**vars(args), 'ids': ids, 'id_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        'code_revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'working_diff_sha256': hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest(),
        'model_revision': getattr(llm.model.config,'_commit_hash',None),
        'embedder_revision': getattr(embedder.model[0].auto_model.config,'_commit_hash',None),
        'packages': {x:importlib.metadata.version(x) for x in ['torch','transformers','sentence-transformers','numpy']},
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'reader_system': READER_SYSTEM, 'critic_system': CRITIC_SYSTEM,
        'metric_version':'HotpotQA-special-answers-v1', 'chat_template':True,
        'note':'oracle is evaluation-only; all arms use identical generation settings'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    all_metrics = {}
    for arm in args.arms:
        seed()
        dest=out/arm;dest.mkdir()
        rows=[]
        with (dest/'traces.jsonl').open('w') as f:
            for i, ex in enumerate(loader):
                start=time.perf_counter()
                row={'id':ex.id,'question':ex.question,'gold_answer':ex.answer,'arm':arm}
                try:
                    paragraphs=[dict(x,_pid=j) for j,x in enumerate(ex.candidate_paragraphs)]
                    # Metadata baseline inspired by contextual retrieval. No generated
                    # context, no gold titles and no question-specific text in indexing.
                    if args.title_prefix:
                        paragraphs=[dict(x,text=x['title']+'\n'+x['text']) for x in paragraphs]
                    if arm == 'oracle':
                        gold_titles={x[0] for x in ex.supporting_facts}
                        paragraphs=[x for x in paragraphs if x['title'] in gold_titles]
                    result=run_reader(ex.question,paragraphs,arm,ranker,llm,args.max_new_tokens)
                    row.update(result)
                    row.update(score(row['final_answer'],ex.answer))
                    gold=set(tuple(x) for x in ex.supporting_facts)
                    titles={x[0] for x in gold}
                    row['title_recall_at_5']=len(titles & set(row['hops'][0]['titles']))/max(1,len(titles))
                    visible={(s['title'],s['sentence_index']) for s in row['sources'].values()}
                    row['visible_sentence_recall']=len(gold & visible)/max(1,len(gold))
                    obj=row['calls'][-1]['parsed'] or {}
                    cites=citation_ids(obj)
                    if not isinstance(cites,list): cites=[]
                    cited={ (row['sources'][sid]['title'],row['sources'][sid]['sentence_index'])
                           for sid in cites if isinstance(sid,str) and sid in row['sources']}
                    row['citation_fact_recall']=len(gold & cited)/max(1,len(gold))
                    row['generated_tokens']=sum(c['generated_tokens'] for c in row['calls'])
                except Exception as exc:
                    row.update(error=repr(exc),em=0.0,f1=0.0,final_answer='')
                row['latency_seconds']=time.perf_counter()-start
                rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                print(f'{arm} {i+1}/{len(loader)} EM={row["em"]} F1={row["f1"]:.3f} {row.get("error","")}',flush=True)
        metrics=summarize(rows);all_metrics[arm]=metrics
        (dest/'metrics.json').write_text(json.dumps(metrics,indent=2))
        (dest/'predictions.json').write_text(json.dumps({r['id']:r['final_answer'] for r in rows},indent=2))
        print(arm,json.dumps(metrics),flush=True)
    cache.save()
    (out/'summary.json').write_text(json.dumps(all_metrics,indent=2))

if __name__ == '__main__': main()
