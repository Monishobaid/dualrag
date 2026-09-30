#!/usr/bin/env python3
"""Corpus-wide BM25/dense/RRF/reranker experiment, with frozen identical IDs.

Independent of answer-verifier correctness. All evidence labels are evaluation
only, after ranking; no gold paragraph is injected into any candidate list.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.corpus import Corpus, file_hash, digest, fuse, retrieval_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', required=True)
    p.add_argument('--dataset', default='data/hotpotqa/hotpot_dev_distractor_v1.json')
    p.add_argument('--ids', default='research_results/2026-09-25/verification_ids_100_seed20260925.json')
    p.add_argument('--output', required=True)
    p.add_argument('--arms', nargs='+', choices=['bm25','dense','hybrid','reranked'], default=['bm25','dense','hybrid','reranked'])
    p.add_argument('--variants', nargs='+', choices=['plain','contextual'], default=['plain','contextual'])
    p.add_argument('--embedder', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--embedder-revision', default='main')
    p.add_argument('--reranker', default='cross-encoder/ms-marco-MiniLM-L6-v2')
    p.add_argument('--reranker-revision', default='main')
    p.add_argument('--device', default='cpu')
    p.add_argument('--cache', default='runs/corpus-cache')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--shortlist', type=int, default=100)
    p.add_argument('--ks', nargs='+', type=int, default=[5,10,20])
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if min(a.ks) < 1 or a.shortlist < max(a.ks):
        p.error('shortlist must cover positive ks')
    if len(set(a.arms)) != len(a.arms) or len(set(a.variants)) != len(a.variants):
        p.error('duplicate arms/variants')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite ' + str(out))
    raw_ids = json.loads(Path(a.ids).read_text())
    ids = raw_ids['ids'] if isinstance(raw_ids, dict) else raw_ids
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Invalid frozen manifest')
    dataset = {x['_id']: x for x in json.loads(Path(a.dataset).read_text())}
    examples = [dataset[x] for x in ids]
    corpus = Corpus(a.corpus)
    corpus.db.execute('PRAGMA cache_size=-64000')
    chash = file_hash(a.corpus)
    random.seed(a.seed)
    protocol = {**vars(a), 'ids': ids, 'id_sha256': digest(ids), 'ids_file_sha256': file_hash(a.ids),
        'dataset_sha256': file_hash(a.dataset), 'corpus_sha256': chash, 'corpus': corpus.metadata,
        'code_revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'working_diff_sha256': digest(subprocess.check_output(['git','diff','HEAD'],text=True)),
        'untracked_source_hashes': {str(f): file_hash(f) for f in [Path(__file__), Path('src/retrieval/corpus.py'), Path('src/retrieval/dense_corpus.py')]},
        'rrf_constant': 60, 'bm25': 'SQLite FTS5 defaults k1=1.2 b=0.75, unicode61 tokenizer',
        'sufficiency_note': 'complete_support is annotation coverage, not semantic sufficiency',
        'status': 'initializing', 'arms_completed': []}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    model = reranker = None
    needs_dense = any(x != 'bm25' for x in a.arms)
    if needs_dense:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from huggingface_hub import snapshot_download
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        torch.set_num_threads(4)
        # Resolve to a local immutable snapshot before loading; no remote code.
        model_path = snapshot_download(a.embedder, revision=a.embedder_revision,
            allow_patterns=['*.json','*.txt','*.safetensors','pytorch_model.bin','1_Pooling/*','2_Normalize/*'],
            ignore_patterns=['onnx/*','openvino/*'])
        # ST 2.3 expects a directory even for stateless Normalize modules.
        # Hub archives cannot represent empty folders; no weights are changed.
        for module in json.loads((Path(model_path) / 'modules.json').read_text()):
            if module['type'] == 'sentence_transformers.models.Normalize':
                target = (Path(model_path) / module['path']).resolve()
                if Path(model_path).resolve() not in target.parents:
                    raise ValueError('Invalid module path')
                target.mkdir(exist_ok=True)
        model = SentenceTransformer(model_path, device=a.device)
        protocol['embedder_resolved_revision'] = Path(model_path).name
        protocol['packages'] = {x:importlib.metadata.version(x) for x in ['torch','numpy','sentence-transformers','transformers']}
        if 'reranked' in a.arms:
            rerank_path = snapshot_download(a.reranker, revision=a.reranker_revision,
                allow_patterns=['*.json','*.txt','*.safetensors','pytorch_model.bin'],
                ignore_patterns=['onnx/*','openvino/*'])
            reranker = CrossEncoder(rerank_path, device=a.device, max_length=512)
            protocol['reranker_resolved_revision'] = Path(rerank_path).name
    protocol['status'] = 'running'
    all_summary = {}
    for variant in a.variants:
        contextual = variant == 'contextual'
        dense = None
        if model is not None:
            from src.retrieval.dense_corpus import DenseCorpus
            dense = DenseCorpus(corpus, model, chash, protocol['embedder_resolved_revision'], a.cache,
                                contextual=contextual, batch_size=a.batch_size)
            protocol[variant + '_embedding_audit'] = dense.metadata
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        rows = {arm: [] for arm in a.arms}
        files = {arm: (out/(variant+'_'+arm+'.jsonl')).open('w') for arm in a.arms}
        for count, ex in enumerate(examples, 1):
            started = time.perf_counter()
            lexical = corpus.lexical(ex['question'], a.shortlist, contextual)
            lexical_time = time.perf_counter() - started
            vector_results, dense_time = [], 0
            if dense:
                started = time.perf_counter()
                q = ex['question']
                if a.embedder.startswith('BAAI/bge-') and a.embedder.endswith('-en-v1.5'):
                    q = 'Represent this sentence for searching relevant passages: ' + q
                vector = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
                vector_results = dense.search_vector(vector, a.shortlist)
                dense_time = time.perf_counter() - started
            t = time.perf_counter()
            hybrid = fuse([lexical, vector_results], a.shortlist)
            fusion_time = time.perf_counter() - t
            rankings = {'bm25': lexical, 'dense': vector_results, 'hybrid': hybrid}
            times = {'bm25': lexical_time, 'dense': dense_time, 'hybrid': lexical_time+dense_time+fusion_time}
            rerank_overflow = 0
            if reranker:
                t = time.perf_counter()
                chunks = [corpus.get(cid) for cid, _ in hybrid]
                pairs = [(ex['question'], c[variant]) for c in chunks]
                lengths = [len(x) for x in reranker.tokenizer([x[0] for x in pairs], [x[1] for x in pairs], truncation=False)['input_ids']]
                rerank_overflow = sum(x > 512 for x in lengths)
                scores = reranker.predict(pairs, batch_size=a.batch_size, show_progress_bar=False)
                rankings['reranked'] = sorted([(c['id'],float(s)) for c,s in zip(chunks,scores)], key=lambda x:(-x[1],x[0]))
                times['reranked'] = times['hybrid'] + time.perf_counter() - t
            for arm in a.arms:
                ranking = rankings[arm]
                selected = [corpus.get(cid) for cid, _ in ranking[:max(a.ks)]]
                row = {'id': ex['_id'], 'question': ex['question'], 'arm': variant+'_'+arm,
                    'rankings': [{'chunk_id':cid,'score':s} for cid,s in ranking], 'chunks':selected,
                    'query_seconds': times[arm], 'reranker_overflow_pairs':rerank_overflow if arm=='reranked' else 0,
                    'metrics': {str(k):retrieval_metrics(ex,selected[:k],corpus) for k in a.ks}}
                rows[arm].append(row)
                files[arm].write(json.dumps(row,ensure_ascii=False)+'\n'); files[arm].flush()
            print(variant,count,'/',len(examples),flush=True)
        for arm in a.arms:
            files[arm].close()
            run = rows[arm]
            metrics = {str(k): {m:sum(r['metrics'][str(k)][m] for r in run)/len(run)
                for m in ['title_recall','sentence_recall','complete_support','corpus_support_availability']} for k in a.ks}
            metrics.update(num_examples=len(run), mean_query_seconds=sum(r['query_seconds'] for r in run)/len(run),
                           reranker_overflow_pairs=sum(r['reranker_overflow_pairs'] for r in run))
            all_summary[variant+'_'+arm] = metrics
            protocol['arms_completed'].append(variant+'_'+arm)
        (out/'summary.json').write_text(json.dumps(all_summary,indent=2))
    protocol['status'] = 'complete'
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(all_summary,indent=2))


if __name__ == '__main__':
    main()
