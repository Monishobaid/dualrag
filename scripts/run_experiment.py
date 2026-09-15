#!/usr/bin/env python3
"""Main experiment runner."""

import argparse
import logging
from pathlib import Path
import sys
import json
import random
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAGConfig
from src.data.hotpot_loader import HotpotQALoader
from src.retrieval.embedder import Embedder
from src.retrieval.index import EmbeddingCache
from src.llm.qwen import QwenModel
from src.llm.azure_openai import AzureOpenAIModel
from src.rag.iterative_rag import IterativeRAG
from src.rag.baselines import get_baseline
from src.eval.hotpot_eval import evaluate_single, evaluate_predictions
from src.eval.metrics import aggregate_trace_metrics
from src.utils.logging import setup_logger, JSONLWriter, log_experiment_start, log_experiment_end, read_jsonl

logger = None

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_single_experiment(args, method, base_run_id):
    # Set up run ID
    if args.run_id is None or args.method == 'all':
        run_id = f"{method}_{base_run_id}"
    else:
        run_id = args.run_id

    output_path = Path(args.output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    global logger
    logger = setup_logger(f'experiment_{method}', str(output_path / 'experiment.log'))
    
    set_seed(args.seed)
    
    config = RAGConfig(
        model_name=args.model_name,
        embed_model_name=args.embed_model_name,
        device=args.device,
        dataset_path=args.dataset_path,
        split=args.split,
        max_examples=args.max_examples,
        top_k=args.top_k,
        max_hops=args.hops,
        retrieval_mode=args.retrieval_mode,
        dense_weight=args.dense_weight,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        chunk_tokens=args.chunk_tokens,
        convergence_k=args.convergence_k,
        min_hops=args.min_hops,
        overlap_threshold=args.overlap_threshold,
        conf_threshold=args.conf_threshold,
        enable_evidence_critic=args.enable_evidence_critic,
        evidence_critic_threshold=args.evidence_critic_threshold,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        method=method,
        run_id=run_id
    )
    
    config = get_baseline(method, config)
    config.save(str(output_path / 'config.json'))
    
    log_experiment_start(logger, config)
    
    logger.info("Loading dataset...")
    loader = HotpotQALoader(
        filepath=config.dataset_path,
        max_examples=config.max_examples,
        seed=config.seed
    )
    logger.info(f"Loaded {len(loader)} examples")
    
    logger.info("Initializing models...")
    embedder = Embedder(model_name=config.embed_model_name, device=config.device)
    cache = EmbeddingCache(cache_dir=config.cache_dir)
    
    if config.model_name in AzureOpenAIModel.CONFIGS or config.model_name.lower().startswith(('gpt', 'azure')):
        llm = AzureOpenAIModel(model_name=config.model_name, device=config.device)
    else:
        llm = QwenModel(model_name=config.model_name, device=config.device)
    
    rag = IterativeRAG(
        llm_model=llm,
        embedder=embedder,
        embedding_cache=cache,
        config=config
    )
    
    logger.info(f"Running experiments for method: {method}...")
    
    predictions = {}
    gold_answers = {}
    failed_examples = []
    
    with JSONLWriter(str(output_path / 'traces.jsonl')) as trace_writer:
        for idx, example in enumerate(tqdm(loader, desc=f"Processing ({method})"), 1):
            try:
                print(f"\n\n{'#'*80}")
                print(f"📋 METHOD: {method} | EXAMPLE {idx}/{len(loader)}")
                print(f"{'#'*80}")
                print(f"🆔 ID: {example.id}")
                print(f"❓ Question: {example.question}")
                print(f"🎯 Gold Answer: {example.answer}")
                print(f"📚 Available paragraphs: {len(example.candidate_paragraphs)}")
                print(f"{'#'*80}\n")

                # Oracle diagnostic: pass only gold supporting-fact titles to
                # generation. This isolates answer-generation quality from
                # retrieval errors and must never be used for a deployable run.
                run_paragraphs = example.candidate_paragraphs
                run_hops = config.max_hops
                run_top_k = config.top_k
                if method == 'oracle_evidence':
                    gold_titles = {fact[0] for fact in example.supporting_facts if fact}
                    run_paragraphs = [p for p in example.candidate_paragraphs if p['title'] in gold_titles]
                    run_hops, run_top_k = 1, max(1, len(run_paragraphs))
                trace = rag.run(
                    question=example.question,
                    candidate_paragraphs=run_paragraphs,
                    max_hops=run_hops,
                    top_k=run_top_k
                )
                if method == 'oracle_evidence':
                    trace['oracle_evidence'] = True
                    trace['oracle_titles'] = sorted({p['title'] for p in run_paragraphs})

                eval_result = evaluate_single(trace['final_answer'], example.answer)

                print(f"\n{'='*80}")
                print(f"📊 EVALUATION ({method})")
                print(f"{'='*80}")
                print(f"Predicted: {trace['final_answer']}")
                print(f"Gold:      {example.answer}")
                print(f"EM:        {eval_result['em']} {'✅' if eval_result['em'] else '❌'}")
                print(f"F1:        {eval_result['f1']:.3f}")
                print(f"{'='*80}\n")

                predictions[example.id] = trace['final_answer']
                gold_answers[example.id] = example.answer

                trace_log = {
                    'id': example.id,
                    'question': example.question,
                    'gold_answer': example.answer,
                    'method': config.method,
                    'em': eval_result['em'],
                    'f1': eval_result['f1'],
                    **trace
                }
                trace_writer.write(trace_log)

            except Exception as e:
                logger.error(f"Error processing example {example.id}: {e}", exc_info=True)
                print(f"\n❌ ERROR processing {example.id} ({method}): {e}\n")
                failed_examples.append({'id': example.id, 'error': str(e)})
                continue
    
    cache.save()
    
    with open(output_path / 'predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    eval_metrics = evaluate_predictions(predictions, gold_answers)
    # Failed examples are incorrect for benchmark denominators. The runner
    # continues after per-example errors, so do not silently score only the
    # successful subset.
    if failed_examples and len(loader):
        success_fraction = len(predictions) / len(loader)
        eval_metrics['em'] *= success_fraction
        eval_metrics['f1'] *= success_fraction
        eval_metrics['num_examples'] = len(loader)
    traces = read_jsonl(str(output_path / 'traces.jsonl'))
    trace_metrics = aggregate_trace_metrics(traces)
    final_metrics = {**eval_metrics, **trace_metrics,
                     'num_requested_examples': len(loader),
                     'num_failed_examples': len(failed_examples),
                     'failed_example_ids': [item['id'] for item in failed_examples]}
    
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    log_experiment_end(logger, final_metrics)
    
    return final_metrics, output_path


def main():
    parser = argparse.ArgumentParser(description='Run RAG experiment')
    
    # Dataset args
    parser.add_argument('--dataset_path', type=str, default='data/hotpotqa/hotpot_dev_distractor_v1.json')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'train'])
    parser.add_argument('--max_examples', type=int, default=None)
    
    # Method args
    parser.add_argument('--method', type=str, default='fixed_hops',
                        choices=['fixed_hops', 'fixed_hops_reasoning_stop', 
                                'retrieval_stop_only', 'dual_stopping', 'oracle_evidence', 'all'])
    
    # Retrieval args
    parser.add_argument('--hops', type=int, default=4, help='Maximum hops')
    parser.add_argument('--top_k', type=int, default=5, help='Paragraphs per hop')
    parser.add_argument('--retrieval_mode', type=str, default='dense', choices=['dense', 'bm25', 'hybrid'],
                        help='Retriever: dense embeddings, lexical BM25, or normalized hybrid')
    parser.add_argument('--dense_weight', type=float, default=0.5,
                        help='Dense weight for hybrid retrieval (0-1)')
    parser.add_argument('--min_hops', type=int, default=2, help='Minimum hops for retrieval stop')
    parser.add_argument('--overlap_threshold', type=float, default=0.7)
    parser.add_argument('--conf_threshold', type=float, default=0.6)
    parser.add_argument('--enable_evidence_critic', action='store_true',
                        help='Require model-free evidence sufficiency before retrieval stopping')
    parser.add_argument('--evidence_critic_threshold', type=float, default=0.65,
                        help='Evidence critic score threshold in [0, 1]')
    
    # Generation args
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--chunk_tokens', type=int, default=64)
    parser.add_argument('--convergence_k', type=int, default=2)
    
    # Model args
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--embed_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--device', type=str, default='auto')
    
    # Output args
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--cache_dir', type=str, default='runs/cache_embeddings')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_id', type=str, default=None)
    
    args = parser.parse_args()
    
    base_run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.method == 'all':
        methods_to_run = ['fixed_hops', 'retrieval_stop_only', 'fixed_hops_reasoning_stop', 'dual_stopping']
    else:
        methods_to_run = [args.method]
        
    all_results = {}
    
    for method in methods_to_run:
        print(f"\n\n{'='*80}")
        print(f"🚀 STARTING EXPERIMENT FOR METHOD: {method}")
        print(f"{'='*80}\n")
        
        metrics, out_path = run_single_experiment(args, method, base_run_id)
        all_results[method] = metrics
        
    if args.method == 'all' or len(methods_to_run) > 1:
        print(f"\n\n{'='*80}")
        print("🏆 FINAL RESULTS COMPARISON 🏆")
        print(f"{'='*80}")
        
        metrics_to_show = ['em', 'f1', 'avg_hops', 'avg_tokens_generated', 'reasoning_stop_rate', 'retrieval_stop_rate']
        
        header = f"{'Method':<30} | " + " | ".join([f"{m:<20}" for m in metrics_to_show])
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for method, metrics in all_results.items():
            row = f"{method:<30} | "
            for m in metrics_to_show:
                val = metrics.get(m, "N/A")
                if isinstance(val, (float, np.float64, np.float32)):
                    row += f"{val:<20.4f} | "
                else:
                    row += f"{str(val):<20} | "
            print(row)
            
        print("-" * len(header))
        
        summary_path = Path(args.output_dir) / f"summary_{base_run_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
