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
from src.rag.iterative_rag import IterativeRAG
from src.rag.baselines import get_baseline
from src.eval.hotpot_eval import evaluate_single
from src.utils.logging import setup_logger, JSONLWriter, log_experiment_start, log_experiment_end

logger = None


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Run RAG experiment')
    
    # Dataset args
    parser.add_argument('--dataset_path', type=str, default='data/hotpotqa/hotpot_dev_distractor_v1.json')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'train'])
    parser.add_argument('--max_examples', type=int, default=None)
    
    # Method args
    parser.add_argument('--method', type=str, default='fixed_hops',
                        choices=['fixed_hops', 'fixed_hops_reasoning_stop', 
                                'retrieval_stop_only', 'dual_stopping'])
    
    # Retrieval args
    parser.add_argument('--hops', type=int, default=4, help='Maximum hops')
    parser.add_argument('--top_k', type=int, default=5, help='Paragraphs per hop')
    parser.add_argument('--min_hops', type=int, default=2, help='Minimum hops for retrieval stop')
    parser.add_argument('--overlap_threshold', type=float, default=0.7)
    parser.add_argument('--conf_threshold', type=float, default=0.6)
    
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
    
    # Generate run ID
    if args.run_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_id = f"{args.method}_{timestamp}"
    
    # Create output directory
    output_path = Path(args.output_dir) / args.run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    global logger
    logger = setup_logger('experiment', str(output_path / 'experiment.log'))
    
    # Set seed
    set_seed(args.seed)
    
    # Create config
    config = RAGConfig(
        model_name=args.model_name,
        embed_model_name=args.embed_model_name,
        device=args.device,
        dataset_path=args.dataset_path,
        split=args.split,
        max_examples=args.max_examples,
        top_k=args.top_k,
        max_hops=args.hops,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        chunk_tokens=args.chunk_tokens,
        convergence_k=args.convergence_k,
        min_hops=args.min_hops,
        overlap_threshold=args.overlap_threshold,
        conf_threshold=args.conf_threshold,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        method=args.method,
        run_id=args.run_id
    )
    
    # Apply baseline configuration
    config = get_baseline(args.method, config)
    
    # Save config
    config.save(str(output_path / 'config.json'))
    
    log_experiment_start(logger, config)
    
    # Load dataset
    logger.info("Loading dataset...")
    loader = HotpotQALoader(
        filepath=config.dataset_path,
        max_examples=config.max_examples,
        seed=config.seed
    )
    logger.info(f"Loaded {len(loader)} examples")
    
    # Initialize models
    logger.info("Initializing models...")
    embedder = Embedder(model_name=config.embed_model_name, device=config.device)
    cache = EmbeddingCache(cache_dir=config.cache_dir)
    llm = QwenModel(model_name=config.model_name, device=config.device)
    
    # Initialize RAG engine
    rag = IterativeRAG(
        llm_model=llm,
        embedder=embedder,
        embedding_cache=cache,
        config=config
    )
    
    # Run experiments
    logger.info("Running experiments...")
    
    predictions = {}
    gold_answers = {}
    
    with JSONLWriter(str(output_path / 'traces.jsonl')) as trace_writer:
        for idx, example in enumerate(tqdm(loader, desc="Processing examples"), 1):
            try:
                print(f"\n\n{'#'*80}")
                print(f"📋 EXAMPLE {idx}/{len(loader)}")
                print(f"{'#'*80}")
                print(f"🆔 ID: {example.id}")
                print(f"❓ Question: {example.question}")
                print(f"🎯 Gold Answer: {example.answer}")
                print(f"📚 Available paragraphs: {len(example.candidate_paragraphs)}")
                print(f"{'#'*80}\n")

                # Run RAG
                trace = rag.run(
                    question=example.question,
                    candidate_paragraphs=example.candidate_paragraphs,
                    max_hops=config.max_hops,
                    top_k=config.top_k
                )

                # Evaluate
                eval_result = evaluate_single(trace['final_answer'], example.answer)

                # Display evaluation
                print(f"\n{'='*80}")
                print(f"📊 EVALUATION")
                print(f"{'='*80}")
                print(f"Predicted: {trace['final_answer']}")
                print(f"Gold:      {example.answer}")
                print(f"EM:        {eval_result['em']} {'✅' if eval_result['em'] else '❌'}")
                print(f"F1:        {eval_result['f1']:.3f}")
                print(f"{'='*80}\n")

                # Store prediction
                predictions[example.id] = trace['final_answer']
                gold_answers[example.id] = example.answer

                # Write trace
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
                print(f"\n❌ ERROR processing {example.id}: {e}\n")
                continue
    
    # Save predictions
    with open(output_path / 'predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Compute final metrics
    from src.eval.hotpot_eval import evaluate_predictions
    from src.eval.metrics import aggregate_trace_metrics
    from src.utils.logging import read_jsonl
    
    eval_metrics = evaluate_predictions(predictions, gold_answers)
    
    traces = read_jsonl(str(output_path / 'traces.jsonl'))
    trace_metrics = aggregate_trace_metrics(traces)
    
    final_metrics = {**eval_metrics, **trace_metrics}
    
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    log_experiment_end(logger, final_metrics)
    
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Run ID: {args.run_id}")


if __name__ == "__main__":
    main()
