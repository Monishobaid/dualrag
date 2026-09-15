import os

filepath = 'colab_dual_stopping_rag.py'

with open(filepath, 'r') as f:
    content = f.read()

# 1. Fix missing import sentence_transformers CrossEncoder
content = content.replace(
    'from sentence_transformers import SentenceTransformer\n', 
    'from sentence_transformers import SentenceTransformer, CrossEncoder\n'
)

# 2. Fix empty try block for AzureOpenAI
content = content.replace(
    'try:\nexcept ImportError:\n    AzureOpenAI = None\n    tiktoken = None',
    'try:\n    pass\nexcept ImportError:\n    AzureOpenAI = None\n    tiktoken = None'
)

# 3. Update defaults in RAGConfig
content = content.replace(
    'model_name: str = "Qwen/Qwen2.5-3B-Instruct"',
    'model_name: str = "Qwen/Qwen2.5-7B-Instruct"'
)
content = content.replace(
    'embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"',
    'embed_model_name: str = "facebook/contriever-msmarco"\n    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"'
)

# 4. Update defaults in argparse
content = content.replace(
    "parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')",
    "parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct')"
)
content = content.replace(
    "parser.add_argument('--embed_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')",
    "parser.add_argument('--embed_model_name', type=str, default='facebook/contriever-msmarco')\n    parser.add_argument('--reranker_model_name', type=str, default='BAAI/bge-reranker-v2-m3')"
)

# 5. Embedder default init
content = content.replace(
    'def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"):',
    'def __init__(self, model_name: str = "facebook/contriever-msmarco", device: str = "auto"):'
)

# 6. ParagraphRanker init
content = content.replace(
    'def __init__(self, embedder: Embedder, cache: Optional[EmbeddingCache] = None):',
    'def __init__(self, embedder: Embedder, cache: Optional[EmbeddingCache] = None, cross_encoder: Optional[CrossEncoder] = None):'
)
old_ranker_init = '''        """
        self.embedder = embedder
        self.cache = cache'''
new_ranker_init = '''        """
        self.embedder = embedder
        self.cache = cache
        self.cross_encoder = cross_encoder'''
content = content.replace(old_ranker_init, new_ranker_init)

# 7. ParagraphRanker.rank logic
old_rank = '''        # Build index and search
        index = ParagraphIndex(paragraphs, para_embeddings)
        results = index.search(query_embedding, top_k=top_k)'''

new_rank = '''        # Build index and search
        retrieve_k = top_k * 3 if self.cross_encoder else top_k
        index = ParagraphIndex(paragraphs, para_embeddings)
        results = index.search(query_embedding, top_k=retrieve_k)
        
        # Rerank with CrossEncoder if available
        if self.cross_encoder and results:
            cross_inp = [[query, r['paragraph']['text']] for r in results]
            scores = self.cross_encoder.predict(cross_inp)
            for r, s in zip(results, scores):
                r['score'] = float(s)
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:top_k]'''
content = content.replace(old_rank, new_rank)

# 8. IterativeRAG init integration
old_iterative_init = '''        self.ranker = ParagraphRanker(embedder, cache=embedding_cache)'''
new_iterative_init = '''        # Initialize reranker if configured
        import torch
        self.cross_encoder = None
        if self.config and getattr(self.config, 'reranker_model_name', None):
            self.cross_encoder = CrossEncoder(
                self.config.reranker_model_name,
                device=self.config.device if self.config.device != "auto" else (
                    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                )
            )
            
        self.ranker = ParagraphRanker(embedder, cache=embedding_cache, cross_encoder=self.cross_encoder)'''
content = content.replace(old_iterative_init, new_iterative_init)

# 9. Fix Jupyter missing __file__ NameError
content = content.replace(
    'sys.path.insert(0, str(Path(__file__).parent.parent))',
    '# sys.path.insert(0, str(Path(__file__).parent.parent)) # Removed for Colab compatibility'
)

# 10. Fix Jupyter argparse crash due to kernel arguments
content = content.replace(
    'args = parser.parse_args()',
    'args, _ = parser.parse_known_args()'
)

# 11. Fix get_baseline alias missing from import stripping
content = content.replace(
    'config = get_baseline(args.method, config)',
    'config = get_baseline_config(args.method, config)'
)

# 12. Add auto-download logic if dataset missing
download_snippet = '''    import os
    import requests
    import urllib.request
    
    # Auto-download dataset if missing in Colab
    if not os.path.exists(config.dataset_path):
        logger.info(f"Dataset not found at {config.dataset_path}. Downloading...")
        os.makedirs(os.path.dirname(config.dataset_path), exist_ok=True)
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
        
        try:
            urllib.request.urlretrieve(url, config.dataset_path)
            logger.info("Download complete.")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise e

    # Load dataset'''

content = content.replace(
    '    # Load dataset',
    download_snippet
)

# 13. Ensure cache gets saved at the end of execution
content = content.replace(
    '''    # Save predictions
    with open(output_path / 'predictions.json', 'w') as f:''',
    '''    # Save embedding cache
    cache.save()
    
    # Save predictions
    with open(output_path / 'predictions.json', 'w') as f:'''
)

with open(filepath, 'w') as f:
    f.write(content)

print("Patching complete.")
