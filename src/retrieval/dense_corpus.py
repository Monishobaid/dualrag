"""Batched dense corpus indexing; exact blockwise search with a disk matrix.

MiniLM baseline uses normalized sentence embeddings. Cache identity includes
corpus bytes, model commit, tokenizer limit and contextualization variant.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from src.retrieval.corpus import digest


class DenseCorpus:
    def __init__(self, corpus, model, corpus_hash, model_revision, cache_dir,
                 contextual=False, batch_size=64):
        self.model = model
        key = digest([corpus_hash, model_revision, model.max_seq_length, contextual])
        root = Path(cache_dir) / key
        root.mkdir(parents=True, exist_ok=True)
        self.path = root
        meta = root / 'complete.json'
        start = time.perf_counter()
        if not meta.exists():
            n = corpus.metadata['chunks']
            matrix = np.lib.format.open_memmap(root / 'vectors.npy', mode='w+', dtype=np.float32,
                                               shape=(n, model.get_sentence_embedding_dimension()))
            ids, texts, offset, overflow = [], [], 0, 0
            field = 'contextual' if contextual else 'plain'
            def encode_batch(batch, offset):
                lengths = [len(x) for x in model.tokenizer(batch, truncation=False)['input_ids']]
                matrix[offset:offset + len(batch)] = model.encode(batch, batch_size=batch_size,
                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                return sum(x > model.max_seq_length for x in lengths)
            for chunk in corpus.chunks():
                ids.append(chunk['id'])
                texts.append(chunk[field])
                if len(texts) == batch_size:
                    overflow += encode_batch(texts, offset)
                    offset += len(texts)
                    texts = []
                    if offset % (batch_size * 20) == 0:
                        print('Encoded', offset, '/', n, 'contextual=', contextual, flush=True)
            if texts:
                overflow += encode_batch(texts, offset)
            matrix.flush()
            np.save(root / 'ids.npy', np.asarray(ids, dtype=np.int64))
            meta.write_text(json.dumps({'corpus_sha256': corpus_hash, 'model_revision': model_revision,
                'max_seq_length': model.max_seq_length, 'contextual': contextual,
                'overflow_chunks': overflow, 'num_chunks': n,
                'build_seconds': time.perf_counter() - start}, indent=2))
        self.metadata = json.loads(meta.read_text())
        self.vectors = np.load(root / 'vectors.npy', mmap_mode='r')
        self.ids = np.load(root / 'ids.npy', mmap_mode='r')

    def search_vector(self, vector, k=100):
        candidates = []
        for offset in range(0, len(self.ids), 32768):
            scores = self.vectors[offset:offset + 32768] @ vector
            # Lexsort defines deterministic ties; never depend on argpartition ties.
            order = np.lexsort((self.ids[offset:offset + len(scores)], -scores))[:k]
            candidates.extend((int(self.ids[offset + i]), float(scores[i])) for i in order)
        return sorted(candidates, key=lambda x: (-x[1], x[0]))[:k]
