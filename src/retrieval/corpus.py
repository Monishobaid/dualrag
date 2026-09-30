"""Persistent, question-independent evidence corpus.

HotpotQA supplies sentence addresses (Yang et al., 2018). Anthropic contextual
retrieval motivates document context before indexing; our deterministic title +
lead prefix is a baseline, not a reproduction of its generated contextualization.
SQLite FTS5 provides corpus-wide BM25; RRF follows Cormack et al., SIGIR 2009.
"""
from __future__ import annotations
import bz2
import hashlib
import io
import json
import re
import shutil
import sqlite3
import tarfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def context_documents(paths):
    """Only read document context: never questions, answers or gold support."""
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for ex in data:
            for title, sentences in ex['context']:
                yield {'title': title, 'sentences': sentences}


def wiki_documents(archive):
    """Stream official intro tar.bz2 without extracting untrusted member paths.

    Official fullwiki intro text is already segmented; preserve those indices.
    Full-article dumps require a separate paragraph addressing scheme and are
    intentionally not accepted as equivalent to the introductory corpus.
    """
    with tarfile.open(archive, 'r|bz2') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.bz2'):
                continue
            with bz2.open(io.BytesIO(tar.extractfile(member).read()), 'rt', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    sentences = obj['text']
                    if sentences and isinstance(sentences[0], list):
                        raise ValueError('Expected intro corpus, not full article text')
                    yield {'title': obj['title'], 'sentences': sentences}


def build_corpus(path, documents, scope, source_hashes, chunk_words=120):
    """Deduplicate exact title+sentences, split at sentences, retain parent lead.

    A long sentence stays intact and its tokenizer overflow is measured later.
    Same-title different-text versions remain separate to avoid silent data loss.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    if partial.exists():
        raise FileExistsError('Prior incomplete build exists: ' + str(partial))
    db = sqlite3.connect(partial)
    db.executescript('''
      CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT, sentences TEXT);
      CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id TEXT, title TEXT,
          start INTEGER, sentences TEXT, plain TEXT, contextual TEXT);
      CREATE VIRTUAL TABLE plain_index USING fts5(text, tokenize='unicode61');
      CREATE VIRTUAL TABLE contextual_index USING fts5(text, tokenize='unicode61');
      CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
      CREATE INDEX documents_title ON documents(title);
    ''')
    n = 0
    for document in documents:
        title, sentences = document['title'], document['sentences']
        if not isinstance(title, str) or not all(isinstance(s, str) for s in sentences):
            raise ValueError('Invalid document schema')
        if not sentences:
            continue
        docid = digest([title, sentences])
        inserted = db.execute('INSERT OR IGNORE INTO documents VALUES (?,?,?)',
                              (docid, title, json.dumps(sentences))).rowcount
        if not inserted:
            continue
        start, current, size = 0, [], 0
        groups = []
        for i, sentence in enumerate(sentences):
            words = len(sentence.split())
            if current and size + words > chunk_words:
                groups.append((start, current))
                start, current, size = i, [], 0
            current.append(sentence)
            size += words
        if current:
            groups.append((start, current))
        for start, group in groups:
            text = ' '.join(group)
            contextual = 'Document: ' + title + '\nLead: ' + sentences[0] + '\nPassage: ' + text
            cur = db.execute('INSERT INTO chunks(document_id,title,start,sentences,plain,contextual) VALUES (?,?,?,?,?,?)',
                             (docid, title, start, json.dumps(group), text, contextual))
            cid = cur.lastrowid
            db.execute('INSERT INTO plain_index(rowid,text) VALUES (?,?)', (cid, text))
            db.execute('INSERT INTO contextual_index(rowid,text) VALUES (?,?)', (cid, contextual))
        n += 1
        if n % 10000 == 0:
            if shutil.disk_usage(path.parent).free < 2 * 1024**3:
                raise OSError('Less than 2 GiB free: stopping corpus build')
            db.commit()
            print('Indexed documents:', n, flush=True)
    metadata = {'scope': scope, 'source_hashes': source_hashes, 'chunk_words': chunk_words,
                'documents': n, 'chunks': db.execute('SELECT count(*) FROM chunks').fetchone()[0],
                'contextualization': 'deterministic title + first sentence; not LLM contextualization',
                'format_version': 1}
    for k, v in metadata.items():
        db.execute('INSERT INTO metadata VALUES (?,?)', (k, json.dumps(v)))
    db.commit()
    db.close()
    partial.rename(path)
    return metadata


class Corpus:
    def __init__(self, path):
        self.path = str(path)
        self.db = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row
        self.metadata = {r['key']: json.loads(r['value']) for r in self.db.execute('SELECT * FROM metadata')}

    def chunks(self):
        for row in self.db.execute('SELECT * FROM chunks ORDER BY id'):
            yield self.decode(row)

    @staticmethod
    def decode(row):
        d = dict(row)
        d['sentences'] = json.loads(d['sentences'])
        return d

    def get(self, cid):
        return self.decode(self.db.execute('SELECT * FROM chunks WHERE id=?', (int(cid),)).fetchone())

    def lexical(self, query, k=100, contextual=False):
        # Quote tokens so user queries cannot inject FTS operators or SQL.
        terms = sorted(set(re.findall(r'\w+', query.lower())))
        if not terms:
            return []
        match = ' OR '.join('"' + t.replace('"', '""') + '"' for t in terms)
        table = 'contextual_index' if contextual else 'plain_index'
        rows = self.db.execute(f'SELECT rowid,bm25({table}) AS score FROM {table} '
                               f'WHERE {table} MATCH ? ORDER BY score,rowid LIMIT ?', (match, k))
        return [(r['rowid'], -r['score']) for r in rows]


def fuse(rankings, k=100, constant=60):
    """RRF combines ranks, avoiding fragile per-query dense/BM25 score scales."""
    scores = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]


def retrieval_metrics(example, chunks, corpus):
    """Exact support text+title matching survives duplicate article versions.

    Original support indices refer to each example's source version. A matching
    title with different content must not silently count as supporting evidence.
    Complete support coverage is a diagnostic proxy, not semantic sufficiency.
    """
    contexts = {title: sentences for title, sentences in example['context']}
    gold_titles = {t for t, _ in example['supporting_facts']}
    gold = {(t, contexts[t][i].strip()) for t, i in example['supporting_facts']}
    found = {(c['title'], s.strip()) for c in chunks for s in c['sentences']}
    titles = {c['title'] for c in chunks}
    available = set()
    for title in gold_titles:
        for row in corpus.db.execute('SELECT sentences FROM documents WHERE title=?', (title,)):
            available.update((title, s.strip()) for s in json.loads(row[0]))
    return {'title_recall': len(gold_titles & titles) / max(1, len(gold_titles)),
            'sentence_recall': len(gold & found) / max(1, len(gold)),
            'complete_support': bool(gold) and gold <= found,
            'corpus_support_availability': len(gold & available) / max(1, len(gold)),
            'missing_support': [{'title': t, 'text': s} for t, s in sorted(gold - found)]}
