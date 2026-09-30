#!/usr/bin/env python3
"""Build a question-independent corpus from pooled contexts or official fullwiki."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.corpus import build_corpus, context_documents, wiki_documents, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--contexts', nargs='+')
    group.add_argument('--wiki-intro-archive')
    p.add_argument('--output', required=True)
    p.add_argument('--chunk-words', type=int, default=120)
    a = p.parse_args()
    paths = a.contexts or [a.wiki_intro_archive]
    docs = context_documents(paths) if a.contexts else wiki_documents(paths[0])
    scope = 'pooled-contexts' if a.contexts else 'hotpotqa-official-fullwiki-intro'
    result = build_corpus(a.output, docs, scope, {x: file_hash(x) for x in paths}, a.chunk_words)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
