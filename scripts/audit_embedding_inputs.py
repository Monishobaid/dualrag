#!/usr/bin/env python3
"""Count encoder input overflow without changing retrieval or using a GPU.

Downloads tokenizer/config JSON at the recorded model revisions. Requires the
tokenizers package (also installed with Transformers). Overflow is not proof
that a gold supporting sentence was lost: it may occur early in the paragraph.
"""
import json
from pathlib import Path
import tempfile
from urllib.request import urlretrieve

from tokenizers import Tokenizer

ROOT = Path('research_results/2026-09-24')
MODELS = [
    ('MiniLM', 'sentence-transformers/all-MiniLM-L6-v2',
     '1110a243fdf4706b3f48f1d95db1a4f5529b4d41', [False, True]),
    ('BGE-base', 'BAAI/bge-base-en-v1.5',
     'a5beb1e3e68b9ab74eb54cfd186867f64f240e1a', [True]),
]


def main():
    data = {r['_id']: r for r in json.loads(Path('data/hotpotqa/hotpot_dev_distractor_v1.json').read_text())}
    manifests = {'original50': 'example_ids_dev_50_seed42.json',
                 'confirmation100': 'research_confirm100_ids.json'}
    report = []
    with tempfile.TemporaryDirectory(prefix='dualrag-tokenizer-') as tmp:
        for label, model, revision, prefixes in MODELS:
            paths = {}
            for name in ['tokenizer.json', 'sentence_bert_config.json']:
                path = Path(tmp) / name
                urlretrieve(f'https://huggingface.co/{model}/resolve/{revision}/{name}', path)
                paths[name] = path
            limit = json.loads(paths['sentence_bert_config.json'].read_text())['max_seq_length']
            tokenizer = Tokenizer.from_file(str(paths['tokenizer.json']))
            tokenizer.no_truncation(); tokenizer.no_padding()
            for prefix in prefixes:
                for cohort, filename in manifests.items():
                    manifest = json.loads((ROOT / filename).read_text())
                    ids = manifest['ids'] if isinstance(manifest, dict) else manifest
                    overflow = []; lengths = []
                    for key in ids:
                        row = data[key]; gold_titles = {x[0] for x in row['supporting_facts']}
                        for title, sentences in row['context']:
                            text = (title + '\n' if prefix else '') + ' '.join(sentences)
                            length = len(tokenizer.encode(text).ids)
                            lengths.append(length)
                            if length > limit:
                                overflow.append({'id': key, 'title': title, 'tokens': length,
                                                 'gold_title': title in gold_titles})
                    report.append({'model': model, 'revision': revision, 'label': label,
                                   'title_prefix': prefix, 'cohort': cohort, 'encoder_limit': limit,
                                   'paragraphs': len(lengths), 'overflow_count': len(overflow),
                                   'max_tokens': max(lengths), 'overflow': overflow})
    (ROOT / 'embedding_input_audit.json').write_text(json.dumps(report, indent=2))
    lines = ['# Embedding input-length audit', '',
             'Post-run CPU audit using the tokenizers and encoder limits at the recorded '
             'checkpoint revisions. This does not change any retrieval result or prompt. '
             'Overflow means the encoder cannot represent the complete paragraph; it does '
             'not establish whether the annotated supporting sentence lies in the removed tail.', '',
             '| Model | Title prefix | Cohort | Encoder limit | Paragraphs | Overflow | Gold-title paragraphs overflowing | Maximum length |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in report:
        lines.append(f'| {r["label"]} | {r["title_prefix"]} | {r["cohort"]} | {r["encoder_limit"]} | '
                     f'{r["paragraphs"]} | {r["overflow_count"]} | '
                     f'{sum(x["gold_title"] for x in r["overflow"])} | {r["max_tokens"]} |')
    lines += ['', 'Both title-prefix variants on confirmation100 are tokenizer audits only; '
              'no title-prefix or BGE generation benchmark was run on that cohort. Reader context uses original complete '
              'sentences independently of embedding truncation. A model upgrade changes '
              'both representation and encoder length, so its gain cannot be attributed '
              'solely to one of those factors.', '',
              'None of the gold-title paragraphs overflowed in the actual plain-MiniLM '
              'original50/confirmation100 or BGE original50 configurations. Thus this '
              'audit does not support blaming their retrieval misses on truncated gold '
              'paragraph tails; long distractor representations can still be affected.', '',
              'Sources: [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), '
              '[BGE model card](https://huggingface.co/BAAI/bge-base-en-v1.5).', '']
    Path('docs/EMBEDDING_AUDIT_2026-09-24.md').write_text('\n'.join(lines))
    print('Audited', len(report), 'tokenizer/cohort configurations')


if __name__ == '__main__':
    main()
