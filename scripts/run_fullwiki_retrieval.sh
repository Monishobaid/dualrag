#!/usr/bin/env bash
# True official fullwiki; never relabel pooled evidence as fullwiki results.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/runs/hf-cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p data/corpus outputs
archive=data/corpus/fullwiki-intro.tar.bz2
if [ ! -f "$archive" ]; then
  if [ ! -f "${archive}.part" ]; then
    curl -fL --connect-timeout 20 --max-time 3600 -o "${archive}.part" https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
  fi
  python - "${archive}.part" <<'PY'
import hashlib,sys
h=hashlib.md5()
with open(sys.argv[1],'rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
assert h.hexdigest()=='01edf64cd120ecc03a2745352779514c', 'Incomplete or wrong official archive'
PY
  mv "${archive}.part" "$archive"
fi
if [ ! -f data/corpus/fullwiki.sqlite ]; then
  python scripts/build_corpus.py --wiki-intro-archive "$archive" --output data/corpus/fullwiki.sqlite
fi
# Run CPU lexical arms first; GPU corpus embedding is a separately budgeted run.
python scripts/run_corpus_retrieval.py --corpus data/corpus/fullwiki.sqlite --output outputs/fullwiki_20260930_bm25 --arms bm25
python scripts/analyze_corpus_retrieval.py --run outputs/fullwiki_20260930_bm25 --output research_results/2026-09-30/fullwiki_bm25
