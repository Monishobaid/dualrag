#!/usr/bin/env bash
# Finish the already launched experiment and back up only completed artifacts.
# Arguments are the current pooled-reader and fullwiki-lexical shell PIDs.
set -euo pipefail
cd "$(dirname "$0")/.."
reader_pid="${1:?reader process ID required}"
wiki_pid="${2:?fullwiki build process ID required}"
[[ "$reader_pid" =~ ^[0-9]+$ && "$wiki_pid" =~ ^[0-9]+$ ]]
status=outputs/evidence_completion_status.txt
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "FAILED exit=$rc; inspect outputs/evidence_completion_console.log" > "$status"; fi' EXIT
wait_for() { while kill -0 "$1" 2>/dev/null; do sleep 20; done; }
publish() {
  git add "$1" RESEARCH_LOG.md
  git commit -m "$2"
  git push origin v-2
}
record_report() {
  python - "$1" "$2" <<'PY'
from pathlib import Path
import sys
report, label = Path(sys.argv[1]), sys.argv[2]
with Path('RESEARCH_LOG.md').open('a') as f:
    f.write('\n### Completed artifact: '+label+'\n\n'+report.read_text()+'\n')
PY
}
echo 'Waiting for pooled reader' > "$status"
wait_for "$reader_pid"
python scripts/analyze_corpus_reader.py --run outputs/evidence_20260930_renku_reader --output research_results/2026-09-30/pooled_reader
python scripts/archive_retrieval_logs.py --run outputs/evidence_20260930_renku_retrieval --output research_results/2026-09-30/evidence_20260930_renku_retrieval
record_report research_results/2026-09-30/pooled_reader/REPORT.md 'pooled reader and sufficiency audit'
git add research_results/2026-09-30/evidence_20260930_renku_retrieval
publish research_results/2026-09-30/pooled_reader 'Record pooled corpus reader results and raw traces'
echo 'Waiting for official fullwiki lexical evaluation' > "$status"
wait_for "$wiki_pid"
python scripts/archive_retrieval_logs.py --run outputs/fullwiki_20260930_bm25 --output research_results/2026-09-30/fullwiki_bm25
record_report research_results/2026-09-30/fullwiki_bm25/REPORT.md 'official fullwiki BM25'
publish research_results/2026-09-30/fullwiki_bm25 'Record official fullwiki lexical retrieval results'
echo 'Running official fullwiki dense, hybrid, reranker and reader' > "$status"
bash scripts/run_fullwiki_dense.sh
python scripts/archive_retrieval_logs.py --run outputs/fullwiki_20260930_all --output research_results/2026-09-30/fullwiki_all
record_report research_results/2026-09-30/fullwiki_all/REPORT.md 'official fullwiki all retrieval arms'
record_report research_results/2026-09-30/fullwiki_reader/REPORT.md 'official fullwiki reader and sufficiency audit'
git add research_results/2026-09-30/fullwiki_all
publish research_results/2026-09-30/fullwiki_reader 'Record matched fullwiki retrieval and answer evaluation'
echo 'COMPLETE; all measured artifacts backed up to GitLab v-2' > "$status"
