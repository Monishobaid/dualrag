#!/usr/bin/env bash
# Run from repository root on Renku with .venv active. Each output is immutable.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python -m unittest discover -s tests -q
python - <<'PY'
import json, random, hashlib
from pathlib import Path
old=json.loads(Path('outputs/example_ids_dev_50_seed42.json').read_text())
old=old['ids'] if isinstance(old,dict) else old
p=Path('outputs/research_confirm100_ids.json')
data=json.loads(Path('data/hotpotqa/hotpot_dev_distractor_v1.json').read_text())
ids=random.Random(20260924).sample(sorted(x['_id'] for x in data if x['_id'] not in set(old)),100)
if p.exists():
 assert json.loads(p.read_text())['ids']==ids
else:
 p.write_text(json.dumps({'ids':ids,'seed':20260924,'excluded_ids':old,'role':'frozen disjoint development confirmation'},indent=2))
print('CONFIRMATION_MANIFEST',hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
PY
python scripts/research_experiment.py --dataset data/hotpotqa/research_train5.json --ids outputs/research_train5_ids.json --output outputs/research_train5_fixed --arms adaptive
python scripts/run_experiment.py --method fixed_hops --model_name Qwen/Qwen2.5-7B-Instruct --example_ids_file outputs/example_ids_dev_50_seed42.json --hops 3 --top_k 5 --max_new_tokens 192 --temperature 0 --top_p 1 --max_evidence_paragraphs 15 --use_chat_template --run_id research_chat_legacy50 > outputs/research_chat_legacy50.log 2>&1
python scripts/research_experiment.py --ids outputs/example_ids_dev_50_seed42.json --output outputs/research7b_50
python scripts/research_experiment.py --ids outputs/example_ids_dev_50_seed42.json --output outputs/research7b_titlehybrid50 --retrieval hybrid --title-prefix --arms top5
python scripts/research_experiment.py --ids outputs/example_ids_dev_50_seed42.json --output outputs/research7b_bgehybrid50 --retrieval hybrid --title-prefix --embedder BAAI/bge-base-en-v1.5 --arms top5 adaptive
python scripts/research_experiment.py --ids outputs/example_ids_dev_50_seed42.json --output outputs/research14b_50 --model Qwen/Qwen2.5-14B-Instruct
# No configuration is selected/tuned using this confirmation sample.
python scripts/research_experiment.py --ids outputs/research_confirm100_ids.json --output outputs/research7b_confirm100 --arms top5 all_context adaptive
python scripts/research_experiment.py --ids outputs/research_confirm100_ids.json --output outputs/research14b_confirm100 --model Qwen/Qwen2.5-14B-Instruct --arms top5 all_context adaptive
python scripts/export_research_results.py
