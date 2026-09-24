#!/usr/bin/env python3
"""Copy compact, auditable results into the repo; full prompts remain on Renku."""
import json
from pathlib import Path
import shutil
import subprocess
root=Path('research_results/2026-09-24');root.mkdir(parents=True,exist_ok=True)
historical = ['dense_fixed50_e9f673f','hybrid_fixed50_e9f673f','hybrid_rerank_fixed50_e9f673f','oracle_evidence50_e9f673f','dense_reader50_615a16f']
for run in sorted(Path('outputs').glob('research*')) + [Path('outputs')/name for name in historical]:
    if not run.is_dir(): continue
    dest=root/run.name;dest.mkdir(exist_ok=True)
    for name in ['protocol.json','summary.json','config.json','metrics.json','predictions.json']:
        if (run/name).is_file(): shutil.copyfile(run/name,dest/name)
    for arm in [run]+[x for x in run.iterdir() if x.is_dir()]:
        if not (arm/'traces.jsonl').exists() or not (arm/'metrics.json').exists(): continue
        d=dest if arm==run else dest/arm.name;d.mkdir(exist_ok=True)
        for name in ['metrics.json','predictions.json']:
            if (arm/name).exists():shutil.copyfile(arm/name,d/name)
        with (d/'audit.jsonl').open('w') as out:
            for line in (arm/'traces.jsonl').read_text().splitlines():
                r=json.loads(line)
                for c in r.get('calls',[]):
                    c.pop('prompt',None);c.pop('system',None)
                # Sentence sources are retained for error inspection. The full
                # formatted input is recoverable from the original cluster trace.
                out.write(json.dumps(r)+'\n')
for name in ['research_confirm100_ids.json','example_ids_dev_50_seed42.json','research_train5_ids.json']:
    if (Path('outputs')/name).exists():shutil.copyfile(Path('outputs')/name,root/name)
print('EXPORTED',root)
