#!/usr/bin/env python3
"""Build an evidence-backed report from completed exports, without tuning runs."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paired_research_analysis import compare
from scripts.research_experiment import score

ROOT=Path('research_results/2026-09-24')
LABELS={
 'dense_fixed50_e9f673f':'Historical raw-prompt / fixed hops / 7B',
 'dense_reader50_615a16f':'Historical heuristic candidate reader / 7B',
 'research_raw_legacy50':'Matched raw-prompt legacy / fixed hops / 7B',
 'research7b_budget_confirm100':'Confirmation 7B / static budget controls',
 'research_chat_legacy50':'Chat-formatted legacy / fixed hops / 7B',
 'research7b_50':'Corrected 7B / MiniLM',
 'research7b_titlehybrid50':'Corrected 7B / MiniLM + title + hybrid',
 'research7b_bgehybrid50':'Corrected 7B / BGE + title + hybrid',
 'research14b_50':'Corrected 14B / MiniLM',
 'research7b_confirm100':'Confirmation 7B / MiniLM',
 'research14b_confirm100':'Confirmation 14B / MiniLM'}


def read_rows(folder):
    rows=[json.loads(x) for x in (folder/'audit.jsonl').read_text().splitlines()]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate example IDs')
    return {r['id']:r for r in rows}


def main():
    lines=['# Renku research results — 24 September 2026','',
      'All numbers below come from exported completed artifacts. Original50 is exploratory; '
      'confirmation100 was frozen separately before new dev results. These are HotpotQA '
      'distractor experiments, not open-domain or market-wide benchmarks.','',
      '## Completed comparisons','',
      '| Configuration | Reader arm | N | EM | F1 | Output tokens/example | Seconds/example | Hops |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    runs={}
    for name,label in LABELS.items():
        run=ROOT/name
        if not run.exists():continue
        dirs=[run] if (run/'metrics.json').exists() else sorted(x for x in run.iterdir() if x.is_dir())
        for d in dirs:
            if not (d/'metrics.json').exists() or not (d/'audit.jsonl').exists():continue
            m=json.loads((d/'metrics.json').read_text());rows=read_rows(d)
            if len(rows)!=m['num_examples']:raise ValueError(f'Denominator mismatch: {d}')
            em=sum(score(r['final_answer'],r['gold_answer'])['em'] for r in rows.values())/len(rows)
            f1=sum(score(r['final_answer'],r['gold_answer'])['f1'] for r in rows.values())/len(rows)
            arm=d.name if d!=run else 'legacy'
            runs[name+'/'+arm]=rows
            toks=m.get('mean_generated_tokens',m.get('avg_tokens_generated',0));lat=m.get('mean_latency_seconds',m.get('avg_latency_seconds',0));hops=m.get('mean_hops',m.get('avg_hops',0))
            lines.append(f'| {label} | {arm} | {len(rows)} | {em:.1%} | {f1:.2%} | {toks:.1f} | {lat:.2f} | {hops:.2f} |')
    lines+=['','EM/F1 are rescored using the same HotpotQA answer rule, including yes/no/noanswer mismatches. '
       'Output-token counts include all critic calls. Latency excludes model loading, is measured in '
       'sequential warm-process runs, and is not a repeated randomized timing benchmark.','',
       '## Paired differences','',
       'Bootstrap intervals use 10,000 paired resamples, seed42. They do not correct for multiple '
       'comparisons or the repeated inspection of original50. A confidence interval containing zero '
       'does not establish an improvement.','',
       '| Control → candidate | N | Gains / losses | EM change [95% CI] | F1 change [95% CI] |',
       '|---|---:|---:|---|---|']
    pairs=[('dense_fixed50_e9f673f/legacy','dense_reader50_615a16f/legacy'),
           ('dense_fixed50_e9f673f/legacy','research_chat_legacy50/legacy'),
           ('research_raw_legacy50/legacy','research_chat_legacy50/legacy'),
           ('research7b_budget_confirm100/top9','research7b_confirm100/adaptive'),
           ('research7b_budget_confirm100/ranked_all','research7b_confirm100/adaptive'),
           ('research_chat_legacy50/legacy','research7b_50/all_context'),
           ('research7b_titlehybrid50/top5','research7b_bgehybrid50/top5'),
           ('research7b_50/all_context','research14b_50/all_context')]
    for name in ['research7b_50','research14b_50','research7b_confirm100','research14b_confirm100']:
        pairs.extend([(name+'/top5',name+'/adaptive'),(name+'/all_context',name+'/adaptive')])
    comparisons={}
    for a,b in pairs:
        if a not in runs or b not in runs:continue
        result=compare(runs[a],runs[b]);comparisons[a+'__'+b]=result
        def interval(k):
            x=result['effects'][k];lo,hi=x['paired_bootstrap_95_ci']
            return f'{100*x["mean"]:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] pp'
        lines.append(f'| {a} → {b} | {result["n"]} | {result["gains"]} / {result["losses"]} | {interval("em_delta")} | {interval("f1_delta")} |')
    lines+=['','## Critic diagnostics','',
        'Missing annotated support is a proxy, not a human sufficiency label: alternative evidence '
        'may be valid. Likewise, valid citation IDs do not establish entailment.','']
    for name,rows in runs.items():
        if not name.endswith('/adaptive'):continue
        stopped=[r for r in rows.values() if r.get('stop_reason')=='critic_sufficient']
        wrong=[r for r in stopped if score(r['final_answer'],r['gold_answer'])['em']==0]
        incomplete=[r for r in wrong if r.get('visible_sentence_recall',1)<1]
        lines.append(f'- **{name}:** sufficient stops {len(stopped)}/{len(rows)}; '
                     f'{len(wrong)} of those answers fail EM; {len(incomplete)} also lack some annotated supporting sentences.')
    lines+=['','## Format and execution health','',
        '| Run | Failed examples | Calls | Non-strict JSON | Invalid citation IDs | Budget hits |',
        '|---|---:|---:|---:|---:|---:|']
    for name,rows in runs.items():
        calls=[c for r in rows.values() for c in r.get('calls',[])]
        if not calls:continue
        lines.append(f'| {name} | {sum(bool(r.get("error")) for r in rows.values())} | {len(calls)} | '
            f'{sum(not c["strict_json"] for c in calls)} | {sum(not c["citation_ids_valid"] for c in calls)} | '
            f'{sum(c["hit_token_limit"] for c in calls)} |')
    lines+=['','An insufficient-context critic may legitimately return no citations; the invalid-ID '
        'column also counts empty citation lists. See individual calls before classifying a failure.','',
        '## Artifacts and limits','',
        '- Complete original prompts and evidence order remain in Renku `outputs/*/traces.jsonl`.',
        '- Exported `audit.jsonl` files retain answers, calls, model outputs and sentence sources.',
        '- `protocol.json` records model revision, packages, dataset and cohort hashes, and settings.',
        '- A larger generator changes model capacity and cost; its gain is not evidence for the critic.',
        '- Oracle gold paragraphs are an evaluation-only reference, not a mathematical maximum. '
          'Removing distractors or changing order can also change answers.',
        '- The critic is prompted, not learned or calibrated. No train/dev threshold search was performed.',
        '- A claim of novelty or superiority needs additional datasets, open-domain retrieval, '
          'matched compute and stronger external baselines.','']
    Path('docs/RESULTS_2026-09-24.md').write_text('\n'.join(lines))
    (ROOT/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
    print('Wrote report for',len(runs),'completed runs')

if __name__=='__main__':main()
