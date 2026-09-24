#!/usr/bin/env python3
"""Build an evidence-backed report from completed exports, without tuning runs."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paired_research_analysis import compare
from scripts.research_experiment import score
from src.rag.sufficiency import citation_ids, parse_object

ROOT=Path('research_results/2026-09-24')
LABELS={
 'dense_fixed50_e9f673f':'Historical raw-prompt / fixed hops / 7B',
 'hybrid_fixed50_e9f673f':'Historical hybrid / fixed hops / 7B',
 'hybrid_rerank_fixed50_e9f673f':'Historical hybrid + reranker / fixed hops / 7B',
 'oracle_evidence50_e9f673f':'Historical oracle / 7B',
 'dense_reader50_615a16f':'Historical heuristic candidate reader / 7B',
 'research_raw_legacy50':'Matched raw-prompt legacy / fixed hops / 7B',
 'research7b_budget_confirm100':'Confirmation 7B / static budget controls',
 'research14b_budget_confirm100':'Post-hoc 14B / static budget controls on confirmation100',
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
      'distractor experiments, not open-domain or market-wide benchmarks. The 14B top9/ranked-all '
      'controls were added after observing its confirmation result and are explicitly post hoc.','',
      '## Completed comparisons','',
      '| Configuration | Reader arm | N | EM | F1 | Output tokens/example | Input tokens/example | Seconds/example | Hops |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    runs={}; metrics={}
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
            if arm != 'legacy' and (abs(em-m['em']) > 1e-10 or abs(f1-m['f1']) > 1e-10):
                raise ValueError(f'Exported metrics disagree with rescored predictions: {d}')
            runs[name+'/'+arm]=rows
            metrics[name+'/'+arm]=m
            toks=m.get('mean_generated_tokens',m.get('avg_tokens_generated',0));lat=m.get('mean_latency_seconds',m.get('avg_latency_seconds',0));hops=m.get('mean_hops',m.get('avg_hops',0))
            calls=[c for r in rows.values() for c in r.get('calls',[])]
            legacy_meta=[h.get('generation_metadata') for r in rows.values() for h in r.get('hops',[])]
            inputs=calls or (legacy_meta if legacy_meta and all(m and 'input_tokens' in m for m in legacy_meta) else [])
            input_tokens=f'{sum(c["input_tokens"] for c in inputs)/len(rows):.1f}' if inputs else 'unavailable'
            lines.append(f'| {label} | {arm} | {len(rows)} | {em:.1%} | {f1:.2%} | {toks:.1f} | {input_tokens} | {lat:.2f} | {hops:.2f} |')
    lines+=['','EM/F1 are rescored using the same HotpotQA answer rule, including yes/no/noanswer mismatches. '
       'Input/output-token counts include all critic calls, including rereading accumulated evidence. Latency excludes model loading, is measured in '
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
           ('research14b_budget_confirm100/top9','research14b_confirm100/adaptive'),
           ('research14b_budget_confirm100/ranked_all','research14b_confirm100/adaptive'),
           ('research7b_confirm100/all_context','research7b_budget_confirm100/ranked_all'),
           ('research14b_confirm100/all_context','research14b_budget_confirm100/ranked_all'),
           ('research7b_budget_confirm100/ranked_all','research14b_budget_confirm100/ranked_all'),
           ('research_chat_legacy50/legacy','research7b_50/all_context'),
           ('research7b_titlehybrid50/top5','research7b_bgehybrid50/top5'),
           ('research7b_50/all_context','research14b_50/all_context'),
           ('research7b_confirm100/all_context','research14b_confirm100/all_context'),
           ('research7b_confirm100/adaptive','research14b_confirm100/adaptive')]
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
    lines+=['','## Evidence and stopping','',
        '| Run | Initial title recall@5 | Visible support-sentence recall | Cited support-fact recall | Call parse failures | Stop reasons |',
        '|---|---:|---:|---:|---:|---|']
    for name,m in metrics.items():
        if 'visible_supporting_sentence_recall' not in m:continue
        recall=m['supporting_title_recall_at_5']
        initial='n/a' if recall is None else f'{recall:.1%}'
        reasons=', '.join(f'{k}: {v}' for k,v in m['stop_reasons'].items())
        lines.append(f'| {name} | {initial} | {m["visible_supporting_sentence_recall"]:.1%} | '
                     f'{m["citation_supporting_fact_recall"]:.1%} | {m["parse_failure_rate"]:.1%} | {reasons} |')
    lines+=['','Initial title recall uses five retrieved paragraphs. Visible recall measures the annotated '
        'sentences actually sent to the final reader. Cited recall measures which annotated facts the '
        'answer cites; none of these alone verifies logical entailment.','',
        '## Critic diagnostics','',
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
        '| Run | Failed examples | Calls | Non-strict JSON | Schema errors | Empty citations | Invalid nonempty IDs | Budget hits |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,rows in runs.items():
        calls=[c for r in rows.values() for c in r.get('calls',[])]
        if not calls:continue
        def schema_error(c):
            o=c.get('parsed') or {}; ids=o.get('evidence_ids')
            valid=isinstance(ids,list) and all(isinstance(x,str) for x in ids)
            if c['role']=='reader':
                return not (valid and isinstance(o.get('answer'),str) and isinstance(o.get('explanation'),str))
            return not (valid and type(o.get('sufficient')) is bool and isinstance(o.get('query'),str)
                        and isinstance(o.get('missing_information'),str))
        lines.append(f'| {name} | {sum(bool(r.get("error")) for r in rows.values())} | {len(calls)} | '
            f'{sum(not c["strict_json"] for c in calls)} | {sum(schema_error(c) for c in calls)} | '
            f'{sum(not citation_ids(c.get("parsed")) for c in calls)} | '
            f'{sum(bool(citation_ids(c.get("parsed"))) and not c["citation_ids_valid"] for c in calls)} | '
            f'{sum(c["hit_token_limit"] for c in calls)} |')
    lines+=['','An insufficient-context critic may legitimately return no citations. Empty lists and invalid '
        'nonempty IDs are therefore separated. Schema checks also catch a string or integer list where '
        'a list of string IDs was requested. These are reporting diagnostics; no predictions are repaired.','',
        '### Matched legacy raw-output inspection','',
        '| Formatting | Hop outputs | Strict JSON objects | Recovered complete objects | No complete object | Budget hits |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ['research_raw_legacy50/legacy','research_chat_legacy50/legacy']:
        if name not in runs:continue
        metadata=[h['generation_metadata'] for r in runs[name].values() for h in r['hops']]
        objects=[parse_object(c['raw_output']) for c in metadata]
        strict=sum(o is not None and s for o,s in objects)
        recovered=sum(o is not None and not s for o,s in objects)
        missing=sum(o is None for o,s in objects)
        lines.append(f'| {name} | {len(metadata)} | {strict} | {recovered} | {missing} | '
                     f'{sum(c["hit_token_limit"] for c in metadata)} |')
    lines+=['','A legacy regex fallback can extract an answer field even when no complete JSON object '
        'is present. Its historical reported parse-failure metric therefore measures a different '
        'event from strict object validity. Predictions are unchanged by this inspection.','',
        '## Artifacts and limits','',
        '- Complete original prompts and evidence order remain in Renku `outputs/*/traces.jsonl`.',
        '- Exported `audit.jsonl` files retain answers, calls, model outputs and sentence sources.',
        '- `protocol.json` records model revision, packages, dataset and cohort hashes, and settings.',
        '- A larger generator changes model capacity and cost; its gain is not evidence for the critic.',
        '- Adaptive arms use the same checkpoint for critic and reader. Switching 7B to 14B changes both roles, '
          'so that comparison does not isolate reader capacity from controller behavior.',
        '- Oracle gold paragraphs are an evaluation-only reference, not a mathematical maximum. '
          'Removing distractors or changing order can also change answers.',
        '- The critic is prompted, not learned or calibrated. No train/dev threshold search was performed.',
        '- A claim of novelty or superiority needs additional datasets, open-domain retrieval, '
          'matched compute and stronger external baselines.','']
    Path('docs/RESULTS_2026-09-24.md').write_text('\n'.join(lines))
    (ROOT/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
    failures={name:[{'id':r['id'],'question':r['question'],'gold':r['gold_answer'],
        'answer':r['final_answer'],**score(r['final_answer'],r['gold_answer']),
        'stop_reason':r.get('stop_reason'),'visible_sentence_recall':r.get('visible_sentence_recall'),
        'execution_error':r.get('error')} for r in rows.values()
        if not score(r['final_answer'],r['gold_answer'])['em'] or r.get('error')]
        for name,rows in runs.items()}
    (ROOT/'failed_predictions.json').write_text(json.dumps(failures,indent=2))
    print('Wrote report for',len(runs),'completed runs')

if __name__=='__main__':main()
