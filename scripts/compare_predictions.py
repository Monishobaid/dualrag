#!/usr/bin/env python3
"""Compare matched normal and oracle-evidence predictions example by example.

The oracle run uses HotpotQA supporting-fact labels, so it is an evaluation
ceiling rather than a deployable method.  This report separates retrieval
failures (oracle correct, normal wrong) from generation failures (both wrong
or oracle wrong) instead of relying only on aggregate EM/F1.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval.hotpot_eval import evaluate_single


def _load_predictions(path: Path):
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return {str(k): str(v or "") for k, v in payload.items()}
    if isinstance(payload, list):
        return {str(row["id"]): str(row.get("prediction", row.get("answer", ""))) for row in payload}
    raise ValueError(f"Unsupported predictions format: {path}")


def _load_traces(path: Path):
    traces = {}
    if not path.exists():
        return traces
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("id") is not None:
                traces[str(row["id"])] = row
    return traces


def build_report(normal_dir: Path, oracle_dir: Path):
    normal = _load_predictions(normal_dir / "predictions.json")
    oracle = _load_predictions(oracle_dir / "predictions.json")
    normal_traces = _load_traces(normal_dir / "traces.jsonl")
    oracle_traces = _load_traces(oracle_dir / "traces.jsonl")
    ids = sorted(set(normal) | set(oracle))
    rows = []
    for example_id in ids:
        n_answer = normal.get(example_id, "")
        o_answer = oracle.get(example_id, "")
        n_trace = normal_traces.get(example_id, {})
        o_trace = oracle_traces.get(example_id, {})
        gold = n_trace.get("gold_answer", o_trace.get("gold_answer", ""))
        n_eval = evaluate_single(n_answer, gold)
        o_eval = evaluate_single(o_answer, gold)
        rows.append({
            "id": example_id,
            "question": n_trace.get("question", o_trace.get("question", "")),
            "gold_answer": gold,
            "normal_answer": n_answer,
            "oracle_answer": o_answer,
            "normal_em": int(n_eval["em"]),
            "oracle_em": int(o_eval["em"]),
            "normal_f1": n_eval["f1"],
            "oracle_f1": o_eval["f1"],
            "answer_changed": n_answer.strip() != o_answer.strip(),
            "category": (
                "retrieval_rescued" if not n_eval["em"] and o_eval["em"]
                else "oracle_regressed" if n_eval["em"] and not o_eval["em"]
                else "both_correct" if n_eval["em"] and o_eval["em"]
                else "both_wrong"
            ),
            "normal_supporting_title_recall": n_trace.get("supporting_title_recall"),
            "normal_hops": n_trace.get("hops_taken"),
            "oracle_hops": o_trace.get("hops_taken"),
        })
    summary = {
        "normal_examples": len(normal),
        "oracle_examples": len(oracle),
        "matched_examples": len(rows),
        "normal_em": sum(r["normal_em"] for r in rows) / len(rows) if rows else 0.0,
        "oracle_em": sum(r["oracle_em"] for r in rows) / len(rows) if rows else 0.0,
        "normal_f1": sum(r["normal_f1"] for r in rows) / len(rows) if rows else 0.0,
        "oracle_f1": sum(r["oracle_f1"] for r in rows) / len(rows) if rows else 0.0,
        "categories": {},
    }
    for row in rows:
        summary["categories"][row["category"]] = summary["categories"].get(row["category"], 0) + 1
    return {"summary": summary, "examples": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_dir", required=True, type=Path)
    parser.add_argument("--oracle_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.normal_dir, args.oracle_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    md_path = args.output.with_suffix(".md")
    s = report["summary"]
    lines = [
        "# Normal versus oracle-evidence predictions",
        "",
        "The oracle run uses gold supporting-fact titles and is an evaluation-only ceiling.",
        "",
        f"- Matched examples: {s['matched_examples']}",
        f"- Normal EM/F1: {s['normal_em']:.4f} / {s['normal_f1']:.4f}",
        f"- Oracle EM/F1: {s['oracle_em']:.4f} / {s['oracle_f1']:.4f}",
        f"- Categories: {json.dumps(s['categories'], sort_keys=True)}",
        "",
        "| ID | Normal | Oracle | Gold | Category |",
        "|---|---|---|---|---|",
    ]
    for row in report["examples"]:
        def cell(value):
            return str(value or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {cell(row['id'])} | {cell(row['normal_answer'])} | "
            f"{cell(row['oracle_answer'])} | {cell(row['gold_answer'])} | {row['category']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(s, indent=2))
    print(f"Wrote {args.output} and {md_path}")


if __name__ == "__main__":
    main()
