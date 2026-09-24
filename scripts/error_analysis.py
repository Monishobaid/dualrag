#!/usr/bin/env python3
"""Create a conservative failure taxonomy for a matched normal/oracle run.

The labels are diagnostic heuristics, not gold annotations. Each example can
have several flags, while ``primary_category`` uses the first actionable cause.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.hotpot_loader import HotpotQALoader
from src.eval.hotpot_eval import evaluate_single
from src.rag.reader import GroundedAnswerVerifier


def _traces(path: Path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("id") is not None:
                    out[str(row["id"])] = row
    return out


def _predictions(path: Path):
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return {str(k): str(v or "") for k, v in payload.items()}
    return {str(row["id"]): str(row.get("prediction", row.get("answer", ""))) for row in payload}


def analyze(normal_dir: Path, oracle_dir: Path, dataset_path: Path, ids_file=None):
    normal = _predictions(normal_dir / "predictions.json")
    oracle = _predictions(oracle_dir / "predictions.json")
    ntrace = _traces(normal_dir / "traces.jsonl")
    otrace = _traces(oracle_dir / "traces.jsonl")
    ids = None
    if ids_file:
        payload = json.loads(ids_file.read_text())
        ids = payload.get("ids", payload) if isinstance(payload, dict) else payload
    loader = HotpotQALoader(str(dataset_path), example_ids=ids) if ids else HotpotQALoader(str(dataset_path))
    questions = {ex.id: ex.question for ex in loader}
    verifier = GroundedAnswerVerifier()
    rows = []
    for example_id in sorted(set(normal) | set(oracle)):
        nt = ntrace.get(example_id, {})
        ot = otrace.get(example_id, {})
        gold = nt.get("gold_answer", ot.get("gold_answer", ""))
        n_answer = normal.get(example_id, "")
        o_answer = oracle.get(example_id, "")
        ne = evaluate_single(n_answer, gold)
        oe = evaluate_single(o_answer, gold)
        if ne["em"]:
            continue

        question = nt.get("question", questions.get(example_id, ""))
        hops = nt.get("hops", [])
        flags = []
        parse_issue = any(bool(h.get("parse_failed")) for h in hops)
        if parse_issue:
            flags.append("parser_issue")

        recall = nt.get("supporting_title_recall")
        first_hop_recall = nt.get("supporting_fact_recall_at_k")
        if first_hop_recall is not None and float(first_hop_recall) < 1.0:
            flags.append("missed_evidence")

        verification = verifier.verify(
            question=question,
            candidate=n_answer,
            evidence=[
                {"title": p.get("title", ""), "text": p.get("text_snippet", "")}
                for h in hops for p in h.get("retrieved_paragraphs", [])
            ],
        )
        if verification["type_score"] == 0.0:
            flags.append("wrong_answer_type")

        grounding = [float(h.get("grounding_score", 0.0)) for h in hops if h.get("answer")]
        if grounding and max(grounding) == 0.0:
            flags.append("hallucination_or_unsupported")

        if not flags and recall is not None and float(recall) >= 1.0:
            flags.append("incomplete_multihop_reasoning")
        if not flags:
            flags.append("other_generation_error")

        priority = [
            "parser_issue", "missed_evidence", "wrong_answer_type",
            "hallucination_or_unsupported", "incomplete_multihop_reasoning",
            "other_generation_error",
        ]
        primary = next(label for label in priority if label in flags)
        rows.append({
            "id": example_id,
            "question": question,
            "gold_answer": gold,
            "normal_answer": n_answer,
            "oracle_answer": o_answer,
            "normal_em": ne["em"],
            "normal_f1": ne["f1"],
            "oracle_em": oe["em"],
            "oracle_f1": oe["f1"],
            "oracle_rescued": bool(not ne["em"] and oe["em"]),
            "flags": flags,
            "primary_category": primary,
            "supporting_title_recall": recall,
            "supporting_fact_recall_at_k": first_hop_recall,
            "hops": nt.get("hops_taken"),
            "parse_issue": parse_issue,
            "max_hop_grounding_score": max(grounding) if grounding else None,
            "type_check": verification,
        })

    counts = {}
    for row in rows:
        counts[row["primary_category"]] = counts.get(row["primary_category"], 0) + 1
    return {"summary": {"failed_examples": len(rows), "primary_categories": counts}, "examples": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--normal_dir", type=Path, required=True)
    p.add_argument("--oracle_dir", type=Path, required=True)
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--ids_file", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = analyze(args.normal_dir, args.oracle_dir, args.dataset_path, args.ids_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    md = ["# Frozen-run error analysis", "", "These categories are trace-based heuristics, not gold annotations.", "", f"- Failed examples: {report['summary']['failed_examples']}", f"- Primary categories: {json.dumps(report['summary']['primary_categories'], sort_keys=True)}", "", "| ID | Normal | Oracle | Gold | Category | Flags |", "|---|---|---|---|---|---|"]
    for row in report["examples"]:
        cell = lambda x: str(x or "").replace("|", "\\|").replace("\n", " ")
        md.append(f"| {row['id']} | {cell(row['normal_answer'])} | {cell(row['oracle_answer'])} | {cell(row['gold_answer'])} | {row['primary_category']} | {', '.join(row['flags'])} |")
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
