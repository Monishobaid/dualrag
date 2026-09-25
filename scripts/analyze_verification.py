#!/usr/bin/env python3
"""Create paired diagnostics for a verification experiment.

The report keeps answer gains and regressions visible instead of reducing a
revision method to one aggregate score. It consumes only saved traces and
predictions; it never retunes a threshold or uses a gold title in routing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_traces(path: Path) -> dict:
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def pair(base: dict, candidate: dict) -> dict:
    base_em = bool(base.get("em"))
    cand_em = bool(candidate.get("em"))
    if cand_em and not base_em:
        outcome = "revision_gain"
    elif base_em and not cand_em:
        outcome = "revision_regression"
    elif cand_em:
        outcome = "both_correct"
    else:
        outcome = "both_wrong"
    return {
        "id": candidate["id"], "outcome": outcome,
        "base_answer": base.get("final_answer", ""),
        "candidate_answer": candidate.get("final_answer", ""),
        "gold_answer": candidate.get("gold_answer", ""),
        "question": candidate.get("question", ""),
        "base_issues": base.get("relation_checks", {}).get("issues", []),
        "candidate_issues": candidate.get("relation_checks", {}).get("issues", []),
        "verification": candidate.get("verification", {}),
        "revision": candidate.get("revision", {}),
        "base_titles": base.get("visible_titles", []),
        "candidate_titles": candidate.get("visible_titles", []),
        "candidate_calls": [c.get("role") for c in candidate.get("calls", [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--base", default="static")
    parser.add_argument("--candidate", default="revised")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.experiment)
    base = load_traces(root / args.base / "traces.jsonl")
    candidate = load_traces(root / args.candidate / "traces.jsonl")
    missing = sorted(set(base) ^ set(candidate))
    if missing:
        raise ValueError(f"Mismatched trace IDs: {missing[:5]}")
    rows = [pair(base[item], candidate[item]) for item in base]
    counts = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    report = {
        "base_arm": args.base, "candidate_arm": args.candidate,
        "num_examples": len(rows), "outcome_counts": counts,
        "revision_attempts": sum(r["revision"].get("attempted", False) for r in rows),
        "revision_accepts": sum(r["revision"].get("accepted", False) for r in rows),
        "rows": rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("base_arm", "candidate_arm", "num_examples", "outcome_counts", "revision_attempts", "revision_accepts")}, indent=2))


if __name__ == "__main__":
    main()

