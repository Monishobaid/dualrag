#!/usr/bin/env python3
"""Run matched relation-aware answer verification arms.

This is an isolated experiment runner. The historical stopping runner is left
unchanged so its results remain reproducible. Use a frozen IDs manifest and a
new output directory for every run; the script refuses to overwrite artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.hotpot_loader import HotpotQALoader
from src.rag.claim_verifier import VERIFIER_SYSTEM, parse_verification
from src.rag.relation_checks import run_relation_checks
from src.rag.revision import REVISION_SYSTEM, revision_prompt
from src.rag.structured_answer import STRUCTURED_READER_SYSTEM, parse_structured, validate_structured
from src.rag.sufficiency import READER_SYSTEM, evidence_view, parse_object, short_answer
from src.utils.text import compute_exact_match, compute_f1, normalize_text


def score(pred: str, gold: str) -> dict:
    p, g = normalize_text(pred), normalize_text(gold)
    f1 = compute_f1(pred, gold)
    if p != g and ({p, g} & {"yes", "no", "noanswer"}):
        f1 = 0.0
    return {"em": compute_exact_match(pred, gold), "f1": f1}


def load_ids(path: str) -> list[str]:
    obj = json.loads(Path(path).read_text())
    ids = obj.get("ids") if isinstance(obj, dict) else obj
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("Manifest must contain unique, non-empty IDs")
    return ids


def call_json(llm, question: str, paragraphs: list[dict], system: str,
              role: str, budget: int, parser):
    context, sources = evidence_view(paragraphs)
    prompt = f"EVIDENCE:\n{context}\n\nQUESTION: {question}"
    started = time.perf_counter()
    raw, tokens = llm.generate_with_token_count(
        prompt, system_message=system, max_new_tokens=budget, temperature=0, top_p=1
    )
    obj, strict = parser(raw)
    call = {
        "role": role, "prompt": prompt, "system": system, "raw_output": raw,
        "parsed": obj, "strict_json": strict, "parse_failed": obj is None,
        "generated_tokens": tokens, "latency_seconds": time.perf_counter() - started,
        **llm.last_generation,
    }
    return obj, sources, context, call


def _as_structured(obj: dict | None, question: str) -> dict:
    """Add conservative fields for the unchanged reader arm."""
    obj = dict(obj or {})
    answer = obj.get("answer", "") if isinstance(obj.get("answer"), str) else ""
    obj.setdefault("answer_type", "other")
    obj.setdefault("operation", "other")
    obj.setdefault("claims", [{"text": answer, "evidence_ids": obj.get("evidence_ids", [])}]
                    if answer else [])
    obj.setdefault("explanation", "")
    return obj


def build_evidence(ex, arm: str, ranker) -> list[dict]:
    paragraphs = [dict(p, _pid=i) for i, p in enumerate(ex.candidate_paragraphs)]
    if arm == "oracle":
        gold_titles = {x[0] for x in ex.supporting_facts}
        return [p for p in paragraphs if p["title"] in gold_titles]
    ranked = ranker.rank(ex.question, paragraphs, top_k=len(paragraphs))
    return [x["paragraph"] for x in ranked]


def run_one(ex, arm: str, ranker, llm, max_new_tokens: int) -> dict:
    evidence = build_evidence(ex, arm, ranker)
    calls = []
    started = time.perf_counter()
    row = {"id": ex.id, "question": ex.question, "gold_answer": ex.answer,
           "arm": arm, "visible_titles": [p["title"] for p in evidence]}
    try:
        if arm in {"static", "oracle"}:
            obj, sources, context, call = call_json(
                llm, ex.question, evidence, READER_SYSTEM, "reader", max_new_tokens,
                parse_object
            )
            calls.append(call)
            structured = _as_structured(obj, ex.question)
        else:
            structured, sources, context, call = call_json(
                llm, ex.question, evidence, STRUCTURED_READER_SYSTEM, "structured_reader",
                max_new_tokens, parse_structured
            )
            calls.append(call)
            structured = structured or {}

        validation = validate_structured(structured, sources)
        checks = run_relation_checks(ex.question, structured, sources)
        verification = {"claims": [], "parse_failed": False, "any_unsupported": False,
                        "any_uncertain": False, "mode": "none"}
        revision = {"attempted": False, "accepted": False, "reason": "not_requested"}

        if arm in {"verified", "revised"}:
            claims = structured.get("claims") if isinstance(structured.get("claims"), list) else []
            verifier_obj, _, _, vcall = call_json(
                llm, ex.question,
                evidence,
                VERIFIER_SYSTEM,
                "verifier",
                max_new_tokens,
                lambda raw: (parse_verification(raw, len(claims), sources), True),
            )
            calls.append(vcall)
            verification = verifier_obj or verification
            verification["mode"] = "prompted"

        if arm == "revised":
            should_revise = bool(checks.get("issues")) or verification.get("any_unsupported", False)
            revision["attempted"] = should_revise
            revision["reason"] = "checks_or_verifier_flag" if should_revise else "no_flag"
            if should_revise:
                prompt = revision_prompt(ex.question, structured, checks, verification, context)
                rev_started = time.perf_counter()
                raw, tokens = llm.generate_with_token_count(
                    prompt, system_message=REVISION_SYSTEM, max_new_tokens=max_new_tokens,
                    temperature=0, top_p=1
                )
                revised, strict = parse_structured(raw)
                rev_call = {
                    "role": "revision", "prompt": prompt, "system": REVISION_SYSTEM,
                    "raw_output": raw, "parsed": revised, "strict_json": strict,
                    "parse_failed": revised is None, "generated_tokens": tokens,
                    "latency_seconds": time.perf_counter() - rev_started,
                    **llm.last_generation,
                }
                calls.append(rev_call)
                if revised and isinstance(revised.get("answer"), str):
                    structured = revised
                    validation = validate_structured(structured, sources)
                    checks = run_relation_checks(ex.question, structured, sources)
                    revision["accepted"] = True
                    revision["reason"] = "revision_returned_structured_answer"

        final_answer = short_answer(structured)
        row.update({
            "final_answer": final_answer, "structured_answer": structured,
            "validation": validation, "relation_checks": checks,
            "verification": verification, "revision": revision,
            "sources": sources, "calls": calls,
            "num_hops": 1, "stop_reason": "single_pass_verification",
        })
        row.update(score(final_answer, ex.answer))
        gold = {(title, index) for title, index in ex.supporting_facts}
        visible = {(x["title"], x["sentence_index"]) for x in sources.values()}
        cited_ids = []
        for claim in (structured.get("claims", []) if isinstance(structured, dict) else []):
            if isinstance(claim, dict):
                cited_ids.extend(claim.get("evidence_ids", []) or [])
        cited = {(sources[sid]["title"], sources[sid]["sentence_index"])
                 for sid in cited_ids if sid in sources}
        row["visible_supporting_sentence_recall"] = len(gold & visible) / max(1, len(gold))
        row["citation_supporting_fact_recall"] = len(gold & cited) / max(1, len(gold))
    except Exception as exc:
        row.update({"error": repr(exc), "final_answer": "", "em": 0.0, "f1": 0.0,
                    "calls": calls, "relation_checks": {"issues": ["execution_error"]},
                    "verification": {}, "revision": {"attempted": False}})
    row["generated_tokens"] = sum(c.get("generated_tokens", 0) for c in row.get("calls", []))
    row["latency_seconds"] = time.perf_counter() - started
    return row


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    calls = [c for row in rows for c in row.get("calls", [])]
    issue_counts = Counter(x for row in rows for x in row.get("relation_checks", {}).get("issues", []))
    return {
        "num_examples": len(rows), "num_successful": sum("error" not in r for r in rows),
        "em": sum(r.get("em", 0) for r in rows) / n,
        "f1": sum(r.get("f1", 0) for r in rows) / n,
        "mean_hops": sum(r.get("num_hops", 0) for r in rows) / n,
        "mean_generated_tokens": sum(r.get("generated_tokens", 0) for r in rows) / n,
        "mean_latency_seconds": sum(r.get("latency_seconds", 0) for r in rows) / n,
        "visible_supporting_sentence_recall": sum(r.get("visible_supporting_sentence_recall", 0) for r in rows) / n,
        "citation_supporting_fact_recall": sum(r.get("citation_supporting_fact_recall", 0) for r in rows) / n,
        "parse_failure_rate": sum(c.get("parse_failed", False) for c in calls) / max(1, len(calls)),
        "schema_invalid_rate": sum(not r.get("validation", {}).get("valid", False) for r in rows if "validation" in r) / max(1, sum("validation" in r for r in rows)),
        "unsupported_claim_rate": sum(bool(r.get("verification", {}).get("any_unsupported")) for r in rows) / n,
        "revision_attempt_rate": sum(bool(r.get("revision", {}).get("attempted")) for r in rows) / n,
        "revision_accept_rate": sum(bool(r.get("revision", {}).get("accepted")) for r in rows) / n,
        "relation_issue_counts": dict(issue_counts),
        "failed_examples": [r["id"] for r in rows if r.get("error")],
        "num_model_calls": len(calls),
        "stop_reasons": dict(Counter(r.get("stop_reason", "error") for r in rows)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    p.add_argument("--ids", required=True, help="Frozen JSON list or {ids: [...]} manifest")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--retrieval", choices=["dense", "bm25", "hybrid"], default="dense")
    p.add_argument("--dense-weight", type=float, default=0.5)
    p.add_argument("--reranker", default=None)
    p.add_argument("--arms", nargs="+", choices=["static", "structured", "checked", "verified", "revised", "oracle"],
                   default=["static", "structured", "checked", "verified", "revised", "oracle"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=192)
    args = p.parse_args()

    import numpy as np
    import torch
    from src.llm.qwen import QwenModel
    from src.retrieval.embedder import Embedder
    from src.retrieval.index import EmbeddingCache
    from src.retrieval.ranker import ParagraphRanker

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    ids = load_ids(args.ids)
    loader = HotpotQALoader(args.dataset, example_ids=ids)

    def seed_all() -> None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    seed_all()
    llm = QwenModel(args.model, use_chat_template=True, max_input_tokens=8192)
    embedder = Embedder(args.embedder)
    cache = EmbeddingCache(namespace=f"verification|{args.embedder}")
    ranker = ParagraphRanker(embedder, cache, args.retrieval, dense_weight=args.dense_weight,
                             reranker_model_name=args.reranker)
    protocol = {
        **vars(args), "ids": ids,
        "id_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
        "model_revision": getattr(llm.model.config, "_commit_hash", None),
        "packages": {x: importlib.metadata.version(x) for x in ["torch", "transformers", "sentence-transformers", "numpy"]},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "experiment": "relation-aware structured claims, deterministic checks, prompted verification, one bounded revision",
        "gold_use": "oracle arm only; no gold labels enter deployable arms",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    summary = {}
    for arm in args.arms:
        seed_all()
        arm_dir = out / arm
        arm_dir.mkdir()
        rows = []
        with (arm_dir / "traces.jsonl").open("w") as trace_file:
            for index, ex in enumerate(loader, 1):
                row = run_one(ex, arm, ranker, llm, args.max_new_tokens)
                rows.append(row)
                trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                trace_file.flush()
                print(f"{arm} {index}/{len(loader)} EM={row.get('em', 0):.0f} F1={row.get('f1', 0):.3f} {row.get('error', '')}", flush=True)
        metrics = summarize(rows)
        summary[arm] = metrics
        (arm_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (arm_dir / "predictions.json").write_text(json.dumps({r["id"]: r.get("final_answer", "") for r in rows}, indent=2))
        print(arm, json.dumps(metrics), flush=True)
    cache.save()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
