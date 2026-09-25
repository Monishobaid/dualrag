"""Claim verifier adapters for bounded grounded correction.

The prompted adapter is a comparison baseline, not a trained Stop-RAG
controller. The optional NLI adapter follows the independent-checker idea in
Provenance and must be calibrated on train before routing test revisions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from src.rag.sufficiency import parse_object, citation_ids


VERIFIER_SYSTEM = (
    "You are an evidence verifier. For each claim, decide supported, "
    "unsupported, or uncertain using only the supplied sentences. Check the "
    "requested relation and answer type, not just word overlap. Return one "
    "JSON object with key claims, a list of objects containing claim_index, "
    "label, evidence_ids, confidence, and error. Use no outside knowledge."
)


def verification_prompt(question: str, answer: Dict[str, Any], context: str) -> str:
    return (
        f"EVIDENCE:\n{context}\n\nQUESTION: {question}\n\n"
        f"CANDIDATE_JSON:\n{json.dumps(answer, ensure_ascii=False)}\n\n"
        "Verify every candidate claim and preserve claim order."
    )


def parse_verification(raw: str, num_claims: int, sources: Dict[str, Any]) -> Dict[str, Any]:
    obj, strict = parse_object(raw)
    records: List[Dict[str, Any]] = []
    if isinstance(obj, dict) and isinstance(obj.get("claims"), list):
        for item in obj["claims"]:
            if not isinstance(item, dict):
                continue
            idx = item.get("claim_index")
            if not isinstance(idx, int):
                continue
            ids = citation_ids({"evidence_ids": item.get("evidence_ids", [])})
            label = item.get("label") if item.get("label") in {"supported", "unsupported", "uncertain"} else "uncertain"
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            records.append({"claim_index": idx, "label": label,
                            "evidence_ids": ids, "valid_evidence_ids": [x for x in ids if x in sources],
                            "confidence": max(0.0, min(1.0, confidence)),
                            "error": str(item.get("error", ""))[:120]})
    by_index = {x["claim_index"]: x for x in records}
    normalized = [by_index.get(i, {"claim_index": i, "label": "uncertain", "evidence_ids": [],
                                    "valid_evidence_ids": [], "confidence": 0.0,
                                    "error": "missing_verifier_record"}) for i in range(num_claims)]
    return {"claims": normalized, "strict_json": strict, "parse_failed": not isinstance(obj, dict),
            "any_unsupported": any(x["label"] == "unsupported" for x in normalized),
            "any_uncertain": any(x["label"] == "uncertain" for x in normalized)}

