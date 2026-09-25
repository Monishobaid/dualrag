"""Structured answer fields for the relation-aware verification experiment.

The schema adapts the atomic-claim idea from FActScore and the grounded
revision idea from RARR/Re-Ex to the local HotpotQA evidence pool. It is an
observable, concise representation; it is deliberately not a request for
hidden chain-of-thought.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.rag.sufficiency import citation_ids, parse_object


STRUCTURED_READER_SYSTEM = (
    "Answer the question only from the supplied evidence. Return exactly one "
    "JSON object with keys answer, answer_type, operation, claims, and "
    "explanation. answer is short and contains no explanation. answer_type "
    "must be entity, date, number, yes_no, list, or other. operation must be "
    "lookup, comparison, intersection, count, boolean, or other. claims is "
    "a short list of factual statements, each with evidence_ids using the "
    "printed sentence IDs. Keep the explanation to two short sentences. If "
    "the evidence is insufficient, use an empty answer and say so in the "
    "explanation. Do not use outside knowledge."
)


def parse_structured(raw: str) -> Tuple[Dict[str, Any] | None, bool]:
    """Parse a structured object, preserving strict/non-strict status."""
    obj, strict = parse_object(raw)
    if not isinstance(obj, dict):
        return None, strict
    return obj, strict


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_claims(obj: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    """Normalize claim records without inventing missing evidence."""
    if not obj or not isinstance(obj.get("claims"), list):
        return []
    claims: List[Dict[str, Any]] = []
    for claim in obj["claims"]:
        if not isinstance(claim, dict):
            continue
        ids = claim.get("evidence_ids", [])
        if not isinstance(ids, list):
            ids = []
        claims.append({
            "text": _clean_text(claim.get("text")),
            "evidence_ids": [x for x in ids if isinstance(x, str)],
        })
    return claims


def validate_structured(obj: Dict[str, Any] | None, sources: Dict[str, Any]) -> Dict[str, Any]:
    """Return structural errors; support/entailment is checked elsewhere."""
    errors: List[str] = []
    if not obj:
        errors.append("parse_failed")
        return {"valid": False, "errors": errors, "claims": []}
    answer = _clean_text(obj.get("answer"))
    if not answer:
        errors.append("empty_answer")
    allowed_types = {"entity", "date", "number", "yes_no", "list", "other"}
    allowed_ops = {"lookup", "comparison", "intersection", "count", "boolean", "other"}
    if obj.get("answer_type") not in allowed_types:
        errors.append("invalid_answer_type")
    if obj.get("operation") not in allowed_ops:
        errors.append("invalid_operation")
    claims = normalize_claims(obj)
    if not claims:
        errors.append("no_claims")
    cited = set(citation_ids({"evidence_ids": [sid for c in claims for sid in c["evidence_ids"]]}))
    invalid = sorted(cited - set(sources))
    if invalid:
        errors.append("invalid_evidence_ids")
    return {"valid": not errors, "errors": errors, "claims": claims,
            "answer": answer, "answer_type": obj.get("answer_type"),
            "operation": obj.get("operation"), "cited_ids": sorted(cited)}

