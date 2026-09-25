"""Deterministic answer/relation checks for the verification experiment.

The checks operationalize the relation and answer-type failure patterns found
in the manual review. They are inspired by FActScore-style atomic checking and
Provenance-style source tracing, but use no gold answers or gold supporting
titles and are intentionally reported as diagnostics rather than an oracle.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from src.rag.sufficiency import citation_ids


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "to", "was",
    "what", "when", "where", "which", "who", "with", "were", "does",
    "did", "than", "that", "this", "these", "those", "their", "they",
}


def tokens(text: str) -> List[str]:
    return [x for x in re.findall(r"[a-z0-9]+", (text or "").lower()) if x not in STOPWORDS]


def infer_question_type(question: str) -> str:
    q = (question or "").strip().lower()
    if q.startswith(("is ", "are ", "was ", "were ", "does ", "did ", "can ", "has ", "have ")):
        return "yes_no"
    if q.startswith(("how many", "how much")) or "number" in q or "population" in q:
        return "number"
    if any(x in q for x in ("what year", "when did", "when was", "what date")):
        return "date"
    if any(x in q for x in ("both", "common to", "in each", "intersection")):
        return "list"
    return "entity"


def infer_operation(question: str) -> str:
    q = (question or "").lower()
    if any(x in q for x in ("both", "common", "shared", "in each")):
        return "intersection"
    if any(x in q for x in ("more", "less", "earlier", "later", "older", "younger", "larger", "smaller", "between")):
        return "comparison"
    if q.startswith(("how many", "how much")) or "number of" in q:
        return "count"
    if q.startswith(("is ", "are ", "was ", "were ", "does ", "did ")):
        return "boolean"
    return "lookup"


def _numbers(text: str) -> List[str]:
    return re.findall(r"(?<![a-z])\d+(?:[,.]\d+)?(?:%|[a-z]+)?", (text or "").lower())


def _claim_support(claim: str, evidence: str) -> float:
    claim_terms = set(tokens(claim))
    evidence_terms = set(tokens(evidence))
    if not claim_terms:
        return 0.0
    return len(claim_terms & evidence_terms) / len(claim_terms)


def run_relation_checks(question: str, obj: Dict[str, Any] | None,
                        sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Run explainable checks on a parsed structured answer."""
    obj = obj or {}
    claims = obj.get("claims") if isinstance(obj.get("claims"), list) else []
    cited = citation_ids({"evidence_ids": [sid for c in claims if isinstance(c, dict)
                                              for sid in (c.get("evidence_ids") or [])]})
    valid_ids = [sid for sid in cited if sid in sources]
    evidence_text = " ".join(sources[sid].get("text", "") for sid in valid_ids)
    answer = obj.get("answer", "") if isinstance(obj.get("answer"), str) else ""
    requested_type = infer_question_type(question)
    requested_operation = infer_operation(question)
    answer_type = obj.get("answer_type")
    operation = obj.get("operation")
    type_ok = answer_type in {requested_type, "other"} or (requested_type == "list" and answer_type == "entity")
    operation_ok = operation in {requested_operation, "other"}
    citations_ok = bool(valid_ids) and len(valid_ids) == len(cited)
    claim_support = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        text = claim.get("text", "") if isinstance(claim.get("text"), str) else ""
        ids = citation_ids({"evidence_ids": claim.get("evidence_ids", [])})
        joined = " ".join(sources[sid].get("text", "") for sid in ids if sid in sources)
        claim_support.append({"text": text, "evidence_ids": ids,
                              "lexical_support": _claim_support(text, joined)})
    all_claims_supported = bool(claim_support) and all(x["lexical_support"] >= 0.35 for x in claim_support)
    answer_in_evidence = bool(tokens(answer)) and set(tokens(answer)).issubset(set(tokens(evidence_text)))
    numbers_in_answer = _numbers(answer)
    numbers_in_explanation = _numbers(obj.get("explanation", "") if isinstance(obj.get("explanation"), str) else "")
    numeric_consistent = not numbers_in_answer or set(numbers_in_answer).issubset(set(numbers_in_explanation) | set(_numbers(evidence_text)))
    issues: List[str] = []
    if not answer.strip():
        issues.append("empty_answer")
    if not citations_ok:
        issues.append("invalid_or_missing_citations")
    if not type_ok:
        issues.append("wrong_type")
    if not operation_ok:
        issues.append("wrong_operation")
    if not all_claims_supported:
        issues.append("unsupported_claim")
    if requested_operation in {"comparison", "intersection", "count"} and not operation_ok:
        issues.append("missing_relation")
    if not numeric_consistent:
        issues.append("numeric_contradiction")
    return {
        "question_type": requested_type,
        "question_operation": requested_operation,
        "answer_type_ok": type_ok,
        "operation_ok": operation_ok,
        "citations_valid": citations_ok,
        "answer_tokens_in_evidence": answer_in_evidence,
        "claim_support": claim_support,
        "all_claims_lexically_supported": all_claims_supported,
        "numeric_consistent": numeric_consistent,
        "issues": sorted(set(issues)),
        "passed": not issues,
    }

