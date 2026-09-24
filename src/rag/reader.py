"""Auditable answer-candidate reader for multi-hop RAG.

The default pipeline keeps Qwen's answer-selection behaviour for a clean
control.  ``GroundedCandidateReader`` is an optional reader ablation: it
collects the short answer, intermediate hypothesis, and any alternatives
returned at each hop, then verifies them with deterministic checks.  It does
not use gold answers or supporting-fact labels.

Example: if one hop proposes ``Rome`` and a later hop proposes ``Italy``, the
reader prefers the candidate that is explicitly present in the evidence and
matches the question type, while retaining every score in the trace.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from src.utils.text import normalize_text


_QUESTION_TYPE_PATTERNS = {
    "person": (r"^\s*who\b", r"^\s*whom\b"),
    "place": (r"^\s*where\b", r"\bwhat city\b", r"\bwhat country\b"),
    "number": (r"^\s*how many\b", r"^\s*how much\b", r"\bwhat number\b"),
    "date": (r"^\s*when\b", r"\bwhat year\b", r"\bwhat date\b"),
    "attribute": (
        r"\bwhat (?:nationality|ethnicity|heritage|language|religion|genre|occupation)\b",
        r"\bwhich (?:nationality|ethnicity|heritage|language|religion|genre|occupation)\b",
    ),
}

_NUMBER_RE = re.compile(r"\b(?:\d+(?:[,.]\d+)*|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand)\b", re.I)
_DATE_RE = re.compile(r"\b(?:1\d{3}|20\d{2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)


def question_type(question: str) -> str:
    """Infer a coarse answer type from the question wording."""
    text = (question or "").strip().lower()
    for kind, patterns in _QUESTION_TYPE_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            return kind
    return "entity"


def _clean_candidate(value: Any) -> str:
    """Remove answer wrappers while retaining the model's short span."""
    if value is None:
        return ""
    text = str(value).strip().strip("\"'")
    text = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*", "", text, flags=re.I)
    text = re.split(r"\s*(?:\n|[.!?])\s*", text, maxsplit=1)[0].strip(" \"'")
    return text


def _evidence_text(evidence: Iterable[Dict[str, Any]]) -> str:
    return " ".join(
        f"{item.get('title', '')} {item.get('text', '')}".lower()
        for item in evidence or []
    )


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


class GroundedAnswerVerifier:
    """Verify candidate type, evidence support, and cross-hop consistency."""

    def verify(
        self,
        question: str,
        candidate: str,
        evidence: List[Dict[str, Any]],
        prior_candidates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        candidate = _clean_candidate(candidate)
        kind = question_type(question)
        normalized = normalize_text(candidate)
        evidence_norm = normalize_text(_evidence_text(evidence))
        exact_support = bool(normalized and normalized in evidence_norm)

        cand_tokens = _token_set(candidate)
        ev_tokens = _token_set(evidence_norm)
        token_support = (
            sum(token in ev_tokens for token in cand_tokens) / len(cand_tokens)
            if cand_tokens else 0.0
        )
        # Exact phrase support is required for the strongest score.  Partial
        # token overlap remains visible but cannot masquerade as exact support.
        evidence_score = 1.0 if exact_support else 0.5 * token_support

        type_score = 1.0
        type_reason = "compatible_or_open_type"
        if not candidate:
            type_score, type_reason = 0.0, "empty_candidate"
        elif kind == "person":
            type_score = 0.0 if _NUMBER_RE.search(candidate) or len(candidate.split()) > 10 else 1.0
            type_reason = "person_shape" if type_score else "looks_like_number_or_sentence"
        elif kind == "number":
            type_score = 1.0 if _NUMBER_RE.search(candidate) else 0.0
            type_reason = "contains_number" if type_score else "missing_number"
        elif kind == "date":
            type_score = 1.0 if _DATE_RE.search(candidate) else 0.0
            type_reason = "contains_date" if type_score else "missing_date"
        elif kind == "place":
            type_score = 0.0 if _NUMBER_RE.fullmatch(candidate.strip()) else 1.0
            type_reason = "place_shape" if type_score else "number_for_place"
        elif kind == "attribute":
            # Attribute semantics are open-ended; reject only obvious full
            # sentence leakage and keep the evidence check decisive.
            type_score = 0.0 if len(candidate.split()) > 12 else 1.0
            type_reason = "attribute_span" if type_score else "sentence_leakage"

        prior_norm = {normalize_text(item) for item in (prior_candidates or []) if item}
        repeated = normalized in prior_norm if normalized else False
        consistency_score = 1.0 if repeated else 0.5 if exact_support else 0.0

        # Model confidence is added by ``score_candidate``; these components
        # are intentionally explicit so a paper audit can explain a choice.
        return {
            "candidate": candidate,
            "question_type": kind,
            "exact_evidence_support": exact_support,
            "token_evidence_support": round(float(token_support), 4),
            "evidence_score": round(float(evidence_score), 4),
            "type_score": round(float(type_score), 4),
            "type_reason": type_reason,
            "repeated_across_hops": repeated,
            "consistency_score": round(float(consistency_score), 4),
        }


class GroundedCandidateReader:
    """Collect and select answer candidates using the deterministic verifier."""

    def __init__(self) -> None:
        self.verifier = GroundedAnswerVerifier()

    def candidates_from_hop(
        self,
        question: str,
        hop_result: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        hop: int,
        prior_candidates: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Create auditable candidates from one Qwen response."""
        values: List[tuple[str, str]] = []
        answer = _clean_candidate(hop_result.get("answer"))
        if answer:
            values.append(("answer", answer))
        hypothesis = _clean_candidate(hop_result.get("current_hypothesis"))
        if hypothesis and normalize_text(hypothesis) != normalize_text(answer):
            values.append(("current_hypothesis", hypothesis))
        alternatives = hop_result.get("answer_candidates") or []
        if isinstance(alternatives, list):
            for alternative in alternatives:
                cleaned = _clean_candidate(alternative)
                if cleaned:
                    values.append(("answer_candidates", cleaned))

        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for source, value in values:
            key = normalize_text(value)
            if not key or key in seen:
                continue
            seen.add(key)
            verification = self.verifier.verify(
                question=question,
                candidate=value,
                evidence=evidence,
                prior_candidates=prior_candidates,
            )
            # Confidence remains the model's signal, but unsupported or
            # type-incompatible candidates cannot win on confidence alone.
            confidence = max(0.0, min(1.0, float(hop_result.get("confidence", 0.0))))
            score = (
                0.45 * confidence
                + 0.30 * verification["evidence_score"]
                + 0.15 * verification["type_score"]
                + 0.10 * verification["consistency_score"]
            )
            if not verification["exact_evidence_support"]:
                score -= 0.10
            if verification["type_score"] == 0.0:
                score -= 0.25
            rows.append({
                "hop": hop,
                "source": source,
                "confidence": round(confidence, 4),
                "selection_score": round(float(score), 4),
                **verification,
            })
        return rows

    def select(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return the highest-scoring candidate and the full audit trail."""
        if not candidates:
            return {"answer": "", "selected": None, "candidates": []}
        selected = max(candidates, key=lambda row: (row["selection_score"], row["confidence"], -row["hop"]))
        return {
            "answer": selected["candidate"],
            "selected": selected,
            "candidates": candidates,
        }
