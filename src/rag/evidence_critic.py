"""Lightweight evidence sufficiency checks for iterative RAG.

The critic is deliberately model-free.  It scores whether the generated answer
is supported by the retrieved passages and whether the model supplied a quote
that can be found in those passages.  This makes it useful as a safe controller
in benchmarks where training a learned stop model is not yet practical.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "with",
}


def _tokens(text: str) -> List[str]:
    return [
        token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if token not in _STOPWORDS and len(token) > 1
    ]


class EvidenceSufficiencyCritic:
    """Score answer support using lexical evidence and quote checks.

    The score is in ``[0, 1]`` and is intended as a conservative signal.  An
    answer must have lexical support and a score above ``threshold`` before it
    can mark a retrieval hop as evidence-sufficient.
    """

    def __init__(self, threshold: float = 0.65):
        self.threshold = max(0.0, min(1.0, float(threshold)))

    @staticmethod
    def _evidence_text(evidence: Iterable[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for item in evidence or []:
            parts.append(str(item.get("title", "")))
            parts.append(str(item.get("text", "")))
        return " ".join(parts).lower()

    def evaluate(
        self,
        question: str,
        answer: str,
        evidence: List[Dict[str, Any]],
        supporting_quotes: Iterable[str] | None = None,
    ) -> Dict[str, Any]:
        answer = (answer or "").strip()
        evidence_text = self._evidence_text(evidence)
        answer_tokens = _tokens(answer)
        question_tokens = set(_tokens(question))
        evidence_tokens = set(_tokens(evidence_text))

        if not answer_tokens or not evidence_text:
            return {
                "score": 0.0,
                "sufficient": False,
                "answer_grounded": False,
                "answer_token_coverage": 0.0,
                "question_token_coverage": 0.0,
                "quote_support": 0.0,
                "reason": "empty_answer_or_evidence",
            }

        answer_coverage = sum(token in evidence_tokens for token in answer_tokens) / len(answer_tokens)
        phrase_grounded = answer.lower().strip("\"'") in evidence_text
        if phrase_grounded:
            answer_coverage = 1.0

        question_coverage = (
            sum(token in evidence_tokens for token in question_tokens) / len(question_tokens)
            if question_tokens else 0.0
        )

        quotes = [str(q).strip().lower() for q in (supporting_quotes or []) if str(q).strip()]
        quote_support = (
            sum(1 for quote in quotes if quote in evidence_text) / len(quotes)
            if quotes else 0.0
        )

        # Direct answer grounding dominates. Quotes and question coverage are
        # secondary because model-produced quotes can be absent or paraphrased.
        score = 0.70 * answer_coverage + 0.20 * quote_support + 0.10 * question_coverage
        answer_grounded = answer_coverage >= 0.5
        sufficient = bool(answer_grounded and score >= self.threshold)

        if sufficient:
            reason = "answer_grounded_with_support"
        elif not answer_grounded:
            reason = "answer_not_grounded"
        elif quote_support == 0.0 and quotes:
            reason = "supporting_quotes_not_found"
        else:
            reason = "critic_score_below_threshold"

        return {
            "score": round(float(score), 4),
            "sufficient": sufficient,
            "answer_grounded": answer_grounded,
            "answer_token_coverage": round(float(answer_coverage), 4),
            "question_token_coverage": round(float(question_coverage), 4),
            "quote_support": round(float(quote_support), 4),
            "reason": reason,
        }
