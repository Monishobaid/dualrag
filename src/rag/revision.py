"""One-call evidence-only revision for the answer-verification experiment.

RARR, Re-Ex, and RAC motivate revision after locating unsupported claims. This
module keeps the intervention bounded: supported claims are preserved and the
model receives only the retrieved evidence, so repeated autonomous revision is
not hidden inside the experiment.
"""

from __future__ import annotations

import json


REVISION_SYSTEM = (
    "Revise a candidate answer only when the supplied verification flags a "
    "problem. Use only the supplied evidence. Preserve supported claims, fix "
    "unsupported or contradictory claims, and return exactly one JSON object "
    "with answer, answer_type, operation, claims, and explanation. Keep the "
    "answer short and cite sentence IDs. If the evidence cannot support a "
    "correction, return the original answer unchanged. Do not use outside "
    "knowledge or hidden chain-of-thought."
)


def revision_prompt(question: str, original: dict, checks: dict, verification: dict,
                    context: str) -> str:
    return (
        f"EVIDENCE:\n{context}\n\nQUESTION: {question}\n\n"
        f"ORIGINAL_JSON:\n{json.dumps(original, ensure_ascii=False)}\n\n"
        f"DETERMINISTIC_CHECKS:\n{json.dumps(checks, ensure_ascii=False)}\n\n"
        f"VERIFICATION:\n{json.dumps(verification, ensure_ascii=False)}\n\n"
        "Return the corrected or unchanged JSON object only."
    )

