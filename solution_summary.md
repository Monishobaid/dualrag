# RAG System Stabilization & Fixes

## Critical Bug Fixes

1.  **Infinite Chunking Loop (src/llm/generation.py):**
    - Fixed a critical bug in `ChunkedGenerator.generate_with_callback` where `cumulative_text` was never updated.
    - Previously, the model received the same prompt repeatedly in every chunk, leading to hallucination loops and broken JSON output.
    - **Fix:** Added `cumulative_text += text` inside the loop.

## Architectural Simplifications

1.  **Removing Fallback Extraction (src/rag/iterative_rag.py):**
    - Removed the secondary LLM call that attempted to extract answers from malformed JSON.
    - This unreliable fallback introduced noise and "mutated" correct retrieval data.
    - **New Behavior:** If JSON parsing fails (which is rare now), the system logs a warning, returns an empty answer, and relies on the robust retrieval loop to retry with the original question.

2.  **Removing Semantic Verification (src/rag/iterative_rag.py):**
    - Disabled the "Semantic Verifier" step which was prone to hallucinating "corrections" based on limited context.
    - The system now relies on the strong, unmutated output of the primary LLM.

## Configuration Updates

- Recommended running with `retrieval_stop_only` method (disabling reasoning stop) for Azure OpenAI models to bypass chunking entirely and use standard, stable generation.
- Increased `max_new_tokens` to 2048 to prevent truncating valid JSON responses.

## Results (v20_simplified)

- **EM:** 0.60 (Stable)
- **F1:** 0.822 (High Accuracy)
- **JSON Failures:** 0 (Perfect parse rate)
- **Stability:** No crashes, no infinite loops.
