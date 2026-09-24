"""Prompt templates used by the custom HotpotQA reader.

The JSON fields (answer, evidence flag, quotes, follow-up query, confidence,
and short reasoning) are an experiment interface designed for this repository.
They are not a verbatim prompt from any one uploaded paper. Keeping the
interface structured makes each hop auditable and lets the stopping code read a
complete ``answer`` field safely.
"""

from typing import List, Dict, Any, Optional


def format_paragraphs(paragraphs: List[Dict[str, Any]], max_paragraphs: Optional[int] = None) -> str:
    """
    Format paragraphs for prompt.
    
    Args:
        paragraphs: List of paragraph dicts with 'title' and 'text'
        max_paragraphs: Optional maximum paragraphs to include. ``None`` keeps
            every paragraph.  The old implicit ten-paragraph cap silently
            discarded later-hop evidence, so callers now choose a budget
            explicitly when one is needed.
    
    Returns:
        Formatted evidence string
    """
    if max_paragraphs is not None:
        if max_paragraphs < 0:
            raise ValueError("max_paragraphs must be non-negative or None")
        paragraphs = paragraphs[:max_paragraphs]
    
    formatted = []
    for i, para in enumerate(paragraphs, 1):
        title = para.get('title', f'Paragraph {i}')
        text = para.get('text', '')
        formatted.append(f"[{i}] Title: {title}\n{text}")
    
    return "\n\n".join(formatted)


def get_multihop_reasoning_prompt(
    question: str,
    evidence: List[Dict[str, Any]],
    hop: int,
    max_evidence_paragraphs: Optional[int] = None,
    generate_candidates: bool = False,
) -> str:
    """
    Build the reader prompt for one retrieval hop.

    ``current_hypothesis`` is intentionally an intermediate field; the final
    ``answer`` must be a short span. For example, a hop may mention a person in
    one paragraph but still need a follow-up query to find that person's birth
    place.
    
    Args:
        question: Original question
        evidence: Retrieved paragraphs
        hop: Current hop number
    
    Returns:
        Formatted prompt
    """
    evidence_str = format_paragraphs(evidence, max_paragraphs=max_evidence_paragraphs)
    
    candidate_field = ''
    candidate_instruction = ''
    if generate_candidates:
        candidate_field = '    "answer_candidates": ["<up to three short answer spans>"],\n'
        candidate_instruction = (
            '- "answer_candidates": Give up to three distinct short spans supported by the evidence. '
            'Use this only for plausible alternatives; do not write sentences.\n'
        )

    candidate_rules = candidate_instruction
    candidate_rules += ('- Do not add an answer_candidates field in legacy mode.\n' if not generate_candidates else '')
    candidate_rules += ('- When present, answer_candidates must contain short entity, attribute, date, or number spans, never explanations.\n' if generate_candidates else '')

    prompt = f"""You are answering a multi-hop question that may require information from multiple sources.

Question: {question}

Evidence (Retrieved Information):
{evidence_str}

Based on the evidence above, provide your reasoning and structured response in a single valid JSON object. Do not use Markdown formatting (e.g. ```json).
Format:
{{
    "current_hypothesis": "<Brief answer based on current knowledge>",
    "answer": "<Insert your final answer here (SHORT SPAN ONLY, NO SENTENCES)>",
{candidate_field}    "has_enough_evidence": <true or false>,
    "supporting_quotes": ["<quote 1>", "<quote 2>"],
    "followup_query": "<Insert follow-up question here, or empty string if done>",
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<Concise reasoning (MAX 20 WORDS).>"
}}

Instructions:
- "reasoning": Analyze briefly (MAX 20 WORDS). BE CONCISE.
- "current_hypothesis": Brief answer based on current knowledge.
{candidate_rules}
- SHORT ANSWER: Return ONLY the entity, name, date, or number. Do NOT write full sentences like "The answer is...".
- TYPE CONSISTENCY: Ensure your answer matches the specific type requested.
- ATTRIBUTE SELECTION: If asked for an attribute of an entity, provide the attribute, not the entity itself.
- CONSTRAINT SATISFACTION: If the question implies a singular answer, do not provide a list.
- MISSING INFO: If the retrieved text mentions an entity but doesn't explicitly answer the question about that entity, set "has_enough_evidence" to false and write a "followup_query" specifically asking about that missing detail.
- DO NOT GUESS: If you cannot find the EXACT answer in the text, you MUST ask a follow-up query. Focus on what details are missing.
- INTERMEDIATE HOPPING: If your "current_hypothesis" is a Place/Object but the Question asks "Who" (or vice versa), you MUST form a follow-up query to bridge the gap.
- CONFIDENCE SCORING: If you have to guess, or if you are asking a follow-up query, set confidence < 0.5.
- CRITICAL: Return ONLY the JSON object. No preamble, no postscript.

Response:"""
    
    return prompt


def get_answer_extraction_prompt(text: str) -> str:
    """
    Prompt to extract final answer from generated text.
    
    Args:
        text: Generated text containing reasoning
    
    Returns:
        Extraction prompt
    """
    prompt = f"""Extract the final answer from the following text based on the user's implicit question type.
    
Text:
{text}

Instructions:
1. Identify if the user is asking for a Person, Place, Number, Date, or Attribute.
2. Select the candidate answer that matches this type.
3. RULES:
   - For PERSONS: Provide the FULL NAME (e.g., "Salma Hayek Pinault", not just "Salma Hayek").
   - For NUMBERS: Include the UNIT if appropriate (e.g., "40 members", not just "40").
   - For LISTS: If the question implies a single entity, pick the most prominent one.
4. Return ONLY the short, direct answer.

Final Answer:"""
    
    return prompt


def get_followup_query_prompt(
    question: str,
    current_answer: str,
    evidence_so_far: List[Dict[str, Any]]
) -> str:
    """
    Prompt to generate follow-up query.
    
    Args:
        question: Original question
        current_answer: Current best answer
        evidence_so_far: Evidence retrieved so far
    
    Returns:
        Follow-up query generation prompt
    """
    evidence_str = format_paragraphs(evidence_so_far, max_paragraphs=5)
    
    prompt = f"""You are answering: {question}

Current answer: {current_answer}

Evidence so far:
{evidence_str}

What specific information do you still need to answer this question completely? Generate ONE focused follow-up question that would help you find the missing information.

If you have enough information already, respond with "NONE".

Follow-up question:"""
    
    return prompt


def get_answer_verification_prompt(question: str, answer: str) -> str:
    """
    Prompt to verify if the answer matches the question's semantic type constraint.
    """
    prompt = f"""Check if the Proposed Answer satisfies the specific semantic constraints of the Question (e.g., specific Nationality, Heritage, Genre, Number, etc).
If the Proposed Answer is incorrect (e.g. wrong type, wrong entity), correct it using the Question context. If it is already correct, repeat it.

Question: {question}
Proposed Answer: {answer}

Examples:
Q: "Who starred in the film?" A: "Brad Pitt" -> Correct (Person)
Q: "Starred the actress of what heritage?" A: "Brad Pitt" -> INCORRECT (Person, not Heritage). Correct: "American"
Q: "How many members?" A: "The Senate" -> INCORRECT (Entity). Correct: "100"
Q: "Who is the actress?" A: "Salma Hayek" -> VERIFIED: Salma Hayek Pinault (Prefer Full Name)

Task:
1. Identify the TYPE requested by the Question (e.g. Person, Heritage, Number).
2. CRITICAL: Check if Proposed Answer matches that TYPE. If NO, Correct it immediately.
3. ONLY IF Type matches: Check specific constraints (e.g. Full Name for Persons, Units for Numbers).
4. If yes, output: VERIFIED: <Answer>
5. If no, output: CORRECTION: <Corrected Answer based on context or type>

Response:"""
    return prompt


def get_confidence_prompt(question: str, answer: str, evidence: List[Dict[str, Any]]) -> str:
    """
    Prompt to estimate confidence in answer.
    
    Args:
        question: Question
        answer: Current answer
        evidence: Available evidence
    
    Returns:
        Confidence estimation prompt
    """
    evidence_str = format_paragraphs(evidence, max_paragraphs=5)
    
    prompt = f"""Question: {question}
Answer: {answer}

Evidence:
{evidence_str}

On a scale of 0.0 to 1.0, how confident are you that this answer is correct based on the evidence?
- 1.0 = Completely certain, evidence directly supports the answer
- 0.5 = Somewhat confident, some supporting evidence
- 0.0 = Not confident, insufficient evidence

Respond with ONLY a number between 0.0 and 1.0.

Confidence:"""
    
    return prompt


def get_simple_qa_prompt(question: str, evidence: List[Dict[str, Any]]) -> str:
    """
    Simple QA prompt without JSON output (for baseline).
    
    Args:
        question: Question  
        evidence: Evidence paragraphs
    
    Returns:
        Simple QA prompt
    """
    evidence_str = format_paragraphs(evidence)
    
    prompt = f"""Answer the following question based on the provided evidence.

Question: {question}

Evidence:
{evidence_str}

Provide a concise, direct answer:

Answer:"""
    
    return prompt


# System prompts
SYSTEM_PROMPT = """You are a helpful AI assistant specialized in answering multi-hop questions. You analyze evidence carefully and provide accurate, concise answers."""

JSON_SYSTEM_PROMPT = """You are a helpful AI assistant that always responds in valid JSON format. You analyze questions and evidence carefully to provide accurate answers."""
