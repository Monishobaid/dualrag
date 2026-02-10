"""Prompt templates for RAG system."""

from typing import List, Dict, Any


def format_paragraphs(paragraphs: List[Dict[str, Any]], max_paragraphs: int = 10) -> str:
    """
    Format paragraphs for prompt.
    
    Args:
        paragraphs: List of paragraph dicts with 'title' and 'text'
        max_paragraphs: Maximum paragraphs to include
    
    Returns:
        Formatted evidence string
    """
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
    hop: int
) -> str:
    """
    Prompt for multi-hop reasoning.
    
    Args:
        question: Original question
        evidence: Retrieved paragraphs
        hop: Current hop number
    
    Returns:
        Formatted prompt
    """
    evidence_str = format_paragraphs(evidence)
    
    prompt = f"""You are answering a multi-hop question that may require information from multiple sources.

Question: {question}

Evidence (Retrieved Information):
{evidence_str}

Based on the evidence above, first provide your step-by-step reasoning in a <THOUGHT> block.
Then, provide your structured response in a <JSON> block.

Format:
<THOUGHT>
Your detailed step-by-step reasoning here.
When you form a hypothesis, state: "Current Hypothesis: <your answer>"
At the end of your thought, state: "Therefore, the answer is: <your answer>"

Example:
<THOUGHT>
The user is asking about the capital of France.
Checking retrieval... found Paris.
Current Hypothesis: Paris
Paris is the capital.
Therefore, the answer is: Paris
</THOUGHT>
</THOUGHT>
<JSON>
{{
    "current_hypothesis": "<Brief answer based on current knowledge>",
    "answer": "<Insert your final answer here>",
    "has_enough_evidence": <true or false>,
    "supporting_quotes": ["<quote 1>", "<quote 2>"],
    "followup_query": "<Insert follow-up question here, or empty string if done>",
    "confidence": <float between 0.0 and 1.0>
}}
</JSON>

Instructions:
- Use the <THOUGHT> block to analyze the evidence and converge on an answer
- If the evidence contains enough information, set has_enough_evidence to true
- If you need more specific information, create a focused follow-up query
- Only output the XML-like blocks as specified
- End your output with the exact string: <END_JSON>

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
    prompt = f"""Extract the final answer from the following text. Return ONLY the short, direct answer with no explanation.

Text:
{text}

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
