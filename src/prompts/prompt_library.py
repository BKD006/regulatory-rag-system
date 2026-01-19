"""
Central prompt library for the retrieval and generation pipeline.

Design goals:
- Single source of truth for prompts
- No hard-coded strings across the codebase
- LangGraph-compatible
- Auditable & versionable (important for compliance use-cases)
"""

from enum import Enum
from typing import List


# ------------------------------------------------------------------
# Prompt Identifiers
# ------------------------------------------------------------------

class PromptID(str, Enum):
    """
    Unique identifiers for prompts used in the system.
    """
    RETRIEVAL_SYSTEM = "retrieval_system"
    ANSWER_SYSTEM = "answer_system"


# ------------------------------------------------------------------
# System Prompts
# ------------------------------------------------------------------

RETRIEVAL_SYSTEM_PROMPT = """
You are a retrieval-augmented reasoning assistant for regulatory and technical documents.

Your task is to answer the user’s question using ONLY the provided context chunks.

Rules you must follow:
1. Do NOT use prior knowledge.
2. Do NOT hallucinate missing information.
3. If the answer is not explicitly present, say:
   “The provided documents do not contain this information.”
4. Prefer precise, clause-level answers over summaries.
5. Preserve technical language exactly as written.
6. If multiple chunks provide related information:
   - Merge them coherently
   - Do not repeat content
7. Cite sources using document title and section context when available.

You are operating in a compliance-critical environment.
Accuracy and traceability are more important than fluency.
""".strip()


# ------------------------------------------------------------------
# Prompt Registry (Single Source of Truth)
# ------------------------------------------------------------------

PROMPT_REGISTRY = {
    PromptID.RETRIEVAL_SYSTEM: RETRIEVAL_SYSTEM_PROMPT,
}


# ------------------------------------------------------------------
# User Prompt Builders
# ------------------------------------------------------------------

def build_user_prompt(
    question: str,
    context_chunks: List[str],
) -> str:
    """
    Builds a user prompt using retrieved context chunks.

    Args:
        question: Original user query
        context_chunks: List of retrieved chunk contents

    Returns:
        A formatted prompt string ready for LLM invocation
    """
    context = "\n\n---\n\n".join(context_chunks)

    return f"""
Question:
{question}

Context:
{context}

Answer:
""".strip()
