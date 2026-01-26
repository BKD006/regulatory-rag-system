"""
Answer Generation with Citations for Hybrid RAG.

- Consumes retrieved chunks
- Produces a grounded answer
- Attaches citations (document + chunk IDs)
"""

from typing import List
from dataclasses import dataclass
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq  # or any LLM you use
from utils.models import RetrievedChunk
from src.prompts.prompt_library import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT_TEMPLATE,
)
from utils.observability import langfuse_callback
@dataclass
class AnswerResult:
    answer: str
    citations: List[RetrievedChunk]
# ------------------------------------------------------------------
# Answer Generator
# ------------------------------------------------------------------

class AnswerGenerator:
    """
    Generates a grounded answer with citations from retrieved chunks.
    """

    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
    ):
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            callbacks=[langfuse_callback],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
            self,
            question: str,
            retrieved_chunks: List[RetrievedChunk],
        ) -> AnswerResult:

            if not retrieved_chunks:
                return AnswerResult(
                    answer="I could not find this information in the provided documents.",
                    citations=[],
                )

            # Build numbered source blocks
            source_blocks = []
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                source_blocks.append(
                    f"[{idx}] Source: {chunk.source}\n{chunk.content}"
                )

            sources_text = "\n\n".join(source_blocks)

            user_prompt = ANSWER_USER_PROMPT_TEMPLATE.format(
                question=question,
                sources=sources_text,
            )

            messages = [
                SystemMessage(content=ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            response = await self.llm.ainvoke(messages)

            return AnswerResult(
                answer=response.content.strip(),
                citations=retrieved_chunks,
            )