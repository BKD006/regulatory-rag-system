"""
Answer Generation with Citations for Hybrid RAG.

- Consumes retrieved chunks
- Produces a grounded answer
- Attaches citations (document + chunk IDs)
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from utils.models import AnswerOutput, Citation
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq  # or any LLM you use

from src.prompts.prompt_library import (
    PromptID,
    PROMPT_REGISTRY,
    build_user_prompt,
)

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
        )

        self.system_prompt = PROMPT_REGISTRY[PromptID.RETRIEVAL_SYSTEM]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        question: str,
        retrieved_chunks: List[Any],  # RetrievedChunk objects
    ) -> AnswerOutput:
        """
        Generate answer + citations from retrieved chunks.
        """

        if not retrieved_chunks:
            return AnswerOutput(
                answer="The provided documents do not contain this information.",
                evidence=[],
                citations=[],
            )

        # -----------------------------
        # Build context + citation map
        # -----------------------------
        context_chunks = []
        citation_map = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            tagged_chunk = (
                f"[CHUNK {idx}]\n"
                f"Source: {chunk.source}\n"
                f"{chunk.content}"
            )
            context_chunks.append(tagged_chunk)

            citation_map.append(
                Citation(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                )
            )

        user_prompt = build_user_prompt(
            question=question,
            context_chunks=context_chunks,
        )

        # -----------------------------
        # LLM call
        # -----------------------------
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)
        answer_text = response.content.strip()

        # -----------------------------
        # Evidence extraction (simple)
        # -----------------------------
        evidence = [
            f"{c.source} (chunk {i+1})"
            for i, c in enumerate(citation_map)
        ]

        return AnswerOutput(
            answer=answer_text,
            evidence=evidence,
            citations=citation_map,
        )
