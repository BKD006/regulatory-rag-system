"""
Answer Generation with Citations for Hybrid RAG.

- Consumes GUARDED chunks only
- Produces a grounded answer
- Enforces single-document usage
- Attaches citations safely
"""

from typing import List
from dataclasses import dataclass
from langchain_core.messages import SystemMessage, HumanMessage
from utils.models import RetrievedChunk
from src.prompts.prompt_library import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT_TEMPLATE,
)
from utils.model_loader import ModelLoader
from utils.observability import langfuse_callback
from logger import GLOBAL_LOGGER as log

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

@dataclass
class AnswerResult:
    answer: str
    citations: List[RetrievedChunk]


# ------------------------------------------------------------------
# Answer Generator
# ------------------------------------------------------------------

class AnswerGenerator:
    """
    Generates a grounded answer from a SINGLE document
    using pre-validated chunks.
    """

    def __init__(self):
        self.llm = ModelLoader().load_llm()
        log.info("answer_generator_initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> AnswerResult:
        """
        retrieved_chunks MUST already be guardrail-validated.
        """

        if not retrieved_chunks:
            log.warning("answer_generation_no_chunks")

            return AnswerResult(
                answer=(
                    "I could not find this information "
                    "in the provided document."
                ),
                citations=[],
            )

        # --------------------------------------------------
        # HARD safety: enforce single document at runtime
        # --------------------------------------------------
        sources = {c.source for c in retrieved_chunks}
        if len(sources) != 1:
            log.warning(
                "multiple_sources_detected_in_answer_generation",
                sources=list(sources),
            )

            return AnswerResult(
                answer=(
                    "This question cannot be answered safely "
                    "because multiple documents were detected."
                ),
                citations=[],
            )

        document_title = next(iter(sources))

        log.info(
            "answer_generation_started",
            document=document_title,
            chunk_count=len(retrieved_chunks),
        )

        # --------------------------------------------------
        # Build strictly controlled source blocks
        # --------------------------------------------------
        source_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source_blocks.append(
                f"[{idx}] ({document_title})\n{chunk.content}"
            )

        sources_text = "\n\n".join(source_blocks)

        # --------------------------------------------------
        # User prompt (STRICT)
        # --------------------------------------------------
        user_prompt = ANSWER_USER_PROMPT_TEMPLATE.format(
            question=question,
            sources=sources_text,
        )

        # --------------------------------------------------
        # System prompt (HARD instructions)
        # --------------------------------------------------
        system_prompt = (
            ANSWER_SYSTEM_PROMPT
            + "\n\n"
            + f"""
                IMPORTANT RULES (NON-NEGOTIABLE):
                - You MUST answer using ONLY the document titled "{document_title}"
                - Do NOT reference any other document
                - If the answer is not present, say so clearly
                - Do NOT guess or infer beyond the provided text
                """
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # --------------------------------------------------
        # LLM call
        # --------------------------------------------------
        response = await self.llm.ainvoke(messages)
        answer_text = response.content.strip()

        log.info(
            "answer_generation_completed",
            document=document_title,
            chunk_count=len(retrieved_chunks),
        )

        return AnswerResult(
            answer=answer_text,
            citations=retrieved_chunks,
        )
