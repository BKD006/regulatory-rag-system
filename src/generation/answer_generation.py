from typing import List, Optional, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from utils.models import RetrievedChunk, AnswerResult
from src.prompts.prompt_library import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT_TEMPLATE,
)
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log


# ------------------------------------------------------------------
# Answer Generator
# ------------------------------------------------------------------

class AnswerGenerator:
    """
    Generates answers using an LLM based on retrieved document chunks.

    Applies strict constraints to ensure answers are grounded in a single document
    and limits context size to avoid exceeding token limits.

    Attributes:
        llm (object): Loaded language model used for answer generation.
    """

    def __init__(self):
        """
        Initializes the AnswerGenerator by loading the LLM.

        Raises:
            Exception: If LLM loading fails.
        """
        self.llm = ModelLoader().load_llm()
        log.info("answer_generator_initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        chat_history: Optional[List[Dict]] = None,
        previous_answer: Optional[str] = None
    ) -> AnswerResult:
        """
        Generates an answer to a question using retrieved document chunks.

        Enforces constraints such as:
        - Single document grounding
        - Context size limitation
        - Strict prompt rules to avoid hallucination

        Args:
            question (str): User query.
            retrieved_chunks (List[RetrievedChunk]): Relevant chunks retrieved from documents.
            chat_history (Optional[List[Dict]]): Prior conversation history.
            previous_answer (Optional[str]): Previously generated answer for refinement.

        Returns:
            AnswerResult: Generated answer along with associated citations.
        """

        # --------------------------------------------------
        # Handle empty retrieval case
        # --------------------------------------------------
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
        # HARD safety: enforce single document constraint
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

        # --------------------------------------------------
        # TOKEN-SAFE CONTEXT LIMITER
        # --------------------------------------------------
        MAX_CHARS = 8000  # Approximate safe threshold
        total_chars = 0
        limited_chunks = []

        for c in retrieved_chunks:
            total_chars += len(c.content)

            if total_chars > MAX_CHARS:
                break

            limited_chunks.append(c)

        # Ensure minimum context is preserved
        if not limited_chunks:
            limited_chunks = retrieved_chunks[:2]

        retrieved_chunks = limited_chunks

        log.info(
            "context_limited",
            chunk_count=len(retrieved_chunks),
            total_chars=total_chars,
        )

        log.info(
            "answer_generation_started",
            document=document_title,
            chunk_count=len(retrieved_chunks),
        )

        # --------------------------------------------------
        # Build structured source blocks
        # --------------------------------------------------
        source_blocks = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source_blocks.append(
                f"[{idx}] ({document_title})\n{chunk.content}"
            )

        sources_text = "\n\n".join(source_blocks)

        history_block = ""
        if chat_history:
            history_block = "\n\nConversation History:\n" + "\n".join(
                f"{h['role']}: {h['content']}" for h in chat_history
            )

        previous_answer_block = ""
        if previous_answer:
            previous_answer_block = f"\n\nPrevious Answer:\n{previous_answer}"

        # --------------------------------------------------
        # Construct user prompt
        # --------------------------------------------------
        user_prompt = ANSWER_USER_PROMPT_TEMPLATE.format(
            question=question,
            sources=sources_text,
        ) + history_block + previous_answer_block

        # --------------------------------------------------
        # Construct system prompt with strict rules
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

        # Prepare LLM messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # --------------------------------------------------
        # LLM invocation
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