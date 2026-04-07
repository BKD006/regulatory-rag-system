import json
from typing import Optional, Dict, Any, List
from utils import db_utils
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class ConversationStore:
    """
    Handles storage and retrieval of conversation history (QA logs).

    Provides methods to persist question-answer pairs and fetch recent
    conversation history for session-based context.

    Attributes:
        None
    """

    # ------------------------------------------
    # SAVE QA
    # ------------------------------------------

    async def save_qa(
        self,
        query: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves a question-answer pair into the database.

        Args:
            query (str): User query.
            answer (str): Generated answer.
            citations (Optional[List[Dict]]): List of citation metadata.
            metadata (Optional[Dict[str, Any]]): Additional metadata (e.g., session_id).

        Returns:
            None

        Raises:
            RegulatoryRAGException: If database insertion fails.
        """

        try:
            async with db_utils.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO public.qa_logs
                    (query, answer, citations, metadata)
                    VALUES ($1, $2, $3, $4)
                    """,
                    query,
                    answer,
                    json.dumps(citations) if citations else None,
                    json.dumps(metadata) if metadata else None,
                )

            log.info(
                "qa_log_saved",
                query_length=len(query),
            )

        except Exception as e:
            log.error(
                "qa_log_save_failed",
                error=str(e),
            )
            raise RegulatoryRAGException(e)
        
    # ------------------------------------------

    async def get_recent_history(
        self,
        session_id: str,
        limit: int = 4
    ) -> List[Dict[str, str]]:
        """
        Retrieves recent conversation history for a given session.

        Returns messages formatted for chat-based LLM input.

        Args:
            session_id (str): Identifier for the conversation session.
            limit (int): Number of recent QA pairs to fetch.

        Returns:
            List[Dict[str, str]]:
                List of messages in chronological order with roles:
                [{"role": "user", "content": ...},
                 {"role": "assistant", "content": ...}]

        Notes:
            - Results are returned in chronological order.
            - Returns an empty list if retrieval fails.
        """

        try:
            async with db_utils.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT query, answer
                    FROM public.qa_logs
                    WHERE metadata->>'session_id' = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    session_id,
                    limit
                )

            history = []
            for r in reversed(rows):  # maintain chronological order
                history.append({"role": "user", "content": r["query"]})
                history.append({"role": "assistant", "content": r["answer"]})

            return history

        except Exception as e:
            log.error("fetch_history_failed", error=str(e))
            return []