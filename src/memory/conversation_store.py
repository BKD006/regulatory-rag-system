import json
from typing import Optional, Dict, Any, List
from utils import db_utils
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class ConversationStore:

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