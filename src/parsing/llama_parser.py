"""
LlamaParse document parser wrapper.

Replaces Docling DocumentConverter.

Responsibilities:
- Parse PDF using LlamaParse
- Return markdown text
- Hide parser implementation from ingestion pipeline
"""

import os
import sys
import asyncio
from typing import List

from dotenv import load_dotenv
from llama_parse import LlamaParse

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class LlamaDocumentParser:
    """
    Wrapper around LlamaParse.

    Returns markdown text suitable for chunking.
    """

    def __init__(self):

        load_dotenv()

        self.api_key = os.getenv("LLAMA_CLOUD_API_KEY")

        if not self.api_key:
            log.error(
                "llama_parser_missing_api_key"
            )
            raise RegulatoryRAGException(
                "LLAMA_CLOUD_API_KEY not set",
                sys,
            )

        try:
            self.parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",  # IMPORTANT
                verbose=False,
            )

            log.info(
                "llama_parser_initialized",
                result_type="markdown",
            )

        except Exception as e:

            log.error(
                "llama_parser_init_failed",
                error=str(e),
            )

            raise RegulatoryRAGException(e, sys)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    async def parse(
        self,
        file_path: str,
    ) -> str:
        """
        Parse document using LlamaParse.

        Returns:
            markdown text
        """

        if not os.path.exists(file_path):
            raise RegulatoryRAGException(
                f"File not found: {file_path}",
                sys,
            )

        log.info(
            "llama_parsing_started",
            file=file_path,
        )

        try:

            # LlamaParse is sync → run in thread
            documents = await asyncio.to_thread(
                self.parser.load_data,
                file_path,
            )

            text = self._join_documents(documents)

            log.info(
                "llama_parsing_completed",
                file=file_path,
                length=len(text),
            )

            return text

        except Exception as e:

            log.error(
                "llama_parsing_failed",
                file=file_path,
                error=str(e),
            )

            raise RegulatoryRAGException(e, sys)

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _join_documents(
        self,
        documents: List,
    ) -> str:
        """
        Merge multiple LlamaParse docs into one string.
        """

        texts = []

        for doc in documents:

            # LlamaParse returns objects with .text
            if hasattr(doc, "text") and doc.text:
                texts.append(doc.text)

        return "\n\n".join(texts)