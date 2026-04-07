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
    Parser class for extracting document content using LlamaParse API.

    Attributes:
        api_key (str): API key for authenticating with Llama Cloud.
        parser (LlamaParse): Instance used to parse documents into markdown format.
    """

    def __init__(self):
        """
        Initializes the LlamaDocumentParser with API credentials and parser client.

        Raises:
            RegulatoryRAGException:
                - If the API key is missing.
                - If parser initialization fails.
        """

        load_dotenv()

        # Fetch API key from environment
        self.api_key = os.getenv("LLAMA_CLOUD_API_KEY")

        if not self.api_key:
            log.error("llama_parser_missing_api_key")

            raise RegulatoryRAGException(
                "LLAMA_CLOUD_API_KEY not set",
                sys,
            )

        try:
            # Initialize LlamaParse client
            self.parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",  # Ensures output is chunking-friendly
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
        Parses a document file asynchronously using LlamaParse.

        Converts the document into markdown format and merges all parsed segments
        into a single string.

        Args:
            file_path (str): Path to the input document file.

        Returns:
            str: Extracted document content as a single markdown string.

        Raises:
            RegulatoryRAGException:
                - If the file does not exist.
                - If parsing fails due to API or processing errors.
        """

        # --------------------------
        # Validate file existence
        # --------------------------
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
            # LlamaParse API is synchronous → run in thread
            documents = await asyncio.to_thread(
                self.parser.load_data,
                file_path,
            )

            # Merge all returned segments into one string
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
        Combines multiple parsed document segments into a single string.

        Args:
            documents (List): List of document objects returned by LlamaParse.

        Returns:
            str: Concatenated text content from all valid document segments.
        """

        texts = []

        for doc in documents:
            # Each document object contains `.text`
            if hasattr(doc, "text") and doc.text:
                texts.append(doc.text)

        return "\n\n".join(texts)