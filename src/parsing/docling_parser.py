"""
Docling parser wrapper.

Used by ingest.py

Responsibilities:
- Parse PDF using Docling
- Return markdown + DoclingDocument
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


load_dotenv()


class DoclingDocumentParser:
    """
    Wrapper around Docling DocumentConverter.
    """

    def __init__(self):

        try:

            self.converter = DocumentConverter()

            log.info(
                "docling_parser_initialized"
            )

        except Exception as e:

            log.error(
                "docling_parser_init_failed",
                error=str(e),
            )

            raise RegulatoryRAGException(e, sys)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def parse(
        self,
        file_path: str,
    ) -> Tuple[str, Optional[object]]:
        """
        Parse document.

        Returns:
            content (markdown),
            docling_doc
        """

        if not os.path.exists(file_path):

            raise RegulatoryRAGException(
                f"File not found: {file_path}",
                sys,
            )

        ext = Path(file_path).suffix.lower()

        try:

            if ext == ".pdf":

                log.info(
                    "docling_parsing_started",
                    file=file_path,
                )

                result = self.converter.convert(
                    file_path
                )

                content = (
                    result.document.export_to_markdown()
                )

                docling_doc = result.document

                log.info(
                    "docling_parsing_completed",
                    file=file_path,
                    length=len(content),
                )

                return content, docling_doc

            # --------------------------
            # TXT / MD fallback
            # --------------------------

            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as f:

                text = f.read()

            return text, None

        except Exception as e:

            log.error(
                "docling_parsing_failed",
                file=file_path,
                error=str(e),
            )

            raise RegulatoryRAGException(
                e,
                sys,
            )