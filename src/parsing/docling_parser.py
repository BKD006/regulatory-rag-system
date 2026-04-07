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
    Parser class for extracting structured content from documents using Docling.

    Attributes:
        converter (DocumentConverter): Instance used to convert documents into structured format.
    """

    def __init__(self):
        """
        Initializes the DoclingDocumentParser with a DocumentConverter instance.

        Raises:
            RegulatoryRAGException: If initialization of the converter fails.
        """
        try:
            # Core Docling converter for parsing documents
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
        Parses a document file and extracts its content.

        Supports PDF files via Docling for structured parsing and falls back to
        plain text reading for TXT/MD files.

        Args:
            file_path (str): Path to the input document file.

        Returns:
            Tuple[str, Optional[object]]:
                - Extracted content as a string (markdown for PDFs, raw text otherwise).
                - Parsed Docling document object if PDF, else None.

        Raises:
            RegulatoryRAGException:
                - If the file does not exist.
                - If parsing fails due to any internal error.
        """

        # --------------------------
        # Validate file existence
        # --------------------------
        if not os.path.exists(file_path):
            raise RegulatoryRAGException(
                f"File not found: {file_path}",
                sys,
            )

        ext = Path(file_path).suffix.lower()

        try:
            # --------------------------
            # PDF parsing (Docling)
            # --------------------------
            if ext == ".pdf":

                log.info(
                    "docling_parsing_started",
                    file=file_path,
                )

                # Convert PDF → structured document
                result = self.converter.convert(file_path)

                # Export structured content to markdown
                content = result.document.export_to_markdown()

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