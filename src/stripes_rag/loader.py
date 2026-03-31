"""Docling DocumentConverter wrapper for single-file conversion."""

from __future__ import annotations

import logging
from pathlib import Path

from docling.document_converter import DocumentConverter

logging.getLogger("RapidOCR").setLevel(logging.WARNING)
logging.getLogger("docling.datamodel.document").setLevel(logging.CRITICAL)
from docling_core.types.doc.document import DoclingDocument


_converter: DocumentConverter | None = None


def get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def convert_file(path: Path) -> DoclingDocument:
    """Convert a document file (PDF, DOCX, XLSX, PPTX, HTML, or MD) to a DoclingDocument.

    Raises on conversion failure.
    """
    converter = get_converter()
    result = converter.convert(path)
    return result.document
