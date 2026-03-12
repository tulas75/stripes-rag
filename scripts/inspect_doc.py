#!/usr/bin/env python3
"""Inspect what Docling extracts from a PDF, DOCX, XLSX, PPTX, HTML, or Markdown file.

Converts the document and prints the full markdown output plus a summary
of tables, pictures, and pages found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.labels import DocItemLabel


def _build_converter(describe_images: bool) -> DocumentConverter:
    """Build a DocumentConverter, optionally with image description enabled."""
    if not describe_images:
        return DocumentConverter()

    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        PictureDescriptionVlmEngineOptions,
    )
    from docling.document_converter import InputFormat, PdfFormatOption

    vlm_options = PictureDescriptionVlmEngineOptions.from_preset(
        "smolvlm",
        picture_area_threshold=0.0,
    )
    pipeline_options = PdfPipelineOptions(
        do_picture_description=True,
        picture_description_options=vlm_options,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect what Docling extracts from a PDF, DOCX, XLSX, PPTX, HTML, or Markdown file.",
    )
    parser.add_argument("file", type=Path, help="PDF, DOCX, XLSX, PPTX, HTML, or Markdown file to inspect")
    parser.add_argument(
        "-d", "--describe-images",
        action="store_true",
        help="Generate image descriptions using SmolVLM (downloads ~500MB model on first run)",
    )
    args = parser.parse_args()

    path: Path = args.file
    if not path.is_file():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {path.name} ...")
    converter = _build_converter(args.describe_images)
    result = converter.convert(path)
    doc = result.document

    # --- Markdown output ---
    md = doc.export_to_markdown()
    print("\n" + "=" * 60)
    print("MARKDOWN OUTPUT")
    print("=" * 60 + "\n")
    print(md)

    # --- Summary ---
    tables = [item for item, _level in doc.iterate_items() if item.label == DocItemLabel.TABLE]
    pictures = [item for item, _level in doc.iterate_items() if item.label == DocItemLabel.PICTURE]
    pages = doc.pages or {}

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Pages:    {len(pages)}")
    print(f"  Tables:   {len(tables)}")
    print(f"  Pictures: {len(pictures)}")


if __name__ == "__main__":
    main()
