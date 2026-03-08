"""HybridChunker config + metadata extraction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length is longer than the specified maximum sequence length",
)

from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument

from stripes_rag.config import settings

_chunker: HybridChunker | None = None


def get_chunker() -> HybridChunker:
    global _chunker
    if _chunker is None:
        tokenizer = HuggingFaceTokenizer.from_pretrained(
            settings.embedding_model,
            max_tokens=settings.chunk_max_tokens,
        )
        _chunker = HybridChunker(tokenizer=tokenizer)
    return _chunker


@dataclass
class ChunkResult:
    text: str
    headings: str | None
    page_numbers: str | None
    chunk_index: int


def chunk_document(
    doc: DoclingDocument, source_file: Path, file_hash: str
) -> list[ChunkResult]:
    """Chunk a DoclingDocument and extract metadata for each chunk."""
    chunker = get_chunker()
    results: list[ChunkResult] = []

    for idx, chunk in enumerate(chunker.chunk(doc)):
        enriched_text = chunker.serialize(chunk).replace("\x00", "")

        headings = None
        if chunk.meta.headings:
            headings = " > ".join(chunk.meta.headings)

        page_numbers = None
        pages: set[int] = set()
        for item in chunk.meta.doc_items:
            for prov in item.prov:
                pages.add(prov.page_no)
        if pages:
            page_numbers = ",".join(str(p) for p in sorted(pages))

        results.append(
            ChunkResult(
                text=enriched_text,
                headings=headings,
                page_numbers=page_numbers,
                chunk_index=idx,
            )
        )

    return results
