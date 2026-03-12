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

# bge-m3 max sequence length is 8192 tokens; ~4 chars/token is a safe estimate.
# Oversized chunks (e.g. large tables emitted as a single block by HybridChunker)
# get split into sub-chunks of this size so no content is lost.
_MAX_CHUNK_CHARS = 8192 * 4


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

        # If the chunk is too large for the embedding model, split it into
        # sub-chunks so no content is lost (HybridChunker can emit oversized
        # atomic blocks like large tables).
        if len(enriched_text) > _MAX_CHUNK_CHARS:
            for start in range(0, len(enriched_text), _MAX_CHUNK_CHARS):
                results.append(
                    ChunkResult(
                        text=enriched_text[start:start + _MAX_CHUNK_CHARS],
                        headings=headings,
                        page_numbers=page_numbers,
                        chunk_index=idx,
                    )
                )
        else:
            results.append(
                ChunkResult(
                    text=enriched_text,
                    headings=headings,
                    page_numbers=page_numbers,
                    chunk_index=idx,
                )
            )

    return results
