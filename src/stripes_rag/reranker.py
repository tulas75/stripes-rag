"""Optional cross-encoder reranker via TEI /rerank endpoint."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

from stripes_rag.config import settings

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    docs_with_scores: list[tuple[Document, float]],
    top_k: int,
) -> list[tuple[Document, float]]:
    """Rerank documents using a cross-encoder via TEI.

    If reranker is not configured or the service is unreachable,
    returns the input unchanged (graceful fallback).

    Returns list of (doc, reranker_score) tuples sorted by score descending.
    """
    if not settings.reranker_url:
        return docs_with_scores[:top_k]

    if not docs_with_scores:
        return docs_with_scores

    import httpx

    texts = [doc.page_content for doc, _ in docs_with_scores]

    try:
        resp = httpx.post(
            f"{settings.reranker_url}/rerank",
            json={"query": query, "texts": texts, "return_text": False},
            timeout=30.0,
        )
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.ConnectError) as e:
        logger.warning("Reranker unavailable (%s), falling back to vector scores", e)
        return docs_with_scores[:top_k]

    ranked = resp.json()
    # TEI returns [{"index": int, "score": float}, ...]
    ranked.sort(key=lambda r: r["score"], reverse=True)

    results: list[tuple[Document, float]] = []
    for r in ranked[:top_k]:
        idx = r["index"]
        doc, _ = docs_with_scores[idx]
        results.append((doc, r["score"]))

    return results
