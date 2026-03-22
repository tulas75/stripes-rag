"""Optional cross-encoder reranker via TEI or LiteLLM."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

from stripes_rag.config import settings

logger = logging.getLogger(__name__)


def is_reranker_available() -> bool:
    """Check whether a reranker is configured (TEI or LiteLLM)."""
    return settings.resolved_reranker_provider in ("tei", "litellm")


def _rerank_tei(
    query: str,
    docs_with_scores: list[tuple[Document, float]],
    top_k: int,
) -> tuple[list[tuple[Document, float]], bool]:
    """Rerank via TEI /rerank HTTP endpoint."""
    import httpx

    texts = [doc.page_content for doc, _ in docs_with_scores]
    batch_size = settings.reranker_batch_size

    url = f"{settings.reranker_url}/rerank"
    logger.warning("Reranker POST %s with %d texts (batch_size=%d)", url, len(texts), batch_size)

    all_scored: list[dict] = []
    try:
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            resp = httpx.post(
                url,
                json={"query": query, "texts": batch, "return_text": False},
                timeout=settings.reranker_timeout,
            )
            resp.raise_for_status()
            for r in resp.json():
                all_scored.append({"index": offset + r["index"], "score": r["score"]})
    except (httpx.HTTPError, httpx.ConnectError) as e:
        logger.warning("Reranker unavailable (%s), falling back to vector scores", e)
        return docs_with_scores[:top_k], False

    all_scored.sort(key=lambda r: r["score"], reverse=True)

    results: list[tuple[Document, float]] = []
    for r in all_scored[:top_k]:
        doc, _ = docs_with_scores[r["index"]]
        results.append((doc, r["score"]))

    return results, True


def _rerank_litellm(
    query: str,
    docs_with_scores: list[tuple[Document, float]],
    top_k: int,
) -> tuple[list[tuple[Document, float]], bool]:
    """Rerank via LiteLLM (DeepInfra, Cohere, etc.)."""
    import litellm

    texts = [doc.page_content for doc, _ in docs_with_scores]
    logger.warning("LiteLLM rerank model=%s with %d texts", settings.reranker_model, len(texts))

    try:
        resp = litellm.rerank(
            model=settings.reranker_model,
            query=query,
            documents=texts,
            top_n=top_k,
        )
    except Exception as e:
        logger.warning("LiteLLM reranker failed (%s), falling back to vector scores", e)
        return docs_with_scores[:top_k], False

    results: list[tuple[Document, float]] = []
    for r in resp.results:
        idx = r["index"] if isinstance(r, dict) else r.index
        score = r["relevance_score"] if isinstance(r, dict) else r.relevance_score
        doc, _ = docs_with_scores[idx]
        results.append((doc, score))

    return results, True


def rerank(
    query: str,
    docs_with_scores: list[tuple[Document, float]],
    top_k: int,
) -> tuple[list[tuple[Document, float]], bool]:
    """Rerank documents using the configured provider.

    If reranker is not configured or the service is unreachable,
    returns the input unchanged (graceful fallback).

    Returns (results, reranked) where reranked is True only if
    cross-encoder scoring actually succeeded.
    """
    provider = settings.resolved_reranker_provider

    if provider == "none":
        return docs_with_scores[:top_k], False

    if not docs_with_scores:
        return docs_with_scores, False

    if provider == "litellm":
        return _rerank_litellm(query, docs_with_scores, top_k)

    return _rerank_tei(query, docs_with_scores, top_k)
