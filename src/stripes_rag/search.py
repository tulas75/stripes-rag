"""Shared query service used by CLI, API, and MCP server."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from stripes_rag.config import settings
from stripes_rag.db import get_engine, get_vectorstore
from stripes_rag.embeddings import get_embeddings
from stripes_rag.reranker import is_reranker_available, rerank
from stripes_rag.tracker import FileTracker


@dataclass
class SearchResult:
    content: str
    score: float
    vector_similarity: float
    reranker_score: float | None
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResponse:
    results: list[SearchResult]
    reranked: bool
    query: str
    k: int


class SearchService:
    """Holds heavyweight resources (engine, embeddings, vectorstore) and
    exposes query methods.  Intended to be created once per process."""

    def __init__(self) -> None:
        engine = get_engine()
        embeddings = get_embeddings()
        self._vectorstore = get_vectorstore(engine, embeddings)

    def search(
        self,
        query: str,
        k: int = 5,
        source_file: str | None = None,
        use_reranker: bool | None = None,
    ) -> SearchResponse:
        filter_dict = None
        if source_file:
            resolved = str(Path(source_file).resolve())
            filter_dict = {"source_file": resolved}

        reranker_on = use_reranker if use_reranker is not None else is_reranker_available()
        k_fetch = k * settings.reranker_top_k_multiplier if reranker_on else k

        raw = self._vectorstore.similarity_search_with_score(
            query, k=k_fetch, filter=filter_dict,
        )

        reranked = False
        if reranker_on and raw:
            for doc, distance in raw:
                doc.metadata["_vector_distance"] = distance
            raw, reranked = rerank(query, raw, top_k=k)

        results: list[SearchResult] = []
        for doc, score in raw:
            meta = dict(doc.metadata)
            vector_distance = meta.pop("_vector_distance", 1 - score)
            vector_sim = 1 - vector_distance if reranked else 1 - score

            results.append(SearchResult(
                content=doc.page_content,
                score=score if reranked else vector_sim,
                vector_similarity=vector_sim,
                reranker_score=score if reranked else None,
                metadata=meta,
            ))

        return SearchResponse(results=results, reranked=reranked, query=query, k=k)

    def list_files(
        self,
        name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        tracker = FileTracker()
        if name:
            records = tracker.find_by_name(name)
            return [
                {
                    "file_path": r.file_path,
                    "file_name": Path(r.file_path).name,
                    "status": r.status,
                    "file_size": r.file_size,
                    "chunk_count": r.chunk_count,
                    "indexed_at": r.indexed_at.isoformat() if r.indexed_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in records[offset:offset + limit]
            ]
        records = tracker.all_records(limit=limit, offset=offset)
        return [
            {
                "file_path": r.file_path,
                "file_name": Path(r.file_path).name,
                "status": r.status,
                "file_size": r.file_size,
                "chunk_count": r.chunk_count,
                "indexed_at": r.indexed_at.isoformat() if r.indexed_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            }
            for r in records
        ]

    def status(self) -> dict:
        tracker = FileTracker()
        return tracker.stats()


_service: SearchService | None = None


def get_search_service() -> SearchService:
    global _service
    if _service is None:
        _service = SearchService()
    return _service
