"""FastAPI app for querying indexed documents."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from stripes_rag.config import settings
from stripes_rag.reranker import is_reranker_available
from stripes_rag.search import get_search_service


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=50)
    source_file: str | None = None
    rerank: bool | None = None


class SearchResultItem(BaseModel):
    content: str
    score: float
    vector_similarity: float
    reranker_score: float | None
    source_file: str
    file_name: str
    page_numbers: str | None
    headings: str | None
    chunk_index: int
    file_hash: str


class SearchResponseModel(BaseModel):
    query: str
    k: int
    reranked: bool
    results: list[SearchResultItem]


class FileItem(BaseModel):
    file_path: str
    file_name: str
    status: str
    file_size: int
    chunk_count: int
    indexed_at: str | None
    updated_at: str | None


class StatusResponse(BaseModel):
    file_count: int
    pending_count: int
    error_count: int
    total_chunks: int
    total_size: int
    first_indexed: str | None
    last_updated: str | None
    reranker_available: bool
    embedding_provider: str
    embedding_model: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_search_service()  # front-load model loading
    yield


app = FastAPI(
    title="Stripes RAG API",
    version="0.1.0",
    description="Read-only query API for Stripes RAG indexed documents.",
    lifespan=lifespan,
)


@app.post("/search", response_model=SearchResponseModel)
def search(request: SearchRequest):
    service = get_search_service()
    response = service.search(
        query=request.query,
        k=request.k,
        source_file=request.source_file,
        use_reranker=request.rerank,
    )
    return SearchResponseModel(
        query=response.query,
        k=response.k,
        reranked=response.reranked,
        results=[
            SearchResultItem(
                content=r.content,
                score=r.score,
                vector_similarity=r.vector_similarity,
                reranker_score=r.reranker_score,
                source_file=r.metadata.get("source_file", ""),
                file_name=Path(r.metadata.get("source_file", "unknown")).name,
                page_numbers=r.metadata.get("page_numbers"),
                headings=r.metadata.get("headings"),
                chunk_index=r.metadata.get("chunk_index", 0),
                file_hash=r.metadata.get("file_hash", ""),
            )
            for r in response.results
        ],
    )


@app.get("/files", response_model=list[FileItem])
def list_files(
    name: str | None = Query(default=None, description="Filter files by name (ILIKE search)"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    service = get_search_service()
    return service.list_files(name=name, limit=limit, offset=offset)


@app.get("/status", response_model=StatusResponse)
def status():
    service = get_search_service()
    stats = service.status()
    return StatusResponse(
        file_count=stats["file_count"],
        pending_count=stats["pending_count"],
        error_count=stats["error_count"],
        total_chunks=stats["total_chunks"],
        total_size=stats["total_size"],
        first_indexed=str(stats["first_indexed"]) if stats["first_indexed"] else None,
        last_updated=str(stats["last_updated"]) if stats["last_updated"] else None,
        reranker_available=is_reranker_available(),
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
