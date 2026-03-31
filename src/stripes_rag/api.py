"""FastAPI app for querying indexed documents and RAG chat."""

from __future__ import annotations

import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    model: str = "ollama_chat/qwen3.5:4b-q8_0"
    api_base: str | None = None
    api_key: str | None = None
    profile: str = "Classic RAG"
    language: str = "ENG"
    temperature: float = 0.3
    k: int = 5
    max_steps: int = 6
    use_reranker: bool = False


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
    description="Query API and RAG chat for Stripes RAG indexed documents.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Search / files / status / health (existing)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Chat / models / profiles (new)
# ---------------------------------------------------------------------------

@app.get("/models")
def models():
    from stripes_rag.models import list_models
    return list_models()


@app.get("/profiles")
def profiles():
    from stripes_rag.prompts import PROMPT_REGISTRY
    return [
        {"name": p.name, "description": p.description}
        for p in PROMPT_REGISTRY.values()
    ]


def _split_follow_ups(text: str) -> tuple[str, list[str]]:
    pattern = r"##\s*Follow[\-\s]?up\s+Questions?"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return text.strip(), []
    body = text[:match.start()].strip()
    follow_up_section = text[match.end():]
    follow_ups = [
        line.lstrip("-*\u2022 ").strip()
        for line in follow_up_section.splitlines()
        if line.strip() and line.strip().lstrip("-*\u2022 ")
    ]
    return body, follow_ups


def _run_chat(request: ChatRequest):
    """Generator that yields SSE events for the chat endpoint."""
    from smolagents import CodeAgent, LiteLLMModel, Tool

    from stripes_rag.db import get_engine, get_vectorstore
    from stripes_rag.embeddings import get_embeddings
    from stripes_rag.prompts import get_profile
    from stripes_rag.reranker import rerank

    yield f"event: status\ndata: Searching knowledge base...\n\n"

    # Build vectorstore (reuse search service's engine/embeddings)
    service = get_search_service()
    vectorstore = service._vectorstore

    # Build retriever tool (same as app.py)
    class RetrieverTool(Tool):
        name = "retriever"
        description = (
            "Searches the organizational knowledge base for relevant past projects, "
            "experiences, methodologies, and lessons learned."
        )
        inputs = {
            "query": {
                "type": "string",
                "description": "The search query.",
            }
        }
        output_type = "string"

        def __init__(self, vectorstore, k, use_reranker, **kwargs):
            super().__init__(**kwargs)
            self.vectorstore = vectorstore
            self.k = k
            self.use_reranker = use_reranker
            self.retrieved_chunks = []
            self.retrieval_time = 0.0
            self.retrieval_calls = 0

        def forward(self, query: str) -> str:
            from stripes_rag.config import settings as _settings

            k_fetch = self.k * _settings.reranker_top_k_multiplier if self.use_reranker else self.k
            t0 = time.perf_counter()
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k_fetch)

            reranked = False
            if self.use_reranker:
                for doc, distance in docs_with_scores:
                    doc.metadata["_vector_distance"] = distance
                docs_with_scores, reranked = rerank(query, docs_with_scores, top_k=self.k)

            self.retrieval_time += time.perf_counter() - t0
            self.retrieval_calls += 1

            result = "\nRetrieved documents:\n"
            for i, (doc, score) in enumerate(docs_with_scores):
                vector_sim = 1 - doc.metadata.pop("_vector_distance", 1 - score) if reranked else 1 - score
                reranker_score = score if reranked else None
                similarity = reranker_score if reranker_score is not None else vector_sim

                self.retrieved_chunks.append({
                    "source": Path(doc.metadata.get("source_file", "unknown")).name,
                    "similarity": round(vector_sim, 4),
                    "reranker_score": round(reranker_score, 4) if reranker_score is not None else None,
                    "headings": doc.metadata.get("headings", ""),
                    "pages": doc.metadata.get("page_numbers", ""),
                    "content": doc.page_content,
                })

                result += (
                    f"\n\n===== Document {i} (similarity: {similarity:.4f}) =====\n"
                    f"Source: {doc.metadata.get('source_file', 'unknown')}\n"
                    f"Similarity: {similarity:.4f}\n"
                )
                headings = doc.metadata.get("headings", "")
                pages = doc.metadata.get("page_numbers", "")
                if headings:
                    result += f"Headings: {headings}\n"
                if pages:
                    result += f"Pages: {pages}\n"
                result += f"\n{doc.page_content}"

            return result

    retriever_tool = RetrieverTool(vectorstore, k=request.k, use_reranker=request.use_reranker)

    profile = get_profile(request.profile)
    agent = CodeAgent(
        tools=[retriever_tool],
        model=LiteLLMModel(
            model_id=request.model,
            api_base=request.api_base,
            api_key=request.api_key,
            temperature=request.temperature,
        ),
        max_steps=request.max_steps,
        stream_outputs=False,
        additional_authorized_imports=["json"],
        verbosity_level=0,
    )

    yield f"event: status\ndata: Synthesizing answer...\n\n"

    t_start = time.perf_counter()
    try:
        answer = agent.run(
            request.message,
            additional_args=dict(
                additional_notes=profile.template.format(language=request.language)
            ),
        )
    except Exception as e:
        yield f"event: error\ndata: {json.dumps(str(e))}\n\n"
        return

    elapsed = time.perf_counter() - t_start

    # Deduplicate and sort chunks
    unique_chunks = []
    seen = set()
    for c in retriever_tool.retrieved_chunks:
        if c["content"] not in seen:
            seen.add(c["content"])
            unique_chunks.append(c)
    if any(c.get("reranker_score") is not None for c in unique_chunks):
        unique_chunks.sort(key=lambda c: c["reranker_score"] or 0, reverse=True)
    else:
        unique_chunks.sort(key=lambda c: c["similarity"], reverse=True)

    # Send chunks
    yield f"event: chunks\ndata: {json.dumps(unique_chunks)}\n\n"

    # Split answer and follow-ups
    body, follow_ups = _split_follow_ups(str(answer))
    yield f"event: answer\ndata: {json.dumps(body)}\n\n"

    if follow_ups:
        yield f"event: follow_ups\ndata: {json.dumps(follow_ups)}\n\n"

    # Stats
    token_usage = agent.monitor.get_total_token_counts()
    retr_time = retriever_tool.retrieval_time
    retr_calls = retriever_tool.retrieval_calls
    stats = {
        "elapsed": round(elapsed, 1),
        "retrieval_time": round(retr_time, 1),
        "retrieval_calls": retr_calls,
        "llm_time": round(elapsed - retr_time, 1),
        "input_tokens": token_usage.input_tokens,
        "output_tokens": token_usage.output_tokens,
    }
    yield f"event: stats\ndata: {json.dumps(stats)}\n\n"
    yield f"event: done\ndata: {{}}\n\n"


@app.post("/chat")
def chat(request: ChatRequest):
    return StreamingResponse(
        _run_chat(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Static files (frontend) — must be last to not shadow API routes
# ---------------------------------------------------------------------------

_static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
