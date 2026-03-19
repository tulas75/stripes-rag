"""Embeddings wrapper — local sentence-transformers or remote TEI server."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from stripes_rag.config import settings


class _BatchedEndpointEmbeddings(Embeddings):
    """Wraps HuggingFaceEndpointEmbeddings with client-side batching."""

    def __init__(self, inner: Embeddings, batch_size: int) -> None:
        self._inner = inner
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            results.extend(self._inner.embed_documents(texts[i : i + self._batch_size]))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


def get_embeddings() -> Embeddings:
    if settings.embedding_server_url:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        inner = HuggingFaceEndpointEmbeddings(
            model=settings.embedding_server_url,
        )
        return _BatchedEndpointEmbeddings(inner, settings.embedding_batch_size)

    # Local sentence-transformers (default)
    from langchain_huggingface import HuggingFaceEmbeddings

    kwargs = {
        "model_name": settings.embedding_model,
        "model_kwargs": {
            "device": settings.embedding_device,
            "local_files_only": True,
        },
        "encode_kwargs": {"batch_size": settings.embedding_batch_size},
    }
    try:
        return HuggingFaceEmbeddings(**kwargs)
    except OSError:
        # Model not cached locally yet — fall back to downloading
        kwargs["model_kwargs"]["local_files_only"] = False
        return HuggingFaceEmbeddings(**kwargs)
