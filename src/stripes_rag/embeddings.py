"""Embeddings wrapper — local sentence-transformers or remote TEI server."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from stripes_rag.config import settings


def get_embeddings() -> Embeddings:
    if settings.embedding_server_url:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        return HuggingFaceEndpointEmbeddings(
            model=settings.embedding_server_url,
        )

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
