"""HuggingFaceEmbeddings wrapper configured for EmbeddingGemma on MPS."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from stripes_rag.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"batch_size": settings.embedding_batch_size},
    )
