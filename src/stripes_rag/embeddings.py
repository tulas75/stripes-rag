"""Embeddings wrapper — local sentence-transformers or LiteLLM (TEI, DeepInfra, OpenAI, etc.)."""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from stripes_rag.config import settings

logger = logging.getLogger(__name__)


class _LiteLLMEmbeddings(Embeddings):
    """Wraps litellm.embedding() for any OpenAI-compatible provider."""

    def __init__(
        self,
        model: str,
        batch_size: int,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._extra: dict = {}
        if api_base:
            self._extra["api_base"] = api_base
        if api_key:
            self._extra["api_key"] = api_key

    def _embed_single(self, text: str) -> list[float]:
        """Embed a single text with retry: prepend a prefix on failure.

        Ollama's bge-m3 produces NaN for certain token combinations.
        Prepending "passage: " changes the tokenization enough to work around
        the bug while preserving semantic meaning.
        """
        import litellm

        try:
            resp = litellm.embedding(
                model=self._model, input=[text], encoding_format="float", **self._extra
            )
            return resp.data[0]["embedding"]
        except Exception:
            prefixed = f"passage: {text}"
            logger.warning("Embedding failed, retrying with prefix: %s", repr(text[:80]))
            resp = litellm.embedding(
                model=self._model, input=[prefixed], encoding_format="float", **self._extra
            )
            return resp.data[0]["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import litellm

        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                resp = litellm.embedding(
                    model=self._model, input=batch, encoding_format="float", **self._extra
                )
                results.extend([item["embedding"] for item in resp.data])
            except Exception:
                # Batch failed — retry items individually with prefix fallback
                for text in batch:
                    results.append(self._embed_single(text))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> Embeddings:
    if settings.embedding_provider == "litellm":
        return _LiteLLMEmbeddings(
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            api_base=settings.embedding_api_base,
            api_key=settings.embedding_api_key,
        )

    # "local" — sentence-transformers (default)
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


def get_embedding_dim(embeddings: Embeddings) -> int:
    """Return the embedding dimension, probing the model if not configured."""
    if settings.embedding_dim is not None:
        return settings.embedding_dim
    vec = embeddings.embed_query("dimension probe")
    return len(vec)
