import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lookup: explicit STRIPES_ENV_FILE env var first, then CWD, then
# package source root.  The explicit var is the reliable path for MCP
# servers launched by Claude Desktop.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
_explicit = os.environ.get("STRIPES_ENV_FILE")
_env_files = (_explicit,) if _explicit else (_PACKAGE_ROOT / ".env", ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_files,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL
    postgres_user: str = "stripes"
    postgres_password: str = "stripes"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "stripes_rag"

    # Embedding provider: "local" (sentence-transformers) or "litellm" (TEI, DeepInfra, OpenAI, etc.)
    embedding_provider: Literal["local", "litellm"] = "local"
    embedding_dim: int | None = None  # auto-detected if None

    # Embedding model
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 64
    # LiteLLM settings
    embedding_api_base: str | None = None   # e.g. "http://localhost:8080/v1" or "https://api.deepinfra.com/v1/openai"
    embedding_api_key: str | None = None

    # Chunking
    tokenizer_model: str | None = None  # HF repo id for chunker tokenizer; defaults to embedding_model
    chunk_max_tokens: int = 512

    # Parser mode: "hiquality" (Docling always), "quality" (Docling + fast fallback), "fast" (fast always)
    parser_mode: Literal["hiquality", "quality", "fast"] = "quality"
    parser_page_threshold: int = 100   # PDF: fall back to fast parser above this page count
    parser_size_threshold_mb: int = 10  # DOCX/PPTX/HTML/MD: fall back above this file size

    # Indexing
    index_batch_size: int = 128
    max_file_size_mb: int = 50

    # Vector index type: "hnsw" (default, high recall) or "ivfflat" (faster build, large datasets)
    vector_index_type: Literal["hnsw", "ivfflat"] = "hnsw"

    # Reranker provider: "none" (disabled), "tei" (TEI server), "litellm" (any LiteLLM provider), "llamacpp" (llama.cpp server)
    reranker_provider: Literal["none", "tei", "litellm", "llamacpp"] = "none"

    # Reranker settings
    reranker_url: str | None = None          # e.g. "http://localhost:8081"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k_multiplier: int = 3       # retrieve k*N candidates, rerank to k
    reranker_timeout: float = 120.0           # seconds; CPU reranking can be slow
    reranker_batch_size: int = 64             # TEI max batch size

    @property
    def resolved_reranker_provider(self) -> Literal["none", "tei", "litellm", "llamacpp"]:
        """Auto-resolve provider from URL if left at default."""
        if self.reranker_provider == "none" and self.reranker_url:
            return "tei"
        return self.reranker_provider

    @property
    def async_connection_string(self) -> str:
        """For PGEngine (SQLAlchemy async via asyncpg)."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_connection_string(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
