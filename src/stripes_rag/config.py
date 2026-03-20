from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # PostgreSQL
    postgres_user: str = "stripes"
    postgres_password: str = "stripes"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "stripes_rag"

    # Embedding model
    #embedding_model: str = "google/embeddinggemma-300m"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 64
    embedding_server_url: str | None = None  # e.g. "http://localhost:8080"

    # Chunking
    chunk_max_tokens: int = 512

    # Indexing
    index_batch_size: int = 128
    max_file_size_mb: int = 50

    # Vector index type: "hnsw" (default, high recall) or "ivfflat" (faster build, large datasets)
    vector_index_type: Literal["hnsw", "ivfflat"] = "hnsw"

    # Reranker (optional)
    reranker_url: str | None = None          # e.g. "http://localhost:8081"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k_multiplier: int = 3       # retrieve k*N candidates, rerank to k

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
