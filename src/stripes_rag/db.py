"""PGEngine + vectorstore table init + vectorstore factory."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore

from stripes_rag.config import settings

TABLE_NAME = "document_chunks"
VECTOR_DIM = 1024

METADATA_COLUMNS = [
    Column("source_file", "TEXT", nullable=False),
    Column("page_numbers", "TEXT", nullable=True),
    Column("headings", "TEXT", nullable=True),
    Column("chunk_index", "INTEGER", nullable=False),
    Column("file_hash", "TEXT", nullable=False),
]


def get_engine() -> PGEngine:
    return PGEngine.from_connection_string(settings.async_connection_string)


def init_vectorstore_table(engine: PGEngine) -> None:
    from sqlalchemy import text

    try:
        engine.init_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=VECTOR_DIM,
            metadata_columns=METADATA_COLUMNS,
            overwrite_existing=False,
        )
    except Exception as e:
        if "DuplicateTableError" in type(e).__name__ or "already exists" in str(e):
            pass  # Table already exists, nothing to do
        else:
            raise

    # Ensure indexes exist (idempotent)
    async def _create_indexes():
        async with engine._pool.connect() as conn:
            index_type = settings.vector_index_type
            index_name = f"idx_{TABLE_NAME}_embedding"

            # Drop existing vector index if it exists (type may have changed)
            await conn.execute(text(
                f'DROP INDEX IF EXISTS {index_name}'
            ))

            # Create vector index based on configured type
            if index_type == "ivfflat":
                await conn.execute(text(
                    f'CREATE INDEX {index_name} '
                    f'ON "{TABLE_NAME}" USING ivfflat (embedding vector_cosine_ops) '
                    f'WITH (lists = 100)'
                ))
            else:  # hnsw (default)
                await conn.execute(text(
                    f'CREATE INDEX {index_name} '
                    f'ON "{TABLE_NAME}" USING hnsw (embedding vector_cosine_ops) '
                    f'WITH (m = 16, ef_construction = 64)'
                ))

            # B-tree index for source_file filtering and delete operations
            await conn.execute(text(
                f'CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_source_file '
                f'ON "{TABLE_NAME}" (source_file)'
            ))
            await conn.commit()

    engine._run_as_sync(_create_indexes())


def get_vectorstore(engine: PGEngine, embeddings: Embeddings) -> PGVectorStore:
    return PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name=TABLE_NAME,
        metadata_columns=[col.name for col in METADATA_COLUMNS],
    )


def delete_file_chunks(engine: PGEngine, source_file: str) -> None:
    """Delete all chunks for a given source file."""
    from sqlalchemy import text

    async def _delete():
        async with engine._pool.connect() as conn:
            await conn.execute(
                text(f'DELETE FROM "{TABLE_NAME}" WHERE source_file = :sf'),
                {"sf": source_file},
            )
            await conn.commit()

    engine._run_as_sync(_delete())
