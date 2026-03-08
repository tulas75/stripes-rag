"""File tracking via SHA-256 hashes in PostgreSQL for crash-safe resumability."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from stripes_rag.config import settings

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS file_tracking (
    file_path   TEXT PRIMARY KEY,
    file_hash   TEXT NOT NULL,
    file_size   BIGINT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    status      TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT
);
"""

MIGRATE_ADD_STATUS = [
    "ALTER TABLE file_tracking ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'indexed'",
    "ALTER TABLE file_tracking ADD COLUMN IF NOT EXISTS error_message TEXT",
]


@dataclass
class FileRecord:
    file_path: str
    file_hash: str
    file_size: int
    chunk_count: int
    indexed_at: datetime
    updated_at: datetime
    status: str = "indexed"
    error_message: str | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):  # 1MB blocks
            h.update(block)
    return h.hexdigest()


class FileTracker:
    def __init__(self, conn_string: str | None = None):
        self._conn_string = conn_string or settings.sync_connection_string
        self._ensure_table()

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._conn_string)

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_TABLE)
            for stmt in MIGRATE_ADD_STATUS:
                conn.execute(stmt)

    def needs_indexing(self, path: Path, *, retry_errors: bool = False) -> bool:
        """Return True if file is new or changed since last indexing.

        Files with status='error' are skipped unless retry_errors is True.
        """
        current_hash = _sha256(path)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_hash, status FROM file_tracking WHERE file_path = %s",
                (str(path.resolve()),),
            ).fetchone()
        if row is None:
            return True
        stored_hash, status = row
        if status == "error":
            return retry_errors
        return stored_hash != current_hash

    def upsert_record(self, path: Path, chunk_count: int) -> None:
        """Record a successfully indexed file."""
        resolved = str(path.resolve())
        file_hash = _sha256(path)
        file_size = path.stat().st_size
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO file_tracking
                    (file_path, file_hash, file_size, chunk_count, indexed_at, updated_at, status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, 'indexed', NULL)
                ON CONFLICT (file_path) DO UPDATE SET
                    file_hash = EXCLUDED.file_hash,
                    file_size = EXCLUDED.file_size,
                    chunk_count = EXCLUDED.chunk_count,
                    updated_at = EXCLUDED.updated_at,
                    status = 'indexed',
                    error_message = NULL
                """,
                (resolved, file_hash, file_size, chunk_count, now, now),
            )

    def upsert_error(self, path: Path, error_message: str) -> None:
        """Record a file that failed to index."""
        resolved = str(path.resolve())
        try:
            file_hash = _sha256(path)
            file_size = path.stat().st_size
        except OSError:
            file_hash = ""
            file_size = 0
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO file_tracking
                    (file_path, file_hash, file_size, chunk_count, indexed_at, updated_at, status, error_message)
                VALUES (%s, %s, %s, 0, %s, %s, 'error', %s)
                ON CONFLICT (file_path) DO UPDATE SET
                    file_hash = EXCLUDED.file_hash,
                    file_size = EXCLUDED.file_size,
                    chunk_count = 0,
                    updated_at = EXCLUDED.updated_at,
                    status = 'error',
                    error_message = EXCLUDED.error_message
                """,
                (resolved, file_hash, file_size, now, now, error_message),
            )

    def get_record(self, path: Path) -> FileRecord | None:
        resolved = str(path.resolve())
        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_path, file_hash, file_size, chunk_count, indexed_at, updated_at, status, error_message "
                "FROM file_tracking WHERE file_path = %s",
                (resolved,),
            ).fetchone()
        if row is None:
            return None
        return FileRecord(*row)

    def all_records(self) -> list[FileRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file_path, file_hash, file_size, chunk_count, indexed_at, updated_at, status, error_message "
                "FROM file_tracking ORDER BY updated_at DESC"
            ).fetchall()
        return [FileRecord(*r) for r in rows]

    def error_records(self) -> list[FileRecord]:
        """Return all files with status='error'."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file_path, file_hash, file_size, chunk_count, indexed_at, updated_at, status, error_message "
                "FROM file_tracking WHERE status = 'error' ORDER BY updated_at DESC"
            ).fetchall()
        return [FileRecord(*r) for r in rows]

    def stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'indexed') as file_count,
                    COALESCE(SUM(chunk_count) FILTER (WHERE status = 'indexed'), 0) as total_chunks,
                    COALESCE(SUM(file_size) FILTER (WHERE status = 'indexed'), 0) as total_size,
                    MIN(indexed_at) as first_indexed,
                    MAX(updated_at) as last_updated,
                    COUNT(*) FILTER (WHERE status = 'error') as error_count
                FROM file_tracking
                """
            ).fetchone()
        return {
            "file_count": row[0],
            "total_chunks": row[1],
            "total_size": row[2],
            "first_indexed": row[3],
            "last_updated": row[4],
            "error_count": row[5],
        }

    def tracked_paths(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file_path FROM file_tracking WHERE status = 'indexed' ORDER BY file_path"
            ).fetchall()
        return [r[0] for r in rows]

    def delete_record(self, file_path: str) -> None:
        """Delete a tracking record by file_path."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM file_tracking WHERE file_path = %s",
                (file_path,),
            )

    def find_by_name(self, name: str) -> list[FileRecord]:
        """Find records where the file path contains the given name."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file_path, file_hash, file_size, chunk_count, indexed_at, updated_at, status, error_message "
                "FROM file_tracking WHERE file_path ILIKE %s ORDER BY file_path",
                (f"%{name}%",),
            ).fetchall()
        return [FileRecord(*r) for r in rows]

