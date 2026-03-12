#!/usr/bin/env python3
"""Rebase file paths stored in the DB from one prefix to another.

Useful when switching between host paths and Docker-mounted paths:

    python scripts/rebase_paths.py --from /Users/tulas/Projects/stripes-rag/docs --to /docs
    python scripts/rebase_paths.py --from /docs --to /Users/tulas/Projects/stripes-rag/docs
"""

from __future__ import annotations

import argparse
import sys

import psycopg

CHUNKS_TABLE = "document_chunks"


def get_connection_string() -> str:
    try:
        from stripes_rag.config import settings
        return settings.sync_connection_string
    except ImportError:
        return "postgresql://stripes:stripes@localhost:5432/stripes_rag"


def main():
    parser = argparse.ArgumentParser(description="Rebase file paths in the DB.")
    parser.add_argument("--from", dest="old_prefix", required=True, help="Old path prefix to replace")
    parser.add_argument("--to", dest="new_prefix", required=True, help="New path prefix")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    conn_str = get_connection_string()

    try:
        conn = psycopg.connect(conn_str)
    except Exception as e:
        print(f"Cannot connect to database: {e}", file=sys.stderr)
        print("Is PostgreSQL running? Try: docker compose up -d", file=sys.stderr)
        sys.exit(1)

    with conn:
        tracking_count = conn.execute(
            "SELECT COUNT(*) FROM file_tracking WHERE file_path LIKE %s || '%%'",
            (args.old_prefix,),
        ).fetchone()[0]
        chunks_count = conn.execute(
            f'SELECT COUNT(*) FROM "{CHUNKS_TABLE}" WHERE source_file LIKE %s || \'%%\'',
            (args.old_prefix,),
        ).fetchone()[0]

    if tracking_count == 0 and chunks_count == 0:
        print(f"No paths matching prefix '{args.old_prefix}' found.")
        sys.exit(0)

    print(f"Rebase: {args.old_prefix} -> {args.new_prefix}")
    print(f"  file_tracking:  {tracking_count} rows")
    print(f"  {CHUNKS_TABLE}: {chunks_count} rows")

    if not args.yes:
        answer = input("\nApply? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    with conn:
        conn.execute(
            "UPDATE file_tracking "
            "SET file_path = %s || SUBSTRING(file_path FROM LENGTH(%s) + 1) "
            "WHERE file_path LIKE %s || '%%'",
            (args.new_prefix, args.old_prefix, args.old_prefix),
        )
        conn.execute(
            f'UPDATE "{CHUNKS_TABLE}" '
            f"SET source_file = %s || SUBSTRING(source_file FROM LENGTH(%s) + 1) "
            f"WHERE source_file LIKE %s || '%%'",
            (args.new_prefix, args.old_prefix, args.old_prefix),
        )

    conn.close()
    print("Done. Paths rebased successfully.")


if __name__ == "__main__":
    main()
