#!/usr/bin/env python3
"""Show chunk size distribution from the document_chunks table."""

import psycopg
from stripes_rag.config import settings


def main():
    with psycopg.connect(settings.sync_connection_string) as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE length(content) = 0) as cat0,
                COUNT(*) FILTER (WHERE length(content) <= 2048) as cat1,
                COUNT(*) FILTER (WHERE length(content) > 2048 AND length(content) <= 3000) as cat2,
                COUNT(*) FILTER (WHERE length(content) > 3000 AND length(content) <= 4000) as cat3,
                COUNT(*) FILTER (WHERE length(content) > 4000) as cat4,
                AVG(length(content))::int as avg_chars,
                MIN(length(content)) as min_chars,
                MAX(length(content)) as max_chars
            FROM document_chunks
        """).fetchone()

    total, conn_zero, cat1, cat2, cat3, cat4, avg_c, min_c, max_c = row

    if total == 0:
        print("No chunks found.")
        return

    print(f"Total chunks: {total:,}")
    print()
    print(f"  {'Bucket':<40} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*40} {'-'*8}  {'-'*6}")
    for label, count in [
        ("0 chars (empty)", conn_zero),
        ("≤512 tokens (≤2048 chars)", cat1),
        ("512–750 tokens (2049–3000 chars)", cat2),
        ("750–1000 tokens (3001–4000 chars)", cat3),
        (">1000 tokens (>4000 chars)", cat4),
    ]:
        print(f"  {label:<40} {count:>8,}  {count/total*100:>5.1f}%")

    print()
    print(f"  avg: {avg_c:,} chars, min: {min_c:,}, max: {max_c:,}")


if __name__ == "__main__":
    main()
