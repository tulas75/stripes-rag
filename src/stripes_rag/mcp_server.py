"""MCP server exposing Stripes RAG search tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from stripes_rag.search import get_search_service

mcp = FastMCP("stripes-rag")


def _fmt_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


@mcp.tool()
def search(
    query: str,
    k: int = 5,
    source_file: str | None = None,
    rerank: bool | None = None,
) -> str:
    """Search indexed documents using semantic similarity.

    Args:
        query: The search query text.
        k: Number of results to return (default 5, max 50).
        source_file: Optional file path to search within a specific document.
        rerank: Use cross-encoder reranking. None means auto-detect from config.
    """
    service = get_search_service()
    response = service.search(
        query=query,
        k=min(k, 50),
        source_file=source_file,
        use_reranker=rerank,
    )

    if not response.results:
        return "No results found."

    parts = [f"Found {len(response.results)} results (reranked: {'yes' if response.reranked else 'no'})"]
    for i, r in enumerate(response.results, 1):
        meta = r.metadata
        source = meta.get("source_file", "unknown").rsplit("/", 1)[-1]
        pages = meta.get("page_numbers", "?")
        headings = meta.get("headings", "")

        lines = [f"\n--- Result {i} (score: {r.score:.4f}) ---"]
        lines.append(f"Source: {source}")
        lines.append(f"Pages: {pages}")
        if headings:
            lines.append(f"Headings: {headings}")
        lines.append(f"Content: {r.content[:500]}{'...' if len(r.content) > 500 else ''}")
        parts.append("\n".join(lines))

    return "\n".join(parts)


@mcp.tool()
def list_files(
    name: str | None = None,
    limit: int = 50,
) -> str:
    """List indexed files. Without arguments returns the first 50 files.

    Args:
        name: Optional search filter (case-insensitive, partial match).
        limit: Maximum number of files to return (default 50).
    """
    service = get_search_service()
    files = service.list_files(name=name, limit=min(limit, 500))

    if not files:
        return "No files found." if name else "No indexed files."

    parts = [f"{'Matching' if name else 'Indexed'} files ({len(files)}):"]
    for f in files:
        parts.append(
            f"  {f['file_name']}  [{f['status']}]  "
            f"{f['chunk_count']} chunks  {_fmt_size(f['file_size'])}"
        )

    return "\n".join(parts)


@mcp.tool()
def status() -> str:
    """Show indexing statistics: file counts, chunks, and storage size."""
    service = get_search_service()
    stats = service.status()

    return (
        f"Files indexed: {stats['file_count']}\n"
        f"Files pending: {stats['pending_count']}\n"
        f"Files errored: {stats['error_count']}\n"
        f"Total chunks: {stats['total_chunks']}\n"
        f"Total size: {_fmt_size(stats['total_size'])}\n"
        f"First indexed: {stats['first_indexed'] or 'N/A'}\n"
        f"Last updated: {stats['last_updated'] or 'N/A'}"
    )


def main(transport: str = "stdio", host: str = "0.0.0.0", port: int = 8001):
    """Entry point for ``stripes mcp`` command."""
    get_search_service()  # front-load model loading
    if transport == "sse":
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport="sse")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
