"""Orchestrates load -> chunk -> embed -> store with pipeline parallelism.

Multiple worker processes parse+chunk files in parallel.  The main thread
consumes results and embeds+stores them serially.  A sliding window of
submitted futures (workers+1) limits memory.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from stripes_rag.db import (
    TABLE_NAME,
    delete_file_chunks,
    get_engine,
    get_vectorstore,
    init_vectorstore_table,
)
from stripes_rag.embeddings import get_embeddings
from stripes_rag.tracker import FileTracker, _sha256

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


@dataclass
class FileResult:
    path: Path
    status: str  # "indexed", "skipped", "error"
    chunks: int = 0
    error: str | None = None
    parse_time: float = 0.0
    chunk_time: float = 0.0
    embed_time: float = 0.0


def discover_files(directory: Path, recursive: bool = False) -> list[Path]:
    """Find all PDF/DOCX files in directory, skipping macOS resource forks."""
    files: list[Path] = []
    if recursive:
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(directory.glob(f"*{ext}"))
    # Filter out macOS ._ resource fork files
    files = [f for f in files if not f.name.startswith("._")]
    return sorted(files)


# ---------------------------------------------------------------------------
# Worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _parse_and_chunk(file_path: Path) -> dict:
    """Parse + chunk a single file.  Returns serializable data."""
    import time

    from stripes_rag.chunker import chunk_document
    from stripes_rag.loader import convert_file
    from stripes_rag.tracker import _sha256

    resolved = str(file_path.resolve())
    file_hash = _sha256(file_path)

    t0 = time.perf_counter()
    doc = convert_file(file_path)
    parse_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks = chunk_document(doc, file_path, file_hash)
    del doc

    lc_doc_data = [
        {
            "page_content": c.text,
            "metadata": {
                "source_file": resolved,
                "page_numbers": c.page_numbers,
                "headings": c.headings,
                "chunk_index": c.chunk_index,
                "file_hash": file_hash,
            },
        }
        for c in chunks
    ]
    chunk_time = time.perf_counter() - t0

    return {
        "file_path": file_path,
        "resolved": resolved,
        "lc_doc_data": lc_doc_data,
        "parse_time": parse_time,
        "chunk_time": chunk_time,
        "num_chunks": len(chunks),
    }


# ---------------------------------------------------------------------------
# Pipeline (main process)
# ---------------------------------------------------------------------------

def _embed_and_store(
    data: dict,
    engine,
    vectorstore,
    tracker: FileTracker,
) -> FileResult:
    """Embed + store results from a worker, update tracker."""
    file_path = data["file_path"]
    resolved = data["resolved"]
    lc_doc_data = data["lc_doc_data"]
    parse_time = data["parse_time"]
    chunk_time = data["chunk_time"]
    num_chunks = data["num_chunks"]

    if not lc_doc_data:
        tracker.upsert_record(file_path, 0)
        return FileResult(
            path=file_path, status="indexed", chunks=0,
            parse_time=parse_time, chunk_time=chunk_time,
        )

    lc_docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in lc_doc_data
    ]

    t0 = time.perf_counter()
    delete_file_chunks(engine, resolved)
    vectorstore.add_documents(lc_docs)
    embed_time = time.perf_counter() - t0

    tracker.upsert_record(file_path, num_chunks)

    return FileResult(
        path=file_path, status="indexed", chunks=num_chunks,
        parse_time=parse_time, chunk_time=chunk_time, embed_time=embed_time,
    )


def _run_pipeline(
    files_to_process: list[Path],
    engine,
    vectorstore,
    tracker: FileTracker,
    workers: int = 2,
    progress_callback=None,
    result_callback=None,
) -> list[FileResult]:
    """Process files with parallel parse+chunk and serial embed+store."""
    if not files_to_process:
        return []

    results: list[FileResult] = []
    file_iter = iter(files_to_process)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Sliding window: keep workers+1 futures in flight
        pending: dict = {}
        for _ in range(min(workers + 1, len(files_to_process))):
            fp = next(file_iter)
            pending[pool.submit(_parse_and_chunk, fp)] = fp

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)

            for future in done:
                file_path = pending.pop(future)

                if progress_callback:
                    progress_callback(file_path, len(results), len(files_to_process))

                try:
                    data = future.result()
                    result = _embed_and_store(data, engine, vectorstore, tracker)
                except Exception as e:
                    try:
                        tracker.upsert_error(file_path, str(e))
                    except Exception:
                        pass  # DB may be down; still record the error locally
                    result = FileResult(
                        path=file_path, status="error", error=str(e),
                    )

                results.append(result)
                if result_callback:
                    result_callback(result)

                # Submit next file to keep the window full
                try:
                    fp = next(file_iter)
                    pending[pool.submit(_parse_and_chunk, fp)] = fp
                except StopIteration:
                    pass

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_directory(
    directory: Path,
    recursive: bool = False,
    force: bool = False,
    retry_errors: bool = False,
    workers: int = 4,
    progress_callback=None,
    result_callback=None,
) -> list[FileResult]:
    """Index all PDF/DOCX files in a directory.

    Uses process-pool pipeline parallelism for throughput.
    Per-file atomic commits for crash-safe resumability.
    """
    engine = get_engine()
    init_vectorstore_table(engine)
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(engine, embeddings)
    tracker = FileTracker()

    files = discover_files(directory, recursive)
    results: list[FileResult] = []
    files_to_process: list[Path] = []

    for file_path in files:
        if not force and not tracker.needs_indexing(file_path, retry_errors=retry_errors):
            result = FileResult(path=file_path, status="skipped")
            results.append(result)
            if result_callback:
                result_callback(result)
        else:
            files_to_process.append(file_path)

    pipeline_results = _run_pipeline(
        files_to_process, engine, vectorstore, tracker,
        workers=workers,
        progress_callback=progress_callback,
        result_callback=result_callback,
    )
    results.extend(pipeline_results)

    return results


def reindex_all(workers: int = 4, progress_callback=None) -> list[FileResult]:
    """Re-index all previously tracked files."""
    engine = get_engine()
    init_vectorstore_table(engine)
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(engine, embeddings)
    tracker = FileTracker()

    paths = tracker.tracked_paths()
    results: list[FileResult] = []
    files_to_process: list[Path] = []

    for path_str in paths:
        file_path = Path(path_str)
        if not file_path.exists():
            tracker.upsert_error(file_path, "File not found")
            results.append(
                FileResult(path=file_path, status="error", error="File not found")
            )
            continue
        files_to_process.append(file_path)

    pipeline_results = _run_pipeline(
        files_to_process, engine, vectorstore, tracker,
        workers=workers,
        progress_callback=progress_callback,
    )
    results.extend(pipeline_results)

    return results
