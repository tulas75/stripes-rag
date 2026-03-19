"""Click CLI with Rich progress bars and styled output."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from stripes_rag.config import settings

console = Console()


@click.group()
def cli():
    """Stripes RAG - Document indexing & semantic search."""
    pass


@cli.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--recursive", "-r", is_flag=True, help="Scan subdirectories")
@click.option("--force", "-f", is_flag=True, help="Re-index all files regardless of changes")
@click.option("--retry-errors", is_flag=True, help="Retry previously failed files")
@click.option("-j", "--workers", default=2, show_default=True, help="Parallel parse workers")
@click.option("--device", type=click.Choice(["cpu", "mps", "cuda"]), default=None,
              help="Embedding device (overrides EMBEDDING_DEVICE env var)")
@click.option("--skip-reindex", is_flag=True, help="Skip rebuilding the vector index after indexing")
def index(directory: Path | None, recursive: bool, force: bool, retry_errors: bool, workers: int, device: str | None, skip_reindex: bool):
    """Index documents from a directory, or resume processing pending files.

    With DIRECTORY: scan + process (register new/changed files, then index them).
    Without DIRECTORY: resume processing any pending files from a previous run.
    """
    if device:
        settings.embedding_device = device

    from stripes_rag.indexer import index_pending, setup_pipeline
    from stripes_rag.tracker import FileTracker

    rebuild = not skip_reindex
    rebuild_status = None

    def on_rebuild(event):
        nonlocal rebuild_status
        if event == "start":
            rebuild_status = console.status("[bold cyan]Rebuilding vector index...[/bold cyan]")
            rebuild_status.start()
        elif event == "done" and rebuild_status:
            rebuild_status.stop()

    if directory is None:
        # Resume mode: process pending files only
        tracker = FileTracker()
        pending = tracker.pending_files()
        if not pending:
            console.print("[green]No pending files to process.[/green]")
            return

        console.print(f"Resuming [bold]{len(pending)}[/bold] pending files")

        with console.status("[bold cyan]Initializing pipeline...[/bold cyan]"):
            engine, _, vectorstore = setup_pipeline()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing...", total=len(pending))

            def on_progress(file_path, done, total):
                progress.update(task, description=f"[cyan]{file_path.name}[/cyan]")

            def on_result(result):
                progress.advance(task)

            results = index_pending(
                workers=workers,
                progress_callback=on_progress,
                result_callback=on_result,
                engine=engine,
                vectorstore=vectorstore,
                rebuild_index=rebuild,
                rebuild_callback=on_rebuild,
            )
            progress.update(task, completed=len(pending))

        _print_results(results)
        return

    # Scan + process mode
    from stripes_rag.indexer import discover_files

    files = discover_files(directory, recursive)
    if not files:
        console.print("[yellow]No supported document files found.[/yellow]")
        return

    console.print(f"Found [bold]{len(files)}[/bold] files on disk")

    with console.status("[bold cyan]Scanning for changes...[/bold cyan]"):
        tracker = FileTracker()
        new_pending, already_pending, skipped_count = tracker.register_pending(
            files, force=force, retry_errors=retry_errors,
        )

    total_pending = new_pending + already_pending

    if not total_pending:
        console.print("[green]All files are up to date.[/green]")
        return

    parts: list[str] = []
    if skipped_count:
        parts.append(f"{skipped_count} up to date")
    if new_pending:
        parts.append(f"{new_pending} to process")
    if already_pending:
        parts.append(f"{already_pending} resuming from previous run")
    console.print(f"  [dim]{', '.join(parts)}[/dim]")

    with console.status("[bold cyan]Initializing pipeline...[/bold cyan]"):
        engine, _, vectorstore = setup_pipeline()

    total = total_pending + skipped_count

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing...", total=total, completed=skipped_count)

        def on_progress(file_path, done, total):
            progress.update(task, description=f"[cyan]{file_path.name}[/cyan]")

        def on_result(result):
            progress.advance(task)

        results = index_pending(
            workers=workers,
            progress_callback=on_progress,
            result_callback=on_result,
            engine=engine,
            vectorstore=vectorstore,
            rebuild_index=rebuild,
            rebuild_callback=on_rebuild,
        )
        progress.update(task, completed=total)

    _print_results(results)


@cli.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--recursive", "-r", is_flag=True, help="Scan subdirectories")
@click.option("--force", "-f", is_flag=True, help="Re-queue all files regardless of changes")
@click.option("--retry-errors", is_flag=True, help="Re-queue previously failed files")
@click.option("--dry-run", is_flag=True, help="Show what would be queued without making changes")
def scan(directory: Path, recursive: bool, force: bool, retry_errors: bool, dry_run: bool):
    """Scan a directory and register files as pending for indexing.

    Use 'stripes index' (without directory) to process the pending queue.
    """
    from stripes_rag.indexer import discover_files
    from stripes_rag.tracker import FileTracker

    files = discover_files(directory, recursive)
    if not files:
        console.print("[yellow]No supported document files found.[/yellow]")
        return

    console.print(f"Found [bold]{len(files)}[/bold] files on disk")

    if dry_run:
        tracker = FileTracker()
        with console.status("[bold cyan]Checking files...[/bold cyan]"):
            from stripes_rag.tracker import _sha256

            resolved_map = {str(f.resolve()): f for f in files}
            with tracker._connect() as conn:
                rows = conn.execute(
                    "SELECT file_path, file_hash, status FROM file_tracking "
                    "WHERE file_path = ANY(%s)",
                    (list(resolved_map.keys()),),
                ).fetchall()
            existing = {row[0]: (row[1], row[2]) for row in rows}

            would_queue: list[Path] = []
            already_pending: int = 0
            for resolved, path in resolved_map.items():
                rec = existing.get(resolved)
                if rec is None:
                    would_queue.append(path)
                elif rec[1] == "pending" and not force:
                    already_pending += 1
                elif force:
                    would_queue.append(path)
                elif rec[1] == "error" and retry_errors:
                    would_queue.append(path)
                elif rec[1] != "error":
                    file_hash = _sha256(path)
                    if rec[0] != file_hash:
                        would_queue.append(path)

        if not would_queue and not already_pending:
            console.print("[green]All files are up to date.[/green]")
            return

        if would_queue:
            console.print(f"\n[bold]{len(would_queue)}[/bold] files would be queued:")
            for f in sorted(would_queue)[:50]:
                console.print(f"  [cyan]{f.name}[/cyan]")
            if len(would_queue) > 50:
                console.print(f"  [dim]... and {len(would_queue) - 50} more[/dim]")
        if already_pending:
            console.print(f"  [dim]{already_pending} already pending from previous run[/dim]")
        if not would_queue:
            console.print("[green]No new files to queue.[/green]")
        console.print(f"\n[dim]Run without --dry-run to register them as pending.[/dim]")
        return

    with console.status("[bold cyan]Scanning for changes...[/bold cyan]"):
        tracker = FileTracker()
        new_pending, already_pending, skipped_count = tracker.register_pending(
            files, force=force, retry_errors=retry_errors,
        )

    total_pending = new_pending + already_pending

    if not total_pending:
        console.print("[green]All files are up to date.[/green]")
        return

    if new_pending:
        console.print(f"  [bold]{new_pending}[/bold] files registered as pending")
    if already_pending:
        console.print(f"  [dim]{already_pending} already pending from previous run[/dim]")
    if skipped_count:
        console.print(f"  [dim]{skipped_count} up to date[/dim]")
    console.print(f"\n[dim]Run 'stripes index' to process {total_pending} pending files.[/dim]")


@cli.command()
@click.argument("query")
@click.option("-k", default=5, help="Number of results", show_default=True)
@click.option("--file", "source_file", default=None, help="Filter by source file path")
def search(query: str, k: int, source_file: str | None):
    """Semantic search across indexed documents."""
    from stripes_rag.db import get_engine, get_vectorstore, init_vectorstore_table
    from stripes_rag.embeddings import get_embeddings

    engine = get_engine()
    init_vectorstore_table(engine)
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(engine, embeddings)

    filter_dict = None
    if source_file:
        resolved = str(Path(source_file).resolve())
        filter_dict = {"source_file": resolved}

    with console.status("[bold cyan]Searching...[/bold cyan]"):
        results = vectorstore.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, (doc, distance) in enumerate(results, 1):
        similarity = 1.0 - distance
        meta = doc.metadata
        source = Path(meta.get("source_file", "unknown")).name
        pages = meta.get("page_numbers", "?")
        headings = meta.get("headings", "")

        table = Table(
            title=f"Result {i}",
            title_style="bold",
            show_header=False,
            padding=(0, 1),
        )
        table.add_column("Field", style="dim")
        table.add_column("Value")
        table.add_row("Similarity", f"{similarity:.4f}")
        table.add_row("Source", source)
        table.add_row("Pages", str(pages))
        if headings:
            table.add_row("Headings", headings)
        table.add_row("Text", doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
        console.print(table)
        console.print()


@cli.command()
def status():
    """Show indexing statistics."""
    from stripes_rag.tracker import FileTracker

    try:
        tracker = FileTracker()
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        console.print("[dim]Is PostgreSQL running? Try: docker compose up -d[/dim]")
        return

    stats = tracker.stats()

    def _fmt_size(size_bytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    table = Table(title="Stripes RAG Status", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Files indexed", str(stats["file_count"]))
    table.add_row("Files pending", str(stats["pending_count"]))
    table.add_row("Files errored", str(stats["error_count"]))
    table.add_row("Total chunks", str(stats["total_chunks"]))
    table.add_row("Total size", _fmt_size(stats["total_size"]))
    table.add_row("First indexed", str(stats["first_indexed"] or "N/A"))
    table.add_row("Last updated", str(stats["last_updated"] or "N/A"))
    console.print(table)
    if stats["pending_count"] > 0:
        console.print(f"\n[dim]Run 'stripes index' to process {stats['pending_count']} pending files.[/dim]")


@cli.command(name="list")
def list_files():
    """List all indexed files."""
    from stripes_rag.tracker import FileTracker

    try:
        tracker = FileTracker()
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        console.print("[dim]Is PostgreSQL running? Try: docker compose up -d[/dim]")
        return

    records = tracker.all_records()
    if not records:
        console.print("[yellow]No tracked files found.[/yellow]")
        return

    def _fmt_size(size_bytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    table = Table(title=f"Tracked Files ({len(records)})")
    table.add_column("#", style="dim", justify="right")
    table.add_column("File", style="cyan")
    table.add_column("Status")
    table.add_column("Size", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Updated", style="dim")

    for i, rec in enumerate(records, 1):
        if rec.status == "error":
            status_text = "[red]error[/red]"
        elif rec.status == "pending":
            status_text = "[yellow]pending[/yellow]"
        else:
            status_text = "[green]indexed[/green]"
        table.add_row(
            str(i),
            Path(rec.file_path).name,
            status_text,
            _fmt_size(rec.file_size),
            str(rec.chunk_count),
            rec.updated_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@cli.command()
def errors():
    """List all files that failed to index."""
    from stripes_rag.tracker import FileTracker

    try:
        tracker = FileTracker()
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        console.print("[dim]Is PostgreSQL running? Try: docker compose up -d[/dim]")
        return

    records = tracker.error_records()
    if not records:
        console.print("[green]No errored files.[/green]")
        return

    table = Table(title=f"Errored Files ({len(records)})")
    table.add_column("#", style="dim", justify="right")
    table.add_column("File", style="cyan")
    table.add_column("Error", style="red")
    table.add_column("Updated", style="dim")

    for i, rec in enumerate(records, 1):
        table.add_row(
            str(i),
            Path(rec.file_path).name,
            rec.error_message or "Unknown error",
            rec.updated_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
    console.print(f"\n[dim]Retry with: stripes index <directory> -r --retry-errors[/dim]")


@cli.command()
@click.argument("filename")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def delete(filename: str, yes: bool):
    """Delete an indexed file's vectors and tracking record.

    FILENAME can be a full path or a partial filename to match.
    """
    from stripes_rag.db import get_engine, delete_file_chunks
    from stripes_rag.tracker import FileTracker

    try:
        tracker = FileTracker()
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        console.print("[dim]Is PostgreSQL running? Try: docker compose up -d[/dim]")
        return

    # Find matching files
    matches = tracker.find_by_name(filename)
    if not matches:
        console.print(f"[yellow]No indexed file matching '{filename}' found.[/yellow]")
        return

    # If multiple matches, ask the user to be more specific
    if len(matches) > 1:
        console.print(f"[yellow]Multiple files match '{filename}':[/yellow]")
        for i, rec in enumerate(matches, 1):
            console.print(f"  {i}. [cyan]{Path(rec.file_path).name}[/cyan] ({rec.chunk_count} chunks)")
        console.print("\n[dim]Please provide a more specific filename.[/dim]")
        return

    record = matches[0]
    file_name = Path(record.file_path).name

    if not yes:
        console.print()
        console.print(
            f"[bold red]⚠  WARNING:[/bold red] This will permanently delete:"
        )
        console.print(f"  • [cyan]{file_name}[/cyan] — {record.chunk_count} chunks")
        console.print()
        if not click.confirm("Are you sure you want to continue?"):
            console.print("[dim]Aborted.[/dim]")
            return

    engine = get_engine()
    delete_file_chunks(engine, record.file_path)
    tracker.delete_record(record.file_path)

    console.print(f"[green]✓[/green] Deleted [cyan]{file_name}[/cyan] ({record.chunk_count} chunks removed)")


@cli.command()
@click.option("-j", "--workers", default=2, show_default=True, help="Parallel parse workers")
@click.option("--device", type=click.Choice(["cpu", "mps", "cuda"]), default=None,
              help="Embedding device (overrides EMBEDDING_DEVICE env var)")
@click.option("--skip-reindex", is_flag=True, help="Skip rebuilding the vector index after indexing")
def reindex(workers: int, device: str | None, skip_reindex: bool):
    """Re-index all previously tracked files."""
    if device:
        settings.embedding_device = device

    from stripes_rag.indexer import index_pending, setup_pipeline
    from stripes_rag.tracker import FileTracker

    tracker = FileTracker()
    paths = tracker.tracked_paths()
    if not paths:
        console.print("[yellow]No previously indexed files found.[/yellow]")
        return

    console.print(f"Re-indexing [bold]{len(paths)}[/bold] files")

    rebuild = not skip_reindex
    rebuild_status = None

    def on_rebuild(event):
        nonlocal rebuild_status
        if event == "start":
            rebuild_status = console.status("[bold cyan]Rebuilding vector index...[/bold cyan]")
            rebuild_status.start()
        elif event == "done" and rebuild_status:
            rebuild_status.stop()

    with console.status("[bold cyan]Registering files as pending...[/bold cyan]"):
        files = [Path(p) for p in paths if Path(p).exists()]
        tracker.register_pending(files, force=True)

    with console.status("[bold cyan]Initializing pipeline...[/bold cyan]"):
        engine, _, vectorstore = setup_pipeline()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Re-indexing...", total=len(paths))

        def on_progress(file_path, done, total):
            progress.update(task, description=f"[cyan]{file_path.name}[/cyan]")

        def on_result(result):
            progress.advance(task)

        results = index_pending(
            workers=workers,
            progress_callback=on_progress,
            result_callback=on_result,
            engine=engine,
            vectorstore=vectorstore,
            rebuild_index=rebuild,
            rebuild_callback=on_rebuild,
        )
        progress.update(task, completed=len(paths))

    _print_results(results)


@cli.command(name="rebuild-index")
def rebuild_index_cmd():
    """Rebuild the vector similarity index.

    Drops and recreates the HNSW/IVFFlat index. Useful after bulk indexing
    with --skip-reindex, or if you changed vector_index_type.
    """
    from stripes_rag.db import get_engine, init_vectorstore_table, rebuild_vector_index

    engine = get_engine()
    init_vectorstore_table(engine)

    with console.status("[bold cyan]Rebuilding vector index...[/bold cyan]"):
        rebuild_vector_index(engine)

    console.print("[green]Vector index rebuilt.[/green]")


@cli.command()
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def reset(yes: bool):
    """Drop all indexed data (vectors and tracking records)."""
    import psycopg

    from stripes_rag.db import TABLE_NAME

    if not yes:
        console.print()
        console.print(
            "[bold red]⚠  WARNING:[/bold red] This will permanently delete "
            "[bold]all[/bold] indexed data:",
        )
        console.print(f"  • Drop table [cyan]{TABLE_NAME}[/cyan] (all embeddings)")
        console.print("  • Truncate table [cyan]file_tracking[/cyan] (all tracking records)")
        console.print()
        if not click.confirm("Are you sure you want to continue?"):
            console.print("[dim]Aborted.[/dim]")
            return

    try:
        with psycopg.connect(settings.sync_connection_string) as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{TABLE_NAME}"')
            conn.execute("TRUNCATE file_tracking")
        console.print("[green]✓[/green] All indexed data has been removed.")
        console.print("[dim]Run 'stripes index <directory>' or 'stripes reindex' to rebuild.[/dim]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[dim]Is PostgreSQL running? Try: docker compose up -d[/dim]")


def _find_postgres_container() -> str:
    """Find the running postgres container ID."""
    import subprocess

    for image in ("pgvector/pgvector:pg17", "pgvector/pgvector"):
        result = subprocess.run(
            ["docker", "ps", "--filter", f"ancestor={image}", "--format", "{{.ID}}"],
            capture_output=True, text=True,
        )
        container_id = result.stdout.strip().split("\n")[0].strip()
        if container_id:
            return container_id
    raise click.ClickException(
        "Could not find running postgres container. "
        "Is Docker running? Try: docker compose up -d"
    )


@cli.command(name="export")
@click.argument("file", type=click.Path(path_type=Path))
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def export_cmd(file: Path, yes: bool):
    """Export indexed data (chunks + file tracking) to a dump file."""
    import subprocess

    import psycopg

    from stripes_rag.db import TABLE_NAME

    def _fmt_size(size_bytes: int | float) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    try:
        with psycopg.connect(settings.sync_connection_string) as conn:
            chunks_count = conn.execute(
                f'SELECT COUNT(*) FROM "{TABLE_NAME}"'
            ).fetchone()[0]
            files_count = conn.execute(
                "SELECT COUNT(*) FROM file_tracking"
            ).fetchone()[0]
            chunks_size = conn.execute(
                f"SELECT pg_total_relation_size('{TABLE_NAME}')"
            ).fetchone()[0]
            files_size = conn.execute(
                "SELECT pg_total_relation_size('file_tracking')"
            ).fetchone()[0]
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        console.print("[dim]Is PostgreSQL running? Try: docker compose up -d[/dim]")
        return

    if chunks_count == 0 and files_count == 0:
        console.print("[yellow]No data to export.[/yellow]")
        return

    total_size = chunks_size + files_size

    table = Table(title="Export Summary", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Chunks", str(chunks_count))
    table.add_row("Tracked files", str(files_count))
    table.add_row("Estimated size", _fmt_size(total_size))
    table.add_row("Output file", str(file))
    console.print(table)

    if not yes:
        if not click.confirm("\nProceed with export?"):
            console.print("[dim]Aborted.[/dim]")
            return

    container_id = _find_postgres_container()

    with console.status("[bold cyan]Exporting...[/bold cyan]"):
        result = subprocess.run(
            [
                "docker", "exec",
                "-e", f"PGPASSWORD={settings.postgres_password}",
                container_id,
                "pg_dump",
                "-U", settings.postgres_user,
                "-d", settings.postgres_db,
                "-Fc",
                f"--table={TABLE_NAME}",
                "--table=file_tracking",
            ],
            capture_output=True,
        )

    if result.returncode != 0:
        console.print(f"[red]pg_dump failed:[/red] {result.stderr.decode()}")
        return

    file.write_bytes(result.stdout)
    file_size = file.stat().st_size

    console.print(
        f"[green]\u2713[/green] Exported to [cyan]{file}[/cyan] ({_fmt_size(file_size)})"
    )


@cli.command(name="import")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--replace", is_flag=True, help="Clear existing data before import (default: merge)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def import_cmd(file: Path, replace: bool, yes: bool):
    """Import indexed data from a dump file.

    Default mode merges data (skips existing rows).
    Use --replace to clear all data and restore from the dump.
    """
    import subprocess

    import psycopg

    from stripes_rag.db import TABLE_NAME, get_engine, init_vectorstore_table, rebuild_vector_index

    def _fmt_size(size_bytes: int | float) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    file_size = file.stat().st_size

    try:
        with psycopg.connect(settings.sync_connection_string) as conn:
            chunks_count = conn.execute(
                f'SELECT COUNT(*) FROM "{TABLE_NAME}"'
            ).fetchone()[0]
            files_count = conn.execute(
                "SELECT COUNT(*) FROM file_tracking"
            ).fetchone()[0]
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        console.print("[dim]Is PostgreSQL running? Try: docker compose up -d[/dim]")
        return

    table = Table(title="Import Summary", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Dump file", str(file))
    table.add_row("File size", _fmt_size(file_size))
    table.add_row("Existing chunks", str(chunks_count))
    table.add_row("Existing files", str(files_count))
    table.add_row(
        "Mode",
        "[red]replace[/red] (clear + restore)" if replace else "merge (skip duplicates)",
    )
    console.print(table)

    if replace and (chunks_count > 0 or files_count > 0):
        console.print(
            f"\n[bold red]\u26a0  WARNING:[/bold red] Replace mode will delete all "
            f"{chunks_count} chunks and {files_count} file records before importing."
        )

    if not yes:
        if not click.confirm("\nProceed with import?"):
            console.print("[dim]Aborted.[/dim]")
            return

    container_id = _find_postgres_container()

    docker_exec_prefix = [
        "docker", "exec",
        "-e", f"PGPASSWORD={settings.postgres_password}",
        "-i", container_id,
    ]

    if replace:
        restore_args = [
            *docker_exec_prefix,
            "pg_restore",
            "-U", settings.postgres_user,
            "-d", settings.postgres_db,
            "--clean", "--if-exists",
            "--no-owner", "--no-acl",
        ]
    else:
        # Ensure tables exist before data-only restore
        engine = get_engine()
        init_vectorstore_table(engine)
        from stripes_rag.tracker import FileTracker
        FileTracker()

        restore_args = [
            *docker_exec_prefix,
            "pg_restore",
            "-U", settings.postgres_user,
            "-d", settings.postgres_db,
            "--data-only", "--disable-triggers",
            "--no-owner", "--no-acl",
        ]

    with console.status("[bold cyan]Importing...[/bold cyan]"):
        with open(file, "rb") as f:
            result = subprocess.run(restore_args, stdin=f, capture_output=True)

    if result.returncode != 0:
        stderr = result.stderr.decode()
        if replace:
            console.print(f"[red]pg_restore failed:[/red] {stderr}")
            return
        else:
            # In merge mode, duplicate key errors are expected
            lines = [
                l for l in stderr.strip().splitlines()
                if "duplicate key" not in l.lower()
                and "already exists" not in l.lower()
            ]
            if lines:
                console.print(f"[yellow]pg_restore warnings:[/yellow]")
                for line in lines[:10]:
                    console.print(f"  [dim]{line}[/dim]")

    with console.status("[bold cyan]Rebuilding vector index...[/bold cyan]"):
        if replace:
            engine = get_engine()
            init_vectorstore_table(engine)
        rebuild_vector_index(engine)

    # Report final counts
    try:
        with psycopg.connect(settings.sync_connection_string) as conn:
            new_chunks = conn.execute(
                f'SELECT COUNT(*) FROM "{TABLE_NAME}"'
            ).fetchone()[0]
            new_files = conn.execute(
                "SELECT COUNT(*) FROM file_tracking"
            ).fetchone()[0]
    except Exception:
        console.print("[green]\u2713[/green] Import complete.")
        return

    if replace:
        console.print(
            f"[green]\u2713[/green] Restored {new_chunks} chunks and {new_files} tracked files"
        )
    else:
        added_chunks = new_chunks - chunks_count
        added_files = new_files - files_count
        console.print(
            f"[green]\u2713[/green] Imported {added_chunks} new chunks and "
            f"{added_files} new file records "
            f"(total: {new_chunks} chunks, {new_files} files)"
        )


def _print_results(results):
    """Print a summary table of indexing results."""
    indexed = [r for r in results if r.status == "indexed"]
    skipped = [r for r in results if r.status == "skipped"]
    errors = [r for r in results if r.status == "error"]

    console.print()
    table = Table(title="Indexing Results")
    table.add_column("File", style="cyan")
    table.add_column("Status")
    table.add_column("Chunks", justify="right")

    for r in results:
        status_style = {
            "indexed": "[green]Indexed[/green]",
            "skipped": "[dim]Skipped (unchanged)[/dim]",
            "error": f"[red]Error: {r.error}[/red]",
        }
        table.add_row(
            r.path.name,
            status_style.get(r.status, r.status),
            str(r.chunks) if r.status == "indexed" else "-",
        )

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] {len(indexed)} indexed, "
        f"{len(skipped)} skipped, {len(errors)} errors"
    )

    # Timing breakdown for indexed files
    if indexed:
        total_parse = sum(r.parse_time for r in indexed)
        total_chunk = sum(r.chunk_time for r in indexed)
        total_embed = sum(r.embed_time for r in indexed)
        total_wall = total_parse + total_chunk + total_embed

        if total_wall > 0:
            console.print()
            ttable = Table(title="Timing Breakdown")
            ttable.add_column("Phase", style="bold")
            ttable.add_column("Time (s)", justify="right")
            ttable.add_column("% of Total", justify="right")

            for label, val in [("Parse", total_parse), ("Chunk", total_chunk), ("Embed+Store", total_embed)]:
                pct = val / total_wall * 100
                ttable.add_row(label, f"{val:.1f}", f"{pct:.0f}%")

            ttable.add_row("[bold]Total[/bold]", f"[bold]{total_wall:.1f}[/bold]", "[bold]100%[/bold]")
            console.print(ttable)
