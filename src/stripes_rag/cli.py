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
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--recursive", "-r", is_flag=True, help="Scan subdirectories")
@click.option("--force", "-f", is_flag=True, help="Re-index all files regardless of changes")
@click.option("--retry-errors", is_flag=True, help="Retry previously failed files")
@click.option("-j", "--workers", default=2, show_default=True, help="Parallel parse workers")
def index(directory: Path, recursive: bool, force: bool, retry_errors: bool, workers: int):
    """Index PDF/DOCX files from a directory."""
    from stripes_rag.indexer import discover_files, index_directory

    files = discover_files(directory, recursive)
    if not files:
        console.print("[yellow]No PDF/DOCX files found.[/yellow]")
        return

    console.print(f"Found [bold]{len(files)}[/bold] files on disk")

    # Pre-check which files need indexing so we can show accurate counts
    with console.status("[bold cyan]Checking which files need indexing...[/bold cyan]"):
        from stripes_rag.tracker import FileTracker
        tracker = FileTracker()
        to_process = [f for f in files if force or tracker.needs_indexing(f, retry_errors=retry_errors)]

    if not to_process:
        console.print("[green]All files are up to date.[/green]")
        return

    skipped = len(files) - len(to_process)
    if skipped:
        console.print(f"  [dim]{skipped} already indexed, {len(to_process)} to process[/dim]")
    else:
        console.print(f"  [dim]{len(to_process)} to process[/dim]")

    already_done = len(files) - len(to_process)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing...", total=len(files), completed=already_done)

        def on_progress(file_path, done, total):
            progress.update(task, description=f"[cyan]{file_path.name}[/cyan]")

        def on_result(result):
            if result.status != "skipped":
                progress.advance(task)

        results = index_directory(
            directory,
            recursive=recursive,
            force=force,
            retry_errors=retry_errors,
            workers=workers,
            progress_callback=on_progress,
            result_callback=on_result,
        )
        progress.update(task, completed=len(files))

    _print_results(results)


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
    table.add_row("Files errored", str(stats["error_count"]))
    table.add_row("Total chunks", str(stats["total_chunks"]))
    table.add_row("Total size", _fmt_size(stats["total_size"]))
    table.add_row("First indexed", str(stats["first_indexed"] or "N/A"))
    table.add_row("Last updated", str(stats["last_updated"] or "N/A"))
    console.print(table)


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
def reindex(workers: int):
    """Re-index all previously tracked files."""
    from stripes_rag.indexer import reindex_all
    from stripes_rag.tracker import FileTracker

    tracker = FileTracker()
    paths = tracker.tracked_paths()
    if not paths:
        console.print("[yellow]No previously indexed files found.[/yellow]")
        return

    console.print(f"Re-indexing [bold]{len(paths)}[/bold] files")

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
            progress.update(task, completed=done, description=f"[cyan]{file_path.name}[/cyan]")

        results = reindex_all(workers=workers, progress_callback=on_progress)
        progress.update(task, completed=len(paths))

    _print_results(results)


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
