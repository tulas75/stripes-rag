# Stripes RAG

A CLI tool and web interface for indexing documents (PDF, DOCX, XLSX, PPTX, HTML, MD) into PostgreSQL with pgvector for semantic search and retrieval-augmented generation (RAG).

## Features

- **Document parsing** with [Docling](https://github.com/DS4SD/docling) — layout-aware extraction from PDF, DOCX, XLSX, PPTX, HTML, and MD files
- **Hybrid chunking** — structural + semantic chunking with heading and page number metadata
- **Embedding** with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (1024-dim, multilingual) — local or via remote [TEI](https://github.com/huggingface/text-embeddings-inference) server
- **pgvector storage** — HNSW or IVFFlat indexing for fast similarity search
- **Crash-safe resumability** — pending state tracking with per-file atomic commits; interrupted runs resume from where they left off
- **Parallel processing** — `ProcessPoolExecutor` for parsing/chunking with configurable worker count
- **Streamlit chat UI** — interactive RAG chat with multiple LLM providers via LiteLLM and smolagents

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- ~2 GB disk for the embedding model on first run

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd stripes-rag
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env as needed (defaults work out of the box for local dev)
```

### 3. Start PostgreSQL

```bash
docker compose up -d
```

This starts PostgreSQL 17 with the pgvector extension on port 5432.

### 4. Index documents

```bash
stripes index /path/to/documents -r             # scan + process (cpu, default)
stripes index /path/to/documents -r --device mps # use Apple GPU on macOS
stripes index                                    # resume pending files from a previous run
```

### 5. Search

```bash
stripes search "your query here"
```

Or launch the web interface:

```bash
streamlit run app.py
```

## CLI Reference

### `stripes index [directory]`

Index documents from a directory, or resume processing pending files.

- **With directory**: scans for new/changed files, registers them as pending, then processes the queue.
- **Without directory**: resumes processing any pending files from a previous interrupted run.

```bash
stripes index docs/ -r              # scan + process
stripes index                       # resume: process pending only
stripes index docs/ -r --device mps # scan + process with Apple GPU
```

| Flag | Description |
|------|-------------|
| `-r, --recursive` | Scan subdirectories |
| `-f, --force` | Re-index all files regardless of changes |
| `--retry-errors` | Retry previously failed files |
| `-j, --workers N` | Parallel parse workers (default: 2) |
| `--device DEVICE` | Embedding device: `cpu`, `mps`, or `cuda` (overrides `EMBEDDING_DEVICE`) |
| `--skip-reindex` | Skip rebuilding the vector index after indexing |

### `stripes scan <directory>`

Scan a directory and register files as pending without processing them. Useful for previewing what would be indexed.

```bash
stripes scan docs/ -r              # register pending files
stripes scan docs/ -r --dry-run    # preview what would be queued
```

| Flag | Description |
|------|-------------|
| `-r, --recursive` | Scan subdirectories |
| `-f, --force` | Re-queue all files regardless of changes |
| `--retry-errors` | Re-queue previously failed files |
| `--dry-run` | Show what would be queued without making changes |

### `stripes search <query>`

Run a semantic search against indexed documents.

| Flag | Description |
|------|-------------|
| `-k N` | Number of results (default: 5) |
| `--file PATH` | Filter by source file path |

### `stripes status`

Show indexing statistics — file counts (indexed, pending, errored), chunk totals, and timestamps.

### `stripes list`

List all tracked files with status (indexed, pending, error), size, and chunk count.

### `stripes errors`

List files that failed to index, with error messages.

### `stripes delete <filename>`

Delete a file's vectors and tracking record. Accepts full path or partial match.

| Flag | Description |
|------|-------------|
| `-y, --yes` | Skip confirmation prompt |

### `stripes reindex`

Re-index all previously tracked files.

| Flag | Description |
|------|-------------|
| `-j, --workers N` | Parallel parse workers (default: 2) |
| `--device DEVICE` | Embedding device: `cpu`, `mps`, or `cuda` (overrides `EMBEDDING_DEVICE`) |
| `--skip-reindex` | Skip rebuilding the vector index after indexing |

### `stripes rebuild-index`

Rebuild the vector similarity index. Useful after running `index` or `reindex` with `--skip-reindex`, or after changing `VECTOR_INDEX_TYPE`.

```bash
stripes rebuild-index
```

### `stripes reset`

Drop all indexed data (vectors and tracking records).

| Flag | Description |
|------|-------------|
| `-y, --yes` | Skip confirmation prompt |

## Configuration

All settings are managed via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `stripes` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `stripes` | PostgreSQL password |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `stripes_rag` | Database name |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace embedding model |
| `EMBEDDING_DEVICE` | `cpu` | Compute device (`cpu`, `mps`, `cuda`) |
| `EMBEDDING_BATCH_SIZE` | `64` | Embedding batch size |
| `EMBEDDING_SERVER_URL` | *(unset)* | TEI server URL (omit for local sentence-transformers) |
| `CHUNK_MAX_TOKENS` | `512` | Max tokens per chunk |
| `INDEX_BATCH_SIZE` | `128` | DB insertion batch size |
| `VECTOR_INDEX_TYPE` | `hnsw` | `hnsw` (high recall) or `ivfflat` (faster build) |

For the Streamlit chat app, additional API keys can be set for LLM providers (DeepSeek, Mistral, Fireworks, etc.).

## Embedding Server (TEI)

By default, embeddings are computed locally with sentence-transformers. You can optionally offload embedding to a [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) (TEI) server — useful for dedicating GPU resources, sharing the model across services, or keeping the indexing machine lightweight.

### Start TEI

The docker-compose file includes an opt-in TEI service:

```bash
docker compose --profile tei up -d
```

This starts TEI on port 8080 (configurable via `TEI_PORT`) serving the same `BAAI/bge-m3` model.

### Point stripes-rag at TEI

Add to your `.env`:

```
EMBEDDING_SERVER_URL=http://localhost:8080
```

That's it — both the CLI indexer and the Streamlit app will use TEI for embeddings. Remove the variable to switch back to local.

### Batch size

TEI limits how many texts it accepts per request (`--max-client-batch-size`, default 32). The client automatically splits requests into batches of `EMBEDDING_BATCH_SIZE` (default 64). If you hit batch size errors, either lower `EMBEDDING_BATCH_SIZE` or increase TEI's `--max-client-batch-size`.

### Important

The TEI model must produce the same embedding dimensions (1024 for bge-m3) as the local model. Mixing models between indexing and search will produce incorrect results.

## Architecture

```
PDF, DOCX, XLSX, PPTX, HTML, or MD files
     │
     ▼
┌─────────────────────────┐
│  ProcessPoolExecutor     │  ← -j workers
│  ┌───────────────────┐  │
│  │ Docling parse      │  │
│  │ HybridChunker      │  │
│  │ metadata extraction │  │
│  └───────────────────┘  │
└──────────┬──────────────┘
           │ chunks + metadata
           ▼
┌─────────────────────────┐
│  Main thread             │
│  ┌───────────────────┐  │
│  │ bge-m3 embedding   │──┼──→ local or TEI server
│  │ PGVectorStore      │  │
│  │ FileTracker commit  │  │
│  └───────────────────┘  │
└──────────┬──────────────┘
           │
           ▼
    PostgreSQL + pgvector
```

- **Pipeline parallelism**: Workers parse and chunk files in subprocesses while the main thread embeds and stores results
- **Sliding window**: Keeps `workers + 1` futures in flight to bound memory usage
- **Two-phase indexing**: `scan` registers files as pending in the DB; `index` processes the pending queue. This allows `stripes index` (without a directory) to resume after a crash
- **Atomic commits**: Each file is committed independently — interrupted runs resume from remaining pending files
- **Change detection**: SHA-256 file hashes tracked in `file_tracking` table; unchanged files are skipped

## Chunking Strategy

Documents are split using Docling's **HybridChunker**, which combines structural and semantic awareness for higher-quality chunks than naive text splitting.

### How it works

1. **Docling parses the document** into a structured `DoclingDocument` — preserving headings, tables, lists, paragraphs, and layout hierarchy from the original document.

2. **HybridChunker splits on document structure first.** It respects natural boundaries (sections, paragraphs, list items) rather than cutting at arbitrary token counts. Sibling elements (e.g. consecutive paragraphs under the same heading) are merged together up to the token limit.

3. **Token budget is enforced via the embedding model's tokenizer.** The chunker uses a `HuggingFaceTokenizer` initialized from the same model used for embedding (`BAAI/bge-m3`), with a max of `CHUNK_MAX_TOKENS` (default: 512). This ensures chunks align with what the model can actually encode.

4. **`chunker.serialize(chunk)`** prepends heading context to the chunk text, so each chunk is self-contained — a paragraph deep in section 3.2.1 carries its full heading path in the text that gets embedded.

### Metadata extraction

Each chunk carries structured metadata extracted from Docling's provenance data:

| Field | Source | Example |
|-------|--------|---------|
| `headings` | `chunk.meta.headings` | `"Introduction > Background > Related Work"` |
| `page_numbers` | `chunk.meta.doc_items[*].prov[*].page_no` | `"3,4"` |
| `chunk_index` | Enumeration order | `0`, `1`, `2`, ... |

- **Headings** are joined with ` > ` to represent the full section hierarchy
- **Page numbers** are deduplicated and sorted — a chunk spanning pages 3-4 stores `"3,4"`
- **Chunk index** preserves the original document order for retrieval context

### Why HybridChunker over naive splitting

| | Naive (RecursiveCharacterTextSplitter) | HybridChunker |
|---|---|---|
| Split points | Character/token count | Document structure (headings, paragraphs) |
| Tables | Broken mid-row | Kept intact as single chunks |
| Context | Lost after split | Heading path prepended to each chunk |
| Overlap | Sliding window duplication | No duplication — structural merging instead |

## Database Schema

**`document_chunks`** — vector store table

| Column | Type | Description |
|--------|------|-------------|
| `id` | `BIGSERIAL` | Primary key |
| `embedding` | `vector(1024)` | Document chunk embedding |
| `document` | `TEXT` | Chunk text content |
| `source_file` | `TEXT` | Source file path |
| `page_numbers` | `TEXT` | Page number(s) |
| `headings` | `TEXT` | Section headings (`>` separated) |
| `chunk_index` | `INTEGER` | Position within source file |
| `file_hash` | `TEXT` | SHA-256 of source file |

**`file_tracking`** — indexing state

| Column | Type | Description |
|--------|------|-------------|
| `file_path` | `TEXT` (PK) | Absolute file path |
| `file_hash` | `TEXT` | SHA-256 hash |
| `file_size` | `BIGINT` | File size in bytes |
| `chunk_count` | `INTEGER` | Number of chunks |
| `status` | `TEXT` | `pending`, `indexed`, or `error` |
| `error_message` | `TEXT` | Error details (if any) |
| `indexed_at` | `TIMESTAMPTZ` | First indexed |
| `updated_at` | `TIMESTAMPTZ` | Last updated |

## Streamlit Chat App

The web interface (`app.py`) provides:

- **Multiple RAG profiles**: Classic RAG, Project Architect, Study Companion
- **Model selection**: Ollama, DeepSeek, Mistral, DeepInfra, Groq, and more via LiteLLM
- **Language toggle**: Italian / English
- **Source references**: Expandable sections showing retrieved chunks with similarity scores
- **Auto-generated follow-up questions**

```bash
streamlit run app.py
```

## Project Structure

```
src/stripes_rag/
├── cli.py          # Click CLI with Rich progress bars
├── config.py       # Pydantic Settings (.env support)
├── db.py           # PGEngine + vectorstore init
├── tracker.py      # SHA-256 file change tracking
├── embeddings.py   # HuggingFace embeddings wrapper
├── loader.py       # Docling DocumentConverter
├── chunker.py      # HybridChunker + metadata extraction
├── indexer.py       # Pipeline orchestration
└── prompts.py      # RAG prompt profiles
app.py              # Streamlit chat interface
docker-compose.yml  # PostgreSQL 17 + pgvector
```
