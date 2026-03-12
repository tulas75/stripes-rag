# Stripes RAG

A CLI tool and web interface for indexing documents (PDF, DOCX, XLSX, PPTX, HTML, MD) into PostgreSQL with pgvector for semantic search and retrieval-augmented generation (RAG).

## Features

- **Document parsing** with [Docling](https://github.com/DS4SD/docling) — layout-aware extraction from PDF, DOCX, XLSX, PPTX, HTML, and MD files
- **Hybrid chunking** — structural + semantic chunking with heading and page number metadata
- **Embedding** with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (1024-dim, multilingual)
- **pgvector storage** — HNSW or IVFFlat indexing for fast similarity search
- **Crash-safe resumability** — per-file atomic commits with SHA-256 change detection
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
stripes index /path/to/documents
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

### `stripes index <directory>`

Index documents from a directory (supports PDF, DOCX, XLSX, PPTX, HTML, MD).

| Flag | Description |
|------|-------------|
| `-r, --recursive` | Scan subdirectories |
| `-f, --force` | Re-index all files regardless of changes |
| `--retry-errors` | Retry previously failed files |
| `-j, --workers N` | Parallel parse workers (default: 2) |

### `stripes search <query>`

Run a semantic search against indexed documents.

| Flag | Description |
|------|-------------|
| `-k N` | Number of results (default: 5) |
| `--file PATH` | Filter by source file path |

### `stripes status`

Show indexing statistics — file counts, chunk totals, and timestamps.

### `stripes list`

List all indexed files with status, size, and chunk count.

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
| `EMBEDDING_DEVICE` | `mps` | Compute device (`mps`, `cpu`, `cuda`) |
| `EMBEDDING_BATCH_SIZE` | `64` | Embedding batch size |
| `CHUNK_MAX_TOKENS` | `512` | Max tokens per chunk |
| `INDEX_BATCH_SIZE` | `128` | DB insertion batch size |
| `VECTOR_INDEX_TYPE` | `hnsw` | `hnsw` (high recall) or `ivfflat` (faster build) |

For the Streamlit chat app, additional API keys can be set for LLM providers (DeepSeek, Mistral, Fireworks, etc.).

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
│  │ bge-m3 embedding   │  │
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
- **Atomic commits**: Each file is committed independently — interrupted runs resume from where they left off
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
| `status` | `TEXT` | `indexed` or `error` |
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
