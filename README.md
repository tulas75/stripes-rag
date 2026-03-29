# Stripes RAG

A CLI tool and web interface for indexing documents (PDF, DOCX, XLSX, PPTX, HTML, MD) into PostgreSQL with pgvector for semantic search and retrieval-augmented generation (RAG).

## Features

- **Document parsing** with [Docling](https://github.com/DS4SD/docling) — layout-aware extraction from PDF, DOCX, XLSX, PPTX, HTML, and MD files
- **Hybrid chunking** — structural + semantic chunking with heading and page number metadata
- **Embedding** with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (1024-dim, multilingual) — local sentence-transformers, remote [TEI](https://github.com/huggingface/text-embeddings-inference) server, or any provider via [LiteLLM](https://github.com/BerriAI/litellm) (DeepInfra, OpenAI, Cohere, etc.)
- **pgvector storage** — HNSW or IVFFlat indexing for fast similarity search
- **Crash-safe resumability** — pending state tracking with per-file atomic commits; interrupted runs resume from where they left off
- **Parallel processing** — `ProcessPoolExecutor` for parsing/chunking with configurable worker count
- **Optional reranker** — cross-encoder reranking via [TEI](https://github.com/huggingface/text-embeddings-inference) or LiteLLM for higher-precision results (graceful fallback when unavailable)
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

When a reranker is configured (`RERANKER_PROVIDER` or `RERANKER_URL`), the search command over-fetches `k * RERANKER_TOP_K_MULTIPLIER` candidates, reranks them with the cross-encoder, and returns the top `k`. Results display the reranker relevance score instead of the vector similarity score.

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

### `stripes export <file>`

Export indexed data (chunks + file tracking) to a PostgreSQL custom-format dump file. Runs `pg_dump` inside the Docker postgres container.

```bash
stripes export backup.dump          # interactive confirmation
stripes export backup.dump -y       # skip confirmation
```

| Flag | Description |
|------|-------------|
| `-y, --yes` | Skip confirmation prompt |

### `stripes import <file>`

Import indexed data from a dump file. Default mode merges (skips existing rows). Use `--replace` to clear all data and restore from the dump.

```bash
stripes import backup.dump             # merge: skip duplicates
stripes import backup.dump --replace   # clear + full restore
stripes import backup.dump -y          # skip confirmation
```

| Flag | Description |
|------|-------------|
| `--replace` | Clear existing data before import (default: merge) |
| `-y, --yes` | Skip confirmation prompt |

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
| `POSTGRES_PORT` | `5432` | PostgreSQL port (host-published; inside Docker the app connects on 5432 automatically) |
| `POSTGRES_DB` | `stripes_rag` | Database name |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `litellm` (TEI, DeepInfra, OpenAI, Cohere, etc.) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model name. For LiteLLM use `openai/<model>` with `EMBEDDING_API_BASE`. |
| `EMBEDDING_DIM` | *(auto)* | Embedding dimension. Auto-detected by probing the model at startup if omitted. |
| `EMBEDDING_DEVICE` | `cpu` | Compute device for local embeddings (`cpu`, `mps`, `cuda`) |
| `EMBEDDING_BATCH_SIZE` | `64` | Embedding batch size |
| `EMBEDDING_API_BASE` | *(unset)* | LiteLLM: API base URL (e.g. `http://localhost:8080/v1` for TEI, `https://api.deepinfra.com/v1/openai` for DeepInfra) |
| `EMBEDDING_API_KEY` | *(unset)* | LiteLLM: API key |
| `TOKENIZER_MODEL` | *(unset)* | HuggingFace tokenizer model for chunking. Required when `EMBEDDING_MODEL` is not a HF repo id (e.g. Ollama's `bge-m3:latest`). Defaults to `EMBEDDING_MODEL`. |
| `CHUNK_MAX_TOKENS` | `512` | Max tokens per chunk |
| `INDEX_BATCH_SIZE` | `128` | DB insertion batch size |
| `VECTOR_INDEX_TYPE` | `hnsw` | `hnsw` (high recall) or `ivfflat` (faster build) |
| `RERANKER_PROVIDER` | `none` | `none` (disabled), `tei` (TEI server), or `litellm` (any LiteLLM provider). Auto-detected as `tei` when `RERANKER_URL` is set. |
| `RERANKER_URL` | *(unset)* | TEI reranker server URL (also auto-sets provider to `tei`) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model. For LiteLLM, use provider prefix (e.g. `deepinfra/nvidia/llama-nemotron-rerank-vl-1b-v2`). |
| `RERANKER_TOP_K_MULTIPLIER` | `3` | Over-fetch multiplier (retrieve `k * N` candidates, rerank to `k`) |

For LiteLLM providers, set the appropriate API key env var (e.g. `DEEPINFRA_API_KEY`, `OPENAI_API_KEY`, `COHERE_API_KEY`) — LiteLLM reads these automatically.

For the Streamlit chat app, additional API keys can be set for LLM providers (DeepSeek, Mistral, Fireworks, etc.).

## Embedding Providers

By default, embeddings are computed locally with sentence-transformers. Set `EMBEDDING_PROVIDER=litellm` to use any OpenAI-compatible server — TEI, DeepInfra, OpenAI, Cohere, etc. — via [LiteLLM](https://docs.litellm.ai/docs/embedding/supported_embedding).

### Local (default)

No extra config needed. The `BAAI/bge-m3` model is downloaded on first run (~2 GB).

```
EMBEDDING_MODEL=BAAI/bge-m3
```

### TEI Server

[TEI](https://github.com/huggingface/text-embeddings-inference) exposes an OpenAI-compatible `/v1/embeddings` endpoint. The docker-compose file includes an opt-in TEI service:

```bash
docker compose --profile tei up -d
```

```
EMBEDDING_PROVIDER=litellm
EMBEDDING_MODEL=openai/BAAI/bge-m3
EMBEDDING_API_BASE=http://localhost:8080/v1
```

### DeepInfra / other cloud providers

DeepInfra (and similar providers) also expose an OpenAI-compatible endpoint:

```
EMBEDDING_PROVIDER=litellm
EMBEDDING_MODEL=openai/BAAI/bge-m3
EMBEDDING_API_BASE=https://api.deepinfra.com/v1/openai
EMBEDDING_API_KEY=your-deepinfra-key
```

> **Why `openai/` prefix?** LiteLLM routes `openai/` through its OpenAI-compatible handler. The `deepinfra/` prefix only supports chat and rerank in LiteLLM — not embeddings. Since TEI, DeepInfra, and most providers all speak the OpenAI embeddings API, `openai/` + `EMBEDDING_API_BASE` is the universal approach.

For providers with native LiteLLM embedding support (OpenAI, Cohere, Voyage, etc.), use their prefix directly:

```
EMBEDDING_PROVIDER=litellm
EMBEDDING_MODEL=cohere/embed-multilingual-v3.0
COHERE_API_KEY=your-key-here
```

### Dimension auto-detection

The embedding dimension is probed automatically at startup by embedding a short test string. To skip the probe, set `EMBEDDING_DIM` explicitly:

```
EMBEDDING_DIM=1024
```

**Important:** All indexed documents must use the same embedding model/dimension. Switching models requires a full re-index (`stripes reset` + `stripes index`).

## Reranker (optional)

A **reranker** is a cross-encoder model that rescores vector search candidates by jointly encoding the query and each document. This produces more accurate relevance scores than embedding similarity alone, at the cost of an extra inference step.

The pipeline: retrieve `k * multiplier` candidates via vector search, rerank with the cross-encoder, return top `k`.

### TEI Reranker

The docker-compose file includes an opt-in TEI reranker service:

```bash
docker compose --profile reranker up -d
```

This starts a second TEI instance on port 8081 (configurable via `RERANKER_PORT`) serving `BAAI/bge-reranker-v2-m3`.

Add to your `.env`:

```
RERANKER_URL=http://localhost:8081
```

The provider is auto-detected as `tei` from the URL. Remove the variable to disable.

### LiteLLM Reranker

Use any [LiteLLM rerank provider](https://docs.litellm.ai/docs/rerank) (DeepInfra, Cohere, etc.):

```
RERANKER_PROVIDER=litellm
RERANKER_MODEL=deepinfra/nvidia/llama-nemotron-rerank-vl-1b-v2
DEEPINFRA_API_KEY=your-key-here
```

### Graceful fallback

If the reranker service is unreachable (TEI container down, API error, etc.), both the CLI and the Streamlit app log a warning and fall back to plain vector similarity results. No errors are raised.

### Tuning

- `RERANKER_TOP_K_MULTIPLIER` (default: 3) controls over-fetching. With `-k 5`, the system retrieves 15 candidates and reranks to 5. Higher values improve recall but increase latency.
- The Streamlit sidebar includes a **Reranker** checkbox to toggle reranking on/off at runtime (only enabled when a reranker provider is configured).

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
│  │ bge-m3 embedding   │──┼──→ local or LiteLLM (TEI, DeepInfra, …)
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

### Chunk size outcomes

```
Document
  → HybridChunker (target: 512 tokens)
    → most chunks: ≤512 tokens ✓
    → some chunks: 513–32768 chars (oversized atomic blocks) → passed through, bge-m3 handles up to 8192 tokens
    → rare chunks: >32768 chars → force-split into sub-chunks of 32768 chars (~8192 tokens each)
```

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
| `embedding` | `vector(N)` | Document chunk embedding (dimension auto-detected from model) |
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
- **Reranker toggle**: Enable/disable cross-encoder reranking from the sidebar
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
├── embeddings.py   # Embedding provider dispatch (local, LiteLLM)
├── reranker.py     # Optional reranker dispatch (TEI, LiteLLM)
├── loader.py       # Docling DocumentConverter
├── chunker.py      # HybridChunker + metadata extraction
├── indexer.py       # Pipeline orchestration
└── prompts.py      # RAG prompt profiles
app.py              # Streamlit chat interface
docker-compose.yml  # PostgreSQL 17 + pgvector
```
