# RAG — Multi-Document Retrieval Augmented Generation

A local CLI tool for building searchable knowledge bases from your documents and querying them with an LLM.

Supports multiple named databases, folder ingestion, content deduplication via hashing, query expansion, and source attribution. Uses [FAISS](https://github.com/facebookresearch/faiss) for vector search and any model available via [OpenRouter](https://openrouter.ai).

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) — dependencies are declared inline per PEP 723
- An [OpenRouter](https://openrouter.ai) API key

## Setup

```bash
export OPENROUTER_API_KEY=your_key_here
```

Add to `~/.zshrc` or `~/.bashrc` to make it permanent.

## Usage

### `rag.py` — Multi-document system (recommended)

```bash
# Create or switch to a named database
uv run rag.py set research

# Add a single document
uv run rag.py add notes.md

# Add a folder recursively
uv run rag.py add ./docs --recursive

# Query
uv run rag.py query "What is the main argument?"
uv run rag.py q "Summarise the key findings"   # alias

# Get more results from the last query
uv run rag.py more

# List all databases
uv run rag.py list-dbs

# Show active database info
uv run rag.py info
```

### `simple_rag.py` — Single-document system with caching

```bash
uv run simple_rag.py path/to/document.md
uv run simple_rag.py path/to/document.md --test    # run with included sample data
uv run simple_rag.py path/to/document.md --verbose
```

### `simple_rag_v0.py` — Minimal reference implementation

Basic RAG without caching or configuration persistence. Good for understanding the core approach.

```bash
uv run simple_rag_v0.py
```

## Architecture

- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2` by default)
- **Vector search**: FAISS (cosine similarity)
- **LLM**: Configurable via OpenRouter — defaults to free-tier models with fallback chain
- **Storage**: `.rag_databases/` directory (gitignored) — one subdirectory per named database
- **Deduplication**: SHA-256 content hashing; unchanged documents are not reprocessed

## Sample Data

`Sample data/` contains a truncated research document for testing. Run `uv run simple_rag.py --test` to use it.

## Model Configuration

On first run, `rag.py` creates `.rag_config.json` with default free-tier OpenRouter models. Override per-query:

```bash
uv run rag.py q "question" --gen-model anthropic/claude-3-haiku
```

## Notes

- `.rag_databases/` and `.rag_cache/` are gitignored — they contain your indexed documents and are generated at runtime
- Query expansion is on by default (uses the LLM to generate alternative phrasings before searching)
