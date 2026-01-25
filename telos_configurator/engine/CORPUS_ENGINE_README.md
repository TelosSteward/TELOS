# TELOS Corpus Engine

Complete backend module for corpus document management, embedding generation, and semantic search.

## Overview

The `corpus_engine.py` module provides a self-contained, thread-safe engine for:
- Loading documents from multiple formats (JSON, PDF, TXT, MD, DOCX, XLSX)
- Managing a corpus of documents with metadata
- Generating embeddings using Ollama's `nomic-embed-text` model
- Performing semantic search with cosine similarity
- Persisting corpus state to/from JSON files

## Installation Requirements

```bash
pip install numpy requests PyPDF2 python-docx openpyxl
```

**Ollama Setup:**
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the embedding model
ollama pull nomic-embed-text

# Start Ollama server (runs on localhost:11434)
ollama serve
```

## Quick Start

```python
from telos_configurator.engine.corpus_engine import get_corpus_engine

# Get singleton engine instance
engine = get_corpus_engine()

# Add a document (from Streamlit UploadedFile)
success, message, doc_id = engine.add_document(
    uploaded_file,
    category="policy",
    source="upload"
)

# Embed all documents
success_count, failure_count, failed = engine.embed_all()

# Search the corpus
results = engine.search("patient privacy requirements", top_k=3)

# Save corpus to file
engine.save_corpus("/path/to/corpus.json")

# Load corpus from file
engine.load_corpus("/path/to/corpus.json")
```

## Public API Summary

### Document Management
- `add_document(uploaded_file, category, source)` → (success, message, doc_id)
- `remove_document(doc_id)` → (success, message)
- `list_documents()` → List[Dict]
- `get_document(doc_id)` → Optional[Dict]
- `clear_corpus()` → (success, message)

### Embedding Generation
- `embed_document(doc_id, progress_callback)` → (success, message)
- `embed_all(progress_callback)` → (success_count, failure_count, failed_filenames)
- `get_embedding_status()` → Dict

### Search/Retrieval
- `search(query_text, top_k)` → List[Dict]
- `search_by_embedding(query_embedding, top_k)` → List[Dict]

### Persistence
- `save_corpus(filepath)` → (success, message)
- `load_corpus(filepath)` → (success, message)

### Utilities
- `get_stats()` → Dict

## Supported File Formats

| Format | Extension | Requirements |
|--------|-----------|--------------|
| JSON | `.json` | Built-in |
| PDF | `.pdf` | `PyPDF2` |
| Text | `.txt` | Built-in |
| Markdown | `.md` | Built-in |
| Word | `.docx` | `python-docx` |
| Excel | `.xlsx` | `openpyxl` |

## Testing

```bash
cd /Users/brunnerjf/Desktop/TELOS_Master/telos_configurator/engine
python3 test_corpus_engine.py
```

For complete API documentation, see inline docstrings in `corpus_engine.py`.
