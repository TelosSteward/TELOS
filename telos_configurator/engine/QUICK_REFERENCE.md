# Corpus Engine Quick Reference

## File Location
`/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator/engine/corpus_engine.py`

## Import
```python
from telos_configurator.engine.corpus_engine import get_corpus_engine, CorpusDocument
```

## Common Operations

### Initialize Engine
```python
engine = get_corpus_engine()  # Singleton instance
```

### Add Documents
```python
# From Streamlit file uploader
success, msg, doc_id = engine.add_document(
    uploaded_file,
    category="policy",
    source="healthcare_system"
)
```

### List Documents
```python
docs = engine.list_documents()
# Returns: [{"doc_id": "...", "filename": "...", "title": "...", ...}, ...]
```

### Embed Documents
```python
# Single document
success, msg = engine.embed_document(doc_id)

# All documents with progress
def progress(current, total, filename):
    print(f"[{current}/{total}] {filename}")

success, failed, failed_list = engine.embed_all(progress_callback=progress)
```

### Search
```python
# By text query
results = engine.search("patient privacy requirements", top_k=3)

# By embedding vector
import numpy as np
query_emb = np.array([...])
results = engine.search_by_embedding(query_emb, top_k=5)

# Results format
for result in results:
    print(f"{result['title']}: {result['similarity']:.3f}")
    print(result['key_provisions'])
```

### Save/Load
```python
# Save
engine.save_corpus("/path/to/corpus.json")

# Load
engine.load_corpus("/path/to/corpus.json")
```

### Get Stats
```python
stats = engine.get_stats()
print(f"Total: {stats['total_documents']}")
print(f"Embedded: {stats['embedded_documents']}")
print(f"Categories: {stats['categories']}")
```

### Get Embedding Status
```python
status = engine.get_embedding_status()
print(f"{status['embedded']}/{status['total_documents']} embedded")
print(f"{status['percentage']:.1f}% complete")
```

### Remove Documents
```python
# Single document
success, msg = engine.remove_document(doc_id)

# All documents
success, msg = engine.clear_corpus()
```

### Get Full Document
```python
doc = engine.get_document(doc_id)
if doc:
    print(doc['text_content'])
    print(doc['key_provisions'])
    print(doc['escalation_triggers'])
```

## Streamlit Integration Pattern

```python
import streamlit as st
from telos_configurator.engine.corpus_engine import get_corpus_engine

# Initialize once in session state
if 'corpus_engine' not in st.session_state:
    st.session_state.corpus_engine = get_corpus_engine()

engine = st.session_state.corpus_engine

# File upload
uploaded = st.file_uploader("Upload", type=["json", "pdf", "txt", "md", "docx", "xlsx"])

if uploaded:
    if st.button("Add to Corpus"):
        success, msg, doc_id = engine.add_document(uploaded, category="policy")
        if success:
            st.success(msg)
        else:
            st.error(msg)

# Display stats
stats = engine.get_stats()
col1, col2 = st.columns(2)
col1.metric("Documents", stats['total_documents'])
col2.metric("Embedded", stats['embedded_documents'])

# Search
query = st.text_input("Search")
if query:
    results = engine.search(query, top_k=3)
    for result in results:
        with st.expander(f"{result['title']} ({result['similarity']:.3f})"):
            st.write(result['text_preview'])
```

## Error Handling Pattern

```python
success, message, doc_id = engine.add_document(file)

if not success:
    st.error(message)
    # Handle error:
    # - Display user-friendly message
    # - Log error
    # - Retry if appropriate
else:
    st.success(message)
    # Continue with doc_id
```

## Testing

```bash
cd /Users/brunnerjf/Desktop/TELOS_Master/telos_configurator/engine
python3 test_corpus_engine.py
```

## Dependencies

```bash
pip install numpy requests PyPDF2 python-docx openpyxl
```

## Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull embedding model
ollama pull nomic-embed-text

# Start server
ollama serve  # Runs on localhost:11434
```

## Configuration Constants

Located in `corpus_engine.py`:

```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
OLLAMA_ENDPOINT = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_TIMEOUT = 30  # seconds
```

## Supported Formats

- JSON (`.json`) - Built-in
- PDF (`.pdf`) - Requires PyPDF2
- Text (`.txt`) - Built-in
- Markdown (`.md`) - Built-in
- Word (`.docx`) - Requires python-docx
- Excel (`.xlsx`) - Requires openpyxl

## JSON Document Schema (Optional)

```json
{
  "title": "Document Title",
  "text_content": "Full text...",
  "key_provisions": ["Provision 1", "Provision 2"],
  "escalation_triggers": ["Trigger 1"]
}
```

Fields `title`, `text_content`, `key_provisions`, `escalation_triggers` are optional.
