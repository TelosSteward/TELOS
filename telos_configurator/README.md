# TELOS Corpus Configurator MVP

A comprehensive wizard-based interface for configuring and deploying three-tier TELOS governance frameworks.

## Overview

The TELOS Corpus Configurator provides a step-by-step workflow for:

1. **Domain Selection** - Choose from pre-configured templates (Healthcare, Financial, Legal) or create custom configurations
2. **Corpus Upload & Management** - Upload policy documents in multiple formats (JSON, PDF, TXT, MD, DOCX, XLSX)
3. **PA Configuration** - Define your Primacy Attractor (purpose, scope, exclusions, prohibitions)
4. **Threshold Calibration** - Configure three-tier governance thresholds with visual tier zones
5. **Activation** - Activate the governance engine with readiness checks
6. **Dashboard & Testing** - Monitor metrics, test queries, and review audit logs

## Quick Start

### Prerequisites

- Python 3.9+
- Ollama running locally with `nomic-embed-text` model
- Required Python packages (see below)

### Install Dependencies

```bash
pip install streamlit pandas numpy requests PyPDF2 python-docx openpyxl
```

### Launch Application

```bash
cd /Users/brunnerjf/Desktop/TELOS_Master/telos_configurator
streamlit run main.py --server.port 8502
```

The application will open at `http://localhost:8502`

## Architecture

### Core Components

- **`main.py`** - Streamlit entry point with multi-step wizard UI
- **`state_manager.py`** - Session state orchestration and persistence
- **`config/styles.py`** - TELOS visual language (colors, glassmorphism, CSS)
- **`engine/corpus_engine.py`** - Document parsing, embedding, and search
- **`engine/governance_engine.py`** - Three-tier governance logic and PA management
- **`components/`** - 9 reusable UI components

### Component Architecture

```
components/
├── domain_selector.py          # Pre-configured domain templates
├── corpus_uploader.py          # File upload interface
├── corpus_manager.py           # Document list and embedding controls
├── pa_configurator.py          # Primacy Attractor definition
├── threshold_config.py         # Tier threshold sliders with visualization
├── activation_panel.py         # Readiness checklist and activation
├── dashboard_metrics.py        # Live governance statistics
├── test_query_interface.py    # Interactive query testing
└── audit_panel.py              # Audit log viewer and export
```

## Features

### Domain Templates

Pre-configured templates for common use cases:

- **Healthcare (HIPAA)** - Patient privacy and medical data protection
- **Financial Services** - Regulatory compliance and fraud prevention
- **Legal Services** - Attorney-client privilege and ethical standards
- **Custom** - Build from scratch

### Multi-Format Document Support

- **JSON** - Structured policy documents
- **PDF** - Scanned or digital PDFs (via PyPDF2)
- **Text** - Plain text and Markdown files
- **DOCX** - Microsoft Word documents (via python-docx)
- **XLSX** - Excel spreadsheets (via openpyxl)

### Three-Tier Governance

- **Tier 1: PA Mathematical Block** - High fidelity queries blocked
- **Tier 2: RAG Policy Retrieval** - Medium fidelity retrieves relevant policies
- **Tier 3: Expert Escalation** - Low fidelity escalated to humans

### State Persistence

- Save/load full configuration state
- Export corpus with embeddings
- Export audit logs in JSON format

## Workflow

### Step 1: Domain Selection

Choose a template or custom configuration:

```
Healthcare → HIPAA compliance template
Financial → Financial regulations template
Legal → Legal ethics template
Custom → Build from scratch
```

### Step 2: Corpus Upload

Upload policy documents:

1. Click "Choose a file" and select document
2. Set category (Policy, Regulation, Guideline, etc.)
3. Set source (Manual Upload or custom)
4. Click "Add to Corpus"
5. Once all documents uploaded, click "Embed All"

### Step 3: PA Configuration

Define your Primacy Attractor:

1. **Name**: Descriptive identifier
2. **Purpose**: Clear purpose statement
3. **Scope**: In-scope topics (one per line)
4. **Exclusions**: Out-of-scope topics (one per line)
5. **Prohibitions**: Prohibited actions (one per line)
6. Click "Generate PA Embedding"

### Step 4: Threshold Calibration

Configure tier thresholds:

- **Tier 1 Threshold**: PA block threshold (default: 0.65)
- **Tier 2 Lower**: RAG zone lower bound (default: 0.35)
- **RAG Relevance**: Minimum similarity for retrieval (default: 0.50)

View visual tier zone diagram for reference.

### Step 5: Activation

Review readiness checklist:

- ✓ PA configured and embedded
- ✓ Corpus loaded with documents
- ✓ Corpus embeddings generated
- ✓ Thresholds configured and valid

Click "ACTIVATE GOVERNANCE" when ready.

### Step 6: Dashboard & Testing

Three tabs available:

1. **Metrics Dashboard**: Overall statistics, tier distribution, recent queries
2. **Test Query Interface**: Enter test queries to see tier classification
3. **Audit Log**: Filter, search, and export governance decisions

## Configuration Management

### Save Configuration

```python
# Saves:
# - Domain selection
# - PA definition
# - Current step
# - UI state
```

Click "Save Configuration" in sidebar to export as JSON.

### Load Configuration

Upload previously saved configuration JSON to restore state.

**Note:** Corpus and PA embeddings must be regenerated after loading.

## Visual Language

### TELOS Brand Colors

- **Gold**: `#F4D03F` - Primary brand color
- **Tier 1**: `#e74c3c` (Red) - Mission critical
- **Tier 2**: `#f39c12` (Yellow) - Important
- **Tier 3**: `#3498db` (Blue) - Standard

### Glassmorphism

All cards use TELOS glassmorphism effect:

- Frosted glass appearance
- Subtle gradient overlay
- Backdrop blur filter
- Colored borders with glow
- Layered shadows

## API Reference

### State Manager

```python
from state_manager import (
    initialize_state,      # Initialize all session state
    get_current_step,      # Get current step index
    navigate_to_step,      # Navigate to specific step
    save_configuration,    # Save config to JSON
    load_configuration,    # Load config from JSON
    reset_all_state        # Full reset
)
```

### Corpus Engine

```python
from engine.corpus_engine import CorpusEngine

engine = CorpusEngine()
success, msg, doc_id = engine.add_document(uploaded_file, category, source)
success_count, fail_count, failed = engine.embed_all(progress_callback)
results = engine.search(query_text, top_k=3)
stats = engine.get_stats()
```

### Governance Engine

```python
from engine.governance_engine import GovernanceEngine, create_pa, embed_pa

pa = create_pa(name, purpose, scope, exclusions, prohibitions)
embed_pa(pa)

engine = GovernanceEngine()
success, error = engine.configure(pa, thresholds, corpus_docs, corpus_embeddings)
result = engine.process(query, top_k=3)
stats = engine.get_statistics()
audit_log = engine.export_audit_log()
```

## Troubleshooting

### Ollama Connection Issues

**Error:** "Cannot connect to Ollama API"

**Solution:**
```bash
# Start Ollama
ollama serve

# Pull embedding model
ollama pull nomic-embed-text
```

### File Upload Failures

**Error:** "No text content could be extracted"

**Solution:**
- Verify file is not corrupted
- Check file format is supported
- For PDFs, ensure text is extractable (not just images)

### Embedding Failures

**Error:** "Failed to embed PA" or "Failed to embed documents"

**Solution:**
- Ensure Ollama is running (`http://localhost:11434`)
- Check `nomic-embed-text` model is installed
- Verify network connectivity

### State Reset

If application behaves unexpectedly:

1. Click "Reset All" in sidebar
2. Confirm reset
3. Restart from Step 1

## Port Configuration

Default port: **8502** (different from main TELOS Observatory on 8501)

To use a different port:

```bash
streamlit run main.py --server.port YOUR_PORT
```

## Development

### Adding New Domain Templates

Edit `components/domain_selector.py`:

```python
DOMAIN_TEMPLATES = {
    "your_domain": {
        "name": "Your Domain Name",
        "description": "Description here",
        "pa_template": {
            "name": "PA Name",
            "purpose": "Purpose statement",
            "scope": ["item1", "item2"],
            "exclusions": ["exclusion1"],
            "prohibitions": ["prohibition1"]
        },
        "thresholds": {
            "tier_1": 0.70,
            "tier_2_lower": 0.40,
            "rag_relevance": 0.55
        }
    }
}
```

### Adding New Components

1. Create component file in `components/`
2. Define `render_*` function
3. Add to `components/__init__.py`
4. Import in `main.py`
5. Integrate into appropriate step

## License

TELOS AI Labs Inc. - Proprietary

## Contact

- **Email**: contact@telos-labs.ai
- **Primary Contact**: JB@telos-labs.ai

## Version History

- **v1.0.0** (2026-01-23) - MVP Release
  - Multi-step wizard interface
  - Domain templates (Healthcare, Financial, Legal)
  - Multi-format document support
  - Three-tier governance framework
  - State persistence
  - Audit logging
