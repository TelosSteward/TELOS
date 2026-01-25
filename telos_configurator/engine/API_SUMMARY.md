# TELOS Governance Engine - Public API Summary

## Module Location
`/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator/engine/governance_engine.py`

## Import

```python
from telos_configurator.engine import GovernanceEngine, PrimacyAttractor, ThresholdConfig
```

---

## Core Classes

### 1. `GovernanceEngine`
Main governance orchestrator.

**Key Methods:**
- `configure(pa, thresholds, corpus_docs, corpus_embeddings)` → `(bool, str|None)`
- `is_active()` → `bool`
- `process(query, top_k=3)` → `GovernanceResult`
- `get_statistics()` → `Dict[str, Any]`
- `export_audit_log()` → `List[Dict]`
- `clear_log()` → `None`

### 2. `PrimacyAttractor`
PA configuration dataclass.

**Fields:**
- `name: str`
- `purpose_statement: str`
- `scope: List[str]`
- `exclusions: List[str]`
- `prohibitions: List[str]`
- `embedding: Optional[np.ndarray]`
- `created_at: Optional[str]`

**Methods:**
- `get_combined_text()` → `str`
- `to_dict()` → `Dict`
- `from_dict(data)` → `PrimacyAttractor` (classmethod)

### 3. `ThresholdConfig`
Threshold configuration dataclass.

**Fields:**
- `tier_1_threshold: float = 0.65`
- `tier_2_lower: float = 0.35`
- `tier_2_upper: float = 0.65`
- `rag_relevance: float = 0.50`

**Methods:**
- `validate()` → `(bool, str|None)`
- `to_dict()` → `Dict`
- `from_dict(data)` → `ThresholdConfig` (classmethod)

### 4. `GovernanceResult`
Governance decision result dataclass.

**Fields:**
- `query: str`
- `fidelity: float`
- `tier: int`
- `tier_name: str`
- `action: str`
- `retrieved_policies: List[Dict] = []`
- `blocking_reason: Optional[str] = None`
- `timestamp: Optional[str] = None`

**Methods:**
- `to_dict()` → `Dict`

---

## Function API

### PA Management
- `create_pa(name, purpose, scope, exclusions, prohibitions)` → `PrimacyAttractor`
- `embed_pa(pa)` → `bool`
- `save_pa(pa, filepath)` → `bool`
- `load_pa(filepath)` → `PrimacyAttractor|None`

### Fidelity Computation
- `compute_fidelity(query_embedding, pa_embedding)` → `float`
- `classify_tier(fidelity, thresholds)` → `int`

### RAG Retrieval
- `retrieve_relevant_policies(query_embedding, corpus_embeddings, corpus_docs, top_k=3, relevance_threshold=0.50)` → `List[Dict]`

### End-to-End Processing
- `process_query(query, pa, corpus_embeddings, corpus_docs, thresholds, top_k=3)` → `GovernanceResult`

### Utilities
- `get_embedding(text)` → `np.ndarray|None`
- `cosine_similarity(a, b)` → `float`

---

## Constants

```python
TIER_NAMES = {1: "PA_Block", 2: "RAG_Policy", 3: "Expert_Escalation"}
TIER_ACTIONS = {1: "BLOCKED", 2: "POLICY_RETRIEVED", 3: "ESCALATED"}

DEFAULT_TIER_1_THRESHOLD = 0.65
DEFAULT_TIER_2_LOWER = 0.35
DEFAULT_TIER_2_UPPER = 0.65
DEFAULT_RAG_RELEVANCE = 0.50
```

---

## Usage Pattern

```python
# 1. Setup
from telos_configurator.engine import GovernanceEngine, create_pa, embed_pa

pa = create_pa(name="...", purpose="...", scope=[...], exclusions=[...], prohibitions=[...])
embed_pa(pa)

# 2. Initialize
engine = GovernanceEngine()
engine.configure(pa=pa, thresholds=ThresholdConfig(), corpus_docs=[], corpus_embeddings=[])

# 3. Process
result = engine.process("user query")

# 4. Handle result
if result.tier == 1:
    print(f"BLOCKED: {result.blocking_reason}")
elif result.tier == 2:
    print(f"Retrieved {len(result.retrieved_policies)} policies")
elif result.tier == 3:
    print(f"ESCALATED: {result.blocking_reason}")

# 5. Audit
stats = engine.get_statistics()
audit_log = engine.export_audit_log()
```

---

## Tier Classification

| Tier | Fidelity Range | Name | Action | Description |
|------|----------------|------|--------|-------------|
| 1 | ≥ 0.65 | PA_Block | BLOCKED | High fidelity - PA mathematically blocks |
| 2 | 0.35-0.65 | RAG_Policy | POLICY_RETRIEVED | Medium fidelity - RAG retrieves policies |
| 3 | < 0.35 | Expert_Escalation | ESCALATED | Low fidelity - Human expert review |

---

## Thread Safety

All `GovernanceEngine` methods are thread-safe via internal locking. Safe for Streamlit multi-session use.

---

## Error Handling

- Functions return `None` or negative values on error
- `GovernanceResult.fidelity == -1.0` indicates error
- Check `GovernanceResult.blocking_reason` for details
- All functions include try/except with logging

---

## Dependencies

- `numpy` - Vector operations
- `requests` - Ollama API
- Standard library: `json`, `os`, `threading`, `dataclasses`, `typing`, `datetime`

---

## Configuration

Requires Ollama running locally:
```bash
ollama pull nomic-embed-text
ollama serve  # default: http://localhost:11434
```

Modify constants at top of `governance_engine.py`:
```python
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
EMBEDDING_TIMEOUT = 30
```

---

**Author:** TELOS AI Labs Inc.  
**Contact:** contact@telos-labs.ai  
**Date:** 2026-01-23
