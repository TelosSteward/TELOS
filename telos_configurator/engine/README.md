# TELOS Governance Engine

A comprehensive governance module implementing the three-tier TELOS framework for the Corpus Configurator MVP.

## Overview

The Governance Engine provides mathematical governance for conversational AI through three progressive tiers:

- **Tier 1: PA Mathematical Block** - High fidelity queries (≥0.65) are blocked by the Primacy Attractor
- **Tier 2: RAG Policy Retrieval** - Medium fidelity queries (0.35-0.65) retrieve relevant policies from corpus
- **Tier 3: Expert Escalation** - Low fidelity queries (<0.35) escalate to human review

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Governance Engine                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Query Input → Embed → Compute Fidelity → Classify Tier     │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐        │
│  │ Tier 1   │   │ Tier 2   │   │ Tier 3           │        │
│  │ PA Block │   │ RAG      │   │ Expert Escalate  │        │
│  │ f≥0.65   │   │ 0.35≤f<  │   │ f<0.35          │        │
│  │          │   │ 0.65     │   │                  │        │
│  │ BLOCKED  │   │ POLICY_  │   │ ESCALATED        │        │
│  │          │   │ RETRIEVED│   │                  │        │
│  └──────────┘   └──────────┘   └──────────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
cd /Users/brunnerjf/Desktop/TELOS_Master/telos_configurator
```

Ensure Ollama is running with nomic-embed-text:
```bash
ollama pull nomic-embed-text
ollama serve
```

## Quick Start

```python
from telos_configurator.engine import (
    GovernanceEngine,
    create_pa,
    embed_pa,
    ThresholdConfig
)

# 1. Create Primacy Attractor
pa = create_pa(
    name="Healthcare HIPAA",
    purpose="Ensure HIPAA compliance in healthcare conversations",
    scope=["patient privacy", "medical data protection"],
    exclusions=["general medical info"],
    prohibitions=["disclosing PHI", "unauthorized sharing"]
)

# 2. Embed PA
embed_pa(pa)

# 3. Prepare corpus
corpus_docs = [
    {"document_id": "HIPAA-001", "title": "Privacy Rule", ...},
    {"document_id": "HIPAA-002", "title": "Security Rule", ...}
]
corpus_embeddings = [get_embedding(doc['title']) for doc in corpus_docs]

# 4. Initialize engine
engine = GovernanceEngine()
engine.configure(
    pa=pa,
    thresholds=ThresholdConfig(),
    corpus_docs=corpus_docs,
    corpus_embeddings=corpus_embeddings
)

# 5. Process queries
result = engine.process("Can I share patient records?")

print(f"Tier: {result.tier}")
print(f"Action: {result.action}")
print(f"Fidelity: {result.fidelity:.4f}")
if result.retrieved_policies:
    print(f"Retrieved: {len(result.retrieved_policies)} policies")
```

## Public API

### Core Classes

#### `GovernanceEngine`
Main governance engine class.

**Methods:**
- `configure(pa, thresholds, corpus_docs, corpus_embeddings)` - Configure engine
- `is_active()` - Check if configured
- `process(query, top_k=3)` - Process query through governance
- `get_statistics()` - Get tier distribution, avg fidelity
- `export_audit_log()` - Export all governance decisions
- `clear_log()` - Clear query log
- `get_pa_info()` - Get PA configuration
- `get_threshold_info()` - Get threshold configuration
- `get_corpus_info()` - Get corpus metadata

#### `PrimacyAttractor`
Primacy Attractor configuration.

**Fields:**
- `name: str` - PA identifier
- `purpose_statement: str` - Purpose description
- `scope: List[str]` - In-scope topics
- `exclusions: List[str]` - Out-of-scope topics
- `prohibitions: List[str]` - Prohibited actions
- `embedding: np.ndarray` - PA embedding vector
- `created_at: str` - ISO timestamp

**Methods:**
- `get_combined_text()` - Get concatenated text for embedding
- `to_dict()` - Convert to JSON-serializable dict
- `from_dict(data)` - Create from dictionary

#### `ThresholdConfig`
Threshold configuration for tier classification.

**Fields:**
- `tier_1_threshold: float = 0.65` - PA block threshold
- `tier_2_lower: float = 0.35` - RAG lower bound
- `tier_2_upper: float = 0.65` - RAG upper bound
- `rag_relevance: float = 0.50` - Minimum retrieval relevance

**Methods:**
- `validate()` - Validate configuration
- `to_dict()` - Convert to dict
- `from_dict(data)` - Create from dict

#### `GovernanceResult`
Result of governance decision.

**Fields:**
- `query: str` - Original query
- `fidelity: float` - Fidelity score
- `tier: int` - Tier number (1, 2, 3)
- `tier_name: str` - "PA_Block", "RAG_Policy", "Expert_Escalation"
- `action: str` - "BLOCKED", "POLICY_RETRIEVED", "ESCALATED"
- `retrieved_policies: List[Dict]` - Retrieved policies (Tier 2)
- `blocking_reason: str` - Reason for decision
- `timestamp: str` - ISO timestamp

### PA Functions

#### `create_pa(name, purpose, scope, exclusions, prohibitions)`
Create a new Primacy Attractor.

**Returns:** `PrimacyAttractor`

#### `embed_pa(pa)`
Generate embedding for PA.

**Returns:** `bool` - True if successful

#### `save_pa(pa, filepath)`
Save PA to JSON file.

**Returns:** `bool` - True if successful

#### `load_pa(filepath)`
Load PA from JSON file.

**Returns:** `PrimacyAttractor` or `None`

### Fidelity Functions

#### `compute_fidelity(query_embedding, pa_embedding)`
Compute cosine similarity between query and PA.

**Returns:** `float` - Fidelity score [0, 1]

#### `classify_tier(fidelity, thresholds)`
Classify query into tier based on fidelity.

**Returns:** `int` - Tier number (1, 2, 3)

### RAG Functions

#### `retrieve_relevant_policies(query_embedding, corpus_embeddings, corpus_docs, top_k=3, relevance_threshold=0.50)`
Retrieve top-k relevant policies from corpus.

**Returns:** `List[Dict]` - Retrieved documents with similarity scores

#### `process_query(query, pa, corpus_embeddings, corpus_docs, thresholds, top_k=3)`
Process query through complete governance pipeline.

**Returns:** `GovernanceResult`

### Utility Functions

#### `get_embedding(text)`
Generate embedding using Ollama.

**Returns:** `np.ndarray` or `None`

#### `cosine_similarity(a, b)`
Compute cosine similarity between vectors.

**Returns:** `float` - Similarity score [-1, 1]

## Constants

```python
TIER_NAMES = {
    1: "PA_Block",
    2: "RAG_Policy",
    3: "Expert_Escalation"
}

TIER_ACTIONS = {
    1: "BLOCKED",
    2: "POLICY_RETRIEVED",
    3: "ESCALATED"
}

DEFAULT_TIER_1_THRESHOLD = 0.65
DEFAULT_TIER_2_LOWER = 0.35
DEFAULT_TIER_2_UPPER = 0.65
DEFAULT_RAG_RELEVANCE = 0.50
```

## Examples

### Example 1: Basic Usage

```python
from telos_configurator.engine import GovernanceEngine, create_pa, embed_pa

# Create and embed PA
pa = create_pa(
    name="Finance Compliance",
    purpose="Ensure financial compliance",
    scope=["financial regulations", "trading rules"],
    exclusions=["general finance"],
    prohibitions=["insider trading", "market manipulation"]
)
embed_pa(pa)

# Initialize engine
engine = GovernanceEngine()
engine.configure(pa=pa, thresholds=ThresholdConfig(), corpus_docs=[], corpus_embeddings=[])

# Process query
result = engine.process("What are the insider trading rules?")
print(f"Tier: {result.tier}, Action: {result.action}")
```

### Example 2: Corpus Integration

```python
from telos_configurator.engine import GovernanceEngine, get_embedding

# Prepare corpus
corpus_docs = [
    {"document_id": "SEC-001", "title": "Insider Trading Prohibition"},
    {"document_id": "SEC-002", "title": "Market Manipulation Rules"}
]

# Embed corpus
corpus_embeddings = []
for doc in corpus_docs:
    emb = get_embedding(doc['title'])
    if emb is not None:
        corpus_embeddings.append(emb)

# Configure with corpus
engine.configure(pa=pa, thresholds=ThresholdConfig(), 
                 corpus_docs=corpus_docs, corpus_embeddings=corpus_embeddings)

# Process query - will retrieve policies in Tier 2
result = engine.process("Can I trade based on non-public information?")
if result.tier == 2:
    for policy in result.retrieved_policies:
        print(f"  {policy['document_id']}: {policy['title']} (sim: {policy['similarity']})")
```

### Example 3: Custom Thresholds

```python
from telos_configurator.engine import ThresholdConfig

# More conservative thresholds (wider RAG zone)
custom_thresholds = ThresholdConfig(
    tier_1_threshold=0.75,  # Higher PA block
    tier_2_lower=0.25,      # Lower RAG floor
    tier_2_upper=0.75,      # Match tier 1
    rag_relevance=0.40      # Lower retrieval threshold
)

engine.configure(pa=pa, thresholds=custom_thresholds, 
                 corpus_docs=corpus_docs, corpus_embeddings=corpus_embeddings)
```

### Example 4: Audit Trail

```python
# Process multiple queries
queries = [
    "What is insider trading?",
    "Can I share material information?",
    "What's the weather?"
]

for q in queries:
    engine.process(q)

# Get statistics
stats = engine.get_statistics()
print(f"Total Queries: {stats['total_queries']}")
print(f"Avg Fidelity: {stats['avg_fidelity']:.4f}")
print(f"Tier Distribution: {stats['tier_distribution']}")

# Export audit log
audit_log = engine.export_audit_log()
with open('audit.json', 'w') as f:
    json.dump(audit_log, f, indent=2)
```

### Example 5: PA Persistence

```python
from telos_configurator.engine import save_pa, load_pa

# Save PA
save_pa(pa, "/path/to/pa_config.json")

# Load PA
loaded_pa = load_pa("/path/to/pa_config.json")
if loaded_pa:
    print(f"Loaded PA: {loaded_pa.name}")
```

## Thread Safety

The `GovernanceEngine` class is thread-safe and can be used in Streamlit applications:

```python
import streamlit as st

# Initialize once in session state
if 'engine' not in st.session_state:
    st.session_state.engine = GovernanceEngine()
    # configure...

# Use in multiple threads
result = st.session_state.engine.process(user_query)
```

## Error Handling

All functions include comprehensive error handling:

```python
result = engine.process(query)

if result.fidelity == -1.0:
    # Embedding failed or engine not configured
    print(f"Error: {result.blocking_reason}")
elif result.tier == 3 and "No relevant policies" in result.blocking_reason:
    # No policies found - true escalation
    print("Escalating to expert review")
```

## Testing

Run the built-in test suite:

```bash
cd /Users/brunnerjf/Desktop/TELOS_Master/telos_configurator/engine
python3 governance_engine.py
```

Expected output:
```
======================================================================
TELOS Governance Engine - Test
======================================================================

[1] Creating test Primacy Attractor...
  PA Name: Healthcare HIPAA
  Purpose: Ensure all healthcare conversations comply with HIPAA regulations

[2] Embedding PA...
  PA embedded successfully (dim: (768,))

[3] Creating mock corpus...
  Embedding corpus documents...
    Embedded: HIPAA-001
    Embedded: HIPAA-002

[4] Initializing Governance Engine...
  Engine configured successfully

[5] Testing queries...

  Query 1: What are the requirements for protecting patient data?
    Fidelity: 0.7234
    Tier: 1 (PA_Block)
    Action: BLOCKED
    Reason: Query fidelity (0.7234) >= tier 1 threshold (0.65)

  Query 2: Can I share medical records with family members?
    Fidelity: 0.5431
    Tier: 2 (RAG_Policy)
    Action: POLICY_RETRIEVED
    Retrieved Policies: 2

  Query 3: What is the weather today?
    Fidelity: 0.1234
    Tier: 3 (Expert_Escalation)
    Action: ESCALATED
    Reason: Query fidelity (0.1234) < tier 2 threshold (0.35)

[6] Governance Statistics:
  Total Queries: 3
  Avg Fidelity: 0.4633
  Tier Distribution:
    Tier 1: 1 (33.3%)
    Tier 2: 1 (33.3%)
    Tier 3: 1 (33.3%)

======================================================================
Test complete!
======================================================================
```

## License

Copyright (c) 2026 TELOS AI Labs Inc.
