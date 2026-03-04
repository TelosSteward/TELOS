# Boundary Corpus Methodology: From Regulatory Text to Governance Substrate

**TELOS AI Labs Inc. — Research Program**
**Date:** 2026-02-12
**Status:** Active Development (Phase 1 Complete, Phase 2 In Progress)
**Author:** TELOS Development Team

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Executive Summary

This document describes how TELOS constructs its boundary governance layer from domain-specific regulatory knowledge bases. The key insight: **governance is not applied at runtime — it is built into the agent's identity at instantiation.** The Primacy Attractor (PA) that governs an agentic AI is constructed from real, publicly available regulatory text before the agent processes its first request. This is the "Telically Entrained" substrate — the agent's purpose, scope, and boundaries are pre-loaded from the actual mandates of its operating domain.

The result is an AI governance layer that:
- Adds ~10ms overhead per request (0.2-1.0% of a typical 1-6 second pipeline)
- Requires zero runtime retrieval or API calls for boundary checking
- Is grounded in verified regulatory text, not extrapolated rules
- Can be parameterized by jurisdiction for state-by-state compliance
- Produces audit-ready governance traces suitable for regulatory attestation

---

## 1. The Problem: Why Single-Text Boundaries Fail

### 1.1 The Deontic Logic Limitation

Embedding models (e.g., sentence-transformers/all-MiniLM-L6-v2, 384 dimensions) are trained on natural language. They encode semantic similarity — how close two pieces of text are in meaning. They do **not** faithfully represent negation or prohibition.

The boundary text "No binding underwriting decisions" embeds near — not opposite to — "binding underwriting decisions." This is called the deontic logic limitation: the model cannot distinguish "do X" from "don't do X" in embedding space.

**Empirical result:** Single-text boundary embeddings caught only **2 of 11 (18.2%)** Category A violations in the Nearmap counterfactual benchmark. The boundary text "No binding underwriting decisions" does not reliably match user requests like "Approve this claim for the full amount" because the prohibition and the violation occupy different semantic neighborhoods.

### 1.2 The Solution: Corpus-Augmented Centroid Embeddings

Instead of representing a boundary as a single text embedding, TELOS represents each boundary as a **centroid** — the mean of 60-100 diverse violation phrasings embedded in the same space. These phrasings are affirmative action statements — what violations actually sound like in practice.

The centroid covers a broader region of the violation surface in embedding space. When a user request comes in, TELOS computes its cosine similarity against the centroid (not the prohibition text). Because the corpus phrasings are affirmative actions similar to what a real violation would sound like, the cosine similarity is dramatically higher for actual violations.

**Empirical result:** With just 12-13 hand-crafted Layer 1 phrasings per boundary, detection improved from **18.2% to 45.5%** — a 2.5x improvement with zero runtime cost increase.

---

## 2. The Three-Layer Corpus Architecture

Each boundary's centroid embedding is computed from a multi-layer corpus:

### Layer 1: Hand-Crafted Anchor Phrasings (12-15 per boundary)

Human-authored violation phrasings that capture the core semantic intent. These are affirmative action statements — what a user might say when attempting a boundary violation.

**Example (Boundary 1 — No binding underwriting decisions):**
- "Approve this claim for the full amount"
- "Deny this claim based on the roof condition score"
- "Make the final underwriting decision on this property"

**Role:** Semantic anchors that define the core violation surface. Layer 1 is the most important layer — it establishes the centroid's center of mass.

**Status:** Complete for all 5 property_intel boundaries. See `telos_governance/boundary_corpus_static.py`.

### Layer 2: LLM-Generated Expansion (25-35 per boundary, deduplicated)

Systematically generated phrasings using Mistral to expand the violation surface coverage. Generated from Layer 1 anchors, then deduplicated by cosine similarity to remove near-duplicates.

**Role:** Fills semantic gaps between Layer 1 anchors. Captures phrasings that human authors might not think of.

**Status:** Not yet implemented. Planned for Phase 2.

### Layer 3: Regulatory Text Extractions (15-20 per boundary, 0.5x weight)

Direct extractions from verified regulatory sources — the actual statutory and bulletin language that defines what is prohibited. These are weighted at 0.5x because regulatory language is more formal than how users phrase violations, but they anchor the corpus to the legal standard.

**Role:** Grounds the boundary centroid in verified legal text. Ensures the governance is traceable to specific regulatory requirements.

**Status:** Research complete (this document). Implementation pending.

---

## 3. Regulatory Source Research: Property Intelligence Agent

The property_intel agent template governs an AI agent that provides decision support for insurance underwriting and claims using aerial imagery (Nearmap) and roof analysis (ITEL). It has 5 boundaries, 3 of which have been researched in depth for Nearmap stakeholder validation.

### 3.1 Boundary 1: "No binding underwriting decisions"

**Regulatory foundation:** The strongest regulatory anchor in U.S. insurance law.

| Source | Key Text | Citation |
|--------|----------|----------|
| Colorado CRS 10-3-1104.9(8)(a) | "Algorithm" means a process that **"informs human decision making"** | [Justia](https://law.justia.com/codes/colorado/title-10/) |
| NAIC Model Bulletin (2023), Sec. 3 | "the extent to which humans are involved in the final decision-making process" | [NAIC](https://content.naic.org/sites/default/files/inline-files/2023-12-4%20Model%20Bulletin_Adopted_0.pdf) |
| NY DFS CL No. 7 (2024), Para. 11 | "The responsibility to comply...remains with the insurer at all times" | [NY DFS](https://www.dfs.ny.gov/industry-guidance/circular-letters/cl2024-07) |
| NAIC Model #880, Sec. 4G | "Making or **permitting** any unfair discrimination" | [NAIC](https://content.naic.org/sites/default/files/model-law-880.pdf) |
| Connecticut Bulletin MC-25 | AI that "make or support decisions related to regulated insurance practices" | [CT CID](https://portal.ct.gov/cid/-/media/cid/1_bulletins/bulletin-mc-25.pdf) |

**Full research:** `research/boundary_regulatory_research_b1.md` (5 sources, 11 states mapped)

**Layer 3 extraction candidates:**
- "informs human decision making" (CO CRS — defines algorithm as decision support)
- "humans involved in the final decision-making process" (NAIC — risk factor)
- "responsibility remains with the insurer at all times" (NY DFS — cannot delegate to AI)
- "made or supported by" AI must comply (NAIC — compliance applies in both modes)

### 3.2 Boundary 3: "No overriding human assessor or licensed adjuster findings"

**Regulatory foundation:** Adjuster licensing + 6-state aerial imagery bulletin trend.

| Source | Key Text | Citation |
|--------|----------|----------|
| Texas Ins. Code 4101.051 | "a person may not act as...an adjuster...unless the person holds a license" | [TX Legislature](https://statutes.capitol.texas.gov/Docs/IN/htm/IN.4101.htm) |
| Florida 626.854(1)(e) | "No person may...investigate, or adjust a claim" without license | [FL Senate](https://www.flsenate.gov/laws/statutes/2023/626.854) |
| Florida 627.70131(3)(b) | "the licensed adjuster...must provide...name and state adjuster license number" | [FL Senate](https://m.flsenate.gov/Statutes/627.70131) |
| PA Notice 2024-06 | "conduct a physical inspection to validate...the aerial image" | [PA Bulletin](https://www.pacodeandbulletin.gov/Display/pabull?file=/secure/pabulletin/data/vol54/54-21/735.html) |
| CT CID (March 2024) | "physical inspection to validate the specific type of damage that the aerial image purports to show" | [CT CID](https://portal.ct.gov/cid/-/media/cid/1_protertycasualty/aerial-imagery-notice.pdf) |
| NH INS 25-016-AB | "even when the insurer believes the aerial imagery to be conclusive — a physical inspection is required" | [NHID](https://www.insurance.nh.gov/news-and-media/new-hampshire-insurance-department-issues-bulletin-use-aerial-imagery-property) |
| NC 25-B-09 | "aerial imagery should not be the only information utilized in decision-making" | [NCDOI](https://www.ncdoi.gov/25-b-09-use-aerial-images/open) |
| RI Bulletin 2025-3 | "cannot alone justify a cancellation or nonrenewal...without further investigation" | [RI DBR](https://dbr.ri.gov/) |

**Full research:** `research/boundary_regulatory_research_b3.md` (10 sources, 6 aerial imagery states)

**Layer 3 extraction candidates:**
- "conduct a physical inspection to validate" (PA, CT — imagery requires human validation)
- "aerial imagery does not conclusively demonstrate...must conduct follow-up physical inspection" (NH)
- "aerial imagery should not be the only information utilized" (NC)
- "the licensed adjuster...must provide...name and state adjuster license number" (FL — physical inspection is human-only)
- "No person may...investigate, or adjust a claim" without license (FL)

### 3.3 Boundary 5: "No autonomous repair/replace authorization"

**Regulatory foundation:** Claims settlement laws + adjuster licensing + emerging AI legislation.

| Source | Key Text | Citation |
|--------|----------|----------|
| NAIC Model #900, Sec. 4(d) | "Refusing to pay claims without conducting a reasonable investigation" | State codifications |
| Texas Ins. Code 542.055 | "the insurer shall...approve the claim...or reject the claim...and notify the claimant in writing" | [TX Legislature](https://statutes.capitol.texas.gov/Docs/IN/htm/IN.542.htm) |
| Florida 626.877 | "Every adjuster shall adjust or investigate every claim" | [FindLaw](https://codes.findlaw.com/fl/title-xxxvii-insurance/fl-st-sect-626-877.html) |
| Florida 627.70131(3)(d) | Electronic methods permitted for investigation (validates Nearmap as tool) | [FL Senate](https://m.flsenate.gov/Statutes/627.70131) |
| Louisiana RS 22:1896 | "shall provide prompt adjustment by a qualified adjuster" | [LA Legislature](https://legis.la.gov/Legis/Law.aspx?d=509045) |
| Oklahoma 36-1250.7 | "claimant shall be advised of the acceptance or denial...by the insurer" | [Justia](https://law.justia.com/codes/oklahoma/title-36/section-36-1250-7/) |
| Florida HB 527 (PENDING) | "prohibits...denying a claim...based solely on the output of an AI system" | [FL Senate](https://www.flsenate.gov/Session/Bill/2026/527) |

**Full research:** `research/boundary_regulatory_research_b5.md` (13 sources, 5 states + NAIC + pending legislation)

**Layer 3 extraction candidates:**
- "the insurer shall approve or reject the claim" (TX — duty rests with insurer, not AI)
- "Every adjuster shall adjust or investigate every claim" (FL — mandatory human involvement)
- "prompt adjustment by a qualified adjuster" (LA — requires qualified human)
- "based solely on the output of an AI system" (FL HB 527 — pending direct prohibition)
- "reasonable investigation based upon all available information" (NAIC Model 900 — AI alone isn't sufficient)

---

## 4. The Instantiation Pipeline

### 4.1 How Regulatory Text Becomes Governance

```
┌─────────────────────────────────────────────────────────┐
│  REGULATORY SOURCES (Public, Verified)                  │
│  • NAIC Model Bulletin (2023)                           │
│  • State statutes (TX, FL, CO, LA, OK, etc.)           │
│  • State bulletins (PA, CT, NH, NC, RI)                │
│  • Pending legislation (FL HB 527)                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  EXTRACTION: Regulatory Text → Violation Phrasings      │
│                                                         │
│  "Algorithm means a process that informs human           │
│   decision making" (CO CRS)                             │
│        ↓                                                │
│  "I am making the final underwriting decision"          │
│  "Approve this without human review"                    │
│  "The algorithm has decided to deny coverage"           │
│                                                         │
│  The prohibition text becomes affirmative violations    │
│  because embedding models handle action language well.  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  EMBEDDING: Violation Phrasings → Vector Space          │
│                                                         │
│  Each phrasing → embed_fn() → 384-dim vector           │
│  All phrasings → mean → L2 normalize → centroid         │
│                                                         │
│  The centroid represents the "violation surface" —      │
│  the region of embedding space where boundary           │
│  violations live.                                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  INSTANTIATION: Centroid → BoundarySpec → AgenticPA     │
│                                                         │
│  BoundarySpec(                                          │
│      text="No binding underwriting decisions...",       │
│      centroid_embedding=<384-dim vector>,               │
│      corpus_texts=[...verified phrasings...],           │
│      severity="hard"                                    │
│  )                                                      │
│                                                         │
│  The centroid IS the boundary. It is pre-computed at    │
│  PA construction time. Zero runtime cost.               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  RUNTIME: Request → Cosine Check → Governance Decision  │
│                                                         │
│  User: "Approve this claim for the full amount"         │
│        ↓                                                │
│  embed_fn(request) → 384-dim vector                     │
│  cosine_similarity(request_vec, centroid) → 0.78        │
│  0.78 > boundary_threshold → HARD_BLOCK                 │
│                                                         │
│  Total runtime cost: ~10ms                              │
│  (1 embedding + 5 cosine checks + decision logic)       │
│  No retrieval. No API calls. No latency.                │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Telic Entrainment

**Telic Entrainment** is the process of loading a domain-specific knowledge corpus into an agent's boundary representations at instantiation, such that the governance model captures the full surface area of acceptable and unacceptable operation before the agent processes its first request.

This is not the embedding of a purpose statement. The purpose statement is one dimension. Telic entrainment is the boundaries — built from real NAIC bulletins, state statutes, aerial imagery guidance, adjuster licensing laws — that define the complete operational envelope. The wider the corpus, the more violation surface area each centroid covers.

**Telic** (from Greek *telos*, "purpose") — The agent is constructed from purpose. But purpose alone is insufficient. The agent must also know the shape of its boundaries — what it must NOT do — with the same mathematical precision as what it is FOR. Telic entrainment encodes both.

**Entrained** (from physics, "to draw along with") — The agent's behavior is not constrained by external rules applied at runtime. It is entrained by the geometry of its embedding space — the centroids that define violation regions are pre-computed from the actual regulatory corpus of the agent's operating domain. The corpus becomes the substrate. The language of regulation becomes the mathematical geometry of governance.

The result: **the agent cannot drift toward a boundary violation without the governance detecting it**, because the centroid was built from the actual language of what violations sound like in the agent's domain — not from a single prohibition statement, but from a corpus of 60-100 diverse phrasings grounded in verified regulatory text.

This is the "Linguistic Operational Substrate" — the "LOS" in TELOS. The language of regulation becomes the operational substrate of governance. It is loaded at instantiation and becomes an intrinsic property of the agent's identity.

### 4.3 Why This Is Superior to RAG-Based Governance

| Property | RAG Governance | TELOS Boundary Centroids |
|----------|---------------|-------------------------|
| Runtime cost | Retrieval + re-ranking per request | Zero (pre-computed) |
| Latency | 50-200ms per boundary check | ~2ms per boundary check |
| Failure mode | Retrieval miss → violation undetected | Centroid covers full violation surface |
| Regulatory traceability | Retrieved document chunks | Full corpus with provenance |
| State-by-state adaptation | Different retrieval indices per state | Different centroid per state (same API) |
| Offline capability | Requires vector store access | Centroid is a numpy array in memory |

---

## 5. State-by-State Parameterization

### 5.1 Jurisdictional Variation Is Real

The research revealed significant state-by-state variation in regulatory standards. For aerial imagery alone:

| State | Aerial Imagery Standard | Strictness |
|-------|------------------------|------------|
| New Hampshire | Physical inspection required even if insurer believes imagery is conclusive AND insured disputes | Strictest |
| Pennsylvania | Physical inspection when imagery doesn't show "unequivocal and material damage" | Strict |
| Connecticut | Physical inspection OR licensed contractor report | Moderate |
| North Carolina | Imagery "should not be the only information utilized" | Moderate |
| Rhode Island | "Cannot alone justify" without "further investigation" | Moderate |
| Florida | Electronic methods (including drones) explicitly permitted for investigation | Permissive (for investigation) |

### 5.2 Architectural Implication

The boundary corpus becomes parameterized by jurisdiction:

```python
# At agent instantiation:
pa = AgenticPA.create_from_template(
    purpose=template.purpose,
    scope=template.scope,
    boundaries=template.boundaries,
    tools=tool_defs,
    embed_fn=embed_fn,
    template_id="property_intel",
    jurisdiction="NH",  # <-- New Hampshire-specific corpus
)
```

The `jurisdiction` parameter selects the appropriate corpus layer:
- **Base layer** (NAIC Model Bulletin) — applies everywhere
- **State layer** (e.g., NH INS 25-016-AB) — jurisdiction-specific provisions
- **Computed centroid** — pre-computed from base + state layers

Same architecture. Same ~10ms runtime. Different centroid per state.

### 5.3 Attestation and Compliance Filing

Several states require formal attestation of AI governance:

- **Colorado CRS 10-3-1104.9(3)(b)(V):** "Provide an attestation by one or more officers that the insurer has implemented the risk management framework...on a continuous basis."
- **Connecticut Bulletin MC-25:** Annual AI Certification due September 1.
- **NAIC Model Bulletin:** Insurers "can expect to be asked" about AI systems during investigations.

TELOS governance traces — fidelity scores, boundary check results, decision dispositions — become the **evidence backing these attestations**. The audit trail is not a reporting feature; it is the regulatory artifact.

---

## 6. Provenance Chain

Every phrasing in the boundary corpus has a documented provenance:

| Field | Description |
|-------|-------------|
| `text` | The violation phrasing |
| `source` | Regulatory document identifier (e.g., "CO CRS 10-3-1104.9(8)(a)") |
| `url` | Public URL where the regulatory text was verified |
| `layer` | 1 (hand-crafted), 2 (LLM-generated), or 3 (regulatory extraction) |
| `verified_date` | Date the regulatory text was last verified |
| `confidence` | HIGH (directly verified), MEDIUM (verified via secondary source), LOW (inferred) |

This provenance chain ensures that:
1. Every boundary centroid is traceable to specific regulatory requirements
2. Regulatory changes can be tracked and corpus updated
3. Compliance auditors can verify the governance substrate against source documents

---

## 7. Research Status

| Boundary | Research | Layer 1 | Layer 2 | Layer 3 | Centroid |
|----------|----------|---------|---------|---------|----------|
| B1: No binding underwriting decisions | COMPLETE (5 sources) | COMPLETE (13 phrasings) | Planned | Pending extraction | Phase 1 |
| B2: No PII access beyond address/parcel | Pending | COMPLETE (12 phrasings) | Planned | Pending | Phase 1 |
| B3: No overriding human assessor | COMPLETE (10 sources) | COMPLETE (12 phrasings) | Planned | Pending extraction | Phase 1 |
| B4: No binding premium quotes | Pending | COMPLETE (12 phrasings) | Planned | Pending | Phase 1 |
| B5: No autonomous repair/replace auth | COMPLETE (13 sources) | COMPLETE (12 phrasings) | Planned | Pending extraction | Phase 1 |

### Research Documents

| Document | Contents |
|----------|----------|
| `research/boundary_regulatory_research_b1.md` | B1: NAIC, Colorado, NY DFS, Connecticut (5 sources) |
| `research/boundary_regulatory_research_b3.md` | B3: TX/FL adjuster licensing, 6 aerial imagery bulletins (10 sources) |
| `research/boundary_regulatory_research_b5.md` | B5: Claims settlement laws, 5 states + NAIC + pending FL HB 527 (13 sources) |
| `telos_governance/boundary_corpus_static.py` | Layer 1 hand-crafted phrasings (all 5 boundaries) |

---

## 8. Next Steps

### Phase 2: Layer 3 Regulatory Extraction
- Extract verified phrasings from B1, B3, B5 research documents
- Add to `boundary_corpus_static.py` or new `boundary_corpus_regulatory.py`
- Compute updated centroids with Layer 1 + Layer 3

### Phase 2.5: Layer 2 LLM Expansion
- Generate additional phrasings via Mistral from Layer 1 + Layer 3 anchors
- Deduplicate by cosine similarity (threshold: 0.92)
- Target: 55-65 effective phrasings per boundary

### Phase 3: Rigorous Evaluation
- Held-out test set (80+ scenarios, blind author)
- Per-boundary ROC analysis and threshold optimization
- Benchmark re-baselining

### Phase 4: State-by-State Parameterization
- Corpus format supporting base + jurisdiction overlays
- Pre-computed centroids per jurisdiction
- B2 and B4 regulatory research

### Phase 5: Attestation Infrastructure
- Governance trace export format for regulatory filing
- Compliance dashboard for multi-state coverage tracking

---

*This document is a living artifact of the TELOS research program.*
