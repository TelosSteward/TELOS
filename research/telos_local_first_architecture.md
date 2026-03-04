# TELOS Local-First Governance Architecture
## Privacy by Design, Control by Cryptography, Compliance by Measurement
Jeffrey Brunner | TELOS AI Labs Inc. | JB@telos-labs.ai | February 2026

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## 1. The Architecture in One Sentence

TELOS is a governance engine that runs entirely on the client's infrastructure, measures every AI decision against a bespoke boundary corpus, produces cryptographically signed governance receipts, and gates all boundary changes through multi-party cryptographic co-signing — so that TELOS never sees client data and the client cannot unilaterally weaken their own governance.

---

## 2. What Runs Where

TELOS operates as middleware between the client's application and their AI capability surface. The governance engine intercepts requests, measures them against pre-computed boundary centroids using embedding geometry, renders a governance decision (EXECUTE, CLARIFY, SUGGEST, ESCALATE), and logs a signed governance receipt — all in under 10ms.

| Component | Location | Who Controls It |
|-----------|----------|----------------|
| Governance engine (5-dimension fidelity scoring) | Client infrastructure | TELOS code, client operation |
| Boundary corpus (what the AI must not do) | Client infrastructure, loaded by TELOS | Joint control via TKeys co-signing |
| Embedding model (MiniLM, 384-dim) | Client infrastructure, bundled with engine | TELOS provides, client runs |
| Governance receipts (per-decision audit trail) | Client infrastructure | Client owns entirely |
| Accumulated governance data | Client infrastructure | Client owns entirely |
| TKeys signing infrastructure | Distributed (each party holds their key) | Neither party acts alone |

**What TELOS never sees:** Client data. Governance traces. Fidelity metrics. Domain-specific patterns. Proprietary workflows. None of it. The architecture enforces this structurally — not by policy, not by contract, but by the fact that TELOS has no access path to client infrastructure.

**What the client never modifies alone:** The boundary corpus. The governance rules that define what the AI must not do are loaded by TELOS and gated through cryptographic co-signing. The client cannot unilaterally weaken boundaries, which prevents "governance washing" — deploying a governance system with boundaries so loose it provides the appearance of compliance without the substance.

---

## 3. How the Governance Engine Works

The engine operates on embedding geometry. It does not inspect model internals, modify model weights, or require access to the AI model's reasoning process. Any AI model that produces text can be governed.

**The process:**

1. A request arrives (e.g., "Assess roof condition for 742 Evergreen Terrace")
2. The engine embeds the request using a local embedding model (MiniLM, 384 dimensions)
3. The embedding is measured against pre-computed boundary centroids using cosine similarity — a mathematical measurement of how closely the request's meaning aligns with known violation patterns
4. Five governance dimensions are scored: purpose fidelity, scope fidelity, tool selection, chain continuity, and boundary enforcement
5. A governance decision is rendered: EXECUTE (proceed), CLARIFY (verify intent), SUGGEST (offer alternatives), or ESCALATE (require human review)
6. A governance receipt is generated and signed with HMAC-SHA512

**Performance:** ~3-6ms total governance overhead per request on CPU. For a request that takes 500-2000ms for LLM inference, governance adds less than 1% latency. The engine is invisible when the AI stays on task.

**Model-agnostic:** The engine does not need to know whether the client runs GPT-4, Claude, Llama, Mistral, or a proprietary model. It only needs the text of the request. The governance embedding model (MiniLM) is TELOS's internal concern, bundled with the engine, version-pinned by SHA-256 hash to ensure boundary embeddings and governance measurements are always computed against the same model.

---

## 4. Bespoke Boundary Corpus

Each client deployment receives a domain-specific boundary corpus — the set of rules defining what the AI must not do. The corpus is not a keyword list. It is a set of pre-computed centroid embeddings derived from regulatory text through a three-layer methodology:

| Layer | Source | Purpose |
|-------|--------|---------|
| Layer 1: Hand-crafted anchor phrasings | Human-authored, 12-15 per boundary | Define the core violation surface — what violations actually sound like |
| Layer 2: LLM-generated expansion | Model-generated gap-fillers, 25-35 per boundary | Cover phrasing variants and edge cases |
| Layer 3: Regulatory extractions | Verbatim regulatory text with provenance | Ground boundaries in specific statutes and bulletins |

The centroid embedding for each boundary is the geometric mean of all its phrasings — covering a broad region of violation space rather than a single point. When a request arrives, the engine measures its cosine similarity against each boundary centroid. A high similarity score means the request is semantically close to what a violation sounds like.

**Why this matters:** Single-text boundary definitions (e.g., "No binding underwriting decisions") catch only 18.2% of actual violations because embedding models cannot faithfully represent negation. Corpus-augmented centroids catch significantly more because they are built from affirmative action statements — what violations actually sound like in practice.

**Bespoke per client:** The Nearmap deployment has 5 insurance-specific boundaries mapped to NAIC FACTS and Colorado SB 24-205. A healthcare deployment would have different boundaries mapped to HIPAA and clinical decision-support regulations. A financial services deployment would map to MiFID II and Basel requirements. The governance engine is the same. The boundaries are bespoke.

---

## 5. TKeys: Cryptographic Co-Signing for Boundary Control

All boundary corpus changes require multi-party cryptographic authorization. Neither TELOS nor the client can act unilaterally.

### How it works

Think of it like a safe deposit box that requires two keys — the bank's key and the customer's key — to open. Neither party alone can change what's inside.

1. TELOS prepares a boundary update package (new phrasings, pre-computed embeddings, version tag)
2. TELOS signs the package with TELOS's private key
3. The package is delivered to the client
4. The client reviews the package and co-signs with the client's private key
5. The local governance engine verifies both signatures before loading the new boundaries
6. If either signature is missing or invalid, the engine rejects the update and continues operating with the existing boundary corpus

### Why multi-sig matters

- **Prevents vendor overreach:** TELOS cannot silently change what the governance engine considers a violation
- **Prevents governance washing:** The client cannot unilaterally weaken boundaries to avoid flagging inconvenient violations
- **Creates an auditable change management trail:** Every boundary change has a documented justification, both signatures, and a timestamp — the full chain of custody from regulatory requirement to deployed boundary
- **Structural revenue relationship:** Regulatory changes (NAIC updates, EU AI Act enforcement evolution, new state laws) require boundary corpus updates, which require TELOS's co-signature to deploy. The relationship is structural, not contractual.

### Current implementation status

| Capability | Status |
|-----------|--------|
| Session-bound encryption (AES-256-GCM) | Implemented |
| HMAC-SHA512 governance receipt signing | Implemented |
| HKDF key derivation with CSPRNG entropy floor | Implemented |
| Forward secrecy (session keys destroyed after use) | Implemented |
| Asymmetric key pairs for multi-sig (Ed25519) | Planned — requires `boundary_signer.py` |
| Key provisioning ceremony (public key exchange) | Planned |
| Dual-signature boundary package verification | Planned |
| Key revocation mechanism | Planned |

The existing TKeys infrastructure handles runtime governance trace encryption and signing. The multi-signature boundary co-signing is a separate cryptographic subsystem that extends TKeys with asymmetric key pairs — a different concern (configuration integrity vs. runtime data protection).

---

## 6. What the Client Gets: A Compliance Research Engine

Every governance receipt stays on the client's infrastructure. Over time, this accumulation becomes a domain-specific compliance research engine:

**Immediate value (day one):**
- Per-decision governance receipts proving what the AI did and did not do
- Audit trail sufficient for NAIC Model Bulletin attestation, Colorado SB 24-205 compliance, and EU AI Act Article 72 post-market monitoring
- Evidence of documented governance effort at every AI decision point

**Accumulated value (over months):**
- Patterns in which boundaries are most frequently approached
- Identification of workflows or user behaviors that produce higher governance risk
- Data to support tightening controls in specific areas based on observed behavior
- A growing body of evidence demonstrating continuous, systematic governance — the exact standard regulators look for

**What this means practically:** The client does not just have a governance engine. They have a running record of every AI decision their system has ever made, measured against the boundaries that define their compliance posture. When a regulator, carrier, or auditor asks "prove your AI is governed," the client produces the receipts. Same request, same receipt, every time. Deterministic and reproducible.

---

## 7. Demonstrable Due Diligence

TELOS provides governance measurement, not governance perfection. No system can guarantee that an AI will never make an error. What TELOS provides is documented evidence that governance was active, measuring, and enforcing boundaries at the moment of every AI decision.

This is the standard that regulators and courts actually apply. Every major AI regulation uses process-oriented language:

| Regulation | Standard |
|-----------|----------|
| EU AI Act, Article 9 | "Establish, implement, document and maintain" a risk management system |
| EU AI Act, Article 72 | "Active and systematic" post-market monitoring |
| NAIC Model Bulletin, Section III | Controls "reflective of, and commensurate with" the risk |
| Colorado CRS 10-3-1104.9 | "Reasonably designed" risk management framework |
| NY DFS Circular Letter No. 7 | Responsibility for "decisions or actions impacting consumers" |

None of these require perfection. All require documented process. TELOS governance receipts — cryptographically signed, per-decision, 5-dimensional measurements of what the AI did — provide exactly that documentation.

**What this is:** Evidence of reasonable, documented governance effort. A demonstrable record that the organization had governance in place, it was measuring every decision, and it acted on what it found.

**What this is not:** A guarantee. An insurance policy. A legal shield against all liability. TELOS measures governance. The client is responsible for acting on what the measurements show.

---

## 8. Responsibility Architecture

The local-first model creates a clear separation of control and responsibility:

| Domain | Who Controls It | Who Is Responsible |
|--------|----------------|-------------------|
| Client data and workflows | Client | Client |
| AI model selection and operation | Client | Client |
| Acting on governance findings | Client | Client |
| Infrastructure maintenance | Client | Client |
| Boundary corpus content and adequacy | TELOS (with client co-signing) | Shared — see below |
| Governance engine accuracy | TELOS | TELOS |
| Regulatory monitoring for boundary updates | TELOS | TELOS |
| Timeliness of boundary updates | TELOS | TELOS |

### Shared responsibility for boundaries

The boundary corpus is the most consequential component of the governance system. If boundaries miss a violation category, the engine will not detect it regardless of how well the math works. TELOS controls the boundary corpus because regulatory interpretation requires domain expertise. But control must come with accountability:

- TELOS commits to constructing boundary corpora using documented methodology (three-layer corpus with regulatory provenance)
- TELOS commits to disclosing measured detection rates to clients before boundary authorization
- TELOS commits to proactive regulatory monitoring — identifying when new regulations require boundary updates, not waiting for clients to ask
- The multi-sig co-signing process ensures the client participates in boundary authorization, creating shared ownership of the governance configuration

### Change management process

1. A regulatory change occurs (new NAIC bulletin, state law, EU AI Act implementing act) — or the client identifies a governance gap
2. TELOS evaluates the change, drafts boundary corpus updates, and reviews with legal counsel
3. TELOS presents the update package to the client with documentation of what changed and why
4. The client reviews and co-signs (or requests modifications)
5. Both signatures are verified by the local governance engine before loading the new boundaries
6. The old boundary corpus version, the new version, both signatures, and the justification are logged in the change management audit trail

### Response time commitments

| Change Type | Target Response |
|-------------|----------------|
| Pre-approved regulatory patterns (state adopts existing NAIC bulletin) | 48 hours |
| Standard regulatory updates (new state-specific requirements) | 5 business days |
| Novel regulatory requirements (new category of boundary) | 15 business days |

---

## 9. Provider-Level Commitment

Under the EU AI Act, a "provider" is an entity that develops an AI system and places it on the market. Providers face heavier obligations than "deployers" (entities that use AI systems). Because TELOS controls the boundary corpus through cryptographic co-signing — not just licensing passive software — TELOS occupies a role closer to provider than passive infrastructure supplier.

TELOS voluntarily holds itself to provider-level standards:

| EU AI Act Obligation | How TELOS Satisfies It |
|---------------------|----------------------|
| Article 9: Risk management system | Boundary corpus methodology with three-layer regulatory grounding |
| Article 11: Technical documentation | Open-source governance engine, published benchmark methodology |
| Article 12: Record-keeping | GovernanceEvent audit trails with per-decision context |
| Article 13: Transparency | Per-dimension explanations in every governance receipt |
| Article 14: Human oversight | Graduated response (CLARIFY → SUGGEST → ESCALATE) preserving human decision authority |
| Article 15: Accuracy, robustness, cybersecurity | TKeys cryptographic infrastructure, version-pinned embedding models |
| Article 72: Post-market monitoring | Continuous governance measurement at every decision point, change management audit trail |

This is not just a compliance posture. It is a competitive position. When a prospect asks "who is accountable for this governance system?", the answer is clear: TELOS holds itself to the highest standard the regulation defines.

---

## 10. What TELOS Is and Is Not

| TELOS Is | TELOS Is Not |
|---------|-------------|
| A governance measurement engine | A compliance guarantee |
| Privacy-by-architecture (never sees client data) | A data processor or data controller |
| Model-agnostic (governs any AI via embedding geometry) | An AI model or AI replacement |
| A bespoke boundary configuration for each client's domain | A one-size-fits-all product |
| A structural relationship gated by cryptographic co-signing | A software license that can be replaced by a competitor overnight |
| Evidence of documented governance effort | Legal immunity or regulatory safe harbor |
| Open-source (governance engine on github.com/TelosSteward/telos-hardened) | A black box |

---

## Summary

The client runs everything locally. TELOS never sees their data. The governance engine measures every AI decision against a bespoke boundary corpus built from regulatory text. Every decision produces a cryptographically signed governance receipt. The boundary corpus can only be changed when both TELOS and the client co-sign — neither party acts alone. The accumulated governance data belongs to the client and stays on their infrastructure.

The result: plug-and-play governance infrastructure that is model-agnostic, enterprise-agnostic, and privacy-preserving by architecture. The client gets a running record of every AI decision their system has ever made. TELOS gets a structural relationship gated by the cryptographic keys that authorize boundary changes. Regulators get the documented governance evidence they actually require.

*Actuarial proof of intent and execution for every AI decision within bounded work across its entire life cycle.*

**JB@telos-labs.ai** | github.com/TelosSteward/telos-hardened
