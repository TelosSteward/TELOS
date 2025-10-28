# Section 7: Intellectual Property

**Status**: Patent Disclosure Prepared
**Owner**: Origin Industries PBC (to be formed Q1 2026)
**Core Innovation**: T-Keys Cryptographic Attribution Layer

---

## Overview

The TELOS platform contains patentable innovations centered on **T-Keys** (Telic Keys), a cryptographic layer for attributing AI governance decisions to specific regulatory attractors.

**Patent Disclosure**: Full technical specification available in separate document
**Filing Timeline**: Q1 2026 (provisional), Q1 2027 (full utility)
**Jurisdiction**: United States (primary), international (PCT)

---

## Core Patent: T-Keys Cryptographic Attribution

### Invention Summary

**Title**: "Cryptographic Attribution of Multi-Attractor AI Governance Decisions"

**Abstract**:
A system and method for cryptographically attributing AI governance decisions to specific regulatory attractors in a multi-stakeholder governance environment. The invention enables tamper-proof audit trails showing which attractor(s) influenced each governance decision, with cryptographic proof of decision provenance suitable for regulatory compliance and legal admissibility.

**Problem Solved**:
In multi-stakeholder AI governance, when multiple regulatory attractors (medical, financial, legal) evaluate the same conversation, it's unclear which attractor's judgment led to a specific governance action. This creates:
- Accountability gaps
- Compliance verification challenges
- Difficulty assigning legal responsibility
- Inability to audit regulatory effectiveness

**Solution**:
T-Keys system provides:
- Unique cryptographic key per attractor
- Signed governance decisions
- Tamper-proof attribution chain
- Verifiable audit trail
- Non-repudiation of regulatory judgments

---

## Seven Core Claims

### Claim 1: Multi-Attractor Attribution System

**Claim**:
A system for attributing governance decisions to multiple regulatory attractors comprising:
- A plurality of governance attractors, each associated with a unique cryptographic key pair
- A salience extraction module that processes conversation state
- A parallel evaluation engine that applies each attractor to the extracted salience
- A decision attribution module that cryptographically signs each attractor's evaluation
- An audit trail generator that creates tamper-proof records of attractor contributions

**Innovation**: First system to provide cryptographic attribution in multi-stakeholder AI governance

### Claim 2: Signed Fidelity Evaluations

**Claim**:
A method for cryptographically signing fidelity evaluations comprising:
- Calculating fidelity score for a message relative to an attractor
- Generating a hash of (message, fidelity, timestamp, attractor_id)
- Signing the hash with the attractor's private key
- Storing the signed evaluation in an immutable audit trail

**Innovation**: Makes fidelity evaluations non-repudiable and legally admissible

### Claim 3: Attribution Chain

**Claim**:
A method for creating verifiable chains of governance decisions comprising:
- Initial message evaluation signed by all active attractors
- Intervention decision signed by triggering attractor(s)
- Counterfactual experiments signed by comparison engine
- Complete chain linking message → evaluation → decision → outcome

**Innovation**: Enables complete accountability from input to governance action

### Claim 4: Consensus Verification

**Claim**:
A method for cryptographically verifying consensus decisions comprising:
- Individual attractor evaluations, each signed
- Consensus computation algorithm
- Consensus result signed by consensus engine
- Verification that consensus correctly aggregates individual signatures

**Innovation**: Proves consensus wasn't manipulated after individual evaluations

### Claim 5: Counterfactual Attribution

**Claim**:
A method for attributing counterfactual branch outcomes to specific attractors comprising:
- Fork point snapshot signed by session manager
- Baseline branch turns, each signed by evaluation engine
- TELOS branch turns, each signed by intervening attractor
- ΔF calculation signed by comparison engine
- Complete attribution: which attractor caused which improvement

**Innovation**: Provides cryptographic proof of governance efficacy attribution

### Claim 6: Regulatory Audit Interface

**Claim**:
A system for regulatory audit of multi-attractor governance comprising:
- Public key registry for all regulatory attractors
- Signature verification engine
- Audit trail query interface
- Compliance report generator with signature chains

**Innovation**: Enables regulators to independently verify governance decisions

### Claim 7: Federated Attractor Verification

**Claim**:
A method for federating attractor governance across organizational boundaries comprising:
- Attractor configuration signed by regulatory authority
- Remote attestation of attractor deployment
- Cross-organizational signature verification
- Proof that deployed attractor matches approved configuration

**Innovation**: Enables multi-organization governance with provable compliance

---

## Technical Architecture

### T-Keys Infrastructure

```
┌─────────────────────────────────────────────────────────┐
│                     T-Keys System                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐  ┌────────────────┐                 │
│  │ Medical        │  │ Financial      │                 │
│  │ Attractor      │  │ Attractor      │  ...            │
│  │ + Private Key  │  │ + Private Key  │                 │
│  └───────┬────────┘  └───────┬────────┘                 │
│          │                    │                          │
│          ├────────────────────┴─────────────┐            │
│          │                                  │            │
│    ┌─────▼──────────────────────────────────▼──────┐    │
│    │      Parallel Evaluation Engine              │    │
│    │  (Salience → Fidelity → Signature)           │    │
│    └─────┬────────────────────────────────────────┘    │
│          │                                              │
│    ┌─────▼──────────────────────────────────────┐      │
│    │    Consensus Engine                        │      │
│    │  (Aggregate + Sign Consensus)              │      │
│    └─────┬──────────────────────────────────────┘      │
│          │                                              │
│    ┌─────▼──────────────────────────────────────┐      │
│    │    Attribution Chain Builder               │      │
│    │  (Link signatures into audit trail)        │      │
│    └─────┬──────────────────────────────────────┘      │
│          │                                              │
│    ┌─────▼──────────────────────────────────────┐      │
│    │    Immutable Audit Trail                   │      │
│    │  (Blockchain-style linked signatures)      │      │
│    └────────────────────────────────────────────┘      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Signature Format

```python
@dataclass
class SignedEvaluation:
    """Cryptographically signed attractor evaluation."""
    message_hash: str  # SHA-256 of message
    attractor_id: str
    fidelity: float
    timestamp: float
    metadata: Dict[str, Any]
    signature: str  # RSA signature of above fields
    public_key_id: str  # For verification

    def verify(self, public_key: str) -> bool:
        """Verify signature with attractor's public key."""
        # Reconstruct signed data
        data = f"{self.message_hash}|{self.attractor_id}|{self.fidelity}|{self.timestamp}"

        # Verify RSA signature
        return crypto.verify(data, self.signature, public_key)
```

### Attribution Chain

```python
@dataclass
class AttributionChain:
    """Complete governance decision chain with signatures."""
    chain_id: str
    message: str
    message_hash: str

    # Individual evaluations (signed)
    evaluations: List[SignedEvaluation]

    # Consensus (signed)
    consensus_fidelity: float
    consensus_signature: str

    # Decision (signed)
    decision: str  # "continue", "intervene", etc.
    decision_signature: str
    deciding_attractor_ids: List[str]

    # Counterfactual (if applicable)
    counterfactual_experiment_id: Optional[str]
    delta_f_signature: Optional[str]

    def verify_chain(self, public_keys: Dict[str, str]) -> bool:
        """Verify entire attribution chain."""
        # 1. Verify all individual evaluations
        for eval in self.evaluations:
            if not eval.verify(public_keys[eval.attractor_id]):
                return False

        # 2. Verify consensus signature
        if not self._verify_consensus(public_keys):
            return False

        # 3. Verify decision signature
        if not self._verify_decision(public_keys):
            return False

        # 4. Verify counterfactual (if present)
        if self.counterfactual_experiment_id:
            if not self._verify_counterfactual(public_keys):
                return False

        return True  # All signatures valid
```

---

## Implementation Timeline

### Phase 1: Cryptographic Infrastructure (Q1 2026)

**Objectives**:
- Implement key generation and management
- Build signature system
- Create verification layer

**Deliverables**:
- Key management module (~200 lines)
- Signature utilities (~150 lines)
- Verification engine (~100 lines)

### Phase 2: Attractor Integration (Q2 2026)

**Objectives**:
- Integrate T-Keys with existing attractors
- Modify evaluation pipeline for signing
- Update audit trail storage

**Deliverables**:
- Modified PrimacyAttractor with signing (~50 lines added)
- Updated ParallelStewardManager (~100 lines added)
- Enhanced WebSessionManager for signature storage (~75 lines added)

### Phase 3: Audit Interface (Q3 2026)

**Objectives**:
- Build regulatory audit dashboard
- Create signature verification tools
- Implement compliance reporting

**Deliverables**:
- Audit dashboard UI (~400 lines)
- Verification CLI tool (~200 lines)
- Compliance report generator (~150 lines)

### Phase 4: Federated Deployment (Q4 2026)

**Objectives**:
- Enable cross-organizational verification
- Remote attestation protocols
- Distributed key management

**Deliverables**:
- Federation protocol (~300 lines)
- Remote attestation module (~200 lines)
- Distributed PKI (~250 lines)

**Total T-Keys System**: ~2,175 additional lines

---

## Patent Filing Strategy

### Provisional Application (Q1 2026)

**Target Date**: March 2026

**Scope**:
- All 7 core claims
- Technical architecture
- Implementation examples
- Use cases (medical, financial, legal)

**Cost**: $2,000 - $5,000 (including attorney fees)

**Purpose**: Establish priority date, 12-month window for full filing

### Full Utility Application (Q1 2027)

**Target Date**: March 2027 (within 12 months of provisional)

**Scope**:
- Refined claims based on implementation experience
- Additional dependent claims
- Broader prior art analysis
- Detailed embodiments

**Cost**: $10,000 - $20,000 (including prosecution)

**Timeline**: 18-36 months to first office action

### International Protection (2027-2028)

**PCT Application** (within 12 months of US filing):
- Target: March 2027
- Scope: Major jurisdictions (EU, UK, CA, AU, JP)
- Cost: $15,000 - $30,000

**National Phase Entries** (within 30 months of PCT):
- Individual country filings
- Translation costs
- Local attorney fees
- Estimated: $50,000 - $100,000 total

---

## Prior Art Analysis

### Existing Systems (Non-Infringing)

**Multi-Party Signatures** (e.g., Bitcoin):
- Different: Consensus among identical actors, not regulatory attractors
- Not applicable: No governance profile differentiation

**AI Decision Explainability** (e.g., LIME, SHAP):
- Different: Feature attribution, not regulatory attractor attribution
- Not applicable: No cryptographic proof

**Audit Logging Systems** (e.g., blockchain audit trails):
- Different: Generic event logging, not governance-specific
- Not applicable: No multi-attractor architecture

**Medical Device Audit Trails** (FDA requirements):
- Different: Single-organization compliance, not multi-stakeholder
- Not applicable: No cryptographic attribution to competing regulations

### Novel Aspects

**T-Keys is novel in**:
1. Multi-stakeholder regulatory attribution
2. Cryptographic proof of attractor contributions
3. Counterfactual governance efficacy attribution
4. Federated attractor verification
5. Regulatory audit interface with signature chains

---

## Ownership and Licensing

### Owner: Origin Industries PBC

**Entity Type**: Delaware Public Benefit Corporation
**Formation**: Q1 2026
**Purpose**: Develop and license TELOS governance infrastructure

**Ownership Structure**:
- Founders: TBD
- Early team: Equity allocation TBD
- Investors: TBD (if applicable)

### Licensing Strategy

**Option 1: Open Core**
- Core TELOS platform: Open source (Apache 2.0)
- T-Keys layer: Proprietary, licensed
- Revenue: Licensing fees for production deployments

**Option 2: Full Proprietary**
- Entire system: Proprietary
- Revenue: SaaS model, usage-based pricing
- Risk: Lower adoption, less community contribution

**Option 3: Patent Pool**
- License to regulatory partners royalty-free
- License to commercial entities with fees
- Revenue: Enterprise licensing

**Decision**: TBD in corporate formation discussions (Q1 2026)

### Licensing Terms (Proposed)

**Regulatory Partners**:
- Royalty-free license for governance configuration
- Non-exclusive rights
- Attribution required

**Commercial Entities**:
- Usage-based licensing (per 1,000 evaluations)
- Annual minimums for production use
- Support and updates included

**Open Source Projects**:
- Apache 2.0 for core platform
- T-Keys: Restricted license (non-commercial use OK)

---

## Defensive Publications

To protect freedom to operate, we will publish:

1. **TELOS Mathematical Foundation** (pre-patent)
   - Published: TBD (open source release)
   - Prevents others from patenting basic attractor math

2. **TELOSCOPE Counterfactual Methodology** (pre-patent)
   - Published: Research paper (Q1 2026)
   - Prevents others from patenting evidence generation

3. **Parallel Architecture** (pre-patent)
   - Published: Technical blog (Q4 2025)
   - Prevents others from patenting shared salience extraction

**Strategy**: Patent T-Keys (core innovation), publish rest as prior art

---

## Trade Secrets

Some aspects will remain trade secrets rather than patented:

1. **Attractor Optimization Heuristics**: Methods for tuning basin radius
2. **Consensus Weighting Algorithms**: Specific formulas for multi-attractor consensus
3. **Performance Optimizations**: Caching, batching, efficiency tricks
4. **Partner Configurations**: Domain-specific attractor configurations

**Reasoning**: Trade secrets don't expire, patents do (20 years)

---

## IP Risk Mitigation

### Patent Search (Q4 2025)

**Activities**:
- Prior art search for T-Keys concepts
- Freedom to operate analysis
- Competitive landscape review

**Budget**: $3,000 - $5,000 (professional search)

### Patent Attorney Engagement (Q1 2026)

**Selection Criteria**:
- Experience with software patents
- Expertise in cryptography/security
- AI/ML patent background preferred

**Budget**: $5,000 retainer + hourly

### IP Insurance (Q2 2026)

**Coverage**:
- Patent infringement defense
- IP litigation costs
- License dispute resolution

**Budget**: $5,000 - $10,000/year

---

## Cross-Reference

**TASKS.md Section 7**: IP-related tasks
**TELOS_BUILD_MANIFEST.md**: Main navigation
**Section 8: Corporate Structure**: Entity formation for IP ownership

---

## Summary

**Core Innovation**: T-Keys cryptographic attribution layer

**Patent Strategy**:
- Provisional: Q1 2026 ($5K)
- Full Utility: Q1 2027 ($20K)
- International: 2027-2028 ($100K)

**7 Core Claims**:
1. Multi-attractor attribution system
2. Signed fidelity evaluations
3. Attribution chain
4. Consensus verification
5. Counterfactual attribution
6. Regulatory audit interface
7. Federated attractor verification

**Owner**: Origin Industries PBC (to be formed Q1 2026)

**Licensing**: TBD (open core vs proprietary vs patent pool)

🔐 **Purpose: Protect core innovation while enabling regulatory co-development**
