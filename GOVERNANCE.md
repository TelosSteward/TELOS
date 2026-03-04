# Governance of the Governance Engine

This document describes how TELOS governs its own development -- the meta-governance
process that ensures changes to the scoring engine, boundary detection cascade, calibration
thresholds, and cryptographic layer are validated before they reach any deployment.

## Advisory Review Process

Significant architectural decisions undergo structured advisory review before implementation.
Each review convenes a panel of domain experts (governance theory, systems engineering, data
integrity, regulatory compliance, AI safety, and cybersecurity) who assess the proposed
change independently, then synthesize convergence points (3+ agree) and divergences.

Findings are classified by priority:

| Priority | Meaning | Requirement |
|----------|---------|-------------|
| P0 | Must fix before merge | Blocks release |
| P1 | Should fix before merge | Blocks release unless explicitly deferred with rationale |
| P2 | Improvement | Tracked, not blocking |

All advisory assessments are numbered (A1, A2, ...) and archived with their verdicts,
actions taken, and any deferred items.

## Change Validation Requirements

### Scoring Engine Changes

Any modification to `telos_core/` (primacy math, fidelity engine, constants, embedding
provider) or `telos_governance/` (agentic fidelity, governance protocol, SetFit classifier,
response manager) must satisfy:

1. **Full test suite passes** -- 1,600+ tests, zero failures, zero skips on modified code
2. **Benchmark regression gate** -- no category may regress below its established floor:
   - Category A (purpose fidelity): >= 80%
   - Category B (boundary detection): >= 90%
   - Category C (graduated response): >= 87%
   - Category D (scope compliance): descriptive
   - Category E (chain continuity): >= 60%
   - False Positive Rate: <= 10%
3. **Adversarial holdout** -- changes must not introduce new attack surfaces in the
   2,550-scenario adversarial benchmark (AILuminate, MedSafetyBench, HarmBench, SB 243)
4. **Agentic holdout** -- changes must not regress the 1,468-scenario agentic benchmark
   (PropensityBench, AgentHarm, AgentDojo)

### Threshold, Calibration, and Corpus Changes

Modifications to verdict thresholds, dimension weights, optimizer parameters, or boundary
corpus additionally require optimizer governance charter compliance (TELOS-RGC-001:
four-gate ratchet), before/after metrics in the commit message, and -- for corpus changes
-- retrained models with documented AUC and new terms traced to sourced CVEs or regulatory
requirements.

### Cryptographic Layer Changes

Modifications to TKeys, receipt signing, bundle format, or PA signing additionally require
cybersecurity advisory review and no reduction in cryptographic strength.

## Two-Gate Architecture

TELOS uses a two-gate scoring architecture to prevent single-point-of-failure in governance
decisions:

- **Gate 1 (Fidelity):** Six-dimension composite scoring against the Primacy Attractor
  (purpose, scope, boundary, tool, chain, risk). Produces a numeric fidelity score and
  maps it to a graduated verdict.
- **Gate 2 (Boundary):** Four-layer boundary detection cascade (L0 keyword, L1 cosine
  similarity, L1.5 SetFit ML classifier, L2 optional LLM). Produces an independent
  boundary violation signal.

A request must pass both gates to receive an EXECUTE verdict. Either gate can independently
block or escalate. This eliminates the failure mode where a single scoring mechanism is
gamed or miscalibrated.

## TELOSCOPE: Independent Measurement

TELOSCOPE is the observation and research instrument for the TELOS governance engine. It
provides independent measurement capabilities that operate outside the governance scoring
pipeline:

- **Corpus analysis** -- inspect and validate the boundary corpus independently
- **Rescore and sweep** -- replay governance decisions with modified parameters
- **Timeline and compare** -- track governance behavior across sessions and configurations
- **Validate** -- verify cryptographic receipt integrity and audit trail completeness
- **Annotate and report** -- structured research output for third-party review

TELOSCOPE cannot modify governance behavior. It is read-only by design. This separation
ensures that the measurement instrument does not influence the system it measures.

## SAAI Framework Compliance

TELOS development follows the Structured Access for AI Integrity (SAAI) framework:

- **Graduated access** -- five-verdict system maps directly to SAAI's graduated sanctions
- **Deference under uncertainty** -- ESCALATE verdict defers to human review when the
  governance engine cannot determine alignment with sufficient confidence
- **Audit trail** -- every governance decision produces a signed, immutable receipt
- **Scope restriction** -- the governance engine cannot self-authorize expanded scope;
  PA configurations are externally defined and cryptographically signed by the customer
- **Failure mode** -- when uncertain, TELOS restricts rather than permits

## Contributing

Changes to governance-critical code (anything in `telos_core/`, `telos_governance/`,
`telos_privacy/`, or `telos_adapters/`) require:

1. A passing test suite with the change applied
2. Benchmark results demonstrating no regression
3. A clear description of what changed and why in the pull request

For changes that modify scoring behavior, threshold values, or cryptographic protocols,
please open an issue first to discuss the proposed change before submitting code.
