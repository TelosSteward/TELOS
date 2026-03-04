# Security Policy

## Architecture Transparency (Kerckhoffs's Principle)

TELOS follows Kerckhoffs's principle: the governance architecture is fully public. Security
does not depend on secrecy of the design. All scoring algorithms, cascade logic, verdict
thresholds, cryptographic protocols, and integration patterns are published in this repository
and in the accompanying academic paper.

The following components are withheld from the public repository. Their absence does not
compromise the security of the architecture -- it protects the commercial value of calibrated
deployments.

## What Is Withheld

### Trained Model Weights (SetFit ONNX)

The boundary detection classifiers (healthcare AUC 0.980, OpenClaw AUC 0.990) are trained
SetFit models exported to ONNX format. Pre-trained weights are available on the
[HuggingFace Hub](https://huggingface.co/TelosSteward) for research and evaluation.
Production-grade models with domain-specific fine-tuning are available through TELOS Enforce.

### Calibration Thresholds and Boundary Corpus

The fidelity thresholds (EXECUTE/CLARIFY/SUGGEST/INERT/ESCALATE boundaries), dimension
weights, and the curated boundary corpus that trains the L1.5 SetFit cascade layer are
calibrated per-domain using the Governance Configuration Optimizer (14 tunable parameters,
5,212 scenarios, four-gate ratchet). These represent significant research investment and are
available through TELOS Enforce commercial licensing.

### Production Daemon Configuration

The OpenClaw adapter daemon configuration, IPC socket parameters, watchdog settings, and
production presets are deployment-specific. Reference configurations and integration
documentation are provided in this repository.

### Governance Traces and Forensic Reports

Ed25519-signed governance receipts, forensic HTML reports, and audit JSONL streams are
generated per-deployment by the governance engine. They contain customer-specific operational
data and are never aggregated or stored by TELOS AI Labs.

## Responsible Disclosure

If you discover a vulnerability in the TELOS governance engine -- including but not limited
to governance bypass techniques, cryptographic weaknesses in TKeys or receipt signing,
adversarial inputs that evade the boundary detection cascade, or authentication/authorization
flaws in the daemon IPC protocol -- we ask that you report it responsibly.

**Governance bypass discoveries should be reported, not exploited.** The purpose of this
engine is to keep AI agents operating within their defined boundaries. Circumventing that
purpose undermines safety for all deployments.

### How to Report

- **Email:** security@telosailabs.com
- **GitHub:** [Security Advisories](https://github.com/TelosSteward/TELOS/security/advisories) (private by default)

Please include:

1. Description of the vulnerability and affected component
2. Steps to reproduce (or proof of concept)
3. Potential impact assessment
4. Your suggested severity (Critical / High / Medium / Low)

### Response Timeline

| Stage | Target |
|-------|--------|
| Acknowledgment | 48 hours |
| Initial assessment | 5 business days |
| Fix or mitigation | 30 days (critical), 90 days (non-critical) |
| Public disclosure | Coordinated with reporter after fix is deployed |

### Scope

In scope: all code in this repository, published ONNX models, CLI tooling, daemon IPC,
TKeys cryptographic layer, bundle format, and governance receipt verification.

Out of scope: third-party dependencies (report upstream), social engineering, denial of
service against test infrastructure.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | Yes       |
| < 2.0   | No        |
