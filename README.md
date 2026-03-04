# TELOS

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org) [![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](docs/TELOS_Academic_Paper.pdf) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18370263-blue.svg)](https://doi.org/10.5281/zenodo.18370263) [![SAAI Compliant](https://img.shields.io/badge/SAAI-Framework_Compliant-green.svg)](research/regulatory/saai_requirement_mapping.md)

**A mathematical governance framework for AI agents.**

TELOS (Telically Entrained Linguistic Operational Substrate) uses Primacy Attractors — embedding-space representations of user-defined purpose — to detect and direct drift in autonomous AI agents. Every tool call is scored against a fixed specification using cosine similarity, producing a graduated governance verdict (proceed, verify intent, offer alternatives, block, or escalate to human review) with an Ed25519-signed cryptographic receipt.

No LLM required for governance decisions. Deterministic. Auditable. 15-25ms per decision.

**Version:** 2.0.0 | **License:** Apache 2.0 | **Contact:** JB@telos-labs.ai

---

## See It Work (3 Commands)

```bash
# Setup
git clone https://github.com/TelosSteward/TELOS.git
cd TELOS && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Score a single request against a governance specification
telos score "Assess roof condition for underwriting" -c templates/property_intel.yaml -v

# 2. Run the 4-domain PA swap demo (insurance -> solar -> healthcare -> civic)
#    Same governance engine, 4 radically different purposes, 29 turns, 12 boundary interventions
DEMO_FAST=1 python3 demos/nearmap_live_demo.py

# 3. Run full test suite (1,600+ tests)
pytest tests/ -v
```

For a comprehensive technical overview of the architecture, development history, and research program, see [CLAUDE.md](CLAUDE.md).

---

## What TELOS Does

An AI agent receives a tool call request. Before execution, TELOS:

1. **Measures** the request against the Primacy Attractor (purpose embedding) across 6 dimensions: purpose fidelity, scope fidelity, boundary fidelity, tool fidelity, chain continuity, and risk
2. **Detects** boundary violations via a 4-layer cascade: L0 keyword match, L1 cosine similarity, L1.5 SetFit ML classifier, L2 LLM (optional)
3. **Decides** using graduated thresholds:

| Verdict | Fidelity | Action |
|---------|----------|--------|
| **EXECUTE** | >= 0.85 | Proceed — tool runs, receipt signed |
| **CLARIFY** | 0.70-0.84 | Verify intent — governance context injected, tool runs with caution |
| **SUGGEST** | 0.50-0.69 | Offer alternatives — tool blocked, dimension-aware redirection |
| **INERT** | < 0.50 | Block — request falls outside defined scope |
| **ESCALATE** | < 0.50 + boundary | Escalate to human review — all tools blocked, human_required flag |

4. **Signs** a governance receipt (Ed25519) with full dimension breakdown, decision rationale, and config hash

All governance math runs locally via ONNX inference. No API calls. No cloud dependency. Deterministic: same input always produces the same decision.

---

## Core Innovation

**The "lost in the middle" problem (Liu et al., 2024):** LLMs exhibit strong primacy and recency biases — attending well to information at the beginning and end of context but poorly to middle positions. As conversations extend, constitutional constraints drift into poorly-attended regions. The model forgets its purpose as context accumulates.

**TELOS solves this** by establishing an external fixed reference point — the Primacy Attractor — that exists outside the model's context window entirely. Every exchange is measured against it using cosine similarity in embedding space. The governance signal never goes stale, never drifts into the middle of a forgotten context, never competes with the model's attention. Statistical Process Control (Shewhart, 1931) adapted into semantic space: continuous measurement at every decision point.

**The structural insight:** Conversational governance is a convergence problem (are two semantic signals staying aligned?). Agentic governance is a compliance problem (is this instrument operating within its defined specification?). The math is identical; the operational surface is more constrained and classifiable. This is why agentic governance achieves higher precision — tool calls are discrete events with known signatures, not fluid semantic territory.

Full analysis: `research/papers/convergence_to_compliance.md`

---

## Domains

TELOS governance adapts to any domain via PA swap — the same engine, different purpose specifications:

### Property Intelligence (Nearmap)
Insurance underwriting agent with aerial imagery analysis, roof condition assessment, vegetation risk scoring. 8 tools, 6 boundaries. Validated against 235 benchmark scenarios.

### Solar Site Assessment
Solar installation feasibility agent with irradiance lookup, shading analysis, utility rate comparison. 8 tools, 6 boundaries, 22 violation keywords.

### Healthcare (7 configurations)

| Config | Agent Type | Tools | Boundaries |
|--------|-----------|-------|------------|
| `healthcare_ambient` | Clinical AI scribe | 8 | 8 |
| `healthcare_call_center` | Scheduling/Rx/billing | 8 | 7 |
| `healthcare_coding` | ICD-10/CPT coding assist | 8 | 7 |
| `healthcare_diagnostic_ai` | Imaging/lab triage | 8 | 6 |
| `healthcare_patient_facing` | Portal Q&A/intake | 8 | 7 |
| `healthcare_predictive` | Sepsis/deterioration | 7 | 6 |
| `healthcare_therapeutic` | Treatment CDS/dosing | 8 | 8 |

Domain-specific SetFit boundary classifier (AUC 0.980). Validated against 280 benchmark scenarios across 12 attack families.

### Civic Services
Municipal government services agent — permits, utilities, voter registration, service requests. 6 tools, 5 boundaries. Demonstrates PA swap: same governance engine, radically different purpose.

### OpenClaw Autonomous Agent Governance
Runtime governance for [OpenClaw](https://github.com/nicepkg/openclaw) — the most widely deployed autonomous AI agent (200K+ stars, zero native governance). Domain-specific SetFit classifier (AUC 0.990). 10 tool groups, 6 attack families, 4 risk tiers. Every boundary traced to a sourced CVE or security incident.

---

## Cryptographic Trust Layer

### TKeys
Session-bound AES-256-GCM encryption using governance telemetry as supplementary entropy for HKDF key derivation. Per-turn key rotation with 128-bit CSPRNG floor. HMAC-SHA512 signing on every governance delta. Independent cryptographic primitive — publishable separately.

Full security review: `research/papers/tkeys_cryptographic_review.md` | Novelty analysis: `research/papers/tkeys_novelty_analysis.md`

### TKeys Activation Protocol (Liability Separation)
The governance engine is **inert by default**. Without a valid TKey signature on the PA configuration, the engine refuses to start.

1. Customer creates their PA configuration (purpose, boundaries, tools, constraints)
2. Customer signs with their TKey (`telos pa sign config.yaml`) — Ed25519 signature covers SHA-256 hash of full config
3. Activation ping to TELOS Labs for counter-signature (dual-attestation)
4. Only after valid signature does the engine activate

**What this proves:** WHO defined the governance boundaries, WHAT they defined, WHEN they signed, and that TELOS had zero involvement in the substance. Violations are the customer's responsibility — TELOS measures what the customer defined.

### Bundle Delivery
`.telos` binary format: `[TELO magic][version][manifest][64B labs_sig][64B deploy_sig][AES-256-GCM payload]`. One-command customer provisioning via `telos bundle provision`. Ed25519-signed license tokens for offline capability-scoped enforcement.

### Governance Receipts
Every decision produces an Ed25519-signed receipt: decision + config hash + timestamp + dimension breakdown + signer identity. Immutable audit trail. Designed to support post-market monitoring workflows consistent with EU AI Act Article 72 requirements.

---

## CLI

```bash
pip install -e .

# Governance scoring
telos version                                    # Show version
telos score "request text" -c config.yaml -v     # Score a single request
telos config validate config.yaml                # Validate a governance config
telos config list                                # List available templates
telos init                                       # Interactive config creation

# Demos
telos demo nearmap                               # Property intelligence (4 domains)
telos demo healthcare                            # Interactive healthcare menu (7 configs)
telos demo openclaw                              # Interactive OpenClaw menu (9 surfaces)
telos demo openclaw --all --fast                  # All 90 scenarios

# Benchmarks
telos benchmark run -b nearmap --forensic        # 235 scenarios
telos benchmark run -b healthcare --forensic     # 280 scenarios
telos benchmark run -b openclaw --forensic       # 100 scenarios

# Bundle delivery
telos bundle provision --agent-id ID --customer-id ID --output dir
telos bundle activate bundle.telos
telos pa sign config.yaml                        # Sign PA with customer TKey
telos pa verify config.yaml                      # Verify dual-attestation

# Agent governance (OpenClaw daemon)
telos agent init --detect                        # Auto-detect agent config
telos agent status                               # Show daemon status
telos service install                            # Install as launchd/systemd service
```

See [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) for all commands.

---

## Validation

### Adversarial (2,550 attacks, 0% ASR)

| Benchmark | Scenarios | Result | Venue | DOI |
|-----------|-----------|--------|-------|-----|
| AILuminate | 1,200 | 100% blocked | MLCommons | [10.5281/zenodo.18370263](https://doi.org/10.5281/zenodo.18370263) |
| MedSafetyBench | 900 | 100% blocked | NeurIPS 2024 | [10.5281/zenodo.18370659](https://doi.org/10.5281/zenodo.18370659) |
| HarmBench | 400 | 100% blocked | CAIS | [10.5281/zenodo.18370659](https://doi.org/10.5281/zenodo.18370659) |
| SB 243 Child Safety | 50 | 100% blocked | CA Legislature | [10.5281/zenodo.18370504](https://doi.org/10.5281/zenodo.18370504) |

### Agentic (1,468 scenarios, 100% DSR)

| Benchmark | Scenarios | Defense Success Rate | Venue | DOI |
|-----------|-----------|---------------------|-------|-----|
| PropensityBench | 977 | 100% | Scale AI / CMU / UK AISI | [10.5281/zenodo.18562833](https://doi.org/10.5281/zenodo.18562833) |
| AgentHarm | 352 | 100% | Gray Swan AI / ICLR 2025 | [10.5281/zenodo.18564855](https://doi.org/10.5281/zenodo.18564855) |
| AgentDojo | 139 | 100% | ETH Zurich / NeurIPS 2024 | [10.5281/zenodo.18565869](https://doi.org/10.5281/zenodo.18565869) |

### Domain-Specific

| Benchmark | Scenarios | Configs | Attack Families |
|-----------|-----------|---------|-----------------|
| Nearmap Property Intel | 235 | 1 | 9 |
| Healthcare | 280 | 7 | 12 |
| OpenClaw | 100 | 1 | 6 |
| Civic Services | 75 | 1 | — |

### Calibration

| Benchmark | Metric | Result |
|-----------|--------|--------|
| XSTest | False Positive Rate | 8.0% (vs 24.8% generic) |
| Governance Benchmark | Sessions | 46 multi-session, 8 domains |

### Optimizer

Governance Configuration Optimizer: 14 tunable parameters, Optuna TPE, 7 benchmarks, 5,212 total scenarios, four-gate ratchet (Cat A regression + holdout + less-restrictive block + GDD). Research Governance Charter: `research/architecture/optimizer_governance_charter.md`.

---

## Package Architecture

```
./
├── telos_core/                  # Pure mathematical engine (ZERO framework deps)
│   ├── constants.py             #   All calibration thresholds (single source of truth)
│   ├── primacy_math.py          #   Attractor geometry, basin membership
│   ├── fidelity_engine.py       #   Two-layer fidelity calculation
│   └── embedding_provider.py    #   Multi-model embeddings (MiniLM, MPNet, ONNX)
├── telos_governance/            # Governance gates + CLI + delivery system
│   ├── scoring/                 #   Fidelity scoring + governance protocol
│   │   ├── agentic_pa.py        #     Primacy Attractor construction + sub-centroid clustering
│   │   ├── agentic_fidelity.py  #     6-dimension composite fidelity scoring
│   │   ├── fidelity_gate.py     #     Two-tier conversational governance gate
│   │   ├── governance_protocol.py #   Audit-trail governance protocol
│   │   ├── threshold_config.py  #     ThresholdConfig dataclass (optimizer params)
│   │   └── tool_selection_gate.py #   Semantic tool ranking
│   ├── corpus/                  #   Boundary corpus (3 layers)
│   │   ├── boundary_corpus_static.py    # L1: 61 hand-crafted boundary phrasings
│   │   ├── boundary_corpus_llm.py       # L2: 121 LLM-generated gap-fillers
│   │   ├── boundary_corpus_regulatory.py # L3: 48 regulatory extractions
│   │   └── boundary_corpus_safe.py      # Safe / false-positive corpus
│   ├── crypto/                  #   Cryptographic layer (TKeys + signing)
│   │   ├── crypto_layer.py      #     AES-256-GCM encryption-at-rest for PA config IP
│   │   ├── receipt_signer.py    #     Ed25519 + HMAC-SHA512 governance receipt signing
│   │   ├── signing.py           #     Ed25519 key management for .telos bundles
│   │   ├── pa_signing.py        #     TKey PA approval + dual-attestation protocol
│   │   └── gate_signer.py       #     Time-bounded gate token signing
│   ├── bundle/                  #   Bundle delivery system
│   │   ├── bundle.py            #     .telos binary format (build, sign, encrypt, verify)
│   │   ├── licensing.py         #     Offline Ed25519-signed license tokens
│   │   └── bundle_pipeline.py   #     One-command customer delivery provisioning
│   ├── telemetry/               #   Intelligence + telemetry pipeline
│   │   ├── intelligence_layer.py #    Opt-in governance telemetry (off/metrics/full)
│   │   ├── telemetry_pipeline.py #    Telemetry processing pipeline
│   │   └── interpreter.py       #     Governance trace interpreter
│   ├── teloscope/               #   TELOSCOPE analysis + audit tools
│   │   ├── annotate.py          #     Session annotation
│   │   ├── compare.py           #     Config comparison
│   │   ├── inspect.py           #     Receipt inspection
│   │   ├── report.py            #     Governance report generation
│   │   ├── rescore.py           #     Replay + rescore sessions
│   │   ├── stats.py             #     Governance statistics
│   │   ├── sweep.py             #     Parameter sweep analysis
│   │   ├── timeline.py          #     Decision timeline
│   │   ├── validate.py          #     Config validation
│   │   └── teloscope_audit.py   #     Full session audit
│   ├── pa/                      #   Primacy Attractor construction + templates
│   │   ├── pa_constructor.py    #     PA builder from config
│   │   ├── pa_context.py        #     PA context management
│   │   ├── pa_extractor.py      #     PA extraction from documents
│   │   └── agent_templates.py   #     Pre-built agent PA templates
│   ├── adapters/                #   Internal adapter utilities
│   │   ├── action_chain.py      #     SCI tracking for multi-step actions
│   │   ├── response_manager.py  #     Agentic session orchestrator
│   │   ├── setfit_classifier.py #     SetFit ONNX boundary classifier (L1.5 cascade)
│   │   └── tool_semantics.py    #     Tool semantic definitions
│   ├── demo/                    #   Demo support modules
│   │   ├── mock_tools.py        #     MockToolExecutor for benchmarks + demo mode
│   │   ├── demo_audit_bridge.py #     Demo-to-audit bridge
│   │   └── demo_teloscope_analysis.py # Demo TELOSCOPE analysis
│   ├── cli.py                   #   CLI entry point (score, benchmark, demo, bundle, agent)
│   ├── config.py                #   YAML config schema, loader, validation
│   ├── session.py               #   GovernanceSessionContext lifecycle
│   └── report_generator.py      #   9-section forensic reports (HTML + JSONL + CSV)
├── telos_adapters/              # Framework adapters
│   ├── langgraph/               #   LangGraph wrapper, supervisor, swarm
│   ├── generic/                 #   @telos_governed decorator
│   └── openclaw/                #   OpenClaw autonomous agent governance adapter
│       ├── governance_hook.py   #     Core scoring bridge (tool call -> verdict)
│       ├── ipc_server.py        #     Unix Domain Socket server (0.05-0.2ms latency)
│       ├── daemon.py            #     Entry point: config -> hook -> IPC -> watchdog
│       └── plugin/              #     TypeScript OpenClaw hook plugin
├── telos_gateway/               # FastAPI API gateway (OpenAI-compatible)
├── telos_observatory/           # Streamlit UI (DEMO + BETA + AGENTIC)
├── telos_privacy/               # TKeys cryptographic layer
│   └── cryptography/            #   Session-bound AES-256-GCM + HKDF + HMAC-SHA512
├── tools/                       # Analysis scripts + research utilities
│   ├── governance_optimizer.py  #   Multi-seed threshold optimizer (Optuna TPE, 14 params)
│   ├── governance_comparison.py #   Cross-config comparison
│   └── run_backtest.py          #   Backtest runner
├── demos/                       # Live governance demos (4 domains)
├── templates/                   # YAML governance configs (10 templates)
├── models/                      # ONNX SetFit models (healthcare, openclaw)
├── tests/                       # 1,600+ tests
├── validation/                  # Benchmark datasets (5,200+ scenarios)
│   ├── nearmap/                 #   235 scenarios, 9 attack families
│   ├── healthcare/              #   280 scenarios, 12 attack families, 7 configs
│   └── openclaw/                #   100 scenarios, 6 attack families, 4 risk tiers
├── research/                    # Active research program (see research/README.md)
│   ├── papers/                  #   Core technical papers + design docs
│   ├── experiments/             #   MVEs, closures, experimental designs
│   ├── benchmarks/              #   Benchmark roadmaps, hypotheses, results
│   ├── regulatory/              #   Regulatory mappings + compliance analysis
│   ├── architecture/            #   Systems architecture + governance design
│   ├── data/                    #   Research datasets + validation artifacts
│   ├── scripts/                 #   Research scripts + utilities
│   └── planning/                #   Work orders, advisory reports, planning docs
├── supabase/                    # Activation backend (Edge Function + migration)
└── docs/                        # CLI reference, config reference, integration guide
```

---

## Theoretical Grounding

| Theorist | Concept | TELOS Implementation |
|----------|---------|---------------------|
| Jensen & Meckling (1976) | Principal-agent monitoring | PA = contract, fidelity scoring = monitoring, receipts = audit |
| Bovens (2007) | Actor-forum accountability | Governance receipts = accountable actor, ESCALATE = consequences |
| Ostrom (1990) | Graduated sanctions (DP5) | 5-verdict system (EXECUTE/CLARIFY/SUGGEST/INERT/ESCALATE) |
| Russell (2019) | Deference-under-uncertainty | ESCALATE + Permission Controller = defer when uncertain |
| Shewhart (1931) | Statistical Process Control | Continuous measurement at every decision point |

Regulatory mapping (self-assessed): IEEE 7000, SAAI, EU AI Act, NIST AI RMF, OWASP Agentic Top 10, UC Berkeley CLTC Agentic AI Profile, Singapore IMDA Framework. Mappings in `docs/` and `research/`.

---

## Research Program

Key documents in `research/` (see `research/README.md` for full index):

| Document | Contents |
|----------|----------|
| `benchmarks/agentic_governance_hypothesis.md` | Core hypothesis + H6-H10 autonomous agent hypotheses |
| `papers/convergence_to_compliance.md` | Why agentic governance is a compliance problem, not convergence |
| `papers/telos_agentic_architecture.md` | Full agentic architecture design |
| `papers/tkeys_cryptographic_review.md` | 5-agent TKeys cryptographic security review |
| `papers/tkeys_novelty_analysis.md` | TKeys novelty, provenance, and security analysis |
| `experiments/setfit_mve_phase2_closure.md` | SetFit boundary classifier experimental record (AUC 0.980) |
| `architecture/optimizer_governance_charter.md` | Research Governance Charter (TELOS-RGC-001) |
| `benchmarks/agentic_benchmark_roadmap.md` | 13 external benchmarks across 3 phases |

---

## Key Documentation

| Document | Description |
|----------|-------------|
| [Academic Paper (PDF)](docs/TELOS_Academic_Paper.pdf) | Peer-review ready paper with full methodology |
| [Whitepaper v3.0](docs/TELOS_Whitepaper_v3.0.md) | Complete mathematical specification |
| [Technical Brief (PDF)](docs/TELOS_Technical_Brief.pdf) | Concise 3-page summary |
| [CLI Reference](docs/CLI_REFERENCE.md) | All CLI commands with examples |
| [Config Reference](docs/CONFIG_REFERENCE.md) | Full YAML governance schema |
| [Integration Guide](docs/INTEGRATION_GUIDE.md) | 5 integration patterns |
| [Berkeley CLTC Mapping](docs/BERKELEY_CLTC_MAPPING.md) | NIST AI RMF agentic extension alignment |
| [CLAUDE.md](CLAUDE.md) | Full development guide, architecture, and research team spec |

---

## Citation

```bibtex
@dataset{brunner_2025_telos_adversarial,
  author       = {Brunner, Jeffrey},
  orcid        = {0009-0003-6848-8014},
  title        = {{TELOS Adversarial Validation Dataset}},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18370659}
}
```

Additional citations for PropensityBench, AgentHarm, AgentDojo, SB 243, and Governance Benchmark datasets are available on [Zenodo](https://zenodo.org/search?q=TELOS%20Brunner).

---

## Running Tests

```bash
# Full test suite (1,600+ tests)
pytest tests/ -v

# TKeys cryptographic verification (22 tests)
python3 -m telos_privacy.cryptography.test_verify_crypto

# Nearmap benchmark (235 scenarios)
python3 validation/nearmap/run_nearmap_benchmark.py --forensic -v

# Healthcare benchmark (280 scenarios, 7 configs)
python3 validation/healthcare/run_healthcare_benchmark.py --forensic -v

# OpenClaw benchmark (100 scenarios)
python3 validation/openclaw/run_openclaw_benchmark.py --forensic -v
```

---

---

## Scope and Limitations

TELOS enforces behavioral fidelity to a declared Primacy Attractor (PA). It does not evaluate whether the PA itself is safe, beneficial, or aligned with human values. PA authorship requires external value alignment processes not provided by this framework.

**What TELOS is:** A behavioral specification enforcement tool with embedding-space monitoring and tiered response. Deterministic governance decisions at the tool-call level.

**What TELOS is not:** A solution to the AI alignment problem, a formal safety guarantee, or a substitute for human judgment in defining agent purpose. It is a research-stage framework under active development.

**Benchmark scope:** Validation results reported above reflect performance on specific, fixed benchmark sets evaluated by the authors. They do not generalize to novel attack vectors not represented in the evaluation data. Benchmark performance on known attack sets should not be interpreted as guarantees of real-world safety. See [VALIDATION_STATUS.md](VALIDATION_STATUS.md) for data availability and reproduction instructions.

**Regulatory status:** Regulatory mappings in this repository (EU AI Act, NIST AI RMF, IEEE 7000) represent the authors' self-assessment of alignment. They have not been reviewed or certified by any regulatory body. Organizations using TELOS in regulated deployments are responsible for their own compliance determinations.


## License

Apache License 2.0 — See [LICENSE](LICENSE) file.

## Author

**Jeffrey Brunner**
TELOS AI Labs Inc.
ORCID: [0009-0003-6848-8014](https://orcid.org/0009-0003-6848-8014)
Email: JB@telos-labs.ai
