# OpenClaw Autonomous Agent — Regulatory Mapping

**Product:** TELOS-OpenClaw Governance Adapter (`telos-openclaw`)
**Organization:** TELOS AI Labs Inc.
**Contact:** JB@telos-labs.ai
**Date:** February 18, 2026
**Version:** 1.0
**Scope:** Regulatory requirements specific to autonomous AI agent governance, extending existing TELOS mappings for always-on, 24/7 operation

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## 1. Executive Summary

OpenClaw is the most widely deployed autonomous AI agent (200K+ GitHub stars, 24,478 exposed instances). Unlike session-based conversational AI, OpenClaw operates continuously — executing tool calls, managing scheduled tasks, and interacting across messaging channels without human oversight of individual decisions.

This document extends the existing TELOS regulatory mappings (IEEE 7000-2021, SAAI, EU AI Act, NIST AI RMF) to address requirements specific to **autonomous agent governance**. Existing mappings cover conversational/session-based AI. Autonomous agents introduce additional regulatory considerations around continuous monitoring, multi-channel operation, persistence mechanisms, and cumulative authority.

**Key differences from session-based governance:**
- **Temporal scope:** Minutes/hours per session → weeks/months of continuous operation
- **Decision volume:** ~10-50 per session → ~30,000 per month
- **Authority accumulation:** Bounded by session → unbounded over time
- **Channel multiplicity:** Single interface → 11 tool groups across 4 risk tiers
- **Failure consequence:** Session-scoped impact → persistent system-level compromise

---

## 2. CVE-to-Regulatory Mapping

Every documented OpenClaw security incident maps to specific regulatory requirements. This traceability ensures governance boundaries are rooted in real-world risk.

| Incident | CVE/Source | Regulatory Requirements Triggered |
|----------|-----------|----------------------------------|
| **RCE via WebSocket hijacking** | CVE-2026-25253 (CVSS 8.8) | EU AI Act Art. 9 (risk management), Art. 15 (robustness), NIST AI RMF GOVERN 1.2, IEEE 7000 EVR |
| **OS command injection** | CVE-2026-25157 | EU AI Act Art. 9, Art. 15, SAAI SFR-Security, IEEE 7002 (input validation) |
| **API token exposure** | Moltbook breach (Wiz Research) | EU AI Act Art. 12 (record-keeping), Art. 73 (incident reporting), IEEE 7002 (data protection), SAAI SFR-Transparency |
| **Malicious skill marketplace** | ClawHavoc (Cisco) — 341 skills, 12% of ClawHub | EU AI Act Art. 9 (supply chain risk), NIST AI RMF MAP 3.1, SAAI SFR-Security, IEEE 7000 (stakeholder impact) |
| **Agent identity theft** | Infostealer campaigns (Techzine, Hacker News) | EU AI Act Art. 15 (cybersecurity), IEEE 7002 (credential protection), NIST AI RMF MANAGE 2.3 |
| **Exposed instances** | Censys/Shodan — 24,478 instances | EU AI Act Art. 15, Art. 72 (post-market monitoring), NIST AI RMF GOVERN 2.1 |
| **Internal security ban** | Meta banned OpenClaw internally | EU AI Act Art. 9 (organizational risk assessment), SAAI SFR-Corrigibility |
| **Over-permissioned skills** | Cyera Research — 336 skills request Google Workspace, 127+ demand raw secrets | EU AI Act Art. 14 (human oversight), SAAI SFR-Corrigibility, IEEE 7000 (proportionality) |

---

## 3. EU AI Act — Autonomous Agent Extension

The existing TELOS EU AI Act mapping covers Articles 9, 12, 14, 72 for healthcare. The following extends this to autonomous agents operating 24/7.

### Article 9 — Risk Management System

> *"A risk management system shall be established, implemented, documented and maintained."*

| Requirement | Conversational (Existing) | Autonomous Agent (New) | Implementation |
|-------------|--------------------------|----------------------|----------------|
| **Risk identification** | Per-session PA boundaries | Per-tool-group risk tiers (CRITICAL/HIGH/MEDIUM/LOW) | `ActionClassifier.classify()` maps every tool to a risk tier |
| **Risk estimation** | Fidelity score per turn | Composite fidelity score per action (6 dimensions) | `GovernanceHook.score_action()` |
| **Risk mitigation** | Conversational intervention | EXECUTE/CLARIFY/ESCALATE/BLOCK verdict per action | `GovernanceVerdict` with fail-policy per preset |
| **Residual risk** | Session-bounded | Cumulative — aggregate authority granted over time | **NEW:** Cumulative authority tracking in `GovernanceReceipt` |
| **Supply chain risk** | N/A | Malicious skills, tool poisoning (ClawHavoc: 341 skills) | `ActionClassifier` detects cross-group chains, shadow tools |

### Article 12 — Record-Keeping

> *"High-risk AI systems shall technically allow for the automatic recording of events (logs)."*

| Requirement | Conversational (Existing) | Autonomous Agent (New) | Implementation |
|-------------|--------------------------|----------------------|----------------|
| **Event logging** | JSONL governance traces | Ed25519-signed `GovernanceReceipt` per action | `receipt_signer.py` with 11 new OpenClaw-specific fields |
| **Log duration** | Per session | Continuous — weeks/months of operation | Daemon-level log rotation, `telos agent history` query |
| **Log integrity** | File storage (GAP-001) | Ed25519 signatures on every receipt | `GovernanceReceipt.signature` field (GAP-001 resolved for OpenClaw) |
| **Queryability** | Manual file review | CLI query with filters: `telos agent history --tool-group runtime --verdict ESCALATE` | `cli.py` agent history command |

### Article 14 — Human Oversight

> *"High-risk AI systems shall be designed and developed in such a way... as to be effectively overseen by natural persons."*

| Requirement | Conversational (Existing) | Autonomous Agent (New) | Implementation |
|-------------|--------------------------|----------------------|----------------|
| **Override capability** | Session-level PA modification | Per-action override via ESCALATE verdict + human approval | `GovernancePreset` with escalation policy |
| **Understanding state** | Fidelity display in UI | `telos agent monitor` live TUI + `telos agent status` | Rich-based monitoring dashboard |
| **Intervention** | Redirect/block in conversation | Block tool execution before it runs via `before_tool_call` | TypeScript plugin intercepts all actions |
| **Stopping the system** | End session | `telos service uninstall` or `telos agent block-policy --emergency-stop` | launchd/systemd service management |

### Article 15 — Accuracy, Robustness, and Cybersecurity

| Requirement | Autonomous Agent Implementation | Evidence |
|-------------|-------------------------------|----------|
| **Accuracy** | 4-layer cascade: L0 keyword, L1 cosine, L1.5 SetFit, L2 LLM | Phase I benchmark: 75.5% violation detection (pre-calibration) |
| **Robustness** | Fail-policy per preset (strict+balanced=closed, permissive=open) | `GovernancePreset` configuration |
| **Cybersecurity** | Boundaries sourced from CVE-2026-25253, CVE-2026-25157, ClawHavoc | `validation/openclaw/PROVENANCE.md` |

### Article 72 — Post-Market Monitoring

> *"Providers shall establish and document a post-market monitoring system."*

| Requirement | Autonomous Agent Implementation | Evidence |
|-------------|-------------------------------|----------|
| **Continuous monitoring** | Real-time scoring of every tool call (~30K decisions/month) | `GovernanceHook` in daemon process |
| **Drift detection** | CUSUM charts per tool group (`CUSUMMonitorBank` — auto-creates monitors per group, adaptive 20-observation baseline) + session-level `AgenticDriftTracker` (10/15/20% graduated sanctions) | `cusum_monitor.py`, `daemon.py` drift integration |
| **Incident collection** | Governance receipts with CVE-mapped boundary triggers | `GovernanceReceipt.boundary_triggered` field |
| **Systematic investigation** | Forensic reports via `telos benchmark run -b openclaw --forensic` | `report_generator.py` 9-section reports |
| **Corrective action** | Boundary updates via `telos agent block-policy` | CLI policy management |

### Article 73 — Reporting of Serious Incidents

| Requirement | Autonomous Agent Implementation |
|-------------|-------------------------------|
| **Incident detection** | ESCALATE verdicts with `severity: critical` flag |
| **Incident recording** | Ed25519-signed GovernanceReceipt with full action context |
| **Reporting timeline** | Webhook/email notification sinks for real-time alerting |
| **Root cause analysis** | Forensic report with per-tool-group breakdown + cascade activation trace |

---

## 4. IEEE 7000-2021 — Autonomous Agent Extension

The existing TELOS IEEE 7000 alignment matrix maps core process areas. The following extends for always-on agent governance.

### New Ethical Concerns for Autonomous Agents

| Ethical Concern | IEEE 7000 Process Area | TELOS-OpenClaw Implementation |
|----------------|----------------------|------------------------------|
| **Cumulative authority** | Value Disposition | GovernanceReceipt tracks total authority granted per session; `AgenticDriftTracker` (10/15/20% graduated sanctions) + CUSUM per-tool-group drift detection flags authority accumulation |
| **Cross-channel contamination** | Value-Based Requirements | ActionClassifier detects cross-group chains (e.g., `fs` → `runtime` → `web`); separate risk tier enforcement per channel |
| **Persistence mechanisms** | Ethical Value Register | Boundaries sourced from gateway manipulation incidents; `automation` group at CRITICAL risk tier |
| **Skill supply chain** | Stakeholder Impact | ClawHavoc-sourced boundaries; ActionClassifier detects shadow tool registration patterns |
| **Agent identity protection** | Value Disposition | Boundaries sourced from infostealer campaigns; credential access paths blocked |

### EVR Extension for OpenClaw

The Primacy Attractor serves as the Ethical Value Register for autonomous agents:

```yaml
# EVR extension in templates/openclaw.yaml
purpose:
  statement: >-
    Execute user-directed tasks within the agent's configured scope
    and workspace, using only authorized tools with appropriate
    permissions, without accessing credentials, exfiltrating data,
    executing destructive commands, or exceeding the boundaries of
    the assigned task
```

This PA statement encodes 5 ethical values:
1. **User direction** — Actions serve the principal's intent (goal alignment)
2. **Scope limitation** — Agent operates within configured boundaries (proportionality)
3. **Authorization** — Only approved tools with correct permissions (least privilege)
4. **Data protection** — No credential access or exfiltration (privacy)
5. **Boundary respect** — No destructive or out-of-scope actions (safety)

---

## 5. IEEE 7001-2021 — Transparency Extension

| Transparency Requirement | Conversational (Existing) | Autonomous Agent (New) |
|--------------------------|--------------------------|----------------------|
| **Real-time transparency** | Fidelity display per turn | `telos agent monitor` live TUI with fidelity sparkline, cascade activation, decision stream |
| **Retrospective transparency** | Session-level JSONL traces | Continuous JSONL + Ed25519-signed receipts, queryable via `telos agent history` |
| **Audit trail integrity** | File storage (GAP-001) | Ed25519 signatures resolve GAP-001 for autonomous agent context |
| **Decision explainability** | Semantic interpreter | Per-action governance verdict with 6-dimension score breakdown (purpose, scope, boundary, tool, chain, risk) |

---

## 6. SAAI — Autonomous Agent Extension

The existing TELOS SAAI mapping achieves 88% compliance (39/47 applicable requirements). For autonomous agents, Schaake's M0 analysis identified a reinterpretation to 66% compliance (autonomous agents introduce stricter requirements), rising to 94% post-implementation.

### New SAAI Claims for Autonomous Agents (TELOS-SAAI-009 through 014)

| Claim ID | Category | Requirement | Claim |
|----------|----------|-------------|-------|
| **TELOS-SAAI-009** | continuous_monitoring | always_on_governance | TELOS-OpenClaw scores every tool call in real-time via `before_tool_call` hook, with no gaps in coverage during 24/7 operation |
| **TELOS-SAAI-010** | action_classification | tool_group_risk_tiering | Every OpenClaw tool is classified into one of 11 tool groups across 4 risk tiers (CRITICAL, HIGH, MEDIUM, LOW), with governance depth proportional to risk |
| **TELOS-SAAI-011** | boundary_provenance | sourced_boundaries | All governance boundaries trace to documented CVEs and security incidents with full provenance chain; zero inferred or assumed boundaries |
| **TELOS-SAAI-012** | fail_safety | configurable_fail_policy | Governance presets (strict, balanced, permissive, custom) define fail behavior; balanced and strict presets default to fail-closed on governance process failure |
| **TELOS-SAAI-013** | supply_chain_security | skill_poisoning_detection | ActionClassifier detects malicious skill patterns from ClawHavoc campaign (341 documented malicious skills, 12% of ClawHub marketplace) |
| **TELOS-SAAI-014** | cross_channel_governance | multi_group_chain_detection | Cross-group action chains (e.g., read credentials → execute shell → exfiltrate via web) are detected and escalated as CRITICAL risk regardless of individual tool group tiers |

### GAP-001 Status Update (Autonomous Agents)

| Gap | Original Status | Autonomous Agent Status |
|-----|----------------|------------------------|
| **TELOS-GAP-001** (cryptographic log integrity) | Open — standard file storage | **Partially resolved** — Ed25519-signed GovernanceReceipts for OpenClaw. Full hash chain not yet implemented for conversational traces. |
| **TELOS-GAP-002** (super-normal stimuli controls) | Medium priority | **Elevated to HIGH** for always-on agents — continuous operation without interaction limits requires engagement frequency monitoring (Schaake M0) |

---

## 7. NIST AI RMF / 600-1 — Autonomous Agent Extension

| NIST Function | Autonomous Agent Requirement | TELOS-OpenClaw Implementation |
|---------------|------------------------------|------------------------------|
| **GOVERN 1.2** | Risk management for autonomous operation | 4 governance presets with configurable fail policy; `telos agent block-policy` for runtime modification |
| **GOVERN 2.1** | Continuous risk awareness | Real-time scoring of ~30K decisions/month; CUSUM drift detection per tool group (`CUSUMMonitorBank`, adaptive baseline); structured NDJSON audit trail (`AuditWriter`) |
| **MAP 3.1** | Supply chain risk mapping | ClawHavoc-sourced boundaries; `ActionClassifier` detects tool poisoning patterns from 341 documented malicious skills |
| **MEASURE 2.1** | AI system performance monitoring | `telos benchmark run -b openclaw --forensic`; per-tool-group, per-risk-tier, per-attack-family accuracy breakdowns |
| **MANAGE 2.3** | Incident response | ESCALATE verdicts → webhook/email notifications; `telos agent history` for forensic investigation |

---

## 8. GovernanceReceipt Fields for OpenClaw

Every governance decision produces a `GovernanceReceipt` with the following fields (11 new for autonomous agents, marked with *):

| Field | Type | Description |
|-------|------|-------------|
| `receipt_id` | string | Unique receipt identifier (UUID v4) |
| `timestamp` | ISO 8601 | Decision timestamp |
| `session_id` | string | OpenClaw session identifier |
| `action_type` | string | OpenClaw tool name (e.g., `Bash`, `Read`, `Write`) |
| `*tool_group` | string | TELOS tool group classification (e.g., `runtime`, `fs`, `web`) |
| `*risk_tier` | enum | CRITICAL, HIGH, MEDIUM, LOW |
| `request_text` | string | Natural language description of the action |
| `verdict` | enum | EXECUTE, CLARIFY, ESCALATE, BLOCK |
| `fidelity_score` | float | Composite 6-dimension fidelity score |
| `*purpose_score` | float | Purpose alignment dimension (weight: 0.35) |
| `*scope_score` | float | Scope alignment dimension (weight: 0.20) |
| `*boundary_score` | float | Boundary violation detection (weight: -0.10) |
| `*tool_score` | float | Tool fidelity dimension (weight: 0.20) |
| `*chain_score` | float | Chain continuity / SCI (weight: 0.15) |
| `*cascade_level` | enum | L0, L1, L1.5, L2 — which cascade layer made the decision |
| `*boundary_triggered` | string[] | List of boundary IDs triggered (if any) |
| `*fail_policy` | enum | closed, open — active fail policy at decision time |
| `*governance_preset` | enum | strict, balanced, permissive, custom |
| `signature` | bytes | Ed25519 signature over receipt contents |
| `signing_key_fingerprint` | string | SHA-256 fingerprint of signing key |

---

## 9. OWASP Top 10 for Agentic Applications (2026)

The OWASP Top 10 for Agentic Applications (ASI01-ASI10) is a peer-reviewed framework developed by 100+ industry experts identifying the most critical security risks for autonomous AI systems. Palo Alto Networks published the *OWASP Agentic Top 10 Survival Guide* as a practical operationalization.

**Source:** [OWASP GenAI Security Project](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/) | [Palo Alto Survival Guide](https://www.paloaltonetworks.com/resources/ebooks/owasp-agentic-top-10-survival-guide)

| OWASP Risk | Description | TELOS-OpenClaw Coverage | Implementation |
|------------|-------------|------------------------|----------------|
| **ASI01 — Agent Goal Hijack** | Attackers alter agent objectives via malicious content in emails, documents, or web pages | **Strong** | PA as constitutional anchor; `before_tool_call` scores every action against fixed reference point; boundary corpus includes prompt injection scenarios (Cat E) |
| **ASI02 — Tool Misuse and Exploitation** | Agents use legitimate tools unsafely due to ambiguous prompts or manipulated input | **Strong** | ActionClassifier maps ~40 tools to 11 groups; 4-layer cascade scores every tool call; 4 risk tiers with depth proportional to risk |
| **ASI03 — Identity and Privilege Abuse** | Agents inherit user credentials and high-privilege access that can be escalated | **Strong** | Cross-group chain detection (SAAI-014); cumulative authority tracking (H8); credential access paths blocked per infostealer-sourced boundaries |
| **ASI04 — Agentic Supply Chain** | Compromised tools, plugins, or prompt templates alter agent behavior | **Strong** | ClawHavoc-sourced boundaries (SAAI-013); 341 documented malicious skills; shadow tool registration detection in ActionClassifier |
| **ASI05 — Unexpected Code Execution** | Agents generate/execute code and commands unsafely | **Strong** | `runtime` group at CRITICAL risk tier; CVE-2026-25253 (RCE) and CVE-2026-25157 (command injection) sourced boundaries; shell metacharacter detection |
| **ASI06 — Memory and Context Poisoning** | Attackers poison agent memory systems to influence future decisions | **Partial** | PA is external (AI cannot access or modify); `memory` group governed at LOW tier. Gap: no deep inspection of RAG/embedding poisoning in retrieved context |
| **ASI07 — Insecure Inter-Agent Communication** | Unencrypted/unauthenticated agent-to-agent communication | **Partial** | `nodes` group governed; UDS IPC is local-only (no network exposure). Gap: no cross-network agent authentication protocol for multi-host deployments |
| **ASI08 — Cascading Failures** | Errors in one agent propagate across planning, execution, and downstream systems | **Strong** | Fail-closed default (balanced/strict presets); watchdog with graduated restart (30s heartbeat); detached daemon survives OpenClaw restarts |
| **ASI09 — Human-Agent Trust Exploitation** | Users over-trust agent recommendations, enabling manipulation | **Strong** | ESCALATE verdict forces human review for uncertain decisions; 4 governance presets with configurable escalation thresholds; GovernanceReceipt audit trail |
| **ASI10 — Rogue Agents** | Compromised agents act harmfully while appearing legitimate | **Strong** | Continuous monitoring of every action (~30K decisions/month); Ed25519-signed ALL GovernanceVerdicts (not just escalation receipts); CUSUM drift detection per tool group; structured NDJSON audit trail; `AgenticDriftTracker` graduated sanctions; forensic reporting |

**Summary:** 8/10 strong coverage, 2/10 partial coverage (ASI06, ASI07). The partial gaps are architectural — ASI06 requires deep RAG/embedding poisoning detection (beyond scope of tool-call governance), and ASI07 requires cross-network agent authentication (relevant only for multi-host deployments, which is a post-v2.0 concern).

---

## 10. Regulatory Compliance Summary (All Frameworks)

| Framework | Conversational Compliance | Autonomous Agent Compliance (Pre-OpenClaw) | Autonomous Agent Compliance (Post-OpenClaw) |
|-----------|--------------------------|-------------------------------------------|---------------------------------------------|
| **IEEE 7000-2021** | Strong alignment | Partial — no tool group risk mapping, no persistence monitoring | Strong alignment — EVR extended, 5 new ethical concerns addressed |
| **IEEE 7001-2021** | Strong alignment | Partial — GAP-001 (crypto logs) unresolved for continuous operation | Strong alignment — Ed25519 receipts, live monitoring TUI |
| **IEEE 7002-2022** | Strong alignment | Partial — credential protection not enforced | Strong alignment — infostealer-sourced boundaries, credential path blocking |
| **SAAI** | 88% (39/47) | 66% (autonomous reinterpretation) | 94% (6 new claims: TELOS-SAAI-009 through 014) |
| **EU AI Act** | Art. 9, 12, 14, 72 mapped | Art. 15, 73 unmapped; Art. 72 not implemented for continuous operation | All 6 articles mapped with autonomous agent specifics |
| **NIST AI RMF** | Mapped | GOVERN/MAP/MEASURE/MANAGE partially addressed | All 5 functions implemented with OpenClaw-specific measures |
| **OWASP Agentic Top 10** | N/A (framework postdates conv. mapping) | Not mapped | 8/10 strong, 2/10 partial (ASI06 memory poisoning, ASI07 inter-agent auth) |

---

## 11. Sources

All regulatory mappings are grounded in:

1. **CVE data:** CVE-2026-25253 (Belgium CCB advisory, Adversa.ai), CVE-2026-25157 (GitHub GHSA-q284-4pvr-m585)
2. **Security research:** Moltbook breach (Wiz Research), ClawHavoc campaign (Cisco Skill Scanner), infostealer campaigns (Techzine, The Hacker News), Cyera Research (over-permissioned skills)
3. **Exposure data:** Censys/Shodan (24,478 exposed instances)
4. **Organizational actions:** Meta internal OpenClaw ban
5. **Regulatory frameworks:** EU AI Act (2024/1689), IEEE 7000-2021, IEEE 7001-2021, IEEE 7002-2022, SAAI (Dr. Nell Watson), NIST AI RMF 1.0, NIST 600-1, OWASP Top 10 for Agentic Applications 2026 (genai.owasp.org), Palo Alto Networks OWASP Agentic Top 10 Survival Guide
6. **TELOS internal:** M0 14-agent research phase (Russell, Gebru, Karpathy, Schaake, Nell), M4 benchmark results
7. **Existing TELOS mappings:** `research/ieee_7000_alignment_matrix.md`, `research/saai_requirement_mapping.md`, `research/saai_machine_readable_claims.json`

---

*This document extends (not replaces) the existing TELOS regulatory mappings. For session-based/conversational governance compliance, see the original documents listed in Sources §7.*
