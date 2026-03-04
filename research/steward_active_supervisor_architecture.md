# TELOS Steward — Active Supervisor Architecture (Future Growth)

**Status:** Conceptual — not scheduled for implementation
**Author:** JB
**Date:** 2026-02-15
**Contact:** JB@telos-labs.ai

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Executive Summary

TELOS v2 operates as a passive governance instrument — it measures fidelity, flags violations, and blocks or escalates. It does not participate in the causal chain of decision-making. This document captures the architectural vision for a future evolution where TELOS Steward becomes an active LLM supervisor inside the governed agent's action chain, providing semantic course correction in natural language rather than binary scoring.

This is a growth pathway, not a current priority. Staying nimble and simple is better than complex and robust when complexity invites liability. This document preserves the thinking for when the market, the product, and the legal posture are ready.

---

## The Core Concept

Today's architecture:

```
[Governed Agent LLM] → [Action] → [TELOS scores action] → flag/block/escalate
```

Future Steward architecture:

```
[Governed Agent LLM] ←→ [TELOS Steward LLM] → semantic redirect
                      ↓
                  [Action proceeds or corrects]
```

The supervisor moves from an external observer to a first-class participant in the chain. Steward doesn't just measure drift — it *talks to the governed agent* and semantically redirects it before the action reaches the human or the system.

LLM-to-LLM governance: one model watching another, speaking in natural language, grounded in the customer's own governance corpus.

---

## The Liability Spectrum

This is the critical design constraint. There is a clear line between measurement and direction, and crossing it changes TELOS's legal posture fundamentally.

### Level 0 — Passive Scoring (Current v2)

- Cosine fidelity measurement, binary flags, block/escalate decisions
- No language output to the governed agent
- No causal participation in decisions
- **Liability:** Zero. TELOS is an instrument. We measure, we don't decide.

### Level 1 — Semantic Flagging

- Instead of a raw score, Steward explains *why* something is drifting
- Example: "This action is moving toward boundary: no autonomous ordering of treatments"
- Still measurement — richer signal, natural language explanation, but no direction
- **Liability:** Low. We're describing what we observe, not prescribing what to do.

### Level 2 — Corpus Reflection

- Steward echoes the customer's own rules back at the governed agent
- Example: "Your configuration states: no clinical diagnosis or treatment advice. This response contains a treatment recommendation."
- The corpus is theirs. The words are theirs. We're a mirror, not a guide.
- Steward's LLM context is loaded exclusively with the customer's YAML config — boundaries, exemplars, tool definitions. It can only speak from their corpus.
- **Liability:** Low-to-moderate. We're reflecting their governance posture in natural language. The interpretation is mechanical, not judgmental.

### Level 3 — Constrained Redirection

- Steward suggests alternatives from the customer's pre-approved action space
- Example: "Instead of generating a diagnosis, your config defines these safe actions: schedule follow-up, route to specialist, document finding for clinician review."
- We select from their menu. We don't write our own.
- All suggested alternatives trace to the customer's tool definitions and safe exemplars.
- **Liability:** Moderate. We're choosing among their options, which implies judgment about which option fits. If we redirect to the wrong one, we participated in the decision chain.

### Level 4 — Full Semantic Direction

- Steward generates novel guidance: "You should do Y because of guideline Z"
- We become a decision participant, injecting domain knowledge into the action chain
- **Liability:** High. We are in the causal chain of decision-making. If Y is wrong, we own it.
- **Position:** We do not want to be here. It is not our domain. We are not the experts. We govern the governed — we do not govern the domain.

---

## The Corpus Defense

The key architectural insight for Levels 1-3: if Steward's LLM context window is loaded exclusively with the customer's own governance corpus, then Steward is not injecting TELOS judgment into their domain. It is reflecting their own governance posture back at their own agent in natural language.

This is meaningfully different from "TELOS told our agent what to do." The distinction:

- **TELOS directs:** "Prescribe amoxicillin instead of azithromycin" → we are in the clinical decision chain, we carry liability
- **Corpus reflection:** "Your boundary states: no off-label recommendations without explicit off-label designation. This recommendation lacks that designation." → we are reflecting their own rule, they carry the domain liability

The corpus-only constraint is not just a technical choice — it is a liability firewall. Steward never speaks from knowledge TELOS brings to the table. It speaks from the customer's YAML, their boundaries, their exemplars, their tool definitions. We are their governance voice, not our own.

### Implementation Constraint

```yaml
steward:
  corpus_only: true  # Steward LLM context limited to this config's corpus
```

When `corpus_only: true`, the Steward LLM's system prompt and context window contain only:
- The customer's YAML config (purpose, boundaries, tools, exemplars, constraints)
- The current action chain state (what the governed agent has done so far)
- The current request being evaluated

No external knowledge bases. No TELOS-supplied guidelines. No domain expertise beyond what the customer encoded in their config. If the customer's config doesn't cover a scenario, Steward escalates — it does not improvise.

---

## Where Steward Lives in the Stack

The natural home for Steward is the therapeutic knowledge base domain (`healthcare_therapeutic.yaml`). This is where semantic course correction has the highest value:

- Diagnostic AI: binary detection — the image has a finding or it doesn't. Passive scoring is sufficient.
- Coding AI: rule-based compliance — the code matches documentation or it doesn't. Passive scoring is sufficient.
- Therapeutic AI: pathway selection — the right treatment depends on patient context, guidelines, contraindications, and clinical judgment. This is where semantic direction adds value over binary flags.

However, the first testbed could be property intelligence (Nearmap) precisely because the stakes are lower. Prove the architecture works in a domain where a wrong redirect costs a bad property report, not a bad clinical outcome.

---

## Potential Config Schema (Future)

```yaml
steward:
  mode: reflect           # observe | reflect | constrain | direct
  corpus_only: true       # LLM context limited to this config's corpus
  intervention_threshold: 0.45  # only activate Steward below this fidelity score
  max_redirects_per_chain: 2    # limit redirect attempts before escalating to human
  log_all_interventions: true   # every Steward utterance is logged for audit
  liability_acceptance: false   # customer must explicitly accept liability for 'constrain' or 'direct' modes
```

### Mode Descriptions

| Mode | Steward Behavior | TELOS Liability | Customer Action Required |
|------|-----------------|-----------------|------------------------|
| `observe` | Passive scoring only (v2 behavior) | None | None |
| `reflect` | Explains why drift was detected, cites customer's own boundaries | Low | None |
| `constrain` | Suggests alternatives from customer's pre-approved action space | Moderate | `liability_acceptance: true` |
| `direct` | Generates novel semantic guidance | High | Not recommended — may never ship |

---

## Why Not Now

1. **Simplicity is a feature.** TELOS v2's value proposition is clean: we measure, we score, we flag. Customers understand it immediately. Adding an active LLM supervisor introduces complexity that requires explanation, trust-building, and legal review.

2. **Liability posture.** As a measurement instrument, TELOS carries zero liability for the governed agent's decisions. The moment Steward actively participates in the decision chain — even at Level 2 (corpus reflection) — there is a legal argument that TELOS influenced the outcome. We need legal counsel before crossing that line.

3. **Market timing.** The current healthcare AI market is still figuring out basic governance — confidence thresholds, drift detection, audit trails. They're not ready for LLM-to-LLM supervisory architecture. Sell what they need today (passive governance), build trust, then introduce Steward when they ask "can TELOS do more?"

4. **Technical maturity.** Steward requires TELOS to operate its own LLM (or a tightly controlled third-party LLM). That introduces latency in the action chain, cost per intervention, and a new failure mode (what if Steward hallucinates?). The governance of the governor is a hard problem.

5. **Nimble beats robust.** A simple system that ships and works is better than a complex system that's theoretically superior but introduces edge cases, liability questions, and customer confusion. Ship Level 0, prove value, earn the right to Level 1.

---

## Growth Pathway

### Phase 1: Prove Passive Value (Now — v2)

- Ship 6-dimension fidelity scoring across healthcare AI domains
- Demonstrate governance gaps in deployed systems (ESM, ambient scribes, coding AI)
- Build customer trust through measurement accuracy and zero false positives
- Establish TELOS as the governance layer hospitals adopt

### Phase 2: Semantic Flagging (v3 Candidate)

- Add natural language explanations to governance flags
- "This action scored 0.38 fidelity because it approaches boundary: [boundary text]"
- No course correction — just richer signal for the human operator
- Test with Nearmap first (low stakes), then healthcare
- Legal review of liability implications before healthcare deployment

### Phase 3: Corpus Reflection (v4 Candidate)

- Steward LLM loaded with customer's YAML corpus
- Reflects customer's own governance rules back at the governed agent in natural language
- Corpus-only constraint enforced at the architecture level
- Requires customer opt-in and liability framework
- Requires TELOS to operate or contract an LLM for Steward

### Phase 4: Constrained Redirection (v5+ Candidate)

- Steward suggests alternatives from the customer's pre-approved action space
- Requires explicit customer liability acceptance
- Requires extensive testing for redirect accuracy
- May require domain-specific Steward fine-tuning on the customer's corpus
- Highest value in therapeutic knowledge base domain

### Phase 5: Evaluate Full Direction (Research Only)

- Assess whether Level 4 (full semantic direction) is ever appropriate
- Likely answer: only in domains where TELOS has genuine expertise (governance methodology itself, not clinical or domain expertise)
- May manifest as Steward advising on *governance configuration* rather than *domain decisions*
- Example: "Your boundary set doesn't cover this scenario — consider adding a boundary for [gap]"
- This keeps Steward in the governance domain (ours) rather than the clinical domain (theirs)

---

## Key Principle

**We govern the governed. We do not govern the domain.**

Steward speaks the customer's language back to the customer's agent. The domain expertise stays on their side of the line. TELOS's expertise is governance methodology — fidelity measurement, drift detection, boundary enforcement. The moment we claim domain expertise (clinical, legal, financial, property), we accept domain liability. That is not our business.

The only domain where TELOS can legitimately direct is governance itself — advising customers on how to configure their governance posture, not on what clinical or business decisions to make.

---

## Open Questions for Future Exploration

1. **Steward's LLM:** Do we train/fine-tune our own, or use a controlled third-party (GPT-4, Claude, Mistral) with strict system prompts? Own model = more control, higher cost. Third-party = faster to market, dependency risk.

2. **Latency budget:** Steward adds an LLM inference step to the action chain. What latency is acceptable? For ambient scribes (seconds matter), probably too slow. For coding AI (batch processing), easily tolerable.

3. **Governance of the governor:** If Steward hallucinates a boundary that doesn't exist in the customer's config, we've introduced a governance failure. How do we validate Steward's own fidelity? TELOS scoring TELOS — recursive governance.

4. **Pricing model:** Steward as active supervisor is significantly more expensive than passive scoring (LLM inference per intervention). Is this a premium tier? Per-intervention pricing? Flat rate with caps?

5. **Regulatory posture:** Does an active AI supervisor in the clinical decision chain trigger SaMD classification for TELOS itself? If Steward redirects a therapeutic AI's recommendation, is TELOS now medical device software? This needs regulatory counsel before any healthcare deployment.

6. **Customer trust:** Customers adopting TELOS for passive governance may resist giving an external LLM active participation in their decision chain. The trust ladder is: measure → explain → suggest → direct. Each step requires earned trust from the previous step.

---

*This document captures architectural thinking for future growth. No implementation is planned or scheduled. The current priority is shipping v2 passive governance, proving value through measurement accuracy, and building customer trust that earns the right to evolve.*
