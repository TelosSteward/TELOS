# Open Philanthropy Technical AI Safety RFP - Expression of Interest (Alternative: Jailbreak Focus)

**Research Areas:** Jailbreaks and unintentional misalignment (*), Alternatives to adversarial training (*), Activation monitoring

**Project Title:** Detecting Competent Goal-Directed Violations with Geometric Governance

## Proposal Summary (300 words)

Your RFP prioritizes finding inputs that trigger "competent, goal-directed violations" — behaviors where LLMs pursue goals developers tried to prevent. TELOS detects these violations by monitoring geometric drift in embedding space, catching goal-directed behavior even when individual outputs appear benign.

**The Competent Violations Problem:** Existing jailbreak detection focuses on refusals or single-turn violations. But competent violations are multi-step: an agent pursues an undesired goal across variations in situation, flexibly optimizing its actions. Current methods miss this because they evaluate outputs individually, not trajectories.

**TELOS Solution:** We define a "primacy attractor" encoding the developer's intended purpose. As an LLM agent acts over multiple timesteps, TELOS measures whether its semantic trajectory remains within the basin of attraction. Systematic drift toward alternative goals — even when each individual action seems compliant — triggers detection.

**Key Advantages:**

1. **Trajectory-level detection:** Identifies goal-pursuit patterns across multiple timesteps, not just single-turn violations
2. **Unintended misbehavior focus:** Catches accidental misalignment, not just adversarial jailbreaks
3. **Unambiguous specification:** Works best when developers have clearly specified rules (exactly your "nice-to-have" criterion)
4. **Mathematical rigor:** Quantifiable fidelity scores with provable bounds

**Research Plan:**

Phase 1: Build evaluation dataset of competent violations using agent scenarios from AgentHarm and control evaluation literature
Phase 2: Measure TELOS detection rates vs. baseline monitoring (prompt-based, output filtering)
Phase 3: Test against jailbreak techniques (Andriushchenko et al., Kumar et al., @elder_plinius methods)
Phase 4: Investigate whether TELOS generalizes to detect previously unseen violation patterns

**Deliverables:** Open-source benchmark for competent violations, empirical comparison of detection methods, theoretical analysis of when geometric monitoring succeeds/fails.

**Requested funding:** $150K-400K, 18 months. Compute for agent scenarios, API credits, red-team contractor budget.
