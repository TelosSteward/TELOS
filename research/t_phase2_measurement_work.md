## T Work Block: Measurement Infrastructure + A8 Gaps

S is working on the remaining StewartBot hard blockers (data retention, watchdog). You have 4 parallel tasks — all engine-side, all unblocking the A8 research team findings.

Read `research/openclaw_calibration_log.md` if you haven't recently — it's the artifact being reviewed.

---

### TASK 1: Measurement Codebook (6 Dimensions x 5 Verdicts)

This is the single most-requested item from the A8 review. Nell and Gebru both said it's BLOCKING for baseline data collection.

Create `research/measurement_codebook.md` with:

**For each of the 6 scoring dimensions** (purpose, scope, boundary, tool, chain, composite):

1. **Operational definition** — What exactly does this dimension measure? What observable behavior maps to what score range? Be precise enough that two independent raters (human or machine) would produce the same score.

2. **Scoring method** — How is this dimension computed? Which code path, which model, which similarity metric?

3. **Score interpretation guide** — What does 0.2 mean vs 0.5 vs 0.8 for this dimension? Concrete examples for each range.

4. **Known limitations** — Where does this dimension fail? (e.g., purpose fidelity conflates intent with arguments per Gebru's finding)

**For each of the 5 verdicts** (EXECUTE, CLARIFY, SUGGEST, INERT, ESCALATE):

1. **Threshold** — What composite score triggers this verdict?
2. **Behavioral definition** — What does the system DO when this verdict fires?
3. **Expected frequency** — In normal operation, how often should each verdict appear?
4. **Anchor examples** — 3-5 concrete tool calls that should produce this verdict, with reasoning

**Cross-reference table:** 6 dimensions x 5 verdicts — for each cell, describe how that dimension's score contributes to that verdict outcome.

This codebook is what makes the SCED study possible. Without it, "accuracy" has no definition.

---

### TASK 2: Weight Provenance Documentation

The composite weights are: purpose=0.35, scope=0.20, boundary=0.20, tool=0.15, chain=-0.10.

The A8 review (Russell, Schaake) flagged that these weights have no documented derivation. Add a section to the calibration log OR create `research/weight_provenance.md` documenting:

1. **Origin** — Where did these specific values come from? Were they theoretically derived, empirically tuned, or chosen by convention?
2. **Rationale** — Why is purpose weighted highest? Why is chain a penalty rather than a positive dimension?
3. **Sensitivity analysis** — How much do verdict outcomes change if you perturb each weight by +/- 0.05? (You can reference existing optimizer benchmark data if it covers this.)
4. **Change control** — Who can modify these weights? Through what process? (Link to The Ratchet / PA framework.)
5. **Approval chain** — Jeffrey approved these values (or they were derived from a process Jeffrey approved). Document this.

This is a regulatory requirement (Schaake: EU AI Act Art 9 — documented risk management system requires provenance for all scoring parameters).

---

### TASK 3: Held-Out Evaluation Split

Karpathy: AUC 0.9905 on N=171 with no held-out split is "not trustworthy." Gebru: isotonic ECE=0.0 is likely overfitting.

Do one of:

**Option A (preferred):** Split existing 171 examples into train (80%) and held-out eval (20%). Re-run SetFit training on train only. Report AUC on held-out set. Document the split methodology.

**Option B:** Keep current 171 as training. Create 30-50 NEW examples as a held-out eval set. These must be written independently (not derived from existing examples). Score them. Report results.

Either way, document in the calibration log:
- Split methodology
- Held-out AUC, precision, recall, F1
- Whether isotonic calibration still shows ECE=0.0 on held-out (if yes, it's overfitting)

---

### TASK 4: Cascade Failure Mode Specification

The 4-layer scoring cascade (L0 keyword → L1 cosine → L1.5 SetFit → L2 LLM judge) has no failure mode documentation (Karpathy finding).

Create a section in the calibration log or a standalone `research/cascade_failure_modes.md` documenting for each layer:

1. **What triggers this layer?** (entry conditions)
2. **What happens if this layer crashes?** (does it fall through? fail closed? return a default?)
3. **What happens if this layer is slow?** (timeout behavior, latency budget)
4. **What is the expected latency?** (p50, p95, p99 if you have telemetry; estimates if not)
5. **Dependencies** — What external resources does each layer need? (models, embeddings, ONNX runtime, etc.)

Also address Bengio's safety concern: the max-pool gate at 0.50 creates a discontinuity. Document the gate behavior and acknowledge the adversarial exploitation risk (inputs tuned to land on either side of the gate).

---

### Priority Order

1. Task 1 (measurement codebook) — this is the most blocking item
2. Task 2 (weight provenance) — can be quick if you know the history
3. Task 3 (held-out eval) — requires compute but methodology is straightforward
4. Task 4 (cascade failure modes) — documentation task
5. Commit each separately
6. Run tests after each to confirm 1553/1553 pass
