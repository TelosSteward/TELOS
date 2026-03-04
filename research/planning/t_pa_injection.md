## PA System Prompt Injection — Measurement Infrastructure (A10 Phase 1)

Read `research/A10_PA_SEMANTIC_INJECTION_REVIEW.md` for the full 6-agent review and design rationale. All 6 advisors recommend building this. This task covers Phase 1 (measurement setup) and Phase 2 (implementation).

### Context

Dual-layer alignment architecture: pre-generation semantic injection (PA in system prompt every turn) + post-generation mathematical scoring (TELOS fidelity measurement). The semantic layer reduces how often the math needs to intervene. The math catches what the semantics miss.

Critical reframe from Karpathy: this is NOT RAG retrieval. The PA is a known, fixed document. Direct system prompt injection with hash verification. Precompute at session start since it only changes via Ed25519 ceremony.

### TASK 1: Governance Event Schema Update

Add `pa_injected: bool` field to the governance event schema. Every scored event must record whether PA was in the model's context when it generated the response. This is the covariate that makes all subsequent measurement valid.

### TASK 2: PAContext Class

Create a `PAContext` class (or add to existing module — wherever it fits best in the engine architecture):

- `get_injection_block(pa_path, compressed=False) -> str` — Returns formatted PA text for system prompt injection
- `get_injection_hash(pa_path) -> str` — Returns SHA-256 hash of the injection block
- `verify_injection_integrity(injection_block, signed_pa_hash) -> bool` — Verifies the injection block matches the signed PA
- Compressed mode (~2k tokens) for Stewart/MLX (32k context window). Full mode for OpenClaw/Claude (200k window).
- Precompute and cache at session start. Invalidate only on PA change (Ed25519 ceremony).

The hash chain matters: signed PA → injection block → hash verification every turn. This creates the PA Injection Integrity Record that Schaake identified as a novel compliance artifact.

### TASK 3: Alternating-Treatment Toggle

Add a session-level boolean toggle for PA injection. This enables Nell's alternating-treatment experimental design:

- `pa_injection_enabled: bool` in session/scoring config
- When enabled, PA injection block is included in the system prompt
- When disabled, PA injection block is omitted
- The `pa_injected` field in the governance event records which condition each turn was under
- For the alternating-treatment experiment: even turns injected, odd turns not (or configurable)

This must be trivially reversible within a session with zero hysteresis — the model has no memory between turns, so toggling is clean.

### TASK 4: System Invariant Documentation

Add to the TELOS system invariants (wherever those are documented — CLAUDE.md, design docs, or a new invariants file):

**"Never fine-tune on PA-injected outputs."** If PA-injected outputs become training data (fine-tuning, RLHF, DPO), surface features of compliance get baked into weights. The agent would appear aligned without governance infrastructure. This is Bengio's finding and is as critical as the Ed25519 signing requirement.

### TASK 5: Parrot-vs-Understanding Experiment Design

Using the backtest data (`research/backtest_forensic_dataset.jsonl` — 2,071 events, 11 scorable), design the experiment:

- Score the 11 scorable events WITH PA injection (PA in the scoring context)
- Compare to existing scores WITHOUT PA injection (already in backtest results)
- If scores increase uniformly across all event types (including the boundary violations), the model is parroting — PA injection inflates scores without improving alignment
- If scores increase for in-scope events but NOT for boundary violations, PA injection provides genuine constraint guidance
- Document the experiment design in `research/pa_injection_experiment.md`

This tells us whether threshold recalibration is needed and how much measurement inflation PA injection produces.

### Priority

1. Task 1 (schema) — foundation for everything else
2. Task 2 (PAContext) — the implementation
3. Task 3 (toggle) — measurement capability
4. Task 4 (invariant) — one paragraph, do it now
5. Task 5 (experiment design) — document, don't run yet

Commit each separately, tests must pass after each.
