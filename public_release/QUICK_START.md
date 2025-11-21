# Runtime Governance - Quick Start

Get up and running in 5 minutes.

## 1. Install Dependencies (30 seconds)

```bash
pip install sentence-transformers numpy
```

## 2. Define Your PA (2 minutes)

Edit `.claude_project.md` and add:

```markdown
## 🔭 RUNTIME GOVERNANCE - ACTIVE

**PA Baseline:**
[Write 2-4 sentences describing your project's purpose, scope, and boundaries]

Example:
"Build a REST API for user authentication by end of Q1. Features include
OAuth2, JWT tokens, rate limiting, and PostgreSQL backend. Maintain test
coverage >80% and follow OWASP security best practices."
```

## 3. Start Your Session (10 seconds)

```bash
python3 telos_session_start.py
```

You'll see:
```
🔭 TELOS Runtime Governance - Session Initialization
============================================================

📋 Extracting PA baseline from .claude_project.md...
✅ PA extracted (234 chars)

✅ TELOS Runtime Governance INITIALIZED
```

## 4. Work Normally

Just use Claude Code as usual. After each turn, Claude should run:

```bash
python3 telos_turn_checkpoint.py --user "your message" --assistant "Claude's response"
```

This happens automatically if Claude is configured properly.

## 5. Check Your Progress

At any time:
```bash
# View session stats
cat .telos_active_session.json

# Export full session
python3 telos_session_export.py
```

## That's It!

Your Claude Code sessions are now tracked and measured automatically.

---

## Cost Estimate

**100 conversation turns:**
- Local embeddings: $0.00 (free)
- Mistral embeddings: $0.002 (~2/10 of a cent)
- OpenAI embeddings: $0.002 (~2/10 of a cent)

**Storage:** Free (local JSON + Memory MCP)

**Total cost for a full day of Claude Code usage:** Less than a penny.

---

## Example Output

After Turn 5:
```
============================================================
📊 Turn 5: F=0.834 ✅
============================================================

Status: ON TRACK
Fidelity: 0.834 (threshold: 0.80)

Session Stats:
  Total turns: 5
  Mean fidelity: 0.812
  Turns on track: 4/5 (80%)
```

---

## Troubleshooting

**"No PA baseline found"**
→ Add PA section to `.claude_project.md`

**"Embedding provider failed"**
→ Install: `pip install sentence-transformers`
→ Or set Mistral API key: `export MISTRAL_API_KEY="..."`

**"Memory MCP not found"**
→ Ensure Memory MCP is enabled in Claude Code settings

---

## Next Steps

- Read full docs: `RUNTIME_GOVERNANCE_README.md`
- Configure: Copy `governance_config.example.json` to `governance_config.json`
- Export data: `python3 telos_session_export.py --format dashboard`
- View session history in Memory MCP

**You now have a checksum for Claude Code alignment.** 🎯
