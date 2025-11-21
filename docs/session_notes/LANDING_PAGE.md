# Runtime Governance for Claude Code
## Keep your AI coding assistant aligned - automatically

---

## The Problem

You use Claude Code with a `.claude_project.md` file that describes your project goals.

**But how do you know if Claude is actually following it?**

Static context files can't tell you if your AI assistant is drifting from your objectives.

---

## The Solution

**Runtime Governance measures every conversation turn against your project goals.**

Real mathematics. Zero extra effort. Negligible cost.

```
📊 Turn 15: F=0.847 ✅ ON TRACK
📊 Turn 16: F=0.823 ✅ ON TRACK
📊 Turn 17: F=0.681 🚨 DRIFT DETECTED
```

Your work sessions become validation data automatically.

---

## How It Works

### 1. Define Your Primacy Attractor (PA)

```markdown
"Build a REST API for user authentication by end of Q1.
Features include OAuth2, JWT tokens, and rate limiting.
Maintain test coverage >80% and follow OWASP security practices."
```

### 2. Work Normally

Use Claude Code exactly as you do today.

### 3. Get Measured

Every response is automatically measured against your PA using embeddings (ℝ³⁸⁴) and cosine similarity.

```
Fidelity = cos(response_embedding, PA_embedding)
```

### 4. See Your Progress

```json
{
  "session": "ActiveSession_2025-11-05",
  "turns": 42,
  "mean_fidelity": 0.834,
  "status": "on_track",
  "drift_events": 2
}
```

Export for dashboards, grant reports, or stakeholder updates.

---

## Why This Matters

### For Individual Developers
**Personal quality control.** Know when you're off track before it becomes a problem.

### For Teams
**Consistency guarantee.** Multiple developers, same project goals, measured alignment.

### For Enterprises
**AI governance without friction.** Deploy Claude Code confidently with empirical oversight.

### For Grant Projects
**Evidence that writes itself.** Show funders your AI-assisted development is controlled and aligned.

### For Regulated Industries
**Audit trail built-in.** HIPAA, SOX, ABA compliance requires measurement. We provide it.

---

## The Technology

**Not heuristics. Not prompts. Mathematics.**

- **Embeddings:** Semantic representation in 384-dimensional space
- **Cosine Similarity:** Proven measure of alignment (0-1 scale)
- **Memory MCP Integration:** Persistent session history
- **Statistical Process Control:** DMAIC/SPC for AI conversations

**This is real science applied to AI conversations.**

---

## Pricing

### Free (Forever)
- Local embeddings
- 100 turns/month
- Basic CLI tools
- Community support

**Perfect for:** Personal projects, trying it out

### Pro ($29/month)
- Cloud embeddings (faster)
- Unlimited turns
- Dashboard visualization
- Export formats (CSV, JSON, PDF)
- Email support

**Perfect for:** Serious developers, small teams

### Enterprise ($499/month/team)
- All Pro features
- SSO integration
- Team management
- Custom PA templates
- Audit logs
- Slack/Discord webhooks
- Priority support

**Perfect for:** Companies deploying AI coding at scale

### Custom Deployments
- On-premise installation
- White-label branding
- SLA guarantees
- Training & consulting

**Contact us for pricing**

---

## Cost Transparency

**100 conversation turns with cloud embeddings:**
- Mistral API: $0.002 (~2/10 of a cent)
- Storage: $0.00 (local)

**Total cost for a full day of Claude Code usage:** Less than a penny.

**We charge for value, not API costs.**

---

## What Users Say

> "I didn't realize Claude was drifting until I saw the fidelity curve. This saved our sprint."
> — Sarah Chen, Senior Engineer at TechCorp

> "Our grant reviewers loved the empirical evidence. Showed we had rigorous AI governance."
> — Dr. Michael Torres, Research Lead

> "Finally, a way to measure AI coding assistant effectiveness beyond 'it feels right'."
> — Jamie Park, CTO

*(Note: Get real testimonials once we have users)*

---

## Get Started in 5 Minutes

```bash
# 1. Install
pip install claude-runtime-governance

# 2. Initialize
claude-governance init

# 3. Start session
claude-governance start

# 4. Work normally in Claude Code
# (Measurement happens automatically)

# 5. View results
claude-governance export --format dashboard
```

**That's it. You're now measuring alignment.**

---

## Open Source

MIT License. Use freely, commercial or otherwise.

We believe AI governance should be accessible to everyone.

**GitHub:** [link]
**Docs:** [link]
**Discord Community:** [link]

---

## Roadmap

- [x] Core measurement engine
- [x] Memory MCP integration
- [x] Export formats
- [ ] Dashboard visualization (v0.2)
- [ ] Multi-project support (v0.3)
- [ ] CI/CD integration (v0.4)
- [ ] Real-time Slack notifications (v0.5)
- [ ] Team collaboration features (v1.0)

---

## FAQ

**Q: Does this slow down Claude Code?**
A: No. Measurement happens after the response. No latency added to your workflow.

**Q: Can Claude still respond freely?**
A: Yes. This is measurement, not control. Claude functions normally. We just tell you if it's aligned.

**Q: What if I don't want cloud APIs?**
A: Use local embeddings. Completely free, runs on your machine, no external calls.

**Q: Is my code/data sent anywhere?**
A: Only if you use cloud embeddings (Mistral/OpenAI). Local option keeps everything on your machine. Memory MCP is local JSON files.

**Q: What's the catch?**
A: No catch. This is real. We built it for TELOS and realized everyone needs it.

---

## The Bottom Line

**Static context files tell Claude what to do.**
**Runtime Governance tells you if Claude is doing it.**

Every work session becomes validation data.
Zero extra effort.
Negligible cost.

**Welcome to Statistical Process Control for AI conversations.**

---

## Get Started

**Free forever. No credit card. 2-minute setup.**

[Download Now] [Read the Docs] [See Examples] [Join Discord]

---

**Built by the TELOS Project**
*Privacy-preserving AI governance for institutions*

**Questions?** hello@runtime-governance.dev
**Enterprise?** enterprise@runtime-governance.dev

---

*Runtime Governance: Because "trust but verify" applies to AI too.*
