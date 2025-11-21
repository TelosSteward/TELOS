# TELOS Repository Classification

## Purpose: Security & IP Protection

This document defines what belongs in PUBLIC vs PRIVATE repositories.

---

## 🌐 PUBLIC REPO: `telos_purpose`

### Purpose
- Research community engagement
- Developer adoption
- Academic credibility
- Open-source goodwill

### Contains
✅ **Generic algorithms:**
- Basic fidelity measurement (cosine similarity)
- Generic embedding generation
- Runtime governance scripts (already sanitized in `public_release/`)

✅ **Sanitized documentation:**
- How to use Runtime Governance
- Research methodology (high-level)
- Installation guides
- API documentation

✅ **Research artifacts:**
- Published papers (after publication)
- Aggregate validation results (no proprietary metrics)
- Case studies (sanitized)

### Does NOT Contain
❌ **Proprietary innovations:**
- Dual Attractor implementation details
- Lock-on derivation formulas
- SPC/DMAIC for AI framework specifics
- Adaptive weighting algorithms
- Progressive PA extraction logic

❌ **Business/platform code:**
- Observatory platform
- Telemetric Keys
- OriginMind
- Institutional deployment configs

❌ **Proprietary metrics:**
- "+85.32%" improvement numbers
- "45+ studies" specifics
- Performance benchmarks vs competitors

❌ **Partnership details:**
- GMU deployment specifics
- Grant applications
- Institutional agreements
- Pricing/contracts

---

## 🔒 PRIVATE REPO: `telos_observatory`

### Purpose
- Full system development
- Institutional deployment
- IP protection
- Partner/contributor collaboration

### Contains
✅ **Complete platform:**
- Observatory V3 (Streamlit)
- All UI components
- Beta consent system
- Steward PM (full version)

✅ **Proprietary innovations:**
- Full Dual PA implementation
- Lock-on derivation code
- SPC/DMAIC integration
- Telemetric Keys (when built)
- OriginMind (when built)
- Progressive PA Extractor

✅ **Business/deployment:**
- Institutional deployment configs
- Partner-specific customizations
- Grant applications
- Strategy documents
- Partnership agreements

✅ **Research data:**
- Detailed validation studies
- Proprietary performance metrics
- Internal analysis
- Competitive research

✅ **Infrastructure:**
- Steward sanitization scripts
- Deployment automation
- Monitoring/logging
- Admin dashboards

### Access Control
- **Owner:** You
- **Contributors:** Selected developers (case-by-case)
- **Read access:** Institutional partners (GMU, etc.) via separate deploy keys
- **No access:** General public

---

## 📋 Sanitization Rules

### Before ANY commit to PUBLIC repo:

**1. Run Steward Sanitization Check:**
```bash
python3 steward_sanitization_check.py path/to/files
```

**2. Check for HIGH-severity terms:**
- Dual Attractor / Dual PA / AI PA
- Lock-on derivation / Lock-on formula
- DMAIC for AI / SPC for AI
- Telemetric Keys / OriginMind
- Progressive PA Extractor
- "+85.32%" / "+85%" / "45+ studies"
- Specific performance numbers

**3. Check for MEDIUM-severity terms:**
- DMAIC cycle (generic OK, "for AI" NOT OK)
- Process capability analysis / Cpk for governance
- Progressive PA (context-dependent)
- Adaptive weighting / Proportional intervention
- GMU partnership details
- Grant specifics (LTFF, EV amounts)

**4. Manual review:**
- Read the diff carefully
- Ask: "Does this reveal proprietary methodology?"
- When in doubt, keep it private

### Sanitization Examples

**❌ BLOCKED (proprietary):**
```python
# Uses lock-on derivation to compute AI PA from User PA
ai_pa = compute_lock_on(user_pa, alpha=0.3)
dual_fidelity = (0.6 * f_user) + (0.4 * f_ai)
```

**✅ ALLOWED (generic):**
```python
# Calculates alignment between response and baseline
fidelity = cosine_similarity(response_emb, baseline_emb)
```

---

**❌ BLOCKED (specific metrics):**
```markdown
TELOS achieves +85.32% improvement over baseline across 45 validation studies.
```

**✅ ALLOWED (generalized):**
```markdown
Empirical validation shows significant improvement over baseline approaches.
```

---

**❌ BLOCKED (implementation details):**
```markdown
The Dual PA architecture uses adaptive weighting with lock-on derivation...
```

**✅ ALLOWED (high-level concept):**
```markdown
TELOS uses multi-objective governance to balance competing constraints.
```

---

## 🚀 Deployment Paths

### PUBLIC Repo → GitHub Pages / PyPI / npm
- Runtime Governance package
- Documentation site
- Research landing page
- **Goal:** Developer adoption, citations

### PRIVATE Repo → Streamlit Cloud / AWS / Institutional VMs
- Observatory V3 (beta.telos.app)
- GMU deployment (gmu.telos.app)
- Partner instances (partner-specific subdomains)
- **Goal:** Institutional pilots, paying customers

---

## 🔐 Security Checklist

Before pushing to PUBLIC repo:

- [ ] Ran `steward_sanitization_check.py` (no HIGH findings)
- [ ] No API keys, credentials, secrets
- [ ] No user data, PII, session logs
- [ ] No proprietary term violations
- [ ] No specific performance numbers
- [ ] No partnership/grant details
- [ ] No Observatory platform code
- [ ] Documentation is generic, not implementation-specific
- [ ] License is clear (MIT for public)

Before pushing to PRIVATE repo:

- [ ] No plain-text secrets (use environment variables)
- [ ] No user PII in beta_consents/ (gitignored)
- [ ] No session data (gitignored)
- [ ] `.env` files excluded
- [ ] Streamlit secrets in dashboard (not repo)

---

## 📊 Current State

### Already Sanitized (Ready for PUBLIC):
- ✅ `public_release/` directory → becomes `telos_purpose` repo
- ✅ Runtime Governance scripts
- ✅ Generic embedding provider
- ✅ Documentation (QUICK_START, README, etc.)

### Needs to Stay PRIVATE:
- ✅ `telos_observatory_v3/` (full platform)
- ✅ `telos_purpose/core/` (full implementations)
- ✅ `.claude_project.md` (strategy, grants)
- ✅ Steward PM
- ✅ All session/validation data

---

## 🎯 Summary

**PUBLIC = Community + Adoption**
**PRIVATE = Platform + Revenue**

Both are necessary. Both are valuable.

**Sanitization protects IP while building community.**

---

*This classification is enforced by Steward PM sanitization checks.*
*Review quarterly as IP strategy evolves.*
