# Opus Verification Checklist
**Purpose:** Double-check Sonnet's pre-deployment work
**Sonnet Report:** See `SONNET_WORK_COMPLETION_REPORT.md`
**Full Audit Spec:** See `OPUS_AUDIT_BRIEF.md`
**Date:** November 5, 2025

---

## Context

Sonnet (Claude Sonnet 4.5) completed pre-deployment preparation work. This checklist guides Opus through verification of that work to ensure nothing was missed.

**Sonnet's Claims:**
- ✅ Sanitized `public_release/` (removed dual_attractor.py, SPC/DMAIC)
- ✅ Replaced 4 hardcoded API keys with environment variables
- ✅ Removed 4 debug print statements
- ✅ Created requirements.txt for Observatory V3
- ✅ Verified gitignore coverage
- ✅ Public release passed sanitization check (0 violations)

**Opus's Task:** Verify these claims and look for anything Sonnet missed.

---

## Verification Tasks

### 1. Sanitization Verification (CRITICAL)

**Goal:** Confirm public_release/ contains NO proprietary IP

#### Step 1: Run Sanitization Check
```bash
python3 steward_sanitization_check.py public_release/
```

**Expected Result:** `✅ CLEAN - 11 files scanned, no proprietary terms found`

**If BLOCKED:**
- Review each HIGH-severity finding
- Check if it's a false positive or real leak
- Document findings for user

#### Step 2: Manual Review of public_release/

**Check these files specifically:**
- [ ] `README.md` - No "SPC/DMAIC for AI" references?
- [ ] `RELEASE_NOTES.md` - No "Statistical Process Control for AI"?
- [ ] `runtime_governance_checkpoint.py` - No dual PA logic?
- [ ] `runtime_governance_start.py` - Generic, no proprietary?
- [ ] `runtime_governance_export.py` - No proprietary export formats?
- [ ] `embedding_provider.py` - Generic transformer model only?

**Forbidden Terms (must NOT appear):**
- "Dual Attractor" / "Dual PA" / "AI PA"
- "Lock-on derivation"
- "SPC for AI" / "DMAIC for AI"
- "Telemetric Keys"
- "OriginMind"
- "Progressive PA Extractor"
- "+85.32%" or "+85%" (specific metrics)

#### Step 3: Verify Deletion
```bash
ls public_release/dual_attractor.py
```

**Expected Result:** File not found

**If file exists:** CRITICAL - Sonnet failed to delete it

#### Step 4: Check for Proprietary Logic

**Read these files and verify no proprietary algorithms:**
- [ ] `public_release/runtime_governance_checkpoint.py`
  - Should use simple cosine similarity only
  - No dual fidelity calculations
  - No adaptive weighting

**Red flags:**
- Multiple PA references (User PA + AI PA)
- Weighted fidelity formulas
- Lock-on computation
- SPC/DMAIC frameworks

---

### 2. Security Verification (CRITICAL)

**Goal:** Confirm NO hardcoded secrets in codebase

#### Step 1: Scan for Hardcoded API Keys
```bash
grep -rn "NxFBck0mkmGhM9vn0bvJzHf1scagv44f" . --exclude-dir=.git
```

**Expected Result:** Only 1 match in `.env` file (which is gitignored)

**If found elsewhere:** CRITICAL security leak

#### Step 2: Check Specific Files

**Files Sonnet Modified (verify changes):**

1. **`telos_observatory_v3/components/conversation_display.py`**
   - [ ] Line ~992: Uses `st.secrets.get("MISTRAL_API_KEY")` NOT hardcoded?
   - [ ] Line ~1031: Uses `st.secrets.get("MISTRAL_API_KEY")` NOT hardcoded?
   - [ ] Error handling present if key missing?

2. **`telos_observatory_v3/core/state_manager.py`**
   - [ ] Line ~736: Uses `st.secrets.get("MISTRAL_API_KEY")` NOT hardcoded?
   - [ ] Line ~390: Uses `st.secrets.get("MISTRAL_API_KEY")` NOT hardcoded?
   - [ ] Proper error handling (raises ValueError if missing)?

3. **`telos_observatory_advanced/components/conversation_display.py`**
   - [ ] Line ~353: Uses `st.secrets.get("MISTRAL_API_KEY")` NOT hardcoded?

#### Step 3: Verify Gitignore Coverage

**Check `.gitignore` includes:**
- [ ] `.env` and `*.env` patterns
- [ ] `*_API_KEY*` pattern
- [ ] `credentials.json`
- [ ] `secrets/` directory
- [ ] `.streamlit/secrets.toml`
- [ ] `.mcp.json`
- [ ] `telos_observatory_v3/beta_consents/`
- [ ] `sessions/` and `*.db` files
- [ ] `.telos_*.json` runtime files

**Test gitignore:**
```bash
git check-ignore .env
git check-ignore credentials.json
git check-ignore .streamlit/secrets.toml
```

**Expected:** All should be ignored

---

### 3. Production Code Verification (HIGH)

**Goal:** Confirm debug statements removed

#### Step 1: Scan for Debug Prints
```bash
grep -rn "\[DEBUG\]" telos_observatory_v3/
```

**Expected Result:** 0 matches

**If found:** Document locations for cleanup

#### Step 2: Check state_manager.py Specifically

**Read:** `telos_observatory_v3/core/state_manager.py` around line 465-488

**Verify removed:**
- [ ] `print(f"[DEBUG add_user_message] Building history...")`
- [ ] `print(f"[DEBUG add_user_message] Skipping loading turn")`
- [ ] `print(f"[DEBUG add_user_message] Total messages...")`
- [ ] `print(f"[DEBUG add_user_message] User message...")`

---

### 4. Dependency Verification (MEDIUM)

**Goal:** Confirm requirements.txt files are complete

#### Step 1: Check Observatory V3 Requirements
```bash
cat telos_observatory_v3/requirements.txt
```

**Verify includes:**
- [ ] streamlit>=1.28.0
- [ ] mistralai>=1.0.0
- [ ] sentence-transformers>=2.2.0
- [ ] torch>=2.0.0
- [ ] numpy>=1.24.0
- [ ] pandas>=2.0.0
- [ ] anthropic>=0.25.0 (optional)
- [ ] python-dotenv>=1.0.0

#### Step 2: Check Root Requirements
```bash
cat requirements.txt
```

**Verify comprehensive** (should match above plus dev tools)

---

### 5. Documentation Verification (MEDIUM)

**Goal:** Ensure docs match current state

#### Check for Staleness

**Files to review:**
- [ ] `README.md` - Accurate deployment instructions?
- [ ] `REPO_CLASSIFICATION.md` - Matches actual split?
- [ ] `public_release/README.md` - No outdated claims?
- [ ] `public_release/QUICK_START.md` - Installation works?

**Look for:**
- Date references (should be 2025, not 2024)
- Feature claims (do they exist?)
- Installation steps (do they work?)

---

### 6. Competitive Positioning Analysis (STRATEGIC)

**Goal:** Assess if public release helps or hurts us

**Analyze `public_release/` contents:**

**What we're giving away:**
- Runtime Governance (fidelity measurement)
- Cosine similarity in R^384
- Memory MCP integration pattern
- Generic embedding provider

**What we're keeping:**
- Dual PA architecture
- Lock-on derivation formula
- SPC/DMAIC for AI frameworks
- Observatory platform
- Telemetric Keys / OriginMind

**Questions to answer:**
1. **Time to replicate public release:** How long for competitor to rebuild?
   - Estimate: ___ weeks

2. **Value gap:** How much better is our full system vs. this?
   - Rating: Significant / Moderate / Minimal

3. **Reverse engineering risk:** Could they deduce Dual PA from this?
   - Assessment: Low / Medium / High

4. **Recommendation:** Should we release this publicly?
   - ✅ Yes, strategic advantage
   - ⚠️ Yes, with modifications (list them)
   - 🚨 No, too risky (explain why)

---

### 7. Architecture Health Check (MEDIUM)

**Goal:** Identify any production blockers

**Quick scans:**

#### Circular Dependencies
```bash
grep -r "from telos_observatory_v3" telos_observatory_v3/ | grep "import.*state_manager"
```

**Look for:** Circular import patterns

#### Missing Error Handling

**Check:** `telos_observatory_v3/core/state_manager.py`
- [ ] API key loading has error handling?
- [ ] File operations wrapped in try/except?
- [ ] User inputs validated?

#### TODO Comments
```bash
grep -rn "TODO" telos_observatory_v3/
```

**Document:** Any critical TODOs that block deployment

---

### 8. Beta Readiness Check (MEDIUM)

**Goal:** Can Observatory V3 actually deploy to Streamlit Cloud?

**Check:**
- [ ] `telos_observatory_v3/main.py` has proper entry point?
- [ ] Secrets loaded from `st.secrets` (not hardcoded)?
- [ ] Beta consent system functional?
- [ ] No localhost references?
- [ ] Error messages user-friendly?

**Test locally (if possible):**
```bash
cd telos_observatory_v3
streamlit run main.py
```

**Verify:**
- [ ] App launches without errors
- [ ] Beta onboarding appears
- [ ] Steward panel loads
- [ ] Can add turns (even without real data)

---

## Verification Checklist Summary

### CRITICAL (Must Fix Before Deployment)
- [ ] **Sanitization**: Public release has zero proprietary terms
- [ ] **Secrets**: No hardcoded API keys in code
- [ ] **Gitignore**: All secrets excluded
- [ ] **File Deletion**: dual_attractor.py removed from public_release

### HIGH (Fix Before Launch)
- [ ] **Debug Prints**: All removed from production code
- [ ] **Requirements**: Complete and accurate
- [ ] **Error Handling**: API key loading robust

### MEDIUM (Can Defer)
- [ ] **Documentation**: Up to date and accurate
- [ ] **Architecture**: No critical health issues
- [ ] **Beta Readiness**: App can deploy to Streamlit Cloud

### STRATEGIC (For User Decision)
- [ ] **Competitive Analysis**: Public release risk/benefit assessment
- [ ] **IP Protection**: Dual PA derivation not reverse-engineerable

---

## Opus Deliverables

After verification, create:

### 1. Verification Report (1-2 pages)
```markdown
# Opus Verification Report

## Sonnet Work Verified
- [x/  ] Sanitization complete
- [x/  ] Secrets removed
- [x/  ] Debug prints removed
- [x/  ] Dependencies complete

## Issues Found
1. [Issue title]
   - Severity: CRITICAL / HIGH / MEDIUM / LOW
   - Location: file.py:line
   - Problem: [description]
   - Fix: [recommendation]

## Competitive Assessment
- Public release risk: LOW / MEDIUM / HIGH
- Rationale: [2-3 sentences]
- Recommendation: APPROVE / APPROVE WITH CHANGES / REJECT

## Deployment Recommendation
- Status: READY / NEEDS WORK / BLOCKED
- Critical blockers: [list]
- Estimated time to ready: [hours]
```

### 2. Updated Action Plan (if issues found)

**Format:**
```markdown
## Pre-Deployment Fixes

### Critical (Do Now)
1. [Fix description] - [file:line] - [time estimate]

### High Priority (Before Launch)
2. [Fix description] - [file:line] - [time estimate]

### Medium (Can Defer)
3. [Fix description] - [file:line] - [time estimate]
```

---

## Success Criteria

**Verification is successful if:**
- ✅ Sonnet's work is accurate (or issues documented)
- ✅ No critical security vulnerabilities found
- ✅ Clear deployment recommendation provided
- ✅ Competitive risk assessed
- ✅ Action plan created (if needed)

---

## Final Note to Opus

**Be thorough but pragmatic:**
- Focus on deployment blockers, not perfection
- Think like an adversary for security
- Think like a CEO for strategy
- Document everything you find
- Give clear go/no-go recommendation

**Remember:**
- User wants to deploy this week
- Beta testing with real users
- Nothing pushed to GitHub yet (clean slate)
- GMU partnership depends on successful launch

**Good luck!** 🚀
