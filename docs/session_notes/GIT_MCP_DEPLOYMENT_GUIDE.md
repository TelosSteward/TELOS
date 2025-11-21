# Git MCP Deployment Guide
**Purpose:** Step-by-step guide for deploying TELOS to dual repositories
**Date:** November 5, 2025
**Prerequisites:** Opus verification complete, all critical issues resolved

---

## Overview

**Strategy:** Simultaneous deployment of two repos from single codebase
- **PUBLIC:** `telos_purpose` (Runtime Governance - sanitized foundation)
- **PRIVATE:** `telos_observatory` (Observatory Platform - full system)

**Approach:** Use Git MCP (Model Context Protocol) for repository operations

---

## Pre-Deployment Checklist

**Before proceeding, ensure:**
- [ ] Opus verification report reviewed
- [ ] All CRITICAL issues resolved
- [ ] `SONNET_WORK_COMPLETION_REPORT.md` reviewed
- [ ] GitHub organizations/accounts ready
- [ ] Streamlit Cloud account configured

---

## Part 1: Prepare PUBLIC Repo (telos_purpose)

### Step 1: Create Clean Directory for Public Release

**Create staging directory:**
```bash
cd /Users/brunnerjf/Desktop
mkdir telos_purpose_staging
cp -r telos_privacy/public_release/* telos_purpose_staging/
cd telos_purpose_staging
```

**Verify contents:**
```bash
ls -la
# Should show:
# - runtime_governance_start.py
# - runtime_governance_checkpoint.py
# - runtime_governance_export.py
# - embedding_provider.py
# - README.md
# - QUICK_START.md
# - RELEASE_NOTES.md
# - INTERPRETATION_GUIDE.md
# - requirements.txt
# - governance_config.example.json
# - LICENSE (if present)
```

### Step 2: Initialize Git with Git MCP

**Using Claude Code with Git MCP:**
```
# In Claude Code chat:
"Please use Git MCP to initialize a new repository in telos_purpose_staging/"
```

**Git MCP commands Claude will use:**
```python
# Initialize repo
mcp__git__init(repo_path="/Users/brunnerjf/Desktop/telos_purpose_staging")

# Copy PUBLIC gitignore
cp /Users/brunnerjf/Desktop/telos_privacy/.gitignore_PUBLIC .gitignore

# Stage all files
mcp__git__add(
    repo_path="/Users/brunnerjf/Desktop/telos_purpose_staging",
    files=["."]
)

# Check status
mcp__git__status(repo_path="/Users/brunnerjf/Desktop/telos_purpose_staging")
```

### Step 3: Create Initial Commit

**Commit message (via Git MCP):**
```python
mcp__git__commit(
    repo_path="/Users/brunnerjf/Desktop/telos_purpose_staging",
    message="""Initial release: Runtime Governance v0.1.0

Runtime Governance provides turn-by-turn fidelity measurement for Claude Code
sessions. Keep your work aligned with project goals using real mathematics
(embeddings in R^384, cosine similarity).

Features:
- Automatic fidelity measurement
- 100% local (zero cost, no external APIs)
- Memory MCP integration
- Session export (JSON, CSV, grant reports)

Built by the TELOS Project.
License: MIT

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
"""
)
```

### Step 4: Connect to GitHub

**Create remote repository on GitHub:**
- Go to https://github.com/new
- Repository name: `telos_purpose`
- Description: "Runtime Governance for Claude Code - Keep AI sessions aligned with project goals"
- Visibility: **PUBLIC**
- Do NOT initialize with README (we have one)

**Connect local repo to GitHub:**
```bash
# Using Git MCP:
git remote add origin https://github.com/[your-username]/telos_purpose.git

# Note: Git MCP doesn't have remote management yet, so use bash:
cd telos_purpose_staging
git remote add origin https://github.com/[your-username]/telos_purpose.git
git branch -M main
git push -u origin main
```

---

## Part 2: Prepare PRIVATE Repo (telos_observatory)

### Step 1: Clean Current Repository

**Current state:**
- Working directory: `/Users/brunnerjf/Desktop/telos_privacy`
- Branch: `DualAttractorCanonicalImplementation`
- Modified files: 7
- Untracked files: 19

**Stage Sonnet's work:**
```python
# Using Git MCP in Claude Code:
mcp__git__add(
    repo_path="/Users/brunnerjf/Desktop/telos_privacy",
    files=[
        ".gitignore_PRIVATE",
        ".gitignore_PUBLIC",
        "REPO_CLASSIFICATION.md",
        "steward_sanitization_check.py",
        "steward_pm.py",
        "telos_observatory_v3/requirements.txt",
        "SONNET_WORK_COMPLETION_REPORT.md",
        "OPUS_AUDIT_BRIEF.md",
        "OPUS_VERIFICATION_CHECKLIST.md",
        "GIT_MCP_DEPLOYMENT_GUIDE.md",
        "telos_observatory_v3/components/conversation_display.py",
        "telos_observatory_v3/core/state_manager.py",
        "telos_observatory_advanced/components/conversation_display.py"
    ]
)
```

**Review changes:**
```python
mcp__git__diff_staged(repo_path="/Users/brunnerjf/Desktop/telos_privacy")
```

### Step 2: Commit Pre-Deployment Work

**Commit message:**
```python
mcp__git__commit(
    repo_path="/Users/brunnerjf/Desktop/telos_privacy",
    message="""Pre-deployment security & sanitization

Prepare codebase for dual-repository deployment (public/private split).

Security fixes:
- Replace 4 hardcoded API keys with environment variables
- Verify gitignore coverage for secrets
- Remove debug print statements from production code

IP protection:
- Sanitize public_release/ (remove dual_attractor.py, SPC/DMAIC)
- Create steward_sanitization_check.py scanner
- Document repository classification strategy

Infrastructure:
- Create requirements.txt for Observatory V3
- Create .gitignore_PRIVATE and .gitignore_PUBLIC
- Document Git MCP deployment workflow

Documentation:
- SONNET_WORK_COMPLETION_REPORT.md (Sonnet's work)
- OPUS_VERIFICATION_CHECKLIST.md (for Opus audit)
- REPO_CLASSIFICATION.md (public/private strategy)

All changes verified. Ready for Opus verification.

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
"""
)
```

### Step 3: Replace .gitignore

**Switch to PRIVATE gitignore:**
```bash
cp .gitignore_PRIVATE .gitignore
git add .gitignore
git commit -m "Switch to private repo gitignore (comprehensive exclusions)"
```

### Step 4: Create New GitHub Repository

**Create remote repository on GitHub:**
- Go to https://github.com/new
- Repository name: `telos_observatory`
- Description: "TELOS Observatory Platform - AI Governance Infrastructure (Private)"
- Visibility: **PRIVATE**
- Do NOT initialize with README

**Connect and push:**
```bash
# Remove old origin if exists:
git remote remove origin

# Add new origin:
git remote add origin https://github.com/[your-username]/telos_observatory.git

# Push to GitHub:
git branch -M main
git push -u origin main
```

---

## Part 3: Streamlit Cloud Deployment

### Step 1: Deploy Observatory V3

**Create new Streamlit app:**
1. Go to https://share.streamlit.io/deploy
2. Connect to GitHub repository: `telos_observatory`
3. Branch: `main`
4. Main file path: `telos_observatory_v3/main.py`

**Configure secrets:**
1. Go to app settings → Secrets
2. Add:
```toml
MISTRAL_API_KEY = "your_actual_key_here"
```

**Environment variables:**
```toml
# .streamlit/secrets.toml (on Streamlit Cloud dashboard)
MISTRAL_API_KEY = "your_actual_key_here"
```

### Step 2: Test Beta Deployment

**Access app:**
- URL: `https://[app-name].streamlit.app`
- Test: Beta onboarding flow
- Test: Demo mode conversation
- Test: Steward panel
- Test: Export functionality

**Verify:**
- [ ] App loads without errors
- [ ] Beta consent form appears
- [ ] Steward responds to queries
- [ ] No exposed secrets in logs
- [ ] Error messages user-friendly

---

## Part 4: Post-Deployment Verification

### Step 1: Security Check

**Verify no secrets in GitHub:**
```bash
# Clone public repo and scan:
cd /tmp
git clone https://github.com/[your-username]/telos_purpose.git
cd telos_purpose
grep -r "NxFBck0mkmGhM9vn0bvJzHf1scagv44f" .
grep -r "api_key.*=.*\"" .
# Should return 0 results

# Clone private repo and scan:
cd /tmp
git clone https://github.com/[your-username]/telos_observatory.git
cd telos_observatory
grep -r "NxFBck0mkmGhM9vn0bvJzHf1scagv44f" .
# Should only find .env (which is gitignored)
```

### Step 2: Public Release Validation

**Test installation from public repo:**
```bash
cd /tmp
git clone https://github.com/[your-username]/telos_purpose.git
cd telos_purpose
pip install -r requirements.txt
python3 runtime_governance_start.py --help
# Should work without errors
```

**Run sanitization check:**
```bash
# From private repo:
python3 steward_sanitization_check.py /tmp/telos_purpose/
# Should return: ✅ CLEAN
```

### Step 3: Documentation Check

**Verify README accuracy:**
- [ ] Installation instructions work?
- [ ] Example commands execute?
- [ ] Links point to correct locations?
- [ ] No references to private repo?

---

## Part 5: Ongoing Maintenance

### Syncing Public Release

**When updating public release:**
1. Make changes in `telos_privacy/public_release/`
2. Run sanitization check:
   ```bash
   python3 steward_sanitization_check.py public_release/
   ```
3. If clean, copy to public repo:
   ```bash
   cp -r public_release/* ../telos_purpose_staging/
   cd ../telos_purpose_staging
   git add .
   git commit -m "Update: [describe changes]"
   git push
   ```

### Updating Private Platform

**Normal workflow:**
```python
# Stage changes
mcp__git__add(repo_path="/Users/brunnerjf/Desktop/telos_privacy", files=[...])

# Commit
mcp__git__commit(
    repo_path="/Users/brunnerjf/Desktop/telos_privacy",
    message="[description]"
)

# Push
git push origin main
```

**Streamlit will auto-deploy** from main branch

---

## Git MCP Commands Reference

### Status & Info
```python
# Check repository status
mcp__git__git_status(repo_path="/path/to/repo")

# View commit history
mcp__git__git_log(repo_path="/path/to/repo", max_count=10)

# Show specific commit
mcp__git__git_show(repo_path="/path/to/repo", revision="commit_sha")
```

### Staging & Commits
```python
# Stage files
mcp__git__git_add(repo_path="/path/to/repo", files=["file1.py", "file2.md"])

# Unstage all
mcp__git__git_reset(repo_path="/path/to/repo")

# Commit
mcp__git__git_commit(repo_path="/path/to/repo", message="Commit message")
```

### Diffs
```python
# View unstaged changes
mcp__git__git_diff_unstaged(repo_path="/path/to/repo")

# View staged changes
mcp__git__git_diff_staged(repo_path="/path/to/repo")

# Compare branches
mcp__git__git_diff(repo_path="/path/to/repo", target="main...feature-branch")
```

### Branches
```python
# List branches
mcp__git__git_branch(repo_path="/path/to/repo", branch_type="all")

# Create new branch
mcp__git__git_create_branch(
    repo_path="/path/to/repo",
    branch_name="feature-xyz",
    base_branch="main"
)

# Switch branches
mcp__git__git_checkout(repo_path="/path/to/repo", branch_name="feature-xyz")
```

---

## Rollback Plan

**If deployment fails:**

### Public Repo Issues
1. Delete GitHub repository
2. Fix issues locally in `public_release/`
3. Re-run sanitization check
4. Restart from Part 1

### Private Repo Issues
1. Don't delete repository (contains history)
2. Create new branch: `git checkout -b hotfix-deployment`
3. Fix issues
4. Test locally
5. Merge to main and push

### Streamlit Deployment Issues
1. App settings → Reboot app
2. Check secrets configuration
3. Review logs for errors
4. Test locally first: `streamlit run telos_observatory_v3/main.py`

---

## Success Criteria

**Deployment is successful if:**
- ✅ Public repo accessible at github.com/[user]/telos_purpose
- ✅ Private repo accessible at github.com/[user]/telos_observatory
- ✅ Observatory V3 running at [app-name].streamlit.app
- ✅ No secrets exposed in public repo
- ✅ Public release passes sanitization check
- ✅ Beta users can access app and consent
- ✅ Documentation accurate and complete

---

## Timeline Estimate

**Assuming Opus verification complete:**
- Part 1 (Public repo): 30 minutes
- Part 2 (Private repo): 20 minutes
- Part 3 (Streamlit deploy): 20 minutes
- Part 4 (Verification): 30 minutes
- **Total: ~2 hours**

**If issues found:** Add 1-4 hours per critical issue

---

## Support Resources

**Git MCP Documentation:**
- Available tools: status, add, commit, diff, log, branch, checkout, show
- Not available: push, pull, remote (use bash for these)

**Streamlit Cloud:**
- Docs: https://docs.streamlit.io/streamlit-community-cloud
- Secrets: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management

**GitHub:**
- Creating repos: https://docs.github.com/en/repositories/creating-and-managing-repositories

---

## Final Checklist

**Before going live:**
- [ ] Opus verification complete (CRITICAL)
- [ ] All critical issues resolved
- [ ] `.env` file NOT committed to ANY repo
- [ ] Public repo sanitization verified (0 violations)
- [ ] Streamlit secrets configured
- [ ] Beta consent system tested
- [ ] Error handling robust
- [ ] Documentation reviewed

**After going live:**
- [ ] Monitor GitHub Issues on public repo
- [ ] Track Streamlit app logs for errors
- [ ] Collect beta user feedback
- [ ] Update documentation as needed

---

**Ready to deploy!** 🚀

Follow this guide step-by-step after Opus verification is complete.
