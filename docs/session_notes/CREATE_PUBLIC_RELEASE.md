# Creating the Public Release

## Step 1: Create Separate Public Repo

```bash
# On your machine (outside this repo)
mkdir ~/runtime-governance
cd ~/runtime-governance
git init

# Copy and rebrand files
cp /path/to/telos/telos_session_start.py ./runtime_governance_start.py
cp /path/to/telos/telos_turn_checkpoint.py ./runtime_governance_checkpoint.py
cp /path/to/telos/telos_session_export.py ./runtime_governance_export.py
cp /path/to/telos/embedding_provider.py ./embedding_provider.py
cp /path/to/telos/dual_attractor.py ./dual_attractor.py

# Add documentation
cp /path/to/telos/RUNTIME_GOVERNANCE_README.md ./README.md
cp /path/to/telos/QUICK_START.md ./QUICK_START.md
# etc.

# Create on GitHub
gh repo create telos-project/runtime-governance --public
git add .
git commit -m "Initial release: Runtime Governance for Claude Code"
git push origin main
```

## Step 2: Users Install It

### Actual Steps for End Users:

#### Method 1: Direct Clone (Recommended)

```bash
# User is already in their Claude Code project directory
cd ~/my-claude-code-project

# Clone the governance scripts into a subdirectory
git clone https://github.com/telos-project/runtime-governance.git .runtime-governance

# Or use sparse checkout to get just the scripts in root
curl -O https://raw.githubusercontent.com/telos-project/runtime-governance/main/runtime_governance_start.py
curl -O https://raw.githubusercontent.com/telos-project/runtime-governance/main/runtime_governance_checkpoint.py
curl -O https://raw.githubusercontent.com/telos-project/runtime-governance/main/runtime_governance_export.py
curl -O https://raw.githubusercontent.com/telos-project/runtime-governance/main/embedding_provider.py
curl -O https://raw.githubusercontent.com/telos-project/runtime-governance/main/dual_attractor.py

# Install dependencies
pip install sentence-transformers numpy

# Done - scripts are now in their project
```

#### Method 2: pip Package (Future - More Polished)

```bash
pip install claude-runtime-governance

# Installs globally, can run from any directory
runtime-governance init
```

## Step 3: User Configuration

**User edits THEIR `.claude_project.md`:**

```markdown
## 🔭 RUNTIME GOVERNANCE - ACTIVE

**PA Baseline:**
"Build a React e-commerce site by Q2 2025 with Stripe payments,
user authentication, and product catalog. Maintain test coverage >80%."

**Fidelity Thresholds:**
- F ≥ 0.8: ✅ ON TRACK
- 0.7 ≤ F < 0.8: ⚠️ WARNING
- F < 0.7: 🚨 DRIFT
```

## Step 4: User Starts Session

```bash
# User runs in their project directory
python3 runtime_governance_start.py
```

Output:
```
🔭 Runtime Governance - Session Initialization
============================================================

📋 Extracting PA baseline from .claude_project.md...
✅ PA extracted (147 chars)

⚠️  CLAUDE: Create Memory MCP entity now
[Instructions for Claude to create session entity]

💾 Session info saved to .telos_active_session.json
✅ Runtime Governance INITIALIZED
```

## Step 5: User Works with Claude Code

**Claude Code automatically runs checkpoints** after each turn (if configured).

Or user manually runs:
```bash
python3 runtime_governance_checkpoint.py --user "msg" --assistant "response"
```

## Step 6: User Exports Data

```bash
python3 runtime_governance_export.py
```

Gets JSON with all turns, fidelities, session stats.

---

## The Key Point

**Users download scripts TO their existing Claude Code project.**

They don't clone your entire TELOS repo.
They just get the generic governance scripts.

### File Structure in User's Project:

```
their-project/
├── .claude_project.md          (THEIR PA definition)
├── runtime_governance_start.py  (from your repo)
├── runtime_governance_checkpoint.py (from your repo)
├── runtime_governance_export.py (from your repo)
├── embedding_provider.py        (from your repo)
├── dual_attractor.py           (from your repo)
├── .telos_active_session.json  (generated locally)
├── .telos_checkpoint_*.json    (generated locally)
└── [their actual code files]
```

**Everything stays local to them.**

Your TELOS repo stays private with `telos_*` naming.
Public repo has generic `runtime_governance_*` naming.

---

## Should We Build This?

I can create:
1. Rebranded copies of the scripts (generic naming)
2. Installation script for users
3. Updated docs with correct file names
4. Sample `.claude_project.md` template

This would be a clean public release separate from your TELOS infrastructure.

Want me to build that now?
