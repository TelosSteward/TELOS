# TELOS Repository Cleanup Plan
## Creating Clean GitHub Repository from Local Files

### Current Situation Analysis

**Repository Size:**
- Total files: ~1000+ files
- Many duplicates, archives, and deprecated versions
- Mixed personal notes with production code
- Multiple versions of same documents

### Categorization Strategy

## 🟢 ESSENTIAL FOR GITHUB (Goes to Privacy_PreCommit)

### Core Implementation
```
telos/
├── __init__.py
├── core/
│   ├── primacy_attractor.py
│   ├── primacy_math.py
│   ├── orchestration.py
│   ├── telemetry.py
│   └── intervention_controller.py
├── utils/
│   ├── conversation_manager.py
│   └── mistral_client.py
└── tests/
    ├── test_primacy.py
    └── test_orchestration.py
```

### Primary Documentation
```
docs/
├── README.md
├── TELOS_Whitepaper_v2.3.md (latest version only)
├── TELOS_Academic_Paper.md
├── TELOS_EU_Article72_Submission.md
├── TELOS_Implementation_Guide.md
├── TELOS_TECHNICAL_PAPER.md
└── QUICK_START_GUIDE.md
```

### Observatory (Production UI)
```
telos_observatory_v3/
├── main.py
├── components/
├── pages/
└── requirements.txt
```

### Essential Configurations
```
config/
├── config.example.json
├── governance_config.example.json
└── healthcare_pa_example.json
```

### Setup & Installation
```
setup.py
requirements.txt
README.md
LICENSE
.gitignore
```

---

## 🔴 STAYS LOCAL ONLY (Not for GitHub)

### Archives & Old Versions
```
archive/
docs/archive/
duplicates/
txt-versions/
```

### Personal/Internal Documents
```
JB_PROTOCOLS.md
OPUS_AUDIT_BRIEF.md
MONETIZATION_ANALYSIS.md
grant_applications/
planning_tools/
```

### Session Files & Temporary Data
```
.telos_*.json
sessions/
claude_code_session_*.json
.claude/
```

### Multiple Document Versions
```
TELOS_Whitepaper.md
TELOS_Whitepaper_v2.1.md
TELOS_Whitepaper_v2.2.md
(keep only v2.3)
```

### Work-in-Progress
```
SONNET_WORK_COMPLETION_REPORT.md
CREATE_PUBLIC_RELEASE.md
REPO_CLASSIFICATION.md
```

### Internal Tools
```
steward.py (unless production-ready)
concatenate_for_opus.py
extract_essential_telos.py
```

---

## 🟡 REVIEW BEFORE DECISION

### Might Be Useful
```
demo_mode/ (if polished)
public_release/ (if ready)
steward/ (if production-ready)
```

### Documentation to Review
```
docs/positioning/ (maybe public)
docs/guides/ (select best ones)
docs/implementation/ (consolidate)
```

---

## Directory Structure for Privacy_PreCommit

```
Privacy_PreCommit/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── .gitignore
│
├── telos/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── primacy_attractor.py
│   │   ├── primacy_math.py
│   │   ├── orchestration.py
│   │   ├── intervention_controller.py
│   │   └── telemetry.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── conversation_manager.py
│   │   ├── mistral_client.py
│   │   └── embedding_provider.py
│   │
│   └── tests/
│       ├── __init__.py
│       ├── test_primacy.py
│       ├── test_orchestration.py
│       └── test_integration.py
│
├── telos_observatory/
│   ├── README.md
│   ├── main.py
│   ├── requirements.txt
│   ├── components/
│   ├── pages/
│   └── assets/
│
├── docs/
│   ├── README.md
│   ├── whitepapers/
│   │   ├── TELOS_Whitepaper_v2.3.md
│   │   ├── TELOS_Academic_Paper.md
│   │   └── TELOS_Technical_Paper.md
│   │
│   ├── guides/
│   │   ├── QUICK_START.md
│   │   ├── INSTALLATION_GUIDE.md
│   │   └── DEPLOYMENT_GUIDE.md
│   │
│   └── regulatory/
│       └── TELOS_EU_Article72_Submission.md
│
├── examples/
│   ├── basic_usage.py
│   ├── healthcare_pa.py
│   └── config_examples/
│       ├── config.example.json
│       └── governance.example.json
│
└── scripts/
    ├── setup.sh
    ├── test.sh
    └── deploy.sh
```

---

## GitHub Repository Cleanup Actions

### 1. Files to Remove from GitHub
```bash
# Remove all archive folders
git rm -r archive/
git rm -r docs/archive/
git rm -r docs/fixes/  # Old fixes, now integrated

# Remove duplicate/old versions
git rm docs/TELOS_Whitepaper.md
git rm docs/TELOS_Whitepaper_v2.1.md
git rm docs/TELOS_Whitepaper_v2.2.md

# Remove personal/internal files
git rm JB_PROTOCOLS.md
git rm OPUS_AUDIT_BRIEF.md
git rm MONETIZATION_ANALYSIS.md

# Remove session/temporary files
git rm -r sessions/
git rm .telos_*.json
```

### 2. Files to Move/Rename
```bash
# Consolidate documentation
docs/TELOS_Whitepaper_v2.3.md → docs/whitepapers/TELOS_Whitepaper.md
docs/guides/QUICK_START_GUIDE.md → docs/QUICK_START.md
```

### 3. New Files to Add
```bash
# Add proper README
README.md (professional, with badges, clear instructions)

# Add LICENSE
LICENSE (MIT or Apache 2.0)

# Add proper .gitignore
.gitignore (Python, Node, IDE files)
```

---

## Execution Steps

### Phase 1: Create Privacy_PreCommit
1. Create new folder structure
2. Copy essential files only
3. Clean up file names and paths
4. Add proper documentation

### Phase 2: Test Privacy_PreCommit
1. Verify all core functionality works
2. Test installation process
3. Validate documentation accuracy
4. Run test suite

### Phase 3: Prepare GitHub Commands
1. Create branch for cleanup
2. Generate git rm commands for deprecated files
3. Prepare commit message
4. Get approval before push

### Phase 4: Execute GitHub Cleanup (WITH APPROVAL)
1. Push clean version to new branch
2. Create PR for review
3. Merge to main after approval
4. Tag release version

---

## File Count Estimation

### Current State
- Total files: ~1000+
- Useful files: ~200
- GitHub-ready: ~100

### After Cleanup
- Privacy_PreCommit: ~100 files
- GitHub repository: ~100 files
- Local archive: ~900 files (kept safe)

---

## Safety Measures

1. **Everything stays local** - Nothing deleted from telos_privacy/
2. **New folder for clean version** - Privacy_PreCommit/
3. **No GitHub push without approval**
4. **Create backup branch first**
5. **Test everything before committing**

---

## Next Actions

1. ✅ Review this plan
2. Create Privacy_PreCommit folder structure
3. Copy essential files
4. Test the clean version
5. Prepare git cleanup commands
6. Get approval
7. Execute GitHub cleanup