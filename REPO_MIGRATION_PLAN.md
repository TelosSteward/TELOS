# TELOS Repository Migration Plan

## Overview

This document outlines the migration plan for splitting the current TELOS research repository into two clean, production-ready repositories:
- **telos-purpose**: Purpose alignment and governance architecture
- **telos-privacy**: Privacy-preserving AI interactions

**Timeline**: Before public rollout
**Status**: Planning phase (active research/validation ongoing)

---

## Current State

**Current Repository**: `/Users/brunnerjf/Desktop/telos`
- Mixed research, validation, and production code
- Multiple experimental iterations (observatory v1/v2/v3)
- Single PA and Dual PA implementations
- Test sessions, validation studies, research briefs

**Purpose**: Research and validation environment

---

## Target Architecture

### Repository 1: `telos-purpose`

**Purpose**: Production-ready purpose alignment framework with Dual PA architecture

**Structure**:
```
telos-purpose/
├── telos_purpose/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── dual_pa.py                    # Dual Primacy Attractor (THE architecture)
│   │   ├── unified_orchestrator.py       # Clean orchestrator (dual PA only)
│   │   ├── governance_config.py          # Configuration system
│   │   └── fidelity_engine.py           # Fidelity measurement
│   ├── interventions/
│   │   ├── __init__.py
│   │   ├── intervention_engine.py        # Intervention system
│   │   └── intervention_strategies.py    # Strategy implementations
│   ├── llm_clients/
│   │   ├── __init__.py
│   │   ├── base_client.py               # Abstract base
│   │   ├── mistral_client.py            # Mistral implementation
│   │   └── openai_client.py             # OpenAI implementation
│   └── utils/
│       ├── __init__.py
│       └── embedding_utils.py
├── examples/
│   ├── basic_usage.py                    # Simple getting started
│   ├── custom_pa_config.py              # Advanced configuration
│   └── intervention_demo.py             # Intervention showcase
├── tests/
│   ├── test_dual_pa.py
│   ├── test_orchestrator.py
│   └── test_interventions.py
├── validation/
│   ├── methodology.md                    # How validation was conducted
│   ├── results_summary.md               # Key findings
│   └── research_briefs/                 # 46 dual PA validation briefs
│       ├── sharegpt_sessions/           # 45 ShareGPT briefs
│       └── claude_conversation/         # 1 Claude conversation brief
├── docs/
│   ├── README.md                        # Main documentation
│   ├── architecture.md                  # Dual PA architecture
│   ├── api_reference.md                 # API documentation
│   ├── quickstart.md                    # Getting started guide
│   └── concepts.md                      # Core concepts
├── pyproject.toml                        # Poetry/modern packaging
├── README.md                            # Professional README
├── LICENSE                              # License file
└── .gitignore
```

### Repository 2: `telos-privacy`

**Purpose**: Privacy-preserving AI interaction framework

**Structure**:
```
telos-privacy/
├── telos_privacy/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── privacy_engine.py
│   │   └── data_governance.py
│   ├── anonymization/
│   ├── encryption/
│   └── utils/
├── examples/
├── tests/
├── docs/
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Migration Strategy

### Phase 1: Inventory & Categorization

**Files to MIGRATE to telos-purpose**:

#### Core Architecture (MUST HAVE)
- `telos_purpose/core/dual_pa.py` - The dual PA implementation
- `telos_purpose/core/unified_orchestrator_steward.py` - Orchestrator (rename to `unified_orchestrator.py`)
- `telos_purpose/core/governance_config.py` - Configuration system
- `telos_purpose/core/fidelity_engine.py` - Fidelity measurement
- `telos_purpose/core/intervention_engine.py` - Intervention system

#### LLM Clients (MUST HAVE)
- `telos_purpose/llm_clients/mistral_client.py` - Mistral implementation
- `telos_purpose/llm_clients/base_llm_client.py` - Base client interface
- Any other production LLM clients

#### Validation Evidence (SHOULD HAVE)
- `dual_pa_research_briefs/` - All 46 research briefs
- `dual_pa_proper_comparison_results.json` - ShareGPT validation data
- `claude_conversation_dual_pa_fresh_results.json` - Claude validation data

#### Documentation Source Material
- Key methodology from research scripts
- Validation approach documentation

**Files to LEAVE BEHIND (archive as telos-research)**:

#### Observatory & Experiments
- `telos_observatory/` - All versions (v1, v2, v3)
- `telos_observatory_v2/`
- `telos_observatory_v3/`

#### Test Sessions & Data
- `test_sessions/` - All test session data
- `sharegpt_sessions/` - Raw ShareGPT data
- Any other test/experimental data directories

#### Single PA Code (deprecated)
- Any single PA specific implementations
- Phase 1/Phase 2 validation code (keep methodology, not code)

#### Research Scripts
- `run_proper_dual_pa_comparison.py`
- `regenerate_claude_starters_dual_pa.py`
- `generate_dual_pa_research_briefs.py`
- `summarize_dual_pa_results.py`
- `convert_claude_to_starters_only.py`
- Any other one-off analysis scripts

#### Experimental/WIP Code
- Any experimental features not yet validated
- Prototype code
- Debug scripts

---

## Phase 2: Code Cleanup

### Changes Required for Clean Migration

#### 1. Remove Single PA Code Paths
**File**: `unified_orchestrator_steward.py`
- Remove all single PA fallback logic
- Remove `mode` parameter (always dual PA)
- Simplify to assume dual PA only

**Before** (current):
```python
def initialize_governance(self):
    if self.governance_config.mode == "dual_pa":
        # Dual PA logic
    else:
        # Single PA fallback
```

**After** (clean):
```python
def initialize_governance(self):
    # Always dual PA
    self.dual_pa.derive_ai_pa(...)
```

#### 2. Rename Files for Clarity
- `unified_orchestrator_steward.py` → `unified_orchestrator.py` (remove "steward")
- Simplify any confusing naming

#### 3. Configuration Simplification
**File**: `governance_config.py`

**Before**:
```python
@staticmethod
def dual_pa_config(strict_mode=False):
    return GovernanceConfig(mode="dual_pa", ...)
```

**After**:
```python
def __init__(self, strict_mode=False, ...):
    # Dual PA is the only mode
    self.strict_mode = strict_mode
    ...
```

#### 4. Clean Up Imports
- Remove unused imports
- Remove experimental feature imports
- Ensure all imports are production-ready

#### 5. Documentation Updates
- Update all docstrings to reflect dual PA only
- Remove references to single PA
- Add clear examples in docstrings

---

## Phase 3: New Repository Setup

### telos-purpose Repository Creation

#### 1. Initialize Repository
```bash
cd ~/Desktop
mkdir telos-purpose
cd telos-purpose
git init
```

#### 2. Create Modern Python Package Structure
```bash
# Use Poetry for modern packaging
poetry init
# Configure pyproject.toml with dependencies
```

#### 3. Copy Migration Files
Follow the structure outlined in Target Architecture section

#### 4. Write Professional README.md

**Structure**:
```markdown
# TELOS Purpose - AI Purpose Alignment Framework

[![License](badge)]() [![Python](badge)]()

TELOS Purpose is a production-ready framework for maintaining AI alignment with user intent through dual primacy attractor (PA) architecture.

## What is TELOS?

TELOS prevents AI drift and misalignment by using a two-attractor governance system:
- **User PA**: Governs WHAT to discuss (user's purpose)
- **AI PA**: Governs HOW to help (assistant's supportive role)

## Key Features

- 85%+ improvement in purpose alignment over baseline
- Zero-config dual PA architecture
- Built-in intervention system
- Support for multiple LLM providers
- Production-tested on 45+ real-world conversations

## Quick Start

[Installation and basic usage]

## Validation

TELOS has been validated across 45 real-world conversations, achieving:
- Mean fidelity improvement: +85.32%
- Perfect correlation in dual PA alignment
- Minimal intervention requirements

See `validation/` directory for full research briefs.

## Documentation

[Links to docs]

## License

[License info]
```

#### 5. Create Documentation

**Files to create**:
- `docs/architecture.md` - Explain dual PA architecture
- `docs/quickstart.md` - Getting started guide
- `docs/api_reference.md` - API documentation
- `docs/concepts.md` - Core concepts (PA, fidelity, interventions)
- `validation/methodology.md` - How validation was conducted
- `validation/results_summary.md` - Key findings from 46 briefs

#### 6. Create Examples

**Files to create**:
- `examples/basic_usage.py` - Simple example
- `examples/custom_pa_config.py` - Advanced configuration
- `examples/intervention_demo.py` - Intervention showcase

**Example Content** (`examples/basic_usage.py`):
```python
"""
Basic TELOS Purpose Usage Example

This example shows how to set up dual PA governance
for a simple conversation.
"""
from telos_purpose.core.unified_orchestrator import UnifiedOrchestrator
from telos_purpose.core.governance_config import GovernanceConfig
from telos_purpose.llm_clients.mistral_client import MistralClient

# Define your user's purpose
user_pa = {
    "purpose": ["Help me write Python code for data analysis"],
    "scope": ["Python programming", "Data analysis", "Best practices"],
    "boundaries": ["No unrelated topics", "Focus on practical solutions"]
}

# Initialize orchestrator (dual PA is automatic)
config = GovernanceConfig(strict_mode=False)
orchestrator = UnifiedOrchestrator(
    governance_config=config,
    user_pa_config=user_pa,
    llm_client=MistralClient(api_key="your-key")
)

# Initialize governance (derives AI PA automatically)
await orchestrator.initialize_governance()

# Start conversation
orchestrator.start_session()

# Generate governed response
result = orchestrator.generate_governed_response(
    user_input="How do I read a CSV file in pandas?",
    conversation_context=[]
)

print(result['governed_response'])
print(f"User PA fidelity: {result['dual_pa_metrics']['user_fidelity']:.4f}")
print(f"AI PA fidelity: {result['dual_pa_metrics']['ai_fidelity']:.4f}")
```

---

## Phase 4: Validation Evidence Migration

### Copy Research Briefs
```bash
# In new telos-purpose repo
mkdir -p validation/research_briefs/sharegpt_sessions
mkdir -p validation/research_briefs/claude_conversation

# Copy briefs
cp ~/Desktop/telos/dual_pa_research_briefs/research_brief_01_*.md \
   validation/research_briefs/sharegpt_sessions/
# ... (all 45 ShareGPT briefs)

cp ~/Desktop/telos/dual_pa_research_briefs/research_brief_46_*.md \
   validation/research_briefs/claude_conversation/
```

### Create Summary Documents

**File**: `validation/methodology.md`
```markdown
# TELOS Validation Methodology

## Overview
TELOS dual PA architecture was validated using isolated session regeneration
across 45 real-world conversations from ShareGPT plus 1 high-stakes conversation
where drift was originally observed.

## Approach
1. Extract conversation starters (user inputs only)
2. Regenerate ALL responses with dual PA governance
3. Compare fidelity metrics against single PA baseline
4. Document interventions and alignment maintenance

## Results
- 85.32% improvement in purpose alignment
- Perfect PA correlation across sessions
- Minimal intervention requirements
- Zero drift in previously problematic conversation

See individual research briefs for detailed session analysis.
```

**File**: `validation/results_summary.md`
```markdown
# TELOS Validation Results Summary

## Key Findings

### ShareGPT Comparison (45 sessions)
- Total sessions: 45
- Dual PA success rate: 100%
- Mean User PA fidelity: [value from stats]
- Mean improvement over baseline: +85.32%

### Claude Conversation (drift scenario)
- Session ID: claude_conversation
- Turns: 51
- User PA fidelity: 1.0000 (perfect)
- AI PA fidelity: 1.0000 (perfect)
- Interventions: 0 (none needed)
- PA correlation: 1.0000

## Implications
Dual PA architecture successfully prevents drift and maintains
alignment across diverse conversation types, including conversations
that originally exhibited alignment problems.

## Research Briefs
See `research_briefs/` directory for detailed per-session analysis.
```

---

## Phase 5: Testing & Validation

### Test Checklist

- [ ] All core imports work
- [ ] Basic example runs successfully
- [ ] Tests pass (migrate relevant tests)
- [ ] Documentation builds correctly
- [ ] Package installs via pip/poetry
- [ ] No references to single PA remain
- [ ] All validation briefs accessible
- [ ] API is clean and intuitive

### Quality Checks

- [ ] Code formatting (black, isort)
- [ ] Type hints where appropriate
- [ ] Docstrings complete
- [ ] No TODO comments in production code
- [ ] License headers if required
- [ ] No hardcoded secrets/keys

---

## Phase 6: Git & GitHub Setup

### Initialize Git History
```bash
cd ~/Desktop/telos-purpose
git add .
git commit -m "Initial commit: TELOS Purpose v1.0 - Dual PA Architecture

- Core dual PA implementation
- Validated across 45+ real-world conversations
- 85%+ improvement in purpose alignment
- Production-ready orchestrator and intervention system
- Complete validation evidence and research briefs
"
```

### Create GitHub Repository
```bash
# On GitHub: Create new repo 'telos-purpose'
git remote add origin git@github.com:yourusername/telos-purpose.git
git branch -M main
git push -u origin main
```

### Repository Settings
- Add description: "AI Purpose Alignment Framework - Dual Primacy Attractor Architecture"
- Add topics: `ai-alignment`, `llm`, `governance`, `purpose-alignment`
- Add license: [Choose appropriate license]
- Enable Issues
- Enable Discussions (for user feedback)

---

## Phase 7: Documentation & Discoverability

### Documentation Website (Optional)
Consider using:
- GitHub Pages with MkDocs
- ReadTheDocs
- Docusaurus

### Blog Post / Announcement
Announce the framework with:
- Problem statement (AI drift)
- Solution (dual PA architecture)
- Validation results (85%+ improvement)
- Getting started guide
- Link to GitHub

### Social Media / Community
- AI/ML communities
- Reddit (r/MachineLearning, r/LocalLLaMA)
- Twitter/X
- HackerNews (if appropriate)

---

## telos-privacy Migration (Separate Plan)

Similar process for privacy-focused repository:
1. Identify privacy-specific modules
2. Create clean repo structure
3. Migrate code
4. Write documentation
5. Validate and test
6. Publish

**Note**: This can be done in parallel or after telos-purpose is complete.

---

## Archive Strategy

### Option 1: Keep Current Repo as telos-research
```bash
cd ~/Desktop/telos
git remote rename origin old-origin
# Keep as local research archive
```

### Option 2: Archive to Private Repo
```bash
# Create private GitHub repo 'telos-research-archive'
git remote add origin git@github.com:yourusername/telos-research-archive.git
git push -u origin main
```

### Option 3: Local Archive Only
```bash
cd ~/Desktop
mv telos telos-research-archive-2024
# Keep locally but don't push to GitHub
```

**Recommendation**: Keep as research archive (Option 1 or 2) for reference.

---

## Timeline Estimate

**When you're ready to split**:

- **Phase 1** (Inventory): 1-2 hours
- **Phase 2** (Code Cleanup): 3-4 hours
- **Phase 3** (New Repo Setup): 2-3 hours
- **Phase 4** (Validation Migration): 1-2 hours
- **Phase 5** (Testing): 2-3 hours
- **Phase 6** (Git/GitHub): 1 hour
- **Phase 7** (Documentation): 3-5 hours

**Total**: ~13-20 hours of focused work

---

## Success Criteria

Migration is successful when:

- [ ] telos-purpose repository is production-ready
- [ ] All core functionality works without research artifacts
- [ ] Documentation is complete and professional
- [ ] Examples run out of the box
- [ ] Validation evidence is accessible
- [ ] No references to deprecated single PA architecture
- [ ] Package is installable via standard Python tools
- [ ] README clearly communicates value proposition
- [ ] Ready for public rollout

---

## Notes

- **Do NOT rush this migration** - Better to wait and do it right when splitting purpose/privacy
- Current repo can continue as research/validation environment
- Migration should happen in one clean sweep, not incrementally
- Focus on making telos-purpose the "ideal" public-facing repo
- telos-privacy can follow same pattern once privacy modules are ready

---

**Status**: Planning document (ready for execution when repos split)
**Last Updated**: 2024-11-02
**Next Step**: Complete current validation work, then execute migration when ready for public rollout
