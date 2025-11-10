# Observatory Unified Mode Architecture - Structured Planning Document

**Created**: 2025-11-09T18:31:35.718973

---

## Overview

This document outlines the comprehensive plan for Observatory Unified Mode Architecture.

---

## Phases

### Phase 1: Architecture Analysis

**Description**: Analyze current duplicated mode rendering and plan unified approach

**Duration Estimate**: Complete (already analyzed)

**Dependencies**: None

**Steps**:

1. Document current architecture: 4 duplicated if/elif blocks (DEMO/BETA/TELOS/DEVOPS)
2. Identify redundant code: ~76 lines doing nearly identical component calls
3. Define unified rendering function with mode-based feature flags
4. Document component-level conditional logic (pa_converged gates)
5. Run structured_planning.py to generate formal planning artifacts

---

### Phase 2: Unified Rendering Implementation

**Description**: Create single master rendering function with feature flags

**Duration Estimate**: Complete (already implemented)

**Dependencies**: Architecture Analysis

**Steps**:

1. Create render_mode_content(mode: str) function in main.py
2. Define mode-specific feature flags: show_observation_deck, show_teloscope, show_devops_header
3. Set telos_demo_mode flag based on current mode
4. Replace duplicated if/elif blocks with single function call
5. Maintain component-level conditional logic (components check pa_converged internally)

---

### Phase 3: Mode Validation & Testing

**Description**: Test all modes to ensure unified rendering works correctly

**Duration Estimate**: 30 minutes

**Dependencies**: Unified Rendering Implementation

**Steps**:

1. Start Streamlit application
2. Test DEMO mode: verify demo PA loaded, no Observation Deck, no TELOSCOPE
3. Test BETA mode: verify progressive PA extraction, Observation Deck available
4. Test TELOS mode: verify full Observatory with TELOSCOPE controls
5. Test DEVOPS mode: verify debug header, full access, progressive PA
6. Test PA convergence gates: verify UI shows 'Calibrating...' until ~10 turns
7. Test PA established state: verify metrics appear after convergence

---

### Phase 4: Documentation & Code Comments

**Description**: Document the unified architecture pattern for maintainability

**Duration Estimate**: 15 minutes

**Dependencies**: Mode Validation & Testing

**Steps**:

1. Add docstring to render_mode_content() explaining feature flags
2. Add inline comments explaining mode-specific behaviors
3. Document the principle: 'Components contain conditional logic, modes control features'
4. Update any relevant README or architecture documentation

---

### Phase 5: Version Control

**Description**: Commit changes and sync across repositories

**Duration Estimate**: 10 minutes

**Dependencies**: Documentation & Code Comments

**Steps**:

1. Review all changes with git status and git diff
2. Create commit: 'Refactor: Unify mode rendering with feature flags'
3. Push to TELOS-Observatory repo (https://github.com/TelosSteward/TELOS-Observatory)
4. Push to TELOS main repo (https://github.com/TelosSteward/TELOS)
5. Verify both remotes updated successfully

---

## Phase Dependencies

- **Architecture Analysis** → **Unified Rendering Implementation**: Must understand current architecture before refactoring
- **Unified Rendering Implementation** → **Mode Validation & Testing**: Must implement unified rendering before testing it

---

## Risk Analysis

### LOW: Feature flags might not cover all mode-specific behaviors

**Mitigation**: Test each mode thoroughly to ensure parity with original behavior

### LOW: PA convergence gates might break in edge cases

**Mitigation**: Component-level checks remain intact; mode layer only controls visibility

### MEDIUM: Code duplication might reappear if pattern not documented

**Mitigation**: Add clear docstrings and comments explaining the unified architecture principle

---

## Success Criteria

- **Single unified rendering function used by all modes**
  - Measurement: Verify render_mode_content() called by both Steward panel and full-width layouts
- **All modes maintain correct behavior**
  - Measurement: Manual testing confirms DEMO/BETA/TELOS/DEVOPS all work as before refactor
- **Code reduction achieved**
  - Measurement: Reduced from ~76 duplicated lines to ~32 unified lines (58% reduction)
- **PA convergence gates work correctly**
  - Measurement: UI shows 'Calibrating...' until ~10 turns, then displays metrics
- **Components remain modular**
  - Measurement: Components contain conditional logic; modes just control feature visibility

---

## Next Steps

1. Review this plan with stakeholders
2. Validate dependencies and timeline
3. Begin Phase 1 implementation
4. Track progress against success criteria

---

*Generated on 2025-11-09 18:31:35*
