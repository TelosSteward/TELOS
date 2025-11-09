# TELOS BETA Implementation Fix - Structured Planning Document

**Created**: 2025-11-09T17:04:11.620370

---

## Overview

This document outlines the comprehensive plan for TELOS BETA Implementation Fix.

---

## Phases

### Phase 1: Enable Core TELOS Governance

**Description**: Fix the fundamental issue preventing TELOS governance from functioning - interventions are disabled and PA is hardcoded instead of extracted.

**Duration Estimate**: 2-3 days

**Dependencies**: None

**Steps**:

1. Enable interventions in state_manager.py (change enable_interventions=False to True at line 408)
2. Test that UnifiedGovernanceSteward is actually being called on each turn
3. Add telemetry logging to track when interventions fire
4. Verify intervention logic is working with simple test case
5. Document intervention triggering conditions and thresholds

---

### Phase 2: Implement Real PA Extraction

**Description**: Replace hardcoded generic PA with progressive calibration that learns from actual user conversations during first 10 turns.

**Duration Estimate**: 3-5 days

**Dependencies**: Enable Core TELOS Governance

**Steps**:

1. Remove hardcoded generic PA from state_manager.py lines 388-396
2. Implement progressive PA extraction using conversation analysis
3. Extract Purpose from first 3-5 turns using semantic clustering
4. Extract Scope by identifying key topics and themes
5. Extract Boundaries by detecting user preferences and constraints
6. Lock PA at turn 11 and mark as calibrated
7. Add telemetry to track PA evolution across turns
8. Test with TELOS project conversation to verify PA captures "AI governance project" as purpose

---

### Phase 3: Calculate Real Fidelity Scores

**Description**: Implement actual fidelity calculation after each response and remove all hardcoded 0.850 values from displays.

**Duration Estimate**: 2-3 days

**Dependencies**: Implement Real PA Extraction

**Steps**:

1. After streaming response completes, embed the full response text
2. Calculate cosine similarity between response embedding and PA centroid
3. Store real fidelity score in turn data (replace None values)
4. Remove hardcoded 0.85 from conversation_display.py lines 1636, 2375
5. Update display to show actual calculated fidelity from turn data
6. Add fidelity calculation logging for debugging
7. Test that fidelity varies based on actual conversation content
8. Verify low fidelity scores trigger interventions

---

### Phase 4: Fix PA Display

**Description**: Replace placeholder PA text with actual extracted PA (Purpose, Scope, Boundaries) from conversation analysis.

**Duration Estimate**: 1-2 days

**Dependencies**: Implement Real PA Extraction

**Steps**:

1. Identify where PA display pulls data in conversation_display.py
2. Replace placeholder text ("Establish conversation purpose", etc.) with real PA data
3. Format PA display to show: Purpose as bullet points, Scope as topic list, Boundaries as constraint statements
4. Add calibration status indicator (Calibrating turns 1-10, Locked turn 11+)
5. Test display shows actual extracted PA from test conversations
6. Verify PA updates during calibration phase (turns 1-10)
7. Confirm PA locks and stays stable after turn 11

---

### Phase 5: Remove Fake Displays

**Description**: Remove counterfactual display showing demo data and gray out Observatory section until real data is available.

**Duration Estimate**: 1 day

**Dependencies**: None

**Steps**:

1. Remove counterfactual display component entirely from BETA interface
2. Gray out Observatory section with "Coming Soon" message
3. Clean up any references to counterfactual in BETA code
4. Update UI to show only: PA Display (with real data) and Conversation (with real fidelity)
5. Test that removed displays do not appear in BETA mode
6. Document what displays are available in BETA vs full Observatory

---

### Phase 6: Drift Detection & Intervention Logic

**Description**: Implement logic that detects when conversation drifts from PA and applies interventions to redirect.

**Duration Estimate**: 3-4 days

**Dependencies**: Calculate Real Fidelity Scores, Enable Core TELOS Governance

**Steps**:

1. Define drift threshold (e.g., fidelity < 0.75 = drift)
2. Implement intervention response generation when drift detected
3. Intervention should redirect to PA purpose without being preachy
4. Add intervention telemetry (what triggered it, what response was generated)
5. Test with Thanksgiving conversation - should detect drift and redirect to TELOS
6. Verify intervention appears in conversation with visual indicator
7. Measure intervention effectiveness (does fidelity recover?)
8. Document intervention strategy and response patterns

---

### Phase 7: Integration Testing

**Description**: Test entire fixed BETA with real conversations to verify all components work together correctly.

**Duration Estimate**: 2-3 days

**Dependencies**: Fix PA Display, Drift Detection & Intervention Logic, Remove Fake Displays

**Steps**:

1. Test Case 1: TELOS project conversation (should extract TELOS PA)
2. Test Case 2: Drift to Thanksgiving (should detect and intervene)
3. Test Case 3: Extended 20-turn conversation (PA should lock at turn 11)
4. Test Case 4: Multiple users (each should get unique PA)
5. Verify all displays show real data (no placeholders, no hardcoded values)
6. Check telemetry logs for complete data capture
7. Measure fidelity variation across different conversation types
8. Run Playwright diagnostic script to automate regression testing
9. Document any bugs found and fix before completion

---

### Phase 8: Documentation & Validation

**Description**: Document all fixes made, create testing guide, and validate BETA is ready for adversarial testing.

**Duration Estimate**: 1-2 days

**Dependencies**: Integration Testing

**Steps**:

1. Document all code changes made with file paths and line numbers
2. Create BETA testing guide for manual validation
3. Document PA extraction algorithm and calibration process
4. Document intervention triggering conditions and thresholds
5. Create telemetry interpretation guide
6. Validate BETA meets success criteria (real PA, real fidelity, interventions work)
7. Generate final validation report showing before/after comparison
8. Confirm BETA is ready for adversarial compliance validation (Phase 2 of original plan)

---

## Phase Dependencies

- **Enable Core TELOS Governance** → **Implement Real PA Extraction**: Cannot extract PA if governance loop is not executing
- **Implement Real PA Extraction** → **Calculate Real Fidelity Scores**: Fidelity calculation requires real PA centroid, not hardcoded generic PA
- **Implement Real PA Extraction** → **Fix PA Display**: Cannot display real PA until it is actually being extracted
- **Calculate Real Fidelity Scores** → **Drift Detection & Intervention Logic**: Drift detection depends on real fidelity scores, not hardcoded values
- **Enable Core TELOS Governance** → **Drift Detection & Intervention Logic**: Interventions must be enabled for drift logic to function
- **Fix PA Display** → **Integration Testing**: All displays must show real data before integration testing
- **Drift Detection & Intervention Logic** → **Integration Testing**: Core functionality must work before comprehensive testing
- **Remove Fake Displays** → **Integration Testing**: UI must be clean before testing user experience
- **Integration Testing** → **Documentation & Validation**: Cannot document until all fixes are tested and validated

---

## Risk Analysis

### HIGH: PA extraction may not capture user intent accurately from first 10 turns

**Mitigation**: Implement conservative extraction with human review option. Start with explicit intent statements. Add refinement mechanism if PA seems off-target.

### MEDIUM: Fidelity calculation may be too sensitive or not sensitive enough

**Mitigation**: Test with diverse conversations to calibrate thresholds. Make intervention threshold configurable. Log all fidelity scores for analysis.

### MEDIUM: Interventions may feel preachy or disrupt conversation flow

**Mitigation**: Design intervention responses to be gentle redirects, not lectures. Test user experience with multiple intervention styles. Use natural language transitions.

### LOW: Enabling interventions may cause unexpected side effects in existing code

**Mitigation**: Enable interventions incrementally with extensive logging. Test in isolated environment first. Have rollback plan ready.

### MEDIUM: Timeline may extend if PA extraction algorithm is more complex than expected

**Mitigation**: Start with simple semantic clustering approach. Can iterate and improve after basic version works. Set decision point at Day 3 of Phase 2.

### MEDIUM: Testing may reveal additional broken components not yet discovered

**Mitigation**: Phase 7 integration testing is designed to catch this. Budget extra time for unexpected fixes. Prioritize critical path items first.

---

## Success Criteria

- **Interventions are enabled and firing when drift detected**
  - Measurement: Thanksgiving conversation test: AI detects drift (fidelity < 0.75) and redirects to TELOS topic. Telemetry shows intervention triggered and logged.
- **Real PA extracted from user conversations**
  - Measurement: TELOS project conversation: PA Purpose includes "AI governance" or "TELOS project". PA is unique per user, not generic hardcoded text.
- **Real fidelity scores calculated and displayed**
  - Measurement: Fidelity varies across conversation (not stuck at 0.850). Drift conversation shows fidelity drop. Display shows actual calculated values from turn data.
- **PA display shows real extracted data**
  - Measurement: PA section displays actual Purpose/Scope/Boundaries from conversation analysis, not placeholder text like "Establish conversation purpose".
- **Fake displays removed**
  - Measurement: Counterfactual display completely removed from BETA. Observatory section grayed out. No demo/placeholder data visible to users.
- **PA locks at turn 11 and remains stable**
  - Measurement: 20-turn conversation: PA changes during turns 1-10, locks at turn 11, does not change afterward. Calibration status indicator updates correctly.
- **Complete telemetry capture for all TELOS operations**
  - Measurement: Every turn has logged: PA state, fidelity score, intervention decision, drift detection. No None values in critical fields. Logs are parseable and complete.
- **BETA ready for adversarial testing**
  - Measurement: All 7 success criteria above met. Manual testing confirms TELOS governance is functioning. Stakeholder approval to proceed with adversarial validation.

---

## Next Steps

1. Review this plan with stakeholders
2. Validate dependencies and timeline
3. Begin Phase 1 implementation
4. Track progress against success criteria

---

*Generated on 2025-11-09 17:04:11*
