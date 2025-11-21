# Memory MCP Setup for Dual PA + Primacy State Research

**Date:** November 15, 2025
**Purpose:** Track Primacy State formalization as strategic TELOS research initiative

---

## Memory MCP Entity Creation

### Entity: "Dual PA + Primacy State Formalization Research"

**Commands to execute in Claude Code with Memory MCP:**

```
# Create main research entity
mcp__memory__create_entities([{
  "name": "Dual_PA_Primacy_State_Research",
  "entityType": "research_initiative",
  "observations": [
    "Mathematical formalization of Primacy State as derived constant from dual PA dynamics",
    "Status: Exploratory - Feasibility Phase (November 2025)",
    "Motivation: 85.32% improvement from dual PA validation, perfect 1.0000 fidelity achievable",
    "Goal: Determine if PS = ρ_PA · (2·F_user·F_AI)/(F_user+F_AI) provides better governance",
    "Key questions: Computational feasibility, diagnostic value, interpretability improvement"
  ]
}])

# Create relationship to Dual PA Validation
mcp__memory__create_relations([{
  "from": "Dual_PA_Primacy_State_Research",
  "to": "Dual_PA_Validation_Results",
  "relationType": "builds_upon"
}])

# Create relationship to BETA A/B Testing
mcp__memory__create_relations([{
  "from": "Dual_PA_Primacy_State_Research",
  "to": "BETA_AB_Testing",
  "relationType": "potential_integration_target"
}])

# Create relationship to Whitepaper
mcp__memory__create_relations([{
  "from": "Dual_PA_Primacy_State_Research",
  "to": "TELOS_Whitepaper_v2.3",
  "relationType": "requires_update_if_validated"
}])
```

---

## Key Observations to Track

**Add these observations as research progresses:**

### Phase 1: Planning Complete
```
mcp__memory__add_observations({
  "entityName": "Dual_PA_Primacy_State_Research",
  "observations": [
    "Research plan created: DUAL_PA_PRIMACY_STATE_RESEARCH_PLAN.md",
    "Feasibility test designed: test_primacy_state_feasibility.py",
    "Hypotheses structured: 4 testable hypotheses defined",
    "Integration analysis: 8-10 Supabase columns identified",
    "Stakeholder communication prepared",
    "Next: Execute feasibility testing on 46 existing dual PA sessions"
  ]
})
```

###

 Phase 2: Feasibility Results (After Testing)
```
mcp__memory__add_observations({
  "entityName": "Dual_PA_Primacy_State_Research",
  "observations": [
    "Feasibility test executed on 46 dual PA validation sessions",
    "PS computation time: X ms average (acceptable: <50ms)",
    "PS decomposition diagnostics: {F_user, F_AI, ρ_PA} provide actionable info: YES/NO",
    "Hypothesis H1 result: PA correlation predicts stability - SUPPORTED/REJECTED",
    "Hypothesis H2 result: PS enables earlier drift detection - SUPPORTED/REJECTED",
    "Decision: GO/NO-GO on full implementation based on criteria"
  ]
})
```

### Phase 3: Implementation Decision
```
mcp__memory__add_observations({
  "entityName": "Dual_PA_Primacy_State_Research",
  "observations": [
    "Decision: [PROCEED/DEFER/ABANDON]",
    "Rationale: [Based on feasibility results]",
    "Timeline: [If proceeding: 4-6 weeks implementation]",
    "BETA integration: [IMMEDIATE/PARALLEL/DEFERRED/NO]",
    "Whitepaper section: [ADD NOW/ADD AFTER VALIDATION/NO]"
  ]
})
```

---

## Strategic Dependencies

**This research initiative depends on:**
1. ✅ Dual PA validation results (already complete - 85.32% improvement)
2. ✅ Existing codebase with dual PA implementation (dual_attractor.py)
3. ⏳ Feasibility testing execution (planned, not yet run)
4. ⏳ Stakeholder feedback on interpretability (needed for decision)

**This research initiative enables:**
1. 🔮 Stronger mathematical foundation (if validated)
2. 🔮 Better diagnostic capabilities (PS decomposition shows which PA fails)
3. 🔮 Predictive drift detection (ΔV_dual forecasts Primacy State collapse)
4. 🔮 Clearer regulatory evidence ("PS = 0.943" vs vague "aligned")

**This research initiative may block:**
- Nothing currently - kept separate from active BETA A/B testing
- Implementation decision may affect future whitepaper revisions

---

## Query Examples

**Check status:**
```
mcp__memory__search_nodes(query="Primacy State")
mcp__memory__open_nodes(names=["Dual_PA_Primacy_State_Research"])
```

**Check dependencies:**
```
mcp__memory__search_nodes(query="Dual PA validation")
# Look for relations: builds_upon, potential_integration_target
```

**Track progress:**
```
# After each phase, add observations showing progress
# Memory MCP maintains history of all observations
```

---

## Integration with Steward PM

**Query Steward PM about this research:**

```bash
# Full status including this research
python3 steward_pm.py status

# Analyze this specific research initiative
python3 steward_pm.py analyze "Primacy State research"

# Get priority recommendation (where does this fit in overall TELOS roadmap?)
python3 steward_pm.py next
```

**Steward PM will use Memory MCP to:**
- Show how this research relates to grants (does it strengthen LTFF application?)
- Assess impact on February 2026 institutional deployment timeline
- Identify dependencies (does GMU partnership need this? Does EU grant?)
- Recommend priority (vs. other research tasks like validation studies)

---

## Timeline Tracking

**Key Milestones to track in Memory MCP:**

| Milestone | Target Date | Status | Memory Observation |
|-----------|-------------|--------|-------------------|
| Research plan complete | Nov 15, 2025 | ✅ DONE | "Planning phase complete" |
| Feasibility test run | Nov 18-22, 2025 | ⏳ PENDING | "Feasibility executed, results: ..." |
| Go/No-Go decision | Nov 25, 2025 | ⏳ PENDING | "Decision: [GO/NO-GO], rationale: ..." |
| Implementation (if GO) | Dec 2025 - Jan 2026 | 🔮 CONDITIONAL | "Implementation started/complete" |
| Whitepaper update | Jan 2026 | 🔮 CONDITIONAL | "Section 2.3 added to whitepaper" |

---

## Success Criteria (Track in Memory MCP)

**For GO decision, need ALL of these:**
- ✅ PS formula computable on existing data without errors
- ✅ PS decomposition provides actionable diagnostic information
- ✅ Computational overhead acceptable (<50ms added latency)
- ✅ Delta interpreter narratives rated "clear" by 3+ reviewers
- ✅ At least 2 of 4 hypotheses show supporting evidence

**Memory MCP will track:** Which criteria pass/fail, enabling data-driven decision

---

## Related Entities to Create (Optional)

**For more granular tracking:**

```
# Individual hypotheses as entities
mcp__memory__create_entities([
  {
    "name": "PS_Hypothesis_H1_PA_Correlation_Stability",
    "entityType": "research_hypothesis",
    "observations": ["Higher ρ_PA predicts lower ΔV variance"]
  },
  {
    "name": "PS_Hypothesis_H2_Earlier_Detection",
    "entityType": "research_hypothesis",
    "observations": ["PS components fail before overall fidelity"]
  }
])

# Link to parent research
mcp__memory__create_relations([{
  "from": "PS_Hypothesis_H1_PA_Correlation_Stability",
  "to": "Dual_PA_Primacy_State_Research",
  "relationType": "tests"
}])
```

**Benefit:** Can track hypothesis results individually in Memory MCP

---

## How This Integrates with Overall TELOS Strategy

**Memory MCP enables Steward PM to answer:**

**Q: Does Primacy State research strengthen grant applications?**
- A: Query relationship to LTFF/EV grant entities
- Analysis: Does mathematical rigor help academic reviewers?
- Answer tracked in Memory: YES/NO with rationale

**Q: Is this on critical path for February 2026 institutional deployment?**
- A: Query relationship to deployment milestones
- Analysis: Do institutions need PS, or is current dual PA sufficient?
- Answer tracked in Memory: CRITICAL/IMPORTANT/NICE-TO-HAVE

**Q: How does this compare to other research priorities?**
- A: Query all research initiatives, check dependencies
- Analysis: Is feasibility testing higher priority than more validation studies?
- Answer: Steward PM recommendation based on Memory MCP state

---

## Execution Instructions

**To set up Memory MCP tracking:**

1. **In Claude Code session with Memory MCP enabled:**
   ```
   # Copy entity creation commands from above
   # Execute mcp__memory__create_entities()
   # Execute mcp__memory__create_relations()
   # Execute mcp__memory__add_observations() with planning complete status
   ```

2. **Verify setup:**
   ```
   mcp__memory__search_nodes(query="Primacy State")
   # Should return: Dual_PA_Primacy_State_Research entity
   ```

3. **Check Steward PM integration:**
   ```
   python3 steward_pm.py status
   # Should include this research in project overview
   ```

4. **After feasibility testing:**
   ```
   mcp__memory__add_observations({entity: "...", observations: ["Results: ..."]})
   ```

---

**Status:** Ready for Memory MCP setup
**Next:** Create entity, then generate all research planning documents
