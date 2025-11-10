#!/usr/bin/env python3
"""
Planning document for Observatory Unified Mode Architecture Refactor.
Uses StructuredPlanner to formally document the architectural changes.
"""

from structured_planning import StructuredPlanner


def main():
    planner = StructuredPlanner(
        "Observatory Unified Mode Architecture",
        output_dir="./planning_output"
    )

    # Phase 1: Analysis & Planning
    planner.add_phase(
        name="Architecture Analysis",
        description="Analyze current duplicated mode rendering and plan unified approach",
        steps=[
            "Document current architecture: 4 duplicated if/elif blocks (DEMO/BETA/TELOS/DEVOPS)",
            "Identify redundant code: ~76 lines doing nearly identical component calls",
            "Define unified rendering function with mode-based feature flags",
            "Document component-level conditional logic (pa_converged gates)",
            "Run structured_planning.py to generate formal planning artifacts"
        ],
        duration_estimate="Complete (already analyzed)"
    )

    # Phase 2: Refactoring Implementation
    planner.add_phase(
        name="Unified Rendering Implementation",
        description="Create single master rendering function with feature flags",
        steps=[
            "Create render_mode_content(mode: str) function in main.py",
            "Define mode-specific feature flags: show_observation_deck, show_teloscope, show_devops_header",
            "Set telos_demo_mode flag based on current mode",
            "Replace duplicated if/elif blocks with single function call",
            "Maintain component-level conditional logic (components check pa_converged internally)"
        ],
        duration_estimate="Complete (already implemented)",
        dependencies=["Architecture Analysis"]
    )

    # Phase 3: Validation & Testing
    planner.add_phase(
        name="Mode Validation & Testing",
        description="Test all modes to ensure unified rendering works correctly",
        steps=[
            "Start Streamlit application",
            "Test DEMO mode: verify demo PA loaded, no Observation Deck, no TELOSCOPE",
            "Test BETA mode: verify progressive PA extraction, Observation Deck available",
            "Test TELOS mode: verify full Observatory with TELOSCOPE controls",
            "Test DEVOPS mode: verify debug header, full access, progressive PA",
            "Test PA convergence gates: verify UI shows 'Calibrating...' until ~10 turns",
            "Test PA established state: verify metrics appear after convergence"
        ],
        duration_estimate="30 minutes",
        dependencies=["Unified Rendering Implementation"]
    )

    # Phase 4: Documentation
    planner.add_phase(
        name="Documentation & Code Comments",
        description="Document the unified architecture pattern for maintainability",
        steps=[
            "Add docstring to render_mode_content() explaining feature flags",
            "Add inline comments explaining mode-specific behaviors",
            "Document the principle: 'Components contain conditional logic, modes control features'",
            "Update any relevant README or architecture documentation"
        ],
        duration_estimate="15 minutes",
        dependencies=["Mode Validation & Testing"]
    )

    # Phase 5: Git Commit & Push
    planner.add_phase(
        name="Version Control",
        description="Commit changes and sync across repositories",
        steps=[
            "Review all changes with git status and git diff",
            "Create commit: 'Refactor: Unify mode rendering with feature flags'",
            "Push to TELOS-Observatory repo (https://github.com/TelosSteward/TELOS-Observatory)",
            "Push to TELOS main repo (https://github.com/TelosSteward/TELOS)",
            "Verify both remotes updated successfully"
        ],
        duration_estimate="10 minutes",
        dependencies=["Documentation & Code Comments"]
    )

    # Dependencies
    planner.add_dependency(
        from_phase="Architecture Analysis",
        to_phase="Unified Rendering Implementation",
        reason="Must understand current architecture before refactoring"
    )

    planner.add_dependency(
        from_phase="Unified Rendering Implementation",
        to_phase="Mode Validation & Testing",
        reason="Must implement unified rendering before testing it"
    )

    # Risks
    planner.add_risk(
        risk="Feature flags might not cover all mode-specific behaviors",
        mitigation="Test each mode thoroughly to ensure parity with original behavior",
        severity="low"
    )

    planner.add_risk(
        risk="PA convergence gates might break in edge cases",
        mitigation="Component-level checks remain intact; mode layer only controls visibility",
        severity="low"
    )

    planner.add_risk(
        risk="Code duplication might reappear if pattern not documented",
        mitigation="Add clear docstrings and comments explaining the unified architecture principle",
        severity="medium"
    )

    # Success Criteria
    planner.add_success_criterion(
        criterion="Single unified rendering function used by all modes",
        measurement="Verify render_mode_content() called by both Steward panel and full-width layouts"
    )

    planner.add_success_criterion(
        criterion="All modes maintain correct behavior",
        measurement="Manual testing confirms DEMO/BETA/TELOS/DEVOPS all work as before refactor"
    )

    planner.add_success_criterion(
        criterion="Code reduction achieved",
        measurement="Reduced from ~76 duplicated lines to ~32 unified lines (58% reduction)"
    )

    planner.add_success_criterion(
        criterion="PA convergence gates work correctly",
        measurement="UI shows 'Calibrating...' until ~10 turns, then displays metrics"
    )

    planner.add_success_criterion(
        criterion="Components remain modular",
        measurement="Components contain conditional logic; modes just control feature visibility"
    )

    # Generate outputs
    planner.print_summary()
    planner.save_plan()
    planner.generate_markdown_report()


if __name__ == "__main__":
    main()
