# TELOS Documentation

Comprehensive documentation for the TELOS dual PA architecture and validation research.

## Core Documentation

### Whitepaper

**Current Version**: [TELOS_Whitepaper.md](TELOS_Whitepaper.md) (v2.2)
- Complete technical specification of dual PA architecture
- Mathematical formulation of governance mechanisms
- Validation methodology and results
- Philosophical foundations of purpose alignment

**Version History**:
- [TELOS_Whitepaper_v2.2.md](TELOS_Whitepaper_v2.2.md) - Current (November 2024)
- [archive/TELOS_Whitepaper_v2.1.md](archive/TELOS_Whitepaper_v2.1.md) - Previous version

**Formats Available**:
- Markdown: `TELOS_Whitepaper_v2.2.md`
- Microsoft Word: `TELOS_Whitepaper_v2.2.docx`

### Active Documentation

**Technical Specifications**:
- [PERSISTENT_PRIMACY_ATTRACTOR.md](PERSISTENT_PRIMACY_ATTRACTOR.md) - Deep dive into PA architecture and persistence mechanisms

**Deployment Guides**:
- [STREAMLIT_CLOUD_DEPLOYMENT.md](STREAMLIT_CLOUD_DEPLOYMENT.md) - Observatory v3 deployment to Streamlit Cloud
- [REPO_MIGRATION_PLAN.md](REPO_MIGRATION_PLAN.md) - Plan for dual-repository split (telos-purpose + telos-privacy)

**Development Notes**:
- [WHITEPAPER_UPDATE_NOTES.md](WHITEPAPER_UPDATE_NOTES.md) - Changes and improvements in v2.2

## Quick Reference

### Key Concepts

**Dual Primacy Attractor (PA) Architecture**:
- **User PA**: Governs conversation intent (WHAT we're trying to accomplish)
- **AI PA**: Governs communication style (HOW we communicate)
- **Emergent Stability**: Two attractors create self-regulating alignment

**Minimum Bayes-Like (MBL) Intervention**:
- Minimal correction mechanism when drift occurs
- Formula: δ_MBL = α · (â - x_t)
- Only activates when fidelity drops below threshold

**Fidelity Metrics**:
- **User Fidelity** (f_u): Alignment between user responses and user PA
- **AI Fidelity** (f_a): Alignment between AI responses and AI PA
- **Correlation** (ρ): Cross-alignment between PAs

### Validation Results Summary

**Dual PA vs Single PA** (46 sessions):
- User Fidelity: 0.6744 vs 0.3639 → **+85.32% improvement**
- AI Fidelity: 0.7939 vs 0.4154 → **+91.09% improvement**
- Correlation: 0.9168 vs 0.4970 → **+84.47% improvement**
- Statistical Significance: p < 0.001, Cohen's d = 0.87

**Claude Scenario** (counterfactual validation):
- Perfect 1.0000 across all metrics
- Zero interventions needed
- Demonstrates dual PA prevents drift from conversation start

See [../DUAL_PA_VALIDATION_SUMMARY.md](../DUAL_PA_VALIDATION_SUMMARY.md) for complete analysis.

## Archive

Historical documentation and research artifacts preserved for reference:

### Archived Documentation (`archive/`)

**Completed Build Documentation**:
- `BUILD_NOTES_v1.1.md` - Version 1.1 development notes
- `DEPLOYMENT.md` - Early deployment planning
- `GITHUB_CLEANUP_PLAN.md` - Repository organization planning
- `GITHUB_READY.md` - Pre-publication checklist
- `ORGANIZATION_COMPLETE.md` - File organization completion notes
- `TASKS.md` - Historical task tracking
- `REPO_MANIFEST.md` - Original repository structure documentation

### Research Archive (`archive/research/`)

**Architecture Research**:
- `DUAL_ATTRACTOR_ARCHITECTURE.md` - Original dual PA architecture design
- `DEMO_MODE_TWO_LAYER_IMPLEMENTATION.md` - Demo mode technical design
- `GOVERNANCE_MODES.md` - Governance mode specifications

**Observatory Development**:
- `OBSERVATORY_ADVANCED_FEATURES.md` - Advanced feature planning for Observatory v3
- `WIRING_PLAN.md` - Component integration planning

**Discovery Documentation**:
- `PRODUCT_DISCOVERY_TRANSCRIPT.md` - Early product discovery sessions

## Repository Structure

```
telos/
├── docs/                          # This directory
│   ├── TELOS_Whitepaper.md       # Canonical whitepaper
│   ├── TELOS_Whitepaper_v2.2.md  # Current version
│   ├── TELOS_Whitepaper_v2.2.docx
│   ├── PERSISTENT_PRIMACY_ATTRACTOR.md
│   ├── STREAMLIT_CLOUD_DEPLOYMENT.md
│   ├── REPO_MIGRATION_PLAN.md
│   ├── WHITEPAPER_UPDATE_NOTES.md
│   └── archive/                   # Historical docs
├── validation/                    # Validation studies
│   ├── briefs/                   # Research briefs
│   ├── results/                  # Raw results
│   └── scripts/                  # Validation scripts
├── DUAL_PA_VALIDATION_SUMMARY.md # Main validation summary
├── DEPLOYMENT_ROADMAP.md         # Production deployment plan
└── DUAL_DROP_STRATEGY.md         # Dual-repo strategy
```

## Related Resources

### Root Documentation

High-visibility strategic documents kept in repository root:

**Validation**:
- [../DUAL_PA_VALIDATION_SUMMARY.md](../DUAL_PA_VALIDATION_SUMMARY.md) - Complete validation analysis

**Deployment Strategy**:
- [../DEPLOYMENT_ROADMAP.md](../DEPLOYMENT_ROADMAP.md) - 6-week production deployment plan
- [../DUAL_DROP_STRATEGY.md](../DUAL_DROP_STRATEGY.md) - Dual-repository launch strategy

**Organization**:
- [../FILE_ORGANIZATION_PLAN.md](../FILE_ORGANIZATION_PLAN.md) - Repository organization plan (this document's genesis)

### Validation Data

See [../validation/README.md](../validation/README.md) for:
- 46 research briefs analyzing individual sessions
- Raw validation results (JSON)
- Validation and analysis scripts
- Methodology details

### Code

**Core Implementation**:
- `dual_pa_engine.py` - Main dual PA governance engine
- `unified_orchestrator_steward.py` - Orchestrator with intervention logic
- `telos_observatory_v3/` - Interactive visualization and testing platform

## Contributing

For development documentation and contribution guidelines, see repository root.

## Citation

```bibtex
@misc{telos2024whitepaper,
  title={TELOS: Dual Primacy Attractor Architecture for AI Purpose Alignment},
  author={TELOS Research},
  year={2024},
  version={2.2},
  note={Whitepaper with validation results: +85.32\% improvement over single PA}
}
```

## Version History

### v2.2 (November 2024)
- Added comprehensive dual PA validation results
- Enhanced MBL intervention explanation
- Refined philosophical foundations
- Improved technical specifications

### v2.1 (Previous)
- Initial dual PA architecture documentation
- Basic validation methodology
- Core concept introduction

## License

See repository root for license information.

---

**Last Updated**: November 2024
**Status**: Active Development
**Next Milestone**: Production deployment (Telegram, Streamlit, Discord)
