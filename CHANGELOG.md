# Changelog

All notable changes to TELOS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SECURITY.md for vulnerability reporting policy
- CHANGELOG.md for version tracking

### Changed
- Updated documentation to reference Zenodo validation data
- Fixed broken import paths in documentation

### Removed
- Unused security/forensics directory
- Broken CI workflow (security-validation.yml)

## [1.0.0] - 2025-11-24

### Added
- **TELOSCOPE Observatory**: Interactive Streamlit-based governance interface
- **Dual Attractor Model**: PA (Primacy Attractor) and SA (Secondary Attractor) dynamics
- **Demo Mode**: 12-slide guided demonstration of TELOS principles
- **BETA Mode**: Full governance establishment for real-world testing
- **Validation Framework**: 1,300 adversarial attack validation suite
  - MedSafetyBench (900 attacks)
  - HarmBench (400 attacks)
- **Statistical Analysis**: Wilson Score CI, Bayesian analysis, power analysis
- **LLM Integration**: Mistral API and Ollama local support
- **Telemetric Keys**: Cryptographic verification of governance decisions

### Security
- 0% Attack Success Rate across 1,300 adversarial attacks
- 99.9% confidence interval [0%, 0.28%]
- Bayes Factor: 2.7 x 10^17

### Documentation
- Technical Paper (TELOS_Technical_Paper.md)
- Statistical Validity whitepaper
- Telemetric Keys Foundations
- Reproduction Guide
- Hardware Requirements

## [0.9.0] - 2025-10-27 (Pre-release)

### Added
- Initial TELOSCOPE prototype
- Basic PA establishment flow
- Conversation interface with governance overlay
- Initial validation against MedSafetyBench subset

### Changed
- Refactored governance engine for performance
- Improved embedding-based similarity matching

## [0.5.0] - 2025-10-15 (Alpha)

### Added
- Core dual attractor architecture
- Unified Steward controller
- Basic Streamlit interface
- Initial documentation

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2025-11-24 | Production release, full validation, 0% ASR |
| 0.9.0 | 2025-10-27 | Pre-release with TELOSCOPE prototype |
| 0.5.0 | 2025-10-15 | Alpha with core architecture |

---

**Maintained by**: Jeffrey Brunner
**License**: MIT
