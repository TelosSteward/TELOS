# Changelog

All notable changes to TELOS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open source infrastructure (CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md)
- GitHub Actions CI/CD workflow

## [3.0.0] - 2025-12-22

### Added
- **Expanded Adversarial Validation**: 1,300 attacks tested (up from 54)
  - HarmBench: 400 standard + 400 contextual behaviors
  - Med-Safety-Bench: 500 medical safety scenarios
  - Attack Success Rate: 0.0% across all benchmarks
- **Governance Observability System**: Full forensic trace logging
  - 11 event types (SessionStart, FidelityCalculated, InterventionTriggered, etc.)
  - Privacy modes: FULL, HASHED, DELTAS_ONLY
  - JSONL export for audit trails
- **Pytest Test Suite**: Unit tests for core mathematical components
- **PBC Governance Documentation**: Delaware Public Benefit Corporation framework

### Changed
- Technical Paper rewritten to v3.0
- Updated validation claims to 1,300 attacks

## [2.3.0] - 2025-12-15

### Added
- Beta testing framework with consent flows
- Session summarization with AI-powered insights
- HTML governance report generation

## [2.0.0] - 2025-12-01

### Added
- TELOSCOPE Observatory Interface
- Primacy Attractor Templates
- Proportional Controller (K=1.5)

## [1.0.0] - 2025-11-15

### Added
- Initial TELOS framework implementation
- Core fidelity mathematics
- Constitutional Filter
