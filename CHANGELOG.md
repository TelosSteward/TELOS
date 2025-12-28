# Changelog

All notable changes to TELOS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open source infrastructure (CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md)
- GitHub Actions CI/CD workflow

## [3.1.0] - 2025-12-27

### Added
- **Steward Context Awareness**: Full session visibility for Steward explanations
  - AI responses added to turn history (previously only user input)
  - Session summary statistics (total turns, average fidelity, intervention count)
  - TELOS_IMPLEMENTATION.md RAG corpus with exact implementation details
- **Adaptive Context System (v3.1)**: Intelligent follow-up message handling
  - Message type classification (ANAPHORA, CLARIFICATION, FOLLOW_UP, DIRECT)
  - Context-aware fidelity boosting with type-specific multipliers
  - Multi-tier context buffer with phase detection
- **SB 243 Child Safety Validation Dataset**: California child safety compliance
- **EU AI Act Compliance Documentation**: Positioning document for Article 72

### Changed
- **Refactored beta_response_manager.py**: Decomposed into single-responsibility modules
  - Separated embedding, fidelity calculation, and intervention logic
  - Improved maintainability and testability
- **Architectural cleanup**: Threshold consolidation to single source of truth
- **Fidelity display**: Switched from decimals to percentages for better UX
- **Threshold recalibration**: Relaxed thresholds to reduce false positives
  - Improved accuracy for natural conversation flow

### Performance
- **Cold start optimization**: Reduced Railway startup from 60s to <10s
  - Cached embedding provider instances
  - Pre-downloaded SentenceTransformer models during build
  - Removed MPNet pre-warming overhead
- **Response speed**: Skip AI fidelity calculation in GREEN zone
- **Streaming display**: True token-by-token rendering for better perceived performance

### Fixed
- Conversation history now included in redirect responses for PA context
- Checkbox styling and percentage formatting in UI
- Demo navigation flow improvements
- Railway deployment configuration for Streamlit

### Removed
- Redundant module files (bloat cleanup)

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
