# Contributing to TELOS

Thank you for your interest in contributing to TELOS! This project aims to advance AI governance through Statistical Process Control and quantum-resistant cryptography.

## How to Contribute

### Reporting Security Issues

**IMPORTANT**: If you discover a security vulnerability, please DO NOT file a public issue. Instead, email security@telos-project.org with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Clear description
   - Reproduction steps
   - Environment details
   - Expected vs actual behavior

### Suggesting Features

1. Use the feature request template
2. Explain how it improves:
   - SPC calibration
   - Telemetric Keys security
   - System observability
   - Performance metrics

### Code Contributions

#### Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/telos-privacy.git
cd telos-privacy

# Create a feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt
```

#### Code Standards

- **Python**: Follow PEP 8
- **Documentation**: Update docstrings and README
- **Tests**: Maintain 100% critical path coverage
- **Security**: No hardcoded credentials or keys
- **SPC**: Document any changes to control limits or metrics

#### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run security validation
python security/forensics/DETAILED_ANALYSIS/strix_attack_with_fallback.py

# Check SPC calibration
python telos/core/spc_validation.py
```

#### Commit Messages

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Testing
- `perf:` Performance
- `security:` Security enhancement
- `spc:` SPC calibration changes

Example:
```
feat: Add EWMA drift detection to SPC module

- Implements exponentially weighted moving average
- Calibrates detection threshold at 2.5σ
- Reduces false positive rate by 15%
```

### Pull Request Process

1. **Branch**: Create from `main`
2. **Test**: Ensure all tests pass
3. **Document**: Update relevant docs
4. **Review**: Request review from maintainers
5. **Sign**: Sign the CLA if required

### Review Criteria

PRs are evaluated on:
- **Security Impact**: No vulnerabilities introduced
- **SPC Compliance**: Maintains calibration standards
- **Performance**: No degradation in response times
- **Documentation**: Clear and complete
- **Tests**: Comprehensive coverage

## Development Guidelines

### SPC Calibration

When modifying SPC parameters:
1. Document baseline metrics
2. Run sensitivity analysis
3. Validate with 1000+ test cases
4. Update control charts
5. Verify Cpk > 1.33

### Telemetric Keys

Cryptographic changes require:
1. Security analysis document
2. Quantum resistance verification
3. Performance benchmarks
4. Backward compatibility plan

### Architecture Principles

- **Defense in Depth**: Three-tier architecture
- **Statistical Rigor**: Evidence-based decisions
- **Transparency**: Observable and auditable
- **Performance**: <10ms overhead target
- **Reliability**: 99.9% uptime goal

## Community

### Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/). Please:
- Be respectful and inclusive
- Focus on constructive criticism
- Help others learn and grow
- Prioritize project goals

### Getting Help

- **Documentation**: Start with `/docs`
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions
- **Email**: research@telos-project.org

### Recognition

Contributors are recognized in:
- Release notes
- Contributors file
- Academic publications (with permission)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to reach out:
- GitHub Issues for public questions
- research@telos-project.org for research collaboration
- security@telos-project.org for security matters

Thank you for helping make AI governance mathematically enforceable!