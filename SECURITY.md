# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in TELOS, please report it responsibly:

1. **Do NOT open a public GitHub issue** for security vulnerabilities
2. **GitHub Private Vulnerability Reporting**: Use GitHub's private vulnerability reporting feature if available

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Target**: Within 30 days for critical issues

## Security Validation Results

TELOS has been validated against **1,300 adversarial attacks** with the following results:

| Metric | Result |
|--------|--------|
| Attack Success Rate | **0%** |
| Statistical Confidence | 99.9% CI [0%, 0.28%] |
| Bayes Factor | 2.7 x 10^17 |
| Power Analysis | > 0.99 |

### Benchmarks Used

- **MedSafetyBench** (900 attacks): NeurIPS 2024 healthcare safety benchmark
- **HarmBench** (400 attacks): CAIS general adversarial benchmark

### Validation Data

Full validation data is publicly available:
- **Zenodo**: https://doi.org/10.5281/zenodo.17702890
- **Reproduction Guide**: See `REPRODUCTION_GUIDE.md`

## Security Architecture

TELOS implements a dual-attractor governance model:

1. **Primacy Attractor (PA)**: User-established purpose, scope, and boundaries
2. **Secondary Attractor (SA)**: Dynamic contextual adaptation within PA constraints
3. **Telemetric Validation**: Cryptographic verification of governance decisions

### Key Security Properties

- **No prompt injection bypass**: PA boundaries cannot be overridden by user input
- **Continuous governance**: Every response is validated against established PA
- **Transparent operation**: All governance decisions are logged and auditable
- **Environment variable security**: No hardcoded API keys or secrets

## Responsible Disclosure

We are committed to working with security researchers to improve TELOS. Researchers who:

1. Follow responsible disclosure practices
2. Give us reasonable time to address issues
3. Avoid accessing or modifying user data

Will be acknowledged in our security advisories (unless they prefer anonymity).

## Contact

- **GitHub Issues**: https://github.com/TelosSteward/TELOS/issues (non-security issues only)
- **Security Contact**: [To be established]

---

**Last Updated**: December 2025
**Policy Version**: 1.0
