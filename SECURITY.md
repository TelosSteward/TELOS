# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 3.x     | :white_check_mark: |
| 2.x     | :x:                |
| 1.x     | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in TELOS, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email security concerns to the project maintainers
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution Timeline**: Depends on severity, typically 30-90 days

### Scope

Security concerns include:
- Vulnerabilities in the TELOS governance framework
- Issues that could compromise user data or privacy
- Bypass of fidelity detection mechanisms
- Authentication or authorization flaws

### Out of Scope

- Vulnerabilities in third-party dependencies (report to upstream)
- Social engineering attacks
- Denial of service attacks
- Issues requiring physical access

## Security Best Practices

When deploying TELOS:

1. **API Keys**: Never commit API keys. Use environment variables.
2. **Privacy Modes**: Use `DELTAS_ONLY` mode for production deployments
3. **Access Control**: Restrict access to governance trace logs
4. **Updates**: Keep dependencies updated

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve TELOS.
