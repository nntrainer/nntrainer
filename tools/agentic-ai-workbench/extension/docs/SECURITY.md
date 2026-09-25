# Security Documentation - AI Compiler Workbench

This document provides comprehensive security information for the AI Compiler Workbench (Agentic Graph Visualizer) VS Code extension.

---

## 🛡️ Security Status

| Phase | Status | Date |
|-------|--------|------|
| Phase 1 (CRITICAL) | ✅ **COMPLETE** | 2026-08-24 |
| Phase 2 (HIGH) | 🔄 In Progress | - |
| Phase 3 (MEDIUM) | ⏳ Pending | - |
| Phase 4 (LOW) | ⏳ Pending | - |

---

## 📋 Security Fixes Implemented

### Phase 1: CRITICAL Fixes ✅

#### Fix #1: API Key Exposure → SecretStorage
- **Issue**: API keys passed via environment variables
- **Fix**: API keys now retrieved from VS Code SecretStorage
- **Location**: `extension.js` line 209
- **Backward Compatible**: Falls back to config if SecretStorage empty

#### Fix #2: Command Injection via Model Name
- **Issue**: User model names passed directly to subprocess
- **Fix**: Model names validated against allowlist regex
- **Functions**: `validateModelName()`, `validatePath()`
- **Location**: `extension.js` lines 17-44, 154-160

#### Fix #4: Unsafe subprocess.spawn
- **Issue**: subprocess calls without shell: false
- **Fix**: All subprocess.spawn now use `shell: false`, sanitized env
- **Function**: `sanitizeEnv()`
- **Location**: `extension.js` lines 46-53, 227-232

---

## 🔒 Security Architecture

### Defense in Depth

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: VS Code Extension (extension.js)                  │
│  - Model name validation (regex + shell meta check)         │
│  - Path validation (canonicalization + allowlist)           │
│  - Environment sanitization (remove sensitive vars)         │
│  - subprocess with shell: false                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Python Orchestrator (orchestrator_main.py)        │
│  - argparse with typed arguments                            │
│  - Path validation                                          │
│  - Environment variable validation                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Python Agents (agents/*.py)                       │
│  - Input validation at agent boundary                       │
│  - subprocess with shell: false                             │
│  - Error handling with fail-secure defaults                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Security-Relevant Files

| File | Security Concerns | Mitigations |
|------|-------------------|-------------|
| `extension.js` | Command injection, API key exposure | Input validation, SecretStorage, shell: false |
| `orchestrator_main.py` | Argument injection | argparse, typed arguments |
| `docker_builder.py` | Docker socket access | Path validation, volume restrictions |
| `auto_fix.py` | LLM prompt injection | Prompt sanitization (TODO) |
| `chat_agent.py` | LLM prompt injection | Prompt sanitization (TODO) |
| `main.html` | XSS via webview | CSP with nonce, no unsafe-eval |

---

## 🔐 Secret Management

### Storing API Keys

```bash
# In VS Code, use the command palette:
# 1. Ctrl+Shift+P → "Store Secret"
# 2. Enter key name: "anthropicApiKey"
# 3. Enter your API key value
```

### Programmatic Access

```typescript
// extension.js
const apiKey = await context.secrets.get("anthropicApiKey");
```

---

## 🚨 Known Vulnerabilities (Remaining)

| ID | Severity | Issue | Status |
|----|----------|-------|--------|
| SEC-005 | HIGH | Docker socket access restrictions | TODO |
| SEC-006 | HIGH | File path input validation (partial) | Partial |
| SEC-007 | HIGH | Secure temporary file handling | TODO |
| SEC-008 | HIGH | LLM prompt injection prevention | TODO |
| SEC-010 | MEDIUM | Rate limiting on LLM calls | TODO |
| SEC-011 | MEDIUM | Structured error handling | TODO |
| SEC-012 | MEDIUM | Webview CSP hardening | TODO |

---

## 🧪 Security Testing

### Manual Testing Checklist

- [ ] Model name validation rejects shell metacharacters
- [ ] Path validation rejects path traversal attempts
- [ ] API keys not visible in process listings
- [ ] subprocess.spawn uses shell: false
- [ ] Environment sanitized before subprocess

### Automated Tests

```bash
# Run security unit tests (TODO: create tests)
npm test -- --grep "security"
```

---

## 📦 Dependency Security

### Current Status

| Ecosystem | Tool | Status |
|-----------|------|--------|
| npm | `npm audit` | ⚠️ 8 vulnerabilities (6 moderate, 2 high) |
| pip | `pip-audit` | ⏳ Pending |

### Recommended Actions

1. Run `npm audit fix` regularly
2. Pin dependency versions in `package.json` and `requirements.txt`
3. Consider using Dependabot for automated updates

---

## 🏗️ Secure Development Guidelines

### For Contributors

1. **Never** use `shell: true` with user input
2. **Always** validate external input (model names, paths)
3. **Never** log API keys or secrets
4. **Always** use SecretStorage for sensitive data
5. **Fail securely** - errors should not expose sensitive information

### Code Review Checklist

- [ ] Input validation at trust boundaries
- [ ] No shell:true in subprocess calls
- [ ] Secrets from SecretStorage, not config
- [ ] Error messages don't leak sensitive data
- [ ] Paths validated and canonicalized

---

## 📞 Security Contact

For security issues or vulnerabilities, please report to:
- **Email**: [security-contact@example.com](mailto:security-contact@example.com)
- **GitHub**: Use private vulnerability reporting feature

---

## 📅 Security Review Schedule

| Activity | Frequency | Last Review |
|----------|-----------|-------------|
| Dependency audit | Weekly | - |
| Security code review | Monthly | 2026-08-24 |
| Penetration testing | Quarterly | - |
| Threat model update | Annually | - |

---

## 🔗 References

- [VS Code Extension Security](https://code.visualstudio.com/api/working-with-extensions/security)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Node.js Security Best Practices](https://nodejs.org/en/docs/guides/security/)
- [Python Security Best Practices](https://docs.python.org/3/library/security.html)

---

**Last Updated**: 2026-08-24  
**Document Version**: 1.0  
**Next Review**: 2026-09-24
