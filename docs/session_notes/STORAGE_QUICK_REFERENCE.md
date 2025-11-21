# TELOS Observatory V3: Storage Architecture - Quick Reference

## One-Line Summary
**Local JSON file storage only. No databases, no cloud. In-memory only until explicitly saved.**

## Storage Locations

```
/Users/brunnerjf/Desktop/telos_privacy/telos_observatory_v3/
├── saved_sessions/
│   ├── session_index.json (registry, 45 sessions currently)
│   └── *.json (individual session files with full conversations)
└── beta_consents/
    └── consent_log.json (audit trail, 45 consents recorded)
```

## What Gets Saved (Automatic)
- ✓ **Consent records** → `beta_consents/consent_log.json` (immediately on consent)
- Session data → `saved_sessions/*.json` (ONLY when user clicks "Save Current")

## What Gets Saved (User-Initiated)
- Entire conversation history (user messages + AI responses)
- TELOS metrics (fidelity, distance, interventions)
- Session metadata (ID, mode, timestamps)
- Primacy Attractor configuration

## What Does NOT Get Saved Automatically
- Session state is RAM-only (st.session_state)
- Lost on page refresh, browser close, or app restart
- No automatic persistence between sessions
- No automatic sync to remote servers

## Session Lifecycle

| Event | Storage | Status |
|-------|---------|--------|
| User arrives | None | Empty session created in RAM |
| User consents | consent_log.json | Recorded automatically |
| User types message | RAM only | Lost if page refreshes |
| User clicks "Save" | saved_sessions/*.json | Persisted to disk |
| Page refreshes | None | All RAM data lost |
| User clicks "Load" | JSON read from disk | Restored to RAM |

## File Formats

### Session File Structure
```json
{
  "session_id": "session_1234567890",
  "timestamp": "ISO timestamp",
  "mode": "demo|open",
  "total_turns": N,
  "avg_fidelity": 0.82,
  "turns": [
    {
      "turn": 1,
      "user_input": "User message",
      "response": "AI response",
      "fidelity": 0.85,
      "distance": 0.15,
      "intervention_applied": false,
      "drift_detected": false
    }
  ],
  "primacy_attractor": { ... },
  "metadata": { ... }
}
```

### Consent Record Structure
```json
{
  "session_id": "session_1762316828",
  "timestamp": "2025-11-04T23:27:14.333998",
  "consent_statement": "...",
  "version": "2.1"
}
```

## Critical Issues for Compliance

| Issue | Severity | Status |
|-------|----------|--------|
| Consent text says "stored on our servers" but means "stored locally" | CRITICAL | NEEDS FIX |
| No encryption for JSON files | HIGH | NOT IMPLEMENTED |
| No data deletion mechanism | HIGH | NOT IMPLEMENTED |
| No data retention policy | HIGH | NOT DOCUMENTED |
| Evidence export shows message but doesn't download | MEDIUM | NOT FUNCTIONAL |

## Key Code Locations

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Session saving | `sidebar_actions.py` | 145-195 | `_save_current_session()` |
| Session loading | `sidebar_actions.py` | 220-244 | `_load_session_by_id()` |
| Consent logging | `beta_onboarding.py` | 23-59 | `_log_consent()` |
| State management | `core/state_manager.py` | 20-120 | `StateManager` class |
| Main session init | `main.py` | 28-54 | `initialize_session()` |

## Database/API Integrations

| System | Status | Data Flow |
|--------|--------|-----------|
| Supabase | NOT INTEGRATED | - |
| Firebase | NOT INTEGRATED | - |
| PostgreSQL | NOT INTEGRATED | - |
| Mistral LLM API | INTEGRATED | Conversation → Mistral → Response |
| Claude API | IN REQUIREMENTS | Not actively used |
| Local Embeddings | LOCAL ONLY | No upload |

## Data Persistence Flow

```
Session Start
    ↓
In-Memory State (RAM only via st.session_state)
    ↓
User Types Messages
    ↓
Conversation Data in Memory
    ↓
User Saves → JSON to Disk
    ↓
Page Refresh → RAM Data Lost
    ↓
User Loads → JSON from Disk → Back to Memory
```

## Privacy Safeguards

| Safeguard | Present | Notes |
|-----------|---------|-------|
| Encryption at rest | NO | Plain-text JSON |
| Access control | NO | Filesystem permissions only |
| Data retention policy | NO | Indefinite storage |
| Automatic cleanup | NO | Manual deletion required |
| Data export tool | PARTIAL | Tool exists but doesn't work |

## For Regulatory Audits

**Store Location**: Local filesystem only  
**Encryption**: None  
**Retention**: Indefinite (until manual delete)  
**Access Logs**: None  
**Data Classification**: Mixed (conversations + metrics stored together)  
**Consent Records**: Yes (maintained in consent_log.json)  
**User Rights (GDPR)**:
- Access: Not implemented (must read JSON manually)
- Deletion: Not implemented (must delete files manually)  
- Export: Partially implemented (shows message, no file)

## Current Risk Assessment

**DEPLOYMENT CONTEXT MATTERS:**

### Self-Hosted (Local Machine)
- Risk: LOW (only accessible to machine owner)
- Suitable for: Research, development, local testing
- Compliance: Acceptable with updated consent language

### Cloud-Hosted (Streamlit Cloud, AWS, etc.)
- Risk: HIGH (data on third-party servers)
- Suitable for: NOT RECOMMENDED in current state
- Compliance: Requires encryption, DPA, explicit data sharing terms

## Immediate Actions Required

1. **Update Consent Language** - Change "stored on our servers" to "stored locally on your device"
2. **Fix Evidence Export** - Actually download file instead of just showing success message
3. **Add Encryption** - Use cryptography library to encrypt JSON files
4. **Add Deletion UI** - Provide "Delete Session" buttons with confirmation
5. **Document Retention** - Create PRIVACY.md explaining data lifecycle

## Testing Persistence

```bash
# Check what's actually saved
ls -la telos_observatory_v3/saved_sessions/
cat telos_observatory_v3/beta_consents/consent_log.json

# Verify no remote uploads
grep -r "http\|request\|upload\|cloud" telos_observatory_v3/ --include="*.py"
# (Only Mistral API calls, not session storage)
```

---

**Date**: November 14, 2025  
**Status**: Thoroughly Analyzed  
**Report**: See TELOS_STORAGE_ARCHITECTURE.md for full details
