# TELOS Observatory V3: Data Storage & Persistence Architecture Report
**Thoroughness Level: VERY THOROUGH**  
**Date: November 14, 2025**  
**Focus: Critical for Compliance/Consent Language**

---

## EXECUTIVE SUMMARY

The TELOS Observatory V3 uses a **HYBRID LOCAL-FILE + MEMORY-ONLY architecture** with **NO remote database integration**. All session data is persisted to local JSON files on the filesystem. There is NO cloud-based storage, NO Supabase integration, NO database servers—only file system storage and Streamlit's in-memory session state.

### Key Finding
**Critical Distinction for Consent Language**: The beta onboarding consent currently states "session data is stored on our servers," BUT this is technically incorrect. Data is stored only on the LOCAL FILESYSTEM where the Streamlit app is running. There is NO server-based storage unless the filesystem itself resides on a cloud-hosted server infrastructure.

---

## 1. CURRENT STORAGE MECHANISM

### 1.1 Storage Hierarchy

```
TELOS Observatory V3 Data Flow:
┌─────────────────────────────────────────────────────────────┐
│                   USER INTERACTION                           │
│              (Streamlit Web Interface)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        v                         v
┌──────────────────┐      ┌──────────────────┐
│  IN-MEMORY STATE │      │  FILE SYSTEM     │
│  (Session-only)  │      │  (Persistent)    │
│                  │      │                  │
│ • session_state  │      │ JSON Files:      │
│ • ObservatoryState│      │ • saved_sessions/│
│ • UI toggles     │      │ • beta_consents/ │
│ • Turn data      │      │                  │
└──────────────────┘      └──────────────────┘
        │                         │
        └────────────┬────────────┘
                     │
          (Explicit save via UI)
          (Load on restart)
```

### 1.2 Two-Tier Architecture

**TIER 1: EPHEMERAL IN-MEMORY**
- All real-time state lives in Streamlit's `st.session_state` (RAM)
- Survives: Single browser session (until page refresh)
- Lost on: Browser refresh, app restart, server shutdown

**TIER 2: PERSISTENT FILE SYSTEM**
- User-initiated saves → JSON files on local filesystem
- Automatic consent logging → JSON files on local filesystem
- Survives: App restarts, browser restarts
- Accessible: Only if you have direct filesystem access

---

## 2. WHAT DATA IS ACTUALLY PERSISTED

### 2.1 Session Data (When Explicitly Saved)

**Location**: `/Users/brunnerjf/Desktop/telos_privacy/telos_observatory_v3/saved_sessions/`

**File Format**: JSON

**Data Structure**:
```json
{
  "session_id": "session_1234567890",
  "timestamp": "2025-11-14T10:30:45.123456",
  "mode": "demo" | "open",
  "current_turn": 5,
  "total_turns": 15,
  "avg_fidelity": 0.82,
  "total_interventions": 2,
  "drift_warnings": 1,
  "turns": [
    {
      "turn": 1,
      "timestamp": 2.5,
      "user_input": "User's message",
      "response": "AI's response",
      "fidelity": 0.85,
      "distance": 0.15,
      "threshold": 0.8,
      "intervention_applied": false,
      "drift_detected": false,
      "status": "✓",
      "status_text": "Good",
      "in_basin": true,
      "phase2_comparison": null,
      "is_loading": false
    },
    ...
  ],
  "primacy_attractor": {
    "purpose": [...],
    "scope": [...],
    "boundaries": [...],
    "privacy_level": 0.8,
    "constraint_tolerance": 0.2,
    "task_priority": 0.7
  },
  "metadata": {
    "beta_consent": true,
    "beta_consent_timestamp": "2025-11-14T10:25:00.000000"
  },
  "pa_config": { ... }  // Only in Demo Mode
}
```

**What Gets Saved**:
- Complete conversation history (user inputs + AI responses)
- TELOS metrics (fidelity scores, distance measurements, intervention flags)
- Session metadata (ID, timestamps, mode)
- Primacy Attractor configuration
- Beta consent status

**What Does NOT Get Saved** (until explicitly saved):
- Session data exists only in memory
- Lost on: Page refresh, browser close, app crash
- Accessible only within the active browser session

### 2.2 Consent Logs (Automatic)

**Location**: `/Users/brunnerjf/Desktop/telos_privacy/telos_observatory_v3/beta_consents/consent_log.json`

**File Format**: JSON

**Data Structure**:
```json
{
  "consents": [
    {
      "session_id": "session_1762316828",
      "timestamp": "2025-11-04T23:27:14.333998",
      "consent_statement": "I understand and consent to participate in TELOS Beta testing. I agree to share governance deltas to help improve the system. I understand that session data (including conversations) is stored during beta testing and that governance metrics are extracted from this data for research purposes.",
      "version": "2.1"
    },
    ...
  ]
}
```

**Automatic Persistence**:
- Logged when user clicks "Continue to Beta"
- Stored automatically by `_log_consent()` method
- Includes: session_id, timestamp, consent statement, version
- ALWAYS persisted regardless of whether session is saved

### 2.3 Session Index (Metadata Registry)

**Location**: `/Users/brunnerjf/Desktop/telos_privacy/telos_observatory_v3/saved_sessions/session_index.json`

**Purpose**: Maintains a registry of all saved sessions for the "Load Session" dropdown

**Data Structure**:
```json
{
  "total_sessions": 45,
  "sessions": [
    {
      "id": "sharegpt_filtered_1",
      "name": "Phase 2: sharegpt_filtered_1",
      "date": "2025-10-30T18:18:22.892491",
      "type": "phase2_validation",
      "file": "/path/to/session/file.json"
    },
    ...
  ],
  "failed": [],
  "created_at": "2025-11-01T00:28:38.544458"
}
```

---

## 3. SESSION LIFECYCLE & RESTORATION

### 3.1 Session Initialization (Fresh Start)

```python
# From main.py:initialize_session()
empty_data = {
    'session_id': f"session_{int(datetime.now().timestamp())}",
    'turns': [],
    'total_turns': 0,
    'current_turn': 0,
    'avg_fidelity': 0.0,
    'total_interventions': 0,
    'drift_warnings': 0
}
state_manager.initialize(empty_data)
st.session_state.state_manager = state_manager
```

**Behavior**: Every page load creates a new empty session
- Unique session_id generated from Unix timestamp
- No persistent data carried forward
- User starts with blank conversation

### 3.2 User Session Progression

```
START (Fresh Page Load)
  ↓
Initialize StateManager with empty data
  ↓
Render Beta Onboarding (if no consent)
  ↓
User Gives Consent → logged to consent_log.json automatically
  ↓
Conversation Display Available
  ↓
User Types Messages → stored in memory (st.session_state.state_manager.state.turns)
  ↓
User Clicks "Save Current" → JSON written to saved_sessions/
  ↓
Page Refreshes/App Restarts
  ↓
All in-memory data LOST
  ↓
User Must Click "Load Session" to restore from filesystem
```

### 3.3 Session Restoration (From Saved File)

**Triggered By**: User clicking "📂" button next to saved session in sidebar

**Code Flow**:
```python
# From sidebar_actions.py:_load_session_by_id()
def _load_session_by_id(self, session_id):
    saved_sessions_dir = Path(__file__).parent.parent / 'saved_sessions'
    session_file = saved_sessions_dir / f"{session_id}.json"
    
    with open(session_file, 'r') as f:
        session_data = json.load(f)  # Read JSON from disk
    
    self.state_manager.load_from_session(session_data)  # Populate memory
    st.rerun()  # Trigger UI refresh
```

**Restoration Process**:
1. User selects session from "Saved Sessions" list (populated from session_index.json)
2. Sidebar action loads the corresponding JSON file
3. StateManager.load_from_session() reconstructs all state in memory
4. UI reruns and displays the conversation

**Behavior**:
- All turns, metrics, and metadata restored to memory
- Data becomes available for viewing/editing
- User can navigate turns, view analysis panels
- Conversation is "live" again until next page refresh

---

## 4. FILE I/O OPERATIONS INVENTORY

### 4.1 Write Operations

| Operation | Location | Trigger | Format | Frequency |
|-----------|----------|---------|--------|-----------|
| Save Session | `saved_sessions/<id>_<timestamp>.json` | User clicks "Save Current" | JSON | Manual, on-demand |
| Log Consent | `beta_consents/consent_log.json` | User clicks "Continue to Beta" | JSON | Once per session |
| Update Index | `saved_sessions/session_index.json` | After saving session | JSON | With each save |

**Code References**:
- Session save: `sidebar_actions.py:_save_current_session()` (lines 145-195)
- Consent log: `beta_onboarding.py:_log_consent()` (lines 23-59)
- Index update: (automatic on save)

### 4.2 Read Operations

| Operation | Location | Trigger | Use |
|-----------|----------|---------|-----|
| Load Sessions | `saved_sessions/session_index.json` | Sidebar render | Populate dropdown |
| Load Session | `saved_sessions/<id>.json` | User clicks load | Restore to memory |
| Whitepaper | `../../public/TELOSCOPE_Prototype_Whitepaper.md` | User clicks "Whitepaper" link | Display docs |

**Code References**:
- Index load: `sidebar_actions.py:_get_saved_sessions()` (lines 197-218)
- Session load: `sidebar_actions.py:_load_session_by_id()` (lines 220-244)

### 4.3 Directory Structure

```
telos_observatory_v3/
├── saved_sessions/
│   ├── session_index.json              # Registry of all sessions
│   ├── sharegpt_filtered_1.json         # Actual session data
│   ├── sharegpt_filtered_2.json
│   ├── session_1234567890_20251114.json # User-saved session
│   └── ... (45+ sessions)
├── beta_consents/
│   └── consent_log.json                # Consent audit trail
├── main.py
├── core/state_manager.py
└── components/
    ├── sidebar_actions.py              # Handles save/load
    └── beta_onboarding.py              # Handles consent logging
```

---

## 5. DATABASE/REMOTE STORAGE INTEGRATION

### 5.1 Supabase: NOT INTEGRATED
**Status**: No Supabase imports, no Supabase API calls found

### 5.2 Firebase: NOT INTEGRATED
**Status**: No Firebase imports, no Firebase configuration found

### 5.3 SQL Databases: NOT INTEGRATED
**Status**: No SQLAlchemy, no PostgreSQL/MySQL imports, no database drivers

### 5.4 Cloud Storage: NOT INTEGRATED
**Status**: No AWS S3, no GCS, no Azure Blob Storage imports

### 5.5 API Integrations PRESENT

**Mistral LLM API**:
- Used for: Generating AI responses
- Credentials: MISTRAL_API_KEY from st.secrets or environment
- Data sent: Conversation history, system prompts
- Impact: Conversation content goes to Mistral servers

**Anthropic Claude API** (Optional):
- In requirements.txt but not actively used in V3
- Would be for comparison/extended features

**No Other Remote Storage Found**:
```bash
$ grep -r "requests\|http\|upload\|cloud\|server\|database\|postgres\|mysql\|mongo" \
  telos_observatory_v3/ --include="*.py" | grep -v "MISTRAL\|response\|Mistral"
# Returns: No remote database operations
```

---

## 6. STREAMLIT SESSION STATE MECHANICS

### 6.1 How st.session_state Works

**Scope**: Per-browser-session (RAM only)
**Lifespan**: Single page session (survives f5/reload within same tab; lost on close)
**Persistence**: ZERO automatic persistence to disk

**Variables Stored** (from code analysis):
```python
st.session_state.state_manager                # Core state manager
st.session_state.beta_consent_given           # Consent flag
st.session_state.beta_consent_timestamp       # Consent timestamp
st.session_state.telos_demo_mode              # Demo vs Open mode
st.session_state.active_tab                   # Current tab (DEMO/BETA/TELOS)
st.session_state.saved_sessions_expanded      # UI toggle states
st.session_state.confirm_reset                # Confirmation states
st.session_state.settings_expanded            # Settings panel state
st.session_state.steward_panel_open           # Steward panel state
st.session_state.enable_intro_examples        # Settings
st.session_state.enable_async                 # Performance flags
st.session_state.enable_parallel              # Performance flags
```

**Critical Issue**: These disappear on page refresh unless explicitly saved to disk

### 6.2 ObservatoryState Dataclass

**Defined in**: `core/state_manager.py` lines 20-60

**Fields** (all in-memory only):
```python
@dataclass
class ObservatoryState:
    current_turn: int = 0
    total_turns: int = 0
    session_id: str = "unknown"
    turns: List[Dict[str, Any]] = []  # ALL CONVERSATION DATA
    primacy_attractor: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    # Calibration state
    user_pa_established: bool = False
    ai_pa_established: bool = False
    calibration_phase: bool = True
    
    # UI state
    deck_expanded: bool = False
    teloscope_playing: bool = False
    teloscope_expanded: bool = False
    # ... etc
```

---

## 7. PRIVACY & SECURITY IMPLICATIONS

### 7.1 LOCAL STORAGE RISKS

**Data at Rest**:
- Conversation data stored in plain-text JSON files
- No encryption applied
- Anyone with filesystem access can read
- File permissions: Readable by user owner

**Mitigation NOT Present**:
- No file encryption
- No access control lists
- No data anonymization
- No automatic cleanup/purge

### 7.2 In-Memory Risks

**Volatile Storage**:
- Data in RAM is cryptographically accessible via process memory inspection
- No data wiping on session end
- Potential vulnerability if system is compromised

### 7.3 Consent Language CRITICAL ISSUE

**Current Statement** (beta_onboarding.py line 48):
```
"I understand that session data (including conversations) is stored 
during beta testing and that governance metrics are extracted from 
this data for research purposes."
```

**The Problem**:
This implies data is stored "on our servers" BUT:
- No server-based storage exists in the code
- Data is stored on the LOCAL FILESYSTEM
- "Our servers" is misleading if the app runs on user's machine
- If deployed on a hosting service, this becomes true, but code has NO upload logic

**Corrected Statement Should Be**:
```
"During this beta session, conversation data is stored in memory 
while you're using the application. If you save a session, that 
data is stored locally on your computer/device as a JSON file. 
No data is automatically sent to remote servers for storage. 
Governance metrics (fidelity scores, intervention data) are 
calculated locally and only shared with us if you explicitly 
export them."
```

### 7.4 Data Retention

**Session Data**:
- Kept indefinitely in `saved_sessions/` directory
- No automatic expiration
- No cleanup mechanism
- User must manually delete

**Consent Logs**:
- Kept indefinitely in `beta_consents/consent_log.json`
- Audit trail intact
- No auto-deletion
- Historical record preserved

### 7.5 Data Minimization

**What's Collected**:
- Every user message (full text)
- Every AI response (full text)
- TELOS metrics for each turn
- User consent information
- Session metadata

**Governance Deltas vs Session Data**:
- Deltas = Metrics only (numbers)
- Session Data = Full conversations (text)
- Both currently stored together locally
- No distinction in JSON files

---

## 8. SESSION RESTORATION FLOW (DETAILED)

### 8.1 Complete Restoration Sequence

```
USER RETURNS TO APPLICATION
│
├─→ Is session still in browser memory? (same tab, no refresh)
│   └─→ YES: Use existing st.session_state.state_manager
│       └─→ Conversation instantly available
│       └─→ User can continue from where they left off
│
├─→ Is session lost? (page refresh, new tab, browser restart)
│   └─→ YES: Initialize empty session (see 3.1)
│   └─→ Sidebar shows "Saved Sessions" dropdown
│   └─→ User must click "📂" to load previous session
│       └─→ Triggers: _load_session_by_id()
│       └─→ Reads: JSON file from saved_sessions/
│       └─→ Restores: All turns, metrics, metadata to memory
│       └─→ Display: Full conversation reconstructed on screen
```

### 8.2 Conversation History Restoration Example

**Saved File Content**:
```json
{
  "session_id": "session_1234567890",
  "turns": [
    {"turn": 1, "user_input": "What is TELOS?", "response": "TELOS is...", ...},
    {"turn": 2, "user_input": "How does drift detection work?", "response": "...", ...},
    {"turn": 3, "user_input": "Can you explain the math?", "response": "...", ...}
  ]
}
```

**After Loading**:
```python
state_manager.state.turns = [
    Turn(1, "What is TELOS?", "TELOS is...", ...),
    Turn(2, "How does drift detection work?", "...", ...),
    Turn(3, "Can you explain the math?", "...", ...),
]
state_manager.state.total_turns = 3
state_manager.state.current_turn = 2  # Last turn viewed
```

**Display**:
- Full conversation appears in UI
- User can navigate with "Prev/Next" buttons
- TELOSCOPE Controls show turn slider at position 2
- Metrics re-rendered from restored data

---

## 9. DATA EXPORT & EVIDENCE PACKAGE

### 9.1 Evidence Export

**Triggered By**: User clicks "Export Evidence" button

**Code**: `sidebar_actions.py:_export_evidence()` (lines 331-351)

**Current Implementation**:
```python
def _export_evidence(self):
    interventions = [
        turn for turn in self.state_manager.state.turns
        if turn.get('intervention_applied', False)
    ]
    
    evidence = {
        'session_id': self.state_manager.state.session_id,
        'export_date': datetime.now().isoformat(),
        'total_turns': len(self.state_manager.state.turns),
        'intervention_count': len(interventions),
        'interventions': interventions
    }
    
    st.success(f"Evidence package ready: {len(interventions)} interventions documented")
```

**Problem**: Currently shows success message but doesn't actually export to file
- Data structure is built but not serialized
- No file download mechanism
- User gets UI feedback but no actual file

**Expected Behavior** (what should happen):
- Generate evidence JSON
- Trigger browser download
- User gets `evidence_<session_id>_<timestamp>.json`

---

## 10. STREAMLIT DEPLOYMENT CONSIDERATIONS

### 10.1 Streamlit Cloud Behavior

If deployed on Streamlit Cloud:
- App runs on Streamlit's servers
- Local filesystem is Streamlit Cloud's filesystem
- "Local storage" becomes cloud storage
- Each rerun may reset filesystem (ephemeral containers)
- Current implementation may NOT WORK on cloud

**Current Code Assumption**: Persistent local filesystem
**Cloud Reality**: Ephemeral filesystem between reruns

### 10.2 Self-Hosted Behavior

If self-hosted on personal server:
- Filesystem is the server's filesystem
- Data persists across restarts
- Accessible only to server operator
- Consent statement about "servers" becomes accurate

---

## 11. NO CACHING DECORATORS FOUND

```bash
$ grep -r "@st.cache\|@st.experimental_memo\|@cache\|cache_resource" \
  telos_observatory_v3/ --include="*.py"
# Result: No matches
```

**Implication**: No Streamlit-level caching of sessions
- Every load from filesystem is a fresh read
- No memory-level deduplication
- No cache-based session persistence

---

## 12. COMPLIANCE GAPS & RECOMMENDATIONS

### 12.1 Issues Requiring Immediate Attention

| Issue | Severity | Fix |
|-------|----------|-----|
| Misleading consent language about "servers" | CRITICAL | Clarify that data is stored locally, not uploaded |
| No encryption for stored JSON files | HIGH | Implement file encryption (cryptography library) |
| No data retention policy documented | HIGH | Add automatic purge after X days or explicit deletion |
| Evidence export not functional | MEDIUM | Implement actual file download mechanism |
| No distinction between deltas and full data in storage | MEDIUM | Separate storage or add data-type metadata |
| Session data format includes full conversations | MEDIUM | Consider storing deltas separately from conversations |

### 12.2 Privacy-First Recommendations

1. **Update Consent Language**:
   ```
   "Session data is stored locally during your session and exists only 
   in your browser's memory. If you save a session, it is stored as a 
   JSON file on your device. Governance metrics (mathematical 
   measurements of alignment) are calculated locally. No conversation 
   data is sent to external servers unless you explicitly authorize it."
   ```

2. **Implement Automatic Data Cleanup**:
   ```python
   def cleanup_old_sessions(days=30):
       """Delete saved sessions older than N days"""
       cutoff = datetime.now() - timedelta(days=days)
       for session_file in saved_sessions_dir.glob("*.json"):
           if datetime.fromtimestamp(session_file.stat().st_mtime) < cutoff:
               session_file.unlink()
   ```

3. **Implement File Encryption**:
   ```python
   from cryptography.fernet import Fernet
   cipher = Fernet(key)
   encrypted_data = cipher.encrypt(json_data.encode())
   # Store encrypted_data instead of plaintext
   ```

4. **Add Data Export Control**:
   - Make governance deltas extractable separately
   - Option to delete conversation while keeping metrics
   - Clear "delete all" button with confirmation

5. **Document Data Lifecycle**:
   - Create PRIVACY_POLICY.md specifying:
     - What data is collected
     - How long it's retained
     - Where it's stored
     - User rights (access, deletion, export)

---

## 13. ARCHITECTURE DIAGRAM

```
┌──────────────────────────────────────────────────────────────┐
│              TELOS OBSERVATORY V3 ARCHITECTURE               │
└──────────────────────────────────────────────────────────────┘

LAYER 1: USER INTERFACE
┌──────────────────────────────────────────────────────────────┐
│ Streamlit Web App (Dark Theme, Gold UI)                      │
│ ├─ Conversation Display (ChatGPT-style)                      │
│ ├─ Observation Deck (Metrics)                                │
│ ├─ TELOSCOPE Controls (Playback)                             │
│ └─ Sidebar Actions (Save/Load/Reset)                         │
└────────────────────────┬─────────────────────────────────────┘

LAYER 2: STATE MANAGEMENT
┌────────────────────────────────────────────────────────────┐
│ StateManager (In-Memory)                                     │
│ ├─ ObservatoryState (all current session data)              │
│ ├─ Turn management (navigation)                              │
│ └─ Component visibility toggles                              │
│                                                              │
│ Streamlit session_state (Browser Session)                   │
│ ├─ UI state flags                                            │
│ ├─ Consent tracking                                          │
│ └─ Settings preferences                                      │
└────────────┬──────────────────────┬────────────────────────┘
             │                      │
   (explicit save)      (on consent/autoload)
             │                      │
LAYER 3: PERSISTENCE                │
┌────────────────────────────────────────────────────────────┐
│ FILE SYSTEM (Local JSON)                                     │
│                                                              │
│ saved_sessions/                                              │
│ ├─ session_index.json (registry)                            │
│ ├─ session_1234567890_20251114.json (conversations)         │
│ └─ demo_session_data.json                                   │
│                                                              │
│ beta_consents/                                               │
│ └─ consent_log.json (audit trail)                           │
└────────────────────────────────────────────────────────────┘

LAYER 4: EXTERNAL SERVICES (API ONLY)
┌────────────────────────────────────────────────────────────┐
│ Mistral API                                                  │
│ └─ LLM responses (conversation content sent outbound)       │
│                                                              │
│ Sentence Transformers (Local)                                │
│ └─ Embeddings computed locally, not uploaded                │
└────────────────────────────────────────────────────────────┘

DATA FLOW ON USER MESSAGE:
1. User types message → st.session_state.state_manager.state
2. Message sent to Mistral API → external
3. Response received → st.session_state.state_manager.state
4. TELOS metrics computed locally
5. Data displayed in UI (in memory)
6. User clicks "Save" → JSON written to saved_sessions/
7. Page refresh → all in-memory data lost
8. User clicks "Load" → JSON read from disk → back to step 1

NO AUTOMATIC PERSISTENCE BETWEEN SESSIONS
```

---

## 14. FINAL COMPLIANCE CHECKLIST

- [ ] **Consent Language Inaccurate**: Update to clarify local storage vs server storage
- [ ] **No Encryption**: Add file encryption for stored sessions
- [ ] **No Retention Policy**: Document or implement automatic cleanup
- [ ] **No Data Access Control**: No way for users to see what was saved about them
- [ ] **No Right to Deletion**: No bulk delete mechanism
- [ ] **No Audit Trail of Access**: Who accessed saved sessions not logged
- [ ] **Evidence Export Incomplete**: Mechanism shows message but doesn't export
- [ ] **No Data Processing Agreement**: If sharing metrics, need DPA
- [ ] **Consent Tracking Works**: Consent logs are properly maintained
- [ ] **Session Data Unencrypted**: Files readable by anyone with filesystem access

---

## 15. RECOMMENDED IMMEDIATE ACTIONS

1. **Update beta_onboarding.py Consent Text** (CRITICAL):
   - Change "stored on our servers" to "stored locally"
   - Be explicit about where JSON files are saved
   - Clarify no automatic upload to external servers

2. **Implement File Encryption** (HIGH):
   - Use `cryptography` library (already available)
   - Encrypt before writing to disk
   - Require user confirmation/password

3. **Complete Evidence Export** (HIGH):
   - Actually download file instead of just showing message
   - Implement proper JSON serialization and MIME type headers

4. **Add Data Deletion UI** (HIGH):
   - "Delete Saved Session" button for each session
   - "Delete All Sessions" with confirmation
   - "Delete Consent Record" (if required)

5. **Document Data Handling** (MEDIUM):
   - Create PRIVACY.md explaining architecture
   - Add data retention guidelines
   - Explain Mistral API data sharing

---

## CONCLUSION

**TELOS Observatory V3 uses a LOCAL-FILE-ONLY persistence model with NO remote database integration.** All data persists to JSON files on the filesystem where the app runs. Session state exists only in browser memory until explicitly saved.

**The consent language needs immediate correction** to accurately reflect that data is stored locally, not on external servers. The technical architecture is straightforward and auditable, but privacy safeguards (encryption, retention policies, deletion mechanisms) are not yet implemented.

**For compliance/audit purposes**: This architecture is suitable for research/beta use with LOCAL deployment. For production or cloud deployment, significant additional work on encryption, access controls, and data governance is required.

