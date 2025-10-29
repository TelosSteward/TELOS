# Steward Research Panel Architecture

## Overview

Steward is a specialized AI research instrument embedded in TELOSCOPE for analyzing TELOS conversation sessions. It treats each session as a "micro research environment" and provides both quantitative metrics and AI-powered qualitative insights.

## Key Design Decisions

### 1. Single API Integration (Mistral)

**Decision:** Use Mistral API for both TELOS governance AND Steward analysis.

**Why:**
- **Cost Efficiency**: One API account instead of two (no Anthropic needed)
- **Consistency**: Same LLM analyzes the governance it creates
- **Simplicity**: Reuses existing `MistralClient` instance
- **Better Context**: Mistral understands TELOS since it IS TELOS

**Before:** Steward required separate `ANTHROPIC_API_KEY` → everyone using Steward consumed your tokens

**After:** Steward uses existing `MISTRAL_API_KEY` → same API powering the core system

### 2. Clear Boundaries & Scope

**Steward is a Research Analysis Tool, NOT a general assistant.**

**CAN DO:**
- Analyze governance patterns (TELOS on/off effects)
- Extract canonical inputs and primacy attractors
- Identify conversation patterns
- Generate session summaries with metrics
- Answer research questions about session data
- Compare sessions for research insights

**CANNOT DO:**
- Modify code or files
- Run system commands
- Access external resources
- Perform actions outside research analysis
- Answer general questions unrelated to TELOS research

**Implementation:**
- Scope validation in `_check_scope()` method
- Out-of-scope keyword detection
- Clear error messages redirecting to valid use cases
- System prompts with explicit boundaries for AI

### 3. Reusable Client Architecture

```python
# In Streamlit UI - reuse existing client
mistral_client = st.session_state.get('llm', None)
analyzer = StewardAnalyzer(mistral_client=mistral_client)

# Single client instance powers:
# 1. TELOS governance
# 2. Steward research analysis
```

**Benefits:**
- No duplicate API connections
- Efficient resource usage
- Consistent configuration

## Architecture Components

### 1. Backend: `steward_analysis.py`

**Core Class:** `StewardAnalyzer`

**Analysis Methods:**
- `analyze_governance_impact()` - TELOS on/off patterns
- `extract_canonical_inputs()` - Key inputs and primacy attractors
- `analyze_conversation_patterns()` - Turn-taking, message lengths
- `analyze_primacy_attractor_evolution()` - How focus evolved
- `generate_session_summary()` - Comprehensive report

**AI Methods:**
- `chat_with_steward()` - Natural language research queries
- `_get_ai_governance_analysis()` - AI interpretation of governance
- `_get_ai_canonical_analysis()` - AI categorization of inputs
- `_get_ai_pattern_analysis()` - AI pattern recognition
- `_get_ai_executive_summary()` - AI-generated overview

**Scope Control:**
- `_check_scope()` - Validates queries are research-focused
- `SCOPE_KEYWORDS` - Defines in-scope vs out-of-scope terms

### 2. Frontend: `streamlit_live_comparison.py`

**UI Component:** `render_steward_research_panel()`

**Features:**
- Toggle to activate Steward mode (overlays normal sidebar)
- Session selector (current or saved sessions)
- Quick analysis dropdown (6 analysis types)
- Chat interface for natural language queries
- Results display with metrics and AI insights

**User Flow:**
1. Toggle "🤖 STEWARD Research" in sidebar
2. Select session to analyze (current or saved)
3. Either:
   - Choose quick analysis → click "Run Analysis"
   - Ask natural language question → click "Ask Steward"
4. View results with metrics + AI interpretation

### 3. CLI Tool: `steward.py`

**Commands:**
- `status` - Show project progress across trackers
- `next` - AI suggestions for what to work on
- `complete "task"` - Mark tasks complete
- `auto-update` - Detect completed work from git
- `report` - Generate weekly status report (AI)
- `analyze "topic"` - Deep dive analysis (AI)

**Now uses Mistral:** Updated to use same API as web interface

## API Integration

### Mistral API Usage

**Configuration:**
```bash
export MISTRAL_API_KEY='your-key-here'
```

**Client Initialization:**
```python
from telos_purpose.llm_clients.mistral_client import MistralClient

client = MistralClient(model="mistral-small-latest")
response = client.generate(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1500,
    temperature=0.7
)
```

**Cost Estimation:**
- Typical analysis: 500-2000 tokens per query
- Model: `mistral-small-latest` (most cost-effective)
- Same token pool as TELOS conversations

## Research Capabilities

### Quantitative Metrics

**Governance Analysis:**
- Total turns
- TELOS enabled/disabled counts
- Governance switches
- TELOS usage ratio

**Conversation Patterns:**
- User/Assistant turn counts
- Average message lengths
- Turn-taking ratios

**Canonical Inputs:**
- Number of canonical inputs identified
- Primacy attractors tracked
- Timestamp data

### Qualitative Insights (AI-Powered)

**Governance Impact:**
- How TELOS vs native responses differed
- When governance was most effective
- Governance switch patterns and reasons

**Canonical Analysis:**
- Thematic categorization of inputs
- Research significance assessment
- Evolution of canonical themes

**Pattern Recognition:**
- Conversation flow dynamics
- Topic shift identification
- Engagement patterns

**Executive Summaries:**
- High-level session overview
- Key research findings
- Recommended next steps

## Session Data Format

Each session contains:
```json
{
  "session_id": "20250129_143022",
  "timestamp": "2025-01-29T14:30:22",
  "messages": [
    {
      "role": "user|assistant",
      "content": "message text",
      "governance_enabled": true,
      "metadata": {
        "primacy_attractor": "...",
        "is_canonical": true
      },
      "timestamp": "..."
    }
  ]
}
```

## Future Enhancements

**Potential Additions:**
1. **Real-time monitoring** - Live analysis during conversations
2. **Session comparison** - Side-by-side analysis of multiple sessions
3. **Export reports** - PDF/CSV export of analysis results
4. **Custom metrics** - User-defined analysis parameters
5. **Longitudinal analysis** - Patterns across time periods
6. **Collaborative annotations** - Research team notes on sessions

## Security & Privacy

**Data Handling:**
- All session data stays local (saved in `saved_sessions/`)
- API calls only send analysis prompts, not full conversation history
- No data stored on external servers
- Researchers control what sessions are analyzed

**API Key Security:**
- Keys stored in `.env` file (gitignored)
- Never hardcoded in source
- Single API key for entire system

## Testing Steward

**Basic Test (No AI):**
```bash
# Run without MISTRAL_API_KEY set
python3 steward.py status  # Should work
python3 steward.py next    # Falls back to basic mode
```

**AI Test (With API):**
```bash
export MISTRAL_API_KEY='your-key'
python3 steward.py next    # Should give AI recommendations
python3 steward.py analyze "testing suite"
```

**Web UI Test:**
1. Start Streamlit: `streamlit run telos_purpose/dev_dashboard/streamlit_live_comparison.py`
2. Have a conversation with TELOS
3. Toggle "🤖 STEWARD Research" in sidebar
4. Select "Governance Impact Assessment"
5. Click "Run Analysis"
6. Verify metrics and AI interpretation appear

## Summary

Steward is now a **production-ready research instrument** that:
- Uses single API (Mistral) for efficiency
- Has clear boundaries (research analysis only)
- Provides both quantitative and qualitative insights
- Works in both web UI and command-line
- Treats sessions as analyzable research environments
- Enables researchers to ask natural language questions about their data

This transforms TELOSCOPE from a governance platform into a **complete research instrument** with embedded analysis capabilities.
