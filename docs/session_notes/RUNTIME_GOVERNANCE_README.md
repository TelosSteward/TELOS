# Claude Code Runtime Governance

**Keep Claude Code aligned with your project goals - automatically.**

Every conversation turn is measured against your project's Primacy Attractor (PA) using real mathematics (embeddings in ℝ³⁸⁴, cosine similarity). Your work sessions become validation data with zero extra effort.

## What This Does

- **Automatic Fidelity Measurement**: Every Claude response measured against your PA baseline
- **Turn-by-Turn Tracking**: Complete session history stored in Memory MCP
- **Statistical Process Control**: SPC/DMAIC for AI conversations
- **Validation Data Generation**: Every work session produces evidence of alignment
- **Cost-Effective**: ~$0.001 per turn using Mistral embeddings

## Why This Matters

Traditional `.claude_project.md` files are static. They can't tell you if Claude is drifting from your goals. This system provides:

- **Real-time alignment barometer** (✅ on track, ⚠️ warning, 🚨 drift)
- **Session analytics** (mean fidelity, drift patterns, turn-by-turn curves)
- **Empirical evidence** for stakeholders/grants
- **True checksum** for project trajectory

**This is leagues better than static context files.**

## Requirements

- Claude Code with Memory MCP enabled
- Python 3.9+
- Mistral API key (for embeddings)
- `sentence-transformers` package (optional local embeddings)

## Installation

1. Copy these files to your project root:
   - `telos_session_start.py`
   - `telos_turn_checkpoint.py`
   - `telos_session_export.py`
   - `embedding_provider.py`
   - `dual_attractor.py`

2. Install dependencies:
```bash
pip install sentence-transformers numpy
```

3. Set Mistral API key (optional, for cloud embeddings):
```bash
export MISTRAL_API_KEY="your_key_here"
```

4. Add to `.gitignore`:
```
.telos_active_session.json
.telos_checkpoint_*.json
.telos_current_turn.json
```

## Setup Your Project

### 1. Define Your PA Baseline

Add to your `.claude_project.md`:

```markdown
## 🔭 RUNTIME GOVERNANCE - ACTIVE

**PA Baseline (Stable):**
[Your project purpose/scope/boundaries in 2-4 sentences]

Example:
"Ship ProductX to production by Q2 2025 through systematic execution on
backend APIs, frontend UI, and deployment infrastructure while maintaining
code quality, test coverage >80%, and security best practices. Build working
systems that deliver user value and meet business requirements."

**Fidelity Thresholds:**
- F ≥ 0.8: ✅ ON TRACK
- 0.7 ≤ F < 0.8: ⚠️ WARNING
- F < 0.7: 🚨 DRIFT DETECTED

**After every turn:**
1. Store turn in Memory MCP
2. Calculate fidelity (actual mathematics)
3. Report: 📊 Turn X: F=0.XXX ✅/⚠️/🚨
```

### 2. Initialize Session

At start of each work session:
```bash
python3 telos_session_start.py
```

This:
- Extracts PA from `.claude_project.md`
- Creates `ActiveSession_[date]` in Memory MCP
- Initializes turn counter

### 3. Run Checkpoints

After each conversation turn, Claude Code should run:
```bash
python3 telos_turn_checkpoint.py --user "user message" --assistant "assistant response"
```

Or with a JSON file:
```bash
python3 telos_turn_checkpoint.py --turn-json turn_data.json
```

### 4. Export Session Data

```bash
# Standard format
python3 telos_session_export.py

# Dashboard format (CSV + summary)
python3 telos_session_export.py --format dashboard

# Grant report format
python3 telos_session_export.py --format grant
```

## Cost Analysis

**Mistral Embeddings (mistral-embed):**
- $0.10 per 1M tokens
- Average turn: ~200 tokens (user + response)
- **Cost per turn: ~$0.00002** (two-hundredths of a cent)
- **100 turns: ~$0.002** (two-tenths of a cent)

**Local Embeddings (sentence-transformers):**
- Free
- Runs on CPU
- Slightly slower but no API costs

**Storage (Memory MCP):**
- Free (local JSON files)

**Bottom line:** Even with 1000 turns/month, you're spending ~$0.02 on embeddings. Negligible.

## How It Works

### The Mathematics

1. **Embedding Generation**: Convert text to 384-dimensional vectors (ℝ³⁸⁴)
   - Response text → embedding
   - PA baseline → embedding

2. **Fidelity Calculation**: Cosine similarity
   ```
   F = (response · PA) / (||response|| × ||PA||)
   ```

3. **Status Determination**:
   - F ≥ 0.8: ON TRACK ✅
   - 0.7 ≤ F < 0.8: WARNING ⚠️
   - F < 0.7: DRIFT 🚨

### The Data Flow

```
Turn happens → Checkpoint runs → Fidelity calculated →
Turn stored in Memory MCP → Session stats updated →
Report displayed → Work continues
```

### Key Design Principles

- **Pure measurement** (not adaptive control)
- **PA is baseline** (not progressively evolved)
- **Feedback control** (adjust PA manually when needed)
- **Zero intervention** (measures after response, doesn't intercept)

## Use Cases

### 1. Long-Running Projects
Ensure Claude stays aligned with project goals across multiple sessions spanning weeks/months.

### 2. Grant Applications
Generate empirical evidence of alignment: "Our AI development process maintains 0.87 mean fidelity across 200+ conversation turns."

### 3. Team Collaboration
Multiple people use Claude Code on same project - governance ensures consistency.

### 4. High-Stakes Domains
Medical, legal, financial applications where drift detection is critical.

### 5. SPC/DMAIC for AI
Statistical Process Control for AI-assisted software development.

## Configuration Options

### Embedding Provider

Edit `embedding_provider.py` to choose:

**Local (Free):**
```python
embeddings = EmbeddingProvider(
    provider='local',
    deterministic=False
)
```

**Mistral Cloud:**
```python
embeddings = EmbeddingProvider(
    provider='mistral',
    api_key=os.getenv('MISTRAL_API_KEY')
)
```

**OpenAI:**
```python
embeddings = EmbeddingProvider(
    provider='openai',
    model='text-embedding-3-small',
    api_key=os.getenv('OPENAI_API_KEY')
)
```

### Checkpoint Cadence

**Every turn (recommended):**
- Maximum validation data
- Negligible cost with Mistral
- Enables detailed fidelity curves

**Every N turns:**
Edit checkpoint script to check `turn_count % N == 0`

### Thresholds

Customize fidelity thresholds in `.claude_project.md`:
```markdown
**Fidelity Thresholds:**
- F ≥ 0.85: ✅ EXCELLENT
- 0.75 ≤ F < 0.85: ⚠️ ACCEPTABLE
- F < 0.75: 🚨 NEEDS ATTENTION
```

## Memory MCP Integration

All session data stored as entities:

**Session Entity:**
```json
{
  "name": "ActiveSession_2025-11-05",
  "entityType": "Session",
  "observations": [
    "session_id: session_20251105_103645",
    "started_at: 2025-11-05T10:36:45",
    "pa_baseline: [your PA text]",
    "turn_count: 42",
    "status: active"
  ]
}
```

**Turn Entities:**
```json
{
  "name": "Turn_5",
  "entityType": "Turn",
  "observations": [
    "turn_number: 5",
    "fidelity: 0.834",
    "status: on_track",
    "user_message: [text]",
    "timestamp: [ISO]"
  ]
}
```

**Relations:**
```
ActiveSession_2025-11-05 --[has_turn]--> Turn_1
ActiveSession_2025-11-05 --[has_turn]--> Turn_2
...
```

## Querying Session Data

```python
# Get session stats
python3 -c "
import json
session = json.load(open('.telos_active_session.json'))
print(f'Turn count: {session[\"turn_count\"]}')
print(f'Last fidelity: {session[\"last_fidelity\"]:.3f}')
"
```

Or query Memory MCP:
```python
mcp__memory__search_nodes(query="ActiveSession")
```

## Exporting for Analysis

```bash
# Export all session turns to JSON
python3 telos_session_export.py --format standard

# Export for dashboard visualization
python3 telos_session_export.py --format dashboard

# Export for grant/stakeholder report
python3 telos_session_export.py --format grant
```

## Limitations

1. **No real-time intervention**: Cannot intercept Claude responses before delivery (architectural limitation of Claude Code)
2. **Feedback control only**: Can adjust `.claude_project.md` for next turn, not current turn
3. **Mistral API dependency**: For cloud embeddings (local option available)
4. **Memory MCP required**: Core feature of Claude Code

## Roadmap

- [ ] Dashboard visualization (Streamlit/Plotly)
- [ ] Automated PA refinement suggestions
- [ ] Multi-project support
- [ ] Integration with CI/CD
- [ ] Slack/Discord notifications for drift
- [ ] Comparative analysis across sessions

## License

MIT License - Use freely, commercial or otherwise.

## Support

This was built as part of the TELOS project (privacy-preserving AI governance).

Questions? Check the example implementation at: [your repo]

## The Bottom Line

**You now have a barometer for Claude Code alignment.**

Static `.claude_project.md` files tell Claude what to do. Runtime Governance tells you if Claude is actually doing it.

Every work session becomes validation data. Zero extra effort. Negligible cost.

**Welcome to Statistical Process Control for AI conversations.**
