# TELOS Comprehensive Control Panel

Production-ready Streamlit dashboard for real-time monitoring and analysis of TELOS mathematical governance.

## Features

### 🎯 Core Capabilities

1. **Live Conversation Interface**
   - Interactive chat with TELOS-governed LLM
   - Real-time response processing
   - Conversation history with metrics
   - Intervention indicators

2. **Real-Time Metrics Dashboard**
   - Turn-by-turn fidelity tracking
   - Lyapunov function monitoring
   - Error signal visualization
   - Basin membership status
   - Drift distance measurements

3. **Trajectory Visualization**
   - 2D projection of conversation path in embedding space (PCA)
   - Attractor center and basin boundary overlay
   - Color-coded turn progression
   - Intervention markers
   - Interactive plotly charts

4. **Intervention Logging**
   - Comprehensive intervention history
   - Detailed reasoning and metrics
   - Fidelity before/after tracking
   - Export capabilities

5. **Mathematical State Explanations**
   - Plain English descriptions of metrics
   - Configuration parameters
   - Threshold indicators
   - Stability analysis

## Quick Start

### Option 1: Launch Script (Recommended)

```bash
cd ~/Desktop/telos
./launch_dashboard.sh
```

### Option 2: Manual Launch

```bash
cd ~/Desktop/telos
source venv/bin/activate
export MISTRAL_API_KEY="your_key_here"
streamlit run telos_purpose/dev_dashboard/streamlit_live_comparison.py
```

## Dashboard Tabs

### 💬 Conversation
- Live chat interface
- Message input and processing
- Current state metrics (F, V(x), ε, Basin)
- Conversation history
- Intervention indicators

### 📊 Metrics Dashboard
- 4-panel comprehensive view:
  - Fidelity over time
  - Drift distance from attractor
  - Error signal ε
  - Basin membership
- Individual metric plots with annotations
- Intervention markers on all plots

### 🎯 Trajectory
- 2D PCA projection of embedding space
- Attractor center (gold star)
- Basin boundary (purple circle)
- Conversation path (blue line with color gradient)
- Intervention points (red X markers)
- Trajectory statistics

### ⚡ Interventions
- Tabular intervention log
- Turn, timestamp, type, reason
- Error signals and fidelity changes
- Detailed expansion view

### ❓ Help
- Complete documentation
- Metric explanations
- Intervention types
- Analysis tips

## Sidebar Controls

### System Status
- Operational indicator
- Session turn counter
- Last update timestamp

### Quick Stats
- Current fidelity
- Average fidelity
- Intervention count and rate
- Basin membership status

### Controls
- **Reset Session**: Clear all data and start fresh
- **Export Data**: Download complete session JSON

### Configuration
- Load default config.json
- Upload custom configuration
- Initialize TELOS button

## Metrics Explained

### Telic Fidelity (F)
**Range**: [0, 1]

Measures semantic alignment with governance objectives.

- **F > 0.8**: Excellent alignment ✅
- **0.5 < F < 0.8**: Moderate alignment ⚠️
- **F < 0.5**: Poor alignment ❌

### Lyapunov Function V(x)
**Formula**: V(x) = ||x - x*||²

Energy function measuring drift from ideal state.

- Lower values = closer to attractor
- V(x) < R² indicates stability (inside basin)
- Increases when conversation drifts off-topic

### Error Signal ε
**Range**: [0, 1]

Proportional control signal for interventions.

- **ε < ε_min**: Stable, no action (MONITOR)
- **ε_min ≤ ε < ε_max**: Warning zone (CORRECT)
- **ε ≥ ε_max**: Critical, intervention required (INTERVENE/ESCALATE)

### Basin Membership
**Binary**: Inside/Outside

Stability indicator based on distance from attractor.

- **Inside**: ||x - x*|| < R (stable)
- **Outside**: ||x - x*|| ≥ R (drifting)

## Intervention Types

1. **MONITOR**: Passive observation, no system action
2. **CORRECT**: Gentle reminder in system prompt
3. **INTERVENE**: Active correction with response regeneration
4. **ESCALATE**: Critical intervention with strong correction

## Data Export

Click "Export Data" in sidebar to download JSON containing:

- Session ID and timestamp
- Complete conversation history
- All metric time series
- Full intervention log
- Configuration used

Export format:
```json
{
  "session_id": "session_1761392002",
  "timestamp": "2025-10-25T07:33:29.123456",
  "total_turns": 5,
  "total_interventions": 1,
  "conversation_history": [...],
  "metrics": {
    "fidelity_history": [...],
    "lyapunov_history": [...],
    "error_signal_history": [...],
    "drift_distance_history": [...]
  },
  "intervention_log": [...],
  "config": {...}
}
```

## Configuration

The dashboard uses `config.json` with the following structure:

```json
{
  "governance_profile": {
    "purpose": [...],
    "scope": [...],
    "boundaries": [...]
  },
  "attractor_parameters": {
    "constraint_tolerance": 0.2,
    "privacy_level": 0.8,
    "task_priority": 0.9
  },
  "intervention_thresholds": {
    "epsilon_min": 0.5,
    "epsilon_max": 0.8
  },
  "validation_settings": {
    "use_real_embeddings": true,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

## Analysis Tips

### Identifying Stable Conversations
- High fidelity (F > 0.8)
- Low Lyapunov values
- Consistently inside basin
- Low error signals
- Minimal interventions

### Detecting Drift
- Declining fidelity trend
- Rising error signals
- Moving outside basin
- Increasing Lyapunov values
- Trajectory moving away from attractor

### Evaluating Interventions
- Return to basin after correction
- Fidelity improvement
- Error signal reduction
- Trajectory correction toward attractor

### System Health Indicators
- **Healthy**: <20% intervention rate, F_avg > 0.8
- **Acceptable**: 20-40% intervention rate, F_avg > 0.6
- **Needs Tuning**: >40% intervention rate or F_avg < 0.6

## Troubleshooting

### Dashboard Won't Start
- Check virtual environment activated: `source venv/bin/activate`
- Verify Streamlit installed: `pip install streamlit`
- Check for port conflicts: Use `--server.port 8502` flag

### API Key Issues
- Ensure `MISTRAL_API_KEY` environment variable set
- Verify key is valid and has credits
- Check internet connection

### Visualization Issues
- Ensure scikit-learn installed: `pip install scikit-learn`
- Check plotly version: `pip install plotly --upgrade`
- Verify at least 2 turns processed for trajectory plot

### Performance Issues
- Use deterministic embeddings for testing: `"use_real_embeddings": false`
- Reduce max_tokens in LLM calls
- Clear session history periodically

## Development

### File Structure
```
telos_purpose/dev_dashboard/
├── streamlit_live_comparison.py  # Main dashboard application
├── README.md                      # This file
└── __init__.py
```

### Adding Custom Metrics
1. Add metric to session state in `init_session_state()`
2. Compute and store in `process_turn()`
3. Create visualization function
4. Add to appropriate tab in `main()`

### Extending Visualizations
- Use plotly for interactive charts
- Add to existing tabs or create new ones
- Follow naming convention: `create_*_plot()`

## Requirements

- Python 3.9+
- streamlit
- plotly
- pandas
- numpy
- scikit-learn
- telos_purpose package (installed)

## Performance

- **Initial load**: ~5-10 seconds (model loading)
- **Per turn**: ~2-5 seconds (LLM + governance)
- **Dashboard update**: ~100-500ms (visualization rendering)

## Known Limitations

- PCA projection loses some information (typically captures 60-80% variance)
- Trajectory visualization requires 2+ turns
- Large sessions (>100 turns) may slow down
- Export data grows with conversation length

## Future Enhancements

- [ ] 3D trajectory visualization
- [ ] Real-time streaming updates
- [ ] Comparative baseline view
- [ ] Historical session comparison
- [ ] Advanced statistical analysis
- [ ] Custom metric definitions
- [ ] Export to CSV/Excel
- [ ] Annotation and note-taking
- [ ] Session replay functionality

## Support

For issues or questions:
- Check TELOS main README
- Review Internal Test 0 documentation
- Consult `TEST0_SUCCESS.md` for setup verification

## License

MIT License - See LICENSE file in repository root

---

**Version**: 2.0
**Last Updated**: 2025-10-25
**Framework**: TELOS (Telically Entrained Linguistic Operational Substrate)
