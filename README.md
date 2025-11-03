# TELOS Observatory
**Runtime Governance Transparency for Large Language Models**

![Version](https://img.shields.io/badge/version-1.0--experimental-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![Status](https://img.shields.io/badge/status-experimental--launch-gold)

## What is TELOS Observatory?

TELOS Observatory is the first mathematically governed AI conversation platform. Watch your conversations stay on purpose in real-time, measure governance fidelity turn-by-turn, and export evidence-grade telemetry for research or compliance.

**The Problem:** LLMs drift from their purpose across conversations. Studies show 20-40% governance loss in multi-turn sessions.

**The Solution:** Mathematical primacy attractors + real-time intervention + observable transparency.

## Try It Now

**Demo Mode**: 5 free messages, no signup required
- [Launch Demo](https://your-app.streamlit.app) (Streamlit Cloud)
- See governance in action with live TELOS demo conversation
- Upgrade anytime with your Anthropic API key for unlimited usage

## Key Features

### 🔍 Observable Governance
- **Steward Lens**: Real-time governance metrics and intervention tracking
- **TELOSCOPE**: Mathematical transparency window showing fidelity calculations
- **Side-by-Side Comparison**: Toggle between Native Mistral and TELOS Steward responses
- **Turn Navigation**: Time-travel through conversation history

### 📊 Advanced Analytics
- **Cross-Session Trends**: Track governance effectiveness over multiple conversations
- **Comparative Analysis**: Session A vs Session B metrics with statistical significance
- **Pattern Detection**: Automated identification of intervention patterns, drift triggers, and anomalies
- **Statistical Summaries**: Publication-ready statistics with confidence intervals

### 🎁 Evidence Export
- **Multiple Formats**: JSON, CSV, Markdown transcripts, HTML reports, ZIP packages
- **LaTeX Tables**: Publication-ready statistical tables
- **Turn-Level Data**: Complete session telemetry for R/Python/SPSS analysis
- **Reproducibility**: All data needed to verify governance claims

## Quick Start

### Try Online (Easiest)
1. Visit [Demo Mode](https://your-app.streamlit.app) (no signup)
2. Get 5 free governed messages to explore
3. Add your Anthropic API key for unlimited usage

### Run Locally

```bash
git clone https://github.com/TelosSteward/TELOS-Observatory.git
cd telos
pip install -r requirements.txt
streamlit run telos_observatory_v3/main.py
```

Opens in browser at `http://localhost:8501`

### First Session

**Demo Mode** (default):
1. App loads with TELOS demo conversation
2. Send messages to see governance in real-time
3. Watch fidelity metrics in Observation Deck
4. 5-message limit (add API key for more)

**Open Mode** (with API key):
1. Click "Exit Demo Mode" in sidebar
2. Enter your Anthropic API key
3. Configure your Primacy Attractor (purpose/scope/boundaries)
4. Start governed conversation - unlimited messages

## Architecture

```
TELOS Observatory
├── Mathematical Framework (Primacy Attractor)
│   ├── Purpose, Scope, Boundaries → Mitigation Layer
│   ├── Basin Dynamics (Lyapunov Stability)
│   └── Fidelity Measurement (turn-by-turn)
│
├── Observable Interface (Streamlit Dashboard)
│   ├── Chat Interface (ChatGPT-style minimalistic)
│   ├── Steward Lens (real-time governance)
│   ├── TELOSCOPE (mathematical transparency)
│   └── Analytics (cross-session analysis)
│
└── Evidence Infrastructure
    ├── Session Export (JSON/CSV/HTML/ZIP)
    ├── Statistical Analysis (LaTeX tables)
    └── Pattern Detection (automated insights)
```

## For Researchers

### Export Session Data

Click "Export Evidence" in dashboard:
- **JSON**: Complete machine-readable session
- **CSV**: Turn-by-turn telemetry for statistical analysis
- **HTML Report**: Visual governance summary
- **ZIP Package**: All formats with README (recommended)

### Statistical Analysis

Analytics tab provides:
- Descriptive statistics (mean, median, std dev)
- 95% confidence intervals
- Distribution visualizations
- LaTeX table export for papers

### Reproducibility

All exports include:
- Session configuration
- Turn-by-turn fidelity scores
- Intervention metadata
- Timestamps
- Mathematical parameters

## Mathematical Foundation

**Primacy Attractor**: Governance is defined as an attractor in embedding space:
```
â = (τ·p + (1-τ)·s) / ‖τ·p + (1-τ)·s‖

Where:
- p = purpose embedding
- s = scope embedding
- τ = task_priority parameter
```

**Fidelity Measurement**:
```
F(x) = 1 if ‖x - â‖ < r, else 0

Where:
- x = response embedding
- â = attractor center
- r = basin radius
```

**Basin Radius**:
```
r = 2/max(ρ, 0.25)
ρ = 1 - constraint_tolerance
```

See `/docs/mathematical_foundations.md` for complete derivations.

## Current Status

**Version**: 1.0-experimental (Dual Attractor Launch)
**Branch**: `experimental/dual-attractor`
**Status**: Public beta - experimental dual PA architecture
**License**: MIT

### Completed Features
✅ Observatory v3 interface with demo mode
✅ 5-message demo cap with upgrade path
✅ Real-time governance observation (Observation Deck)
✅ Mathematical transparency (TELOSCOPE controls)
✅ Single PA mode (proven, stable)
✅ Dual PA mode (experimental - see DUAL_ATTRACTOR_ARCHITECTURE.md)
✅ UnifiedOrchestratorSteward with fallback logic (9/9 tests passing)
✅ Streamlit Cloud deployment ready

### Experimental Features (v1.2-dual-attractor)
⚡ **Dual Primacy Attractor**: Governs BOTH topic AND AI behavior
⚡ **Lock-On Derivation**: AI PA auto-computed from User PA
⚡ **Dual Fidelity Metrics**: Track user purpose + AI role adherence
⚡ **Smart Fallback**: Degrades gracefully to single PA if needed

### Next Milestones
🎯 **Validate Dual PA** via public beta testing
🎯 **Discord Bot** - Autonomous TELOS governance for Discord servers
🎯 **Production Release** - Promote dual PA to stable (if validated)

## Deployment

Want to deploy your own instance?

- **Streamlit Cloud** (easiest): See [DEPLOYMENT.md](DEPLOYMENT.md) for step-by-step guide
- **Local Development**: `streamlit run telos_observatory_v3/main.py`
- **Docker** (coming soon): Containerized deployment for production

## Documentation

- **Architecture**: [DUAL_ATTRACTOR_ARCHITECTURE.md](DUAL_ATTRACTOR_ARCHITECTURE.md) - Evolution from single to dual PA
- **Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md) - Streamlit Cloud setup guide
- **Tests**: `/tests` - Unit tests for orchestration layer (9/9 passing)

## Citation

If you use TELOS Observatory in your research:
```bibtex
@software{telos_observatory_2025,
  title = {TELOS Observatory: Mathematically Governed AI Conversations},
  author = {TELOS Labs},
  year = {2025},
  url = {https://github.com/TelosSteward/TELOS-Observatory},
  version = {1.0-experimental},
  note = {Dual Primacy Attractor Architecture}
}
```

## License

MIT License - See LICENSE file

## Contact & Feedback

- **Demo**: https://your-app.streamlit.app
- **Issues**: https://github.com/TelosSteward/TELOS-Observatory/issues
- **Email**: telos.steward@gmail.com

---

**Built for researchers who demand evidence, not promises.**

**Experimental Launch**: This is experimental dual PA architecture. Single PA mode is stable and proven. Dual PA is in public beta for validation.
