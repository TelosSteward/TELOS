# TELOS Observatory
**Runtime Governance Transparency for Large Language Models**

![Version](https://img.shields.io/badge/version-0.9.7-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## What is TELOS Observatory?

TELOS Observatory is a research-grade dashboard for observing, measuring, and analyzing AI governance in real-time. Built on the TELOS mathematical framework, it provides unprecedented visibility into how governance persists or drifts across multi-turn LLM conversations.

**The Problem:** Studies show 20-40% governance fidelity loss in multi-turn sessions as context drifts.

**The Solution:** Mathematical measurement + visual transparency + exportable evidence.

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

### Installation
```bash
git clone https://github.com/TelosSteward/TELOS-Observatory.git
cd telos
pip install -e .
```

### Run Dashboard
```bash
streamlit run telos_purpose/dev_dashboard/streamlit_live_comparison.py
```

The dashboard will open in your browser at `http://localhost:8501`

### First Conversation

1. Type a message in the chat input
2. Watch governance in action via Steward Lens
3. Toggle between Native and TELOS responses
4. Navigate through turns to see evolution
5. Export evidence when done

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

**Version**: 0.9.7 (89% complete)
**Status**: Production-ready UI, research validation phase
**License**: MIT

### Completed Features
✅ ChatGPT-style interface
✅ Real-time governance observation
✅ Side-by-side Native vs TELOS comparison
✅ Turn navigation (time travel)
✅ Visual analytics (trends, metrics, timelines)
✅ Evidence export (5 formats)
✅ Cross-session analytics
✅ Pattern detection
✅ Statistical summaries

### Roadmap to V1.0
⏳ Documentation completion
⏳ Comprehensive testing
⏳ Final polish

## Citation

If you use TELOS Observatory in your research:
```bibtex
@software{telos_observatory_2025,
  title = {TELOS Observatory: Runtime Governance Transparency for LLMs},
  author = {TELOS Labs},
  year = {2025},
  url = {https://github.com/TelosSteward/TELOS-Observatory},
  version = {0.9.7}
}
```

## License

MIT License - See LICENSE file

## Contact

- **Issues**: https://github.com/TelosSteward/TELOS-Observatory/issues
- **Email**: telos.steward@gmail.com

---

**Built for researchers who demand evidence, not promises.**
