# TELOS Validation Data

This directory contains all validation studies for the dual PA architecture.

## Structure

```
validation/
├── briefs/
│   └── dual_pa_research_briefs/    # 46 session-level research briefs
├── results/
│   └── dual_pa/                    # Raw validation results (JSON)
└── scripts/                        # Validation and analysis scripts
```

## Dual PA Validation (November 2024)

**Results**: +85.32% improvement over single PA baseline
**Statistical Significance**: p < 0.001, Cohen's d = 0.87
**Test Corpus**: 46 sessions across 8 diverse domains
**Status**: v1.0.0-dual-pa-canonical

### Key Findings

- **User Fidelity**: 0.6744 (dual PA) vs 0.3639 (single PA) → +85.32% improvement
- **AI Fidelity**: 0.7939 (dual PA) vs 0.4154 (single PA) → +91.09% improvement
- **Correlation**: 0.9168 (dual PA) vs 0.4970 (single PA) → +84.47% improvement
- **Perfect Score**: Claude scenario achieved 1.0000 across all metrics

See [../DUAL_PA_VALIDATION_SUMMARY.md](../DUAL_PA_VALIDATION_SUMMARY.md) for complete analysis.

## Directory Contents

### Research Briefs (`briefs/dual_pa_research_briefs/`)

46 individual session analysis files, each containing:
- Session metadata (domain, PA definitions)
- Fidelity metrics (user, AI, correlation)
- Key insights and observations
- Comparison to single PA baseline

**Domains Covered**:
- Educational (tutoring, learning)
- Professional (job search, career advice)
- Technical (coding, debugging)
- Personal growth (fitness, habit formation)
- Creative (writing, brainstorming)
- Research (scientific inquiry)
- Social (etiquette, relationships)
- Mental health (counseling, support)

### Results Files (`results/dual_pa/`)

**Primary Results**:
- `dual_pa_proper_comparison_results.json` (772 KB) - Main validation study comparing dual vs single PA across 46 sessions
- `claude_conversation_dual_pa_fresh_results.json` (290 KB) - Claude scenario counterfactual validation (perfect 1.0000 scores)
- `claude_conversation_starters_only.json` (45 KB) - Baseline conversation starters
- `dual_pa_counterfactual_results.json` (7.7 KB) - Summary counterfactual results

### Scripts (`scripts/`)

**Primary Analysis**:
- `run_proper_dual_pa_comparison.py` - Main validation script comparing dual vs single PA
- `summarize_dual_pa_results.py` - Statistical analysis and summary generation
- `generate_dual_pa_research_briefs.py` - Individual session brief generation

**Specialized Validation**:
- `run_claude_conversation_with_dual_pa.py` - Claude scenario counterfactual validation
- `generate_claude_conversation_starters.py` - Conversation starter generation
- `run_dual_pa_counterfactual.py` - General counterfactual validation

**Analysis Utilities**:
- `analyze_dual_pa_sessions.py` - Session-level analysis
- `compute_dual_pa_aggregate_metrics.py` - Aggregate metric computation
- `compare_single_vs_dual_pa.py` - Direct comparison utilities

**Observatory Integration**:
- `generate_observatory_sessions.py` - Session data for Observatory v3
- `prepare_observatory_data.py` - Data preparation for visualization

## Running Validation

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export MISTRAL_API_KEY="your_mistral_key"
export OPENAI_API_KEY="your_openai_key"  # Optional
```

### Run Full Validation

```bash
cd validation/scripts

# Run main comparison (46 sessions)
python run_proper_dual_pa_comparison.py

# Generate summary statistics
python summarize_dual_pa_results.py

# Generate research briefs
python generate_dual_pa_research_briefs.py
```

### Run Counterfactual Validation

```bash
# Claude scenario (perfect score case)
python run_claude_conversation_with_dual_pa.py

# General counterfactual
python run_dual_pa_counterfactual.py
```

## Validation Methodology

**Test Design**: Matched-pair comparison
- Each session tested with both single PA and dual PA
- Same conversation starters used for both conditions
- AI generates both user and assistant responses (counterfactual)

**Fidelity Measurement**:
- User fidelity: Alignment between user responses and user PA
- AI fidelity: Alignment between AI responses and AI PA
- Correlation: Cross-alignment (user→AI PA, AI→user PA)

**Statistical Analysis**:
- Paired t-tests for within-session comparisons
- Effect size calculation (Cohen's d)
- Distribution analysis across domains

## Next Steps

### Planned Validation Extensions

1. **Runtime Validation** (50-100 sessions)
   - Real human users in "open mode"
   - Measure MBL intervention effectiveness
   - Track drift and correction patterns

2. **Expanded Test Corpus** (500+ sessions)
   - More diverse domains
   - Edge cases and challenging scenarios
   - Multi-turn complexity studies

3. **Long-Context Validation**
   - Extended conversations (50+ turns)
   - Context window stress testing
   - Persistent alignment tracking

4. **Cross-LLM Validation**
   - Test across multiple LLM providers
   - Provider-specific fidelity patterns
   - Model size impact analysis

## Citation

```bibtex
@misc{telos2024dualpa,
  title={TELOS: Dual Primacy Attractor Architecture for AI Purpose Alignment},
  author={TELOS Research},
  year={2024},
  note={Validation study: 46 sessions, +85.32\% improvement}
}
```

## License

See repository root for license information.
