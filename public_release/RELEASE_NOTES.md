# Runtime Governance v0.1.0

**Release Date:** November 5, 2025

## What is Runtime Governance?

Keep your Claude Code sessions aligned with project goals - automatically.

Every conversation turn is measured against your Primacy Attractor (PA) using real mathematics (embeddings in ℝ³⁸⁴, cosine similarity). Your work sessions become validation data with zero extra effort.

## Key Features

- ✅ **Automatic Fidelity Measurement** - Every turn measured against PA baseline
- ✅ **100% Local** - No external API calls, zero cost
- ✅ **Real Mathematics** - Embeddings + cosine similarity (not heuristics)
- ✅ **Memory MCP Integration** - Persistent session history
- ✅ **Export Formats** - Standard JSON, Dashboard CSV, Grant Reports
- ✅ **Zero Configuration** - Works out of the box

## Requirements

- Python 3.9+
- Claude Code with Memory MCP enabled
- sentence-transformers package
- numpy package

## Installation

```bash
# Download/clone this repository
git clone https://github.com/telos-project/runtime-governance.git
cd runtime-governance

# Run installation script
./install.sh

# Or install manually
pip install -r requirements.txt
```

## Quick Start

1. **Add PA to your `.claude_project.md`:**
   ```markdown
   ## 🔭 RUNTIME GOVERNANCE - ACTIVE

   **PA Baseline:**
   "Build a REST API for authentication by Q1..."
   ```

2. **Initialize session:**
   ```bash
   python3 runtime_governance_start.py
   ```

3. **Work normally** - Claude Code runs checkpoints automatically

4. **Export data:**
   ```bash
   python3 runtime_governance_export.py
   ```

See `QUICK_START.md` for detailed instructions.

## Architecture

**Fully self-contained:**
- Local sentence-transformers for embeddings
- Local Memory MCP for storage
- No cloud dependencies
- Zero API costs

**The flow:**
```
Claude responds →
Checkpoint runs (local) →
Fidelity calculated (ℝ³⁸⁴ embeddings, cosine similarity) →
Stored in Memory MCP (local) →
Display: 📊 Turn X: F=0.XXX ✅/⚠️/🚨
```

## Cost

**$0.00** - Everything runs locally.

## Files Included

- `runtime_governance_start.py` - Session initialization
- `runtime_governance_checkpoint.py` - Turn-by-turn measurement
- `runtime_governance_export.py` - Data export
- `embedding_provider.py` - Local embedding generation
- `README.md` - Complete documentation
- `QUICK_START.md` - 5-minute setup guide
- `claude_project_template.md` - Template for your project
- `install.sh` - One-command installation
- `requirements.txt` - Python dependencies
- `LICENSE` - MIT License

## What's New in v0.1.0

Initial release with:
- Core measurement engine
- Memory MCP integration
- Three export formats (standard, dashboard, grant)
- Complete documentation
- MIT license (use freely)

## Roadmap

- [ ] Dashboard visualization (v0.2)
- [ ] Multi-project support (v0.3)
- [ ] CI/CD integration (v0.4)
- [ ] Real-time notifications (v0.5)
- [ ] Team collaboration features (v1.0)

## Support

**Documentation:** See `README.md` and `QUICK_START.md`
**Issues:** Report on GitHub
**Questions:** Open a discussion

## License

MIT License - Use freely, commercial or otherwise.

## Credits

Built by the TELOS Project as part of privacy-preserving AI governance research.

---

**Welcome to Runtime Governance for AI conversations.**

Static context files tell Claude what to do.
Runtime Governance tells you if Claude is doing it.
