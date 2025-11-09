#!/bin/bash
# TELOS Full Project Recall Script
# Adapted for TELOS_CLEAN repository structure
# Use this for comprehensive project context and status

echo "📋 TELOS FULL PROJECT RECALL"
echo "============================================================"
echo ""
echo "## 1. PROJECT CONTEXT"
echo ""
if [ -f ".claude_project.md" ]; then
    cat .claude_project.md
else
    echo "No .claude_project.md found - checking README..."
    head -50 README.md
fi
echo ""
echo "============================================================"
echo "## 2. GIT STATUS"
echo ""
git status
echo ""
echo "Recent commits:"
git log --oneline -10
echo ""
echo "Branches:"
git branch -a
echo ""
echo "============================================================"
echo "## 3. VALIDATION DATA STATUS"
echo ""
echo "📊 Baseline Conversations Available:"
ls -la tests/validation_data/baseline_conversations/ 2>/dev/null | head -5
echo ""
echo "Total baseline conversations: $(ls tests/validation_data/baseline_conversations/*.json 2>/dev/null | wc -l)"
echo ""
echo "============================================================"
echo "## 4. TESTING FRAMEWORK STATUS"
echo ""
echo "✅ Available Testing Components:"
echo "  - Baseline Runners: $(ls tests/validation/baseline_runners.py 2>/dev/null && echo 'Found' || echo 'Missing')"
echo "  - Comparative Test: $(ls tests/validation/comparative_test.py 2>/dev/null && echo 'Found' || echo 'Missing')"
echo "  - Counterfactual Manager: $(ls telos/core/counterfactual_manager.py 2>/dev/null && echo 'Found' || echo 'Missing')"
echo ""
echo "============================================================"
echo "## 5. OBSERVATORY STATUS"
echo ""
# Check if Observatory is running
if pgrep -f "streamlit.*observatory/main.py" > /dev/null; then
    echo "🔭 Observatory is RUNNING at http://localhost:8501"
else
    echo "⚠️  Observatory is NOT running"
    echo "   To start: python3 -m streamlit run observatory/main.py --server.port 8501"
fi
echo ""
echo "============================================================"
echo "## 6. ENVIRONMENT STATUS"
echo ""
echo "Working Directory: $(pwd)"
echo "Python Version: $(python3 --version)"
echo "Virtual Environment: $(if [ -d "venv" ]; then echo "✅ Available"; else echo "❌ Not found"; fi)"
echo ""
echo "Key Dependencies:"
if [ -f "requirements.txt" ]; then
    echo "  - numpy: $(grep numpy requirements.txt | head -1)"
    echo "  - torch: $(grep torch requirements.txt | head -1)"
    echo "  - streamlit: $(grep streamlit requirements.txt | head -1)"
    echo "  - anthropic: $(grep anthropic requirements.txt | head -1)"
fi
echo ""
echo "============================================================"
echo "## 7. KEY FINDINGS TO REMEMBER"
echo ""
echo "📊 Validation Results (from previous studies):"
echo "  - Baseline improvement: +85.32% over single PA"
echo "  - 46 ShareGPT conversations + 1 Claude conversation"
echo "  - Counterfactual testing framework ready"
echo "  - DMAIC/SPC implementation complete"
echo ""
echo "🔑 Key Innovations:"
echo "  - Dual Primacy Attractor Architecture"
echo "  - DMAIC cycle per conversation turn"
echo "  - Telemetric Keys cryptographic system"
echo "  - Progressive PA Extractor"
echo ""
echo "============================================================"
echo "## 8. INITIALIZING STEWARD UNIFIED"
echo ""
echo "🤖 Waking up the Unified Steward Project Manager..."
echo ""
python3 steward/steward_unified.py
echo ""
echo "============================================================"
echo "✅ TELOS RECALL COMPLETE"
echo "============================================================"
echo ""
echo "Ready to continue work. Key areas:"
echo "  1. Run new validation studies with clean data"
echo "  2. Observatory interface improvements"
echo "  3. Documentation updates"
echo "  4. Grant application preparation"
echo ""