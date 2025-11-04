#!/bin/bash
# TELOS Full Project Recall Script
# Use this instead of /telos command until slash commands work

echo "📋 TELOS FULL PROJECT RECALL"
echo "============================================================"
echo ""
echo "## 1. PROJECT CONTEXT"
echo ""
cat .claude_project.md
echo ""
echo "============================================================"
echo "## 2. GIT STATUS"
echo ""
git status
echo ""
git log --oneline -5
echo ""
git branch -a
echo ""
echo "============================================================"
echo "## 3. ESTABLISHING GOVERNANCE"
echo ""
python3 telos_init.py
echo ""
echo "============================================================"
echo "✅ TELOS SESSION READY"
echo "============================================================"
echo ""
echo "## 4. LAUNCHING TELOS OBSERVATORY"
echo ""
echo "🔭 Starting TELOS Observatory v3 (Demo Mode)..."
echo "   Cleaning up any existing Streamlit processes..."

# Kill any existing streamlit processes
pkill -f streamlit 2>/dev/null
sleep 1

# Set Python path
export PYTHONPATH=/Users/brunnerjf/Desktop/telos_privacy

# Launch Observatory in background
echo "   Launching at http://localhost:8501"
echo ""
nohup /Users/brunnerjf/Library/Python/3.9/bin/streamlit run telos_observatory_v3/main.py --server.port 8501 > /dev/null 2>&1 &

# Wait a moment for startup
sleep 3

echo "✅ TELOS Observatory running at: http://localhost:8501"
echo ""
echo "============================================================"
echo "🚀 READY TO WORK"
echo "============================================================"
