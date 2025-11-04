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
