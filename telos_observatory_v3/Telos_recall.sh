#!/bin/bash
# TELOS Recall Script - Critical Context for Claude Code Sessions
# This script contains essential documentation that should be read when restarting Claude Code sessions

echo "========================================"
echo "TELOS RECALL - Essential Context"
echo "========================================"
echo ""
echo "📊 PROJECT RESOURCES"
echo "--------------------"
echo "Zenodo DOI: 10.5281/zenodo.17702890"
echo "Validation Dataset: 1,986 attacks | 0% ASR | 99.9% CI [0%, 0.38%]"
echo "Dataset URL: https://doi.org/10.5281/zenodo.17702890"
echo ""
echo "Loading critical TELOS documentation..."
echo ""

# Display key documentation files
echo "1. BETA STATUS SUMMARY"
cat BETA_STATUS_SUMMARY.md
echo ""
echo "========================================"
echo ""

echo "2. BETA EXPERIENCE MASTER FLOW"
cat BETA_EXPERIENCE_MASTER_FLOW.md
echo ""
echo "========================================"
echo ""

echo "3. TELOS MARKET POSITION"
cat business/TELOS_Market_Position_Reality_Check.md
echo ""
echo "========================================"
echo ""

echo "4. REGULATORY FORCING FUNCTION"
cat business/Regulatory_Forcing_Function.md
echo ""
echo "========================================"
echo ""

echo "TELOS RECALL COMPLETE"
echo "All critical context has been loaded."
echo ""
