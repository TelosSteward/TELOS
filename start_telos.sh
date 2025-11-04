#!/bin/bash
#
# TELOS Observatory Launcher
# Simple script to start the TELOS Observatory Demo Mode interface
#
# Usage: ./start_telos.sh
#

echo "🔭 Starting TELOS Observatory..."
echo "================================"
echo ""

# Kill any existing streamlit processes
echo "Cleaning up any existing Streamlit processes..."
pkill -f streamlit 2>/dev/null
sleep 1

# Set Python path
export PYTHONPATH=/Users/brunnerjf/Desktop/telos_privacy

# Start the Observatory
echo "Launching TELOS Observatory v3 (Demo Mode)..."
echo ""
echo "✅ Observatory will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

/Users/brunnerjf/Library/Python/3.9/bin/streamlit run telos_observatory_v3/main.py --server.port 8501
