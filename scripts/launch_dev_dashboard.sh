#!/bin/bash

# Launch script for TELOS Dev Dashboard
# This script starts the Streamlit dashboard with optimal settings

echo "======================================================================="
echo "🚀 Launching TELOS Dev Dashboard"
echo "======================================================================="
echo ""

# Check if Streamlit is installed
if ! python3 -m streamlit --version &> /dev/null 2>&1; then
    echo "❌ Streamlit is not installed. Installing now..."
    pip3 install streamlit
    echo ""
else
    echo "✅ Streamlit is installed"
fi

# Check if psutil is installed (for health monitoring)
if ! python3 -c "import psutil" &> /dev/null 2>&1; then
    echo "⚠️  psutil not installed. Installing for health monitoring..."
    pip3 install psutil
    echo ""
fi

# Check if anthropic is installed (for Claude integration)
if ! python3 -c "import anthropic" &> /dev/null 2>&1; then
    echo "⚠️  Anthropic SDK not installed. Claude features will be disabled."
    echo "   To enable: pip3 install anthropic"
    echo ""
fi

# Check for Claude API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ℹ️  No API key detected - Claude Bridge mode enabled"
    echo "   This mode works with your Claude.ai subscription!"
    echo "   Features: Command templates, natural language builder, web bridge"
    echo ""
else
    echo "✅ Claude API configured - Full API integration enabled"
    echo ""
fi

# Launch the dashboard
echo "🌐 Starting dashboard server..."
echo "📍 Dashboard will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "======================================================================="

# Run Streamlit with custom settings
# Use python3 -m to ensure we use the right installation
python3 -m streamlit run dev_dashboard/main.py \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.base="dark" \
    --theme.primaryColor="#FFD700" \
    --theme.backgroundColor="#1a1a1a" \
    --theme.secondaryBackgroundColor="#2a2a2a" \
    --theme.textColor="#e0e0e0"