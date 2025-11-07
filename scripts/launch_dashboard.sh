#!/bin/bash
#
# TELOS Dashboard Launcher
# Quick script to launch the comprehensive control panel
#

echo "🎯 TELOS Comprehensive Control Panel"
echo "====================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -e ."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if MISTRAL_API_KEY is set
if [ -z "$MISTRAL_API_KEY" ]; then
    echo ""
    echo "⚠️  MISTRAL_API_KEY not set in environment"
    echo "Would you like to set it now? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        echo "Enter your Mistral API key:"
        read -r api_key
        export MISTRAL_API_KEY="$api_key"
        echo "✅ API key set for this session"
    else
        echo "You can set it later with: export MISTRAL_API_KEY='your_key_here'"
    fi
    echo ""
fi

# Check for required dependencies
echo "Checking dependencies..."
python -c "import streamlit, plotly, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing dependencies. Installing..."
    pip install streamlit plotly scikit-learn -q
fi

echo "✅ All dependencies ready"
echo ""
echo "🚀 Launching TELOS Control Panel..."
echo "   Dashboard will open in your browser"
echo "   Press Ctrl+C to stop"
echo ""

# Launch Streamlit
streamlit run telos_purpose/dev_dashboard/streamlit_live_comparison.py
