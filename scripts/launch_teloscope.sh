#!/bin/bash
# TELOSCOPE Observatory Launch Script
# =====================================

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔭 TELOSCOPE Observatory${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "config.json" ]; then
    echo -e "${RED}❌ Error: config.json not found${NC}"
    echo "Please run this script from the telos project root directory"
    exit 1
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo -e "${GREEN}✅ Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${RED}❌ Error: Virtual environment not found${NC}"
    echo "Please create venv first: python -m venv venv"
    exit 1
fi

# Check for MISTRAL_API_KEY
if [ -z "$MISTRAL_API_KEY" ]; then
    echo -e "${RED}⚠️  Warning: MISTRAL_API_KEY not set${NC}"
    echo "Please set it with: export MISTRAL_API_KEY='your_key_here'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for required dependencies
echo -e "${GREEN}✅ Checking dependencies...${NC}"
python -c "import streamlit, plotly, scipy, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Missing dependencies${NC}"
    echo "Installing required packages..."
    pip install streamlit plotly scipy pandas
fi

# Launch Streamlit
echo ""
echo -e "${GREEN}✅ Launching TELOSCOPE Observatory...${NC}"
echo ""
echo -e "${BLUE}📊 Dashboard will open at: http://localhost:8501${NC}"
echo -e "${BLUE}📝 Press Ctrl+C to stop${NC}"
echo ""

streamlit run telos_purpose/dev_dashboard/streamlit_teloscope.py
