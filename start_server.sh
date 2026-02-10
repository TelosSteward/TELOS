#!/bin/bash
# Local development startup script for TELOS Observatory
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
source .env 2>/dev/null || export MISTRAL_API_KEY=$(cat .env 2>/dev/null | grep MISTRAL_API_KEY | cut -d'=' -f2)
streamlit run telos_observatory/main.py --server.port 8501
