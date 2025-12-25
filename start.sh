#!/bin/bash
# Railway startup script - pre-downloads embedding models before starting Streamlit

echo "=== TELOS Observatory Startup ==="
echo "Pre-downloading SentenceTransformer models..."

# Download models (they get cached in the container)
python3 -c "
from sentence_transformers import SentenceTransformer
print('Downloading all-MiniLM-L6-v2...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Downloading all-mpnet-base-v2...')
SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print('Models ready!')
"

echo "Starting Streamlit..."
streamlit run telos_observatory_v3/main.py --server.port $PORT --server.address 0.0.0.0
