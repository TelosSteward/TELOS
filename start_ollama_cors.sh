#!/bin/bash
# Start Ollama with CORS enabled for Chrome extension

echo "Stopping any existing Ollama instances..."
pkill -9 ollama
sleep 2

echo "Starting Ollama with CORS enabled..."
export OLLAMA_ORIGINS="*"
/Applications/Ollama.app/Contents/Resources/ollama serve

# Note: This will run in foreground. To run in background, add & at the end
# and redirect output: ... serve > /tmp/ollama.log 2>&1 &
