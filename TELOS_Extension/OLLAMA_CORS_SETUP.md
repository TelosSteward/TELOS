# Ollama CORS Setup for Chrome Extension

## The Problem

By default, Ollama doesn't allow requests from Chrome extensions due to CORS (Cross-Origin Resource Sharing) restrictions. This causes "Forbidden" errors when the extension tries to connect to `localhost:11434`.

## Solution: Start Ollama with CORS Enabled

### Option 1: Terminal Command (Recommended for Development)

1. **Stop Ollama** (if running):
   ```bash
   pkill -9 ollama
   ```

2. **Start with CORS enabled**:
   ```bash
   OLLAMA_ORIGINS="*" /Applications/Ollama.app/Contents/Resources/ollama serve
   ```

   This will run in the foreground. Keep the terminal open.

3. **Or run in background**:
   ```bash
   OLLAMA_ORIGINS="*" /Applications/Ollama.app/Contents/Resources/ollama serve > /tmp/ollama.log 2>&1 &
   ```

### Option 2: Use the Startup Script

Run the provided script:
```bash
cd /Users/brunnerjf/Desktop/Privacy_PreCommit
./start_ollama_cors.sh &
```

### Option 3: Set Environment Variable Globally (Permanent)

Add to your `~/.zshrc` or `~/.bash_profile`:
```bash
export OLLAMA_ORIGINS="*"
```

Then restart Ollama from the macOS app.

## Verification

Test that Ollama is accessible:
```bash
curl http://localhost:11434/api/tags
```

Should return JSON with available models.

## Security Note

`OLLAMA_ORIGINS="*"` allows ALL origins to access Ollama. This is fine for local development but less secure. For production, use:
```bash
OLLAMA_ORIGINS="chrome-extension://YOUR_EXTENSION_ID"
```

Get your extension ID from `chrome://extensions/`.

## Alternative: Use Ollama via Python Backend

If CORS continues to be problematic, consider:
1. Creating a Python FastAPI server that talks to Ollama
2. Extension connects to your Python server (easier CORS control)
3. Python server proxies requests to Ollama

This is more robust for production deployments.

---

**Current Status**: CORS not yet working. Extension shows "Forbidden" error.
**Next Step**: Manually run Option 1 in a terminal window, then test extension.
