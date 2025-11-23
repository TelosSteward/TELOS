# TELOS Chrome Extension

**Version**: 1.0.0
**Status**: Rebuilt from specifications (November 21, 2025)
**Purpose**: Local LLM governance with Ollama, bypassing API rate limits

---

## Overview

The TELOS Chrome Extension enables local governance of LLM conversations using Ollama, eliminating API rate limits and providing cryptographic validation through telemetric signatures.

### Key Features

✅ **Local Ollama Integration** - Run governance on your own hardware
✅ **3-Tier Governance System** - PA/RAG/Expert escalation
✅ **Telemetric Signatures** - Quantum-resistant cryptographic signing
✅ **Zero Rate Limits** - All processing happens locally
✅ **Privacy-First** - No data sent to external APIs
✅ **Beautiful UI** - Cyberpunk-inspired interface

---

## Architecture

```
┌─────────────┐
│   Popup UI  │ (User Interface)
└──────┬──────┘
       │
       v
┌─────────────┐
│ Background  │ (Service Worker - Coordinates everything)
│   Worker    │
└──────┬──────┘
       │
       ├──> Ollama (localhost:11434) - LLM inference
       ├──> Telemetric Signer - Cryptographic validation
       └──> Chrome Storage - Session state
```

### File Structure

```
TELOS_Extension/
├── manifest.json                    # Extension configuration (Manifest V3)
├── background.js                    # Service worker (governance logic)
├── content.js                       # Content script (page injection)
├── popup.html                       # Extension popup UI
├── popup.js                         # Popup controller
├── lib/
│   └── telemetric-signatures-mvp.js # Cryptographic signing
├── icons/
│   ├── icon16.png                   # (Placeholder - add your icons)
│   ├── icon48.png
│   └── icon128.png
└── README.md                        # This file
```

---

## Installation

### 1. Prerequisites

- **Ollama** running locally:
  ```bash
  ollama serve
  ollama pull mistral:latest
  ```

- **Chrome** or **Edge** browser (Chromium-based)

### 2. Load Extension

1. Open Chrome
2. Navigate to `chrome://extensions/`
3. Enable **Developer mode** (top right)
4. Click **Load unpacked**
5. Select the `TELOS_Extension` folder
6. Extension icon should appear in toolbar

### 3. Add Icons (Optional)

The extension will work without icons, but for a polished look:
- Add 16x16px icon as `icons/icon16.png`
- Add 48x48px icon as `icons/icon48.png`
- Add 128x128px icon as `icons/icon128.png`

Suggested design: TELOS logo with cyberpunk aesthetic (blue/purple gradient)

---

## Usage

### Initialize Session

1. Click the TELOS extension icon
2. Verify "Ollama: Connected" status
3. Click **Initialize Session**
4. Session is now active with telemetric signatures enabled

### Test Governance

1. Enter a test message in the popup
2. Click **Send & Govern**
3. View governance tier, fidelity, and response
4. Each turn is cryptographically signed

### Session Info

- **Session ID**: Unique identifier
- **Turns Processed**: Number of governed messages
- **Telemetric Signatures**: Always enabled for IP protection

---

## Technical Details

### Governance Tiers

| Tier | Name | Fidelity Range | Method |
|------|------|----------------|--------|
| 1 | PA Autonomous | ≥ 0.18 | Primacy Attractor alignment |
| 2 | RAG Enhanced | 0.12 - 0.18 | RAG + PA guidance |
| 3 | Expert Escalation | < 0.12 | Human review required |

### Telemetric Signatures

Based on `telemetric_keys_quantum.py`:
- **Algorithm**: SHA-512 + HMAC
- **Security**: 256-bit post-quantum resistance
- **Entropy Sources**:
  - Response timing (delta_t)
  - Fidelity scores
  - Embedding distances
  - Message lengths
  - Browser performance metrics
  - Cryptographic random
- **Forward Secrecy**: One-way key rotation per turn
- **IP Protection**: Non-reproducible signatures prove prior art

### API Endpoints (Ollama)

- `http://localhost:11434/api/generate` - Text generation
- `http://localhost:11434/api/embeddings` - Fidelity calculation (future)
- `http://localhost:11434/api/tags` - List available models

---

## Future Enhancements

### Phase 2
- [ ] ChatGPT/Claude.ai page injection
- [ ] Intercept API calls and govern locally
- [ ] Visual governance overlay on chat interfaces
- [ ] Export session signatures for verification

### Phase 3
- [ ] Multi-model support (Llama, Phi, Gemma)
- [ ] RAG corpus integration
- [ ] Real-time fidelity calculation with embeddings
- [ ] Supabase sync for cross-device sessions

### Phase 4
- [ ] Expert review queue UI
- [ ] Collaborative governance with team
- [ ] Advanced telemetric analysis dashboard
- [ ] Chrome Web Store publication

---

## Development

### Testing Locally

```bash
# Start Ollama
ollama serve

# Pull required model
ollama pull mistral:latest

# Load extension in Chrome (see Installation)

# Open popup and test
```

### Debugging

- **Background worker logs**: `chrome://extensions` → TELOS → "service worker" link
- **Content script logs**: Browser DevTools Console (F12)
- **Popup logs**: Right-click popup → "Inspect"

### Modifying

- **Change governance thresholds**: Edit `background.js` → `tier1Threshold`, `tier2Threshold`
- **Update UI**: Edit `popup.html` and `popup.css`
- **Add models**: Update `MISTRAL_MODEL` in `background.js`

---

## Security Notes

### What This Extension Does

✅ Runs locally - no external API calls
✅ Processes messages through Ollama (localhost)
✅ Generates cryptographic signatures
✅ Stores session state in Chrome storage

### What This Extension Does NOT Do

❌ Send data to external servers
❌ Track your browsing
❌ Access sensitive information
❌ Modify webpage content (yet)

### Permissions Explained

- `storage` - Save session state and preferences
- `activeTab` - Future: Inject governance into chat pages
- `host_permissions` - Connect to local Ollama instance

---

## Credits

**Rebuilt**: November 21, 2025
**Original Design**: TELOS Session Handoff Documentation
**Cryptography**: Based on `telemetric_keys_quantum.py`
**Framework**: TELOS 3-Tier Governance System

---

## Support

- **Issues**: Check Ollama is running (`ollama serve`)
- **Rate Limits**: None! All local processing
- **Performance**: Depends on your hardware (Ollama speed)

---

## License

Part of the TELOS project. See main repository for license details.

---

**Status**: Ready for testing
**Next Steps**: Add icons, test with Ollama, iterate on UI
**Deploy**: Chrome Web Store (Phase 4)
