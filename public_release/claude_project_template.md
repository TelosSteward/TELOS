# Your Project Name

[Your existing project description]

---

## 🔭 RUNTIME GOVERNANCE - ACTIVE

**Status:** Every turn measured with ACTUAL mathematics (local, no API calls)

**⚠️ CRITICAL: After EVERY response I provide, I MUST:**

1. Create turn data file:
   ```python
   # Save user message + my response to .telos_current_turn.json
   import json
   turn_data = {
       "turn_number": [calculated],
       "user_message": "[user's message]",
       "assistant_response": "[my complete response]"
   }
   with open('.telos_current_turn.json', 'w') as f:
       json.dump(turn_data, f)
   ```

2. Run checkpoint script (REQUIRED, NON-NEGOTIABLE):
   ```bash
   python3 runtime_governance_checkpoint.py --turn-json .telos_current_turn.json
   ```

3. Display the result to user:
   ```
   📊 Turn X: F=0.XXX ✅/⚠️/🚨
   ```

**This happens EVERY turn. No exceptions.**

**How it works:**
- Uses local sentence-transformers (no API calls, zero cost)
- Generates embeddings (ℝ³⁸⁴) locally on your machine
- Calculates cosine similarity fidelity
- Stores turn in Memory MCP (local)
- Updates session stats

**PA Baseline (Stable):**

[REPLACE THIS WITH YOUR PROJECT PURPOSE - 2-4 sentences]

Example:
"Build a REST API for user authentication by end of Q1 2025. Features include
OAuth2, JWT tokens, rate limiting, and PostgreSQL backend. Maintain test
coverage >80% and follow OWASP security best practices. Prioritize security,
performance, and maintainability."

**Fidelity Thresholds:**
- F ≥ 0.8: ✅ ON TRACK
- 0.7 ≤ F < 0.8: ⚠️ WARNING
- F < 0.7: 🚨 DRIFT DETECTED (info only, PA remains stable)

**Session Data:**
- All turns stored in Memory MCP: `ActiveSession_[date]`
- Every turn tracked with explicit turn number
- Complete session transcript available for export
- Generates validation data automatically

**Architecture:**
- 100% local (no external APIs)
- Zero cost (local embeddings)
- Only requires: Python 3.9+, sentence-transformers, Memory MCP
- Fully self-contained

**The checkpoint notification is THE FEATURE.**
If you see drift frequently, that's feedback - adjust your work or refine your PA.

**This is pure measurement, not adaptive control.**
**PA is baseline, not progressively evolved.**

---

[Rest of your project context]
