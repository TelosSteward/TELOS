#!/bin/bash
# Package Runtime Governance for public release

echo "📦 Packaging Runtime Governance for Release"
echo "============================================"
echo ""

# Create release directory
RELEASE_DIR="claude_runtime_governance_v0.1"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

echo "1. Copying core files..."

# Core governance files
cp telos_session_start.py "$RELEASE_DIR/"
cp telos_turn_checkpoint.py "$RELEASE_DIR/"
cp telos_session_export.py "$RELEASE_DIR/"

# Dependencies
cp embedding_provider.py "$RELEASE_DIR/"
cp dual_attractor.py "$RELEASE_DIR/"

# Documentation
cp RUNTIME_GOVERNANCE_README.md "$RELEASE_DIR/README.md"
cp QUICK_START.md "$RELEASE_DIR/"
cp MONETIZATION_ANALYSIS.md "$RELEASE_DIR/"
cp LANDING_PAGE.md "$RELEASE_DIR/"

# Config examples
cp governance_config.example.json "$RELEASE_DIR/"

# License
cat > "$RELEASE_DIR/LICENSE" << 'EOL'
MIT License

Copyright (c) 2025 TELOS Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOL

# Create requirements.txt
cat > "$RELEASE_DIR/requirements.txt" << 'EOL'
sentence-transformers>=2.2.0
numpy>=1.24.0
EOL

# Create setup script
cat > "$RELEASE_DIR/install.sh" << 'EOL'
#!/bin/bash
echo "🔧 Installing Runtime Governance Dependencies"
pip install -r requirements.txt
echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Add PA baseline to your .claude_project.md"
echo "  2. Run: python3 telos_session_start.py"
echo "  3. Start using Claude Code"
echo ""
echo "See QUICK_START.md for detailed instructions."
EOL

chmod +x "$RELEASE_DIR/install.sh"

# Create example .claude_project.md template
cat > "$RELEASE_DIR/claude_project_template.md" << 'EOL'
# Your Project Name

## 🔭 RUNTIME GOVERNANCE - ACTIVE

**PA Baseline (Stable):**
[YOUR PROJECT PURPOSE HERE - 2-4 sentences describing purpose, scope, boundaries]

Example:
"Build a REST API for user authentication by end of Q1. Features include
OAuth2, JWT tokens, rate limiting, and PostgreSQL backend. Maintain test
coverage >80% and follow OWASP security best practices."

**Fidelity Thresholds:**
- F ≥ 0.8: ✅ ON TRACK
- 0.7 ≤ F < 0.8: ⚠️ WARNING
- F < 0.7: 🚨 DRIFT DETECTED

**After every turn:**
1. Store turn in Memory MCP
2. Calculate fidelity (actual mathematics)
3. Report: 📊 Turn X: F=0.XXX ✅/⚠️/🚨

---

[REST OF YOUR PROJECT CONTEXT]
EOL

echo "✅ Core files copied"
echo ""

echo "2. Creating archive..."
tar -czf "${RELEASE_DIR}.tar.gz" "$RELEASE_DIR"
echo "✅ Archive created: ${RELEASE_DIR}.tar.gz"
echo ""

echo "3. File listing:"
echo "---"
cd "$RELEASE_DIR"
ls -lh
cd ..
echo "---"
echo ""

echo "4. Release checklist:"
echo "   [ ] Test installation on fresh system"
echo "   [ ] Verify all scripts run"
echo "   [ ] Check documentation completeness"
echo "   [ ] Update version numbers"
echo "   [ ] Create GitHub release"
echo "   [ ] Announce on HN/Reddit/Twitter"
echo ""

echo "📦 Package ready: ${RELEASE_DIR}.tar.gz"
echo ""
echo "To distribute:"
echo "  - Upload to GitHub releases"
echo "  - Share download link"
echo "  - Post announcement with README contents"
echo ""
echo "Installation for users:"
echo "  1. Download and extract: tar -xzf ${RELEASE_DIR}.tar.gz"
echo "  2. cd ${RELEASE_DIR}"
echo "  3. ./install.sh"
echo "  4. Follow QUICK_START.md"
echo ""
echo "✅ READY FOR RELEASE"
