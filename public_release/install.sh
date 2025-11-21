#!/bin/bash
# Runtime Governance Installation Script

echo "🔧 Installing Runtime Governance"
echo "=================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

echo "Checking Python version..."
echo "Found: Python $PYTHON_VERSION"

# Simple version check
if [[ "$PYTHON_VERSION" < "$REQUIRED_VERSION" ]]; then
    echo "❌ Python 3.9+ required. Please upgrade Python."
    exit 1
fi

echo "✅ Python version OK"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip3 install sentence-transformers numpy

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "=================================="
echo "Next steps:"
echo "=================================="
echo ""
echo "1. Add PA baseline to your .claude_project.md"
echo "   (See claude_project_template.md for example)"
echo ""
echo "2. Initialize a session:"
echo "   python3 runtime_governance_start.py"
echo ""
echo "3. Start using Claude Code normally"
echo "   Checkpoints will run automatically after each response"
echo ""
echo "4. Export session data:"
echo "   python3 runtime_governance_export.py"
echo ""
echo "See QUICK_START.md for detailed instructions."
echo ""
