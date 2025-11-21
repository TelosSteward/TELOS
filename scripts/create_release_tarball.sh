#!/bin/bash
# Create distributable tarball for Runtime Governance

cd /Users/brunnerjf/Desktop/telos_privacy

RELEASE_NAME="runtime-governance-v0.1.0"
RELEASE_DIR="public_release"

echo "📦 Creating Release Tarball"
echo "==========================="
echo ""

# Create tarball
echo "Packaging $RELEASE_DIR..."
tar -czf "${RELEASE_NAME}.tar.gz" -C "$RELEASE_DIR" .

if [ $? -eq 0 ]; then
    echo "✅ Tarball created: ${RELEASE_NAME}.tar.gz"
    echo ""

    # Show file size
    SIZE=$(du -h "${RELEASE_NAME}.tar.gz" | cut -f1)
    echo "Size: $SIZE"
    echo ""

    # Show contents
    echo "Contents:"
    tar -tzf "${RELEASE_NAME}.tar.gz" | head -20
    echo ""

    echo "==========================="
    echo "✅ READY FOR DISTRIBUTION"
    echo "==========================="
    echo ""
    echo "To use:"
    echo "  1. Upload to GitHub releases"
    echo "  2. Users download: ${RELEASE_NAME}.tar.gz"
    echo "  3. Extract: tar -xzf ${RELEASE_NAME}.tar.gz"
    echo "  4. Run: ./install.sh"
    echo ""
    echo "Or create GitHub repo:"
    echo "  1. cd public_release"
    echo "  2. git init"
    echo "  3. git add ."
    echo "  4. git commit -m 'Initial release: Runtime Governance v0.1.0'"
    echo "  5. gh repo create telos-project/runtime-governance --public"
    echo "  6. git push origin main"
    echo ""
else
    echo "❌ Failed to create tarball"
    exit 1
fi
