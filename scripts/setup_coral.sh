#!/bin/bash
# Coralus — Coral Source Registration Script
# Run this once to set up all Coral data sources.

set -e
export PAGER=cat

echo "========================================="
echo "  Coralus — Coral Setup"
echo "========================================="

# Check Coral installation
if ! command -v coral &> /dev/null; then
    echo "   Coral CLI not found."
    echo "   Install with: brew install withcoral/tap/coral"
    echo ""
    echo "   After installing, run this script again."
    exit 1
fi

echo " Coral CLI found: $(coral --version 2>/dev/null || echo 'version unknown')"

# Load GITHUB_TOKEN from .env if not set
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -z "$GITHUB_TOKEN" ] && [ -f "$SCRIPT_DIR/../.env" ]; then
    echo " GITHUB_TOKEN not in environment, loading from $SCRIPT_DIR/../.env..."
    # Read GITHUB_TOKEN from .env
    GITHUB_TOKEN=$(grep -E "^GITHUB_TOKEN=" "$SCRIPT_DIR/../.env" | head -n 1 | cut -d'=' -f2-)
    # Strip quotes if present
    GITHUB_TOKEN=$(echo "$GITHUB_TOKEN" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
    export GITHUB_TOKEN
fi

# 1. Add GitHub connector
echo ""
echo "--- Step 1: Add GitHub connector ---"
if [ -z "$GITHUB_TOKEN" ] || [ "$GITHUB_TOKEN" = "your_github_token_here" ] || [ -z "${GITHUB_TOKEN// }" ]; then
    echo " GITHUB_TOKEN not set or is still a placeholder."
    echo "   Please update GITHUB_TOKEN in your .env file."
    exit 1
else
    echo "Re-configuring GitHub source in Coral..."
    # Remove old github source if exists to update credentials
    coral source remove github &>/dev/null || true
    if coral source add github; then
        echo " GitHub source configured successfully!"
    else
        echo " Failed to add GitHub source."
        exit 1
    fi
fi

# 2. Add PapersWithCode custom source
echo ""
echo "--- Step 2: Add PapersWithCode custom source ---"
SPEC_PATH="$SCRIPT_DIR/../sources/paperswithcode.yaml"

if [ -f "$SPEC_PATH" ]; then
    coral source remove paperswithcode &>/dev/null || true
    if coral source add paperswithcode --spec "$SPEC_PATH" &>/dev/null; then
        echo " PapersWithCode source added"
    else
        echo "  PapersWithCode source add failed or already exists"
    fi
else
    echo "  Source spec not found at $SPEC_PATH"
fi

# 3. Verify sources
echo ""
echo "--- Step 3: Verify sources ---"
coral source list || echo "  Could not list sources"

# 4. Test a simple query
echo ""
echo "--- Step 4: Test query ---"
echo "Testing GitHub connection..."
if coral sql "SELECT number, title, state, created_at FROM github.issues WHERE owner = 'withcoral' AND repo = 'coral' AND state = 'open' LIMIT 5"; then
    echo " GitHub queries working!"
else
    echo " Test query failed — check your GITHUB_TOKEN or internet connection."
    exit 1
fi

echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/test_full_pipeline.py"
echo "  2. Or start the UI: streamlit run ui/app.py"
echo ""
echo "To start Coral MCP server for agent integration:"
echo "  coral mcp start"
