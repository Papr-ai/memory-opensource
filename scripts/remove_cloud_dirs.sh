#!/bin/bash
# Remove cloud-specific directories for open source preparation

set -e

echo "🗑️  Removing cloud-specific directories..."
echo "=========================================="
echo ""

# Check if directories exist
if [ -d "cloud_plugins" ]; then
    echo "Removing cloud_plugins/..."
    rm -rf cloud_plugins/
    echo "✅ Removed cloud_plugins/"
else
    echo "ℹ️  cloud_plugins/ does not exist"
fi

if [ -d "cloud_scripts" ]; then
    echo "Removing cloud_scripts/..."
    rm -rf cloud_scripts/
    echo "✅ Removed cloud_scripts/"
else
    echo "ℹ️  cloud_scripts/ does not exist"
fi

echo ""
echo "=========================================="
echo "✅ Cloud directories removed!"
echo ""
echo "⚠️  Next steps:"
echo "   1. Verify tests pass: PAPR_EDITION=opensource poetry run pytest"
echo "   2. Build Docker: docker-compose -f docker-compose-open-source.yaml build"
echo "   3. Check for any remaining cloud references"

