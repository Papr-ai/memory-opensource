#!/bin/bash
# Script to organize docs and scripts into proper folder structure

set -e

echo "📁 Organizing Documentation and Scripts"
echo "========================================"
echo ""

# Create docs folder structure
echo "📚 Creating docs folder structure..."
mkdir -p docs/features/{schemas,multi_tenant,subscriptions,telemetry,temporal,documents,acl,implementation}
mkdir -p docs/guides
mkdir -p docs/troubleshooting
mkdir -p docs/api

# Create scripts folder structure
echo "🔧 Creating scripts folder structure..."
mkdir -p scripts/{setup,migration,testing,deployment,debugging,maintenance,opensource,generators,utils}

echo "✅ Folder structure created"
echo ""

# Note: Actual file moves will be done manually to avoid breaking references
echo "📋 Next steps:"
echo "1. Review the organization plan in docs/ORGANIZATION_PLAN.md"
echo "2. Move files according to categorization"
echo "3. Update any imports/references"
echo "4. Create README files in each folder"

