#!/bin/bash
# Wrapper script to run pydantic_to_graphql.py with Poetry environment

set -e

cd "$(dirname "$0")/.."

echo "🔄 Generating GraphQL schema from Pydantic models..."
echo ""

poetry run python scripts/pydantic_to_graphql.py "$@"

echo ""
echo "✅ Done!"
