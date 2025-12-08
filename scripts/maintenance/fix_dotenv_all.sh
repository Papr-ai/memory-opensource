#!/bin/bash
# Script to fix all remaining load_dotenv() calls to respect USE_DOTENV

echo "🔧 Fixing all remaining load_dotenv() calls to respect USE_DOTENV..."

# List of files already fixed
FIXED_FILES=(
    "main.py"
    "config/features.py"
    "services/stripe_service.py"
    "tests/test_add_memory_fastapi.py"
    "app_factory.py"
    "memory/memory_graph.py"
    "services/user_utils.py"
    "routers/v1/document_routes_v2.py"
    "routers/v1/memory_routes_v1.py"
    "start_all_workers.py"
    "start_temporal_worker.py"
    "start_document_worker.py"
    "cloud_plugins/temporal/activities/document_activities.py"
    "api_handlers/chat_gpt_completion.py"
    "routers/v1/sync_routes.py"
    "routers/v1/feedback_routes.py"
    "routers/v1/graphql_routes.py"
    "routers/v1/schema_routes_v1.py"
    "core/document_processing/provider_manager.py"
    "core/document_processing/providers/tensorlake.py"
    "core/document_processing/llm_memory_generator.py"
)

# Function to check if file is already fixed
is_fixed() {
    local file="$1"
    for fixed in "${FIXED_FILES[@]}"; do
        if [[ "$file" == *"$fixed" ]]; then
            return 0
        fi
    done
    return 1
}

# Find all Python files with load_dotenv
find . -name "*.py" -type f -exec grep -l "load_dotenv" {} \; | while read -r file; do
    # Skip already fixed files
    if is_fixed "$file"; then
        echo "✅ Already fixed: $file"
        continue
    fi
    
    echo "🔧 Fixing: $file"
    
    # Create backup
    cp "$file" "${file}.backup"
    
    # Use sed to replace load_dotenv patterns
    # Pattern 1: Simple load_dotenv(ENV_FILE) after find_dotenv
    sed -i '' 's/ENV_FILE = find_dotenv()/use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"\nif use_dotenv:\n    ENV_FILE = find_dotenv()/g' "$file"
    
    # Pattern 2: load_dotenv(ENV_FILE) after if ENV_FILE:
    sed -i '' 's/if ENV_FILE:\n    load_dotenv(ENV_FILE)/if use_dotenv:\n    if ENV_FILE:\n        load_dotenv(ENV_FILE)/g' "$file"
    
    # Pattern 3: Direct load_dotenv() calls
    sed -i '' 's/load_dotenv()/use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"\nif use_dotenv:\n    load_dotenv()/g' "$file"
    
    echo "✅ Fixed: $file"
done

echo "🎉 All load_dotenv() calls should now respect USE_DOTENV!"
echo "⚠️  Please review the changes and test thoroughly!"
