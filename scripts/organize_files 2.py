#!/usr/bin/env python3
"""
Organize docs and scripts into proper folder structure
"""
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Docs organization mapping
DOCS_MOVES = {
    # Schema files
    "CUSTOM_SCHEMA_IMPLEMENTATION_GUIDE.md": "features/schemas/",
    "CUSTOM_SCHEMA_QUICK_START.md": "features/schemas/",
    "SCHEMA_AUTO_DISCOVERY_VS_CUSTOM.md": "features/schemas/",
    "SCHEMA_DOCUMENT_WORKFLOW_FIX.md": "features/schemas/",
    "SCHEMA_FLOW_DIAGRAM.md": "features/schemas/",
    "SCHEMA_ID_IMPLEMENTATION_SUMMARY.md": "features/schemas/",
    "SCHEMA_IMPLEMENTATION_SUMMARY.md": "features/schemas/",
    "schema_lifecycle_flow.md": "features/schemas/",
    "parse_schema_setup.md": "features/schemas/",
    "README_SCHEMA_TESTING.md": "features/schemas/",
    
    # Subscription files
    "API_KEY_LAST_USED_UPDATE.md": "features/subscriptions/",
    "RATE_LIMITS_REFERENCE.md": "features/subscriptions/",
    "RATE_LIMITS_UPDATE_SUMMARY.md": "features/subscriptions/",
    "SUBSCRIPTION_LIMITS_IMPLEMENTATION.md": "features/subscriptions/",
    "SUBSCRIPTION_LIMITS_UPDATE_SUMMARY.md": "features/subscriptions/",
    "TIER_LIMITS_FIX_SUMMARY.md": "features/subscriptions/",
    
    # Document files
    "DOCUMENT_INGESTION_FEES.md": "features/documents/",
    "DOCUMENT_UPLOAD_WITH_SCHEMA.md": "features/documents/",
    "HIERARCHICAL_CHUNKING_FLOW.md": "features/documents/",
    "PROVIDER_SDK_INTEGRATION_GUIDE.md": "features/documents/",
    "PROVIDER_TEST_RESULTS.md": "features/documents/",
    "PROVIDER_UNIT_TESTS_README.md": "features/documents/",
    "RUN_PROVIDER_TESTS.md": "features/documents/",
    "IMAGE_URL_COMPLETE_FLOW.md": "features/documents/",
    
    # Implementation summaries
    "IMPLEMENTATION_COMPLETE.md": "features/implementation/",
    "IMPLEMENTATION_COMPLETE_OPTION_A.md": "features/implementation/",
    "PHASE1_COMPLETE_SUMMARY.md": "features/implementation/",
    "PHASE1_COMPLETE.md": "features/implementation/",
    "PHASE1_IMPLEMENTATION_COMPLETE.md": "features/implementation/",
    "BATCH_IMPLEMENTATION_SUMMARY.md": "features/implementation/",
    "COMPLETE_PARAMETER_UPDATE_SUMMARY.md": "features/implementation/",
    "INTERACTION_LIMITS_CONFIG_UPDATE.md": "features/implementation/",
    "INTERACTION_LIMITS_PROGRESS.md": "features/implementation/",
    "INTERACTION_POINTERS_UPDATE.md": "features/implementation/",
    "OPERATION_TRACKING.md": "features/implementation/",
    "PR_SUMMARY.md": "features/implementation/",
    "PYPROJECT_UPDATES.md": "features/implementation/",
    
    # Guides
    "DOCKER_COMMANDS_QUICKREF.md": "guides/",
    "DOCKER_DEPLOYMENT.md": "guides/",
    "DOCKER_QUICK_START.md": "guides/",
    "DEPLOYMENT.md": "guides/",
    "AZURE_DEPLOYMENT_ARCHITECTURE.md": "guides/",
    "INTEGRATION_EXAMPLE.md": "guides/",
    "WEBHOOK_GUIDE.md": "guides/",
    "MIGRATION_GUIDE.md": "guides/",
    "FOLDER_MIGRATION_GUIDE.md": "guides/",
    
    # Troubleshooting
    "PARSE_SERVER_FILE_UPLOAD_TROUBLESHOOTING.md": "troubleshooting/",
    "MONGODB_SSL_FIX.md": "troubleshooting/",
    "MEMORY_CHUNK_IDS_FIX_SUMMARY.md": "troubleshooting/",
    "RELEVANCE_SCORE_FIX.md": "troubleshooting/",
    "VERSIONING_FIX.md": "troubleshooting/",
    "TENSORLAKE_CONTENT_FIX_SUMMARY.md": "troubleshooting/",
    "TENSORLAKE_FIX_SUMMARY.md": "troubleshooting/",
    
    # Other features
    "GROQ_LLM_OPTIMIZATION.md": "features/",
    "SYNC_EMBEDDING_IMPROVEMENTS.md": "features/",
    "LLM_CALLS_DOCUMENTATION.md": "features/",
    "NODE_CONSTRAINT_BEHAVIOR_RULES.md": "features/",
    "confidence_weighting_proof.md": "features/",
    "reducto_memory_optimization_strategy.md": "features/",
    "WORKER_FIXED.md": "troubleshooting/",
}

# Scripts organization mapping
SCRIPTS_MOVES = {
    # Setup
    "bootstrap_opensource_user.py": "opensource/",
    "init_parse_schema_opensource.py": "opensource/",
    "init_qdrant_collections_opensource.py": "opensource/",
    "docker_entrypoint_opensource.sh": "opensource/",
    "setup_api_operation_tracking.py": "setup/",
    "create_api_key_index.py": "setup/",
    "add_parse_indexes.py": "setup/",
    
    # Migration
    "migrate_to_multi_tenant.py": "migration/",
    "migrate_to_multi_tenant_v2.py": "migration/",
    "migrate_batch_memory_request.py": "migration/",
    "cleanup_v1_migration.py": "migration/",
    "add_organization_to_developer_users.py": "migration/",
    "add_organization_to_workspaces.py": "migration/",
    "add_relationships.py": "migration/",
    "add_relationships_v2.py": "migration/",
    
    # Testing
    "test_migration_safe.py": "testing/",
    "test_schema_enforcement.py": "testing/",
    "test_document_with_schema_id.py": "testing/",
    "test_api_latency.py": "testing/",
    "test_production_config.py": "testing/",
    "validate_multi_tenant.py": "testing/",
    "verify_phase1.py": "testing/",
    "verify_temporal_integration.py": "testing/",
    
    # Deployment
    "deploy_workers_azure.sh": "deployment/",
    "start_services.sh": "deployment/",
    "stop_all_services.sh": "deployment/",
    
    # Debugging
    "debug_agentic_search.py": "debugging/",
    "debug_env_vars.py": "debugging/",
    "debug_mongo_connection.py": "debugging/",
    "diagnose_mongodb_latency.py": "debugging/",
    "diagnose_startup.sh": "debugging/",
    "analyze_retrieval_sources.py": "debugging/",
    "check_neo4j_memory.py": "debugging/",
    "check_worker_logs.py": "debugging/",
    "check_worker_logs.sh": "debugging/",
    
    # Maintenance
    "cleanup_duplicate_batch_posts.py": "maintenance/",
    "fix_duplicate_api_keys.py": "maintenance/",
    "fix_dotenv_all.sh": "maintenance/",
    "optimize_mongodb_connection.py": "maintenance/",
    
    # Generators
    "generate_api_key.py": "generators/",
    "generate_graphql.sh": "generators/",
    "generate_openapi.py": "generators/",
    "generate_parse_schema_from_api.py": "generators/",
    "generate_parse_schema_from_atlas.py": "generators/",
    "generate_test_jwt.py": "generators/",
    "pydantic_to_graphql.py": "generators/",
    
    # Utils
    "calculate_memory_storage_size.py": "utils/",
    "calculate_memory_token_size.py": "utils/",
    "calculate_total_tokens.py": "utils/",
    "calculate_workspace_storage_count.py": "utils/",
    "check_api_keys.sh": "utils/",
    "check_provider_sdks.py": "utils/",
    "add_interaction_limits_template.py": "utils/",
    "add_agent_learning.py": "utils/",
    "create_test_schemas.py": "utils/",
    "show_schema_id_payload.py": "utils/",
    "quick_schema_test.py": "utils/",
    "detect_duplicates.py": "utils/",
    "mongodb_health_check.py": "utils/",
    "prepare_open_source.py": "utils/",
    "eval_timeout_config.py": "utils/",
}

def move_files(moves_dict, base_dir, file_type="docs"):
    """Move files according to the mapping"""
    moved = 0
    skipped = 0
    errors = 0
    
    for filename, target_dir in moves_dict.items():
        source = base_dir / filename
        target = base_dir / target_dir / filename
        
        if not source.exists():
            skipped += 1
            continue
            
        try:
            # Create target directory if it doesn't exist
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(source), str(target))
            print(f"✅ Moved {filename} → {target_dir}")
            moved += 1
        except Exception as e:
            print(f"❌ Error moving {filename}: {e}")
            errors += 1
    
    print(f"\n📊 Summary: {moved} moved, {skipped} skipped, {errors} errors")
    return moved, skipped, errors

def main():
    print("📁 Organizing Documentation and Scripts")
    print("=" * 50)
    
    # Organize docs
    print("\n📚 Organizing documentation...")
    docs_moved, docs_skipped, docs_errors = move_files(DOCS_MOVES, DOCS_DIR, "docs")
    
    # Organize scripts
    print("\n🔧 Organizing scripts...")
    scripts_moved, scripts_skipped, scripts_errors = move_files(SCRIPTS_MOVES, SCRIPTS_DIR, "scripts")
    
    print("\n" + "=" * 50)
    print("✅ Organization complete!")
    print(f"   Docs: {docs_moved} files moved")
    print(f"   Scripts: {scripts_moved} files moved")

if __name__ == "__main__":
    main()

