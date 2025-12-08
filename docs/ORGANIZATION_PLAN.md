# Documentation and Scripts Organization Plan

## Current State
- **Docs**: 132 markdown files, 68 at root level
- **Scripts**: 67 files, mostly flat structure

## Proposed Structure

### Documentation Organization

```
docs/
├── features/              # Implemented features documentation
│   ├── schemas/          # Schema-related features
│   ├── multi_tenant/     # Multi-tenant features (move from root)
│   ├── subscriptions/    # Subscription features (move from root)
│   ├── telemetry/        # Telemetry features (move from root)
│   ├── temporal/         # Temporal workflows (move from root)
│   ├── documents/        # Document ingestion (move from root)
│   └── acl/              # ACL features (move from root)
│
├── guides/               # How-to guides and quickstarts
│   ├── quickstart.md
│   ├── docker.md
│   ├── deployment.md
│   └── integration.md
│
├── architecture/         # Architecture docs (keep existing)
│
├── roadmap/             # Future features (keep and expand)
│   ├── graphql/         # Already exists
│   └── namespace_deployment/  # Already exists
│
├── troubleshooting/      # Troubleshooting guides
│
└── api/                 # API documentation
```

### Scripts Organization

```
scripts/
├── setup/               # Initialization and setup
│   ├── bootstrap_opensource_user.py
│   ├── init_parse_schema_opensource.py
│   ├── init_qdrant_collections_opensource.py
│   ├── setup_api_operation_tracking.py
│   └── docker_entrypoint_opensource.sh
│
├── migration/          # Migration scripts
│   ├── migrate_to_multi_tenant.py
│   ├── migrate_to_multi_tenant_v2.py
│   ├── migrate_batch_memory_request.py
│   ├── cleanup_v1_migration.py
│   └── add_organization_to_*.py
│
├── testing/           # Test and validation scripts
│   ├── test_*.py
│   ├── validate_multi_tenant.py
│   ├── verify_*.py
│   └── check_*.py
│
├── deployment/        # Deployment scripts
│   ├── deploy_workers_azure.sh
│   └── start_services.sh
│
├── debugging/        # Debug and diagnostic scripts
│   ├── debug_*.py
│   ├── diagnose_*.py
│   └── analyze_*.py
│
├── maintenance/      # Cleanup and maintenance
│   ├── cleanup_*.py
│   ├── fix_*.py
│   └── optimize_*.py
│
├── opensource/      # Open source specific
│   ├── bootstrap_opensource_user.py
│   ├── init_parse_schema_opensource.py
│   └── init_qdrant_collections_opensource.py
│
├── generators/      # Code generation scripts
│   ├── generate_*.py
│   └── generate_*.sh
│
├── custom_schema/   # Keep existing
│
└── utils/           # Utility scripts
    ├── calculate_*.py
    └── check_*.sh
```

## Migration Strategy

1. Create new folder structure
2. Move files to appropriate folders
3. Update any references/imports
4. Create README files in each folder explaining purpose
5. Update main README with new structure

