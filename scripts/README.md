# Scripts

This directory contains utility scripts organized by purpose.

## Structure

- **setup/** - Initialization and setup scripts (Parse schema, Qdrant collections, indexes)
- **migration/** - Data migration scripts (multi-tenant, batch memory, relationships)
- **testing/** - Test and validation scripts
- **deployment/** - Deployment scripts (Azure workers, service management)
- **debugging/** - Debug and diagnostic scripts
- **maintenance/** - Cleanup and maintenance scripts
- **opensource/** - Open source specific setup scripts
- **generators/** - Code generation scripts (API keys, GraphQL, OpenAPI)
- **utils/** - Utility scripts (calculations, checks, helpers)
- **custom_schema/** - Custom schema creation scripts

## Usage

Most scripts can be run directly:

```bash
# Setup scripts
python scripts/setup/init_parse_schema_opensource.py

# Migration scripts
python scripts/migration/migrate_to_multi_tenant_v2.py

# Testing scripts
python scripts/testing/validate_multi_tenant.py

# Debugging scripts
python scripts/debugging/debug_mongo_connection.py
```

## Script Categories

### Setup Scripts
Scripts for initializing services, creating indexes, and setting up the environment.

### Migration Scripts
Scripts for migrating data between versions or adding new features to existing data.

### Testing Scripts
Scripts for validating functionality, testing migrations, and verifying configurations.

### Deployment Scripts
Scripts for deploying services, managing workers, and controlling service lifecycle.

### Debugging Scripts
Scripts for diagnosing issues, analyzing performance, and checking service health.

### Maintenance Scripts
Scripts for cleanup, optimization, and fixing data issues.

### Generators
Scripts that generate code, schemas, or configuration files.

### Utils
General utility scripts for calculations, checks, and helper functions.
