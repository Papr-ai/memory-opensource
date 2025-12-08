# Open Source Repository Management Guide

## Overview

This guide explains how to manage the open-source edition of Papr Memory as a separate repository while maintaining code synchronization with the cloud edition.

## Repository Strategy

### Recommended Approach: Separate Repositories with Shared Core

```
papr-memory-cloud/          # Private cloud repository
├── cloud_plugins/          # Cloud-only features
├── cloud_scripts/          # Cloud maintenance scripts
├── config/cloud.yaml       # Cloud configuration
└── [all shared code]

papr-memory/                # Public open-source repository
├── config/opensource.yaml  # Open-source configuration
└── [shared code only]
```

### Benefits

1. **Clear Separation**: Open-source users don't see cloud-specific code
2. **Security**: Cloud secrets/configs never exposed
3. **Simpler OSS**: Cleaner codebase for contributors
4. **Independent Versioning**: OSS can have its own release cycle

## What Can Be Safely Removed for Open-Source

### ✅ Safe to Remove

#### 1. **Directories**
- `cloud_plugins/` - All cloud-only plugins (Stripe, Temporal, Azure, etc.)
- `cloud_scripts/` - Cloud maintenance and admin scripts

#### 2. **Configuration Files**
- `config/cloud.yaml` - Cloud-specific configuration

#### 3. **Docker Files** (if you have cloud-specific ones)
- `docker-compose.yaml` (keep `docker-compose-open-source.yaml`)

#### 4. **CI/CD Files** (cloud-specific)
- `azure-pipelines.yml` or other cloud CI configs

### ⚠️ Conditional Imports (Already Safe)

These files use conditional imports, so they're safe to keep:
- `routers/v1/document_routes_v2.py` - Uses `try/except` for Temporal
- `services/batch_processor.py` - Conditionally loads Temporal
- `tests/test_add_memory_fastapi.py` - Tests skip if cloud plugins unavailable

### ✅ Keep These Files

- `config/features.py` - Handles edition detection (needs `cloud.yaml` reference removed)
- `config/opensource.yaml` - Open-source configuration
- `config/base.yaml` - Shared configuration
- All core services and routers (they use feature flags)

## Migration Steps

### Step 1: Prepare Open-Source Distribution

Use the existing preparation script:

```bash
poetry run python scripts/prepare_open_source.py --output ../memory-oss
```

This script will:
- ✅ Remove `cloud_plugins/` directory
- ✅ Remove `cloud_scripts/` directory  
- ✅ Remove `config/cloud.yaml`
- ✅ Scan for potential secrets
- ✅ Create OSS-specific files (SECURITY.md, etc.)

### Step 2: Update `config/features.py`

After removing `cloud.yaml`, update the config loader to handle missing file gracefully:

```python
# In config/features.py, line 69-72
if self.edition == "cloud":
    self.edition_config = self._load_config("cloud.yaml")
    if not self.edition_config:
        logger.warning("cloud.yaml not found - cloud features disabled")
        self.edition_config = self._load_config("opensource.yaml")  # Fallback
else:
    self.edition_config = self._load_config("opensource.yaml")
```

### Step 3: Create Open-Source Repository

```bash
# Create new repository
cd ../memory-oss
git init
git add .
git commit -m "Initial open-source release"
git remote add origin https://github.com/Papr-ai/memory.git
git push -u origin main
```

### Step 4: Set Up Branching Strategy

#### Option A: Monorepo with Branches (Current Approach)
```
main (cloud)           # Cloud edition
  └── opensource       # Open-source branch
```

#### Option B: Separate Repositories (Recommended)
```
papr-memory-cloud/     # Private repo
papr-memory/           # Public repo
```

## Code Synchronization Strategy

### From Cloud → Open-Source

1. **Core Improvements** (always sync):
   - Bug fixes in shared code
   - Performance improvements
   - Security patches
   - API improvements

2. **Feature Flags** (conditional sync):
   - New features that work in OSS → Add to `opensource.yaml`
   - Cloud-only features → Keep in cloud only

3. **Process**:
   ```bash
   # In cloud repo
   git checkout main
   git pull
   
   # Cherry-pick commits that should be in OSS
   git log --oneline --grep="fix" --grep="perf" --grep="security"
   
   # Create OSS branch or sync to OSS repo
   git checkout opensource
   git cherry-pick <commit-hash>
   ```

### From Open-Source → Cloud

1. **Community Contributions**:
   - Bug fixes from OSS users
   - Performance improvements
   - Documentation improvements
   - Test improvements

2. **Process**:
   ```bash
   # In OSS repo
   git checkout main
   git pull
   
   # Find commits to merge
   git log --oneline
   
   # Merge into cloud repo
   cd ../papr-memory-cloud
   git checkout main
   git remote add oss ../memory-oss
   git fetch oss
   git merge oss/main --no-ff -m "Merge OSS improvements"
   ```

### Automated Sync (Optional)

Create a GitHub Action to sync specific paths:

```yaml
# .github/workflows/sync-to-oss.yml
name: Sync to Open-Source

on:
  push:
    branches: [main]
    paths:
      - 'services/**'
      - 'routers/**'
      - 'models/**'
      - 'config/opensource.yaml'
      - 'config/base.yaml'
      - 'tests/**'

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Sync to OSS repo
        run: |
          # Script to push changes to OSS repo
          # Only syncs non-cloud-specific files
```

## File Removal Checklist

### ✅ Safe to Remove Immediately

- [ ] `cloud_plugins/` directory
- [ ] `cloud_scripts/` directory
- [ ] `config/cloud.yaml`
- [ ] Cloud-specific CI/CD files

### ⚠️ Update Before Removing

- [ ] `config/features.py` - Handle missing `cloud.yaml` gracefully
- [ ] `README.md` - Update for open-source audience
- [ ] `.github/workflows/` - Remove cloud-specific workflows

### ✅ Keep These

- [ ] `config/opensource.yaml`
- [ ] `config/base.yaml`
- [ ] `config/features.py` (with updates)
- [ ] `docker-compose-open-source.yaml`
- [ ] All core services and routers

## Verification

After removing cloud-specific files, verify:

```bash
# 1. Check for any remaining cloud imports
grep -r "from cloud_plugins" . --exclude-dir=.git
grep -r "import cloud_plugins" . --exclude-dir=.git

# 2. Verify feature flags work
PAPR_EDITION=opensource poetry run python -c "from config import get_features; f = get_features(); print(f.is_cloud)"  # Should print False

# 3. Run tests
PAPR_EDITION=opensource poetry run pytest tests/ -v

# 4. Check Docker build
docker-compose -f docker-compose-open-source.yaml build
```

## Managing Improvements Between Repos

### Weekly Sync Process

1. **Monday**: Review OSS contributions → Merge to cloud
2. **Wednesday**: Review cloud improvements → Sync compatible changes to OSS
3. **Friday**: Test both editions → Release if stable

### Versioning Strategy

- **Cloud**: `v1.2.3-cloud` (includes cloud features)
- **OSS**: `v1.2.3` (matches cloud version, minus cloud features)

### Release Notes

When syncing changes, document:
- What changed
- Why it changed
- Impact on OSS users
- Migration steps (if needed)

## Example: Syncing a Bug Fix

### Scenario: Fix memory leak in `memory_graph.py`

1. **Fix in Cloud**:
   ```bash
   cd papr-memory-cloud
   git checkout -b fix/memory-leak
   # Fix the bug
   git commit -m "fix: memory leak in memory_graph.py"
   git push
   ```

2. **Sync to OSS**:
   ```bash
   cd papr-memory
   git checkout main
   git remote add cloud ../papr-memory-cloud
   git fetch cloud
   git cherry-pick cloud/fix/memory-leak
   git push
   ```

3. **Verify**:
   ```bash
   # Test in OSS
   poetry run pytest tests/test_memory_graph.py
   ```

## Troubleshooting

### Issue: Import errors after removing cloud_plugins

**Solution**: Check for missing `try/except` blocks:
```python
# Bad
from cloud_plugins.temporal.client import get_temporal_client

# Good
try:
    from cloud_plugins.temporal.client import get_temporal_client
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
```

### Issue: Config errors after removing cloud.yaml

**Solution**: Update `config/features.py` to handle missing file:
```python
if self.edition == "cloud":
    self.edition_config = self._load_config("cloud.yaml")
    if not self.edition_config:
        logger.warning("cloud.yaml not found, using opensource.yaml")
        self.edition_config = self._load_config("opensource.yaml")
```

## Next Steps

1. ✅ Run `prepare_open_source.py` script
2. ✅ Update `config/features.py` for missing `cloud.yaml`
3. ✅ Create new GitHub repository for OSS
4. ✅ Set up sync process (manual or automated)
5. ✅ Document OSS-specific setup in README
6. ✅ Create CONTRIBUTING.md for OSS contributors

