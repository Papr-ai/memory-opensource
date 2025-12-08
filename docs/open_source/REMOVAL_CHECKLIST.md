# Open-Source Removal Checklist

Quick reference for what to remove when creating the open-source repository.

## ✅ Safe to Remove

### Directories
- [ ] `cloud_plugins/` - Entire directory (Stripe, Temporal, Azure plugins)
- [ ] `cloud_scripts/` - Entire directory (admin/maintenance scripts)

### Configuration Files
- [ ] `config/cloud.yaml` - Cloud-specific configuration

### Docker Files (if cloud-specific)
- [ ] `docker-compose.yaml` (if different from `docker-compose-open-source.yaml`)

### CI/CD Files
- [ ] `.github/workflows/cloud-*.yml` - Cloud-specific workflows
- [ ] `azure-pipelines.yml` - Azure DevOps pipelines

### Environment Files
- [ ] `.env` - Contains secrets (keep `.env.example`)
- [ ] `.env.local` - Local overrides
- [ ] `.env.production` - Production secrets

## ⚠️ Update Before Removing

### Configuration
- [x] `config/features.py` - Updated to handle missing `cloud.yaml` gracefully

### Documentation
- [ ] `README.md` - Update for open-source audience
- [ ] `CONTRIBUTING.md` - Add OSS contribution guidelines
- [ ] `LICENSE` - Ensure proper open-source license

## ✅ Keep These Files

### Core Configuration
- [x] `config/opensource.yaml` - Open-source configuration
- [x] `config/base.yaml` - Shared base configuration
- [x] `config/features.py` - Feature flag system (updated)

### Docker
- [x] `docker-compose-open-source.yaml` - OSS Docker setup

### Core Code
- [x] All files in `services/` - Core services (use feature flags)
- [x] All files in `routers/` - API routes (use feature flags)
- [x] All files in `models/` - Data models
- [x] All files in `memory/` - Core memory logic
- [x] All files in `tests/` - Test suite

## Verification Commands

After removal, run these to verify:

```bash
# 1. Check for remaining cloud imports (should be empty or only in try/except blocks)
grep -r "from cloud_plugins" . --exclude-dir=.git --exclude-dir=__pycache__ | grep -v "try:" | grep -v "except ImportError"

# 2. Verify feature flags work
PAPR_EDITION=opensource poetry run python -c "from config import get_features; f = get_features(); print('Is Cloud:', f.is_cloud)"  # Should print False

# 3. Verify cloud.yaml is not referenced (except in features.py)
grep -r "cloud.yaml" . --exclude-dir=.git --exclude="features.py"

# 4. Run tests
PAPR_EDITION=opensource poetry run pytest tests/ -v

# 5. Build Docker image
docker-compose -f docker-compose-open-source.yaml build
```

## Quick Removal Script

```bash
#!/bin/bash
# Quick removal script for cloud-specific files

echo "Removing cloud-specific files..."

# Remove directories
rm -rf cloud_plugins/
rm -rf cloud_scripts/

# Remove config
rm -f config/cloud.yaml

# Remove cloud-specific docker (if exists and different)
# rm -f docker-compose.yaml  # Only if different from docker-compose-open-source.yaml

# Remove cloud CI/CD
rm -f azure-pipelines.yml
rm -f .github/workflows/cloud-*.yml 2>/dev/null

# Remove environment files with secrets
rm -f .env .env.local .env.production

echo "✅ Cloud-specific files removed"
echo "⚠️  Remember to:"
echo "   1. Update README.md for OSS audience"
echo "   2. Verify tests pass: PAPR_EDITION=opensource poetry run pytest"
echo "   3. Build Docker: docker-compose -f docker-compose-open-source.yaml build"
```

