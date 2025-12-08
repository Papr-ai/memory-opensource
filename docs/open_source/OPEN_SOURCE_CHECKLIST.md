# Open Source Preparation Checklist

Complete checklist for preparing Papr Memory for open source release.

## Phase 1: Architecture Setup ✅

- [x] Create config system with feature flags
  - [x] `/config/base.yaml` - Shared configuration
  - [x] `/config/opensource.yaml` - OSS edition config
  - [x] `/config/cloud.yaml` - Cloud edition config
  - [x] `/config/features.py` - Feature flag system

- [x] Create core directory structure
  - [x] `/core/` directory for OSS code
  - [x] `/core/services/telemetry.py` - Privacy-first telemetry

- [x] Create plugin directories
  - [ ] `/cloud_plugins/` - Cloud-only features
  - [ ] `/plugins/` - Community plugins

- [x] Create automated cleanup script
  - [x] `scripts/prepare_open_source.py`

## Phase 2: Dependencies

- [x] Add PostHog for OSS telemetry
  ```bash
  poetry add posthog
  ```

- [x] Verify PyYAML is installed (for config files)

- [ ] Test all dependencies work in OSS mode

## Phase 3: Code Separation

### Cloud-Specific Services

- [ ] Move Stripe integration
  - [ ] Create `/cloud_plugins/stripe/`
  - [ ] Move `services/stripe_service.py` → `cloud_plugins/stripe/service.py`
  - [ ] Move `services/user_utils.py` Stripe code → cloud plugin
  - [ ] Remove hardcoded Stripe price IDs

- [ ] Move Auth0 integration
  - [ ] Create `/cloud_plugins/auth0/`
  - [ ] Extract Auth0 code from `app_factory.py`
  - [ ] Make Auth0 optional via feature flags

- [ ] Move Azure services
  - [ ] Create `/cloud_plugins/azure/`
  - [ ] Move `services/azure_webhook_consumer.py`
  - [ ] Move `services/webhook_service.py` Azure code

- [ ] Move Amplitude analytics
  - [ ] Create `/cloud_plugins/amplitude/`
  - [ ] Wrapper around amplitude-analytics package

### Replace Amplitude with TelemetryService

Update these files:

- [ ] `services/utils.py` - Replace `log_amplitude_event()`
- [ ] `services/memory_service.py` - Replace Amplitude imports
- [ ] `app_factory.py` - Replace Amplitude client initialization
- [ ] `routers/v1/memory_routes_v1.py` - Replace Amplitude tracking
- [ ] `routers/v1/feedback_routes.py` - Replace Amplitude tracking
- [ ] `routes/memory_routes.py` - Replace Amplitude tracking (if used)

Pattern to follow:

```python
# Before
from amplitude import Amplitude, BaseEvent
amplitude_client.track(BaseEvent(...))

# After
from core.services.telemetry import get_telemetry
await get_telemetry().track("event_name", properties)
```

## Phase 4: Scripts & Files

### Move Cloud Scripts

Create `/cloud_scripts/` and move:

- [ ] `scripts/stripe/` → `cloud_scripts/stripe/`
- [ ] `scripts/add_developer_flags.py`
- [ ] `scripts/backfill_memory_counters.py`
- [ ] `scripts/copy_lost_data.py`
- [ ] `scripts/fix_duplicate_api_keys.py`
- [ ] `scripts/generate_missing_api_keys.py`
- [ ] `scripts/mirror_parse_to_mongo.py`
- [ ] `scripts/sync_neo_to_parse.py`
- [ ] `scripts/test_production_config.py`
- [ ] `scripts/update_feedback_analytics.py`

### Delete Sensitive Files

- [ ] Delete `scripts/Cohort_dataloss_users.csv` (user data)
- [ ] Review all scripts for hardcoded secrets
- [ ] Remove `azure-pipelines.yml` or mark as cloud-only

### Configuration Files

- [x] Create `.env.example` (done via terminal)
- [ ] Update `docker-compose-open-source.yaml`
  - [ ] Remove Azure Container Registry references
  - [ ] Add PostHog container (optional)
  - [ ] Set `PAPR_EDITION=opensource`

## Phase 5: Documentation

### Create New Docs

- [x] `docs/TELEMETRY.md` - Telemetry transparency
- [x] `docs/MIGRATION_GUIDE.md` - Migration guide
- [ ] `docs/SELF_HOSTING.md` - Self-hosting guide
- [ ] `SECURITY.md` - Security policy

### Update Existing Docs

- [ ] Update `README.md`
  - [ ] Add Open Source vs Cloud comparison table
  - [ ] Update installation instructions
  - [ ] Add telemetry disclosure
  - [ ] Add self-hosting instructions

- [ ] Update `CONTRIBUTING.md`
  - [ ] Add plugin development guide
  - [ ] Clarify cloud vs OSS contributions

## Phase 6: Security Review

### Scan for Secrets

- [ ] Run `scripts/prepare_open_source.py` and review warnings
- [ ] Search for hardcoded API keys
  ```bash
  grep -r "sk_live\|pk_live\|price_\|prod_" --include="*.py" .
  ```

- [ ] Review all environment variables in code
- [ ] Ensure no database credentials are hardcoded
- [ ] Check for Auth0 client secrets

### Remove Sensitive Data

- [ ] Remove all `.log` files
- [ ] Clear `logs/` directory
- [ ] Remove any user data files
- [ ] Check git history for secrets (use `git-secrets` or `truffleHog`)

## Phase 7: Testing

### Test Open Source Edition

```bash
# Set OSS mode
export PAPR_EDITION=opensource
export TELEMETRY_ENABLED=false

# Start server
poetry run python main.py

# Test endpoints work
curl http://localhost:5001/health
curl http://localhost:5001/docs
```

### Test Cloud Edition

```bash
# Set cloud mode
export PAPR_EDITION=cloud

# Start server
poetry run python main.py

# Verify cloud features load
```

### Test Both Editions

- [ ] Memory creation works in both editions
- [ ] Search works in both editions
- [ ] OSS has no Stripe references
- [ ] Cloud has Stripe integration
- [ ] Telemetry works (PostHog in OSS, Amplitude in cloud)
- [ ] Opt-out works: `TELEMETRY_ENABLED=false`

## Phase 8: Create OSS Distribution

### Run Preparation Script

```bash
# Create clean OSS copy
poetry run python scripts/prepare_open_source.py --output ../memory-oss

# Review output
cd ../memory-oss
ls -la
```

### Verify OSS Distribution

- [ ] No `cloud_plugins/` directory
- [ ] No `cloud_scripts/` directory
- [ ] No `azure-pipelines.yml`
- [ ] No cloud-specific files
- [ ] `.env.example` exists
- [ ] `docker-compose.yaml` exists (not docker-compose-open-source.yaml)
- [ ] All docs are present

### Test OSS Build

```bash
cd ../memory-oss

# Test with Docker
docker-compose up -d

# Test manually
poetry install
poetry run python main.py
```

## Phase 9: Repository Setup

### Create OSS Repository

- [ ] Create new repo: `github.com/Papr-ai/memory`
- [ ] Set license: AGPL-3.0 (or your choice)
- [ ] Add description
- [ ] Add topics: `memory`, `ai`, `vector-database`, `knowledge-graph`

### Initial Commit

```bash
cd ../memory-oss
git init
git add .
git commit -m "Initial open source release"
git remote add origin git@github.com:Papr-ai/memory.git
git push -u origin main
```

### GitHub Settings

- [ ] Enable Issues
- [ ] Enable Discussions
- [ ] Add CODE_OF_CONDUCT.md
- [ ] Add SECURITY.md
- [ ] Create issue templates
- [ ] Set up GitHub Actions for CI
- [ ] Add community health files

## Phase 10: Launch Preparation

### Pre-Launch

- [ ] Set up project website (optional)
- [ ] Create demo/playground (optional)
- [ ] Record demo video
- [ ] Write launch blog post
- [ ] Prepare social media posts
- [ ] Create Product Hunt page (optional)

### Launch Checklist

- [ ] Announce on Twitter/X
- [ ] Post on Hacker News
- [ ] Post on Reddit (r/selfhosted, r/programming)
- [ ] Share in Discord communities
- [ ] Email existing users (if any)
- [ ] Update company website

### Post-Launch

- [ ] Monitor GitHub issues
- [ ] Respond to community questions
- [ ] Track analytics (PostHog)
- [ ] Gather feedback
- [ ] Plan first community PR

## Phase 11: Ongoing Maintenance

### Regular Tasks

- [ ] Triage new issues weekly
- [ ] Review PRs promptly
- [ ] Update documentation as needed
- [ ] Release notes for each version
- [ ] Security patches

### Community Building

- [ ] Welcome first-time contributors
- [ ] Create "good first issue" labels
- [ ] Host community calls (optional)
- [ ] Showcase community projects
- [ ] Recognize top contributors

---

## Quick Reference

### Test Commands

```bash
# Install dependencies
poetry install

# Run in OSS mode
PAPR_EDITION=opensource poetry run python main.py

# Run in cloud mode  
PAPR_EDITION=cloud poetry run python main.py

# Test telemetry opt-out
TELEMETRY_ENABLED=false poetry run python main.py

# Run tests
poetry run pytest

# Create OSS distribution
poetry run python scripts/prepare_open_source.py --output ../memory-oss
```

### Important Files

| File | Purpose |
|------|---------|
| `.env.example` | Environment template |
| `config/features.py` | Feature flag system |
| `config/opensource.yaml` | OSS configuration |
| `core/services/telemetry.py` | Privacy-first telemetry |
| `scripts/prepare_open_source.py` | OSS distribution creator |
| `docs/TELEMETRY.md` | Telemetry transparency |
| `docs/MIGRATION_GUIDE.md` | Migration instructions |

---

## Need Help?

- 💬 Internal team chat
- 📧 Email: dev@papr.ai
- 📝 Check `/docs/MIGRATION_GUIDE.md`

**Progress**: ✅ = Complete | ⏳ = In Progress | ❌ = Blocked | 📝 = Needs Review

