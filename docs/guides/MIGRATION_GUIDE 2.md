# Migration Guide: Preparing for Open Source

This guide helps you migrate from the current codebase structure to the new open source-ready architecture.

## Overview

The new architecture separates:
- **Core** - Open source functionality (`/core`)
- **Cloud Plugins** - Proprietary features (`/cloud_plugins`)
- **Community Plugins** - Open source plugins (`/plugins`)
- **Feature Flags** - Config-based feature management (`/config`)

## Migration Steps

### Step 1: Install Dependencies

```bash
# Add PostHog for OSS telemetry
poetry add posthog

# Add PyYAML for config files (if not installed)
poetry add pyyaml
```

### Step 2: Feature Flag System

The new feature flag system uses YAML config files instead of hardcoded checks.

**Before:**
```python
# Hardcoded cloud check
if os.getenv("ENVIRONMENT") == "production":
    from services.stripe_service import StripeService
    stripe = StripeService()
```

**After:**
```python
# Feature flag check
from config import get_features

features = get_features()

if features.has_stripe:
    from cloud_plugins.stripe.service import StripeService
    stripe = StripeService()
else:
    # OSS fallback (no restrictions)
    from core.services.subscription import SubscriptionService
    stripe = SubscriptionService()  # Default unlimited
```

### Step 3: Replace Amplitude with TelemetryService

**Before:**
```python
from amplitude import Amplitude, BaseEvent

amplitude = Amplitude(os.getenv("AMPLITUDE_API_KEY"))

event = BaseEvent(
    event_type="memory_created",
    user_id=user_id,
    event_properties={"type": "text"}
)
amplitude.track(event)
```

**After:**
```python
from core.services.telemetry import get_telemetry

telemetry = get_telemetry()

await telemetry.track("memory_created", {
    "type": "text"
})
```

The TelemetryService:
- Automatically uses PostHog for OSS, Amplitude for cloud
- Handles anonymization
- Fails silently
- Respects opt-out

### Step 4: Move Cloud-Specific Code

#### Stripe Integration

**Before:** `services/stripe_service.py`

**After:** Move to `cloud_plugins/stripe/`

```bash
mkdir -p cloud_plugins/stripe
mv services/stripe_service.py cloud_plugins/stripe/service.py
```

Create `cloud_plugins/stripe/__init__.py`:

```python
"""Stripe payment integration (cloud only)"""

from config import get_features

if get_features().has_stripe:
    from .service import StripeService
    __all__ = ["StripeService"]
else:
    __all__ = []
```

#### Auth0 Integration

Move Auth0-specific code to `cloud_plugins/auth0/`:

```bash
mkdir -p cloud_plugins/auth0
# Move Auth0 code here
```

#### Azure Services

Move to `cloud_plugins/azure/`:

```bash
mkdir -p cloud_plugins/azure
mv services/azure_webhook_consumer.py cloud_plugins/azure/webhook_consumer.py
```

### Step 5: Update Imports

Update imports throughout your codebase:

```python
# Before
from services.stripe_service import StripeService

# After (with graceful fallback)
from config import get_features

if get_features().has_stripe:
    from cloud_plugins.stripe.service import StripeService
else:
    # OSS fallback
    StripeService = None
```

Or use a plugin loader:

```python
# core/app_factory.py
def load_subscription_service():
    features = get_features()
    
    if features.has_stripe:
        try:
            from cloud_plugins.stripe.service import StripeService
            return StripeService()
        except ImportError:
            logger.warning("Stripe plugin not available")
    
    # OSS default (no restrictions)
    from core.services.subscription import SubscriptionService
    return SubscriptionService()
```

### Step 6: Update Environment Variables

Create `.env.example` with both OSS and cloud variables:

```bash
# Already created by prepare_open_source.py
cp .env.example.template .env.example
```

Update your `.env`:

```bash
# Add these new variables
PAPR_EDITION=cloud  # or 'opensource'
TELEMETRY_ENABLED=true
TELEMETRY_PROVIDER=amplitude  # or 'posthog' for OSS
```

### Step 7: Move Cloud Scripts

```bash
# Create cloud_scripts directory
mkdir -p cloud_scripts

# Move cloud-specific scripts
mv scripts/stripe cloud_scripts/
mv scripts/*production* cloud_scripts/
mv scripts/sync_neo_to_parse.py cloud_scripts/
# ... etc
```

### Step 8: Update Docker Configuration

Rename and update:

```bash
# Keep cloud version as docker-compose.yaml
mv docker-compose.yaml docker-compose-cloud.yaml

# Make OSS version the default
cp docker-compose-open-source.yaml docker-compose.yaml
```

Update `docker-compose.yaml` to remove Azure registry:

```yaml
services:
  web:
    build: .
    image: papr-memory:latest  # Not Azure registry
    environment:
      - PAPR_EDITION=opensource
```

### Step 9: Testing

Test both editions:

```bash
# Test open source edition
PAPR_EDITION=opensource poetry run python main.py

# Test cloud edition
PAPR_EDITION=cloud poetry run python main.py
```

Verify telemetry:

```bash
# Check telemetry status
curl http://localhost:5001/telemetry/status
```

### Step 10: Run Automated Cleanup

Use the prepare script to create OSS distribution:

```bash
poetry run python scripts/prepare_open_source.py --output ../memory-oss
```

This will:
- ✅ Copy only OSS files
- ✅ Exclude cloud_plugins/
- ✅ Exclude cloud scripts
- ✅ Scan for potential secrets
- ✅ Create OSS-specific files

## Code Patterns

### Pattern 1: Feature Flag Guard

```python
from config import get_features

features = get_features()

@app.post("/billing/subscribe")
async def subscribe():
    if not features.has_stripe:
        raise HTTPException(
            status_code=404,
            detail="Billing not available in open source edition"
        )
    
    # Stripe logic here
```

### Pattern 2: Conditional Import

```python
from config import get_features

# Try to import cloud feature
if get_features().has_stripe:
    try:
        from cloud_plugins.stripe import StripeService
        stripe_available = True
    except ImportError:
        stripe_available = False
else:
    stripe_available = False
```

### Pattern 3: Plugin Registry

```python
# core/plugins/registry.py
class PluginRegistry:
    _plugins = {}
    
    @classmethod
    def register(cls, name: str, plugin):
        cls._plugins[name] = plugin
    
    @classmethod
    def get(cls, name: str):
        return cls._plugins.get(name)

# cloud_plugins/stripe/__init__.py
from core.plugins.registry import PluginRegistry
from .service import StripeService

if get_features().has_stripe:
    PluginRegistry.register("stripe", StripeService())
```

## Rollback Plan

If you need to rollback:

1. **Keep old code in a branch**:
   ```bash
   git checkout -b pre-oss-migration
   git commit -am "Backup before OSS migration"
   ```

2. **Revert imports**: Change new imports back to old ones

3. **Restore Amplitude**: Switch TelemetryService calls back to Amplitude

## Checklist

- [ ] Install new dependencies (posthog, pyyaml)
- [ ] Create config files (base.yaml, opensource.yaml, cloud.yaml)
- [ ] Create feature flag system
- [ ] Create TelemetryService
- [ ] Replace all Amplitude calls
- [ ] Move Stripe to cloud_plugins/
- [ ] Move Auth0 to cloud_plugins/
- [ ] Move Azure services to cloud_plugins/
- [ ] Update all imports
- [ ] Create .env.example
- [ ] Move cloud scripts to cloud_scripts/
- [ ] Update docker-compose files
- [ ] Test both editions
- [ ] Run prepare_open_source.py
- [ ] Review for hardcoded secrets
- [ ] Update documentation

## Support

Questions? Issues?

- 💬 [GitHub Discussions](https://github.com/Papr-ai/memory/discussions)
- 📧 Email: dev@papr.ai

## Next Steps

After migration:

1. **Create OSS repo**: `github.com/Papr-ai/memory`
2. **Set up CI/CD**: GitHub Actions for OSS
3. **Publish docs**: Update docs for community
4. **Announce**: Blog post, Twitter, HN
5. **Monitor**: Track OSS adoption and feedback

---

**Need help?** Create an issue and tag it `migration-help`

