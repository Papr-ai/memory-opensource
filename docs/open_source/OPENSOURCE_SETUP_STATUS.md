# Open Source Setup - Status & TODOs

## Completed

### 1. Conditional Route Loading
- Modified `routers/v1/__init__.py` to conditionally load routes based on edition
- Modified `app_factory.py` to conditionally load JWKS routes
- **Cloud-only routes EXCLUDED:**
  - `/v1/documents/*` - Temporal document processing
  - `/v1/graphql/*` - Neo4j Aura GraphQL
  - `/.well-known/jwks.json` - Auth0 JWKS

### 2. Docker Configuration
- Updated `docker-compose-open-source.yaml`:
  - Removed Azure Container Registry reference
  - Set `PAPR_EDITION=opensource`
  - Added healthcheck
  - Configured Docker-appropriate SSL paths

### 3. Environment Configuration
- Created `.env.example` for open-source users
- Created `.env.opensource` for testing
- Configured Parse Server credentials consistently
- Set `AMPLITUDE_API_KEY=` (empty) to prevent initialization

### 4. Testing
- Docker compose brings up all services successfully
- Health endpoint responds: `/health`
- API docs accessible: `/docs`
- Core routes confirmed working (memory, user, feedback, schemas, etc.)

---

## Known Issues to Fix

### 1. Amplitude Still Initializing in OSS **[HIGH PRIORITY]**

**Problem:** Several route files directly instantiate `Amplitude()` without checking feature flags.

**Files that need refactoring:**
```
routes/memory_routes.py:53
routers/v1/schema_routes_v1.py:36
routers/v1/feedback_routes.py:38
routers/v1/memory_routes_v1.py:71
```

**Current workaround:** Set `AMPLITUDE_API_KEY=` (empty) in `.env`

**Proper fix needed:**
```python
# Instead of this:
from amplitude import Amplitude
client = Amplitude(env.get("AMPLITUDE_API_KEY"))

# Should be:
from core.services.telemetry import get_telemetry
telemetry = get_telemetry()  # Uses PostHog in OSS, Amplitude in cloud
```

**Tasks:**
- [ ] Refactor `routes/memory_routes.py` to use telemetry service
- [ ] Refactor `routers/v1/schema_routes_v1.py` to use telemetry service
- [ ] Refactor `routers/v1/feedback_routes.py` to use telemetry service
- [ ] Refactor `routers/v1/memory_routes_v1.py` to use telemetry service

---

### 2. Parse Server Schema Initialization **[HIGH PRIORITY]**

**Problem:** Parse Server needs schema/classes created before the API can work properly.

**Required Parse Classes:**
- `_User` (built-in, auto-created)
- `_Session` (built-in, auto-created)
- `_Role` (built-in, auto-created)
- `Organization` (custom)
- `Namespace` (custom)
- `Workspace` (custom)
- `workspace_follower` (custom)
- `Tenant` (custom)
- Memory-related classes (TBD - check codebase)

**Existing script:** `scripts/add_parse_indexes.py` adds indexes but doesn't create schema.

**Need to create:** `scripts/init_parse_schema.py`

**Script should:**
1. Connect to Parse Server via REST API
2. Create all required classes with proper fields
3. Set appropriate CLPs (Class Level Permissions)
4. Run the index creation script
5. Be idempotent (safe to run multiple times)

**Example structure:**
```python
#!/usr/bin/env python3
"""
Initialize Parse Server schema for Papr Memory open source.
This script creates all required Parse classes, fields, and permissions.
"""

import os
import requests
import sys

PARSE_SERVER_URL = os.getenv("PARSE_SERVER_URL", "http://localhost:1337/parse")
PARSE_APP_ID = os.getenv("PARSE_APPLICATION_ID")
PARSE_MASTER_KEY = os.getenv("PARSE_MASTER_KEY")

def create_schema():
    """Create Parse Server schema"""
    headers = {
        "X-Parse-Application-Id": PARSE_APP_ID,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }

    # Define schemas for each class
    schemas = {
        "Organization": {
            "className": "Organization",
            "fields": {
                "name": {"type": "String"},
                "slug": {"type": "String"},
                "settings": {"type": "Object"},
                # ... more fields
            },
            "classLevelPermissions": {
                "find": {"*": True},
                "get": {"*": True},
                "create": {"requiresAuthentication": True},
                "update": {"requiresAuthentication": True},
                "delete": {"requiresAuthentication": True}
            }
        },
        # ... more classes
    }

    for class_name, schema in schemas.items():
        response = requests.post(
            f"{PARSE_SERVER_URL}/schemas/{class_name}",
            headers=headers,
            json=schema
        )
        # ... handle response
```

**Tasks:**
- [ ] Audit codebase to find all Parse classes used
- [ ] Document required fields for each class
- [ ] Create `scripts/init_parse_schema.py`
- [ ] Test schema creation with fresh Parse Server
- [ ] Add schema init to Docker startup (optional)
- [ ] Document manual setup in README

---

### 3. Cloud-Only Service Dependencies

**Problem:** Some code still tries to connect to cloud services.

**Observed in logs:**
```
PARSE_SERVER_URL: https://c43c027ef3e5.ngrok.app  # Should be http://parse-server:1337/parse
hotglue_api_key: ...  # Shouldn't be used in OSS
```

**Tasks:**
- [ ] Ensure Parse Server URL is correctly set in Docker
- [ ] Review services that shouldn't initialize in OSS

---

## Quick Start (Current State)

### For Testing
```bash
cd /Users/shawkatkabbara/Documents/GitHub/memory

# Use the test environment
docker-compose -f docker-compose-open-source.yaml --env-file .env.opensource up -d

# Check health
curl http://localhost:5001/health

# View logs
docker-compose -f docker-compose-open-source.yaml logs -f web
```

### For Contributors (when .env.example is finalized)
```bash
# Clone repo
git clone https://github.com/Papr-ai/memory.git
cd memory

# Configure
cp .env.example .env
# Edit .env - add OPENAI_API_KEY

# Start
docker-compose -f docker-compose-open-source.yaml up -d

# Initialize Parse schema (NEEDS TO BE CREATED)
python scripts/init_parse_schema.py

# Check
curl http://localhost:5001/docs
```

---

## Next Steps

### Before Open Source Launch

1. **Fix Amplitude initialization** (1-2 hours)
   - Refactor 4 route files to use telemetry service

2. **Create Parse schema init script** (2-4 hours)
   - Audit Parse classes
   - Write init script
   - Test with fresh instance

3. **Update README** (1 hour)
   - Add open source quickstart
   - Document schema initialization
   - Add troubleshooting section

4. **Test end-to-end** (1 hour)
   - Fresh Docker install
   - Schema initialization
   - API calls (create memory, search, etc.)

### Nice to Have

- [ ] Automated schema initialization in Docker entrypoint
- [ ] Health check that verifies Parse schema
- [ ] Example API calls in README
- [ ] Docker Compose profile for development (with dashboard)

---

## Contact

For questions about open source setup:
- GitHub Issues: https://github.com/Papr-ai/memory/issues
- Discord: https://discord.gg/sWpR5a3H
