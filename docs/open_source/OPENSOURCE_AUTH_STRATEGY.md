# Open Source Authentication Strategy

## Problem Statement

**Cloud version uses:**
- Auth0 for OAuth signup/login
- Dashboard UI for API key management
- Multi-tenant organization structure
- Complex user → organization → namespace → API key flow

**Open source goal:**
- Simple onboarding (no Auth0)
- No dashboard UI (keep proprietary)
- Enable developers to test easily
- NOT make it trivial to replicate cloud features

## Solution: Simplified Self-Service Bootstrap

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLOUD VERSION                            │
├─────────────────────────────────────────────────────────────┤
│ User signs up → Auth0 OAuth                                 │
│              → Dashboard creates Organization               │
│              → Dashboard creates Namespace                  │
│              → Dashboard generates API keys                 │
│              → Full multi-tenant features                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  OPEN SOURCE VERSION                         │
├─────────────────────────────────────────────────────────────┤
│ User runs bootstrap script (CLI)                            │
│              → Creates Parse _User (no Auth0)               │
│              → Creates Organization                         │
│              → Creates default Namespace                    │
│              → Generates API key                            │
│              → Basic single-tenant setup                    │
└─────────────────────────────────────────────────────────────┘
```

### Data Model (Open Source Subset)

```
_User (Parse Server)
  ├── organization → Organization
  └── user_type: "CREATOR"

Organization
  ├── name, slug
  ├── plan_tier: "FREE"
  ├── rate_limits: { requests_per_hour: 1000 }
  └── default_namespace → Namespace

Namespace
  ├── name, slug
  ├── organization → Organization
  └── environment: "production"

APIKey
  ├── api_key: "pmem_oss_..."
  ├── user → _User
  ├── organization → Organization
  ├── namespace → Namespace
  ├── enabled: true
  └── rate_limit: 1000
```

## Implementation: Two Scripts

### 1. Bootstrap Script (Recommended)
**Purpose:** Complete user onboarding
**Creates:** User + Organization + Namespace + API Key
**Use case:** First-time setup

```bash
python scripts/bootstrap_opensource_user.py \
    --email dev@example.com \
    --name "Developer Name" \
    --organization "My Company"
```

**Output:**
- Parse User credentials (for Parse Dashboard)
- API key (for API requests)
- Organization and Namespace IDs

**Benefits:**
- Full multi-tenant structure
- Can view data in Parse Dashboard
- Proper organization/namespace isolation
- Matches cloud data model (easier migration)

### 2. Simple API Key Generator (Alternative)
**Purpose:** Quick API key for testing
**Creates:** API Key only (minimal setup)
**Use case:** Quick testing, additional keys

```bash
python scripts/generate_api_key.py \
    --email dev@example.com \
    --name "My Project"
```

**Output:**
- API key only
- No user/org/namespace creation

**Benefits:**
- Faster for testing
- Less setup required
- Good for generating additional keys

## Authentication Methods

### Method 1: API Key (Primary)

```bash
curl -H "X-API-Key: pmem_oss_..." http://localhost:5001/v1/memory
```

**Features:**
- Simple header-based auth
- Rate limited per key
- No session management
- Stateless

### Method 2: Parse Session Token (Advanced)

For applications that need user-specific features:

```bash
# 1. Create Parse User via bootstrap script
python scripts/bootstrap_opensource_user.py ...

# 2. Login to get session token
curl -X POST http://localhost:1337/parse/login \
  -H "X-Parse-Application-Id: papr-oss-app-id" \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "password"}'

# 3. Use session token in requests
curl -H "X-Parse-Session-Token: r:abc123..." http://localhost:5001/v1/memory
```

## Cloud vs Open Source: Authentication Comparison

| Feature | Open Source | Cloud |
|---------|-------------|-------|
| **Signup Method** | CLI script | Auth0 OAuth + Dashboard |
| **User Creation** | Parse _User only | Parse _User + Auth0 profile |
| **API Key Generation** | CLI script | Dashboard UI |
| **Multi-Tenancy** | Single org per script run | Full multi-org support |
| **Team Management** | Manual Parse ops | Dashboard UI |
| **SSO/OAuth** | ❌ | ✅ Auth0 |
| **Role-based Access** | Basic Parse ACL | Advanced RBAC |
| **API Key Rotation** | Manual | Dashboard |
| **Usage Analytics** | Basic (Parse queries) | Advanced dashboard |

## Why This Protects Cloud Value

### Open Source Has Friction:
1. **Manual setup** - Users must run CLI scripts
2. **No UI** - All management via Parse Dashboard or CLI
3. **Single tenant** - Each script run = one org
4. **Basic features** - No OAuth, no SSO, no team management
5. **Self-managed** - Users handle their own infrastructure

### Cloud Adds Value:
1. **Instant signup** - Auth0 OAuth, Google/GitHub login
2. **Dashboard UI** - Point-and-click API key management
3. **Multi-tenant** - Multiple orgs per account
4. **Advanced features** - Teams, SSO, RBAC, analytics
5. **Managed service** - We handle infrastructure, scaling, updates

### Migration Path:
- Open source users who need more features → upgrade to cloud
- Data model compatible (same Organization/Namespace structure)
- Can export from open source → import to cloud
- Easy transition without code changes

## Security Considerations

### Open Source:
- API keys stored in Parse Server (MongoDB)
- No OAuth complexity
- Users responsible for securing Parse Server
- Basic rate limiting
- Manual key rotation

### Cloud:
- API keys + OAuth tokens
- Auth0 handles security
- We manage security updates
- Advanced rate limiting
- Automated key rotation
- Audit logs

## Developer Experience Flow

### Open Source:
```
1. Clone repo
2. docker-compose up
3. python bootstrap_opensource_user.py
4. Copy API key
5. Start building
```

**Time to first API call: ~5 minutes**

### Cloud:
```
1. Visit memory.papr.ai
2. Click "Sign up with Google"
3. Click "Generate API Key"
4. Copy API key
5. Start building
```

**Time to first API call: ~1 minute**

## Rate Limits & Tiers

### Open Source:
- **FREE tier only**
- 1000 requests/hour per API key
- Unlimited storage (self-hosted)
- No usage analytics
- No support SLA

### Cloud:
- **Multiple tiers**: Free, Pro, Enterprise
- Higher rate limits
- Managed storage
- Usage analytics dashboard
- Priority support
- SLA guarantees

## Recommendation

**For open source users:**
1. Start with `bootstrap_opensource_user.py` for full setup
2. Use `generate_api_key.py` for additional keys
3. Document that cloud version has easier onboarding
4. Emphasize cloud features (dashboard, teams, OAuth, analytics)

**For cloud marketing:**
- Highlight dashboard UI as key differentiator
- Emphasize "5 minutes to deploy" vs "1 minute to deploy"
- Show team collaboration features
- Promote managed service benefits

This strategy lets developers try the core technology while keeping cloud compelling for production use.
