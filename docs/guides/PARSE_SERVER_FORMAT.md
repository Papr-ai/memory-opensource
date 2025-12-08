# Parse Server Format Requirements

## Why New Collections Don't Show in Parse Dashboard

Parse Server has specific format requirements for documents to be visible in the dashboard.

### Problem: MongoDB Format vs Parse Format

**What we created (v1 migration):**
```javascript
// MongoDB format (invisible in Parse Dashboard)
{
  "objectId": "org_abc123",  // ❌ Parse doesn't recognize this
  "name": "Acme Corp",
  "owner_user_id": "user_123",
  "ACL": {...}  // ❌ Parse expects _acl, _rperm, _wperm
}
```

**What Parse expects:**
```javascript
// Parse Server format (visible in Parse Dashboard)
{
  "_id": "org_abc123",  // ✅ Use _id as the objectId
  "name": "Acme Corp",
  "owner_user_id": "user_123",
  "_acl": {...},  // ✅ ACL stored as _acl
  "_rperm": ["user1", "user2"],  // ✅ Read permissions array
  "_wperm": ["user1"],  // ✅ Write permissions array
  "_created_at": ISODate("2025-01-01T00:00:00Z"),  // ✅ Dates as MongoDB ISODate
  "_updated_at": ISODate("2025-01-01T00:00:00Z")
}
```

## Parse Server Field Mappings

| Public API Field | MongoDB Storage | Parse Dashboard Display |
|------------------|-----------------|-------------------------|
| `objectId` | `_id` | `objectId` |
| `ACL` | `_acl`, `_rperm`, `_wperm` | `ACL` |
| `createdAt` | `_created_at` | `createdAt` |
| `updatedAt` | `_updated_at` | `updatedAt` |

## Fixed Migration (v2)

The enhanced migration script (`migrate_to_multi_tenant_v2.py`) creates documents in Parse format:

```python
org = {
    "_id": org_id,  # ✅ Direct _id field
    "name": "Acme Corp",
    "owner_user_id": user_id,
    "team_members": team_member_ids,
    "team_members_info": team_members_info,  # NEW: Detailed member info
    "workspace_id": workspace_id,  # NEW: Link to original workspace
    "_created_at": datetime.now(),  # ✅ MongoDB datetime
    "_updated_at": datetime.now(),
    
    # Parse Server ACL format
    "_acl": {
        "86cRDG7c4z": {"read": True, "write": True}
    },
    "_wperm": ["86cRDG7c4z"],  # ✅ Write permissions
    "_rperm": ["86cRDG7c4z"]   # ✅ Read permissions
}
```

## Enhanced Migration Features

### 1. Workspace Member Discovery

The v2 migration properly discovers team members:

```
Developer (Shawkat)
  ↓ has isSelectedWorkspaceFollower
workspace_follower (6mu7xHjNXH)
  ↓ has workspace pointer
WorkSpace (EaAJm7b1zN)
  ↓ has many workspace_followers
[workspace_follower1, workspace_follower2, ...]
  ↓ each points to user
[Team Member 1, Team Member 2, ...]
```

### 2. End-User Filtering

Properly excludes end-users from team members:

```python
# Skip end-users (developer's customers)
if user.get("type") == "developerUser":
    continue  # Don't add to team_members

# Include team members (organization members)
else:
    team_members.append(user_id)
```

### 3. Role Mapping

Maps workspace roles to organization roles:

| Workspace Role | Organization Role |
|---------------|-------------------|
| `owner-<workspace_id>` | owner |
| `admin-<workspace_id>` | admin |
| `moderator-<workspace_id>` | moderator |
| (default) | member |

## Organization Schema (Enhanced)

```javascript
{
  "_id": "org_86cRDG7c4z",
  "name": "shawkat.kabbara",
  "owner_user_id": "86cRDG7c4z",
  
  // Team members with roles
  "team_members": ["86cRDG7c4z", "user123", "user456"],
  "team_members_info": [
    {
      "user_id": "86cRDG7c4z",
      "email": "shawkat@papr.ai",
      "displayName": "Shawkat Kabbara",
      "role": "owner"
    },
    {
      "user_id": "user123",
      "email": "amir@papr.ai",
      "displayName": "Amir",
      "role": "admin"
    }
  ],
  
  // Link to original workspace (backward compat)
  "workspace_id": "EaAJm7b1zN",
  
  "plan_tier": "trial",
  "subscription_id": null,
  "settings": {
    "default_namespace": "ns_org_86cRDG7c4z_production"
  },
  
  // Parse ACL format
  "_acl": {
    "86cRDG7c4z": {"read": true, "write": true},
    "user123": {"read": true, "write": false}
  },
  "_rperm": ["86cRDG7c4z", "user123"],
  "_wperm": ["86cRDG7c4z"],
  "_created_at": ISODate("2025-10-01T07:49:10.000Z"),
  "_updated_at": ISODate("2025-10-01T07:49:10.000Z")
}
```

## How to Verify Parse Dashboard Visibility

After running v2 migration, check Parse Dashboard:

1. Go to Parse Dashboard → Browse
2. Look for "Organization" in class list
3. Click on it - you should see all organizations
4. Each org should have:
   - `objectId` = the `_id` value
   - `ACL` = the `_acl` value
   - `team_members` array
   - `workspace_id` link

## Migration Comparison

| Feature | v1 Migration | v2 Migration (Enhanced) |
|---------|--------------|-------------------------|
| Parse visibility | ❌ Uses objectId field | ✅ Uses _id field |
| ACL format | ❌ Uses ACL | ✅ Uses _acl, _rperm, _wperm |
| Team members | ❌ Only owner | ✅ All workspace members |
| End-user filtering | ❌ No filtering | ✅ Excludes type="developerUser" |
| Role mapping | ❌ No roles | ✅ Maps workspace roles |
| Workspace link | ❌ No link | ✅ Stores workspace_id |

## Running Enhanced Migration

```bash
# Clean up old organizations (if needed)
# This is safe - only removes what v1 created, not your data
poetry run python -c "
from pymongo import MongoClient
import certifi, os
from dotenv import load_dotenv
load_dotenv()
client = MongoClient(os.getenv('MONGO_URI'), tlsCAFile=certifi.where())
db = client.get_default_database()

# Remove v1 organizations (they have objectId field)
result = db.Organization.delete_many({'objectId': {'\$exists': True}})
print(f'Removed {result.deleted_count} v1 organizations')
"

# Run enhanced migration
poetry run python scripts/migrate_to_multi_tenant_v2.py
```

## Backward Compatibility

✅ **Workspace still works** for chat app users
✅ **Organization adds developer dashboard** functionality
✅ **Both systems coexist** peacefully

```
Chat App Users → Use Workspace (existing system)
Developer Dashboard → Use Organization (new system)
```

For developers who have both workspace AND organization:
- `workspace_id` field in Organization links them
- Team members from workspace become org members
- End-users stay as DeveloperUser (not added to team_members)

