# Test User Credentials - Verified from Parse Server

## ✅ Verified Parse Server User (2026-02-07)

### User Details
- **User ID**: `2Mignn79LT`
- **Username**: `opensource@papr.ai`
- **Display Name**: Open Source User
- **Session Token**: `r:4ac9ac563aaeef8057f4be91d8e3107a`
- **API Key**: `pmem_oss_7dPghxOIueTm_9x7ZgFCAbIjZNiZgax4Q0Oens8mBqE`
- **User Type**: CREATOR
- **Is Developer**: true

### Organization
- **Organization ID**: `cakBkdOCKL` ✅ (corrected from `IUqhEA3Sg8`)
- **Name**: Papr Open Source
- **Slug**: papr-open-source
- **Plan**: FREE

### Namespace
- **Namespace ID**: `uh2IcLjbD2` ✅ (corrected from `rR5fGqyJNF`)
- **Name**: Papr Open Source - Default
- **Organization**: cakBkdOCKL
- **Environment**: production
- **Is Active**: false

### Workspace (Tenant)
- **Workspace/Tenant ID**: `aqafCD1JAr`
- **Name**: Papr Open Source Workspace
- **Owner**: 2Mignn79LT

## 🔧 Updated Files

1. **`.env`** - Updated test credentials
2. **`docker-compose.yaml`** - Updated test-runner environment variables

## 📝 Key Corrections

The original `.env` had incorrect IDs:
- ❌ Old `TEST_NAMESPACE_ID`: `rR5fGqyJNF`
- ✅ New `TEST_NAMESPACE_ID`: `uh2IcLjbD2`

- ❌ Old `TEST_ORGANIZATION_ID`: `IUqhEA3Sg8`
- ✅ New `TEST_ORGANIZATION_ID`: `cakBkdOCKL`

These incorrect IDs were causing the **Search/ACL test failures**:
- "Expected organization_id IUqhEA3Sg8, got None"
- "No relevant items found" (ACL filtering with wrong org/namespace)

## 🎯 Impact on Tests

### Fixed Tests (Expected):
- ✅ Search Memory - Organization and Namespace Filter
- ✅ Search Memory - User ID ACL
- ✅ Search Memory - External User ID ACL
- ✅ Other ACL-related search tests

These tests were failing because they were looking for memories with:
- `organization_id = IUqhEA3Sg8` (doesn't exist)
- `namespace_id = rR5fGqyJNF` (doesn't exist)

Now they will use the correct IDs from the actual Parse Server user.

## 🧪 Next Steps

Run the Search/ACL failing tests to verify fixes:

```bash
# Search with organization/namespace filter
./tests/run_single_test.sh "tests/test_add_memory_fastapi.py::test_v1_search_with_organization_and_namespace_filter"

# Search with User ID ACL
./tests/run_single_test.sh "tests/test_add_memory_fastapi.py::test_v1_search_with_user_id_acl"

# Search with External User ID ACL
./tests/run_single_test.sh "tests/test_add_memory_fastapi.py::test_v1_search_with_external_user_id_acl"
```

## Query Commands Used

```bash
# Get user info
curl -X GET "http://localhost:1337/parse/users/me" \
  -H "X-Parse-Application-Id: papr-oss-app-id" \
  -H "X-Parse-Session-Token: r:4ac9ac563aaeef8057f4be91d8e3107a"

# Get organization
curl -X GET "http://localhost:1337/parse/classes/Organization/cakBkdOCKL" \
  -H "X-Parse-Application-Id: papr-oss-app-id" \
  -H "X-Parse-Master-Key: papr-oss-master-key"

# Get namespace  
curl -X GET "http://localhost:1337/parse/classes/Namespace/uh2IcLjbD2" \
  -H "X-Parse-Application-Id: papr-oss-app-id" \
  -H "X-Parse-Master-Key: papr-oss-master-key"

# Get workspace
curl -X GET "http://localhost:1337/parse/classes/WorkSpace" \
  -H "X-Parse-Application-Id: papr-oss-app-id" \
  -H "X-Parse-Master-Key: papr-oss-master-key"
```
