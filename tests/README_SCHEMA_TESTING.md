# 🧪 End-to-End Schema Testing Guide

This guide shows you how to test the complete user-defined schema functionality, including:
1. **Schema Creation** via API endpoint
2. **Memory Addition** with GPT-5-mini schema selection
3. **Memory Search** with optimized parallel schema fetching

## 🚀 Quick Start

### Option 1: Bash/Curl Test (Easiest)
```bash
# 1. Edit the configuration in test_schema_curl.sh
vim test_schema_curl.sh
# Update: API_KEY, SESSION_TOKEN, and BASE_URL

# 2. Run the test
./test_schema_curl.sh
```

### Option 2: Python Test (More Detailed)
```bash
# 1. Edit the configuration in quick_schema_test.py
vim quick_schema_test.py
# Update: API_KEY, SESSION_TOKEN, and BASE_URL

# 2. Run the test
poetry run python quick_schema_test.py
```

### Option 3: Comprehensive Test (Full Featured)
```bash
# 1. Set environment variables
export TEST_API_KEY="your-actual-api-key"
export TEST_SESSION_TOKEN="your-actual-session-token"
export API_BASE_URL="http://localhost:8000"  # optional

# 2. Run the comprehensive test
poetry run python test_end_to_end_schema.py
```

## 🔧 Prerequisites

### 1. API Server Running
Make sure your API server is running and accessible:
```bash
# Start your server (example)
poetry run uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Authentication Credentials
You need valid authentication credentials:
- **API Key**: Your developer API key
- **Session Token**: A valid session token for the user

### 3. Environment Setup
Make sure required environment variables are set:
```bash
# Required for schema functionality
export ENABLE_LLM_SCHEMA_SELECTION=true
export OPENAI_SCHEMA_SELECTOR_MODEL=gpt-4o-mini
export OPENAI_API_KEY=your-openai-api-key

# Parse Server configuration
export PARSE_SERVER_URL=your-parse-server-url
export PARSE_APPLICATION_ID=your-parse-app-id
export PARSE_REST_API_KEY=your-parse-rest-key
export PARSE_MASTER_KEY=your-parse-master-key
```

## 📋 Test Scenarios

### Test 1: Schema Creation
**Endpoint:** `POST /v1/schemas`

**What it tests:**
- Schema validation and storage in Parse Server
- User-defined node types and relationships
- Schema status management

**Expected Result:**
- HTTP 201 Created
- Schema stored with unique ID
- Schema available for future operations

### Test 2: Memory Addition with Schema Selection
**Endpoint:** `POST /v1/memories`

**What it tests:**
- GPT-5-mini intelligently selects the E-commerce schema
- Memory content: "Customer Sarah purchased iPhone for $1199..."
- Graph generation using selected schema

**Expected Result:**
- HTTP 200 OK
- Memory stored with ID
- Graph nodes created: Customer(Sarah) -[PURCHASED]-> Product(iPhone)
- GPT-5-mini logs show E-commerce schema selection

### Test 3: Schema-Aware Search
**Endpoint:** `POST /v1/memories/search`

**What it tests:**
- Parallel schema fetching during authentication
- GPT-5-mini selects E-commerce schema for search
- Neo4j query enhanced with user-defined schema
- Search finds relevant memories and graph nodes

**Expected Result:**
- HTTP 200 OK
- Memories found matching the search query
- Graph nodes returned from Neo4j
- Performance improvement from parallel schema fetching

## 📊 Expected Performance Improvements

With the parallel schema fetching optimization:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Auth + Schema Fetch | 350ms | 200ms | **43% faster** |
| Total Search Time | 950ms | 800ms | **16% faster** |
| Parse Server Calls | 2 calls | 1 call | **50% reduction** |

## 🔍 Monitoring and Debugging

### 1. Check Server Logs
Look for these log messages:
```
INFO: Using LLM-selected schema for user user_123
INFO: LLM selected schema: E-commerce Schema (confidence: 0.92)
INFO: Using pre-fetched schemas for search enhancement: 2 schemas
INFO: Schema fetch completed in 145.32ms - found 2 active schemas
```

### 2. Verify Parse Server Data
Check that schemas are stored in Parse:
- Class: `UserGraphSchema`
- Fields: `name`, `node_types`, `relationship_types`, `status`, etc.

### 3. Check Neo4j Graph
Verify graph nodes and relationships were created:
```cypher
MATCH (c:Customer)-[r:PURCHASED]->(p:Product) 
WHERE c.name CONTAINS "Sarah" 
RETURN c, r, p
```

### 4. Monitor Performance
Watch for improved search latency:
- Authentication timing should show parallel schema fetching
- Total search time should be reduced
- Fewer Parse Server calls during search operations

## 🐛 Troubleshooting

### Common Issues

**1. Schema Creation Fails (HTTP 400)**
- Check schema validation errors
- Ensure node types don't conflict with system types
- Verify relationship constraints are valid

**2. Memory Addition Fails (HTTP 500)**
- Check OpenAI API key is valid
- Verify GPT-5-mini model is available
- Check Parse Server connectivity

**3. Search Returns No Results**
- Allow time for memory processing (3-5 seconds)
- Check if Neo4j is running and accessible
- Verify graph nodes were created

**4. Schema Not Selected**
- Check `ENABLE_LLM_SCHEMA_SELECTION=true`
- Verify OpenAI API key and credits
- Ensure content clearly matches schema domain

### Debug Commands

**Check server health:**
```bash
curl -X GET "$BASE_URL/health" \
  -H "X-API-Key: $API_KEY"
```

**List user schemas:**
```bash
curl -X GET "$BASE_URL/v1/schemas" \
  -H "X-API-Key: $API_KEY" \
  -H "X-Session-Token: $SESSION_TOKEN"
```

**Check memory by ID:**
```bash
curl -X GET "$BASE_URL/v1/memories/$MEMORY_ID" \
  -H "X-API-Key: $API_KEY" \
  -H "X-Session-Token: $SESSION_TOKEN"
```

## 🎯 Success Criteria

A successful end-to-end test should show:

1. **✅ Schema Creation**
   - HTTP 201 response
   - Valid schema ID returned
   - Schema stored in Parse Server

2. **✅ Intelligent Memory Addition**
   - HTTP 200 response
   - Memory ID returned
   - GPT-5-mini selects correct schema
   - Graph nodes created in Neo4j

3. **✅ Optimized Schema Search**
   - HTTP 200 response
   - Relevant memories found
   - Graph nodes returned
   - Parallel schema fetching works
   - Performance improvements observed

4. **✅ System Integration**
   - All components work together
   - No additional Parse calls during search
   - GPT-5-mini makes consistent schema choices
   - Backward compatibility maintained

## 🚀 Next Steps

After successful testing:

1. **Deploy to Production**
   - Update environment variables
   - Monitor performance improvements
   - Track schema selection accuracy

2. **Create Additional Schemas**
   - Test with CRM, HR, and other domains
   - Verify GPT-5-mini selects appropriate schemas
   - Monitor system performance with multiple schemas

3. **Advanced Testing**
   - Test with multiple concurrent users
   - Verify schema access control
   - Test schema updates and versioning

4. **Performance Monitoring**
   - Set up alerts for search latency
   - Monitor Parse Server load reduction
   - Track GPT-5-mini selection accuracy

---

**📞 Need Help?**
- Check server logs for detailed error messages
- Verify all environment variables are set correctly
- Ensure Parse Server and Neo4j are accessible
- Test with simple schemas first, then add complexity


