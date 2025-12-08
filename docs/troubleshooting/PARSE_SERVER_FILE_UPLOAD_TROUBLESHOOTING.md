# Parse Server File Upload Troubleshooting Guide

## Error: "File storage failed: Parse Server file storage failed"

This error occurs when the document upload endpoint (`/v1/document`) fails to store the file in Parse Server. This guide helps you diagnose and fix the root cause.

## Root Cause Analysis

The error can occur at several points in the file upload flow:

1. **Missing or incorrect Parse Server configuration**
2. **Parse Server connectivity issues**
3. **Authentication failures**
4. **Parse Server file adapter not configured**
5. **File size limits exceeded**
6. **SSL/TLS certificate issues**
7. **Network timeout issues**

## Diagnostic Steps

### Step 1: Check Parse Server Configuration

Verify these environment variables are set correctly in your production environment:

```bash
# Required environment variables
PARSE_SERVER_URL=https://your-parse-server.com/parse
PARSE_APPLICATION_ID=your-parse-app-id
PARSE_MASTER_KEY=your-parse-master-key
```

**Common Issues:**
- `PARSE_SERVER_URL` missing trailing `/parse` path
- `PARSE_MASTER_KEY` not set or incorrect
- Environment variables not loaded in production

**How to verify:**
1. Check your production logs for these values (they should be logged at startup)
2. Test Parse Server connectivity: `curl -X GET "${PARSE_SERVER_URL}/health"`

### Step 2: Check Parse Server Logs

The most important diagnostic step is to check your Parse Server logs for the actual error:

```bash
# If using Docker
docker logs papr-parse-server

# If using Kubernetes
kubectl logs -l app=parse-server

# Check for errors around the time of the upload attempt
```

**Look for:**
- HTTP status codes (401, 403, 413, 500)
- File adapter errors
- Authentication errors
- File size limit errors

### Step 3: Verify Parse Server File Adapter Configuration

Parse Server requires a file adapter to be configured. Common adapters:

1. **GridFS (MongoDB)** - Default, requires no additional config
2. **S3** - Requires AWS credentials
3. **GCS** - Requires Google Cloud credentials
4. **Filesystem** - Requires writable directory

**Check your Parse Server configuration:**

```javascript
// parse-server-config.json or environment variables
{
  "filesAdapter": {
    "module": "@parse/fs-files-adapter",  // or grid-store, s3, etc.
    "options": {
      // adapter-specific options
    }
  }
}
```

**Common Issue:** File adapter not configured, causing 500 errors on file upload.

### Step 4: Test Parse Server File Upload Directly

Test if Parse Server accepts file uploads directly:

```bash
curl -X POST "${PARSE_SERVER_URL}/files/test.pdf" \
  -H "X-Parse-Application-Id: ${PARSE_APPLICATION_ID}" \
  -H "X-Parse-Master-Key: ${PARSE_MASTER_KEY}" \
  -H "Content-Type: application/pdf" \
  --data-binary @test.pdf
```

**Expected Response:**
```json
{
  "name": "test.pdf",
  "url": "https://your-parse-server.com/parse/files/test.pdf"
}
```

**If this fails:**
- Check Parse Server file adapter configuration
- Verify master key permissions
- Check Parse Server logs for detailed error

### Step 5: Check File Size Limits

Parse Server has default file size limits. Check your configuration:

```javascript
// In Parse Server config
{
  "maxUploadSize": "20mb"  // Default is usually 20mb
}
```

**Common Issue:** Large PDF files exceeding the limit cause 413 errors.

### Step 6: Check Authentication

Verify that the API key authentication is working:

```bash
# Test with master key
curl -X GET "${PARSE_SERVER_URL}/classes/_User" \
  -H "X-Parse-Application-Id: ${PARSE_APPLICATION_ID}" \
  -H "X-Parse-Master-Key: ${PARSE_MASTER_KEY}"
```

**If this fails:**
- Master key is incorrect
- Master key IP restrictions (check `PARSE_SERVER_MASTER_KEY_IPS`)
- Parse Server security settings blocking the request

### Step 7: Check Network Connectivity

Verify that your application server can reach Parse Server:

```bash
# From your application server
curl -v "${PARSE_SERVER_URL}/health"
```

**Common Issues:**
- Firewall blocking outbound connections
- DNS resolution failures
- SSL certificate validation failures
- Network timeouts (especially for large files)

### Step 8: Review Application Logs

Check your application logs for detailed error messages:

```bash
# Look for these log messages
grep "Failed to store file in Parse Server" logs/app_*.log
grep "Failed to upload file to Parse Server" logs/app_*.log
grep "Parse Server file upload" logs/app_*.log
```

**The improved error logging will show:**
- HTTP status code from Parse Server
- Response text from Parse Server
- File size and filename
- Parse Server URL being used
- Authentication method (API key vs session token)

## Common Error Scenarios

### Error: "PARSE_SERVER_URL is not configured"
**Solution:** Set `PARSE_SERVER_URL` environment variable in production

### Error: "PARSE_APPLICATION_ID is not configured"
**Solution:** Set `PARSE_APPLICATION_ID` environment variable in production

### Error: "PARSE_MASTER_KEY is not configured"
**Solution:** Set `PARSE_MASTER_KEY` environment variable in production

### Error: HTTP 401 (Unauthorized)
**Causes:**
- Incorrect `PARSE_APPLICATION_ID`
- Incorrect `PARSE_MASTER_KEY`
- Master key IP restrictions

**Solution:**
- Verify credentials are correct
- Check `PARSE_SERVER_MASTER_KEY_IPS` allows your server IP
- Test authentication with curl (see Step 6)

### Error: HTTP 403 (Forbidden)
**Causes:**
- Master key not allowed for file operations
- ACL restrictions
- Parse Server security settings

**Solution:**
- Check Parse Server configuration for master key permissions
- Verify `PARSE_SERVER_ENABLE_EXPERIMENTAL_DIRECT_ACCESS` is set if needed
- Review Parse Server ACL settings

### Error: HTTP 413 (Payload Too Large)
**Causes:**
- File exceeds Parse Server `maxUploadSize` limit

**Solution:**
- Increase `maxUploadSize` in Parse Server config
- Or compress files before upload
- Or use chunked upload for large files

### Error: HTTP 500 (Internal Server Error)
**Causes:**
- Parse Server file adapter not configured
- File adapter misconfiguration
- Parse Server bug or crash

**Solution:**
- Check Parse Server logs for detailed error
- Verify file adapter is properly configured
- Test file upload directly to Parse Server (see Step 4)
- Check Parse Server version compatibility

### Error: Timeout
**Causes:**
- Large file uploads taking too long
- Network latency
- Parse Server slow to respond

**Solution:**
- Increase timeout values (currently 600s for read/write)
- Check Parse Server performance
- Consider using async file upload with status polling

### Error: SSL/TLS Certificate Issues
**Causes:**
- Invalid SSL certificate
- Certificate validation failures
- Missing CA certificates

**Solution:**
- Verify `SSL_CERT_FILE` points to valid certificate bundle
- Check Parse Server SSL certificate is valid
- Test SSL connection: `openssl s_client -connect your-parse-server.com:443`

## Quick Fix Checklist

- [ ] Verify `PARSE_SERVER_URL` is set and accessible
- [ ] Verify `PARSE_APPLICATION_ID` is correct
- [ ] Verify `PARSE_MASTER_KEY` is correct and has permissions
- [ ] Check Parse Server logs for detailed errors
- [ ] Test file upload directly to Parse Server
- [ ] Verify Parse Server file adapter is configured
- [ ] Check file size doesn't exceed limits
- [ ] Verify network connectivity to Parse Server
- [ ] Check SSL certificate validity
- [ ] Review application logs for detailed error messages

## Next Steps After Diagnosis

1. **If configuration issue:** Update environment variables and restart application
2. **If Parse Server issue:** Fix Parse Server configuration and restart Parse Server
3. **If file adapter issue:** Configure appropriate file adapter in Parse Server
4. **If network issue:** Fix network/firewall rules or DNS configuration
5. **If file size issue:** Increase limits or implement chunked upload

## Getting Help

If you've completed all diagnostic steps and still can't resolve the issue:

1. Collect the following information:
   - Parse Server logs (around the time of failure)
   - Application logs (with improved error messages)
   - Parse Server configuration (sanitize secrets)
   - Environment variables (sanitize secrets)
   - File size and type that failed
   - HTTP status code from Parse Server response

2. Check Parse Server GitHub issues for similar problems
3. Review Parse Server documentation for file adapter configuration

