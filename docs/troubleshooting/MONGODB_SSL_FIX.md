# MongoDB SSL Certificate Fix

## Issue
When connecting to MongoDB Atlas from macOS, you may encounter:
```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
```

## Quick Fix (Development)

Both migration scripts have been updated to use `tlsAllowInvalidCertificates=True` for development:

```python
client = MongoClient(
    MONGO_URI,
    tlsAllowInvalidCertificates=True,  # For development only
    serverSelectionTimeoutMS=10000
)
```

## Production Fix (Recommended)

### Option 1: Install Python Certifi
```bash
poetry add certifi
```

Then use it in your connection:
```python
import certifi

client = MongoClient(
    MONGO_URI,
    tlsCAFile=certifi.where()
)
```

### Option 2: Install macOS Certificates
```bash
# Install certificates from Python
/Applications/Python\ 3.11/Install\ Certificates.command

# Or via Homebrew
brew install openssl
```

### Option 3: Use MongoDB's CA Certificate
Download MongoDB Atlas CA certificate and specify it:
```python
client = MongoClient(
    MONGO_URI,
    tlsCAFile="/path/to/mongodb-atlas-ca.pem"
)
```

## .env File Warning

If you see:
```
WARNING - Python-dotenv could not parse statement starting at line 153
```

This means there's a malformed line in your `.env` file. Common issues:
- Unquoted values with special characters
- Missing equals sign
- Line continuation issues

**Fix:**
1. Open your `.env` file
2. Go to line 153
3. Ensure proper format: `KEY=value` or `KEY="value with spaces"`
4. Remove any trailing spaces or special characters

**Example good formats:**
```bash
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/db
MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/db"
API_KEY=abc123xyz
```

**Example bad formats:**
```bash
MONGO_URI = mongodb+srv://...  # No spaces around =
MONGO_URI mongodb+srv://...    # Missing =
MONGO_URI=mongodb+srv://... \  # Don't use line continuation
  ?param=value
```

## Testing

After the fix, run:
```bash
# Test the connection
poetry run python scripts/test_migration_safe.py

# You should see:
# INFO - ============================================================
# INFO - MIGRATION SAFETY CHECK (DRY RUN)
# INFO - ============================================================
```

## Security Note

⚠️ **`tlsAllowInvalidCertificates=True` is for development only!**

For production:
- Use proper SSL certificates
- Use `certifi` package
- Never disable certificate verification in production

