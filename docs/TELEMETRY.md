# Telemetry & Privacy Policy

## Overview

Papr Memory includes **optional, privacy-first telemetry** to help us improve the software. This document explains exactly what data is collected, how it's used, and how to opt out.

## Our Commitment

We are committed to privacy and transparency:
- ✅ Telemetry is **opt-out** (can be easily disabled)
- ✅ All data is **anonymous** by default
- ✅ **No personal information** is ever collected
- ✅ Open source users can **self-host their own analytics** with PostHog
- ✅ All telemetry code is **open source** and auditable

## What We Collect

### ✅ Data We DO Collect

1. **Feature Usage**
   - Which API endpoints are called
   - Which features are used
   - Feature adoption rates
   - Example: "search endpoint was called 100 times today"

2. **Performance Metrics**
   - Response times (bucketed, not exact)
   - Query performance
   - Error rates
   - Example: "Average search took 100-500ms"

3. **Error Information**
   - Anonymous error types
   - Error frequency
   - Stack traces (with PII removed)
   - Example: "Database connection error occurred 3 times"

4. **Technical Context**
   - Python version (major.minor only)
   - Edition (opensource or cloud)
   - Version number
   - Example: "Python 3.11, opensource edition, v1.0.0"

### ❌ Data We NEVER Collect

- ❌ **Memory content** - We never see what you store
- ❌ **Search queries** - Your searches are private
- ❌ **Personal information** - No emails, names, or user data
- ❌ **IP addresses** - Your location stays private
- ❌ **File paths or names** - Your file structure is private
- ❌ **Unique device identifiers** - No device tracking
- ❌ **User IDs** - Only hashed anonymous IDs

## How to Opt Out

You can disable telemetry in **multiple ways**:

### Method 1: Environment Variable
```bash
# Add to your .env file
TELEMETRY_ENABLED=false
```

### Method 2: Command Line Flag
```bash
# Start with telemetry disabled
TELEMETRY_ENABLED=false python main.py
```

### Method 3: Config File
```yaml
# Edit config/opensource.yaml
telemetry:
  enabled: false
```

### Verify Telemetry Status
```bash
# Check if telemetry is enabled
curl http://localhost:5001/telemetry/status
```

## Self-Hosted Analytics

Open source users can use **PostHog** for self-hosted analytics:

```bash
# Add to .env
TELEMETRY_PROVIDER=posthog
POSTHOG_HOST=http://your-posthog-instance:8000
POSTHOG_API_KEY=your-key

# Or disable completely
TELEMETRY_ENABLED=false
```

### Deploy Your Own PostHog

```bash
# Self-host PostHog with Docker
git clone https://github.com/PostHog/posthog
cd posthog
docker-compose up -d
```

## Transparency

All telemetry code is open source:
- `/core/services/telemetry.py` - Main telemetry service
- `/config/features.py` - Feature flags
- `/config/opensource.yaml` - Open source config

You can audit exactly what data is sent by reviewing these files.

## Data Retention

- Raw telemetry events: Deleted after **90 days**
- Aggregated statistics: Kept indefinitely (anonymous)
- No individual-level data is retained long-term

## Questions?

If you have questions about telemetry or privacy:
- Open an issue: https://github.com/Papr-ai/memory/issues
- Email: privacy@papr.ai

**Thank you for helping us improve Papr Memory!** 🙏
