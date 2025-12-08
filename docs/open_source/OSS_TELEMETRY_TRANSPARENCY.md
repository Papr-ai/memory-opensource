# Open Source Telemetry: Full Transparency

## What OSS Users See vs Cloud Users

```
┌─────────────────────────────────────────────────────────────┐
│                    Cloud Edition (Amplitude)                 │
├─────────────────────────────────────────────────────────────┤
│ Primary ID: developer_abc123                                │
│ Properties: {                                               │
│   "end_user_id": "user_1234",                              │
│   "developer_id": "developer_abc123",                      │
│   "email": "dev@company.com",                              │
│   "country": "United States",                              │
│   "type": "text",                                          │
│   "client_type": "api"                                     │
│ }                                                           │
│                                                             │
│ Developer sees: Full analytics, user tracking, cohorts     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                OSS Edition (PostHog/Self-Hosted)            │
├─────────────────────────────────────────────────────────────┤
│ Primary ID: sha256:a1b2c3d4e5f6... (anonymous hash)        │
│ Properties: {                                               │
│   "type": "text",                    // What was created   │
│   "has_metadata": true,              // Count, not content │
│   "edition": "opensource",           // Which edition      │
│   "version": "1.0.0",                // Software version   │
│   "os": "Linux",                     // Operating system   │
│   "arch": "x64"                      // Architecture       │
│ }                                                           │
│                                                             │
│ Developer sees: Aggregate stats only, NO personal data     │
└─────────────────────────────────────────────────────────────┘
```

---

## First-Run Experience (OSS)

### Initial Prompt

When someone first runs the OSS version, they see:

```
╔════════════════════════════════════════════════════════════╗
║                   Welcome to Papr Memory!                   ║
╠════════════════════════════════════════════════════════════╣
║                                                             ║
║  Help us improve Papr by sharing anonymous usage data?     ║
║                                                             ║
║  What we collect:                                          ║
║    ✓ Feature usage (e.g., "search performed")             ║
║    ✓ Error types (e.g., "database connection failed")     ║
║    ✓ Performance metrics (e.g., "query took 1-5s")        ║
║    ✓ System info (OS, architecture, version)              ║
║                                                             ║
║  What we DON'T collect:                                    ║
║    ✗ Your content or data                                  ║
║    ✗ User IDs, emails, or personal info                   ║
║    ✗ IP addresses or location                             ║
║    ✗ File names or paths                                  ║
║                                                             ║
║  You can opt out anytime: TELEMETRY_ENABLED=false          ║
║  View our code: core/services/telemetry.py                 ║
║  Read more: docs/TELEMETRY.md                              ║
║                                                             ║
║  [Y] Yes, help improve Papr    [N] No, disable telemetry  ║
║                                                             ║
╚════════════════════════════════════════════════════════════╝

Your choice: _
```

### Implementation

```python
# core/services/first_run.py
import os
from pathlib import Path
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)

FIRST_RUN_FILE = Path.home() / ".papr" / "first_run_complete"

async def check_first_run() -> bool:
    """Check if this is the first run"""
    return not FIRST_RUN_FILE.exists()

async def prompt_telemetry_consent():
    """Prompt user for telemetry consent on first run"""
    
    if not await check_first_run():
        return  # Already prompted
    
    # Check if already configured via env
    if os.getenv("TELEMETRY_ENABLED") is not None:
        logger.info("Telemetry already configured via environment variable")
        await mark_first_run_complete()
        return
    
    # Print the prompt
    print("""
╔════════════════════════════════════════════════════════════╗
║                   Welcome to Papr Memory!                   ║
╠════════════════════════════════════════════════════════════╣
║                                                             ║
║  Help us improve Papr by sharing anonymous usage data?     ║
║                                                             ║
║  What we collect:                                          ║
║    ✓ Feature usage (e.g., "search performed")             ║
║    ✓ Error types (e.g., "database connection failed")     ║
║    ✓ Performance metrics (e.g., "query took 1-5s")        ║
║    ✓ System info (OS, architecture, version)              ║
║                                                             ║
║  What we DON'T collect:                                    ║
║    ✗ Your content or data                                  ║
║    ✗ User IDs, emails, or personal info                   ║
║    ✗ IP addresses or location                             ║
║    ✗ File names or paths                                  ║
║                                                             ║
║  You can opt out anytime: TELEMETRY_ENABLED=false          ║
║  View our code: core/services/telemetry.py                 ║
║  Read more: docs/TELEMETRY.md                              ║
║                                                             ║
║  [Y] Yes, help improve Papr    [N] No, disable telemetry  ║
║                                                             ║
╚════════════════════════════════════════════════════════════╝
""")
    
    # Get user input
    choice = input("Your choice [Y/n]: ").strip().lower()
    
    # Update .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            lines = f.readlines()
        
        # Find and update TELEMETRY_ENABLED line
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("TELEMETRY_ENABLED="):
                lines[i] = f"TELEMETRY_ENABLED={'true' if choice != 'n' else 'false'}\n"
                updated = True
                break
        
        # Add if not found
        if not updated:
            lines.append(f"\n# Telemetry (set to false to opt out)\n")
            lines.append(f"TELEMETRY_ENABLED={'true' if choice != 'n' else 'false'}\n")
        
        with open(env_file, "w") as f:
            f.writelines(lines)
    
    # Mark first run complete
    await mark_first_run_complete()
    
    print(f"\nTelemetry {'enabled' if choice != 'n' else 'disabled'}. Thank you!")
    print(f"You can change this anytime in .env\n")

async def mark_first_run_complete():
    """Mark that first run setup is complete"""
    FIRST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIRST_RUN_FILE.touch()
```

---

## Runtime Visibility

### CLI Command to See What's Tracked

```bash
# Add to your CLI
$ papr telemetry status

Telemetry Status:
  Enabled: Yes
  Provider: PostHog (self-hosted)
  Host: http://localhost:8000
  Anonymous ID: sha256:a1b2c3d4e5f6...

Recent Events (last 24 hours):
  - memory_created (15 times)
  - search_performed (8 times)
  - error_occurred (2 times)

To opt out: TELEMETRY_ENABLED=false in .env
To view data: http://localhost:8000 (PostHog dashboard)
```

### Implementation

```python
# cli/telemetry.py
import click
from core.services.telemetry import get_telemetry

@click.group()
def telemetry():
    """Manage telemetry settings"""
    pass

@telemetry.command()
def status():
    """Show telemetry status"""
    telemetry = get_telemetry()
    status = telemetry.get_status()
    
    click.echo(f"""
Telemetry Status:
  Enabled: {status['enabled']}
  Provider: {status['provider']}
  Host: {status.get('host', 'N/A')}
  Anonymous ID: {status['anonymous_id']}

To opt out: TELEMETRY_ENABLED=false in .env
To view data: {status.get('dashboard_url', 'N/A')}
""")

@telemetry.command()
def disable():
    """Disable telemetry"""
    # Update .env
    update_env_file("TELEMETRY_ENABLED", "false")
    click.echo("✓ Telemetry disabled")

@telemetry.command()
def enable():
    """Enable telemetry"""
    # Update .env
    update_env_file("TELEMETRY_ENABLED", "true")
    click.echo("✓ Telemetry enabled")
```

---

## What Developers See in PostHog (Self-Hosted)

### Dashboard Example

```
PostHog Dashboard (http://localhost:8000)

┌─────────────────────────────────────────────────────────┐
│ Events (Last 7 Days)                                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  memory_created          1,234 events                  │
│  search_performed          456 events                  │
│  error_occurred             12 events                  │
│  feature_used              789 events                  │
│                                                         │
│  [View Details]                                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ System Distribution                                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  OS:                                                   │
│    Linux:    65%                                       │
│    macOS:    25%                                       │
│    Windows:  10%                                       │
│                                                         │
│  Version:                                              │
│    1.0.0:    80%                                       │
│    0.9.5:    20%                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘

NO USER-LEVEL DATA VISIBLE
All tracking is anonymous and aggregated
```

---

## Example: Actual Telemetry Events (OSS)

### Event 1: Memory Created

```json
{
  "event": "memory_created",
  "distinct_id": "sha256:a1b2c3d4e5f6789...",
  "properties": {
    "type": "text",
    "has_metadata": true,
    "edition": "opensource",
    "version": "1.0.0",
    "os": "Linux",
    "arch": "x64",
    "python_version": "3.11.5",
    "$lib": "posthog-python",
    "$lib_version": "3.0.0"
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Notice**: No content, no user ID, no identifying information!

### Event 2: Search Performed

```json
{
  "event": "search_performed",
  "distinct_id": "sha256:a1b2c3d4e5f6789...",
  "properties": {
    "result_count_bucket": "10-50",  // Not exact count
    "duration_bucket": "100-500ms",  // Bucketed for privacy
    "edition": "opensource",
    "version": "1.0.0",
    "os": "Linux"
  },
  "timestamp": "2025-01-15T10:32:15Z"
}
```

**Notice**: Bucketed data, not exact values!

### Event 3: Error Occurred

```json
{
  "event": "error_occurred",
  "distinct_id": "sha256:a1b2c3d4e5f6789...",
  "properties": {
    "error_type": "database_connection_failed",
    "component": "neo4j",
    "edition": "opensource",
    "version": "1.0.0"
  },
  "timestamp": "2025-01-15T10:35:00Z"
}
```

**Notice**: Error type only, no error messages or stack traces!

---

## Developer Inspection Tools

### 1. Environment Variables (Full Control)

```bash
# .env
TELEMETRY_ENABLED=true          # Toggle on/off
TELEMETRY_PROVIDER=posthog      # Choose provider
TELEMETRY_DEBUG=true            # See what's being sent
POSTHOG_HOST=http://localhost:8000  # Use self-hosted
```

### 2. Debug Mode (See Events Before Sending)

```python
# When TELEMETRY_DEBUG=true
DEBUG: Telemetry Event
  Event: memory_created
  Anonymous ID: sha256:a1b2c3d4...
  Properties: {
    "type": "text",
    "has_metadata": true,
    "edition": "opensource",
    "os": "Linux"
  }
  ✓ No PII detected
  ✓ Ready to send
```

### 3. Network Inspection

```bash
# Developers can intercept and inspect
$ tcpdump -i any port 8000 -A

POST /capture HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "api_key": "...",
  "event": "memory_created",
  "properties": {
    "type": "text",
    "edition": "opensource"
  }
}

# They can see EXACTLY what's being sent
```

---

## Documentation in README

```markdown
## Telemetry (OSS Edition)

Papr Memory collects **anonymous** usage statistics to help us improve the software.

### What We Collect

- ✅ **Feature usage**: Which features are used (e.g., "search", "create memory")
- ✅ **Error types**: General error categories (e.g., "database connection failed")
- ✅ **Performance**: Aggregated metrics (e.g., "query took 1-5 seconds")
- ✅ **System info**: OS, architecture, software version

### What We DON'T Collect

- ❌ **Your data**: Memory content, search queries, file names
- ❌ **Personal info**: User IDs, emails, IP addresses
- ❌ **Identifiable data**: Anything that could link to you

### How to Opt Out

```bash
# In your .env file
TELEMETRY_ENABLED=false
```

Or at first run, choose "No" when prompted.

### Transparency

- **View the code**: `core/services/telemetry.py`
- **See exactly what's sent**: Set `TELEMETRY_DEBUG=true`
- **Use self-hosted analytics**: Configure your own PostHog instance
- **Read our policy**: `docs/TELEMETRY.md`

### Self-Hosted PostHog (Recommended)

For complete control, run your own PostHog:

```bash
docker run -d --name posthog \
  -p 8000:8000 \
  posthog/posthog:latest
```

Then configure:
```bash
POSTHOG_HOST=http://localhost:8000
POSTHOG_API_KEY=  # Leave empty for self-hosted
```

Now YOU control all telemetry data!
```

---

## Comparison: What Each Edition Sees

### Cloud (Your View)

```python
# Amplitude Dashboard
User: developer_abc123
Events:
  - memory_created (user_1234, email: alice@example.com, country: US)
  - memory_created (user_5678, email: bob@example.com, country: CA)
  - search_performed (user_1234, query_length: 15, results: 42)

# You see EVERYTHING to optimize your product
```

### OSS (Their View)

```python
# PostHog Self-Hosted Dashboard
Anonymous User: sha256:a1b2c3...
Events:
  - memory_created (type: text, has_metadata: true)
  - memory_created (type: document, has_metadata: false)
  - search_performed (result_count_bucket: "10-50", duration_bucket: "100-500ms")

# They see NOTHING identifying, aggregate stats only
```

---

## Trust Building

### Open Source = Auditable

```python
# Anyone can read the code and verify
# core/services/telemetry.py

def _anonymize_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
    """Remove any PII from properties"""
    sensitive_keys = [
        'email', 'ip_address', 'username', 'content', 'query', 
        'filepath', 'filename', 'user_id', 'developer_id'
    ]
    
    # Create a new dictionary with only non-sensitive keys
    anonymized_properties = {
        k: v for k, v in properties.items() 
        if k not in sensitive_keys
    }
    
    # Apply differential privacy to numerical values
    if 'duration_ms' in anonymized_properties:
        anonymized_properties['duration_bucket'] = self._get_duration_bucket(
            anonymized_properties['duration_ms']
        )
        del anonymized_properties['duration_ms']
    
    return anonymized_properties
```

**Developers can verify**: The code does what it says it does!

---

## Bottom Line

For OSS users:

1. **They see the prompt** on first run
2. **They can opt out** easily (TELEMETRY_ENABLED=false)
3. **They can inspect** what's being sent (debug mode, network capture)
4. **They can self-host** PostHog for complete control
5. **They can audit** the source code
6. **NO personal data** is ever collected

For cloud users:

1. **You see everything** (developer_id, end_user_id, emails, geo)
2. **You can optimize** based on real usage data
3. **You can track** developer growth and retention
4. **Full analytics** for product decisions

**Both editions work perfectly** - the telemetry service handles it transparently!

