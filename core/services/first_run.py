"""
First Run Experience for OSS Edition

Prompts users to consent to telemetry on first run.
Provides transparency about what data is collected.
"""

import os
from pathlib import Path
from typing import Optional
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)

# Store first run status in user's home directory
FIRST_RUN_FILE = Path.home() / ".papr" / "first_run_complete"

def check_first_run() -> bool:
    """
    Check if this is the first run of Papr Memory.
    
    Returns:
        True if first run, False if already configured
    """
    return not FIRST_RUN_FILE.exists()

def mark_first_run_complete():
    """Mark that first run setup is complete"""
    FIRST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIRST_RUN_FILE.touch()
    logger.info("First run setup completed")

def prompt_telemetry_consent() -> Optional[bool]:
    """
    Prompt user for telemetry consent on first run.
    
    Returns:
        True if user consents, False if they opt out, None if already configured
    """
    
    # Skip if not first run
    if not check_first_run():
        return None
    
    # Skip if already configured via environment variable
    if os.getenv("TELEMETRY_ENABLED") is not None:
        logger.info("Telemetry already configured via environment variable")
        mark_first_run_complete()
        return os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"
    
    # Only prompt in OSS edition
    edition = os.getenv("PAPR_EDITION", "opensource").lower()
    if edition != "opensource":
        mark_first_run_complete()
        return None
    
    # Print the consent prompt
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
║  You can:                                                  ║
║    • Opt out anytime: TELEMETRY_ENABLED=false              ║
║    • View our code: core/services/telemetry.py             ║
║    • Read our policy: docs/TELEMETRY.md                    ║
║    • Self-host PostHog for full control                    ║
║                                                             ║
║  [Y] Yes, help improve Papr    [N] No, disable telemetry  ║
║                                                             ║
╚════════════════════════════════════════════════════════════╝
""")
    
    # Get user input (default to No if non-interactive)
    try:
        choice = input("Your choice [Y/n]: ").strip().lower()
    except EOFError:
        # Non-interactive mode (e.g., Docker, systemd)
        logger.info("Non-interactive mode detected, defaulting to telemetry disabled")
        choice = 'n'
    
    enabled = choice != 'n' and choice != 'no'
    
    # Update .env file if it exists
    _update_env_file(enabled)
    
    # Mark first run complete
    mark_first_run_complete()
    
    print(f"\n✓ Telemetry {'enabled' if enabled else 'disabled'}. Thank you!")
    print(f"  You can change this anytime in .env: TELEMETRY_ENABLED={'true' if enabled else 'false'}\n")
    
    return enabled

def _update_env_file(enabled: bool):
    """
    Update .env file with telemetry preference.
    
    Args:
        enabled: Whether telemetry should be enabled
    """
    env_file = Path(".env")
    
    if not env_file.exists():
        # Create .env if it doesn't exist
        logger.info("Creating .env file")
        with open(env_file, "w") as f:
            f.write(f"# Telemetry Settings (set to false to opt out)\n")
            f.write(f"TELEMETRY_ENABLED={'true' if enabled else 'false'}\n")
        return
    
    # Read existing .env
    with open(env_file, "r") as f:
        lines = f.readlines()
    
    # Find and update TELEMETRY_ENABLED line
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith("TELEMETRY_ENABLED="):
            lines[i] = f"TELEMETRY_ENABLED={'true' if enabled else 'false'}\n"
            updated = True
            break
    
    # Add if not found
    if not updated:
        # Add a comment and the setting
        lines.append(f"\n# Telemetry Settings (set to false to opt out)\n")
        lines.append(f"TELEMETRY_ENABLED={'true' if enabled else 'false'}\n")
    
    # Write back
    with open(env_file, "w") as f:
        f.writelines(lines)
    
    logger.info(f"Updated .env: TELEMETRY_ENABLED={'true' if enabled else 'false'}")

def show_telemetry_status():
    """
    Show current telemetry status.
    Useful for CLI commands.
    """
    from core.services.telemetry import get_telemetry
    
    telemetry = get_telemetry()
    status = telemetry.get_status()
    
    print("""
╔════════════════════════════════════════════════════════════╗
║                   Telemetry Status                          ║
╠════════════════════════════════════════════════════════════╣
""")
    print(f"║  Enabled:      {str(status['enabled']).ljust(45)} ║")
    print(f"║  Provider:     {str(status['provider']).ljust(45)} ║")
    print(f"║  Edition:      {str(status['edition']).ljust(45)} ║")
    
    if status.get('host'):
        print(f"║  Host:         {str(status['host']).ljust(45)} ║")
    
    print(f"║  Anonymous ID: {str(status['anonymous_id'][:30]).ljust(45)} ║")
    
    print("""╠════════════════════════════════════════════════════════════╣
║                                                             ║
║  To change:                                                ║
║    • Edit .env: TELEMETRY_ENABLED=true/false               ║
║    • Or set env var: export TELEMETRY_ENABLED=false        ║
║                                                             ║
║  For self-hosted analytics:                                ║
║    • Set POSTHOG_HOST=http://your-posthog-instance         ║
║    • Leave POSTHOG_API_KEY empty for self-hosted           ║
║                                                             ║
║  Learn more:                                               ║
║    • docs/TELEMETRY.md                                     ║
║    • core/services/telemetry.py (view source)              ║
║                                                             ║
╚════════════════════════════════════════════════════════════╝
""")

def disable_telemetry():
    """Disable telemetry (for CLI command)"""
    _update_env_file(False)
    print("✓ Telemetry disabled")
    print("  Restart the service for changes to take effect")

def enable_telemetry():
    """Enable telemetry (for CLI command)"""
    _update_env_file(True)
    print("✓ Telemetry enabled")
    print("  Restart the service for changes to take effect")

