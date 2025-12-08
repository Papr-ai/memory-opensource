"""
Utility to read the app version from pyproject.toml.
Used by worker scripts to construct build IDs.
"""
import tomllib
from pathlib import Path


def get_app_version() -> str:
    """Read semantic version from pyproject.toml."""
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    
    if not pyproject_path.exists():
        # Fallback if pyproject.toml not found
        return "0.1.0"
    
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            return data.get("tool", {}).get("poetry", {}).get("version", "0.1.0")
    except Exception:
        # Fallback on any parse error
        return "0.1.0"


__version__ = get_app_version()

