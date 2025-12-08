"""
Configuration and Feature Flags for Papr Memory

This module provides a simple, config-file based feature flag system.
No external services required - just YAML configs and environment variables.
"""

from .features import FeatureFlags, get_features

__all__ = ["FeatureFlags", "get_features"]

