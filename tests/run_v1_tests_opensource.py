#!/usr/bin/env python3
"""
Simple runner for V1 Endpoints Sequential Test Suite - Open Source Edition

This script provides an easy way to run all v1 endpoint tests sequentially
for the open source edition. Only includes routes available in OSS.

Run with:
    poetry run python tests/run_v1_tests_opensource.py
    python tests/run_v1_tests_opensource.py
"""

import os
# Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION before any imports that might load protobuf/sentencepiece
# This must be set BEFORE importing any modules that use protobuf
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import asyncio
import sys
from pathlib import Path

# Add the project root and tests directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import from the same directory
import test_v1_endpoints_sequential_opensource

if __name__ == "__main__":
    print("🚀 Starting V1 Endpoints Sequential Test Suite (Open Source Edition)")
    print("=" * 60)
    print("This will run all v1 endpoint tests sequentially for OSS routes:")
    print("  ✅ Memory, User, Feedback, Sync, Telemetry, Schema")
    print("  ❌ Excludes: Document (cloud-only), GraphQL (cloud-only)")
    print("=" * 60)
    
    try:
        asyncio.run(test_v1_endpoints_sequential_opensource.main())
        print("\n✅ Test suite completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)
