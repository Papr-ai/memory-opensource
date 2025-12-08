#!/usr/bin/env python3
"""
Simple runner for V1 Endpoints Sequential Test Suite

This script provides an easy way to run all v1 endpoint tests sequentially.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root and tests directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import from the same directory
import test_v1_endpoints_sequential

if __name__ == "__main__":
    print("🚀 Starting V1 Endpoints Sequential Test Suite")
    print("=" * 60)
    print("This will run all v1 endpoint tests sequentially to avoid")
    print("parallel execution issues and provide detailed reporting.")
    print("=" * 60)
    
    try:
        asyncio.run(test_v1_endpoints_sequential.main())
        print("\n✅ Test suite completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1) 