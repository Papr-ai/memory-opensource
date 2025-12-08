#!/usr/bin/env python3
"""
Simple test runner for Feedback End-to-End Test Only

This script runs only the feedback end-to-end test to verify the fix.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from tests.test_feedback_end_to_end import test_feedback_end_to_end

async def main():
    """Run only the feedback end-to-end test."""
    print("🧪 Running Feedback End-to-End Test Only")
    print("=" * 50)
    
    try:
        await test_feedback_end_to_end()
        print("✅ Feedback end-to-end test passed!")
    except Exception as e:
        print(f"❌ Feedback end-to-end test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 