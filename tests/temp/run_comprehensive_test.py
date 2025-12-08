#!/usr/bin/env python3
"""
Comprehensive Test Runner for Code Schema End-to-End Testing

This script runs the enhanced pytest with comprehensive validation and reporting.
It demonstrates the new validation framework that checks:
- HTTP status codes
- Pydantic model compliance  
- Schema structure validation
- Memory processing completion
- Graph node creation
- Relationship validation
- Search functionality
- Data integrity
"""

import asyncio
import subprocess
import sys
from pathlib import Path

def run_test():
    """Run the comprehensive test with proper environment setup"""
    
    # Change to the project directory
    project_root = Path(__file__).parent
    
    print("🚀 Starting Comprehensive Code Schema End-to-End Test")
    print("=" * 60)
    print()
    print("This test will validate:")
    print("✅ Schema creation with Pydantic model compliance")
    print("✅ Memory processing with intelligent waiting")
    print("✅ Custom node and relationship generation")
    print("✅ Graph structure validation")
    print("✅ Search functionality with agentic graph")
    print("✅ Data integrity across all operations")
    print()
    print("⏱️  Expected duration: 2-4 minutes")
    print("🔄 The test will wait intelligently for async processing")
    print()
    
    # Build the command to run all tests in the class
    cmd = [
        "poetry", "run", "python", "-m", "pytest",
        "tests/test_code_schema_end_to_end.py::TestCodeSchemaEndToEnd",
        "-v", "-s", "--tb=short"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Run the test
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        print("\n" + "=" * 60)
        if result.returncode == 0:
            print("🎉 Test completed successfully!")
            print("📊 Check the detailed validation report above")
        else:
            print("❌ Test failed - check the output above for details")
            print("💡 Common issues:")
            print("   - Server not running (start with: poetry run uvicorn main:app --host 0.0.0.0 --port 8000)")
            print("   - Environment variables not set (source .env)")
            print("   - SSL certificate issues (check SSL_CERT_FILE)")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_test()
    sys.exit(exit_code)
