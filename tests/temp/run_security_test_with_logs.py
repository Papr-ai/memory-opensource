#!/usr/bin/env python3
"""
Script to run security schema Neo4j test with proper log capture
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def main():
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Log file paths
    test_log = logs_dir / f"security_test_{timestamp}.log"
    test_output = logs_dir / f"security_test_output_{timestamp}.txt"
    
    print(f"🚀 Running security schema Neo4j test...")
    print(f"📝 Test output will be saved to: {test_output}")
    print(f"📋 Server logs are in: logs/app_2025-10-28.log")
    print("=" * 60)
    
    try:
        # Run the test and capture both stdout and stderr
        result = subprocess.run([
            "poetry", "run", "python", "tests/test_security_schema_neo4j.py"
        ], 
        capture_output=True, 
        text=True, 
        timeout=300  # 5 minute timeout
        )
        
        # Write output to file
        with open(test_output, 'w') as f:
            f.write("=== SECURITY SCHEMA NEO4J TEST OUTPUT ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
        
        # Also print to console
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        print(f"\n📝 Full test output saved to: {test_output}")
        
        if result.returncode == 0:
            print("✅ Test completed successfully!")
        else:
            print(f"❌ Test failed with exit code: {result.returncode}")
            
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return 1
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())




