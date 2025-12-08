#!/usr/bin/env python3
"""
Script to check Temporal worker logs and status
Usage: python scripts/check_worker_logs.py
"""

import subprocess
import sys
import json
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=10
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)

def print_header(text):
    """Print a section header"""
    print("\n" + "=" * 50)
    print(f"   {text}")
    print("=" * 50 + "\n")

def print_section(number, title):
    """Print a numbered section title"""
    print(f"\n{number}  {title}:")
    print("─" * 50)

def main():
    print_header("PAPR Memory - Worker Status Check")
    
    # Check if Docker is running
    code, _, _ = run_command("docker info")
    if code != 0:
        print("❌ Docker is not running. Please start Docker Desktop.")
        sys.exit(1)
    
    # Determine which compose file to use
    compose_file = "docker-compose.yaml"
    code, out, _ = run_command("docker-compose -f docker-compose-split.yaml ps")
    if code == 0 and "memory-worker" in out:
        compose_file = "docker-compose-split.yaml"
        print("📦 Using: docker-compose-split.yaml (split services)")
    else:
        print("📦 Using: docker-compose.yaml (all-in-one)")
    
    # 1. Container Status
    print_section("1️⃣ ", "Container Status")
    code, out, _ = run_command(f"docker-compose -f {compose_file} ps")
    if code == 0:
        print(out)
    else:
        print("❌ Could not get container status")
    
    # 2. Memory Worker Logs
    print_section("2️⃣ ", "Memory Worker Logs (last 50 lines)")
    code, out, _ = run_command(f"docker-compose -f {compose_file} ps")
    if code == 0 and "memory-worker" in out:
        code, logs, _ = run_command(f"docker-compose -f {compose_file} logs --tail=50 memory-worker")
        if logs:
            print(logs)
        else:
            print("⚠️  No logs yet (worker might be starting)")
    else:
        print("⚠️  Memory worker not found (might be in all-in-one mode)")
    
    # 3. Document Worker Logs
    print_section("3️⃣ ", "Document Worker Logs (last 50 lines)")
    code, out, _ = run_command(f"docker-compose -f {compose_file} ps")
    if code == 0 and "document-worker" in out:
        code, logs, _ = run_command(f"docker-compose -f {compose_file} logs --tail=50 document-worker")
        if logs:
            print(logs)
        else:
            print("⚠️  No logs yet (worker might be starting)")
    else:
        print("⚠️  Document worker not found (might be in all-in-one mode)")
    
    # 4. Web Server Health
    print_section("4️⃣ ", "Web Server Health")
    code, out, _ = run_command("curl -s http://localhost:5001/health")
    if code == 0:
        print("✅ Web server is healthy")
        try:
            health = json.loads(out)
            print(json.dumps(health, indent=2))
        except:
            print(out)
    else:
        print("❌ Web server not responding")
    
    # 5. Resource Usage
    print_section("5️⃣ ", "Resource Usage")
    code, out, _ = run_command('docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"')
    if code == 0:
        print(out)
    else:
        print("❌ Could not get resource usage")
    
    # Follow instructions
    print_header("Follow Live Logs")
    print(f"All services:     docker-compose -f {compose_file} logs -f")
    print(f"Memory worker:    docker-compose -f {compose_file} logs -f memory-worker")
    print(f"Document worker:  docker-compose -f {compose_file} logs -f document-worker")
    print(f"Web server:       docker-compose -f {compose_file} logs -f web")
    print()

if __name__ == "__main__":
    main()

