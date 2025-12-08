import subprocess
import datetime
import os

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f"logs/schema_id_test_output_{timestamp}.txt"

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

print(f"Running focused schema ID test. Full output will be saved to: {log_file_path}")

try:
    # Use poetry run python to ensure the correct environment
    result = subprocess.run(
        ["poetry", "run", "python", "test_schema_id_only.py"],
        capture_output=True,
        text=True,
        check=True,
        cwd="/Users/amirkabbara/Documents/GitHub/memory"  # Ensure correct working directory
    )
    stdout = result.stdout
    stderr = result.stderr
    exit_code = result.returncode

    with open(log_file_path, "w") as f:
        f.write(f"=== FOCUSED SCHEMA ID TEST OUTPUT ===\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Exit code: {exit_code}\n\n")
        f.write(f"=== STDOUT ===\n{stdout}\n")
        if stderr:
            f.write(f"=== STDERR ===\n{stderr}\n")

    print(f"\n📝 Full test output saved to: {log_file_path}")
    print(f"\n=== LIVE OUTPUT ===")
    print(stdout)
    if stderr:
        print(f"\n=== ERRORS ===")
        print(stderr)
    
    if exit_code == 0:
        print("✅ Test completed successfully!")
    else:
        print(f"❌ Test completed with errors. See {log_file_path} for details.")

except subprocess.CalledProcessError as e:
    with open(log_file_path, "w") as f:
        f.write(f"=== FOCUSED SCHEMA ID TEST OUTPUT ===\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Exit code: {e.returncode}\n\n")
        f.write(f"=== STDOUT ===\n{e.stdout}\n")
        f.write(f"=== STDERR ===\n{e.stderr}\n")
    print(f"\n❌ Test failed with CalledProcessError. See {log_file_path} for details.")
    print(f"Exit code: {e.returncode}")
    if e.stdout:
        print(f"STDOUT: {e.stdout}")
    if e.stderr:
        print(f"STDERR: {e.stderr}")
except FileNotFoundError:
    print("Error: 'poetry' command not found. Ensure Poetry is installed and in your PATH.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

