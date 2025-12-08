#!/usr/bin/env python3
"""
Find and report duplicate scripts in scripts/ directory
"""
import os
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

# Get all Python and shell scripts in root
root_scripts = set()
for ext in ['.py', '.sh']:
    for file in SCRIPTS_DIR.glob(f'*{ext}'):
        if file.is_file():
            root_scripts.add(file.name)

# Get all scripts in subfolders
subfolder_scripts = {}
subfolders = ['setup', 'migration', 'testing', 'deployment', 'debugging', 
              'maintenance', 'opensource', 'generators', 'utils', 'custom_schema']

for subfolder in subfolders:
    subfolder_path = SCRIPTS_DIR / subfolder
    if subfolder_path.exists():
        for ext in ['.py', '.sh']:
            for file in subfolder_path.glob(f'*{ext}'):
                if file.is_file():
                    if file.name not in subfolder_scripts:
                        subfolder_scripts[file.name] = []
                    subfolder_scripts[file.name].append(subfolder)

# Find duplicates
print("🔍 Checking for duplicate scripts...")
print("=" * 60)
print()

duplicates_found = False
for script in sorted(root_scripts):
    if script in subfolder_scripts:
        duplicates_found = True
        locations = subfolder_scripts[script]
        print(f"⚠️  DUPLICATE: {script}")
        print(f"   Root: scripts/{script}")
        for loc in locations:
            print(f"   Also in: scripts/{loc}/{script}")
        print()

if not duplicates_found:
    print("✅ No duplicates found!")
else:
    print(f"\n📊 Found {len([s for s in root_scripts if s in subfolder_scripts])} duplicate(s)")

