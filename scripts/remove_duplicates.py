#!/usr/bin/env python3
"""
Remove duplicate scripts from root, keeping only the ones in subfolders
"""
import os
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

# Subfolders to check
subfolders = ['setup', 'migration', 'testing', 'deployment', 'debugging', 
              'maintenance', 'opensource', 'generators', 'utils', 'custom_schema']

# Get all scripts in subfolders
subfolder_scripts = set()
for subfolder in subfolders:
    subfolder_path = SCRIPTS_DIR / subfolder
    if subfolder_path.exists():
        for ext in ['.py', '.sh']:
            for file in subfolder_path.glob(f'*{ext}'):
                if file.is_file():
                    subfolder_scripts.add(file.name)

# Find and remove duplicates from root
print("🗑️  Removing duplicate scripts from root...")
print("=" * 60)
print()

removed = 0
kept = 0

# Get all scripts in root
for ext in ['.py', '.sh']:
    for file in SCRIPTS_DIR.glob(f'*{ext}'):
        if file.name in subfolder_scripts:
            # This is a duplicate - remove it
            try:
                file.unlink()
                print(f"✅ Removed: {file.name} (exists in subfolder)")
                removed += 1
            except Exception as e:
                print(f"❌ Error removing {file.name}: {e}")
        else:
            # Keep this file (it's not in any subfolder)
            kept += 1
            if file.name not in ['find_duplicates.py', 'remove_duplicates.py', 'organize_files.py', 'organize_docs_and_scripts.sh']:
                print(f"ℹ️  Keeping: {file.name} (not in subfolders)")

print()
print("=" * 60)
print(f"📊 Summary: {removed} removed, {kept} kept in root")
print()
print("✅ Duplicate cleanup complete!")

