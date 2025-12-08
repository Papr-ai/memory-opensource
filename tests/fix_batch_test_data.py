#!/usr/bin/env python3
"""
Script to fix the batch_quick_add_test_data.json file by properly escaping content.
This reads the malformed JSON and attempts to parse it as Python literal eval,
then saves it as proper JSON.
"""
import json
import re

input_file = "tests/batch_quick_add_test_data.json"
output_file = "tests/batch_quick_add_test_data_fixed.json"

print(f"Reading {input_file}...")

# Read the raw content
with open(input_file, 'r', encoding='utf-8') as f:
    raw_content = f.read()

# The issue is that the content field has unescaped newlines
# We need to fix this by reading it carefully

# Strategy: Replace literal newlines inside string values with \n
# This is tricky because we need to detect when we're inside a string value

def fix_json_newlines(text):
    """Fix unescaped newlines in JSON string values."""
    result = []
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            result.append(char)
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            result.append(char)
            continue
            
        if char == '"':
            # Toggle string state
            in_string = not in_string
            result.append(char)
            continue
            
        if char == '\n' and in_string:
            # This is an unescaped newline inside a string - fix it
            result.append('\\n')
            continue
            
        result.append(char)
    
    return ''.join(result)

print("Fixing unescaped newlines...")
fixed_content = fix_json_newlines(raw_content)

print("Parsing JSON...")
try:
    data = json.loads(fixed_content)
    print(f"✅ Successfully parsed JSON!")
    
    # Validate structure
    if isinstance(data, dict):
        print(f"   - batch_id: {data.get('batch_id', 'N/A')}")
        if 'batch_request' in data and 'memories' in data['batch_request']:
            memories = data['batch_request']['memories']
            print(f"   - memories count: {len(memories)}")
            if memories:
                first_mem = memories[0]
                print(f"   - first memory content length: {len(first_mem.get('content', ''))}")
    
    # Save the fixed JSON
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved fixed JSON to {output_file}")
    print(f"\nYou can now use this file in your test!")
    
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing failed: {e}")
    print(f"   Line {e.lineno}, Column {e.colno}")
    
    # Show context around the error
    lines = fixed_content.split('\n')
    start = max(0, e.lineno - 3)
    end = min(len(lines), e.lineno + 2)
    print(f"\n   Context:")
    for i in range(start, end):
        marker = " >>> " if i == e.lineno - 1 else "     "
        print(f"{marker}{i+1:4d}: {lines[i][:100]}")

