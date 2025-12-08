#!/usr/bin/env python3
"""
Calculate total storageSize across all memories
"""
import json
from pathlib import Path

def format_bytes(bytes_value):
    """Format bytes into human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:,.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:,.2f} PB"

def main():
    # Get the JSON file path
    script_dir = Path(__file__).parent
    json_file = script_dir / "storageCountMemories.json"
    
    print("=" * 80)
    print("Memories Storage Size Analysis")
    print("=" * 80)
    print()
    print(f"Reading file: {json_file}")
    
    if not json_file.exists():
        print(f"❌ Error: File not found: {json_file}")
        return
    
    # Read and parse JSON
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return
    
    # Extract edges from memories
    edges = data.get('data', {}).get('memories', {}).get('edges', [])
    memory_count = len(edges)
    
    print(f"Found {memory_count} memories")
    print()
    
    # Calculate total storage size
    total_storage_size = 0
    max_storage_size = 0
    min_storage_size = float('inf')
    
    for edge in edges:
        storage_size = edge.get('node', {}).get('storageSize', 0)
        total_storage_size += storage_size
        max_storage_size = max(max_storage_size, storage_size)
        if storage_size > 0:  # Only consider non-zero values for min
            min_storage_size = min(min_storage_size, storage_size)
    
    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"📊 Total Memories:                {memory_count:,}")
    print(f"💾 Total storageSize (bytes):     {total_storage_size:,}")
    print(f"💾 Total storageSize (readable):  {format_bytes(total_storage_size)}")
    print()
    
    if memory_count > 0:
        avg_storage_size = total_storage_size / memory_count
        print(f"📈 Average storageSize per memory: {avg_storage_size:,.2f} bytes ({format_bytes(avg_storage_size)})")
        
        if min_storage_size != float('inf'):
            print(f"📉 Minimum storageSize:            {min_storage_size:,} bytes ({format_bytes(min_storage_size)})")
        else:
            print(f"📉 Minimum storageSize:            N/A (all zeros)")
            
        print(f"📈 Maximum storageSize:            {max_storage_size:,} bytes ({format_bytes(max_storage_size)})")
    
    print()
    print("=" * 80)
    print(f"✅ Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

