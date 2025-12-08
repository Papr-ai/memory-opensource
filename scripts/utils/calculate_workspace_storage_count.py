#!/usr/bin/env python3
"""
Calculate total storageCount across all workspace_followers
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
    json_file = script_dir / "storageCountWorkspaceFollowers.json"
    
    print("=" * 80)
    print("Workspace Followers Storage Count Analysis")
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
    
    # Extract edges from workspace_followers
    edges = data.get('data', {}).get('workspace_followers', {}).get('edges', [])
    follower_count = len(edges)
    
    print(f"Found {follower_count} workspace followers")
    print()
    
    # Calculate total storage count
    total_storage_count = 0
    max_storage_count = 0
    min_storage_count = float('inf')
    
    for edge in edges:
        storage_count = edge.get('node', {}).get('storageCount', 0)
        total_storage_count += storage_count
        max_storage_count = max(max_storage_count, storage_count)
        if storage_count > 0:  # Only consider non-zero values for min
            min_storage_count = min(min_storage_count, storage_count)
    
    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"📊 Total Workspace Followers:       {follower_count:,}")
    print(f"💾 Total storageCount (bytes):      {total_storage_count:,}")
    print(f"💾 Total storageCount (readable):   {format_bytes(total_storage_count)}")
    print()
    
    if follower_count > 0:
        avg_storage_count = total_storage_count / follower_count
        print(f"📈 Average storageCount per follower: {avg_storage_count:,.2f} bytes ({format_bytes(avg_storage_count)})")
        
        if min_storage_count != float('inf'):
            print(f"📉 Minimum storageCount:              {min_storage_count:,} bytes ({format_bytes(min_storage_count)})")
        else:
            print(f"📉 Minimum storageCount:              N/A (all zeros)")
            
        print(f"📈 Maximum storageCount:              {max_storage_count:,} bytes ({format_bytes(max_storage_count)})")
    
    print()
    print("=" * 80)
    print(f"✅ Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

