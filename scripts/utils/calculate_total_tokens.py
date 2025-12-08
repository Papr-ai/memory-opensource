#!/usr/bin/env python3
"""
Calculate total addMemoryTokenCount across all workspace_followers
"""
import json
from pathlib import Path

def main():
    # Get the JSON file path
    script_dir = Path(__file__).parent
    json_file = script_dir / "AddMemoryTokenCount.json"
    
    print("=" * 80)
    print("Workspace Followers Token Count Analysis")
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
    
    # Extract edges
    edges = data.get('data', {}).get('workspace_followers', {}).get('edges', [])
    follower_count = len(edges)
    
    print(f"Found {follower_count} workspace followers")
    print()
    
    # Calculate total tokens
    total_tokens = 0
    max_tokens = 0
    min_tokens = float('inf')
    
    for edge in edges:
        token_count = edge.get('node', {}).get('addMemoryTokenCount', 0)
        total_tokens += token_count
        max_tokens = max(max_tokens, token_count)
        min_tokens = min(min_tokens, token_count)
    
    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"📊 Total Workspace Followers:    {follower_count:,}")
    print(f"🔢 Total addMemoryTokenCount:    {total_tokens:,}")
    print()
    
    if follower_count > 0:
        avg_tokens = total_tokens / follower_count
        print(f"📈 Average tokens per follower:  {avg_tokens:,.2f}")
        print(f"📉 Minimum tokens:               {min_tokens:,}")
        print(f"📈 Maximum tokens:               {max_tokens:,}")
    
    print()
    print("=" * 80)
    print(f"✅ Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
