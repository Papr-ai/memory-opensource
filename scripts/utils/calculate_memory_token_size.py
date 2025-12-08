#!/usr/bin/env python3
"""
Calculate total tokenSize across all memories
"""
import json
from pathlib import Path

def main():
    # Get the JSON file path
    script_dir = Path(__file__).parent
    json_file = script_dir / "tokenSizeMemories.json"
    
    print("=" * 80)
    print("Memories Token Size Analysis")
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
    
    # Calculate total token size
    total_token_size = 0
    max_token_size = 0
    min_token_size = float('inf')
    
    for edge in edges:
        token_size = edge.get('node', {}).get('tokenSize', 0)
        total_token_size += token_size
        max_token_size = max(max_token_size, token_size)
        if token_size > 0:  # Only consider non-zero values for min
            min_token_size = min(min_token_size, token_size)
    
    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"📊 Total Memories:              {memory_count:,}")
    print(f"🔢 Total tokenSize:             {total_token_size:,}")
    print()
    
    if memory_count > 0:
        avg_token_size = total_token_size / memory_count
        print(f"📈 Average tokenSize per memory: {avg_token_size:,.2f}")
        
        if min_token_size != float('inf'):
            print(f"📉 Minimum tokenSize:            {min_token_size:,}")
        else:
            print(f"📉 Minimum tokenSize:            N/A (all zeros)")
            
        print(f"📈 Maximum tokenSize:            {max_token_size:,}")
    
    print()
    print("=" * 80)
    print(f"✅ Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

