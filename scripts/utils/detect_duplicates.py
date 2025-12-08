#!/usr/bin/env python3
"""
Script to detect duplicate memories in Parse Server and Qdrant for a specific user.
Usage: python scripts/detect_duplicates.py --user_id <user_id>
"""
import os
import sys
import asyncio
import argparse
from collections import defaultdict
import hashlib

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
from dotenv import find_dotenv, load_dotenv

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)
    print(f"Loaded environment from: {ENV_FILE}")
else:
    print("No .env file found")

# Debug environment variables
print(f"PARSE_SERVER_URL: {os.getenv('PARSE_SERVER_URL', 'NOT SET')}")
print(f"QDRANT_URL: {os.getenv('QDRANT_URL', 'NOT SET')}")

from datastore.neo4jconnection import Neo4jConnection
from memory.memory_graph import MemoryGraph
import httpx
from qdrant_client import AsyncQdrantClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuplicateDetector:
    def __init__(self):
        self.parse_url = os.getenv('PARSE_SERVER_URL')
        self.parse_app_id = os.getenv('PARSE_APPLICATION_ID')
        self.parse_master_key = os.getenv('PARSE_MASTER_KEY')
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        self.qdrant_collection = os.getenv('QDRANT_COLLECTION_QWEN4B', 'memories')
        
        # Validate required environment variables
        if not self.parse_url:
            raise ValueError("PARSE_SERVER_URL environment variable is required")
        if not self.parse_url.startswith(('http://', 'https://')):
            self.parse_url = 'https://' + self.parse_url
            print(f"Added https:// protocol to Parse URL: {self.parse_url}")
        
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL environment variable is required")
        if not self.qdrant_url.startswith(('http://', 'https://')):
            self.qdrant_url = 'https://' + self.qdrant_url
            print(f"Added https:// protocol to Qdrant URL: {self.qdrant_url}")
            
        print(f"Using Parse URL: {self.parse_url}")
        print(f"Using Qdrant URL: {self.qdrant_url}")
        
    async def check_parse_duplicates(self, user_id: str):
        """Check for duplicate memories in Parse Server"""
        logger.info(f"Checking Parse Server for duplicates for user: {user_id}")
        
        headers = {
            'X-Parse-Application-Id': self.parse_app_id,
            'X-Parse-Master-Key': self.parse_master_key,
            'Content-Type': 'application/json'
        }
        
        # Fetch all memories for the user
        url = f"{self.parse_url}/parse/classes/Memory"
        params = {
            'where': f'{{"user_read_access":{{"$in":["{user_id}"]}}}}',
            'limit': 1000,
            'keys': 'objectId,content,memoryId,createdAt,metadata'
        }
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.error(f"Failed to connect to Parse Server: {e}")
            logger.warning("Parse Server unavailable - returning empty results")
            return {
                'total_memories': 0,
                'content_duplicates': {},
                'memoryid_duplicates': {},
                'error': str(e)
            }
            
        memories = data.get('results', [])
        logger.info(f"Found {len(memories)} memories in Parse Server")
        
        # Group by content hash to find duplicates
        content_groups = defaultdict(list)
        memoryid_groups = defaultdict(list)
        
        for memory in memories:
            content = memory.get('content', '')
            memory_id = memory.get('memoryId', '')
            object_id = memory.get('objectId', '')
            created_at = memory.get('createdAt', '')
            
            # Hash content for duplicate detection
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            content_groups[content_hash].append({
                'objectId': object_id,
                'memoryId': memory_id,
                'content_preview': content[:100] + '...' if len(content) > 100 else content,
                'createdAt': created_at
            })
            
            # Group by memoryId
            if memory_id:
                memoryid_groups[memory_id].append({
                    'objectId': object_id,
                    'content_preview': content[:100] + '...' if len(content) > 100 else content,
                    'createdAt': created_at
                })
        
        # Find duplicates
        content_duplicates = {k: v for k, v in content_groups.items() if len(v) > 1}
        memoryid_duplicates = {k: v for k, v in memoryid_groups.items() if len(v) > 1}
        
        logger.info(f"Found {len(content_duplicates)} content duplicate groups")
        logger.info(f"Found {len(memoryid_duplicates)} memoryId duplicate groups")
        
        return {
            'total_memories': len(memories),
            'content_duplicates': content_duplicates,
            'memoryid_duplicates': memoryid_duplicates
        }
    
    async def check_qdrant_duplicates(self, user_id: str):
        """Check for duplicate memories in Qdrant"""
        logger.info(f"Checking Qdrant for duplicates for user: {user_id}")
        
        client = AsyncQdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=60.0
        )
        
        try:
            # Get all points for the user
            # We'll scroll through all points and filter by user_id in payload
            scroll_result = await client.scroll(
                collection_name=self.qdrant_collection,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            user_points = []
            points = scroll_result[0]  # First element is the list of points
            
            for point in points:
                payload = point.payload or {}
                if payload.get('user_id') == user_id:
                    user_points.append(point)
            
            # Continue scrolling if there are more points
            next_page_offset = scroll_result[1]  # Second element is the next page offset
            while next_page_offset:
                scroll_result = await client.scroll(
                    collection_name=self.qdrant_collection,
                    limit=1000,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False
                )
                points = scroll_result[0]
                next_page_offset = scroll_result[1]
                
                for point in points:
                    payload = point.payload or {}
                    if payload.get('user_id') == user_id:
                        user_points.append(point)
            
            logger.info(f"Found {len(user_points)} points in Qdrant for user")
            
            # Group by content and memoryId
            content_groups = defaultdict(list)
            memoryid_groups = defaultdict(list)
            
            for point in user_points:
                payload = point.payload or {}
                content = payload.get('content', '')
                memory_id = payload.get('memoryId', '')
                chunk_id = payload.get('chunk_id', '')
                
                # Hash content for duplicate detection
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                content_groups[content_hash].append({
                    'point_id': str(point.id),
                    'chunk_id': chunk_id,
                    'memoryId': memory_id,
                    'content_preview': content[:100] + '...' if len(content) > 100 else content
                })
                
                # Group by memoryId
                if memory_id:
                    memoryid_groups[memory_id].append({
                        'point_id': str(point.id),
                        'chunk_id': chunk_id,
                        'content_preview': content[:100] + '...' if len(content) > 100 else content
                    })
            
            # Find duplicates
            content_duplicates = {k: v for k, v in content_groups.items() if len(v) > 1}
            memoryid_duplicates = {k: v for k, v in memoryid_groups.items() if len(v) > 1}
            
            logger.info(f"Found {len(content_duplicates)} content duplicate groups in Qdrant")
            logger.info(f"Found {len(memoryid_duplicates)} memoryId duplicate groups in Qdrant")
            
            return {
                'total_points': len(user_points),
                'content_duplicates': content_duplicates,
                'memoryid_duplicates': memoryid_duplicates
            }
            
        finally:
            await client.close()
    
    async def detect_all_duplicates(self, user_id: str):
        """Detect duplicates in both Parse and Qdrant"""
        logger.info(f"Starting duplicate detection for user: {user_id}")
        
        # Run both checks in parallel
        parse_task = self.check_parse_duplicates(user_id)
        qdrant_task = self.check_qdrant_duplicates(user_id)
        
        parse_results, qdrant_results = await asyncio.gather(parse_task, qdrant_task)
        
        return {
            'user_id': user_id,
            'parse_server': parse_results,
            'qdrant': qdrant_results
        }

async def main():
    parser = argparse.ArgumentParser(description='Detect duplicate memories')
    parser.add_argument('--user_id', required=True, help='User ID to check for duplicates')
    args = parser.parse_args()
    
    detector = DuplicateDetector()
    results = await detector.detect_all_duplicates(args.user_id)
    
    print("\n" + "="*80)
    print(f"DUPLICATE DETECTION RESULTS FOR USER: {args.user_id}")
    print("="*80)
    
    # Parse Server Results
    parse_data = results['parse_server']
    print(f"\n📊 PARSE SERVER:")
    print(f"   Total memories: {parse_data['total_memories']}")
    print(f"   Content duplicate groups: {len(parse_data['content_duplicates'])}")
    print(f"   MemoryId duplicate groups: {len(parse_data['memoryid_duplicates'])}")
    
    if parse_data['content_duplicates']:
        print(f"\n🔍 Sample Parse content duplicates:")
        for i, (hash_key, duplicates) in enumerate(list(parse_data['content_duplicates'].items())[:3]):
            print(f"   Group {i+1}: {len(duplicates)} duplicates")
            for dup in duplicates:
                print(f"     - {dup['objectId']} | {dup['memoryId']} | {dup['content_preview']}")
    
    # Qdrant Results
    qdrant_data = results['qdrant']
    print(f"\n📊 QDRANT:")
    print(f"   Total points: {qdrant_data['total_points']}")
    print(f"   Content duplicate groups: {len(qdrant_data['content_duplicates'])}")
    print(f"   MemoryId duplicate groups: {len(qdrant_data['memoryid_duplicates'])}")
    
    if qdrant_data['content_duplicates']:
        print(f"\n🔍 Sample Qdrant content duplicates:")
        for i, (hash_key, duplicates) in enumerate(list(qdrant_data['content_duplicates'].items())[:3]):
            print(f"   Group {i+1}: {len(duplicates)} duplicates")
            for dup in duplicates:
                print(f"     - {dup['point_id']} | {dup['chunk_id']} | {dup['memoryId']} | {dup['content_preview']}")
    
    # Summary
    total_parse_dupes = sum(len(dupes) for dupes in parse_data['content_duplicates'].values())
    total_qdrant_dupes = sum(len(dupes) for dupes in qdrant_data['content_duplicates'].values())
    
    print(f"\n🚨 SUMMARY:")
    print(f"   Parse Server: {total_parse_dupes} total duplicate memories")
    print(f"   Qdrant: {total_qdrant_dupes} total duplicate points")
    print(f"   Expected memories: ~143 (from your eval)")
    print(f"   Actual Parse memories: {parse_data['total_memories']}")
    print(f"   Actual Qdrant points: {qdrant_data['total_points']}")
    
    if parse_data['total_memories'] > 200 or qdrant_data['total_points'] > 200:
        print(f"   ⚠️  DUPLICATION CONFIRMED - Way more than expected!")
    
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main()) 