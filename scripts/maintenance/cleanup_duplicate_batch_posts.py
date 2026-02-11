"""
Cleanup script to delete duplicate batch_memories Posts.

After our fix, batch memories are now stored within the document Post,
so standalone batch_memories Posts created BEFORE the fix are now redundant.

This script safely deletes Posts with:
- type: "batch_memories" 
- No extractionResultFile (not a document Post)
- Created before a certain date (to avoid deleting valid new Posts)
"""

import asyncio
import httpx
import json
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import os

# Load environment variables conditionally
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()

PARSE_SERVER_URL = os.getenv("PARSE_SERVER_URL", "").replace("/parse", "") + "/parse"
PARSE_APPLICATION_ID = os.getenv("PARSE_APPLICATION_ID")
PARSE_MASTER_KEY = os.getenv("PARSE_MASTER_KEY")


def get_parse_headers() -> Dict[str, str]:
    """Get Parse Server headers with master key"""
    return {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }


async def find_duplicate_batch_posts(
    cutoff_date: Optional[str] = None,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find batch_memories Posts that are duplicates (standalone, not merged with document).
    
    Args:
        cutoff_date: ISO date string (e.g., "2025-10-22T03:54:00.000Z"). 
                     Only find Posts created BEFORE this date.
        user_id: Optional user_id to filter by specific user
    
    Returns:
        List of Post objects that are duplicates
    """
    # Build query: type=batch_memories AND no extractionResultFile AND no uploadId
    where_clause = {
        "type": "batch_memories",
        "extractionResultFile": {"$exists": False},
        "uploadId": {"$exists": False}
    }
    
    if cutoff_date:
        where_clause["createdAt"] = {"$lt": {"__type": "Date", "iso": cutoff_date}}
    
    if user_id:
        where_clause["user"] = {
            "__type": "Pointer",
            "className": "_User",
            "objectId": user_id
        }
    
    params = {
        "where": json.dumps(where_clause),
        "limit": 1000,  # Adjust if you have more
        "order": "createdAt",
        "keys": "objectId,type,createdAt,batchMetadata,content,uploadId,extractionResultFile"
    }
    
    url = f"{PARSE_SERVER_URL}/parse/classes/Post"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params, headers=get_parse_headers())
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print(f"✅ Found {len(results)} duplicate batch_memories Posts")
            return results
        else:
            print(f"❌ Failed to query Posts: {response.status_code} - {response.text}")
            return []


async def delete_post(post_id: str) -> bool:
    """Delete a Post by objectId"""
    url = f"{PARSE_SERVER_URL}/parse/classes/Post/{post_id}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.delete(url, headers=get_parse_headers())
        
        if response.status_code == 200:
            return True
        else:
            print(f"❌ Failed to delete Post {post_id}: {response.status_code} - {response.text}")
            return False


async def delete_duplicate_posts(
    cutoff_date: Optional[str] = None,
    user_id: Optional[str] = None,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Delete duplicate batch_memories Posts.
    
    Args:
        cutoff_date: Only delete Posts created before this date (ISO format)
        user_id: Only delete Posts for this user
        dry_run: If True, only print what would be deleted (don't actually delete)
    
    Returns:
        Summary with counts of deleted Posts
    """
    print("=" * 80)
    print("🔍 DUPLICATE BATCH_MEMORIES POST CLEANUP")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (no deletions)' if dry_run else 'LIVE (will delete)'}")
    print(f"Cutoff date: {cutoff_date or 'None (all dates)'}")
    print(f"User filter: {user_id or 'All users'}")
    print()
    
    # Find duplicates
    duplicates = await find_duplicate_batch_posts(cutoff_date, user_id)
    
    if not duplicates:
        print("✅ No duplicate Posts found!")
        return {"found": 0, "deleted": 0}
    
    print(f"\n📋 Found {len(duplicates)} duplicate Posts:")
    print("-" * 80)
    for post in duplicates:
        post_id = post.get("objectId")
        created = post.get("createdAt")
        count = (post.get("batchMetadata") or {}).get("count", "?")
        print(f"  • {post_id} | Created: {created} | Batch size: {count}")
    
    if dry_run:
        print("\n⚠️  DRY RUN: No Posts were deleted.")
        print("   Run with dry_run=False to actually delete these Posts.")
        return {"found": len(duplicates), "deleted": 0}
    
    # Confirm deletion
    print("\n⚠️  WARNING: You are about to DELETE these Posts permanently!")
    confirm = input("Type 'DELETE' to confirm: ")
    
    if confirm != "DELETE":
        print("❌ Deletion cancelled.")
        return {"found": len(duplicates), "deleted": 0}
    
    # Delete Posts
    print("\n🗑️  Deleting Posts...")
    deleted_count = 0
    failed_count = 0
    
    for post in duplicates:
        post_id = post.get("objectId")
        success = await delete_post(post_id)
        if success:
            deleted_count += 1
            print(f"  ✅ Deleted {post_id}")
        else:
            failed_count += 1
            print(f"  ❌ Failed to delete {post_id}")
    
    print("\n" + "=" * 80)
    print(f"✅ Deletion complete!")
    print(f"   - Found: {len(duplicates)}")
    print(f"   - Deleted: {deleted_count}")
    print(f"   - Failed: {failed_count}")
    print("=" * 80)
    
    return {
        "found": len(duplicates),
        "deleted": deleted_count,
        "failed": failed_count
    }


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Delete duplicate batch_memories Posts created before our fix"
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        help="Only delete Posts created before this date (ISO format, e.g., '2025-10-22T03:54:00.000Z')",
        default="2025-10-22T03:54:00.000Z"  # Default: Before the last test (where fix worked)
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="Only delete Posts for this user (Parse objectId)",
        default=None
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run mode (default: True). Pass --no-dry-run to actually delete."
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Disable dry run and actually delete Posts"
    )
    
    args = parser.parse_args()
    
    result = await delete_duplicate_posts(
        cutoff_date=args.cutoff_date,
        user_id=args.user_id,
        dry_run=args.dry_run
    )
    
    return result


if __name__ == "__main__":
    asyncio.run(main())

