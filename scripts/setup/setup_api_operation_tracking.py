#!/usr/bin/env python3
"""
Setup API Operation Tracking

This script:
1. Adds necessary fields to Interaction class in Parse Server
2. Creates indexes for efficient querying
3. Initializes MongoDB Time Series collection for detailed logs
4. Verifies the setup

Run with: poetry run python scripts/setup_api_operation_tracking.py
"""

import os
import sys
import asyncio
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv
import logging
import certifi

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables conditionally
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
if not MONGO_URI:
    logger.error("MONGO_URI not set")
    sys.exit(1)


async def setup_api_operation_tracking():
    """Setup API operation tracking infrastructure"""
    
    logger.info("=" * 60)
    logger.info("Setting up API Operation Tracking")
    logger.info("=" * 60)
    
    # Connect to MongoDB
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client.get_default_database()
    
    logger.info(f"\n✓ Connected to MongoDB: {db.name}")
    
    # 1. Create indexes for Interaction class
    logger.info("\n=== Creating Interaction Indexes ===")
    
    interaction_collection = db["Interaction"]
    
    # Existing index (keep it)
    interaction_collection.create_index([
        ("_p_user", ASCENDING),
        ("_p_workspace", ASCENDING),
        ("type", ASCENDING),
        ("month", ASCENDING),
        ("year", ASCENDING)
    ], name="interaction_lookup")
    logger.info("✓ Created interaction_lookup index")
    
    # NEW: Index for organization-level queries
    interaction_collection.create_index([
        ("_p_organization", ASCENDING),
        ("type", ASCENDING),
        ("month", ASCENDING),
        ("year", ASCENDING)
    ], name="interaction_org_lookup")
    logger.info("✓ Created interaction_org_lookup index")
    
    # NEW: Index for API operation queries
    interaction_collection.create_index([
        ("route", ASCENDING),
        ("method", ASCENDING),
        ("isMemoryOperation", ASCENDING),
        ("month", ASCENDING),
        ("year", ASCENDING)
    ], name="interaction_api_operation_lookup")
    logger.info("✓ Created interaction_api_operation_lookup index")
    
    # NEW: Index for memory operation counts
    interaction_collection.create_index([
        ("_p_organization", ASCENDING),
        ("isMemoryOperation", ASCENDING),
        ("month", ASCENDING),
        ("year", ASCENDING)
    ], name="interaction_memory_ops_lookup")
    logger.info("✓ Created interaction_memory_ops_lookup index")
    
    # 2. Create Time Series collection for detailed logs
    logger.info("\n=== Creating Time Series Collection ===")
    
    collections = db.list_collection_names()
    
    if "api_operation_logs" in collections:
        logger.info("⊙ api_operation_logs collection already exists")
    else:
        # Create time series collection
        db.create_collection(
            "api_operation_logs",
            timeseries={
                "timeField": "timestamp",
                "metaField": "metadata",
                "granularity": "seconds"
            },
            expireAfterSeconds=7776000  # 90 days retention
        )
        logger.info("✓ Created api_operation_logs time series collection")
    
    # Create indexes for time series collection
    logger.info("\n=== Creating Time Series Indexes ===")
    
    time_series_collection = db["api_operation_logs"]
    
    time_series_collection.create_index([
        ("user_id", ASCENDING),
        ("timestamp", DESCENDING)
    ], name="ts_user_time")
    logger.info("✓ Created ts_user_time index")
    
    time_series_collection.create_index([
        ("organization_id", ASCENDING),
        ("timestamp", DESCENDING)
    ], name="ts_org_time")
    logger.info("✓ Created ts_org_time index")
    
    time_series_collection.create_index([
        ("route", ASCENDING),
        ("method", ASCENDING),
        ("timestamp", DESCENDING)
    ], name="ts_route_method_time")
    logger.info("✓ Created ts_route_method_time index")
    
    time_series_collection.create_index([
        ("is_memory_operation", ASCENDING),
        ("timestamp", DESCENDING)
    ], name="ts_memory_op_time")
    logger.info("✓ Created ts_memory_op_time index")
    
    time_series_collection.create_index([
        ("workspace_id", ASCENDING),
        ("is_memory_operation", ASCENDING),
        ("timestamp", DESCENDING)
    ], name="ts_workspace_memory_op_time")
    logger.info("✓ Created ts_workspace_memory_op_time index")
    
    # 3. Verify setup
    logger.info("\n=== Verification ===")
    
    # Check Interaction indexes
    interaction_indexes = list(interaction_collection.list_indexes())
    logger.info(f"✓ Interaction collection has {len(interaction_indexes)} indexes")
    
    # Check Time Series collection
    time_series_indexes = list(time_series_collection.list_indexes())
    logger.info(f"✓ api_operation_logs collection has {len(time_series_indexes)} indexes")
    
    # Show sample Interaction document structure
    logger.info("\n=== Sample Interaction Document Structure ===")
    sample_interaction = {
        "_id": "sample_id",
        "_p_user": "_User$user_123",
        "_p_workspace": "WorkSpace$workspace_456",
        "_p_organization": "Organization$org_789",  # NEW
        "type": "api_operation",  # NEW enum value
        "route": "v1/memory",  # NEW
        "method": "POST",  # NEW
        "isMemoryOperation": True,  # NEW
        "month": 10,
        "year": 2025,
        "count": 42,
        "_created_at": "2025-10-01T00:00:00.000Z",
        "_updated_at": "2025-10-03T12:00:00.000Z"
    }
    
    logger.info("\nInteraction document should include:")
    for key, value in sample_interaction.items():
        marker = "  # NEW" if key in ["_p_organization", "route", "method", "isMemoryOperation"] or (key == "type" and value == "api_operation") else ""
        logger.info(f"  {key}: {value}{marker}")
    
    # Show sample APIOperationLog structure
    logger.info("\n=== Sample APIOperationLog Document Structure ===")
    sample_log = {
        "timestamp": "2025-10-03T12:00:00.000Z",
        "user_id": "user_123",
        "workspace_id": "workspace_456",
        "organization_id": "org_789",
        "developer_id": "dev_abc",
        "route": "v1/memory",
        "method": "POST",
        "operation_type": "add_memory",
        "is_memory_operation": True,
        "memory_id": "mem_xyz",
        "latency_ms": 123.45,
        "status_code": 200,
        "api_key": "hashed_key_123",
        "client_type": "papr_plugin",
        "metadata": {}
    }
    
    logger.info("\nAPIOperationLog document includes:")
    for key, value in sample_log.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Setup Complete!")
    logger.info("=" * 60)
    
    logger.info("\nNext Steps:")
    logger.info("1. Update your API routes to use get_api_operation_tracker()")
    logger.info("2. See docs/API_OPERATION_TRACKING.md for integration examples")
    logger.info("3. Test with: poetry run python scripts/test_api_tracking.py")
    
    client.close()


if __name__ == "__main__":
    asyncio.run(setup_api_operation_tracking())

