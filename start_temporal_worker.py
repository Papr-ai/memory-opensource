#!/usr/bin/env python3
"""
Temporal Worker for PAPR Memory Server

This worker processes Temporal workflows for batch memory processing.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Force protobuf pure-Python implementation to avoid C-extension descriptor errors in some environments
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Load environment variables from .env early so all imports see them (conditionally based on USE_DOTENV)
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    _ENV_FILE = find_dotenv()
    if _ENV_FILE:
        load_dotenv(_ENV_FILE)

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.converter import DataConverter
from temporalio.contrib.pydantic import pydantic_data_converter
from version import __version__ as APP_VERSION
from cloud_plugins.temporal.client import get_temporal_client
from cloud_plugins.temporal.workflows.batch_memory import (
    ProcessBatchMemoryWorkflow,
    ProcessBatchMemoryFromPostWorkflow,
    ProcessBatchMemoryFromRequestWorkflow,
)
from cloud_plugins.temporal.workflows.document_processing import DocumentProcessingWorkflow
from cloud_plugins.temporal.activities import memory_activities, document_activities
from config import get_features
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


async def main():
    """Main function to start the Temporal worker"""

    # Check if Temporal is enabled
    features = get_features()
    if not features.is_enabled("temporal"):
        logger.error("Temporal is not enabled in current edition")
        return

    try:
        # CRITICAL: Initialize MongoDB client BEFORE any MemoryGraph instances are created
        # This ensures the shared MongoDB singleton is available when workers create MemoryGraph
        from services.mongo_client import get_mongo_db
        logger.info("Pre-initializing MongoDB client singleton for Temporal worker...")
        shared_db = get_mongo_db()
        if shared_db is not None:
            logger.info(f"✅ MongoDB client initialized successfully: {shared_db.name}")
        else:
            logger.error("❌ MongoDB client initialization returned None - workers will fallback to Parse Server!")
        
        # Get Temporal client
        client = await get_temporal_client()
        logger.info("Successfully connected to Temporal")

        # Get task queue from config
        task_queue = "memory-processing"

        # Build ID format: v{semver}+{feature}.{timestamp}
        # See start_all_workers.py for full explanation
        feature_id = "batch-default"
        timestamp = "20251117"  # Date of batch-default release
        default_build_id = f"v{APP_VERSION}+{feature_id}.{timestamp}"
        
        build_id = os.getenv("TEMPORAL_BUILD_ID", default_build_id)
        logger.info(f"🏗️  Worker build ID: {build_id}")

        # Create worker
        worker = Worker(
            client,
            task_queue=task_queue,
            build_id=build_id,
            workflows=[
                ProcessBatchMemoryWorkflow,
                DocumentProcessingWorkflow,
                ProcessBatchMemoryFromPostWorkflow,
                ProcessBatchMemoryFromRequestWorkflow,
            ],
            activities=[
                # Memory processing activities
                memory_activities.add_memory_quick,
                memory_activities.batch_add_memory_quick,  # ✅ NEW: Batch version
                memory_activities.index_and_enrich_memory,
                memory_activities.update_relationships,
                memory_activities.process_memory_batch,
                memory_activities.send_webhook_notification,
                memory_activities.process_batch_memories_from_parse_reference,
                memory_activities.fetch_batch_memories_from_post,
                memory_activities.fetch_and_process_batch_request,
                # New fine-grained indexing activities
                memory_activities.idx_index_grouped_memory,
                memory_activities.idx_generate_graph_schema,
                memory_activities.idx_update_metrics,
                memory_activities.link_batch_memories_to_post,
                # Document processing activities
                document_activities.validate_document,
                document_activities.process_document_with_provider,
                document_activities.store_in_memory_batch,
                document_activities.store_in_parse_only,
                document_activities.store_in_memory_and_parse,
                document_activities.send_status_update,
                document_activities.send_webhook_notification,
                document_activities.cleanup_failed_processing,
                # New file reference-based activities for large documents
                document_activities.download_and_validate_file,
                document_activities.process_document_with_provider_from_reference,
                document_activities.create_memory_batch_for_pages,
                document_activities.store_document_in_parse,
                # New hierarchical chunking activities
                document_activities.process_document_with_hierarchical_chunking,
                document_activities.extract_structured_content_from_provider,
                document_activities.create_hierarchical_memory_batch,
                document_activities.chunk_document_elements,
                document_activities.analyze_document_structure,
                document_activities.generate_llm_optimized_memory_structures,
                document_activities.fetch_llm_result_from_post,
                document_activities.store_batch_memories_in_parse_for_processing,
                document_activities.update_post_pipeline_start,
            ],
        )

        logger.info(f"Starting Temporal worker on task queue: {task_queue}")
        logger.info("Worker will process batch memory and document workflows")

        # Run the worker
        await worker.run()

    except KeyboardInterrupt:
        logger.info("Temporal worker interrupted by user")
        # Don't raise - graceful shutdown
    except SystemExit:
        logger.info("Temporal worker shutting down")
        # Don't raise - graceful shutdown
    except Exception as e:
        # Log error but don't crash - allow worker to restart or handle gracefully
        logger.error(f"❌ Temporal worker error: {e}", exc_info=True)
        # In production, we might want to restart the worker instead of crashing
        # For now, we log the error and re-raise so the process manager can handle it
        raise


if __name__ == "__main__":
    asyncio.run(main())