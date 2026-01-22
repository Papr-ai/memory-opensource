#!/usr/bin/env python3
"""
Temporal Worker for Document Processing

Registers the document processing workflow and related activities
on the 'document-processing' task queue.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Force protobuf pure-Python implementation
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Load environment variables (conditionally based on USE_DOTENV)
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    _ENV_FILE = find_dotenv()
    if _ENV_FILE:
        load_dotenv(_ENV_FILE)

sys.path.insert(0, str(Path(__file__).parent))

from temporalio.worker import Worker
from version import __version__ as APP_VERSION
from cloud_plugins.temporal.client import get_temporal_client
from cloud_plugins.temporal.workflows.document_processing import DocumentProcessingWorkflow
from cloud_plugins.temporal.activities import document_activities, memory_activities
from services.logger_singleton import LoggerSingleton
from config import get_features

logger = LoggerSingleton.get_logger(__name__)


async def main():
    features = get_features()
    if not features.is_enabled("temporal"):
        logger.error("Temporal is not enabled in current edition")
        return

    try:
        # CRITICAL: Initialize MongoDB client BEFORE any MemoryGraph instances are created
        # This ensures the shared MongoDB singleton is available when workers create MemoryGraph
        from services.mongo_client import get_mongo_db
        logger.info("Pre-initializing MongoDB client singleton for Document worker...")
        shared_db = get_mongo_db()
        if shared_db is not None:
            logger.info(f"✅ MongoDB client initialized successfully: {shared_db.name}")
        else:
            logger.error("❌ MongoDB client initialization returned None - workers will fallback to Parse Server!")
        
        client = await get_temporal_client()
        logger.info("Successfully connected to Temporal")

        task_queue = "document-processing-v2"

        # Build ID format: v{semver}+{feature}.{timestamp}
        # See start_all_workers.py for full explanation
        feature_id = "batch-default"
        timestamp = "20251117"  # Date of batch-default release
        default_build_id = f"v{APP_VERSION}+{feature_id}.{timestamp}"
        
        build_id = os.getenv("TEMPORAL_BUILD_ID", default_build_id)
        logger.info(f"🏗️  Worker build ID: {build_id}")

        worker = Worker(
            client,
            task_queue=task_queue,
            build_id=build_id,
            workflows=[DocumentProcessingWorkflow],
            activities=[
                # Document processing activities only - NO memory activities
                # (memory activities are handled by memory_worker on memory-processing queue)
                document_activities.download_and_validate_file,
                document_activities.process_document_with_provider_from_reference,
                document_activities.process_document_with_hierarchical_chunking,
                document_activities.extract_structured_content_from_provider,
                document_activities.chunk_document_elements,
                document_activities.generate_llm_optimized_memory_structures,
                document_activities.fetch_llm_result_from_post,
                document_activities.download_and_reupload_provider_images,
                document_activities.extract_and_upload_images_from_pdf,
                document_activities.extract_and_upload_images_from_pdf_stored,
                document_activities.create_hierarchical_memory_batch,
                document_activities.create_memory_batch_for_pages,
                document_activities.store_document_in_parse,
                document_activities.store_batch_memories_in_parse_for_processing,
                document_activities.send_webhook_notification,
                document_activities.send_status_update,
                document_activities.cleanup_failed_processing,
                document_activities.update_post_pipeline_start,
                document_activities.update_post_llm_extraction,
                document_activities.update_post_indexing_results,
            ],
        )

        logger.info(f"Starting Document Temporal worker on task queue: {task_queue}")
        await worker.run()

    except KeyboardInterrupt:
        logger.info("Document worker interrupted by user")
        # Don't raise - graceful shutdown
    except SystemExit:
        logger.info("Document worker shutting down")
        # Don't raise - graceful shutdown
    except Exception as e:
        # Log error but don't crash - allow worker to restart or handle gracefully
        logger.error(f"❌ Document worker error: {e}", exc_info=True)
        # In production, we might want to restart the worker instead of crashing
        # For now, we log the error and re-raise so the process manager can handle it
        raise


if __name__ == "__main__":
    asyncio.run(main())


