#!/usr/bin/env python3
"""
Start both Temporal workers (memory and document) in the same process.
This is used when deploying workers separately from the web server.
"""

import asyncio
import os
import sys
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
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

# Memory worker imports
from cloud_plugins.temporal.workflows.batch_memory import (
    ProcessBatchMemoryWorkflow,
    ProcessBatchMemoryFromPostWorkflow,
    ProcessBatchMemoryFromRequestWorkflow,
)
from cloud_plugins.temporal.activities import memory_activities

# Document worker imports
from cloud_plugins.temporal.workflows.document_processing import DocumentProcessingWorkflow
from cloud_plugins.temporal.activities import document_activities

from services.logger_singleton import LoggerSingleton
from config import get_features

logger = LoggerSingleton.get_logger(__name__)

# Health check server for Cloud Run
workers_running = False
health_server_started = False

class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for Cloud Run health checks"""
    def do_GET(self):
        global workers_running, health_server_started
        
        # Cloud Run startup probe: return 200 as soon as server is listening
        # This allows Cloud Run to mark the container as started
        # The actual worker readiness is checked separately
        if self.path == '/health' or self.path == '/':
            # Always return 200 once server is started (for Cloud Run startup probe)
            # Cloud Run just needs to know the container is alive and listening
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            if workers_running:
                self.wfile.write(b'OK - Workers running')
            else:
                self.wfile.write(b'OK - Starting workers...')
        elif self.path == '/ready':
            # Readiness check: returns 200 only when workers are actually running
            if workers_running:
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Ready')
            else:
                self.send_response(503)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Not ready - Workers starting...')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def start_health_server():
    """Start a simple HTTP server for Cloud Run health checks in a background thread"""
    global health_server_started
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    
    def run_server():
        global health_server_started
        logger.info(f"✅ Health check server started on port {port}")
        health_server_started = True
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Wait a moment to ensure server is actually listening
    import time
    time.sleep(0.5)
    
    return server

async def main():
    """Start both memory and document workers"""
    global workers_running
    
    # CRITICAL: Start health server IMMEDIATELY for Cloud Run startup probe
    # This must happen before any async operations to ensure Cloud Run can connect
    logger.info("🚀 Starting health check server for Cloud Run...")
    health_server = start_health_server()
    logger.info("✅ Health check server started, Cloud Run can now probe port 8080")
    
    features = get_features()
    if not features.is_enabled("temporal"):
        logger.error("Temporal is not enabled in current edition")
        # Health server is running, so Cloud Run won't fail, but workers won't start
        return

    try:
        client = await get_temporal_client()
        logger.info("✅ Successfully connected to Temporal")

        # Memory worker configuration
        # Use -v2 queue to avoid conflicts with old worker registrations
        memory_task_queue = "memory-processing"
        
        # Document worker configuration
        document_task_queue = "document-processing-v2"  # Use v2 to avoid stuck workflows

        logger.info(f"🔧 Starting Memory Worker on task queue: {memory_task_queue}")
        logger.info(f"🔧 Starting Document Worker on task queue: {document_task_queue}")

        # Build ID format: v{semver}+{feature}.{timestamp}
        # - semver: Semantic version from pyproject.toml (0.1.0)
        # - feature: Short identifier for the main change (batch-default)
        # - timestamp: Deployment date for uniqueness
        # 
        # Examples:
        #   v0.1.0+batch-default.20251117  ← Current (batch processing by default)
        #   v0.2.0+graph-memory.20251201   ← Hypothetical future version
        #
        # Benefits:
        # - Clear semantic version for compatibility tracking
        # - Feature tag explains what changed
        # - Timestamp for deployment tracking
        # - Can be overridden via TEMPORAL_BUILD_ID env var for custom builds
        #
        # This follows Temporal's best practice of using build IDs that:
        # 1. Are unique per deployment
        # 2. Are human-readable
        # 3. Sort chronologically
        # 4. Indicate compatibility (via semver major.minor)
        #
        # IMPORTANT: Worker Versioning requires BOTH:
        # 1. Workers registering with build_id (done here)
        # 2. Task queue configured with default build ID in Temporal Cloud, OR
        #    workflows started with versioning_intent to route to versioned workers
        #
        # To configure default build ID in Temporal Cloud:
        #   temporal task-queue update-build-ids add-new-default \
        #     --task-queue memory-processing-v2 \
        #     --build-id "v0.2.2+batch-default.20251117"
        #
        # Note: The old build_id API will be deprecated March 2026 in favor of
        # Worker Deployments. See: https://docs.temporal.io/worker-versioning
        #
        # Set TEMPORAL_USE_VERSIONING=false to run unversioned (for local dev/testing
        # if task queue default build ID is not configured in Temporal Cloud).
        # Default is true to match current production behavior.
        use_versioning = os.getenv("TEMPORAL_USE_VERSIONING", "true").lower() == "true"
        
        feature_id = "batch-default"
        timestamp = "20251117"  # Date of batch-default release
        default_build_id = f"v{APP_VERSION}+{feature_id}.{timestamp}"
        build_id = os.getenv("TEMPORAL_BUILD_ID", default_build_id) if use_versioning else None
        
        if build_id:
            logger.info(f"🏗️  Worker build ID: {build_id} (versioned mode)")
            logger.info(f"⚠️  Ensure task queue has default build ID configured in Temporal Cloud!")
        else:
            logger.info(f"🏗️  Worker version: {APP_VERSION} (unversioned mode)")

        # Create both workers
        # build_id is only passed if TEMPORAL_USE_VERSIONING=true
        worker_kwargs = {"client": client, "task_queue": memory_task_queue}
        if build_id:
            worker_kwargs["build_id"] = build_id
        
        memory_worker = Worker(
            **worker_kwargs,
            workflows=[
                ProcessBatchMemoryWorkflow,
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
                # Fine-grained indexing activities
                memory_activities.idx_index_grouped_memory,
                memory_activities.idx_generate_graph_schema,
                memory_activities.idx_update_metrics,
                memory_activities.link_batch_memories_to_post,
            ],
        )

        doc_worker_kwargs = {"client": client, "task_queue": document_task_queue}
        if build_id:
            doc_worker_kwargs["build_id"] = build_id
        
        document_worker = Worker(
            **doc_worker_kwargs,
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

        logger.info("✅ Both workers configured successfully")
        logger.info("📊 Memory Worker:")
        logger.info(f"   - Workflows: {len([ProcessBatchMemoryWorkflow, ProcessBatchMemoryFromPostWorkflow, ProcessBatchMemoryFromRequestWorkflow])}")
        logger.info(f"   - Activities: 14 (including link_batch_memories_to_post)")
        logger.info(f"   - Task Queue: {memory_task_queue}")
        logger.info("")
        logger.info("📄 Document Worker:")
        logger.info(f"   - Workflows: 1 (DocumentProcessingWorkflow)")
        logger.info(f"   - Activities: 21 (document activities only)")
        logger.info(f"   - Task Queue: {document_task_queue}")
        logger.info("")
        logger.info("🚀 Starting both workers... (Press CTRL+C to quit)")

        # Mark workers as running (health server already started above)
        workers_running = True

        # Run both workers concurrently with error handling
        try:
            await asyncio.gather(
                memory_worker.run(),
                document_worker.run()
            )
        except KeyboardInterrupt:
            logger.info("Workers interrupted by user")
            # Don't raise - graceful shutdown
        except SystemExit:
            logger.info("Workers shutting down")
            # Don't raise - graceful shutdown
        except Exception as e:
            # Log error but don't crash - allow workers to restart or handle gracefully
            logger.error(f"❌ Worker runtime error: {e}", exc_info=True)
            # Re-raise so the process manager can handle it
            raise
        finally:
            # Cleanup health server
            # Note: workers_running is already declared as global at top of main()
            workers_running = False
            if 'health_server' in locals():
                health_server.shutdown()

    except Exception as e:
        logger.error(f"❌ Failed to start workers: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())

