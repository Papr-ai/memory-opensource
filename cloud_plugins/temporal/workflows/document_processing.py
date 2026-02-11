"""
Temporal workflow for durable document processing
"""

from temporalio import workflow, activity
from temporalio.common import RetryPolicy
from datetime import timedelta
from typing import Dict, Any, Optional, List
from models.shared_types import MemoryMetadata, PreferredProvider
from models.temporal_models import SchemaSpecification
import asyncio
import uuid
from services.logger_singleton import LoggerSingleton


# Don't import activities directly to avoid sandbox restrictions
# Use string names instead


@workflow.defn
class DocumentProcessingWorkflow:
    """Temporal workflow for durable document processing"""

    def __init__(self):
        self.upload_id = None
        self.organization_id = None
        self.namespace_id = None
        self.status_updates = []
        # Timing helpers
        self._workflow_start_ts = None
        self._last_step_ts = None

    @workflow.run
    async def run(
        self,
        upload_id: str,
        organization_id: Optional[str],
        namespace_id: Optional[str],
        file_reference: Dict[str, Any],  # File reference instead of content
        user_id: str,
        workspace_id: Optional[str],
        metadata: MemoryMetadata,
        preferred_provider: Optional[PreferredProvider] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        hierarchical_enabled: bool = True,
        schema_specification: Optional[SchemaSpecification] = None  # Schema specification for graph extraction
    ) -> Dict[str, Any]:
        """Main workflow execution using file references for large file support"""

        self.upload_id = upload_id
        self.organization_id = organization_id
        self.namespace_id = namespace_id
        # Initialize timing at workflow start
        self._workflow_start_ts = workflow.now()
        self._last_step_ts = self._workflow_start_ts
        
        # Extract schema specification data
        schema_id = schema_specification.schema_id if schema_specification else None
        graph_override = schema_specification.graph_override if schema_specification else None
        property_overrides = schema_specification.property_overrides if schema_specification else None

        try:
            # Step 1: Download and validate file from Parse Server storage
            validation_result = await workflow.execute_activity(
                "download_and_validate_file",
                args=[file_reference, organization_id, namespace_id],
                start_to_close_timeout=timedelta(minutes=10),  # Increased for large files
                retry_policy=RetryPolicy(maximum_attempts=3)
            )

            if not validation_result["valid"]:
                await self._update_status("failed", 0.0, error=validation_result["error"])
                await self._cleanup_on_failure(validation_result["error"])
                return {"status": "failed", "error": validation_result["error"]}

            await self._update_status("processing", 0.1)

            # Step 2: Process document with provider (handles large files)
            workflow.logger.info(f"Processing document with provider: {preferred_provider}")

            # PIPELINE TRACKING: Mark pipeline start time
            # This will be set on the Post created by process_document_with_provider_from_reference
            pipeline_start_time = workflow.now()

            # CRITICAL: This activity creates the Post in Parse Server
            # It now includes validation to ensure the Post actually exists after creation
            # This prevents workflow replay issues where old Post IDs are reused
            processing_result = await workflow.execute_activity(
                "process_document_with_provider_from_reference",
                args=[
                    file_reference,
                    upload_id,
                    organization_id,
                    namespace_id,
                    workspace_id,
                    preferred_provider,
                    user_id
                ],
                start_to_close_timeout=timedelta(hours=2),  # Much longer for 1000-page documents
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    backoff_coefficient=2.0,
                    initial_interval=timedelta(minutes=1)
                ),
                heartbeat_timeout=timedelta(minutes=10)  # Regular heartbeats for long processing
            )

            # CRITICAL: Validate Post was created
            post_id_ref = (processing_result.get("post") or {}).get("objectId")
            if not post_id_ref:
                error_msg = f"Document processing completed but no Post ID was returned for upload_id {upload_id}"
                workflow.logger.error(error_msg)
                raise Exception(error_msg)
            
            workflow.logger.info(f"✅ Document processed successfully with Post ID: {post_id_ref}")

            # Set pipeline_start on the Post (with idempotency check inside activity)
            if post_id_ref:
                try:
                    await workflow.execute_activity(
                        "update_post_pipeline_start",
                        args=[post_id_ref, pipeline_start_time.isoformat()],
                        start_to_close_timeout=timedelta(seconds=30),
                        retry_policy=RetryPolicy(maximum_attempts=2)
                    )
                except Exception as e:
                    # Log but don't fail workflow - pipeline_start is metadata only
                    workflow.logger.warning(f"⚠️ Failed to set pipeline_start (non-fatal): {e}")

            # Decide memory creation strategy: hierarchical LLM chunking vs per-page
            memory_items: List[Dict[str, Any]] = []
            total_pages = processing_result.get("stats", {}).get("total_pages", 0)
            # Get the actual provider that was used (preferred_provider=None defaults to Reducto)
            actual_provider = processing_result.get("stats", {}).get("provider", "").lower()
            storage_result: Dict[str, Any] = {}

            if hierarchical_enabled:
                # Hierarchical pipeline: extract structured content → LLM generation → batch → create Post
                workflow.logger.info(f"Creating memories for hierarchical pipeline with {total_pages} pages")
                await self._update_status("analyzing_structure", 0.65, None, total_pages)

                # First extract structured elements by fetching provider JSON via post_id
                post_id_ref = (processing_result.get("post") or {}).get("objectId")
                workflow.logger.info(f"Extracting structured content from provider with post_id: {post_id_ref}")
                extraction = await workflow.execute_activity(
                    "extract_structured_content_from_provider",
                    args=[
                        {"post": {"objectId": post_id_ref}},
                        processing_result.get("stats", {}).get("provider", "unknown"),
                        metadata,
                        organization_id,
                        namespace_id
                    ],
                    start_to_close_timeout=timedelta(minutes=10),
                    retry_policy=RetryPolicy(maximum_attempts=3)
                )

                # Step 2.4: Re-upload temporary provider images BEFORE chunking (so chunked extraction has permanent URLs)
                # Providers like Reducto, TensorLake may provide temporary signed URLs that expire after 1 hour
                # If extraction was stored (has_temp_images=True), re-upload images to Parse for permanent URLs
                extraction_post_id = extraction.get("post_id", post_id_ref)
                extraction_stored = extraction.get("extraction_stored", False)
                
                # Only re-upload if extraction was stored (which happens when has_temp_images=True)
                if extraction_stored and extraction_post_id:
                    workflow.logger.info(f"Extraction stored in Parse - checking for temporary image URLs to re-upload (provider: {actual_provider})")
                    try:
                        reupload_result = await workflow.execute_activity(
                            "download_and_reupload_provider_images",
                            args=[extraction_post_id, actual_provider or "unknown"],
                            start_to_close_timeout=timedelta(minutes=15),
                            retry_policy=RetryPolicy(maximum_attempts=2)
                        )
                        workflow.logger.info(f"✅ Re-uploaded {reupload_result.get('images_reuploaded', 0)} temporary images to Parse (before chunking)")
                    except Exception as reupload_err:
                        workflow.logger.warning(f"Image re-upload failed (non-fatal): {reupload_err}")

                # Step 2.5: Apply hierarchical semantic chunking to group elements optimally
                # This creates 1-2 page chunks instead of many small fragments
                # Note: Chunking now works with permanent Parse image URLs (if Reducto was used)
                
                extraction_stored = extraction.get("extraction_stored", False)
                element_count = len(extraction.get("structured_elements", []))

                if extraction_stored:
                    workflow.logger.info(f"Hierarchical chunking enabled - fetching elements from Parse Post {extraction_post_id}")
                else:
                    workflow.logger.info(f"Hierarchical chunking enabled - chunking {element_count} inline elements")

                chunking_result = await workflow.execute_activity(
                    "chunk_document_elements",
                    args=[
                        extraction.get("structured_elements", []),  # Empty if extraction_stored=True
                        None,  # Use default chunking config
                        extraction_post_id,  # post_id for fetching if needed
                        extraction_stored  # flag to fetch from Parse
                    ],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=RetryPolicy(maximum_attempts=2)
                )

                # Update extraction with chunked elements
                # If extraction was stored, the chunking activity will have re-stored the chunked version
                if not extraction_stored:
                    extraction["structured_elements"] = chunking_result.get("chunked_elements", [])
                else:
                    # Chunked elements are stored back in Parse under the same post_id
                    # The LLM activity will fetch them
                    extraction["extraction_stored"] = True  # Still stored, but now chunked

                chunking_stats = chunking_result.get("stats", {})
                workflow.logger.info(f"Chunking complete: {chunking_stats.get('chunked_count')} chunks from {chunking_stats.get('original_count')} elements ({chunking_stats.get('reduction_percent', 0):.1f}% reduction)")
            

                decision = extraction.get("decision", "complex")
                workflow.logger.info(f"Document processing path: {decision}")

                # Send status update after extraction (image extraction already done in Step 2.5)
                await self._update_status("creating_memories", 0.75, None, total_pages, page_id=post_id_ref)

                # Branch: simple path skips LLM, complex path uses LLM generation
                if decision == "simple":
                    # Simple: use memory_requests directly from extraction (already chunked Markdown)
                    memory_requests = extraction.get("memory_requests", [])
                    workflow.logger.info(f"Simple path: using {len(memory_requests)} memory requests from Markdown extraction")
                else:
                    # Complex: generate LLM-optimized memory structures from structured elements
                    # Check if extraction was stored in Parse (for large documents)
                    extraction_stored = extraction.get("extraction_stored", False)
                    extraction_post_id = extraction.get("post_id", post_id_ref)
                    
                    workflow.logger.info(f"Complex path: extraction_stored={extraction_stored}, will {'fetch from Parse' if extraction_stored else 'use inline elements'}")
                    
                    # Step 2.6: Extract and upload images from PDF (for providers that don't provide image URLs)
                    # This extracts actual image bytes from the PDF and uploads them to Azure Storage
                    pdf_url = file_reference.get("file_url") if file_reference else None
                    provider_extracted_images = actual_provider == 'reducto'  # Reducto images already handled above
                    
                    # Check if we should extract images from PDF (PDF files only, and provider hasn't already done it)
                    if pdf_url and pdf_url.lower().endswith('.pdf') and not provider_extracted_images:
                        workflow.logger.info(f"PDF file detected, checking if image extraction is needed")
                        
                        # For stored extractions, we need to trigger image extraction via a separate activity
                        # that will fetch elements from Parse, extract images, and update them back
                        if extraction_stored:
                            workflow.logger.info(f"Extraction stored in Parse, calling image extraction with Post ID {extraction_post_id}")
                            try:
                                # Call activity that will fetch elements from Parse, extract images, and update
                                image_extraction = await workflow.execute_activity(
                                    "extract_and_upload_images_from_pdf_stored",
                                    args=[
                                        pdf_url,
                                        extraction_post_id,
                                        organization_id,
                                        namespace_id,
                                        workspace_id
                                    ],
                                    start_to_close_timeout=timedelta(minutes=15),
                                    retry_policy=RetryPolicy(maximum_attempts=2)
                                )
                                workflow.logger.info(f"Image extraction complete: {image_extraction.get('images_extracted', 0)} images uploaded")
                            except Exception as img_err:
                                workflow.logger.warning(f"Image extraction failed (non-fatal): {img_err}")
                        else:
                            # For inline extractions, pass structured_elements directly
                            structured_elements = extraction.get("structured_elements", [])
                            if structured_elements:
                                workflow.logger.info(f"Inline extraction with {len(structured_elements)} elements, extracting images from {pdf_url}")
                                try:
                                    image_extraction = await workflow.execute_activity(
                                        "extract_and_upload_images_from_pdf",
                                        args=[
                                            pdf_url,
                                            structured_elements,
                                            organization_id,
                                            namespace_id,
                                            workspace_id,
                                            extraction_post_id
                                        ],
                                        start_to_close_timeout=timedelta(minutes=15),
                                        retry_policy=RetryPolicy(maximum_attempts=2)
                                    )
                                    workflow.logger.info(f"Image extraction complete: {image_extraction.get('images_extracted', 0)} images uploaded")
                                    
                                    # Update structured_elements with extracted image URLs
                                    if image_extraction.get('images_extracted', 0) > 0:
                                        # The activity updates elements in-place, so structured_elements is already updated
                                        extraction['structured_elements'] = structured_elements
                                        workflow.logger.info(f"Updated extraction with {image_extraction.get('images_extracted')} image URLs")
                                except Exception as img_err:
                                    workflow.logger.warning(f"Image extraction failed (non-fatal): {img_err}")
                            else:
                                workflow.logger.info("No structured elements found for inline extraction")
                    elif provider_extracted_images:
                        workflow.logger.info(f"Provider '{actual_provider}' already extracted and uploaded images (found in structured elements), skipping redundant extraction")
                    else:
                        workflow.logger.info(f"Skipping image extraction: pdf_url={pdf_url is not None}, is_pdf={pdf_url.lower().endswith('.pdf') if pdf_url else False}")
                    
                    llm_gen = await workflow.execute_activity(
                        "generate_llm_optimized_memory_structures",
                        args=[
                            extraction.get("structured_elements", []),  # Empty if extraction_stored=True
                            getattr(metadata, "domain", None),
                            metadata,
                            organization_id,
                            namespace_id,
                            True,  # use_llm
                            extraction_post_id,  # post_id for fetching if needed
                            extraction_stored  # flag to fetch from Parse
                        ],
                        start_to_close_timeout=timedelta(minutes=20),
                        retry_policy=RetryPolicy(maximum_attempts=3)
                    )
                    
                    # PIPELINE TRACKING: Update llm_enhanced_extraction in Post (Stage 3)
                    # Only pass minimal metadata to avoid Temporal payload limits
                    if post_id_ref and isinstance(llm_gen, dict):
                        try:
                            await workflow.execute_activity(
                                "update_post_llm_extraction",
                                args=[
                                    post_id_ref,
                                    llm_gen.get("llm_result_stored", False),
                                    llm_gen.get("total_generated", 0),
                                    llm_gen.get("llm_result_id"),
                                    llm_gen.get("generation_summary", {})
                                ],
                                start_to_close_timeout=timedelta(seconds=30),
                                retry_policy=RetryPolicy(maximum_attempts=2)
                            )
                        except Exception as e:
                            workflow.logger.warning(f"Failed to update llm_enhanced_extraction: {e}")
                    
                    # Extract memory requests from LLM generation result
                    # Check if LLM result was stored in Parse (for large documents)
                    llm_result_stored = llm_gen.get("llm_result_stored", False) if isinstance(llm_gen, dict) else False
                    
                    if llm_result_stored:
                        # Large result: fetch from Parse Server
                        llm_post_id = llm_gen.get("post_id", post_id_ref)
                        workflow.logger.info(f"LLM result stored in Parse, will fetch from Post {llm_post_id}")
                        
                        # Fetch LLM result from Parse
                        llm_fetch = await workflow.execute_activity(
                            "fetch_llm_result_from_post",
                            args=[llm_post_id],
                            start_to_close_timeout=timedelta(minutes=5),
                            retry_policy=RetryPolicy(maximum_attempts=3)
                        )
                        
                        memory_requests = llm_fetch.get("memory_requests", []) if isinstance(llm_fetch, dict) else []
                        workflow.logger.info(f"Fetched {len(memory_requests)} memory requests from Parse")
                    else:
                        # Small result: use inline memory_requests
                        batch_req = llm_gen.get("batch_request") if isinstance(llm_gen, dict) else None
                        memory_requests = []
                        if batch_req and isinstance(batch_req, dict):
                            memory_requests = batch_req.get("memories", []) or []
                        else:
                            memory_requests = (llm_gen.get("memory_requests", []) if isinstance(llm_gen, dict) else [])
                        workflow.logger.info(f"Complex path (LLM): generated {len(memory_requests)} memory requests")

                # Create memory batch if we have any memories (from either simple or complex path)
                if memory_requests:
                    workflow.logger.info(f"Processing {len(memory_requests)} memories with full indexing pipeline")
                    # Get document Post ID to reuse
                    document_post_id = (processing_result.get("post") or {}).get("objectId") or post_id_ref
                    workflow.logger.info(f"Using existing document Post ID: {document_post_id}")
                    
                    # Store memories in Parse to avoid Temporal payload limits
                    # Extract file URL for sourceUrl field
                    file_url = file_reference.get("file_url") if file_reference else None
                    workflow.logger.info(f"Storing {len(memory_requests)} memories in Parse Post (sourceUrl: {file_url})")
                    store_result = await workflow.execute_activity(
                        "store_batch_memories_in_parse_for_processing",
                        args=[
                            memory_requests,
                            organization_id,
                            namespace_id,
                            user_id,
                            workspace_id,
                            document_post_id,  # Reuse document Post
                            schema_specification.model_dump() if schema_specification else None,  # Pass schema specification as dict for indexing
                            file_url  # PDF/document URL for sourceUrl
                        ],
                        start_to_close_timeout=timedelta(minutes=5),
                        retry_policy=RetryPolicy(maximum_attempts=3)
                    )
                    
                    batch_post_id = store_result.get("post_id")
                    workflow.logger.info(f"Memories stored in Post {batch_post_id}, starting batch memory child workflow")
                    
                    # Include page_id in status update for user-facing tracking
                    await self._update_status("indexing_memories", 0.8, page_id=batch_post_id)
                    
                    # Start ProcessBatchMemoryFromPostWorkflow as child workflow
                    # This runs the full multi-stage indexing pipeline:
                    # Stage 1: add_memory_quick (parallel quick adds to Qdrant + Parse)
                    # Stage 2: idx_generate_graph_schema (parallel LLM schema generation with custom schema enforcement)
                    # Stage 3: update_relationships (parallel Neo4j relationship building)
                    # Stage 4: idx_update_metrics (parallel metrics updates)
                    from cloud_plugins.temporal.workflows.batch_memory import ProcessBatchMemoryFromPostWorkflow
                    
                    workflow.logger.info(f"Starting child workflow ProcessBatchMemoryFromPostWorkflow for Post {batch_post_id}")
                    
                    # Use versioned task queue to ensure we get the new worker
                    # This avoids conflicts with old workers still registered on the default queue
                    memory_task_queue = "memory-processing"  # v2 for batch-default version
                    
                    batch_workflow_handle = await workflow.start_child_workflow(
                        ProcessBatchMemoryFromPostWorkflow.run,
                        args=[{
                            "batch_id": f"doc-{upload_id}",
                            "post_id": batch_post_id,
                            "organization_id": organization_id,
                            "namespace_id": namespace_id,
                            "user_id": user_id,
                            "workspace_id": workspace_id,
                            "schema_specification": schema_specification.model_dump() if schema_specification else None  # Pass schema specification as dict for custom schema enforcement
                        }],
                        id=f"batch-memory-doc-{upload_id}",
                        task_queue=memory_task_queue
                    )
                    
                    # Wait for child workflow to complete
                    workflow.logger.info(f"Waiting for batch memory workflow to complete...")
                    indexing_result = await batch_workflow_handle
                    
                    workflow.logger.info(
                        f"Batch indexing complete: {indexing_result.get('successful', 0)}/{indexing_result.get('total_processed', 0)} successful"
                    )
                    
                    # PIPELINE TRACKING: Update indexing_results and pipeline_end in Post (Stage 5)
                    if post_id_ref and isinstance(indexing_result, dict):
                        try:
                            pipeline_end_time = workflow.now()
                            await workflow.execute_activity(
                                "update_post_indexing_results",
                                args=[
                                    post_id_ref,
                                    indexing_result,
                                    pipeline_start_time.isoformat(),
                                    pipeline_end_time.isoformat()
                                ],
                                start_to_close_timeout=timedelta(seconds=30),
                                retry_policy=RetryPolicy(maximum_attempts=2)
                            )
                        except Exception as e:
                            workflow.logger.warning(f"Failed to update indexing_results: {e}")

                    await self._update_status("storing_document", 0.9, page_id=document_post_id)
                    storage_result = {"parse_records": {"post_id": document_post_id}}
                    
                    # Note: Linking memories to Post is now handled inside ProcessBatchMemoryFromPostWorkflow

                else:
                    # Fallback: skip page-based path when only a post reference exists
                    fallback_post_id = (processing_result.get("post") or {}).get("objectId") or post_id_ref
                    await self._update_status("storing_document", 0.95, page_id=fallback_post_id)
                    storage_result = {"parse_records": {"post_id": fallback_post_id}}
                    # Note: Linking memories to Post is now handled inside ProcessBatchMemoryFromPostWorkflow
                    
            else:
                # Non-hierarchical path: skip page batching when only post_id exists
                non_hier_post_id = (processing_result.get("post") or {}).get("objectId") or post_id_ref
                await self._update_status("storing_document", 0.9, page_id=non_hier_post_id)
                storage_result = {"parse_records": {"post_id": non_hier_post_id}}
                # Note: Linking memories to Post is now handled inside ProcessBatchMemoryFromPostWorkflow

            # Use post_id from storage_result or fall back to post_id_ref
            final_post_id = storage_result.get("parse_records", {}).get("post_id") if storage_result else post_id_ref
            await self._update_status("completed", 1.0, page_id=final_post_id)

            # Step 5: Send webhook notification if configured
            if webhook_url:
                await workflow.execute_activity(
                    "send_document_webhook_notification",
                    args=[{
                        "upload_id": upload_id,
                        "status": "completed",
                        "webhook_url": webhook_url,
                        "webhook_secret": webhook_secret,
                        "results": {
                            "memory_items": memory_items,
                            "parse_records": storage_result.get("parse_records", {}),
                            "processing_stats": processing_result.get("stats", {}),
                            "total_pages": total_pages
                        }
                    }],
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=RetryPolicy(maximum_attempts=3)
                )

            return {
                "status": "completed",
                "upload_id": upload_id,
                "memory_items": memory_items,
                "parse_records": storage_result.get("parse_records", {}),
                "processing_stats": processing_result.get("stats", {}),
                "total_pages": total_pages,
                "total_memory_items": len(memory_items)
            }

        except Exception as e:
            await self._update_status("failed", 0.0, error=str(e))
            await self._cleanup_on_failure(str(e))

            # Send failure webhook if configured
            if webhook_url:
                try:
                    await workflow.execute_activity(
                        "send_document_webhook_notification",
                        args=[{
                            "upload_id": upload_id,
                            "status": "failed",
                            "webhook_url": webhook_url,
                            "webhook_secret": webhook_secret,
                            "error": str(e)
                        }],
                        start_to_close_timeout=timedelta(minutes=2),
                        retry_policy=RetryPolicy(maximum_attempts=1)
                    )
                except:
                    pass  # Don't fail workflow if webhook fails

            raise

    async def _update_status(
        self,
        status: str,
        progress: float,
        current_page: Optional[int] = None,
        total_pages: Optional[int] = None,
        error: Optional[str] = None,
        page_id: Optional[str] = None  # User-facing Post ID (called page_id for consistency with user-facing API)
    ):
        """Update processing status"""
        now_ts = workflow.now()
        # Compute simple timings
        total_elapsed_ms = None
        step_elapsed_ms = None
        try:
            if self._workflow_start_ts:
                total_elapsed_ms = int((now_ts - self._workflow_start_ts).total_seconds() * 1000)
            if self._last_step_ts:
                step_elapsed_ms = int((now_ts - self._last_step_ts).total_seconds() * 1000)
        except Exception:
            pass

        update = {
            "upload_id": self.upload_id,
            "status": status,
            "progress": progress,
            # current_page intentionally omitted; page granularity is not meaningful here
            "total_pages": total_pages,
            "error": error,
            "timestamp": now_ts,
            "organization_id": self.organization_id,
            "namespace_id": self.namespace_id,
            # Timing metadata for observability
            "step_elapsed_ms": step_elapsed_ms,
            "total_elapsed_ms": total_elapsed_ms,
            # User-facing page ID (Post ID in Parse)
            "page_id": page_id if page_id else None,
        }

        self.status_updates.append(update)

        # Send status update via activity (non-blocking)
        try:
            await workflow.execute_activity(
                "send_status_update",
                args=[update],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=2)
            )
        except Exception as e:
            # Don't fail workflow if status update fails
            workflow.logger.warning(f"Failed to send status update: {e}")
        finally:
            # Advance step timer
            self._last_step_ts = now_ts

    async def _cleanup_on_failure(self, error: str):
        """Cleanup resources on failure"""
        try:
            await workflow.execute_activity(
                "cleanup_failed_processing",
                args=[self.upload_id, self.organization_id, self.namespace_id, error],
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=2)
            )
        except Exception as e:
            workflow.logger.warning(f"Cleanup failed: {e}")

    @workflow.signal
    async def cancel_processing(self, reason: str):
        """Signal to cancel processing"""
        await self._update_status("cancelled", 0.0, error=f"Cancelled: {reason}")
        await self._cleanup_on_failure(f"Cancelled: {reason}")
        workflow.continue_as_new(
            args=[],
            run_timeout=timedelta(seconds=1)
        )

    @workflow.query
    def get_status(self) -> Dict[str, Any]:
        """Query current processing status"""
        if self.status_updates:
            latest = self.status_updates[-1]
            return {
                "upload_id": self.upload_id,
                "status": latest["status"],
                "progress": latest["progress"],
                "current_page": latest.get("current_page"),
                "total_pages": latest.get("total_pages"),
                "error": latest.get("error"),
                "timestamp": latest["timestamp"],
                "total_updates": len(self.status_updates),
                "page_id": latest.get("page_id")  # User-facing Post ID
            }
        return {
            "upload_id": self.upload_id,
            "status": "unknown",
            "progress": 0.0
        }