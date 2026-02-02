#!/usr/bin/env python3
"""
Sequential V1 Endpoints Test Runner

This script runs all v1 endpoint tests sequentially to avoid parallel execution issues
and provides detailed logging and reporting.

Usage:
    python test_v1_endpoints_sequential.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app
from tests.test_add_memory_fastapi import (
    # Add Memory Tests
    test_v1_add_memory_1,
    test_v1_add_memory_with_api_key,
    test_v1_add_memory_with_external_user_id_and_custom_metadata,
    test_v1_add_memory_with_external_user_id_only,
    test_v1_add_memory_with_external_user_id_and_acl,
    test_v1_add_memory_with_user_id_from_created_user,
    test_v1_add_memory_with_org_namespace_top_level,
    test_v1_add_memory_with_deprecated_org_namespace,
    
    # Batch Add Memory Tests
    test_v1_add_memory_batch_1,
    test_v1_add_memory_batch_with_user_id,
    test_v1_add_memory_batch_with_external_user_id,
    test_v1_add_memory_batch_webhook_immediate_when_skip_background,
    test_v1_add_memory_batch_webhook_with_background_processing,
    
    # Webhook Tests
    test_v1_add_memory_batch_with_webhook_success,
    test_v1_add_memory_batch_with_webhook_partial_success,
    test_v1_add_memory_batch_with_webhook_no_url,
    test_v1_add_memory_batch_with_webhook_azure_fallback,
    test_v1_add_memory_batch_webhook_payload_structure,
    
    # Update Memory Tests
    test_v1_update_memory_1,
    test_v1_update_memory_with_api_key,
    test_v1_update_memory_acl_with_api_key_and_real_users,
    
    # Get Memory Tests
    test_v1_get_memory,
    
    # Search Memory Tests
    test_v1_search_1,
    test_v1_search_with_user_id_acl,
    test_v1_search_with_external_user_id_acl,
    test_v1_search_new_user_qwen_route,
    test_v1_search_with_custom_metadata_filter_qwen_only,
    test_v1_search_with_numeric_custom_metadata_filter,
    test_v1_search_with_list_custom_metadata_filter,
    test_v1_search_with_boolean_custom_metadata_filter,
    test_v1_search_with_mixed_custom_metadata_types,
    test_v1_search_with_organization_and_namespace_filter,
    test_search_v1_agentic_graph,
    test_v1_search_predicted_grouping_logging,
    test_e2e_developer_marking_apikey_sets_flag,
    test_e2e_developer_marking_bearer_does_not_set_flag,
    test_e2e_anon_user_not_marked_developer_when_dev_api_key_used,
    test_v1_search_fixed_user_cache_test,
    test_v1_search_performance_under_500ms_low_similarity,
    
    # Delete Memory Tests
    test_v1_delete_memory_1,
    test_v1_delete_memory_with_api_key,
    
    # Multi-tenant Tests
    test_multi_tenant_auth_models,
    test_memory_models_multi_tenant_fields,
    test_batch_memory_multi_tenant_scoping,
    test_backward_compatibility,
)
from tests.test_memory_policy_end_to_end import (
    TestLinkToDSLEndToEnd,
    TestFullMemoryPolicyEndToEnd,
    TestSchemaLevelPolicyInheritance,
    TestMemoryLevelPolicyOverride,
    TestPolicyMerging,
    TestControlledVocabulary,
    TestEdgeConstraintsEndToEnd,
    TestGraphQLValidation,
    TestErrorHandling,
    unique_id as memory_policy_unique_id,
    api_headers as memory_policy_api_headers,
)

# Import delete all memories tests
from tests.test_delete_all_memories import (
    test_delete_all_memories_complete_workflow,
    test_delete_all_memories_with_external_user_id,
    test_delete_all_memories_no_memories_found,
)

# Import user endpoint tests
from tests.test_user_v1_integration import (
    # Create User Tests
    test_create_user_v1_integration,
    test_create_anonymous_user_v1_integration,
    test_create_user_batch_v1_integration,
    
    # Get User Tests
    test_get_user_v1_integration,
    
    # Update User Tests
    test_update_user_v1_integration,
    
    # Delete User Tests
    test_delete_user_v1_integration,
    test_delete_user_by_external_id_integration,
    
    # List Users Tests
    test_list_users_v1_integration,
)

# Import feedback endpoint tests
from tests.test_feedback_end_to_end import (
    test_feedback_end_to_end,
    test_get_feedback_by_id_v1,
)
from tests.test_sync_v1 import (
    test_v1_sync_tiers_basic,
    test_v1_sync_tiers_with_embeddings_small_limits,
    test_v1_sync_delta_cursor_pagination,
)
from tests.test_query_log_integration import (
    test_memory_metadata_with_query_log_fields,
    test_classification_data_detection,
    test_search_with_query_log_integration,
    test_agentic_graph_log_model_creation,
    test_user_feedback_log_model_creation,
    test_agentic_graph_log_with_relations,
    test_user_feedback_log_minimal,
    test_agentic_graph_log_storage_function,
    test_user_feedback_log_storage_function,
    test_agentic_graph_log_data_preparation,
    test_user_feedback_log_data_preparation,
    test_memory_retrieval_log_predicted_grouping,
    test_query_log_persisted_with_classification,
    test_real_query_log_creation_and_memory_increment,
    test_cache_hits_increment_on_repeated_search,
    test_fused_confidence_matches_weight_delta,
    test_backfill_retrieval_counters_small_batch,
    test_sync_tiers_returns_ranked_tier1,
)

# Import security schema tests
from tests.test_security_schema_v1 import (
    test_v1_create_security_schema,
    test_v1_add_memory_with_schema_id,
    test_v1_wait_for_memory_processing,
    test_v1_search_verify_neo4j_nodes,
    test_v1_search_with_agentic_graph,
    test_v1_add_memory_with_graph_override,
    test_v1_security_schema_full_workflow,
)


# Import graph generation mode tests
from tests.test_graph_generation_modes import (
    test_auto_mode_with_schema_id,
    test_auto_mode_with_property_overrides,
    test_manual_mode_with_explicit_graph,
)

# Import document processing tests
from tests.test_document_processing_v2 import (
    test_document_upload_v2_with_api_key,
    test_document_upload_v2_with_session_token,
    test_document_upload_v2_with_real_pdf_file,
    test_document_upload_v2_with_real_pdf_file_custom_schema,
    test_document_upload_v2_invalid_file_type,
    test_document_upload_v2_malicious_content,
    test_document_upload_v2_large_file,
    test_document_upload_v2_with_webhook,
    test_document_upload_v2_multi_tenant,
    test_document_upload_v2_provider_preference,
    test_reducto_provider_direct_and_route,
    test_document_upload_v2_authentication_failure,
    test_document_status_endpoint,
    test_document_cancel_endpoint,
    test_reducto_provider_simple_upload_and_parse,
    test_document_upload_v2_with_gemini_provider,
    test_document_upload_v2_with_tensorlake_provider,
    # PaddleOCR and DeepSeek-OCR providers out of scope for v1 test run
    # test_document_upload_v2_with_paddleocr_provider,
    # test_document_upload_v2_with_deepseek_ocr_provider,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Wrapper for tests requiring caplog fixture
try:
    from contextlib import nullcontext
except ImportError:
    nullcontext = None  # Python <3.7 not supported here, but keep guard

class DummyCaplog:
    def __init__(self):
        self.records = []
    def at_level(self, level):
        return nullcontext() if nullcontext else None

async def test_v1_search_performance_under_500ms_low_similarity_wrapper(app_instance):
    """Wrapper to run low-similarity performance test without pytest caplog fixture."""
    dummy_caplog = DummyCaplog()
    await test_v1_search_performance_under_500ms_low_similarity(app_instance, dummy_caplog)

async def test_real_query_log_creation_and_memory_increment_wrapper(app_instance):
    """Wrapper to run QueryLog creation + memory increment test with AsyncClient."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_real_query_log_creation_and_memory_increment(async_client)

async def test_cache_hits_increment_on_repeated_search_wrapper(app_instance):
    """Wrapper to run repeated search cache-hit increment test with AsyncClient."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_cache_hits_increment_on_repeated_search(async_client)

async def test_fused_confidence_matches_weight_delta_wrapper(app_instance):
    """Wrapper to validate fused confidence against observed weighted delta with AsyncClient."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_fused_confidence_matches_weight_delta(async_client)

class V1EndpointTester:
    """Sequential test runner for v1 endpoints with detailed logging and reporting."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
        
    async def run_test(self, test_name: str, test_func, app_instance) -> Dict[str, Any]:
        """Run a single test and return results."""
        start_time = time.time()
        result = {
            "test_name": test_name,
            "status": "unknown",
            "duration": 0,
            "error": None,
            "details": None
        }
        
        try:
            logger.info(f"Starting test: {test_name}")
            # Call the test function with the app instance
            await test_func(app_instance)
            result["status"] = "passed"
            result["duration"] = time.time() - start_time
            logger.info(f"✅ Test passed: {test_name} (took {result['duration']:.2f}s)")
        except pytest.skip.Exception as e:  # Handle pytest skips explicitly
            result["status"] = "skipped"
            result["duration"] = time.time() - start_time
            result["details"] = str(e)
            logger.info(f"⏭️ Test skipped: {test_name} - {str(e)}")
        except BaseException as e:
            # Handle pytest failures and regular exceptions
            # pytest.fail() raises pytest.outcomes.Failed (BaseException), so catch BaseException
            # but allow KeyboardInterrupt/SystemExit to halt the run.
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            result["status"] = "failed"
            result["duration"] = time.time() - start_time
            result["error"] = str(e)
            
            # Log with traceback for better debugging, but don't crash
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"❌ Test failed: {test_name} - {str(e)}")
            logger.debug(f"Test failure traceback:\n{error_trace}")
            
        return result
    
    async def run_add_memory_tests(self, app_instance):
        """Run all add memory related tests."""
        logger.info("🧪 Running Add Memory Tests...")
        
        add_memory_tests = [
            ("Add Memory - Basic", test_v1_add_memory_1),
            ("Add Memory - API Key", test_v1_add_memory_with_api_key),
            ("Add Memory - External User ID + Custom Metadata", test_v1_add_memory_with_external_user_id_and_custom_metadata),
            ("Add Memory - External User ID Only", test_v1_add_memory_with_external_user_id_only),
            ("Add Memory - External User ID + ACL", test_v1_add_memory_with_external_user_id_and_acl),
            ("Add Memory - User ID from Created User", test_v1_add_memory_with_user_id_from_created_user),
            ("Add Memory - Top-Level Org/Namespace", test_v1_add_memory_with_org_namespace_top_level),
            ("Add Memory - Deprecated Org/Namespace", test_v1_add_memory_with_deprecated_org_namespace),
        ]
        
        for test_name, test_func in add_memory_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_batch_add_memory_tests(self, app_instance):
        """Run all batch add memory related tests."""
        logger.info("🧪 Running Batch Add Memory Tests...")
        
        batch_tests = [
            ("Batch Add Memory - Basic", test_v1_add_memory_batch_1),
            ("Batch Add Memory - User ID", test_v1_add_memory_batch_with_user_id),
            ("Batch Add Memory - External User ID", test_v1_add_memory_batch_with_external_user_id),
            
            # Webhook Tests
            ("Batch Add Memory - Webhook Success", test_v1_add_memory_batch_with_webhook_success),
            ("Batch Add Memory - Webhook Partial Success", test_v1_add_memory_batch_with_webhook_partial_success),
            ("Batch Add Memory - Webhook No URL", test_v1_add_memory_batch_with_webhook_no_url),
            ("Batch Add Memory - Webhook Azure Fallback", test_v1_add_memory_batch_with_webhook_azure_fallback),
            ("Batch Add Memory - Webhook Payload Structure", test_v1_add_memory_batch_webhook_payload_structure),
            ("Batch Add Memory - Webhook Immediate (skip_background=True)", test_v1_add_memory_batch_webhook_immediate_when_skip_background),
            ("Batch Add Memory - Webhook With Background Processing", test_v1_add_memory_batch_webhook_with_background_processing),
        ]
        
        for test_name, test_func in batch_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_update_memory_tests(self, app_instance):
        """Run all update memory related tests."""
        logger.info("🧪 Running Update Memory Tests...")
        
        update_tests = [
            ("Update Memory - Basic", test_v1_update_memory_1),
            ("Update Memory - API Key", test_v1_update_memory_with_api_key),
            ("Update Memory - ACL with Real Users", test_v1_update_memory_acl_with_api_key_and_real_users),
        ]
        
        for test_name, test_func in update_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_get_memory_tests(self, app_instance):
        """Run all get memory related tests."""
        logger.info("🧪 Running Get Memory Tests...")
        
        get_tests = [
            ("Get Memory - Basic", test_v1_get_memory),
        ]
        
        for test_name, test_func in get_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_search_memory_tests(self, app_instance):
        """Run all search memory related tests."""
        logger.info("🧪 Running Search Memory Tests...")
        
        search_tests = [
            ("Search Memory - Basic", test_v1_search_1),
            ("Search Memory - User ID ACL", test_v1_search_with_user_id_acl),
            ("Search Memory - External User ID ACL", test_v1_search_with_external_user_id_acl),
            ("Search Memory - New User Qwen Route", test_v1_search_new_user_qwen_route),
            ("Search Memory - Custom Metadata Filter (Qwen Only)", test_v1_search_with_custom_metadata_filter_qwen_only),
            ("Search Memory - Numeric Custom Metadata Filter", test_v1_search_with_numeric_custom_metadata_filter),
            ("Search Memory - List Custom Metadata Filter", test_v1_search_with_list_custom_metadata_filter),
            ("Search Memory - Boolean Custom Metadata Filter", test_v1_search_with_boolean_custom_metadata_filter),
            ("Search Memory - Mixed Custom Metadata Types", test_v1_search_with_mixed_custom_metadata_types),
            ("Search Memory - Organization and Namespace Filter", test_v1_search_with_organization_and_namespace_filter),
            ("Search Memory - Agentic Graph", test_search_v1_agentic_graph_wrapper),
            ("Search Memory - Predicted Grouping Logging", test_v1_search_predicted_grouping_logging),
            ("Search Memory - Fixed User Cache Test", test_v1_search_fixed_user_cache_test_wrapper),
            ("Search Memory - Low-similarity Performance", test_v1_search_performance_under_500ms_low_similarity_wrapper),
            ("Auth/Dev Marking - APIKey Sets Developer", test_e2e_developer_marking_apikey_sets_flag),
            ("Auth/Dev Marking - Bearer Only Does Not Set Developer", test_e2e_developer_marking_bearer_does_not_set_flag),
            ("Auth/Dev Marking - Anon End-User Not Marked Developer When Dev API Key Used", test_e2e_anon_user_not_marked_developer_when_dev_api_key_used),
        ]
        
        for test_name, test_func in search_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)

    async def run_memory_policy_tests(self, app_instance):
        """Run memory policy end-to-end tests."""
        logger.info("🧪 Running Memory Policy End-to-End Tests...")

        memory_policy_tests = [
            ("Memory Policy - link_to String", test_memory_policy_link_to_string_form_wrapper),
            ("Memory Policy - link_to List", test_memory_policy_link_to_list_form_wrapper),
            ("Memory Policy - link_to Dict create=never", test_memory_policy_link_to_dict_form_wrapper),
            ("Memory Policy - link_to Exact Match", test_memory_policy_link_to_exact_match_wrapper),
            ("Memory Policy - link_to Semantic Threshold", test_memory_policy_link_to_semantic_threshold_wrapper),
            ("Memory Policy - Auto Mode", test_memory_policy_auto_mode_wrapper),
            ("Memory Policy - Manual Mode", test_memory_policy_manual_mode_wrapper),
            ("Memory Policy - OMO Safety", test_memory_policy_omo_safety_wrapper),
            ("Memory Policy - Schema Inheritance", test_memory_policy_schema_inheritance_wrapper),
            ("Memory Policy - Override Schema", test_memory_policy_override_schema_wrapper),
            ("Memory Policy - link_to + policy", test_memory_policy_link_to_with_policy_wrapper),
            ("Memory Policy - link_to merge constraints", test_memory_policy_link_to_merge_constraints_wrapper),
            ("Memory Policy - create never blocks", test_memory_policy_create_never_blocks_wrapper),
            ("Memory Policy - mixed create policies", test_memory_policy_mixed_create_wrapper),
            ("Memory Policy - edge arrow syntax", test_memory_policy_edge_arrow_wrapper),
            ("Memory Policy - edge create never", test_memory_policy_edge_create_never_wrapper),
            ("Memory Policy - GraphQL validation", test_memory_policy_graphql_validation_wrapper),
            ("Memory Policy - invalid link_to syntax", test_memory_policy_invalid_link_to_wrapper),
            ("Memory Policy - invalid policy mode", test_memory_policy_invalid_mode_wrapper),
        ]

        for test_name, test_func in memory_policy_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_delete_memory_tests(self, app_instance):
        """Run all delete memory related tests."""
        logger.info("🧪 Running Delete Memory Tests...")
        
        delete_tests = [
            ("Delete Memory - Basic", test_v1_delete_memory_1),
            ("Delete Memory - API Key", test_v1_delete_memory_with_api_key),
            ("Delete All Memories - Complete Workflow", test_delete_all_memories_complete_workflow_wrapper),
            ("Delete All Memories - External User ID", test_delete_all_memories_with_external_user_id_wrapper),
            ("Delete All Memories - No Memories Found", test_delete_all_memories_no_memories_found_wrapper),
        ]
        
        for test_name, test_func in delete_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_upload_document_tests(self, app_instance):
        """Run all upload document related tests."""
        logger.info("🧪 Running Upload Document Tests (V2)...")
        
        upload_tests = [
            ("Upload Document V2 - API Key", test_document_upload_v2_with_api_key_wrapper),
            ("Upload Document V2 - Session Token", test_document_upload_v2_with_session_token_wrapper),
            ("Upload Document V2 - Real PDF File", test_document_upload_v2_with_real_pdf_file_wrapper),
            ("Upload Document V2 - Real PDF with Custom Schema", test_document_upload_v2_with_real_pdf_file_custom_schema_wrapper),
            ("Upload Document V2 - Invalid File Type", test_document_upload_v2_invalid_file_type_wrapper),
            ("Upload Document V2 - Malicious Content", test_document_upload_v2_malicious_content_wrapper),
            ("Upload Document V2 - Large File", test_document_upload_v2_large_file_wrapper),
            ("Upload Document V2 - With Webhook", test_document_upload_v2_with_webhook_wrapper),
            ("Upload Document V2 - Multi-tenant", test_document_upload_v2_multi_tenant_wrapper),
            ("Upload Document V2 - Provider Preference", test_document_upload_v2_provider_preference_wrapper),
            ("Upload Document V2 - Authentication Failure", test_document_upload_v2_authentication_failure_wrapper),
            ("Document Status Endpoint", test_document_status_endpoint_wrapper),
            ("Document Cancel Endpoint", test_document_cancel_endpoint_wrapper),
            ("Reducto Provider - Direct and Route", test_reducto_provider_direct_and_route_wrapper),
            ("Reducto Provider - Simple Upload", test_reducto_provider_simple_upload_and_parse_wrapper),
            # Provider-specific tests
            ("Upload Document V2 - Gemini Provider", test_document_upload_v2_with_gemini_provider_wrapper),
            ("Upload Document V2 - TensorLake Provider", test_document_upload_v2_with_tensorlake_provider_wrapper),
            # PaddleOCR and DeepSeek-OCR providers out of scope for v1 test run
            # ("Upload Document V2 - PaddleOCR Provider", test_document_upload_v2_with_paddleocr_provider_wrapper),
            # ("Upload Document V2 - DeepSeek-OCR Provider", test_document_upload_v2_with_deepseek_ocr_provider_wrapper),
        ]
        
        for test_name, test_func in upload_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_create_user_tests(self, app_instance):
        """Run all create user related tests."""
        logger.info("🧪 Running Create User Tests...")
        
        create_user_tests = [
            ("Create User - Basic", test_create_user_v1_integration_wrapper),
            ("Create User - Anonymous", test_create_anonymous_user_v1_integration_wrapper),
            ("Create User - Batch", test_create_user_batch_v1_integration_wrapper),
        ]
        
        for test_name, test_func in create_user_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_get_user_tests(self, app_instance):
        """Run all get user related tests."""
        logger.info("🧪 Running Get User Tests...")
        
        get_user_tests = [
            ("Get User - Basic", test_get_user_v1_integration_wrapper),
        ]
        
        for test_name, test_func in get_user_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_update_user_tests(self, app_instance):
        """Run all update user related tests."""
        logger.info("🧪 Running Update User Tests...")
        
        update_user_tests = [
            ("Update User - Basic", test_update_user_v1_integration_wrapper),
        ]
        
        for test_name, test_func in update_user_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_delete_user_tests(self, app_instance):
        """Run all delete user related tests."""
        logger.info("🧪 Running Delete User Tests...")
        
        delete_user_tests = [
            ("Delete User - Basic", test_delete_user_v1_integration_wrapper),
            ("Delete User - By External ID", test_delete_user_by_external_id_integration_wrapper),
        ]
        
        for test_name, test_func in delete_user_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_list_users_tests(self, app_instance):
        """Run all list users related tests."""
        logger.info("🧪 Running List Users Tests...")
        
        list_users_tests = [
            ("List Users - Basic", test_list_users_v1_integration_wrapper),
        ]
        
        for test_name, test_func in list_users_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_feedback_tests(self, app_instance):
        """Run all feedback related tests."""
        logger.info("🧪 Running Feedback Tests...")
        
        feedback_tests = [
            ("Feedback - End to End", test_feedback_end_to_end_wrapper),
            ("Feedback - Get by ID", test_get_feedback_by_id_v1_wrapper),
        ]
        
        for test_name, test_func in feedback_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)

    async def run_sync_tests(self, app_instance):
        """Run sync endpoint tests."""
        logger.info("🧪 Running Sync Tests...")
        sync_tests = [
            ("Sync Tiers - Basic", test_v1_sync_tiers_basic),
            ("Sync Tiers - With Embeddings (small)", test_v1_sync_tiers_with_embeddings_small_limits),
            ("Sync Delta - Cursor Pagination", test_v1_sync_delta_cursor_pagination),
        ]
        for test_name, test_func in sync_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)

    async def run_query_log_tests(self, app_instance):
        """Run Query Log integration tests (search + feedback correlation)."""
        logger.info("🧪 Running Query Log Integration Tests...")

        querylog_tests = [
            ("QueryLog - Memory Metadata with Query Log Fields", test_memory_metadata_with_query_log_fields_wrapper),
            ("QueryLog - Classification Data Detection", test_classification_data_detection),
            ("QueryLog - Search with Query Log Integration", test_search_with_query_log_integration_wrapper),
            ("QueryLog - Agentic Graph Log Model Creation", test_agentic_graph_log_model_creation),
            ("QueryLog - User Feedback Log Model Creation", test_user_feedback_log_model_creation),
            ("QueryLog - Agentic Graph Log with Relations", test_agentic_graph_log_with_relations),
            ("QueryLog - User Feedback Log Minimal", test_user_feedback_log_minimal),
            ("QueryLog - Agentic Graph Log Storage Function", test_agentic_graph_log_storage_function),
            ("QueryLog - User Feedback Log Storage Function", test_user_feedback_log_storage_function),
            ("QueryLog - Agentic Graph Log Data Preparation", test_agentic_graph_log_data_preparation),
            ("QueryLog - User Feedback Log Data Preparation", test_user_feedback_log_data_preparation),
            ("QueryLog - Memory Retrieval Log Predicted Grouping", test_memory_retrieval_log_predicted_grouping_wrapper),
            ("QueryLog - Persisted with Classification", test_query_log_persisted_with_classification_wrapper),
            ("QueryLog - Real creation and memory increment", test_real_query_log_creation_and_memory_increment_wrapper),
            ("QueryLog - Cache hits increment on repeated search", test_cache_hits_increment_on_repeated_search_wrapper),
            ("QueryLog - Fused confidence matches weight delta", test_fused_confidence_matches_weight_delta_wrapper),
            ("QueryLog - Backfill Retrieval Counters Small Batch", test_backfill_retrieval_counters_small_batch_wrapper),
            ("QueryLog - Sync Tiers Returns Ranked Tier1", test_sync_tiers_returns_ranked_tier1_wrapper),
        ]

        for test_name, test_func in querylog_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)

    async def run_security_schema_tests(self, app_instance):
        """Run custom security schema end-to-end tests."""
        logger.info("🧪 Running Security Schema Tests...")

        security_schema_tests = [
            ("Custom Schema - Create Security Schema", test_v1_create_security_schema),
            ("Custom Schema - Add Memory with schema_id", test_v1_add_memory_with_schema_id),
            ("Custom Schema - Wait for Memory Processing", test_v1_wait_for_memory_processing),
            ("Custom Schema - Search Verify Neo4j Nodes", test_v1_search_verify_neo4j_nodes),
            ("Custom Schema - Search with Agentic Graph", test_v1_search_with_agentic_graph),
            ("Custom Schema - Add Memory with graph_override", test_v1_add_memory_with_graph_override),
            ("Custom Schema - Full Workflow Validation", test_v1_security_schema_full_workflow),
        ]

        for test_name, test_func in security_schema_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_graph_generation_tests(self, app_instance):
        """Run graph generation mode tests."""
        logger.info("🧪 Running Graph Generation Mode Tests...")

        graph_generation_tests = [
            ("Graph Generation - Auto Mode with Schema ID", test_auto_mode_with_schema_id),
            ("Graph Generation - Auto Mode with Property Overrides", test_auto_mode_with_property_overrides),
            ("Graph Generation - Manual Mode with Explicit Graph", test_manual_mode_with_explicit_graph),
        ]

        for test_name, test_func in graph_generation_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_multi_tenant_tests(self, app_instance):
        """Run multi-tenant authentication and scoping tests."""
        logger.info("🧪 Running Multi-Tenant Tests...")

        multi_tenant_tests = [
            ("Multi-Tenant - Auth Models", test_multi_tenant_auth_models_wrapper),
            ("Multi-Tenant - Memory Models Fields", test_memory_models_multi_tenant_fields_wrapper),
            ("Multi-Tenant - Batch Memory Scoping", test_batch_memory_multi_tenant_scoping_wrapper),
            ("Multi-Tenant - Backward Compatibility", test_backward_compatibility_wrapper),
        ]

        for test_name, test_func in multi_tenant_tests:
            result = await self.run_test(test_name, test_func, app_instance)
            self.results.append(result)
    
    async def run_all_tests(self):
        """Run all v1 endpoint tests sequentially."""
        logger.info("🚀 Starting V1 Endpoints Sequential Test Suite")
        self.start_time = time.time()
        suite_error = None
        
        try:
            app_instance = app
            # Run memory tests by endpoint group
            await self.run_add_memory_tests(app_instance)
            await self.run_batch_add_memory_tests(app_instance)
            await self.run_update_memory_tests(app_instance)
            await self.run_get_memory_tests(app_instance)
            await self.run_search_memory_tests(app_instance)
            await self.run_memory_policy_tests(app_instance)
            await self.run_delete_memory_tests(app_instance)
            await self.run_upload_document_tests(app_instance)
            
            # Run user tests by endpoint group
            await self.run_create_user_tests(app_instance)
            await self.run_get_user_tests(app_instance)
            await self.run_update_user_tests(app_instance)
            await self.run_delete_user_tests(app_instance)
            await self.run_list_users_tests(app_instance)
            
            # Run feedback tests
            await self.run_feedback_tests(app_instance)
            # Run sync tests
            await self.run_sync_tests(app_instance)
            # Run query log integration tests
            await self.run_query_log_tests(app_instance)
            # Run security schema tests
            await self.run_security_schema_tests(app_instance)
            # Run graph generation mode tests
            await self.run_graph_generation_tests(app_instance)
            # Run multi-tenant tests
            await self.run_multi_tenant_tests(app_instance)
        except Exception as e:
            # Catch any unhandled exceptions during test execution
            suite_error = str(e)
            logger.error(f"❌ Test suite crashed with unhandled exception: {e}", exc_info=True)
            # Add error to results so it's tracked in the report
            self.results.append({
                "test_name": "TEST_SUITE_ERROR",
                "status": "failed",
                "duration": time.time() - self.start_time,
                "error": f"Unhandled exception: {suite_error}",
                "details": None
            })
        finally:
            # Always generate report, even if there was an error
            self.end_time = time.time()
            self.generate_report()
            
            # Re-raise the exception after generating report so it's visible in terminal
            if suite_error:
                raise Exception(f"Test suite failed: {suite_error}")
    
    def generate_report(self):
        """Generate detailed test report."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"] == "passed"])
        failed_tests = len([r for r in self.results if r["status"] == "failed"])
        skipped_tests = len([r for r in self.results if r["status"] == "skipped"])
        total_duration = self.end_time - self.start_time
        
        # Calculate statistics
        passed_duration = sum(r["duration"] for r in self.results if r["status"] == "passed")
        failed_duration = sum(r["duration"] for r in self.results if r["status"] == "failed")
        
        # Generate report data
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "skipped_tests": skipped_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_duration": total_duration,
                "average_duration": total_duration / total_tests if total_tests > 0 else 0,
                "passed_duration": passed_duration,
                "failed_duration": failed_duration
            },
            "results": self.results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Print summary
        logger.info("=" * 80)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests} ✅")
        logger.info(f"Failed: {failed_tests} ❌")
        logger.info(f"Skipped: {skipped_tests} ⏭️")
        logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        logger.info(f"Total Duration: {total_duration:.2f}s")
        logger.info(f"Average Duration: {report['summary']['average_duration']:.2f}s")
        
        # Print failed tests
        if failed_tests > 0:
            logger.info("\n❌ FAILED TESTS:")
            for result in self.results:
                if result["status"] == "failed":
                    logger.info(f"  - {result['test_name']}: {result['error']}")
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report - use path relative to this script
        report_dir = Path(__file__).parent / "test_reports"
        report_dir.mkdir(exist_ok=True)
        json_file = report_dir / f"v1_endpoints_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 JSON report saved: {json_file}")
        
        # Save text report
        txt_file = report_dir / f"v1_endpoints_log_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write("V1 ENDPOINTS SEQUENTIAL TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Passed: {passed_tests}\n")
            f.write(f"Failed: {failed_tests}\n")
            f.write(f"Skipped: {skipped_tests}\n")
            f.write(f"Success Rate: {report['summary']['success_rate']:.1f}%\n")
            f.write(f"Total Duration: {total_duration:.2f}s\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 30 + "\n")
            for result in self.results:
                status_icon = "✅" if result["status"] == "passed" else ("⏭️" if result["status"] == "skipped" else "❌")
                f.write(f"{status_icon} {result['test_name']} ({result['duration']:.2f}s)\n")
                if result["error"]:
                    f.write(f"    Error: {result['error']}\n")
                if result["status"] == "skipped" and result.get("details"):
                    f.write(f"    Skipped: {result['details']}\n")
                f.write("\n")
        
        logger.info(f"📄 Text report saved: {txt_file}")
        logger.info("=" * 80)

# Create wrapper functions for user tests to work with the sequential test runner
async def test_create_user_v1_integration_wrapper(app_instance):
    """Wrapper for test_create_user_v1_integration to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_create_user_v1_integration(async_client)

async def test_create_anonymous_user_v1_integration_wrapper(app_instance):
    """Wrapper for test_create_anonymous_user_v1_integration to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_create_anonymous_user_v1_integration(async_client)

async def test_get_user_v1_integration_wrapper(app_instance):
    """Wrapper for test_get_user_v1_integration to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_get_user_v1_integration(async_client)

async def test_update_user_v1_integration_wrapper(app_instance):
    """Wrapper for test_update_user_v1_integration to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_update_user_v1_integration(async_client)

async def test_delete_user_v1_integration_wrapper(app_instance):
    """Wrapper for test_delete_user_v1_integration to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_delete_user_v1_integration(async_client)

async def test_delete_user_by_external_id_integration_wrapper(app_instance):
    """Wrapper for test_delete_user_by_external_id_integration to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_delete_user_by_external_id_integration(async_client)

async def test_list_users_v1_integration_wrapper(app_instance):
    """Wrapper for test_list_users_v1_integration to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_list_users_v1_integration(async_client)

async def test_create_user_batch_v1_integration_wrapper(app_instance):
    """Wrapper for test_create_user_batch_v1_integration to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_create_user_batch_v1_integration(async_client)

async def test_feedback_end_to_end_wrapper(app_instance):
    """Wrapper for test_feedback_end_to_end to work with sequential test runner."""
    # The test function creates its own client internally, so we just call it directly
    await test_feedback_end_to_end()

async def test_get_feedback_by_id_v1_wrapper(app_instance):
    """Wrapper for test_get_feedback_by_id_v1 to work with sequential test runner."""
    # The test function creates its own client internally, so we just call it directly
    await test_get_feedback_by_id_v1()

async def test_delete_all_memories_complete_workflow_wrapper(app_instance):
    """Wrapper for test_delete_all_memories_complete_workflow to work with sequential test runner."""
    # The test function creates its own client internally, so we just call it directly
    await test_delete_all_memories_complete_workflow()

async def test_delete_all_memories_with_external_user_id_wrapper(app_instance):
    """Wrapper for test_delete_all_memories_with_external_user_id to work with sequential test runner."""
    # The test function creates its own client internally, so we just call it directly
    await test_delete_all_memories_with_external_user_id()

async def test_delete_all_memories_no_memories_found_wrapper(app_instance):
    """Wrapper for test_delete_all_memories_no_memories_found to work with sequential test runner."""
    # The test function creates its own client internally, so we just call it directly
    await test_delete_all_memories_no_memories_found()

# Document processing test wrappers
async def test_document_upload_v2_with_api_key_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_with_api_key to work with sequential test runner."""
    await test_document_upload_v2_with_api_key(app_instance)

async def test_document_upload_v2_with_session_token_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_with_session_token to work with sequential test runner."""
    await test_document_upload_v2_with_session_token(app_instance)

async def test_document_upload_v2_invalid_file_type_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_invalid_file_type to work with sequential test runner."""
    await test_document_upload_v2_invalid_file_type(app_instance)

async def test_document_upload_v2_malicious_content_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_malicious_content to work with sequential test runner."""
    await test_document_upload_v2_malicious_content(app_instance)

async def test_document_upload_v2_large_file_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_large_file to work with sequential test runner."""
    await test_document_upload_v2_large_file(app_instance)

async def test_document_upload_v2_with_webhook_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_with_webhook to work with sequential test runner."""
    await test_document_upload_v2_with_webhook(app_instance)

async def test_document_upload_v2_multi_tenant_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_multi_tenant to work with sequential test runner."""
    await test_document_upload_v2_multi_tenant(app_instance)

async def test_document_upload_v2_provider_preference_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_provider_preference to work with sequential test runner."""
    await test_document_upload_v2_provider_preference(app_instance)

async def test_reducto_provider_direct_and_route_wrapper(app_instance):
    """Wrapper for test_reducto_provider_direct_and_route to work with sequential test runner."""
    await test_reducto_provider_direct_and_route(app_instance)

async def test_document_upload_v2_authentication_failure_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_authentication_failure to work with sequential test runner."""
    await test_document_upload_v2_authentication_failure(app_instance)

async def test_document_status_endpoint_wrapper(app_instance):
    """Wrapper for test_document_status_endpoint to work with sequential test runner."""
    await test_document_status_endpoint(app_instance)

async def test_document_cancel_endpoint_wrapper(app_instance):
    """Wrapper for test_document_cancel_endpoint to work with sequential test runner."""
    await test_document_cancel_endpoint(app_instance)

async def test_reducto_provider_simple_upload_and_parse_wrapper(app_instance):
    """Wrapper for test_reducto_provider_simple_upload_and_parse to work with sequential test runner."""
    # Note: test_reducto_provider_simple_upload_and_parse takes no arguments
    await test_reducto_provider_simple_upload_and_parse()

async def test_document_upload_v2_with_gemini_provider_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_with_gemini_provider to work with sequential test runner."""
    await test_document_upload_v2_with_gemini_provider(app_instance)

async def test_document_upload_v2_with_tensorlake_provider_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_with_tensorlake_provider to work with sequential test runner."""
    await test_document_upload_v2_with_tensorlake_provider(app_instance)

# PaddleOCR and DeepSeek-OCR wrapper functions - out of scope for v1 test run
# async def test_document_upload_v2_with_paddleocr_provider_wrapper(app_instance):
#     """Wrapper for test_document_upload_v2_with_paddleocr_provider to work with sequential test runner."""
#     await test_document_upload_v2_with_paddleocr_provider(app_instance)
# 
# async def test_document_upload_v2_with_deepseek_ocr_provider_wrapper(app_instance):
#     """Wrapper for test_document_upload_v2_with_deepseek_ocr_provider to work with sequential test runner."""
#     await test_document_upload_v2_with_deepseek_ocr_provider(app_instance)

async def test_document_upload_v2_with_real_pdf_file_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_with_real_pdf_file to work with sequential test runner."""
    await test_document_upload_v2_with_real_pdf_file(app_instance)

async def test_document_upload_v2_with_real_pdf_file_custom_schema_wrapper(app_instance):
    """Wrapper for test_document_upload_v2_with_real_pdf_file_custom_schema to work with sequential test runner."""
    await test_document_upload_v2_with_real_pdf_file_custom_schema(app_instance)

# Query log test wrappers
async def test_memory_metadata_with_query_log_fields_wrapper(app_instance):
    """Wrapper for test_memory_metadata_with_query_log_fields to work with sequential test runner."""
    await test_memory_metadata_with_query_log_fields()

async def test_search_with_query_log_integration_wrapper(app_instance):
    """Wrapper for test_search_with_query_log_integration to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_search_with_query_log_integration(async_client)

async def test_memory_retrieval_log_predicted_grouping_wrapper(app_instance):
    """Wrapper for test_memory_retrieval_log_predicted_grouping to work with sequential test runner."""
    # This test requires monkeypatch, we'll skip it for now
    pytest.skip("Requires monkeypatch fixture")

async def test_query_log_persisted_with_classification_wrapper(app_instance):
    """Wrapper for test_query_log_persisted_with_classification to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_query_log_persisted_with_classification(async_client)

async def test_backfill_retrieval_counters_small_batch_wrapper(app_instance):
    """Wrapper for test_backfill_retrieval_counters_small_batch to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_backfill_retrieval_counters_small_batch(async_client)

async def test_sync_tiers_returns_ranked_tier1_wrapper(app_instance):
    """Wrapper for test_sync_tiers_returns_ranked_tier1 to work with sequential test runner."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance), base_url="http://test") as async_client:
        await test_sync_tiers_returns_ranked_tier1(async_client)

# Search test wrappers for caplog tests
class DummyCaplog:
    def __init__(self):
        self.records = []
    def at_level(self, level):
        return nullcontext() if nullcontext else None
    def clear(self):
        self.records = []

async def test_search_v1_agentic_graph_wrapper(app_instance):
    """Wrapper for test_search_v1_agentic_graph to work with sequential test runner."""
    dummy_caplog = DummyCaplog()
    await test_search_v1_agentic_graph(app_instance, dummy_caplog)

async def test_v1_search_fixed_user_cache_test_wrapper(app_instance):
    """Wrapper for test_v1_search_fixed_user_cache_test to work with sequential test runner."""
    dummy_caplog = DummyCaplog()
    await test_v1_search_fixed_user_cache_test(app_instance, dummy_caplog)

# Memory policy end-to-end test wrappers
def _memory_policy_fixtures():
    return memory_policy_unique_id(), memory_policy_api_headers()

async def test_memory_policy_link_to_string_form_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestLinkToDSLEndToEnd().test_link_to_string_form(unique_id, headers)

async def test_memory_policy_link_to_list_form_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestLinkToDSLEndToEnd().test_link_to_list_form(unique_id, headers)

async def test_memory_policy_link_to_dict_form_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestLinkToDSLEndToEnd().test_link_to_dict_form_with_create_never(unique_id, headers)

async def test_memory_policy_link_to_exact_match_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestLinkToDSLEndToEnd().test_link_to_with_exact_match(unique_id, headers)

async def test_memory_policy_link_to_semantic_threshold_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestLinkToDSLEndToEnd().test_link_to_with_semantic_threshold(unique_id, headers)

async def test_memory_policy_auto_mode_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestFullMemoryPolicyEndToEnd().test_memory_policy_auto_mode_creates_nodes(unique_id, headers)

async def test_memory_policy_manual_mode_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestFullMemoryPolicyEndToEnd().test_memory_policy_manual_mode_exact_nodes(unique_id, headers)

async def test_memory_policy_omo_safety_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestFullMemoryPolicyEndToEnd().test_memory_policy_with_omo_safety(unique_id, headers)

async def test_memory_policy_schema_inheritance_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestSchemaLevelPolicyInheritance().test_schema_policy_inheritance(unique_id, headers)

async def test_memory_policy_override_schema_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestMemoryLevelPolicyOverride().test_memory_policy_overrides_schema(unique_id, headers)

async def test_memory_policy_link_to_with_policy_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestPolicyMerging().test_link_to_with_memory_policy(unique_id, headers)

async def test_memory_policy_link_to_merge_constraints_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestPolicyMerging().test_link_to_constraints_merge_with_memory_policy(unique_id, headers)

async def test_memory_policy_create_never_blocks_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestControlledVocabulary().test_create_never_blocks_new_nodes(unique_id, headers)

async def test_memory_policy_mixed_create_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestControlledVocabulary().test_mixed_create_policies(unique_id, headers)

async def test_memory_policy_edge_arrow_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestEdgeConstraintsEndToEnd().test_edge_arrow_syntax(unique_id, headers)

async def test_memory_policy_edge_create_never_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestEdgeConstraintsEndToEnd().test_edge_with_create_never(unique_id, headers)

async def test_memory_policy_graphql_validation_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestGraphQLValidation().test_validate_nodes_via_graphql(unique_id, headers)

async def test_memory_policy_invalid_link_to_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestErrorHandling().test_invalid_link_to_syntax_returns_error(unique_id, headers)

async def test_memory_policy_invalid_mode_wrapper(app_instance):
    unique_id, headers = _memory_policy_fixtures()
    await TestErrorHandling().test_invalid_memory_policy_mode_returns_error(unique_id, headers)

# Multi-tenant test wrappers (these don't need app parameter)
async def test_multi_tenant_auth_models_wrapper(app_instance):
    """Wrapper for test_multi_tenant_auth_models - doesn't need app."""
    await test_multi_tenant_auth_models()

async def test_memory_models_multi_tenant_fields_wrapper(app_instance):
    """Wrapper for test_memory_models_multi_tenant_fields - doesn't need app."""
    await test_memory_models_multi_tenant_fields()

async def test_batch_memory_multi_tenant_scoping_wrapper(app_instance):
    """Wrapper for test_batch_memory_multi_tenant_scoping - doesn't need app."""
    await test_batch_memory_multi_tenant_scoping()

async def test_backward_compatibility_wrapper(app_instance):
    """Wrapper for test_backward_compatibility - doesn't need app."""
    await test_backward_compatibility()

async def main():
    """Main entry point."""
    tester = V1EndpointTester()
    try:
        await tester.run_all_tests()
        # Always generate report, even if some tests failed
        tester.generate_report()
    except Exception as e:
        # Log the error to ensure it's captured
        logger.error(f"Test suite main() failed: {e}", exc_info=True)
        # Generate report before re-raising so we can see what tests ran
        try:
            tester.generate_report()
        except Exception as report_error:
            logger.error(f"Failed to generate report: {report_error}")
        # Don't re-raise - we want the suite to complete and show the report
        # The individual test failures are already logged

if __name__ == "__main__":
    asyncio.run(main()) 