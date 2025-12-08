"""
Test batch_add_memory_quick with real workflow data structure.

This test simulates the exact data structure that comes from ProcessBatchMemoryFromPostWorkflow.
"""
import pytest
import json


class TestBatchQuickAddRealWorkflow:
    """Test batch_add_memory_quick with exact workflow data structure."""

    @pytest.mark.asyncio
    async def test_batch_quick_add_from_post_workflow(self):
        """
        Test batch_add_memory_quick with data structure from ProcessBatchMemoryFromPostWorkflow.
        
        This test uses the EXACT input structure that failed in production:
        - Post ID: G5WELDfPmv
        - Organization: Ky6jxP0yxI
        - Namespace: XhnQ7Y762d  
        - User: mhnkVbAdgG
        - Workspace: 85DF0bmVaO
        - Schema: Dh6EivRmo8
        """
        from cloud_plugins.temporal.activities.memory_activities import batch_add_memory_quick
        
        # Load real test data
        with open("tests/batch_quick_add_test_data_fixed.json", 'r') as f:
            batch_request_data = json.load(f)
        
        # Build batch_data_list in the EXACT format that ProcessBatchMemoryWorkflow.run creates
        # See batch_memory.py lines 418-480
        memories = batch_request_data["batch_request"]["memories"][:10]  # Test with 10 memories
        
        # Auth response from workflow
        auth_response = {
            "developer_id": batch_request_data["batch_request"]["user_id"],
            "end_user_id": batch_request_data["batch_request"]["user_id"],
            "workspace_id": "85DF0bmVaO",
            "organization_id": batch_request_data["batch_request"]["organization_id"],
            "namespace_id": batch_request_data["batch_request"]["namespace_id"],
            "is_qwen_route": False
        }
        
        # Schema specification (this is what's passed from ProcessBatchMemoryFromPostWorkflow)
        schema_specification = {
            "schema_id": "Dh6EivRmo8",
            "simple_schema_mode": False,
            "graph_override": None,
            "property_overrides": None
        }
        
        # Create batch_data_list matching workflow structure
        batch_data_list = []
        for idx in range(len(memories)):
            batch_data = {
                "batch_data": {
                    "batch_id": batch_request_data["batch_id"],
                    "api_key": "temporal_internal",
                    "legacy_route": True,
                    "auth_response": auth_response,
                    "batch_request": {
                        "user_id": batch_request_data["batch_request"]["user_id"],
                        "external_user_id": batch_request_data["batch_request"]["external_user_id"],
                        "organization_id": batch_request_data["batch_request"]["organization_id"],
                        "namespace_id": batch_request_data["batch_request"]["namespace_id"],
                        "memories": memories  # ALL memories
                    },
                    "schema_specification": schema_specification  # Include schema spec in batch_data
                },
                "index": idx
            }
            batch_data_list.append(batch_data)
        
        print(f"\n🚀 Testing batch_add_memory_quick with {len(batch_data_list)} memories")
        print(f"   Organization: {auth_response['organization_id']}")
        print(f"   Namespace: {auth_response['namespace_id']}")
        print(f"   Schema ID: {schema_specification['schema_id']}")
        
        # Call activity
        results = await batch_add_memory_quick(batch_data_list)
        
        # Verify
        assert len(results) == len(memories), f"Expected {len(memories)} results, got {len(results)}"
        
        successful = sum(1 for r in results if r.get("memory_id"))
        print(f"\n✅ Results:")
        print(f"   Total: {len(results)}")
        print(f"   Successful: {successful}/{len(results)}")
        print(f"   Success rate: {(successful/len(results)*100):.1f}%")
        
        # With real workspace/auth, we expect high success rate
        assert successful >= len(results) * 0.8, f"Success rate too low: {successful}/{len(results)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

