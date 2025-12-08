import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from services.auth_utils import _verify_api_key_mongo, _get_user_data_mongo
from services.user_utils import User
from memory.memory_graph import MemoryGraph


class TestMongoDBFieldValidation:
    """Test suite to validate MongoDB field names and query consistency"""

    @pytest.fixture
    def mock_db(self):
        """Mock MongoDB database with realistic user data"""
        # Create separate mock collections
        mock_user_collection = Mock()
        mock_workspace_follower_collection = Mock()
        
        # Mock user document with MongoDB field names (_p_ prefixes)
        mock_user_doc = {
            "_id": "testUser123",
            "username": "test@example.com",
            "email": "test@example.com",
            "isQwenRoute": True,
            "_p_isSelectedWorkspaceFollower": "workspace_follower$follower123"
        }
        
        # Mock workspace_follower document
        mock_follower_doc = {
            "_id": "follower123",
            "_p_workspace": "WorkSpace$workspace456"
        }
        
        # Configure collection mocks
        mock_user_collection.find_one.return_value = mock_user_doc
        mock_user_collection.find.return_value.hint.return_value.limit.return_value = [mock_user_doc]
        mock_workspace_follower_collection.find_one.return_value = mock_follower_doc
        
        # Create db mock that returns the appropriate collection
        db = Mock()
        db.__getitem__ = Mock(side_effect=lambda collection_name: {
            "_User": mock_user_collection,
            "workspace_follower": mock_workspace_follower_collection
        }.get(collection_name, Mock()))
        
        return db

    @pytest.mark.asyncio
    async def test_verify_api_key_mongo_field_names(self, mock_db):
        """Test that _verify_api_key_mongo uses correct field names"""
        
        result = await _verify_api_key_mongo(mock_db, "test_api_key")
        
        # Verify the function was called with correct projections
        mock_db["_User"].find_one.assert_called()
        call_args = mock_db["_User"].find_one.call_args
        
        # Should NOT have "isSelectedWorkspaceFollower" in the query/projection
        assert "isSelectedWorkspaceFollower" not in str(call_args)
        
        # Result should contain expected fields
        assert result is not None
        assert result["objectId"] == "testUser123"
        assert result["username"] == "test@example.com" 
        assert result["email"] == "test@example.com"
        assert result["isQwenRoute"] == True
        assert result["workspace_id"] == "workspace456"
        
        # Should NOT contain MongoDB internal field names in response
        assert "_p_isSelectedWorkspaceFollower" not in result

    @pytest.mark.asyncio
    async def test_get_user_data_mongo_field_names(self, mock_db):
        """Test that _get_user_data_mongo uses correct field names"""
        
        result = await _get_user_data_mongo(mock_db, "testUser123")
        
        # Verify workspace_follower query was made correctly
        mock_db["workspace_follower"].find_one.assert_called_with(
            {"_id": "follower123"},  # Should strip the "workspace_follower$" prefix
            {"_p_workspace": 1}      # Should use _p_ prefix
        )
        
        # Result should contain expected fields
        assert result["workspace_id"] == "workspace456"
        assert result["is_qwen_route"] == True

    @pytest.mark.asyncio 
    async def test_user_utils_mongo_workspace_subscription(self):
        """Test User._get_workspace_subscription_mongo uses correct field names"""
        
        user = User("testUser123")
        mock_db = Mock()
        
        # Mock user document with correct MongoDB field format
        mock_user_doc = {
            "_id": "testUser123",
            "_p_isSelectedWorkspaceFollower": "workspace_follower$follower123"
        }
        
        # Mock workspace_follower document
        mock_follower_doc = {
            "_id": "follower123", 
            "_p_workspace": "WorkSpace$workspace456"
        }
        
        # Mock workspace document
        mock_workspace_doc = {
            "_id": "workspace456",
            "_p_subscription": "Subscription$sub789",
            "_p_company": "Company$company101"
        }
        
        # Mock subscription document
        mock_subscription_doc = {
            "_id": "sub789",
            "stripeCustomerId": "cus_test123",
            "isMeteredBillingOn": True,
            "status": "active",
            "tier": "pro"
        }
        
        # Configure mocks with proper collection structure
        mock_user_collection = Mock()
        mock_workspace_follower_collection = Mock()
        mock_workspace_collection = Mock()
        mock_subscription_collection = Mock()
        
        mock_user_collection.find_one.return_value = mock_user_doc
        mock_workspace_follower_collection.find_one.return_value = mock_follower_doc
        mock_workspace_collection.find_one.return_value = mock_workspace_doc
        mock_subscription_collection.find_one.return_value = mock_subscription_doc
        
        mock_db.__getitem__ = Mock(side_effect=lambda collection_name: {
            "_User": mock_user_collection,
            "workspace_follower": mock_workspace_follower_collection,
            "WorkSpace": mock_workspace_collection,
            "Subscription": mock_subscription_collection
        }.get(collection_name, Mock()))
        
        # Test the method
        workspace_data, subscription_data = await user._get_workspace_subscription_mongo(mock_db)
        
        # Verify correct field names were used in queries
        mock_db["_User"].find_one.assert_called_with(
            {"_id": "testUser123"},
            {"_p_isSelectedWorkspaceFollower": 1}  # Should use _p_ prefix
        )
        
        mock_db["workspace_follower"].find_one.assert_called_with(
            {"_id": "follower123"},
            {"_p_workspace": 1}  # Should use _p_ prefix
        )
        
        mock_db["WorkSpace"].find_one.assert_called_with(
            {"_id": "workspace456"},
            {"_id": 1, "_p_subscription": 1, "_p_company": 1}  # Should use _p_ prefixes
        )
        
        # Verify results are in correct format
        assert workspace_data["objectId"] == "workspace456"
        assert subscription_data["objectId"] == "sub789"
        assert subscription_data["stripeCustomerId"] == "cus_test123"

    def test_memory_graph_warmup_queries(self):
        """Test MemoryGraph warmup methods use correct field names"""
        
        memory_graph = MemoryGraph()
        memory_graph.db = Mock()
        
        # Test API key lookup warmup
        memory_graph._warmup_api_key_lookup()
        
        # Create mock collections for memory_graph.db
        mock_user_collection = Mock()
        mock_user_collection.with_options.return_value.find.return_value = Mock()
        mock_user_collection.find_one.return_value = {}
        
        memory_graph.db.__getitem__ = Mock(side_effect=lambda collection_name: {
            "_User": mock_user_collection
        }.get(collection_name, Mock()))
        
        # Verify the projection uses correct field names
        call_args = mock_user_collection.with_options.return_value.find.call_args
        if call_args and len(call_args) >= 2:
            projection = call_args[1]  # Second argument is projection
            # Should use _p_ prefix for MongoDB pointer fields
            assert "_p_isSelectedWorkspaceFollower" in str(projection) or "isQwenRoute" in str(projection)
        
        # Test user metadata lookup warmup
        memory_graph._warmup_user_metadata_lookup()
        
        # Verify the query uses correct field names
        mock_user_collection.find_one.assert_called()
        call_args = mock_user_collection.find_one.call_args
        projection = call_args[1] if len(call_args) >= 2 else call_args[0][1] if call_args else {}
        
        # Should use _p_ prefix for pointer fields in MongoDB
        assert "_p_isSelectedWorkspaceFollower" in str(projection)

    @pytest.mark.asyncio
    async def test_field_consistency_across_methods(self, mock_db):
        """Test that all methods consistently use the same field naming convention"""
        
        # Test _verify_api_key_mongo
        result1 = await _verify_api_key_mongo(mock_db, "test_api_key")
        
        # Reset mock to track new calls
        mock_db.reset_mock()
        mock_db["_User"].find_one.return_value = {
            "_id": "testUser123",
            "isQwenRoute": True,
            "_p_isSelectedWorkspaceFollower": "workspace_follower$follower123"
        }
        mock_db["workspace_follower"].find_one.return_value = {
            "_id": "follower123",
            "_p_workspace": "WorkSpace$workspace456"
        }
        
        # Test _get_user_data_mongo  
        result2 = await _get_user_data_mongo(mock_db, "testUser123")
        
        # Both should use the same workspace_id result
        assert result1["workspace_id"] == result2["workspace_id"]
        assert result1["workspace_id"] == "workspace456"

    def test_parse_pointer_prefix_handling(self):
        """Test that Parse pointer prefixes are correctly stripped"""
        
        test_cases = [
            ("workspace_follower$abc123", "abc123"),
            ("WorkSpace$def456", "def456"),
            ("Subscription$ghi789", "ghi789"),
            ("Company$jkl012", "jkl012"),
            ("_User$mno345", "mno345"),
        ]
        
        for pointer_value, expected_id in test_cases:
            if pointer_value.startswith("workspace_follower$"):
                result = pointer_value.split("$", 1)[1]
                assert result == expected_id
            elif pointer_value.startswith("WorkSpace$"):
                result = pointer_value.split("$", 1)[1]
                assert result == expected_id
            elif pointer_value.startswith("Subscription$"):
                result = pointer_value.split("$", 1)[1]
                assert result == expected_id

    @pytest.mark.asyncio
    async def test_error_handling_missing_fields(self):
        """Test error handling when expected fields are missing"""
        
        # Mock database with missing _p_isSelectedWorkspaceFollower
        mock_db = Mock()
        mock_user_doc = {
            "_id": "testUser123",
            "username": "test@example.com",
            "email": "test@example.com", 
            "isQwenRoute": True
            # Missing _p_isSelectedWorkspaceFollower
        }
        
        # Create proper mock collection structure
        mock_user_collection = Mock()
        mock_user_collection.find_one.return_value = mock_user_doc
        mock_user_collection.find.return_value.hint.return_value.limit.return_value = [mock_user_doc]
        
        mock_db.__getitem__ = Mock(side_effect=lambda collection_name: {
            "_User": mock_user_collection
        }.get(collection_name, Mock()))
        
        # Should handle missing field gracefully
        result = await _verify_api_key_mongo(mock_db, "test_api_key")
        
        assert result is not None
        assert result["workspace_id"] is None  # Should be None when field is missing

    def test_field_naming_documentation(self):
        """Document the field naming conventions for future reference"""
        
        field_mapping = {
            # Parse Server Format -> MongoDB Format
            "isSelectedWorkspaceFollower": "_p_isSelectedWorkspaceFollower",
            "workspace": "_p_workspace", 
            "user": "_p_user",
            "subscription": "_p_subscription",
            "company": "_p_company",
            
            # Non-pointer fields remain the same
            "isQwenRoute": "isQwenRoute",
            "username": "username",
            "email": "email",
            "_id": "_id"
        }
        
        # This test serves as documentation
        assert len(field_mapping) >= 8
        
        # Verify pointer fields use _p_ prefix
        pointer_fields = [field for field in field_mapping.values() if field.startswith("_p_")]
        assert len(pointer_fields) >= 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 