#!/usr/bin/env python3
"""
Test script to verify MongoDB error handling works correctly.
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory.memory_graph import MongoDBConnectionWrapper
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure
from services.logging_config import get_logger
import time

# Initialize logger
logger = get_logger(__name__)

class TestMongoDBErrorHandling:
    """Test class for MongoDB error handling functionality"""
    
    def test_mongodb_wrapper_no_client(self):
        """Test MongoDB wrapper with no client"""
        wrapper = MongoDBConnectionWrapper()
        assert wrapper.is_connection_healthy() == False
        
    def test_safe_execute_no_client(self):
        """Test safe_execute with no MongoDB client"""
        wrapper = MongoDBConnectionWrapper()
        result = wrapper.safe_execute("test_operation", lambda: "test")
        assert result is None
        
    def test_error_counting(self):
        """Test error counting functionality"""
        wrapper = MongoDBConnectionWrapper()
        wrapper.error_count = 0
        wrapper.last_error_time = time.time()
        
        # Create a mock MongoDB client to trigger error handling
        from unittest.mock import Mock
        mock_client = Mock()
        mock_db = Mock()
        wrapper.mongo_client = mock_client
        wrapper.db = mock_db
        
        # Simulate errors
        for i in range(3):
            wrapper.safe_execute("test_operation", lambda: (_ for _ in ()).throw(AutoReconnect("Test error")))
        
        # The error count should be 3 since we're simulating errors
        assert wrapper.error_count == 3
        
    def test_error_window_reset(self):
        """Test error window reset functionality"""
        wrapper = MongoDBConnectionWrapper()
        wrapper.error_count = 0
        wrapper.last_error_time = time.time() - 70  # Outside 60-second window
        
        # Create a mock MongoDB client to trigger error handling
        from unittest.mock import Mock
        mock_client = Mock()
        mock_db = Mock()
        wrapper.mongo_client = mock_client
        wrapper.db = mock_db
        
        # Simulate an error
        wrapper.safe_execute("test_operation", lambda: (_ for _ in ()).throw(AutoReconnect("Test error")))
        
        # Should reset to 1 since we're outside the error window
        assert wrapper.error_count == 1
        
    def test_max_errors_reached(self):
        """Test max errors reached functionality"""
        wrapper = MongoDBConnectionWrapper()
        wrapper.error_count = 0
        wrapper.last_error_time = time.time()
        
        # Create a mock MongoDB client to trigger error handling
        from unittest.mock import Mock
        mock_client = Mock()
        mock_db = Mock()
        wrapper.mongo_client = mock_client
        wrapper.db = mock_db
        
        # Simulate more errors than max_errors
        for i in range(6):  # More than max_errors (5)
            wrapper.safe_execute("test_operation", lambda: (_ for _ in ()).throw(AutoReconnect("Test error")))
        
        # Should reach max errors
        assert wrapper.error_count >= wrapper.max_errors

class TestAsyncMongoDBErrorHandling:
    """Test class for async MongoDB error handling functionality"""
    
    @pytest.mark.asyncio
    async def test_async_safe_execute_no_client(self):
        """Test async safe_execute with no MongoDB client"""
        wrapper = MongoDBConnectionWrapper()
        result = await wrapper.safe_execute_async("test_async_operation", lambda: asyncio.sleep(0))
        assert result is None
        
    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test async error handling"""
        wrapper = MongoDBConnectionWrapper()
        
        async def failing_operation():
            raise AutoReconnect("Test async error")
        
        result = await wrapper.safe_execute_async("test_async_operation", failing_operation)
        assert result is None

class TestExceptionHandlers:
    """Test class for exception handler functionality"""
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        from services.error_handlers import is_error_rate_limited
        
        # Test rate limiting
        assert is_error_rate_limited("test_error") == False
        
    def test_error_rate_limiting(self):
        """Test error rate limiting functionality"""
        from services.error_handlers import is_error_rate_limited
        
        # Test error rate limiting
        for i in range(15):  # More than max_errors_per_window
            is_error_rate_limited("test_error")
        
        assert is_error_rate_limited("test_error") == True
        
    def test_exception_handler_functions_exist(self):
        """Test that exception handler functions exist"""
        from services.error_handlers import (
            mongodb_exception_handler,
            neo4j_exception_handler,
            qdrant_exception_handler,
            general_database_exception_handler,
            register_exception_handlers,
            DatabaseConnectionError,
            safe_database_operation
        )
        
        # Verify all functions exist
        assert callable(mongodb_exception_handler)
        assert callable(neo4j_exception_handler)
        assert callable(qdrant_exception_handler)
        assert callable(general_database_exception_handler)
        assert callable(register_exception_handlers)
        assert issubclass(DatabaseConnectionError, Exception)
        assert callable(safe_database_operation)

class TestMongoDBWrapperIntegration:
    """Test class for MongoDB wrapper integration"""
    
    def test_wrapper_initialization(self):
        """Test MongoDB wrapper initialization"""
        wrapper = MongoDBConnectionWrapper()
        assert wrapper.mongo_client is None
        assert wrapper.db is None
        assert wrapper.error_count == 0
        assert wrapper.max_errors == 5
        assert wrapper.error_window == 60
        
    def test_wrapper_with_mongo_client(self):
        """Test MongoDB wrapper with actual client (if available)"""
        # This test would require a real MongoDB connection
        # For now, we'll just test the structure
        wrapper = MongoDBConnectionWrapper()
        
        # Test that the wrapper can handle operations gracefully
        result = wrapper.safe_execute("test_operation", lambda: "success")
        assert result is None  # Because no client is available
        
    def test_health_check_without_client(self):
        """Test health check without MongoDB client"""
        wrapper = MongoDBConnectionWrapper()
        assert wrapper.is_connection_healthy() == False

def test_mongodb_error_types():
    """Test that we can import and use MongoDB error types"""
    from pymongo.errors import AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure
    
    # Test that we can create error instances
    auto_reconnect_error = AutoReconnect("Test auto reconnect")
    server_selection_error = ServerSelectionTimeoutError("Test server selection")
    connection_failure_error = ConnectionFailure("Test connection failure")
    
    assert isinstance(auto_reconnect_error, AutoReconnect)
    assert isinstance(server_selection_error, ServerSelectionTimeoutError)
    assert isinstance(connection_failure_error, ConnectionFailure)

@pytest.mark.asyncio
async def test_full_error_handling_workflow():
    """Test the complete error handling workflow"""
    wrapper = MongoDBConnectionWrapper()
    
    # Create a mock MongoDB client to trigger error handling
    from unittest.mock import Mock
    mock_client = Mock()
    mock_db = Mock()
    wrapper.mongo_client = mock_client
    wrapper.db = mock_db
    
    # Test sync operation
    result = wrapper.safe_execute("test_sync", lambda: (_ for _ in ()).throw(AutoReconnect("Test error")))
    assert result is None
    
    # Test async operation
    async def async_failing_operation():
        raise ServerSelectionTimeoutError("Test async error")
    
    result = await wrapper.safe_execute_async("test_async", async_failing_operation)
    assert result is None
    
    # Verify error counting
    assert wrapper.error_count > 0

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 