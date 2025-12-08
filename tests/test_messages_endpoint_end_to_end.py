#!/usr/bin/env python3
"""
End-to-End Messages Endpoint Test

Tests the complete workflow of:
1. Creating chat messages via POST /v1/messages
2. Storing messages in Parse PostMessage class with Chat session tracking
3. Verifying messages are stored and retrievable by sessionId
4. Testing message processing pipeline (when enabled)
5. Validating role-based memory categorization

This test validates that the messages endpoint works end-to-end with proper:
- Message storage in Parse PostMessage class
- Chat session creation and management
- Message retrieval by sessionId
- Processing pipeline integration
- Role-based memory creation
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import httpx
import pytest

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.message_models import MessageRequest, MessageResponse
from models.shared_types import MessageRole, MemoryMetadata
from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

# Test configuration - Using same pattern as security test
BASE_URL = "http://localhost:8000"
TEST_API_KEY = "f80c5a2940f21882420b41690522cb2c"
TEST_SESSION_TOKEN = "r:578db0db09b3159b7ec98e0043b2af9a"

# Headers for API requests
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": TEST_API_KEY,
    "Authorization": f"Bearer {TEST_SESSION_TOKEN}"
}

@dataclass
class ValidationResult:
    """Result of a validation check"""
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None

@dataclass
class TestReport:
    """Comprehensive test report"""
    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    validations: List[ValidationResult] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def passed_validations(self) -> int:
        return sum(1 for v in self.validations if v.passed)
    
    @property
    def failed_validations(self) -> int:
        return sum(1 for v in self.validations if not v.passed)
    
    @property
    def success_rate(self) -> float:
        if not self.validations:
            return 0.0
        return self.passed_validations / len(self.validations)
    
    def add_validation(self, test_name: str, passed: bool, message: str, details: Optional[Dict[str, Any]] = None):
        """Add a validation result"""
        self.validations.append(ValidationResult(test_name, passed, message, details))
        
        status = "✅" if passed else "❌"
        logger.info(f"   {status} {test_name}: {message}")
        if details and not passed:
            logger.info(f"      Details: {details}")
    
    def finish(self):
        """Mark test as finished and log summary"""
        self.end_time = datetime.now()
        
        logger.info(f"\n📊 Test Report: {self.test_name}")
        logger.info(f"   Duration: {self.duration:.2f}s")
        logger.info(f"   Validations: {self.passed_validations}/{len(self.validations)} passed ({self.success_rate:.1%})")
        
        if self.failed_validations > 0:
            logger.warning(f"   ⚠️ {self.failed_validations} validations failed:")
            for v in self.validations:
                if not v.passed:
                    logger.warning(f"      - {v.test_name}: {v.message}")
        
        logger.info(f"   Artifacts: {list(self.artifacts.keys())}")

class TestMessagesEndToEnd:
    """Test class for end-to-end messages endpoint functionality"""
    
    @pytest.mark.asyncio
    async def test_message_storage_and_retrieval(self):
        """Test storing messages and retrieving them by sessionId"""
        report = TestReport("Message Storage and Retrieval", datetime.now())
        
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            try:
                # Generate unique session ID for this test
                test_session_id = f"test_session_{int(time.time())}"
                report.artifacts["session_id"] = test_session_id
                
                logger.info(f"💬 Testing message storage and retrieval for session: {test_session_id}")
                
                # Test messages to store - proper user/assistant alternating pattern
                # Using structured content format (as used in actual chat)
                test_messages = [
                    {
                        "content": [{"type": "text", "text": "I need help setting up automated database backups for our PostgreSQL production server"}],
                        "role": MessageRole.USER,
                        "process_messages": True,
                        "expected_status": "pending"
                    },
                    {
                        "content": [{"type": "text", "text": "I can help you set up automated PostgreSQL backups. What's your current backup strategy?"}],
                        "role": MessageRole.ASSISTANT,
                        "process_messages": True,
                        "expected_status": "pending"
                    },
                    {
                        "content": [{"type": "text", "text": "We currently don't have any automated backups, just manual exports once a week"}],
                        "role": MessageRole.USER,
                        "process_messages": True,
                        "expected_status": "pending"
                    },
                    {
                        "content": [{"type": "text", "text": "Got it. I recommend using pg_dump with cron jobs. Would you like me to walk you through the setup?"}],
                        "role": MessageRole.ASSISTANT,
                        "process_messages": True,
                        "expected_status": "pending"
                    }
                ]
                
                stored_message_ids = []
                
                # Step 1: Store messages
                logger.info("📝 Storing messages...")
                for i, msg_data in enumerate(test_messages, 1):
                    message_request = MessageRequest(
                        content=msg_data["content"],
                        role=msg_data["role"],
                        sessionId=test_session_id,
                        process_messages=msg_data["process_messages"],
                        metadata=MemoryMetadata(
                            topics=["database", "postgresql", "backups"],
                            location="Office"
                        )
                    )

                    # Extract text from structured content for logging
                    content_text = msg_data['content'][0]['text'] if isinstance(msg_data['content'], list) else msg_data['content']
                    logger.info(f"   Message {i}: {msg_data['role'].value} - {content_text[:50]}...")
                    
                    response = await client.post(
                        "/v1/messages",
                        headers=HEADERS,
                        json=message_request.model_dump()
                    )
                    
                    report.add_validation(
                        f"message_{i}_storage",
                        response.status_code == 200,
                        f"Message {i} storage returned {response.status_code}"
                    )
                    
                    if response.status_code == 200:
                        message_response = response.json()
                        
                        # Validate response structure
                        report.add_validation(
                            f"message_{i}_response_structure",
                            "objectId" in message_response,
                            f"Message {i} response contains objectId"
                        )
                        
                        if "objectId" in message_response:
                            message_id = message_response["objectId"]
                            stored_message_ids.append(message_id)
                            
                            # Validate processing status
                            actual_status = message_response.get("processing_status", "unknown")
                            expected_status = msg_data["expected_status"]
                            report.add_validation(
                                f"message_{i}_processing_status",
                                actual_status == expected_status,
                                f"Message {i} processing status: expected '{expected_status}', got '{actual_status}'"
                            )
                            
                            logger.info(f"   ✅ Stored with ID: {message_id}, Status: {actual_status}")
                        else:
                            logger.error(f"   ❌ No objectId in response: {message_response}")
                    else:
                        logger.error(f"   ❌ Failed to store message {i}: {response.text}")
                        report.artifacts[f"message_{i}_error"] = response.text
                
                report.artifacts["stored_message_ids"] = stored_message_ids
                logger.info(f"📊 Successfully stored {len(stored_message_ids)} messages")
                
                # Step 2: Wait a moment for Parse Server to index
                await asyncio.sleep(2)
                
                # Step 3: Retrieve messages by sessionId
                logger.info("📥 Retrieving messages by sessionId...")
                
                response = await client.get(
                    f"/v1/messages/sessions/{test_session_id}",
                    headers=HEADERS,
                    params={"limit": 10}
                )
                
                report.add_validation(
                    "message_retrieval",
                    response.status_code == 200,
                    f"Message retrieval returned {response.status_code}"
                )
                
                if response.status_code == 200:
                    retrieval_response = response.json()
                    
                    # Validate response structure
                    report.add_validation(
                        "retrieval_response_structure",
                        "messages" in retrieval_response,
                        "Retrieval response contains messages array"
                    )
                    
                    if "messages" in retrieval_response:
                        retrieved_messages = retrieval_response["messages"]
                        
                        report.add_validation(
                            "message_count_match",
                            len(retrieved_messages) == len(test_messages),
                            f"Retrieved {len(retrieved_messages)} messages, expected {len(test_messages)}"
                        )
                        
                        # Validate message order (should be chronological)
                        if len(retrieved_messages) >= 2:
                            first_msg_time = retrieved_messages[0].get("createdAt", "")
                            last_msg_time = retrieved_messages[-1].get("createdAt", "")
                            
                            report.add_validation(
                                "message_chronological_order",
                                first_msg_time <= last_msg_time,
                                "Messages returned in chronological order"
                            )
                        
                        # Validate message content and roles
                        for i, retrieved_msg in enumerate(retrieved_messages):
                            expected_msg = test_messages[i]

                            # Check content - handle both structured and string content
                            retrieved_content = retrieved_msg.get("content", retrieved_msg.get("message", ""))

                            # Extract text from expected structured content
                            if isinstance(expected_msg["content"], list):
                                expected_text = expected_msg["content"][0]["text"]
                            else:
                                expected_text = expected_msg["content"]

                            # Check if expected text is in retrieved content
                            # Handle case where retrieved content might be structured or string
                            if isinstance(retrieved_content, dict) and "data" in retrieved_content:
                                # Structured content wrapped
                                retrieved_text = retrieved_content["data"][0]["text"] if retrieved_content["data"] else ""
                            elif isinstance(retrieved_content, list):
                                # Direct structured content
                                retrieved_text = retrieved_content[0]["text"] if retrieved_content else ""
                            else:
                                # String content
                                retrieved_text = str(retrieved_content)

                            content_match = expected_text in retrieved_text
                            
                            report.add_validation(
                                f"retrieved_message_{i+1}_content",
                                content_match,
                                f"Message {i+1} content matches expected"
                            )
                            
                            # Check role
                            retrieved_role = retrieved_msg.get("role", retrieved_msg.get("messageRole", ""))
                            expected_role = expected_msg["role"].value
                            role_match = retrieved_role == expected_role
                            
                            report.add_validation(
                                f"retrieved_message_{i+1}_role",
                                role_match,
                                f"Message {i+1} role: expected '{expected_role}', got '{retrieved_role}'"
                            )

                            logger.info(f"   📝 Message {i+1}: {retrieved_role} - {retrieved_text[:50]}...")
                        
                        logger.info("✅ Message retrieval validation completed")
                    else:
                        logger.error("❌ No messages array in retrieval response")
                        report.artifacts["retrieval_error"] = retrieval_response
                else:
                    logger.error(f"❌ Message retrieval failed: {response.text}")
                    report.artifacts["retrieval_error"] = response.text
                
                # Step 4: Test conversation history format
                logger.info("💬 Testing conversation history format...")
                
                if "messages" in locals() and retrieved_messages:
                    conversation_pairs = []
                    current_pair = {}

                    for msg in retrieved_messages:
                        role = msg.get("role", msg.get("messageRole", ""))
                        content = msg.get("content", msg.get("message", ""))

                        # Extract text from structured content
                        if isinstance(content, dict) and "data" in content:
                            text_content = content["data"][0]["text"] if content["data"] else ""
                        elif isinstance(content, list):
                            text_content = content[0]["text"] if content else ""
                        else:
                            text_content = str(content)

                        if role == "user":
                            if current_pair:
                                conversation_pairs.append(current_pair)
                            current_pair = {"user": text_content}
                        elif role == "assistant" and "user" in current_pair:
                            current_pair["assistant"] = text_content
                            conversation_pairs.append(current_pair)
                            current_pair = {}

                    if current_pair:
                        conversation_pairs.append(current_pair)
                    
                    report.add_validation(
                        "conversation_pairs_formed",
                        len(conversation_pairs) >= 1,
                        f"Formed {len(conversation_pairs)} conversation pairs"
                    )
                    
                    for i, pair in enumerate(conversation_pairs, 1):
                        user_msg = pair.get("user", "")[:40]
                        assistant_msg = pair.get("assistant", "")[:40]
                        logger.info(f"   💬 Pair {i}:")
                        logger.info(f"      User: {user_msg}...")
                        if assistant_msg:
                            logger.info(f"      Assistant: {assistant_msg}...")
                
                report.finish()
                
                # Ensure critical validations passed
                critical_checks = ["message_retrieval", "message_count_match"]
                failed_critical = [v for v in report.validations if v.test_name in critical_checks and not v.passed]
                
                if failed_critical:
                    assert False, f"Critical message validations failed: {[v.test_name for v in failed_critical]}"
                
                return test_session_id, stored_message_ids
                
            except Exception as e:
                report.add_validation("test_execution", False, f"Test execution error: {e}")
                report.finish()
                raise
    
    @pytest.mark.asyncio
    async def test_process_messages_flag(self):
        """Test the process_messages flag functionality"""
        report = TestReport("Process Messages Flag Test", datetime.now())
        
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            try:
                test_session_id = f"process_test_{int(time.time())}"
                report.artifacts["session_id"] = test_session_id
                
                logger.info(f"🎛️ Testing process_messages flag for session: {test_session_id}")
                
                # Test both modes - using structured content format
                test_cases = [
                    {
                        "content": [{"type": "text", "text": "This message should be processed into memories"}],
                        "role": MessageRole.USER,
                        "process_messages": True,
                        "expected_status": "pending",
                        "description": "Full processing mode"
                    },
                    {
                        "content": [{"type": "text", "text": "I'll help you with that. Let me explain the process."}],
                        "role": MessageRole.ASSISTANT,
                        "process_messages": True,
                        "expected_status": "pending",
                        "description": "Assistant processing mode"
                    },
                    {
                        "content": [{"type": "text", "text": "This message should only be stored without processing"}],
                        "role": MessageRole.USER,
                        "process_messages": False,
                        "expected_status": "stored_only",
                        "description": "Storage only mode"
                    }
                ]
                
                for i, test_case in enumerate(test_cases, 1):
                    logger.info(f"   Test {i}: {test_case['description']}")
                    
                    message_request = MessageRequest(
                        content=test_case["content"],
                        role=test_case["role"],
                        sessionId=test_session_id,
                        process_messages=test_case["process_messages"],
                        metadata=MemoryMetadata(topics=["testing"])
                    )
                    
                    response = await client.post(
                        "/v1/messages",
                        headers=HEADERS,
                        json=message_request.model_dump()
                    )
                    
                    report.add_validation(
                        f"process_flag_test_{i}_request",
                        response.status_code == 200,
                        f"Test {i} request returned {response.status_code}"
                    )
                    
                    if response.status_code == 200:
                        message_response = response.json()
                        actual_status = message_response.get("processing_status", "unknown")
                        expected_status = test_case["expected_status"]
                        
                        report.add_validation(
                            f"process_flag_test_{i}_status",
                            actual_status == expected_status,
                            f"Test {i} status: expected '{expected_status}', got '{actual_status}'"
                        )
                        
                        logger.info(f"   ✅ {test_case['description']}: Status = {actual_status}")
                    else:
                        logger.error(f"   ❌ Test {i} failed: {response.text}")
                
                report.finish()
                return test_session_id
                
            except Exception as e:
                report.add_validation("test_execution", False, f"Test execution error: {e}")
                report.finish()
                raise
    
    @pytest.mark.asyncio
    async def test_session_status_endpoint(self):
        """Test the session status endpoint"""
        report = TestReport("Session Status Endpoint Test", datetime.now())
        
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            try:
                # First create some messages
                test_session_id, message_ids = await self.test_message_storage_and_retrieval()
                report.artifacts["session_id"] = test_session_id
                report.artifacts["message_ids"] = message_ids
                
                logger.info(f"📊 Testing session status for: {test_session_id}")
                
                # Test session status endpoint
                response = await client.get(
                    f"/v1/messages/sessions/{test_session_id}/status",
                    headers=HEADERS
                )
                
                report.add_validation(
                    "status_endpoint_request",
                    response.status_code == 200,
                    f"Status endpoint returned {response.status_code}"
                )
                
                if response.status_code == 200:
                    status_response = response.json()
                    
                    # Validate response structure
                    expected_fields = ["session_id", "message_count", "processing_summary"]
                    for field in expected_fields:
                        report.add_validation(
                            f"status_field_{field}",
                            field in status_response,
                            f"Status response contains '{field}' field"
                        )
                    
                    # Validate session ID matches
                    if "session_id" in status_response:
                        report.add_validation(
                            "status_session_id_match",
                            status_response["session_id"] == test_session_id,
                            "Status response session ID matches request"
                        )
                    
                    # Validate message count
                    if "message_count" in status_response:
                        expected_count = len(message_ids)
                        actual_count = status_response["message_count"]
                        report.add_validation(
                            "status_message_count",
                            actual_count == expected_count,
                            f"Message count: expected {expected_count}, got {actual_count}"
                        )
                    
                    logger.info(f"✅ Session status: {status_response}")
                else:
                    logger.error(f"❌ Status endpoint failed: {response.text}")
                    report.artifacts["status_error"] = response.text
                
                report.finish()
                return test_session_id
                
            except Exception as e:
                report.add_validation("test_execution", False, f"Test execution error: {e}")
                report.finish()
                raise
    
    @pytest.mark.asyncio
    async def test_full_messages_workflow(self):
        """Test the complete messages workflow end-to-end"""
        overall_report = TestReport("Full Messages Workflow", datetime.now())
        
        try:
            logger.info("🚀 Starting full messages endpoint end-to-end test...")
            
            # Step 1: Test message storage and retrieval
            session_id, message_ids = await self.test_message_storage_and_retrieval()
            overall_report.add_validation("storage_retrieval", True, f"Message storage/retrieval completed ({len(message_ids)} messages)")
            overall_report.artifacts["main_session_id"] = session_id
            overall_report.artifacts["main_message_ids"] = message_ids
            
            # Step 2: Test process_messages flag
            flag_session_id = await self.test_process_messages_flag()
            overall_report.add_validation("process_flag_test", True, f"Process messages flag test completed ({flag_session_id})")
            overall_report.artifacts["flag_test_session_id"] = flag_session_id
            
            # Step 3: Test session status endpoint
            status_session_id = await self.test_session_status_endpoint()
            overall_report.add_validation("session_status_test", True, f"Session status test completed ({status_session_id})")
            
            overall_report.finish()
            
            logger.info("🎉 Full messages workflow test completed successfully!")
            
            return {
                "main_session_id": session_id,
                "message_ids": message_ids,
                "flag_test_session_id": flag_session_id,
                "report": overall_report
            }
            
        except Exception as e:
            overall_report.add_validation("workflow_execution", False, f"Workflow error: {e}")
            overall_report.finish()
            raise

if __name__ == "__main__":
    """Run the test directly for debugging"""
    import pytest
    pytest.main([__file__, "-v", "-s"])


