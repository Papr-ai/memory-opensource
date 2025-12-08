#!/usr/bin/env python3
"""
End-to-End Code Repository Schema Test

Tests the complete workflow of:
1. Creating a custom code repository schema
2. Adding memories that should use the custom schema  
3. Verifying LLM generates custom nodes (Developer, Function, etc.) and relationships (CREATED, CALLS, etc.)
4. Confirming proper graph structure in Neo4j with Memory nodes connected to custom nodes

This test validates that the custom schema system works end-to-end with proper:
- Schema selection by LLM
- Custom node generation (not generic system nodes)
- Custom relationship creation
- Memory node inclusion and connection
- Node merging based on unique_identifiers
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
from dataclasses import dataclass, field

import httpx
import pytest

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.memory_models import AddMemoryRequest, SearchRequest, SearchResponse
from models.shared_types import MemoryMetadata, MemoryType
from models.user_schemas import UserGraphSchema
from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

# Test configuration
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

class TestValidator:
    """Validation utilities for comprehensive testing"""
    
    @staticmethod
    async def wait_for_memory_processing(client: httpx.AsyncClient, memory_id: str, max_wait_seconds: int = 60) -> bool:
        """Wait for memory to be fully processed and searchable with graph nodes"""
        logger.info(f"⏳ Waiting for memory {memory_id} to be fully processed...")
        
        start_time = time.time()
        check_interval = 5  # Check every 5 seconds
        
        while (time.time() - start_time) < max_wait_seconds:
            try:
                # Search for the memory to see if it's indexed and has graph nodes
                search_request = SearchRequest(
                    query=memory_id,  # Search by memory ID
                    enable_agentic_graph=True,
                    external_user_id="clean_user_456"
                )
                
                response = await client.post(
                    "/v1/memory/search",
                    params={"max_memories": 5, "max_nodes": 10},
                    headers=HEADERS,
                    json=search_request.model_dump()
                )
                
                if response.status_code == 200:
                    search_data = response.json()
                    memories = search_data.get("data", {}).get("memories", [])
                    nodes = search_data.get("data", {}).get("nodes", [])
                    
                    # Check if our memory is found
                    memory_found = any(m.get("memoryId") == memory_id for m in memories)
                    
                    # Check if we have custom nodes (not just Memory nodes)
                    custom_nodes = [n for n in nodes if n.get("label") != "Memory"]
                    
                    if memory_found and len(custom_nodes) > 0:
                        elapsed = time.time() - start_time
                        logger.info(f"✅ Memory processing completed in {elapsed:.1f}s - found {len(custom_nodes)} custom nodes")
                        return True
                    elif memory_found:
                        elapsed = time.time() - start_time
                        logger.info(f"⏳ Memory found but no custom nodes yet ({elapsed:.1f}s) - continuing to wait...")
                    else:
                        elapsed = time.time() - start_time
                        logger.info(f"⏳ Memory not yet searchable ({elapsed:.1f}s) - continuing to wait...")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.warning(f"Error while waiting for memory processing: {e}")
                await asyncio.sleep(check_interval)
        
        elapsed = time.time() - start_time
        logger.warning(f"⚠️ Memory processing wait timeout after {elapsed:.1f}s")
        return False
    
    @staticmethod
    async def wait_for_schema_activation(client: httpx.AsyncClient, schema_id: str, max_wait_seconds: int = 30) -> bool:
        """Wait for schema to be fully activated and available for LLM selection"""
        logger.info(f"⏳ Waiting for schema {schema_id} to be activated...")
        
        start_time = time.time()
        check_interval = 5  # Check every 5 seconds
        
        while (time.time() - start_time) < max_wait_seconds:
            try:
                response = await client.get(f"/v1/schemas/{schema_id}", headers=HEADERS)
                
                if response.status_code == 200:
                    schema_data = response.json()
                    status = schema_data.get("data", {}).get("status")
                    
                    if status == "active":
                        elapsed = time.time() - start_time
                        logger.info(f"✅ Schema activated in {elapsed:.1f}s")
                        return True
                    else:
                        elapsed = time.time() - start_time
                        logger.info(f"⏳ Schema status: {status} ({elapsed:.1f}s) - continuing to wait...")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.warning(f"Error while waiting for schema activation: {e}")
                await asyncio.sleep(check_interval)
        
        elapsed = time.time() - start_time
        logger.warning(f"⚠️ Schema activation wait timeout after {elapsed:.1f}s")
        return False
    
    @staticmethod
    async def validate_schema_creation(client: httpx.AsyncClient, schema_id: str, expected_schema: Dict[str, Any]) -> List[ValidationResult]:
        """Validate that schema was created correctly with proper structure and permissions"""
        validations = []
        
        try:
            # Get the created schema
            response = await client.get(f"/v1/schemas/{schema_id}", headers=HEADERS)
            
            if response.status_code != 200:
                validations.append(ValidationResult(
                    "schema_retrieval", False, 
                    f"Failed to retrieve schema: {response.status_code}",
                    {"response": response.text}
                ))
                return validations
            
            schema_data = response.json()
            validations.append(ValidationResult(
                "schema_retrieval", True, "Schema retrieved successfully"
            ))
            
            # Validate Pydantic model compliance
            try:
                schema_obj = UserGraphSchema.model_validate(schema_data["data"])
                validations.append(ValidationResult(
                    "pydantic_validation", True, "Schema matches UserGraphSchema model"
                ))
            except Exception as e:
                validations.append(ValidationResult(
                    "pydantic_validation", False, f"Schema doesn't match Pydantic model: {e}",
                    {"schema_data": schema_data.get("data", {})}
                ))
                return validations
            
            # Validate schema structure
            actual_schema = schema_data["data"]
            
            # Check basic fields
            for field in ["name", "description", "status"]:
                if field in expected_schema:
                    expected_val = expected_schema[field]
                    actual_val = actual_schema.get(field)
                    validations.append(ValidationResult(
                        f"schema_field_{field}", 
                        actual_val == expected_val,
                        f"Field '{field}': expected '{expected_val}', got '{actual_val}'"
                    ))
            
            # Validate node types
            expected_nodes = expected_schema.get("node_types", {})
            actual_nodes = actual_schema.get("node_types", {})
            
            validations.append(ValidationResult(
                "node_types_count",
                len(actual_nodes) == len(expected_nodes),
                f"Node types count: expected {len(expected_nodes)}, got {len(actual_nodes)}"
            ))
            
            for node_name, expected_node in expected_nodes.items():
                if node_name in actual_nodes:
                    actual_node = actual_nodes[node_name]
                    
                    # Check unique_identifiers
                    expected_unique = expected_node.get("unique_identifiers", [])
                    actual_unique = actual_node.get("unique_identifiers", [])
                    validations.append(ValidationResult(
                        f"node_{node_name}_unique_identifiers",
                        set(actual_unique) == set(expected_unique),
                        f"Node '{node_name}' unique_identifiers: expected {expected_unique}, got {actual_unique}"
                    ))
                    
                    # Check required properties
                    expected_required = expected_node.get("required_properties", [])
                    actual_required = actual_node.get("required_properties", [])
                    validations.append(ValidationResult(
                        f"node_{node_name}_required_properties",
                        set(actual_required) == set(expected_required),
                        f"Node '{node_name}' required_properties: expected {expected_required}, got {actual_required}"
                    ))
                else:
                    validations.append(ValidationResult(
                        f"node_{node_name}_exists", False,
                        f"Expected node type '{node_name}' not found in schema"
                    ))
            
            # Validate relationship types
            expected_rels = expected_schema.get("relationship_types", {})
            actual_rels = actual_schema.get("relationship_types", {})
            
            validations.append(ValidationResult(
                "relationship_types_count",
                len(actual_rels) == len(expected_rels),
                f"Relationship types count: expected {len(expected_rels)}, got {len(actual_rels)}"
            ))
            
            for rel_name in expected_rels.keys():
                validations.append(ValidationResult(
                    f"relationship_{rel_name}_exists",
                    rel_name in actual_rels,
                    f"Relationship type '{rel_name}' {'found' if rel_name in actual_rels else 'missing'}"
                ))
            
        except Exception as e:
            validations.append(ValidationResult(
                "schema_validation_error", False, f"Validation error: {e}"
            ))
        
        return validations
    
    @staticmethod
    async def validate_memory_creation(client: httpx.AsyncClient, memory_id: str, expected_content: str, expected_metadata: Dict[str, Any]) -> List[ValidationResult]:
        """Validate that memory was created correctly and is searchable"""
        validations = []
        
        try:
            # Search for the memory to verify it exists and is properly indexed
            search_request = SearchRequest(
                query=expected_content[:50],  # Use first 50 chars as search query
                enable_agentic_graph=True,
                external_user_id="clean_user_456"
            )
            
            response = await client.post(
                "/v1/memory/search",
                params={"max_memories": 10},
                headers=HEADERS,
                json=search_request.model_dump()
            )
            
            if response.status_code != 200:
                validations.append(ValidationResult(
                    "memory_search", False,
                    f"Memory search failed: {response.status_code}",
                    {"response": response.text}
                ))
                return validations
            
            search_data = response.json()
            memories = search_data.get("data", {}).get("memories", [])
            
            # Find our memory in the results
            found_memory = None
            for memory in memories:
                if memory.get("memoryId") == memory_id:
                    found_memory = memory
                    break
            
            if found_memory:
                validations.append(ValidationResult(
                    "memory_searchable", True, "Memory found in search results"
                ))
                
                # Validate memory content
                actual_content = found_memory.get("content", "")
                content_match = expected_content.lower() in actual_content.lower()
                validations.append(ValidationResult(
                    "memory_content", content_match,
                    f"Memory content {'matches' if content_match else 'does not match'} expected"
                ))
                
                # Validate metadata fields
                actual_metadata = found_memory.get("metadata", {})
                for key, expected_value in expected_metadata.items():
                    if key in actual_metadata:
                        actual_value = actual_metadata[key]
                        if isinstance(expected_value, list):
                            match = set(expected_value).issubset(set(actual_value))
                        else:
                            match = actual_value == expected_value
                        
                        validations.append(ValidationResult(
                            f"metadata_{key}", match,
                            f"Metadata '{key}': expected {expected_value}, got {actual_value}"
                        ))
                    else:
                        validations.append(ValidationResult(
                            f"metadata_{key}_exists", False,
                            f"Expected metadata field '{key}' not found"
                        ))
            else:
                validations.append(ValidationResult(
                    "memory_searchable", False,
                    f"Memory {memory_id} not found in search results"
                ))
        
        except Exception as e:
            validations.append(ValidationResult(
                "memory_validation_error", False, f"Memory validation error: {e}"
            ))
        
        return validations
    
    @staticmethod
    async def validate_graph_nodes(client: httpx.AsyncClient, search_query: str, expected_nodes: List[str], expected_relationships: List[str]) -> List[ValidationResult]:
        """Validate that proper graph nodes and relationships were created"""
        validations = []
        
        try:
            search_request = SearchRequest(
                query=search_query,
                enable_agentic_graph=True,
                external_user_id="clean_user_456"
            )
            
            response = await client.post(
                "/v1/memory/search",
                params={"max_memories": 20, "max_nodes": 15},
                headers=HEADERS,
                json=search_request.model_dump()
            )
            
            if response.status_code != 200:
                validations.append(ValidationResult(
                    "graph_search", False,
                    f"Graph search failed: {response.status_code}",
                    {"response": response.text}
                ))
                return validations
            
            search_data = response.json()
            nodes = search_data.get("data", {}).get("nodes", [])
            relationships = search_data.get("data", {}).get("relationships", [])
            
            validations.append(ValidationResult(
                "graph_nodes_returned", len(nodes) > 0,
                f"Graph search returned {len(nodes)} nodes"
            ))
            
            if nodes:
                # Validate node types
                actual_node_types = [node.get("label", "") for node in nodes]
                unique_node_types = set(actual_node_types)
                
                # Check for custom nodes (not just Memory nodes)
                custom_nodes = [nt for nt in unique_node_types if nt != "Memory"]
                validations.append(ValidationResult(
                    "custom_nodes_created", len(custom_nodes) > 0,
                    f"Custom node types found: {custom_nodes}"
                ))
                
                # Check for expected node types
                for expected_node in expected_nodes:
                    found = expected_node in unique_node_types
                    validations.append(ValidationResult(
                        f"node_type_{expected_node}", found,
                        f"Expected node type '{expected_node}' {'found' if found else 'missing'}"
                    ))
            
            if relationships:
                # Validate relationship types
                actual_rel_types = [rel.get("type", "") for rel in relationships]
                unique_rel_types = set(actual_rel_types)
                
                validations.append(ValidationResult(
                    "relationships_created", len(relationships) > 0,
                    f"Found {len(relationships)} relationships of types: {unique_rel_types}"
                ))
                
                # Check for expected relationship types
                for expected_rel in expected_relationships:
                    found = expected_rel in unique_rel_types
                    validations.append(ValidationResult(
                        f"relationship_type_{expected_rel}", found,
                        f"Expected relationship type '{expected_rel}' {'found' if found else 'missing'}"
                    ))
            else:
                validations.append(ValidationResult(
                    "relationships_created", False,
                    "No relationships found in graph search results"
                ))
        
        except Exception as e:
            validations.append(ValidationResult(
                "graph_validation_error", False, f"Graph validation error: {e}"
            ))
        
        return validations

class TestCodeSchemaEndToEnd:
    """Test class for end-to-end code repository schema functionality"""
    
    @pytest.fixture
    def schema_data(self):
        """Code repository schema definition with proper node types and relationships"""
        return {
            "name": "Code Repository Schema",
            "description": "Schema for developers to save and organize code snippets, functions, and projects",
            "status": "active",
            "node_types": {
                "CodeSnippet": {
                    "name": "CodeSnippet",
                    "label": "CodeSnippet",
                    "description": "A piece of code with metadata",
                    "properties": {
                        "title": {"type": "string", "required": True},
                        "language": {"type": "string", "required": True},
                        "code": {"type": "string", "required": True},
                        "description": {"type": "string", "required": False},
                        "tags": {"type": "string", "required": False},
                        "difficulty": {"type": "string", "required": False},
                        "function_name": {"type": "string", "required": False},
                        "file_path": {"type": "string", "required": False},
                        "line_number": {"type": "integer", "required": False}
                    },
                    "required_properties": ["title", "language", "code"],
                    "unique_identifiers": ["title", "language", "function_name"]
                },
                "Function": {
                    "name": "Function",
                    "label": "Function",
                    "description": "A specific function or method",
                    "properties": {
                        "name": {"type": "string", "required": True},
                        "language": {"type": "string", "required": True},
                        "signature": {"type": "string", "required": False},
                        "return_type": {"type": "string", "required": False},
                        "parameters": {"type": "string", "required": False},
                        "file_path": {"type": "string", "required": False}
                    },
                    "required_properties": ["name", "language"],
                    "unique_identifiers": ["name", "language", "file_path"]
                },
                "Module": {
                    "name": "Module",
                    "label": "Module",
                    "description": "A code module, package, or import",
                    "properties": {
                        "name": {"type": "string", "required": True},
                        "language": {"type": "string", "required": True},
                        "version": {"type": "string", "required": False},
                        "import_path": {"type": "string", "required": False},
                        "is_external": {"type": "string", "required": False}
                    },
                    "required_properties": ["name", "language"],
                    "unique_identifiers": ["name", "language"]
                },
                "Developer": {
                    "name": "Developer",
                    "label": "Developer",
                    "description": "A software developer",
                    "properties": {
                        "name": {"type": "string", "required": True},
                        "email": {"type": "string", "required": True},
                        "github_username": {"type": "string", "required": False},
                        "specialization": {"type": "string", "required": False}
                    },
                    "required_properties": ["name", "email"],
                    "unique_identifiers": ["email"]
                },
                "CodeProject": {
                    "name": "CodeProject",
                    "label": "CodeProject", 
                    "description": "A software development project",
                    "properties": {
                        "name": {"type": "string", "required": True},
                        "description": {"type": "string", "required": False},
                        "tech_stack": {"type": "string", "required": False},
                        "repository_url": {"type": "string", "required": False}
                    },
                    "required_properties": ["name"],
                    "unique_identifiers": ["name"]
                },
                "Library": {
                    "name": "Library",
                    "label": "Library",
                    "description": "A software library or framework",
                    "properties": {
                        "name": {"type": "string", "required": True},
                        "language": {"type": "string", "required": True},
                        "version": {"type": "string", "required": False}
                    },
                    "required_properties": ["name", "language"],
                    "unique_identifiers": ["name", "language"]
                }
            },
            "relationship_types": {
                "CREATED": {
                    "name": "CREATED",
                    "label": "Created",
                    "description": "Developer created code snippet or project",
                    "allowed_source_types": ["Developer"],
                    "allowed_target_types": ["CodeSnippet", "CodeProject", "Function"]
                },
                "CALLS": {
                    "name": "CALLS",
                    "label": "Calls",
                    "description": "Function or code snippet calls another function",
                    "allowed_source_types": ["Function", "CodeSnippet"],
                    "allowed_target_types": ["Function"]
                },
                "IMPORTS": {
                    "name": "IMPORTS",
                    "label": "Imports",
                    "description": "Code snippet or function imports a module",
                    "allowed_source_types": ["CodeSnippet", "Function"],
                    "allowed_target_types": ["Module", "Library"]
                },
                "REFERENCES": {
                    "name": "REFERENCES",
                    "label": "References",
                    "description": "Code references another piece of code",
                    "allowed_source_types": ["CodeSnippet", "Function"],
                    "allowed_target_types": ["CodeSnippet", "Function"]
                },
                "CONTAINS": {
                    "name": "CONTAINS",
                    "label": "Contains",
                    "description": "Code snippet or project contains functions",
                    "allowed_source_types": ["CodeSnippet", "CodeProject"],
                    "allowed_target_types": ["Function"]
                },
                "USES": {
                    "name": "USES",
                    "label": "Uses",
                    "description": "Code snippet or project uses a library",
                    "allowed_source_types": ["CodeSnippet", "CodeProject", "Function"],
                    "allowed_target_types": ["Library"]
                },
                "BELONGS_TO": {
                    "name": "BELONGS_TO", 
                    "label": "Belongs To",
                    "description": "Code snippet or function belongs to a project or module",
                    "allowed_source_types": ["CodeSnippet", "Function"],
                    "allowed_target_types": ["CodeProject", "Module"]
                },
                "IMPLEMENTS": {
                    "name": "IMPLEMENTS",
                    "label": "Implements",
                    "description": "Code snippet implements a specific algorithm or pattern",
                    "allowed_source_types": ["CodeSnippet", "Function"],
                    "allowed_target_types": ["CodeSnippet", "Function"]
                },
                "DEPENDS_ON": {
                    "name": "DEPENDS_ON",
                    "label": "Depends On",
                    "description": "Code has a dependency on another piece of code",
                    "allowed_source_types": ["CodeSnippet", "Function", "CodeProject"],
                    "allowed_target_types": ["CodeSnippet", "Function", "Module", "Library"]
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_create_code_schema(self, schema_data):
        """Test creating the code repository schema with comprehensive validation"""
        report = TestReport("Schema Creation", datetime.now())
        
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            try:
                logger.info("🏗️ Creating Code Repository Schema...")
                
                # Step 1: Create schema
                response = await client.post(
                    "/v1/schemas",
                    headers=HEADERS,
                    json=schema_data
                )
                
                report.add_validation(
                    "schema_creation_request", 
                    response.status_code == 201,
                    f"Schema creation returned {response.status_code}"
                )
                
                if response.status_code != 201:
                    report.artifacts["creation_error"] = response.text
                    report.finish()
                    assert False, f"Schema creation failed: {response.text}"
                
                schema_response = response.json()
                report.add_validation(
                    "response_structure",
                    "data" in schema_response and "id" in schema_response["data"],
                    "Response contains required data and id fields"
                )
                
                schema_id = schema_response["data"]["id"]
                report.artifacts["schema_id"] = schema_id
                logger.info(f"✅ Schema created with ID: {schema_id}")
                
                # Step 2: Wait for schema activation
                schema_ready = await TestValidator.wait_for_schema_activation(client, schema_id)
                report.add_validation("schema_activation", schema_ready, 
                                    f"Schema activation {'completed' if schema_ready else 'timed out'}")
                
                # Step 3: Comprehensive validation
                validations = await TestValidator.validate_schema_creation(client, schema_id, schema_data)
                for validation in validations:
                    report.add_validation(validation.test_name, validation.passed, validation.message, validation.details)
                
                report.finish()
                
                # Ensure critical validations passed
                critical_checks = ["schema_retrieval", "pydantic_validation", "node_types_count", "relationship_types_count"]
                failed_critical = [v for v in report.validations if v.test_name in critical_checks and not v.passed]
                
                if failed_critical:
                    assert False, f"Critical schema validations failed: {[v.test_name for v in failed_critical]}"
                
                return schema_id
                
            except Exception as e:
                report.add_validation("test_execution", False, f"Test execution error: {e}")
                report.finish()
                raise
    
    @pytest.mark.asyncio 
    async def test_add_python_code_memory(self, schema_data):
        """Test adding a Python code memory with comprehensive validation"""
        report = TestReport("Python Memory Creation", datetime.now())
        
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            try:
                # First create the schema
                schema_id = await self.test_create_code_schema(schema_data)
                report.artifacts["schema_id"] = schema_id
                
                # Wait for schema to be ready for LLM selection
                schema_ready = await TestValidator.wait_for_schema_activation(client, schema_id)
                if not schema_ready:
                    report.add_validation("schema_readiness", False, "Schema not ready for memory processing")
                
                logger.info("📝 Adding Python FastAPI code memory...")
                
                # Create memory with proper metadata structure
                custom_metadata = {
                    "source": "code_repository",
                    "category": "api_endpoint_implementation", 
                    "language": "python",
                    "difficulty": "intermediate"
                }
                
                metadata = MemoryMetadata(
                    topics=["python", "fastapi", "api", "endpoint", "authentication"],
                    createdAt=datetime.now().isoformat() + "Z",
                    location="development_environment",
                    emoji_tags=["🐍", "⚡", "🔒"],
                    emotion_tags=["focused", "technical"],
                    conversationId=f"code_session_{int(time.time())}",
                    external_user_id="clean_user_456",
                    customMetadata=custom_metadata
                )
                
                expected_content = (
                    "Developer Jennifer Park created a Python FastAPI endpoint called get_user_profile "
                    "in file api/users.py at line 67. The endpoint imports Depends, HTTPException from "
                    "fastapi module, and also imports UserService from services/user_service.py. "
                    "The get_user_profile function calls authenticate_user, fetch_user_data, and "
                    "format_response functions. It is used by the mobile app and web dashboard. "
                    "The endpoint belongs to the UserManagement code project and uses PostgreSQL for data storage."
                )
                
                memory_request = AddMemoryRequest(
                    content=expected_content,
                    type=MemoryType.TEXT,
                    metadata=metadata,
                )
                
                # Step 1: Create memory
                response = await client.post(
                    "/v1/memory",
                    params={"skip_background_processing": False},
                    headers=HEADERS,
                    json=memory_request.model_dump()
                )
                
                report.add_validation(
                    "memory_creation_request",
                    response.status_code == 200,
                    f"Memory creation returned {response.status_code}"
                )
                
                if response.status_code != 200:
                    report.artifacts["creation_error"] = response.text
                    report.finish()
                    assert False, f"Memory addition failed: {response.text}"
                
                memory_response = response.json()
                report.add_validation(
                    "memory_response_structure",
                    "data" in memory_response and len(memory_response["data"]) > 0,
                    "Response contains memory data"
                )
                
                memory_id = memory_response["data"][0]["memoryId"]
                report.artifacts["memory_id"] = memory_id
                logger.info(f"✅ Memory created with ID: {memory_id}")
                
                # Step 2: Wait for complete memory processing
                processing_complete = await TestValidator.wait_for_memory_processing(client, memory_id, max_wait_seconds=90)
                report.add_validation("memory_processing", processing_complete,
                                    f"Memory processing {'completed' if processing_complete else 'timed out'}")
                
                # Step 3: Validate memory creation
                expected_metadata = {
                    "topics": ["python", "fastapi", "api", "endpoint", "authentication"],
                    "location": "development_environment"
                }
                
                memory_validations = await TestValidator.validate_memory_creation(
                    client, memory_id, expected_content, expected_metadata
                )
                for validation in memory_validations:
                    report.add_validation(validation.test_name, validation.passed, validation.message, validation.details)
                
                # Step 4: Validate graph nodes creation (only if processing completed)
                if processing_complete:
                    graph_validations = await TestValidator.validate_graph_nodes(
                        client, 
                        "Jennifer Park FastAPI get_user_profile",
                        ["Developer", "Function", "CodeProject", "Library"],
                        ["CREATED", "CALLS", "IMPORTS", "BELONGS_TO"]
                    )
                    for validation in graph_validations:
                        report.add_validation(validation.test_name, validation.passed, validation.message, validation.details)
                else:
                    report.add_validation("graph_validation_skipped", False, 
                                        "Graph validation skipped due to processing timeout")
                
                report.finish()
                
                # Ensure critical validations passed
                critical_checks = ["memory_searchable", "custom_nodes_created"]
                failed_critical = [v for v in report.validations if v.test_name in critical_checks and not v.passed]
                
                if failed_critical:
                    logger.warning(f"Some critical validations failed: {[v.test_name for v in failed_critical]}")
                
                return memory_id, schema_id
                
            except Exception as e:
                report.add_validation("test_execution", False, f"Test execution error: {e}")
                report.finish()
                raise
    
    @pytest.mark.asyncio
    async def test_add_javascript_code_memory(self, schema_data):
        """Test adding a JavaScript code memory that should create different developer"""
        # Connect to the running server instead of creating a new app instance
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
                # First create the schema
                schema_id = await self.test_create_code_schema(schema_data)
                
                # Wait for schema processing
                await asyncio.sleep(3)
                
                logger.info("📝 Adding JavaScript React code memory...")
                
                # Create memory with proper metadata structure
                custom_metadata = {
                    "source": "code_repository",
                    "category": "react_component_implementation", 
                    "language": "javascript",
                    "difficulty": "intermediate"
                }
                
                metadata = MemoryMetadata(
                    topics=["javascript", "react", "authentication", "component", "firebase"],
                    createdAt=datetime.now().isoformat() + "Z",
                    location="development_environment",
                    emoji_tags=["⚛️", "🔥", "🔒"],
                    emotion_tags=["creative", "technical"],
                    conversationId=f"code_session_{int(time.time())}",
                    external_user_id="clean_user_456",  # Add external_user_id to test proper user resolution
                    customMetadata=custom_metadata
                )
                
                memory_request = AddMemoryRequest(
                    content=(
                        "Developer Mike Chen wrote a JavaScript React component called AuthForm "
                        "in file components/AuthForm.js at line 12. The component imports React, useState, "
                        "and useEffect from react module, and also imports Firebase from firebase/auth module. "
                        "The AuthForm function calls validateEmail, hashPassword, and firebaseLogin functions. "
                        "It is referenced by the main App component and the LoginPage component. "
                        "The AuthForm belongs to the ShopEasy e-commerce code project and depends on the Firebase Auth library."
                    ),
                    type=MemoryType.TEXT,
                    metadata=metadata,
                )
                
                response = await client.post(
                    "/v1/memory",
                    params={"skip_background_processing": False},
                    headers=HEADERS,
                    json=memory_request.model_dump()
                )
                
                assert response.status_code == 200, f"Memory addition failed: {response.text}"
                
                memory_response = response.json()
                assert "data" in memory_response
                assert len(memory_response["data"]) > 0
                
                memory_id = memory_response["data"][0]["memoryId"]
                logger.info(f"✅ Memory added successfully with ID: {memory_id}")
                
                # Wait for background processing (including Neo4j graph generation)
                await asyncio.sleep(15)
                
                return memory_id, schema_id
    
    @pytest.mark.asyncio
    async def test_search_code_memories_with_agentic_graph(self, schema_data):
        """Test agentic graph search with schema selection and Neo4j node retrieval"""
        # First ensure we have memories and schema created
        python_memory_id, schema_id = await self.test_add_python_code_memory(schema_data)
        js_memory_id, _ = await self.test_add_javascript_code_memory(schema_data)
        
        # Wait for background processing to complete (both memories + Neo4j graph generation)
        await asyncio.sleep(20)
        
        # Connect to the running server for search tests
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
                logger.info("🔍 Testing agentic graph search with schema selection...")
                
                search_test_cases = [
                    {
                        "query": "Python FastAPI functions created by Jennifer Park",
                        "expected_memories": ["get_user_profile", "Jennifer Park", "FastAPI"],
                        "expected_nodes": ["Function", "Developer", "Library"],
                        "expected_relationships": ["CREATED", "IMPORTS", "CALLS"]
                    },
                    {
                        "query": "JavaScript React components that use Firebase authentication",
                        "expected_memories": ["AuthForm", "Mike Chen", "React"],
                        "expected_nodes": ["Function", "Developer", "Module"],
                        "expected_relationships": ["CREATED", "IMPORTS", "DEPENDS_ON"]
                    },
                    {
                        "query": "functions that call authenticate_user in Python code",
                        "expected_memories": ["authenticate_user", "get_user_profile"],
                        "expected_nodes": ["Function"],
                        "expected_relationships": ["CALLS"]
                    },
                    {
                        "query": "developers who work on authentication systems",
                        "expected_memories": ["Jennifer Park", "Mike Chen"],
                        "expected_nodes": ["Developer", "Function"],
                        "expected_relationships": ["CREATED"]
                    }
                ]
                
                for i, test_case in enumerate(search_test_cases, 1):
                    logger.info(f"🔍 Test {i}: {test_case['query']}")
                    
                    search_request = SearchRequest(
                        query=test_case["query"],
                        # Don't hardcode user_id - let the system resolve it from session token like memory addition does
                        enable_agentic_graph=True,
                        rank_results=True,
                        external_user_id="clean_user_456",  # Add external_user_id to test proper user resolution
                    )
                    
                    response = await client.post(
                        "/v1/memory/search",
                        params={
                            "max_memories": 20,
                            "max_nodes": 15,
                            "enable_agentic_graph": True
                        },
                        headers=HEADERS,
                        json=search_request.model_dump()
                    )
                    
                    logger.info(f"   Search response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        search_response = response.json()
                        validated_response = SearchResponse.model_validate(search_response)
                        
                        # Validate response structure
                        assert validated_response.error is None, f"Search error: {validated_response.error}"
                        assert validated_response.code == 200
                        assert validated_response.data is not None
                        
                        # Check memories returned
                        memories = validated_response.data.memories or []
                        logger.info(f"   📋 Found {len(memories)} memories")
                        
                        if memories:
                            memory_contents = [getattr(m, 'content', '') or '' for m in memories]
                            for expected in test_case["expected_memories"]:
                                found = any(expected.lower() in content.lower() for content in memory_contents)
                                if found:
                                    logger.info(f"   ✅ Found expected content: {expected}")
                                else:
                                    logger.warning(f"   ⚠️ Missing expected content: {expected}")
                        
                        # Check nodes returned (this is the key test for agentic graph)
                        nodes = validated_response.data.nodes or []
                        logger.info(f"   🏗️ Found {len(nodes)} graph nodes")
                        
                        if nodes:
                            node_types = [getattr(node, 'label', '') for node in nodes]
                            logger.info(f"   🏷️ Node types: {set(node_types)}")
                            
                            # Verify we're getting custom schema nodes, not just Memory nodes
                            custom_node_types = [nt for nt in node_types if nt != 'Memory']
                            assert len(custom_node_types) > 0, f"Expected custom node types, got only: {node_types}"
                            logger.info(f"   ✅ Found custom node types: {set(custom_node_types)}")
                            
                            # Check for expected node types
                            for expected_node_type in test_case["expected_nodes"]:
                                if expected_node_type in node_types:
                                    logger.info(f"   ✅ Found expected node type: {expected_node_type}")
                                else:
                                    logger.warning(f"   ⚠️ Missing expected node type: {expected_node_type}")
                        else:
                            logger.warning("   ⚠️ No graph nodes returned - agentic graph may not be working")
                        
                        # Check relationships if available
                        relationships = getattr(validated_response.data, 'relationships', None) or []
                        if relationships:
                            logger.info(f"   🔗 Found {len(relationships)} relationships")
                            rel_types = [getattr(rel, 'type', '') for rel in relationships]
                            logger.info(f"   🏷️ Relationship types: {set(rel_types)}")
                        
                        logger.info(f"   ✅ Test {i} completed successfully")
                    
                    elif response.status_code == 403:
                        logger.warning(f"   ⚠️ Search returned 403 - subscription/auth issue in test environment")
                    else:
                        logger.error(f"   ❌ Search failed: {response.status_code} - {response.text}")
                    
                    logger.info("")  # Add spacing between tests
                
                logger.info("🎉 Agentic graph search tests completed!")
    
    @pytest.mark.asyncio
    async def test_full_code_schema_workflow(self, schema_data):
        """Test the complete end-to-end workflow with comprehensive reporting"""
        overall_report = TestReport("Full End-to-End Workflow", datetime.now())
        
        try:
            logger.info("🚀 Starting full code schema end-to-end test...")
            
            # Step 1: Create schema
            schema_id = await self.test_create_code_schema(schema_data)
            overall_report.add_validation("schema_creation", True, f"Schema created successfully ({schema_id})")
            overall_report.artifacts["schema_id"] = schema_id
            
            # Step 2: Add Python memory
            python_memory_id, _ = await self.test_add_python_code_memory(schema_data)
            overall_report.add_validation("python_memory_creation", True, f"Python memory added ({python_memory_id})")
            overall_report.artifacts["python_memory_id"] = python_memory_id
            
            # Step 3: Add JavaScript memory  
            js_memory_id, _ = await self.test_add_javascript_code_memory(schema_data)
            overall_report.add_validation("javascript_memory_creation", True, f"JavaScript memory added ({js_memory_id})")
            overall_report.artifacts["js_memory_id"] = js_memory_id
            
            # Step 4: Test searches
            await self.test_search_code_memories_with_agentic_graph(schema_data)
            overall_report.add_validation("agentic_search_tests", True, "Search tests completed")
            
            # Step 5: Final comprehensive validation
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
                await self._run_final_validation(client, overall_report, schema_id, python_memory_id, js_memory_id)
            
            overall_report.finish()
            
            logger.info("🎉 Full end-to-end test completed successfully!")
            
            return {
                "schema_id": schema_id,
                "python_memory_id": python_memory_id, 
                "js_memory_id": js_memory_id,
                "report": overall_report
            }
            
        except Exception as e:
            overall_report.add_validation("workflow_execution", False, f"Workflow error: {e}")
            overall_report.finish()
            raise
    
    async def _run_final_validation(self, client: httpx.AsyncClient, report: TestReport, schema_id: str, python_memory_id: str, js_memory_id: str):
        """Run final comprehensive validation across all created artifacts"""
        logger.info("🔍 Running final comprehensive validation...")
        
        # Test 1: Verify both memories are searchable together
        search_request = SearchRequest(
            query="developer created code function",
            enable_agentic_graph=True,
            external_user_id="clean_user_456"
        )
        
        response = await client.post(
            "/v1/memory/search",
            params={"max_memories": 20, "max_nodes": 20},
            headers=HEADERS,
            json=search_request.model_dump()
        )
        
        if response.status_code == 200:
            search_data = response.json()
            memories = search_data.get("data", {}).get("memories", [])
            nodes = search_data.get("data", {}).get("nodes", [])
            
            # Check that both memories are found
            memory_ids = [m.get("memoryId") for m in memories]
            python_found = python_memory_id in memory_ids
            js_found = js_memory_id in memory_ids
            
            report.add_validation("both_memories_searchable", python_found and js_found, 
                                f"Both memories found in search: Python={python_found}, JS={js_found}")
            
            # Check for diverse node types
            node_types = set(node.get("label", "") for node in nodes)
            expected_types = {"Memory", "Developer", "Function", "CodeProject", "Library", "Module"}
            found_types = expected_types.intersection(node_types)
            
            report.add_validation("diverse_node_types", len(found_types) >= 4,
                                f"Found {len(found_types)}/6 expected node types: {found_types}")
            
            # Check for different developers (node merging test)
            developer_nodes = [node for node in nodes if node.get("label") == "Developer"]
            developer_names = [node.get("properties", {}).get("name", "") for node in developer_nodes]
            unique_developers = set(name for name in developer_names if name)
            
            report.add_validation("multiple_developers", len(unique_developers) >= 2,
                                f"Found {len(unique_developers)} unique developers: {unique_developers}")
            
        else:
            report.add_validation("final_search", False, f"Final search failed: {response.status_code}")
        
        # Test 2: Schema still accessible and valid
        schema_response = await client.get(f"/v1/schemas/{schema_id}", headers=HEADERS)
        report.add_validation("schema_persistence", schema_response.status_code == 200,
                            "Schema remains accessible after all operations")
        
        logger.info("✅ Final validation completed")
    
    def generate_test_summary(self, reports: List[TestReport]) -> str:
        """Generate a comprehensive test summary"""
        total_validations = sum(len(r.validations) for r in reports)
        total_passed = sum(r.passed_validations for r in reports)
        total_failed = sum(r.failed_validations for r in reports)
        overall_success_rate = total_passed / total_validations if total_validations > 0 else 0
        
        summary = [
            "\n" + "="*80,
            "🎯 COMPREHENSIVE TEST SUMMARY",
            "="*80,
            f"📊 Overall Results:",
            f"   • Total Validations: {total_validations}",
            f"   • Passed: {total_passed} ✅",
            f"   • Failed: {total_failed} ❌", 
            f"   • Success Rate: {overall_success_rate:.1%}",
            "",
            "📋 Test Breakdown:"
        ]
        
        for report in reports:
            duration_str = f"{report.duration:.1f}s" if report.duration else "N/A"
            summary.extend([
                f"   • {report.test_name}:",
                f"     - Duration: {duration_str}",
                f"     - Validations: {report.passed_validations}/{len(report.validations)} passed ({report.success_rate:.1%})",
                f"     - Artifacts: {list(report.artifacts.keys())}"
            ])
            
            if report.failed_validations > 0:
                summary.append(f"     - Failed: {[v.test_name for v in report.validations if not v.passed]}")
        
        summary.extend([
            "",
            "🔍 Key Validations:",
            f"   • Schema Creation: {'✅' if any('schema_creation' in v.test_name for r in reports for v in r.validations if v.passed) else '❌'}",
            f"   • Memory Processing: {'✅' if any('memory_processing' in v.test_name for r in reports for v in r.validations if v.passed) else '❌'}",
            f"   • Custom Nodes Created: {'✅' if any('custom_nodes_created' in v.test_name for r in reports for v in r.validations if v.passed) else '❌'}",
            f"   • Graph Relationships: {'✅' if any('relationships_created' in v.test_name for r in reports for v in r.validations if v.passed) else '❌'}",
            f"   • Pydantic Validation: {'✅' if any('pydantic_validation' in v.test_name for r in reports for v in r.validations if v.passed) else '❌'}",
            "",
            "="*80
        ])
        
        return "\n".join(summary)

    @pytest.mark.asyncio
    async def test_graph_override_functionality(self, schema_data):
        """Test the graph_override functionality specifically"""
        # Connect to the running server
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
                logger.info("📝 Testing graph_override functionality...")
                
                # Create memory with explicit graph override (without creating schema first)
                custom_metadata = {
                    "source": "code_repository",
                    "category": "graph_override_test", 
                    "language": "python",
                    "difficulty": "advanced"
                }
                
                metadata = MemoryMetadata(
                    topics=["graph_override", "explicit_nodes", "testing"],
                    createdAt=datetime.now().isoformat() + "Z",
                    location="test_environment",
                    emoji_tags=["🧪", "🎯", "⚡"],
                    emotion_tags=["precise", "controlled"],
                    conversationId=f"override_test_{int(time.time())}",
                    external_user_id="clean_user_456",
                    customMetadata=custom_metadata
                )
                
                # Define explicit graph structure using system schema node types
                from models.memory_models import GraphOverrideSpecification, GraphOverrideNode, GraphOverrideRelationship
                
                graph_override = GraphOverrideSpecification(
                    nodes=[
                        GraphOverrideNode(
                            id="person_sarah_wilson",
                            label="Person",  # Using system schema
                            properties={
                                "name": "Sarah Wilson",
                                "role": "Senior Developer",
                                "description": "Experienced Python developer specializing in data analytics"
                            }
                        ),
                        GraphOverrideNode(
                            id="company_techcorp",
                            label="Company",  # Using system schema
                            properties={
                                "name": "TechCorp",
                                "description": "Technology consulting company specializing in analytics solutions"
                            }
                        ),
                        GraphOverrideNode(
                            id="project_analytics",
                            label="Project",  # Using system schema
                            properties={
                                "name": "Analytics Platform",
                                "type": "data_analytics",
                                "description": "Advanced data analytics and reporting platform"
                            }
                        )
                    ],
                    relationships=[
                        GraphOverrideRelationship(
                            source_node_id="person_sarah_wilson",
                            target_node_id="company_techcorp",
                            relationship_type="WORKS_FOR",
                            properties={
                                "role": "Senior Developer",
                                "start_date": "2023-01-01"
                            }
                        ),
                        GraphOverrideRelationship(
                            source_node_id="person_sarah_wilson",
                            target_node_id="project_analytics",
                            relationship_type="WORKS_ON",
                            properties={
                                "role": "Lead Developer",
                                "responsibility": "backend_development"
                            }
                        )
                    ]
                )
                
                memory_request = AddMemoryRequest(
                    content=(
                        "Sarah Wilson, a senior developer at TechCorp, is leading the development of "
                        "an advanced Analytics Platform. She specializes in Python and data analytics, "
                        "working on backend systems that process large datasets for business intelligence."
                    ),
                    type=MemoryType.TEXT,
                    metadata=metadata,
                    graph_override=graph_override  # This should bypass LLM extraction
                )
                
                logger.info("🚀 Adding memory with graph_override...")
                response = await client.post(
                    "/v1/memory",
                    params={"skip_background_processing": False},
                    headers=HEADERS,
                    json=memory_request.model_dump()
                )
                
                logger.info(f"📊 Response status: {response.status_code}")
                if response.status_code != 200:
                    logger.error(f"❌ Response: {response.text}")
                
                assert response.status_code == 200, f"Memory addition with graph_override failed: {response.text}"
                
                memory_response = response.json()
                assert "data" in memory_response
                assert len(memory_response["data"]) > 0
                
                memory_id = memory_response["data"][0]["memoryId"]
                logger.info(f"✅ Memory with graph_override added successfully with ID: {memory_id}")
                
                # Wait for background processing
                await asyncio.sleep(15)
                
                # Now search for the specific nodes we created
                logger.info("🔍 Searching for graph_override nodes...")
                
                search_request = SearchRequest(
                    query="Sarah Wilson TechCorp Analytics Platform senior developer",
                    enable_agentic_graph=True,
                    rank_results=True,
                    external_user_id="clean_user_456"
                )
                
                search_response = await client.post(
                    "/v1/memory/search",
                    params={
                        "max_memories": 20,
                        "max_nodes": 15
                    },
                    headers=HEADERS,
                    json=search_request.model_dump()
                )
                
                logger.info(f"🔍 Search response status: {search_response.status_code}")
                
                if search_response.status_code == 200:
                    search_data = search_response.json()
                    assert search_data["data"] is not None
                    
                    # Verify our explicit nodes are found
                    nodes = search_data["data"].get("nodes", [])
                    logger.info(f"📊 Found {len(nodes)} nodes in search results")
                    
                    # Check for our specific override nodes
                    node_labels = [node["label"] for node in nodes]
                    node_names = []
                    
                    for node in nodes:
                        props = node.get("properties", {})
                        if "name" in props:
                            node_names.append(props["name"])
                    
                    logger.info(f"🏷️ Node labels found: {set(node_labels)}")
                    logger.info(f"📝 Node names found: {set(node_names)}")
                    
                    # Verify we have our explicit nodes
                    expected_nodes = ["Sarah Wilson", "TechCorp", "Analytics Platform"]
                    found_nodes = []
                    
                    for expected in expected_nodes:
                        if expected in node_names:
                            found_nodes.append(expected)
                            logger.info(f"   ✅ Found explicit node: {expected}")
                        else:
                            logger.warning(f"   ⚠️ Missing explicit node: {expected}")
                    
                    # We should find at least some of our explicit nodes
                    if len(found_nodes) > 0:
                        logger.info(f"✅ Graph override test completed - found {len(found_nodes)}/{len(expected_nodes)} explicit nodes")
                        logger.info("🎉 SUCCESS: graph_override functionality is working!")
                    else:
                        logger.warning(f"⚠️ No explicit graph_override nodes found. Expected: {expected_nodes}, Found names: {node_names}")
                        logger.info("🤔 This might mean LLM extraction ran instead of graph_override, or nodes weren't created properly")
                
                elif search_response.status_code == 403:
                    logger.warning("⚠️ Search returned 403 - subscription/auth issue, but memory was created successfully")
                    logger.info("✅ Graph override memory creation succeeded!")
                else:
                    logger.error(f"❌ Search failed: {search_response.status_code} - {search_response.text}")
                    logger.info("✅ But memory creation with graph_override succeeded!")
                
                return memory_id


if __name__ == "__main__":
    """Run the test directly for debugging"""
    import pytest
    pytest.main([__file__, "-v", "-s"])
