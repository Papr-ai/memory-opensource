import sys
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal, TYPE_CHECKING, Dict, Any, Optional, Tuple, Set, Union, TypeVar, Awaitable
from pinecone import Pinecone, ServerlessSpec, Vector
import numpy as np
from neo4j.graph import Node as Neo4jNode, Relationship as Neo4jRelationship
from numpy.typing import NDArray
from uuid import uuid4
from models.embedding_model import EmbeddingModel
from memory.memory_item import MemoryItem, memory_item_from_dict
from api_handlers.chat_gpt_completion import ChatGPTCompletion
from services.user_utils import User
from fastapi import APIRouter, BackgroundTasks, Depends
from httpx import AsyncClient
import contextlib 
import asyncio
import re
import uuid
from services.memory_management import (
    convert_acl, 
    update_memory_item_parse, 
    map_metadata_to_parse_fields,
    add_list_of_usecases,
    add_list_of_goals,
    get_user_goals,
    get_user_usecases,
    get_user_memGraph_schema,
    create_memory_graph,
    extract_goal_titles,
    extract_usecases,
    extract_relationship_types,
    extract_node_names,
    update_memory_item,
    add_list_of_usecases_async,
    add_list_of_goals_async,
    get_user_goals_async,
    get_user_usecases_async,
    get_user_memGraph_schema_async,
    create_memory_graph_async,
    extract_node_names, 
    extract_relationship_types, 
    store_generic_memory_item, 
    convert_neo_item_to_memory_item, 
    flatten_neo_item_to_parse_item, 
    retrieve_memory_item_with_user, 
    delete_memory_item_parse, convert_comma_string_to_list
)
from models.structured_outputs import (
    LLMGraphNode, LLMGraphRelationship, NodeLabel,
    PersonNode, CompanyNode, CustomerNode, ProjectNode,
    TaskNode, InsightNode, MeetingNode, CodeNode,
    PersonProperties, CompanyProperties, CustomerProperties,
    ProjectProperties, TaskProperties, InsightProperties,
    MeetingProperties, CodeProperties,
    NodeReference, ProcessMemoryResponse, MemoryMetrics, ProcessMemoryData
)
from models.memory_models import RelationshipType
from services.url_utils import clean_url
from models.acl import ACLCondition, ACLFilter
from models.memory_models import MemorySourceInfo, MemoryIDSourceLocation,MemorySourceLocation, NodeConverter, memory_item_to_node, NeoNode, NeoPersonNode, NeoCompanyNode, NeoProjectNode, NeoTaskNode, NeoInsightNode, NeoMeetingNode, NeoOpportunityNode, NeoCodeNode, SearchRequest, ContextItem, MemoryNodeProperties, RelatedMemoryResult, RerankingConfig, RerankingProvider, RelationshipItem
from models.shared_types import MemoryMetadata
from typing import Tuple, List, Dict, Any, Optional
from models.neo_path import GraphPath, PathSegment, QueryResult
from models.memory_models import MemoryNodeProperties, RelatedMemoryResult
import copy
# Add the 'services' directory to the sys.path
services_dir = str(Path(__file__).parent.parent / 'services')
if services_dir not in sys.path:
    sys.path.append(services_dir)

from models.parse_server import PineconeMatch, ErrorDetail, SystemUpdateStatus, PineconeQueryResponse, ParseStoredMemory, ParseUserPointer, MemoryRetrievalResult, RelatedMemoriesSuccess, RelatedMemoriesError, DeleteMemoryResponse, DeleteMemoryResult, DeletionStatus, UpdateMemoryItem, UpdateMemoryResponse, MemoryParseServerUpdate, ParsePointer, DeveloperUserPointer
from services.memory_management import store_memory_item, retrieve_memory_item_by_qdrant_id, retrieve_multiple_memory_items,retrieve_memory_item_parse, batch_store_memories, batch_store_memories_async, retrieve_memory_items_with_users_async
from memory.memory_item import MemoryItem, TextMemoryItem, CodeSnippetMemoryItem, WebpageMemoryItem, CodeFileMemoryItem, MeetingMemoryItem, PluginMemoryItem, DocumentMemoryItem, IssueMemoryItem, CustomerMemoryItem, memory_item_to_dict
from neo4j import GraphDatabase
from scipy import spatial
from dotenv import find_dotenv, load_dotenv
from os import environ as env
import json
import concurrent.futures
from utils.converter import convert_sets_to_lists
from services.user_utils import User, UserEncoder
from services.logging_config import get_logger
from datastore.neo4jconnection import Neo4jConnection, AsyncNeo4jConnection, CircuitBreaker
from config import get_features
import ssl
from typing import Protocol, Any, Dict, Optional
from services.logger_singleton import LoggerSingleton
import urllib3
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure
from services.mongo_client import get_mongo_db
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import PointStruct, PointsSelector, UpdateStatus, UpdateResult

# Global task tracking for memory processing
_memory_task_status: Dict[str, str] = {}  # 'pending', 'running', 'completed', 'failed'
_memory_batch_tasks: Dict[str, List[str]] = {}  # batch_id -> list of task_ids



# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

T = TypeVar('T')

# Load environment variables (conditionally based on USE_DOTENV)
use_dotenv = env.get("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)

# Initialize Neo4j client credentials and pinecone
NEO4J_URL = clean_url(env.get("NEO4J_URL"))
NEO4J_SECRET = clean_url(env.get("NEO4J_SECRET"))

PINECONE_KEY = clean_url(env.get("PINECONE_KEY"))
PINECONE_ENV = clean_url(env.get("PINECONE_ENV"))

#logger.info(f"ENV_FILE: {ENV_FILE}")
logger.debug(f"NEO4J_URL: {NEO4J_URL}")
#logger.info(f"NEO4J_SECRET: {NEO4J_SECRET}")
# Initialize Parse client
parse_application_id = clean_url(env.get("PARSE_APPLICATION_ID"))
parse_rest_api_key = clean_url(env.get("PARSE_REST_API_KEY"))

PARSE_SERVER_URL = clean_url(env.get("PARSE_SERVER_URL"))
HEADERS = {
    "X-Parse-Application-Id": parse_application_id,
    "X-Parse-REST-API-Key": parse_rest_api_key,
    "Content-Type": "application/json"
}

api_key = clean_url(env.get("OPENAI_API_KEY"))
organization_id = clean_url(env.get("OPENAI_ORGANIZATION"))

class MongoDBConnectionWrapper:
    """Enhanced wrapper to handle MongoDB connection errors gracefully with circuit breaker pattern"""
    
    def __init__(self, mongo_client=None, db=None):
        self.mongo_client = mongo_client
        self.db = db
        self.last_error_time = 0
        self.error_count = 0
        self.max_errors = 5
        self.error_window = 60  # seconds
        
        # Circuit breaker for MongoDB operations
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=300)
        self.fallback_mode = False
        self.last_fallback_time = 0
        self.connection_retry_count = 0
        self.max_connection_retries = 3
        
        # Health monitoring
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        self.connection_healthy = True
        
    async def is_connection_healthy(self) -> bool:
        """Enhanced health check with circuit breaker and timeout protection"""
        if not self.mongo_client or not self.db:
            self.connection_healthy = False
            return False
            
        # Check circuit breaker first
        if not await self.circuit_breaker.can_try():
            logger.warning("MongoDB circuit breaker is open, skipping health check")
            self.connection_healthy = False
            return False
            
        try:
            # Quick health check with timeout
            await asyncio.wait_for(
                asyncio.to_thread(self.db.command, "ping"),
                timeout=5.0  # 5 second timeout
            )
            self.connection_healthy = True
            self.connection_retry_count = 0  # Reset retry count on success
            return True
        except asyncio.TimeoutError:
            logger.warning("MongoDB health check timed out")
            await self.circuit_breaker.record_failure()
            self.connection_healthy = False
            return False
        except (AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure) as e:
            logger.warning(f"MongoDB health check failed: {e}")
            await self.circuit_breaker.record_failure()
            self.connection_healthy = False
            return False
        except Exception as e:
            logger.error(f"Unexpected MongoDB error: {e}")
            await self.circuit_breaker.record_failure()
            self.connection_healthy = False
            return False
    
    async def safe_execute(self, operation_name: str, operation_func, *args, **kwargs):
        """Enhanced safe execution with circuit breaker and timeout protection"""
        if not self.mongo_client or not self.db:
            logger.error(f"MongoDB not available for {operation_name}")
            return None
            
        # Check circuit breaker first
        if not await self.circuit_breaker.can_try():
            logger.warning(f"MongoDB circuit breaker is open, skipping {operation_name}")
            self.fallback_mode = True
            return None
            
        # Check if we're in fallback mode
        if self.fallback_mode:
            logger.warning(f"MongoDB in fallback mode, skipping {operation_name}")
            return None
            
        try:
            # Execute with timeout protection
            result = await asyncio.wait_for(
                asyncio.to_thread(operation_func, *args, **kwargs),
                timeout=30.0  # 30 second timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"MongoDB {operation_name} timed out after 30 seconds")
            await self.circuit_breaker.record_failure()
            self.fallback_mode = True
            return None
        except (AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure) as e:
            current_time = time.time()
            
            # Reset error count if outside error window
            if current_time - self.last_error_time > self.error_window:
                self.error_count = 0
                
            self.error_count += 1
            self.last_error_time = current_time
            
            logger.error(f"MongoDB connection error in {operation_name}: {e}")
            await self.circuit_breaker.record_failure()
            
            if self.error_count >= self.max_errors:
                logger.critical(f"Too many MongoDB errors ({self.error_count}), entering fallback mode")
                self.fallback_mode = True
                self.last_fallback_time = current_time
                
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {operation_name}: {e}")
            await self.circuit_breaker.record_failure()
            return None
    
    async def safe_execute_async(self, operation_name: str, operation_func, *args, **kwargs):
        """Enhanced async safe execution with circuit breaker and timeout protection"""
        if not self.mongo_client or not self.db:
            logger.error(f"MongoDB not available for {operation_name}")
            return None
            
        # Check circuit breaker first
        if not await self.circuit_breaker.can_try():
            logger.warning(f"MongoDB circuit breaker is open, skipping {operation_name}")
            self.fallback_mode = True
            return None
            
        # Check if we're in fallback mode
        if self.fallback_mode:
            logger.warning(f"MongoDB in fallback mode, skipping {operation_name}")
            return None
            
        try:
            # Execute with timeout protection
            result = await asyncio.wait_for(
                operation_func(*args, **kwargs),
                timeout=30.0  # 30 second timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"MongoDB {operation_name} timed out after 30 seconds")
            await self.circuit_breaker.record_failure()
            self.fallback_mode = True
            return None
        except (AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure) as e:
            current_time = time.time()
            
            # Reset error count if outside error window
            if current_time - self.last_error_time > self.error_window:
                self.error_count = 0
                
            self.error_count += 1
            self.last_error_time = current_time
            
            logger.error(f"MongoDB connection error in {operation_name}: {e}")
            await self.circuit_breaker.record_failure()
            
            if self.error_count >= self.max_errors:
                logger.critical(f"Too many MongoDB errors ({self.error_count}), entering fallback mode")
                self.fallback_mode = True
                self.last_fallback_time = current_time
                
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {operation_name}: {e}")
            await self.circuit_breaker.record_failure()
            return None
    
    async def attempt_recovery(self):
        """Attempt to recover from fallback mode by testing connection health"""
        if not self.fallback_mode:
            return True
            
        current_time = time.time()
        
        # Don't attempt recovery too frequently
        if current_time - self.last_fallback_time < 60:  # Wait at least 1 minute
            return False
            
        logger.info("Attempting MongoDB recovery from fallback mode...")
        
        # Test connection health
        if await self.is_connection_healthy():
            self.fallback_mode = False
            self.connection_retry_count = 0
            logger.info("MongoDB recovery successful, exiting fallback mode")
            return True
        else:
            logger.warning("MongoDB recovery attempt failed, staying in fallback mode")
            return False
    
    async def get_connection_stats(self):
        """Get MongoDB connection statistics for monitoring"""
        if not self.mongo_client:
            return {"status": "disconnected", "error": "No MongoDB client"}
            
        try:
            # Get server status
            server_status = await asyncio.wait_for(
                asyncio.to_thread(self.db.command, "serverStatus"),
                timeout=10.0
            )
            
            return {
                "status": "connected",
                "healthy": self.connection_healthy,
                "fallback_mode": self.fallback_mode,
                "circuit_breaker_open": not await self.circuit_breaker.can_try(),
                "error_count": self.error_count,
                "uptime": server_status.get("uptime", 0),
                "connections": server_status.get("connections", {}),
                "memory": server_status.get("mem", {})
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "healthy": self.connection_healthy,
                "fallback_mode": self.fallback_mode
            }

async def _add_monitored_memory_task(
    background_tasks: BackgroundTasks,
    task_func,
    batch_id: str,
    task_name: str,
    *args,
    **kwargs
) -> str:
    """
    Add a monitored background task for memory processing.
    
    Returns:
        Task ID for tracking
    """
    task_id = f"{batch_id}_{task_name}_{uuid.uuid4().hex[:8]}"
    
    async def monitored_task():
        try:
            _memory_task_status[task_id] = 'running'
            logger.info(f"Starting monitored memory task: {task_id}")
            
            # Execute the actual task
            result = await task_func(*args, **kwargs)
            
            _memory_task_status[task_id] = 'completed'
            logger.info(f"Completed memory task: {task_id}")
            return result
            
        except Exception as e:
            _memory_task_status[task_id] = 'failed'
            logger.error(f"Memory task {task_id} failed: {e}")
            raise
    
    # Add the monitored task to background tasks
    background_tasks.add_task(monitored_task)
    _memory_task_status[task_id] = 'pending'
    
    # Track this task as part of the batch
    if batch_id not in _memory_batch_tasks:
        _memory_batch_tasks[batch_id] = []
    _memory_batch_tasks[batch_id].append(task_id)
    
    logger.info(f"Added monitored memory task: {task_id} for batch: {batch_id}")
    return task_id

async def wait_for_memory_processing_completion(
    batch_id: str, 
    timeout_seconds: int = 30,
    poll_interval: float = 0.5
) -> bool:
    """
    Wait for memory processing background tasks to complete.
    
    Args:
        batch_id: Unique identifier for this batch
        timeout_seconds: Maximum time to wait
        poll_interval: How often to check task status
    
    Returns:
        True if all tasks completed successfully, False if timeout or failure
    """
    start_time = datetime.now()
    timeout = timedelta(seconds=timeout_seconds)
    
    logger.info(f"Starting memory processing monitoring for batch {batch_id}")
    
    while datetime.now() - start_time < timeout:
        # Check if all tasks for this batch are completed
        batch_tasks = _memory_batch_tasks.get(batch_id, [])
        
        if not batch_tasks:
            logger.warning(f"No memory processing tasks found for batch {batch_id}")
            return True
        
        completed_tasks = sum(1 for task_id in batch_tasks if _memory_task_status.get(task_id) in ['completed', 'failed'])
        
        if completed_tasks == len(batch_tasks):
            # All tasks are done, check if any failed
            failed_tasks = [task_id for task_id in batch_tasks if _memory_task_status.get(task_id) == 'failed']
            if failed_tasks:
                logger.error(f"Some memory processing tasks failed for batch {batch_id}: {failed_tasks}")
                return False
            else:
                logger.info(f"All memory processing tasks completed successfully for batch {batch_id}")
                return True
        
        # Log progress
        logger.debug(f"Batch {batch_id}: {completed_tasks}/{len(batch_tasks)} memory tasks completed")
        await asyncio.sleep(poll_interval)
    
    logger.warning(f"Timeout waiting for memory processing tasks to complete for batch {batch_id}")
    return False

class AsyncSession(Protocol):
    async def __aenter__(self) -> 'AsyncSession': ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...
    async def run(self, query: str, parameters: Optional[Dict[str, Any]] = None): ...
    async def close(self) -> None: ...
    async def begin_transaction(self) -> 'AsyncTransaction': ...

class AsyncTransaction(Protocol):
    async def __aenter__(self) -> 'AsyncTransaction': ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...
    async def run(self, query: str, parameters: Optional[Dict[str, Any]] = None): ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...

try:
    from neo4j.time import DateTime as Neo4jDateTime
except ImportError:
    Neo4jDateTime = None

def convert_datetimes(obj):
    if isinstance(obj, dict):
        return {k: convert_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetimes(v) for v in obj]
    elif isinstance(obj, datetime) or (Neo4jDateTime and isinstance(obj, Neo4jDateTime)):
        return obj.isoformat()
    else:
        return obj
    
class MemoryGraph:
    def __init__(self, embedding_model=None):
        # Create Neo4j connections (sync and async)
     
        # Initialize async Neo4j connection
        
        # NEW: Property collection configuration
        self.qdrant_property_collection = None
        self.property_indexes_created = False  # Track if property indexes have been created
        try:
            self.async_neo_conn = AsyncNeo4jConnection(
                uri=NEO4J_URL,
                user="neo4j",
                pwd=NEO4J_SECRET,
                retries=5,
                delay=2
            )
            # Initialize with fallback mode false
            self.async_neo_conn.fallback_mode = False
        except Exception as e:
            logger.error(f"Failed to initialize AsyncNeo4jConnection: {e}")
            # Set to None so ensure_async_connection() can attempt to initialize later
            self.async_neo_conn = None
            # Create connection object in fallback mode
            self.async_neo_conn = AsyncNeo4jConnection(
                uri=NEO4J_URL,
                user="neo4j",
                pwd=NEO4J_SECRET,
                retries=5,
                delay=2
            )
            self.async_neo_conn.fallback_mode = True
            # Initialize fallback storage since Neo4j is unavailable
            self.fallback_storage = {}

        # Initialize Pinecone with error handling and modern configuration
        # Only initialize Pinecone in cloud edition (not needed in open source)
        features = get_features()
        pc = None
        self.pinecone_circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=300)
        self.pinecone_fallback_mode = True
        
        if features.is_cloud:
            try:
                # Use modern Pinecone initialization with connection pooling
                pc = Pinecone(
                    api_key=PINECONE_KEY, 
                    environment=PINECONE_ENV,
                    # Add connection configuration for better stability
                    pool_connections=10,
                    pool_maxsize=20,
                    max_retries=3,
                    timeout=30
                )
                # Initialize Pinecone circuit breaker
                self.pinecone_fallback_mode = False
                logger.info("Pinecone initialized with modern SDK configuration")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}")
                pc = None
                self.pinecone_fallback_mode = True
        else:
            logger.info("Pinecone skipped - open source edition (cloud-only feature)")
        # Initialize MongoClient with enhanced robustness and error handling
        # Try to reuse the shared MongoDB client from services.mongo_client to avoid creating multiple connection pools
        shared_db = get_mongo_db()
        if shared_db is not None:
            # Reuse the shared MongoDB client and database
            self.mongo_client = shared_db.client
            self.db = shared_db
            logger.info(f"Reusing shared MongoDB client - database: {self.db.name}, server: {self.mongo_client.address}")
        elif not env.get("MONGO_URI") and not env.get("DATABASE_URI"):
            logger.warning("MemoryGraph.__init__: Neither MONGO_URI nor DATABASE_URI set, MongoDB operations will be unavailable")
            self.mongo_client = None
            self.db = None
        else:
            # Fallback: Create new client only if shared client is not available
            try:
                # Enhanced connection settings for maximum robustness
                optimized_params = {
                    'serverSelectionTimeoutMS': 10000,   # 10 second server selection
                    'connectTimeoutMS': 10000,           # 10 second connection timeout
                    'socketTimeoutMS': 60000,            # 60 second socket timeout
                    'maxPoolSize': 100,                   # Increased connection pool size
                    'minPoolSize': 10,                    # Minimum connections
                    'maxIdleTimeMS': 300000,              # 5 minutes idle timeout
                    'waitQueueTimeoutMS': 10000,          # 10 second wait timeout
                    'retryWrites': False,                 # DocumentDB doesn't support this
                    'readPreference': 'primaryPreferred', # Prefer primary but allow secondary
                    'heartbeatFrequencyMS': 10000,        # 10 second heartbeat
                    'maxConnecting': 20,                  # Max concurrent connections
                    'compressors': 'zlib',                # Enable compression
                    'zlibCompressionLevel': 6,            # Compression level
                    'retryReads': True,                   # Enable retry reads
                    'directConnection': False,            # Use replica set discovery
                }
                
                logger.info("Initializing new MongoDB connection (shared client not available)...")
                self.mongo_client = MongoClient(env.get("MONGO_URI"), **optimized_params)
                
                # Extract database with error handling
                try:
                    self.db = self.mongo_client.get_default_database()
                    logger.info(f"MongoDB database name: {self.db.name}")
                    logger.info(f"MongoDB server info: {self.mongo_client.address}")
                except Exception as e:
                    logger.error(f"Failed to get default database: {e}")
                    self.db = None
                
                # Test the connection with basic health checks (synchronous)
                if self.db is not None:
                    try:
                        # Test connection with basic commands
                        server_info = self.db.command("ismaster")
                        logger.info(f"MongoDB connection successful - server: {server_info.get('me', 'unknown')}")
                        
                        # Test with ping command
                        ping_result = self.db.command("ping")
                        logger.info(f"MongoDB ping successful: {ping_result}")
                        
                        # List collections to verify we're connected to the right database
                        collections = self.db.list_collection_names()
                        logger.info(f"Available collections: {collections[:10]}...")  # Show first 10 collections
                        
                        # Check if _User collection exists and has data
                        if "_User" in collections:
                            user_count = self.db["_User"].count_documents({})
                            logger.info(f"_User collection has {user_count} documents")
                        else:
                            logger.warning("_User collection not found in this database!")
                            
                    except Exception as e:
                        logger.error(f"MongoDB connection test failed: {e}")
                        # Don't fail completely, but log the issue
                        logger.warning("MongoDB connection test failed, but client will be available for retry")
                        
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB client: {e}")
                self.mongo_client = None
                self.db = None
        
        # Initialize Pinecone indexes with error handling
        if pc:
            try:
                self.index = pc.Index(host=env.get("PINECONE_INDEXHOST_SENTENCE", "https://memory-dev-d1f89ad.svc.us-east-1-aws.pinecone.io"))
                self.bigbird_index_name = "memory-bigbird"
                self.bigbird_index = pc.Index(host=env.get("PINECONE_INDEXHOST_BIGBIRD", "https://memory-bigbird-d1f89ad.svc.apw5-4e34-81fa.pinecone.io"))
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone indexes: {e}")
                self.index = None
                self.bigbird_index = None
                self.pinecone_fallback_mode = True
        else:
            self.index = None
            self.bigbird_index = None
        #self.snowflake_index = pc.Index("memorysnowflake")
        # Replace 'SentenceTransformer' initialization with 'EmbeddingModel'
        # Use provided embedding model or create new one
        self.embedding_model = embedding_model or EmbeddingModel()
        # Initialize Qdrant client
        self.qdrant_client = None
        self.qdrant_collection = None
        self.qdrant_indexes_created = False  # Track if indexes have been created
        
        # Check if QDRANT_URL is set (required for Qdrant initialization)
        qdrant_url = env.get("QDRANT_URL")
        if qdrant_url:
            # Use runtime function to get URL (applies localhost override for open-source local testing)
            from services.url_utils import get_qdrant_url
            qdrant_url = get_qdrant_url()
            
            # Get features to check edition
            features = get_features()
            qdrant_api_key = env.get("QDRANT_API_KEY")
            
            # Cloud edition requires API key, open-source can work without it
            if features.is_cloud:
                # Cloud deployment - API key is required
                if not qdrant_api_key:
                    logger.error("QDRANT_API_KEY is required for cloud deployments but not set")
                    raise ValueError("QDRANT_API_KEY is required for cloud deployments")
                self.qdrant_client = AsyncQdrantClient(
                    url=qdrant_url, 
                    api_key=qdrant_api_key,
                    timeout=60.0  # Increased timeout for batch operations
                )
                logger.info("Qdrant client initialized with API key (cloud mode)")
            else:
                # Open-source edition - API key is optional (local Qdrant doesn't require it)
                if qdrant_api_key:
                    self.qdrant_client = AsyncQdrantClient(
                        url=qdrant_url, 
                        api_key=qdrant_api_key,
                        timeout=60.0  # Increased timeout for batch operations
                    )
                    logger.info(f"Qdrant client initialized with API key (open-source mode): {qdrant_url}")
                else:
                    self.qdrant_client = AsyncQdrantClient(
                        url=qdrant_url,
                        timeout=60.0  # Increased timeout for batch operations
                    )
                    logger.info(f"Qdrant client initialized without API key (open-source mode): {qdrant_url}")
            
            # Set collection name based on environment variables
            if env.get("QDRANT_COLLECTION_QWEN4B"):
                self.qdrant_collection = env.get("QDRANT_COLLECTION_QWEN4B")
                logger.info(f"Qdrant collection set to: {self.qdrant_collection}")
            elif env.get("QDRANT_COLLECTION_QWEN0pt6B"):
                self.qdrant_collection = env.get("QDRANT_COLLECTION_QWEN0pt6B")
                logger.info(f"Qdrant collection set to: {self.qdrant_collection}")
            else:
                logger.warning("No Qdrant collection environment variable found")
        else:
            logger.warning("QDRANT_URL not set - Qdrant client will not be initialized")
        
        # Initialize memory_items dictionary
        self.memory_items = {}
        
        self.fallback_storage = {}  # Simple in-memory fallback
        
        # Track MongoDB warmup status
        self.mongodb_warmed_up = False
                
        # Initialize MongoDB wrapper for safe operations
        self.mongo_wrapper = MongoDBConnectionWrapper(self.mongo_client, self.db)
    
    def reconnect_mongodb(self):
        """Reconnect MongoDB client if connection was closed
        
        Tries to reuse the shared MongoDB client from services.mongo_client first.
        Only creates a new client if the shared one is not available or not working.
        """
        if not env.get("MONGO_URI"):
            logger.warning("MONGO_URI not set, cannot reconnect MongoDB")
            return False
            
        try:
            logger.info("Attempting to reconnect MongoDB...")
            
            # Try to reuse the shared MongoDB client first (preferred approach)
            shared_db = get_mongo_db()
            if shared_db is not None:
                # Reuse the shared client and database
                self.mongo_client = shared_db.client
                self.db = shared_db
                
                # Update wrapper with shared connection
                self.mongo_wrapper.mongo_client = self.mongo_client
                self.mongo_wrapper.db = self.db
                self.mongo_wrapper.connection_healthy = True
                self.mongo_wrapper.fallback_mode = False
                
                # Test the connection
                try:
                    self.db.command("ping")
                    logger.info("MongoDB reconnection successful using shared client")
                    return True
                except Exception as ping_err:
                    logger.warning(f"Shared MongoDB client ping failed: {ping_err}, will try creating new client")
                    # Fall through to create new client if shared one is not working
            
            # Fallback: Create new client only if shared client is not available or not working
            logger.info("Creating new MongoDB client (shared client not available or not working)...")
            # Close existing client if it exists (and it's not the shared one)
            if self.mongo_client and (not shared_db or self.mongo_client != shared_db.client):
                try:
                    self.mongo_client.close()
                except Exception:
                    pass
            
            # Enhanced connection settings for maximum robustness
            optimized_params = {
                'serverSelectionTimeoutMS': 10000,
                'connectTimeoutMS': 10000,
                'socketTimeoutMS': 60000,
                'maxPoolSize': 100,           # Default pool size (matching fallback in __init__)
                'minPoolSize': 10,            # Default minimum
                'maxIdleTimeMS': 300000,      # 5 minutes idle timeout
                'waitQueueTimeoutMS': 10000,
                'retryWrites': False,
                'readPreference': 'primaryPreferred',
                'heartbeatFrequencyMS': 10000,
                'maxConnecting': 20,          # Default max concurrent connections
                'compressors': 'zlib',
                'zlibCompressionLevel': 6,
                'retryReads': True,
                'directConnection': False,
            }
            
            self.mongo_client = MongoClient(env.get("MONGO_URI"), **optimized_params)
            self.db = self.mongo_client.get_default_database()
            
            # Update wrapper with new connection
            self.mongo_wrapper.mongo_client = self.mongo_client
            self.mongo_wrapper.db = self.db
            self.mongo_wrapper.connection_healthy = True
            self.mongo_wrapper.fallback_mode = False
            
            # Test the connection
            self.db.command("ping")
            logger.info("MongoDB reconnection successful with new client")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reconnect MongoDB: {e}")
            self.mongo_client = None
            self.db = None
            if self.mongo_wrapper:
                self.mongo_wrapper.mongo_client = None
                self.mongo_wrapper.db = None
                self.mongo_wrapper.connection_healthy = False
            return False
    
    async def warm_mongodb_connection(self):
        """Warm up MongoDB connection and critical authentication operations to eliminate cold start penalty"""
        if self.mongo_client is None or self.db is None:
            logger.warning("MongoDB client not available - skipping warmup")
            return
        
        # Skip if already warmed to avoid redundant operations
        if getattr(self, 'mongodb_warmed', False):
            logger.debug("MongoDB connection already warmed, skipping warmup")
            return
        
        try:
            logger.info("🔥 Warming up MongoDB connection...")
            start_time = time.time()
            
            # Warm up operations in order of importance (removed unauthorized operations)
            warmup_operations = [
                ("ping", lambda: self.db.command('ping')),
                ("user_count", lambda: self.db['_User'].count_documents({})),
                ("sample_user", lambda: self.db['_User'].find_one({}, {'_id': 1})),
                ("api_key_query", lambda: self.db['_User'].find_one({'userAPIkey': {'$type': 'string'}}, {'_id': 1}))
            ]
            
            # Add authentication-specific warmup operations that actually happen during API requests
            auth_warmup_operations = [
                ("api_key_lookup_with_hint", lambda: self._warmup_api_key_lookup()),
                ("user_metadata_lookup", lambda: self._warmup_user_metadata_lookup()),
                ("workspace_follower_lookup", lambda: self._warmup_workspace_follower_lookup()),
                ("parse_pointer_resolution", lambda: self._warmup_parse_pointer_resolution())
            ]
            
            # Combine all warmup operations
            all_operations = warmup_operations + auth_warmup_operations
            
            for op_name, operation in all_operations:
                try:
                    op_start = time.time()
                    await asyncio.get_event_loop().run_in_executor(None, operation)
                    op_time = (time.time() - op_start) * 1000
                    logger.info(f"  ✅ {op_name}: {op_time:.2f}ms")
                except Exception as e:
                    logger.warning(f"  ⚠️  {op_name} failed: {e}")
            
            total_time = (time.time() - start_time) * 1000
            logger.info(f"🔥 MongoDB warmup completed in {total_time:.2f}ms")
            self.mongodb_warmed = True
            
        except Exception as e:
            logger.error(f"❌ MongoDB warmup failed: {e}")
            self.mongodb_warmed = False

    def _warmup_api_key_lookup(self):
        """Warm up the exact API key lookup operation that happens during authentication"""
        try:
            from pymongo import ReadPreference
            collection = self.db["_User"].with_options(read_preference=ReadPreference.PRIMARY_PREFERRED)
            
            # Simulate the exact query pattern used in _verify_api_key_mongo
            sample_query = {"userAPIkey": {"$type": "string"}}
            projection = {"_id": 1, "username": 1, "email": 1, "isQwenRoute": 1, "_p_isSelectedWorkspaceFollower": 1}
            
            # Try with hint first (like in the real code)
            try:
                result = collection.find(sample_query, projection).hint([("userAPIkey", 1)]).limit(1)
                list(result)  # Force execution
            except Exception:
                # Fallback without hint
                result = collection.find_one(sample_query, projection)
                
        except Exception as e:
            logger.debug(f"API key lookup warmup failed: {e}")

    def _warmup_user_metadata_lookup(self):
        """Warm up user metadata lookup operations"""
        try:
            # Sample user metadata query patterns
            self.db["_User"].find_one({"_id": {"$type": "string"}}, {"isQwenRoute": 1, "_p_isSelectedWorkspaceFollower": 1})
        except Exception as e:
            logger.debug(f"User metadata lookup warmup failed: {e}")

    def _warmup_workspace_follower_lookup(self):
        """Warm up workspace follower lookup operations"""
        try:
            # Warm up workspace_follower collection queries
            self.db["workspace_follower"].find_one({}, {"_p_workspace": 1})
        except Exception as e:
            logger.debug(f"Workspace follower lookup warmup failed: {e}")

    def _warmup_parse_pointer_resolution(self):
        """Warm up Parse pointer resolution patterns"""
        try:
            # Warm up pointer resolution patterns
            self.db["_User"].find_one({"_id": {"$type": "string"}}, {"_p_isSelectedWorkspaceFollower": 1})
        except Exception as e:
            logger.debug(f"Parse pointer resolution warmup failed: {e}")
    
    async def keep_mongodb_warm(self):
        """Enhanced MongoDB keep-warm task with circuit breaker and recovery"""
        if self.mongo_client is None or self.db is None:
            logger.warning("MongoDB client not available for keep-warm task")
            return
        
        logger.info("🔥 Starting enhanced MongoDB keep-warm task...")
        
        while True:
            try:
                # Check if we're in fallback mode and attempt recovery
                if self.mongo_wrapper.fallback_mode:
                    logger.info("MongoDB in fallback mode, attempting recovery...")
                    recovery_success = await self.mongo_wrapper.attempt_recovery()
                    if not recovery_success:
                        await asyncio.sleep(30)  # Wait 30 seconds before next attempt
                        continue
                
                # Wait 30 seconds between pings (reduced from 5 minutes)
                await asyncio.sleep(30)
                
                start_time = time.time()
                
                # Perform a lightweight ping operation with timeout
                await asyncio.wait_for(
                    asyncio.to_thread(self.db.command, "ping"),
                    timeout=10.0
                )
                
                ping_time = (time.time() - start_time) * 1000
                
                # Log performance metrics
                if ping_time > 1000:  # Log if ping takes more than 1 second
                    logger.warning(f"🔥 MongoDB keep-warm ping slow: {ping_time:.2f}ms")
                else:
                    logger.debug(f"🔥 MongoDB keep-warm ping: {ping_time:.2f}ms")
                
                # Update health status
                self.mongo_wrapper.connection_healthy = True
                self.mongo_wrapper.last_health_check = time.time()
                
            except asyncio.CancelledError:
                logger.info("🔥 MongoDB keep-warm task cancelled")
                break
            except asyncio.TimeoutError:
                logger.warning("🔥 MongoDB keep-warm ping timed out")
                await self.mongo_wrapper.circuit_breaker.record_failure()
                await asyncio.sleep(10)  # Shorter sleep on timeout
            except Exception as e:
                error_str = str(e).lower()
                # Check if connection is closed and attempt reconnection
                if "closed" in error_str or "after close" in error_str or "not connected" in error_str:
                    logger.warning(f"🔥 MongoDB connection closed, attempting reconnect: {e}")
                    try:
                        reconnected = await asyncio.to_thread(self.reconnect_mongodb)
                        if reconnected:
                            logger.info("🔥 MongoDB reconnection successful via keep-warm task")
                            continue  # Retry ping immediately after reconnection
                        else:
                            logger.error("🔥 MongoDB reconnection failed")
                            await self.mongo_wrapper.circuit_breaker.record_failure()
                    except Exception as reconnect_err:
                        logger.error(f"🔥 Error during reconnection attempt: {reconnect_err}")
                        await self.mongo_wrapper.circuit_breaker.record_failure()
                else:
                    logger.warning(f"🔥 MongoDB keep-warm ping failed: {e}")
                    await self.mongo_wrapper.circuit_breaker.record_failure()
                # Continue trying - don't break the loop
                await asyncio.sleep(10)  # Shorter sleep on error

    async def create_optimized_qdrant_collection(self, collection_name: str, vector_size: int = 2560):
        """
        Create a Qdrant collection with optimized settings for speed.
        
        Based on Qdrant optimization guide:
        - Use quantization for faster search
        - Configure on_disk settings for memory efficiency
        - Use optimized index parameters
        """
        try:
            # Check if collection already exists
            collections = await self.qdrant_client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name in existing_collections:
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Create collection with optimized settings using the correct API structure
            from qdrant_client import models
            
            await self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                    # Enable quantization for faster search
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True
                        )
                    ),
                    # Optimize for speed
                    on_disk=False,  # Keep vectors in RAM for faster access
                    hnsw_config=models.HnswConfigDiff(
                        m=16,  # Number of connections per layer (default: 16)
                        ef_construct=100,  # Size of the dynamic candidate list (default: 100)
                        full_scan_threshold=10000,  # Threshold for full scan
                        max_indexing_threads=4  # Number of threads for indexing
                    )
                )
            )
            
            logger.info(f"Created optimized Qdrant collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating optimized Qdrant collection {collection_name}: {e}")
            return False

    async def optimize_qdrant_collection(self, collection_name: str = None):
        """
        Optimize an existing Qdrant collection for better performance.
        
        This method can be called to optimize collections that were created
        without the optimized settings.
        """
        try:
            if not collection_name:
                collection_name = self.qdrant_collection
                
            if not collection_name:
                logger.error("No collection name provided for optimization")
                return False
                
            logger.info(f"Optimizing Qdrant collection: {collection_name}")
            
            # Skip getting collection info to avoid validation errors with incompatible fields
            # Just update the collection directly with the optimization settings
            from qdrant_client import models
            
            await self.qdrant_client.update_collection(
                collection_name=collection_name,
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=2,  # Reduce segment count for faster search
                    memmap_threshold=10000     # Keep more data in memory
                )
            )
            
            logger.info(f"Successfully optimized Qdrant collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing Qdrant collection {collection_name}: {e}")
            # Don't fail the entire application if optimization fails
            # The search will still work, just not optimally
            logger.info("Continuing without collection optimization - search will still work")
            return False

    async def init_qdrant(self) -> None:
        try:
            # Only initialize if we have a client and haven't created indexes yet
            if self.qdrant_client and not self.qdrant_indexes_created:
                logger.info("Initializing Qdrant indexes")
                
                # Ensure collection name is set
                if not self.qdrant_collection:
                    if env.get("QDRANT_COLLECTION_QWEN4B"):
                        self.qdrant_collection = env.get("QDRANT_COLLECTION_QWEN4B")
                    elif env.get("QDRANT_COLLECTION_QWEN0pt6B"):
                        self.qdrant_collection = env.get("QDRANT_COLLECTION_QWEN0pt6B")
                    else:
                        logger.error("No Qdrant collection environment variable found")
                        return
                
                logger.info(f"Using Qdrant collection: {self.qdrant_collection}")

                # Create payload indexes only once
                if env.get("QDRANT_COLLECTION_QWEN4B"):
                    # Optimize index fields - only index the most frequently used fields
                    index_fields = [
                        "user_id", "workspace_id", "organization_id", "namespace_id", 
                        "user_read_access", "workspace_read_access", "role_read_access",
                        "organization_read_access", "namespace_read_access",
                        "user_write_access", "workspace_write_access", "role_write_access",
                        "organization_write_access", "namespace_write_access",
                        "metadata.createdAt", "metadata.user_id", "metadata.user_read_access", 
                        "metadata.user_write_access", "metadata.workspace_read_access", 
                        "metadata.workspace_write_access", "metadata.role_read_access",
                        "metadata.role_write_access", "metadata.organization_read_access", 
                        "metadata.organization_write_access", "metadata.namespace_read_access", 
                        "metadata.namespace_write_access", "metadata.external_user_id", 
                        "metadata.external_user_read_access", "metadata.external_user_write_access",
                        "metadata.relatedGoals", "metadata.relatedUseCases", "metadata.relatedSteps",
                        "metadata.goalClassificationScores", "metadata.useCaseClassificationScores", 
                        "metadata.stepClassificationScores", "topics", "hierarchical_structures", 
                        "location", "emoji_tags", "emotion_tags", "conversationId", "chunk_id"
                    ]
                    
                    for field in index_fields:
                        try:
                            await self.qdrant_client.create_payload_index(
                                collection_name=self.qdrant_collection, 
                                field_name=field, 
                                field_schema="keyword"
                            )
                        except Exception as e:
                            # Index might already exist, which is fine
                            logger.debug(f"Index for {field} may already exist: {e}")
                    
                    self.qdrant_indexes_created = True
                    logger.info(f"Created optimized payload indexes for Qdrant collection {self.qdrant_collection}")

                elif env.get("QDRANT_COLLECTION_QWEN0pt6B"):
                    # Create basic payload indexes only once - include ALL ACL fields (read AND write)
                    basic_index_fields = [
                        "user_id", 
                        "user_read_access", "user_write_access",
                        "workspace_read_access", "workspace_write_access",
                        "role_read_access", "role_write_access",
                        "organization_read_access", "organization_write_access",
                        "namespace_read_access", "namespace_write_access",
                        "external_user_read_access", "external_user_write_access"
                    ]
                    
                    for field in basic_index_fields:
                        try:
                            await self.qdrant_client.create_payload_index(
                                collection_name=self.qdrant_collection, 
                                field_name=field, 
                                field_schema="keyword"
                            )
                        except Exception as e:
                            # Index might already exist, which is fine
                            logger.debug(f"Index for {field} may already exist: {e}")
                    
                    self.qdrant_indexes_created = True
                    logger.info(f"Created payload indexes for Qdrant collection {self.qdrant_collection}")
            else:
                logger.info("Qdrant indexes already created or client not available")
            
            # NEW: Initialize property collection
            await self._init_property_collection()
        except Exception as e:
            logger.error(f"Error ensuring payload indexes exist for Qdrant collection {self.qdrant_collection}: {e}")
    
    async def _init_property_collection(self) -> None:
        """Initialize separate Qdrant collection for property indexes"""
        try:
            # Only initialize if we have a client and haven't created indexes yet
            if self.qdrant_client and not self.property_indexes_created:
                logger.info("Initializing property collection indexes")
                
                # Property collection name from environment or default
                property_collection_name = env.get("QDRANT_PROPERTY_COLLECTION", "neo4j_properties")
                
                # Create property collection if it doesn't exist
                success = await self.create_optimized_qdrant_collection(
                    collection_name=property_collection_name,
                    vector_size=384  # Use sentence-bert embeddings (all-MiniLM-L6-v2) for simpler, smaller vectors
                )
                
                if success:
                    self.qdrant_property_collection = property_collection_name
                    
                    # Create property-specific payload indexes
                    await self._create_property_collection_indexes(property_collection_name)
                    
                    self.property_indexes_created = True
                    logger.info(f"Property collection initialized: {property_collection_name}")
                else:
                    logger.warning("Failed to create property collection")
            else:
                logger.info("Property collection indexes already created or client not available")
                
        except Exception as e:
            logger.error(f"Error initializing property collection: {e}")

    async def _create_property_collection_indexes(self, collection_name: str) -> None:
        """Create optimized indexes for property collection"""
        try:
            # Property-specific indexes for fast filtering
            property_indexes = [
                # Core property metadata
                "is_property_index",
                "node_type", 
                "property_name",
                "property_type",
                "property_key",  # Composite node_type.property_name
                "schema_id",
                "schema_name",
                "schema_type",
                "is_system_schema",
                
                # Property value metadata for analytics
                "property_value_length",
                "property_value_word_count",
                "property_value_lowercase",
                
                # ACL indexes (inherit from main collection)
                "user_id", "workspace_id", "organization_id", "namespace_id",
                "user_read_access", "workspace_read_access", "role_read_access",
                "organization_read_access", "namespace_read_access",
                "user_write_access", "workspace_write_access", "role_write_access",
                "organization_write_access", "namespace_write_access", "external_user_read_access",
                
                # Source tracking and context
                "source_node_id",
                "source_memory_id",
                "source_memory_type",
                
                # Indexing metadata
                "indexed_at",
                "indexing_version"
            ]
            
            for field_name in property_indexes:
                try:
                    await self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema="keyword"
                    )
                    # Only log successful creation, not skipped ones
                    logger.debug(f"Created property collection index: {field_name}")
                except Exception as e:
                    # Index might already exist - only log if it's not a "already exists" error
                    if "already exists" not in str(e).lower() and "conflict" not in str(e).lower():
                        logger.debug(f"Property index {field_name} creation issue: {e}")
                    # Otherwise silently skip (index already exists)
                    
        except Exception as e:
            logger.warning(f"Error creating property collection indexes: {e}")

    

    async def _index_node_properties_with_sync_results(self, neo4j_results: List[Dict], memory_item: Dict, 
                                                     workspace_id: Optional[str], user_schema: Optional[Any],
                                                     common_metadata: Optional[Dict[str, Any]] = None):
        """
        Steps 4-5: Async property indexing with Neo4j synchronization results.
        Uses was_created flag to determine create vs update operations.
        
        Args:
            common_metadata: Metadata dict with organization_id/namespace_id correctly extracted from both
                           metadata and customMetadata. This is the same metadata used for node creation.
        """
        try:
            from services.property_indexing_service import PropertyIndexingService
            
            logger.info(f"🔄 SYNC STEPS 4-5: Starting property indexing with {len(neo4j_results)} Neo4j results")
            
            # Get the structured output schema that was used for LLM generation
            structured_output_schema = getattr(self, '_last_memory_graph_schema', None)
            logger.info(f"🔄 SYNC STEPS 4-5: structured_output_schema available={structured_output_schema is not None}")
            
            # Get enhanced schema cache for property indexing
            enhanced_schema_cache = getattr(self, '_cached_schema', None)
            logger.info(f"🔄 SYNC STEPS 4-5: enhanced_schema_cache from _cached_schema={enhanced_schema_cache is not None}")
            
            # Create schema cache with structured output schema for property cross-referencing
            if structured_output_schema:
                logger.info("🔄 SYNC STEPS 4-5: Creating schema cache with structured output schema for property indexing")
                enhanced_schema_cache = {
                    'structured_output_schema': structured_output_schema,
                    'user_schema': user_schema,
                    'schema_id': getattr(user_schema, 'id', 'system_schema') if user_schema else 'system_schema',
                    'schema_name': getattr(user_schema, 'name', 'System Schema') if user_schema else 'System Schema'
                }
                logger.info(f"🔄 SYNC STEPS 4-5: Created enhanced_schema_cache with structured output schema")
            elif not enhanced_schema_cache:
                logger.info(f"🔄 SYNC STEPS 4-5: No schema cache available for property indexing (user_schema={user_schema is not None})")
                return
            
            property_service = PropertyIndexingService(self)
            await property_service.index_node_properties_with_sync(
                neo4j_results=neo4j_results,
                source_memory=memory_item,
                cached_schema=enhanced_schema_cache,
                common_metadata=common_metadata  # Pass the correctly extracted metadata
            )
        except Exception as e:
            logger.error(f"🔄 SYNC STEPS 4-5: Property indexing with sync failed: {e}")
            import traceback
            logger.error(f"🔄 SYNC STEPS 4-5: Property indexing traceback: {traceback.format_exc()}")
            # Don't fail the main request - this is background processing

    async def _index_node_properties_async(self, created_nodes: List[Dict], memory_item: Dict, 
                                         workspace_id: Optional[str], user_schema: Optional[Any]):
        """Legacy async background task to index node properties in separate collection"""
        try:
            from services.property_indexing_service import PropertyIndexingService
            
            logger.info(f"🔍 PROPERTY INDEXING DEBUG: user_schema={user_schema is not None}, created_nodes={len(created_nodes)}")
            
            # Get the structured output schema that was used for LLM generation
            structured_output_schema = getattr(self, '_last_memory_graph_schema', None)
            logger.info(f"🔍 PROPERTY INDEXING DEBUG: structured_output_schema available={structured_output_schema is not None}")
            
            # Get enhanced schema cache for property indexing
            enhanced_schema_cache = getattr(self, '_cached_schema', None)
            logger.info(f"🔍 PROPERTY INDEXING DEBUG: enhanced_schema_cache from _cached_schema={enhanced_schema_cache is not None}")
            
            # Create schema cache with structured output schema for property cross-referencing
            if structured_output_schema:
                logger.info("🔍 Creating schema cache with structured output schema for property indexing")
                enhanced_schema_cache = {
                    'structured_output_schema': structured_output_schema,
                    'user_schema': user_schema,
                    'schema_id': getattr(user_schema, 'id', 'system_schema') if user_schema else 'system_schema',
                    'schema_name': getattr(user_schema, 'name', 'System Schema') if user_schema else 'System Schema'
                }
                logger.info(f"🔍 PROPERTY INDEXING DEBUG: Created enhanced_schema_cache with structured output schema")
            elif not enhanced_schema_cache:
                logger.info(f"🔍 No schema cache available for property indexing (user_schema={user_schema is not None})")
                return
            
            property_service = PropertyIndexingService(self)
            await property_service.index_node_properties(
                nodes=created_nodes,
                source_memory=memory_item,
                cached_schema=enhanced_schema_cache
            )
        except Exception as e:
            logger.error(f"Property indexing failed: {e}")
            import traceback
            logger.error(f"Property indexing traceback: {traceback.format_exc()}")
            # Don't fail the main request - this is background processing

    def _build_indexable_properties_from_schema(self, user_schema):
        """Build indexable properties map from a single user schema"""
        indexable_properties = {}
        
        logger.info(f"🔍 INDEXABLE PROPS DEBUG: user_schema type={type(user_schema)}, has_node_types={hasattr(user_schema, 'node_types') if user_schema else False}")
        
        if not user_schema or not hasattr(user_schema, 'node_types'):
            logger.info(f"🔍 INDEXABLE PROPS DEBUG: Early return - no schema or node_types")
            return indexable_properties
        
        logger.info(f"🔍 INDEXABLE PROPS DEBUG: Processing {len(user_schema.node_types)} node types: {list(user_schema.node_types.keys())}")
            
        for node_name, node_type in user_schema.node_types.items():
            logger.info(f"🔍 INDEXABLE PROPS DEBUG: Node {node_name}: type={type(node_type)}, has_properties={hasattr(node_type, 'properties')}")
            
            if hasattr(node_type, 'properties'):
                logger.info(f"🔍 INDEXABLE PROPS DEBUG: Node {node_name} has {len(node_type.properties)} properties: {list(node_type.properties.keys())}")
                
                for prop_name, prop_def in node_type.properties.items():
                    logger.info(f"🔍 INDEXABLE PROPS DEBUG: Property {node_name}.{prop_name}: type={type(prop_def)}")
                    logger.info(f"🔍 INDEXABLE PROPS DEBUG: Property {node_name}.{prop_name}: has_required={hasattr(prop_def, 'required')}, has_type={hasattr(prop_def, 'type')}, has_enum_values={hasattr(prop_def, 'enum_values')}")
                    
                    if hasattr(prop_def, 'required'):
                        logger.info(f"🔍 INDEXABLE PROPS DEBUG: Property {node_name}.{prop_name}.required = {prop_def.required}")
                    if hasattr(prop_def, 'type'):
                        logger.info(f"🔍 INDEXABLE PROPS DEBUG: Property {node_name}.{prop_name}.type = {prop_def.type}")
                    if hasattr(prop_def, 'enum_values'):
                        logger.info(f"🔍 INDEXABLE PROPS DEBUG: Property {node_name}.{prop_name}.enum_values = {prop_def.enum_values}")
                    
                    # Check if property should be indexed (required string properties without enums)
                    if (hasattr(prop_def, 'required') and prop_def.required and
                        hasattr(prop_def, 'type') and 
                        str(prop_def.type).lower() in ['string', 'str'] and
                        not (hasattr(prop_def, 'enum_values') and prop_def.enum_values)):
                        
                        key = f"{node_name}.{prop_name}"
                        indexable_properties[key] = {
                            'node_type': node_name,
                            'property_name': prop_name,
                            'property_type': str(prop_def.type),
                            'schema_id': getattr(user_schema, 'id', 'unknown'),
                            'schema_name': getattr(user_schema, 'name', 'unknown'),
                            'is_system_schema': False
                        }
                        logger.info(f"🔍 INDEXABLE PROPS DEBUG: ✅ Added indexable property: {key}")
                    else:
                        logger.info(f"🔍 INDEXABLE PROPS DEBUG: ❌ Skipped property: {node_name}.{prop_name}")
            else:
                logger.info(f"🔍 INDEXABLE PROPS DEBUG: Node {node_name} has no properties attribute")
        
        logger.info(f"🔍 Built indexable properties for schema: {len(indexable_properties)} properties")
        logger.info(f"🔍 INDEXABLE PROPS FINAL: {indexable_properties}")
        return indexable_properties

    async def ensure_custom_metadata_indexes(self, custom_metadata: dict):
        """
        Dynamically create Qdrant indexes for custom metadata fields.
        This ensures that custom metadata fields can be filtered on during search.
        """
        if not self.qdrant_client or not custom_metadata:
            return
            
        try:
            for key, value in custom_metadata.items():
                if value is not None:
                    field_name = str(key)
                    
                    # Determine field type for indexing
                    field_type = None
                    if isinstance(value, str):
                        field_type = "keyword"
                    elif isinstance(value, bool):
                        # Check bool before int/float since bool is a subclass of int in Python
                        field_type = "bool"
                    elif isinstance(value, (int, float)):
                        field_type = "integer" if isinstance(value, int) else "float"
                    elif isinstance(value, list):
                        # For lists, we need to determine the type of the first non-None element
                        # Lists in Qdrant are typically indexed as keyword arrays
                        field_type = "keyword"
                    # Add more types as needed (e.g., geo, range)
                    
                    if field_type:
                        try:
                            # Create index for the custom metadata field
                            await self.qdrant_client.create_payload_index(
                                collection_name=self.qdrant_collection,
                                field_name=field_name,
                                field_schema=field_type
                            )
                            logger.info(f"Created Qdrant payload index for custom metadata field: {field_name} (Type: {field_type})")
                        except Exception as e:
                            # Index might already exist, which is fine
                            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                                logger.debug(f"Index for custom metadata field '{field_name}' already exists")
                            else:
                                logger.warning(f"Could not create Qdrant payload index for {field_name}: {e}")
                    else:
                        logger.warning(f"Unsupported type for custom metadata field '{field_name}': {type(value)}. Skipping index creation.")
        except Exception as e:
            logger.error(f"Error creating custom metadata indexes: {e}")
    
    async def ensure_async_connection(self):
        async with self.async_neo_conn._lock:
            """Ensure Neo4j connection is established with fallback handling"""
            try:
                if not self.async_neo_conn:
                    logger.info("Initializing Neo4j async connection")
                    self.async_neo_conn = AsyncNeo4jConnection(
                        uri=NEO4J_URL,
                        user="neo4j",
                        pwd=NEO4J_SECRET
                    )
                
                driver = await self.async_neo_conn.get_driver()
                logger.info(f"Retrieved Neo4j driver: {driver}")
                if not driver:
                    logger.info("Connection Lost - Initializing Neo4j async connection")
                    self.async_neo_conn = AsyncNeo4jConnection(
                        uri=NEO4J_URL,
                        user="neo4j",
                        pwd=NEO4J_SECRET
                    )
                    driver = await self.async_neo_conn.get_driver()
                if driver:
                    try:
                        async with driver.session() as session:
                            result = await session.run("RETURN 1")
                            await result.consume()
                            logger.info("Neo4j connection test successful")
                            # Reset fallback mode if connection is successful
                            if self.async_neo_conn.fallback_mode:
                                logger.info("Resetting fallback mode as connection is restored")
                                self.async_neo_conn.fallback_mode = False
                    except ssl.SSLError as ssl_err:
                        logger.error(f"SSL Error connecting to Neo4j: {ssl_err}")
                        logger.error(f"SSL Configuration: CERT_FILE={env.get('SSL_CERT_FILE')}")
                        self.async_neo_conn.fallback_mode = True
                    except Exception as e:
                        logger.error(f"Error testing Neo4j connection: {e}")
                        self.async_neo_conn.fallback_mode = True
                else:
                    logger.warning("Neo4j connection unavailable, using fallback mode")
                    
                    self.async_neo_conn.fallback_mode = True
                    
                    
            except Exception as e:
                logger.error(f"Error ensuring async connection: {str(e)}")
                if self.async_neo_conn:
                    self.async_neo_conn.fallback_mode = True
                else:
                    # Create connection object in fallback mode if it doesn't exist
                    # We still need to create the connection object even in fallback mode
                    # so that we have a consistent interface and can track the fallback state
                    self.async_neo_conn = AsyncNeo4jConnection(
                        uri=NEO4J_URL,
                        user="neo4j", 
                        pwd=NEO4J_SECRET
                    )
                    self.async_neo_conn.fallback_mode = True

    async def get_safe_session(self, neo_session=None):
        """
        Return a healthy AsyncSession:
        - validates connection if we create the session
        - never leaves an unclosed session on error
        """
        if neo_session:
            return neo_session  # caller owns cleanup
        await self.ensure_async_connection()
        stack = contextlib.AsyncExitStack()
        try:
            session = await stack.enter_async_context(self.async_neo_conn.get_session())
            return session
        except Exception:
            await stack.aclose()
            raise    
        
    async def store_with_fallback(self, memory_id: str, data: dict):
        """Store data with fallback mechanism"""
        if self.async_neo_conn.fallback_mode:
            self.fallback_storage[memory_id] = data
            logger.info(f"Stored memory {memory_id} in fallback storage")
            return True
            
        try:
            # Attempt to store in Neo4j
            # return await self._store_in_neo4j(memory_id, data)
            return True
        except Exception as e:
            logger.error(f"Failed to store in Neo4j: {str(e)}")
            self.fallback_storage[memory_id] = data
            return True

    async def recover_fallback_data(self):
        """Attempt to recover data from fallback storage"""
        if not self.fallback_storage:
            return

        logger.info(f"Attempting to recover {len(self.fallback_storage)} items from fallback storage")
        
        try:
            if not self.async_neo_conn.fallback_mode:
                for memory_id, data in self.fallback_storage.items():
                    try:
                        # await self._store_in_neo4j(memory_id, data)
                        # del self.fallback_storage[memory_id]
                        logger.info(f"Recovered memory {memory_id} to Neo4j")
                    except Exception as e:
                        logger.error(f"Failed to recover memory {memory_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Recovery process failed: {str(e)}")

    async def periodic_health_check(self):
        """Periodic health check and recovery attempt"""
        while True:
            try:
                is_healthy = await self.check_neo4j_health()
                if is_healthy and self.fallback_storage:
                   # await self.recover_fallback_data()
                   pass
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                await asyncio.sleep(60)  # Shorter interval if check fails
    
    
    def __iter__(self):
        """Make the MemoryGraph iterable over its memory items"""
        return iter(self.memory_items.values())

    def __len__(self):
        """Return the number of memory items"""
        return len(self.memory_items)

    def get_all_memory_items(self):
        """Get all memory items"""
        return list(self.memory_items.values())
    



    def cosine_similarity(self, vec1, vec2):
        return 1 - spatial.distance.cosine(vec1, vec2)
    
    async def _safe_pinecone_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """
        Safely execute a Pinecone operation with circuit breaker and error handling.
        Only available in cloud edition - returns None immediately in open-source.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: The Pinecone operation function to execute
            *args, **kwargs: Arguments to pass to the operation
            
        Returns:
            Result of the operation or None if failed or in open-source edition
        """
        # Check if we're in cloud edition - Pinecone is cloud-only
        import os
        papr_edition = os.getenv("PAPR_EDITION", "").lower()
        is_opensource = papr_edition == "opensource"
        
        if is_opensource:
            # Open-source edition doesn't use Pinecone - skip operation
            logger.debug(f"Pinecone operation '{operation_name}' skipped - open-source edition (cloud-only feature)")
            return None
        
        # Cloud edition: Use safe Pinecone operation with circuit breaker
        # Check circuit breaker first
        if not await self.pinecone_circuit_breaker.can_try():
            logger.warning(f"Pinecone circuit breaker is open, skipping {operation_name}")
            self.pinecone_fallback_mode = True
            return None
            
        # Check if Pinecone is available
        if not self.index or self.pinecone_fallback_mode:
            logger.warning(f"Pinecone is in fallback mode, skipping {operation_name}")
            return None
            
        try:
            # Execute the operation with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(operation_func, *args, **kwargs),
                timeout=30.0  # 30 second timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"Pinecone {operation_name} timed out after 30 seconds")
            await self.pinecone_circuit_breaker.record_failure()
            self.pinecone_fallback_mode = True
            return None
        except urllib3.exceptions.ProtocolError as e:
            logger.error(f"Pinecone protocol error in {operation_name}: {e}")
            await self.pinecone_circuit_breaker.record_failure()
            self.pinecone_fallback_mode = True
            return None
        except Exception as e:
            logger.error(f"Pinecone error in {operation_name}: {e}")
            await self.pinecone_circuit_breaker.record_failure()
            return None
    
    async def check_and_retrieve_from_pinecone(
        self, 
        session_token: str, 
        embedding: List[float], 
        user_id: str, 
        neo_session: AsyncSession,
        new_metadata: Dict[str, Any],
        user_workspace_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Asynchronously checks for a similar embedding in Pinecone and updates metadata if a match is found.

        Args:
            session_token (str): The session token for authentication.
            embedding (List[float]): The embedding vector to query.
            user_id (str): The user ID associated with the embedding.
            neo_session (AsyncSession): The Neo4j session to use for the query.
            new_metadata (Dict[str, Any]): The metadata to update in Pinecone if a match is found. Must have internal user IDs resolved for ACL fields.

        Returns:
            Optional[str]: The ID of the matched vector if found, else None.
        """
        # Ensure the embedding is a list of floats
        if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
            logger.error(f"Invalid embedding format: {embedding}")
            return None

        # Log the embedding to verify its format
        logger.info(f"Embedding to be queried (first 5 elements): {embedding[:5]}")
        logger.info(f"Embedding length: {len(embedding)}")
        logger.info(f"new_metadata: {new_metadata}")

        # Ensure the embedding is a list of floats
        embedding = [float(x) for x in embedding]

        # Create a new User instance using the user_id
        user_instance = User.get(user_id)
        
        # Get user roles and workspace IDs
        if user_workspace_ids is None:
            # If workspace IDs weren't provided, fetch both roles and workspaces
            user_roles, user_workspace_ids = await asyncio.gather(
                user_instance.get_roles_async(),
                User.get_workspaces_for_user_async(user_id)
            )
        else:
            # If workspace IDs were provided, only fetch roles
            user_roles = await user_instance.get_roles_async()
        
        logger.debug(f'user_roles {user_roles}')
        logger.debug(f'user_workspace_ids {user_workspace_ids}')

        # Get organization and namespace info (if available)
        user_organization_id = getattr(user_instance, 'organization_id', None)
        user_namespace_id = getattr(user_instance, 'namespace_id', None)
        user_organization_access = getattr(user_instance, 'organization_read_access', [])
        user_namespace_access = getattr(user_instance, 'namespace_read_access', [])

        # Setup the ACL filter with organization and namespace support
        acl_conditions = [
            {"user_id": {"$eq": str(user_id)}},
            {"user_read_access": {"$in": [str(user_id)]}},
            {"workspace_read_access": {"$in": [str(workspace_id) for workspace_id in user_workspace_ids]}},
            {"role_read_access": {"$in": user_roles}},
        ]
        
        # Add organization conditions if available (only access arrays, not direct IDs)
        if user_organization_access:
            acl_conditions.append({"organization_read_access": {"$in": user_organization_access}})
            
        # Add namespace conditions if available (only access arrays, not direct IDs)
        if user_namespace_access:
            acl_conditions.append({"namespace_read_access": {"$in": user_namespace_access}})
        
        acl_filter = {"$or": acl_conditions}

        # Perform a similarity search in Pinecone using safe operation
        query_result = await self._safe_pinecone_operation(
            "query",
            self.index.query,
            vector=embedding,
            top_k=1,
            include_values=False,
            include_metadata=True,
            filter=acl_filter
        )
        
        if query_result is None:
            logger.warning("Pinecone query failed, returning None")
        # Perform a similarity search in Pinecone using safe operation
        query_result = await self._safe_pinecone_operation(
            "query",
            self.index.query,
            vector=embedding,
            top_k=1,
            include_values=False,
            include_metadata=True,
            filter=acl_filter
        )
        
        if query_result is None:
            logger.warning("Pinecone query failed, returning None")
            return None
            
        logger.info(f"Query result: {query_result}")
            
        logger.info(f"Query result: {query_result}")

        if not query_result or not query_result.matches:
            logger.info("No matches found in Pinecone")
            return None

        score = query_result['matches'][0]['score']
        matched_id = query_result['matches'][0]['id']
        logger.info(f"Found match with ID: {matched_id} and similarity score: {score}")
        
        if score > 0.97:
            logger.info(f"Score {score} > 0.97, using existing vector")
            # At this point, new_metadata must already have user_read_access/user_write_access resolved to internal user ObjectIds.
            # Do not attempt to resolve external user IDs here; this must be done at the API/top layer.
            pinecone_metadata = MemoryGraph.pinecone_compatible_metadata(new_metadata)
            parse_metadata = new_metadata  # original, with customMetadata
            await self.update_memory_metadata(session_token, matched_id, neo_session, pinecone_metadata, parse_metadata, api_key=api_key)
            return matched_id
        else:
            logger.info(f"Score {score} <= 0.97, will create new vector")
            return None

    async def update_pinecone_metadata(self, vector_id: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Asynchronously updates the metadata of an existing vector in Pinecone.
        Only available in cloud edition - returns True immediately in open-source.

        Args:
            vector_id (str): The ID of the vector to update.
            new_metadata (Dict[str, Any]): The new metadata to set.

        Returns:
            bool: True if update was successful or in open-source, False otherwise.
        """
        # Check if Pinecone is available (cloud-only feature)
        if not hasattr(self, 'index') or self.index is None:
            # Open-source edition doesn't have Pinecone - skip update
            logger.debug(f"Pinecone not available (open-source edition), skipping metadata update for vector ID: {vector_id}")
            return True  # Return True to indicate "handled" (no error)
        
        # Cloud edition: Use safe Pinecone operation for metadata update
        result = await self._safe_pinecone_operation(
            "update_metadata",
            self.index.update,
            id=vector_id,
            set_metadata=new_metadata
        )
        
        if result is None:
            logger.warning(f"Pinecone metadata update failed for vector ID: {vector_id}")
            return False
            
        logger.info(f"Successfully updated metadata for vector ID: {vector_id}")
        return True
    
    async def update_neo_metadata(self, memory_id: str, new_metadata: Dict[str, Any], neo_session: AsyncSession) -> bool:
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, skipping metadata update")
            # Store in fallback storage instead
            self.fallback_storage[memory_id] = new_metadata
            return True  # Return True to indicate "handled" (even if in fallback)

             
        try:
            # Flatten the metadata dictionary if necessary
            flat_metadata = self.validate_metadata(new_metadata)

            # Get base ID (remove chunk suffix if present)
            base_memory_id = memory_id.split('_')[0]

            # Main update query - modified to return all properties explicitly
            update_query = """
                MATCH (m:Memory)
                WHERE m.id = $memory_id 
                OR m.id = $base_memory_id
                OR $memory_id IN coalesce(m.memoryChunkIds, [])
                OR $base_memory_id IN coalesce(m.memoryChunkIds, [])
                SET m += $flat_metadata
                RETURN properties(m) as props, m.id as id, labels(m) as labels
            """

            result = await neo_session.run(
                update_query,
                memory_id=str(memory_id),
                base_memory_id=str(base_memory_id),
                flat_metadata=flat_metadata
            )
            updated_record = await result.single()
            # Consume the result to avoid "result has been consumed" errors
            # Note: single() already consumes the result, but we consume explicitly for safety
            try:
                await result.consume()
            except Exception:
                pass  # Result may already be consumed by single()

            if updated_record and updated_record.get("props"):
                props = updated_record.get("props", {})
                node_id = props.get("id") or updated_record.get("id")  # Try both locations
                labels = updated_record.get("labels", [])
                logger.info(f"Successfully updated node with ID: {node_id}")
                logger.info(f"Node properties: {props}")
                logger.info(f"Node labels: {labels}")
                return True
            # If we get here, try to verify if the node exists
            verify_query = """
            MATCH (m:Memory)
            WHERE m.id = $base_memory_id
            RETURN properties(m) as props
            """
            verify_result = await neo_session.run(verify_query, base_memory_id=str(base_memory_id))
            verify_record = await verify_result.single()
            # Consume the verify result to avoid "result has been consumed" errors (open-source only)
            import os
            papr_edition = os.getenv("PAPR_EDITION", "").lower()
            is_opensource = papr_edition == "opensource"
            if is_opensource:
                try:
                    await verify_result.consume()
                except Exception:
                    pass  # Result may already be consumed by single()

            if verify_record and verify_record.get("props"):
                logger.info(f"Node exists but update didn't return data. Node properties: {verify_record.get('props')}")
                return True

            logger.error(f"No memory item found with ID: {memory_id} or base ID: {base_memory_id} in Neo4j")
            logger.error(f"Debug: memory_id={memory_id}, base_memory_id={base_memory_id}")
            return False

        except Exception as e:
            logger.error(f"Error updating metadata in Neo4j for memory ID {memory_id}: {e}")
            # Store in fallback storage if Neo4j update fails
            self.async_neo_conn.fallback_mode = True
            self.fallback_storage[memory_id] = new_metadata
            return True


    async def update_parse_metadata(
        self, 
        session_token: str, 
        memory_id: str, 
        new_metadata: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> bool:
        """
        Asynchronously updates the metadata of an existing memory item in Parse Server.

        Args:
            session_token (str): The session token for authentication.
            memory_id (str): The unique ID of the memory item to update.
            new_metadata (Dict[str, Any]): The new metadata to set.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        try:
            # Extract memoryChunkIds from new_metadata if it exists
            memory_chunk_ids = new_metadata.get('memoryChunkIds', [])
            if isinstance(memory_chunk_ids, str):
                try:
                    memory_chunk_ids = json.loads(memory_chunk_ids)
                except json.JSONDecodeError:
                    memory_chunk_ids = [id.strip() for id in memory_chunk_ids.strip('[]').split(',') if id.strip()]
            
            # Ensure memory_chunk_ids is a list of strings
            if isinstance(memory_chunk_ids, list):
                memory_chunk_ids = [str(id).strip() for id in memory_chunk_ids if id]
            else:
                memory_chunk_ids = []
                
            # Retrieve the existing memory item
            memory_item: ParseStoredMemory = await retrieve_memory_item_parse(
                session_token=session_token, 
                memory_item_id=memory_id,
                memory_chunk_ids=memory_chunk_ids,
                api_key=api_key
            )
            
            if not memory_item:
                logger.error(f"No memory item found with ID: {memory_id.split('_')[0]} in Parse Server")
                return False

            # Convert custom ACL to Parse ACL
            parse_acl = convert_acl(new_metadata)

            # Pre-process list fields
            emoji_tags = convert_comma_string_to_list(
                new_metadata.get('emojiTags') or 
                new_metadata.get('emoji_tags') or 
                new_metadata.get('emoji tags')
            )
            emotion_tags = convert_comma_string_to_list(
                new_metadata.get('emotionTags') or 
                new_metadata.get('emotion_tags') or 
                new_metadata.get('emotion tags')
            )
            topics = convert_comma_string_to_list(new_metadata.get('topics'))
            steps = convert_comma_string_to_list(new_metadata.get('steps'))

            # Create update data using MemoryParseServerUpdate
            parse_memory = MemoryParseServerUpdate(
                ACL=parse_acl,
                sourceType=new_metadata.get('sourceType'),
                context=new_metadata.get('context'),
                title=new_metadata.get('title'),
                location=new_metadata.get('location'),
                emojiTags=emoji_tags,
                emotionTags=emotion_tags,
                hierarchicalStructures=(
                    new_metadata.get('hierarchicalStructures') or 
                    new_metadata.get('hierarchical_structures') or 
                    new_metadata.get('hierarchical structures')
                ),
                sourceUrl=new_metadata.get('sourceUrl'),
                conversationId=new_metadata.get('conversationId'),
                topics=topics,
                steps=steps,
                current_step=new_metadata.get('current_step'),
                memoryChunkIds=memory_chunk_ids,
                customMetadata=new_metadata.get('customMetadata'),
                external_user_read_access=new_metadata.get('external_user_read_access') or None,
                external_user_write_access=new_metadata.get('external_user_write_access') or None,
                user_read_access=new_metadata.get('user_read_access') or None,
                user_write_access=new_metadata.get('user_write_access') or None,
                workspace_read_access=new_metadata.get('workspace_read_access') or None,
                workspace_write_access=new_metadata.get('workspace_write_access') or None,
                role_read_access=new_metadata.get('role_read_access') or None,
                role_write_access=new_metadata.get('role_write_access') or None,
                namespace_read_access=new_metadata.get('namespace_read_access') or None,
                namespace_write_access=new_metadata.get('namespace_write_access') or None,
                organization_read_access=new_metadata.get('organization_read_access') or None,
                organization_write_access=new_metadata.get('organization_write_access') or None
            )


            # Convert to dict and remove None values
            update_data = parse_memory.model_dump(
                exclude_none=True,
                exclude={'createdAt', 'updatedAt'}
            )

            logger.info(f"Data to update in Parse Server: {update_data} with objectId: {memory_item.objectId}")

            # Perform the update
            success = await update_memory_item_parse(
                session_token=session_token,
                object_id=memory_item.objectId,
                update_data=update_data,
                api_key=api_key
            )

            if success:
                logger.info(f"Successfully updated memory item with ID: {memory_id.split('_')[0]}")
                return True
            else:
                logger.error(f"Failed to update memory item with ID: {memory_id.split('_')[0]}")
                return False

        except Exception as e:
            logger.error(f"Error updating memory item with ID {memory_id.split('_')[0]}: {e}")
            return False

    async def update_memory_metadata(
        self,
        session_token: str,
        vector_id: str,
        neo_session: AsyncSession,
        pinecone_metadata: Dict[str, Any],
        parse_metadata: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> bool:
        """
        Asynchronously updates metadata across Pinecone, Neo4j, and Parse Server for a given memory item.

        Args:
            session_token (str): The session token for authentication.
            vector_id (str): The ID of the vector in Pinecone to update.
            neo_session (AsyncSession): The Neo4j session to use for the query.
            pinecone_metadata (Dict[str, Any]): The new metadata to set.
            parse_metadata (Dict[str, Any]): The new metadata to set.

        Returns:
            bool: True if all updates were successful, False otherwise.
        """
        try:
            # Build list of update tasks (only include available services)
            # Check if we're in open-source edition - skip Pinecone updates
            import os
            papr_edition = os.getenv("PAPR_EDITION", "").lower()
            is_opensource = papr_edition == "opensource"
            
            update_tasks = [
                self.update_qdrant_metadata(vector_id, pinecone_metadata),
                self.update_neo_metadata(vector_id, pinecone_metadata, neo_session),
                self.update_parse_metadata(session_token, vector_id, parse_metadata, api_key=api_key)
            ]
            
            # Only add Pinecone update if not open-source (Pinecone is cloud-only)
            if not is_opensource:
                # Cloud edition: Add Pinecone update
                update_tasks.append(self.update_pinecone_metadata(vector_id, pinecone_metadata))
            
            await asyncio.gather(*update_tasks)
            logger.info(f"Successfully updated metadata across systems for vector ID: {vector_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating metadata across systems for vector ID {vector_id}: {e}")
            return False

    async def add_memory_item_without_relationships(
        self, 
        session_token: str, 
        memory_item: MemoryItem, 
        neo_session: AsyncSession,
        user_id: str,
        user_workspace_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        developer_user_object_id: Optional[str] = None
    ) -> Tuple[List[ParseStoredMemory], List[MemoryItem]]:
        """
        Asynchronously indexes a memory item (simple index).
        This method performs basic indexing of the memory item, including embedding generation and storage.
        It is designed for fast ingestion and prediction of Q→A pairs, with the ability to look back and connect related memories later.
        Accepts an optional neo_session for robust session management.
        """
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, using fallback storage for memory item")
            self.fallback_storage[memory_item.id] = memory_item
            return [], [memory_item]

        async def _simple_index_memory_item(session):
            # Initialize with proper types
            added_item_properties_list: List[ParseStoredMemory] = []
            memory_item_list: List[MemoryItem] = []        
            
            total_start_time = time.time()
            timings = {
                'metadata_prep': 0,
                'embedding_generation': 0,
                'similarity_check': 0,
                'pinecone_store': 0,
                'parse_server_store': 0,
                'neo4j_store': 0,
                'total': 0
            }

            # Metadata preparation timing
            metadata_start = time.time()
            if 'createdAt' not in memory_item.metadata:
                memory_item.metadata['createdAt'] = datetime.now(timezone.utc).isoformat()

            developer_user_id = memory_item.metadata.get('external_user_id')
            developer_user_pointer = None
            if developer_user_id:
                memory_item.metadata['external_user_id'] = developer_user_id
                # Optionally, set read/write access lists
                memory_item.metadata['external_user_read_access'] = memory_item.metadata.get('external_user_read_access', [])
                memory_item.metadata['external_user_write_access'] = memory_item.metadata.get('external_user_write_access', [])
                developer_user_pointer = DeveloperUserPointer(
                    objectId=developer_user_id,
                    className='DeveloperUser',
                    external_id=developer_user_id
                )
            
            # Use centralized static methods for metadata cleaning
            logger.info(f"memory_item.metadata inside add_memory_item_without_relationships: {memory_item.metadata}")
            memory_item.metadata = MemoryGraph.sanitize_metadata(memory_item.metadata)
            # Use deepcopy to avoid mutating the original metadata
            original_metadata = copy.deepcopy(memory_item.metadata)
            logger.info(f"original_metadata inside add_memory_item_without_relationships: {original_metadata}")
            pinecone_metadata = MemoryGraph.pinecone_compatible_metadata(original_metadata)
            # Ensure individual memories have isGroupedMemories: false (not grouped)
            pinecone_metadata['isGroupedMemories'] = False
            logger.info(f'pinecone_metadata inside add_memory_item_without_relationships: {pinecone_metadata}')
            timings['metadata_prep'] = time.time() - metadata_start
            logger.info(f"Metadata preparation took {timings['metadata_prep']:.4f} seconds")

            # Extract user and workspace information
            # Use the user_id parameter passed to the function (which should be the end_user_id)
            # instead of extracting from metadata
            workspace_id = memory_item.metadata.get('workspace_id')
            logger.info(f'user_id parameter: {user_id}')
            logger.info(f'workspace_id: {workspace_id}')

            # Generate embeddings for all chunks with retry mechanism
            embedding_start = time.time()
            try:
                logger.info(f'Generating embeddings for memory item: {memory_item.content}')
                #embeddings, chunks = await self.embedding_model.get_sentence_embedding(
                #    memory_item.content, max_retries=5, retry_delay=1.0
                #)
                embeddings, chunks = await self.embedding_model.get_qwen_embedding_4b(
                    memory_item.content, max_retries=5, retry_delay=1.0
                )
                
                logger.info(f'Generated {len(embeddings)} embeddings from {len(chunks)} chunks')
            except Exception as e:
                logger.error(f"Failed to generate embeddings after all retries: {e}")
                return ([], [])

            # Ensure embeddings are lists of floats
            embeddings = [list(map(float, embedding)) for embedding in embeddings]
            timings['embedding_generation'] = time.time() - embedding_start
            logger.info(f"Embedding generation took {timings['embedding_generation']:.4f} seconds")

            memoryChunkIds = []  # Track all chunk IDs
            logger.info(f"Initializing empty memoryChunkIds list")
            existing_main_id = None  # Track the first existing ID we find

            # Check existing chunks concurrently
            # Limit concurrent Pinecone queries to avoid rate limiting
            semaphore = asyncio.Semaphore(10)  

            async def limited_check(emb):
                async with semaphore:
                    await self.ensure_async_connection()
                    async with self.async_neo_conn.get_session() as task_session:
                        return await self.check_and_retrieve_from_qdrant(
                            session_token, emb, user_id, task_session, original_metadata, user_workspace_ids, api_key=api_key
                        )

            similarity_start = time.time()
            existing_memory_ids = await asyncio.gather(*[
                limited_check(emb) for emb in embeddings
            ])
            timings['similarity_check'] = time.time() - similarity_start
            logger.info(f"Similarity checks took {timings['similarity_check']:.4f} seconds")
            logger.info(f'existing_memory_ids: {existing_memory_ids}')
            
            # Process existing chunks and prepare new chunks
            new_chunks = []
            for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
                existing_id = existing_memory_ids[idx]
                chunk_metadata = pinecone_metadata.copy()
                
                # Extract custom metadata fields and add them to chunk_metadata for Qdrant indexing
                custom_metadata = None
                if hasattr(memory_item, 'customMetadata'):
                    custom_metadata = memory_item.customMetadata
                elif isinstance(memory_item.metadata, dict) and 'customMetadata' in memory_item.metadata:
                    custom_metadata = memory_item.metadata['customMetadata']
                
                # Add custom metadata fields directly to chunk_metadata for Qdrant filtering
                if custom_metadata and isinstance(custom_metadata, dict):
                    for key, value in custom_metadata.items():
                        if value is not None:  # Only add non-None values
                            chunk_metadata[key] = value
                            logger.info(f"Added custom metadata field '{key}' = '{value}' to Qdrant payload")
                    
                    # Ensure Qdrant indexes exist for custom metadata fields
                    await self.ensure_custom_metadata_indexes(custom_metadata)
                
                chunk_metadata.update({
                    'chunk_index': idx,
                    'total_chunks': len(chunks),
                    'content': chunk  # Add the actual chunk content for search and debugging
                })

                if existing_id:
                    logger.info(f'Memory chunk with similar embedding exists in Pinecone with id {existing_id}')
                    # Get the base ID (remove chunk suffix if present)
                    base_id = existing_id.split('_')[0]
                    logger.info(f'Using base ID: {base_id} from chunk ID: {existing_id}')
                    
                    # First check Neo4j for the base ID
                    logger.info(f'Checking Neo4j for memory item with ID: {base_id} before calling get_memory_item')
                    neo4j_item = await self.get_memory_item(base_id, session)
                    if neo4j_item:
                        logger.info(f'Found existing memory in Neo4j with ID: {base_id}')
                        
                        # Since it exists in Neo4j, check Parse Server
                        existing_parse_item = await retrieve_memory_item_by_qdrant_id(
                            session_token, 
                            base_id,  # Use base_id, not chunk ID
                            api_key=api_key
                        )
                        
                        if existing_parse_item:
                            logger.info(f'Found existing memory in Parse Server with ID: {base_id}')
                            logger.info(f'existing_parse_item: {existing_parse_item}')
                            logger.info(f'existing_parse_item memoryChunkIds: {existing_parse_item.get("memoryChunkIds")}')
                            logger.info(f'existing_parse_item metadata: {existing_parse_item.get("metadata")}')
                            # Extract memoryChunkIds from existing item
                            existing_chunk_ids = []
                            if isinstance(existing_parse_item.get('metadata'), str):
                                try:
                                    metadata = json.loads(existing_parse_item['metadata'])
                                    existing_chunk_ids = metadata.get('memoryChunkIds', [])
                                except json.JSONDecodeError:
                                    logger.warning(f"Could not parse metadata JSON for existing memory {base_id}")
                            elif isinstance(existing_parse_item.get('metadata'), dict):
                                existing_chunk_ids = existing_parse_item['metadata'].get('memoryChunkIds', [])
                            
                            # Also check direct memoryChunkIds field
                            if not existing_chunk_ids and existing_parse_item.get('memoryChunkIds'):
                                existing_chunk_ids = existing_parse_item['memoryChunkIds']
                            
                            # For legacy memories, use the memoryId as the chunk ID if no chunks found
                            if not existing_chunk_ids:
                                existing_chunk_ids = [base_id]  # Use base_id as the single chunk ID
                                logger.info(f"Legacy memory found - using memoryId as chunk ID: {existing_chunk_ids}")
                                
                            logger.info(f"Found existing chunk IDs: {existing_chunk_ids}")
                            # Create a properly formatted ParseStoredMemory object

                            custom_metadata = None
                            if hasattr(memory_item, 'customMetadata'):
                                custom_metadata = memory_item.customMetadata
                            elif isinstance(memory_item.metadata, dict) and 'customMetadata' in memory_item.metadata:
                                custom_metadata = memory_item.metadata['customMetadata']
                            
                            # Ensure Qdrant indexes exist for custom metadata fields even for existing memories
                            if custom_metadata and isinstance(custom_metadata, dict):
                                await self.ensure_custom_metadata_indexes(custom_metadata)
                            developer_user_data_existing = existing_parse_item.get('developerUser')
                            developer_user_pointer_existing = None
                            if developer_user_data_existing:
                                developer_user_pointer_existing = DeveloperUserPointer(
                                    objectId=developer_user_data_existing.get('objectId'),
                                    className='DeveloperUser',
                                    external_id=developer_user_data_existing.get('external_id'),
                                    metadata=developer_user_data_existing.get('metadata'),
                                    email=developer_user_data_existing.get('email')
                                )

                            parse_stored_memory = ParseStoredMemory(
                                objectId=existing_parse_item.get('objectId'),
                                createdAt=existing_parse_item.get('createdAt'),
                                updatedAt=existing_parse_item.get('updatedAt'),
                                ACL=existing_parse_item.get('ACL', {}),
                                content=existing_parse_item.get('content', ''),
                                type=existing_parse_item.get('type', 'text'),
                                metadata=existing_parse_item.get('metadata', '{}'),
                                customMetadata=custom_metadata,
                                memoryId=base_id,  # Use base_id
                                memoryChunkIds=existing_chunk_ids,  
                                user=ParseUserPointer(
                                    objectId=existing_parse_item.get('user', {}).get('objectId'),
                                    className='_User'
                                ),
                                developerUser=developer_user_pointer_existing,
                                external_user_read_access=existing_parse_item.get('external_user_read_access', []),
                                external_user_write_access=existing_parse_item.get('external_user_write_access', []),
                                user_read_access=existing_parse_item.get('user_read_access', []),
                                user_write_access=existing_parse_item.get('user_write_access', []),
                                workspace_read_access=existing_parse_item.get('workspace_read_access', []),
                                workspace_write_access=existing_parse_item.get('workspace_write_access', []),
                                role_read_access=existing_parse_item.get('role_read_access', []),
                            )
                            
                            # Update memory item with Parse Server data
                            memory_item.objectId = existing_parse_item.get('objectId')
                            memory_item.createdAt = existing_parse_item.get('createdAt')
                            memory_item.id = base_id  # Ensure consistent ID
                            
                            added_item_properties_list.append(parse_stored_memory)
                            memory_item_list.append(memory_item)
                            logger.info(f'Returning existing memory item from all systems with ID: {base_id}')
                            return added_item_properties_list, memory_item_list
                    
                    # If we didn't find it in Neo4j/Parse, continue with new item creation
                    logger.info(f'Memory {base_id} not found in Neo4j/Parse, will create new')
                    memory_item.id = base_id  # Use the base_id for consistency
                
                # Only add new chunk if we didn't find existing item in Neo4j/Parse
                chunk_id = str(memory_item.id) + f"_{idx}"  # Ensure clean string format from the start
                chunk_metadata['chunk_id'] = chunk_id
                new_chunks.append((chunk_id, embedding, chunk_metadata))
                memoryChunkIds.append(chunk_id)  # Add clean string ID
                logger.info(f"Added chunk_id to memoryChunkIds: {chunk_id}")
                logger.info(f"Current memoryChunkIds list: {memoryChunkIds}")

            # Batch upsert new chunks to Pinecone
            # Batch upsert new chunks to Qdrant
            if new_chunks:
                qdrant_start = time.time()
                try:
                    # Use add_qdrant_point for each chunk with proper chunk_id
                    successful_upserts = 0
                    for chunk_id, embedding, metadata in new_chunks:
                        try:
                            result = await self.add_qdrant_point(chunk_id, embedding, metadata)
                            # Debug logging for result structure
                            logger.debug(f"Qdrant result type: {type(result)}, result: {result}")
                            
                            if result and hasattr(result, 'status'):
                                logger.debug(f"Result status type: {type(result.status)}, value: {result.status}")
                                # Check for both enum and string values for backward compatibility
                                if (result.status == UpdateStatus.COMPLETED or 
                                    (hasattr(result.status, 'value') and result.status.value == 'completed') or
                                    str(result.status) == 'completed'):
                                    successful_upserts += 1
                                    logger.info(f"Successfully added Qdrant point with chunk_id: {chunk_id}")
                                else:
                                    logger.warning(f"Qdrant upsert returned unexpected status for chunk_id: {chunk_id}")
                                    logger.warning(f"Expected: {UpdateStatus.COMPLETED} or 'completed', Got: {result.status}")
                                    logger.warning(f"Qdrant result details: operation_id={getattr(result, 'operation_id', 'N/A')} status={result.status}")
                            elif result is None:
                                logger.warning(f"Qdrant upsert failed for chunk_id: {chunk_id} - returned None")
                            else:
                                logger.warning(f"Qdrant upsert returned result without status for chunk_id: {chunk_id}")
                                logger.warning(f"Qdrant result details: {result}")
                        except Exception as e:
                            logger.error(f"Failed to add Qdrant point with chunk_id {chunk_id}: {e}")
                            raise e

                    logger.info(f'Successfully upserted {successful_upserts}/{len(new_chunks)} new chunks to Qdrant')
                    
                except Exception as e:
                    logger.error(f"Failed to upsert or verify chunks in Qdrant: {e}", exc_info=True)
                    return ([], [])
                
                timings['qdrant_store'] = time.time() - qdrant_start
            logger.info(f"Qdrant storage took {timings['qdrant_store']:.4f} seconds")
            # Store single node in Neo4j with all chunk IDs
            neo4j_start = time.time()
            try:
                # Use Pinecone-compatible metadata for Neo4j
                neo4j_metadata = MemoryGraph.pinecone_compatible_metadata(memory_item.metadata)
                memory_item.metadata['memoryChunkIds'] = memoryChunkIds
                memory_item.memoryChunkIds = memoryChunkIds

                logger.info(f"Storing memory item with chunk IDs: {memoryChunkIds}")
                logger.info(f"Preparing to store in Neo4j with memoryChunkIds: {memoryChunkIds}")
                if developer_user_id:
                    memory_item.metadata['external_user_id'] = developer_user_id
                    # Optionally, set read/write access lists
                    memory_item.metadata['external_user_read_access'] = memory_item.metadata.get('external_user_read_access', [])
                    memory_item.metadata['external_user_write_access'] = memory_item.metadata.get('external_user_write_access', [])

                # Preserve organization_id and namespace_id for multi-tenant scoping
                # These are now top-level fields in MemoryMetadata, not in customMetadata
                org_id = memory_item.metadata.get('organization_id')
                ns_id = memory_item.metadata.get('namespace_id')
                if org_id:
                    neo4j_metadata['organization_id'] = org_id
                if ns_id:
                    neo4j_metadata['namespace_id'] = ns_id

                # Preserve access control lists for multi-tenant scoping
                for access_field in ['organization_read_access', 'organization_write_access',
                                     'namespace_read_access', 'namespace_write_access']:
                    access_list = memory_item.metadata.get(access_field)
                    if access_list:
                        neo4j_metadata[access_field] = access_list

                # --- FIX: temporarily set metadata to neo4j_metadata ---
                old_metadata = memory_item.metadata
                memory_item.metadata = neo4j_metadata
                neo4j_result = await self.add_memory_item_to_neo4j(memory_item, neo_session, memoryChunkIds)
                # --- Restore original metadata for Parse Server ---
                memory_item.metadata = old_metadata
                
                # 🔒 CRITICAL: If an existing Memory node was found in Neo4j, use its ID
                # This ensures relationships point to the correct node
                if neo4j_result and 'id' in neo4j_result:
                    actual_neo4j_id = neo4j_result['id']
                    original_id = str(memory_item.id)
                    if actual_neo4j_id != original_id:
                        logger.info(f"📍 Using existing Neo4j Memory node ID: {actual_neo4j_id} (original ID was {original_id})")
                        memory_item.id = actual_neo4j_id
                    else:
                        logger.debug(f"📍 Created new Neo4j Memory node with ID: {actual_neo4j_id}")
                timings['neo4j_store'] = time.time() - neo4j_start
                logger.info(f"Neo4j storage took {timings['neo4j_store']:.4f} seconds")
            except Exception as e:
                logger.error(f"Failed to add memory node to Neo4j: {e}", exc_info=True)

            # Store in Parse Server with all chunk IDs
            parse_start = time.time()
            try:
                # Ensure memoryChunkIds are in metadata before storing
                memory_item.metadata['memoryChunkIds'] = memoryChunkIds
                memory_item.memoryChunkIds = memoryChunkIds  
                logger.info(f"Storing memory item with chunk IDs: {memoryChunkIds}")
                logger.info(f"Preparing to store in Parse Server with memoryChunkIds: {memoryChunkIds}")
                logger.info(f"Memory item metadata before Parse Server storage: {memory_item.metadata}")
                
                added_item_properties = await store_memory_item(user_id, session_token, memory_item, api_key=api_key, developer_user_object_id=developer_user_object_id)
                logger.info(f"Calling store_memory_item with memory_item metadata: {memory_item.metadata}")
                
                if not added_item_properties:
                    logger.error("Failed to store memory item in Parse server")
                else:
                    # Add detailed logging of the response before conversion
                    logger.info(f"Raw added_item_properties before conversion: {added_item_properties}")
                    logger.info(f"Raw added_item_properties type: {type(added_item_properties)}")
                    if isinstance(added_item_properties, dict):
                        logger.info(f"Dict keys: {added_item_properties.keys()}")
                        logger.info(f"Metadata in response: {added_item_properties.get('metadata', 'No metadata')}")
                    
                    # Check if added_item_properties is a Pydantic model
                    if hasattr(added_item_properties, 'model_copy'):
                        logger.info("Processing Pydantic model response")
                        updated_properties = added_item_properties.model_copy(
                            update={
                                'memoryId': str(memory_item.id),
                                'memoryChunkIds': memoryChunkIds  # Ensure chunk IDs are included
                            }
                        )
                        logger.info(f"Updated Pydantic properties: {updated_properties}")
                        added_item_properties_list.append(updated_properties)
                    else:
                        # Handle dictionary response
                        logger.info("Processing dictionary response")
                        updated_properties = {
                            **added_item_properties,
                            'memoryId': str(memory_item.id),
                            'memoryChunkIds': memoryChunkIds  # Ensure chunk IDs are included
                        }
                        if 'customMetadata' in memory_item.metadata:
                            updated_properties['customMetadata'] = memory_item.metadata['customMetadata']
                        if 'customMetadata' in updated_properties and not (isinstance(updated_properties['customMetadata'], dict) or updated_properties['customMetadata'] is None):
                            updated_properties['customMetadata'] = None
                        logger.info(f"Final properties before ParseStoredMemory creation: {updated_properties}")
                        added_item_properties_list.append(ParseStoredMemory(**updated_properties))
                        
                    memory_item_list.append(memory_item)
                    logger.info(f'Added memory item with id {memory_item.id} and chunk IDs {memoryChunkIds} to Parse Server')
                    
                    # Update the memory_item with the Parse Server data
                    if added_item_properties_list:
                        parse_stored_memory = added_item_properties_list[-1]  # Get the latest one
                        # Update memory_item with Parse Server data
                        memory_item.objectId = parse_stored_memory.objectId
                        memory_item.createdAt = parse_stored_memory.createdAt
                        memory_item.memoryChunkIds = parse_stored_memory.memoryChunkIds
                        logger.info(f'Updated memory_item with Parse Server data - memoryChunkIds: {memory_item.memoryChunkIds}')
            except Exception as e:
                logger.error(f"Failed to store in Parse Server: {e}", exc_info=True)
                # Return empty lists instead of None to maintain return type consistency
                return ([], [])
                
            timings['parse_server_store'] = time.time() - parse_start
            logger.info(f"Parse Server storage took {timings['parse_server_store']:.4f} seconds")

            # Calculate total time
            timings['total'] = time.time() - total_start_time
            logger.info("Memory item processing timings:")
            for operation, duration in timings.items():
                logger.info(f"  {operation}: {duration:.4f} seconds")

            return added_item_properties_list, memory_item_list

        try:
            return await _simple_index_memory_item(neo_session)
        except Exception as session_error:
            logger.warning(f"Session error encountered: {session_error}, retrying with a new session.")
            try:
               return await _simple_index_memory_item(neo_session)
            except Exception as e:
                logger.error(f"Failed to get Neo4j session after retry: {e}")
                self.async_neo_conn.fallback_mode = True
                logger.warning("Switching to fallback storage for memory item")
                self.fallback_storage[memory_item.id] = memory_item
                return [], [memory_item]

    async def batch_add_memory_items_without_relationships(
        self,
        session_token: str,
        memory_items: List[MemoryItem],
        neo_session: AsyncSession,
        user_id: str,
        user_workspace_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        developer_user_object_id: Optional[str] = None
    ) -> Tuple[List[ParseStoredMemory], List[MemoryItem]]:
        """
        Batch version of add_memory_item_without_relationships.
        Processes multiple memories in a single operation for better performance.
        
        Args:
            session_token: Authentication token
            memory_items: List of MemoryItem objects to process
            neo_session: Neo4j async session
            user_id: User ID
            user_workspace_ids: Optional workspace IDs
            api_key: Optional API key
            developer_user_object_id: Optional developer user object ID
            
        Returns:
            Tuple of (List[ParseStoredMemory], List[MemoryItem])
        """
        if self.async_neo_conn.fallback_mode:
            logger.warning(f"Neo4j in fallback mode, using fallback storage for {len(memory_items)} memory items")
            for memory_item in memory_items:
                self.fallback_storage[memory_item.id] = memory_item
            return [], memory_items

        logger.info(f"📦 Batch processing {len(memory_items)} memories")
        total_start_time = time.time()
        
        try:
            # Phase 1: Prepare metadata for all memories
            logger.info("Phase 1: Preparing metadata")
            developer_user_pointer = None
            if developer_user_object_id:
                developer_user_pointer = DeveloperUserPointer(
                    objectId=developer_user_object_id,
                    className='DeveloperUser',
                    external_id=None
                )
            
            for memory_item in memory_items:
                if 'createdAt' not in memory_item.metadata:
                    memory_item.metadata['createdAt'] = datetime.now(timezone.utc).isoformat()
                
                # Sanitize and prepare metadata
                memory_item.metadata = MemoryGraph.sanitize_metadata(memory_item.metadata)
                memory_item.metadata['isGroupedMemories'] = False
            
            # Phase 2: Generate embeddings for all memories
            logger.info("Phase 2: Generating embeddings")
            embedding_start = time.time()
            
            all_chunks = []
            all_embeddings = []
            all_memoryChunkIds = []
            
            for memory_item in memory_items:
                try:
                    embeddings, chunks = await self.embedding_model.get_qwen_embedding_4b(
                        memory_item.content, max_retries=5, retry_delay=1.0
                    )
                    embeddings = [list(map(float, embedding)) for embedding in embeddings]
                    
                    # Generate chunk IDs
                    chunk_ids = [f"{memory_item.id}_{idx}" for idx in range(len(chunks))]
                    
                    all_chunks.extend(list(zip(chunk_ids, chunks, [memory_item] * len(chunks))))
                    all_embeddings.extend(list(zip(chunk_ids, embeddings)))
                    all_memoryChunkIds.append(chunk_ids)
                    
                    # Store chunk IDs in memory item
                    memory_item.memoryChunkIds = chunk_ids
                    memory_item.metadata['memoryChunkIds'] = chunk_ids
                    
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for memory {memory_item.id}: {e}")
                    all_memoryChunkIds.append([str(memory_item.id)])
                    memory_item.memoryChunkIds = [str(memory_item.id)]
            
            embedding_time = time.time() - embedding_start
            logger.info(f"✅ Generated {len(all_embeddings)} embeddings in {embedding_time:.2f}s")
            
            # Phase 3: Batch upsert to Qdrant
            logger.info("Phase 3: Upserting to Qdrant")
            qdrant_start = time.time()
            
            if all_embeddings:
                # Prepare Qdrant points
                qdrant_points = []
                for (chunk_id, embedding), (_, chunk, memory_item) in zip(all_embeddings, all_chunks):
                    original_metadata = copy.deepcopy(memory_item.metadata)
                    payload = MemoryGraph.pinecone_compatible_metadata(original_metadata)
                    payload['content'] = chunk
                    payload['chunk_id'] = chunk_id
                    payload['isGroupedMemories'] = False
                    
                    # Add custom metadata fields
                    custom_metadata = memory_item.metadata.get('customMetadata', {})
                    if custom_metadata and isinstance(custom_metadata, dict):
                        for key, value in custom_metadata.items():
                            if value is not None:
                                payload[key] = value
                        await self.ensure_custom_metadata_indexes(custom_metadata)
                    
                    qdrant_points.append((chunk_id, embedding, payload))
                
                # Batch upsert
                success = await self.batch_upsert_qdrant_points(qdrant_points)
                if not success:
                    logger.error("Qdrant batch upsert failed")
                    return [], []
            
            qdrant_time = time.time() - qdrant_start
            logger.info(f"✅ Qdrant batch upsert completed in {qdrant_time:.2f}s")
            
            # Phase 4: Batch create Neo4j nodes
            logger.info("Phase 4: Creating Neo4j nodes")
            neo4j_start = time.time()
            
            await self.batch_create_memory_nodes(
                memory_items,
                neo_session,
                all_memoryChunkIds
            )
            
            neo4j_time = time.time() - neo4j_start
            logger.info(f"✅ Neo4j batch creation completed in {neo4j_time:.2f}s")
            
            # Phase 5: Batch store in Parse Server
            logger.info("Phase 5: Storing in Parse Server")
            parse_start = time.time()
            
            stored_memories = await batch_store_memories_async(
                user_id=user_id,
                session_token=session_token,
                memory_items=memory_items,
                api_key=api_key,
                developer_user_object_id=developer_user_object_id
            )
            
            parse_time = time.time() - parse_start
            logger.info(f"✅ Parse Server batch store completed in {parse_time:.2f}s")
            
            # Update memory_items with Parse Server data
            for memory_item, stored_memory in zip(memory_items, stored_memories):
                memory_item.objectId = stored_memory.objectId
                memory_item.createdAt = stored_memory.createdAt
            
            total_time = time.time() - total_start_time
            logger.info(f"📊 Batch processing completed in {total_time:.2f}s:")
            logger.info(f"  - Embedding: {embedding_time:.2f}s")
            logger.info(f"  - Qdrant: {qdrant_time:.2f}s")
            logger.info(f"  - Neo4j: {neo4j_time:.2f}s")
            logger.info(f"  - Parse: {parse_time:.2f}s")
            
            return stored_memories, memory_items
            
        except Exception as e:
            logger.error(f"❌ Error in batch_add_memory_items_without_relationships: {e}", exc_info=True)
            # Fallback to individual processing
            logger.warning("Falling back to individual processing")
            all_stored = []
            all_items = []
            for memory_item in memory_items:
                try:
                    stored, items = await self.add_memory_item_without_relationships(
                        session_token, memory_item, neo_session, user_id,
                        user_workspace_ids, api_key, developer_user_object_id
                    )
                    all_stored.extend(stored)
                    all_items.extend(items)
                except Exception as item_error:
                    logger.error(f"Failed to process memory {memory_item.id}: {item_error}")
            return all_stored, all_items

    async def update_memory_item_with_relationships(
        self, 
        memory_item: 'MemoryItem',  # Use string literal for forward reference
        relationships_json: List[RelationshipItem], 
        workspace_id: Optional[str], 
        user_id: str,
        neo_session: Optional[AsyncSession] = None,
        legacy_route: bool = True
    ) -> Dict[str, Union[bool, List[LLMGraphRelationship], Optional[str]]]:
        """
        Creates relationships between memory items and returns relationship status.
        If neo_session is None, create a new session for this operation (background task pattern).
        If neo_session is provided, use it directly (sub-method pattern).
        """
        memory_item_ids: List[str] = []
        created_relationships: List[LLMGraphRelationship] = []
        success = True
        error = None

        # Extract tenant scoping from memory_item metadata
        metadata = (json.loads(memory_item.metadata)
                   if isinstance(memory_item.metadata, str)
                   else memory_item.metadata)
        organization_id = metadata.get('organization_id')
        namespace_id = metadata.get('namespace_id')

        # Session management logic
        if neo_session is None:
            await self.ensure_async_connection()
            async with self.async_neo_conn.get_session() as session:
                return await self.update_memory_item_with_relationships(
                    memory_item,
                    relationships_json,
                    workspace_id,
                    user_id,
                    neo_session=session,
                    legacy_route=legacy_route
                )

        # --- Existing logic below, using the provided neo_session ---
        try:
            # Check if 'context' key exists in memory_item
            # Only check context if it exists and is not empty in the memory_dict
            if memory_item.context and len(memory_item.context) > 0:
                memory_item_ids = await self.get_memory_item_ids_from_conversation_history(memory_item.context, user_id, legacy_route)
                logger.info(f'Got memory item ids from conversation history: {memory_item_ids}')
            else:
                logger.info("No context provided in memory_dict, skipping conversation history processing.")

                            # Iterate over the relationships in the JSON structure
            for relationship in relationships_json:
                    try:
                        # Handle both dictionary and RelationshipItem object formats
                        if isinstance(relationship, dict):
                            # Extract from dictionary format
                            related_item_id = relationship.get('related_item_id')
                            relation_type = relationship.get('relation_type')
                            relationship_type_enum = relationship.get('relationship_type')
                            related_item_type = relationship.get('related_item_type', 'Memory')
                            metadata = relationship.get('metadata', {})
                            
                            logger.info(f"Processing relationship as dictionary: {relationship}")
                        else:
                            # Extract from RelationshipItem object format
                            related_item_id = relationship.related_item_id
                            relation_type = relationship.relation_type
                            relationship_type_enum = relationship.relationship_type
                            related_item_type = relationship.related_item_type
                            metadata = relationship.metadata
                            
                            logger.info(f"Processing relationship as RelationshipItem object")
                        
                        # Log all extracted fields
                        logger.info(f"Related item ID: {related_item_id}")
                        logger.info(f"Relation type: {relation_type}")
                        logger.info(f"Relationship type enum: {relationship_type_enum}")
                        logger.info(f"Related item type: {related_item_type}")
                        logger.info(f"Metadata: {metadata}")
                        
                        # Ensure both values are strings
                        if related_item_id is not None:
                            related_item_id = str(related_item_id)
                        if relation_type is not None:
                            relation_type = str(relation_type)

                        # Handle the case where related_item_id is None but relationship_type is set
                        if (related_item_id is None or related_item_id == 'None') and relationship_type_enum:
                            related_item_id = str(relationship_type_enum.value)
                        
                        # If we have a related_item_id but no relationship_type_enum, set it to LINK_TO_ID
                        if related_item_id and not relationship_type_enum:
                            relationship_type_enum = RelationshipType.LINK_TO_ID
                            logger.info(f'Set relationship_type_enum to LINK_TO_ID for related_item_id: {related_item_id}')

                        if not related_item_id or not relation_type:
                            logger.warning(f"Skipping invalid relationship: {relationship}")
                            continue
                    except Exception as e:
                        logger.error(f"Error extracting relationship data from {relationship}: {e}")
                        logger.error(f"Relationship type: {type(relationship)}")
                        logger.error(f"Relationship content: {relationship}")
                        continue
            

                    # Handle special relationship cases using enum
                    if relationship_type_enum == RelationshipType.PREVIOUS_MEMORY_ITEM:
                        if memory_item_ids:
                            related_item_id = str(memory_item_ids[-1])
                            logger.info(f'Using most recent memory item ID: {related_item_id}')
                        else:
                            logger.info('No previous memory items found, skipping relationship')
                            continue

                    elif relationship_type_enum == RelationshipType.ALL_PREVIOUS_MEMORY_ITEMS:
                        for prev_memory_id in memory_item_ids:
                            success &= await self._create_single_relationship(
                                str(prev_memory_id),
                                str(memory_item.id),
                                relation_type,
                                workspace_id,
                                user_id,
                                created_relationships,
                                neo_session=neo_session,
                                organization_id=organization_id,
                                namespace_id=namespace_id
                            )
                        continue
                    
                    # If relationship_type_enum is None, treat as regular relationship
                    elif relationship_type_enum is None:
                        logger.info(f'Processing regular relationship with relation_type: {relation_type}')
                        # Continue to regular relationship creation below
                    
                    # Handle LINK_TO_ID case - use the related_item_id directly
                    elif relationship_type_enum == RelationshipType.LINK_TO_ID:
                        logger.info(f'Processing LINK_TO_ID relationship with related_item_id: {related_item_id}')
                        # Continue to regular relationship creation below

                    # Create regular relationship
                    relationship_success = await self._create_single_relationship(
                        str(memory_item.id),
                        related_item_id,
                        relation_type,
                        workspace_id,
                        user_id,
                        created_relationships,
                        neo_session=neo_session,
                        organization_id=organization_id,
                        namespace_id=namespace_id
                    )
                    success = success and relationship_success


        except Exception as e:
            logger.error(f"Error creating relationships: {e}")
            logger.error(f"Relationship data: {relationships_json}")
            logger.error(f"Memory item: {memory_item}")
            success = False
            error = str(e)

        return {
            "success": success and len(created_relationships) > 0,
            "relationships": created_relationships,
            "error": error
        }

    async def _create_single_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        workspace_id: Optional[str],
        user_id: str,
        created_relationships: List[LLMGraphRelationship],
        neo_session: AsyncSession,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None
    ) -> bool:
        """Helper method to create a single relationship and update the relationships list"""
        try:
            result = await self.link_memory_items_async(
                source_id,
                target_id,
                relation_type,
                neo_session=neo_session,
                workspace_id=workspace_id,
                user_id=user_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if result:
                created_relationships.append(LLMGraphRelationship(
                    type=relation_type,
                    direction="->",
                    source=NodeReference(
                        label="Memory",
                        id=source_id
                    ),
                    target=NodeReference(
                        label="Memory",
                        id=target_id
                    )
                ))
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error creating relationship between {source_id} and {target_id}: {e}")
            return False

    async def _safe_neo4j_run(self, session, query: str, params: dict = None, operation_name: str = "Neo4j operation"):
        """
        Safely execute a Neo4j query with connection error handling.
        
        Args:
            session: Neo4j session
            query: Cypher query to execute
            params: Query parameters
            operation_name: Name of the operation for logging
            
        Returns:
            Result object or None if connection failed
        """
        try:
            result = await session.run(query, params or {})
            if result is None:
                logger.error(f"Neo4j session returned None result in {operation_name}, connection may be closed")
                # Record failure in circuit breaker
                await self.async_neo_conn.circuit_breaker.record_failure()
                self.async_neo_conn.fallback_mode = True
                return None
            return result
        except Exception as e:
            # Record failure in circuit breaker for any exception
            await self.async_neo_conn.circuit_breaker.record_failure()
            
            # Check if it's a connection error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['closed connection', 'connection', 'driver', 'session', 'complete']):
                logger.error(f"Neo4j connection error in {operation_name}: {e}")
                self.async_neo_conn.fallback_mode = True
            else:
                logger.error(f"Error in {operation_name}: {e}")
            return None

    async def memory_item_exists_async(self, session, memory_item_id: str) -> bool:
        """Async check if memory item exists in Neo4j"""
        result = await self._safe_neo4j_run(
            session, 
            "MATCH (a:Memory) WHERE a.id = $id RETURN a",
            {"id": memory_item_id},
            "memory_item_exists_async"
        )
        if result is None:
            return False
        
        try:
            record = await result.single()
            return record is not None
        except Exception as e:
            logger.error(f"Error processing result in memory_item_exists_async: {e}")
            return False
    
    def _strip_chunk_id(self, memory_id: str) -> str:
        """Strips chunk identifier (_0, _1, etc.) from memory ID."""
        if not memory_id:
            return memory_id
        return memory_id.split('_')[0]

    async def link_memory_items_async(
        self, 
        item_id_1: str, 
        item_id_2: str, 
        relation_type: str, 
        neo_session: AsyncSession,
        workspace_id: Optional[str] = None, 
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None
    ) -> bool:
        """
        Async version of link_memory_items with tenant scoping and ACL checks.
        
        SECURITY: Ensures both nodes:
        1. Exist in Neo4j
        2. Belong to the same tenant (workspace_id, organization_id, namespace_id - MUST match with AND)
        3. User has write access to both nodes (user_id OR *_write_access - SHOULD match with OR)
        
        Follows the same pattern as _node_exists and _merge_node_by_unique_props:
        - Tenant scoping (workspace_id, organization_id, namespace_id): MUST match (AND logic)
        - ACL checks (user_id, *_write_access): SHOULD match (OR logic)
        """
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, cannot create relationships")
            return False

        try:
            # Strip chunk identifiers from memory IDs
            clean_id_1 = self._strip_chunk_id(str(item_id_1))
            clean_id_2 = self._strip_chunk_id(str(item_id_2))
            
            logger.info(f"Original IDs: {item_id_1}, {item_id_2}")
            logger.info(f"Cleaned IDs: {clean_id_1}, {clean_id_2}")

            # Build tenant scoping conditions (MUST - AND logic)
            tenant_conditions_a = []
            tenant_conditions_b = []
            
            if workspace_id:
                tenant_conditions_a.append("a.workspace_id = $workspace_id")
                tenant_conditions_b.append("b.workspace_id = $workspace_id")
            if organization_id:
                tenant_conditions_a.append("a.organization_id = $organization_id")
                tenant_conditions_b.append("b.organization_id = $organization_id")
            if namespace_id:
                tenant_conditions_a.append("a.namespace_id = $namespace_id")
                tenant_conditions_b.append("b.namespace_id = $namespace_id")
            
            # Build ACL conditions (SHOULD - OR logic) for each node
            acl_conditions_a = []
            acl_conditions_b = []
            
            if user_id:
                acl_conditions_a.append("a.user_id = $user_id")
                acl_conditions_a.append("$user_id IN a.user_write_access")
                acl_conditions_b.append("b.user_id = $user_id")
                acl_conditions_b.append("$user_id IN b.user_write_access")
            
            if workspace_id:
                acl_conditions_a.append("$workspace_id IN a.workspace_write_access")
                acl_conditions_b.append("$workspace_id IN b.workspace_write_access")
            
            if organization_id:
                acl_conditions_a.append("$organization_id IN a.organization_write_access")
                acl_conditions_b.append("$organization_id IN b.organization_write_access")
            
            if namespace_id:
                acl_conditions_a.append("$namespace_id IN a.namespace_write_access")
                acl_conditions_b.append("$namespace_id IN b.namespace_write_access")
            
            # Combine tenant scoping (MUST) and ACL (SHOULD) conditions
            where_parts = []
            
            # Add tenant scoping for both nodes (MUST)
            if tenant_conditions_a:
                where_parts.append(f"({' AND '.join(tenant_conditions_a)})")
            if tenant_conditions_b:
                where_parts.append(f"({' AND '.join(tenant_conditions_b)})")
            
            # Add ACL checks for both nodes (SHOULD)
            if acl_conditions_a:
                where_parts.append(f"({' OR '.join(acl_conditions_a)})")
            if acl_conditions_b:
                where_parts.append(f"({' OR '.join(acl_conditions_b)})")
            
            where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
            
            # Verify both nodes exist WITH tenant scoping and ACL checks
            verify_query = f"""
            MATCH (a:Memory)
            WHERE a.id = $item_id_1
            WITH a
            MATCH (b:Memory)
            WHERE b.id = $item_id_2
            WITH a, b
            {where_clause}
            RETURN a.id as id1, b.id as id2, a.workspace_id as workspace_id, 
                   a.organization_id as organization_id, a.namespace_id as namespace_id
            """
            
            parameters = {
                'item_id_1': str(clean_id_1), 
                'item_id_2': str(clean_id_2),
                'workspace_id': workspace_id,
                'user_id': user_id,
                'organization_id': organization_id,
                'namespace_id': namespace_id
            }

            logger.info(f"🔒 Executing verify query with tenant scoping (MUST) and ACL checks (SHOULD)")
            logger.info(f"Parameters: workspace_id={workspace_id}, org_id={organization_id}, namespace_id={namespace_id}, user_id={user_id}")
            result = await neo_session.run(verify_query, parameters)
            record = await result.single()
            logger.info(f"Verify query record: {record}")

            # Check if both nodes were found with proper access
            if not record or not record.get("id1") or not record.get("id2"):
                error_msg = f"Memory items {clean_id_1} and/or {clean_id_2} not found, not in same tenant, or user lacks write access"
                logger.error(error_msg)
                return False
            
            logger.info(f"✅ Both nodes exist in same tenant (workspace={record.get('workspace_id')}, org={record.get('organization_id')}, namespace={record.get('namespace_id')}) and user has write access")

            # Build tenant scoping for create query (same conditions)
            create_where_a = ["a.id = $item_id_1"]
            create_where_b = ["b.id = $item_id_2"]
            
            if workspace_id:
                create_where_a.append("a.workspace_id = $workspace_id")
                create_where_b.append("b.workspace_id = $workspace_id")
            if organization_id:
                create_where_a.append("a.organization_id = $organization_id")
                create_where_b.append("b.organization_id = $organization_id")
            if namespace_id:
                create_where_a.append("a.namespace_id = $namespace_id")
                create_where_b.append("b.namespace_id = $namespace_id")
            
            # If we get here, both nodes exist with proper access, create relationship
            # Build relationship properties - only include non-NULL tenant IDs
            # (Neo4j cannot MERGE on NULL properties)
            rel_props = []
            rel_params = {
                'item_id_1': str(clean_id_1),
                'item_id_2': str(clean_id_2),
                'relation_type': relation_type
            }
            
            if workspace_id is not None:
                rel_props.append("workspace_id: $workspace_id")
                rel_params['workspace_id'] = workspace_id
            if organization_id is not None:
                rel_props.append("organization_id: $organization_id")
                rel_params['organization_id'] = organization_id
            if namespace_id is not None:
                rel_props.append("namespace_id: $namespace_id")
                rel_params['namespace_id'] = namespace_id
            if user_id is not None:
                rel_props.append("user_id: $user_id")
                rel_params['user_id'] = user_id
            
            # Always include type and created_at
            rel_props.append("type: $relation_type")
            rel_props.append("created_at: datetime()")
            
            create_query = f"""
            MATCH (a:Memory)
            WHERE {' AND '.join(create_where_a)}
            MATCH (b:Memory)
            WHERE {' AND '.join(create_where_b)}
            MERGE (a)-[r:{relation_type} {{
                {', '.join(rel_props)}
            }}]->(b)
            RETURN type(r) as rel_type
            """
            parameters = rel_params
            logger.info(f"Executing create query with tenant scoping")
            result = await neo_session.run(create_query, parameters)
            record = await result.single()
            logger.info(f"Create query record: {record}")

            # Check if relationship was created
            if not record or not record.get("rel_type"):
                error_msg = f"Failed to create relationship between {clean_id_1} and {clean_id_2}"
                logger.error(error_msg)
                return False
            
            logger.info(f"✅ Successfully created {relation_type} relationship between {clean_id_1} and {clean_id_2}")
            return True
        except Exception as e:
            logger.error(f"Error in link_memory_items_async: {str(e)}")
            self.async_neo_conn.fallback_mode = True
            return False

    
    async def add_grouped_memory_item_to_qdrant(self, memory_item: Dict[str, Any], related_memories: List[ParseStoredMemory]):
       
        """
        Get embeddings for memory item and related memories, and upsert to Qdrant.
        Creates a NEW grouped memory entry without overwriting the original single memory.
        
        Args:
            memory_item (Dict[str, Any]): The primary memory item to index
            related_memories (List[ParseStoredMemory]): List of related memory items
        """
         
        if not related_memories:
            logger.info("No related memories provided to add_grouped_memory_item_to_qdrant. Skipping.")
            return
            
        # 1. Aggregate text content
        aggregated_text = memory_item.get('content', '')
        for related_memory in related_memories:
            if hasattr(related_memory, 'content') and related_memory.content:
                aggregated_text += " " + related_memory.content
            elif isinstance(related_memory, dict) and related_memory.get('content'):
                aggregated_text += " " + related_memory.get('content')
        
        if not aggregated_text.strip():
            logger.warning("No text content to process in add_grouped_memory_item_to_qdrant. Skipping.")
            return

        # 2. Get embeddings for aggregated text
        try:
            embeddings, chunks = await self.embedding_model.get_qwen_embedding_4b(aggregated_text)
        except Exception as e:
            logger.error(f"Failed to get embeddings for Qdrant: {e}", exc_info=True)
            return

        # 3. Create UNIQUE ID for grouped memory (don't overwrite original)
        original_id = str(memory_item['id'])
        grouped_base_id = f"{original_id}_grouped"
        
        # Get related memory IDs safely
        related_memory_ids = []
        # Collect custom metadata from all memories for grouped filtering
        all_custom_metadata = {}
        
        # Start with primary memory custom metadata
        if 'customMetadata' in memory_item:
            all_custom_metadata.update(memory_item['customMetadata'])
        
        for mem in related_memories:
            if isinstance(mem, dict):
                mem_id = mem.get('memoryId')
                mem_custom_meta = mem.get('customMetadata', {})
            else:
                mem_id = getattr(mem, 'memoryId', None)
                mem_custom_meta = getattr(mem, 'customMetadata', {})
            
            if mem_id:
                related_memory_ids.append(mem_id)
            
            # Merge custom metadata from related memories
            if mem_custom_meta and isinstance(mem_custom_meta, dict):
                for key, value in mem_custom_meta.items():
                    if key not in all_custom_metadata:
                        all_custom_metadata[key] = value
                    elif isinstance(all_custom_metadata[key], list) and isinstance(value, list):
                        # For list fields, combine unique values
                        all_custom_metadata[key] = list(set(all_custom_metadata[key] + value))
                    elif isinstance(all_custom_metadata[key], (int, float)) and isinstance(value, (int, float)):
                        # For numeric fields, use the primary memory value (could be enhanced with aggregation logic)
                        pass
                    # For string fields, keep primary memory value

        # 4. Add grouped memory points to Qdrant with NEW IDs
        successful_upserts = 0
        for i, (embedding_vector, chunk_text) in enumerate(zip(embeddings, chunks)):
            # Use grouped_base_id instead of original base_id
            chunk_id = f"{grouped_base_id}_{i}"
            
            # Prepare metadata (payload) for this chunk
            payload = memory_item.get('metadata', {}).copy()
            payload['chunk_id'] = chunk_id
            payload['content'] = chunk_text
            payload['original_memory_id'] = original_id  # Reference to original
            payload['grouped_memory_count'] = len(related_memories) + 1  # Total memories in group
            
            # Add related memory IDs and mark as grouped
            if i == 0:
                payload['relatedMemoryIds'] = related_memory_ids
                payload['isGroupedMemories'] = True
                payload['primary_memory_id'] = original_id
                logger.info(f"Creating grouped memory {chunk_id} with {len(related_memory_ids)} related memories")
            
            # Add custom metadata from all memories for filtering
            if all_custom_metadata:
                for key, value in all_custom_metadata.items():
                    if value is not None:
                        payload[key] = value
                        logger.info(f"Added custom metadata field '{key}' = '{value}' to grouped memory payload")
                
                # Ensure custom metadata indexes exist for grouped memories
                await self.ensure_custom_metadata_indexes(all_custom_metadata)

            # Ensure embedding is a list of floats
            if isinstance(embedding_vector, np.ndarray):
                embedding_vector = embedding_vector.tolist()
            
            try:
                result = await self.add_qdrant_point(chunk_id, embedding_vector, self.sanitize_metadata(payload))
                # Debug logging for result structure
                logger.debug(f"Qdrant grouped result type: {type(result)}, result: {result}")
                
                if result and hasattr(result, 'status'):
                    logger.debug(f"Grouped result status type: {type(result.status)}, value: {result.status}")
                    # Check for both enum and string values for backward compatibility
                    if (result.status == UpdateStatus.COMPLETED or 
                        (hasattr(result.status, 'value') and result.status.value == 'completed') or
                        str(result.status) == 'completed'):
                        successful_upserts += 1
                        logger.info(f"Successfully added grouped Qdrant point with chunk_id: {chunk_id}")
                    else:
                        logger.warning(f"Qdrant grouped upsert returned unexpected status for chunk_id: {chunk_id}")
                        logger.warning(f"Expected: {UpdateStatus.COMPLETED} or 'completed', Got: {result.status}")
                        logger.warning(f"Qdrant result details: operation_id={getattr(result, 'operation_id', 'N/A')} status={result.status}")
                elif result is None:
                    logger.warning(f"Qdrant grouped upsert failed for chunk_id: {chunk_id} - returned None")
                else:
                    logger.warning(f"Qdrant grouped upsert returned result without status for chunk_id: {chunk_id}")
                    logger.warning(f"Qdrant result details: {result}")
            except Exception as e:
                logger.error(f"Failed to add Qdrant point with chunk_id {chunk_id}: {e}")
                raise e

        logger.info(f'Successfully upserted {successful_upserts}/{len(embeddings)} grouped memory points to Qdrant for original_id {original_id} as grouped_id {grouped_base_id}')

    @staticmethod
    def sanitize_metadata(metadata: dict) -> dict:
        """Sanitizes metadata to ensure it's JSON-serializable."""
        def clean_value(v):
            if isinstance(v, (str, int, float, bool, type(None))):
                return v
            if isinstance(v, datetime):
                return v.isoformat()
            if isinstance(v, list):
                return [clean_value(i) for i in v]
            if isinstance(v, dict):
                # Qdrant supports nested dicts
                return {str(k): clean_value(v_item) for k, v_item in v.items()}
            # For other complex types, convert to string as a fallback
            return str(v)

        if not isinstance(metadata, dict):
            return {}
        
        return {str(k): clean_value(v) for k, v in metadata.items()}

    @staticmethod
    def clean_metadata_for_pinecone(metadata: dict) -> dict:
        def clean_value(v):
            if isinstance(v, dict):
                return MemoryGraph.clean_metadata_for_pinecone(v)
            elif isinstance(v, list):
                return [clean_value(i) for i in v if i is not None and i != 'None']
            else:
                return v
        return {k: clean_value(v) for k, v in metadata.items() if v is not None and v != 'None'}

    @staticmethod
    def pinecone_compatible_metadata(metadata) -> dict:
        # Accepts MemoryMetadata or dict
        import json
        if hasattr(metadata, 'flatten'):
            flat = metadata.flatten()
        else:
            base = dict(metadata)
            custom = base.pop('customMetadata', {})
            # Robustly handle customMetadata legacy types
            if custom is None or custom == 'None':
                custom = {}
            elif isinstance(custom, str):
                try:
                    custom = json.loads(custom)
                    if not isinstance(custom, dict):
                        custom = {}
                except Exception:
                    custom = {}
            from models.shared_types import flatten_dict
            flat_custom = flatten_dict(custom)
            # Ensure topics, emoji_tags, emotion_tags are always lists
            for field in ['topics', 'emoji_tags', 'emotion_tags']:
                val = base.get(field)
                if isinstance(val, str):
                    base[field] = [item.strip() for item in val.split(',') if item.strip()]
                elif val is None:
                    base[field] = []
            flat = {**base, **flat_custom}
        compatible = {}
        for k, v in flat.items():
            if isinstance(v, (str, int, float, bool)):
                compatible[k] = v
            elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                compatible[k] = v
        return compatible
    
    async def update_pinecone(
        self, 
        vector_id: str, 
        new_metadata: dict,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Asynchronously updates metadata of an existing vector in Pinecone.
        If embedding is provided, also updates the embedding.

        Args:
            vector_id (str): The ID of the vector to update
            new_metadata (Dict[str, Any]): The new metadata to set
            embedding (Optional[List[float]]): The new embedding vector (optional for metadata-only updates)

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Ensure new_metadata is Pinecone-compatible
            compatible_new_metadata = self.pinecone_compatible_metadata(new_metadata)
            
            if embedding is not None:
                # Update both vector and metadata using upsert
              
                result = await self._safe_pinecone_operation(
                    "upsert",
                    self.index.upsert,
                    vectors=[(vector_id, embedding, compatible_new_metadata)]
                )
                if result is None:
                    logger.warning(f"Pinecone upsert failed for vector {vector_id}")
                else:
                    logger.info(f"Updated Pinecone vector {vector_id} with new embedding and metadata")
            else:
                # Update metadata only using Pinecone's update method
                result = await self._safe_pinecone_operation(
                    "update",
                    self.index.update,
                    id=vector_id,
                    set_metadata=compatible_new_metadata
                )
                if result is None:
                    logger.warning(f"Pinecone metadata update failed for vector {vector_id}")
                else:
                    logger.info(f"Updated Pinecone vector {vector_id} metadata only (no embedding update)")
            
            return True

        except Exception as e:
            logger.error(f"Error updating vector in Pinecone for vector ID {vector_id}: {e}")
            logger.error("Full traceback:", exc_info=True)
            return False
    
    def update_memory_item_in_bigbird(self, memory_item: dict):
        # Similar adjustment as add_memory_item_to_bigbird to ensure correct text content is passed
        if isinstance(memory_item, MemoryItem):  # Adjust based on your class structure
            text_content = memory_item.content
        else:
            text_content = memory_item

        # Check if Pinecone is available before attempting update
        if not self.bigbird_index or self.pinecone_fallback_mode:
            logger.warning(f"Pinecone BigBird index is in fallback mode, skipping update for {memory_item.id}")
            return

        # Check if Pinecone is available before attempting update
        if not self.bigbird_index or self.pinecone_fallback_mode:
            logger.warning(f"Pinecone BigBird index is in fallback mode, skipping update for {memory_item.id}")
            return

        #embedding, chunks = self.embedding_model.get_bigbird_embedding(text_content)
        embedding = self.embedding_model.get_qwen_embedding(text_content)
        for chunkembedding in embedding:
            try:
                # Use direct update with error handling (not async)
                self.bigbird_index.update(
                    id=str(memory_item.id),
                    values=chunkembedding
                )
                logger.info(f'Updated memory item with id {memory_item.id} in BigBird Pinecone index')
            except Exception as e:
                logger.error(f'BigBird Pinecone update failed for memory item {memory_item.id}: {e}')
                # Record failure in circuit breaker
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule the circuit breaker update
                        asyncio.create_task(self.pinecone_circuit_breaker.record_failure())
                    else:
                        loop.run_until_complete(self.pinecone_circuit_breaker.record_failure())
                except:
                    # If we can't record the failure, just log it
                    logger.warning("Could not record circuit breaker failure")
            try:
                # Use direct update with error handling (not async)
                self.bigbird_index.update(
                    id=str(memory_item.id),
                    values=chunkembedding
                )
                logger.info(f'Updated memory item with id {memory_item.id} in BigBird Pinecone index')
            except Exception as e:
                logger.error(f'BigBird Pinecone update failed for memory item {memory_item.id}: {e}')
                # Record failure in circuit breaker
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule the circuit breaker update
                        asyncio.create_task(self.pinecone_circuit_breaker.record_failure())
                    else:
                        loop.run_until_complete(self.pinecone_circuit_breaker.record_failure())
                except:
                    # If we can't record the failure, just log it
                    logger.warning("Could not record circuit breaker failure")

    def update_memory_item_in_embeddinglocal(self, memory_item: dict):
        # Similar adjustment as add_memory_item_to_bigbird to ensure correct text content is passed
        if isinstance(memory_item, MemoryItem):  # Adjust based on your class structure
            text_content = memory_item.content
        else:
            text_content = memory_item

        embedding = self.embedding_model.get_embeddinglocal(text_content)
        self.snowflake_index.update(id=str(memory_item.id), values=embedding.tolist())  # Ensure embedding is correctly formatted
        logger.info(f'Updated memory item with id {memory_item.id} in BigBird Pinecone index')

    def get_memory_item_from_bigbird(self, memory_item_id: str) -> Optional[Dict[str, Any]]:
        results = self.bigbird_index.fetch(ids=[memory_item_id])
        if results:
            return results[memory_item_id]
        else:
            return None
    
    async def get_memory_item_ids_from_conversation_history(
        self, 
        conversation_history: list[dict], 
        user_id: str,
        legacy_route: bool = True
    ) -> List[str]:
        # Extract the content from each item in the conversation history
        conversation_content = [
            item['content'] for item in conversation_history 
            if 'content' in item
        ] if conversation_history else []
        
        # Get user info for ACL filter
        user_instance = User.get(user_id)
        user_roles = user_instance.get_roles()
        user_workspace_ids = User.get_workspaces_for_user(user_id)
        
        # Get organization and namespace info (if available)
        user_organization_id = getattr(user_instance, 'organization_id', None)
        user_namespace_id = getattr(user_instance, 'namespace_id', None)
        user_organization_access = getattr(user_instance, 'organization_read_access', [])
        user_namespace_access = getattr(user_instance, 'namespace_read_access', [])
        
        logger.debug(f'user_roles {user_roles}')
        logger.info(f'user_workspace_ids {user_workspace_ids}')
        logger.debug(f'user_organization_id {user_organization_id}')
        logger.debug(f'user_namespace_id {user_namespace_id}')

        # Setup the ACL filter with organization and namespace support
        acl_conditions = [
            {"user_id": {"$eq": str(user_id)}},
            {"user_read_access": {"$in": [str(user_id)]}},
            {"workspace_read_access": {"$in": [str(workspace_id) for workspace_id in user_workspace_ids]}},
            {"role_read_access": {"$in": user_roles}},
        ]
        
        # Add organization conditions if available
        if user_organization_id:
            acl_conditions.append({"organization_id": {"$eq": str(user_organization_id)}})
        if user_organization_access:
            acl_conditions.append({"organization_read_access": {"$in": user_organization_access}})
            
        # Add namespace conditions if available
        if user_namespace_id:
            acl_conditions.append({"namespace_id": {"$eq": str(user_namespace_id)}})
        if user_namespace_access:
            acl_conditions.append({"namespace_read_access": {"$in": user_namespace_access}})
        
        acl_filter = {"$or": acl_conditions}
        
        logger.info(f'ACL filter for conversation history: {acl_filter}')
        logger.info(f'User roles: {user_roles}')
        logger.info(f'User workspace IDs: {user_workspace_ids}')

        memory_item_ids = set()  # Use a set to avoid duplicates

        if legacy_route:
            # Legacy route: Run both sentence and qwen embeddings in parallel
            logger.info("Using legacy route - running both sentence and qwen embeddings in parallel")
            
            async def get_sentence_embeddings():
                """Get sentence embeddings for legacy Pinecone search"""
                sentence_embeddings = []
                for content in conversation_content:
                    embedding, chunks = await self.embedding_model.get_sentence_embedding(content)
                    if isinstance(embedding, list):
                        sentence_embeddings.extend(embedding)
                    else:
                        sentence_embeddings.append(embedding)
                return sentence_embeddings
            
            async def get_qwen_embeddings():
                """Get qwen embeddings for Qdrant search"""
                qwen_embeddings = []
                for content in conversation_content:
                    embedding = await self.embedding_model.get_qwen_embedding_4b(content)
                    if isinstance(embedding, list):
                        qwen_embeddings.extend(embedding)
                    else:
                        qwen_embeddings.append(embedding)
                return qwen_embeddings
            
            # Run both embedding generations in parallel
            sentence_embeddings, qwen_embeddings = await asyncio.gather(
                get_sentence_embeddings(),
                get_qwen_embeddings()
            )
            
            # Create search tasks for both Pinecone and Qdrant
            search_tasks = []
            
            # Add Pinecone search tasks (if Pinecone is available)
            if hasattr(self, 'index') and self.index:
                for embedding in sentence_embeddings:
                    search_tasks.append(
                        asyncio.to_thread(
                            self.index.query,
                            namespace="",
                            top_k=5,
                            include_values=True,
                            include_metadata=True,
                            vector=embedding,
                            filter=acl_filter
                        )
                    )
            
            # Add Qdrant search tasks
            if self.qdrant_client and self.qdrant_collection:
                for embedding in qwen_embeddings:
                    search_tasks.append(
                        self._qdrant_search_async(
                            collection_name=self.qdrant_collection,
                            query_vector=embedding,
                            query_filter=acl_filter,
                            limit=5,
                            with_payload=True,
                            with_vectors=False
                        )
                    )
            
            # Wait for all searches to complete
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                try:
                    if isinstance(result, Exception):
                        logger.error(f"Error in search query: {result}")
                        continue
                    
                    # Handle different result types (Pinecone vs Qdrant)
                    if hasattr(result, 'matches'):  # Pinecone result
                        matches = result.matches
                        for match in matches:
                            memory_item_id = str(match.id)
                            memory_item_ids.add(memory_item_id)
                            logger.info(f'Found memory item id {memory_item_id} from Pinecone for conversation content')
                    elif hasattr(result, 'points'):  # Qdrant result (query_points format)
                        points = result.points
                        for point in points:
                            memory_item_id = str(point.id)
                            memory_item_ids.add(memory_item_id)
                            logger.info(f'Found memory item id {memory_item_id} from Qdrant for conversation content')
                    elif isinstance(result, list):  # Qdrant _qdrant_search_async / search returns list of points
                        for point in result:
                            memory_item_id = str(getattr(point, 'id', point))
                            memory_item_ids.add(memory_item_id)
                            logger.info(f'Found memory item id {memory_item_id} from Qdrant for conversation content')
                    else:
                        logger.warning(f"Unknown result type: {type(result)}")
                        
                except Exception as e:
                    logger.error(f"Error processing search result: {e}")
                    continue
                    
        else:
            # New route: Only use qwen embeddings for Qdrant
            logger.info("Using new route - running only qwen embeddings for Qdrant")
            
            qwen_embeddings = []
            for content in conversation_content:
                embedding = await self.embedding_model.get_qwen_embedding_4b(content)
                if isinstance(embedding, list):
                    qwen_embeddings.extend(embedding)
                else:
                    qwen_embeddings.append(embedding)
            
            # Create Qdrant search tasks
            search_tasks = []
            if self.qdrant_client and self.qdrant_collection:
                for embedding in qwen_embeddings:
                    search_tasks.append(
                        self._qdrant_search_async(
                            collection_name=self.qdrant_collection,
                            query_vector=embedding,
                            query_filter=acl_filter,
                            limit=5,
                            with_payload=True,
                            with_vectors=False
                        )
                    )
            
            # Wait for all searches to complete
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                try:
                    if isinstance(result, Exception):
                        logger.error(f"Error in Qdrant search query: {result}")
                        continue
                    
                    # Handle Qdrant result
                    if hasattr(result, 'points'):
                        points = result.points
                        for point in points:
                            memory_item_id = str(point.id)
                            memory_item_ids.add(memory_item_id)
                            logger.info(f'Found memory item id {memory_item_id} from Qdrant for conversation content')
                    elif isinstance(result, list):  # _qdrant_search_async / search returns list of points
                        for point in result:
                            memory_item_id = str(getattr(point, 'id', point))
                            memory_item_ids.add(memory_item_id)
                            logger.info(f'Found memory item id {memory_item_id} from Qdrant for conversation content')
                    else:
                        logger.warning(f"Unknown Qdrant result type: {type(result)}")
                        
                except Exception as e:
                    logger.error(f"Error processing Qdrant search result: {e}")
                    continue

        logger.info(f'Total unique memory item ids found: {len(memory_item_ids)}')
        return list(memory_item_ids)  # Convert set back to list before returning

    def add_memory_item(self, memory_item: MemoryItem, relationships_json: dict, sessionToken: str, user_id: str,  imageGenerationCategory=None, add_to_pinecone=True, workspace_id: str = None, api_key: Optional[str] = None, graph_override: Optional[Dict[str, Any]] = None, schema_id: Optional[str] = None, property_overrides: Optional[Dict[str, Dict[str, Any]]] = None):
        import asyncio
        from services.user_utils import User

        if memory_item and hasattr(memory_item, 'id'):
            self.memory_items[memory_item.id] = memory_item
        
        # If no workspace_id provided, try to get it from selected workspace follower
        if not workspace_id:
            workspace_id = User.get_selected_workspace_id(user_id, sessionToken)
            if workspace_id:
                logger.info(f"Using selected workspace ID: {workspace_id}")
                memory_item.metadata['workspace_id'] = workspace_id
            else:
                logger.warning("No workspace_id provided and no selected workspace found")
        
        # Define the categories for which we want to generate images
        IMAGE_GENERATION_CATEGORIES = {
            "narrative_element",
            "rpg_action",
            "object_description",
            "dream_or_fantasy",
            "art_idea",
            "historical_event",
            "biological_concept",
            "cultural_reference",
            "mood_or_emotion",
            "travel"
        }

        logger.info(f'imageGenerationCategory: {imageGenerationCategory}')

        # Initialize variables for the return values
        added_item_properties = None
        memory_item_obj = None

        if add_to_pinecone:
            # Add memory item without relationships first and wait for the result
            added_item_properties, memory_list = self.add_memory_item_without_relationships(sessionToken, memory_item, api_key=api_key)
            added_item_properties: List[ParseStoredMemory] = added_item_properties
            memory_list: List[MemoryItem] = memory_list

            if added_item_properties and memory_list:
                # Since we're now dealing with single items, get the first item
                added_item: ParseStoredMemory = added_item_properties[0]
                memory_item_obj: MemoryItem = memory_list[0]
                
                # Update the memory_item with objectId and createdAt from Parse response
                memory_item_obj.objectId = added_item.objectId
                memory_item_obj.createdAt = added_item.createdAt

                # Convert memory_item to a fully serializable dictionary
                memory_item_dict = memory_item_to_dict(memory_item_obj)

                # If the category is one for which we want to generate images
                if imageGenerationCategory and imageGenerationCategory in IMAGE_GENERATION_CATEGORIES:
                    logger.info(f'Image generation would be triggered here for category: {imageGenerationCategory}')
            
                if memory_item_obj:
                    # Create an async task for processing the memory item
                    async def process_memory_async():
                        await self.process_memory_item_async(
                            session_token=sessionToken,
                            memory_dict=memory_item_dict,
                            relationships_json=relationships_json,
                            workspace_id=workspace_id,
                            user_id=user_id,
                            api_key=api_key,
                            legacy_route=False,  # Use new route by default - this should be passed from caller
                            graph_override=graph_override,  # Pass graph_override to processing
                            schema_id=schema_id,  # Pass schema_id for enforcement
                            property_overrides=property_overrides  # Pass property_overrides for node customization
                        )

                    # Run the async task in a background thread
                    def run_async_task():
                        try:
                            # Create new event loop for this thread
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            # Run the async task
                            loop.run_until_complete(process_memory_async())
                            logger.info("Memory processing completed successfully")
                        except Exception as e:
                            logger.error(f"Error in memory processing: {e}")
                        finally:
                            loop.close()

                    # Start the background thread
                    import threading
                    thread = threading.Thread(target=run_async_task)
                    thread.start()
                    logger.info(f'Started async task to process memory item in background thread')

                    if memory_item_obj.context and len(memory_item_obj.context) > 0:
                        logger.info(f'Context for memory item exists: {memory_item_obj.context}')
                        self.update_memory_item_with_relationships(memory_item_obj, relationships_json, workspace_id, user_id, legacy_route=True)

        return [added_item_properties] if added_item_properties else []
    
    def _extract_graph_generation_config(self, memory_dict: dict) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Extract graph generation configuration from memory_dict.

        Returns:
            tuple: (graph_override, schema_id, property_overrides)
        """
        graph_generation = memory_dict.get('graph_generation')
        # Check for direct schema_id and property_overrides in memory_dict (added by add_memory_item_async)
        direct_schema_id = memory_dict.get('schema_id')
        direct_property_overrides = memory_dict.get('property_overrides')

        if direct_schema_id or direct_property_overrides:
            logger.info(f"✅ EXTRACT: Found direct parameters - schema_id: {direct_schema_id}, property_overrides: {bool(direct_property_overrides)}")
            return (None, direct_schema_id, direct_property_overrides)

        # Handle default auto mode or explicit configuration
        if not graph_generation:
            logger.info("🤖 DEFAULT GRAPH GENERATION: No graph_generation specified, defaulting to auto mode")
            # Return auto mode configuration as default
            return (None, None, None)  # This will trigger LLM generation in process_memory_item_async

        # Handle both dict format (from API) and Pydantic object format (from default_factory)
        if isinstance(graph_generation, dict):
            mode = graph_generation.get('mode', 'auto')

            if mode == 'manual':
                manual_config = graph_generation.get('manual', {})
                return (
                    manual_config,  # The manual config IS the graph_override
                    None,  # No schema_id in manual mode
                    None   # No property_overrides in manual mode
                )
            else:  # auto mode
                auto_config = graph_generation.get('auto', {}) or {}
                return (
                    None,  # No graph_override in auto mode
                    auto_config.get('schema_id'),
                    auto_config.get('property_overrides')
                )
        else:
            # Pydantic GraphGeneration object
            mode = graph_generation.mode.value if hasattr(graph_generation.mode, 'value') else str(graph_generation.mode)

            if mode == 'manual':
                manual_config = graph_generation.manual.model_dump() if graph_generation.manual else {}
                return (
                    manual_config,  # The manual config IS the graph_override
                    None,  # No schema_id in manual mode
                    None   # No property_overrides in manual mode
                )
            else:  # auto mode (default)
                auto_config = graph_generation.auto.model_dump() if graph_generation.auto else {}
                return (
                    None,  # No graph_override in auto mode
                    auto_config.get('schema_id'),
                    auto_config.get('property_overrides')
                )

        # This should never be reached now
        return (None, None, None)
    
    async def process_memory_item_async(self, session_token: str, memory_dict: dict, relationships_json: List[RelationshipItem] = None, workspace_id: str = None, user_id: str = None, user_workspace_ids: Optional[List[str]] = None, api_key: Optional[str] = None, neo_session: Optional[AsyncSession] = None, legacy_route: bool = True, graph_override: Optional[Dict[str, Any]] = None, schema_id: Optional[str] = None, property_overrides: Optional[Dict[str, Dict[str, Any]]] = None, developer_user_id: Optional[str] = None) -> ProcessMemoryResponse:
        """
        Process a memory item asynchronously.

        Parameters are passed directly from the API request via memory_service.py
        """
        # Use parameters passed directly instead of extracting from memory_dict
        # (The memory_dict doesn't contain graph_generation field - it's extracted in memory_service.py)
        logger.info(f"🔍 PROCESS_MEMORY_ASYNC: Using passed parameters directly")
        logger.info(f"🔍 PASSED PARAMS: graph_override={graph_override is not None}, schema_id={schema_id}, property_overrides={property_overrides is not None}")
        if property_overrides:
            logger.info(f"🔍 PASSED PROPERTY OVERRIDES: {property_overrides}")

        logger.info(f"🔍 PROCESS_MEMORY_ASYNC: fallback_mode={self.async_neo_conn.fallback_mode}")
        logger.info(f"🔍 GRAPH CONFIG: graph_override={graph_override is not None}, schema_id={schema_id}, property_overrides={property_overrides is not None}")
        if property_overrides:
            logger.info(f"🔍 PROPERTY OVERRIDES: {property_overrides}")
        
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, skipping process_memory_item_async")
            return {
                "status_code": 500,
                "success": False,
                "error": "Neo4j in fallback mode",
                "data": None
            }

        try:
            await self.ensure_async_connection()
            async with self.async_neo_conn.get_session() as neo_session:
                return await self._index_memories_and_process(
                    neo_session=neo_session,
                    session_token=session_token,
                    memory_dict=memory_dict,
                    relationships_json=relationships_json,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    user_workspace_ids=user_workspace_ids,
                    api_key=api_key,
                    legacy_route=legacy_route,
                    graph_override=graph_override,
                    schema_id=schema_id,
                    property_overrides=property_overrides,
                    developer_user_id=developer_user_id
                )
        except Exception as e:
            logger.error(f"Error in process_memory_item_async: {e}")
            self.async_neo_conn.fallback_mode = True
            return {
                "status_code": 500,
                "success": False,
                "error": str(e),
                "data": None
            }

    async def _index_memories_and_process(self, neo_session: AsyncSession, session_token: str, memory_dict: dict, relationships_json: List[RelationshipItem] = None, workspace_id: str = None, user_id: str = None, user_workspace_ids: Optional[List[str]] = None, api_key: Optional[str] = None, legacy_route: bool = True, graph_override: Optional[Dict[str, Any]] = None, schema_id: Optional[str] = None, property_overrides: Optional[Dict[str, Dict[str, Any]]] = None, developer_user_id: Optional[str] = None) -> ProcessMemoryResponse:
        # Start timing the entire process
        process_start_time = time.time()
        
        total_cost = 0
        success = True
        error = None

        # Initialize ChatGPTCompletion
        logger.info("Starting process memory item")
        chat_gpt = ChatGPTCompletion(
            env.get('OPENAI_API_KEY'),
            env.get('OPENAI_ORG_ID'),
            env.get('LLM_MODEL'),
            env.get('LLM_LOCATION_CLOUD', default=True),
            env.get('EMBEDDING_MODEL_LOCAL')
        )

        # Constants for embedding costs
        BIGBIRD_EMBEDDING_COST = 0.0009043
        SENTENCE_BERT_COST = 0.0004521

        # Calculate memory item sizes - Use datetime_handler for serialization
        try:
            memory_item_text = json.dumps(memory_dict, default=self.datetime_handler)
            memory_item_storage_size = len(memory_item_text.encode('utf-8'))
            memory_item_token_size = chat_gpt.count_tokens(memory_item_text)
            logger.info(f'Successfully serialized memory_dict: {memory_item_text[:200]}...')  # Log first 200 chars
        except Exception as e:
            logger.error(f"Error serializing memory_dict: {e}")
            logger.error(f"memory_dict keys: {memory_dict.keys()}")
            logger.error(f"memory_dict types: {[(k, type(v)) for k, v in memory_dict.items()]}")
            return {
                "status_code": 500,
                "success": False,
                "error": f"Error serializing memory_dict: {e}",
                "data": None
            }

        logger.info(f'Initial memory item metrics:'
                    f'\n- Token size: {memory_item_token_size}'
                    f'\n- Storage size: {memory_item_storage_size} bytes')

        # Extract schema_id from memory metadata if not provided
        if not schema_id:
            # Try to extract from metadata.customMetadata.schema_id
            metadata = memory_dict.get('metadata', {})
            if isinstance(metadata, dict):
                custom_metadata = metadata.get('customMetadata', {})
                if isinstance(custom_metadata, dict):
                    extracted_schema_id = custom_metadata.get('schema_id')
                    if extracted_schema_id:
                        schema_id = extracted_schema_id
                        logger.info(f"✅ Extracted schema_id from memory metadata: {schema_id}")
                    else:
                        logger.info("ℹ️  No schema_id found in metadata.customMetadata")
                else:
                    logger.info("ℹ️  customMetadata is not a dict")
            else:
                logger.info("ℹ️  metadata is not a dict")
        else:
            logger.info(f"✅ Using provided schema_id: {schema_id}")

        # Step 0: Fetch existing goals and use cases
        goal_object_ids = set()
        new_goal_object_ids = set()
        usecase_object_ids = set()
        new_usecase_object_ids = set()
        existing_goals = await get_user_goals_async(user_id, session_token, api_key=api_key)
        # Add existing goals (from Parse) if they have objectId
        if existing_goals:
            for goal in existing_goals:
                if isinstance(goal, dict) and 'objectId' in goal:
                    goal_object_ids.add(goal['objectId'])
                elif hasattr(goal, 'objectId'):
                    goal_object_ids.add(goal.objectId)

        extracted_goals = extract_goal_titles(existing_goals)
        logger.debug(f'extracted_goals: {extracted_goals}')

        existing_use_cases = await get_user_usecases_async(user_id, session_token, api_key=api_key)
        logger.debug(f'existing_use_cases: {existing_use_cases}')
        # Add existing usecases (from Parse) if they have objectId
        if existing_use_cases:
            for uc in existing_use_cases:
                if isinstance(uc, dict) and 'objectId' in uc:
                    usecase_object_ids.add(uc['objectId'])
                elif hasattr(uc, 'objectId'):
                    usecase_object_ids.add(uc.objectId)

        extracted_use_cases = extract_usecases(existing_use_cases)
        logger.debug(f'extracted_use_cases: {extracted_use_cases}')

        # Get memory graph schema from structured outputs
        memory_graph_schema = self.get_memory_graph_schema()
        logger.debug(f'memory_graph_schema: {memory_graph_schema}')

        # Get simplified schema
        node_names, relationship_types = self.get_simplified_schema(memory_graph_schema)
        logger.debug(f'Node names: {node_names}')
        logger.debug(f'Relationship types: {relationship_types}')

        # Reconstruct memory_graph_schema in the expected format
        memory_graph_schema = {
            "nodes": node_names,
            "relationships": relationship_types
        }
        logger.debug(f'Reconstructed memory_graph_schema: {memory_graph_schema}')

        # Step 1: Generate usecase memory item
        usecase_response = await chat_gpt.generate_usecase_memory_item_async(
            memory_dict,
            memory_dict.get('context'),
            extracted_goals,
            extracted_use_cases
        )

        if not usecase_response:
            return {
                "status_code": 500,
                "success": False,
                "error": "Failed to generate usecase memory item",
                "data": None
            }

        if usecase_response:
            usecase_memory_item = usecase_response["data"]
            usecase_metrics = usecase_response["metrics"]
            logger.info(f'Generate usecase memory item: {usecase_memory_item}')
            logger.info(f'Usecase metrics - Input tokens: {usecase_metrics["usecase_token_count_input"]}, '
                        f'Output tokens: {usecase_metrics["usecase_token_count_output"]}, '
                        f'Total cost: ${usecase_metrics["usecase_total_cost"]:.4f}')

            # Process goals and use cases
            if usecase_memory_item.get('use_cases'):
                new_use_cases = [uc for uc in usecase_memory_item["use_cases"] if uc["status"] == "new"]
                if new_use_cases:
                    created_usecases = await add_list_of_usecases_async(user_id, session_token, new_use_cases, api_key=api_key)
                    for created in created_usecases:
                        if created and 'objectId' in created:
                            usecase_object_ids.add(created['objectId'])
                            new_usecase_object_ids.add(created['objectId'])

            if usecase_memory_item.get('goals'):
                new_goals = [goal for goal in usecase_memory_item["goals"] if goal["status"] == "new"]
                if new_goals:
                    created_goals = await add_list_of_goals_async(user_id, session_token, new_goals, api_key=api_key)
                    for created in created_goals:
                        if created and 'objectId' in created:
                            goal_object_ids.add(created['objectId'])
                            new_goal_object_ids.add(created['objectId'])

        # Step 2: Find related memories and build relationships and index memories in BigBird
        # Pass the current memory ID to exclude from related memories search
        related_memories_response = await chat_gpt.generate_related_memories_async(
            session_token, memory_graph_schema, memory_dict, user_id, neo_session,
            extracted_goals, extracted_use_cases, None,
            exclude_memory_id=memory_dict.get('id'),
            user_workspace_ids=user_workspace_ids,
            api_key=api_key,
            legacy_route=legacy_route
        )

        if not related_memories_response:
            return {
                "status_code": 500,
                "success": False,
                "error": "Failed to generate related memories",
                "data": None
            }

        if related_memories_response:
            related_memories: List[ParseStoredMemory] = related_memories_response["data"]
            generated_queries = related_memories_response["generated_queries"]
            confidence_scores = related_memories_response["confidence_scores"]
            related_memories_metrics = related_memories_response["metrics"]

            logger.info(f'Generated queries for finding related memories: {generated_queries}')
            logger.info(f'Generate list of memories to build relationships with: {related_memories}')
            logger.info(f'Confidence scores for related memories: {confidence_scores}')
            logger.info(f'Related memories metrics - Input tokens: {related_memories_metrics["related_memories_token_count_input"]}, '
                        f'Output tokens: {related_memories_metrics["related_memories_token_count_output"]}, '
                        f'Total cost: ${related_memories_metrics["related_memories_total_cost"]:.6f}')

            # Trim and filter related memories
            trimmed_related_memories = self.trim_and_filter_related_memories(related_memories)

            # Create deterministic relationships with the top 3 related memories
            relationships_json = []
            logger.info(f'Creating relationships for {len(related_memories[:3])} related memories')
            logger.info(f'Legacy route enabled: {legacy_route}')
            
            for i, memory in enumerate(related_memories[:3]):  # Limit to top 3 memories
                related_item_id = None
                if isinstance(memory, dict):
                    related_item_id = memory.get('memoryId')
                    logger.info(f'Memory {i} (dict): memoryId={related_item_id}, score={memory.get("score", "N/A")}')
                elif hasattr(memory, 'memoryId'):
                    related_item_id = memory.memoryId
                    logger.info(f'Memory {i} (object): memoryId={related_item_id}, score={getattr(memory, "score", "N/A")}')
                
                if related_item_id:
                    relationship = {
                        "related_item_id": related_item_id,
                        "relation_type": "RELATED_TO",
                        "metadata": {
                            "similarity_score": getattr(memory, 'score', memory.get('score', 0) if isinstance(memory, dict) else 0),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    }
                    relationships_json.append(relationship)
                    logger.info(f'Created relationship {i}: {relationship}')

            logger.info(f'Final relationships_json: {relationships_json}')

            # When storing in memory_dict, ensure all datetime objects are converted
            if 'createdAt' in memory_dict:
                memory_dict['createdAt'] = self.datetime_handler(memory_dict['createdAt'])

            if 'updatedAt' in memory_dict:
                memory_dict['updatedAt'] = self.datetime_handler(memory_dict['updatedAt'])
            # Add to metadata goals and usecases
            if 'metadata' not in memory_dict or type(memory_dict['metadata']) is not dict:
                memory_dict['metadata'] = {}
            memory_dict['metadata']['goals'] = list(new_goal_object_ids)
            memory_dict['metadata']['usecases'] = list(new_usecase_object_ids)
            # When logging or serializing memory_dict
            memory_item_text = json.dumps(memory_dict, default=self.datetime_handler)
            memory_item_storage_size = len(memory_item_text.encode('utf-8'))

            # Index memories in BigBird / Qdrant Grouped Memories
            # If running under Temporal orchestration, this step may be performed by an activity;
            # honor an explicit flag to avoid duplicate work.
            if memory_dict.get('temporal_orchestrated'):
                logger.info('Temporal orchestrated run: skipping local Qdrant grouped indexing (handled by activity)')
            else:
                bigbird_memory_dict = copy.deepcopy(memory_dict)
                if 'metadata' in bigbird_memory_dict:
                    bigbird_memory_dict['metadata'] = MemoryGraph.pinecone_compatible_metadata(bigbird_memory_dict['metadata'])
                logger.info(f'memory_dict before adding to grouped memory via qdrant: {bigbird_memory_dict}')
                await self.add_grouped_memory_item_to_qdrant(bigbird_memory_dict, related_memories)
            
            # Process relationships between memories
            # Convert memory_dict to MemoryItem before passing to update_memory_item_with_relationships

            if relationships_json:
                memory_item_obj = memory_item_from_dict(memory_dict)
                if memory_dict.get('temporal_orchestrated'):
                    logger.info('Temporal orchestrated run: skipping local relationship update (handled by activity)')
                    relationship_result = {"success": True, "relationships": relationships_json}
                else:
                    relationship_result = await self.update_memory_item_with_relationships(
                        memory_item_obj,
                        relationships_json,
                        workspace_id,
                        user_id,
                        neo_session,
                        legacy_route=legacy_route
                    )

                if not relationship_result["success"]:
                    logger.warning(f"Failed to create some relationships: {relationship_result.get('error')}")
                    # Initialize empty relationship result if it failed
                    relationship_result = {
                        "success": False,
                        "relationships": [],
                        "error": relationship_result.get("error", "Failed to create relationships")
                    }
            else:
                # Initialize empty relationship result if no relationships provided
                relationship_result = {
                    "success": True,
                    "relationships": [],
                    "error": None
                }

            # Create MemoryPredictionLog as background task
            if memory_dict.get('objectId') and related_memories:
                try:
                    from models.parse_server import MemoryPredictionLog, ParsePointer
                    from services.memory_management import store_memory_prediction_log_async
                    
                    # Calculate memory age in hours
                    memory_age_hours = None
                    if memory_dict.get('createdAt'):
                        try:
                            if isinstance(memory_dict['createdAt'], str):
                                created_at = datetime.fromisoformat(memory_dict['createdAt'].replace('Z', '+00:00'))
                            else:
                                created_at = memory_dict['createdAt']
                            memory_age_hours = (datetime.now(created_at.tzinfo) - created_at).total_seconds() / 3600
                        except Exception as e:
                            logger.warning(f"Could not calculate memory age: {e}")

                    # Calculate temporal relationship metrics between new memory and retrieved memories
                    temporal_metrics = {}
                    if related_memories and len(related_memories) > 0:
                        try:
                            # Get creation time of new memory
                            created_at = memory_dict.get('createdAt')
                            if not created_at:
                                logger.warning("Memory dict missing createdAt field, skipping temporal metrics")
                                temporal_metrics = {}
                            else:
                                if isinstance(created_at, str):
                                    new_memory_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                else:
                                    new_memory_time = created_at

                                # Get creation times of retrieved memories
                                retrieved_times = []
                                for memory in related_memories[:3]:  # Top 3 memories
                                    if hasattr(memory, 'createdAt') and memory.createdAt:
                                        if isinstance(memory.createdAt, str):
                                            retrieved_time = datetime.fromisoformat(memory.createdAt.replace('Z', '+00:00'))
                                        else:
                                            retrieved_time = memory.createdAt
                                        retrieved_times.append(retrieved_time)

                                if retrieved_times:
                                    # Calculate age differences in hours
                                    age_differences = []
                                    for retrieved_time in retrieved_times:
                                        age_diff = abs((new_memory_time - retrieved_time).total_seconds() / 3600)
                                        age_differences.append(age_diff)

                                    # Calculate temporal metrics
                                    temporal_metrics = {
                                        'newToOldestMemoryAgeHours': max(age_differences),
                                        'newToNewestMemoryAgeHours': min(age_differences),
                                        'newToMedianMemoryAgeHours': sorted(age_differences)[len(age_differences)//2],
                                        'retrievedMemoriesAgeSpreadHours': max(age_differences) - min(age_differences),
                                        'temporalCoherenceScore': 1.0 - (max(age_differences) - min(age_differences)) / (max(age_differences) + 1)  # 0-1, higher = more clustered
                                    }
                        except Exception as e:
                            logger.warning(f"Could not calculate temporal metrics: {e}")

                        # Create MemoryPredictionLog
                        prediction_log = MemoryPredictionLog(
                            memoryItem=ParsePointer(
                                objectId=memory_dict['objectId'],
                                className="Memory"
                            ),
                            user=ParsePointer(
                                objectId=user_id,
                                className="_User"
                            ),
                            workspace=ParsePointer(
                                objectId=workspace_id,
                                className="WorkSpace"
                            ),
                            embeddingModel="bigbird",
                            generatedSearchQueries=generated_queries,
                            predictedRelatedMemories=[memory.objectId for memory in related_memories[:3]],  # Top 3 memories
                            predictionConfidenceScores=confidence_scores[:3] if confidence_scores and len(confidence_scores) >= 3 else (confidence_scores if confidence_scores else []),  # Top 3 confidence scores or all available
                            predictionMethod="cosine_similarity",
                            predictionProcessingTimeMs=(time.time() - process_start_time) * 1000,  # Actual processing time in milliseconds
                            relationshipCreationCount=3,  # Always 3 for top memories
                            newToOldestMemoryAgeHours=temporal_metrics.get('newToOldestMemoryAgeHours'),
                            newToNewestMemoryAgeHours=temporal_metrics.get('newToNewestMemoryAgeHours'),
                            newToMedianMemoryAgeHours=temporal_metrics.get('newToMedianMemoryAgeHours'),
                            retrievedMemoriesAgeSpreadHours=temporal_metrics.get('retrievedMemoriesAgeSpreadHours'),
                            temporalCoherenceScore=temporal_metrics.get('temporalCoherenceScore')
                        )
                        
                        # Store in background task
                        import asyncio
                        asyncio.create_task(
                            store_memory_prediction_log_async(
                                prediction_log=prediction_log,
                                session_token=session_token,
                                api_key=api_key
                            )
                        )
                        logger.info(f"Created MemoryPredictionLog background task for memory {memory_dict['objectId']}")
                        
                except Exception as e:
                    logger.error(f"Error creating MemoryPredictionLog: {e}")
                    # Continue processing even if prediction log fails

        # Step 3: Generate and store a memory graph for the memory item
        # Fetch user and workspace information
        try:
            from services.user_utils import User as UserService
            user_info = await UserService.get_user_async(user_id)
            logger.info(f"user_info: {user_info}")
            company = await UserService.get_company_async(user_id, workspace_id, session_token, api_key=api_key)
            logger.info(f"company: {company}")

            # Add user and workspace info to memory_dict metadata
            if not memory_dict.get('metadata'):
                memory_dict['metadata'] = {}

            memory_dict['metadata'].update({
                'creator_name': user_info.get('name') if user_info else None,
                'company': company
            })

            logger.info(f"Added creator and workspace info to memory: {memory_dict['metadata']}")
        except Exception as e:
            logger.warning(f"Could not fetch user/workspace info (non-critical): {e}")

        # Check if graph_override is provided - bypass LLM generation if so
        logger.info(f"🔍 DEBUG: graph_override value: {graph_override}")
        logger.info(f"🔍 DEBUG: graph_override type: {type(graph_override)}")
        if graph_override:
            logger.info("🎯 GRAPH OVERRIDE: Bypassing LLM generation and using developer-provided graph structure")
            
            # Convert Pydantic object to dict if needed
            if hasattr(graph_override, 'model_dump'):
                graph_override_dict = graph_override.model_dump()
            else:
                graph_override_dict = graph_override
                
            logger.info(f"🎯 GRAPH OVERRIDE: Nodes: {len(graph_override_dict.get('nodes', []))}, Relationships: {len(graph_override_dict.get('relationships', []))}")
            
            try:
                # Convert graph_override to the expected format
                from models.structured_outputs import LLMGraphNode, LLMGraphRelationship, NodeReference
                
                # Create nodes from graph_override
                nodes = []
                for node_data in graph_override_dict.get('nodes', []):
                    # Node ID is required (validated by Pydantic before reaching here)
                    # IMPORTANT: Store the manual ID in llmGenNodeId so relationships can find the nodes
                    # The 'id' field will be replaced with a UUID during processing, but llmGenNodeId preserves the manual ID
                    properties = node_data['properties'].copy()
                    properties['id'] = node_data['id']
                    properties['llmGenNodeId'] = node_data['id']  # Store manual ID for relationship matching

                    node = LLMGraphNode(
                        label=node_data['label'],
                        properties=properties
                    )
                    nodes.append(node)
                    logger.info(f"🎯 GRAPH OVERRIDE: Created {node.label} node with ID {node_data['id']} (stored in llmGenNodeId)")
                    logger.info(f"🔍 DEBUG: Node properties after ID assignment: {properties}")
                
                # Create relationships from graph_override
                relationships = []
                for rel_data in graph_override_dict.get('relationships', []):
                    relationship = LLMGraphRelationship(
                        type=rel_data['relationship_type'],
                        direction='->',
                        source=NodeReference(
                            label=next((n['label'] for n in graph_override_dict['nodes'] if n.get('id') == rel_data['source_node_id']), 'Unknown'),
                            id=rel_data['source_node_id']
                        ),
                        target=NodeReference(
                            label=next((n['label'] for n in graph_override_dict['nodes'] if n.get('id') == rel_data['target_node_id']), 'Unknown'),
                            id=rel_data['target_node_id']
                        )
                    )
                    relationships.append(relationship)
                    logger.info(f"🎯 GRAPH OVERRIDE: Created {relationship.type} relationship: {rel_data['source_node_id']} -> {rel_data['target_node_id']}")
                
                # Store the graph override nodes and relationships directly
                # For manual graphs, try to find registered schemas for node labels
                # This allows using schema unique_identifiers for better deduplication
                user_schema_for_manual = await self._get_schemas_for_manual_graph(
                    nodes=nodes,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    metadata=memory_dict.get('metadata', {})
                )
                
                await self.store_llm_generated_graph(
                    nodes=nodes,
                    relationships=relationships,
                    memory_item=memory_dict,
                    neo_session=neo_session,
                    workspace_id=workspace_id,
                    user_schema=user_schema_for_manual
                )
                
                logger.info("🎯 GRAPH OVERRIDE: Successfully stored developer-provided graph structure")
                
                # Create automatic EXTRACTED relationships from Memory to all manually created nodes
                memory_id = memory_dict.get('id')
                if memory_id and nodes:
                    logger.info(f"🔗 AUTO-CONNECT (MANUAL): Creating automatic EXTRACTED relationships from Memory {memory_id} to {len(nodes)} manually created nodes")
                    automatic_relationships = []
                    
                    for node in nodes:
                        # Use llmGenNodeId for relationship matching (consistent with manual graph relationships)
                        llm_gen_node_id = node.properties.get('llmGenNodeId')
                        node_uuid = node.properties.get('id')
                        if llm_gen_node_id:
                            # Create LLMGraphRelationship object for automatic connection
                            from models.structured_outputs import NodeReference
                            automatic_relationship = LLMGraphRelationship(
                                type="EXTRACTED",
                                direction="->",
                                source=NodeReference(label="Memory", id=memory_id),
                                target=NodeReference(label=node.label, id=llm_gen_node_id)
                            )
                            automatic_relationships.append(automatic_relationship)
                            logger.info(f"🔗 AUTO-CONNECT (MANUAL): Will create Memory -> EXTRACTED -> {node.label}({llm_gen_node_id}, uuid={node_uuid})")
                    
                    if automatic_relationships:
                        logger.info(f"🔗 AUTO-CONNECT (MANUAL): Creating {len(automatic_relationships)} automatic EXTRACTED relationships")
                        
                        # CRITICAL: Use FULL metadata with all tenant IDs and ACL fields
                        # The relationship creation needs tenant IDs to match nodes created in same session
                        metadata_for_extraction = json.loads(memory_dict['metadata']) if isinstance(memory_dict.get('metadata'), str) else memory_dict.get('metadata', {})
                        extraction_metadata = {
                            "extraction_method": "manual_graph_override",
                            "extracted_at": datetime.now(timezone.utc).isoformat(),
                            "user_id": metadata_for_extraction.get("user_id"),
                            "workspace_id": workspace_id or metadata_for_extraction.get("workspace_id"),
                            "organization_id": metadata_for_extraction.get("organization_id"),
                            "namespace_id": metadata_for_extraction.get("namespace_id"),
                            "user_read_access": metadata_for_extraction.get("user_read_access", []),
                            "user_write_access": metadata_for_extraction.get("user_write_access", []),
                            "workspace_read_access": metadata_for_extraction.get("workspace_read_access", []),
                            "workspace_write_access": metadata_for_extraction.get("workspace_write_access", []),
                            "role_read_access": metadata_for_extraction.get("role_read_access", []),
                            "role_write_access": metadata_for_extraction.get("role_write_access", []),
                            "organization_read_access": metadata_for_extraction.get("organization_read_access", []),
                            "organization_write_access": metadata_for_extraction.get("organization_write_access", []),
                            "namespace_read_access": metadata_for_extraction.get("namespace_read_access", []),
                            "namespace_write_access": metadata_for_extraction.get("namespace_write_access", []),
                            "external_user_read_access": metadata_for_extraction.get("external_user_read_access", []),
                            "external_user_write_access": metadata_for_extraction.get("external_user_write_access", []),
                        }
                        
                        for rel in automatic_relationships:
                            try:
                                result = await self._create_relationship(
                                    neo_session=neo_session, 
                                    relationship=rel, 
                                    common_metadata=extraction_metadata
                                )
                                # Don't log here - _create_relationship already logs success/failure
                            except Exception as rel_error:
                                logger.error(f"🔗 AUTO-CONNECT (MANUAL): ❌ Failed to create relationship {rel.source.id} -> {rel.target.id}: {rel_error}")
                
                # Create a mock schema response to maintain compatibility
                schema_response = {
                    "data": {
                        "nodes": nodes,
                        "relationships": relationships
                    },
                    "metrics": {
                        "schema_token_count_input": 0,
                        "schema_token_count_output": 0,
                        "schema_total_cost": 0,
                        "schema_total_tokens": 0
                    }
                }
                schema_total_cost = 0
                
                # Define schema_metrics for consistency with LLM path
                schema_metrics = {
                    "schema_token_count_input": 0,
                    "schema_token_count_output": 0,
                    "schema_total_cost": 0,
                    "schema_total_tokens": 0
                }
                
            except Exception as e:
                logger.error(f"🎯 GRAPH OVERRIDE ERROR: Failed to process graph override: {e}")
                return {
                    "status_code": 500,
                    "success": False,
                    "error": f"Failed to process graph override: {str(e)}",
                    "data": None
                }
        else:
            # Original LLM generation path
            logger.info("🤖 LLM GENERATION: Using automatic graph extraction")

            try:
                # Get developer's workspace ID for schema selection
                # Use the workspace_id that's already passed in - no need for redundant API call
                developer_workspace_id = workspace_id
                if developer_workspace_id:
                    logger.info(f"🔍 DEVELOPER WORKSPACE ID: {developer_workspace_id} for developer_user_id={developer_user_id}")
                else:
                    logger.warning(f"No workspace_id available for developer_user_id={developer_user_id}")
                
                # Extract organization and namespace context from memory metadata
                organization_id = memory_dict.get("metadata", {}).get("organization_id")
                namespace_id = memory_dict.get("metadata", {}).get("namespace_id")
                
                logger.info(f"🔒 MEMORY GRAPH: Passing multi-tenant context to schema generation - org_id={organization_id}, namespace_id={namespace_id}")
                
                schema_response = await chat_gpt.generate_memory_graph_schema_async(
                    memory_dict,
                    usecase_memory_item,  
                    neo_session,
                    workspace_id,
                    trimmed_related_memories,
                    user_id=user_id,
                    schema_ids=[schema_id] if schema_id else None,  # Pass schema_id as list for enforcement
                    property_overrides=property_overrides,  # Pass property overrides for node customization
                    developer_user_id=developer_user_id,  # Pass developer_user_id for schema selection
                    developer_workspace_id=developer_workspace_id,  # Pass developer's workspace ID for schema selection
                    organization_id=organization_id,  # Pass organization context for multi-tenant schema access
                    namespace_id=namespace_id  # Pass namespace context for multi-tenant schema access
                )

                if not schema_response:
                    return {
                        "status_code": 500,
                        "success": False,
                        "error": "Failed to generate memory graph schema",
                        "data": None
                    }

                # Get metrics directly from the response
                schema_metrics = schema_response.get("metrics", {
                    "schema_token_count_input": 0,
                    "schema_token_count_output": 0,
                    "schema_total_cost": 0,
                    "schema_total_tokens": 0
                })

                schema_total_cost = schema_metrics.get("schema_total_cost", 0)

            except Exception as e:
                logger.error(f"🤖 LLM GENERATION ERROR: {e}")
                return {
                    "status_code": 500,
                    "success": False,
                    "error": f"Failed to generate memory graph schema: {str(e)}",
                    "data": None
            }

        # Continue with common processing after either graph_override or LLM generation
        logger.info(f'Generated memory graph schema: {schema_response.get("data")})')
        logger.info(f'Schema metrics:'
                f'\n- Input tokens: {schema_response.get("metrics", {}).get("schema_token_count_input", 0)}'
                f'\n- Output tokens: {schema_response.get("metrics", {}).get("schema_token_count_output", 0)}'
                f'\n- Total cost: ${schema_total_cost:.8f}'
                f'\n- Total tokens: {schema_response.get("metrics", {}).get("schema_total_tokens", 0)}')

        # Initialize metrics dictionary if it doesn't exist
        if 'metrics' not in memory_dict:
            memory_dict['metrics'] = {'operation_costs': {}}

        # Add schema generation cost to total metrics
        memory_dict['metrics']['operation_costs']['schema_generation'] = schema_total_cost

        # Fix the total cost calculation:
        total_cost = (
            (BIGBIRD_EMBEDDING_COST) +
            (SENTENCE_BERT_COST) +
            usecase_metrics["usecase_total_cost"] +
            related_memories_metrics["related_memories_total_cost"] +
            schema_total_cost  # Use the schema_total_cost we calculated above
        )

        # Add complete metrics to memory_item
        memory_metrics = MemoryMetrics(
            total_cost=total_cost,
            token_size=memory_item_token_size,
            storage_size=memory_item_storage_size,
            operation_costs={
                'usecase_generation': usecase_metrics["usecase_total_cost"],
                'related_memories': related_memories_metrics["related_memories_total_cost"],
                'schema_generation': schema_metrics["schema_total_cost"],
                'bigbird_embedding': BIGBIRD_EMBEDDING_COST,
                'sentence_bert': SENTENCE_BERT_COST
            }
        )
        memory_dict['metrics'] = memory_metrics

        # When logging the final metrics:
        logger.info(f'Memory item metrics:'
                    f'\n- Total cost: ${total_cost:.8f}'  # Changed from .6f to .8f
                    f'\n- Token size: {memory_item_token_size}'
                    f'\n- Storage size: {memory_item_storage_size} bytes'
                    f'\n- Operation costs breakdown:'
                    f'\n  * Usecase generation: ${usecase_metrics["usecase_total_cost"]:.8f}'  # Changed from .6f to .8f
                    f'\n  * Related memories: ${related_memories_metrics["related_memories_total_cost"]:.8f}'  # Changed from .6f to .8f
                    f'\n  * Schema generation: ${schema_metrics["schema_total_cost"]:.8f}'  # Changed from .6f to .8f
                    f'\n  * BigBird embedding: ${BIGBIRD_EMBEDDING_COST:.8f}'  # Changed from .6f to .8f
                    f'\n  * Sentence-BERT: ${SENTENCE_BERT_COST:.8f}')  # Changed from .6f to .8f

        # Update memory item with metrics (skip when Temporal orchestrates metrics activity)
        if not memory_dict.get('temporal_orchestrated'):
            if memory_dict.get('objectId'):
                logger.info(f'Updating memory item with metrics: {total_cost}')
                logger.info(f'memory_dict: {memory_dict}')
                logger.info(f'api_key: {api_key}')
                logger.info(f'session_token: {session_token}')
                await update_memory_item(session_token, memory_dict, None, api_key=api_key)
            elif memory_dict['objectId']:
                logger.info(f'Updating memory item with metrics 2: {total_cost}')
                logger.info(f'memory_dict 2: {memory_dict}')
                logger.info(f'api_key 2: {api_key}')
                logger.info(f'session_token 2: {session_token}')
                await update_memory_item(session_token, memory_dict, None, api_key=api_key)
        else:
            logger.info('Temporal orchestrated run: skipping local metrics update (handled by activity)')

        # Update ActiveNodeRel cache after successful Neo4j graph generation
        if schema_response and schema_response.get("data") and user_id and workspace_id:
            try:
                logger.info(f"🔧 CACHE UPDATE: Updating ActiveNodeRel cache for user {user_id}, workspace {workspace_id}")
                
                # Extract ACL information from memory metadata or fetch from user
                memory_metadata = memory_dict.get('metadata', {})
                user_read_access = memory_metadata.get('user_read_access', [])
                workspace_read_access = memory_metadata.get('workspace_read_access', [])
                role_read_access = memory_metadata.get('role_read_access', [])
                organization_read_access = memory_metadata.get('organization_read_access', [])
                namespace_read_access = memory_metadata.get('namespace_read_access', [])
                
                # If ACL info is not in metadata, try to get it from User instance
                if not any([user_read_access, workspace_read_access, role_read_access, organization_read_access, namespace_read_access]):
                    try:
                        from services.user_utils import User
                        user_instance = User.get(user_id)
                        user_roles = user_instance.get_roles() if user_instance else []
                        user_workspace_ids = User.get_workspaces_for_user(user_id) if user_instance else []
                        
                        # Get organization and namespace info (if available)
                        user_organization_access = getattr(user_instance, 'organization_read_access', []) if user_instance else []
                        user_namespace_access = getattr(user_instance, 'namespace_read_access', []) if user_instance else []
                        
                        # Set ACL arrays
                        user_read_access = [user_id]
                        workspace_read_access = [str(wid) for wid in user_workspace_ids]
                        role_read_access = user_roles
                        organization_read_access = user_organization_access
                        namespace_read_access = user_namespace_access
                        
                        logger.info(f"🔧 ACL INFO: Fetched from user - roles: {len(role_read_access)}, workspaces: {len(workspace_read_access)}, org_access: {len(organization_read_access)}, ns_access: {len(namespace_read_access)}")
                    except Exception as e:
                        logger.warning(f"🔧 ACL WARNING: Failed to fetch user ACL info, using minimal access: {e}")
                        user_read_access = [user_id]
                        workspace_read_access = [workspace_id] if workspace_id else []
                        role_read_access = []
                        organization_read_access = []
                        namespace_read_access = []
                
                # Discover current Neo4j patterns for this user/workspace with full ACL context
                neo4j_patterns = await self._discover_neo4j_patterns_for_cache(
                    neo_session=neo_session,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    user_read_access=user_read_access,
                    workspace_read_access=workspace_read_access,
                    role_read_access=role_read_access,
                    organization_read_access=organization_read_access,
                    namespace_read_access=namespace_read_access
                )
                
                if neo4j_patterns:
                    # Update the ActiveNodeRel cache
                    from services.active_node_rel_service import get_active_node_rel_service
                    cache_service = get_active_node_rel_service()
                    
                    cache_updated = await cache_service.update_cached_schema(
                        user_object_id=user_id,
                        workspace_object_id=workspace_id,
                        neo4j_patterns=neo4j_patterns
                    )
                    
                    if cache_updated:
                        logger.info(f"🔧 CACHE UPDATE SUCCESS: Updated ActiveNodeRel cache with {len(neo4j_patterns)} patterns")
                    else:
                        logger.warning(f"🔧 CACHE UPDATE FAILED: Could not update ActiveNodeRel cache")
                else:
                    logger.info(f"🔧 CACHE UPDATE SKIP: No Neo4j patterns discovered")
                    
            except Exception as e:
                logger.warning(f"🔧 CACHE UPDATE ERROR: Failed to update ActiveNodeRel cache: {e}")
                # Don't fail the entire operation if cache update fails

        # When returning the final response, ensure all datetime objects are converted
        return {
            "status_code": 200,
            "success": True,
            "error": None,
            "data": {
                "goal_usecases": usecase_memory_item,
                "memory_graph": schema_response.get("data", {}),
                "related_memories": [
                    {
                        **memory.model_dump(),
                        'createdAt': self.datetime_handler(memory.createdAt) if hasattr(memory, 'createdAt') else None,
                        'updatedAt': self.datetime_handler(memory.updatedAt) if hasattr(memory, 'updatedAt') else None
                    }
                    for memory in related_memories
                ],
                "related_memories_relationships": relationship_result.get("relationships", []) if relationships_json else [],
                "metrics": memory_metrics
            }
        }

    async def _discover_neo4j_patterns_for_cache(
        self, 
        neo_session: AsyncSession, 
        user_id: str, 
        workspace_id: str,
        user_read_access: List[str] = None,
        workspace_read_access: List[str] = None,
        role_read_access: List[str] = None,
        organization_read_access: List[str] = None,
        namespace_read_access: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover current Neo4j node-relationship patterns for ActiveNodeRel cache.
        
        Returns:
            List of pattern dictionaries with keys: source, relationship, target, count
        """
        try:
            logger.info(f"🔍 PATTERN DISCOVERY: Discovering Neo4j patterns for user {user_id}, workspace {workspace_id}")
            
            # Enhanced query to discover patterns WITH actual property keys from the data
            # This collects all unique property keys for each node type in each pattern
            cypher_query = """
            MATCH (source)-[rel]->(target)
            WHERE (source.user_id = $user_id OR any(x IN coalesce(source.user_read_access, []) WHERE x IN $user_read_access) OR any(x IN coalesce(source.workspace_read_access, []) WHERE x IN $workspace_read_access) OR any(x IN coalesce(source.role_read_access, []) WHERE x IN $role_read_access) OR any(x IN coalesce(source.organization_read_access, []) WHERE x IN $organization_read_access) OR any(x IN coalesce(source.namespace_read_access, []) WHERE x IN $namespace_read_access))
            AND (target.user_id = $user_id OR any(x IN coalesce(target.user_read_access, []) WHERE x IN $user_read_access) OR any(x IN coalesce(target.workspace_read_access, []) WHERE x IN $workspace_read_access) OR any(x IN coalesce(target.role_read_access, []) WHERE x IN $role_read_access) OR any(x IN coalesce(target.organization_read_access, []) WHERE x IN $organization_read_access) OR any(x IN coalesce(target.namespace_read_access, []) WHERE x IN $namespace_read_access))
            WITH labels(source)[0] AS source_label, type(rel) AS relationship_type, labels(target)[0] AS target_label,
                 source, rel, target
            WHERE source_label IS NOT NULL AND relationship_type IS NOT NULL AND target_label IS NOT NULL
            AND source_label <> 'Memory' AND target_label <> 'Memory'
            WITH source_label, relationship_type, target_label,
                 source, rel, target,
                 [key IN keys(source) WHERE NOT key IN ['user_id', 'workspace_id', 'organization_id', 'namespace_id', 'createdAt', 'updatedAt', 'id', 'user_read_access', 'workspace_read_access', 'role_read_access', 'organization_read_access', 'namespace_read_access', 'external_user_read_access', 'user_write_access', 'workspace_write_access', 'role_write_access', 'organization_write_access', 'namespace_write_access', 'external_user_write_access', 'tenant_id', 'developer_id', 'session_id', 'api_key_id']] AS s_keys,
                 [key IN keys(rel) WHERE NOT key IN ['user_id', 'workspace_id', 'organization_id', 'namespace_id', 'createdAt', 'updatedAt', 'id', 'user_read_access', 'workspace_read_access', 'role_read_access', 'organization_read_access', 'namespace_read_access', 'external_user_read_access', 'user_write_access', 'workspace_write_access', 'role_write_access', 'organization_write_access', 'namespace_write_access', 'external_user_write_access', 'tenant_id', 'developer_id', 'session_id', 'api_key_id']] AS r_keys,
                 [key IN keys(target) WHERE NOT key IN ['user_id', 'workspace_id', 'organization_id', 'namespace_id', 'createdAt', 'updatedAt', 'id', 'user_read_access', 'workspace_read_access', 'role_read_access', 'organization_read_access', 'namespace_read_access', 'external_user_read_access', 'user_write_access', 'workspace_write_access', 'role_write_access', 'organization_write_access', 'namespace_write_access', 'external_user_write_access', 'tenant_id', 'developer_id', 'session_id', 'api_key_id']] AS t_keys
            WITH source_label, relationship_type, target_label,
                 collect(s_keys) AS s_key_sets, collect(r_keys) AS r_key_sets, collect(t_keys) AS t_key_sets,
                 count(*) AS pattern_count
            WITH source_label, relationship_type, target_label, pattern_count,
                 reduce(a=[], ks IN s_key_sets | a + ks) AS s_flat,
                 reduce(a=[], ks IN r_key_sets | a + ks) AS r_flat,
                 reduce(a=[], ks IN t_key_sets | a + ks) AS t_flat
            WITH source_label, relationship_type, target_label, pattern_count,
                 reduce(a=[], k IN s_flat | CASE WHEN k IN a THEN a ELSE a + k END) AS source_property_keys,
                 reduce(a=[], k IN r_flat | CASE WHEN k IN a THEN a ELSE a + k END) AS relationship_property_keys,
                 reduce(a=[], k IN t_flat | CASE WHEN k IN a THEN a ELSE a + k END) AS target_property_keys
            RETURN source_label, relationship_type, target_label, pattern_count,
                   source_property_keys, relationship_property_keys, target_property_keys
            ORDER BY pattern_count DESC
            LIMIT 100
            """
            
            result = await neo_session.run(
                cypher_query,
                user_id=user_id,
                workspace_id=workspace_id,
                user_read_access=user_read_access or [],
                workspace_read_access=workspace_read_access or [],
                role_read_access=role_read_access or [],
                organization_read_access=organization_read_access or [],
                namespace_read_access=namespace_read_access or []
            )
            
            patterns = []
            async for record in result:
                # Extract property keys from the enhanced query results (already filtered in Cypher)
                source_properties = record.get("source_property_keys", []) or []
                target_properties = record.get("target_property_keys", []) or []
                
                pattern = {
                    "source": record["source_label"],
                    "relationship": record["relationship_type"], 
                    "target": record["target_label"],
                    "count": record["pattern_count"],
                    "source_properties": source_properties,
                    "target_properties": target_properties
                }
                patterns.append(pattern)
                logger.info(f"🔍 Found pattern: {pattern['source']} -> {pattern['relationship']} -> {pattern['target']} (count: {pattern['count']}, source_props: {len(source_properties)}, target_props: {len(target_properties)})")
            
            logger.info(f"🔍 PATTERN DISCOVERY SUCCESS: Found {len(patterns)} unique patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"🔍 PATTERN DISCOVERY ERROR: Failed to discover Neo4j patterns: {e}")
            return []

    def datetime_handler(self, obj):
        """Handle serialization of datetime objects and other non-serializable types"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        try:
            # Handle Pydantic models
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            # Handle other objects
            return str(obj)
        except Exception as e:
            logger.warning(f"Could not serialize object of type {type(obj)}: {e}")
            return None

    def get_simplified_schema(self, memory_graph_schema):
        """
        Extracts simplified schema information from the complex memory graph schema.
        Returns tuple of (node_names, relationship_types)
        """
        node_names = []
        relationship_types = []

        if memory_graph_schema:
            # Extract node names from schema
            nodes = memory_graph_schema.get('properties', {}).get('nodes', {}).get('items', {}).get('anyOf', [])
            if nodes:
                node_names = [
                    node.get('properties', {}).get('label', {}).get('enum', [])[0]
                    for node in nodes
                    if node.get('properties', {}).get('label', {}).get('enum')
                ]

            # Extract relationship types from schema
            relationships = memory_graph_schema.get('properties', {}).get('relationships', {}).get('items', {})
            if relationships:
                relationship_types = relationships.get('properties', {}).get('type', {}).get('enum', [])

        return node_names, relationship_types

    def trim_and_filter_related_memories(self, related_memories: List[ParseStoredMemory], max_length: int = 600) -> List[Dict[str, str]]:
        """
        Trims the content of related memories to a maximum length and filters to only include content.

        Args:
            related_memories (List[ParseStoredMemory]): List of ParseStoredMemory objects to process
            max_length (int, optional): Maximum length for content. Defaults to 300.

        Returns:
            List[Dict[str, str]]: List of dictionaries containing trimmed memory data
        """
        if not related_memories:
            logger.info("No related memories provided for trimming")
            return []
            
        trimmed_memories: List[Dict[str, str]] = []
        for memory in related_memories:
            logger.debug(f"Processing memory for trimming: {memory.memoryId}")
            
            # Access content directly from ParseStoredMemory object
            content = memory.content
            if not content:
                logger.info(f"Memory {memory.memoryId} has no content, skipping")
                continue
                
            trimmed_memory = {
                'id': memory.memoryId,
                'content': (content[:max_length] + '...') if len(content) > max_length else content
            }
            
            trimmed_memories.append(trimmed_memory)
            logger.debug(f"Added trimmed memory: {trimmed_memory['id']}")

        logger.info(f'Trimmed memories: {trimmed_memories}')
        logger.info(f"Trimmed {len(related_memories)} memories to {len(trimmed_memories)} with content")
        return trimmed_memories
    
    def flatten_dict(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def validate_metadata(self, metadata):
        """Validate and flatten metadata for Neo4j storage."""
        if not metadata:
            logger.warning("Empty metadata received")
            return {}

        # Log the input metadata
        logger.info(f"Validating metadata: {json.dumps(metadata, indent=2)}")

        # Ensure all keys and values are of valid types
        validated_metadata = {}
        for key, value in metadata.items():
            # Skip None values
            if value is None or value == "None":
                    continue
                    
            if not isinstance(key, str):
                logger.warning(f"Skipping invalid key type: {key}")
                continue

            # Handle different value types
            if isinstance(value, (str, int, float, bool)):
                validated_metadata[key] = value
            elif isinstance(value, list):
                # Keep lists as lists for Neo4j
                validated_metadata[key] = value if value else []
            else:
                # Convert other types to strings
                validated_metadata[key] = str(value)

        logger.info(f"Validated metadata: {json.dumps(validated_metadata, indent=2)}")
        return validated_metadata

    async def add_memory_item_to_neo4j(
        self,
        memory_item: MemoryItem,
        neo_session: AsyncSession,
        memoryChunkIds: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Add a memory item to Neo4j, with robust fallback and error handling.
        """
        # In open-source, try to connect even if fallback_mode is set (it might have been set incorrectly)
        # Only skip if we're in cloud and fallback_mode is set
        import os
        papr_edition = os.getenv("PAPR_EDITION", "").lower()
        is_opensource = papr_edition == "opensource"
        
        if self.async_neo_conn.fallback_mode and not is_opensource:
            # Cloud edition: Skip if fallback mode is set
            logger.warning("Neo4j in fallback mode, storing memory item in fallback storage")
            self.fallback_storage[str(memory_item.id)] = memory_item
            return {"id": str(memory_item.id)}
        elif self.async_neo_conn.fallback_mode and is_opensource:
            # Open-source: Try to connect anyway - fallback mode might have been set incorrectly
            logger.info("Neo4j fallback mode is set, but attempting to store anyway (open-source)")

        try:
            memory_id = str(memory_item.id)
            chunk_ids = memoryChunkIds if memoryChunkIds else [memory_id]
            
            # Extract metadata for tenant scoping BEFORE node exists check
            metadata = (json.loads(memory_item.metadata)
                       if isinstance(memory_item.metadata, str)
                       else memory_item.metadata)
            custom_metadata = metadata.get('customMetadata', {})

            # Populate common_metadata to preserve organization_id, namespace_id, and access control lists
            # These are top-level fields in MemoryMetadata, not in customMetadata
            common_metadata = {
                "workspace_id": metadata.get("workspace_id"),
                "organization_id": metadata.get("organization_id"),
                "namespace_id": metadata.get("namespace_id"),
                "user_id": metadata.get("user_id"),
                "organization_read_access": metadata.get("organization_read_access"),
                "organization_write_access": metadata.get("organization_write_access"),
                "namespace_read_access": metadata.get("namespace_read_access"),
                "namespace_write_access": metadata.get("namespace_write_access"),
            }
            
            # Check if node exists WITH TENANT SCOPING
            existing_id = await self._node_exists(
                node_id=memory_id,
                node_type=NodeLabel.Memory,
                node_content=memory_item.content,
                neo_session=neo_session,
                workspace_id=common_metadata.get("workspace_id"),
                organization_id=common_metadata.get("organization_id"),
                namespace_id=common_metadata.get("namespace_id"),
                user_id=common_metadata.get("user_id")
            )
            if existing_id:
                logger.info(f"Memory node already exists with ID: {existing_id}")
                # 🔒 CRITICAL: Keep the existing node's ID - do NOT overwrite it!
                # The existing ID is the stable identifier used in relationships
                # Only update non-ACL/non-scoping properties (content, metadata, updatedAt, etc.)
                try:
                    # Prepare update properties (excluding id, and ACL/scoping fields)
                    memory_node = memory_item_to_node(memory_item, chunk_ids)
                    # memory_node.properties is already a dict (converted in memory_item_to_node)
                    update_props = memory_node.properties if isinstance(memory_node.properties, dict) else memory_node.properties.model_dump(exclude_none=True)
                    
                    # Remove fields that should NOT be updated
                    fields_to_preserve = [
                        'id',  # NEVER change the ID of an existing node
                        'workspace_id', 'organization_id', 'namespace_id',  # Tenant scoping
                        'user_id',  # Owner
                        'user_read_access', 'user_write_access',  # ACL
                        'workspace_read_access', 'workspace_write_access',
                        'organization_read_access', 'organization_write_access',
                        'namespace_read_access', 'namespace_write_access',
                        'role_read_access', 'role_write_access',
                        'external_user_read_access', 'external_user_write_access',
                        'createdAt'  # Don't change creation timestamp
                    ]
                    for field in fields_to_preserve:
                        update_props.pop(field, None)
                    
                    # Add updatedAt timestamp
                    update_props['updatedAt'] = datetime.now(timezone.utc).isoformat()
                    
                    update_query = """
                        MATCH (n:Memory {id: $existing_id})
                        SET n += $update_props
                        RETURN n
                    """
                    result = await neo_session.run(update_query, existing_id=existing_id, update_props=update_props)
                    if result is None:
                        logger.error("Neo4j session returned None result, connection may be closed")
                        self.async_neo_conn.fallback_mode = True
                        return {"id": existing_id}  # Return existing ID, not new memory_id
                    
                    record = await result.single()
                    if record:
                        logger.info(f"✅ Updated existing Memory node {existing_id} (preserved ID, updated content/metadata)")
                        # If operation succeeded and we're in open-source with fallback_mode set, reset it
                        if is_opensource and self.async_neo_conn.fallback_mode:
                            logger.info("Successfully updated memory item in Neo4j - resetting fallback mode")
                            self.async_neo_conn.fallback_mode = False
                        return {"id": existing_id}  # Return existing ID, not new memory_id
                    else:
                        logger.warning(f"Failed to update existing node {existing_id}")
                        return {"id": existing_id}  # Return existing ID, not new memory_id
                except Exception as e:
                    # Check if it's a connection error
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ['closed connection', 'connection', 'driver', 'session']):
                        logger.error(f"Neo4j connection error updating existing node: {e}")
                        # Only set fallback mode if we're in open-source
                        if is_opensource:
                            self.async_neo_conn.fallback_mode = True
                    else:
                        logger.error(f"Error updating existing node: {e}")
                    return {"id": existing_id}  # Return existing ID, not new memory_id
            # Convert MemoryItem to Node
            memory_node = memory_item_to_node(memory_item, chunk_ids)

            # Use existing _create_node method
            result = await self._create_node(
                node=memory_node,
                common_metadata=common_metadata,
                neo_session=neo_session
            )
            # If operation succeeded and we're in open-source with fallback_mode set, reset it
            if is_opensource and self.async_neo_conn.fallback_mode and result:
                logger.info("Successfully stored memory item in Neo4j - resetting fallback mode")
                self.async_neo_conn.fallback_mode = False
            return result
        except Exception as e:
            logger.error(f"Error adding memory node to Neo4j: {e}", exc_info=True)
            # Only set fallback mode if we're in open-source (cloud should handle this differently)
            if is_opensource:
                self.async_neo_conn.fallback_mode = True
                self.fallback_storage[str(memory_item.id)] = memory_item
            return {"id": str(memory_item.id)}

    async def batch_create_memory_nodes(
        self,
        memory_items: List[MemoryItem],
        neo_session: AsyncSession,
        memoryChunkIds_list: Optional[List[List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch create memory nodes in Neo4j using UNWIND for efficiency.
        
        Args:
            memory_items: List of MemoryItem objects to create
            neo_session: Neo4j async session
            memoryChunkIds_list: List of chunk ID lists, one per memory
            
        Returns:
            List of dicts with node IDs
        """
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, storing memory items in fallback storage")
            results = []
            for memory_item in memory_items:
                self.fallback_storage[str(memory_item.id)] = memory_item
                results.append({"id": str(memory_item.id)})
            return results

        try:
            logger.info(f"📦 Batch creating {len(memory_items)} memory nodes in Neo4j")
            
            # Prepare nodes data for UNWIND
            nodes_data = []
            for idx, memory_item in enumerate(memory_items):
                memory_id = str(memory_item.id)
                chunk_ids = (memoryChunkIds_list[idx] if memoryChunkIds_list and idx < len(memoryChunkIds_list) 
                           else [memory_id])
                
                # Extract metadata
                metadata = (json.loads(memory_item.metadata)
                           if isinstance(memory_item.metadata, str)
                           else memory_item.metadata)
                custom_metadata = metadata.get('customMetadata', {})
                
                # Build node data
                node_data = {
                    "id": memory_id,
                    "content": memory_item.content,
                    "type": metadata.get("type", "text"),
                    "memoryChunkIds": chunk_ids,
                    "organization_id": metadata.get("organization_id"),
                    "namespace_id": metadata.get("namespace_id"),
                    "organization_read_access": metadata.get("organization_read_access"),
                    "organization_write_access": metadata.get("organization_write_access"),
                    "namespace_read_access": metadata.get("namespace_read_access"),
                    "namespace_write_access": metadata.get("namespace_write_access"),
                    "workspace_id": metadata.get("workspace_id"),
                    "user_id": metadata.get("user_id"),
                    "createdAt": metadata.get("createdAt", datetime.now(timezone.utc).isoformat()),
                }
                
                # Add optional metadata fields
                for key in ["title", "sourceUrl", "location", "conversationId"]:
                    if key in metadata:
                        node_data[key] = metadata[key]
                
                nodes_data.append(node_data)
            
            # Create all nodes in a single UNWIND transaction
            query = """
            UNWIND $nodes_data AS node_data
            CREATE (m:Memory {
                id: node_data.id,
                content: node_data.content,
                type: node_data.type,
                memoryChunkIds: node_data.memoryChunkIds,
                organization_id: node_data.organization_id,
                namespace_id: node_data.namespace_id,
                workspace_id: node_data.workspace_id,
                user_id: node_data.user_id,
                createdAt: node_data.createdAt,
                title: node_data.title,
                sourceUrl: node_data.sourceUrl,
                location: node_data.location,
                conversationId: node_data.conversationId
            })
            RETURN m.id as id
            """
            
            result = await neo_session.run(query, nodes_data=nodes_data)
            records = await result.values()
            
            logger.info(f"✅ Batch created {len(records)} memory nodes in Neo4j")
            
            return [{"id": record[0]} for record in records]
            
        except Exception as e:
            logger.error(f"❌ Error batch creating memory nodes in Neo4j: {e}", exc_info=True)
            self.async_neo_conn.fallback_mode = True
            # Fallback: store in fallback storage
            results = []
            for memory_item in memory_items:
                self.fallback_storage[str(memory_item.id)] = memory_item
                results.append({"id": str(memory_item.id)})
            return results

    async def update_memory_item_in_neo4j(
        self, 
        memory_item_dict: dict, 
        neo_session: AsyncSession,
        memory_type: Optional[str] = None, 
        memoryChunkIds: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:        
        """
        Updates an existing memory item's metadata in Neo4j.
        Excludes the 'content' field from updates.

        Parameters:
            memory_item (dict): A dictionary representing the memory item.
            neo_session (AsyncSession): The Neo4j session to use for the operation
            memory_type (str, optional): The new type to set for the memory item. Defaults to None.
            memoryChunkIds (list, optional): List of Pinecone chunk IDs to update. Defaults to None.

        Returns:
            dict: The properties of the updated node, or None if no update occurred.
        """
        logger.info(f"Updating memory item in Neo4j with ID: {memory_item_dict['id']}")
        # Define the metadata fields to include
        metadata_fields = [
            "id",
            "user_id",
            "pageId",
            "hierarchical_structures",
            "type",
            "title",
            "topics",
            "conversationId",
            "prompt",
            "imageURL",
            "sourceType",
            "sourceUrl",
            "workspace_id",
            "user_id",
            "user_read_access",
            "user_write_access",
            "workspace_read_access",
            "workspace_write_access",
            "role_read_access",
            "role_write_access",
            "namespace_read_access",
            "namespace_write_access",
            "organization_read_access",
            "organization_write_access",
            "content"
        ]

        # Define key mappings: Parse Server keys to Neo4j keys
        key_mappings = {
            'emojiTags': 'emoji_tags',
            'emoji tags': 'emoji_tags',
            'emotionTags': 'emotion_tags',
            'emotion tags': 'emoji_tags',
            'hierarchicalStructures': 'hierarchical_structures',
            'hierarchical structures': 'hierarchical_structures'
        }

        try:
            # Extract relevant metadata
            relevant_metadata = {field: memory_item_dict.get(field) for field in metadata_fields}
            logger.info(f"Extracted relevant metadata: {relevant_metadata}")

            # Prepare the properties for the updated node
            properties = {}
            
            # Only add properties that exist in memory_item_dict
            possible_properties = [
                'content',
                'title',
                'user_read_access',
                'user_write_access',
                'workspace_read_access',
                'workspace_write_access',
                'role_read_access',
                'role_write_access',
                'sourceUrl',
                'memoryChunkIds',
                "type"
            ]

            for prop in possible_properties:
                if prop in memory_item_dict and memory_item_dict[prop] is not None:
                    properties[prop] = memory_item_dict[prop]

            if memory_type:
                properties["type"] = memory_type
            elif "type" in memory_item_dict:
                properties["type"] = memory_item_dict["type"]

            # Process metadata fields with key mapping
            for field in metadata_fields:
                # Try both original and mapped keys
                value = None
                for parse_key, neo_key in key_mappings.items():
                    if neo_key == field and parse_key in memory_item_dict:
                        value = memory_item_dict[parse_key]
                        break
                if value is None:
                    value = memory_item_dict.get(field)

                if value is not None and field != 'id':
                    # Handle specific field conversions
                    if field in ['emoji_tags', 'emotion_tags']:
                        if isinstance(value, str):
                            properties[field] = [tag.strip() for tag in value.split(',') if tag.strip()]
                        elif isinstance(value, list):
                            properties[field] = value
                    else:
                        properties[field] = value

            # Update memoryChunkIds if provided
            if memoryChunkIds is not None:
                properties["memoryChunkIds"] = memoryChunkIds

            logger.info(f"Properties to update: {properties}")

            # First, check if the node exists
            check_query = """
                MATCH (n:Memory {id: $id})
                RETURN n
            """
            check_result = await neo_session.run(check_query, id=str(memory_item_dict['id']))
            check_record = await check_result.single()
            
            if not check_record:
                logger.error(f"Node with ID {memory_item_dict['id']} not found in Neo4j")
                return None

            logger.info("Found existing node in Neo4j, proceeding with update")
            logger.info(f"Existing node properties: {dict(check_record['n'])}")

            logger.info(f"Properties to update: {properties}")

            # Prepare the Cypher query - using direct SET
            query = """
                MATCH (n:Memory {id: $id})
                SET n += $properties
                RETURN n
            """

            logger.info(f"Executing Cypher query: {query}")
            logger.info(f"Query parameters - id: {str(memory_item_dict['id'])}")
            logger.info(f"Query parameters - properties: {properties}")

            try:
                # Execute the query with merged properties
                result = await neo_session.run(
                    query,
                    id=str(memory_item_dict['id']),
                    properties=properties   
                )
                record = await result.single()
                logger.info(f"Record type: {type(record)}")
                logger.info(f"Record keys: {record.keys() if record else 'No record'}")
                logger.info(f"Record values: {record.values() if record else 'No record'}")
                
                if record:
                    # Try to get node data, handling different record structures
                    node_data = None
                    if 'n' in record:
                        node_data = dict(record['n'])
                    elif len(record) > 0:
                        # If 'n' isn't explicitly keyed but we have data
                        node_data = dict(record[0])
                    
                    if node_data:
                        logger.info(f"Successfully updated node properties: {node_data}")
                        return node_data
                    else:
                        # We got a record but couldn't extract node data
                        logger.warning(f"Update succeeded but couldn't extract node data from record: {record}")
                        # Return the properties we sent since we know they were applied
                        return properties
                else:
                    logger.error(f"No record returned for memory item with ID {memory_item_dict['id']}")
                    return None
            except Exception as query_error:
                logger.error(f"Error executing Neo4j query: {query_error}", exc_info=True)
                raise

        except Exception as e:
            logger.error(f"Error updating memory node in Neo4j: {e}", exc_info=True)
            raise
    
    async def lookup_memory_by_client_msg_id(self, client_msg_id: str, neo_session: AsyncSession) -> Optional[str]:
        """
        Asynchronously lookup a memory by its client message ID.
        
        Args:
            client_msg_id (str): The client message ID to look up
            
        Returns:
            Optional[str]: The memory ID if found, None otherwise
        """
        try:
            # Ensure Neo4j connection is initialized
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, cannot lookup memory by client_msg_id")
                return None
            #driver = await self.async_neo_conn.get_driver()
            
            #async with driver.session() as session:
            result = await neo_session.run(
                "MATCH (m:Memory {client_msg_id: $client_msg_id}) RETURN m.id AS id",
                client_msg_id=client_msg_id
            )
            record = await result.single()
            if record:
                return record['id']
            return None
            
        except Exception as e:
            logger.error(f"Error looking up memory by client_msg_id: {str(e)}")
            return None


    
    async def find_related_memory_items_async(
        self, 
        session_token: str, 
        query: str,  
        user_id: str,  # This should always be the resolved user to search as
        chat_gpt: "ChatGPTCompletion", 
        neo_session: Optional[AsyncSession] = None,  # Made optional
        metadata: Optional[MemoryMetadata] = None, 
        relation_type: str = None, 
        project_id: str = None, 
        skip_neo: bool = True, 
        exclude_memory_id: str = None,
        user_workspace_ids: Optional[List[str]] = None,
        user_roles: Optional[List[str]] = None,
        reranking_config: Optional[RerankingConfig] = None, 
        api_key: Optional[str] = None,
        search_request: Optional[SearchRequest] = None,
        context: Optional[List[ContextItem]] = None,
        top_k: int = 20,
        top_k_neo: int = 15,
        legacy_route: bool = True,
        cached_schema: Optional[Dict[str, Any]] = None  # ActivePatterns for Neo4j Cypher generation
    ) -> RelatedMemoryResult:
        """
        Find related memory items using various sources (Pinecone, BigBird, Neo4j).
        Returns structured results including memory items and neo nodes.
        """
        fetch_start = time.time()  # Initialize at the start of the method
        start_time = time.time()
        neoQuery = None
        mem_source_dict = {}
        if reranking_config:
            reranking_enabled = reranking_config.reranking_enabled
            reranking_model = reranking_config.reranking_model
            reranking_provider = reranking_config.reranking_provider
        else:
            reranking_enabled = False
            reranking_model = None
            reranking_provider = None
        result = RelatedMemoryResult(
            memory_items=[],
            neo_nodes=[],
            neo_context=None,
            neo_query=None,  # Added this field
            memory_source_info=MemorySourceInfo(memory_id_source_location=[])  # Initialize with empty list
        )

        if search_request:
            logger.info(
                "SearchRequest snapshot: org_id=%s, namespace_id=%s, "
                "metadata_org=%s, metadata_namespace=%s, metadata_custom_keys=%s",
                getattr(search_request, "organization_id", None),
                getattr(search_request, "namespace_id", None),
                getattr(metadata, "organization_id", None) if metadata else None,
                getattr(metadata, "namespace_id", None) if metadata else None,
                list((getattr(metadata, "customMetadata", None) or {}).keys()) if metadata else [],
            )
        else:
            logger.info(
                "No SearchRequest provided; metadata org_id=%s, namespace_id=%s, custom_keys=%s",
                getattr(metadata, "organization_id", None) if metadata else None,
                getattr(metadata, "namespace_id", None) if metadata else None,
                list((getattr(metadata, "customMetadata", None) or {}).keys()) if metadata else [],
            )

        # Only ensure connection if we don't have a session
        if not neo_session and not skip_neo:
            start_neo_time = time.time()
            neo_time = (time.time() - start_neo_time) * 1000
            logger.warning(f"Neo4j connection ensure took {neo_time:.2f}ms (fallback_mode: {self.async_neo_conn.fallback_mode})")
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, skipping Neo4j query but continuing with Qdrant search")
                # Set skip_neo to True so we skip Neo4j operations but continue with Qdrant search
                skip_neo = True

        # Only use the query for better accuracy
        query_context_combined = query
        # User info (user_roles and user_workspace_ids) is now passed in as parameters from enhanced authentication
        # This eliminates the need for separate user info retrieval calls
        if user_workspace_ids is None:
            user_workspace_ids = []
        if user_roles is None:
            user_roles = []
            
        logger.info(f"Using user_roles from parameters: {user_roles}")
        logger.info(f"Using user_workspace_ids from parameters: {user_workspace_ids}")
        
        
        
        # Build optimized ACL filter conditions for speed
        # Separate ACL filters (OR logic) from scoping filters (AND logic)
        acl_conditions = []  # User access filters - use OR (user has access if ANY condition matches)
        scoping_conditions = []  # Multi-tenant scoping - use AND (must match ALL conditions)
        
        # Add user_id filter if provided - use exact match for speed
        if user_id and isinstance(user_id, str):
            acl_conditions.append(qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id)))
            acl_conditions.append(qmodels.FieldCondition(key="user_read_access", match=qmodels.MatchAny(any=[user_id])))
        
        # Only add workspace filter if user_workspace_ids is small (for speed)
        if user_workspace_ids and len(user_workspace_ids) <= 10:
            acl_conditions.append(qmodels.FieldCondition(key="workspace_read_access", match=qmodels.MatchAny(any=user_workspace_ids)))
        
        # Only add roles filter if user_roles is small (for speed)
        if user_roles and len(user_roles) <= 10:
            acl_conditions.append(qmodels.FieldCondition(key="role_read_access", match=qmodels.MatchAny(any=user_roles)))
        
        # Add external_user_id filter if provided
        #if search_request and getattr(search_request, 'external_user_id', None):
        #    ext_uid = getattr(search_request, 'external_user_id')
        #    if isinstance(ext_uid, str):
        #        acl_conditions.append(qmodels.FieldCondition(key="external_user_id", match=qmodels.MatchValue(value=ext_uid)))
        
        # Add organization_read_access to ACL conditions (OR logic - user has access to these orgs)
        # NOTE: We do NOT add organization_id as a MUST condition to allow legacy memories without organization_id
        # User access is determined by user_id OR user_read_access, regardless of organization_id
        user_organization_ids = []
        if search_request and getattr(search_request, 'organization_id', None):
            org_id = getattr(search_request, 'organization_id')
            if isinstance(org_id, str):
                user_organization_ids.append(org_id)
                # Add to ACL conditions (OR) but NOT as scoping condition (MUST)
                # This allows memories with matching org_id OR legacy memories without org_id
                logger.info(f"Including organization_read_access filter for org_id={org_id} (optional, includes legacy)")
        
        if user_organization_ids and len(user_organization_ids) <= 10:
            acl_conditions.append(qmodels.FieldCondition(key="organization_read_access", match=qmodels.MatchAny(any=user_organization_ids)))
        
        # Add namespace_read_access to ACL conditions (OR logic - user has access to these namespaces)
        # NOTE: We do NOT add namespace_id as a MUST condition to allow legacy memories without namespace_id
        # User access is determined by user_id OR user_read_access, regardless of namespace_id
        user_namespace_ids = []
        if search_request and getattr(search_request, 'namespace_id', None):
            ns_id = getattr(search_request, 'namespace_id')
            if isinstance(ns_id, str):
                user_namespace_ids.append(ns_id)
                # Add to ACL conditions (OR) but NOT as scoping condition (MUST)
                # This allows memories with matching namespace_id OR legacy memories without namespace_id
                logger.info(f"Including namespace_read_access filter for namespace_id={ns_id} (optional, includes legacy)")
        
        if user_namespace_ids and len(user_namespace_ids) <= 10:
            acl_conditions.append(qmodels.FieldCondition(key="namespace_read_access", match=qmodels.MatchAny(any=user_namespace_ids)))
        
        # Combine filters: ACL conditions with OR, scoping conditions with AND
        # Final filter structure: (acl_condition1 OR acl_condition2 OR ...) AND scoping_condition1 AND scoping_condition2 AND ...
        filter_parts = []
        
        # ACL filter with OR logic (user needs access through ANY of these)
        if acl_conditions:
            if len(acl_conditions) == 1:
                filter_parts.append(acl_conditions[0])
            else:
                filter_parts.append(qmodels.Filter(should=acl_conditions))
        
        # Scoping filters with AND logic (must match ALL scoping conditions)
        filter_parts.extend(scoping_conditions)
        
        # Create final filter with AND logic for all parts
        if len(filter_parts) == 0:
            acl_filter = None
        elif len(filter_parts) == 1:
            # Single filter - use it directly if it's a Filter, or wrap if it's a condition
            if isinstance(filter_parts[0], qmodels.Filter):
                acl_filter = filter_parts[0]
            else:
                acl_filter = qmodels.Filter(must=[filter_parts[0]])
        else:
            # Multiple parts - combine with AND logic
            must_conditions = []
            for part in filter_parts:
                if isinstance(part, qmodels.Filter):
                    must_conditions.append(part)
                else:
                    must_conditions.append(part)
            acl_filter = qmodels.Filter(must=must_conditions)

        # Metadata filter
        metadata_conditions = []
        bigbird_memory_info = [];
        '''
        if metadata:
            meta_dict = metadata.model_dump(exclude_unset=True)
            for k, v in meta_dict.items():
                if k == 'customMetadata' and isinstance(v, dict):
                    for ck, cv in v.items():
                        metadata_conditions.append(qmodels.FieldCondition(key=f"metadata.{ck}", match=qmodels.MatchValue(value=cv)))
                elif v is not None:
                     metadata_conditions.append(qmodels.FieldCondition(key=f"metadata.{k}", match=qmodels.MatchValue(value=v)))
        '''
        if metadata is not None:
                    metadata_conditions = []
        if metadata is not None:
            if getattr(metadata, 'topics', None):
                topics_val = metadata.topics
                # Handle list fields properly - use MatchAny for lists, MatchValue for strings
                if isinstance(topics_val, list):
                    metadata_conditions.append(qmodels.FieldCondition(key="topics", match=qmodels.MatchAny(any=topics_val)))
                else:
                    metadata_conditions.append(qmodels.FieldCondition(key="topics", match=qmodels.MatchValue(value=topics_val)))
            if getattr(metadata, 'hierarchical_structures', None):
                metadata_conditions.append(qmodels.FieldCondition(key="hierarchical_structures", match=qmodels.MatchValue(value=metadata.hierarchical_structures)))
            if getattr(metadata, 'location', None):
                metadata_conditions.append(qmodels.FieldCondition(key="location", match=qmodels.MatchValue(value=metadata.location)))
            if getattr(metadata, 'emoji_tags', None):
                emoji_tags_val = metadata.emoji_tags
                # Handle list fields properly - use MatchAny for lists, MatchValue for strings
                if isinstance(emoji_tags_val, list):
                    metadata_conditions.append(qmodels.FieldCondition(key="emoji_tags", match=qmodels.MatchAny(any=emoji_tags_val)))
                else:
                    metadata_conditions.append(qmodels.FieldCondition(key="emoji_tags", match=qmodels.MatchValue(value=emoji_tags_val)))
            if getattr(metadata, 'emotion_tags', None):
                emotion_tags_val = metadata.emotion_tags
                # Handle list fields properly - use MatchAny for lists, MatchValue for strings
                if isinstance(emotion_tags_val, list):
                    metadata_conditions.append(qmodels.FieldCondition(key="emotion_tags", match=qmodels.MatchAny(any=emotion_tags_val)))
                else:
                    metadata_conditions.append(qmodels.FieldCondition(key="emotion_tags", match=qmodels.MatchValue(value=emotion_tags_val)))
            if getattr(metadata, 'conversationId', None):
                metadata_conditions.append(qmodels.FieldCondition(key="conversationId", match=qmodels.MatchValue(value=metadata.conversationId)))
            if getattr(metadata, 'role', None):
                role_val = metadata.role
                # Handle enum values - convert to string if needed
                if hasattr(role_val, 'value'):
                    role_val = role_val.value
                metadata_conditions.append(qmodels.FieldCondition(key="role", match=qmodels.MatchValue(value=role_val)))
            if getattr(metadata, 'category', None):
                category_val = metadata.category
                # Handle enum values - convert to string if needed
                if hasattr(category_val, 'value'):
                    category_val = category_val.value
                metadata_conditions.append(qmodels.FieldCondition(key="category", match=qmodels.MatchValue(value=category_val)))
            if getattr(metadata, 'customMetadata', None):
                for k, v in (metadata.customMetadata or {}).items():
                    # Skip organization_id and namespace_id - these are handled in ACL conditions, not metadata filters
                    # We want them to be optional (OR logic) not required (AND logic) to include legacy memories
                    if k in ['organization_id', 'namespace_id']:
                        logger.info(f"Skipping {k} from metadata filter - handled in ACL conditions for legacy memory support")
                        continue
                    
                    # Lists: use MatchAny for array membership
                    if isinstance(v, list):
                        metadata_conditions.append(
                            qmodels.FieldCondition(
                                key=f"{k}",
                                match=qmodels.MatchAny(any=v)
                            )
                        )
                        continue

                    # Booleans: exact match
                    if isinstance(v, bool):
                        metadata_conditions.append(
                            qmodels.FieldCondition(
                                key=f"{k}",
                                match=qmodels.MatchValue(value=v)
                            )
                        )
                        continue

                    # Numerics: interpret as a lower-bound (>=) filter by default
                    if isinstance(v, (int, float)):
                        metadata_conditions.append(
                            qmodels.FieldCondition(
                                key=f"{k}",
                                range=qmodels.Range(gte=v)
                            )
                        )
                        continue

                    # Dict with explicit range operators: {gte, lte, gt, lt}
                    if isinstance(v, dict):
                        range_kwargs = {}
                        if 'gte' in v:
                            range_kwargs['gte'] = v['gte']
                        if 'lte' in v:
                            range_kwargs['lte'] = v['lte']
                        if 'gt' in v:
                            range_kwargs['gt'] = v['gt']
                        if 'lt' in v:
                            range_kwargs['lt'] = v['lt']
                        if range_kwargs:
                            metadata_conditions.append(
                                qmodels.FieldCondition(
                                    key=f"{k}",
                                    range=qmodels.Range(**range_kwargs)
                                )
                            )
                            continue

                    # Fallback: exact match (e.g., strings)
                    metadata_conditions.append(
                        qmodels.FieldCondition(
                            key=f"{k}",
                            match=qmodels.MatchValue(value=v)
                        )
                    )
            
        # Combine ACL and metadata filters
        final_filter = acl_filter
        if metadata_conditions:
            final_filter = qmodels.Filter(
                must=[acl_filter] + metadata_conditions
            )
        else:
            final_filter = acl_filter
        
        logger.info(f"Final Qdrant filter for search: {final_filter.model_dump()}")

        logger.info(f"legacy_route flag: {legacy_route}")
        embedding_start = time.time()
        
        # Check if we're in open-source edition - skip BigBird/Pinecone in open-source
        # In open-source, BigBird and sentence embeddings require API URLs that don't exist
        # (HUGGING_FACE_API_URL_BIG_BIRD, HUGGING_FACE_API_URL_SENTENCE_BERT)
        # These are cloud-only features, so we force legacy_route to False in open-source
        # to prevent "Invalid type for url" errors when these URLs are None
        import os
        papr_edition = os.getenv("PAPR_EDITION", "").lower()
        is_opensource = papr_edition == "opensource"
        
        # In open-source, force legacy_route to False (skip BigBird/Pinecone)
        # These features require cloud-only API URLs that don't exist in open-source
        if is_opensource and legacy_route:
            logger.info("Open-source edition detected: Skipping BigBird and Pinecone embeddings (cloud-only features)")
            logger.info("Forcing legacy_route to False to prevent API URL errors")
            legacy_route = False
        
        # Initialize variables for both routes
        memory_item_ids = []
        predicted_grouped_memory_ids = [] #bigbird memory ids in legacy and qdrant qwen with is_predicted_grouped set to true
        bigbird_memory_info = []
        similar_embeddings_results = {}
        neo_memory_ids = []

        import re
        def strip_chunk_suffix(memory_id):
            return re.sub(r'_[0-9]+$', '', memory_id)
        
        if legacy_route:
            # Run all three embeddings in parallel
            query_context_embedding_tuple, bigbird_embedding_tuple, query_qdrant_embedding_list = await asyncio.gather(
                self.embedding_model.get_sentence_embedding(query_context_combined, max_retries=3, retry_delay=1),
                self.embedding_model.get_bigbird_embedding(query_context_combined, max_retries=3, retry_delay=1),
                self.embedding_model.get_qwen_embedding_4b(query_context_combined, max_retries=3, retry_delay=1)
            )
            
            # Extract embeddings from tuples (first element is embeddings, second is chunks)
            query_context_embedding = query_context_embedding_tuple[0][0] if query_context_embedding_tuple and len(query_context_embedding_tuple[0]) > 0 else []
            bigbird_embedding = bigbird_embedding_tuple[0][0] if bigbird_embedding_tuple and len(bigbird_embedding_tuple[0]) > 0 else []
            query_qdrant_embedding = query_qdrant_embedding_list[0][0] if query_qdrant_embedding_list and len(query_qdrant_embedding_list[0]) > 0 else None
            
            # Log first few values of each embedding to confirm they're actual embeddings
            logger.info(f"query_context_embedding first 5 values: {query_context_embedding[:5] if isinstance(query_context_embedding, list) else 'Not a list'}")
            logger.info(f"bigbird_embedding first 5 values: {bigbird_embedding[:5] if isinstance(bigbird_embedding, list) else 'Not a list'}")
            logger.info(f"query_qdrant_embedding first 5 values: {query_qdrant_embedding[:5] if isinstance(query_qdrant_embedding, list) else 'Not a list'}")
            
            # Also log the query text for comparison
            logger.info(f"query_context_combined first 50 chars: {query_context_combined[:50]}")
            embedding_time = (time.time() - embedding_start) * 1000
            logger.warning(f"Embedding generation (legacy route) took {embedding_time:.2f}ms")
            search_start = time.time()
            try:
                logger.info("Running Pinecone, BigBird, and Qdrant searches in parallel with asyncio.gather...")
                
                # Convert Qdrant filter to Pinecone-compatible format
                pinecone_filter = self._convert_qdrant_filter_to_pinecone(final_filter)
                
                # Debug logging to check embedding types
                logger.info(f"query_context_embedding type: {type(query_context_embedding)}, length: {len(query_context_embedding) if isinstance(query_context_embedding, list) else 'N/A'}")
                logger.info(f"bigbird_embedding type: {type(bigbird_embedding)}, length: {len(bigbird_embedding) if isinstance(bigbird_embedding, list) else 'N/A'}")
                logger.info(f"query_qdrant_embedding type: {type(query_qdrant_embedding)}, length: {len(query_qdrant_embedding) if isinstance(query_qdrant_embedding, list) else 'N/A'}")
                
                # Increase top_k for vector sources to account for chunking
                chunk_factor = 3  # Assume average 3 chunks per memory
                vector_top_k = top_k * chunk_factor  # Get 3x more from vector sources
                
                similar_embeddings_results, bigbird_memory_info, qdrant_embeddings_results, qdrant_fallback_results = await asyncio.gather(
                    self.get_pinecone_related_memories_async(
                        query_context_embedding,
                        pinecone_filter,
                        top_k=vector_top_k  # Increased for chunking
                    ),
                    self.get_bigbird_related_memories_async(
                        bigbird_embedding,
                        pinecone_filter,
                        user_id,
                        top_k=vector_top_k  # Normal amount for semantic search
                    ),
                    self.get_qdrant_related_memories_async(
                        query_qdrant_embedding,
                        final_filter,
                        top_k=vector_top_k  # Increased for chunking
                    ),
                    self.get_qdrant_related_memories_async_fallback(
                        final_filter,
                        top_k=vector_top_k  # Increased for chunking
                    )
                )
                
                # Log raw results from each source immediately after gather
                pinecone_count = len(similar_embeddings_results.get('matches', []))
                bigbird_count = len(bigbird_memory_info)
                qdrant_main_count = len(qdrant_embeddings_results.get('matches', []))
                qdrant_fallback_count = len(qdrant_fallback_results.get('matches', []))
                
                logger.info(f"=== LEGACY ROUTE SEARCH RESULTS ===")
                logger.info(f"Requested: vector_top_k={vector_top_k} (top_k={top_k} * chunk_factor={chunk_factor}), semantic_top_k={top_k}")
                logger.info(f"Pinecone results: {pinecone_count} matches (vector source)")
                logger.info(f"BigBird results: {bigbird_count} items (semantic source)") 
                logger.info(f"Qdrant main results: {qdrant_main_count} matches (vector source)")
                logger.info(f"Qdrant fallback results: {qdrant_fallback_count} matches")
                
                # Use fallback Qdrant results if main Qdrant search returned no results
                if not qdrant_embeddings_results.get('matches') and qdrant_fallback_results.get('matches'):
                    logger.warning(f"Legacy route: Main Qdrant search returned no results, using fallback: {len(qdrant_fallback_results.get('matches', []))} matches")
                    qdrant_embeddings_results = qdrant_fallback_results
                    qdrant_final_count = qdrant_fallback_count
                elif qdrant_embeddings_results.get('matches'):
                    logger.info(f"Legacy route: Using main Qdrant results: {len(qdrant_embeddings_results.get('matches', []))} matches")
                    qdrant_final_count = qdrant_main_count
                else:
                    qdrant_final_count = 0
                    
                logger.info(f"Final Qdrant count (after fallback logic): {qdrant_final_count}")
                
            except asyncio.TimeoutError:
                logger.error("Memory search operation timed out (legacy route)")
                return result
            except Exception as e:
                logger.error(f"Error in legacy route search: {e}")
                # Return empty results on error
                similar_embeddings_results = {"matches": []}
                bigbird_memory_info = []
                qdrant_embeddings_results = {"matches": []}
                qdrant_fallback_results = {"matches": []}
            
            search_time = (time.time() - search_start) * 1000
            logger.warning(f"Legacy route search took {search_time:.2f}ms")
            
            # Combine memory ids from all three sources with detailed logging
            logger.info(f"=== COMBINING MEMORY IDS ===")
            
            # Start with Pinecone results
            memory_item_ids = [match['id'] for match in similar_embeddings_results.get('matches', [])]
            logger.info(f"After adding Pinecone: {len(memory_item_ids)} total memory IDs")
            
            # Add Qdrant results
            qdrant_ids = [match['id'] for match in qdrant_embeddings_results.get('matches', [])]
            memory_item_ids += qdrant_ids
            logger.info(f"After adding Qdrant ({len(qdrant_ids)} new): {len(memory_item_ids)} total memory IDs")
            
            # Add BigBird results
            bigbird_ids = [item['id'] for item in bigbird_memory_info if 'id' in item]
            memory_item_ids += bigbird_ids
            logger.info(f"After adding BigBird ({len(bigbird_ids)} new): {len(memory_item_ids)} total memory IDs")
            
            # Remove duplicates and log final count
            unique_memory_ids = list(set(memory_item_ids))
            duplicates_removed = len(memory_item_ids) - len(unique_memory_ids)
            logger.info(f"=== FINAL LEGACY ROUTE SUMMARY ===")
            logger.info(f"Total IDs before deduplication: {len(memory_item_ids)}")
            logger.info(f"Duplicates removed: {duplicates_removed}")
            logger.info(f"Final unique memory IDs: {len(unique_memory_ids)}")
            
            # Use the original logic but update memory_item_ids to be the combined list
            memory_item_ids = unique_memory_ids
            # Bigbird ids
            predicted_grouped_memory_ids = set()
            for memory_info in bigbird_memory_info:
                if 'id' in memory_info and memory_info['id']:
                    predicted_grouped_memory_ids.add(memory_info['id'])
                related_ids = memory_info.get('metadata', {}).get('relatedMemoryIds', [])
                for rid in related_ids:
                    predicted_grouped_memory_ids.add(strip_chunk_suffix(rid))
            
            # Check Qdrant results for grouped memories
            for match in qdrant_embeddings_results.get('matches', []):
                if 'metadata' in match and match['metadata'].get('isGroupedMemories', False):
                    memory_id = match['id']
                    predicted_grouped_memory_ids.add(strip_chunk_suffix(memory_id))
                    logger.info(f"Found grouped memory in Qdrant: {memory_id}")
                    # Also add related memory IDs if they exist
                    related_ids = match['metadata'].get('relatedMemoryIds', [])
                    for rid in related_ids:
                        predicted_grouped_memory_ids.add(strip_chunk_suffix(rid))
                        logger.info(f"Added related memory to grouped set: {rid}")
            
            predicted_grouped_memory_ids = list(predicted_grouped_memory_ids)

            logger.info(f'predicted_grouped_memory_ids: {predicted_grouped_memory_ids}')
            # Process memory IDs to handle both legacy and chunked formats using strip_chunk_suffix
            processed_memory_ids = [strip_chunk_suffix(memory_id) for memory_id in predicted_grouped_memory_ids]
            # Remove duplicates while preserving order
            processed_memory_ids = list(dict.fromkeys(processed_memory_ids))
            logger.debug(f'Processed memory IDs: {processed_memory_ids}')
        else:
            # Only run Qwen/Qdrant
            query_qdrant_embedding_list, _ = await self.embedding_model.get_qwen_embedding_4b(
                query_context_combined, max_retries=3, retry_delay=1
            )
            query_qdrant_embedding = query_qdrant_embedding_list[0] if query_qdrant_embedding_list else None
            embedding_time = (time.time() - embedding_start) * 1000
            logger.warning(f"Embedding generation (qwen only) took {embedding_time:.2f}ms")
            search_start = time.time()
            try:
                # Increase top_k for Qdrant to account for chunking  
                chunk_factor = 3  # Assume average 3 chunks per memory
                vector_top_k = top_k * chunk_factor  # Get 3x more from Qdrant
                
                # Run both main search and fallback search in parallel
                qdrant_embeddings_results, qdrant_fallback_results = await asyncio.gather(
                    self.get_qdrant_related_memories_async(
                        query_qdrant_embedding,
                        final_filter,
                        top_k=vector_top_k  # Increased for chunking
                    ),
                    self.get_qdrant_related_memories_async_fallback(
                        final_filter,
                        top_k=vector_top_k  # Increased for chunking
                    )
                )
                
                # Log raw results from both Qdrant searches
                qdrant_main_count = len(qdrant_embeddings_results.get('matches', []))
                qdrant_fallback_count = len(qdrant_fallback_results.get('matches', []))
                
                logger.info(f"=== QWEN-ONLY ROUTE SEARCH RESULTS ===")
                logger.info(f"Requested: vector_top_k={vector_top_k} (top_k={top_k} * chunk_factor={chunk_factor})")
                logger.info(f"Qdrant main results: {qdrant_main_count} matches")
                logger.info(f"Qdrant fallback results: {qdrant_fallback_count} matches")
                
                # Use main results if we have any, otherwise use fallback
                if qdrant_embeddings_results.get('matches'):
                    similar_embeddings_results = qdrant_embeddings_results
                    logger.info(f"Using main Qdrant results: {len(qdrant_embeddings_results.get('matches', []))} matches")
                    final_count = qdrant_main_count
                elif qdrant_fallback_results.get('matches'):
                    similar_embeddings_results = qdrant_fallback_results
                    logger.warning(f"Main search returned no results, using fallback: {len(qdrant_fallback_results.get('matches', []))} matches")
                    # Update qdrant_embeddings_results to fallback for downstream processing
                    qdrant_embeddings_results = qdrant_fallback_results
                    final_count = qdrant_fallback_count
                else:
                    similar_embeddings_results = {"matches": []}
                    logger.warning("Both main and fallback searches returned no results")
                    final_count = 0
                
                logger.info(f"Final Qdrant count (after fallback logic): {final_count}")
                logger.info(f"=== QWEN-ONLY ROUTE SUMMARY ===")
                logger.info(f"Total memory IDs retrieved: {final_count}")
                
                bigbird_memory_info = []
                predicted_grouped_memory_ids = []
            except asyncio.TimeoutError:
                logger.error("Memory search operation timed out (qwen only)")
                return result
            except Exception as e:
                logger.error(f"Error in qwen only search: {e}")
                # Return empty results on error
                qdrant_embeddings_results = {"matches": []}
                similar_embeddings_results = qdrant_embeddings_results
                bigbird_memory_info = []
                predicted_grouped_memory_ids = []
            
            # Check Qdrant results for grouped memories (for qwen-only route)
            for match in qdrant_embeddings_results.get('matches', []):
                if 'metadata' in match and match['metadata'].get('isGroupedMemories', False):
                    memory_id = match['id']
                    predicted_grouped_memory_ids.append(strip_chunk_suffix(memory_id))
                    logger.info(f"Found grouped memory in Qdrant (qwen-only): {memory_id}")
                    # Also add related memory IDs if they exist
                    related_ids = match['metadata'].get('relatedMemoryIds', [])
                    for rid in related_ids:
                        predicted_grouped_memory_ids.append(strip_chunk_suffix(rid))
                        logger.info(f"Added related memory to grouped set (qwen-only): {rid}")
            search_time = (time.time() - search_start) * 1000
            logger.warning(f"Qwen Qdrant only search took {search_time:.2f}ms")
            memory_item_ids = [match['id'] for match in similar_embeddings_results.get('matches', [])]
            logger.info(f"Qwen-only route final memory_item_ids count: {len(memory_item_ids)}")

        #logger.info(f"bigbird_memory_info: {bigbird_memory_info}")
        

        # Build similarity score dict from bigbird results
        bigbird_similarity_scores = {item['id']: item['score'] for item in bigbird_memory_info if 'id' in item and 'score' in item}
        # Build similarity score dict from pinecone results
        pinecone_similarity_scores = {}
        if 'matches' in similar_embeddings_results:
            for match in similar_embeddings_results['matches']:
                if 'id' in match and 'score' in match:
                    pinecone_similarity_scores[match['id']] = match['score']
        # Build similarity score dict from qdrant results
        qdrant_similarity_scores = {}
        # Check similar_embeddings_results (for qwen-only route)
        if 'matches' in similar_embeddings_results:
            for match in similar_embeddings_results['matches']:
                if 'id' in match and 'score' in match:
                    qdrant_similarity_scores[match['id']] = match['score']
        # Check qdrant_embeddings_results (for legacy route)
        if 'qdrant_embeddings_results' in locals() and 'matches' in qdrant_embeddings_results:
            for match in qdrant_embeddings_results['matches']:
                if 'id' in match and 'score' in match:
                    qdrant_similarity_scores[match['id']] = match['score']
        # Merge all similarity scores
        all_similarity_scores = {**pinecone_similarity_scores, **bigbird_similarity_scores, **qdrant_similarity_scores}
        result.similarity_scores_by_id = all_similarity_scores
        result.bigbird_memory_info = bigbird_memory_info  # Attach for downstream use if needed
        
        # Log the similarity scores for debugging
        logger.info(f"BigBird similarity scores: {bigbird_similarity_scores}")
        logger.info(f"Pinecone similarity scores: {pinecone_similarity_scores}")
        logger.info(f"Qdrant similarity scores: {qdrant_similarity_scores}")
        logger.info(f"Combined similarity scores: {all_similarity_scores}")

        # Process results efficiently using list comprehensions
        # memory_item_ids is already populated from the legacy_route or qwen_only sections above
        
       
        
        # predicted_grouped_memory_ids is already populated from the legacy_route or qwen_only sections above
        
        neo_memory_ids = []
        similar_memory_items = []

        neo_start = time.time()

        if not skip_neo:
            # Check circuit breaker before attempting Neo4j operations
            if not await self.async_neo_conn.circuit_breaker.can_try():
                logger.warning("Circuit breaker is open, skipping Neo4j operations but continuing with Qdrant results")
                self.async_neo_conn.fallback_mode = True
                # Set skip_neo to True so we skip Neo4j operations but continue to fetch Qdrant memory items
                skip_neo = True
            
            # Create Neo4j session if we don't have one, using proper context manager
            if not neo_session:
                async with self.async_neo_conn.get_session() as neo_session:
                    try:
                        # Add timeout wrapper around entire Neo4j operation
                        async def neo4j_operation():
                            return await self.query_neo4j_with_user_query_async(
                                session_token, 
                                query_context_combined, 
                                final_filter,
                                user_id, 
                                chat_gpt, 
                                neo_session,
                                project_id=project_id, 
                                top_k=top_k_neo,
                                api_key=api_key,
                                user_workspace_ids=user_workspace_ids,
                                user_organization_ids=user_organization_ids,
                                user_namespace_ids=user_namespace_ids,
                                cached_schema=cached_schema
                            )
                        
                        # Execute with reduced timeout and circuit breaker failure recording
                        try:
                            memory_nodes, other_nodes, _, text_context = await asyncio.wait_for(
                                neo4j_operation(),
                                timeout=30.0  # Reduced from 60 to 30 seconds
                            )
                        except asyncio.TimeoutError:
                            # Record failure in circuit breaker for timeout
                            await self.async_neo_conn.circuit_breaker.record_failure()
                            logger.error("Neo4j operation timed out after 30 seconds, recording failure")
                            self.async_neo_conn.fallback_mode = True
                            # Set skip_neo to True and continue to fetch Qdrant memory items
                            skip_neo = True
                            # Set empty neo results and variables to prevent errors
                            memory_nodes = []
                            other_nodes = []
                            text_context = None
                            result.neo_nodes = []
                            result.neo_context = None
                            result.neo_query = None
                        else:
                            # Only set these if the operation succeeded (not in timeout handler)
                            result.neo_nodes = other_nodes
                            result.neo_context = text_context
                        
                        # Add memory nodes to memory_items if they exist
                        if memory_nodes:
                            try:
                                neo_memory_items = await self.fetch_memory_items_from_sources_async_fast(
                                    session_token,
                                    [node.id for node in memory_nodes],
                                    user_id,
                                    api_key=api_key
                                )
                                result.memory_items.extend(neo_memory_items)
                                neo_memory_ids = [node.id for node in memory_nodes]
                            except Exception as e:
                                logger.warning(f"Error fetching Neo4j memory items: {e}. Continuing without Neo4j memory items.")
                                # Keep neo_memory_ids empty
                                pass

                    except asyncio.TimeoutError:
                        logger.warning("Neo4j operation timed out after 60 seconds. Continuing without Neo4j results.")
                        # Reset Neo4j related fields to empty/None values
                        memory_nodes = []
                        other_nodes = []
                        text_context = None
                        result.neo_nodes = []
                        result.neo_context = None
                        result.neo_query = None
                        # Keep neo_memory_ids empty
                        pass
                    except Exception as e:
                        logger.warning(f"Error querying Neo4j: {e}. Continuing without Neo4j results.")
                        # Reset Neo4j related fields to empty/None values and initialize variables
                        memory_nodes = []
                        other_nodes = []
                        text_context = None
                        result.neo_nodes = []
                        result.neo_context = None
                        result.neo_query = None
                        # Keep neo_memory_ids empty
                        pass
            else:
                # Use the provided session
                try:
                    # Add timeout wrapper around entire Neo4j operation
                    async def neo4j_operation():
                        return await self.query_neo4j_with_user_query_async(
                            session_token, 
                            query_context_combined, 
                            final_filter,
                            user_id, 
                            chat_gpt, 
                            neo_session,
                            project_id=project_id, 
                            top_k=top_k_neo,
                            api_key=api_key,
                            user_workspace_ids=user_workspace_ids,
                            user_organization_ids=user_organization_ids,
                            user_namespace_ids=user_namespace_ids,
                            cached_schema=cached_schema
                        )
                    
                    # Execute with reduced timeout and circuit breaker failure recording
                    try:
                        memory_nodes, other_nodes, _, text_context = await asyncio.wait_for(
                            neo4j_operation(),
                            timeout=30.0  # Reduced from 60 to 30 seconds
                        )
                    except asyncio.TimeoutError:
                        # Record failure in circuit breaker for timeout
                        await self.async_neo_conn.circuit_breaker.record_failure()
                        logger.error("Neo4j operation timed out after 30 seconds, recording failure")
                        self.async_neo_conn.fallback_mode = True
                        # Set skip_neo to True and continue to fetch Qdrant memory items
                        skip_neo = True
                        # Set empty neo results and variables to prevent errors
                        memory_nodes = []
                        other_nodes = []
                        text_context = None
                        result.neo_nodes = []
                        result.neo_context = None
                        result.neo_query = None
                    else:
                        # Only set these if the operation succeeded (not in timeout handler)
                        result.neo_nodes = other_nodes
                        result.neo_context = text_context
                    
                    # Add memory nodes to memory_items if they exist
                    if memory_nodes:
                        try:
                            neo_memory_items = await self.fetch_memory_items_from_sources_async_fast(
                                session_token,
                                [node.id for node in memory_nodes],
                                user_id,
                                api_key=api_key
                            )
                            result.memory_items.extend(neo_memory_items)
                            neo_memory_ids = [node.id for node in memory_nodes]
                        except Exception as e:
                            logger.warning(f"Error fetching Neo4j memory items: {e}. Continuing without Neo4j memory items.")
                            # Keep neo_memory_ids empty
                            pass

                except asyncio.TimeoutError:
                    logger.warning("Neo4j operation timed out after 60 seconds. Continuing without Neo4j results.")
                    # Reset Neo4j related fields to empty/None values and initialize variables
                    memory_nodes = []
                    other_nodes = []
                    text_context = None
                    result.neo_nodes = []
                    result.neo_context = None
                    result.neo_query = None
                    # Keep neo_memory_ids empty
                    pass
                except Exception as e:
                    logger.warning(f"Error querying Neo4j: {e}. Continuing without Neo4j results.")
                    # Reset Neo4j related fields to empty/None values and initialize variables
                    memory_nodes = []
                    other_nodes = []
                    text_context = None
                    result.neo_nodes = []
                    result.neo_context = None
                    result.neo_query = None
                    # Keep neo_memory_ids empty
                    pass

            neo_time = (time.time() - neo_start) * 1000
            logger.warning(f"Neo4j query took {neo_time:.2f}ms")


        # Process memory IDs to get base IDs without chunk numbers
        def get_base_id(memory_id):
            return memory_id.split('_')[0] if '_' in memory_id else memory_id
        

        
        # Get base IDs for each source
        memory_base_ids = [get_base_id(mid) for mid in memory_item_ids]
        #bigbird_base_ids = [get_base_id(mid) for mid in predicted_grouped_memory_ids]
        neo_base_ids = [get_base_id(mid) for mid in neo_memory_ids]
        
        # Initialize combined_memory_item_ids_unsorted to ensure it's always defined
        # This will be populated by stratified sampling below, or fallback to memory_item_ids if needed
        combined_memory_item_ids_unsorted = []
        
        # STRATIFIED SAMPLING FOR DIVERSITY ACROSS SOURCES
        # Extract IDs from each source separately for quota-based sampling
        if legacy_route:
            # Legacy route: separate Pinecone, BigBird, Qdrant, Neo4j
            pinecone_ids = [match['id'] for match in similar_embeddings_results.get('matches', [])]
            bigbird_ids = [item['id'] for item in bigbird_memory_info if 'id' in item]
            qdrant_ids = [match['id'] for match in qdrant_embeddings_results.get('matches', [])]
            neo_ids = neo_memory_ids.copy()
            
            logger.info(f"=== STRATIFIED SAMPLING (LEGACY ROUTE) ===")
            logger.info(f"Source counts - Pinecone: {len(pinecone_ids)}, BigBird: {len(bigbird_ids)}, Qdrant: {len(qdrant_ids)}, Neo4j: {len(neo_ids)}")
            
            # Quota-based sampling: take top M from each source for diversity
            # Use a multiplicative factor to account for chunked memories that will be deduplicated
            chunk_factor = 3  # Assume average 3 chunks per memory
            num_sources = 4  # Pinecone, BigBird, Qdrant, Neo4j
            base_per_source = max(top_k // num_sources, 5)  # At least 5 per source for diversity
            
            # Apply chunk factor to vector sources (Pinecone/Qdrant) that return chunks
            m_per_source_vector = base_per_source * chunk_factor  # 3x more from vector sources
            m_per_source_semantic = base_per_source  # Normal amount from semantic sources (BigBird, Neo4j)
            
            # Sample top M from each source (they're already sorted by score from the search APIs)
            # Use chunk factor for vector sources to account for deduplication
            sampled_pinecone = pinecone_ids[:m_per_source_vector]
            sampled_bigbird = bigbird_ids[:m_per_source_semantic] 
            sampled_qdrant = qdrant_ids[:m_per_source_vector]
            sampled_neo = neo_ids[:m_per_source_semantic]
            
            logger.info(f"Sampled counts - Pinecone: {len(sampled_pinecone)}/{m_per_source_vector}, BigBird: {len(sampled_bigbird)}/{m_per_source_semantic}, Qdrant: {len(sampled_qdrant)}/{m_per_source_vector}, Neo4j: {len(sampled_neo)}/{m_per_source_semantic}")
            logger.info(f"Chunk factor applied: {chunk_factor}x for vector sources (Pinecone/Qdrant)")
            
            # Assign default similarity scores to Neo4j results (they don't have native scores)
            neo_default_score = 0.7  # Mid-range score to give them a fair shot
            for neo_id in sampled_neo:
                if neo_id not in all_similarity_scores:
                    all_similarity_scores[neo_id] = neo_default_score
            
            # Combine sampled IDs from all sources
            combined_memory_item_ids_unsorted = list(set(sampled_pinecone + sampled_bigbird + sampled_qdrant + sampled_neo))
            
            # ADAPTIVE QUOTA FILLING: If we have fewer than top_k results, fill remaining slots
            current_count = len(combined_memory_item_ids_unsorted)
            if current_count < top_k:
                remaining_slots = top_k - current_count
                logger.info(f"Adaptive filling: need {remaining_slots} more memories to reach top_k={top_k}")
                
                # Calculate remaining results available from each source
                remaining_pinecone = pinecone_ids[len(sampled_pinecone):]
                remaining_bigbird = bigbird_ids[len(sampled_bigbird):]
                remaining_qdrant = qdrant_ids[len(sampled_qdrant):]
                remaining_neo = neo_ids[len(sampled_neo):]
                
                # Create list of (source_name, remaining_ids) for sources with additional results
                sources_with_remaining = []
                if remaining_pinecone: sources_with_remaining.append(("Pinecone", remaining_pinecone))
                if remaining_bigbird: sources_with_remaining.append(("BigBird", remaining_bigbird))
                if remaining_qdrant: sources_with_remaining.append(("Qdrant", remaining_qdrant))
                if remaining_neo: sources_with_remaining.append(("Neo4j", remaining_neo))
                
                logger.info(f"Sources with remaining results: {[(name, len(ids)) for name, ids in sources_with_remaining]}")
                
                # Distribute remaining slots proportionally among sources with results
                additional_ids = []
                if sources_with_remaining:
                    slots_per_source = remaining_slots // len(sources_with_remaining)
                    extra_slots = remaining_slots % len(sources_with_remaining)
                    
                    for i, (source_name, remaining_ids) in enumerate(sources_with_remaining):
                        # Give extra slot to first few sources if we have remainder
                        slots_for_this_source = slots_per_source + (1 if i < extra_slots else 0)
                        additional_from_source = remaining_ids[:slots_for_this_source]
                        additional_ids.extend(additional_from_source)
                        
                        # Assign default scores to additional Neo4j results
                        if source_name == "Neo4j":
                            for neo_id in additional_from_source:
                                if neo_id not in all_similarity_scores:
                                    all_similarity_scores[neo_id] = neo_default_score
                        
                        logger.info(f"Added {len(additional_from_source)} more from {source_name}")
                
                # Add the additional IDs to our result set
                combined_memory_item_ids_unsorted.extend(additional_ids)
                combined_memory_item_ids_unsorted = list(set(combined_memory_item_ids_unsorted))  # Remove any duplicates
                logger.info(f"After adaptive filling: {len(combined_memory_item_ids_unsorted)} total memories")
            
        else:
            # Qwen-only route: just Qdrant + Neo4j
            qdrant_ids = [match['id'] for match in similar_embeddings_results.get('matches', [])]
            neo_ids = neo_memory_ids.copy()
            
            logger.info(f"=== STRATIFIED SAMPLING (QWEN-ONLY ROUTE) ===")
            logger.info(f"Source counts - Qdrant: {len(qdrant_ids)}, Neo4j: {len(neo_ids)}")
            
            # For qwen-only route, take more from each source since we only have 2 sources
            # Apply chunk factor to account for Qdrant chunking
            chunk_factor = 3  # Assume average 3 chunks per memory
            num_sources = 2  # Qdrant, Neo4j
            base_per_source = max(top_k // num_sources, 10)  # At least 10 per source
            
            # Apply chunk factor to Qdrant (vector source), normal amount for Neo4j (semantic source)
            m_per_source_qdrant = base_per_source * chunk_factor  # 3x more from Qdrant
            m_per_source_neo = base_per_source  # Normal amount from Neo4j
            
            sampled_qdrant = qdrant_ids[:m_per_source_qdrant]
            sampled_neo = neo_ids[:m_per_source_neo]
            
            logger.info(f"Sampled counts - Qdrant: {len(sampled_qdrant)}/{m_per_source_qdrant}, Neo4j: {len(sampled_neo)}/{m_per_source_neo}")
            logger.info(f"Chunk factor applied: {chunk_factor}x for Qdrant vector source")
            
            # Assign default similarity scores to Neo4j results
            neo_default_score = 0.7
            for neo_id in sampled_neo:
                if neo_id not in all_similarity_scores:
                    all_similarity_scores[neo_id] = neo_default_score
            
            # Combine sampled IDs
            combined_memory_item_ids_unsorted = list(set(sampled_qdrant + sampled_neo))
            
            # ADAPTIVE QUOTA FILLING for Qwen-only route
            current_count = len(combined_memory_item_ids_unsorted)
            if current_count < top_k:
                remaining_slots = top_k - current_count
                logger.info(f"Adaptive filling (Qwen-only): need {remaining_slots} more memories to reach top_k={top_k}")
                
                # Calculate remaining results available from each source
                remaining_qdrant = qdrant_ids[len(sampled_qdrant):]
                remaining_neo = neo_ids[len(sampled_neo):]
                
                # Create list of (source_name, remaining_ids) for sources with additional results
                sources_with_remaining = []
                if remaining_qdrant: sources_with_remaining.append(("Qdrant", remaining_qdrant))
                if remaining_neo: sources_with_remaining.append(("Neo4j", remaining_neo))
                
                logger.info(f"Sources with remaining results: {[(name, len(ids)) for name, ids in sources_with_remaining]}")
                
                # Distribute remaining slots proportionally among sources with results
                additional_ids = []
                if sources_with_remaining:
                    slots_per_source = remaining_slots // len(sources_with_remaining)
                    extra_slots = remaining_slots % len(sources_with_remaining)
                    
                    for i, (source_name, remaining_ids) in enumerate(sources_with_remaining):
                        # Give extra slot to first few sources if we have remainder
                        slots_for_this_source = slots_per_source + (1 if i < extra_slots else 0)
                        additional_from_source = remaining_ids[:slots_for_this_source]
                        additional_ids.extend(additional_from_source)
                        
                        # Assign default scores to additional Neo4j results
                        if source_name == "Neo4j":
                            for neo_id in additional_from_source:
                                if neo_id not in all_similarity_scores:
                                    all_similarity_scores[neo_id] = neo_default_score
                        
                        logger.info(f"Added {len(additional_from_source)} more from {source_name}")
                
                # Add the additional IDs to our result set
                combined_memory_item_ids_unsorted.extend(additional_ids)
                combined_memory_item_ids_unsorted = list(set(combined_memory_item_ids_unsorted))  # Remove any duplicates
                logger.info(f"After adaptive filling: {len(combined_memory_item_ids_unsorted)} total memories")
        
        logger.info(f'Stratified sampling: {len(combined_memory_item_ids_unsorted)} unique IDs after quota-based sampling')
        
        # Fallback: If stratified sampling didn't populate any IDs (e.g., Neo4j timeout prevented execution),
        # use memory_item_ids directly from Qdrant results
        if not combined_memory_item_ids_unsorted and memory_item_ids:
            logger.warning(f"Stratified sampling produced no results, falling back to memory_item_ids: {len(memory_item_ids)} IDs")
            combined_memory_item_ids_unsorted = memory_item_ids.copy()
        
        # Sort combined memory IDs by similarity score (highest first) for quality within diversity
        # This ensures we get the best results within our diversified sample
        combined_memory_item_ids = sorted(
            combined_memory_item_ids_unsorted,
            key=lambda x: all_similarity_scores.get(x, 0.0),  # Use 0.0 for any missing scores
            reverse=True  # Highest scores first
        )
        
        # Log the stratified sampling improvement
        if combined_memory_item_ids_unsorted and all_similarity_scores:
            top_unsorted_score = all_similarity_scores.get(combined_memory_item_ids_unsorted[0], 0.0)
            top_sorted_score = all_similarity_scores.get(combined_memory_item_ids[0], 0.0)
            logger.info(f'Stratified sorting: unsorted first={top_unsorted_score:.6f}, sorted first={top_sorted_score:.6f}')
        
        # Don't limit to top_k yet - we'll limit after fetching actual results
        # This ensures we get enough results even if some memory IDs fail to fetch
        original_count = len(combined_memory_item_ids)
        
        logger.info(f'Stratified results before fetch: {original_count} diversified IDs (will limit to top_k={top_k} after fetch)')
        
        # Log top 5 similarity scores and their sources for debugging
        if combined_memory_item_ids and all_similarity_scores:
            top_scores = []
            for mid in combined_memory_item_ids[:5]:
                score = all_similarity_scores.get(mid, 0.0)
                source = "Neo4j" if score == neo_default_score else "Vector"
                top_scores.append((mid, score, source))
            logger.info(f'Top 5 stratified scores: {top_scores}')
        
        # Update the memory source info creation
        memory_id_source_locations = []
        
        # Create a mapping of memory IDs to their Qdrant metadata for quick lookup
        qdrant_metadata_map = {}
        # Handle both legacy route (qdrant_embeddings_results) and qwen-only route (similar_embeddings_results)
        qdrant_matches = qdrant_embeddings_results.get('matches', []) if 'qdrant_embeddings_results' in locals() else similar_embeddings_results.get('matches', [])
        for match in qdrant_matches:
            memory_id = strip_chunk_suffix(match['id'])
            qdrant_metadata_map[memory_id] = match.get('metadata', {})
        
        for item in combined_memory_item_ids:
            # Check if this memory is from Qdrant and if it's grouped
            in_qdrant = item in set(memory_item_ids)
            in_qdrant_grouped = False
            
            # If it's in Qdrant, check if it's grouped by looking at the Qdrant metadata
            # Use strip_chunk_suffix to ensure consistent key lookup
            base_item_id = strip_chunk_suffix(item)
            if in_qdrant and base_item_id in qdrant_metadata_map:
                in_qdrant_grouped = qdrant_metadata_map[base_item_id].get('isGroupedMemories', False)
                if in_qdrant_grouped:
                    logger.info(f"Memory {item} (base: {base_item_id}) detected as grouped Qdrant memory")
            
            memory_id_source_locations.append(
                MemoryIDSourceLocation(
                    memory_id=item,
                    source_location=MemorySourceLocation(
                        in_qdrant=in_qdrant,
                        in_qdrant_grouped=in_qdrant_grouped,
                        in_neo=item in set(neo_memory_ids)
                    )
                )
            )
        # Log the source location for each memory_id
        for src in memory_id_source_locations:
            logger.info(f"Memory {src.memory_id} source location: Qdrant={src.source_location.in_qdrant}, "
                        f"QdrantGrouped={src.source_location.in_qdrant_grouped}, "
                        f"Neo4j={src.source_location.in_neo}")
        
        result.memory_source_info = MemorySourceInfo(
            memory_id_source_location=memory_id_source_locations
        )
        total_fetch_time = 0.0
        # Fetch memory items if we have results
        if combined_memory_item_ids:
            fetch_start = time.time()
            try:
                similar_memory_items: List[ParseStoredMemory] = await self.fetch_memory_items_from_sources_async_fast(
                    session_token, 
                    combined_memory_item_ids, 
                    user_id,
                    api_key=api_key
                )

                # Filter out excluded memory ID if specified and update result
                if similar_memory_items:
                    logger.info(f"Before filtering: {len(similar_memory_items)} items from Parse Server")
                    
                    # Log first few memory IDs and their ACL fields for debugging
                    for idx, item in enumerate(similar_memory_items[:5]):
                        logger.info(f"Memory {idx}: memoryId={item.memoryId}, "
                                  f"user_read_access={item.user_read_access}, "
                                  f"organization_read_access={item.organization_read_access}, "
                                  f"namespace_read_access={item.namespace_read_access}, "
                                  f"ACL={item.ACL}")
                    
                    filtered_items = [
                        item for item in similar_memory_items 
                        if not exclude_memory_id or item.memoryId != exclude_memory_id
                    ]
                    
                    logger.info(f"After exclude_memory_id filter: {len(filtered_items)} items (excluded: {len(similar_memory_items) - len(filtered_items)})")

                    # Apply customMetadata filter (developer-provided) only for new route (Qdrant)
                    # Note: indexing already stores customMetadata in Parse and Qdrant payload; here we ensure
                    # that if a developer sends metadata.customMetadata in the search request, the returned items
                    # are constrained accordingly.
                    if metadata is not None and getattr(metadata, 'customMetadata', None):
                        logger.info(f"Applying customMetadata filter: {metadata.customMetadata}")
                        custom_filter = metadata.customMetadata or {}
                        if any(k in custom_filter for k in ('organization_id', 'namespace_id')):
                            logger.warning(
                                "customMetadata filter includes organization_id/namespace_id; "
                                "results must have these exact values (legacy memories without them will be dropped)"
                            )
                        else:
                            logger.info("customMetadata filter contains no org/namespace keys")

                        def value_matches(filter_value, item_value):
                            # Numeric range semantics: if filter is numeric and item is numeric, treat as item_value >= filter_value
                            if isinstance(filter_value, (int, float)):
                                try:
                                    if isinstance(item_value, (int, float)):
                                        return float(item_value) >= float(filter_value)
                                except Exception:
                                    return False
                                return False

                            # Explicit range dict: {gte, lte, gt, lt}
                            if isinstance(filter_value, dict):
                                try:
                                    val = float(item_value) if isinstance(item_value, (int, float, str)) else None
                                except Exception:
                                    val = None
                                if val is None:
                                    return False
                                if 'gte' in filter_value and not (val >= float(filter_value['gte'])):
                                    return False
                                if 'lte' in filter_value and not (val <= float(filter_value['lte'])):
                                    return False
                                if 'gt' in filter_value and not (val > float(filter_value['gt'])):
                                    return False
                                if 'lt' in filter_value and not (val < float(filter_value['lt'])):
                                    return False
                                return True

                            # Scalar exact equality or containment in list for non-numerics
                            if isinstance(filter_value, (str, bool)):
                                if isinstance(item_value, list):
                                    return filter_value in item_value
                                return item_value == filter_value

                            # List[str] -> any intersection
                            if isinstance(filter_value, list):
                                item_list = item_value if isinstance(item_value, list) else ([item_value] if item_value is not None else [])
                                try:
                                    item_list_str = [str(iv) for iv in item_list]
                                    return any(str(v) in item_list_str for v in filter_value)
                                except Exception:
                                    return False
                            return False

                        def item_matches_custom_filter(item):
                            item_cm = getattr(item, 'customMetadata', None) or {}
                            for k, v in custom_filter.items():
                                if k not in item_cm:
                                    return False
                                if not value_matches(v, item_cm.get(k)):
                                    return False
                            return True

                        before_custom = len(filtered_items)
                        filtered_items = [it for it in filtered_items if item_matches_custom_filter(it)]
                        logger.info(f"After customMetadata filter: {len(filtered_items)} items (excluded: {before_custom - len(filtered_items)})")
                    else:
                        logger.info("No customMetadata filter to apply")

                    # Now apply top_k limit to the actual fetched results
                    result.memory_items = filtered_items[:top_k]
                    
                    logger.info(f"SUMMARY: Fetched {len(similar_memory_items)} items from Parse, "
                              f"filtered to {len(filtered_items)} items, "
                              f"limited to top_k={len(result.memory_items)}")
                    
                    # Log which memories survived vs were filtered
                    if len(filtered_items) < len(similar_memory_items):
                        filtered_out_ids = [item.memoryId for item in similar_memory_items if item not in filtered_items]
                        logger.warning(f"Filtered out {len(filtered_out_ids)} memories: {filtered_out_ids[:10]}")
                else:
                    result.memory_items = []

                logger.info(f"Final memory items count after top_k limit: {len(result.memory_items)}")
                logger.debug(f"First memory item type: {type(result.memory_items[0]) if result.memory_items else 'No items'}")
            except Exception as e:
                logger.error(f"Error fetching combined memory items: {e}")
            total_fetch_time = (time.time() - fetch_start) * 1000
        
        # Rerank results if enabled
        if reranking_config and reranking_config.reranking_enabled:
            logger.info(f"Reranking memory items using {reranking_provider.value} provider")
            rerank_start = time.time()
            try:
                if reranking_provider == RerankingProvider.COHERE:
                    # Cohere reranking using v2 API
                    cohere_api_key = env.get("COHERE_API_KEY")
                    if not cohere_api_key:
                        logger.warning("COHERE_API_KEY not set, skipping Cohere reranking")
                    else:
                        # Prepare documents for Cohere rerank API
                        documents = [item.content or "" for item in result.memory_items]
                        
                        # Cohere rerank API v2 endpoint
                        cohere_url = "https://api.cohere.com/v2/rerank"
                        model = reranking_config.reranking_model or "rerank-v3.5"
                        
                        # Cohere API expects query and documents array
                        payload = {
                            "model": model,
                            "query": query,
                            "documents": documents,
                            "top_n": len(documents)  # Return all documents with scores
                        }
                        
                        headers = {
                            "Authorization": f"Bearer {cohere_api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        # Make async request to Cohere API
                        async with AsyncClient(timeout=30.0) as client:
                            response = await client.post(cohere_url, json=payload, headers=headers)
                            
                            if response.status_code == 200:
                                cohere_result = response.json()
                                # Cohere returns results with relevance_score (0-1 scale)
                                # Map to our (score, confidence, item) format
                                reranked_results = cohere_result.get("results", [])
                                
                                # Create mapping of index to (score, confidence, item)
                                # Cohere returns results sorted by relevance_score (highest first)
                                scores = []
                                for rerank_item in reranked_results:
                                    index = rerank_item.get("index")
                                    relevance_score = rerank_item.get("relevance_score", 0.0)
                                    # Cohere relevance_score is 0-1, convert to 1-10 scale for consistency
                                    score = relevance_score * 10.0
                                    # Use relevance_score as confidence (Cohere doesn't provide separate confidence)
                                    confidence = relevance_score
                                    
                                    if 0 <= index < len(result.memory_items):
                                        scores.append((score, confidence, result.memory_items[index]))
                                
                                # Sort by score (already sorted by Cohere, but ensure consistency)
                                sorted_results = sorted(scores, key=lambda x: x[0], reverse=True)
                                result.memory_items = [item for _, _, item in sorted_results]
                                confidence_scores = [confidence for _, confidence, _ in sorted_results]
                                result.confidence_scores = confidence_scores
                                
                                logger.info(f"Successfully reranked {len(result.memory_items)} memory items using Cohere {model}")
                            else:
                                logger.error(f"Cohere rerank API returned status {response.status_code}: {response.text}")
                                # Keep original order if Cohere API fails
                else:
                    # OpenAI reranking (original implementation)
                    def build_msg(query, passage):
                        # Trim passage aggressively to keep within small budgets
                        safe_passage = passage
                        try:
                            safe_passage = chat_gpt.trim_content_to_token_limit(passage, max_tokens=800, buffer_tokens=200)
                        except Exception:
                            pass
                        msgs = [
                            {"role":"system","content":"Return only this exact JSON: {\"score\":<1-10>,\"confidence\":<0.0-1.0>}"},
                            {"role":"user","content":f"Q:{query}\n\nP:{safe_passage}"}
                        ]
                        try:
                            msgs = chat_gpt.trim_messages_to_token_budget(
                                msgs,
                                max_total_tokens=int(env.get("RERANK_MAX_TOTAL_TOKENS", "4000")),
                                reserve_completion_tokens=int(env.get("RERANK_RESERVE_COMPLETION", "64")),
                                buffer_tokens=int(env.get("RERANK_PROMPT_BUFFER", "256")),
                            )
                        except Exception:
                            pass
                        return msgs
                    sem = asyncio.Semaphore(len(result.memory_items))      # tune to your key's RPM limit
                    
                    async def score_one(item):
                        async with sem:
                            # Use fallback wrapper with backoff to survive 429s/quota
                            resp = await chat_gpt._create_completion_with_fallback_async(
                                model=reranking_config.reranking_model,
                                messages=build_msg(query, item.content),
                                max_tokens=int(env.get("RERANK_COMPLETION_TOKENS", "20")),
                                temperature=0,
                                response_format={"type":"json_object"}, 
                            )
                            content = resp.choices[0].message.content                            
                            try:
                                # Try JSON first - now expecting both score and confidence
                                parsed = json.loads(content)
                                score = float(parsed.get("score", 0))
                                confidence = float(parsed.get("confidence", 0.5))  # Default confidence if not provided
                            except Exception:
                                # Fallback: extract numbers with regex
                                matches = re.findall(r"(\d+(\.\d+)?)", content)
                                score = float(matches[0]) if matches else 0
                                confidence = float(matches[1]) if len(matches) > 1 else 0.5
                            return score, confidence, item
                    start_rerank_sem = time.perf_counter()
                    scores = await asyncio.gather(*(score_one(m) for m in result.memory_items))
                    time_rerank_sem = time.perf_counter() - start_rerank_sem
                    logger.warning(f"Reranking with semaphore took {time_rerank_sem:.2f}s")
                    
                    # Sort by score and extract confidence scores
                    sorted_results = sorted(scores, key=lambda x: x[0], reverse=True)
                    result.memory_items = [item for _, _, item in sorted_results]
                    confidence_scores = [confidence for _, confidence, _ in sorted_results]
                    
                    # Store confidence scores in the result for later use
                    result.confidence_scores = confidence_scores
                    
                    logger.info("Successfully reranked memory items using OpenAI")
            except Exception as e:
                logger.error(f"Error during reranking: {e}", exc_info=True)
                # Keep original order if reranking fails
            rerank_time = (time.time() - rerank_start) * 1000
            logger.warning(f"Reranking took {rerank_time:.2f}ms")
        result.log_summary()

        total_time = (time.time() - start_time) * 1000
        logger.warning(f"Total total_fetch_time execution took {total_fetch_time:.2f}ms")
        logger.warning(f"Total find_related_memory_items execution took {total_time:.2f}ms")
        logger.warning(f'len memory_items {len(result.memory_items)}')
        
        return result
    
            
    def fetch_parse_server(self, session_token: str, memory_item_ids: List[str], chunk_base_ids: List[str], api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Worker function to fetch memory items from Parse Server.
        Args:
            session_token (str): Authentication token for Parse Server.
            memory_item_ids (List[str]): List of memory item IDs to fetch.
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                - results: List of memory items fetched from Parse Server
                - missing_memory_ids: List of memory IDs that weren't found
        """
        try:
            response = retrieve_multiple_memory_items(session_token, memory_item_ids, chunk_base_ids, api_key=api_key)
            logger.info(f"Retrieved {len(response['results'])} memory items from Parse Server.")
            logger.info(f"Missing {len(response['missing_memory_ids'])} memory items.")
            return response  # Already contains 'results' and 'missing_memory_ids'
        except Exception as e:
            logger.error(f"Error fetching from Parse Server: {e}")
            return {'results': [], 'missing_memory_ids': memory_item_ids}
        
    async def query_neo4j_async(self, query: str, params: Dict[str, Any], neo_session: AsyncSession) -> List[Dict[str, Any]]:
        """
        Async worker function to query Neo4j.
        Each connection creates its own session.
        
        Args:
            query (str): The Cypher query to execute.
            params (Dict[str, Any]): Parameters for the Cypher query.
            neo_session (AsyncSession): Neo4j session
        Returns:
            List[Dict[str, Any]]: A list of memory items fetched from Neo4j.
        """
        try:
        

            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, returning empty results")
                return []

            result = await neo_session.run(query, params)
            records = []
            async for record in result:
                records.append(record)
            await result.consume()
            
            # Track unique items by both ID and content
            unique_items = {}
            content_map = {}  # Map content to IDs for duplicate detection
            
            for record in records:
                node = record['a']
                node_dict = dict(node)
                node_id = node_dict.get('id')
                content = node_dict.get('content')
                
                if not node_id:
                    logger.warning(f"Memory ID property missing from node: {node_dict}")
                    continue
                
                # Check for content-based duplicates if content exists
                if content:
                    if content in content_map:
                        # Content already exists, log duplicate
                        logger.debug(f"Duplicate content found for IDs: {node_id} and {content_map[content]}")
                        # Keep the record with the earlier ID (assuming string comparison)
                        if node_id < content_map[content]:
                            # Remove old entry
                            old_id = content_map[content]
                            unique_items.pop(old_id, None)
                            # Add new entry
                            content_map[content] = node_id
                            unique_items[node_id] = node_dict
                    else:
                        # New content
                        content_map[content] = node_id
                        unique_items[node_id] = node_dict
                else:
                    # No content, check ID-based duplicate
                    if node_id in unique_items:
                        # Keep entry with content if it exists
                        if not unique_items[node_id].get('content'):
                            unique_items[node_id] = node_dict
                    else:
                        unique_items[node_id] = node_dict
            
            memory_items = list(unique_items.values())
            
            logger.info(f"Processed {len(records)} Neo4j records into {len(memory_items)} unique items")
            logger.info(f"Removed {len(records) - len(memory_items)} duplicates")
            
            return memory_items
                
        except Exception as e:
            logger.error(f"Error querying Neo4j: {e}")
            return []

    async def fetch_parse_server_async(
        self, 
        session_token: str, 
        memory_item_ids: List[str], 
        chunk_base_ids: List[str], 
        memory_class: str = "Memory",
        api_key: Optional[str] = None
    ) -> MemoryRetrievalResult:
        """
        Async worker function to fetch memory items from Parse Server.
        Args:
            session_token (str): Authentication token for Parse Server.
            memory_item_ids (List[str]): List of memory item IDs to fetch.
            chunk_base_ids (List[str]): List of base IDs without chunk numbers.
        
        Returns:
            MemoryRetrievalResult: A dictionary containing:
                - 'results': List[ParseStoredMemory] - List of memory items with user objects
                - 'missing_memory_ids': List[str] - List of memory IDs that weren't found
        """
        try:
            # Assuming retrieve_multiple_memory_items is also async
            response: MemoryRetrievalResult = await retrieve_memory_items_with_users_async(
                session_token, 
                memory_item_ids, 
                chunk_base_ids, 
                memory_class,
                api_key=api_key
            )

            # Access the TypedDict fields properly
            logger.info(f"Retrieved {len(response.get('results', []))} memory items from Parse Server.")
            logger.info(f"Missing {len(response.get('missing_memory_ids', []))} memory items.")
            return response  # Already contains 'results' and 'missing_memory_ids'
        except Exception as e:
            logger.error(f"Error fetching from Parse Server: {e}")
            return {'results': [], 'missing_memory_ids': memory_item_ids}
            
    def normalize_and_merge_memory_items(
        self, 
        similar_memory_items_neo: List[Dict[str, Any]], 
        memory_items_parse: List[ParseStoredMemory]
    ) -> List[ParseStoredMemory]:
        """
        Normalize and merge memory items from Neo4j and Parse Server.
        Always prefer Parse Server items as they are the source of truth.
        Returns Parse Server items without metadata.

        Args:
            similar_memory_items_neo: List of memory items from Neo4j
            memory_items_parse: List of ParseStoredMemory items from Parse Server

        Returns:
            List[ParseStoredMemory]: List of normalized Parse Server memory items without metadata
        """
        logger.info('Starting normalization and merging of memory items.')
        
        # Create sets for tracking IDs
        neo_ids: Set[str] = set()
        parse_ids: Set[str] = set()

        # Track Neo4j items for logging purposes
        logger.info(f"Neo4j fetched {len(similar_memory_items_neo)} items.")
        for item in similar_memory_items_neo:
            memory_id = item.get('id')
            if memory_id:
                neo_ids.add(memory_id)
                logger.debug(f"Neo4j Memory item - Memory ID: {memory_id}")
            else:
                logger.warning(f"Memory ID property missing from Neo4j item: {item}")

        # Process Parse Server items (source of truth)
        unique_parse_items: Dict[str, ParseStoredMemory] = {}
        logger.info(f"Parse Server fetched {len(memory_items_parse)} items.")
        
        for item in memory_items_parse:
            memory_id = item.memoryId
            if memory_id:
                parse_ids.add(memory_id)
                # Create a copy without metadata using the model method
                unique_parse_items[memory_id] = item.without_metadata()
                logger.debug(f"Parse Memory item - Memory ID: {memory_id}, ObjectId: {item.objectId}")

        # Log any discrepancies between Neo4j and Parse Server
        missing_in_parse = neo_ids - parse_ids
        missing_in_neo = parse_ids - neo_ids

        if missing_in_parse:
            logger.warning(f'Memory IDs in Neo4j but missing in Parse: {list(missing_in_parse)}')
        if missing_in_neo:
            logger.debug(f'Memory IDs in Parse but missing in Neo4j: {list(missing_in_neo)}')

        normalized_items = list(unique_parse_items.values())
        logger.info(f'Final normalized items count: {len(normalized_items)}')
        
        return normalized_items

    def handle_missing_parse_items(
        self, 
        similar_memory_items_neo: List[Dict[str, Any]], 
        normalized_memory_items: List[Dict[str, Any]], 
        session_token: str, 
        missing_memory_ids: List[str],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle memory items missing in Parse Server by adding them from Neo4j.
        
        Args:
            similar_memory_items_neo (List[Dict[str, Any]]): Memory items from Neo4j.
            normalized_memory_items (List[Dict[str, Any]]): Normalized memory items from Parse Server.
            session_token (str): Authentication token for Parse Server.
            missing_memory_ids (List[str]): List of memory item IDs missing in Parse Server.
        
        Returns:
            Dict[str, Any]: A dictionary mapping memory IDs to their respective memory items.
        """
        unique_memory_items: Dict[str, Any] = {}

        # Create a mapping of Parse Server items by memoryId
        parse_items_map = {
            item.get('memoryId'): item 
            for item in normalized_memory_items 
            if 'objectId' in item and item.get('memoryId') is not None
        }
        
        # Create a mapping of Neo4j items by memoryId for quick lookup
        neo_items_map = {
            item.get('id'): item 
            for item in similar_memory_items_neo 
            if item.get('id') and item.get('content')
        }

        # Process only the missing_memory_ids
        for memory_id in missing_memory_ids:
            item = neo_items_map.get(memory_id)
            if not item:
                logger.warning(f"Memory item {memory_id} not found in Neo4j or lacks 'content'. Skipping.")
                continue

            try:
                memory_item_obj = memory_item_from_dict(item)

                # Ensure 'content' exists for legacy items
                if not memory_item_obj.content:
                    logger.error(f"Legacy memory item {memory_id} lacks 'content'. Skipping.")
                    continue

                neo_item_user_id = item.get('user_id')

                # Store in Parse Server
                parse_response = store_generic_memory_item(neo_item_user_id, session_token, memory_item_obj, api_key=api_key)
                parse_object = retrieve_memory_item_with_user(session_token, memory_id, api_key=api_key)
                
                if parse_object and 'objectId' in parse_object:
                    unique_memory_items[memory_id] = parse_object
                    logger.info(f"Added memory item {memory_id} to Parse Server with objectId {parse_object['objectId']}.")

                    # Optionally remove content and context from Neo4j after successful addition
                    #try:
                    #    with self.neo_conn.session() as session:
                    #        deletion_result = session.run(
                    #            """
                    #            MATCH (m:Memory {id: $memory_id})
                    #            SET m.content = '', m.context = ''
                    #            RETURN count(m) as updatedCount
                    #            """,
                    #            memory_id=memory_id
                    #            ).single()

                    #            updated_count = deletion_result.get('updatedCount', 0)
                    #            if updated_count > 0:
                    #                logger.info(f"Content and context removed from Neo4j for memory_id {memory_id}.")
                    #else:
                    #    logger.warning(f"No Neo4j nodes updated for memory_id {memory_id}.")
                    #except Exception as e:
                    #    logger.error(f"Error removing content and context from Neo4j for memory_id {memory_id}: {e}")
                
                else:
                    logger.error(f"Failed to retrieve memory item {memory_id} from Parse Server after addition.")
            except Exception as e:
                logger.error(f"Error processing memory item {memory_id}: {e}")
                continue

        return unique_memory_items

    
    async def fetch_memory_items_from_sources_async(
        self, 
        session_token: str, 
        memory_item_ids: List[str], 
        user_id: str,
        neo_session: AsyncSession,
        api_key: Optional[str] = None
    ) -> List[ParseStoredMemory]:
        """
        Fetch memory items from Neo4j and Parse Server concurrently, normalize and merge them.
        
        Args:
            session_token (str): Authentication token for Parse Server.
            memory_item_ids (List[str]): List of memory item IDs to fetch (may include chunked IDs with _#).
            user_id (str): ID of the user requesting the memory items.
            neo_session (AsyncSession): Neo4j session.
            api_key (Optional[str]): API key for Parse Server.
        
        Returns:
            List[ParseStoredMemory]: A list of final memory items after processing.
        """

        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, cannot fetch memory items")
            return []
            
        # Process memory IDs to handle both legacy and chunked formats
        base_memory_ids = set()  # Use set to avoid duplicates
        chunk_id_mapping = {}  # Map to track original chunked IDs

        logger.info(f'Memory item ids: {memory_item_ids}')
        
        for memory_id in memory_item_ids:
            # Check if the ID has a chunk suffix (_#)
            if '_' in memory_id:
                base_id = memory_id.rsplit('_', 1)[0]  # Get the base ID without chunk number
                base_memory_ids.add(base_id)
                chunk_id_mapping[base_id] = memory_id  # Store mapping of base ID to chunked ID
            else:
                base_memory_ids.add(memory_id)
            logger.debug(f'Base memory IDs: {base_memory_ids}')
        processed_memory_ids = list(base_memory_ids)
        logger.debug(f'Processed memory IDs: {processed_memory_ids}')
        logger.debug(f'Chunk ID mapping: {chunk_id_mapping}')

        # Define the Neo4j batch query
        neo_query = '''
            MATCH (a:Memory) 
            WHERE a.id IN $ids 
            OR (a.memoryChunkIds IS NOT NULL AND ANY(chunk_id IN a.memoryChunkIds WHERE chunk_id IN $memoryChunkIds))
            OR (a.id IN $chunk_base_ids)
            RETURN a
        '''
        neo_params = {
            'ids': processed_memory_ids,
            'memoryChunkIds': memory_item_ids,
            'chunk_base_ids': processed_memory_ids
        }

        # Combine original and processed IDs for Parse Server query
        all_memory_ids = list(set(memory_item_ids + processed_memory_ids))

        try:
    
            # Execute Neo4j and Parse Server queries concurrently
            memory_class: str = "Memory"
            similar_memory_items_neo, parse_response = await asyncio.gather(
                self.query_neo4j_async(neo_query, neo_params, neo_session),
                self.fetch_parse_server_async(session_token, all_memory_ids, processed_memory_ids, memory_class, api_key=api_key),
                return_exceptions=True  # This allows gather to complete even if one call fails
            )

            # Handle potential exceptions from gather results
            if isinstance(similar_memory_items_neo, Exception):
                logger.error(f"Neo4j query failed: {similar_memory_items_neo}")
                similar_memory_items_neo = []
            else:
                logger.info(f'Neo4j fetched {len(similar_memory_items_neo)} items.')

            if isinstance(parse_response, Exception):
                logger.error(f"Parse Server query failed: {parse_response}")
                memory_items_parse = []
            else:
                memory_items_parse = parse_response['results']
                missing_memory_ids = parse_response['missing_memory_ids']
                logger.info(f'Parse Server fetched {len(memory_items_parse)} items.')
                logger.info(f'Parse Server missing {len(missing_memory_ids)} items.')

            # If both sources failed, return empty list
            if not similar_memory_items_neo and not memory_items_parse:
                logger.error("Both Neo4j and Parse Server queries failed")
                return []
            
            # Extract Parse Server results
            parse_response: MemoryRetrievalResult = parse_response
            memory_items_parse: List[ParseStoredMemory] = parse_response['results']
            missing_memory_ids: List[str] = parse_response['missing_memory_ids']

            logger.info(f'Neo4j fetched {len(similar_memory_items_neo)} items.')
            logger.info(f'Parse Server fetched {len(memory_items_parse)} items.')
            logger.info(f'Parse Server missing {len(missing_memory_ids)} items.')

            #logger.info(f'Memory items parse: {memory_items_parse}')

            # Normalize Parse Server items
            normalized_memory_items: List[ParseStoredMemory] = self.normalize_and_merge_memory_items(
                similar_memory_items_neo, 
                memory_items_parse
            )
            #logger.info(f'Normalized memory items: {normalized_memory_items}')
            #logger.info(f'similar_memory_items_neo: {similar_memory_items_neo}')
            
            # Process final items with chunk handling
            final_memory_items: List[ParseStoredMemory] = []
            for item in normalized_memory_items:
                # Optionally, set matchingChunkIds if you want to indicate which chunks matched
                if item.memoryChunkIds:
                    matching_chunks = [
                        chunk_id for chunk_id in item.memoryChunkIds 
                        if chunk_id in memory_item_ids
                    ]
                    if matching_chunks:
                        updated_item = item.model_copy()
                        setattr(updated_item, 'matchingChunkIds', matching_chunks)
                        final_memory_items.append(updated_item)
                        continue
                final_memory_items.append(item)

            logger.info(f'Final Memory Items Count: {len(final_memory_items)}')

            # Validation is handled by ParseStoredMemory model, but we can still log if needed
            items_without_content = [
                item.memoryId for item in final_memory_items 
                if not item.content
            ]
            if items_without_content:
                logger.error(f"Final memory items missing content: {items_without_content}")

            logger.info(f'Final memory items: {final_memory_items}')

            return final_memory_items
        
        except Exception as e:
            logger.error(f"Error in fetch_memory_items_from_sources_async: {e}")
            return []

    async def fetch_memory_items_from_sources_async_fast(
        self, 
        session_token: str, 
        memory_item_ids: List[str], 
        user_id: str,
        api_key: Optional[str] = None
    ) -> List[ParseStoredMemory]:
        """
        Optimized version that fetches memory items from Parse Server only (no Neo4j).
        This is faster since all memories have been migrated to Parse Server.
        
        Args:
            session_token (str): Authentication token for Parse Server.
            memory_item_ids (List[str]): List of memory item IDs to fetch (may include chunked IDs with _#).
            user_id (str): ID of the user requesting the memory items.
            api_key (Optional[str]): API key for Parse Server.
        
        Returns:
            List[ParseStoredMemory]: A list of final memory items after processing.
        """
        
        # Process memory IDs to handle both legacy and chunked formats
        base_memory_ids = set()  # Use set to avoid duplicates
        chunk_id_mapping = {}  # Map to track original chunked IDs

        logger.info(f'Memory item ids: {memory_item_ids}')
        
        for memory_id in memory_item_ids:
            # Check if the ID has a chunk suffix (_#)
            if '_' in memory_id:
                base_id = memory_id.rsplit('_', 1)[0]  # Get the base ID without chunk number
                base_memory_ids.add(base_id)
                chunk_id_mapping[base_id] = memory_id  # Store mapping of base ID to chunked ID
            else:
                base_memory_ids.add(memory_id)
            logger.debug(f'Base memory IDs: {base_memory_ids}')
        processed_memory_ids = list(base_memory_ids)
        logger.debug(f'Processed memory IDs: {processed_memory_ids}')
        logger.debug(f'Chunk ID mapping: {chunk_id_mapping}')

        # Combine original and processed IDs for Parse Server query
        all_memory_ids = list(set(memory_item_ids + processed_memory_ids))

        try:
            # Only fetch from Parse Server (no Neo4j)
            memory_class: str = "Memory"
            parse_response = await self.fetch_parse_server_async(
                session_token, 
                all_memory_ids, 
                processed_memory_ids, 
                memory_class, 
                api_key=api_key
            )

            # Handle potential exceptions
            if isinstance(parse_response, Exception):
                logger.error(f"Parse Server query failed: {parse_response}")
                return []

            # Extract Parse Server results
            memory_items_parse: List[ParseStoredMemory] = parse_response['results']
            missing_memory_ids: List[str] = parse_response['missing_memory_ids']

            logger.info(f'Parse Server fetched {len(memory_items_parse)} items.')
            logger.info(f'Parse Server missing {len(missing_memory_ids)} items.')

            # Since we're only using Parse Server, no need to normalize/merge with Neo4j
            # Just use the Parse Server results directly
            normalized_memory_items: List[ParseStoredMemory] = memory_items_parse
            
            # Process final items with chunk handling
            final_memory_items: List[ParseStoredMemory] = []
            for item in normalized_memory_items:
                # Optionally, set matchingChunkIds if you want to indicate which chunks matched
                if item.memoryChunkIds:
                    matching_chunks = [
                        chunk_id for chunk_id in item.memoryChunkIds 
                        if chunk_id in memory_item_ids
                    ]
                    if matching_chunks:
                        updated_item = item.model_copy()
                        setattr(updated_item, 'matchingChunkIds', matching_chunks)
                        final_memory_items.append(updated_item)
                        continue
                final_memory_items.append(item)

            logger.info(f'Final Memory Items Count: {len(final_memory_items)}')

            # Validation is handled by ParseStoredMemory model, but we can still log if needed
            items_without_content = [
                item.memoryId for item in final_memory_items 
                if not item.content
            ]
            if items_without_content:
                logger.error(f"Final memory items missing content: {items_without_content}")

            logger.info(f'Final memory items: {final_memory_items}')

            return final_memory_items
        
        except Exception as e:
            logger.error(f"Error in fetch_memory_items_from_sources_async_fast: {e}")
            return []

    async def get_pinecone_related_memories_async(self, query_embedding, acl_filter: dict, top_k=20):
        """
        Async method to get related memories from Pinecone
        
        Args:
            query_embedding: The embedding vector to query
            acl_filter (dict): Dictionary containing ACL filter conditions
            top_k (int): Number of results to return
            
        Returns:
            Dict containing matches with their metadata and scores
        """
        try:
            # Validate that query_embedding is actually a list of floats
            if not isinstance(query_embedding, list) or not all(isinstance(x, (int, float)) for x in query_embedding):
                logger.error(f"Invalid query_embedding type: {type(query_embedding)}, value: {query_embedding[:10] if isinstance(query_embedding, list) else query_embedding}")
                return {'matches': []}
            
            # Use safe Pinecone operation for query
            results = await self._safe_pinecone_operation(
                "get_related_memories",
                self.index.query,
                namespace="",
                top_k=top_k,
                include_values=False,
                include_metadata=True,
                vector=query_embedding,
                filter=acl_filter
            )
            
            if results is None:
                logger.warning("Pinecone query failed, returning empty results")
                return {'matches': []}


            # Convert QueryResponse to a standard dictionary format
            processed_results = {
                'matches': [
                    {
                        'id': match.id,
                        'score': match.score,
                        'metadata': match.metadata,
                        'values': match.values if hasattr(match, 'values') else None
                    }
                    for match in results.matches
                ]
            }
            
            return processed_results
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return {'matches': []}
        
    async def get_bigbird_related_memories_async(self, query_embedding, acl_filter: dict, user_id: str, top_k: int=10):
        """
        Async method to get related memories from BigBird index
        
        Args:
            query_embedding: The embedding vector to query
            acl_filter (dict): Dictionary containing ACL filter conditions
            user_id (str): The user ID
            top_k (int): Number of results to return
            
        Returns:
            List of memory items with their metadata and scores
        """
        try:
            # Validate that query_embedding is actually a list of floats
            if not isinstance(query_embedding, list) or not all(isinstance(x, (int, float)) for x in query_embedding):
                logger.error(f"Invalid query_embedding type: {type(query_embedding)}, value: {query_embedding[:10] if isinstance(query_embedding, list) else query_embedding}")
                return []
            
            # Use safe Pinecone operation for BigBird query
            results = await self._safe_pinecone_operation(
                "bigbird_query",
                self.bigbird_index.query,
                namespace="",
                top_k=top_k,
                include_values=False,
                include_metadata=True,
                vector=query_embedding,
                filter=acl_filter
            )
            
            if results is None:
                logger.warning("BigBird Pinecone query failed, returning empty results")
                return []
            
            # Convert QueryResponse to a standard dictionary format
            bigbird_memory_info = [
                {
                    'id': match.id,
                    'metadata': match.metadata,
                    'score': match.score
                }
                for match in results.matches
            ]
            
            return bigbird_memory_info
        except Exception as e:
            logger.error(f"Error in get_bigbird_related_memories_async: {e}")
            return []

    def _convert_qdrant_filter_to_pinecone(self, qdrant_filter):
        """Convert Qdrant filter to Pinecone-compatible format"""
        try:
            # Extract user_id from Qdrant filter for Pinecone
            user_id = None
            if qdrant_filter.should:
                for condition in qdrant_filter.should:
                    if hasattr(condition, 'key') and condition.key == 'user_id' and hasattr(condition, 'match'):
                        user_id = condition.match.value
                        break
            
            # Create simple Pinecone filter
            pinecone_filter = {}
            if user_id:
                pinecone_filter['user_id'] = user_id
            
            return pinecone_filter
        except Exception as e:
            logger.warning(f"Error converting Qdrant filter to Pinecone: {e}")
            return {}

    async def _qdrant_search_async(
        self,
        collection_name: str,
        query_vector,
        query_filter=None,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
        search_params=None,
    ):
        """
        Async Qdrant search with fallback for API compatibility.
        Uses search() if available, otherwise query_points() (some qdrant-client versions).
        Returns a list of results (ScoredPoint-like: .id, .score, .payload).
        """
        if not self.qdrant_client:
            return []
        search_fn = getattr(self.qdrant_client, "search", None)
        if search_fn is not None:
            kwargs = dict(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold
            if search_params is not None:
                kwargs["search_params"] = search_params
            return await search_fn(**kwargs)
        query_fn = getattr(self.qdrant_client, "query_points", None)
        if query_fn is not None:
            kwargs = dict(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold
            if search_params is not None:
                kwargs["search_params"] = search_params
            res = await query_fn(**kwargs)
            return getattr(res, "points", res) if res is not None else []
        logger.error(
            "Qdrant client has no 'search' or 'query_points'. "
            "Available: %s",
            [m for m in dir(self.qdrant_client) if not m.startswith("_") and ("search" in m.lower() or "query" in m.lower())],
        )
        return []

    async def get_qdrant_related_memories_async(self, query_embedding, acl_filter: qmodels.Filter, top_k=20):
        """
        Async method to get related memories from Qdrant index with resilient error handling
        
        Args:
            query_embedding: The embedding vector to query
            acl_filter (qmodels.Filter): Qdrant filter object for ACL
            top_k (int): Number of results to return
            
        Returns:
            List of memory items with their metadata and scores
        """
        if not query_embedding:
            logger.warning("No query embedding provided to get_qdrant_related_memories_async.")
            return {"matches": []}
        
        # Check if Qdrant client is available
        if not self.qdrant_client:
            logger.warning("Qdrant client not initialized, cannot perform search.")
            return {"matches": []}
        
        # Retry configuration for connection issues
        max_retries = 3
        base_delay = 1.0  # seconds
        timeout_seconds = 30.0  # Increased timeout for batch operations
        
        for attempt in range(max_retries):
            try:
                # Ensure query_embedding is a flat list
                if isinstance(query_embedding, list) and len(query_embedding) > 0:
                    if isinstance(query_embedding[0], list):
                        # If it's a nested list, take the first element
                        query_embedding = query_embedding[0]
                
                # Time the Qdrant search
                search_start = time.time()
                
                # Log search parameters (only on first attempt to reduce noise)
                if attempt == 0:
                    logger.info(f"Qdrant search params - collection: {self.qdrant_collection}, top_k: {top_k}, embedding_dim: {len(query_embedding) if query_embedding else 'None'}")
                    if acl_filter:
                        logger.info(f"Qdrant ACL filter: {acl_filter}")
                
                # Use _qdrant_search_async (handles search vs query_points API compatibility)
                from qdrant_client import models as qmodels
                results = await asyncio.wait_for(
                    self._qdrant_search_async(
                        collection_name=self.qdrant_collection,
                        query_vector=query_embedding,
                        query_filter=acl_filter,
                        limit=top_k,
                        with_payload=True,
                        with_vectors=False,
                        score_threshold=0.15,
                        search_params=qmodels.SearchParams(hnsw_ef=128, exact=False)
                    ),
                    timeout=timeout_seconds
                )
                search_duration = time.time() - search_start
                
                # Performance monitoring
                if search_duration > 0.1:  # Log slow searches
                    logger.warning(f"SLOW Qdrant search: {search_duration:.3f}s for {len(results)} results")
                else:
                    logger.info(f"Fast Qdrant search: {search_duration:.3f}s for {len(results)} results")
                
                logger.warning(f"Qdrant search returned {len(results)} results in {search_duration:.3f}s")
                
                # Log the first result in detail (only on successful searches)
                if results:
                    first_result = results[0]
                    logger.info(f"=== FIRST QDRANT RESULT ===")
                    logger.info(f"ID: {first_result.id}")
                    logger.info(f"Score (cosine similarity): {first_result.score}")
                    logger.info(f"Payload keys: {list(first_result.payload.keys()) if first_result.payload else 'None'}")
                    if first_result.payload:
                        # Log some key payload fields
                        chunk_id = first_result.payload.get('chunk_id', 'N/A')
                        content = first_result.payload.get('content', 'N/A')[:100] + "..." if first_result.payload.get('content') else 'N/A'
                        user_id = first_result.payload.get('user_id', 'N/A')
                        is_grouped = first_result.payload.get('isGroupedMemories', False)
                        logger.info(f"  - chunk_id: {chunk_id}")
                        logger.info(f"  - content preview: {content}")
                        logger.info(f"  - user_id: {user_id}")
                        logger.info(f"  - isGroupedMemories: {is_grouped}")
                    logger.info(f"=== END FIRST RESULT ===")
                else:
                    logger.warning("No Qdrant results found")
                
                # Convert ScoredPoint to a standard dictionary format compatible with Pinecone
                matches = []
                for match in results:
                    try:
                        chunk_id = match.payload.get('chunk_id', match.id)
                        matches.append({
                            'id': chunk_id,
                            'score': match.score,
                            'metadata': match.payload
                        })
                    except Exception as e:
                        logger.warning(f"Error processing Qdrant result: {e}")
                        continue
                
                return {"matches": matches}
                
            except asyncio.TimeoutError:
                retry_delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Qdrant search timeout (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay:.1f}s...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Qdrant search failed after {max_retries} timeout attempts")
                    break
                    
            except Exception as e:
                # Check if it's a connection-related error that should be retried
                error_str = str(e).lower()
                is_connection_error = any(keyword in error_str for keyword in [
                    'timeout', 'connect', 'connection', 'network', 'unreachable', 
                    'responsehandlingexception', 'httperror', 'connectionerror'
                ])
                
                if is_connection_error and attempt < max_retries - 1:
                    retry_delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Qdrant connection error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay:.1f}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Qdrant search failed (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
                    break
        
        # All retries exhausted
        logger.error("Qdrant search failed after all retry attempts. Continuing with other data sources.")
        return {"matches": []}

    async def retrieve_embeddings_by_chunk_ids_batch(
        self,
        chunk_ids: List[str],
        batch_size: int = 100
    ) -> Dict[str, List[float]]:
        """
        Batch retrieve embeddings from Qdrant by chunk IDs.
        
        Args:
            chunk_ids: List of chunk IDs (from payload.chunk_id field) to retrieve
            batch_size: Number of IDs to retrieve per batch (default: 100)
            
        Returns:
            Dict mapping chunk_id to embedding vector
        """
        if not chunk_ids:
            return {}
        
        if not self.qdrant_client or not self.qdrant_collection:
            logger.warning("Qdrant client or collection not initialized, cannot retrieve embeddings")
            return {}
        
        # Deduplicate chunk IDs
        unique_chunk_ids = list(set(chunk_ids))
        logger.info(f"Retrieving embeddings for {len(unique_chunk_ids)} unique chunk IDs from Qdrant")
        
        embeddings_map = {}
        
        # Batch retrieve to avoid overwhelming Qdrant
        for i in range(0, len(unique_chunk_ids), batch_size):
            batch_chunk_ids = unique_chunk_ids[i:i + batch_size]
            
            try:
                batch_start = time.time()
                
                # Use scroll with filter to find points by chunk_id field in payload
                from qdrant_client import models as qmodels
                
                # Create filter for matching chunk_ids using MatchAny
                chunk_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="chunk_id",
                            match=qmodels.MatchAny(any=batch_chunk_ids)
                        )
                    ]
                )
                
                # Use scroll to retrieve all matching points
                results, _ = await asyncio.wait_for(
                    self.qdrant_client.scroll(
                        collection_name=self.qdrant_collection,
                        scroll_filter=chunk_filter,
                        limit=batch_size,  # Limit to batch size
                        with_payload=True,  # Need payload to get chunk_id
                        with_vectors=True   # Need vectors for embeddings
                    ),
                    timeout=30.0  # 30 second timeout per batch
                )
                
                batch_duration = time.time() - batch_start
                logger.info(f"Qdrant batch scroll ({len(batch_chunk_ids)} chunk IDs) took {batch_duration:.3f}s, got {len(results)} results")
                
                # Extract embeddings from results using chunk_id from payload
                for point in results:
                    if point.payload and 'chunk_id' in point.payload:
                        chunk_id = point.payload['chunk_id']
                        
                        # Handle different vector formats
                        if point.vector and isinstance(point.vector, list):
                            embeddings_map[chunk_id] = point.vector
                        elif point.vector and isinstance(point.vector, dict):
                            # Handle named vectors if needed
                            # Assuming default vector name or first vector
                            for vector_name, vector_data in point.vector.items():
                                embeddings_map[chunk_id] = vector_data
                                break
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout retrieving embeddings batch {i//batch_size + 1}")
                continue
            except Exception as e:
                logger.error(f"Error retrieving embeddings batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info(f"Retrieved {len(embeddings_map)} embeddings out of {len(unique_chunk_ids)} requested")
        return embeddings_map

    async def get_qdrant_related_memories_async_fallback(self, acl_filter: qmodels.Filter, top_k=20):
        """
        Fallback method to get memories from Qdrant using only ACL filter with resilient error handling
        This is used when the main vector search returns no results.
        
        Args:
            acl_filter (qmodels.Filter): Qdrant filter object for ACL
            top_k (int): Number of results to return
            
        Returns:
            List of memory items with their metadata and default scores
        """
        # Retry configuration for connection issues
        max_retries = 2  # Fewer retries for fallback since it's already a fallback
        base_delay = 0.5  # Shorter delay for fallback
        timeout_seconds = 20.0  # Increased timeout for fallback during batch operations
        
        for attempt in range(max_retries):
            try:
                # Time the fallback search
                search_start = time.time()
                
                # Log fallback search parameters (only on first attempt to reduce noise)
                if attempt == 0:
                    logger.info(f"Qdrant FALLBACK search params - collection: {self.qdrant_collection}, top_k: {top_k}")
                    if acl_filter:
                        logger.info(f"Qdrant FALLBACK ACL filter: {acl_filter}")
                
                from qdrant_client import models as qmodels
                
                # Add timeout wrapper around the Qdrant fallback operation
                results, _ = await asyncio.wait_for(
                    self.qdrant_client.scroll(
                        collection_name=self.qdrant_collection,
                        scroll_filter=acl_filter,
                        limit=top_k,
                        with_payload=True,
                        with_vectors=False
                    ),
                    timeout=timeout_seconds
                )
                
                search_duration = time.time() - search_start
                
                logger.warning(f"Qdrant FALLBACK search returned {len(results)} results in {search_duration:.3f}s")
                
                # Log the first fallback result (only on successful searches)
                if results:
                    first_result = results[0]
                    logger.info(f"=== FIRST QDRANT FALLBACK RESULT ===")
                    logger.info(f"ID: {first_result.id}")
                    logger.info(f"Payload keys: {list(first_result.payload.keys()) if first_result.payload else 'None'}")
                    if first_result.payload:
                        chunk_id = first_result.payload.get('chunk_id', 'N/A')
                        content = first_result.payload.get('content', 'N/A')[:100] + "..." if first_result.payload.get('content') else 'N/A'
                        user_id = first_result.payload.get('user_id', 'N/A')
                        logger.info(f"  - chunk_id: {chunk_id}")
                        logger.info(f"  - content preview: {content}")
                        logger.info(f"  - user_id: {user_id}")
                    logger.info(f"=== END FIRST FALLBACK RESULT ===")
                else:
                    logger.warning("No Qdrant fallback results found")
                
                # Convert to standard dictionary format with default scores
                matches = []
                for match in results:
                    try:
                        chunk_id = match.payload.get('chunk_id', match.id)
                        matches.append({
                            'id': chunk_id,
                            'score': 0.1,  # Default fallback score (low but not zero)
                            'metadata': match.payload
                        })
                    except Exception as e:
                        logger.warning(f"Error processing Qdrant fallback result: {e}")
                        continue
                
                return {"matches": matches}
                
            except asyncio.TimeoutError:
                retry_delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Qdrant fallback timeout (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay:.1f}s...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Qdrant fallback failed after {max_retries} timeout attempts")
                    break
                    
            except Exception as e:
                # Check if it's a connection-related error that should be retried
                error_str = str(e).lower()
                is_connection_error = any(keyword in error_str for keyword in [
                    'timeout', 'connect', 'connection', 'network', 'unreachable', 
                    'responsehandlingexception', 'httperror', 'connectionerror'
                ])
                
                if is_connection_error and attempt < max_retries - 1:
                    retry_delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Qdrant fallback connection error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay:.1f}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Qdrant fallback failed (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
                    break
        
        # All retries exhausted
        logger.error("Qdrant fallback search failed after all retry attempts. Continuing with other data sources.")
        return {"matches": []}

    async def get_user_memGraph_schema_neo_async(
        self,
        user_id: str,
        neo_session: AsyncSession,
        acl_filter: Dict[str, Any] = None,
        timeout_seconds: int = 180
    ) -> Dict[str, List[str]]:
        """
        Get the user's memory graph schema from Neo4j using ACL filter - async version
        
        Args:
            user_id: The user's ID
            acl_filter: Dictionary containing ACL filter conditions
            neo_session: Optional existing Neo4j session to reuse
            timeout_seconds: Maximum time to wait for query execution
        Returns:
            Dict containing lists of nodes and relationships
        """
        # Query to get distinct node labels with ACL filtering
        node_labels_query = """
        MATCH (n)
        WHERE (
            n.user_id = $user_id OR
            any(x IN n.user_read_access WHERE x IN $user_read_access) OR
            any(x IN n.workspace_read_access WHERE x IN $workspace_read_access) OR
            any(x IN n.role_read_access WHERE x IN $role_read_access) OR
            any(x IN n.organization_read_access WHERE x IN $organization_read_access) OR
            any(x IN n.namespace_read_access WHERE x IN $namespace_read_access)
        )
        RETURN DISTINCT labels(n) AS labels
        """
        
        relationship_types_query = """
        MATCH (m)-[r]-(n)
        WHERE (
            m.user_id = $user_id OR
            any(x IN m.user_read_access WHERE x IN $user_read_access) OR
            any(x IN m.workspace_read_access WHERE x IN $workspace_read_access) OR
            any(x IN m.role_read_access WHERE x IN $role_read_access) OR
            any(x IN m.organization_read_access WHERE x IN $organization_read_access) OR
            any(x IN m.namespace_read_access WHERE x IN $namespace_read_access)
        )
        RETURN DISTINCT type(r) AS type
        """

        try:
            # Only ensure connection if we don't have a session

            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, cannot get schema")
                return {'nodes': [], 'relationships': []}
                
            # Extract filter values from acl_filter
            filter_params = {
                "user_id": user_id,
                "user_read_access": [],
                "workspace_read_access": [],
                "role_read_access": [],
                "organization_read_access": [],
                "namespace_read_access": []
            }

            if acl_filter and "$or" in acl_filter:
                for condition in acl_filter["$or"]:
                    if "user_read_access" in condition:
                        filter_params["user_read_access"] = condition["user_read_access"].get("$in", [])
                    elif "workspace_read_access" in condition:
                        filter_params["workspace_read_access"] = condition["workspace_read_access"].get("$in", [])
                    elif "role_read_access" in condition:
                        filter_params["role_read_access"] = condition["role_read_access"].get("$in", [])
                    elif "organization_read_access" in condition:
                        filter_params["organization_read_access"] = condition["organization_read_access"].get("$in", [])
                    elif "namespace_read_access" in condition:
                        filter_params["namespace_read_access"] = condition["namespace_read_access"].get("$in", [])
            
            valid_nodes: Set[str] = set()
            valid_relationships: List[str] = []

            try:
                # Add timeout context
                async with asyncio.timeout(timeout_seconds):
                    # Execute queries asynchronously
                    node_labels_result = await neo_session.run(node_labels_query, filter_params)
                    relationship_types_result = await neo_session.run(relationship_types_query, filter_params)

                    # Process node labels
                    async for record in node_labels_result:
                        labels = record.get("labels", [])
                        for label in labels:
                            if label == "Bug":
                                valid_nodes.add("Task")  # Map Bug to Task
                            else:
                                try:
                                    # Validate the label
                                    node_label = NodeLabel(label)
                                    valid_nodes.add(label)
                                except ValueError:
                                    logger.warning(f"Skipping invalid node label: {label}")

                    # Process relationship types
                    async for record in relationship_types_result:
                        rel_type = record.get("type")
                        if rel_type and rel_type in [r.value for r in RelationshipType]:
                            valid_relationships.append(rel_type)

                    # Consume results
                    await node_labels_result.consume()
                    await relationship_types_result.consume()

            except asyncio.TimeoutError:
                logger.error(f"Neo4j schema query timed out after {timeout_seconds} seconds")
                # Record failure in circuit breaker for timeout
                await self.async_neo_conn.circuit_breaker.record_failure()
                # Record failure in circuit breaker for timeout
                await self.async_neo_conn.circuit_breaker.record_failure()
                self.async_neo_conn.fallback_mode = True
                self.async_neo_conn.last_fallback_time = time.time()
                return {'nodes': [], 'relationships': []}
            except Exception as e:
                logger.error(f"Error executing Neo4j schema queries: {str(e)}")
                return {'nodes': [], 'relationships': []}

            logger.info(f'Processed nodes: {valid_nodes}')
            logger.info(f'Processed relationships: {valid_relationships}')

            return {
                'nodes': list(valid_nodes),
                'relationships': valid_relationships
            }

        except Exception as e:
            logger.error(f"Error in get_user_memGraph_schema_neo_async: {str(e)}")
            logger.exception(e)
            return {'nodes': [], 'relationships': []}
    
    def convert_sets_to_lists(self, obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets_to_lists(i) for i in obj]
        return obj


    async def _execute_fallback_query_without_property_filters(
        self,
        neo_session: AsyncSession,
        cipher_query: str,
        parameters: Dict[str, Any],
        enhancement_params: Dict[str, Any]
    ) -> Optional[Tuple[List[MemoryNodeProperties], List[NeoNode], str, str]]:
        """
        Helper method to execute a fallback Neo4j query with property filters removed.
        
        Returns:
            Tuple of (memory_nodes, neo_nodes, query, context) if successful
            None if fallback fails or returns no results
        """
        logger.warning(f"🔄 PROPERTY FILTER FALLBACK: Retrying query without property filters...")
        logger.info(f"🔄 Original query had {len(enhancement_params)} property parameters: {list(enhancement_params.keys())}")
        
        try:
            # Remove property parameters from the parameters dict
            fallback_parameters = {k: v for k, v in parameters.items() if k not in enhancement_params}
            
            # Strip property filter conditions from the query
            fallback_query = cipher_query
            for param_key in enhancement_params.keys():
                import re
                # Pattern to match property conditions with our parameters
                pattern = rf'\s+AND\s+\([^)]*\$({re.escape(param_key)})[^)]*\)'
                fallback_query = re.sub(pattern, '', fallback_query, flags=re.IGNORECASE)
                # Also handle non-parenthesized conditions
                pattern = rf'\s+AND\s+\w+\.\w+\s+\w+\s+\$({re.escape(param_key)})'
                fallback_query = re.sub(pattern, '', fallback_query, flags=re.IGNORECASE)
            
            logger.info(f"🔄 FALLBACK QUERY (without property filters): {fallback_query}")
            logger.info(f"🔄 FALLBACK PARAMETERS: {fallback_parameters}")
            
            # Execute fallback query
            fallback_start = time.time()
            async with asyncio.timeout(180):
                fallback_records = await neo_session.run(fallback_query, fallback_parameters)
                
                fallback_memory_nodes: List[MemoryNodeProperties] = []
                fallback_neo_nodes: List[NeoNode] = []
                fallback_paths: List[GraphPath] = []
                fallback_record_count = 0
                
                async for record in fallback_records:
                    fallback_record_count += 1
                    result_data = record.get('result', record)
                    
                    if 'path' in result_data:
                        path_data = result_data['path']
                        
                        if hasattr(path_data, 'nodes'):
                            for node in path_data.nodes:
                                node_dict = dict(node.items())
                                node_labels = list(node.labels)
                                primary_label = node_labels[0] if node_labels else None
                                
                                try:
                                    converted_node = NodeConverter.convert_to_neo_node(node_dict, primary_label)
                                    if converted_node:
                                        if primary_label == 'Memory':
                                            fallback_memory_nodes.append(converted_node.properties)
                                        else:
                                            fallback_neo_nodes.append(converted_node)
                                except ValidationError:
                                    pass
                        
                        if hasattr(path_data, 'relationships'):
                            segments = []
                            for rel in path_data.relationships:
                                try:
                                    # Convert start and end nodes
                                    start_dict = dict(rel.start_node.items())
                                    start_labels = list(rel.start_node.labels)
                                    start_label = start_labels[0] if start_labels else None
                                    
                                    end_dict = dict(rel.end_node.items())
                                    end_labels = list(rel.end_node.labels)
                                    end_label = end_labels[0] if end_labels else None
                                    
                                    converted_start = NodeConverter.convert_to_neo_node(start_dict, start_label)
                                    converted_end = NodeConverter.convert_to_neo_node(end_dict, end_label)
                                    
                                    if converted_start and converted_end:
                                        rel_type_str = str(rel.type).upper()
                                        if rel_type_str.startswith('RELATIONSHIPTYPE_'):
                                            rel_type_str = rel_type_str[len('RELATIONSHIPTYPE_'):]
                                        
                                        segment = PathSegment(
                                            start_node=converted_start.properties,
                                            relationship=rel_type_str,
                                            end_node=converted_end.properties
                                        )
                                        segments.append(segment)
                                except Exception:
                                    pass
                            
                            if segments:
                                try:
                                    fallback_path = GraphPath(
                                        segments=segments,
                                        length=len(segments)
                                    )
                                    fallback_paths.append(fallback_path)
                                except Exception:
                                    pass
                
                fallback_time = time.time() - fallback_start
                logger.warning(f"🔄 FALLBACK QUERY TIME: {fallback_time:.2f}s")
                logger.warning(f"🔄 FALLBACK RESULTS: Found {fallback_record_count} records, {len(fallback_neo_nodes)} graph nodes")
                
                if len(fallback_neo_nodes) > 0:
                    # Use fallback results
                    logger.info(f"✅ FALLBACK SUCCESS: Using results without property filters")
                    
                    fallback_query_result = QueryResult(
                        paths=fallback_paths,
                        query=fallback_query
                    )
                    fallback_text_context = fallback_query_result.get_related_context()
                    
                    # Remove duplicates
                    fallback_memory_nodes = list({node.id: node for node in fallback_memory_nodes}.values())
                    fallback_neo_nodes = list({node.properties.id: node for node in fallback_neo_nodes}.values())
                    
                    logger.warning(f"🔄 FINAL FALLBACK RESULTS: {len(fallback_memory_nodes)} memory nodes, {len(fallback_neo_nodes)} graph nodes")
                    return fallback_memory_nodes, fallback_neo_nodes, fallback_query, fallback_text_context
                else:
                    logger.warning(f"❌ FALLBACK FAILED: Still no results without property filters")
                    return None
        
        except Exception as fallback_error:
            logger.error(f"❌ FALLBACK ERROR: Failed to execute query without property filters: {fallback_error}")
            return None
    
    async def query_neo4j_with_user_query_async(
        self, 
        session_token: str, 
        query_context_combined: str, 
        acl_filter: Dict[str, Any],  
        user_id: str, 
        chat_gpt: "ChatGPTCompletion",
        neo_session: AsyncSession,
        project_id: str = None, 
        top_k: int = 10,
        api_key: Optional[str] = None,
        cached_schema: Optional[Dict[str, Any]] = None,  # ActivePatterns for enhanced Cypher generation
        user_workspace_ids: Optional[List[str]] = None,  # Add workspace IDs for proper ACL
        user_organization_ids: Optional[List[str]] = None,  # Add organization IDs for proper schema lookup
        user_namespace_ids: Optional[List[str]] = None  # Add namespace IDs for proper schema lookup
    ) -> Tuple[List[MemoryNodeProperties], List[NeoNode], str, str]:
        """
        Run a Neo4j query generated by LLM. If query generation fails, returns empty results as if skip_neo is true.
        """
        start_total = time.time()
        try:
            # Connection check timing
            connection_start = time.time()
            
            # Get the user's memory graph schema from Neo4j - convert to async
            # Schema retrieval timing
            schema_start = time.time()
            
            # Skip expensive schema discovery if we have cached patterns
            if cached_schema and cached_schema.get('patterns') and cached_schema.get('nodes') and cached_schema.get('relationships'):
                logger.info("🚀 FAST CACHE: Using cached schema, skipping expensive Neo4j discovery")
                memory_graph_schema = {
                    'nodes': cached_schema.get('nodes', []),
                    'relationships': cached_schema.get('relationships', [])
                }
                logger.warning(f"Schema retrieval (cached) took: {time.time() - schema_start:.3f}s")
            else:
                logger.info("🔍 SLOW DISCOVERY: No cached schema available, doing expensive Neo4j discovery")
                memory_graph_schema = await self.get_user_memGraph_schema_neo_async(
                    user_id, 
                    neo_session=neo_session,
                    acl_filter=acl_filter            
                )
                logger.warning(f"Schema retrieval (discovery) took: {time.time() - schema_start:.2f}s")
            
            logger.info(f'query_neo4j_with_user_query_async memory_graph_schema: {memory_graph_schema}')
            logger.warning(f"Schema retrieval and conversion took: {time.time() - schema_start:.2f}s")

            # Memory ID processing timing
            process_ids_start = time.time()
            memory_graph_converted = self.convert_sets_to_lists(memory_graph_schema)
            
            # Store enhanced schema cache for property indexing
            self._cached_schema = cached_schema
            
            # Add cached_schema patterns to memory_graph for enhanced Cypher generation
            logger.info(f"🔧 DEBUG: cached_schema type={type(cached_schema)}, is_none={cached_schema is None}, value={cached_schema}")
            if cached_schema and 'patterns' in cached_schema:
                if 'patterns' not in memory_graph_converted:
                    memory_graph_converted['patterns'] = []
                # Merge cached patterns with existing patterns
                cached_patterns = cached_schema['patterns']
                if isinstance(cached_patterns, list):
                    memory_graph_converted['patterns'].extend(cached_patterns)
                    logger.info(f"🔧 Added {len(cached_patterns)} cached ActivePatterns to memory_graph for Cypher generation")
                else:
                    logger.warning(f"🔧 cached_schema patterns is not a list: {type(cached_patterns)}")
            else:
                logger.info(f"🔧 No cached_schema patterns available for Cypher generation")

            logger.warning(f"Memory ID processing took: {time.time() - process_ids_start:.2f}s")

            # Generate the Neo4j cipher query - convert to async if needed
            # LLM Query generation timing
            llm_start = time.time()
            # Convert ACL filter to a plain dict for LLM function (avoid passing qdrant Filter objects)
            acl_filter_dict: Dict[str, Any] = {}
            try:
                if hasattr(acl_filter, 'model_dump'):
                    dumped = acl_filter.model_dump()
                    # qdrant Filter structure uses keys like 'must'/'should' with FieldCondition entries; flatten to $or list
                    or_list: List[Dict[str, Any]] = []
                    conditions = []
                    if 'should' in dumped and isinstance(dumped['should'], list):
                        conditions.extend(dumped['should'])
                    if 'must' in dumped and isinstance(dumped['must'], list):
                        conditions.extend(dumped['must'])
                    for c in conditions:
                        key = c.get('key') or c.get('field')
                        match = c.get('match') or {}
                        if key and isinstance(match, dict):
                            if 'value' in match:
                                or_list.append({key: {'$eq': match['value']}})
                            elif 'any' in match and isinstance(match['any'], list):
                                or_list.append({key: {'$in': match['any']}})
                    if or_list:
                        acl_filter_dict = {'$or': or_list}
                elif isinstance(acl_filter, dict):
                    acl_filter_dict = acl_filter
                else:
                    acl_filter_dict = {}
                
                # Add workspace_id, organization_id, namespace_id as top-level keys for schema service
                # Extract from user_workspace_ids, user_organization_ids, user_namespace_ids
                if user_workspace_ids and len(user_workspace_ids) > 0:
                    acl_filter_dict['workspace_id'] = user_workspace_ids[0]
                if user_organization_ids and len(user_organization_ids) > 0:
                    acl_filter_dict['organization_id'] = user_organization_ids[0]
                if user_namespace_ids and len(user_namespace_ids) > 0:
                    acl_filter_dict['namespace_id'] = user_namespace_ids[0]
                    
                logger.info(f"🔧 ACL FILTER DICT: workspace_id={acl_filter_dict.get('workspace_id')}, org_id={acl_filter_dict.get('organization_id')}, namespace_id={acl_filter_dict.get('namespace_id')}")
            except Exception as e:
                logger.warning(f"Failed to convert ACL filter to dict for LLM query: {e}")
                acl_filter_dict = {}

            cipher_query, is_llm_generated, enhancement_params = await chat_gpt.generate_neo4j_cipher_query_async(
                user_query=query_context_combined,
                acl_filter=acl_filter_dict,
                context=None,
                project_id=project_id,
                user_id=user_id,
                memory_graph=self,  # Pass the actual MemoryGraph instance for property search
                memory_graph_schema=memory_graph_converted,  # Pass the schema dict for schema operations
                top_k=top_k,
                enhanced_schema_cache=cached_schema,  # Pass enhanced schema cache for property enhancement
                neo_session=neo_session  # Pass neo_session for node count checks in property enhancement
            )
            logger.warning(f"LLM query generation took: {time.time() - llm_start:.2f}s")
            
            # ALWAYS log the generated query, even if empty or invalid
            logger.warning(f"🔍 GENERATED CYPHER QUERY (is_llm_generated={is_llm_generated}): {cipher_query}")
            logger.warning(f"🔍 QUERY LENGTH: {len(cipher_query) if cipher_query else 0} characters")
            
            if enhancement_params:
                logger.info(f"🔧 ENHANCEMENT PARAMS: Received {len(enhancement_params)} parameters from property enhancement: {list(enhancement_params.keys())}")

            # If no query was generated, return empty results (like skip_neo)
            if not cipher_query or not is_llm_generated:
                logger.warning("No Cypher query generated or not LLM-generated. Returning empty Neo4j results.")
                return [], [], None, ""

            # ACL and parameter preparation timing
            param_start = time.time()
            # Safely extract ACL parameters with type checking from the converted dict
            acl_or_conditions: List[Dict[str, Any]] = []
            try:
                if isinstance(acl_filter_dict, dict):
                    acl_or_conditions = acl_filter_dict.get('$or', []) or []
                    logger.warning(f"🔍 ACL DEBUG: acl_filter_dict keys: {list(acl_filter_dict.keys())}")
                    logger.warning(f"🔍 ACL DEBUG: acl_or_conditions count: {len(acl_or_conditions)}")
                    if acl_or_conditions:
                        logger.warning(f"🔍 ACL DEBUG: First condition: {acl_or_conditions[0]}")
            except Exception as e:
                logger.warning(f"Failed to read ACL $or conditions: {e}")
            
            # Initialize default empty lists for ACL parameters
            user_read_access: List[str] = []
            workspace_read_access: List[str] = []
            role_read_access: List[str] = []
            organization_read_access: List[str] = []
            namespace_read_access: List[str] = []
            
            # Safely extract values from ACL conditions
            for condition in acl_or_conditions:
                if not isinstance(condition, dict):
                    continue
                logger.warning(f"🔍 ACL DEBUG: Processing condition: {condition}")
                
                if 'user_read_access' in condition:
                    user_read_access = condition.get('user_read_access', {}).get('$in', [])
                    logger.warning(f"🔍 ACL DEBUG: Found user_read_access: {user_read_access}")
                elif 'workspace_read_access' in condition:
                    workspace_read_access = condition.get('workspace_read_access', {}).get('$in', [])
                    logger.warning(f"🔍 ACL DEBUG: Found workspace_read_access: {workspace_read_access}")
                elif 'role_read_access' in condition:
                    role_read_access = condition.get('role_read_access', {}).get('$in', [])
                    logger.warning(f"🔍 ACL DEBUG: Found role_read_access: {role_read_access}")
                elif 'organization_read_access' in condition:
                    organization_read_access = condition.get('organization_read_access', {}).get('$in', [])
                    logger.warning(f"🔍 ACL DEBUG: Found organization_read_access: {organization_read_access}")
                elif 'namespace_read_access' in condition:
                    namespace_read_access = condition.get('namespace_read_access', {}).get('$in', [])
                    logger.warning(f"🔍 ACL DEBUG: Found namespace_read_access: {namespace_read_access}")
                elif 'user_id' in condition:
                    # Handle user_id condition - add to user_read_access if not already there
                    user_value = condition.get('user_id', {}).get('$eq')
                    if user_value and user_value not in user_read_access:
                        user_read_access.append(user_value)
                elif 'organization_id' in condition:
                    # Handle organization_id condition - add to organization_read_access
                    org_value = condition.get('organization_id', {}).get('$eq')
                    if org_value and org_value not in organization_read_access:
                        organization_read_access.append(org_value)
                        logger.warning(f"🔍 ACL DEBUG: Added organization_id to organization_read_access: {org_value}")
                elif 'namespace_id' in condition:
                    # Handle namespace_id condition - add to namespace_read_access
                    namespace_value = condition.get('namespace_id', {}).get('$eq')
                    if namespace_value and namespace_value not in namespace_read_access:
                        namespace_read_access.append(namespace_value)
                        logger.warning(f"🔍 ACL DEBUG: Added namespace_id to namespace_read_access: {namespace_value}")

            # ALWAYS add current user_id to user_read_access for search operations
            if user_id and user_id not in user_read_access:
                user_read_access.append(user_id)
                logger.warning(f"🔍 ACL DEBUG: Added current user_id to user_read_access: {user_id}")
            
            # ALWAYS add current workspace_ids to workspace_read_access for search operations
            if user_workspace_ids:
                for workspace_id in user_workspace_ids:
                    if workspace_id and workspace_id not in workspace_read_access:
                        workspace_read_access.append(workspace_id)
                        logger.warning(f"🔍 ACL DEBUG: Added current workspace_id to workspace_read_access: {workspace_id}")

            # Prepare parameters for the query
            cipher_relationship_types = memory_graph_converted.get('relationships', []) if memory_graph_converted else []
            sanitized_relationship_types = [rel.replace('-', '_') for rel in cipher_relationship_types] if cipher_relationship_types else []

            # Extract ACL parameters from the dictionary structure based on query type
            parameters: Dict[str, Any] = {
                'top_k': top_k,
                'user_id': user_id,
                'user_read_access': user_read_access,
                'workspace_read_access': workspace_read_access,
                'role_read_access': role_read_access,
                'organization_read_access': organization_read_access,
                'namespace_read_access': namespace_read_access
            }

            # Add additional parameters only for fallback query
            if not is_llm_generated:
                parameters.update({
                    'sanitized_relationship_types': sanitized_relationship_types
                })
            
            # Merge property enhancement parameters (safely handles parameterized property values)
            if enhancement_params:
                parameters.update(enhancement_params)
                logger.info(f"🔧 MERGED PARAMS: Added {len(enhancement_params)} enhancement parameters to Cypher query")

            logger.warning(f'Query type: {"LLM Generated" if is_llm_generated else "Fallback"}')
            logger.warning(f'Cipher query: {cipher_query}')
            logger.warning(f'Parameters: {parameters}')
            logger.warning(f"Parameter preparation took: {time.time() - param_start:.2f}s")

            # Execute query using async session
            try:
                query_start_time = time.time()
                # Add timeout context (180 seconds)
                async with asyncio.timeout(180):
                    records = await neo_session.run(cipher_query, parameters)
                
                    memory_nodes: List[MemoryNodeProperties] = []
                    neo_nodes: List[NeoNode] = []
                    paths: List[GraphPath] = []

                    record_count = 0
                    
                    async for record in records:
                        record_count += 1
                        logger.info(f"\n=== Processing record {record_count} ===")
                        
                        # Log the raw record structure
                        logger.info(f"Record keys: {record.keys()}")
                        logger.info(f"Raw record data: {record.data()}")  # Add this to see full record
                        
                        # Check if we have a 'result' key or if the path is directly in the record
                        result_data = record.get('result', record)  # Fallback to record if no 'result' key
                        
                        if 'path' in result_data:
                            path_data = result_data['path']
                            logger.info(f"Path data type: {type(path_data)}")
                            logger.info(f"Path data attributes: {dir(path_data)}")
                            
                            # Extract nodes from path
                            if hasattr(path_data, 'nodes'):
                                for node in path_data.nodes:
                                    node_dict = dict(node.items())
                                    node_labels = list(node.labels)
                                    primary_label = node_labels[0] if node_labels else None
                                    
                                    logger.warning(f"\nProcessing node from path:")
                                    logger.warning(f"Labels: {node_labels}")
                                    logger.warning(f"Properties: {json.dumps(convert_datetimes(node_dict), indent=2, default=str)}")
                                    
                                    try:
                                        converted_node = NodeConverter.convert_to_neo_node(node_dict, primary_label)
                                        if converted_node:
                                            if primary_label == 'Memory':
                                                memory_nodes.append(converted_node.properties)
                                            else:
                                                neo_nodes.append(converted_node)
                                            logger.warning(f"✓ Successfully added {primary_label} node")
                                        else:
                                            logger.info(f"Failed to convert {primary_label} node")
                                            
                                    except Exception as e:
                                        logger.error(f"Error processing {primary_label} node: {str(e)}")
                                        logger.error(f"Node data: {json.dumps(convert_datetimes(node_dict), indent=2)}")
                                        continue

                            # Process relationships from path
                            if hasattr(path_data, 'relationships'):
                                segments = []
                                for i in range(len(path_data.relationships)):
                                    rel = path_data.relationships[i]
                                    start_node_dict = dict(path_data.nodes[i].items())
                                    end_node_dict = dict(path_data.nodes[i + 1].items())

                                    # Get labels for both nodes
                                    start_node_labels = list(path_data.nodes[i].labels)
                                    end_node_labels = list(path_data.nodes[i + 1].labels)
                                    
                                    start_primary_label = start_node_labels[0] if start_node_labels else None
                                    end_primary_label = end_node_labels[0] if end_node_labels else None
                                    
                                    logger.debug("=== Node Label Debug ===")
                                    logger.debug(f"Start Node Primary Label: {start_primary_label}")
                                    logger.debug(f"Start Node Properties: {json.dumps(convert_datetimes(start_node_dict), indent=2)}")
                                    logger.debug(f"Relationship Type: {rel.type}")
                                    logger.debug(f"End Node Primary Label: {end_primary_label}")
                                    logger.debug(f"End Node Properties: {json.dumps(convert_datetimes(end_node_dict), indent=2)}")
                                    
                                    try:
                                        # Convert start node
                                        logger.info(f"start_node_dict before: {start_node_dict}")
                                        logger.info(f"start_primary_label before: {start_primary_label}")
                                        converted_start_node = NodeConverter.convert_to_neo_node(start_node_dict, start_primary_label)
                                        if not converted_start_node:
                                            logger.error(f"Failed to convert start node with label {start_primary_label} and {start_node_dict}")
                                            continue
                                            
                                        # Convert end node
                                        converted_end_node = NodeConverter.convert_to_neo_node(end_node_dict, end_primary_label)
                                        if not converted_end_node:
                                            logger.error(f"Failed to convert end node with label {end_primary_label}")
                                            continue
                                        
                                        # Normalize relationship type (strip unexpected prefixes like RELATIONSHIPTYPE_)
                                        rel_type_str = getattr(rel, 'type', '')
                                        if isinstance(rel_type_str, str):
                                            if rel_type_str.startswith('RELATIONSHIPTYPE_'):
                                                rel_type_str = rel_type_str[len('RELATIONSHIPTYPE_'):]
                                            rel_type_str = rel_type_str.upper()
                                        else:
                                            rel_type_str = str(rel_type_str).upper()

                                        # Create PathSegment with converted nodes
                                        segment = PathSegment(
                                            start_node=converted_start_node.properties,  # Use the properties from NeoNode
                                            relationship=rel_type_str,
                                            end_node=converted_end_node.properties      # Use the properties from NeoNode
                                        )
                                        segments.append(segment)
                                        logger.info(f"✓ Successfully added path segment: {start_primary_label}-[{rel.type}]-{end_primary_label}")
                                        
                                    except Exception as e:
                                        logger.error(f"Error creating path segment: {str(e)}")
                                        logger.error(f"Start node label: {start_primary_label}, End node label: {end_primary_label}")
                                        continue
                                # Add this block to create and append the path
                                if segments:
                                    try:
                                        path = GraphPath(
                                            segments=segments,
                                            length=len(segments)  # Add the required length field
                                        )
                                        paths.append(path)
                                        logger.debug(f"✓ Successfully created path with {len(segments)} segments")
                                    except Exception as e:
                                        logger.error(f"Error creating GraphPath: {str(e)}")
                    
                    query_time = time.time() - query_start_time
                    logger.warning(f"\n=== Query Performance ===")
                    logger.warning(f"Total query execution time: {query_time:.2f}s")
                    logger.info(f"Records processed: {record_count}")
                    logger.info(f"Memory nodes found: {len(memory_nodes)}")
                    logger.info(f"Neo nodes found: {len(neo_nodes)}")
                    logger.info(f"Paths created: {len(paths)}")
                    
                    # Result processing timing
                    process_start = time.time()
                    # Create QueryResult for context generation
                    query_result = QueryResult(
                        paths=paths,
                        query=cipher_query
                    )
                
                    # Get human-readable context
                    text_context = query_result.get_related_context()
                    logger.info(f"Generated context: {text_context}")
                    
                    # Remove duplicates while preserving order
                    memory_nodes = list({node.id: node for node in memory_nodes}.values())
                    neo_nodes = list({node.properties.id: node for node in neo_nodes}.values())
                    
                    logger.info(f"Found {len(memory_nodes)} memory nodes")
                    logger.info(f"Found {len(neo_nodes)} neo nodes")
                    if neo_nodes:
                        logger.info("Sample of neo nodes (first 3):")
                        for i, node in enumerate(neo_nodes[:3]):
                            logger.info(f"Node {i + 1}:")
                            logger.info(f"  Label: {node.label.value}")
                            logger.info(f"  Properties: {json.dumps(convert_datetimes(node.properties.model_dump()), indent=2)}")
                    logger.info(f"Generated context: {text_context}")

                    logger.warning(f"Result processing took: {time.time() - process_start:.2f}s")

                    total_time = time.time() - start_total
                    logger.warning(f"Total Neo4j query pipeline took: {total_time:.2f}s")
                    logger.warning(f"🔍 AGENTIC GRAPH RESULTS: Found {len(memory_nodes)} memory nodes and {len(neo_nodes)} graph nodes")
                    
                    # FALLBACK: If no results and query has property filters (CONTAINS/=), retry without them
                    # Check if query has property filters even if enhancement_params is empty
                    has_property_filters = cipher_query and ('CONTAINS' in cipher_query or '.name =' in cipher_query or '.description =' in cipher_query)
                    if len(neo_nodes) == 0 and has_property_filters:
                        fallback_result = await self._execute_fallback_query_without_property_filters(
                            neo_session=neo_session,
                            cipher_query=cipher_query,
                            parameters=parameters,
                            enhancement_params=enhancement_params
                        )
                        if fallback_result:
                            return fallback_result
                    
                    return memory_nodes, neo_nodes, cipher_query, text_context
            
            except asyncio.TimeoutError:
                logger.error("Neo4j query timed out after 180 seconds")
                logger.error(f"⏱️ TIMEOUT - Cypher query that timed out: {cipher_query}")
                logger.error(f"⏱️ TIMEOUT - Query parameters: {parameters}")
                logger.error(f"⏱️ TIMEOUT - Query length: {len(cipher_query) if cipher_query else 0} characters")
                # Record failure in circuit breaker for timeout
                await self.async_neo_conn.circuit_breaker.record_failure()
                # Record failure in circuit breaker for timeout
                await self.async_neo_conn.circuit_breaker.record_failure()
                self.async_neo_conn.fallback_mode = True
                self.async_neo_conn.last_fallback_time = time.time()
                return [], [], None, ""

            except Exception as e:
                logger.error(f"Error executing Neo4j query: {str(e)}")
                logger.error(f"❌ FAILED - Cypher query: {cipher_query}")
                logger.error(f"❌ FAILED - Query parameters: {parameters}")
                logger.error(f"❌ FAILED - Query length: {len(cipher_query) if cipher_query else 0} characters")
                logger.error(f"❌ FAILED - Exception type: {type(e).__name__}")
                logger.error(f"❌ FAILED - Exception details: {str(e)}")
                
                # FALLBACK: If query failed and we have property filters, retry without them
                if enhancement_params:
                    fallback_result = await self._execute_fallback_query_without_property_filters(
                        neo_session=neo_session,
                        cipher_query=cipher_query,
                        parameters=parameters,
                        enhancement_params=enhancement_params
                    )
                    if fallback_result:
                        # Record success since fallback worked
                        await self.async_neo_conn.circuit_breaker.record_success()
                        return fallback_result
                
                # Record failure in circuit breaker for any exception
                await self.async_neo_conn.circuit_breaker.record_failure()
                # Record failure in circuit breaker for any exception
                await self.async_neo_conn.circuit_breaker.record_failure()
                return [], [], None, ""

        except Exception as e:
            logger.error(f"Error in query_neo4j_with_user_query_async: {str(e)}")
            # Record failure in circuit breaker for any exception
            await self.async_neo_conn.circuit_breaker.record_failure()
            # Record failure in circuit breaker for any exception
            await self.async_neo_conn.circuit_breaker.record_failure()
            return [], [], None, ""
    
    def rank_combined_results(self, combined_results):
        # Implement a ranking algorithm based on relevance to the query
        # This could involve similarity scores from Pinecone and heuristic scores for BigBird groups
        pass
    
    

    async def delete_memory_item(
        self, 
        memory_id: str, 
        session_token: str,
        neo_session: AsyncSession,
        skip_parse: bool = False,
        api_key: Optional[str] = None,
        legacy_route: bool = True,
        user_id: Optional[str] = None,
        user_info: Optional[Dict[str, Any]] = None
    ) -> DeleteMemoryResponse:
        """
        Asynchronously deletes a memory item from storage systems.
        Returns DeleteMemoryResponse for both success and error cases.
        
        Args:
            memory_id: The ID of the memory to delete
            session_token: Session token for authentication
            neo_session: Neo4j session
            skip_parse: Whether to skip Parse Server deletion
            api_key: API key for authentication
            legacy_route: If True, delete from both Pinecone and Qdrant. If False, delete only from Qdrant.
        """
        deletion_status = DeletionStatus()
        exists_somewhere = False
        parse_object_id = ''
        try:
            if not memory_id:
                return DeleteMemoryResponse.failure(
                    error='Memory ID is required',
                    code=400
                )
            
            logger.info(f"Starting delete_memory_item with legacy_route={legacy_route}")
            
            # Get memory information based on skip_parse flag
            memory_chunk_ids = []
            if skip_parse:
                # Get memory info from Neo4j
                neo4j_memory = await self.get_memory_item(memory_id, neo_session)
                if neo4j_memory:
                    exists_somewhere = True
                    memory_chunk_ids = neo4j_memory.get('memoryChunkIds', [memory_id])
                    deletion_status.parse = True
            else:
                # Get from Parse Server as before
                parse_memory_item = await retrieve_memory_item_by_qdrant_id(
                    session_token, 
                    str(memory_id),
                    api_key=api_key
                )
                if parse_memory_item:
                    exists_somewhere = True
                    parse_object_id = parse_memory_item.get('objectId')
                    memory_chunk_ids = parse_memory_item.get('memoryChunkIds', [memory_id])

            # Ensure we have at least one ID to delete by using memory_id as fallback
            if not memory_chunk_ids:
                logger.warning(f"No memory chunk IDs found for {memory_id}, using memory_id as fallback")
                memory_chunk_ids = [memory_id, f"{memory_id}_0"]
            else:
                # Check if we have only one ID without chunk suffix
                if len(memory_chunk_ids) == 1 and not any(id.endswith(f"_{i}") for id in memory_chunk_ids for i in range(10)):
                    base_id = memory_chunk_ids[0]
                    logger.info(f"Single memory ID without chunk suffix found: {base_id}, adding chunked version")
                    memory_chunk_ids.append(f"{base_id}_0")
                # Ensure memory_id is included in chunk_ids if not already present
                if memory_id not in memory_chunk_ids:
                    memory_chunk_ids.append(memory_id)

            logger.info(f"Using memory chunk IDs for deletion: {memory_chunk_ids}")

            # Check vector store existence using memory chunk IDs
            try:
                # Check if vectors exist using our helper method
                for chunk_id in memory_chunk_ids:
                    try:
                        point, qdrant_id = await self.get_qdrant_point(chunk_id)
                        if point:
                            exists_somewhere = True
                            logger.info(f"Found Qdrant vector {qdrant_id} for memory chunk {chunk_id}")
                        else:
                            logger.warning(f"No Qdrant vector found for memory chunk {chunk_id}")
                    except Exception as search_error:
                        logger.error(f"Error checking for vector with chunk ID {chunk_id}: {search_error}")
                    
            except Exception as e:
                logger.error(f"Error checking vector store: {e}")
                
            # Return 404 if memory doesn't exist anywhere
            if not exists_somewhere:
                return DeleteMemoryResponse.failure(
                    error=f'Memory item with ID {memory_id} not found in any system',
                    code=404
                )

            # Delete from vector store (Pinecone or Qdrant) only if legacy_route is True
            if legacy_route:
                try:
                    # For legacy route, we need to handle both Pinecone and Qdrant
                    # Run both deletions in parallel
                    qdrant_task = self.delete_qdrant_points_parallel(memory_chunk_ids)
                    pinecone_task = self.delete_pinecone_points_parallel(memory_chunk_ids)
                    
                    # Wait for both to complete
                    qdrant_results, pinecone_results = await asyncio.gather(
                        qdrant_task, 
                        pinecone_task, 
                        return_exceptions=True
                    )
                    
                    # Handle Qdrant results
                    if isinstance(qdrant_results, Exception):
                        logger.error(f"Qdrant delete failed with exception: {qdrant_results}")
                        qdrant_successful = 0
                    else:
                        qdrant_successful = sum(1 for success in qdrant_results.values() if success)
                        total_chunks = len(memory_chunk_ids)
                        logger.info(f'Successfully deleted {qdrant_successful}/{total_chunks} chunks from Qdrant')
                    
                    # Handle Pinecone results
                    if isinstance(pinecone_results, Exception):
                        logger.error(f"Pinecone delete failed with exception: {pinecone_results}")
                        pinecone_successful = 0
                    else:
                        pinecone_successful = sum(1 for success in pinecone_results.values() if success)
                        total_chunks = len(memory_chunk_ids)
                        logger.info(f'Successfully deleted {pinecone_successful}/{total_chunks} chunks from Pinecone')
                    
                    # Mark as successful if at least one vector store had successful deletions
                    if qdrant_successful > 0 or pinecone_successful > 0:
                        deletion_status.pinecone = True
                        deletion_status.qdrant = True
                    else:
                        logger.warning("No vectors found to delete in either Qdrant or Pinecone")
                        deletion_status.pinecone = True  # Mark as successful since there's nothing to delete
                        deletion_status.qdrant = True
                        
                except Exception as e:
                    logger.error(f"Error deleting from vector stores: {e}")
                    logger.error("Full traceback:", exc_info=True)
            else:
                logger.info(f'Skipping vector store deletion for legacy_route=False')
                deletion_status.pinecone = True  # Mark as successful since we're intentionally skipping

            # Delete from Qdrant (always required for non-legacy routes)
            if not legacy_route:
                try:
                    # Use our new parallel delete method for Qdrant
                    delete_results = await self.delete_qdrant_points_parallel(memory_chunk_ids)
                    
                    # Check if any deletions were successful
                    successful_deletes = sum(1 for success in delete_results.values() if success)
                    total_chunks = len(memory_chunk_ids)
                    
                    if successful_deletes > 0:
                        logger.info(f'Successfully deleted {successful_deletes}/{total_chunks} chunks from Qdrant')
                        deletion_status.qdrant = True
                    else:
                        logger.warning("No Qdrant vectors found to delete")
                        deletion_status.qdrant = True  # Mark as successful since there's nothing to delete
                        
                except Exception as e:
                    logger.error(f"Error deleting from Qdrant: {e}")
                    logger.error("Full traceback:", exc_info=True)
            else:
                # For legacy routes, Qdrant deletion is handled in the vector store section above
                deletion_status.qdrant = True

            # Delete from Neo4j with proper async connection handling
            try:
                if self.async_neo_conn.fallback_mode:
                    logger.warning("Neo4j in fallback mode, skipping Neo4j deletion")
                    deletion_status.neo4j = True
                    return DeleteMemoryResponse.success(
                        memoryId=memory_id,
                        objectId=parse_object_id,
                        deletion_status=deletion_status,
                        code=200,
                        message='Neo4j in fallback mode, skipping Neo4j deletion'
                    )

                try:
                    # First verify the node exists
                    verify_result = await neo_session.run(
                        "MATCH (m:Memory {id: $id}) RETURN m",
                        {"id": memory_id}
                    )
                    verify_records = await verify_result.values()
                    if verify_records:
                        # Node exists, proceed with deletion
                        delete_result = await neo_session.run(
                            "MATCH (m:Memory {id: $id}) DETACH DELETE m",
                            {"id": memory_id}
                        )
                        await delete_result.consume()  # Ensure the operation completes
                        logger.info('Deleted from Neo4j')
                        deletion_status.neo4j = True
                    else:
                        logger.warning(f"Memory node with ID {memory_id} not found in Neo4j")
                        deletion_status.neo4j = True
                except Exception as e:
                    logger.error(f"Error in Neo4j operation: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error deleting from Neo4j: {e}", exc_info=True)

            # Delete from Parse Server
            try:
                # Delete from Parse Server only if not skipped
                if not skip_parse:
                    logger.info(f"Starting Parse Server deletion for memory_id: {memory_id}, parse_object_id: {parse_object_id}")
                    try:
                        parse_result = await delete_memory_item_parse(parse_object_id)
                        if parse_result:
                            logger.info(f'Successfully deleted from Parse Server: {memory_id}')
                            deletion_status.parse = True
                        else:
                            logger.error(f'Failed to delete from Parse Server: {memory_id}')
                            deletion_status.parse = False
                    except Exception as e:
                        logger.error(f"Error deleting from Parse Server: {e}")
                        logger.error("Full traceback:", exc_info=True)
                        deletion_status.parse = False
                else:
                    # Mark Parse deletion as successful when skipped
                    deletion_status.parse = True
                    logger.info(f'Skipped Parse Server deletion for: {memory_id}')

            except Exception as e:
                logger.error(f"Error deleting from Parse Server: {e}")
                logger.error("Full traceback:", exc_info=True)
                deletion_status.parse = False

            # Check if all deletions were successful
            if all(vars(deletion_status).values()):
                route_type = "legacy (Pinecone + Qdrant)" if legacy_route else "new (Qdrant only)"
                return DeleteMemoryResponse.success(
                    memoryId=memory_id,
                    objectId=parse_object_id,
                    deletion_status=deletion_status,
                    code=200,
                    message=f'Memory item successfully deleted from all systems using {route_type} route'
                )
            else:
                route_type = "legacy (Pinecone + Qdrant)" if legacy_route else "new (Qdrant only)"
                return DeleteMemoryResponse.failure(
                    error='Memory item deletion partially successful',
                    code=207,
                    message=f'Memory item deletion partially successful using {route_type} route',
                    details={"deletion_status": deletion_status.model_dump() if hasattr(deletion_status, 'model_dump') else dict(deletion_status)}
                )

        except Exception as e:
            logger.error(f"Unexpected error in delete_memory_item: {e}", exc_info=True)
            return DeleteMemoryResponse.failure(
                error=f'Unexpected error: {str(e)}',
                code=500
            )

    async def get_memory_item(
        self, 
        memory_id: str,
        neo_session: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """
        Asynchronously fetch a memory item from Neo4j by its ID.

        Args:
            memory_id (str): The ID of the memory item to retrieve
            neo_session (AsyncSession): The Neo4j session to use for the operation

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing memory item properties if found,
                                    None if not found or error occurs
        """
        # In open-source, try to connect even if fallback_mode is set (it might have been set incorrectly)
        # Only skip if we're in cloud and fallback_mode is set
        import os
        papr_edition = os.getenv("PAPR_EDITION", "").lower()
        is_opensource = papr_edition == "opensource"
        
        if self.async_neo_conn.fallback_mode and not is_opensource:
            # Cloud edition: Skip if fallback mode is set
            logger.warning("Neo4j in fallback mode, cannot get memory item")
            return None
        elif self.async_neo_conn.fallback_mode and is_opensource:
            # Open-source: Try to connect anyway - fallback mode might have been set incorrectly
            logger.info("Neo4j fallback mode is set, but attempting connection anyway (open-source)")
        
        # Try to get a session (with retry)
        session = None
        try:
            session = await self.get_safe_session(neo_session)
            # If we successfully got a session in open-source and fallback_mode was set, reset it
            if is_opensource and self.async_neo_conn.fallback_mode and session:
                logger.info("Successfully got Neo4j session - resetting fallback mode")
                self.async_neo_conn.fallback_mode = False
        except Exception as session_error:
            logger.warning(f"Session error encountered: {session_error}, retrying with a new session.")
            try:
                session = await self.get_safe_session(None)
                # If we successfully got a session in open-source and fallback_mode was set, reset it
                if is_opensource and self.async_neo_conn.fallback_mode and session:
                    logger.info("Successfully got Neo4j session after retry - resetting fallback mode")
                    self.async_neo_conn.fallback_mode = False
            except Exception as e:
                logger.error(f"Failed to get Neo4j session after retry: {e}")
                # Only set fallback mode if we're in open-source (cloud should handle this differently)
                if is_opensource:
                    self.async_neo_conn.fallback_mode = True
                return None

        try:
            # Check if we're in open-source edition - use different extraction method
            import os
            papr_edition = os.getenv("PAPR_EDITION", "").lower()
            is_opensource = papr_edition == "opensource"
            
            if is_opensource:
                # Open-source: Use properties(m) query for more reliable extraction
                cypher_query = """
                    MATCH (m:Memory)
                    WHERE m.id = $memory_id
                    RETURN properties(m) as props
                """
                parameters = {'memory_id': str(memory_id)}
                result = await session.run(cypher_query, parameters)
                record = await result.single()
                # Consume the result to avoid "result has been consumed" errors
                try:
                    await result.consume()
                except Exception:
                    pass  # Result may already be consumed by single()

                if record is None:
                    logger.info(f"No memory item found with ID: {memory_id}")
                    return None

                # Open-source: Add detailed logging to understand Neo4j record structure
                record_keys = list(record.keys()) if record else []
                record_values = list(record.values()) if record else []
                record_value_types = [type(v).__name__ for v in record_values] if record else []
                
                logger.info(f"🔍 [Open-Source] Neo4j record structure for memory_id={memory_id}:")
                logger.info(f"   Record keys: {record_keys}")
                logger.info(f"   Record value types: {record_value_types}")
                logger.info(f"   Record values: {record_values}")
                
                # Log detailed information about each value
                for i, (key, value) in enumerate(record.items()):
                    logger.info(f"   Record[{key}]: type={type(value).__name__}, value={value}")
                    if hasattr(value, '__dict__'):
                        logger.info(f"      __dict__: {value.__dict__}")
                    if hasattr(value, '__class__'):
                        logger.info(f"      __class__: {value.__class__}")
                    if hasattr(value, 'keys'):
                        try:
                            logger.info(f"      keys(): {list(value.keys())}")
                        except:
                            pass
                    if hasattr(value, 'items'):
                        try:
                            logger.info(f"      items(): {list(value.items())[:5]}...")  # First 5 items
                        except:
                            pass

                # Extract and return the properties of the memory item
                # Handle different record structures - Neo4j may return by alias or by numeric key
                memory_properties = None
                
                # Try accessing by alias 'props' first
                if 'props' in record:
                    memory_properties = record['props']
                    logger.info(f"✅ Found properties via 'props' key")
                # If not found, try accessing by numeric keys (e.g., '0', '1') - some Neo4j drivers return by index
                else:
                    logger.info(f"⚠️ 'props' key not found, trying to extract from keys: {record_keys}")
                    
                    # Try accessing by first key (might be '0', '1', 'props', etc.)
                    for key in record_keys:
                        value = record[key]
                        logger.info(f"   Processing key '{key}': type={type(value).__name__}")
                        
                        # Check if it's a dict (properties)
                        if isinstance(value, dict):
                            memory_properties = value
                            logger.info(f"✅ Found properties as dict via key '{key}'")
                            break
                        # Check if it's a node object that can be converted to dict
                        elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes, int)):
                            try:
                                # Try converting to dict
                                memory_properties = dict(value)
                                logger.info(f"✅ Converted to dict via key '{key}'")
                                break
                            except (TypeError, ValueError) as e:
                                logger.debug(f"   Failed to convert to dict: {e}")
                                # If it's a node, try accessing its properties attribute
                                if hasattr(value, 'items'):
                                    try:
                                        memory_properties = dict(value.items())
                                        logger.info(f"✅ Found properties via items() on key '{key}'")
                                        break
                                    except Exception as e2:
                                        logger.debug(f"   Failed to use items(): {e2}")
                                elif hasattr(value, '_properties'):
                                    try:
                                        memory_properties = dict(value._properties)
                                        logger.info(f"✅ Found properties via _properties on key '{key}'")
                                        break
                                    except Exception as e2:
                                        logger.debug(f"   Failed to use _properties: {e2}")
                                elif hasattr(value, 'get'):
                                    # Might be a node-like object with get method
                                    try:
                                        memory_properties = {k: value.get(k) for k in dir(value) if not k.startswith('_')}
                                        logger.info(f"✅ Found properties via get() on key '{key}'")
                                        break
                                    except Exception as e2:
                                        logger.debug(f"   Failed to use get(): {e2}")
                
                if not memory_properties:
                    logger.error(f"❌ Could not extract memory properties from Neo4j record. Record keys: {record_keys}, Record values types: {record_value_types}, Record values: {record_values}")
                    return None
                
                # Convert to dict if it's not already
                if not isinstance(memory_properties, dict):
                    memory_properties = dict(memory_properties) if hasattr(memory_properties, '__iter__') and not isinstance(memory_properties, (str, bytes)) else {}
                    
                logger.info(f"✅ Memory item properties extracted successfully: {len(memory_properties)} properties")
                return memory_properties
            else:
                # Cloud edition: Use original node-based extraction (preserve existing behavior)
                cypher_query = """
                    MATCH (m:Memory)
                    WHERE m.id = $memory_id
                    RETURN m
                """
                parameters = {'memory_id': str(memory_id)}
                result = await session.run(cypher_query, parameters)
                record = await result.single()
                # Consume the result to avoid "result has been consumed" errors
                try:
                    await result.consume()
                except Exception:
                    pass  # Result may already be consumed by single()

                if record is None:
                    logger.info(f"No memory item found with ID: {memory_id}")
                    return None

                # Extract and return the properties of the memory item
                # Cloud: Access node as record['m'] or record[0]
                memory_properties = None
                if 'm' in record:
                    # Node is keyed as 'm'
                    node = record['m']
                    memory_properties = dict(node) if hasattr(node, '__iter__') and not isinstance(node, (str, bytes)) else node
                elif len(record) > 0:
                    # Try accessing as first column
                    node = record[0]
                    if hasattr(node, '__iter__') and not isinstance(node, (str, bytes, int)):
                        memory_properties = dict(node)
                    else:
                        # If record[0] is not a node, try accessing by key
                        logger.warning(f"record[0] is not a node (type: {type(node)}), trying to access record keys: {list(record.keys())}")
                        # Try to get the node from record values
                        for value in record.values():
                            if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, int)):
                                memory_properties = dict(value)
                                break
                
                if not memory_properties:
                    logger.error(f"Could not extract memory properties from Neo4j record. Record keys: {list(record.keys()) if record else 'None'}")
                    return None
                    
                logger.info(f"Memory item properties: {memory_properties}")
                return memory_properties

        except Exception as e:
            logger.error(f"Error fetching memory item from Neo4j: {e}")
            logger.error("Full traceback:", exc_info=True)
            return None

    async def update_memory_item(
        self,
        session_token: str,
        memory_id: str,
        memory_type: str,
        content: str,
        metadata: dict,
        background_tasks: BackgroundTasks,
        neo_session: AsyncSession,
        context: Optional[List[ContextItem]] = None,
        api_key: Optional[str] = None,
        legacy_route: bool = True
    ) -> UpdateMemoryResponse:
        """
        Update a memory item in all storage systems (Pinecone, Neo4j, Parse Server).
        
        Args:
            session_token (str): The session token for authentication
            memory_id (str): The ID of the memory item to update
            memory_type (str): The type of memory item
            content (str): The new content for the memory item
            metadata (dict): The new metadata for the memory item
            background_tasks (BackgroundTasks): FastAPI background tasks object
            context (str, optional): The context for the memory item
            
        Returns:
            UpdateMemoryResponse: The response containing update status and updated item
        """
        status = SystemUpdateStatus()
        
        try:
            # Get the existing memory item
            existing_item = await self.get_memory_item(memory_id, neo_session=neo_session)
            if not existing_item:
                return UpdateMemoryResponse.failure(
                    error="Memory item not found",
                    code=404
                )

            # Extract the properties of the memory item
            # The existing_item is already a dictionary from Neo4j
            memory_properties = existing_item
            is_metadata_only_update = content is None or content == ""

            existing_memory_type = memory_properties.get('type')
            logger.info(f"existing_memory_type: {existing_memory_type}")
            logger.info(f"existing_item fron neo: {existing_item}")
            memory_type = memory_type if memory_type else existing_memory_type
     
            memory_properties['metadata'] = memory_properties.get('metadata', {})
            if context is not None:
                memory_properties['context'] = context
            
            if isinstance(memory_properties['metadata'], str):
                try:
                    memory_properties['metadata'] = json.loads(memory_properties['metadata'])
                except json.JSONDecodeError:
                    memory_properties['metadata'] = {}
            memory_properties['metadata'].update(metadata)

            # Sanitize metadata
            memory_properties['metadata'] = {
                k: str(v) if not isinstance(v, (str, int, float, bool, list)) else v
                for k, v in memory_properties['metadata'].items()
            }

            logger.info(f"Updated memory properties: {memory_properties}")

            # Get the vector ID from memoryChunkIds
            memory_chunk_ids = memory_properties.get('memoryChunkIds', [])
            if isinstance(memory_chunk_ids, str):
                try:
                    memory_chunk_ids = json.loads(memory_chunk_ids)
                except json.JSONDecodeError:
                    memory_chunk_ids = [id.strip() for id in memory_chunk_ids.split(',') if id.strip()]
            elif isinstance(memory_chunk_ids, list):
                memory_chunk_ids = [str(id).strip().strip("'[]\"") for id in memory_chunk_ids]
            else:
                memory_chunk_ids = []

            if not memory_chunk_ids:
                logger.warning(f"No memoryChunkIds found for memory {memory_id}, using memory_id as fallback")
                vector_id = str(memory_id)
            else:
                # Get the first chunk ID since we're dealing with a single chunk
                vector_id = str(memory_chunk_ids[0])

            if is_metadata_only_update:
                # For metadata-only updates (like ACL changes)
                logger.info("Performing metadata-only update")
                
                # Update the memory properties with new ACL values from metadata by appending rather than overwriting
                if 'user_read_access' in memory_properties['metadata']:
                    existing_user_read = set(memory_properties.get('user_read_access', []))
                    logger.info(f"existing_user_read: {existing_user_read}")
                    new_user_read = set(memory_properties['metadata']['user_read_access'])
                    memory_properties['user_read_access'] = list(existing_user_read | new_user_read)
                    # Update metadata to match
                    memory_properties['metadata']['user_read_access'] = memory_properties['user_read_access']
                    logger.info(f"memory_properties['user_read_access']: {memory_properties['user_read_access']}")

                if 'user_write_access' in memory_properties['metadata']:
                    existing_user_write = set(memory_properties.get('user_write_access', []))
                    logger.info(f"existing_user_write: {existing_user_write}")
                    new_user_write = set(memory_properties['metadata']['user_write_access'])
                    memory_properties['user_write_access'] = list(existing_user_write | new_user_write)
                    # Update metadata to match
                    memory_properties['metadata']['user_write_access'] = memory_properties['user_write_access']
                    logger.info(f"memory_properties['user_write_access']: {memory_properties['user_write_access']}")

                if 'workspace_read_access' in memory_properties['metadata']:
                    existing_workspace_read = set(memory_properties.get('workspace_read_access', []))
                    new_workspace_read = set(memory_properties['metadata']['workspace_read_access'])
                    memory_properties['workspace_read_access'] = list(existing_workspace_read | new_workspace_read)
                    # Update metadata to match
                    memory_properties['metadata']['workspace_read_access'] = memory_properties['workspace_read_access']
                    logger.info(f"memory_properties['workspace_read_access']: {memory_properties['workspace_read_access']}")

                if 'workspace_write_access' in memory_properties['metadata']:
                    existing_workspace_write = set(memory_properties.get('workspace_write_access', []))
                    new_workspace_write = set(memory_properties['metadata']['workspace_write_access'])
                    memory_properties['workspace_write_access'] = list(existing_workspace_write | new_workspace_write)
                    # Update metadata to match
                    memory_properties['metadata']['workspace_write_access'] = memory_properties['workspace_write_access']
                    logger.info(f"memory_properties['workspace_write_access']: {memory_properties['workspace_write_access']}")

                if 'role_read_access' in memory_properties['metadata']:
                    existing_role_read = set(memory_properties.get('role_read_access', []))
                    new_role_read = set(memory_properties['metadata']['role_read_access'])
                    memory_properties['role_read_access'] = list(existing_role_read | new_role_read)
                    # Update metadata to match
                    memory_properties['metadata']['role_read_access'] = memory_properties['role_read_access']
                    logger.info(f"memory_properties['role_read_access']: {memory_properties['role_read_access']}")

                if 'role_write_access' in memory_properties['metadata']:
                    existing_role_write = set(memory_properties.get('role_write_access', []))
                    new_role_write = set(memory_properties['metadata']['role_write_access'])
                    memory_properties['role_write_access'] = list(existing_role_write | new_role_write)
                    # Update metadata to match
                    memory_properties['metadata']['role_write_access'] = memory_properties['role_write_access']
                    logger.info(f"memory_properties['role_write_access']: {memory_properties['role_write_access']}")


                logger.info(f"memory_properties: {memory_properties}")
                
                # Get all chunk IDs for this memory
                memory_chunk_ids = memory_properties.get('memoryChunkIds', [])
                if isinstance(memory_chunk_ids, str):
                    try:
                        memory_chunk_ids = json.loads(memory_chunk_ids)
                    except json.JSONDecodeError:
                        memory_chunk_ids = [id.strip() for id in memory_chunk_ids.split(',') if id.strip()]
                elif isinstance(memory_chunk_ids, list):
                    memory_chunk_ids = [str(id).strip().strip("'[]\"") for id in memory_chunk_ids]
                
                # If no chunk IDs found, use memory_id as fallback
                if not memory_chunk_ids:
                    memory_chunk_ids = [str(memory_id)]
                
                # Run all updates in parallel for better performance
                async def update_vector_stores():
                    """Update all vector stores in parallel"""
                    vector_tasks = []
                    for chunk_id in memory_chunk_ids:
                        task = self.update_vector_store(
                            chunk_id=chunk_id,
                            new_metadata=memory_properties['metadata'],
                            embedding=None,  # No embedding for metadata-only update
                            legacy_route=legacy_route
                        )
                        vector_tasks.append(task)
                    
                    results = await asyncio.gather(*vector_tasks, return_exceptions=True)
                    # Handle exceptions
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"Vector store update failed for chunk {memory_chunk_ids[i]}: {result}")
                            results[i] = False
                    
                    return all(results)

                async def update_neo4j():
                    """Update Neo4j"""
                    try:
                        result = await self.update_memory_item_in_neo4j(
                            memory_properties, 
                            neo_session,
                            memory_type,
                            None
                        )
                        return result is not None
                    except Exception as e:
                        logger.error(f"Neo4j update failed: {e}")
                        return False

                async def update_parse():
                    """Update Parse Server"""
                    try:
                        parse_memory = await retrieve_memory_item_by_qdrant_id(
                            session_token, 
                            vector_id,
                            api_key=api_key
                        )
                        if parse_memory:
                            # Convert ACL for Parse update
                            parse_acl = convert_acl(memory_properties['metadata'])

                            logger.info(f"memory_chunk_ids: {memory_chunk_ids}")
                            logger.info(f"parse_memory: {parse_memory}")

                            parse_update_data = {
                                'objectId': parse_memory.get('objectId'),
                                'memoryId': memory_id,
                                'content': parse_memory.get('content', ''),  # Use existing content for metadata-only updates
                                'metadata': memory_properties['metadata'],
                                'customMetadata': memory_properties['metadata'],
                                'memoryChunkIds': memory_chunk_ids,   
                                'ACL': parse_acl  
                            }
                            logger.info(f"parse_update_data: {parse_update_data}")
                            parse_result = await update_memory_item(session_token, parse_update_data, None, api_key=api_key)
                            logger.info(f"parse_result: {parse_result}")
                            return True if parse_result and parse_result.memory_items and len(parse_result.memory_items) > 0 else False
                        return False
                    except Exception as e:
                        logger.error(f"Parse update failed: {e}")
                        return False

                # Run all updates in parallel
                vector_success, neo4j_success, parse_success = await asyncio.gather(
                    update_vector_stores(),
                    update_neo4j(),
                    update_parse(),
                    return_exceptions=True
                )

                # Handle exceptions
                if isinstance(vector_success, Exception):
                    logger.error(f"Vector store update failed: {vector_success}")
                    vector_success = False
                if isinstance(neo4j_success, Exception):
                    logger.error(f"Neo4j update failed: {neo4j_success}")
                    neo4j_success = False
                if isinstance(parse_success, Exception):
                    logger.error(f"Parse update failed: {parse_success}")
                    parse_success = False

                status.pinecone = vector_success
                status.neo4j = neo4j_success
                status.parse = parse_success

                # If Parse update succeeded, return success response
                if parse_success:
                    # Get the updated memory from Parse to extract objectId and updatedAt
                    parse_memory = await retrieve_memory_item_by_qdrant_id(
                        session_token, 
                        vector_id,
                        api_key=api_key
                    )
                    if parse_memory:
                        # Extract and verify objectId
                        object_id = parse_memory.get('objectId')
                        logger.info(f"Retrieved objectId from parse_memory: {object_id}")

                        updated_item = UpdateMemoryItem(
                            objectId=object_id,
                            memoryId=memory_id,
                            content=parse_memory.get('content', ''),   
                            updatedAt=parse_memory.get('updatedAt'),
                            memoryChunkIds=memory_chunk_ids
                        )
                        
                        logger.info(f"Created UpdateMemoryItem with objectId: {updated_item.objectId} and memoryChunkIds: {memory_chunk_ids}")

                        return UpdateMemoryResponse.success(
                            memory_items=[updated_item],
                            status_obj=status
                        )

                # If Parse update failed
                return UpdateMemoryResponse.failure(
                    error="Failed to update in Parse Server",
                    code=500,
                    status_obj=status
                )
            else:
                # Update the content, metadata and context
                memory_properties['content'] = content

                # Generate new embeddings asynchronously
                #embeddings, chunks = await self.embedding_model.get_sentence_embedding(content)
                embeddings, chunks = await self.embedding_model.get_qwen_embedding_4b(content)
                num_embeddings = len(embeddings)

                if num_embeddings == 1:
                    # Ensure embedding is a list of floats
                    embedding = [float(x) for x in embeddings[0]]
                    
                    # Run all updates in parallel for better performance
                    async def _update_vector_store_internal():
                        """Update vector store"""
                        try:
                            return await self.update_vector_store(
                                chunk_id=vector_id,
                                new_metadata=memory_properties['metadata'],
                                embedding=embedding,
                                legacy_route=legacy_route
                            )
                        except Exception as e:
                            logger.error(f"Vector store update failed: {e}")
                            return False

                    async def update_neo4j():
                        """Update Neo4j"""
                        try:
                            result = await self.update_memory_item_in_neo4j(
                                memory_properties, 
                                neo_session,
                                memory_type
                            )
                            return result is not None
                        except Exception as e:
                            logger.error(f"Neo4j update failed: {e}")
                            return False

                    async def update_parse():
                        """Update Parse Server"""
                        try:
                            parse_memory = await retrieve_memory_item_by_qdrant_id(
                                session_token, 
                                vector_id,
                                api_key=api_key
                            )
                            logger.info(f"parse_memory from retrieve_memory_item_by_qdrant_id: {parse_memory}")

                            if parse_memory:
                                # Get the objectId directly from the response
                                memory_objectId = parse_memory['objectId'] 
                                logger.info(f"memory_objectId: {memory_objectId}")

                                # Convert ACL for Parse update
                                parse_acl = convert_acl(memory_properties['metadata'])

                                parse_update_data = {
                                    'objectId': memory_objectId,
                                    'memoryId': memory_id,
                                    'content': content,
                                    'metadata': memory_properties['metadata'],
                                    'memoryChunkIds': memory_chunk_ids,   
                                    'ACL': parse_acl  
                                }
                                parse_result = await update_memory_item(session_token, parse_update_data, None, api_key=api_key)
                                logger.info(f"parse_result: {parse_result}")
                                return True if parse_result and parse_result.memory_items and len(parse_result.memory_items) > 0 else False
                            return False
                        except Exception as e:
                            logger.error(f"Parse update failed: {e}")
                            return False

                    # Run all updates in parallel
                    vector_success, neo4j_success, parse_success = await asyncio.gather(
                        _update_vector_store_internal(),
                        update_neo4j(),
                        update_parse(),
                        return_exceptions=True
                    )

                    # Handle exceptions
                    if isinstance(vector_success, Exception):
                        logger.error(f"Vector store update failed: {vector_success}")
                        vector_success = False
                    if isinstance(neo4j_success, Exception):
                        logger.error(f"Neo4j update failed: {neo4j_success}")
                        neo4j_success = False
                    if isinstance(parse_success, Exception):
                        logger.error(f"Parse update failed: {parse_success}")
                        parse_success = False

                    status.pinecone = vector_success
                    status.neo4j = neo4j_success
                    status.parse = parse_success

                    # If Parse update succeeded, return success response
                    if parse_success:
                        # Get the updated memory from Parse to extract objectId and updatedAt
                        parse_memory = await retrieve_memory_item_by_qdrant_id(
                            session_token, 
                            vector_id,
                            api_key=api_key
                        )
                        if parse_memory:
                            # Extract and verify objectId
                            object_id = parse_memory.get('objectId')
                            logger.info(f"Retrieved objectId from parse_memory: {object_id}")

                            updated_item = UpdateMemoryItem(
                                objectId=object_id,
                                memoryId=memory_id,
                                content=content,
                                updatedAt=parse_memory.get('updatedAt'),
                                memoryChunkIds=memory_chunk_ids
                            )
                            
                            logger.info(f"Created UpdateMemoryItem with objectId: {updated_item.objectId} and memoryChunkIds: {memory_chunk_ids}")

                            return UpdateMemoryResponse.success(
                                memory_items=[updated_item],
                                status_obj=status
                            )

                    # If Parse update failed
                    return UpdateMemoryResponse.failure(
                        error="Failed to update in Parse Server",
                        code=500,
                        status_obj=status
                    )

                else:
                    # Handle multiple chunks in background task
                    background_tasks.add_task(
                        self.process_memory_chunks_async,
                        session_token=session_token,
                        memory_id=memory_id,
                        memory_properties=memory_properties,
                        memory_type=memory_type,
                        embeddings=embeddings,
                        chunks=chunks,
                        api_key=api_key
                    )
                    
                    return UpdateMemoryResponse.success(
                        memory_items=[],  # Empty list as processing is happening in background
                        status_obj=SystemUpdateStatus(
                            pinecone=True,
                            neo4j=True,
                            parse=True
                        )
                    )

        except Exception as e:
            logger.error(f"Error in update_memory_item: {e}")
            logger.error("Full traceback:", exc_info=True)
            return UpdateMemoryResponse.failure(
                error=str(e),
                code=500
            )

    async def process_memory_chunks_async(
        self,
        session_token: str,
        memory_id: str,
        memory_properties: dict,
        memory_type: str,
        embeddings: List[List[float]],
        chunks: List[str],
        api_key: Optional[str] = None
    ) -> UpdateMemoryResponse:
        """
        Process multiple memory chunks asynchronously in the background.
        Only creates new chunk IDs in Pinecone if necessary, and updates Neo4j and Parse
        with the consolidated memory item.
        """
        try:
            status = SystemUpdateStatus()
            new_chunk_ids = []

            # First, update the main memory item in Pinecone with first chunk
            pinecone_success = await self.update_qdrant(
                str(memory_id),
                embeddings[0],
                memory_properties['metadata']
            )
            status.pinecone = pinecone_success
            logger.info(f"Updated main memory in Qdrant: {memory_id}")

            # Process additional chunks in Pinecone if any
            if len(chunks) > 1:
                for index, (embedding, chunk) in enumerate(zip(embeddings[1:], chunks[1:]), 1):
                    # Generate new chunk ID
                    chunk_id = f"{memory_id}_chunk_{index}"
                    new_chunk_ids.append(chunk_id)
                    logger.info(f"new_chunk_ids: {new_chunk_ids}")
                    logger.info(f"chunk_id: {chunk_id}")                    
                    # Add chunk to Pinecone
                    chunk_success = await self.update_qdrant(
                        chunk_id,
                        embedding,
                        memory_properties['metadata']
                    )
                    status.pinecone = status.pinecone and chunk_success
                    logger.info(f"Added chunk to Qdrant: {chunk_id}")

            # Update memoryChunkIds in properties if we created new chunks
            if new_chunk_ids:
                existing_chunk_ids = memory_properties.get('memoryChunkIds', [])
                # Clean existing chunk IDs
                if isinstance(existing_chunk_ids, str):
                    try:
                        existing_chunk_ids = json.loads(existing_chunk_ids)
                    except json.JSONDecodeError:
                        existing_chunk_ids = [id.strip() for id in existing_chunk_ids.split(',') if id.strip()]
                elif isinstance(existing_chunk_ids, list):
                    existing_chunk_ids = [str(id).strip().strip("'[]\"") for id in existing_chunk_ids]
                else:
                    existing_chunk_ids = []
                memory_properties['memoryChunkIds'] = existing_chunk_ids + new_chunk_ids

            # Clean memory chunk IDs one final time before creating parse_update_data
            memory_chunk_ids = memory_properties.get('memoryChunkIds', [])
            if isinstance(memory_chunk_ids, str):
                try:
                    memory_chunk_ids = json.loads(memory_chunk_ids)
                except json.JSONDecodeError:
                    memory_chunk_ids = [id.strip() for id in memory_chunk_ids.split(',') if id.strip()]
            elif isinstance(memory_chunk_ids, list):
                memory_chunk_ids = [str(id).strip().strip("'[]\"") for id in memory_chunk_ids]
            else:
                memory_chunk_ids = []

            # Get parse_memory before trying to use it
            parse_memory = await retrieve_memory_item_by_qdrant_id(
                session_token,
                str(memory_id),
                api_key=api_key
            )
            if not parse_memory:
                return UpdateMemoryResponse.failure(
                    error="Failed to retrieve memory item from Parse Server",
                    code=404,
                    status_obj=status
                )

            parse_update_data = {
                'objectId': parse_memory.get('objectId'),
                'memoryId': memory_id,
                'content': content,
                'metadata': sanitized_metadata,
                'memoryChunkIds': memory_chunk_ids,  # Now using cleaned memory_chunk_ids
                'type': memory_type
            }

            logger.info(f"parse_update_data inside process_memory_chunks_async: {parse_update_data}")

            # Create the appropriate MemoryItem instance based on the type
            sanitized_metadata = {
                k: str(v) if not isinstance(v, (str, int, float, bool, list)) else v
                for k, v in memory_properties.get('metadata', {}).items()
            }
            memory_context = memory_properties.get('context', {})
            content = memory_properties.get('content', '')

            memory_item = None
            if memory_type == 'TextMemoryItem':
                memory_item = TextMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'CodeSnippetMemoryItem':
                memory_item = CodeSnippetMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type in ['document', 'DocumentMemoryItem']:
                memory_item = DocumentMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'WebpageMemoryItem':
                memory_item = WebpageMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'CodeFileMemoryItem':
                memory_item = CodeFileMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'MeetingMemoryItem':
                memory_item = MeetingMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'PluginMemoryItem':
                memory_item = PluginMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'IssueMemoryItem':
                memory_item = IssueMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'CustomerMemoryItem':
                memory_item = CustomerMemoryItem(content, sanitized_metadata, memory_context)

            if not memory_item:
                return UpdateMemoryResponse.failure(
                    error=f"Unsupported memory type: {memory_type}",
                    code=400,
                    status_obj=status
                )

            # Update in Neo4j with consolidated memory item
            await self.ensure_async_connection()
            async with self.async_neo_conn.get_session() as neo_session:
                neo4j_result = await self.update_memory_item_in_neo4j(
                    memory_item.__dict__,
                    neo_session,
                    memory_type
                )
            status.neo4j = neo4j_result is not None
            logger.info(f"Updated memory in Neo4j: {memory_id}")

            # Get parse_memory for Parse Server update
            parse_memory = await retrieve_memory_item_by_qdrant_id(
                session_token,
                str(memory_id),
                api_key=api_key
            )
            if parse_memory:
                # Create chunk IDs for each chunk
                chunk_ids = [f"{memory_id}_{i}" for i in range(len(chunks))]
                
                parse_update_data = {
                    'objectId': parse_memory.get('objectId'),
                    'memoryId': memory_id,
                    'content': content,
                    'metadata': sanitized_metadata,
                    'memoryChunkIds': chunk_ids,  # Add the chunk IDs explicitly
                    'type': memory_type
                }
                logger.info(f"parse_update_data inside process_memory_chunks_async: {parse_update_data}")
                parse_result = await update_memory_item(session_token, parse_update_data, None, api_key=api_key)
                status.parse = parse_result is not None
                
                if parse_result:
                    updated_item = UpdateMemoryItem(
                        objectId=parse_result.objectId,
                        memoryId=parse_result.memoryId,
                        content=parse_result.content,
                        updatedAt=parse_result.updatedAt,
                        memoryChunkIds=chunk_ids
                    )
                    return UpdateMemoryResponse.success(
                        memory_items=[updated_item],
                        status_obj=status,
                        code=200,
                        message="Memory chunks processed successfully"
                    )
            
            return UpdateMemoryResponse.failure(
                error="Failed to update memory item in Parse Server",
                code=500,
                status_obj=status
            )

        except Exception as e:
            logger.error(f"Error processing memory chunks: {e}")
            logger.error("Full traceback:", exc_info=True)
            return UpdateMemoryResponse.failure(
                error=str(e),
                code=500,
                status_obj=SystemUpdateStatus()
            )
            
    def find_and_delete_duplicates(self, user_id, session_token, neo_session: AsyncSession, api_key: Optional[str] = None):
        # Modified method to find and delete duplicates, then return counts or relevant info
        duplicates = self.identify_duplicates(user_id, api_key=api_key)
        num_duplicates_found = len(duplicates)
        self.delete_duplicate_memories(duplicates, session_token, neo_session, api_key=api_key)
        num_duplicates_deleted = len(duplicates)  # Assuming all found duplicates are deleted successfully
        return num_duplicates_found, num_duplicates_deleted

    def identify_duplicates(self, user_id):
        # Get user info for ACL filter
        user_instance = User.get(user_id)
        user_roles = user_instance.get_roles() if user_instance else []
        user_workspace_ids = User.get_workspaces_for_user(user_id)
        
        # Setup the ACL filter using the working structure
        acl_filter = {
            "$or": [
                {"user_id": {"$eq": str(user_id)}},
                {"user_read_access": {"$in": [str(user_id)]}},
                {"workspace_read_access": {"$in": [str(workspace_id) for workspace_id in user_workspace_ids]}},
                {"role_read_access": {"$in": user_roles}},
            ]
        }
        
        # Fetch all memories' embeddings and IDs for the specified user using ACL filter
        response = self.index.query(
            vector=[0] * 768,  # Dummy vector to get all memories
            filter=acl_filter,
            top_k=1000,  # Adjust based on your needs
            include_values=True
        )
        
        duplicates = []
        checked_ids = set()  # Keep track of memory IDs that have been checked

        # Iterate through each memory to find its nearest neighbors (excluding itself)
        for match in response.matches:
            memory_id = match.id
            embedding = match.values

            if memory_id in checked_ids:
                continue  # Skip if already checked

            # Find similar memories using Pinecone's similarity search with same ACL filter
            similar_memories = self.index.query(
                vector=embedding,
                filter=acl_filter,
                include_metadata=True,
                top_k=20
            )
            
            # Filter out duplicates based on cosine similarity threshold (e.g., > 0.95)
            for sim_memory in similar_memories.matches:
                if sim_memory.score > 0.95 and sim_memory.id != memory_id:
                    duplicates.append(sim_memory.id)

            checked_ids.add(memory_id)

        return list(set(duplicates))  # Return a list of unique duplicate IDs
        return list(set(duplicates))  # Return a list of unique duplicate IDs

    def delete_duplicate_memories(self, duplicate_ids, session_token, neo_session: AsyncSession, api_key: Optional[str] = None, legacy_route: bool = True):
        for memory_id in duplicate_ids:
            # Use Pinecone's delete method to remove the memory
            self.delete_memory_item(memory_id, session_token, neo_session, api_key=api_key, legacy_route=True)
            logger.info(f"Deleted duplicate memory with ID from both pine-cone and neo {memory_id}")

    
   
    async def _node_exists(
        self, 
        node_id: str, 
        neo_session: AsyncSession, 
        node_type: Optional[NodeLabel] = None, 
        node_content: Optional[str] = None,
        workspace_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Union[str, bool]:
        """
        Check if a node exists in Neo4j by ID or content, with tenant scoping and ACL checks.
        
        CRITICAL: For Memory nodes, tenant scoping (workspace_id, organization_id, namespace_id) 
        is REQUIRED to prevent cross-tenant data leakage.
        """
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, skipping _node_exists")
            return False


        try:
            if node_type and node_content:
                # Simplified queries that check by content/name first
                if node_type == NodeLabel.Memory:
                    # CRITICAL: Memory nodes MUST have tenant scoping to prevent cross-tenant matching
                    if not workspace_id and not organization_id and not namespace_id:
                        logger.error(f"🚨 SECURITY: _node_exists called for Memory node without tenant scoping! This will match across ALL tenants.")
                    
                    # Build MUST conditions (content + tenant scoping with AND)
                    must_conditions = ["n.content = $node_content"]
                    
                    # Add tenant scoping - all provided tenant IDs MUST match (AND logic)
                    tenant_conditions = []
                    if workspace_id:
                        tenant_conditions.append("n.workspace_id = $workspace_id")
                    if organization_id:
                        tenant_conditions.append("n.organization_id = $organization_id")
                    if namespace_id:
                        tenant_conditions.append("n.namespace_id = $namespace_id")
                    
                    if tenant_conditions:
                        must_conditions.extend(tenant_conditions)
                        logger.info(f"🔒 Memory node exists check with tenant scoping (MUST): {tenant_conditions}")
                    
                    # Build ACL write access conditions (SHOULD match - OR logic)
                    # User has write access if they are the owner OR in any write access list
                    acl_conditions = []
                    if user_id:
                        acl_conditions.append("n.user_id = $user_id")
                        acl_conditions.append("$user_id IN n.user_write_access")
                    if workspace_id:
                        acl_conditions.append("$workspace_id IN n.workspace_write_access")
                    if organization_id:
                        acl_conditions.append("$organization_id IN n.organization_write_access")
                    if namespace_id:
                        acl_conditions.append("$namespace_id IN n.namespace_write_access")
                    
                    # Combine: MUST conditions AND (any ACL condition)
                    if acl_conditions:
                        acl_clause = f"({' OR '.join(acl_conditions)})"
                        where_clause = f"{' AND '.join(must_conditions)} AND {acl_clause}"
                        logger.info(f"🔒 Memory node exists check with ACL (SHOULD match any): {len(acl_conditions)} conditions")
                    else:
                        # No ACL conditions provided - just use tenant scoping
                        where_clause = " AND ".join(must_conditions)
                        logger.warning(f"⚠️ Memory node exists check without ACL conditions - only tenant scoping applied")
                    
                    query = f"""
                    MATCH (n:Memory) 
                    WHERE {where_clause}
                    RETURN n.id as existing_id, COUNT(n) as count
                    """
                elif node_type in [NodeLabel.Person, NodeLabel.Company, NodeLabel.Project]:
                    query = f"""
                    MATCH (n:{node_type.value}) 
                    WHERE n.name = $node_content
                    RETURN n.id as existing_id, COUNT(n) as count
                    """
                else:
                    # For custom types, try content-based deduplication first if content is available
                    if node_content:
                        # Check by content or name for custom node types
                        query = f"""
                        MATCH (n:{node_type.value}) 
                        WHERE (n.content = $node_content OR n.name = $node_content)
                        RETURN n.id as existing_id, COUNT(n) as count
                        """
                        logger.debug(f"Using content-based deduplication for custom node type {node_type.value}")
                    else:
                        # Fallback to ID check if no content available
                        query = "MATCH (n) WHERE n.id = $node_id RETURN n.id as existing_id, COUNT(n) as count"
                        logger.debug(f"Using ID-based check for custom node type {node_type.value} (no content available)")
            else:
                # If no type/content provided, just check by ID
                query = "MATCH (n) WHERE n.id = $node_id RETURN n.id as existing_id, COUNT(n) as count"

            # Build params dict with all available parameters
            params = {"node_id": node_id}
            if node_content:
                params["node_content"] = node_content
            if workspace_id:
                params["workspace_id"] = workspace_id
            if organization_id:
                params["organization_id"] = organization_id
            if namespace_id:
                params["namespace_id"] = namespace_id
            if user_id:
                params["user_id"] = user_id
            logger.debug(f"Node exists query: {query}")
            logger.debug(f"Node exists params: {params}")
            
            # If no session provided, open a short-lived session and consume within context
            if neo_session is None:
                from contextlib import asynccontextmanager
                try:
                    async with self.async_neo_conn.get_session() as session:
                        result = await self._safe_neo4j_run(session, query, params, "_node_exists")
                        if result is None:
                            return False
                        records = await result.values()
                except Exception as e:
                    logger.error(f"Error running _node_exists with temporary session: {e}")
                    return False
            else:
                result = await self._safe_neo4j_run(neo_session, query, params, "_node_exists")
                if result is None:
                    return False
                try:
                    records = await result.values()
                except Exception as e:
                    logger.error(f"Error processing result in _node_exists: {e}")
                    return False

            if records is None:
                logger.error("Neo4j result.values() returned None in _node_exists")
                return False

            for record in records:
                if record and len(record) >= 2 and record[1] > 0:
                    logger.info(f"✅ Found existing {node_type.value if node_type else 'node'} with content '{node_content}', id: {record[0]}")
                    return record[0]  # Return existing_id
            return False
        except Exception as e:
            logger.error(f"Unexpected error in _node_exists: {e}")
            return False

    async def _merge_node_with_unique_identifiers(
        self,
        node: LLMGraphNode,
        common_metadata: Dict[str, Any],
        neo_session: AsyncSession,
        workspace_id: Optional[str] = None,
        user_schema: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Schema-aware node merging with Qdrant synchronization (Steps 1-3).
        
        Step 1: Get unique identifiers from schema
        Step 2: Search Qdrant for existing entity by unique IDs  
        Step 3: Neo4j MERGE with canonical values + selective property updates
        """
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, skipping _merge_node_with_unique_identifiers")
            return None

        try:
            # Get node properties
            if hasattr(node.properties, 'model_dump'):
                props = node.properties.model_dump(exclude_none=True)
            else:
                props = node.properties if isinstance(node.properties, dict) else dict(node.properties)

            # Step 1: Get unique identifiers for this node type from schema (before filtering)
            unique_identifiers = self._get_unique_identifiers_for_node_type(node.label, user_schema)
            
            # ANTI-HALLUCINATION: Filter out null properties from LLM output
            # BUT: Preserve unique identifiers even if None - they're needed for MERGE operations
            # Note: Null unique identifiers are acceptable - property_overrides can fill them later
            # If they remain null, MERGE will skip and we'll fall back to content-based creation (which is OK)
            preserved_unique_ids = {}
            if unique_identifiers:
                for uid in unique_identifiers:
                    if uid in props:
                        preserved_unique_ids[uid] = props[uid]  # Preserve even if None
            
            # Filter null properties (this removes properties with None values)
            props = {k: v for k, v in props.items() if v is not None}
            
            # Restore unique identifiers if they were filtered out (even if None - needed for MERGE)
            # This allows property_overrides to fill them later, or MERGE will skip if still null (acceptable behavior)
            for uid in unique_identifiers:
                if uid in preserved_unique_ids:
                    props[uid] = preserved_unique_ids[uid]  # Can be None - that's OK

            # Add common metadata
            props.update({k: v for k, v in common_metadata.items() if v is not None})
            
            if unique_identifiers:
                logger.info(f"🔄 SYNC STEP 1: Using schema-aware MERGE for {node.label} with unique_identifiers: {unique_identifiers}")
                
                # Step 2: Search Qdrant for existing entity by unique identifier similarity
                existing_entity = await self._search_qdrant_for_existing_entity(node.label, props, unique_identifiers, user_schema)
                
                # Step 3: Neo4j MERGE with canonical values or new entity
                return await self._merge_node_with_sync_results(
                    node.label, props, unique_identifiers, existing_entity, neo_session
                )
            else:
                # Fall back to content-based approach for nodes without unique_identifiers
                logger.info(f"No unique_identifiers found for {node.label}, falling back to content-based approach")
                return await self._create_node_with_content_check(node, common_metadata, neo_session)

        except Exception as e:
            logger.error(f"Error in _merge_node_with_unique_identifiers: {e}")
            # Fall back to content-based approach on error
            return await self._create_node_with_content_check(node, common_metadata, neo_session)

    async def _get_schemas_for_manual_graph(
        self,
        nodes: List[Any],
        user_id: str,
        workspace_id: str,
        metadata: Dict[str, Any]
    ) -> Optional[Any]:
        """
        For manual graph override, try to find registered schemas that match the node labels.
        This allows manual graphs to benefit from schema-defined unique_identifiers.
        
        Returns a combined schema object that includes node_types from any matching registered schemas.
        """
        try:
            # Extract multi-tenant context from metadata
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata)
            
            organization_id = metadata.get('organization_id')
            namespace_id = metadata.get('namespace_id')
            
            # Get all registered schemas for this user/workspace
            from services.schema_service import SchemaService
            schema_service = SchemaService()
            
            user_schemas = await schema_service.get_active_schemas(
                user_id=user_id,
                workspace_id=workspace_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if not user_schemas:
                logger.info("📋 MANUAL GRAPH: No registered schemas found for this user/workspace")
                return None
            
            logger.info(f"📋 MANUAL GRAPH: Found {len(user_schemas)} registered schemas")
            
            # Extract unique node labels from manual graph
            manual_node_labels = set(node.label for node in nodes)
            logger.info(f"📋 MANUAL GRAPH: Node labels in manual graph: {manual_node_labels}")
            
            # Build a combined schema with matching node types
            combined_node_types = {}
            
            for schema in user_schemas:
                if hasattr(schema, 'node_types'):
                    for node_name, node_type in schema.node_types.items():
                        # If this schema defines a node type that's in the manual graph
                        if node_name in manual_node_labels:
                            combined_node_types[node_name] = node_type
                            unique_ids = getattr(node_type, 'unique_identifiers', [])
                            logger.info(f"📋 MANUAL GRAPH: Found schema for {node_name} with unique_identifiers: {unique_ids}")
            
            if not combined_node_types:
                logger.info("📋 MANUAL GRAPH: No matching schemas found for manual node labels")
                return None
            
            # Create a simple object to hold the combined schema
            class CombinedSchema:
                def __init__(self, node_types):
                    self.node_types = node_types
            
            combined_schema = CombinedSchema(combined_node_types)
            logger.info(f"📋 MANUAL GRAPH: Created combined schema with {len(combined_node_types)} node types")
            return combined_schema
            
        except Exception as e:
            logger.warning(f"📋 MANUAL GRAPH: Error fetching schemas for manual graph: {e}")
            logger.debug(f"Schema fetch error details", exc_info=True)
            return None

    def _get_unique_identifiers_for_node_type(self, node_label: str, user_schema: Optional[Any]) -> List[str]:
        """
        Get unique_identifiers for a specific node type from the user schema.
        """
        if not user_schema or not hasattr(user_schema, 'node_types'):
            return []

        try:
            # Look for the node type in the schema
            for node_name, node_type in user_schema.node_types.items():
                if node_name == node_label:
                    unique_ids = getattr(node_type, 'unique_identifiers', [])
                    logger.info(f"Found unique_identifiers for {node_label}: {unique_ids}")
                    return unique_ids
            
            logger.info(f"No schema definition found for node type: {node_label}")
            return []
        except Exception as e:
            logger.error(f"Error getting unique_identifiers for {node_label}: {e}")
            return []

    def _should_search_property_in_qdrant(
        self, 
        node_label: str, 
        prop_name: str, 
        prop_value: Any, 
        user_schema: Optional[Any] = None
    ) -> bool:
        """
        Determine if a property should be searched in Qdrant by reusing existing PropertyIndexingService logic.
        This ensures consistency with the property indexing system.
        """
        try:
            # Create a temporary PropertyIndexingService to reuse existing logic
            from services.property_indexing_service import PropertyIndexingService
            property_service = PropertyIndexingService(self)
            
            # Build indexable properties using existing schema extraction logic
            if user_schema:
                indexable_properties = {}
                # Get the structured output schema that was used for LLM generation (same as property indexing)
                structured_schema = getattr(self, '_last_memory_graph_schema', None)
                if structured_schema:
                    indexable_properties = property_service._extract_indexable_properties_from_schema(structured_schema)
                
                # Check if this property is considered indexable by the existing logic
                prop_key = f"{node_label}.{prop_name}"
                if prop_key in indexable_properties:
                    # Use existing validation logic
                    should_index = property_service._should_index_property_value(
                        node_label, prop_name, prop_value, indexable_properties
                    )
                    if should_index:
                        logger.info(f"🔄 SYNC STEP 2: Including property '{prop_name}' in Qdrant search (schema-based)")
                        return True
                    else:
                        logger.info(f"🔄 SYNC STEP 2: Skipping property '{prop_name}' from Qdrant search (failed validation)")
                        return False
                else:
                    logger.info(f"🔄 SYNC STEP 2: Skipping property '{prop_name}' from Qdrant search (not indexable)")
                    return False
            
            # Fallback: if no schema, use basic validation
            if not isinstance(prop_value, str) or len(str(prop_value).strip()) == 0:
                return False
            
            # Skip obvious ID fields
            prop_name_lower = prop_name.lower()
            if any(id_pattern in prop_name_lower for id_pattern in ['id', 'uuid', 'objectid']):
                logger.info(f"🔄 SYNC STEP 2: Skipping ID-like field '{prop_name}' from Qdrant search (heuristic)")
                return False
            
            logger.info(f"🔄 SYNC STEP 2: Including string property '{prop_name}' in Qdrant search (heuristic)")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking property indexability for {node_label}.{prop_name}: {e}")
            # Safe fallback: only include obvious string properties, skip IDs
            if isinstance(prop_value, str) and len(str(prop_value).strip()) > 0:
                prop_name_lower = prop_name.lower()
                if not any(id_pattern in prop_name_lower for id_pattern in ['id', 'uuid', 'objectid']):
                    return True
            return False

    async def _generate_batch_embeddings_for_deduplication(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple unique identifier texts using HuggingFace API batch processing.
        Used during entity deduplication (Step 2).
        
        Args:
            texts: List of formatted unique identifier strings (e.g., "Node: Company, Property: name: Papr")
            
        Returns:
            List of embeddings (or None for failed texts)
        """
        try:
            import httpx
            from os import environ as env
            
            api_url = env.get("HUGGING_FACE_API_URL_SENTENCE_BERT")
            access_token = env.get("HUGGING_FACE_ACCESS_TOKEN")
            
            if not api_url or not access_token:
                logger.error("HuggingFace API URL or token not configured for deduplication")
                return [None] * len(texts)
            
            headers = {"Authorization": f"Bearer {access_token}"}
            payload = {"inputs": texts}  # Send array of texts for batch processing
            
            async with httpx.AsyncClient(timeout=10.0) as client:  # Fast timeout for deduplication
                response = await client.post(api_url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    embeddings = response.json()
                    logger.info(f"🔄 SYNC STEP 2: ✅ Generated {len(embeddings)} embeddings in batch")
                    return embeddings
                else:
                    logger.error(f"🔄 SYNC STEP 2: Batch embedding API error: {response.status_code} - {response.text}")
                    return [None] * len(texts)
                    
        except Exception as e:
            logger.error(f"🔄 SYNC STEP 2: Batch embedding error: {e}")
            return [None] * len(texts)
    
    async def _search_qdrant_for_existing_entity(
        self, 
        node_label: str, 
        props: Dict[str, Any], 
        unique_identifiers: List[str],
        user_schema: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Step 2: Search Qdrant for existing entity by checking ALL unique identifier similarity.
        Returns canonical entity info only if ALL string unique identifiers have matches in Qdrant.
        """
        try:
            # Check if property indexing service is available
            if not hasattr(self, 'qdrant_client') or not self.qdrant_client:
                logger.warning("🔄 SYNC STEP 2: Qdrant client not available, treating as new entity")
                return None
            
            # Ensure property collection is initialized (same pattern as content-based approach)
            from os import environ as env
            property_collection_name = env.get("QDRANT_PROPERTY_COLLECTION", "neo4j_properties")
            
            # Set property collection if not already set
            if not hasattr(self, 'qdrant_property_collection') or not self.qdrant_property_collection:
                self.qdrant_property_collection = property_collection_name
                logger.info(f"🔄 SYNC STEP 2: Set property collection to: {property_collection_name}")
            
            # Ensure collection exists (same pattern as content-based approach)
            try:
                await self.qdrant_client.get_collection(self.qdrant_property_collection)
                logger.info(f"🔄 SYNC STEP 2: Property collection '{self.qdrant_property_collection}' exists")
            except Exception as e:
                logger.info(f"🔄 SYNC STEP 2: Property collection '{self.qdrant_property_collection}' doesn't exist, creating it")
                try:
                    # Use the same collection creation method as property indexing service
                    await self.create_optimized_qdrant_collection(
                        collection_name=self.qdrant_property_collection,
                        vector_size=384  # Sentence-BERT dimensions
                    )
                    logger.info(f"🔄 SYNC STEP 2: Successfully created property collection '{self.qdrant_property_collection}'")
                except Exception as create_error:
                    logger.error(f"🔄 SYNC STEP 2: Failed to create property collection: {create_error}")
                    return None
            
            # Extract unique identifier values from node properties (only semantic-searchable strings for Qdrant)
            string_unique_values = {}
            all_unique_values = {}
            
            for uid_name in unique_identifiers:
                if uid_name in props and props[uid_name] is not None:
                    value = props[uid_name]
                    all_unique_values[uid_name] = value
                    
                    # Check if this property should be searched in Qdrant based on schema type
                    should_search_in_qdrant = self._should_search_property_in_qdrant(
                        node_label, uid_name, value, user_schema
                    )
                    
                    if should_search_in_qdrant:
                        string_unique_values[uid_name] = str(value)
            
            if not string_unique_values:
                logger.info(f"🔄 SYNC STEP 2: No semantic-searchable unique identifiers found for {node_label}")
                logger.info(f"🔄 SYNC STEP 2: All unique identifiers are exact-match only (IDs/UUIDs), skipping Qdrant search")
                logger.info(f"🔄 SYNC STEP 2: Will use exact Neo4j MERGE with unique identifier values: {list(all_unique_values.keys())}")
                return None
            
            logger.info(f"🔄 SYNC STEP 2: Checking ALL string unique IDs for {node_label}: {list(string_unique_values.keys())}")
            
            # BATCH OPTIMIZATION: Generate all embeddings in a single API call
            uid_contents = []
            uid_mapping = []  # Keep track of which content maps to which uid
            for uid_name, uid_value in string_unique_values.items():
                uid_content = f"Node: {node_label}, Property: {uid_name}: {uid_value}"
                uid_contents.append(uid_content)
                uid_mapping.append((uid_name, uid_value))
            
            logger.info(f"🔄 SYNC STEP 2: 🚀 Generating {len(uid_contents)} embeddings in batch")
            
            # Generate embeddings using batch HuggingFace API
            uid_embeddings = await self._generate_batch_embeddings_for_deduplication(uid_contents)
            
            if not uid_embeddings or all(emb is None for emb in uid_embeddings):
                logger.warning(f"🔄 SYNC STEP 2: Failed to generate any embeddings, falling back to Neo4j MERGE with LLM values")
                return None
            
            # Search for each string unique identifier with semantic similarity
            canonical_values = {}
            canonical_node_id = None
            
            # ALL string unique identifiers must have matches for entity to be considered existing
            for (uid_name, uid_value), uid_embedding in zip(uid_mapping, uid_embeddings):
                if uid_embedding is None:
                    logger.warning(f"🔄 SYNC STEP 2: Failed to generate embedding for {uid_name}, skipping")
                    return None  # If any embedding fails, fall back to creating new entity
                
                logger.info(f"🔄 SYNC STEP 2: Searching for {uid_name} = '{uid_value}'")
                
                from qdrant_client.http import models
                
                # Build ACL conditions (same pattern as content-based approach)
                user_id = props.get('user_id')
                workspace_id = props.get('workspace_id')
                organization_id = props.get('organization_id')
                namespace_id = props.get('namespace_id')
                role_read_access = props.get('role_read_access', [])
                
                # CRITICAL: namespace_id and user_id are REQUIRED for multi-tenant isolation
                if not namespace_id:
                    raise ValueError(f"namespace_id is required for Qdrant search but was None. Node: {node_label}, uid_name: {uid_name}")
                if not user_id:
                    raise ValueError(f"user_id is required for Qdrant search but was None. Node: {node_label}, uid_name: {uid_name}")
                
                # Build must conditions: property_key AND namespace_id AND organization_id AND workspace_id (tenant scoping)
                # NOTE: user_id is NOT in MUST - it's in SHOULD for write access check (consistent with Neo4j)
                must_conditions = [
                    models.FieldCondition(key="property_key", match=models.MatchValue(value=f"{node_label}.{uid_name}")),
                    models.FieldCondition(key="namespace_id", match=models.MatchValue(value=namespace_id))
                ]
                
                # Add organization_id and workspace_id to MUST conditions if available
                if organization_id:
                    must_conditions.append(models.FieldCondition(key="organization_id", match=models.MatchValue(value=organization_id)))
                if workspace_id:
                    must_conditions.append(models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id)))
                
                # Build should conditions (OR): user has WRITE access if ANY of these match
                # This is for deduplication during node creation (write operation), so check *_write_access
                should_conditions = []
                if user_id:
                    should_conditions.append(
                        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
                    )
                    should_conditions.append(
                        models.FieldCondition(key="user_write_access", match=models.MatchAny(any=[user_id]))
                    )
                else:
                    should_conditions.append(
                        models.FieldCondition(key="user_write_access", match=models.MatchAny(any=[]))
                    )
                
                # Add write access ACL filtering (consistent with Neo4j Node MERGE)
                if workspace_id:
                    should_conditions.append(models.FieldCondition(key="workspace_write_access", match=models.MatchAny(any=[workspace_id])))
                if organization_id:
                    should_conditions.append(models.FieldCondition(key="organization_write_access", match=models.MatchAny(any=[organization_id])))
                if namespace_id:
                    should_conditions.append(models.FieldCondition(key="namespace_write_access", match=models.MatchAny(any=[namespace_id])))
                if role_read_access:
                    # Note: role uses read_access for property lookup (roles are for reading, not writing)
                    should_conditions.append(models.FieldCondition(key="role_read_access", match=models.MatchAny(any=role_read_access)))
                
                # Log detailed ACL filter information
                logger.info(f"🔄 SYNC STEP 2: Searching for unique identifier in collection '{self.qdrant_property_collection}'")
                must_fields = f"property_key='{node_label}.{uid_name}' AND namespace_id='{namespace_id}'"
                if organization_id:
                    must_fields += f" AND organization_id='{organization_id}'"
                if workspace_id:
                    must_fields += f" AND workspace_id='{workspace_id}'"
                logger.info(f"🔄 ACL FILTER - MUST (AND - Tenant Scoping): {must_fields}")
                logger.info(f"🔄 ACL FILTER - SHOULD (OR - Write Access): {len(should_conditions)} conditions - user_id='{user_id}', user_write_access=[{user_id}], workspace_write_access=[{workspace_id}], organization_write_access=[{organization_id}], namespace_write_access=[{namespace_id}]")
                
                # ACL filter: must match (property_key AND user_id) AND (user has access via ANY of the should conditions)
                # Handle embedding as list (HuggingFace API) or numpy array (local models)
                uid_embedding_list = uid_embedding.tolist() if hasattr(uid_embedding, 'tolist') else uid_embedding
                
                search_results = await self._qdrant_search_async(
                    collection_name=self.qdrant_property_collection,
                    query_vector=uid_embedding_list,
                    query_filter=models.Filter(
                        must=must_conditions,
                        should=should_conditions  # OR: user has access if ANY ACL condition matches
                    ),
                    limit=1,  # Get top match only
                    score_threshold=0.95,  # High threshold for unique ID matching
                    with_payload=True,
                    with_vectors=False
                )
                
                if search_results and len(search_results) > 0:
                    match = search_results[0]
                    match_node_id = match.payload.get('source_node_id') or match.payload.get('canonical_node_id')
                    canonical_value = match.payload.get('property_value')
                    
                    logger.info(f"🔄 SYNC STEP 2: ✅ Found match for {uid_name}: '{uid_value}' → '{canonical_value}' (score: {match.score:.3f})")
                    
                    # Ensure all matches point to the same canonical entity
                    if canonical_node_id is None:
                        canonical_node_id = match_node_id
                    elif canonical_node_id != match_node_id:
                        logger.warning(f"🔄 SYNC STEP 2: ❌ Unique identifier mismatch! {uid_name} points to different entity ({match_node_id} vs {canonical_node_id})")
                        return None
                    
                    canonical_values[uid_name] = canonical_value
                else:
                    logger.info(f"🔄 SYNC STEP 2: ❌ No match found for {uid_name} = '{uid_value}'")
                    return None  # ALL string unique identifiers must match
            
            # All string unique identifiers matched the same entity
            logger.info(f"🔄 SYNC STEP 2: ✅ ALL string unique IDs matched entity {canonical_node_id}")
            
            # Combine canonical string values with non-string unique identifiers
            final_canonical_values = canonical_values.copy()
            for uid_name, uid_value in all_unique_values.items():
                if uid_name not in canonical_values:  # Non-string unique identifiers
                    final_canonical_values[uid_name] = uid_value
            
            return {
                'canonical_node_id': canonical_node_id,
                'canonical_unique_values': final_canonical_values,
                'string_matches': len(canonical_values),
                'total_unique_ids': len(all_unique_values)
            }
            
        except Exception as e:
            logger.error(f"🔄 SYNC STEP 2: Error searching for existing entity: {e}")
            return None

    async def _merge_node_with_sync_results(
        self,
        node_label: str,
        props: Dict[str, Any],
        unique_identifiers: List[str],
        existing_entity: Optional[Dict[str, Any]],
        neo_session: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """
        Step 3: Neo4j MERGE using existing robust logic with canonical values.
        Returns node data with was_created flag for property indexing.
        """
        try:
            if existing_entity:
                # Use canonical unique identifier values + non-string unique IDs + ACL for MERGE
                canonical_values = existing_entity['canonical_unique_values']
                canonical_node_id = existing_entity['canonical_node_id']
                
                logger.info(f"🔄 SYNC STEP 3: Merging with existing entity {canonical_node_id}")
                logger.info(f"🔄 SYNC STEP 3: Using canonical values: {canonical_values}")
                
                # Create props with canonical values for unique identifiers
                merge_props = props.copy()
                for uid_name, canonical_value in canonical_values.items():
                    merge_props[uid_name] = canonical_value
                    logger.info(f"🔄 SYNC STEP 3: Using canonical {uid_name}: '{canonical_value}'")
                
                # Use existing robust MERGE logic with proper ACL handling
                result = await self._merge_node_by_unique_props(
                    node_label=node_label,
                    props=merge_props,
                    unique_identifiers=unique_identifiers,
                    neo_session=neo_session
                )
                
                if result:
                    # Use the was_created flag from Neo4j MERGE result
                    was_created_in_neo4j = result.get('was_created', False)
                    result['sync_operation'] = 'create' if was_created_in_neo4j else 'update'
                    
                    logger.info(f"🔄 SYNC STEP 3: ✅ {'Created' if was_created_in_neo4j else 'Updated'} {node_label} node: {result.get('id')} (Neo4j was_created={was_created_in_neo4j})")
                    return result
                    
            else:
                # New entity - use original LLM values
                logger.info(f"🔄 SYNC STEP 3: Creating new entity with LLM values")
                
                # Use existing robust MERGE logic with proper ACL handling
                result = await self._merge_node_by_unique_props(
                    node_label=node_label,
                    props=props,
                    unique_identifiers=unique_identifiers,
                    neo_session=neo_session
                )
                
                if result:
                    # Use the was_created flag from Neo4j MERGE result
                    was_created_in_neo4j = result.get('was_created', False)
                    result['sync_operation'] = 'create' if was_created_in_neo4j else 'update'
                    
                    logger.info(f"🔄 SYNC STEP 3: ✅ {'Created' if was_created_in_neo4j else 'Updated'} {node_label} node: {result.get('id')} (Neo4j was_created={was_created_in_neo4j})")
                    return result
            
            logger.error(f"🔄 SYNC STEP 3: MERGE operation failed")
            return None
            
        except Exception as e:
            logger.error(f"🔄 SYNC STEP 3: Error in Neo4j MERGE operation: {e}")
            return None


    async def _merge_node_by_unique_props(
        self,
        node_label: str,
        props: Dict[str, Any],
        unique_identifiers: List[str],
        neo_session: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """
        Execute MERGE query using unique identifier properties.

        IMPORTANT: Always includes workspace_id and user_id in MERGE to ensure multi-tenant isolation.
        This prevents data leakage across users and workspaces.
        """
        try:
            # Extract unique properties for MERGE clause
            # Note: Null unique identifiers are acceptable - property_overrides can fill them later
            # If they remain null after property_overrides, we skip MERGE and fall back (which is OK per Amir's guidance)
            # We don't force values - null is acceptable and nodes will be skipped if unique identifiers remain null
            unique_props = {}
            for prop_name in unique_identifiers:
                if prop_name in props and props[prop_name] is not None:
                    unique_props[prop_name] = props[prop_name]
                else:
                    # Unique identifier is None - this is acceptable, property_overrides may fill it
                    # If it remains None, we'll skip MERGE (acceptable behavior - we don't force values)
                    logger.debug(f"Unique identifier '{prop_name}' is null for {node_label} - property_overrides may fill it, or node will be skipped if it remains null")

            if not unique_props:
                # All unique identifiers are null - skip MERGE and fall back to content-based approach
                # This is acceptable behavior - we don't force values, null is OK (nodes just won't be created if unique IDs are null)
                logger.debug(f"No non-null unique identifier values found for {node_label}, falling back to content-based approach (null is acceptable, node may be skipped)")
                return None

            # CRITICAL: Add workspace_id, organization_id, namespace_id for multi-tenant isolation
            # This ensures nodes are NEVER merged across different tenants
            if 'workspace_id' in props and props['workspace_id'] is not None:
                unique_props['workspace_id'] = props['workspace_id']
            else:
                logger.error(f"workspace_id missing in props for {node_label} - multi-tenant isolation may be compromised!")
            
            if 'organization_id' in props and props['organization_id'] is not None:
                unique_props['organization_id'] = props['organization_id']
            else:
                logger.error(f"organization_id missing in props for {node_label} - multi-tenant isolation may be compromised!")

            if 'namespace_id' in props and props['namespace_id'] is not None:
                unique_props['namespace_id'] = props['namespace_id']
            else:
                logger.error(f"namespace_id missing in props for {node_label} - multi-tenant isolation may be compromised!")

            # ACL Model (TWO-STEP PROCESS):
            # Step 1: Search for existing node with write access check (MATCH + WHERE)
            # Step 2a: If found → Update with MERGE (no WHERE, just tenant scoping)
            # Step 2b: If not found → Create new node with CREATE
            # This avoids Cypher syntax issues with WHERE in complex queries
            
            logger.info(f"🔒 Node operation with tenant scoping: {node_label} using {list(unique_props.keys())}")
            logger.info(f"🔒 Write access check: user_id OR *_write_access arrays")
            
            # Build MATCH conditions for unique properties + tenant scoping
            unique_conditions = ', '.join([f"{k}: ${k}" for k in unique_props.keys()])
            
            # Extract ACL values for WHERE clause
            user_id = props.get('user_id')
            workspace_id = props.get('workspace_id')
            organization_id = props.get('organization_id')
            namespace_id = props.get('namespace_id')
            
            # Build WHERE clause for write access check
            where_conditions = []
            if user_id:
                where_conditions.append("n.user_id = $user_id")
                where_conditions.append("$user_id IN n.user_write_access")
            if workspace_id:
                where_conditions.append("$workspace_id IN n.workspace_write_access")
            if organization_id:
                where_conditions.append("$organization_id IN n.organization_write_access")
            if namespace_id:
                where_conditions.append("$namespace_id IN n.namespace_write_access")
            
            where_clause = f"WHERE {' OR '.join(where_conditions)}" if where_conditions else ""
            
            # Separate properties for CREATE vs UPDATE
            all_properties = props.copy()
            update_properties = {k: v for k, v in props.items() if k not in unique_props}
            update_properties['updatedAt'] = datetime.now(timezone.utc).isoformat()
            
            params = {
                **unique_props,
                'all_properties': all_properties,
                'update_properties': update_properties,
                'user_id': user_id,
                'workspace_id': workspace_id,
                'organization_id': organization_id,
                'namespace_id': namespace_id
            }
            
            # STEP 1: Search for existing node with write access
            search_query = f"""
                OPTIONAL MATCH (n:{node_label} {{{unique_conditions}}})
                {where_clause}
                RETURN n
            """
            
            logger.info(f"🔍 STEP 1: Searching for existing {node_label} with write access")
            logger.debug(f"Search query: {search_query}")
            
            result = await neo_session.run(search_query, params)
            existing_node = await result.single()
            
            if existing_node and existing_node.get('n'):
                # STEP 2a: Node found with write access → Update it
                logger.info(f"✅ STEP 1: Found existing {node_label} - will update")
                
                update_query = f"""
                    MERGE (n:{node_label} {{{unique_conditions}}})
                    SET n += $update_properties
                    RETURN n, false as was_created
                """
                
                logger.info(f"🔄 STEP 2a: Updating existing {node_label}")
                result = await neo_session.run(update_query, params)
                record = await result.single()
            else:
                # STEP 2b: Node not found or no write access → Create new node
                logger.info(f"❌ STEP 1: No existing {node_label} found with write access - will create new")
                
                create_query = f"""
                    CREATE (n:{node_label})
                    SET n = $all_properties
                    RETURN n, true as was_created
                """
                
                logger.info(f"➕ STEP 2b: Creating new {node_label}")
                logger.info(f"🔍 DEBUG: all_properties contains 'id': {'id' in all_properties}")
                logger.info(f"🔍 DEBUG: all_properties['id'] = {all_properties.get('id', 'NOT_FOUND')}")
                result = await neo_session.run(create_query, params)
                record = await result.single()

            if record and record.get('n'):
                node_data = dict(record['n'])
                was_created = record.get('was_created', False)
                
                # Remove the temporary flag from node data (don't persist it)
                if 'was_created_flag' in node_data:
                    del node_data['was_created_flag']
                
                # Add was_created to return data for synchronization
                node_data['was_created'] = was_created
                
                operation = "created" if was_created else "updated"
                logger.info(f"Successfully {operation} {node_label} node with id: {node_data.get('id')} (was_created={was_created})")
                return node_data
            else:
                logger.error(f"MERGE query returned no result for {node_label}")
                return None

        except Exception as e:
            logger.error(f"Error in _merge_node_by_unique_props for {node_label}: {e}")
            return None

    async def _create_node_with_content_check(
        self,
        node: LLMGraphNode,
        common_metadata: Dict[str, Any],
        neo_session: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """
        Enhanced content-based deduplication with Qdrant semantic search.
        
        Flow:
        1. Check Qdrant for semantically similar content (0.95 threshold)
        2. If found, merge in Neo4j with updated content (not canonical like unique IDs)
        3. If not found, create new node
        4. Return result with was_created flag for property indexing sync
        """
        try:
            node_id = node.properties.get('id', str(uuid4()))
            
            # Use priority order: name > title > description > content
            selected_property = None
            node_content = None
            for prop_name in ['name', 'title', 'description', 'content']:
                if node.properties.get(prop_name):
                    node_content = node.properties[prop_name]
                    selected_property = prop_name
                    break

            logger.info(f"🔍 Enhanced Content Check: Processing {node.label} with id={node_id}, selected_property='{selected_property}', content='{node_content[:50] if node_content else 'None'}...'")

            # Step 1: Check Qdrant for semantically similar content
            # Merge common_metadata into props for ACL filtering in Qdrant search
            props_with_acl = {**dict(node.properties), **common_metadata}
            existing_entity = await self._search_qdrant_for_similar_content(
                node_label=node.label,
                content=node_content,
                selected_property=selected_property,
                props=props_with_acl,
                common_metadata=common_metadata
            )

            if existing_entity:
                logger.info(f"🔍 Found semantically similar {node.label} in Qdrant: {existing_entity['canonical_node_id']}")
                
                # Step 2: Merge in Neo4j with UPDATED content (not canonical)
                result = await self._merge_node_with_updated_content(
                    node_label=node.label,
                    new_props=dict(node.properties),
                    existing_node_id=existing_entity['canonical_node_id'],
                    common_metadata=common_metadata,
                    neo_session=neo_session
                )
                
                if result:
                    result['was_created'] = False
                    result['sync_operation'] = 'update'
                    logger.info(f"✅ Merged {node.label} with updated content, id: {result.get('id')}")
                    return result
            
            # Step 3: Fallback to traditional content-based check
            logger.info(f"🔍 No semantic match found, checking traditional content-based deduplication...")
            
            # Handle both system and custom node labels
            try:
                node_type_enum = NodeLabel[node.label]
                logger.info(f"🔍 Found system label enum for {node.label}")
            except KeyError:
                # For custom labels, create a temporary enum-like object
                logger.info(f"🔍 Creating custom label enum for {node.label}")
                node_type_enum = type('CustomNodeLabel', (), {'value': node.label})()

            # For Memory nodes, MUST include tenant scoping to prevent cross-tenant matches
            if node.label == 'Memory' or (hasattr(node_type_enum, 'value') and node_type_enum.value == 'Memory'):
                existing_id = await self._node_exists(
                    node_id=node_id,
                    neo_session=neo_session,
                    node_type=node_type_enum,
                    node_content=node_content,
                    workspace_id=common_metadata.get("workspace_id"),
                    organization_id=common_metadata.get("organization_id"),
                    namespace_id=common_metadata.get("namespace_id"),
                    user_id=common_metadata.get("user_id")
                )
            else:
                existing_id = await self._node_exists(
                    node_id=node_id,
                    neo_session=neo_session,
                    node_type=node_type_enum,
                    node_content=node_content,
                )

            if existing_id:
                logger.info(f"✅ {node.label} node with exact content match already exists with id {existing_id}")
                # Update the existing node with new properties (e.g., llmGenNodeId from manual graph override)
                result = await self._merge_node_with_updated_content(
                    node_label=node.label,
                    new_props=dict(node.properties),
                    existing_node_id=existing_id,
                    common_metadata=common_metadata,
                    neo_session=neo_session
                )
                if result:
                    result['was_created'] = False
                    result['sync_operation'] = 'update'
                    logger.info(f"✅ Updated existing {node.label} node with new properties, id: {result.get('id')}")
                    return result
                else:
                    # Fallback if update fails
                    return {"id": existing_id, "was_created": False, "sync_operation": "existing"}

            # Step 4: Create new node
            logger.info(f"🆕 Creating new {node.label} node with content '{node_content[:50] if node_content else 'None'}...'")
            result = await self._create_node(node=node, common_metadata=common_metadata, neo_session=neo_session)
            if result:
                result['was_created'] = True
                result['sync_operation'] = 'create'
                logger.info(f"✅ Created {node.label} node, result: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ Error in enhanced content check for {node.label}: {e}", exc_info=True)
            return None

    async def _search_qdrant_for_similar_content(
        self,
        node_label: str,
        content: Optional[str],
        selected_property: Optional[str],
        props: Dict[str, Any],
        common_metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Search Qdrant for semantically similar content for content-based deduplication.
        
        Similar to _search_qdrant_for_existing_entity but searches by content similarity
        instead of unique identifier values.
        """
        if not content or not self.qdrant_client:
            return None
            
        # Ensure property collection is initialized (same pattern as property indexing service)
        from os import environ as env
        property_collection_name = env.get("QDRANT_PROPERTY_COLLECTION", "neo4j_properties")
        
        # Set property collection if not already set
        if not hasattr(self, 'qdrant_property_collection') or not self.qdrant_property_collection:
            self.qdrant_property_collection = property_collection_name
            logger.info(f"Set property collection to: {property_collection_name}")
        
        # Ensure collection exists (same pattern as property indexing service)
        try:
            await self.qdrant_client.get_collection(self.qdrant_property_collection)
            logger.info(f"Property collection '{self.qdrant_property_collection}' exists")
        except Exception as e:
            logger.info(f"Property collection '{self.qdrant_property_collection}' doesn't exist, creating it")
            # Create the collection using the same method as main collection
            success = await self.create_optimized_qdrant_collection(
                collection_name=self.qdrant_property_collection,
                vector_size=384  # sentence-bert dimensions
            )
            if not success:
                logger.error(f"Failed to create property collection '{self.qdrant_property_collection}', skipping semantic search")
                return None
            
        # Check if sentence embeddings are available (LOCALPROCESSING or API URL configured)
        # In open-source, skip sentence embeddings if API URL is not configured and LOCALPROCESSING is disabled
        import os
        from os import environ as env
        papr_edition = os.getenv("PAPR_EDITION", "").lower()
        is_opensource = papr_edition == "opensource"
        local_processing = env.get("LOCALPROCESSING", "").lower() == "true"
        hugging_face_api_url_sentence_bert = env.get("HUGGING_FACE_API_URL_SENTENCE_BERT")
        
        # Skip sentence embeddings in open-source if API URL is not configured and local processing is disabled
        if is_opensource and not local_processing and not hugging_face_api_url_sentence_bert:
            logger.debug(f"🔍 SEMANTIC SEARCH: Skipping sentence embeddings in open-source (no API URL configured and LOCALPROCESSING disabled)")
            return None
        
        try:
            # Generate embedding using the same format as property indexing: "Node: NodeType, Property: property_name: value"
            # Use the actual selected property (name, title, description, or content)
            property_name = selected_property or 'content'
            formatted_search_query = f"Node: {node_label}, Property: {property_name}: {content}"
            
            # Generate embedding using HuggingFace API with fast failure
            try:
                embeddings_result, _ = await self.embedding_model.get_sentence_embedding(
                    formatted_search_query,
                    max_retries=1  # Fast failure for content-based deduplication
                )
                if embeddings_result and len(embeddings_result) > 0:
                    content_embedding = embeddings_result[0]
                else:
                    logger.warning(f"🔍 SEMANTIC SEARCH: Failed to generate embedding, skipping content-based deduplication")
                    return None
            except Exception as e:
                logger.warning(f"🔍 SEMANTIC SEARCH: Error generating embedding: {e}, skipping content-based deduplication")
                return None
            
            logger.info(f"🔍 SEMANTIC SEARCH: Generated embedding for formatted query: '{formatted_search_query[:100]}...' (vector size: {len(content_embedding)})")
                
            # Search for similar content in the property collection
            # Look for content or name properties of the same node type
            from qdrant_client import models
            
            # Build ACL filter conditions for end user access
            # Use the same ACL filtering pattern as property search route
            user_id = props.get('user_id')  # End user ID from node properties
            workspace_id = props.get('workspace_id')
            organization_id = props.get('organization_id') 
            namespace_id = props.get('namespace_id')
            
            # Search for content-like properties using property_key format (NodeType.property)
            # For content-based deduplication, focus on name, title, description, and content properties (in priority order)
            content_property_keys = [f"{node_label}.name", f"{node_label}.title", f"{node_label}.description", f"{node_label}.content"]
            
            # CRITICAL: namespace_id and user_id are REQUIRED for multi-tenant isolation
            if not namespace_id:
                raise ValueError(f"namespace_id is required for Qdrant search but was None. Node: {node_label}, props keys: {list(props.keys())}")
            if not user_id:
                raise ValueError(f"user_id is required for Qdrant search but was None. Node: {node_label}, props keys: {list(props.keys())}")
            
            # Build must conditions: property_key AND namespace_id AND user_id (per-user isolation)
            must_conditions = [
                models.FieldCondition(
                    key="property_key",
                    match=models.MatchAny(any=content_property_keys)
                ),
                models.FieldCondition(key="namespace_id", match=models.MatchValue(value=namespace_id)),
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
            ]
            
            # Build should conditions (OR): user has access if ANY of these match
            should_conditions = []
            if user_id:
                should_conditions.append(
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
                )
                should_conditions.append(
                    models.FieldCondition(key="user_read_access", match=models.MatchAny(any=[user_id]))
                )
            else:
                should_conditions.append(
                    models.FieldCondition(key="user_read_access", match=models.MatchAny(any=[]))
                )
            
            # Workspace-level access - check if workspace_id is in workspace_read_access array
            if workspace_id:
                should_conditions.append(
                    models.FieldCondition(key="workspace_read_access", match=models.MatchAny(any=[workspace_id]))
                )
                
            # Organization-level access - check if organization_id is in organization_read_access array
            if organization_id:
                should_conditions.append(
                    models.FieldCondition(key="organization_read_access", match=models.MatchAny(any=[organization_id]))
                )
                
            # Namespace-level access - check if namespace_id is in namespace_read_access array
            if namespace_id:
                should_conditions.append(
                    models.FieldCondition(key="namespace_read_access", match=models.MatchAny(any=[namespace_id]))
                )
            
            # Role-level access - get role_read_access from props if available
            role_read_access = props.get('role_read_access', [])
            if role_read_access:
                should_conditions.append(
                    models.FieldCondition(key="role_read_access", match=models.MatchAny(any=role_read_access))
                )
            
            # ACL filter: must match (property_key AND user_id) AND (user has access via ANY of the should conditions)
            search_filter = models.Filter(
                must=must_conditions,
                should=should_conditions  # OR: user has access if ANY ACL condition matches
            )
            
            # Log detailed ACL filter information
            logger.info(f"🔍 SEMANTIC SEARCH: Searching collection '{self.qdrant_property_collection}'")
            logger.info(f"🔍 ACL FILTER - MUST (AND): property_key IN {content_property_keys} AND namespace_id='{namespace_id}' AND user_id='{user_id}'")
            logger.info(f"🔍 ACL FILTER - SHOULD (OR): {len(should_conditions)} conditions - user_read_access=[{user_id}], workspace_read_access=[{workspace_id}], organization_read_access=[{organization_id}], namespace_read_access=[{namespace_id}], role_read_access={role_read_access}")
            
            # Log detailed search parameters
            # NOTE: Using 0.75 threshold for content-based deduplication (semantic similarity)
            # This is different from unique ID matching (0.95) which requires near-exact matches
            content_similarity_threshold = 0.95
            logger.info(f"🔍 SEMANTIC SEARCH DETAILS: embedding_size={len(content_embedding)}, score_threshold={content_similarity_threshold}, limit=5")
            logger.info(f"🔍 SEMANTIC SEARCH FILTER: {search_filter}")
            
            # Handle embedding as list (HuggingFace API) or numpy array (local models)
            content_embedding_list = content_embedding.tolist() if hasattr(content_embedding, 'tolist') else content_embedding
            
            search_results = await self._qdrant_search_async(
                collection_name=self.qdrant_property_collection,
                query_vector=content_embedding_list,
                query_filter=search_filter,
                limit=5,  # Get top 5 matches for debugging
                score_threshold=content_similarity_threshold,  # Lower threshold for semantic content similarity (not exact matching)
                with_payload=True,
                with_vectors=False
            )
            
            logger.info(f"🔍 SEMANTIC SEARCH: Found {len(search_results)} results with score_threshold={content_similarity_threshold}")
            
            if search_results:
                for i, result in enumerate(search_results):
                    canonical_node_id = result.payload.get('source_node_id')
                    logger.info(f"🔍 SEMANTIC SEARCH: Result {i+1}: score={result.score:.3f}, canonical_node_id={canonical_node_id}, property='{result.payload.get('property_name')}', content='{result.payload.get('property_value', '')[:50]}...'")
                
                # Use the top result
                result = search_results[0]
                canonical_node_id = result.payload.get('source_node_id')
                
                return {
                    'canonical_node_id': canonical_node_id,
                    'similarity_score': result.score,
                    'matched_property': result.payload.get('property_name'),
                    'matched_content': result.payload.get('property_value')
                }
            else:
                # Try a lower threshold search to see if there are ANY matches
                logger.info(f"🔍 SEMANTIC SEARCH: No results with {content_similarity_threshold} threshold, trying 0.5 threshold for debugging...")
                logger.info(f"🔍 DEBUG SEARCH: Using same filter with lower threshold: {search_filter}")
                
                debug_results = await self._qdrant_search_async(
                    collection_name=self.qdrant_property_collection,
                    query_vector=content_embedding_list,
                    query_filter=search_filter,
                    limit=3,
                    score_threshold=0.5,  # Much lower threshold for debugging
                    with_payload=True,
                    with_vectors=False
                )
                
                logger.info(f"🔍 SEMANTIC SEARCH: Found {len(debug_results)} results with score_threshold=0.5")
                
                for i, result in enumerate(debug_results):
                    # Log both the indexed content and the property value for comparison
                    indexed_content = result.payload.get('content', '')  # This should be the formatted content
                    property_value = result.payload.get('property_value', '')
                    logger.info(f"🔍 SEMANTIC SEARCH: Debug result {i+1}: score={result.score:.3f}, node_id={result.payload.get('source_node_id')}, property='{result.payload.get('property_name')}', indexed_content='{indexed_content[:50]}...', raw_value='{property_value[:30]}...'")
                
                # Also try searching for ANY properties of this node type (using property_key exact match)
                logger.info(f"🔍 SEMANTIC SEARCH: Trying search for any {node_label} properties...")
                
                # Search for any property that matches the node type exactly
                # Use MatchAny with all possible property keys for this node type
                possible_property_keys = [f"{node_label}.{prop}" for prop in ["id", "name", "description", "content", "purpose", "version", "ordinal", "required"]]
                any_node_properties_filter = models.Filter(
                    must=[
                        # Use property_key with exact matching instead of text search
                        models.FieldCondition(
                            key="property_key",
                            match=models.MatchAny(any=possible_property_keys)  # Exact match for known property keys
                        )
                    ] + should_conditions
                )
                
                logger.info(f"🔍 ANY PROPERTIES SEARCH: Looking for property_keys={possible_property_keys}")
                logger.info(f"🔍 ANY PROPERTIES FILTER: {any_node_properties_filter}")
                
                any_props_results = await self._qdrant_search_async(
                    collection_name=self.qdrant_property_collection,
                    query_vector=content_embedding_list,
                    query_filter=any_node_properties_filter,
                    limit=3,
                    score_threshold=0.1,  # Very low threshold
                    with_payload=True,
                    with_vectors=False
                )
                
                logger.info(f"🔍 SEMANTIC SEARCH: Found {len(any_props_results)} properties for {node_label} (any property)")
                
                for i, result in enumerate(any_props_results):
                    indexed_content = result.payload.get('content', '')
                    property_value = result.payload.get('property_value', '')
                    logger.info(f"🔍 SEMANTIC SEARCH: Any property {i+1}: score={result.score:.3f}, property='{result.payload.get('property_name')}', indexed_content='{indexed_content[:50]}...', raw_value='{property_value[:30]}...'")
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching Qdrant for similar content: {e}")
            return None

    async def _merge_node_with_updated_content(
        self,
        node_label: str,
        new_props: Dict[str, Any],
        existing_node_id: str,
        common_metadata: Dict[str, Any],
        neo_session: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """
        Merge node in Neo4j with UPDATED content (unlike unique identifiers which stay canonical).
        
        For content-based merging, we update the content to the latest version.
        """
        try:
            # Prepare properties for merge - include all new properties
            merge_props = {**new_props, **common_metadata}
            merge_props['updatedAt'] = datetime.now(timezone.utc).isoformat()
            
            # Use the existing node ID for the merge
            merge_props['id'] = existing_node_id
            
            # Cypher query to merge by ID and update ALL properties with new content
            query = f"""
                MERGE (n:{node_label} {{id: $node_id}})
                ON CREATE SET n += $all_properties, n.was_created_flag = true
                ON MATCH SET n += $all_properties, n.was_created_flag = false
                RETURN n, n.was_created_flag as was_created
            """
            
            params = {
                'node_id': existing_node_id,
                'all_properties': merge_props
            }
            
            logger.info(f"🔄 Content-based merge for {node_label} with id={existing_node_id}")
            
            result = await neo_session.run(query, params)
            record = await result.single()
            
            if record and record.get('n'):
                node_data = dict(record['n'])
                was_created = record.get('was_created', False)
                
                # Remove the temporary flag from node data
                if 'was_created_flag' in node_data:
                    del node_data['was_created_flag']
                
                # Add was_created to return data for synchronization
                node_data['was_created'] = was_created
                
                operation = "created" if was_created else "updated"
                logger.info(f"✅ Content-based {operation} {node_label} node with id: {node_data.get('id')} (was_created={was_created})")
                return node_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error in content-based merge for {node_label}: {e}")
            return None

    @staticmethod
    def _filter_null_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out null values from node properties to prevent storing hallucinated data.
        
        This is part of the anti-hallucination strategy where the LLM is instructed to use
        null for properties where information is not available in the content.
        
        Args:
            properties: Dictionary of node properties that may contain null values
            
        Returns:
            Dictionary with null values removed
        """
        filtered = {}
        for key, value in properties.items():
            if value is not None:
                # For nested dictionaries, recursively filter nulls
                if isinstance(value, dict):
                    filtered_nested = MemoryGraph._filter_null_properties(value)
                    if filtered_nested:  # Only include if not empty after filtering
                        filtered[key] = filtered_nested
                # For lists, filter out null items
                elif isinstance(value, list):
                    filtered_list = [item for item in value if item is not None]
                    if filtered_list:  # Only include if not empty after filtering
                        filtered[key] = filtered_list
                else:
                    filtered[key] = value
        
        logger.info(f"🚫 ANTI-HALLUCINATION: Filtered null properties. Before: {len(properties)} properties, After: {len(filtered)} properties")
        return filtered

    @staticmethod
    def _validate_required_properties(
        node_label: str,
        properties: Dict[str, Any],
        user_schema: Optional[Any] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate that all schema-required properties are present after null filtering.
        
        This enforces the schema's business logic for required fields, which is separate
        from OpenAI's technical requirement that all properties be in the required array.
        
        Schema Required (Business Logic):
        - Properties that MUST have values for node to be valid
        - Defined in UserNodeType.required_properties
        - If missing → node should NOT be created
        
        OpenAI Required (Technical):
        - ALL properties in required array (API constraint)
        - Satisfied by making properties nullable
        
        Args:
            node_label: The label/type of the node being validated
            properties: Properties after null filtering
            user_schema: The UserGraphSchema containing required_properties definition
            
        Returns:
            Tuple of (is_valid, missing_required_fields)
        """
        if not user_schema or not hasattr(user_schema, 'node_types'):
            # No schema or no node types defined - consider valid (system nodes)
            logger.debug(f"✓ VALIDATION: No user schema provided for {node_label}, skipping validation")
            return True, []
        
        # Find the node type definition in schema
        node_type_def = None
        for node_name, node_type in user_schema.node_types.items():
            if node_name == node_label:
                node_type_def = node_type
                break
        
        if not node_type_def:
            # Node type not in schema (system type or different schema) - consider valid
            logger.debug(f"✓ VALIDATION: Node type {node_label} not in schema, skipping validation")
            return True, []
        
        # Get required properties from schema
        required_props = getattr(node_type_def, 'required_properties', [])
        
        if not required_props:
            # No required properties defined - valid
            logger.debug(f"✓ VALIDATION: No required properties for {node_label}")
            return True, []
        
        # Check which required properties are missing
        missing_required = []
        for req_prop in required_props:
            if req_prop not in properties or properties[req_prop] is None:
                missing_required.append(req_prop)
        
        is_valid = len(missing_required) == 0
        
        if is_valid:
            logger.info(f"✓ VALIDATION: {node_label} has all required properties: {required_props}")
        else:
            logger.warning(f"✗ VALIDATION: {node_label} missing required properties: {missing_required}")
            logger.warning(f"✗ VALIDATION: Present properties: {list(properties.keys())}")
        
        return is_valid, missing_required

    async def store_llm_generated_graph(
        self,
        nodes: List[LLMGraphNode],
        relationships: List[LLMGraphRelationship],
        memory_item: Dict[str, Any],
        neo_session: AsyncSession, 
        workspace_id: Optional[str] = None,
        user_schema: Optional[Any] = None
    ):
        """
        Stores the LLM-generated graph structure in Neo4j, applying proper metadata and access controls.
        Integrates with OMO (Open Memory Object) safety standards for consent, risk, and audit tracking.
        """
        # Fallback mode check: skip Neo4j operations if in fallback
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, cannot store graph")
            return

        # Extract metadata from memory item
        metadata = json.loads(memory_item['metadata']) if isinstance(memory_item.get('metadata'), str) else memory_item.get('metadata', {})

        # Extract customMetadata for multi-tenant fields
        custom_metadata = metadata.get('customMetadata', {})

        # OMO Safety Standards: Extract consent, risk, and ACL from metadata
        omo_consent = metadata.get("consent", "implicit")  # Default to implicit consent
        omo_risk = metadata.get("risk", "none")  # Default to no risk
        acl = metadata.get("acl")  # Optional explicit ACL
        external_user_id = metadata.get("external_user_id")
        memory_id = memory_item.get("id") or memory_item.get("memoryId")

        # OMO Consent Enforcement: Skip graph extraction if consent is 'none'
        if omo_consent == "none":
            logger.warning(
                f"Memory {memory_id} has consent='none' - skipping graph storage. "
                "Memories with no consent should not have graph nodes created."
            )
            return

        # OMO Audit Trail: Track extraction metadata
        omo_audit = {
            "source_memory_id": memory_id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "consent": omo_consent,
            "risk": omo_risk,
            "extraction_method": "llm"  # This function handles LLM-extracted graphs
        }

        # Common metadata fields to include for all nodes
        common_metadata = {
            "user_id": metadata.get("user_id"),
            "pageId": metadata.get("pageId"),
            "conversationId": metadata.get("conversationId"),
            "sourceType": metadata.get("sourceType"),
            "sourceUrl": metadata.get("sourceUrl"),
            "workspace_id": workspace_id or metadata.get("workspace_id"),
            # Multi-tenant scoping fields - top-level fields in MemoryMetadata
            "organization_id": metadata.get("organization_id"),
            "namespace_id": metadata.get("namespace_id"),
            "user_read_access": metadata.get("user_read_access", []),
            "user_write_access": metadata.get("user_write_access", []),
            "workspace_read_access": metadata.get("workspace_read_access", []),
            "workspace_write_access": metadata.get("workspace_write_access", []),
            "role_read_access": metadata.get("role_read_access", []),
            "role_write_access": metadata.get("role_write_access", []),
            "organization_read_access": metadata.get("organization_read_access", []),
            "organization_write_access": metadata.get("organization_write_access", []),
            "namespace_read_access": metadata.get("namespace_read_access", []),
            "namespace_write_access": metadata.get("namespace_write_access", []),
            "external_user_read_access": metadata.get("external_user_read_access", []),
            "external_user_write_access": metadata.get("external_user_write_access", []),
            "createdAt": metadata.get("createdAt") or datetime.now(timezone.utc).isoformat(),
            # OMO Safety Standard annotations
            "_omo_consent": omo_consent,
            "_omo_risk": omo_risk,
            "_omo_source_memory_id": memory_id,
            "_omo_audit": json.dumps(omo_audit) if omo_audit else None
        }

        # OMO Risk Enforcement: For flagged content, restrict ACL to owner only
        if omo_risk == "flagged":
            logger.warning(f"Memory {memory_id} has risk='flagged' - restricting ACL to owner only")
            common_metadata["_omo_requires_review"] = True
            if external_user_id:
                # Override ACL to restrict to owner only
                common_metadata["external_user_read_access"] = [external_user_id]
                common_metadata["external_user_write_access"] = [external_user_id]

        # OMO ACL Propagation: Use explicit ACL if provided
        if acl:
            logger.debug(f"Using explicit acl for memory {memory_id}: {acl}")
            if isinstance(acl, dict):
                if acl.get("read"):
                    common_metadata["external_user_read_access"] = acl["read"]
                if acl.get("write"):
                    common_metadata["external_user_write_access"] = acl["write"]

        logger.info(f"OMO Safety: Processing graph for memory {memory_id} with consent={omo_consent}, risk={omo_risk}")

        # Convert dictionary nodes to Node objects if needed
        nodes_objects = [
            node if isinstance(node, LLMGraphNode) else LLMGraphNode(**node)
            for node in nodes
        ]

        # Convert dictionary relationships to Relationship objects if needed  
        relationship_objects = [
            rel if isinstance(rel, LLMGraphRelationship) else LLMGraphRelationship(**rel)
            for rel in relationships
        ]

        # Register custom node and relationship types from the generated nodes
        from models.shared_types import NodeLabel, RelationshipType
        custom_node_labels = list(set([node.label for node in nodes_objects if node.label not in NodeLabel.get_system_labels()]))
        custom_relationship_types = list(set([rel.type for rel in relationship_objects if rel.type not in RelationshipType.get_system_relationships()]))
        
        if custom_node_labels:
            NodeLabel.register_custom_labels(custom_node_labels)
            logger.info(f"🔧 NEO4J STORAGE: Registered custom node labels: {custom_node_labels}")
        
        if custom_relationship_types:
            RelationshipType.register_custom_relationships(custom_relationship_types)
            logger.info(f"🔧 NEO4J STORAGE: Registered custom relationship types: {custom_relationship_types}")

        # Create nodes first and collect results for property indexing
        neo4j_results = []
        
        for node in nodes_objects:
            node_id = node.properties.get('id', str(uuid4()))
            node_content = node.properties.get('content') or node.properties.get('name')

            logger.info(f"Processing {node.label} node with content: {node_content}")
            
            try:
                # Memory nodes: Keep existing content-based deduplication with tenant scoping
                if node.label == 'Memory':
                    logger.info(f"Processing Memory node with existing logic")
                    # Handle both system and custom node labels
                    try:
                        node_type_enum = NodeLabel[node.label]
                    except KeyError:
                        # For custom labels, create a temporary enum-like object
                        logger.info(f"🔧 CUSTOM LABEL: Using custom node label '{node.label}' directly")
                        node_type_enum = type('CustomNodeLabel', (), {'value': node.label})()
                    
                    # CRITICAL: Memory nodes MUST have tenant scoping to prevent cross-tenant matching
                    existing_id = await self._node_exists(
                        node_id=node_id,
                        neo_session=neo_session,
                        node_type=node_type_enum,
                        node_content=node_content,
                        workspace_id=common_metadata.get("workspace_id"),
                        organization_id=common_metadata.get("organization_id"),
                        namespace_id=common_metadata.get("namespace_id"),
                        user_id=common_metadata.get("user_id")
                    )
                    
                    if existing_id:
                        logger.info(f"{node.label} node with content '{node_content}' already exists with id {existing_id}, skipping creation")
                        # Update the node's id to match the existing one for relationship creation
                        node.properties['id'] = existing_id
                        # Add to results for property indexing (existing node)
                        neo4j_results.append({
                            'label': node.label.value if hasattr(node.label, 'value') else str(node.label),
                            'properties': node.properties.model_dump() if hasattr(node.properties, 'model_dump') else dict(node.properties),
                            'was_created': False,
                            'sync_operation': 'existing',
                            'node_id': existing_id
                        })
                        continue

                    logger.info(f"Creating new {node.label} node with content '{node_content}'")
                    create_result = await self._create_node(node=node, common_metadata=common_metadata, neo_session=neo_session)
                    if create_result:
                        # Add to results for property indexing (new node)
                        neo4j_results.append({
                            'label': node.label.value if hasattr(node.label, 'value') else str(node.label),
                            'properties': create_result,
                            'was_created': True,
                            'sync_operation': 'create',
                            'node_id': create_result.get('id')
                        })
                
                else:
                    # Non-Memory nodes: Use schema-aware MERGE with Qdrant synchronization
                    logger.info(f"Processing non-Memory node {node.label} with schema-aware MERGE logic")
                    result = await self._merge_node_with_unique_identifiers(
                        node=node, 
                        common_metadata=common_metadata, 
                        neo_session=neo_session,
                        workspace_id=workspace_id,
                        user_schema=user_schema
                    )
                    
                    if result and result.get('id'):
                        # Update the node's id for relationship creation
                        logger.info(f"🔍 DEBUG: result from merge: {result}")
                        logger.info(f"🔍 DEBUG: result['id'] = {result.get('id')}")
                        logger.info(f"🔍 DEBUG: node.properties before update: {node.properties}")
                        node.properties['id'] = result['id']
                        logger.info(f"🔍 DEBUG: node.properties after update: {node.properties}")
                        logger.info(f"Node {node.label} processed with id: {result['id']}")
                        
                        # Add to results for property indexing with sync info
                        neo4j_results.append({
                            'label': node.label.value if hasattr(node.label, 'value') else str(node.label),
                            'properties': result,
                            'was_created': result.get('was_created', False),
                            'sync_operation': result.get('sync_operation', 'unknown'),
                            'node_id': result.get('id')
                        })

            except Exception as e:
                logger.error(f"Error processing node {node.label} with content '{node_content}': {e}")
                raise

        # Build ID mapping ONLY from successfully created/merged nodes
        # Use neo4j_results which contains only nodes that were successfully processed
        id_mapping = {}
        for result in neo4j_results:
            neo4j_uuid = result.get('node_id')
            properties = result.get('properties', {})
            llm_gen_id = properties.get('llmGenNodeId')
            
            if neo4j_uuid:
                # Map llmGenNodeId to UUID (for auto mode and manual mode with unique identifiers)
                if llm_gen_id:
                    id_mapping[llm_gen_id] = neo4j_uuid
                    logger.debug(f"📍 ID MAPPING: {llm_gen_id} -> {neo4j_uuid}")
                # Also map UUID to itself (for Memory nodes and nodes with custom IDs)
                id_mapping[neo4j_uuid] = neo4j_uuid
        
        # CRITICAL FIX: Add Memory node ID to mapping for EXTRACTED relationships
        # The Memory node was created earlier (in add_memory_item_to_neo4j) and is NOT in neo4j_results
        # EXTRACTED relationships use the Memory node's UUID as source ID
        if memory_item and hasattr(memory_item, 'id'):
            memory_node_id = str(memory_item.id)
            id_mapping[memory_node_id] = memory_node_id
            logger.info(f"📍 ID MAPPING: Added Memory node ID to mapping: {memory_node_id}")
        
        logger.info(f"📍 ID MAPPING: Built mapping for {len(id_mapping)} successfully created node IDs")

        # Filter relationships - only create for nodes that were successfully created
        valid_relationships = []
        skipped_count = 0
        
        for rel in relationship_objects:
            source_original_id = rel.source.id
            target_original_id = rel.target.id
            
            # Check if both source and target nodes were successfully created
            if source_original_id not in id_mapping:
                logger.debug(f"⚠️  Skipping relationship {rel.type}: source node '{source_original_id}' was not created")
                skipped_count += 1
                continue
            if target_original_id not in id_mapping:
                logger.debug(f"⚠️  Skipping relationship {rel.type}: target node '{target_original_id}' was not created")
                skipped_count += 1
                continue
            
            # Update relationship IDs to Neo4j UUIDs
            rel.source.id = id_mapping[source_original_id]
            rel.target.id = id_mapping[target_original_id]
            valid_relationships.append(rel)
        
        if skipped_count > 0:
            logger.warning(f"⚠️  Skipped {skipped_count} relationships - nodes were not created (missing unique identifiers or creation failed)")
        logger.info(f"✅ Creating {len(valid_relationships)} relationships for successfully created nodes")

        # Create only valid relationships
        for rel in valid_relationships:
            await self._create_relationship(neo_session=neo_session, relationship=rel, common_metadata=common_metadata)
        
        # CRITICAL FIX: Make property indexing synchronous (await instead of background task)
        # This ensures properties are indexed BEFORE the response is returned
        # Previously: asyncio.create_task() caused race condition where next request arrived before indexing completed
        if neo4j_results and env.get('ENABLE_PROPERTY_INDEXING', 'true').lower() == 'true':
            logger.info(f"🔧 SYNC PROPERTY INDEXING: Starting with {len(neo4j_results)} Neo4j results")
            await self._index_node_properties_with_sync_results(
                neo4j_results=neo4j_results,
                memory_item=memory_item,
                workspace_id=workspace_id,
                user_schema=user_schema,
                common_metadata=common_metadata  # Pass common_metadata which has organization_id/namespace_id correctly extracted
            )
            logger.info(f"🔧 SYNC PROPERTY INDEXING: ✅ Completed - properties now available for next dedup check")

    async def _create_node(self, node: LLMGraphNode, common_metadata: dict, neo_session: AsyncSession) -> Optional[Dict[str, Any]]:
        """Helper method to create a single node, with robust fallback and error handling."""
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, skipping _create_node")
            return None

        try:
            # --- KEEP ALL EXISTING LOGIC BELOW THIS LINE UNCHANGED ---
            if hasattr(node.properties, 'model_dump'):
                # If it's a Pydantic model
                props = node.properties.model_dump(exclude_none=True)
            else:
                # If it's already a dictionary
                props = node.properties

            # ANTI-HALLUCINATION: Filter out null properties from LLM output
            props = self._filter_null_properties(props)
            
            # SCHEMA VALIDATION: Check that required properties are present
            # Note: This happens BEFORE property overrides are applied (in generate_node_ids)
            # Property overrides can "rescue" nodes by providing missing required fields
            node_label_str = node.label.value if hasattr(node.label, 'value') else str(node.label)
            is_valid, missing_fields = self._validate_required_properties(
                node_label_str, props, None  # user_schema not available at this point
            )
            # Note: Full validation with property overrides happens in generate_node_ids()
            # This is just a preliminary check for obvious issues

            if 'metadata' in props and isinstance(props['metadata'], dict):
                props['metadata'] = json.dumps(props['metadata'])
            
            # Add common metadata
            props.update({k: v for k, v in common_metadata.items() if v is not None})

            # Create node with label and properties
            label = node.label.value if isinstance(node.label, NodeLabel) else node.label
            query = (
                f"CREATE (n:{label} $props) "
                "RETURN n"
            )
            try:
                result = await neo_session.run(query, props=props)
                await result.consume()
                logger.info(f"Created node with label {label} and id {props.get('id')}")
                return props
            except Exception as e:
                logger.error(f"Error creating node: {e}")
                raise
        except Exception as e:
            logger.error(f"Error in _create_node: {e}")
            self.async_neo_conn.fallback_mode = True
            return None

    async def _index_exists_async(self, session, index_name: str) -> bool:
        """Check if an index exists (async version)"""
        result = await session.run("SHOW INDEXES WHERE name = $name", 
                                 {"name": index_name})
        records = await result.values()
        return len(records) > 0

    async def initialize_indexes_async(self):
        """Creates necessary indexes if they don't exist (async version)"""
        driver = await self.async_neo_conn.get_driver()
        if driver is None:
            logger.warning("No Neo4j connection available, skipping index initialization")
            return
            
        try:
            await self.ensure_async_connection()
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, skipping index initialization")
                return
            
            # Base indexes for ID and name
            indexes = {
                "person_id_idx": "CREATE INDEX person_id_idx IF NOT EXISTS FOR (n:Person) ON (n.id)",
                "company_id_idx": "CREATE INDEX company_id_idx IF NOT EXISTS FOR (n:Company) ON (n.id)",
                "customer_id_idx": "CREATE INDEX customer_id_idx IF NOT EXISTS FOR (n:Customer) ON (n.id)",
                "project_id_idx": "CREATE INDEX project_id_idx IF NOT EXISTS FOR (n:Project) ON (n.id)",
                "memory_id_idx": "CREATE INDEX memory_id_idx IF NOT EXISTS FOR (n:Memory) ON (n.id)",
                "task_id_idx": "CREATE INDEX task_id_idx IF NOT EXISTS FOR (n:Task) ON (n.id)",
                "insight_id_idx": "CREATE INDEX insight_id_idx IF NOT EXISTS FOR (n:Insight) ON (n.id)",
                "opportunity_id_idx": "CREATE INDEX opportunity_id_idx IF NOT EXISTS FOR (n:Opportunity) ON (n.id)",
                "code_id_idx": "CREATE INDEX code_id_idx IF NOT EXISTS FOR (n:Code) ON (n.id)",
                "meeting_id_idx": "CREATE INDEX meeting_id_idx IF NOT EXISTS FOR (n:Meeting) ON (n.id)",
                
                # Content-based index for Memory nodes
                "memory_content_idx": "CREATE INDEX memory_content_idx IF NOT EXISTS FOR (n:Memory) ON (n.content)",
                
                # Name-based indexes for entity nodes
                "person_name_idx": "CREATE INDEX person_name_idx IF NOT EXISTS FOR (n:Person) ON (n.name)",
                "company_name_idx": "CREATE INDEX company_name_idx IF NOT EXISTS FOR (n:Company) ON (n.name)",
                "customer_name_idx": "CREATE INDEX customer_name_idx IF NOT EXISTS FOR (n:Customer) ON (n.name)",
                "project_name_idx": "CREATE INDEX project_name_idx IF NOT EXISTS FOR (n:Project) ON (n.name)"
            }
            
            # Add access control indexes for each node type from NodeLabel enum
            for node_type in NodeLabel:
                node_label = node_type.value
                node_name = node_label.lower()
                
                # User ID index
                indexes[f"{node_name}_user_id_idx"] = f"CREATE INDEX {node_name}_user_id_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.user_id)"
                
                # Workspace indexes
                indexes[f"{node_name}_workspace_id_idx"] = f"CREATE INDEX {node_name}_workspace_id_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.workspace_id)"
                indexes[f"{node_name}_workspace_access_idx"] = f"CREATE INDEX {node_name}_workspace_access_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.workspace_read_access)"
                
                # User access index
                indexes[f"{node_name}_user_access_idx"] = f"CREATE INDEX {node_name}_user_access_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.user_read_access)"
                
                # Role access index
                indexes[f"{node_name}_role_access_idx"] = f"CREATE INDEX {node_name}_role_access_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.role_read_access)"
                
                # Organization indexes
                indexes[f"{node_name}_organization_id_idx"] = f"CREATE INDEX {node_name}_organization_id_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.organization_id)"
                indexes[f"{node_name}_organization_access_idx"] = f"CREATE INDEX {node_name}_organization_access_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.organization_read_access)"
                
                # Namespace indexes
                indexes[f"{node_name}_namespace_id_idx"] = f"CREATE INDEX {node_name}_namespace_id_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.namespace_id)"
                indexes[f"{node_name}_namespace_access_idx"] = f"CREATE INDEX {node_name}_namespace_access_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.namespace_read_access)"
            
            #async with driver.session() as session:
            async with self.async_neo_conn.get_session() as session:
                for index_name, create_query in indexes.items():
                    if not await self._index_exists_async(session, index_name):
                        await session.run(create_query)
                        logger.info(f"Created new index: {index_name}")
                    else:
                        logger.info(f"Index already exists: {index_name}")
                        
        except Exception as e:
            logger.error(f"Error managing indexes: {e}")
            raise

    async def _create_custom_schema_indexes(self, user_schemas: List[Dict[str, Any]]):
        """
        Create Neo4j indexes for custom schema node types and their required properties.
        
        Args:
            user_schemas: List of user schema dictionaries containing node_types and their properties
        """
        try:
            from models.user_schemas import PropertyType
            
            indexes = {}
            
            for schema in user_schemas:
                schema_id = schema.get('id', 'unknown')
                node_types = schema.get('node_types', {})
                
                logger.info(f"🔧 Creating indexes for custom schema {schema_id} with {len(node_types)} node types")
                
                for node_name, node_type in node_types.items():
                    # Get required properties and their definitions
                    required_properties = node_type.get('required_properties', [])
                    properties = node_type.get('properties', {})
                    
                    logger.info(f"🔧 Processing node type '{node_name}' with {len(required_properties)} required properties")
                    
                    # Create indexes for required properties based on their types
                    for prop_name in required_properties:
                        if prop_name in properties:
                            prop_def = properties[prop_name]
                            prop_type = prop_def.get('type', 'STRING')
                            
                            # Generate index name using custom naming convention
                            index_name = f"custom_{node_name.lower()}_{prop_name}_idx"
                            
                            # Create type-appropriate index
                            if prop_type in [PropertyType.STRING.value, PropertyType.ARRAY.value]:
                                # Text index for string searches (CONTAINS, STARTS WITH)
                                indexes[index_name] = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{node_name}) ON (n.{prop_name})"
                            elif prop_type in [PropertyType.INTEGER.value, PropertyType.FLOAT.value]:
                                # Range index for numeric comparisons
                                indexes[index_name] = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{node_name}) ON (n.{prop_name})"
                            elif prop_type == PropertyType.DATETIME.value:
                                # Range index for temporal queries
                                indexes[index_name] = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{node_name}) ON (n.{prop_name})"
                            elif prop_type == PropertyType.BOOLEAN.value:
                                # Simple index for boolean properties
                                indexes[index_name] = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{node_name}) ON (n.{prop_name})"
                            else:
                                # Default to simple index for unknown types
                                indexes[index_name] = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{node_name}) ON (n.{prop_name})"
                            
                            logger.info(f"🔧 Prepared index for {node_name}.{prop_name} (type: {prop_type})")
                    
                    # Create ACL indexes for each custom node type
                    acl_properties = [
                        'user_id', 'workspace_id', 'organization_id', 'namespace_id',
                        'user_read_access', 'workspace_read_access', 'role_read_access',
                        'organization_read_access', 'namespace_read_access'
                    ]
                    
                    for acl_prop in acl_properties:
                        index_name = f"custom_{node_name.lower()}_{acl_prop}_idx"
                        indexes[index_name] = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{node_name}) ON (n.{acl_prop})"
            
            # Execute index creation
            if indexes:
                logger.info(f"🔧 Creating {len(indexes)} custom schema indexes")
                async with self.async_neo_conn.get_session() as session:
                    created_count = 0
                    existing_count = 0
                    
                    for index_name, create_query in indexes.items():
                        try:
                            if not await self._index_exists_async(session, index_name):
                                await session.run(create_query)
                                logger.info(f"✅ Created custom index: {index_name}")
                                created_count += 1
                            else:
                                logger.info(f"⏭️ Custom index already exists: {index_name}")
                                existing_count += 1
                        except Exception as e:
                            logger.error(f"❌ Failed to create custom index {index_name}: {e}")
                    
                    logger.info(f"🎯 Custom schema indexing complete: {created_count} created, {existing_count} existing")
            else:
                logger.info("🔧 No custom schema indexes to create")
                
        except Exception as e:
            logger.error(f"❌ Error creating custom schema indexes: {e}")
            # Don't raise - index creation failure shouldn't break schema registration
            
    async def _create_relationship(self, neo_session: AsyncSession, relationship: LLMGraphRelationship, common_metadata: dict):
        """
        Helper method to create a single relationship.

        IMPORTANT: Includes workspace_id and user_id in node MATCH to ensure multi-tenant isolation.
        This prevents creating relationships between nodes from different users or workspaces.
        """
        try:
            # Create a properties dictionary for the relationship
            props = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                **common_metadata
            }

             # Get relationship type
            relationship_type = relationship.type.value if isinstance(relationship.type, RelationshipType) else relationship.type

            # Get source and target information
            source = relationship.source
            target = relationship.target

            # Get source and target IDs and labels
            source_id = source.id if hasattr(source, 'id') else source['id']
            target_id = target.id if hasattr(target, 'id') else target['id']

            # Keep original IDs - they should match the 'id' property stored on nodes

            # Get labels from source and target
            source_label = source.label if hasattr(source, 'label') else source['label']
            target_label = target.label if hasattr(target, 'label') else target['label']

            # Convert labels to their string values if they're enums
            source_label = source_label.value if isinstance(source_label, NodeLabel) else source_label
            target_label = target_label.value if isinstance(target_label, NodeLabel) else target_label

            # Ensure relationship_type is a valid Neo4j relationship type (no dots, spaces, etc.)
            # Convert to uppercase and replace invalid characters with underscores
            safe_relationship_type = str(relationship_type).upper().replace('.', '_').replace(' ', '_')

            # CRITICAL: Extract all tenant isolation and ACL fields for multi-tenant isolation
            workspace_id = props.get('workspace_id')
            user_id = props.get('user_id')
            organization_id = props.get('organization_id')
            namespace_id = props.get('namespace_id')
            
            # Extract ACL fields
            user_read_access = props.get('user_read_access', [])
            user_write_access = props.get('user_write_access', [])
            workspace_read_access = props.get('workspace_read_access', [])
            workspace_write_access = props.get('workspace_write_access', [])
            role_read_access = props.get('role_read_access', [])
            role_write_access = props.get('role_write_access', [])
            organization_read_access = props.get('organization_read_access', [])
            organization_write_access = props.get('organization_write_access', [])
            namespace_read_access = props.get('namespace_read_access', [])
            namespace_write_access = props.get('namespace_write_access', [])

            if not workspace_id or not user_id:
                logger.error(f"Missing workspace_id or user_id for relationship {safe_relationship_type} - multi-tenant isolation may be compromised!")

            logger.info(f"🔒 Creating relationship {safe_relationship_type} with full tenant isolation (workspace={workspace_id}, user={user_id}, org={organization_id}, namespace={namespace_id})")

            # CRITICAL: After the ID mapping fix, source_llm_id and target_llm_id now contain Neo4j UUIDs
            # All nodes (Memory, entities, etc.) are matched by their UUID 'id' field
            # The ID mapping in store_llm_generated_graph ensures we have the correct UUIDs
            
            # Use 'id' field for all nodes (it now contains the Neo4j UUID)
            source_match_field = "id"
            target_match_field = "id"
            
            # CRITICAL: Match nodes by UUID id + namespace_id + organization_id + workspace_id
            # NOTE: user_id is NOT in MATCH - it's in WHERE clause (OR logic)
            # Creating a relationship is a WRITE operation, so we check *_write_access (not read_access)
            # MATCH narrows by tenant (namespace/org/workspace), WHERE checks write permission
            
            # CRITICAL FIX: Handle NULL/missing organization_id and namespace_id
            # In Neo4j, {property: null} doesn't match nodes without that property
            # Build conditional MATCH based on which tenant IDs are present
            match_conditions = [f"{source_match_field}: $source_llm_id"]
            if workspace_id:
                match_conditions.append("workspace_id: $workspace_id")
            if organization_id:
                match_conditions.append("organization_id: $organization_id")
            if namespace_id:
                match_conditions.append("namespace_id: $namespace_id")
            
            source_match_clause = ", ".join(match_conditions)
            target_match_clause = source_match_clause.replace("source_llm_id", "target_llm_id")
            
            query = f"""
            MATCH (source:{source_label} {{{source_match_clause}}})
            WHERE source.user_id = $user_id 
               OR $user_id IN source.user_write_access
               OR $workspace_id IN source.workspace_write_access
               OR $organization_id IN source.organization_write_access
               OR $namespace_id IN source.namespace_write_access
            MATCH (target:{target_label} {{{target_match_clause}}})
            WHERE target.user_id = $user_id
               OR $user_id IN target.user_write_access
               OR $workspace_id IN target.workspace_write_access
               OR $organization_id IN target.organization_write_access
               OR $namespace_id IN target.namespace_write_access
            MERGE (source)-[r:{safe_relationship_type}]->(target)
            ON CREATE SET r += $props
            RETURN r
            """

            result = await neo_session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                workspace_id=workspace_id,
                user_id=user_id,
                organization_id=organization_id,
                namespace_id=namespace_id,
                user_read_access=user_read_access,
                user_write_access=user_write_access,
                workspace_read_access=workspace_read_access,
                workspace_write_access=workspace_write_access,
                role_read_access=role_read_access,
                role_write_access=role_write_access,
                organization_read_access=organization_read_access,
                organization_write_access=organization_write_access,
                namespace_read_access=namespace_read_access,
                namespace_write_access=namespace_write_access,
                props=props
            )
            await result.consume()
            logger.info(f"Created relationship {safe_relationship_type} from {source_id} to {target_id}")
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            raise

   
   
    @staticmethod
    def _make_schema_nullable(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively make all properties in a JSON schema nullable to prevent LLM hallucination.
        
        Uses union type syntax ["type", "null"] as per Azure/OpenAI documentation.
        
        Args:
            schema: JSON schema dictionary to modify
            
        Returns:
            Modified schema with nullable properties
        """
        if not isinstance(schema, dict):
            return schema
        
        # Make a deep copy to avoid modifying the original
        import copy
        schema = copy.deepcopy(schema)
        
        # First, recursively process nested structures (anyOf, items) BEFORE processing properties
        # This ensures we go deep into the schema tree first
        if "anyOf" in schema and isinstance(schema["anyOf"], list):
            schema["anyOf"] = [MemoryGraph._make_schema_nullable(item) for item in schema["anyOf"]]
        
        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = MemoryGraph._make_schema_nullable(schema["items"])
        
        # Now handle property definitions at this level
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_def in schema["properties"].items():
                if isinstance(prop_def, dict):
                    # Recursively process nested objects FIRST
                    if "properties" in prop_def or "anyOf" in prop_def or "items" in prop_def:
                        schema["properties"][prop_name] = MemoryGraph._make_schema_nullable(prop_def)
                    
                    # Then make this property nullable if it has a simple type
                    elif "type" in prop_def and "enum" not in prop_def:
                        prop_type = prop_def["type"]
                        # Skip if already nullable
                        if isinstance(prop_type, list):
                            if "null" not in prop_type:
                                prop_def["type"] = prop_type + ["null"]
                        elif prop_type != "null":
                            # Convert to union type with null
                            prop_def["type"] = [prop_type, "null"]
                            
                            # Add description guidance
                            if "description" in prop_def:
                                description = prop_def["description"]
                                if not description.endswith("Use null if not available in the content."):
                                    prop_def["description"] = f"{description}. Use null if not available in the content."
                            else:
                                prop_def["description"] = "Use null if not available in the content."
                    
                    # Handle enum properties with nullable support
                    # ⚠️ CRITICAL: OpenAI does NOT allow None in enum arrays - use union type instead
                    # Example: {"type": ["string", "null"], "enum": ["F", "C"]} ✅ VALID
                    #          {"type": "string", "enum": ["F", "C", None]} ❌ INVALID
                    elif "enum" in prop_def:
                        # Filter out None from enum if present (OpenAI rejects None in enum arrays)
                        enum_list = [v for v in prop_def["enum"] if v is not None]
                        
                        # Ensure we have a type - default to string if not specified
                        if "type" not in prop_def:
                            base_type = "string"
                        else:
                            base_type = prop_def["type"]
                            # If type is already a list, extract the base type (first non-null type)
                            if isinstance(base_type, list):
                                base_type = next((t for t in base_type if t != "null"), "string")
                        
                        # Use union type to allow null while keeping enum constraint
                        # This follows OpenAI's documented pattern for optional enum parameters
                        prop_def["type"] = [base_type, "null"]
                        prop_def["enum"] = enum_list  # Enum array contains only valid values (no None)
                        
                        # Update description to indicate null is allowed
                        if "description" in prop_def:
                            if "or null" not in prop_def["description"].lower():
                                prop_def["description"] = f"{prop_def['description']} or null if not available"
                        else:
                            prop_def["description"] = "Must be one of the enum values or null if not available"
        
        return schema

    @staticmethod
    def _transform_system_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform the system schema to:
        1. Add llmGenNodeId to all node types
        2. Remove Memory node type (Memory nodes are created separately)
        3. Update relationship enums to exclude Memory
        """
        import copy
        schema = copy.deepcopy(schema)
        
        # Transform nodes
        if 'properties' in schema and 'nodes' in schema['properties']:
            nodes_schema = schema['properties']['nodes']
            if 'items' in nodes_schema and 'anyOf' in nodes_schema['items']:
                node_types = nodes_schema['items']['anyOf']
                
                # Filter out Memory node type and add llmGenNodeId to others
                filtered_nodes = []
                for node_type in node_types:
                    if 'properties' in node_type and 'label' in node_type['properties']:
                        label_enum = node_type['properties']['label'].get('enum', [])
                        
                        # Skip Memory node type
                        if 'Memory' in label_enum:
                            continue
                        
                        # Add llmGenNodeId to node properties
                        if 'properties' in node_type['properties']:
                            node_props = node_type['properties']['properties']
                            if 'properties' in node_props:
                                # Add llmGenNodeId as first property
                                node_props['properties'] = {
                                    'llmGenNodeId': {
                                        'type': 'string',
                                        'description': 'A unique identifier you generate for this node (e.g., "task_1", "person_john"). Required for creating relationships.'
                                    },
                                    **node_props['properties']
                                }
                                
                                # Add llmGenNodeId to required fields
                                if 'required' in node_props:
                                    if 'llmGenNodeId' not in node_props['required']:
                                        node_props['required'] = ['llmGenNodeId'] + node_props['required']
                        
                        filtered_nodes.append(node_type)
                
                nodes_schema['items']['anyOf'] = filtered_nodes
        
        # Transform relationships - remove Memory from source/target enums and rename 'id' to 'llmGenNodeId'
        if 'properties' in schema and 'relationships' in schema['properties']:
            rels_schema = schema['properties']['relationships']
            if 'items' in rels_schema and 'properties' in rels_schema['items']:
                rel_props = rels_schema['items']['properties']
                
                # Update source: remove Memory from enum and rename 'id' to 'llmGenNodeId'
                if 'source' in rel_props and 'properties' in rel_props['source']:
                    source_props = rel_props['source']['properties']
                    if 'type' in source_props and 'enum' in source_props['type']:
                        source_props['type']['enum'] = [
                            t for t in source_props['type']['enum'] if t != 'Memory'
                        ]
                    # Rename 'id' to 'llmGenNodeId'
                    if 'id' in source_props:
                        source_props['llmGenNodeId'] = source_props.pop('id')
                        source_props['llmGenNodeId']['description'] = 'llmGenNodeId of the source node'
                    # Update required array
                    if 'required' in rel_props['source'] and 'id' in rel_props['source']['required']:
                        req_list = rel_props['source']['required']
                        rel_props['source']['required'] = ['llmGenNodeId' if r == 'id' else r for r in req_list]
                
                # Update target: remove Memory from enum and rename 'id' to 'llmGenNodeId'
                if 'target' in rel_props and 'properties' in rel_props['target']:
                    target_props = rel_props['target']['properties']
                    if 'type' in target_props and 'enum' in target_props['type']:
                        target_props['type']['enum'] = [
                            t for t in target_props['type']['enum'] if t != 'Memory'
                        ]
                    # Rename 'id' to 'llmGenNodeId'
                    if 'id' in target_props:
                        target_props['llmGenNodeId'] = target_props.pop('id')
                        target_props['llmGenNodeId']['description'] = 'llmGenNodeId of the target node'
                    # Update required array
                    if 'required' in rel_props['target'] and 'id' in rel_props['target']['required']:
                        req_list = rel_props['target']['required']
                        rel_props['target']['required'] = ['llmGenNodeId' if r == 'id' else r for r in req_list]
        
        return schema
    
    @staticmethod
    def get_memory_graph_schema():
        """
        Returns a fixed memory graph schema from structured outputs with:
        1. llmGenNodeId added to all nodes
        2. Memory node type removed
        3. Relationship enums updated to exclude Memory
        4. All properties made nullable to prevent LLM hallucination
        """
        from models.structured_outputs import MemoryGraphSchema
        base_schema = MemoryGraphSchema.get_fixed_json_schema()
        
        # First, transform the system schema (add llmGenNodeId, remove Memory, etc.)
        transformed_schema = MemoryGraph._transform_system_schema(base_schema)
        
        # Then, make all properties nullable to prevent LLM hallucination
        return MemoryGraph._make_schema_nullable(transformed_schema)
    
    @staticmethod
    def get_node_schema():
        """
        Returns a fixed node schema from structured outputs with nullable properties.
        """
        from models.structured_outputs import LLMGraphNode
        base_schema = LLMGraphNode.get_fixed_json_schema()
        # Make all properties nullable to prevent LLM hallucination
        return MemoryGraph._make_schema_nullable(base_schema)
    
    @staticmethod
    def get_relationship_schema():
        """
        Returns a fixed memory graph schema from structured outputs.
        """
        from models.structured_outputs import LLMGraphRelationship
        return LLMGraphRelationship.get_fixed_json_schema()

    @staticmethod
    def get_memory_only_schema():
        """
        Returns a fixed memory graph schema from structured outputs.
        """
        from models.structured_outputs import MemoryGraphSchema
        return MemoryGraphSchema.get_memory_only_schema()
    
    @staticmethod
    def get_custom_schema_for_structured_output(custom_node_labels: List[str], custom_relationship_types: List[str], user_schemas: Optional[List] = None):
        """
        Generate JSON schema for structured output with custom node types and relationship types.
        This replaces the generic system schema with user-defined custom types.
        
        Args:
            custom_node_labels: List of custom node type names
            custom_relationship_types: List of custom relationship type names  
            user_schemas: List of UserGraphSchema objects containing full property definitions
        
        Note: Memory nodes are handled separately and should not be generated by the LLM.
        """
        # Use only custom node labels - Memory nodes are created separately
        all_node_labels = custom_node_labels
        
        logger.info(f"🚀 CUSTOM SCHEMA: Generating structured output schema with nodes: {all_node_labels}")
        logger.info(f"🚀 CUSTOM SCHEMA: Generating structured output schema with relationships: {custom_relationship_types}")
        
        # Build the anyOf structure for nodes (similar to LLMGraphNode.get_fixed_json_schema)
        node_schemas = []
        
        # Note: Memory nodes are not included in LLM generation - they are created separately
        
        # Create a mapping of node labels to their full definitions
        node_definitions = {}
        if user_schemas:
            for schema in user_schemas:
                for node_name, node_type in schema.node_types.items():
                    node_definitions[node_name] = node_type
        
        # Custom node schemas with full property definitions including enums
        for node_label in custom_node_labels:
            # Skip None or empty labels - they cause schema validation errors
            if not node_label or node_label is None:
                logger.warning(f"Skipping invalid node_label: {node_label}")
                continue
            
            # Start with basic properties - make them nullable to prevent hallucination
            # Using union type syntax ["string", "null"] as per Azure/OpenAI documentation
            # IMPORTANT: llmGenNodeId is NOT nullable - it's required for relationships
            properties_schema = {
                "llmGenNodeId": {
                    "type": "string",
                    "description": "A unique identifier you generate for this node (e.g., 'task_1', 'project_main', 'user_john'). Required for creating relationships. Must be unique within this response."
                },
                "id": {
                    "type": ["string", "null"],
                    "description": "Unique identifier for this node from the content (e.g., ticket number, user ID). Use null if not available in the content."
                },
                "name": {
                    "type": ["string", "null"],
                    "description": "Name or title of this entity. Use null if not available in the content."
                },
                "description": {
                    "type": ["string", "null"],
                    "description": "Description of this entity. Use null if not available in the content."
                }
            }
            required_props = ["llmGenNodeId", "id", "name", "description"]
            
            # Track node type description for schema-level documentation
            node_type_description = None
            
            # Add custom properties from schema definition if available
            if node_label in node_definitions:
                node_def = node_definitions[node_label]
                node_type_description = node_def.description if hasattr(node_def, 'description') and node_def.description else None
                logger.info(f"🔧 SCHEMA: Adding custom properties for {node_label}")
                
                for prop_name, prop_def in node_def.properties.items():
                    # Convert PropertyDefinition to JSON schema format with nullable support
                    # Using union type syntax ["type", "null"] as per Azure/OpenAI documentation
                    base_type = "string"  # Default type
                    
                    # Set type based on PropertyType
                    if hasattr(prop_def, 'type'):
                        prop_type = prop_def.type.value if hasattr(prop_def.type, 'value') else str(prop_def.type)
                        if prop_type == "integer":
                            base_type = "integer"
                        elif prop_type == "float":
                            base_type = "number"
                        elif prop_type == "boolean":
                            base_type = "boolean"
                        elif prop_type == "array":
                            # For arrays, make them nullable using union type
                            prop_schema = {
                                "type": ["array", "null"],
                                "items": {"type": "string"}
                            }
                            base_type = None  # Already set prop_schema
                    
                    # Create nullable schema for non-array types using union type
                    if base_type:
                        prop_schema = {
                            "type": [base_type, "null"]
                        }
                    
                    # Add description with LLM-friendly guidance
                    base_description = ""
                    if hasattr(prop_def, 'description') and prop_def.description:
                        base_description = prop_def.description
                    
                    # Add enum values if specified (enums need special handling for nullable)
                    # IMPORTANT: OpenAI doesn't allow None in string enum arrays - must use union type ["string", "null"]
                    # This allows null values (preventing hallucination) while respecting enum constraints
                    # Note: Null is intentional and acceptable - property_overrides can fill them later, or they remain null (OK)
                    if hasattr(prop_def, 'enum_values') and prop_def.enum_values:
                        # Filter out None from enum list (None cannot be in enum array per OpenAI validation)
                        enum_list = [v for v in prop_def.enum_values if v is not None]
                        if not enum_list:
                            # All enum values were None, skip enum constraint and use nullable string
                            # This allows the property to be null (anti-hallucination behavior)
                            prop_schema = {"type": [base_type, "null"]}
                            if base_description:
                                prop_schema["description"] = f"{base_description}. Use null if not available in the content."
                            else:
                                prop_schema["description"] = "Use null if not available in the content."
                        else:
                            # Use union type to allow null while keeping enum constraint
                            # Union type ["string", "null"] allows null without putting None in enum array
                            # This fixes OpenAI validation error while preserving anti-hallucination behavior
                            prop_schema = {
                                "type": [base_type, "null"],
                                "enum": enum_list  # Enum array contains only valid values (no None)
                            }
                            # Enhance description with enum guidance
                            enum_desc = f"Must be one of: {', '.join(str(v) for v in enum_list)} or null if not available"
                            if base_description:
                                prop_schema["description"] = f"{base_description} ({enum_desc})"
                            else:
                                prop_schema["description"] = enum_desc
                        logger.info(f"🔧 ENUM: Added enum values for {node_label}.{prop_name}: {enum_list} (nullable via union type - null is acceptable)")
                    else:
                        # Add standard description for non-enum properties
                        if base_description:
                            prop_schema["description"] = f"{base_description}. Use null if not available in the content."
                        else:
                            prop_schema["description"] = "Use null if not available in the content."
                    
                    # Add validation constraints (apply to non-null values)
                    # With union types, constraints apply to the first type in the array
                    if base_type and base_type in ["integer", "number"]:
                        if hasattr(prop_def, 'min_value') and prop_def.min_value is not None:
                            prop_schema["minimum"] = prop_def.min_value
                        if hasattr(prop_def, 'max_value') and prop_def.max_value is not None:
                            prop_schema["maximum"] = prop_def.max_value
                    
                    if base_type == "string":
                        if hasattr(prop_def, 'min_length') and prop_def.min_length is not None:
                            prop_schema["minLength"] = prop_def.min_length
                        if hasattr(prop_def, 'max_length') and prop_def.max_length is not None:
                            prop_schema["maxLength"] = prop_def.max_length
                    
                    properties_schema[prop_name] = prop_schema
                
                # Update required properties - preserve default required properties
                if hasattr(node_def, 'required_properties') and node_def.required_properties:
                    # Ensure basic properties are still included along with custom required properties
                    required_props = list(set(["id", "name", "description"] + node_def.required_properties))
                    logger.info(f"🔧 REQUIRED: {node_label} required properties: {required_props}")
                # If no custom required properties, keep the default ones
                # (required_props is already set to ["id", "name", "description"] above)
                
                # ANTI-HALLUCINATION FIX: For OpenAI structured output compliance, ALL properties in the schema 
                # must be included in the required array. However, we make them nullable to prevent hallucination.
                # The LLM can now use null when it doesn't have the information, and we filter nulls downstream.
                all_property_names = list(properties_schema.keys())
                missing_required = [prop for prop in all_property_names if prop not in required_props]
                if missing_required:
                    required_props.extend(missing_required)
                    logger.info(f"🔧 OPENAI COMPLIANCE: Added missing required properties for {node_label}: {missing_required}")
                    logger.info(f"🚫 ANTI-HALLUCINATION: All properties are nullable to prevent LLM from fabricating data")
                
                logger.info(f"🔧 FINAL REQUIRED: {node_label} final required properties: {required_props}")
            
            # Build node schema with optional description
            node_schema_description = f"Node type: {node_label}"
            if node_type_description:
                node_schema_description += f" - {node_type_description}"
            
            custom_node_schema = {
                "type": "object",
                "description": node_schema_description,
                "properties": {
                    # Label property: Must be a valid node label string (cannot be null)
                    # OpenAI doesn't allow None in string enum arrays, so we ensure it's always a valid string
                    # Note: If node_label is None/empty, we skip this schema entirely (see check above)
                    "label": {"type": "string", "enum": [str(node_label)]},
                    "properties": {
                        "type": "object",
                        "properties": properties_schema,
                        "required": required_props,
                        "additionalProperties": False  # OpenAI structured output requires this to be false
                    }
                },
                "required": ["label", "properties"],
                "additionalProperties": False
            }
            node_schemas.append(custom_node_schema)
        
        # Build the relationships schema with detailed descriptions and constraints
        # Create relationship type definitions with source/target constraints
        relationship_details = {}  # Maps rel_name to full details
        if user_schemas:
            for schema in user_schemas:
                for rel_name, rel_type in schema.relationship_types.items():
                    details = {
                        'description': rel_type.description if hasattr(rel_type, 'description') else None,
                        'allowed_source_types': rel_type.allowed_source_types if hasattr(rel_type, 'allowed_source_types') else [],
                        'allowed_target_types': rel_type.allowed_target_types if hasattr(rel_type, 'allowed_target_types') else []
                    }
                    relationship_details[rel_name] = details
        
        # Build relationship schema using type+id pattern with enum constraints
        # Each relationship type gets its own schema with specific source/target type constraints
        relationship_schemas = []
        
        for rel_name in custom_relationship_types:
            # Build description for this specific relationship type
            rel_description = f"{rel_name}"
            if rel_name in relationship_details:
                details = relationship_details[rel_name]
                if details['description']:
                    rel_description = f"{rel_name}: {details['description']}"
                
                # Create schema with consistent type+llmGenNodeId pattern for both source and target
                rel_schema = {
                    "type": "object",
                    "description": rel_description,
                    "properties": {
                        "source": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": details['allowed_source_types']
                                },
                                "llmGenNodeId": {
                                    "type": "string",
                                    "description": "llmGenNodeId of the source node"
                                }
                            },
                            "required": ["type", "llmGenNodeId"],
                            "additionalProperties": False
                        },
                        "target": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": details['allowed_target_types']
                                },
                                "llmGenNodeId": {
                                    "type": "string",
                                    "description": "llmGenNodeId of the target node"
                                }
                            },
                            "required": ["type", "llmGenNodeId"],
                            "additionalProperties": False
                        },
                        "type": {
                            "type": "string",
                            "enum": [rel_name],
                            "description": rel_description
                        }
                    },
                    "required": ["source", "target", "type"],
                    "additionalProperties": False
                }
                relationship_schemas.append(rel_schema)
            else:
                # No detailed constraints available, use basic type+llmGenNodeId schema
                rel_schema = {
                    "type": "object",
                    "description": rel_name,
                    "properties": {
                        "source": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "llmGenNodeId": {"type": "string", "description": "llmGenNodeId of the source node"}
                            },
                            "required": ["type", "llmGenNodeId"],
                            "additionalProperties": False
                        },
                        "target": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "llmGenNodeId": {"type": "string", "description": "llmGenNodeId of the target node"}
                            },
                            "required": ["type", "llmGenNodeId"],
                            "additionalProperties": False
                        },
                        "type": {
                            "type": "string",
                            "enum": [rel_name]
                        }
                    },
                    "required": ["source", "target", "type"],
                    "additionalProperties": False
                }
                relationship_schemas.append(rel_schema)
        
        # Use anyOf to allow any of the defined relationship types
        relationships_schema = {
            "type": "array",
            "description": "List of relationships connecting the identified nodes. Only create relationships that are explicitly stated or strongly implied in the content. Each relationship must match one of the allowed relationship types with their specific source and target node type constraints.",
            "items": {
                "anyOf": relationship_schemas
            }
        }
        
        # Validate and clean up node schemas - remove any None values from label enums
        # IMPORTANT: OpenAI doesn't allow None in string enum arrays - this causes validation errors
        # We filter None from enum arrays but still allow null values via union types for other properties
        # This fixes the OpenAI error while preserving anti-hallucination (null is acceptable, None in enum is not)
        cleaned_node_schemas = []
        for node_schema in node_schemas:
            if not node_schema:
                continue
            
            # Check if label enum has None values (label enum should never have None - it identifies node type)
            if 'properties' in node_schema and 'label' in node_schema['properties']:
                label_prop = node_schema['properties']['label']
                if 'enum' in label_prop:
                    # Filter out None values from enum (None cannot be in enum array per OpenAI)
                    original_enum = label_prop['enum']
                    cleaned_enum = [v for v in original_enum if v is not None and (isinstance(v, str) and v.strip())]
                    
                    if not cleaned_enum:
                        logger.warning(f"Skipping node schema with empty label enum: {node_schema}")
                        continue
                    
                    if len(cleaned_enum) != len(original_enum):
                        logger.debug(f"Filtered None values from label enum (OpenAI validation fix): {original_enum} -> {cleaned_enum}")
                        label_prop['enum'] = cleaned_enum
            
            cleaned_node_schemas.append(node_schema)
        
        # Ensure we have at least one node schema
        if not cleaned_node_schemas:
            logger.error("No valid node schemas after cleaning - cannot create schema")
            raise ValueError("No valid node labels provided - all were None or empty")
        
        # Build the complete schema
        schema = {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "anyOf": cleaned_node_schemas
                    }
                },
                "relationships": relationships_schema
            },
            "required": ["nodes", "relationships"],
            "additionalProperties": False
        }
        
        return schema


    async def batch_add_memory_items_async(
        self,
        memory_items: List[MemoryItem],
        relationships_json_list: List[List[RelationshipItem]],
        sessionToken: str,
        user_id: str,
        background_tasks: BackgroundTasks,
        neo_session: AsyncSession,
        add_to_pinecone: bool = True,
        workspace_id: Optional[str] = None,
        skip_background_processing: bool = False,
        user_workspace_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        developer_user_object_id: Optional[str] = None,
        legacy_route: bool = True,
        developer_user_id: Optional[str] = None,
        graph_override: Optional[Dict[str, Any]] = None,
        schema_id: Optional[str] = None,
        property_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[ParseStoredMemory]:
        """
        Batch version of add_memory_item_async - processes multiple memories in a single database transaction.

        Args:
            memory_items: List of MemoryItem objects to process
            relationships_json_list: List of relationship lists, one per memory
            sessionToken: Authentication token
            user_id: User ID
            background_tasks: FastAPI background tasks
            neo_session: Neo4j async session (shared for entire batch)
            add_to_pinecone: Whether to add to vector database
            workspace_id: Optional workspace ID
            skip_background_processing: If True, process synchronously
            user_workspace_ids: Optional workspace IDs
            api_key: Optional API key
            developer_user_object_id: Optional developer user object ID
            legacy_route: Whether this is a legacy route
            developer_user_id: Developer user ID for schema selection
            graph_override: Optional graph override for manual mode
            schema_id: Optional schema ID for enforcement
            property_overrides: Optional property overrides

        Returns:
            List of ParseStoredMemory objects
        """
        logger.info(f"📦 batch_add_memory_items_async: Processing {len(memory_items)} memories")
        
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, but still using batch processing for Parse/Qdrant")

        try:
            # Store memory items in instance dictionary
            for memory_item in memory_items:
                if memory_item and hasattr(memory_item, 'id'):
                    self.memory_items[memory_item.id] = memory_item
            
            # If no workspace_id provided, try to get it from selected workspace follower
            if not workspace_id:
                async with AsyncClient() as client:
                    workspace_id = await User.get_selected_workspace_id_async(user_id, sessionToken, api_key=api_key)
                    if workspace_id:
                        logger.info(f"Using selected workspace ID: {workspace_id}")
                        for memory_item in memory_items:
                            memory_item.metadata['workspace_id'] = workspace_id
                    else:
                        logger.warning("No workspace_id provided and no selected workspace found")

            # Use the batch processing method
            stored_memories, _ = await self.batch_add_memory_items_without_relationships(
                session_token=sessionToken,
                memory_items=memory_items,
                neo_session=neo_session,
                user_id=user_id,
                user_workspace_ids=user_workspace_ids,
                api_key=api_key,
                developer_user_object_id=developer_user_object_id
            )
            
            # Handle relationships and background processing for each memory
            if not skip_background_processing:
                for i, (memory_item, relationships_json, stored_memory) in enumerate(
                    zip(memory_items, relationships_json_list, stored_memories)
                ):
                    # Schedule background processing for this memory
                    background_tasks.add_task(
                        self.process_relationships_background,
                        memory_item.id,
                        relationships_json,
                        sessionToken,
                        user_id,
                        workspace_id,
                        legacy_route,
                        developer_user_id,
                        graph_override,
                        schema_id,
                        property_overrides
                    )

            logger.info(f"✅ batch_add_memory_items_async: Completed {len(stored_memories)} memories")
            return stored_memories

        except Exception as e:
            logger.error(f"❌ Error in batch_add_memory_items_async: {e}", exc_info=True)
            raise  # Re-raise the exception instead of falling back to individual processing

    async def add_memory_item_async(
        self,
        memory_item: MemoryItem,
        relationships_json: List[RelationshipItem],
        sessionToken: str,
        user_id: str,
        background_tasks: BackgroundTasks,
        neo_session: AsyncSession,
        add_to_pinecone: bool = True,
        workspace_id: Optional[str] = None,
        skip_background_processing: bool = False,
        user_workspace_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        developer_user_object_id: Optional[str] = None,
        legacy_route: bool = True,
        developer_user_id: Optional[str] = None,
        graph_override: Optional[Dict[str, Any]] = None,
        schema_id: Optional[str] = None,
        property_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[ParseStoredMemory]:  # Updated return type
        """
        Async version of add_memory_item that quickly stores the memory and processes relationships in the background.
        Accepts an optional neo_session for robust session management.
        
        Args:
            schema_id: Optional custom schema ID to enforce during graph generation
        """
        logger.info(f"🔍 DEBUG: add_memory_item_async received graph_override: {graph_override}")
        logger.info(f"🔍 DEBUG: add_memory_item_async graph_override type: {type(graph_override)}")
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, skipping add_memory_item_async")
            return []


        try:
            # Store memory item in instance dictionary
            if memory_item and hasattr(memory_item, 'id'):
                self.memory_items[memory_item.id] = memory_item
            
            # If no workspace_id provided, try to get it from memory metadata first
            if not workspace_id:
                # First, check if workspace_id is already in memory metadata
                workspace_id = memory_item.metadata.get('workspace_id') if memory_item and hasattr(memory_item, 'metadata') else None
                
                if workspace_id:
                    logger.info(f"Using workspace_id from memory metadata: {workspace_id}")
                else:
                    # Only fetch from Parse as last resort if absolutely necessary
                    async with AsyncClient() as client:
                        workspace_id = await User.get_selected_workspace_id_async(user_id, sessionToken, api_key=api_key)
                        if workspace_id:
                            logger.info(f"Using selected workspace ID from Parse: {workspace_id}")
                            memory_item.metadata['workspace_id'] = workspace_id
                        else:
                            logger.warning("No workspace_id provided and no selected workspace found")

            # Initialize variables for the return values
            added_item_properties: ParseStoredMemory = None
            memory_item_obj: MemoryItem = None

            if add_to_pinecone:
                # Add memory item without relationships first and wait for the result
                added_item_properties, memory_list = await self.add_memory_item_without_relationships(
                    sessionToken, memory_item, neo_session, user_id, user_workspace_ids, api_key=api_key, developer_user_object_id=developer_user_object_id
                )
                memory_item_list: List[MemoryItem] = memory_list
                logger.info(f'memory_item_list: {memory_item_list}')
                added_item_properties: List[ParseStoredMemory] = added_item_properties
                logger.info(f'added_item_properties: {added_item_properties}')

                if added_item_properties and memory_item_list:
                    # Since we're now dealing with single items, get the first item
                    added_item = added_item_properties[0]  # This is a ParseStoredMemory object
                    logger.info(f'Added item properties memoryId: {added_item.memoryId}')
                    memory_item_obj = memory_item_list[0]
                    logger.info(f'Memory item obj id: {memory_item_obj.id}')
                    
                    # Update the memory_item using proper attribute access
                    memory_item_obj.objectId = added_item.objectId
                    memory_item_obj.createdAt = added_item.createdAt
                    
                    # Get memoryChunkIds safely
                    try:
                        # ParseStoredMemory has memoryChunkIds as a direct field
                        logger.info(f'ParseStoredMemory object: {added_item}')
                        logger.info(f'ParseStoredMemory memoryChunkIds field: {added_item.memoryChunkIds}')
                        logger.info(f'ParseStoredMemory memoryChunkIds type: {type(added_item.memoryChunkIds)}')
                        logger.info(f'ParseStoredMemory hasattr memoryChunkIds: {hasattr(added_item, "memoryChunkIds")}')
                        logger.info(f'ParseStoredMemory dir: {[attr for attr in dir(added_item) if not attr.startswith("_")]}')
                        
                        # Try to get memoryChunkIds directly from the ParseStoredMemory object
                        chunk_ids = added_item.memoryChunkIds if added_item.memoryChunkIds else []
                        logger.info(f'Raw memoryChunkIds from ParseStoredMemory: {added_item.memoryChunkIds}')
                        
                        # If direct access didn't work, try to get from metadata as fallback
                        if not chunk_ids:
                            logger.info('Direct access failed, trying metadata fallback')
                            metadata = added_item.metadata
                            if isinstance(metadata, str):
                                try:
                                    import json
                                    metadata = json.loads(metadata)
                                    logger.info(f'Parsed metadata from string: {metadata}')
                                except json.JSONDecodeError:
                                    logger.error('Failed to parse metadata string')
                                    metadata = {}
                            
                            # Get chunk IDs from metadata or use empty list as fallback
                            chunk_ids = metadata.get('memoryChunkIds', []) if isinstance(metadata, dict) else []
                            logger.info(f'Chunk IDs from metadata fallback: {chunk_ids}')
                        
                        memory_item_obj.memoryChunkIds = chunk_ids
                        logger.info(f'Updated memoryChunkIds: {chunk_ids}')
                    except Exception as e:
                        logger.error(f'Error processing memoryChunkIds: {e}')
                        logger.error(f'Error details: {type(e).__name__}: {str(e)}')
                        memory_item_obj.memoryChunkIds = []
                    
                    logger.info(f'Memory item obj: {memory_item_obj}')

                    # Convert memory_item to a fully serializable dictionary
                    memory_item_dict = memory_item_to_dict(memory_item_obj)
                    
                    # Add schema_id to memory_dict if provided (for graph generation enforcement)
                    if schema_id:
                        logger.info(f"✅ Adding schema_id to memory_dict: {schema_id}")
                        # Add schema_id to the top level for easy access by graph generation
                        memory_item_dict['schema_id'] = schema_id
                        # Also add to metadata.customMetadata for backward compatibility
                        if 'metadata' not in memory_item_dict:
                            memory_item_dict['metadata'] = {}
                        if 'customMetadata' not in memory_item_dict['metadata']:
                            memory_item_dict['metadata']['customMetadata'] = {}
                        memory_item_dict['metadata']['customMetadata']['schema_id'] = schema_id
                    else:
                        logger.info("ℹ️  No schema_id provided for memory_dict")
                    
                    # Add property_overrides to memory_dict if provided (for node customization)
                    if property_overrides:
                        # Convert Pydantic models to dicts for JSON serialization
                        if isinstance(property_overrides, list) and property_overrides and hasattr(property_overrides[0], 'model_dump'):
                            property_overrides_dict = [rule.model_dump() for rule in property_overrides]
                            logger.info(f"✅ Converted Pydantic PropertyOverrideRule objects to dicts")
                        else:
                            property_overrides_dict = property_overrides
                        
                        logger.info(f"✅ Adding property_overrides to memory_dict: {property_overrides_dict}")
                        # Add property_overrides to the top level for easy access by graph generation
                        memory_item_dict['property_overrides'] = property_overrides_dict
                        # Note: Don't add to customMetadata as it causes Pydantic validation errors
                    else:
                        logger.info("ℹ️  No property_overrides provided for memory_dict")
                    
                    logger.info(f'Converted memory_item to dict: {memory_item_dict}')

                    logger.info(f'skip_background_processing: {skip_background_processing}')
                                    
                    # Skip background processing tasks when skip_background_processing is True
                    if not skip_background_processing:
                        logger.info(f'Adding monitored background task for processing')
                        # Add monitored background task for processing
                        
                        # Generate batch ID for tracking if not provided
                        batch_id = getattr(memory_item, 'batch_id', str(uuid.uuid4()))
                        
                        # Add monitored background task for memory processing
                        await _add_monitored_memory_task(
                            background_tasks=background_tasks,
                            task_func=self.process_memory_item_async,
                            batch_id=batch_id,
                            task_name="process_memory_item",
                            session_token=sessionToken,
                            memory_dict=memory_item_dict,
                            relationships_json=relationships_json,
                            workspace_id=workspace_id,
                            user_id=user_id,
                            schema_id=schema_id,  # Pass schema_id to processing
                            api_key=api_key,
                            neo_session=None,
                            legacy_route=legacy_route,
                            graph_override=graph_override,  # Pass graph_override to background processing
                            property_overrides=property_overrides,  # Pass property_overrides for node customization
                            developer_user_id=developer_user_id  # Pass developer_user_id for schema selection
                        )

                        # Process relationships in background
                        if memory_item_obj.context and len(memory_item_obj.context) > 0:
                            logger.info(f'Context for memory item exists: {memory_item_obj.context}')
                            await _add_monitored_memory_task(
                                background_tasks=background_tasks,
                                task_func=self.update_memory_item_with_relationships,
                                batch_id=batch_id,
                                task_name="update_relationships",
                                memory_item_obj=memory_item_obj,
                                relationships_json=relationships_json,
                                workspace_id=workspace_id,
                                user_id=user_id,
                                neo_session=None,  # Let it open its own session
                                legacy_route=legacy_route
                            )
                        
                        
                else:
                    logger.info(f'No added item properties or memory item list')
                return added_item_properties

            return []  # Return empty list if not adding to pinecone
                # --- END OF UNCHANGED LOGIC ---
        except Exception as e:
            logger.error(f"Error in add_memory_item_async: {e}")
            self.async_neo_conn.fallback_mode = True
            return []

    async def check_neo4j_health(self):
        """Check Neo4j connection health"""
        try:
            await self.ensure_async_connection()
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, skipping health check")
                return False
            driver = await self.async_neo_conn.get_driver()
            async with driver.session() as session:
                start = time.time()
                result = await session.run("RETURN 1")
                await result.consume()
                latency = time.time() - start         
                logger.info(f"Neo4j health check successful. Latency: {latency:.2f}s")
                # Reset fallback mode if health check is successful
                if self.async_neo_conn.fallback_mode:
                    logger.info("Resetting fallback mode as connection is healthy")
                    self.async_neo_conn.fallback_mode = False
            return True
        except Exception as e:
            logger.error(f"Neo4j health check failed: {str(e)}")
            self.async_neo_conn.fallback_mode = True
            return False

    async def _find_related_memory_items_with_own_session(
        self,
        session_token: str,
        query: str,
        user_id: str,
        chat_gpt: "ChatGPTCompletion",
        metadata,
        skip_neo: bool = True,
        exclude_memory_id: str = None,
        user_workspace_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        top_k: int = 20,
        reranking_config: Optional[RerankingConfig] = None, 
        legacy_route: bool = True
    ):
        await self.ensure_async_connection()
        async with self.async_neo_conn.get_session() as task_session:
            return await self.find_related_memory_items_async(
                session_token=session_token,
                query=query,
                user_id=user_id,
                chat_gpt=chat_gpt,
                neo_session=task_session,
                metadata=metadata,
                relation_type=None,
                project_id=None,
                skip_neo=skip_neo,
                exclude_memory_id=exclude_memory_id,
                user_workspace_ids=user_workspace_ids,
                api_key=api_key,
                top_k=top_k,
                reranking_config=reranking_config,
                legacy_route=legacy_route
            )

    async def find_related_memories(
        self, 
        session_token: str, 
        memory_graph, 
        memory_item: Dict[str, Any],
        queries: List[str], 
        user_id: str, 
        chat_gpt: "ChatGPTCompletion", 
        metadata, 
        neo_session: AsyncSession,  # Kept for compatibility, but not used in parallel
        skip_neo: bool = True, 
        exclude_memory_id: str = None,
        user_workspace_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        top_k: int = 20,
        reranking_config: Optional[RerankingConfig] = None,
        legacy_route: bool = True
    ) -> Tuple[List[ParseStoredMemory], List[float]]:

        input_memory_id = memory_item.get('memoryId') or memory_item.get('id') or memory_item.get('objectId')
        if not input_memory_id:
            logger.warning(f"Could not find memory ID in input memory item: {memory_item}")

        # Execute all queries in parallel, each with its own session
        query_results = await asyncio.gather(*[
            self._find_related_memory_items_with_own_session(
                session_token=session_token,
                query=query,
                user_id=user_id,
                chat_gpt=chat_gpt,
                metadata=metadata,
                skip_neo=skip_neo,
                exclude_memory_id=exclude_memory_id,
                user_workspace_ids=user_workspace_ids,
                api_key=api_key,
                top_k=top_k,
                reranking_config=reranking_config,
                legacy_route=legacy_route
            )
            for query in queries
        ])
        
        # Extract memory_items and confidence_scores from each RelatedMemoryResult
        all_related_memories: List[ParseStoredMemory] = []
        all_confidence_scores: List[float] = []
        
        for idx, result in enumerate(query_results):
            memory_items = result.memory_items
            confidence_scores = result.confidence_scores if hasattr(result, 'confidence_scores') else [0.8] * len(memory_items)
            logger.info(f'Query {idx + 1} returned {len(memory_items)} items')
            
            # Filter and get top item
            filtered_memories = [
                item for item in memory_items 
                if input_memory_id and item.memoryId != input_memory_id
            ]
            
            if filtered_memories:
                all_related_memories.append(filtered_memories[0])
                # Get corresponding confidence score
                if confidence_scores and len(confidence_scores) > 0:
                    all_confidence_scores.append(confidence_scores[0])
                else:
                    all_confidence_scores.append(0.8)  # Default confidence

        # Remove duplicates while preserving confidence scores
        seen_ids: Set[str] = set()
        seen_contents: Set[str] = set()
        unique_memories: List[ParseStoredMemory] = []
        unique_confidence_scores: List[float] = []
        
        for memory, confidence in zip(all_related_memories, all_confidence_scores):
            if not (memory.memoryId in seen_ids or memory.content in seen_contents):
                seen_ids.add(memory.memoryId)
                seen_contents.add(memory.content)
                unique_memories.append(memory)
                unique_confidence_scores.append(confidence)

        logger.info(f'Found {len(all_related_memories)} total memories, {len(unique_memories)} unique memories')
        return unique_memories, unique_confidence_scores
    
    async def update_qdrant(self, chunk_id: str, new_metadata: dict, embedding: Optional[list] = None) -> bool:
        # Use get_qdrant_point helper to check if point exists and get qdrant_id
        # The fallback logic for trying chunk_id_0 is now handled inside get_qdrant_point
        point, qdrant_id = await self.get_qdrant_point(chunk_id)

        if not point:
            logger.error(f"No Qdrant vector found for chunk_id {chunk_id}")
            return False
        
        # Extract custom metadata and ensure indexes exist for new fields
        custom_metadata = new_metadata.get('customMetadata', {})
        if custom_metadata and isinstance(custom_metadata, dict):
            await self.ensure_custom_metadata_indexes(custom_metadata)
        
        try:
            if embedding is not None:
                await asyncio.wait_for(
                    self.qdrant_client.upsert(
                        collection_name=self.qdrant_collection,
                        points=[{
                            "id": qdrant_id,
                            "vector": embedding,
                            "payload": new_metadata
                        }],
                        wait=True
                    ),
                    timeout=15.0  # 15 second timeout
                )
                logger.info(f"Upserted Qdrant vector {qdrant_id} (from chunk_id {chunk_id}) with new embedding and metadata")
            else:
                await asyncio.wait_for(
                    self.qdrant_client.set_payload(
                        collection_name=self.qdrant_collection,
                        points=[qdrant_id],
                        payload=new_metadata
                    ),
                    timeout=10.0  # 10 second timeout for metadata-only updates
                )
                logger.info(f"Set payload for Qdrant vector {qdrant_id} (from chunk_id {chunk_id}) (metadata only)")
        except asyncio.TimeoutError:
            logger.error(f"Qdrant update operation timed out for chunk_id {chunk_id}")
            return False
        except Exception as e:
            logger.error(f"Qdrant update operation failed for chunk_id {chunk_id}: {e}")
            return False
        return True

    def make_qdrant_id(self, chunk_id: str) -> str:
        # Convert chunk_id to Qdrant-compatible ID
        # For UUID-based IDs, we need to create a valid UUID by using a hash of the chunk_id
        # For simple IDs, we can use a hash to create a valid UUID
        import hashlib
        import uuid
        
        # Create a deterministic UUID from the chunk_id using SHA-256 hash
        # This ensures the same chunk_id always produces the same Qdrant UUID
        hash_object = hashlib.sha256(chunk_id.encode())
        hash_hex = hash_object.hexdigest()
        
        # Create a UUID from the hash (using first 16 bytes)
        # This ensures we get a valid UUID format that Qdrant accepts
        uuid_bytes = bytes.fromhex(hash_hex[:32])
        qdrant_uuid = str(uuid.UUID(bytes=uuid_bytes))
        
        return qdrant_uuid
    
    async def add_qdrant_point(self, chunk_id: str, embedding: list, payload: dict, max_retries: int = 3):
        qdrant_id = self.make_qdrant_id(chunk_id)
        
        for attempt in range(max_retries):
            try:
                # Increase timeout for batch operations and add retry logic
                timeout = 30.0 if attempt == 0 else 45.0  # Longer timeout on retries
                result = await asyncio.wait_for(
                    self.qdrant_client.upsert(
                        collection_name=self.qdrant_collection,
                        points=[{
                            "id": qdrant_id,
                            "vector": embedding,
                            "payload": payload
                        }],
                        wait=True
                    ),
                    timeout=timeout
                )
                logger.info(f"Upserted Qdrant point with id {qdrant_id} (from chunk_id {chunk_id}) on attempt {attempt + 1}")
                return result
                
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    retry_delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Qdrant upsert timeout for chunk_id {chunk_id} (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Qdrant upsert timed out for chunk_id {chunk_id} after {max_retries} attempts")
                    return None
                    
            except Exception as e:
                # Check if it's a connection-related error that should be retried
                error_str = str(e).lower()
                is_connection_error = any(keyword in error_str for keyword in [
                    'timeout', 'connect', 'connection', 'network', 'unreachable', 
                    'responsehandlingexception', 'httperror', 'connectionerror'
                ])
                
                if is_connection_error and attempt < max_retries - 1:
                    retry_delay = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Qdrant connection error for chunk_id {chunk_id} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Qdrant upsert failed for chunk_id {chunk_id} (attempt {attempt + 1}/{max_retries}): {e}")
                    return None
        
        return None
    
    async def batch_upsert_qdrant_points(
        self,
        chunks_data: List[Tuple[str, list, dict]],  # List of (chunk_id, embedding, payload)
        max_retries: int = 3
    ):
        """
        Batch upsert multiple points to Qdrant in a single operation.
        
        Args:
            chunks_data: List of tuples containing (chunk_id, embedding, payload)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Qdrant upsert result or None on failure
        """
        if not chunks_data:
            logger.warning("No chunks provided for batch upsert")
            return None
        
        # Prepare all points for batch upsert
        points = []
        for chunk_id, embedding, payload in chunks_data:
            qdrant_id = self.make_qdrant_id(chunk_id)
            points.append({
                "id": qdrant_id,
                "vector": embedding,
                "payload": payload
            })
        
        logger.info(f"Batch upserting {len(points)} points to Qdrant collection {self.qdrant_collection}")
        
        for attempt in range(max_retries):
            try:
                # Increase timeout for batch operations - scale with batch size
                base_timeout = 30.0
                timeout = base_timeout * (1 + len(points) / 100) if attempt == 0 else base_timeout * 2
                
                result = await asyncio.wait_for(
                    self.qdrant_client.upsert(
                        collection_name=self.qdrant_collection,
                        points=points,
                        wait=True
                    ),
                    timeout=timeout
                )
                logger.info(f"Successfully batch upserted {len(points)} points to Qdrant on attempt {attempt + 1}")
                return result
                
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    retry_delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Qdrant batch upsert timeout for {len(points)} points (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Qdrant batch upsert timed out for {len(points)} points after {max_retries} attempts")
                    raise
                    
            except Exception as e:
                # Check if it's a connection-related error that should be retried
                error_str = str(e).lower()
                is_connection_error = any(keyword in error_str for keyword in [
                    'timeout', 'connect', 'connection', 'network', 'unreachable', 
                    'responsehandlingexception', 'httperror', 'connectionerror'
                ])
                
                if is_connection_error and attempt < max_retries - 1:
                    retry_delay = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Qdrant batch upsert connection error for {len(points)} points (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Qdrant batch upsert failed for {len(points)} points (attempt {attempt + 1}/{max_retries}): {e}")
                    raise
        
        raise Exception(f"Qdrant batch upsert failed after {max_retries} attempts")

    async def get_qdrant_point(self, chunk_id: str):
        qdrant_id = self.make_qdrant_id(chunk_id)
        try:
            result = await asyncio.wait_for(
                self.qdrant_client.retrieve(
                    collection_name=self.qdrant_collection,
                    ids=[qdrant_id],
                    with_vectors=True
                ),
                timeout=10.0  # 10 second timeout for retrieval
            )
            if result and result[0]:
                logger.info(f"Retrieved Qdrant point with id {qdrant_id} (from chunk_id {chunk_id})")
                return result[0], qdrant_id

            # Fallback: if not found and chunk_id doesn't end with _0 or _chunk_, try appending _0
            # This handles legacy single-chunk memories that were stored without suffix
            if not chunk_id.endswith('_0') and '_chunk_' not in chunk_id and '_grouped_' not in chunk_id:
                fallback_chunk_id = f"{chunk_id}_0"
                logger.info(f"Chunk ID {chunk_id} not found, trying fallback: {fallback_chunk_id}")
                fallback_qdrant_id = self.make_qdrant_id(fallback_chunk_id)

                try:
                    fallback_result = await asyncio.wait_for(
                        self.qdrant_client.retrieve(
                            collection_name=self.qdrant_collection,
                            ids=[fallback_qdrant_id],
                            with_vectors=True
                        ),
                        timeout=10.0
                    )
                    if fallback_result and fallback_result[0]:
                        logger.info(f"Found Qdrant point using fallback chunk_id: {fallback_chunk_id}")
                        return fallback_result[0], fallback_qdrant_id
                except Exception as fallback_error:
                    logger.debug(f"Fallback retrieval also failed for {fallback_chunk_id}: {fallback_error}")

            logger.warning(f"No Qdrant point found for id {qdrant_id} (from chunk_id {chunk_id})")
            return None, qdrant_id
        except asyncio.TimeoutError:
            logger.error(f"Qdrant retrieve timed out for chunk_id {chunk_id}")
            return None, qdrant_id
        except Exception as e:
            logger.error(f"Qdrant retrieve failed for chunk_id {chunk_id}: {e}")
            return None, qdrant_id

    async def update_vector_store(
        self,
        chunk_id: str,
        new_metadata: dict,
        embedding: Optional[List[float]] = None,
        legacy_route: bool = True
    ) -> bool:
        """
        Update vector metadata in the appropriate vector store (Pinecone or Qdrant) based on legacy_route flag.
        For legacy_route=True, tries both Pinecone and Qdrant in parallel.
        For legacy_route=False, only updates Qdrant.
        
        Args:
            chunk_id (str): The chunk ID to update
            new_metadata (dict): The new metadata
            embedding (Optional[List[float]]): The embedding vector (only needed for full updates, not metadata-only)
            legacy_route (bool): If True, try both Pinecone and Qdrant. If False, only Qdrant.
            
        Returns:
            bool: True if update was successful in at least one store
        """
        try:
            if legacy_route:
                # For legacy users, try both Pinecone and Qdrant in parallel
                logger.info(f"Legacy route: updating both Pinecone and Qdrant for chunk_id: {chunk_id}")
                
                async def _update_pinecone_internal():
                    try:
                        # In Pinecone, the chunk_id is the actual vector ID
                        # Try to fetch the vector directly by ID
                        fetch_result = await asyncio.to_thread(self.index.fetch, ids=[chunk_id])
                        # FetchResponse has a 'vectors' attribute (not a dict method)
                        if fetch_result and hasattr(fetch_result, 'vectors') and chunk_id in fetch_result.vectors:
                            # Found in Pinecone, update it
                            success = await self.update_pinecone(
                                vector_id=chunk_id,
                                new_metadata=new_metadata,
                                embedding=embedding
                            )
                            if success:
                                logger.info(f"Successfully updated Pinecone vector {chunk_id}")
                                return True
                            else:
                                logger.error(f"Failed to update Pinecone metadata for vector {chunk_id}")
                                return False
                        else:
                            logger.info(f"Vector ID {chunk_id} not found in Pinecone, skipping Pinecone update")
                            return False
                    except Exception as e:
                        logger.error(f"Error updating Pinecone for chunk_id {chunk_id}: {e}")
                        return False
                
                async def _update_qdrant_internal():
                    try:
                        # update_qdrant handles ID conversion and retrieval internally
                        success = await self.update_qdrant(
                            chunk_id=chunk_id,
                            new_metadata=new_metadata,
                            embedding=embedding
                        )
                        if success:
                            logger.info(f"Successfully updated Qdrant vector for chunk_id {chunk_id}")
                            return True
                        else:
                            logger.error(f"Failed to update Qdrant metadata for chunk_id {chunk_id}")
                            return False
                    except Exception as e:
                        logger.error(f"Error updating Qdrant for chunk_id {chunk_id}: {e}")
                        return False
                
                # Run both updates in parallel
                pinecone_result, qdrant_result = await asyncio.gather(
                    _update_pinecone_internal(),
                    _update_qdrant_internal(),
                    return_exceptions=True
                )
                
                # Handle exceptions
                if isinstance(pinecone_result, Exception):
                    logger.error(f"Pinecone update failed with exception: {pinecone_result}")
                    pinecone_result = False
                if isinstance(qdrant_result, Exception):
                    logger.error(f"Qdrant update failed with exception: {qdrant_result}")
                    qdrant_result = False
                
                # Return True if at least one update succeeded
                success = pinecone_result or qdrant_result
                logger.info(f"Vector store update results - Pinecone: {pinecone_result}, Qdrant: {qdrant_result}, Overall: {success}")
                return success
                
            else:
                # For non-legacy users, only update Qdrant
                logger.info(f"Non-legacy route: updating only Qdrant for chunk_id: {chunk_id}")
                
                # update_qdrant handles ID conversion and retrieval internally
                success = await self.update_qdrant(
                    chunk_id=chunk_id,
                    new_metadata=new_metadata,
                    embedding=embedding
                )
                logger.info(f"Qdrant update result for chunk_id {chunk_id}: {success}")
                return success
                    
        except Exception as e:
            logger.error(f"Error in update_vector_store for chunk_id {chunk_id}: {e}")
            logger.error("Full traceback:", exc_info=True)
            return False

    async def update_qdrant_metadata(self, chunk_id: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Update only the metadata of a Qdrant vector using chunk_id, without changing the embedding.
        
        Args:
            chunk_id (str): The chunk ID to update (e.g., 'baseid_01')
            new_metadata (Dict[str, Any]): The new metadata to set
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Use get_qdrant_point helper to check if point exists and get qdrant_id
            point, qdrant_id = await self.get_qdrant_point(chunk_id)
            if not point:
                logger.error(f"No Qdrant vector found for chunk_id {chunk_id}")
                return False
            
            # Extract custom metadata and ensure indexes exist for new fields
            custom_metadata = new_metadata.get('customMetadata', {})
            if custom_metadata and isinstance(custom_metadata, dict):
                await self.ensure_custom_metadata_indexes(custom_metadata)
            
            await self.qdrant_client.set_payload(
                collection_name=self.qdrant_collection,
                points=[qdrant_id],
                payload=new_metadata
            )
            logger.info(f"Successfully updated metadata for Qdrant vector {qdrant_id} (from chunk_id {chunk_id})")
            return True
        except Exception as e:
            logger.error(f"Error updating metadata for Qdrant vector with chunk_id {chunk_id}: {e}")
            return False

    async def delete_qdrant_point(self, chunk_id: str) -> bool:
        """
        Delete a Qdrant vector using chunk_id.
        
        Args:
            chunk_id (str): The chunk ID to delete (e.g., 'baseid_01')
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Use get_qdrant_point helper to check if point exists and get qdrant_id
            point, qdrant_id = await self.get_qdrant_point(chunk_id)
            logger.info(f"qdrant_id: {qdrant_id}")
            if not point:
                logger.warning(f"No Qdrant vector found for chunk_id {chunk_id}")
                return False
            
            # Use the correct Qdrant delete API - pass points directly as a list
            await self.qdrant_client.delete(
                collection_name=self.qdrant_collection,
                points_selector=[qdrant_id]
            )
            logger.info(f"Successfully deleted Qdrant vector {qdrant_id} (from chunk_id {chunk_id})")
            return True
        except Exception as e:
            logger.error(f"Error deleting Qdrant vector with chunk_id {chunk_id}: {e}")
            return False

    async def delete_qdrant_points_parallel(self, chunk_ids: List[str], max_concurrent: int = 10) -> Dict[str, bool]:
        """
        Delete multiple Qdrant vectors in parallel using a semaphore to limit concurrency.
        
        Args:
            chunk_ids (List[str]): List of chunk IDs to delete
            max_concurrent (int): Maximum number of concurrent delete operations
            
        Returns:
            Dict[str, bool]: Mapping of chunk_id to success status
        """
        if not chunk_ids:
            return {}
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}
        
        async def delete_single_chunk(chunk_id: str) -> Tuple[str, bool]:
            async with semaphore:
                success = await self.delete_qdrant_point(chunk_id)
                return chunk_id, success
        
        # Run all deletions in parallel
        tasks = [delete_single_chunk(chunk_id) for chunk_id in chunk_ids]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in chunk_results:
            if isinstance(result, Exception):
                logger.error(f"Exception during parallel delete: {result}")
                continue
            chunk_id, success = result
            results[chunk_id] = success
        
        # Log summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Parallel delete completed: {successful}/{total} chunks deleted successfully")
        
        return results

    async def delete_pinecone_point(self, vector_id: str) -> bool:
        """
        Delete a Pinecone vector using vector_id.
        
        Args:
            vector_id (str): The vector ID to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        # Use safe Pinecone operation for delete
        result = await self._safe_pinecone_operation(
            "delete",
            self.index.delete,
            ids=[vector_id]
        )
        
        if result is None:
            logger.warning(f"Pinecone delete failed for vector {vector_id}")
            return False
            
        logger.info(f"Successfully deleted Pinecone vector {vector_id}")
        return True

    async def delete_pinecone_points_parallel(self, vector_ids: List[str], max_concurrent: int = 10) -> Dict[str, bool]:
        """
        Delete multiple Pinecone vectors in parallel using a semaphore to limit concurrency.
        
        Args:
            vector_ids (List[str]): List of vector IDs to delete
            max_concurrent (int): Maximum number of concurrent delete operations
            
        Returns:
            Dict[str, bool]: Mapping of vector_id to success status
        """
        if not vector_ids:
            return {}
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}
        
        async def delete_single_vector(vector_id: str) -> Tuple[str, bool]:
            async with semaphore:
                success = await self.delete_pinecone_point(vector_id)
                return vector_id, success
        
        # Run all deletions in parallel
        tasks = [delete_single_vector(vector_id) for vector_id in vector_ids]
        vector_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in vector_results:
            if isinstance(result, Exception):
                logger.error(f"Exception during parallel Pinecone delete: {result}")
                continue
            vector_id, success = result
            results[vector_id] = success
        
        # Log summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Parallel Pinecone delete completed: {successful}/{total} vectors deleted successfully")
        
        return results

    async def check_and_retrieve_from_qdrant(
        self,
        session_token: str,
        embedding: list[float],
        user_id: str,
        neo_session: AsyncSession,
        new_metadata: dict,
        user_workspace_ids: Optional[list[str]] = None,
        api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Asynchronously checks for a similar embedding in Qdrant and updates metadata if a match is found.

        Args:
            session_token (str): The session token for authentication.
            embedding (List[float]): The embedding vector to query.
            user_id (str): The user ID associated with the embedding.
            neo_session (AsyncSession): The Neo4j session to use for the query.
            new_metadata (Dict[str, Any]): The metadata to update in Qdrant if a match is found.

        Returns:
            Optional[str]: The ID of the matched vector if found, else None.
        """
        from qdrant_client.http import models as qmodels
        try:
            # Ensure the embedding is a list of floats
            if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
                logger.error(f"Invalid embedding format: {embedding}")
                return None

            logger.info(f"Embedding to be queried (first 5 elements): {embedding[:5]}")
            logger.info(f"Embedding length: {len(embedding)})")
            logger.info(f"new_metadata: {new_metadata}")

            # Get user roles and workspace IDs
            user_instance = User.get(user_id)
            if user_workspace_ids is None:
                user_roles, user_workspace_ids = await asyncio.gather(
                    user_instance.get_roles_async(),
                    User.get_workspaces_for_user_async(user_id)
                )
            else:
                user_roles = await user_instance.get_roles_async()

            logger.debug(f'user_roles {user_roles}')
            logger.debug(f'user_workspace_ids {user_workspace_ids}')
            
            # Get organization and namespace info (if available) - same as Pinecone retrieval
            user_organization_id = getattr(user_instance, 'organization_id', None)
            user_namespace_id = getattr(user_instance, 'namespace_id', None)
            user_organization_access = getattr(user_instance, 'organization_read_access', [])
            user_namespace_access = getattr(user_instance, 'namespace_read_access', [])
            
            logger.debug(f'user_organization_id {user_organization_id}')
            logger.debug(f'user_namespace_id {user_namespace_id}')
            logger.debug(f'user_organization_access {user_organization_access}')
            logger.debug(f'user_namespace_access {user_namespace_access}')
            '''
            # Build Qdrant filter (must use Qdrant's filter format)
            must_conditions = [
                qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=str(user_id))),
                qmodels.FieldCondition(key="user_read_access", match=qmodels.MatchAny(any=[str(user_id)])),
            ]
            if user_workspace_ids:
                must_conditions.append(
                    qmodels.FieldCondition(key="workspace_read_access", match=qmodels.MatchAny(any=[str(wid) for wid in user_workspace_ids]))
                )
            if user_roles:
                must_conditions.append(
                    qmodels.FieldCondition(key="role_read_access", match=qmodels.MatchAny(any=user_roles))
                )
            qdrant_filter = qmodels.Filter(must=must_conditions)
            '''
            # MULTI-TENANT ISOLATION: Enforce namespace_id and organization_id as MUST conditions
            # Build MUST conditions for tenant isolation
            must_conditions = []
            
            # CRITICAL: namespace_id is REQUIRED for multi-tenant isolation (cloud-only)
            # In open-source, namespace_id may be None - this is expected and not a security issue
            namespace_id = new_metadata.get('namespace_id')
            organization_id = new_metadata.get('organization_id')
            
            # Check if we're in cloud edition (multi-tenant features are cloud-only)
            from config.features import get_features
            features = get_features()
            is_cloud = features.is_cloud
            
            if namespace_id:
                must_conditions.append(
                    qmodels.FieldCondition(key="namespace_id", match=qmodels.MatchValue(value=namespace_id))
                )
                logger.info(f"🔒 TENANT FILTER: Added namespace_id={namespace_id} as MUST condition")
            elif is_cloud:
                # Only warn in cloud edition - in open-source, missing namespace_id is expected
                logger.warning(f"⚠️ TENANT FILTER: No namespace_id in metadata - potential cross-tenant leak!")
            
            if organization_id:
                must_conditions.append(
                    qmodels.FieldCondition(key="organization_id", match=qmodels.MatchValue(value=organization_id))
                )
                logger.info(f"🔒 TENANT FILTER: Added organization_id={organization_id} as MUST condition")
            
            # ACL conditions: user has access if ANY condition matches (OR logic)
            acl_conditions = []
            if user_id and isinstance(user_id, (str, int, bool)):
                acl_conditions.append(
                    qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))
                )
                acl_conditions.append(
                    qmodels.FieldCondition(key="user_read_access", match=qmodels.MatchAny(any=[user_id]))
                )
            if user_workspace_ids:
                acl_conditions.append(
                    qmodels.FieldCondition(
                        key="workspace_read_access",
                        match=qmodels.MatchAny(any=[wid for wid in user_workspace_ids if isinstance(wid, (str, int, bool))])
                    )
                )
            # Add organization conditions if available (only access arrays, not direct IDs)
            if user_organization_access:
                acl_conditions.append(
                    qmodels.FieldCondition(
                        key="organization_read_access",
                        match=qmodels.MatchAny(any=[oid for oid in user_organization_access if isinstance(oid, (str, int, bool))])
                    )
                )
            # Add namespace conditions if available (only access arrays, not direct IDs)
            if user_namespace_access:
                acl_conditions.append(
                    qmodels.FieldCondition(
                        key="namespace_read_access",
                        match=qmodels.MatchAny(any=[nid for nid in user_namespace_access if isinstance(nid, (str, int, bool))])
                    )
                )
            if user_roles:
                acl_conditions.append(
                    qmodels.FieldCondition(
                        key="role_read_access",
                        match=qmodels.MatchAny(any=[role for role in user_roles if isinstance(role, (str, int, bool))])
                    )
                )
            
            # CRITICAL FIX: Use MUST + SHOULD for proper tenant isolation
            # Filter structure: (namespace_id AND organization_id) AND (user has access)
            if must_conditions and acl_conditions:
                qdrant_filter = qmodels.Filter(
                    must=must_conditions,
                    should=acl_conditions
                )
            elif must_conditions:
                # Only tenant conditions (shouldn't happen - ACL should always exist)
                logger.warning("⚠️ FILTER: Only tenant conditions present, no ACL!")
                qdrant_filter = qmodels.Filter(must=must_conditions)
            elif acl_conditions:
                # In open-source, ACL-only filtering is expected (no multi-tenant)
                # In cloud, missing tenant isolation is a security issue
                if is_cloud:
                    logger.error("❌ CRITICAL: No tenant isolation conditions - falling back to ACL only!")
                else:
                    logger.debug("Open-source edition: Using ACL-only filtering (multi-tenant not available)")
                qdrant_filter = qmodels.Filter(should=acl_conditions)
            else:
                qdrant_filter = None
            # Perform similarity search in Qdrant with resilient error handling
            search_result = None  # Initialize to avoid variable scoping issues
            try:
                search_result = await asyncio.wait_for(
                    self._qdrant_search_async(
                        collection_name=self.qdrant_collection,
                        query_vector=embedding,
                        query_filter=qdrant_filter,
                        limit=1,
                        with_payload=True,
                        with_vectors=False
                    ),
                    timeout=10.0  # 10 second timeout
                )
            except asyncio.TimeoutError:
                logger.error("Qdrant search timed out in check_and_retrieve_from_qdrant")
                return None
            except Exception as e:
                logger.error(f"Error searching Qdrant: {e}", exc_info=True)
                return None

            logger.info(f"Qdrant search result: {search_result}")
            if not search_result or len(search_result) == 0:
                logger.info("No matches found in Qdrant")
                return None

            match = search_result[0]
            score = match.score
            matched_id = str(match.id)
            logger.info(f"Found match with ID: {matched_id} and similarity score: {score}")

            if score > 0.97:
                logger.info(f"Score {score} > 0.97, using existing vector")
                qdrant_metadata = MemoryGraph.pinecone_compatible_metadata(new_metadata)
                parse_metadata = new_metadata
                await self.update_memory_metadata(session_token, matched_id, neo_session, qdrant_metadata, parse_metadata, api_key=api_key)
                return matched_id
            else:
                logger.info(f"Score {score} <= 0.97, will create new vector")
                return None

        except Exception as e:
            logger.error(f"Unexpected error during Qdrant query: {str(e)}")
            return None

    async def cleanup(self):
        """Cleanup method to properly close client connections"""
        # Close Qdrant client if available
        if hasattr(self, 'qdrant_client') and self.qdrant_client:
            try:
                if hasattr(self.qdrant_client, 'close'):
                    await self.qdrant_client.close()
                    logger.info("Qdrant client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Qdrant client: {e}")
        
        # Close MongoDB client ONLY if it's not the shared singleton
        # The shared singleton should persist for the application lifetime
        if hasattr(self, 'mongo_client') and self.mongo_client:
            try:
                # Check if this is the shared singleton client
                shared_db = get_mongo_db()
                is_shared_client = (shared_db is not None and
                                  self.mongo_client == shared_db.client)

                if not is_shared_client:
                    # Only close if it's a client we created ourselves
                    self.mongo_client.close()
                    logger.info("MongoDB client closed successfully (non-shared client)")
                else:
                    logger.debug("Skipping MongoDB client close (shared singleton - managed externally)")
            except Exception as e:
                logger.warning(f"Error closing MongoDB client: {e}")

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection"""
        # Note: Can't use async in __del__, so this is just for logging
        if hasattr(self, 'qdrant_client') and self.qdrant_client:
            logger.warning("MemoryGraph instance destroyed without proper cleanup. Consider calling cleanup() explicitly.")



def filter_metadata_for_pinecone(metadata: dict) -> dict:
    """Return a dict with only Pinecone-compatible values: str, int, float, bool, or list of str."""
    def is_valid_value(v):
        if isinstance(v, (str, int, float, bool)):
            return True
        if isinstance(v, list) and all(isinstance(i, str) for i in v):
            return True
        return False
    return {k: v for k, v in metadata.items() if is_valid_value(v)}

def clean_metadata_for_parse(metadata: dict) -> dict:
    """Remove None values and unexpected keys from metadata before sending to Parse."""
    allowed_keys = {
        # List all allowed keys for your Parse metadata schema
        "createdAt", "user_id", "external_user_id", "external_user_read_access",
        "external_user_write_access", "user_read_access", "user_write_access",
        "workspace_read_access", "workspace_write_access", "role_read_access",
        "role_write_access", "workspace_id", "memoryChunkIds", "pageId", "sourceType",
        "sourceUrl", "topics", "emojiTags", "emotionTags", "conversationId"
        # ...add any other allowed keys
    }
    return {k: v for k, v in metadata.items() if v is not None and k in allowed_keys}
