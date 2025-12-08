"""
API Operation Tracking Service

Handles dual tracking of API operations:
1. Monthly aggregates in Parse Server (Interaction class)
2. Detailed logs in MongoDB Time Series (APIOperationLog)
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from services.logger_singleton import LoggerSingleton
from models.interaction_models import APIOperationLog, HTTPMethod
from os import environ as env
import httpx

logger = LoggerSingleton.get_logger(__name__)

# Parse Server config
PARSE_SERVER_URL = env.get("PARSE_SERVER_URL", "http://localhost:1337")
PARSE_APPLICATION_ID = env.get("PARSE_APPLICATION_ID")
PARSE_MASTER_KEY = env.get("PARSE_MASTER_KEY")


class APIOperationTracker:
    """
    Tracks API operations for analytics and rate limiting.
    
    Features:
    - Dual storage: Parse Server (aggregates) + MongoDB Time Series (detailed logs)
    - Async non-blocking tracking
    - Memory operation flagging for limit enforcement
    - Performance metrics
    """
    
    def __init__(self, mongo_client: Optional[AsyncIOMotorClient] = None):
        """
        Initialize tracker
        
        Args:
            mongo_client: MongoDB client (if not provided, will use global connection)
        """
        self.mongo_client = mongo_client
        self._time_series_collection = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize time series collection"""
        if self._initialized:
            return
        
        try:
            if not self.mongo_client:
                # Get from memory_graph or create new connection
                from services.utils import get_mongo_client
                self.mongo_client = await get_mongo_client()
            
            db = self.mongo_client.get_default_database()
            
            # Check if collection exists
            collections = await db.list_collection_names()
            
            if "api_operation_logs" not in collections:
                # Create time series collection
                await db.create_collection(
                    "api_operation_logs",
                    timeseries={
                        "timeField": "timestamp",
                        "metaField": "metadata",
                        "granularity": "seconds"  # Can be seconds, minutes, or hours
                    },
                    expireAfterSeconds=7776000  # 90 days retention
                )
                logger.info("✓ Created api_operation_logs time series collection")
            
            self._time_series_collection = db["api_operation_logs"]
            
            # Create indexes for common queries
            await self._time_series_collection.create_index([
                ("user_id", 1),
                ("timestamp", -1)
            ])
            await self._time_series_collection.create_index([
                ("organization_id", 1),
                ("timestamp", -1)
            ])
            await self._time_series_collection.create_index([
                ("route", 1),
                ("method", 1),
                ("timestamp", -1)
            ])
            
            self._initialized = True
            logger.info("✓ APIOperationTracker initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize APIOperationTracker: {e}")
            # Continue without time series logging if it fails
    
    async def track_operation(
        self,
        user_id: str,
        workspace_id: str,
        route: str,
        method: str,
        is_memory_operation: bool,
        organization_id: Optional[str] = None,
        developer_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        batch_size: Optional[int] = None,
        latency_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        api_key: Optional[str] = None,
        client_type: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        increment_monthly_count: bool = True
    ):
        """
        Track an API operation (non-blocking)
        
        Args:
            user_id: User making the request
            workspace_id: Workspace context
            route: API route (e.g., 'v1/memory')
            method: HTTP method (GET, POST, etc.)
            is_memory_operation: Whether this counts against memory operation limits
            organization_id: Organization (for multi-tenant)
            developer_id: API key owner (if different from user_id)
            subscription_id: Subscription context
            operation_type: Type of operation (e.g., 'add_memory', 'search')
            memory_id: Memory ID if applicable
            batch_size: Batch size if batch operation
            latency_ms: Response time in milliseconds
            status_code: HTTP status code
            api_key: API key used (will be hashed)
            client_type: Client type (e.g., 'papr_plugin')
            ip_address: Client IP
            user_agent: User agent
            metadata: Additional metadata
            increment_monthly_count: Whether to increment monthly aggregate count
        """
        # Run tracking in background to not block request
        asyncio.create_task(
            self._track_operation_impl(
                user_id=user_id,
                workspace_id=workspace_id,
                route=route,
                method=method,
                is_memory_operation=is_memory_operation,
                organization_id=organization_id,
                developer_id=developer_id,
                subscription_id=subscription_id,
                operation_type=operation_type,
                memory_id=memory_id,
                batch_size=batch_size,
                latency_ms=latency_ms,
                status_code=status_code,
                api_key=api_key,
                client_type=client_type,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=metadata,
                increment_monthly_count=increment_monthly_count
            )
        )
    
    async def _track_operation_impl(
        self,
        user_id: str,
        workspace_id: str,
        route: str,
        method: str,
        is_memory_operation: bool,
        organization_id: Optional[str] = None,
        developer_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        batch_size: Optional[int] = None,
        latency_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        api_key: Optional[str] = None,
        client_type: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        increment_monthly_count: bool = True
    ):
        """Implementation of operation tracking"""
        try:
            # Ensure initialized
            await self.initialize()
            
            # 1. Log to time series (detailed analytics)
            await self._log_to_time_series(
                user_id=user_id,
                workspace_id=workspace_id,
                route=route,
                method=method,
                is_memory_operation=is_memory_operation,
                organization_id=organization_id,
                developer_id=developer_id,
                operation_type=operation_type or self._infer_operation_type(route, method),
                memory_id=memory_id,
                batch_size=batch_size,
                latency_ms=latency_ms,
                status_code=status_code,
                api_key=self._hash_api_key(api_key) if api_key else None,
                client_type=client_type,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=metadata
            )
            
            # 2. Update monthly aggregate in Parse Server (for rate limiting)
            if increment_monthly_count:
                await self._update_monthly_aggregate(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    route=route,
                    method=method,
                    is_memory_operation=is_memory_operation,
                    organization_id=organization_id,
                    subscription_id=subscription_id
                )
            
            logger.debug(f"✓ Tracked API operation: {method} {route}")
            
        except Exception as e:
            logger.error(f"Failed to track API operation: {e}", exc_info=True)
            # Don't raise - tracking failures shouldn't break the API
    
    async def _log_to_time_series(
        self,
        user_id: str,
        workspace_id: str,
        route: str,
        method: str,
        is_memory_operation: bool,
        organization_id: Optional[str],
        developer_id: Optional[str],
        operation_type: str,
        memory_id: Optional[str],
        batch_size: Optional[int],
        latency_ms: Optional[float],
        status_code: Optional[int],
        api_key: Optional[str],
        client_type: Optional[str],
        ip_address: Optional[str],
        user_agent: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ):
        """Log detailed operation to time series collection"""
        if not self._time_series_collection:
            logger.warning("Time series collection not initialized")
            return
        
        try:
            log = APIOperationLog(
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                workspace_id=workspace_id,
                organization_id=organization_id,
                developer_id=developer_id,
                route=route,
                method=HTTPMethod(method.upper()),
                operation_type=operation_type,
                is_memory_operation=is_memory_operation,
                memory_id=memory_id,
                batch_size=batch_size,
                latency_ms=latency_ms,
                status_code=status_code,
                api_key=api_key,
                client_type=client_type,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=metadata or {}
            )
            
            await self._time_series_collection.insert_one(log.to_mongo_doc())
            
        except Exception as e:
            logger.error(f"Failed to log to time series: {e}")
    
    async def _update_monthly_aggregate(
        self,
        user_id: str,
        workspace_id: str,
        route: str,
        method: str,
        is_memory_operation: bool,
        organization_id: Optional[str],
        subscription_id: Optional[str]
    ):
        """Update monthly aggregate count in Parse Server"""
        try:
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year
            
            # Query for existing interaction
            interaction_url = f"{PARSE_SERVER_URL}/parse/classes/Interaction"
            query = {
                "where": {
                    "user": {
                        "__type": "Pointer",
                        "className": "_User",
                        "objectId": user_id
                    },
                    "workspace": {
                        "__type": "Pointer",
                        "className": "WorkSpace",
                        "objectId": workspace_id
                    },
                    "type": "api_operation",
                    "route": route,
                    "method": method,
                    "month": current_month,
                    "year": current_year
                }
            }
            
            headers = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-Master-Key": PARSE_MASTER_KEY,
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                # Check if exists
                response = await client.get(
                    interaction_url,
                    headers=headers,
                    params={"where": str(query["where"])}
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    if results:
                        # Update existing
                        interaction_id = results[0]["objectId"]
                        update_url = f"{interaction_url}/{interaction_id}"
                        update_data = {
                            "count": {
                                "__op": "Increment",
                                "amount": 1
                            }
                        }
                        await client.put(update_url, headers=headers, json=update_data)
                    else:
                        # Create new
                        new_interaction = {
                            "user": {
                                "__type": "Pointer",
                                "className": "_User",
                                "objectId": user_id
                            },
                            "workspace": {
                                "__type": "Pointer",
                                "className": "WorkSpace",
                                "objectId": workspace_id
                            },
                            "type": "api_operation",
                            "route": route,
                            "method": method,
                            "isMemoryOperation": is_memory_operation,
                            "month": current_month,
                            "year": current_year,
                            "count": 1
                        }
                        
                        if organization_id:
                            new_interaction["organization"] = {
                                "__type": "Pointer",
                                "className": "Organization",
                                "objectId": organization_id
                            }
                        
                        if subscription_id:
                            new_interaction["subscription"] = {
                                "__type": "Pointer",
                                "className": "Subscription",
                                "objectId": subscription_id
                            }
                        
                        await client.post(interaction_url, headers=headers, json=new_interaction)
        
        except Exception as e:
            logger.error(f"Failed to update monthly aggregate: {e}")
    
    def _infer_operation_type(self, route: str, method: str) -> str:
        """Infer operation type from route and method"""
        method = method.upper()
        
        if "memory" in route.lower():
            if method == "POST" and "search" in route.lower():
                return "search_memory"
            elif method == "POST" and "batch" in route.lower():
                return "add_memory_batch"
            elif method == "POST":
                return "add_memory"
            elif method == "PUT":
                return "update_memory"
            elif method == "DELETE":
                return "delete_memory"
            elif method == "GET":
                return "get_memory"
        
        elif "user" in route.lower():
            if method == "POST":
                return "create_user"
            elif method == "GET":
                return "get_user"
            elif method == "PUT":
                return "update_user"
        
        return f"{method.lower()}_{route.replace('/', '_')}"
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for privacy"""
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]


# Singleton instance
_tracker: Optional[APIOperationTracker] = None


def get_api_operation_tracker() -> APIOperationTracker:
    """Get singleton API operation tracker"""
    global _tracker
    if _tracker is None:
        _tracker = APIOperationTracker()
    return _tracker

