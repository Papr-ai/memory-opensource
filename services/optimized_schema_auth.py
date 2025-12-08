#!/usr/bin/env python3
"""
Optimized authentication that fetches user schemas in parallel with user info.
This eliminates the extra Parse call during search operations.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import httpx
from os import environ as env

from models.user_schemas import UserGraphSchema, SchemaStatus
from services.logging_config import get_logger

logger = get_logger(__name__)

class OptimizedSchemaAuth:
    """Enhanced authentication that includes schema fetching for search optimization"""
    
    def __init__(self):
        self.parse_server_url = env.get("PARSE_SERVER_URL")
        self.parse_app_id = env.get("PARSE_APPLICATION_ID")
        self.parse_master_key = env.get("PARSE_MASTER_KEY")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get Parse Server headers with master key"""
        return {
            "X-Parse-Application-Id": self.parse_app_id,
            "X-Parse-Master-Key": self.parse_master_key,
            "Content-Type": "application/json"
        }
    
    async def get_user_info_with_schemas(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None,
        httpx_client: Optional[httpx.AsyncClient] = None
    ) -> Tuple[Dict[str, Any], List[UserGraphSchema]]:
        """
        Fetch user info and schemas in parallel for optimal search performance.
        
        Returns:
            Tuple of (user_info, user_schemas)
        """
        start_time = time.time()
        
        try:
            headers = self._get_headers()
            
            # Prepare parallel tasks
            tasks = []
            
            # Task 1: Get user info with includes
            user_url = f"{self.parse_server_url}/parse/classes/_User/{user_id}"
            user_params = {
                "include": "isSelectedWorkspaceFollower,isSelectedWorkspaceFollower.workspace",
                "keys": "objectId,username,email,isQwenRoute,isSelectedWorkspaceFollower"
            }
            
            if httpx_client:
                user_task = httpx_client.get(user_url, headers=headers, params=user_params)
            else:
                # Create client for this operation
                user_task = self._create_user_task(user_url, headers, user_params)
            
            tasks.append(user_task)
            
            # Task 2: Get user schemas - use multi-tenant query logic
            schemas_url = f"{self.parse_server_url}/parse/classes/UserGraphSchema"
            
            # Build multi-tenant query conditions (same as SchemaService.list_schemas)
            query_conditions = []
            
            # 1. Personal schemas (always accessible to the user)
            query_conditions.append({
                "scope": "personal",
                "user_id": {"__type": "Pointer", "className": "_User", "objectId": user_id}
            })
            
            # 2. Namespace-scoped schemas (if namespace_id available)
            if namespace_id:
                query_conditions.append({
                    "scope": "namespace",
                    "namespace": {"__type": "Pointer", "className": "Namespace", "objectId": namespace_id}
                })
            
            # 3. Organization-scoped schemas (if organization_id available)
            if organization_id:
                query_conditions.append({
                    "scope": "organization",
                    "organization": {"__type": "Pointer", "className": "Organization", "objectId": organization_id}
                })
            
            # 4. Legacy workspace-scoped schemas (backward compatibility)
            if workspace_id:
                query_conditions.append({
                    "scope": "workspace",
                    "workspace_id": {"__type": "Pointer", "className": "WorkSpace", "objectId": workspace_id}
                })
            
            # 5. Global schemas removed from API access (admin-only)
            
            # 6. Explicitly shared schemas (via read_access)
            query_conditions.append({"read_access": {"$in": [user_id]}})
            
            # 7. Legacy fallback for organization schemas without proper pointers
            query_conditions.append({"scope": "organization", "organization": {"$exists": False}})
            
            schemas_where = {"$or": query_conditions}
            schemas_params = {
                "where": json.dumps(schemas_where),
                "order": "-updatedAt",
                "limit": 50  # Reasonable limit for active schemas
            }
            
            if httpx_client:
                schemas_task = httpx_client.get(schemas_url, headers=headers, params=schemas_params)
            else:
                schemas_task = self._create_schemas_task(schemas_url, headers, schemas_params)
            
            tasks.append(schemas_task)
            
            # Execute both tasks in parallel
            logger.info(f"Fetching user info and schemas in parallel for user {user_id}")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process user info result
            user_response = results[0]
            user_info = {}
            
            if isinstance(user_response, Exception):
                logger.error(f"Failed to fetch user info: {user_response}")
            elif hasattr(user_response, 'status_code') and user_response.status_code == 200:
                user_info = user_response.json()
            else:
                logger.warning(f"Unexpected user response: {user_response}")
            
            # Process schemas result
            schemas_response = results[1]
            user_schemas = []
            
            if isinstance(schemas_response, Exception):
                logger.warning(f"Failed to fetch schemas: {schemas_response}")
            elif hasattr(schemas_response, 'status_code') and schemas_response.status_code == 200:
                schemas_data = schemas_response.json()
                user_schemas = self._parse_schemas(schemas_data.get('results', []))
            else:
                logger.warning(f"Unexpected schemas response: {schemas_response}")
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Parallel user+schema fetch completed in {elapsed:.2f}ms")
            logger.info(f"Retrieved {len(user_schemas)} active schemas for user {user_id}")
            
            return user_info, user_schemas
            
        except Exception as e:
            logger.error(f"Error in parallel user+schema fetch: {e}")
            return {}, []
    
    async def _create_user_task(self, url: str, headers: Dict[str, str], params: Dict[str, str]):
        """Create user fetch task with new httpx client"""
        async with httpx.AsyncClient() as client:
            return await client.get(url, headers=headers, params=params)
    
    async def _create_schemas_task(self, url: str, headers: Dict[str, str], params: Dict[str, str]):
        """Create schemas fetch task with new httpx client"""
        async with httpx.AsyncClient() as client:
            return await client.get(url, headers=headers, params=params)
    
    def _parse_schemas(self, schemas_data: List[Dict[str, Any]]) -> List[UserGraphSchema]:
        """Parse raw schema data from Parse into UserGraphSchema objects with lenient validation"""
        parsed_schemas = []
        
        for schema_data in schemas_data:
            try:
                # Convert Parse objectId to id
                if 'objectId' in schema_data:
                    schema_data['id'] = schema_data['objectId']
                
                # Clean up node_types to fix validation issues
                if 'node_types' in schema_data and isinstance(schema_data['node_types'], dict):
                    node_types = schema_data['node_types']
                    for node_name, node_data in node_types.items():
                        if isinstance(node_data, dict):
                            # Clean up required_properties: only keep properties that exist
                            if 'required_properties' in node_data and 'properties' in node_data:
                                required_props = node_data['required_properties']
                                properties = node_data.get('properties', {})
                                if isinstance(required_props, list) and isinstance(properties, dict):
                                    valid_required = [prop for prop in required_props if prop in properties]
                                    if len(valid_required) != len(required_props):
                                        logger.warning(f"Cleaning up node type '{node_name}': removed {len(required_props) - len(valid_required)} invalid required_properties")
                                        node_data['required_properties'] = valid_required
                            
                            # Clean up unique_identifiers: only keep properties that exist
                            if 'unique_identifiers' in node_data and 'properties' in node_data:
                                unique_ids = node_data['unique_identifiers']
                                properties = node_data.get('properties', {})
                                if isinstance(unique_ids, list) and isinstance(properties, dict):
                                    valid_unique = [prop for prop in unique_ids if prop in properties]
                                    if len(valid_unique) != len(unique_ids):
                                        logger.warning(f"Cleaning up node type '{node_name}': removed {len(unique_ids) - len(valid_unique)} invalid unique_identifiers")
                                        node_data['unique_identifiers'] = valid_unique
                            
                            # Limit properties to 10 (keep first 10 if more exist)
                            if 'properties' in node_data and isinstance(node_data['properties'], dict):
                                props = node_data['properties']
                                if len(props) > 10:
                                    logger.warning(f"Node type '{node_name}' has {len(props)} properties (max 10). Keeping first 10.")
                                    limited_props = dict(list(props.items())[:10])
                                    node_data['properties'] = limited_props
                                    # Update required_properties and unique_identifiers
                                    if 'required_properties' in node_data:
                                        node_data['required_properties'] = [p for p in node_data['required_properties'] if p in limited_props]
                                    if 'unique_identifiers' in node_data:
                                        node_data['unique_identifiers'] = [p for p in node_data['unique_identifiers'] if p in limited_props]
                    
                    # Limit node_types to 10 (keep first 10 if more exist)
                    if len(node_types) > 10:
                        logger.warning(f"Schema has {len(node_types)} node types (max 10). Keeping first 10.")
                        schema_data['node_types'] = dict(list(node_types.items())[:10])
                
                # Try normal validation first
                try:
                    schema = UserGraphSchema(**schema_data)
                except Exception as e:
                    # If validation fails, use model_construct for lenient parsing
                    logger.warning(f"Schema validation failed for schema {schema_data.get('id', 'unknown')}: {e}. Using lenient parsing.")
                    schema = UserGraphSchema.model_construct(**schema_data)
                
                # Only include active schemas
                if schema.status == SchemaStatus.ACTIVE:
                    parsed_schemas.append(schema)
                    logger.debug(f"Parsed active schema: {schema.name}")
                
            except Exception as e:
                logger.warning(f"Failed to parse schema data: {e}")
                continue
        
        return parsed_schemas

# Example usage in search endpoint
async def enhanced_search_auth(
    user_id: str,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> Tuple[Dict[str, Any], List[UserGraphSchema]]:
    """
    Enhanced authentication for search that includes schema fetching.
    Use this instead of separate auth + schema calls.
    """
    auth_service = OptimizedSchemaAuth()
    return await auth_service.get_user_info_with_schemas(
        user_id=user_id,
        workspace_id=workspace_id,
        organization_id=organization_id,
        namespace_id=namespace_id,
        httpx_client=httpx_client
    )

# Performance comparison demo
async def demo_performance_comparison():
    """Demonstrate the performance difference between sequential vs parallel fetching"""
    
    print("🚀 Schema Fetching Performance Comparison")
    print("=" * 50)
    
    user_id = "example_user_123"
    workspace_id = "example_workspace_456"
    
    # Simulate sequential fetching (current approach)
    print("\n📊 Sequential Fetching (Current):")
    sequential_start = time.time()
    
    # Step 1: Auth (simulated)
    await asyncio.sleep(0.2)  # 200ms auth
    print("  ✅ User auth completed (200ms)")
    
    # Step 2: Schema fetch (simulated) 
    await asyncio.sleep(0.15)  # 150ms schema fetch
    print("  ✅ Schema fetch completed (150ms)")
    
    sequential_total = (time.time() - sequential_start) * 1000
    print(f"  📈 Total Sequential Time: {sequential_total:.0f}ms")
    
    # Simulate parallel fetching (optimized approach)
    print("\n⚡ Parallel Fetching (Optimized):")
    parallel_start = time.time()
    
    # Both operations in parallel
    await asyncio.gather(
        asyncio.sleep(0.2),   # Auth
        asyncio.sleep(0.15)   # Schema fetch
    )
    print("  ✅ User auth + schema fetch completed in parallel")
    
    parallel_total = (time.time() - parallel_start) * 1000
    print(f"  📈 Total Parallel Time: {parallel_total:.0f}ms")
    
    # Calculate improvement
    improvement = ((sequential_total - parallel_total) / sequential_total) * 100
    print(f"\n🎯 Performance Improvement: {improvement:.1f}% faster!")
    print(f"   Sequential: {sequential_total:.0f}ms")
    print(f"   Parallel:   {parallel_total:.0f}ms")
    print(f"   Saved:      {sequential_total - parallel_total:.0f}ms per search")

if __name__ == "__main__":
    asyncio.run(demo_performance_comparison())


