"""
Service for managing ActiveNodeRel cache operations with Parse server.

This service handles storing, retrieving, and updating cached Neo4j schema patterns
to optimize search performance by eliminating expensive schema discovery queries.
"""

import httpx
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from models.active_node_rel import ActiveNodeRel, NodeRelationshipPattern
from services.user_utils import get_parse_headers

logger = logging.getLogger(__name__)


class ActiveNodeRelParseService:
    """Parse server operations for ActiveNodeRel caching"""
    
    def __init__(self, parse_server_url: str, application_id: str, master_key: str):
        # Normalize base URL: ensure scheme and /parse suffix
        url = (parse_server_url or "").strip()
        if url and not (url.startswith("http://") or url.startswith("https://")):
            url = f"https://{url}"
        if url and not url.endswith('/parse'):
            url = f"{url}/parse"
        self.parse_server_url = url
        self.application_id = application_id
        self.master_key = master_key
        self.class_name = "ActiveNodeRel"
    
    async def get_cached_schema(
        self, 
        user_object_id: str, 
        workspace_object_id: Optional[str] = None,
        httpx_client: Optional[httpx.AsyncClient] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached schema patterns for a user/workspace using Parse pointers.
        
        Args:
            user_object_id: The Parse User objectId
            workspace_object_id: Optional Parse Workspace objectId
            httpx_client: Optional httpx client to reuse
            
        Returns:
            Dictionary with 'nodes', 'relationships', and 'patterns' keys, or None if not found
        """
        try:
            # Check if parse_server_url is configured
            if not self.parse_server_url:
                logger.debug("Parse server URL not configured, skipping cached schema retrieval")
                return None
            
            # Build query with Parse pointers
            where_clause = {
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_object_id
                }
            }
            
            if workspace_object_id:
                where_clause["workspace"] = {
                    "__type": "Pointer", 
                    "className": "WorkSpace",  # Note: capital 'S' to match Parse schema
                    "objectId": workspace_object_id
                }
            
            params = {
                "where": json.dumps(where_clause),
                "order": "-updatedAt",  # Get most recent first
                "limit": 1
            }
            
            headers = get_parse_headers()
            url = f"{self.parse_server_url}/classes/{self.class_name}"
            
            # Use provided client or create new one
            if httpx_client:
                response = await httpx_client.get(url, params=params, headers=headers)
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    result = results[0]
                    logger.info(f"Found cached schema for user {user_object_id}, workspace {workspace_object_id}")
                    
                    # Get active patterns (limit to top 100 for LLM efficiency)
                    active_patterns_raw = result.get("activePatterns", "[]")
                    try:
                        # Deserialize JSON string to list
                        active_patterns = json.loads(active_patterns_raw) if isinstance(active_patterns_raw, str) else active_patterns_raw
                        active_patterns = active_patterns[:100]  # Limit to top 100
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse activePatterns JSON: {e}")
                        active_patterns = []
                    
                    if active_patterns:
                        # Extract nodes and relationships from patterns
                        nodes = set()
                        relationships = set()
                        
                        for pattern in active_patterns:
                            if isinstance(pattern, dict):
                                # Handle both formats: Parse cache uses 'source'/'target'/'relationship', Neo4j discovery uses 'source_label'/'target_label'/'relationship_type'
                                source = pattern.get("source") or pattern.get("source_label")
                                target = pattern.get("target") or pattern.get("target_label")
                                rel_type = pattern.get("relationship") or pattern.get("relationship_type")
                                
                                if source:
                                    nodes.add(source)
                                if target:
                                    nodes.add(target)
                                if rel_type:
                                    relationships.add(rel_type)
                        
                        # Return in search format
                        return {
                            'nodes': sorted(list(nodes)),
                            'relationships': sorted(list(relationships)),
                            'patterns': active_patterns
                        }
                    else:
                        logger.info(f"No active patterns found in cached schema")
                        return None
                else:
                    logger.info(f"No cached schema found for user {user_object_id}, workspace {workspace_object_id}")
                    return None
            else:
                logger.warning(f"Failed to retrieve cached schema: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cached schema: {e}")
            return None
    
    async def update_cached_schema(
        self, 
        user_object_id: str, 
        workspace_object_id: Optional[str],
        neo4j_patterns: List[Dict[str, Any]],
        httpx_client: Optional[httpx.AsyncClient] = None
    ) -> bool:
        """
        Update cached schema patterns after successful Neo4j operations.
        
        Args:
            user_object_id: The Parse User objectId
            workspace_object_id: Optional Parse Workspace objectId
            neo4j_patterns: List of pattern dictionaries from Neo4j schema discovery
            httpx_client: Optional httpx client to reuse
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if parse_server_url is configured
            if not self.parse_server_url:
                logger.debug("Parse server URL not configured, skipping cached schema update")
                return False
            
            # Sort patterns by count (descending) and limit to top 100 for LLM efficiency
            sorted_patterns = sorted(neo4j_patterns, key=lambda x: x.get('count', 0), reverse=True)[:100]
            
            headers = get_parse_headers()
            
            # Convert to Parse format with pointers
            parse_data = {
                "user": {
                    "__type": "Pointer",
                    "className": "_User", 
                    "objectId": user_object_id
                },
                "activePatterns": json.dumps(sorted_patterns)  # Serialize as JSON string
            }
            
            if workspace_object_id:
                parse_data["workspace"] = {
                    "__type": "Pointer",
                    "className": "WorkSpace",  # Note: capital 'S' to match Parse schema
                    "objectId": workspace_object_id
                }
            
            # Check if record already exists
            existing_schema = await self.get_cached_schema(user_object_id, workspace_object_id, httpx_client)
            
            if existing_schema:
                # Update existing record - need to find the objectId first
                where_clause = {
                    "user": {
                        "__type": "Pointer",
                        "className": "_User",
                        "objectId": user_object_id
                    }
                }
                if workspace_object_id:
                    where_clause["workspace"] = {
                        "__type": "Pointer",
                        "className": "WorkSpace",  # Note: capital 'S' to match Parse schema 
                        "objectId": workspace_object_id
                    }
                
                params = {"where": json.dumps(where_clause), "limit": 1}
                url = f"{self.parse_server_url}/classes/{self.class_name}"
                
                # Use provided client or create new one
                if httpx_client:
                    response = await httpx_client.get(url, params=params, headers=headers)
                else:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(url, params=params, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    if results:
                        object_id = results[0]["objectId"]
                        update_url = f"{self.parse_server_url}/classes/{self.class_name}/{object_id}"
                        
                        # Use provided client or create new one
                        if httpx_client:
                            update_response = await httpx_client.put(update_url, json=parse_data, headers=headers)
                        else:
                            async with httpx.AsyncClient() as client:
                                update_response = await client.put(update_url, json=parse_data, headers=headers)
                        
                        if update_response.status_code == 200:
                            logger.info(f"Updated cached schema for user {user_object_id}, workspace {workspace_object_id} with {len(sorted_patterns)} patterns")
                            return True
                        else:
                            logger.error(f"Failed to update cached schema: {update_response.status_code} - {update_response.text}")
                            return False
            else:
                # Create new record
                url = f"{self.parse_server_url}/classes/{self.class_name}"
                
                # Use provided client or create new one
                if httpx_client:
                    response = await httpx_client.post(url, json=parse_data, headers=headers)
                else:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(url, json=parse_data, headers=headers)
                
                if response.status_code == 201:
                    logger.info(f"Created cached schema for user {user_object_id}, workspace {workspace_object_id} with {len(sorted_patterns)} patterns")
                    return True
                else:
                    logger.error(f"Failed to create cached schema: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating cached schema: {e}")
            return False
    
    async def invalidate_cached_schema(
        self, 
        user_id: str, 
        workspace_id: Optional[str] = None,
        httpx_client: Optional[httpx.AsyncClient] = None
    ) -> bool:
        """
        Invalidate/delete cached schema patterns (useful for testing or schema resets).
        
        Args:
            user_id: The user ID
            workspace_id: Optional workspace ID
            httpx_client: Optional httpx client to reuse
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build query
            where_clause = {"user_id": user_id}
            if workspace_id:
                where_clause["workspace_id"] = workspace_id
            
            params = {"where": json.dumps(where_clause)}
            headers = get_parse_headers()
            url = f"{self.parse_server_url}/classes/{self.class_name}"
            
            # Use provided client or create new one
            if httpx_client:
                response = await httpx_client.get(url, params=params, headers=headers)
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Delete all matching records
                for result in results:
                    object_id = result["objectId"]
                    delete_url = f"{self.parse_server_url}/classes/{self.class_name}/{object_id}"
                    
                    # Use provided client or create new one
                    if httpx_client:
                        delete_response = await httpx_client.delete(delete_url, headers=headers)
                    else:
                        async with httpx.AsyncClient() as client:
                            delete_response = await client.delete(delete_url, headers=headers)
                    
                    if delete_response.status_code != 200:
                        logger.warning(f"Failed to delete cached schema record {object_id}: {delete_response.status_code}")
                
                logger.info(f"Invalidated {len(results)} cached schema records for user {user_id}, workspace {workspace_id}")
                return True
            else:
                logger.warning(f"Failed to find cached schema records to delete: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error invalidating cached schema: {e}")
            return False


# Global service instance (will be initialized with environment variables)
_active_node_rel_service: Optional[ActiveNodeRelParseService] = None


def get_active_node_rel_service() -> ActiveNodeRelParseService:
    """Get the global ActiveNodeRelParseService instance"""
    global _active_node_rel_service
    if _active_node_rel_service is None:
        import os
        _active_node_rel_service = ActiveNodeRelParseService(
            parse_server_url=os.getenv("PARSE_SERVER_URL"),  # Use PARSE_SERVER_URL like other services
            application_id=os.getenv("PARSE_APPLICATION_ID"),
            master_key=os.getenv("PARSE_MASTER_KEY")
        )
    return _active_node_rel_service
