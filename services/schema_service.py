from typing import Dict, Any, List, Optional, Union
import asyncio
from models.user_schemas import UserGraphSchema, SchemaStatus, SchemaScope, SchemaResponse, SchemaListResponse
from services.logging_config import get_logger
from services.url_utils import clean_url
from datetime import datetime, timezone
import httpx
import json
from os import environ as env

logger = get_logger(__name__)

class SchemaService:
    """Service for managing user-defined schemas"""
    
    def __init__(self):
        self.parse_server_url = clean_url(env.get("PARSE_SERVER_URL"))
        self.parse_app_id = env.get("PARSE_APPLICATION_ID")
        self.parse_rest_key = env.get("PARSE_REST_API_KEY")
        self.parse_master_key = env.get("PARSE_MASTER_KEY")
        
    def _get_headers(self, use_master_key: bool = False) -> Dict[str, str]:
        """Get Parse Server headers"""
        headers = {
            "X-Parse-Application-Id": self.parse_app_id,
            "Content-Type": "application/json"
        }
        if use_master_key:
            headers["X-Parse-Master-Key"] = self.parse_master_key
        else:
            headers["X-Parse-REST-API-Key"] = self.parse_rest_key
        return headers
    
    def _parse_schema_from_response(self, data: Dict[str, Any]) -> UserGraphSchema:
        """Parse schema from Parse Server response, handling Parse date formats and cleaning up invalid data"""
        # Convert Parse date format to datetime for last_used_at
        if 'last_used_at' in data and isinstance(data['last_used_at'], dict):
            if data['last_used_at'].get('__type') == 'Date':
                from datetime import datetime
                try:
                    iso_string = data['last_used_at']['iso']
                    # Remove 'Z' and parse
                    if iso_string.endswith('Z'):
                        iso_string = iso_string[:-1] + '+00:00'
                    data['last_used_at'] = datetime.fromisoformat(iso_string)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse last_used_at date: {e}")
                    data['last_used_at'] = None
        
        # Convert Parse Server response back to UserGraphSchema object
        # Development server stores objects directly, no JSON string conversion needed
        # But handle both cases for compatibility
        if 'node_types' in data and isinstance(data['node_types'], str):
            data['node_types'] = json.loads(data['node_types'])
        if 'relationship_types' in data and isinstance(data['relationship_types'], str):
            data['relationship_types'] = json.loads(data['relationship_types'])
        
        # Convert Parse Server Pointers back to string IDs
        if 'user_id' in data and isinstance(data['user_id'], dict) and data['user_id'].get('__type') == 'Pointer':
            data['user_id'] = data['user_id']['objectId']
        if 'workspace_id' in data and isinstance(data['workspace_id'], dict) and data['workspace_id'].get('__type') == 'Pointer':
            data['workspace_id'] = data['workspace_id']['objectId']
        if 'organization' in data and isinstance(data['organization'], dict) and data['organization'].get('__type') == 'Pointer':
            data['organization_id'] = data['organization']['objectId']
            del data['organization']
        if 'namespace' in data and isinstance(data['namespace'], dict) and data['namespace'].get('__type') == 'Pointer':
            data['namespace_id'] = data['namespace']['objectId']
            del data['namespace']
        # Handle legacy string fields if present
        if 'organization' in data and isinstance(data['organization'], str):
            data['organization_id'] = data['organization']
            del data['organization']
        if 'namespace' in data and isinstance(data['namespace'], str):
            data['namespace_id'] = data['namespace']
            del data['namespace']
        
        # Map Parse Server fields
        if 'objectId' in data:
            data['id'] = data['objectId']
            del data['objectId']
        if 'createdAt' in data:
            data['created_at'] = data['createdAt']
            del data['createdAt']
        if 'updatedAt' in data:
            data['updated_at'] = data['updatedAt']
            del data['updatedAt']
        
        # Clean up node_types to fix validation issues
        if 'node_types' in data and isinstance(data['node_types'], dict):
            node_types = data['node_types']
            for node_name, node_data in node_types.items():
                if isinstance(node_data, dict):
                    # Clean up required_properties: only keep properties that exist in properties dict
                    if 'required_properties' in node_data and 'properties' in node_data:
                        required_props = node_data['required_properties']
                        properties = node_data.get('properties', {})
                        if isinstance(required_props, list) and isinstance(properties, dict):
                            # Filter out properties that don't exist
                            valid_required = [prop for prop in required_props if prop in properties]
                            if len(valid_required) != len(required_props):
                                logger.warning(f"Cleaning up node type '{node_name}': removed {len(required_props) - len(valid_required)} invalid required_properties")
                                node_data['required_properties'] = valid_required
                    
                    # Clean up unique_identifiers: only keep properties that exist in properties dict
                    if 'unique_identifiers' in node_data and 'properties' in node_data:
                        unique_ids = node_data['unique_identifiers']
                        properties = node_data.get('properties', {})
                        if isinstance(unique_ids, list) and isinstance(properties, dict):
                            # Filter out properties that don't exist
                            valid_unique = [prop for prop in unique_ids if prop in properties]
                            if len(valid_unique) != len(unique_ids):
                                logger.warning(f"Cleaning up node type '{node_name}': removed {len(unique_ids) - len(valid_unique)} invalid unique_identifiers")
                                node_data['unique_identifiers'] = valid_unique
                    
                    # Limit properties to 10 (keep first 10 if more exist)
                    if 'properties' in node_data and isinstance(node_data['properties'], dict):
                        props = node_data['properties']
                        if len(props) > 10:
                            logger.warning(f"Node type '{node_name}' has {len(props)} properties (max 10). Keeping first 10.")
                            # Keep first 10 properties
                            limited_props = dict(list(props.items())[:10])
                            node_data['properties'] = limited_props
                            # Also update required_properties and unique_identifiers
                            if 'required_properties' in node_data:
                                node_data['required_properties'] = [p for p in node_data['required_properties'] if p in limited_props]
                            if 'unique_identifiers' in node_data:
                                node_data['unique_identifiers'] = [p for p in node_data['unique_identifiers'] if p in limited_props]
            
            # Limit node_types to 10 (keep first 10 if more exist)
            if len(node_types) > 10:
                logger.warning(f"Schema has {len(node_types)} node types (max 10). Keeping first 10.")
                data['node_types'] = dict(list(node_types.items())[:10])
        
        # Use model_construct to bypass strict validation for reading existing schemas
        # This allows us to read schemas that don't meet current validation rules
        try:
            # First try normal validation
            return UserGraphSchema(**data)
        except Exception as e:
            # If validation fails, use model_construct to create the object without validation
            # This allows reading existing schemas that don't meet current rules
            logger.warning(f"Schema validation failed for schema {data.get('id', 'unknown')}: {e}. Using lenient parsing.")
            # Create object without validation, but still validate individual fields where possible
            schema = UserGraphSchema.model_construct(**data)
            return schema
    
    async def create_schema(self, schema: UserGraphSchema, user_id: str, workspace_id: Optional[str] = None, 
                           organization_id: Optional[str] = None, namespace_id: Optional[str] = None) -> SchemaResponse:
        """Create a new user-defined schema"""
        try:
            # Validate schema
            validation_result = await self._validate_schema(schema, user_id)
            if not validation_result.success:
                return validation_result
            
            # Set user_id, workspace_id, organization, namespace, and timestamps
            logger.info(f"🔍 TRACE STEP 11 - SCHEMA SERVICE START: user_id={user_id}, workspace_id={workspace_id}, org_id={organization_id}, namespace_id={namespace_id}")
            logger.info(f"🔍 TRACE STEP 12 - SCHEMA OBJECT RECEIVED: user_id={schema.user_id}, workspace_id={schema.workspace_id}")
            
            schema.user_id = user_id
            logger.info(f"🔍 TRACE STEP 13 - SET USER_ID: {user_id}")
            
            if workspace_id:
                schema.workspace_id = workspace_id
                logger.info(f"🔍 TRACE STEP 14 - SET WORKSPACE_ID: {workspace_id}")
            else:
                logger.warning(f"🔍 TRACE STEP 14 - WORKSPACE_ID IS NONE: not setting it")
            
            # Set multi-tenant context
            if organization_id:
                schema.organization_id = organization_id  # Will be converted to pointer later
                logger.info(f"🔍 TRACE STEP 14a - SET ORGANIZATION_ID: {organization_id}")
            
            if namespace_id:
                schema.namespace_id = namespace_id  # Will be converted to pointer later
                logger.info(f"🔍 TRACE STEP 14b - SET NAMESPACE_ID: {namespace_id}")
                
            logger.info(f"🔍 TRACE STEP 15 - SCHEMA AFTER SETTING: user_id={schema.user_id}, workspace_id={schema.workspace_id}, org_id={schema.organization_id}, ns_id={schema.namespace_id}")
            now = datetime.now(timezone.utc)
            schema.created_at = now
            schema.updated_at = now
            
            # Store in Parse Server
            # Convert complex nested objects for Parse Server
            schema_data = schema.model_dump(mode='json', exclude_none=True)
            try:
                schema_dump = json.dumps(schema_data, indent=2)
            except TypeError:
                schema_dump = json.dumps(schema_data, indent=2, default=str)
            logger.info(f"🔍 TRACE STEP 16 - SCHEMA MODEL_DUMP: {schema_dump}")
            
            # Development server expects Objects, not JSON strings
            # Keep node_types and relationship_types as objects

            # Convert user_id to Parse Server Pointer format
            if 'user_id' in schema_data and schema_data['user_id']:
                schema_data['user_id'] = {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": schema_data['user_id']
                }

            # Convert workspace_id to Parse Server Pointer format (legacy)
            logger.info(f"🔍 TRACE STEP 17 - BEFORE WORKSPACE CONVERT: workspace_id in schema_data = {schema_data.get('workspace_id')}")
            if 'workspace_id' in schema_data and schema_data['workspace_id']:
                logger.info(f"🔍 TRACE STEP 18 - CONVERTING WORKSPACE_ID: {schema_data['workspace_id']} to pointer")
                schema_data['workspace_id'] = {
                    "__type": "Pointer",
                    "className": "WorkSpace", 
                    "objectId": schema_data['workspace_id']
                }
                logger.info(f"🔍 TRACE STEP 19 - WORKSPACE_ID CONVERTED: {schema_data['workspace_id']}")
            else:
                logger.warning(f"🔍 TRACE STEP 18 - WORKSPACE_ID NOT FOUND: workspace_id not found or empty in schema_data: {schema_data.get('workspace_id')}")
                # Explicitly set workspace_id to null if not provided
                schema_data['workspace_id'] = None
                logger.warning(f"🔍 TRACE STEP 19 - WORKSPACE_ID SET TO NULL")
            
            # Convert organization_id to Parse Server Pointer format (multi-tenant)
            if 'organization_id' in schema_data and schema_data['organization_id']:
                if isinstance(schema_data['organization_id'], str):
                    logger.info(f"🔍 CONVERTING ORGANIZATION_ID: {schema_data['organization_id']} to pointer")
                    schema_data['organization'] = {
                        "__type": "Pointer",
                        "className": "Organization",
                        "objectId": schema_data['organization_id']
                    }
                    logger.info(f"🔍 ORGANIZATION CONVERTED: {schema_data['organization']}")
                del schema_data['organization_id']
            
            # Convert namespace_id to Parse Server Pointer format (multi-tenant)
            if 'namespace_id' in schema_data and schema_data['namespace_id']:
                if isinstance(schema_data['namespace_id'], str):
                    logger.info(f"🔍 CONVERTING NAMESPACE_ID: {schema_data['namespace_id']} to pointer")
                    schema_data['namespace'] = {
                        "__type": "Pointer",
                        "className": "Namespace",
                        "objectId": schema_data['namespace_id']
                    }
                    logger.info(f"🔍 NAMESPACE CONVERTED: {schema_data['namespace']}")
                del schema_data['namespace_id']
            
            try:
                final_schema_dump = json.dumps(schema_data, indent=2)
            except TypeError:
                final_schema_dump = json.dumps(schema_data, indent=2, default=str)
            logger.info(f"🔍 TRACE STEP 20 - FINAL SCHEMA_DATA: {final_schema_dump}")
            
            async with httpx.AsyncClient() as client:
                response = None
                for attempt in range(2):
                    try:
                        response = await client.post(
                            f"{self.parse_server_url}/parse/classes/UserGraphSchema",
                            headers=self._get_headers(use_master_key=True),
                            json=schema_data,
                            timeout=20.0
                        )
                        break
                    except httpx.RequestError as e:
                        logger.error(
                            "Schema create request failed (attempt %s/2): %s",
                            attempt + 1,
                            e,
                        )
                        if attempt == 0:
                            await asyncio.sleep(0.5)
                        else:
                            raise
                if response is None:
                    raise RuntimeError("Schema create request did not return a response")
                
                if response.status_code != 201:
                    logger.error(f"Failed to create schema: {response.text}")
                    return SchemaResponse(
                        success=False,
                        error="Failed to create schema",
                        code=response.status_code
                    )
                
                result = response.json()
                schema.id = result['objectId']  # Use Parse Server objectId
                
                logger.info(f"Created schema {schema.name} for user {user_id}")
                return SchemaResponse(success=True, data=schema, code=201)
                
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            return SchemaResponse(
                success=False,
                error=str(e),
                code=500
            )
    
    async def get_schema(self, schema_id: str, user_id: str, workspace_id: Optional[str] = None,
                        organization_id: Optional[str] = None, namespace_id: Optional[str] = None) -> SchemaResponse:
        """Get a user's schema by ID with multi-tenant context and permission checks"""
        try:
            logger.info(f"🔍 SCHEMA SERVICE: Looking up schema_id={schema_id}, user_id={user_id}, workspace_id={workspace_id}, org_id={organization_id}, namespace_id={namespace_id}")
            
            # Build the same query conditions as list_schemas but filter by specific objectId
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
            
            # Fallback for legacy queries without multi-tenant context
            if not organization_id and not namespace_id and not workspace_id:
                query_conditions.extend([
                    {"user_id": {"__type": "Pointer", "className": "_User", "objectId": user_id}},
                    {"scope": "organization"}  # Legacy: all org schemas
                ])
            
            # When looking up by specific schema_id, allow both active and draft schemas.
            # If a developer explicitly references a schema by ID, they should be able to use it.
            # Only archived schemas are excluded (they are soft-deleted).
            where_conditions = {
                "$and": [
                    {"objectId": schema_id},  # Filter by specific schema ID
                    {"status": {"$ne": "archived"}},  # Exclude archived (soft-deleted) schemas
                    {"$or": query_conditions} # Apply multi-tenant access rules
                ]
            }
            
            logger.info(f"🔍 SCHEMA SERVICE: Parse query conditions: {where_conditions}")
            
            async with httpx.AsyncClient() as client:
                url = f"{self.parse_server_url}/parse/classes/UserGraphSchema"
                params = {
                    "where": json.dumps(where_conditions),
                    "limit": 1
                }
                
                logger.info(f"🔍 SCHEMA SERVICE: Making Parse request to {url}")
                logger.info(f"🔍 SCHEMA SERVICE: Request params: {params}")
                
                response = await client.get(url, headers=self._get_headers(use_master_key=True), params=params)  # Use master key for server-side queries
                
                logger.info(f"🔍 SCHEMA SERVICE: Response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.warning(f"🔍 SCHEMA SERVICE: Parse request failed with status {response.status_code}")
                    return SchemaResponse(
                        success=False,
                        error="Failed to retrieve schema",
                        code=response.status_code
                    )
                
                data = response.json()
                logger.info(f"🔍 SCHEMA SERVICE: Parse response: {data}")
                results = data.get('results', [])
                
                logger.info(f"🔍 SCHEMA SERVICE: Found {len(results)} schemas matching query")
                if results:
                    schema_data = results[0]
                    logger.info(f"🔍 SCHEMA SERVICE: Schema found - objectId={schema_data.get('objectId')}, name={schema_data.get('name')}, scope={schema_data.get('scope')}")
                
                if not results:
                    logger.warning(f"🔍 SCHEMA SERVICE: No schema found for schema_id={schema_id}")
                    return SchemaResponse(
                        success=False,
                        error="Schema not found or access denied",
                        code=404
                    )
                
                # Parse the first (and only) result
                schema_data = results[0]
                
                # If we got results, access is already validated by the query
                schema = self._parse_schema_from_response(schema_data)
                
                return SchemaResponse(success=True, data=schema)
                
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return SchemaResponse(
                success=False,
                error=str(e),
                code=500
            )
    
    async def list_schemas(self, user_id: str, workspace_id: Optional[str] = None, 
                          organization_id: Optional[str] = None, namespace_id: Optional[str] = None) -> SchemaListResponse:
        """List all schemas accessible to user with multi-tenant context"""
        try:
            # Build query conditions based on available context
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
            
            # Fallback for legacy queries without multi-tenant context
            if not organization_id and not namespace_id and not workspace_id:
                # Legacy fallback: get user's personal schemas and any organization-scoped
                query_conditions.extend([
                    {"user_id": {"__type": "Pointer", "className": "_User", "objectId": user_id}},
                    {"scope": "organization"}  # Legacy: all org schemas (not ideal but maintains compatibility)
                ])
            
            where_conditions = {"$or": query_conditions}
            
            # Log the query for debugging
            logger.info(f"🔍 SCHEMA QUERY: Searching with conditions: {json.dumps(where_conditions, indent=2)}")
            logger.info(f"🔍 SCHEMA CONTEXT: user_id={user_id}, workspace_id={workspace_id}, org_id={organization_id}, namespace_id={namespace_id}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.parse_server_url}/parse/classes/UserGraphSchema",
                    headers=self._get_headers(use_master_key=True),  # Use master key for server-side queries
                    params={
                        "where": json.dumps(where_conditions),
                        "order": "-updatedAt",
                        "limit": 100  # Pagination can be added later
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"🔍 SCHEMA ERROR: Parse returned {response.status_code}: {response.text}")
                    return SchemaListResponse(
                        success=False,
                        error="Failed to list schemas",
                        code=response.status_code
                    )
                
                data = response.json()
                logger.info(f"🔍 SCHEMA RESPONSE: Found {len(data.get('results', []))} total schemas in Parse")
                
                # Log each schema found for debugging
                for idx, item in enumerate(data.get('results', [])):
                    logger.info(f"🔍 SCHEMA RAW {idx+1}: objectId={item.get('objectId')}, scope={item.get('scope')}, status={item.get('status')}, workspace_id={item.get('workspace_id')}")
                
                schemas = []
                for item in data['results']:
                    try:
                        schema = self._parse_schema_from_response(item)
                        schemas.append(schema)
                        logger.info(f"🔍 SCHEMA FOUND: {schema.name} (id={schema.id}, status={schema.status}, scope={item.get('scope', 'unknown')})")
                    except Exception as e:
                        logger.error(f"Error parsing schema {item.get('objectId', 'unknown')}: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        continue
                
                logger.info(f"🔍 SCHEMA RESULT: Returning {len(schemas)} schemas after parsing")
                
                return SchemaListResponse(
                    success=True,
                    data=schemas,
                    total=len(schemas)
                )
                
        except Exception as e:
            logger.error(f"Error listing schemas: {e}")
            return SchemaListResponse(
                success=False,
                error=str(e),
                code=500
            )
    
    async def update_schema(self, schema_id: str, updates: Dict[str, Any], user_id: str) -> SchemaResponse:
        """Update an existing schema"""
        try:
            # First check if user has write access
            current_schema = await self.get_schema(schema_id, user_id)
            if not current_schema.success:
                return current_schema
            
            # Check write permissions
            if (current_schema.data.user_id != user_id and 
                user_id not in current_schema.data.write_access):
                return SchemaResponse(
                    success=False,
                    error="Insufficient permissions to update schema",
                    code=403
                )
            
            # Add update timestamp
            updates['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.parse_server_url}/parse/classes/UserGraphSchema/{schema_id}",
                    headers=self._get_headers(use_master_key=True),
                    json=updates
                )
                
                if response.status_code != 200:
                    return SchemaResponse(
                        success=False,
                        error="Failed to update schema",
                        code=response.status_code
                    )
                
                # Return updated schema
                return await self.get_schema(schema_id, user_id)
                
        except Exception as e:
            logger.error(f"Error updating schema: {e}")
            return SchemaResponse(
                success=False,
                error=str(e),
                code=500
            )
    
    async def update_schema_usage(self, schema_id: str, user_id: str) -> bool:
        """
        Update the usage count and last_used_at timestamp for a schema.
        This tracks which schemas are being used for analytics and optimization.
        
        Args:
            schema_id: The ID of the schema that was used
            user_id: The ID of the user who used the schema
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            from datetime import datetime
            
            # Prepare the update data
            update_data = {
                "usage_count": {"__op": "Increment", "amount": 1},  # Increment usage count
                "last_used_at": {"__type": "Date", "iso": datetime.utcnow().isoformat() + "Z"}  # Update last used timestamp
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.put(
                    f"{self.parse_server_url}/parse/classes/UserGraphSchema/{schema_id}",
                    headers=self._get_headers(use_master_key=True),
                    json=update_data
                )
                
                if response.status_code == 200:
                    logger.info(f"📊 Schema usage updated: {schema_id} (user: {user_id})")
                    return True
                else:
                    logger.warning(f"📊 Failed to update schema usage: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"📊 Error updating schema usage: {e}")
            # Don't raise - usage tracking failure shouldn't break memory addition
            return False
    
    async def delete_schema(self, schema_id: str, user_id: str) -> SchemaResponse:
        """Soft delete a schema"""
        return await self.update_schema(
            schema_id, 
            {"status": SchemaStatus.ARCHIVED.value}, 
            user_id
        )
    
    async def get_active_schemas(self, user_id: str, workspace_id: Optional[str] = None,
                                organization_id: Optional[str] = None, namespace_id: Optional[str] = None) -> List[UserGraphSchema]:
        """Get all active schemas for a user with multi-tenant context"""
        result = await self.list_schemas(user_id, workspace_id, organization_id, namespace_id)
        if result.success:
            logger.info(f"🔍 ACTIVE FILTER: Filtering {len(result.data)} schemas for ACTIVE status")
            active_schemas = [schema for schema in result.data if schema.status == SchemaStatus.ACTIVE]
            logger.info(f"🔍 ACTIVE FILTER: Found {len(active_schemas)} ACTIVE schemas out of {len(result.data)} total")
            for schema in result.data:
                logger.info(f"🔍 SCHEMA STATUS CHECK: {schema.name} (id={schema.id}, status={schema.status}, is_active={schema.status == SchemaStatus.ACTIVE})")
            return active_schemas
        logger.warning(f"🔍 ACTIVE FILTER: list_schemas failed, returning empty list")
        return []
    
    async def get_schemas_by_ids(self, schema_ids: List[str], user_id: str, workspace_id: Optional[str] = None,
                                organization_id: Optional[str] = None, namespace_id: Optional[str] = None) -> List[UserGraphSchema]:
        """Get specific schemas by their IDs with multi-tenant context"""
        schemas = []
        for schema_id in schema_ids:
            result = await self.get_schema(schema_id, user_id, workspace_id, organization_id, namespace_id)
            if result.success and result.data:
                schemas.append(result.data)
        return schemas
    
    async def _validate_schema(self, schema: UserGraphSchema, user_id: str) -> SchemaResponse:
        """Validate schema for conflicts and correctness"""
        try:
            # Check for naming conflicts with protected system entities
            # Only Memory is protected since user schemas are used exclusively when available
            protected_node_types = ["Memory"]
            
            for node_name in schema.node_types.keys():
                if node_name in protected_node_types:
                    return SchemaResponse(
                        success=False,
                        error=f"Node type '{node_name}' is reserved for system use",
                        code=400
                    )
            
            # Validate relationship constraints
            all_node_types = set(schema.node_types.keys())
            
            for rel_name, rel_type in schema.relationship_types.items():
                for source_type in rel_type.allowed_source_types:
                    if source_type not in all_node_types:
                        return SchemaResponse(
                            success=False,
                            error=f"Unknown source node type '{source_type}' in relationship '{rel_name}'",
                            code=400
                        )
                
                for target_type in rel_type.allowed_target_types:
                    if target_type not in all_node_types:
                        return SchemaResponse(
                            success=False,
                            error=f"Unknown target node type '{target_type}' in relationship '{rel_name}'",
                            code=400
                        )
            
            return SchemaResponse(success=True)
            
        except Exception as e:
            logger.error(f"Error validating schema: {e}")
            return SchemaResponse(
                success=False,
                error=str(e),
                code=500
            )
    
    async def create_indexes_for_schema(self, schema: UserGraphSchema, memory_graph) -> bool:
        """
        Create Neo4j indexes for a custom schema when it becomes active.
        
        Args:
            schema: The UserGraphSchema object to create indexes for
            memory_graph: MemoryGraph instance to create indexes with
            
        Returns:
            bool: True if index creation was successful, False otherwise
        """
        try:
            logger.info(f"🔧 Creating indexes for schema '{schema.name}' (id: {schema.id})")
            
            # Convert schema to dictionary format expected by _create_custom_schema_indexes
            schema_dict = {
                'id': schema.id,
                'name': schema.name,
                'node_types': {}
            }
            
            # Convert UserNodeType objects to dictionary format
            for node_name, node_type in schema.node_types.items():
                schema_dict['node_types'][node_name] = {
                    'required_properties': node_type.required_properties,
                    'properties': {}
                }
                
                # Convert PropertyDefinition objects to dictionary format
                for prop_name, prop_def in node_type.properties.items():
                    schema_dict['node_types'][node_name]['properties'][prop_name] = {
                        'type': prop_def.type.value if hasattr(prop_def.type, 'value') else prop_def.type,
                        'required': prop_def.required,
                        'description': prop_def.description
                    }
            
            # Call the MemoryGraph method to create indexes
            await memory_graph._create_custom_schema_indexes([schema_dict])
            
            logger.info(f"✅ Successfully created indexes for schema '{schema.name}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexes for schema '{schema.name}': {e}")
            return False

    async def get_active_schemas_by_object_id(
        self, 
        user_object_id: str, 
        workspace_object_id: str, 
        httpx_client: Optional[httpx.AsyncClient] = None
    ) -> List[UserGraphSchema]:
        """Fetch active UserGraphSchema objects by Parse object IDs for agentic search"""
        
        try:
            # Use provided client or create new one
            client = httpx_client or httpx.AsyncClient()
            close_client = httpx_client is None
            
            try:
                # Query Parse for active schemas
                headers = {
                    "X-Parse-Application-Id": self.parse_app_id,
                    "X-Parse-Master-Key": self.parse_master_key,
                    "Content-Type": "application/json"
                }
                
                # Build query for active schemas
                where_clause = {
                    "user_id": user_object_id,
                    "workspace_id": workspace_object_id,
                    "is_active": True
                }
                
                params = {
                    "where": json.dumps(where_clause),
                    "limit": 1000  # Get all active schemas
                }
                
                response = await client.get(
                    f"{self.parse_server_url}/parse/classes/UserGraphSchema",
                    headers=headers,
                    params=params,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    schemas = []
                    
                    for schema_data in data.get('results', []):
                        try:
                            # Convert Parse object to UserGraphSchema
                            schema = self._parse_schema_from_response(schema_data)
                            schemas.append(schema)
                        except Exception as e:
                            logger.warning(f"Failed to parse schema {schema_data.get('objectId')}: {e}")
                            
                    logger.info(f"🔧 SCHEMA SERVICE: Fetched {len(schemas)} active schemas for agentic search")
                    return schemas
                    
                else:
                    logger.warning(f"Failed to fetch schemas: {response.status_code}")
                    return []
                    
            finally:
                if close_client:
                    await client.aclose()
                    
        except Exception as e:
            logger.error(f"Error fetching schemas by object ID: {e}")
            return []

# Dependency injection for FastAPI
def get_schema_service() -> SchemaService:
    return SchemaService()
