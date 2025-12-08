"""
Property Indexing Service for Neo4j node properties in Qdrant
"""
import asyncio
import uuid
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from models.user_schemas import PropertyDefinition, PropertyType
from services.logging_config import get_logger
from os import environ as env

logger = get_logger(__name__)


class SchemaBasedPropertyClassifier:
    """Classifies properties based on schema definitions for indexing"""
    
    def should_index_property(self, node_type: str, prop_name: str, prop_value: Any, 
                            indexable_properties: Dict[str, List[Dict]]) -> bool:
        """Determine if property should be indexed based on schema definitions"""
        
        prop_key = f"{node_type}.{prop_name}"
        
        # Check if this property is defined as indexable in any schema
        if prop_key not in indexable_properties:
            return False
            
        # Must be a non-empty string
        if not isinstance(prop_value, str) or len(prop_value.strip()) == 0:
            return False
            
        # Skip deterministic values (UUIDs, pure numbers, dates)
        if self._is_deterministic_value(prop_value):
            return False
            
        return True
    
    def _is_deterministic_value(self, value: str) -> bool:
        """Check if value is deterministic (UUID, number, date, etc.)"""
        
        # UUID pattern
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', value.lower()):
            return True
            
        # Pure numbers
        if re.match(r'^\d+$', value):
            return True
            
        # Date patterns
        if re.match(r'^\d{4}-\d{2}-\d{2}', value):
            return True
            
        # Boolean strings
        if value.lower() in ['true', 'false']:
            return True
            
        return False


class PropertyIndexingService:
    """Service for indexing Neo4j node properties in separate Qdrant collection"""
    
    def __init__(self, memory_graph):
        self.memory_graph = memory_graph
        self.classifier = SchemaBasedPropertyClassifier()

    async def index_node_properties(self, nodes: List[Dict], source_memory: Dict, 
                                  cached_schema: Optional[Dict] = None):
        """Create and route property memories to separate collection"""
        
        # Ensure Qdrant client is available
        if not self.memory_graph.qdrant_client:
            logger.warning("Qdrant client not available, skipping property indexing")
            return
        
        # Ensure property collection is initialized (same pattern as main collection)
        from os import environ as env
        property_collection_name = env.get("QDRANT_PROPERTY_COLLECTION", "neo4j_properties")
        
        # Set property collection if not already set
        if not hasattr(self.memory_graph, 'qdrant_property_collection') or not self.memory_graph.qdrant_property_collection:
            self.memory_graph.qdrant_property_collection = property_collection_name
            logger.info(f"Set property collection to: {property_collection_name}")
        
        # Ensure collection exists (same pattern as main Qdrant collection)
        try:
            await self.memory_graph.qdrant_client.get_collection(self.memory_graph.qdrant_property_collection)
            logger.info(f"Property collection '{self.memory_graph.qdrant_property_collection}' exists")
        except Exception as e:
            logger.info(f"Property collection '{self.memory_graph.qdrant_property_collection}' doesn't exist, creating it")
            # Create the collection using the same method as main collection
            success = await self.memory_graph.create_optimized_qdrant_collection(
                collection_name=self.memory_graph.qdrant_property_collection,
                vector_size=384  # sentence-bert dimensions
            )
            if not success:
                logger.error(f"Failed to create property collection '{self.memory_graph.qdrant_property_collection}'")
                return
            
        logger.info(f"🔍 PROPERTY INDEXING: Processing {len(nodes)} nodes for property extraction")
        
        # Extract indexable properties by cross-referencing with structured output schema
        if not cached_schema or not cached_schema.get('structured_output_schema'):
            logger.warning("No structured output schema available for property indexing")
            return
            
        structured_schema = cached_schema['structured_output_schema']
        indexable_properties = self._extract_indexable_properties_from_schema(structured_schema)
        logger.info(f"🔍 PROPERTY INDEXING: Found {len(indexable_properties)} indexable properties from schema")
        
        property_memories = []
        
        for node in nodes:
            node_type = node.get('label', 'Unknown')
            node_properties = node.get('properties', {})
            
            # Skip Memory nodes - they are already stored in separate memory collection
            if node_type == 'Memory':
                logger.info(f"🔍 PROPERTY INDEXING: Skipping Memory node - already stored in separate memory collection")
                continue
            
            logger.info(f"🔍 PROPERTY INDEXING: Processing {node_type} node with {len(node_properties)} properties: {list(node_properties.keys())}")
            
            for prop_name, prop_value in node_properties.items():
                # Check if this property should be indexed based on schema definition
                prop_key = f"{node_type}.{prop_name}"
                if self._should_index_property_value(node_type, prop_name, prop_value, indexable_properties):
                    property_memory = self._create_property_memory(
                        node_type=node_type,
                        property_name=prop_name,
                        property_value=prop_value,
                        source_memory=source_memory,
                        source_node_id=node_properties.get('id'),
                        schema_id=cached_schema.get('schema_id', 'unknown'),
                        schema_name=cached_schema.get('schema_name', 'Generated Schema'),
                        property_definition=indexable_properties[prop_key]
                    )
                    property_memories.append(property_memory)
                    logger.info(f"🔍 PROPERTY INDEXING: ✅ Added property {node_type}.{prop_name} = '{str(prop_value)[:50]}...' (schema-validated)")
                else:
                    reason = "not in schema" if prop_key not in indexable_properties else "failed value check"
                    logger.info(f"🔍 PROPERTY INDEXING: ❌ Skipped property {node_type}.{prop_name} ({reason})")
        
        logger.info(f"🔍 PROPERTY INDEXING: Generated {len(property_memories)} property memories from {len(nodes)} nodes")
        
        if property_memories:
            # Route to property collection instead of main collection
            await self._batch_index_property_memories_to_property_collection(property_memories)

    async def index_node_properties_with_sync(self, neo4j_results: List[Dict], source_memory: Dict, 
                                            cached_schema: Optional[Dict] = None,
                                            common_metadata: Optional[Dict[str, Any]] = None):
        """
        Steps 4-5: Property indexing with Neo4j synchronization results.
        Uses was_created flag to determine create vs update operations.
        """
        
        # Ensure Qdrant client is available
        if not self.memory_graph.qdrant_client:
            logger.info("🔄 SYNC STEPS 4-5: Qdrant client not available, skipping property indexing")
            return
        
        # Ensure property collection is initialized (same pattern as main collection)
        from os import environ as env
        property_collection_name = env.get("QDRANT_PROPERTY_COLLECTION", "neo4j_properties")
        
        # Set property collection if not already set
        if not hasattr(self.memory_graph, 'qdrant_property_collection') or not self.memory_graph.qdrant_property_collection:
            self.memory_graph.qdrant_property_collection = property_collection_name
            logger.info(f"🔄 SYNC STEPS 4-5: Set property collection to: {property_collection_name}")
        
        # Ensure collection exists (same pattern as main Qdrant collection)
        try:
            await self.memory_graph.qdrant_client.get_collection(self.memory_graph.qdrant_property_collection)
            logger.info(f"🔄 SYNC STEPS 4-5: Property collection '{self.memory_graph.qdrant_property_collection}' exists")
        except Exception as e:
            logger.info(f"🔄 SYNC STEPS 4-5: Property collection '{self.memory_graph.qdrant_property_collection}' doesn't exist, creating it")
            # Create the collection using the same method as main collection
            success = await self.memory_graph.create_optimized_qdrant_collection(
                collection_name=self.memory_graph.qdrant_property_collection,
                vector_size=384  # sentence-bert dimensions
            )
            if not success:
                logger.error(f"🔄 SYNC STEPS 4-5: Failed to create property collection '{self.memory_graph.qdrant_property_collection}'")
                return
            
        logger.info(f"🔄 SYNC STEPS 4-5: Processing {len(neo4j_results)} Neo4j results for property extraction")
        
        # Extract indexable properties by cross-referencing with structured output schema
        if not cached_schema or not cached_schema.get('structured_output_schema'):
            logger.warning("🔄 SYNC STEPS 4-5: No structured output schema available for property indexing")
            return
            
        structured_schema = cached_schema['structured_output_schema']
        indexable_properties = self._extract_indexable_properties_from_schema(structured_schema)
        logger.info(f"🔄 SYNC STEPS 4-5: Found {len(indexable_properties)} indexable properties from schema")
        
        property_memories = []
        
        for result in neo4j_results:
            node_type = result.get('label', 'Unknown')
            node_properties = result.get('properties', {})
            
            # Skip Memory nodes - they are already stored in separate memory collection
            if node_type == 'Memory':
                logger.info(f"🔄 SYNC STEPS 4-5: Skipping Memory node - already stored in separate memory collection")
                continue
                
            was_created = result.get('was_created', False)
            sync_operation = result.get('sync_operation', 'unknown')
            node_id = result.get('node_id')
            
            logger.info(f"🔄 SYNC STEPS 4-5: Processing {node_type} node (was_created={was_created}, sync_operation={sync_operation})")
            
            # Extract properties for this node type
            for prop_name, prop_value in node_properties.items():
                if self._should_index_property_value(node_type, prop_name, prop_value, indexable_properties):
                    
                    # Create property memory with full ACL metadata (same as _create_property_memory)
                    property_memory = self._create_property_memory_with_sync(
                        node_type=node_type,
                        property_name=prop_name,
                        property_value=str(prop_value),
                        source_memory=source_memory,
                        source_node_id=node_id,
                        was_created=was_created,
                        sync_operation=sync_operation,
                        canonical_node_id=node_id,
                        schema_id=cached_schema.get('schema_id'),
                        schema_name=cached_schema.get('schema_name'),
                        common_metadata=common_metadata  # Pass common_metadata
                    )
                    
                    property_memories.append(property_memory)
                    logger.info(f"🔄 SYNC STEPS 4-5: ✅ Added property {node_type}.{prop_name}='{str(prop_value)[:50]}...' for {sync_operation}")
        
        logger.info(f"🔄 SYNC STEPS 4-5: Generated {len(property_memories)} property memories from {len(neo4j_results)} Neo4j results")
        
        if len(property_memories) == 0 and len(neo4j_results) > 0:
            logger.warning(f"🔄 SYNC STEPS 4-5: ⚠️  WARNING: 0 properties generated from {len(neo4j_results)} nodes!")
            logger.warning(f"🔄 SYNC STEPS 4-5: This means no properties matched the indexable criteria")
            logger.warning(f"🔄 SYNC STEPS 4-5: Indexable properties found: {len(indexable_properties)}")
        
        if property_memories:
            # Use the existing schema-aware filtering and indexing
            filtered_batch = await self._filter_existing_properties(property_memories)
            
            if filtered_batch:
                await self._index_property_batch(filtered_batch)
                logger.info(f"🔄 SYNC STEPS 4-5: ✅ Successfully indexed {len(filtered_batch)} properties with sync")
            else:
                logger.info("🔄 SYNC STEPS 4-5: All properties already exist after sync filtering")

    def _extract_indexable_properties_from_schema(self, structured_schema: Dict) -> Dict[str, Dict]:
        """Extract indexable properties from structured output schema"""
        indexable_properties = {}
        
        logger.info(f"🔍 SCHEMA EXTRACTION: Processing structured schema type={type(structured_schema)}")
        
        try:
            # Navigate the schema structure: properties -> nodes -> items -> anyOf -> [0] -> properties -> properties
            if not isinstance(structured_schema, dict) or 'properties' not in structured_schema:
                logger.warning("🔍 SCHEMA EXTRACTION: No 'properties' in structured schema")
                return indexable_properties
                
            nodes_schema = structured_schema['properties'].get('nodes', {})
            if 'items' not in nodes_schema:
                logger.warning("🔍 SCHEMA EXTRACTION: No 'items' in nodes schema")
                return indexable_properties
                
            items_schema = nodes_schema['items']
            
            # Handle anyOf structure (multiple node types)
            if 'anyOf' in items_schema:
                logger.info(f"🔍 SCHEMA EXTRACTION: Found anyOf with {len(items_schema['anyOf'])} node type definitions")
                for i, node_def in enumerate(items_schema['anyOf']):
                    if 'properties' in node_def and 'properties' in node_def['properties']:
                        # Extract node type from the schema
                        node_type = self._extract_node_type_from_definition(node_def)
                        if node_type:
                            self._extract_properties_for_node_type(node_def, node_type, indexable_properties)
            
            # Handle direct properties structure (single node type)
            elif 'properties' in items_schema and 'properties' in items_schema['properties']:
                logger.info("🔍 SCHEMA EXTRACTION: Found direct properties structure")
                node_type = self._extract_node_type_from_definition(items_schema)
                if node_type:
                    self._extract_properties_for_node_type(items_schema, node_type, indexable_properties)
                    
        except Exception as e:
            logger.error(f"🔍 SCHEMA EXTRACTION: Error extracting properties from schema: {e}")
            import traceback
            logger.error(f"🔍 SCHEMA EXTRACTION: Traceback: {traceback.format_exc()}")
            
        logger.info(f"🔍 SCHEMA EXTRACTION: Extracted {len(indexable_properties)} indexable properties")
        return indexable_properties
    
    def _is_indexable_string_type(self, prop_type) -> bool:
        """
        Check if property type is an indexable string type.
        Supports both simple 'string' and nullable ['string', 'null'] union types.
        """
        if prop_type == 'string':
            return True
        if isinstance(prop_type, list) and 'string' in prop_type:
            return True  # Handle ['string', 'null'] union types from nullable schema
        return False
    
    def _extract_node_type_from_definition(self, node_def: Dict) -> Optional[str]:
        """Extract node type from a node definition"""
        try:
            # Look for label property with enum values
            if 'properties' in node_def and 'label' in node_def['properties']:
                label_def = node_def['properties']['label']
                if 'enum' in label_def and label_def['enum']:
                    return label_def['enum'][0]  # Take the first enum value as the node type
            return None
        except Exception as e:
            logger.warning(f"🔍 SCHEMA EXTRACTION: Error extracting node type: {e}")
            return None
    
    def _extract_properties_for_node_type(self, node_def: Dict, node_type: str, indexable_properties: Dict):
        """Extract indexable properties for a specific node type"""
        try:
            # Navigate to the actual properties definition
            # The structure is: node_def -> properties -> properties -> {actual_properties}
            if 'properties' not in node_def:
                logger.warning(f"🔍 SCHEMA EXTRACTION: No 'properties' in node_def for {node_type}")
                return
                
            node_properties = node_def['properties']
            if 'properties' not in node_properties:
                logger.warning(f"🔍 SCHEMA EXTRACTION: No nested 'properties' in node_properties for {node_type}")
                return
                
            # This is where the actual property definitions are
            actual_properties_def = node_properties['properties']
            if not isinstance(actual_properties_def, dict) or 'properties' not in actual_properties_def:
                logger.warning(f"🔍 SCHEMA EXTRACTION: No 'properties' in actual_properties_def for {node_type}")
                return
                
            properties_def = actual_properties_def['properties']
            required_props = actual_properties_def.get('required', [])
            
            logger.info(f"🔍 SCHEMA EXTRACTION: Processing {node_type} with {len(properties_def)} properties, {len(required_props)} required")
            logger.info(f"🔍 SCHEMA EXTRACTION: Property names: {list(properties_def.keys())}")
            logger.info(f"🔍 SCHEMA EXTRACTION: Required properties: {required_props}")
            
            for prop_name, prop_def in properties_def.items():
                logger.info(f"🔍 SCHEMA EXTRACTION: Examining property {node_type}.{prop_name}: {prop_def}")
                
                # Check if property type is string or nullable string
                prop_type = prop_def.get('type') if isinstance(prop_def, dict) else None
                is_string_type = self._is_indexable_string_type(prop_type)
                
                # Check if property should be indexed (required string properties without enums)
                # Now supports both 'string' and ['string', 'null'] union types
                if (prop_name in required_props and 
                    isinstance(prop_def, dict) and 
                    is_string_type and 
                    'enum' not in prop_def):
                    
                    prop_key = f"{node_type}.{prop_name}"
                    indexable_properties[prop_key] = {
                        'node_type': node_type,
                        'property_name': prop_name,
                        'property_type': prop_type,
                        'required': True,
                        'has_enum': False
                    }
                    logger.info(f"🔍 SCHEMA EXTRACTION: ✅ Added indexable property: {prop_key}")
                else:
                    reason = []
                    if prop_name not in required_props:
                        reason.append("not required")
                    if not isinstance(prop_def, dict) or not is_string_type:
                        reason.append(f"not string type (type={prop_type})")
                    if isinstance(prop_def, dict) and 'enum' in prop_def:
                        reason.append("has enum")
                    logger.info(f"🔍 SCHEMA EXTRACTION: ❌ Skipped property {node_type}.{prop_name} ({', '.join(reason)})")
                    
        except Exception as e:
            logger.error(f"🔍 SCHEMA EXTRACTION: Error processing properties for {node_type}: {e}")
            import traceback
            logger.error(f"🔍 SCHEMA EXTRACTION: Traceback: {traceback.format_exc()}")

    def _should_index_property_value(self, node_type: str, prop_name: str, prop_value: Any, indexable_properties: Dict[str, Dict]) -> bool:
        """Check if a property value should be indexed based on schema and content"""
        
        prop_key = f"{node_type}.{prop_name}"
        
        # Check if this property is defined as indexable in any schema
        if prop_key not in indexable_properties:
            return False
            
        # Must be a non-empty string
        if not isinstance(prop_value, str) or len(prop_value.strip()) == 0:
            return False
            
        # Skip deterministic values (UUIDs, pure numbers, dates)
        if self._is_deterministic_value(prop_value):
            return False
            
        return True
    
    def _is_deterministic_value(self, value: str) -> bool:
        """Check if value is deterministic (UUID, number, date, etc.)"""
        import re
        
        # UUID pattern
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', value.lower()):
            return True
            
        # Pure numbers
        if re.match(r'^\d+$', value):
            return True
            
        # Date patterns
        if re.match(r'^\d{4}-\d{2}-\d{2}', value):
            return True
            
        # Boolean strings
        if value.lower() in ['true', 'false']:
            return True
            
        return False

    def _truncate_to_token_limit(self, text: str, max_tokens: int = 256) -> str:
        """Truncate text to approximate token limit for sentence-transformers"""
        # Simple approximation: average ~4 characters per token for English text
        # This is conservative to ensure we stay under the limit
        max_chars = max_tokens * 4  # Conservative estimate (4 chars per token)
        
        if len(text) <= max_chars:
            return text
            
        # Truncate at word boundary to avoid cutting words in half
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        
        if last_space > max_chars * 0.8:  # If we can find a space in the last 20%
            return truncated[:last_space]
        else:
            return truncated  # Just truncate at character limit if no good word boundary

    def _create_property_memory(self, node_type: str, property_name: str, property_value: str, 
                              source_memory: Dict, source_node_id: str, 
                              schema_id: Optional[str] = None, schema_name: Optional[str] = None,
                              property_definition: Optional[Dict] = None) -> Dict:
        """Create a property memory following the exact same structure as regular memories"""
        
        # Inherit ALL metadata from source memory (same approach as existing Qdrant implementation)
        source_metadata = source_memory.get('metadata', {})
        
        # Create property-specific content
        property_content = f"Node: {node_type}, Property: {property_name}: {property_value}"
        
        # Build complete metadata following existing memory structure
        property_metadata = {
            # Inherit ALL existing metadata fields
            **source_metadata,
            
            # Property-specific metadata (following same pattern as existing memories)
            'is_property_index': True,
            'node_type': node_type,
            'property_name': property_name,
            'property_value': property_value,
            'source_node_id': source_node_id,
            'source_memory_id': source_memory.get('id'),
            'property_type': 'natural_language',
            'content': property_content,  # Add content to metadata like existing memories
            
            # Enhanced property metadata for better filtering and analytics
            'property_key': f"{node_type}.{property_name}",  # Composite key for easy filtering
            'property_value_length': len(property_value),
            'property_value_word_count': len(property_value.split()),
            'property_value_lowercase': property_value.lower(),  # For case-insensitive search
            
            # Schema metadata for filtering
            'schema_id': schema_id,
            'schema_name': schema_name,
            'is_system_schema': schema_id is None,
            'schema_type': 'system' if schema_id is None else 'user_defined',
            
            # Indexing metadata for debugging and analytics
            'indexed_at': datetime.now(timezone.utc).isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'indexing_version': '1.0',  # For future schema migrations
            
            # Source context for traceability
            'source_memory_type': source_memory.get('type', 'unknown'),
            'source_content_preview': source_memory.get('content', '')[:100] if source_memory.get('content') else '',
            
            # Ensure all required ACL fields are present (following existing pattern)
            'user_id': source_metadata.get('user_id'),
            'workspace_id': source_metadata.get('workspace_id'),
            'organization_id': source_metadata.get('organization_id'),
            'namespace_id': source_metadata.get('namespace_id'),
            
            # All ACL access fields (following exact same structure)
            'user_read_access': source_metadata.get('user_read_access', []),
            'user_write_access': source_metadata.get('user_write_access', []),
            'workspace_read_access': source_metadata.get('workspace_read_access', []),
            'workspace_write_access': source_metadata.get('workspace_write_access', []),
            'role_read_access': source_metadata.get('role_read_access', []),
            'role_write_access': source_metadata.get('role_write_access', []),
            'organization_read_access': source_metadata.get('organization_read_access', []),
            'organization_write_access': source_metadata.get('organization_write_access', []),
            'namespace_read_access': source_metadata.get('namespace_read_access', []),
            'namespace_write_access': source_metadata.get('namespace_write_access', []),
            'external_user_read_access': source_metadata.get('external_user_read_access', []),
            'external_user_write_access': source_metadata.get('external_user_write_access', [])
        }
        
        property_memory = {
            'id': f"prop_{source_node_id}_{property_name}_{uuid.uuid4().hex[:8]}",
            'content': property_content,
            'type': 'PropertyIndex',
            'metadata': property_metadata
        }
        
        return property_memory

    def _create_property_memory_with_sync(self, node_type: str, property_name: str, property_value: str, 
                                        source_memory: Dict, source_node_id: str,
                                        was_created: bool, sync_operation: str, canonical_node_id: str,
                                        schema_id: Optional[str] = None, schema_name: Optional[str] = None,
                                        common_metadata: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Create property memory with full ACL metadata AND sync information.
        This combines _create_property_memory with sync-specific fields.
        
        Args:
            common_metadata: Pre-extracted metadata with organization_id/namespace_id from both
                           metadata and customMetadata. If provided, use this instead of source_metadata.
        """
        
        # Use common_metadata if provided (has organization_id/namespace_id correctly extracted),
        # otherwise fall back to source_memory metadata
        if common_metadata:
            source_metadata = common_metadata
            logger.info(f"🔍 METADATA: Using common_metadata - organization_id={source_metadata.get('organization_id')}, namespace_id={source_metadata.get('namespace_id')}")
        else:
            source_metadata = source_memory.get('metadata', {})
            logger.warning(f"🔍 METADATA: common_metadata not provided, falling back to source_memory - organization_id={source_metadata.get('organization_id')}, namespace_id={source_metadata.get('namespace_id')}")
        
        # Create property content
        property_content = f"Node: {node_type}, Property: {property_name}: {property_value}"
        
        # Build complete metadata following existing memory structure + sync info
        property_metadata = {
            # Inherit ALL existing metadata fields
            **source_metadata,
            
            # Property-specific metadata (following same pattern as existing memories)
            'is_property_index': True,
            'node_type': node_type,
            'property_name': property_name,
            'property_value': property_value,
            'source_node_id': source_node_id,
            'source_memory_id': source_memory.get('id'),
            'property_type': 'natural_language',
            'content': property_content,  # Add content to metadata like existing memories
            
            # Enhanced property metadata for better filtering and analytics
            'property_key': f"{node_type}.{property_name}",  # Composite key for easy filtering
            'property_value_length': len(property_value),
            'property_value_word_count': len(property_value.split()),
            'property_value_lowercase': property_value.lower(),  # For case-insensitive search
            
            # Schema metadata for filtering
            'schema_id': schema_id,
            'schema_name': schema_name,
            'is_system_schema': schema_id is None,
            'schema_type': 'system' if schema_id is None else 'user_defined',
            
            # Indexing metadata for debugging and analytics
            'indexed_at': datetime.now(timezone.utc).isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'indexing_version': '1.0',  # For future schema migrations
            
            # Source context for traceability
            'source_memory_type': source_memory.get('type', 'unknown'),
            'source_content_preview': source_memory.get('content', '')[:100] if source_memory.get('content') else '',
            
            # Ensure all required ACL fields are present (following existing pattern)
            'user_id': source_metadata.get('user_id'),
            'workspace_id': source_metadata.get('workspace_id'),
            'organization_id': source_metadata.get('organization_id'),
            'namespace_id': source_metadata.get('namespace_id'),
            
            # All ACL access fields (following exact same structure)
            'user_read_access': source_metadata.get('user_read_access', []),
            'user_write_access': source_metadata.get('user_write_access', []),
            'workspace_read_access': source_metadata.get('workspace_read_access', []),
            'workspace_write_access': source_metadata.get('workspace_write_access', []),
            'role_read_access': source_metadata.get('role_read_access', []),
            'role_write_access': source_metadata.get('role_write_access', []),
            'organization_read_access': source_metadata.get('organization_read_access', []),
            'organization_write_access': source_metadata.get('organization_write_access', []),
            'namespace_read_access': source_metadata.get('namespace_read_access', []),
            'namespace_write_access': source_metadata.get('namespace_write_access', []),
            'external_user_read_access': source_metadata.get('external_user_read_access', []),
            'external_user_write_access': source_metadata.get('external_user_write_access', []),
            
            # SYNC-SPECIFIC FIELDS: Add the sync information from Neo4j results
            'was_created': was_created,
            'sync_operation': sync_operation,  # 'create' or 'update'
            'canonical_node_id': canonical_node_id,  # Use Neo4j node_id as canonical reference
            'operation': sync_operation  # For compatibility with existing filtering logic
        }
        
        property_memory = {
            'id': f"prop_{source_node_id}_{property_name}_{uuid.uuid4().hex[:8]}",
            'content': property_content,
            'type': 'PropertyIndex',
            'metadata': property_metadata
        }
        
        return property_memory

    async def _batch_index_property_memories_to_property_collection(self, property_memories: List[Dict]):
        """Index property memories to separate Qdrant collection"""
        
        try:
            logger.info(f"Indexing {len(property_memories)} property memories to property collection")
            
            # Process in batches to avoid overwhelming Qdrant
            batch_size = int(env.get('PROPERTY_INDEXING_BATCH_SIZE', '50'))
            for i in range(0, len(property_memories), batch_size):
                batch = property_memories[i:i + batch_size]
                await self._index_property_batch(batch)
                
        except Exception as e:
            logger.error(f"Error batch indexing property memories: {e}")

    async def _index_property_batch(self, property_batch: List[Dict]):
        """Index a batch of property memories using Qdrant's cloud inference with retry logic"""
        
        try:
            if not self.memory_graph.qdrant_client or not self.memory_graph.qdrant_property_collection:
                logger.info("Qdrant client or property collection not available")
                return
            
            # Filter out properties that are already indexed (deduplication)
            filtered_batch = await self._filter_existing_properties(property_batch)
            
            if not filtered_batch:
                logger.info("All properties already indexed, skipping batch")
                return
                
            logger.info(f"Indexing {len(filtered_batch)} new properties (filtered {len(property_batch) - len(filtered_batch)} existing)")
            
            # Prepare documents and generate embeddings using HuggingFace API (consistent with regular memories)
            import asyncio
            
            documents = []
            metadatas = []
            ids = []
            
            for property_memory in filtered_batch:
                content = property_memory['content']
                
                # Limit text to 256 tokens (Sentence-BERT context window limit)
                content = self._truncate_to_token_limit(content, max_tokens=256)
                if len(content) < len(property_memory['content']):
                    logger.info(f"Truncated property content from {len(property_memory['content'])} to {len(content)} characters for 256 token limit")
                
                # Prepare metadata for property collection
                metadata = property_memory['metadata'].copy()
                metadata.update({
                    'memory_id': property_memory['id'],
                    'content': content,
                    'type': property_memory['type'],
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'operation': property_memory.get('operation', 'create'),
                    'canonical_node_id': property_memory.get('canonical_node_id', property_memory.get('source_node_id', ''))
                })
                
                documents.append(content)
                metadatas.append(metadata)
                # Use operation-specific ID handling
                if property_memory.get('operation') == 'update':
                    # For updates, use existing point ID
                    ids.append(property_memory.get('existing_point_id', str(uuid.uuid4())))
                else:
                    # For creates, use new UUID
                    ids.append(str(uuid.uuid4()))
            
            # Generate embeddings using HuggingFace API TRUE BATCH processing
            # HuggingFace API supports sending array of texts in single request
            # This is more efficient than parallel individual requests
            # If HuggingFace API fails (timeout, API error, etc.), those properties won't be indexed
            # This is acceptable - Neo4j will still have the properties, they just won't be in Qdrant
            
            from os import environ as env
            MAX_BATCH_SIZE = int(env.get("MAX_EMBEDDING_CLIENT_BATCH_SIZE", "32"))
            
            logger.info(f"🚀 BATCH EMBEDDING: Generating embeddings for {len(documents)} properties using true batch API")
            
            all_embeddings = []
            
            # Process in batches to respect API limits
            for batch_start in range(0, len(documents), MAX_BATCH_SIZE):
                batch_end = min(batch_start + MAX_BATCH_SIZE, len(documents))
                batch_docs = documents[batch_start:batch_end]
                batch_num = batch_start // MAX_BATCH_SIZE + 1
                total_batches = (len(documents) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
                
                logger.info(f"🚀 BATCH {batch_num}/{total_batches}: Processing {len(batch_docs)} properties")
                
                try:
                    # Use the batch embedding method (to be implemented)
                    batch_embeddings = await self._generate_batch_embeddings(batch_docs)
                    all_embeddings.extend(batch_embeddings)
                    logger.info(f"✅ BATCH {batch_num}/{total_batches}: Successfully generated {len(batch_embeddings)} embeddings")
                except Exception as e:
                    logger.warning(f"⚠️ BATCH {batch_num}/{total_batches} FAILED: {e}, falling back to individual requests")
                    # Fallback: process individually
                    for doc in batch_docs:
                        try:
                            embeddings_result, _ = await self.memory_graph.embedding_model.get_sentence_embedding(doc)
                            if embeddings_result and len(embeddings_result) > 0:
                                all_embeddings.append(embeddings_result[0])
                            else:
                                all_embeddings.append(None)
                        except Exception as individual_error:
                            logger.warning(f"Individual embedding error: {individual_error}")
                            all_embeddings.append(None)
            
            # Filter out documents that failed to generate embeddings
            # Properties that failed will still exist in Neo4j, just not in Qdrant for semantic matching
            valid_items = [(emb, meta, id_val) for emb, meta, id_val in zip(all_embeddings, metadatas, ids) if emb is not None]
            if not valid_items:
                logger.warning(f"No valid embeddings generated for batch of {len(documents)} properties. Properties exist in Neo4j but not indexed in Qdrant.")
                return
            
            if len(valid_items) < len(documents):
                logger.warning(f"Only {len(valid_items)}/{len(documents)} properties successfully generated embeddings. Partial indexing will proceed.")
            
            embeddings, metadatas, ids = zip(*valid_items) if valid_items else ([], [], [])
            
            # Prepare points for batch upsert
            points = []
            for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
                # Use the same ID generation pattern as main collection
                property_id = ids[i]
                qdrant_id = self.memory_graph.make_qdrant_id(property_id)
                
                # Log each property being indexed
                prop_key = metadata.get('property_key', 'unknown')
                prop_value = metadata.get('property_value', '')
                canonical_id = metadata.get('canonical_node_id', 'unknown')
                logger.info(f"🧾 INDEXING: {prop_key}='{str(prop_value)[:50]}...' → node={canonical_id}, point_id={qdrant_id}")
                
                # Handle embedding as list or numpy array (HuggingFace API returns list, local models return numpy array)
                embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                
                points.append({
                    "id": qdrant_id,
                    "vector": embedding_list,
                    "payload": metadata
                })
            
            # Batch upsert with retry logic (similar to add_qdrant_point)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Increase timeout for batch operations and add retry logic
                    timeout = 30.0 if attempt == 0 else 45.0  # Longer timeout on retries
                    result = await asyncio.wait_for(
                        self.memory_graph.qdrant_client.upsert(
                            collection_name=self.memory_graph.qdrant_property_collection,
                            points=points,
                            wait=True
                        ),
                        timeout=timeout
                    )
                    logger.info(f"Successfully indexed {len(points)} property memories to collection '{self.memory_graph.qdrant_property_collection}' on attempt {attempt + 1}")
                    return result
                    
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        retry_delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(f"Property batch upsert timeout (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Property batch upsert timed out after {max_retries} attempts")
                        raise
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        retry_delay = 2 ** attempt
                        logger.warning(f"Property batch upsert error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Property batch upsert failed after {max_retries} attempts: {e}")
                        raise
                
        except Exception as e:
            logger.error(f"Error indexing property batch: {e}")
            import traceback
            logger.error(f"Property indexing traceback: {traceback.format_exc()}")

    async def _filter_existing_properties(self, property_batch: List[Dict]) -> List[Dict]:
        """
        Schema-aware property synchronization using unique identifier matching.
        Steps: 1) Find existing entity by unique IDs, 2) Determine create vs update operations
        """
        try:
            if not self.memory_graph.qdrant_client or not self.memory_graph.qdrant_property_collection:
                return property_batch
            
            # Group properties by entity (same node)
            entities = {}
            for prop_memory in property_batch:
                # Extract from metadata dict (where property indexing stores these fields)
                prop_metadata = prop_memory.get('metadata', {})
                node_type = prop_metadata.get('node_type', '')
                source_node_id = prop_metadata.get('source_node_id', '')
                entity_key = f"{node_type}:{source_node_id}"
                
                if entity_key not in entities:
                    entities[entity_key] = {
                        'node_type': node_type,
                        'source_node_id': source_node_id,
                        'properties': [],
                        'unique_identifiers': self._get_unique_identifiers_for_node_type(node_type)
                    }
                entities[entity_key]['properties'].append(prop_memory)
            
            filtered_batch = []
            
            # Process each entity
            for entity_key, entity_data in entities.items():
                try:
                    # SIMPLIFIED APPROACH: Use canonical_node_id directly from Neo4j
                    # No need to search by unique ID values - we already know the Neo4j node ID
                    canonical_node_id = entity_data['source_node_id']
                    
                    # Get ACL metadata from first property (all properties from same entity have same ACL)
                    acl_metadata = {}
                    if entity_data['properties']:
                        first_prop = entity_data['properties'][0]
                        
                        # ACL fields are inside the metadata dict, not at top level
                        prop_metadata = first_prop.get('metadata', {})
                        
                        logger.info(f"🔍 ACL EXTRACTION: Extracting from prop_metadata - namespace_id={prop_metadata.get('namespace_id')}, organization_id={prop_metadata.get('organization_id')}")
                        
                        acl_metadata = {
                            'user_id': prop_metadata.get('user_id'),
                            'workspace_id': prop_metadata.get('workspace_id'),
                            'organization_id': prop_metadata.get('organization_id'),
                            'namespace_id': prop_metadata.get('namespace_id'),
                            'role_read_access': prop_metadata.get('role_read_access', [])
                        }
                    
                    # Get ALL existing properties for this entity in one query
                    # This works for BOTH unique ID and non-unique ID nodes
                    existing_properties = await self._find_all_properties_for_entity(canonical_node_id, acl_metadata)
                    
                    if existing_properties:
                        # Entity exists in Qdrant - determine which properties to update
                        logger.info(f"🔍 ENTITY SYNC: Found existing entity {entity_key} in Qdrant with {len(existing_properties)} properties")
                        
                        for prop_memory in entity_data['properties']:
                            prop_name = prop_memory.get('property_name', '')
                            
                            # Check if this property exists
                            if prop_name in existing_properties:
                                # Property exists - UPDATE
                                existing_prop = existing_properties[prop_name]
                                prop_memory['operation'] = 'update'
                                prop_memory['existing_point_id'] = existing_prop['point_id']
                                prop_memory['canonical_node_id'] = canonical_node_id
                                logger.info(f"🔍 PROPERTY SYNC: Will UPDATE {entity_key}.{prop_name} (point_id: {existing_prop['point_id']})")
                            else:
                                # New property for existing entity - CREATE
                                prop_memory['operation'] = 'create'
                                prop_memory['canonical_node_id'] = canonical_node_id
                                logger.info(f"🔍 PROPERTY SYNC: Will CREATE new property {entity_key}.{prop_name}")
                            
                            filtered_batch.append(prop_memory)
                    else:
                        # Entity doesn't exist in Qdrant yet - create all properties
                        logger.info(f"🔍 ENTITY SYNC: New entity {entity_key}, will create all properties")
                        
                        for prop_memory in entity_data['properties']:
                            prop_memory['operation'] = 'create'
                            prop_memory['canonical_node_id'] = canonical_node_id
                            filtered_batch.append(prop_memory)
                            
                except Exception as entity_error:
                    logger.error(f"🔍 ENTITY SYNC: Error processing entity {entity_key}: {entity_error}")
                    # Fallback: treat all properties as new
                    for prop_memory in entity_data['properties']:
                        prop_memory['operation'] = 'create'
                        prop_memory['canonical_node_id'] = entity_data['source_node_id']
                        filtered_batch.append(prop_memory)
            
            return filtered_batch
            
        except Exception as e:
            logger.error(f"Error in schema-aware property filtering: {e}")
            # Fallback: treat all as new
            for prop in property_batch:
                prop['operation'] = 'create'
                prop['canonical_node_id'] = prop.get('source_node_id', '')
            return property_batch

    def _generate_property_point_id(self, property_memory: Dict) -> str:
        """
        Generate deterministic point ID based on schema unique_identifiers + property_name.
        This ensures same property for same entity always gets same ID.
        """
        try:
            node_type = property_memory.get('node_type', '')
            property_name = property_memory.get('property_name', '')
            node_id = property_memory.get('node_id', '')
            
            # Get unique identifiers from schema (if available)
            unique_identifiers = self._get_unique_identifiers_for_property(property_memory)
            
            # Build deterministic key components
            key_parts = []
            
            # Always include node_type and property_name
            key_parts.extend([node_type, property_name])
            
            # Add unique identifier values if available
            if unique_identifiers:
                for identifier in sorted(unique_identifiers):  # Sort for consistency
                    value = property_memory.get(identifier, '')
                    key_parts.append(f"{identifier}:{value}")
            else:
                # Fallback to node_id if no unique identifiers
                key_parts.append(f"node_id:{node_id}")
            
            # Add ACL context for multi-tenant isolation
            user_id = property_memory.get('user_id', '')
            workspace_id = property_memory.get('workspace_id', '')
            key_parts.extend([f"user:{user_id}", f"workspace:{workspace_id}"])
            
            # Create deterministic hash
            key_string = "|".join(key_parts)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key_string))
            
            logger.debug(f"🔍 POINT ID: Generated {point_id} from key: {key_string}")
            return point_id
            
        except Exception as e:
            logger.error(f"🔍 POINT ID: Error generating ID: {e}")
            # Fallback to random UUID
            return str(uuid.uuid4())


    def _get_unique_identifiers_for_node_type(self, node_type: str) -> List[str]:
        """Get unique identifiers for a specific node type from schema"""
        try:
            # Default unique identifiers based on node type patterns
            # This should be enhanced to read from actual schema
            default_identifiers = {
                'Project': ['name'],
                'Task': ['title', 'project_id'],
                'Insight': ['title'],
                'KnowledgeNote': ['title'],
                'Company': ['name'],
                'Person': ['name', 'email'],
            }
            
            return default_identifiers.get(node_type, ['id'])  # Fallback to 'id'
            
        except Exception as e:
            logger.error(f"🔍 UNIQUE IDS: Error getting identifiers for {node_type}: {e}")
            return ['id']  # Safe fallback

    async def _find_existing_entity_by_unique_ids(self, entity_data: Dict) -> Optional[Dict]:
        """
        Find existing entity in Qdrant by searching for similar unique identifier values.
        Returns canonical entity info if found, None if new entity.
        """
        try:
            node_type = entity_data['node_type']
            unique_identifiers = entity_data['unique_identifiers']
            properties = entity_data['properties']
            
            # Extract unique identifier values from properties
            unique_values = {}
            for prop_memory in properties:
                prop_name = prop_memory.get('property_name', '')
                if prop_name in unique_identifiers:
                    unique_values[prop_name] = prop_memory.get('property_value', '')
            
            if not unique_values:
                logger.info(f"🔍 ENTITY SEARCH: No unique identifier values found for {node_type}")
                return None
            
            # Search for each unique identifier with semantic similarity
            for uid_name, uid_value in unique_values.items():
                # Search for similar unique identifier values using HuggingFace API
                uid_content = f"Node: {node_type}, Property: {uid_name}: {uid_value}"
                
                try:
                    # Use max_retries=1 for fast failure - entity deduplication is non-critical
                    embeddings_result, _ = await self.memory_graph.embedding_model.get_sentence_embedding(
                        uid_content,
                        max_retries=1  # Fast failure - don't delay memory creation
                    )
                    if embeddings_result and len(embeddings_result) > 0:
                        uid_embedding = embeddings_result[0]
                    else:
                        logger.warning(f"Failed to generate embedding for unique ID search (will create new entity): {uid_content[:80]}...")
                        continue
                except Exception as e:
                    logger.warning(f"Embedding generation error for unique ID search (will create new entity): {e}")
                    continue
                
                search_results = await self.memory_graph.qdrant_client.search(
                    collection_name=self.memory_graph.qdrant_property_collection,
                    query_vector=uid_embedding,
                    filter={
                        "node_type": node_type,
                        "property_name": uid_name
                    },
                    limit=1,  # Get top match only
                    score_threshold=0.95,  # High threshold for unique ID matching
                    with_payload=True,
                    with_vectors=False
                )
                
                if search_results and len(search_results) > 0:
                    match = search_results[0]
                    canonical_node_id = match.payload.get('source_node_id')
                    canonical_value = match.payload.get('property_value')
                    
                    logger.info(f"🔍 ENTITY SEARCH: Found similar {uid_name}: '{uid_value}' matches '{canonical_value}' (score: {match.score:.3f})")
                    
                    return {
                        'canonical_node_id': canonical_node_id,
                        'canonical_unique_values': {uid_name: canonical_value},
                        'similarity_score': match.score
                    }
            
            # No similar entity found
            logger.info(f"🔍 ENTITY SEARCH: No existing entity found for {node_type} with unique IDs: {unique_values}")
            return None
            
        except Exception as e:
            logger.error(f"🔍 ENTITY SEARCH: Error finding existing entity: {e}")
            return None

    async def _generate_batch_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts using HuggingFace API batch processing.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embeddings (or None for failed texts)
        """
        try:
            import httpx
            from os import environ as env
            
            api_url = env.get("HUGGING_FACE_API_URL_SENTENCE_BERT")
            access_token = env.get("HUGGING_FACE_ACCESS_TOKEN")
            
            if not api_url or not access_token:
                logger.error("HuggingFace API URL or token not configured")
                return [None] * len(texts)
            
            headers = {"Authorization": f"Bearer {access_token}"}
            payload = {"inputs": texts}  # Send array of texts
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(api_url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    embeddings = response.json()
                    # API returns array of embeddings matching input array
                    return embeddings
                else:
                    logger.error(f"Batch embedding API error: {response.status_code} - {response.text}")
                    return [None] * len(texts)
                    
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return [None] * len(texts)
    
    async def _find_existing_property_for_entity(self, canonical_node_id: str, property_name: str) -> Optional[Dict]:
        """
        Find existing property point for a specific entity and property name.
        """
        try:
            # Search by exact node ID and property name
            search_results = await self.memory_graph.qdrant_client.scroll(
                collection_name=self.memory_graph.qdrant_property_collection,
                scroll_filter={
                    "source_node_id": canonical_node_id,
                    "property_name": property_name
                },
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            if search_results and len(search_results[0]) > 0:
                point = search_results[0][0]  # First result from scroll
                return {
                    'point_id': str(point.id),
                    'property_value': point.payload.get('property_value', ''),
                    'content': point.payload.get('content', '')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"🔍 PROPERTY SEARCH: Error finding property {property_name} for node {canonical_node_id}: {e}")
            return None

    async def _find_all_properties_for_entity(self, canonical_node_id: str, acl_metadata: Dict[str, Any]) -> Dict[str, Dict]:
        """
        OPTIMIZED: Find ALL existing property points for an entity in one query with ACL filtering.
        
        Args:
            canonical_node_id: The Neo4j node ID (source_node_id in Qdrant)
            acl_metadata: Dict containing user_id, workspace_id, organization_id, namespace_id, role_read_access
        
        Returns: {property_name: {point_id, property_value, content}}
        """
        try:
            from qdrant_client.http import models
            
            # Extract ACL metadata
            user_id = acl_metadata.get('user_id')
            workspace_id = acl_metadata.get('workspace_id')
            organization_id = acl_metadata.get('organization_id')
            namespace_id = acl_metadata.get('namespace_id')
            role_read_access = acl_metadata.get('role_read_access', [])
            
            # CRITICAL: namespace_id is REQUIRED for multi-tenant isolation
            if not namespace_id:
                logger.warning(f"namespace_id is required for Qdrant search but was None. Node: {canonical_node_id}")
                return {}
            
            # Build must conditions: source_node_id AND namespace_id
            must_conditions = [
                models.FieldCondition(key="source_node_id", match=models.MatchValue(value=canonical_node_id)),
                models.FieldCondition(key="namespace_id", match=models.MatchValue(value=namespace_id))
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
            
            # Add ACL filtering
            if workspace_id:
                should_conditions.append(models.FieldCondition(key="workspace_read_access", match=models.MatchAny(any=[workspace_id])))
            if organization_id:
                should_conditions.append(models.FieldCondition(key="organization_read_access", match=models.MatchAny(any=[organization_id])))
            if namespace_id:
                should_conditions.append(models.FieldCondition(key="namespace_read_access", match=models.MatchAny(any=[namespace_id])))
            if role_read_access:
                should_conditions.append(models.FieldCondition(key="role_read_access", match=models.MatchAny(any=role_read_access)))
            
            # ACL filter: must match (source_node_id AND namespace_id) AND (user has access via ANY of the should conditions)
            scroll_filter = models.Filter(
                must=must_conditions,
                should=should_conditions
            )
            
            # Log detailed ACL filter information
            logger.info(f"🔍 BATCH PROPERTY SEARCH: Searching for properties of node {canonical_node_id}")
            logger.info(f"🔍 ACL FILTER - MUST (AND): source_node_id='{canonical_node_id}' AND namespace_id='{namespace_id}'")
            logger.info(f"🔍 ACL FILTER - SHOULD (OR): {len(should_conditions)} conditions - user_id='{user_id}', user_read_access=[{user_id}], workspace_read_access=[{workspace_id}], organization_read_access=[{organization_id}], namespace_read_access=[{namespace_id}], role_read_access={role_read_access}")
            
            # Get ALL properties for this entity in one scroll query
            all_results = []
            offset = None
            
            while True:
                search_results = await self.memory_graph.qdrant_client.scroll(
                    collection_name=self.memory_graph.qdrant_property_collection,
                    scroll_filter=scroll_filter,
                    limit=100,  # Batch size
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, next_offset = search_results
                all_results.extend(points)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            # Build property map
            property_map = {}
            for point in all_results:
                property_name = point.payload.get('property_name', '')
                if property_name:
                    property_map[property_name] = {
                        'point_id': str(point.id),
                        'property_value': point.payload.get('property_value', ''),
                        'content': point.payload.get('content', '')
                    }
            
            logger.info(f"🔍 BATCH PROPERTY SEARCH: Found {len(property_map)} existing properties for entity {canonical_node_id}")
            return property_map
            
        except Exception as e:
            logger.error(f"🔍 BATCH PROPERTY SEARCH: Error finding properties for node {canonical_node_id}: {e}")
            return {}

    async def _store_property_in_qdrant(self, property_id: str, embedding: List[float], metadata: Dict):
        """Store single property memory in Qdrant property collection using same approach as existing memories"""
        
        try:
            # Follow the exact same pattern as add_qdrant_point but route to property collection
            qdrant_id = self.memory_graph.make_qdrant_id(property_id)
            
            # Use same retry logic and timeout as existing implementation
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    timeout = 30.0 if attempt == 0 else 45.0  # Same timeout logic as existing
                    result = await asyncio.wait_for(
                        self.memory_graph.qdrant_client.upsert(
                            collection_name=self.memory_graph.qdrant_property_collection,  # Use property collection
                            points=[{
                                "id": qdrant_id,
                                "vector": embedding,
                                "payload": metadata
                            }],
                            wait=True
                        ),
                        timeout=timeout
                    )
                    logger.debug(f"Upserted property point to {self.memory_graph.qdrant_property_collection} with id {qdrant_id} (from property_id {property_id}) on attempt {attempt + 1}")
                    return result
                    
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        retry_delay = 2 ** attempt  # Same exponential backoff as existing
                        logger.info(f"Property Qdrant upsert timeout for property_id {property_id} (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Property Qdrant upsert timed out for property_id {property_id} after {max_retries} attempts")
                        return None
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        retry_delay = 2 ** attempt
                        logger.info(f"Property Qdrant upsert error for property_id {property_id} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Property Qdrant upsert failed for property_id {property_id} after {max_retries} attempts: {e}")
                        return None
            
        except Exception as e:
            logger.error(f"Error storing property in Qdrant: {e}")
