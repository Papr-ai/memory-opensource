from abc import ABC, abstractmethod
import json
from memory.memory_item_relevance import MemoryItemRelevance
import uuid
from services.logging_config import get_logger
from datetime import datetime, timezone, UTC
from typing import Optional, Dict, Any, List
from services.logger_singleton import LoggerSingleton
from models.shared_types import ContextItem
from services.context_utils import parse_context

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

class MemoryItem(ABC):
    def __init__(self, content: str, metadata: dict, context: Optional[List[ContextItem]] = None, 
                 graph_schema=None, node_and_relationships=None, related_memories=None, 
                 id=None, objectId=None, createdAt=None, relationships_json=None, 
                 memoryChunkIds: Optional[List[str]] = None):
        self.id = id if id else uuid.uuid4()
        self.objectId = objectId  # Parse Server objectId
        self.createdAt = createdAt  # Parse Server createdAt timestamp
        self.content = content
        self.metadata = metadata
        self.context = context
        self.graph_schema = graph_schema
        self.node_and_relationships = node_and_relationships
        self.related_memories = related_memories
        self.relationships_json = relationships_json
        self.memoryChunkIds = memoryChunkIds or []


    @abstractmethod
    def get_relevance_for(self, query):
        pass
class TextMemoryItem(MemoryItem):
    def __init__(self, content: str, metadata: dict, context: Optional[List[ContextItem]] = None, 
                 graph_schema=None, node_and_relationships=None, related_memories=None, 
                 id=None, objectId=None, createdAt=None, relationships_json=None,
                 memoryChunkIds: Optional[List[str]] = None):
        super().__init__(content, metadata, context, graph_schema, node_and_relationships, 
                        related_memories, id, objectId, createdAt, relationships_json,
                        memoryChunkIds=memoryChunkIds)
        self.type = 'TextMemoryItem'

    def __str__(self):
        return f"TextMemoryItem(id={self.id}, content={self.content}, metadata={self.metadata}, context={self.context}, graph_schema={self.graph_schema})"

    def get_relevance_for(self, query, context):
        # The context would usually come from elsewhere in your applications,
        # but I'm using a placeholder here for simplicity

        relevance_calculator = MemoryItemRelevance(self, query, context)
        return relevance_calculator.relevance

    def hello_world(): print("Hello, world!")

class CodeSnippetMemoryItem(MemoryItem):
    def __init__(self, content: str, metadata: dict, context: Optional[List[ContextItem]] = None,
                 graph_schema=None, node_and_relationships=None, related_memories=None,
                 id=None, objectId=None, createdAt=None, relationships_json=None,
                 memoryChunkIds: Optional[List[str]] = None):
        super().__init__(content, metadata, context, graph_schema, node_and_relationships,
                        related_memories, id, objectId, createdAt, relationships_json,
                        memoryChunkIds=memoryChunkIds)
        self.type = 'CodeSnippetMemoryItem'

    def __str__(self):
        return f"CodeSnippetMemoryItem(id={self.id}, content={self.content}, metadata={self.metadata}, context={self.context}, graph_schema={self.graph_schema})"

    def get_relevance_for(self, query, context):
        relevance_calculator = MemoryItemRelevance(self, query, context)
        return relevance_calculator.relevance

class DocumentMemoryItem(MemoryItem):
    def __init__(self, content: str, metadata: dict, context: Optional[List[ContextItem]] = None,
                 graph_schema=None, node_and_relationships=None, related_memories=None,
                 id=None, objectId=None, createdAt=None, relationships_json=None,
                 memoryChunkIds: Optional[List[str]] = None):
        super().__init__(content, metadata, context, graph_schema, node_and_relationships,
                        related_memories, id, objectId, createdAt, relationships_json,
                        memoryChunkIds=memoryChunkIds)
        self.type = 'DocumentMemoryItem'

    def __str__(self):
        return f"DocumentMemoryItem(id={self.id}, content={self.content}, metadata={self.metadata}, context={self.context}, graph_schema={self.graph_schema})"


    def get_relevance_for(self, query, context):
        relevance_calculator = MemoryItemRelevance(self, query, context)
        return relevance_calculator.relevance


class WebpageMemoryItem(MemoryItem):
    def __init__(self, content: str, url: str, timestamp: datetime, tags: list, users: list, 
                 context: Optional[List[ContextItem]]= None, id=None, objectId=None, 
                 createdAt=None, relationships_json=None,
                 memoryChunkIds: Optional[List[str]] = None):
        metadata = {
            'sourceType': 'webpage',
            'url': url,
            'timestamp': timestamp,
            'tags': tags,
            'users': users
        }
        super().__init__(content, metadata, context, id=id, objectId=objectId,
                        createdAt=createdAt, relationships_json=relationships_json,
                        memoryChunkIds=memoryChunkIds)
        self.type = 'WebpageMemoryItem'
    

    def get_relevance_for(self, query, context):
        # Implementation goes here
        relevance_calculator = MemoryItemRelevance(self, query, context)
        return relevance_calculator.relevance

class CodeFileMemoryItem(MemoryItem):
    def __init__(self, content: str, path: str, timestamp: datetime, tags: list, users: list,
                 context: Optional[List[ContextItem]] = None, id=None, objectId=None,
                 createdAt=None, relationships_json=None,
                 memoryChunkIds: Optional[List[str]] = None):
        metadata = {
            'sourceType': 'code_file',
            'path': path,
            'timestamp': timestamp,
            'tags': tags,
            'users': users
        }
        super().__init__(content, metadata, context, id=id, objectId=objectId,
                        createdAt=createdAt, relationships_json=relationships_json,
                        memoryChunkIds=memoryChunkIds)
        self.type = 'CodeFileMemoryItem'

    def get_relevance_for(self, query, context):
        # Implementation goes here
        relevance_calculator = MemoryItemRelevance(self, query, context)
        return relevance_calculator.relevance

class IssueMemoryItem(MemoryItem):
    def __init__(self, content: str, metadata: dict, context: Optional[List[ContextItem]] = None,
                 graph_schema=None, node_and_relationships=None, related_memories=None,
                 id=None, objectId=None, createdAt=None, relationships_json=None,
                 memoryChunkIds: Optional[List[str]] = None):
        super().__init__(content, metadata, context, graph_schema, node_and_relationships,
                        related_memories, id, objectId, createdAt, relationships_json,
                        memoryChunkIds=memoryChunkIds)
        self.type = 'IssueMemoryItem'

    def get_relevance_for(self, query, context):
        relevance_calculator = MemoryItemRelevance(self, query, context)
        return relevance_calculator.relevance


class CustomerMemoryItem(MemoryItem):
    def __init__(self, content: str, metadata: dict, context: Optional[List[ContextItem]]= None,
                 graph_schema=None, node_and_relationships=None, related_memories=None,
                 id=None, objectId=None, createdAt=None, relationships_json=None,
                 memoryChunkIds: Optional[List[str]] = None):
        super().__init__(content, metadata, context, graph_schema, node_and_relationships,
                        related_memories, id, objectId, createdAt, relationships_json,
                        memoryChunkIds=memoryChunkIds)
        self.type = 'CustomerMemoryItem'

    def get_relevance_for(self, query, context):
        relevance_calculator = MemoryItemRelevance(self, query, context)
        return relevance_calculator.relevance

class MeetingMemoryItem(MemoryItem):
    def __init__(self, content: str, metadata: dict, context: Optional[List[ContextItem]]= None,
                 graph_schema=None, node_and_relationships=None, related_memories=None,
                 id=None, objectId=None, createdAt=None, relationships_json=None,
                 memoryChunkIds: Optional[List[str]] = None):
        super().__init__(content, metadata, context, graph_schema, node_and_relationships,
                        related_memories, id, objectId, createdAt, relationships_json,
                        memoryChunkIds=memoryChunkIds)
        self.type = 'MeetingMemoryItem'

    def get_relevance_for(self, query, context):
        relevance_calculator = MemoryItemRelevance(self, query, context)
        return relevance_calculator.relevance

class PluginMemoryItem(MemoryItem):
    def __init__(self, content: str, plugin_name: str, task_details: dict, timestamp: datetime, tags: list, users: list, context: Optional[List[ContextItem]] = None, id=None, objectId=None, createdAt=None, relationships_json=None):
        super().__init__(content, {'sourceType': 'plugin', 'pluginName': plugin_name, 'taskDetails': json.dumps(task_details), 'timestamp': timestamp, 'tags': tags, 'users': users}, context, id, objectId, createdAt, relationships_json)
        self.type = 'PluginMemoryItem'

    def get_relevance_for(self, query, context):
        # Implementation goes here
        relevance_calculator = MemoryItemRelevance(self, query, context)
        return relevance_calculator.relevance

def memory_item_from_dict(memory_item_dict: dict) -> MemoryItem:
    logger.debug(f"Received memory_item_dict: {memory_item_dict}")  # Added logger

    # Ensure we have a valid memory ID
    memory_id = memory_item_dict.get('memoryId') or memory_item_dict.get('id')
    if not memory_id:
        logger.error("No valid memory ID found in memory_item_dict")
        raise ValueError("Memory ID is required")

    # Extract memory item type from dictionary
    memory_item_type = memory_item_dict.get('type')

    # Handle 'message' type by treating it as TextMemoryItem
    if memory_item_type == 'message':
        memory_item_type = 'TextMemoryItem'
    # Default to TextMemoryItem if no type is specified
    elif not memory_item_type:
        memory_item_type = 'TextMemoryItem'
        logger.info(f'No type specified, defaulting to {memory_item_type}')
    
    # Mapping of type strings to classes
    type_mapping = {
        'TextMemoryItem': TextMemoryItem,
        'CodeSnippetMemoryItem': CodeSnippetMemoryItem,
        'DocumentMemoryItem': DocumentMemoryItem,
        'IssueMemoryItem': IssueMemoryItem,
        'CustomerMemoryItem': CustomerMemoryItem,
        'MeetingMemoryItem': MeetingMemoryItem,
        'PluginMemoryItem': PluginMemoryItem,
        # Add other mappings as needed
    }
    
    item_class = type_mapping.get(memory_item_type)
    
    if not item_class:
        logger.warning(f"Unrecognized memory item type: {memory_item_type}, defaulting to TextMemoryItem")
        item_class = TextMemoryItem

    # Prepare metadata dictionary with all possible fields
    metadata = {
        # Base fields
        'memoryId': memory_id,
        'sourceType': memory_item_dict.get('sourceType'),
        'sourceUrl': memory_item_dict.get('sourceUrl'),
        'timestamp': memory_item_dict.get('createdAt'),
        'context': memory_item_dict.get('context'),
        'workspace': memory_item_dict.get('workspace'),
        'project_id': memory_item_dict.get('project_id'),
        'user_id': memory_item_dict.get('user_id'),
        'channel_type': memory_item_dict.get('channel_type'),
        
        # ACL fields
        'user_read_access': memory_item_dict.get('user_read_access', []),
        'user_write_access': memory_item_dict.get('user_write_access', []),
        'role_read_access': memory_item_dict.get('role_read_access', []),
        'role_write_access': memory_item_dict.get('role_write_access', []),
        'workspace_read_access': memory_item_dict.get('workspace_read_access', []),
        'workspace_write_access': memory_item_dict.get('workspace_write_access', []),
        
        # Additional fields
        'title': memory_item_dict.get('title'),
        'location': memory_item_dict.get('location'),
        'conversationId': memory_item_dict.get('conversationId'),
        'memoryChunkIds': memory_item_dict.get('memoryChunkIds', []),
    }

    # Handle emoji tags
    emoji_tags = memory_item_dict.get('emoji_tags') or memory_item_dict.get('emoji tags', [])
    if isinstance(emoji_tags, str):
        emoji_tags = [tag.strip() for tag in emoji_tags.split(',')]
    metadata['emoji_tags'] = emoji_tags

    # Handle hierarchical structures (both versions)
    hierarchical_structures = (
        memory_item_dict.get('hierarchical_structures') or 
        memory_item_dict.get('hierarchicalStructures') or 
        memory_item_dict.get('hierarchical structures')
    )
    if hierarchical_structures:
        metadata['hierarchical_structures'] = hierarchical_structures

    # Handle topics
    topics = memory_item_dict.get('topics')
    if isinstance(topics, str):
        topics = [topic.strip() for topic in topics.split(',')]
    metadata['topics'] = topics

    # Handle role and category
    role = memory_item_dict.get('role')
    if role:
        metadata['role'] = role
    
    category = memory_item_dict.get('category')
    if category:
        metadata['category'] = category

    # Include any additional metadata from the original dictionary
    original_metadata = memory_item_dict.get('metadata', {})
    if isinstance(original_metadata, dict):
        metadata.update(original_metadata)

    # Create the memory item instance with the prepared metadata
    try:
        return item_class(
            content=memory_item_dict.get('content', ''),
            metadata=metadata,
            context=parse_context(memory_item_dict.get('context')),
            id=memory_id,
            objectId=memory_item_dict.get('objectId'),
            createdAt=memory_item_dict.get('createdAt'),
            relationships_json=memory_item_dict.get('relationships_json')
        )
    except Exception as e:
        logger.error(f"Error creating memory item: {e}")
        raise

def memory_item_to_dict(memory_item_obj: MemoryItem) -> dict:
    logger.info(f"Converting memory item to dict. Input object ID: {memory_item_obj.id}")
    logger.info(f"Input object memoryChunkIds: {memory_item_obj.memoryChunkIds}")
    logger.info(f"Input object type: {type(memory_item_obj.memoryChunkIds)}")

    # Check the type of the memory item and convert to dict
    if isinstance(memory_item_obj, TextMemoryItem):
        item_type = 'TextMemoryItem'
    elif isinstance(memory_item_obj, CodeSnippetMemoryItem):
        item_type = 'CodeSnippetMemoryItem'
    elif isinstance(memory_item_obj, DocumentMemoryItem):
        item_type = 'DocumentMemoryItem'
    elif isinstance(memory_item_obj, IssueMemoryItem):
        item_type = 'IssueMemoryItem'
    elif isinstance(memory_item_obj, CustomerMemoryItem):
        item_type = 'CustomerMemoryItem'
    elif isinstance(memory_item_obj, MeetingMemoryItem):
        item_type = 'MeetingMemoryItem'
    else:
        raise ValueError(f"Unrecognized memory item object: {memory_item_obj}")

    # Ensure metadata is a dictionary
    metadata = memory_item_obj.metadata if isinstance(memory_item_obj.metadata, dict) else json.loads(memory_item_obj.metadata)
    logger.info(f"Original metadata: {metadata}")
    
    # Get memoryChunkIds directly from the object, handling string or list
    if isinstance(memory_item_obj.memoryChunkIds, str):
        try:
            # Try to parse as JSON first
            memory_chunk_ids = json.loads(memory_item_obj.memoryChunkIds)
        except json.JSONDecodeError:
            # If it's a string representation of a list, clean and split
            clean_str = memory_item_obj.memoryChunkIds.strip('[]').replace("'", "").replace('"', "")
            memory_chunk_ids = [id.strip() for id in clean_str.split(',') if id.strip()]
    else:
        memory_chunk_ids = memory_item_obj.memoryChunkIds if isinstance(memory_item_obj.memoryChunkIds, list) else []
    
    # Ensure no extra quotes in the IDs
    memory_chunk_ids = [id.strip("'").strip('"') for id in memory_chunk_ids]
    
    logger.info(f"Processed memoryChunkIds: {memory_chunk_ids}")
    
    # Update metadata with memoryChunkIds
    metadata['memoryChunkIds'] = memory_chunk_ids
    logger.info(f"Updated metadata with memoryChunkIds: {metadata['memoryChunkIds']}")

    # Clean metadata for Pinecone compatibility (remove None values)
    metadata = clean_metadata_for_pinecone(metadata)

    # Construct the dictionary
    memory_item_dict = {
        'type': item_type,
        'content': memory_item_obj.content,
        'metadata': metadata,
        'context': json.dumps(memory_item_obj.context),
        'graph_schema': memory_item_obj.graph_schema,
        'node_and_relationships': memory_item_obj.node_and_relationships,
        'related_memories': memory_item_obj.related_memories,
        'id': str(memory_item_obj.id),
        'objectId': memory_item_obj.objectId,
        'createdAt': datetime_handler(memory_item_obj.createdAt),
        'memoryChunkIds': memory_chunk_ids
    }

    logger.info(f"Final dict memoryChunkIds: {memory_item_dict['memoryChunkIds']}")
    logger.info(f"Final dict metadata memoryChunkIds: {memory_item_dict['metadata']['memoryChunkIds']}")
    return memory_item_dict

def datetime_handler(obj):
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

def clean_metadata_for_pinecone(metadata: dict) -> dict:
    """Remove keys with None values from metadata for Pinecone compatibility."""
    return {k: v for k, v in metadata.items() if v is not None}

