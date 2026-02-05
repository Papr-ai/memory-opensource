"""
Automatic registration of default schemas (like AgentLearning) on first use.

This ensures users don't need to manually register schemas - they're created
automatically when first needed.
"""

import asyncio
from typing import Optional
from services.logger_singleton import LoggerSingleton
from services.schema_service import SchemaService

logger = LoggerSingleton.get_logger(__name__)

# AgentLearning Schema Definition (same as register_agent_learning_schema.py)
AGENT_LEARNING_SCHEMA = {
    "name": "AgentLearning",
    "description": "Captures learnings from agent-user interactions and agent execution performance",
    "version": "1.0.0",
    "status": "active",
    "scope": "organization",
    "tags": ["agent_learning", "auto_registered"],
    "node_types": {
        "Learning": {
            "name": "Learning",
            "label": "Learning",
            "description": "A learning captured from agent interactions or execution",
            "properties": {
                "content": {"type": "string", "required": True},
                "role": {"type": "string", "required": True},
                "learning_type": {"type": "string", "required": False},
                "confidence": {"type": "float", "required": False},
                "evidence": {"type": "string", "required": False},
                "scope": {"type": "string", "required": False},
                "context": {"type": "string", "required": False},
                "project_id": {"type": "string", "required": False},
                "goal_id": {"type": "string", "required": False}
            },
            "unique_identifiers": ["content", "role"]
        },
        "User": {
            "name": "User",
            "label": "User",
            "description": "A user in the system",
            "properties": {"id": {"type": "string", "required": True}},
            "unique_identifiers": ["id"]
        },
        "Project": {
            "name": "Project",
            "label": "Project",
            "description": "A project context",
            "properties": {"id": {"type": "string", "required": True}},
            "unique_identifiers": ["id"]
        },
        "Goal": {
            "name": "Goal",
            "label": "Goal",
            "description": "A goal context",
            "properties": {"id": {"type": "string", "required": True}},
            "unique_identifiers": ["id"]
        },
        "MessageSession": {
            "name": "MessageSession",
            "label": "MessageSession",
            "description": "A conversation session with hierarchical summaries",
            "properties": {
                "sessionId": {"type": "string", "required": True},
                "title": {"type": "string", "required": False},
                "short_term_summary": {"type": "string", "required": False},
                "medium_term_summary": {"type": "string", "required": False},
                "long_term_summary": {"type": "string", "required": False},
                "message_count": {"type": "integer", "required": False},
                "topics": {"type": "string", "required": False}
            },
            "unique_identifiers": ["sessionId"]
        },
        "Technology": {
            "name": "Technology",
            "label": "Technology",
            "description": "A technology, framework, or tool",
            "properties": {
                "name": {"type": "string", "required": True},
                "category": {"type": "string", "required": False}
            },
            "unique_identifiers": ["name"]
        },
        "Task": {
            "name": "Task",
            "label": "Task",
            "description": "A task or action being worked on",
            "properties": {
                "description": {"type": "string", "required": True},
                "status": {"type": "string", "required": False}
            },
            "unique_identifiers": ["description"]
        },
        "Person": {
            "name": "Person",
            "label": "Person",
            "description": "A person mentioned in conversations",
            "properties": {
                "name": {"type": "string", "required": True},
                "role": {"type": "string", "required": False}
            },
            "unique_identifiers": ["name"]
        },
        "Agent": {
            "name": "Agent",
            "label": "Agent",
            "description": "An AI agent or sub-agent",
            "properties": {
                "id": {"type": "string", "required": True},
                "name": {"type": "string", "required": False}
            },
            "unique_identifiers": ["id"]
        }
    },
    "relationship_types": {
        "LEARNED_FROM": {
            "name": "LEARNED_FROM",
            "description": "Learning was derived from user interactions",
            "allowed_source_types": ["Learning"],
            "allowed_target_types": ["User"]
        },
        "IN_PROJECT": {
            "name": "IN_PROJECT",
            "description": "Entity is scoped to a project",
            "allowed_source_types": ["Learning", "MessageSession", "Task"],
            "allowed_target_types": ["Project"]
        },
        "FOR_GOAL": {
            "name": "FOR_GOAL",
            "description": "Entity is scoped to a goal",
            "allowed_source_types": ["Learning", "Task"],
            "allowed_target_types": ["Goal"]
        },
        "HAS_LEARNING": {
            "name": "HAS_LEARNING",
            "description": "Session contains a learning",
            "allowed_source_types": ["MessageSession"],
            "allowed_target_types": ["Learning"]
        },
        "IN_SESSION": {
            "name": "IN_SESSION",
            "description": "Entity occurred in this session",
            "allowed_source_types": ["Learning", "Task"],
            "allowed_target_types": ["MessageSession"]
        },
        "USES_TECHNOLOGY": {
            "name": "USES_TECHNOLOGY",
            "description": "Project or session uses a technology",
            "allowed_source_types": ["Project", "MessageSession"],
            "allowed_target_types": ["Technology"]
        },
        "WORKING_ON": {
            "name": "WORKING_ON",
            "description": "Agent or user is working on a task",
            "allowed_source_types": ["Agent", "User", "Person"],
            "allowed_target_types": ["Task"]
        },
        "INVOLVES": {
            "name": "INVOLVES",
            "description": "Session or project involves a person",
            "allowed_source_types": ["MessageSession", "Project"],
            "allowed_target_types": ["Person"]
        },
        "LED_BY": {
            "name": "LED_BY",
            "description": "Session led by an agent",
            "allowed_source_types": ["MessageSession"],
            "allowed_target_types": ["Agent"]
        }
    }
}


async def ensure_agent_learning_schema(
    user_id: str,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> Optional[str]:
    """
    Ensure AgentLearning schema exists for this organization.
    Creates it automatically if it doesn't exist.
    
    Returns:
        Schema ID if found or created, None on error
    """
    try:
        schema_service = SchemaService()
        
        # First, try to find existing schema
        all_schemas = await schema_service.get_active_schemas(
            user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            namespace_id=namespace_id
        )
        
        for schema in all_schemas:
            if schema.name == "AgentLearning" or "agent_learning" in getattr(schema, 'tags', []):
                logger.info(f"✅ Found existing AgentLearning schema: {schema.id}")
                return schema.id
        
        # Schema doesn't exist - create it automatically
        logger.info(f"📝 AgentLearning schema not found for organization {organization_id}, auto-registering...")
        
        created_schema = await schema_service.create_schema(
            schema_data=AGENT_LEARNING_SCHEMA,
            user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            namespace_id=namespace_id
        )
        
        if created_schema and created_schema.id:
            logger.info(f"✅ Auto-registered AgentLearning schema: {created_schema.id}")
            return created_schema.id
        else:
            logger.error(f"❌ Failed to auto-register AgentLearning schema")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error ensuring AgentLearning schema: {e}", exc_info=True)
        return None
