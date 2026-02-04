"""
Message models for chat message storage and processing
"""
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any, Literal, Union, TYPE_CHECKING
from datetime import datetime, timezone
from enum import Enum
from models.shared_types import MemoryMetadata, MessageRole, UserMemoryCategory, AssistantMemoryCategory, MemoryPolicy

# Import GraphGeneration with TYPE_CHECKING to avoid circular import at module load time
# It will be resolved at runtime via model_rebuild()
if TYPE_CHECKING:
    from models.memory_models import GraphGeneration


class MessageRequest(BaseModel):
    """Request model for storing a chat message"""
    content: Union[str, List[Dict[str, Any]]] = Field(
        ...,
        description="The content of the chat message - can be a simple string or structured content objects"
    )
    role: MessageRole = Field(
        ...,
        description="Role of the message sender (user or assistant)"
    )
    sessionId: str = Field(
        ...,
        description="Session ID to group related messages in a conversation"
    )
    metadata: Optional[MemoryMetadata] = Field(
        None,
        description="Optional metadata for the message (topics, location, etc.)"
    )
    context: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Optional context for the message (conversation history or relevant context)"
    )
    relationships_json: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Optional array of relationships for Graph DB (Neo4j)"
    )
    
    # Processing control
    process_messages: bool = Field(
        default=True, 
        description="Whether to process messages into memories (true) or just store them (false). Default is true."
    )
    
    # Graph generation control (same as AddMemoryRequest)
    memory_policy: Optional[MemoryPolicy] = Field(
        default=None,
        description="Unified policy for graph generation and OMO safety. "
                   "Use mode='auto' (LLM extraction), 'manual' (exact nodes), "
                   "or 'hybrid' (LLM with constraints). Includes consent, risk, and ACL settings."
    )
    
    graph_generation: Optional["GraphGeneration"] = Field(
        default=None,
        description="DEPRECATED: Use 'memory_policy' instead. Legacy graph generation configuration.",
        json_schema_extra={"deprecated": True}
    )
    
    # Multi-tenant fields
    organization_id: Optional[str] = Field(
        default=None,
        description="Optional organization ID for multi-tenant message scoping"
    )
    namespace_id: Optional[str] = Field(
        default=None,
        description="Optional namespace ID for multi-tenant message scoping"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "content": "Can you help me plan the Q4 product roadmap?",
                    "role": "user",
                    "sessionId": "session_123",
                    "process_messages": True,
                    "metadata": {
                        "topics": ["product", "planning", "roadmap"],
                        "location": "Office"
                    }
                },
                {
                    "content": [
                        {
                            "type": "text",
                            "text": "Here's a story about a character trapped in an emerald egg..."
                        }
                    ],
                    "role": "user",
                    "sessionId": "session_456",
                    "process_messages": True,
                    "metadata": {
                        "topics": ["creative", "storytelling"],
                        "location": "Home"
                    }
                }
            ]
        }
    )


class MessageResponse(BaseModel):
    """Response model for message storage"""
    objectId: str = Field(..., description="Parse Server objectId of the stored message")
    sessionId: str = Field(..., description="Session ID of the conversation")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Content of the message - can be a simple string or structured content objects")
    createdAt: datetime = Field(..., description="When the message was created")
    processing_status: str = Field(
        default="queued",
        description="Status of background processing (queued, analyzing, completed, failed)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "objectId": "msg_abc123",
                "sessionId": "session_123", 
                "role": "user",
                "content": "Can you help me plan the Q4 product roadmap?",
                "createdAt": "2024-01-15T10:30:00Z",
                "processing_status": "queued"
            }
        }
    )


class ConversationSummaryResponse(BaseModel):
    """Hierarchical conversation summaries for context window compression"""
    short_term: Optional[str] = Field(None, description="Summary of last 15 messages")
    medium_term: Optional[str] = Field(None, description="Summary of last ~100 messages")
    long_term: Optional[str] = Field(None, description="Full session summary")
    topics: List[str] = Field(default_factory=list, description="Key topics discussed")
    last_updated: Optional[datetime] = Field(None, description="When summaries were last updated")


class SessionSummaryResponse(BaseModel):
    """Response model for session summarization endpoint"""
    session_id: str = Field(..., description="Session ID of the conversation")
    summaries: ConversationSummaryResponse = Field(..., description="Hierarchical conversation summaries")
    ai_agent_note: str = Field(
        ..., 
        description="Instructions for AI agents on how to search for more details about this conversation"
    )
    from_cache: bool = Field(..., description="Whether summaries were retrieved from cache (true) or just generated (false)")
    message_count: Optional[int] = Field(None, description="Number of messages summarized (only present if just generated)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "session_123",
                "summaries": {
                    "short_term": "User requested help planning Q4 product roadmap. Assistant offered to help identify key objectives.",
                    "medium_term": "Ongoing product planning discussion for Q4, covering objectives, timelines, and resource allocation.",
                    "long_term": "Product planning and strategy conversation focused on Q4 roadmap development and execution.",
                    "topics": ["product planning", "Q4 roadmap", "objectives", "strategy"],
                    "last_updated": "2024-01-15T10:31:00Z"
                },
                "ai_agent_note": "To find more details about this conversation, search memories with metadata filter: sessionId='session_123'",
                "from_cache": True
            }
        }
    )


class MessageHistoryResponse(BaseModel):
    """Response model for retrieving message history"""
    sessionId: str = Field(..., description="Session ID of the conversation")
    messages: List[MessageResponse] = Field(..., description="List of messages in chronological order")
    total_count: int = Field(..., description="Total number of messages in the session")
    summaries: Optional[ConversationSummaryResponse] = Field(
        None,
        description="Hierarchical conversation summaries for context compression"
    )
    context_for_llm: Optional[str] = Field(
        None,
        description="Pre-formatted compressed context ready for LLM consumption (summaries + recent messages)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sessionId": "session_123",
                "messages": [
                    {
                        "objectId": "msg_abc123",
                        "sessionId": "session_123",
                        "role": "user", 
                        "content": "Can you help me plan the Q4 product roadmap?",
                        "createdAt": "2024-01-15T10:30:00Z",
                        "processing_status": "completed"
                    },
                    {
                        "objectId": "msg_def456",
                        "sessionId": "session_123",
                        "role": "assistant",
                        "content": "I'd be happy to help you plan your Q4 roadmap. Let's start by identifying your key objectives.",
                        "createdAt": "2024-01-15T10:31:00Z", 
                        "processing_status": "completed"
                    }
                ],
                "total_count": 2,
                "summaries": {
                    "short_term": "User requested help planning Q4 product roadmap",
                    "medium_term": "Ongoing product planning discussion for Q4",
                    "long_term": "Product planning and strategy conversation",
                    "topics": ["product", "roadmap", "planning", "Q4"]
                },
                "context_for_llm": "FULL SESSION: Product planning and strategy conversation\nRECENT (last ~100): Ongoing product planning discussion for Q4\nCURRENT (last 15): User requested help planning Q4 product roadmap"
            }
        }
    )


class MessageAnalysisResult(BaseModel):
    """Result from AI analysis of a message"""
    message_id: Optional[str] = Field(None, description="ID of the message that was analyzed")
    session_id: Optional[str] = Field(None, description="Session ID the message belongs to")
    is_memory_worthy: bool = Field(..., description="Whether the message should become a memory")
    memory_content: Optional[str] = Field(None, description="Extracted memory content if worthy")
    category: Optional[str] = Field(None, description="Determined memory category")
    role: MessageRole = Field(..., description="Role that determines category options")
    confidence_score: float = Field(..., description="Confidence in the analysis (0.0-1.0)")
    reasoning: Optional[str] = Field(None, description="AI reasoning for the decision")
    topics: Optional[List[str]] = Field(None, description="Relevant topics/keywords extracted from the message")
    hierarchical_structures: Optional[Union[str, List]] = Field(None, description="Hierarchical categorization")
    
    # User preference learning
    has_user_preference_learning: bool = Field(default=False, description="Whether user preference learning was detected")
    user_learning_content: Optional[str] = Field(None, description="User preference learning content")
    user_learning_type: Optional[str] = Field(None, description="Type of user preference learning")
    user_learning_confidence: float = Field(default=0.0, description="Confidence in user preference learning (0.0-1.0)")
    user_learning_evidence: Optional[str] = Field(None, description="Evidence for user preference learning")
    
    # Agent performance learning
    has_performance_learning: bool = Field(default=False, description="Whether agent performance learning was detected")
    performance_learning_content: Optional[str] = Field(None, description="Agent performance learning content")
    performance_learning_type: Optional[str] = Field(None, description="Type of agent performance learning")
    performance_learning_confidence: float = Field(default=0.0, description="Confidence in agent performance learning (0.0-1.0)")
    inefficient_approach: Optional[str] = Field(None, description="The inefficient approach that was identified")
    efficient_approach: Optional[str] = Field(None, description="The efficient approach discovered")
    performance_context: Optional[str] = Field(None, description="Context for the performance learning")
    performance_scope: Optional[str] = Field(None, description="Scope of performance learning (project, goal, user, global)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_memory_worthy": True,
                "memory_content": "User wants to plan Q4 product roadmap focusing on key objectives and priorities",
                "category": "goal",
                "role": "user",
                "confidence_score": 0.85,
                "reasoning": "This message expresses a clear goal/objective for Q4 planning"
            }
        }
    )


# Update the existing AddMemoryRequest to include role and category fields
class EnhancedAddMemoryRequest(BaseModel):
    """Enhanced AddMemoryRequest with role and category fields for LLM structured output"""
    content: str = Field(..., description="The content of the memory item")
    type: str = Field(default="text", description="Memory type")
    role: Optional[MessageRole] = Field(None, description="Role that generated this memory")
    category: Optional[str] = Field(None, description="Memory category based on role")
    metadata: Optional[MemoryMetadata] = Field(None, description="Memory metadata")
    
    @field_validator('category')
    @classmethod
    def validate_category_for_role(cls, v, info):
        """Validate that category matches the role"""
        if v is None:
            return v
            
        role = info.data.get('role')
        if role == MessageRole.USER:
            if v not in [cat.value for cat in UserMemoryCategory]:
                raise ValueError(f"Invalid category '{v}' for user role. Must be one of: {[cat.value for cat in UserMemoryCategory]}")
        elif role == MessageRole.ASSISTANT:
            if v not in [cat.value for cat in AssistantMemoryCategory]:
                raise ValueError(f"Invalid category '{v}' for assistant role. Must be one of: {[cat.value for cat in AssistantMemoryCategory]}")
        
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "User wants to plan Q4 product roadmap focusing on key objectives and priorities",
                "type": "text",
                "role": "user", 
                "category": "goal",
                "metadata": {
                    "topics": ["product", "planning", "roadmap", "Q4"],
                    "hierarchical_structures": "Business/Planning/Product/Roadmap",
                    "sourceType": "chat_message"
                }
            }
        }
    )


# Rebuild models to resolve forward references
# This must be done after all models are defined
def _rebuild_message_models():
    """Rebuild Pydantic models to resolve forward references"""
    try:
        # Import GraphGeneration at runtime to resolve forward reference
        from models.memory_models import GraphGeneration
        
        # Rebuild the model with the resolved types
        MessageRequest.model_rebuild()
    except Exception as e:
        # Log but don't fail - this is a safety measure
        import logging
        logging.warning(f"Failed to rebuild MessageRequest model: {e}")


# Call rebuild at module load time
_rebuild_message_models()
