"""
Message models for chat message storage and processing
"""
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime, timezone
from enum import Enum
from models.shared_types import MemoryMetadata, MessageRole, UserMemoryCategory, AssistantMemoryCategory


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
        description="Optional metadata for the message"
    )
    
    # Processing control
    process_messages: bool = Field(
        default=True, 
        description="Whether to process messages into memories (true) or just store them (false). Default is true."
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


class MessageHistoryResponse(BaseModel):
    """Response model for retrieving message history"""
    sessionId: str = Field(..., description="Session ID of the conversation")
    messages: List[MessageResponse] = Field(..., description="List of messages in chronological order")
    total_count: int = Field(..., description="Total number of messages in the session")
    
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
                "total_count": 2
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
    hierarchical_structures: Optional[str] = Field(None, description="Hierarchical categorization")
    
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
