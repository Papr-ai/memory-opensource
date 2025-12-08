from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime
from models.parse_server import ParsePointer, UserFeedbackLog
from models.shared_types import FeedbackType, FeedbackSource

class FeedbackData(BaseModel):
    """Developer-friendly feedback data model for requests"""
    userMessage: Optional[ParsePointer] = None  # Pointer to PostMessage
    assistantMessage: Optional[ParsePointer] = None  # Pointer to PostMessage
    feedbackType: FeedbackType
    feedbackValue: Optional[str] = None
    feedbackScore: Optional[float] = None
    feedbackText: Optional[str] = None
    feedbackSource: FeedbackSource
    citedMemoryIds: Optional[List[str]] = Field(default_factory=list)
    citedNodeIds: Optional[List[str]] = Field(default_factory=list)
    feedbackProcessed: Optional[bool] = None
    feedbackImpact: Optional[str] = None

    @field_validator('feedbackScore')
    @classmethod
    def validate_feedback_score(cls, v, values):
        """Validate feedback score based on feedback type"""
        if v is None:
            return v
            
        feedback_type = values.data.get('feedbackType')
        if feedback_type in [FeedbackType.THUMBS_UP, FeedbackType.THUMBS_DOWN]:
            if v not in [-1.0, 1.0]:
                raise ValueError("Thumbs feedback must be -1.0 (down) or +1.0 (up)")
        elif feedback_type == FeedbackType.RATING:
            if not (1.0 <= v <= 5.0):
                raise ValueError("Rating must be between 1.0 and 5.0")
        
        return v


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback on search results"""
    search_id: str = Field(
        ..., 
        description="The search_id from SearchResponse that this feedback relates to"
    )
    feedbackData: FeedbackData = Field(
        ...,
        description="The feedback data containing all feedback information"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Internal user ID (if not provided, will be resolved from authentication)"
    )
    external_user_id: Optional[str] = Field(
        default=None,
        description="External user ID for developer API keys acting on behalf of end users"
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Optional organization ID for multi-tenant feedback scoping. When provided, feedback is scoped to this organization."
    )
    namespace_id: Optional[str] = Field(
        default=None,
        description="Optional namespace ID for multi-tenant feedback scoping. When provided, feedback is scoped to this namespace."
    )

    @field_validator('feedbackData')
    @classmethod
    def validate_feedback_data(cls, v):
        """Validate FeedbackData has required fields"""
        if not v.feedbackType:
            raise ValueError("feedbackType is required in FeedbackData")
        if not v.feedbackSource:
            raise ValueError("feedbackSource is required in FeedbackData")
        return v

    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id - can be None as it will be resolved from authentication"""
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "search_id": "abc123def456",
                "feedbackData": {
                    "userMessage": {
                        "objectId": "abc123def456",
                        "className": "PostMessage",
                        "__type": "Pointer"
                    },
                    "assistantMessage": {
                        "objectId": "abc123def456",
                        "className": "PostMessage",
                        "__type": "Pointer"
                    },
                    "feedbackType": "thumbs_up",
                    "feedbackValue": "helpful",
                    "feedbackScore": 1,
                    "feedbackText": "This answer was very helpful and accurate",
                    "feedbackSource": "inline",
                    "citedMemoryIds": ["mem_123", "mem_456"],
                    "citedNodeIds": ["node_123", "node_456"],
                    "feedbackProcessed": True,
                    "feedbackImpact": "positive"
                },
                "user_id": "abc123def456",
                "external_user_id": "dev_api_key_123"
            }
        }
    )

class BatchFeedbackRequest(BaseModel):
    """Request model for submitting multiple feedback items"""
    feedback_items: List[FeedbackRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of feedback items to submit"
    )
    session_context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Session-level context for batch feedback"
    )

class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    code: int = Field(..., description="HTTP status code")
    status: str = Field(..., description="'success' or 'error'")
    feedback_id: Optional[str] = Field(None, description="Unique feedback ID")
    message: str = Field(..., description="Human-readable message")
    error: Optional[str] = Field(None, description="Error message if status is 'error'")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    @classmethod
    def success(cls, feedback_id: str, message: str = "Feedback submitted successfully", details: Optional[Dict[str, Any]] = None) -> "FeedbackResponse":
        return cls(
            code=200,
            status="success",
            feedback_id=feedback_id,
            message=message,
            error=None,
            details=details
        )

    @classmethod
    def failure(cls, error: str, code: int = 400, details: Optional[Dict[str, Any]] = None) -> "FeedbackResponse":
        return cls(
            code=code,
            status="error",
            feedback_id=None,
            message="Failed to submit feedback",
            error=error,
            details=details
        )

class BatchFeedbackResponse(BaseModel):
    """Response model for batch feedback submission"""
    code: int = Field(..., description="HTTP status code")
    status: str = Field(..., description="'success' or 'error'")
    feedback_ids: List[str] = Field(default_factory=list, description="List of feedback IDs")
    successful_count: int = Field(0, description="Number of successfully processed feedback items")
    failed_count: int = Field(0, description="Number of failed feedback items")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of error details")
    message: str = Field(..., description="Human-readable message")
    error: Optional[str] = Field(None, description="Error message if status is 'error'")

    @classmethod
    def success(cls, feedback_ids: List[str], successful_count: int, failed_count: int = 0, errors: List[Dict[str, Any]] = None) -> "BatchFeedbackResponse":
        return cls(
            code=200 if failed_count == 0 else 207,
            status="success",
            feedback_ids=feedback_ids,
            successful_count=successful_count,
            failed_count=failed_count,
            errors=errors or [],
            message=f"Processed {successful_count} feedback items successfully"
        )

    @classmethod
    def failure(cls, error: str, code: int = 400) -> "BatchFeedbackResponse":
        return cls(
            code=code,
            status="error",
            feedback_ids=[],
            successful_count=0,
            failed_count=0,
            errors=[],
            message="Failed to process batch feedback",
            error=error
        )

