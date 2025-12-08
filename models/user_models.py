from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from datetime import datetime

class UserType(str, Enum):
    DEVELOPER_USER = "developerUser"
    USER = "user"
    AGENT = "agent"

class CreateUserRequest(BaseModel):
    """Request model for creating a user"""
    email: Optional[EmailStr] = None
    external_id: str
    metadata: Optional[Dict[str, Any]] = None
    type: UserType = UserType.DEVELOPER_USER

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "email": "user@example.com",
                "external_id": "user123",
                "metadata": {
                    "name": "John Doe",
                    "preferences": {"theme": "dark"}
                },
                "type": "developerUser"
            }
        }
    )

class UpdateUserRequest(BaseModel):
    """Request model for updating a user"""
    email: Optional[str] = None
    external_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    type: Optional[UserType] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "updated.user@example.com",
                "external_id": "updated_user_123",
                "metadata": {
                    "name": "Updated User",
                    "preferences": {"theme": "light"}
                },
                "type": "developerUser"
            }
        }
    )

class UserResponse(BaseModel):
    """Response model for user operations"""
    code: int = Field(..., description="HTTP status code")
    status: str = Field(..., description="'success' or 'error'")
    user_id: Optional[str] = None
    email: Optional[str] = None
    external_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error: Optional[str] = None
    details: Optional[Any] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": 200,
                "status": "success",
                "user_id": "abc123",
                "email": "user@example.com",
                "external_id": "user123",
                "metadata": {
                    "name": "John Doe",
                    "preferences": {"theme": "dark"}
                },
                "created_at": "2024-03-20T10:00:00.000Z",
                "updated_at": "2024-03-20T10:00:00.000Z",
                "error": None,
                "details": None
            }
        }
    )

    @classmethod
    def success(cls, **kwargs):
        return cls(code=200, status="success", **kwargs)

    @classmethod
    def failure(cls, error: str, code: int = 400, details: Any = None, **kwargs):
        return cls(code=code, status="error", error=error, details=details, **kwargs)

class UserListResponse(BaseModel):
    code: int = Field(..., description="HTTP status code")
    status: str = Field(..., description="'success' or 'error'")
    data: Optional[List[UserResponse]] = None
    total: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None
    error: Optional[str] = None
    details: Optional[Any] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": 200,
                "status": "success",
                "data": [
                    {
                        "user_id": "abc123",
                        "email": "user1@example.com",
                        "external_id": "user123",
                        "metadata": {
                            "name": "John Doe",
                            "preferences": {"theme": "dark"}
                        },
                        "created_at": "2024-03-20T10:00:00.000Z",
                        "updated_at": "2024-03-20T10:00:00.000Z"
                    }
                ],
                "total": 1,
                "page": 1,
                "page_size": 10,
                "error": None,
                "details": None
            }
        }
    )

    @classmethod
    def success(cls, data: List[UserResponse], total: int, page: int, page_size: int):
        return cls(code=200, status="success", data=data, total=total, page=page, page_size=page_size, error=None, details=None)

    @classmethod
    def failure(cls, error: str, code: int = 400, details: Any = None):
        return cls(code=code, status="error", data=None, total=None, page=None, page_size=None, error=error, details=details)

class ErrorDetail(BaseModel):
    """Error response model"""
    code: int
    detail: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": 400,
                "detail": "Invalid request data"
            }
        }
    )

class DeleteUserResponse(BaseModel):
    code: int = Field(..., description="HTTP status code")
    status: str = Field(..., description="'success' or 'error'")
    user_id: Optional[str] = Field(None, description="ID of the user attempted to delete")
    message: Optional[str] = Field(None, description="Success or error message")
    error: Optional[str] = Field(None, description="Error message if failed")
    details: Optional[Any] = Field(None, description="Additional error details or context")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": 200,
                "status": "success",
                "user_id": "abc123",
                "message": "User and association deleted successfully",
                "error": None,
                "details": None
            }
        }
    )

    @classmethod
    def success(cls, user_id: str, message: str = "User and association deleted successfully"):
        return cls(code=200, status="success", user_id=user_id, message=message, error=None, details=None)

    @classmethod
    def failure(cls, user_id: Optional[str], error: str, code: int = 400, details: Any = None):
        return cls(code=code, status="error", user_id=user_id, message=None, error=error, details=details)

class BatchUserCreateRequest(BaseModel):
    users: List[CreateUserRequest] 

class InteractionLimitsErrorResponse(BaseModel):
    """Error response for interaction limits"""
    error: str = Field(..., description="Main error message")
    message: str = Field(..., description="Detailed error message")
    current_count: Optional[int] = Field(None, description="Current usage count")
    limit: Optional[int] = Field(None, description="Usage limit")
    tier: Optional[str] = Field(None, description="Customer tier")
    is_trial: Optional[bool] = Field(None, description="Whether user is on trial")
    subscription_status: Optional[str] = Field(None, description="Stripe subscription status")

class InteractionLimitsWelcomeResponse(BaseModel):
    """Welcome response for new users"""
    message: str = Field(..., description="Welcome message")
    trial_started: Optional[bool] = Field(None, description="Whether trial was started")
    days_remaining: Optional[int] = Field(None, description="Days remaining in trial")
    trial_end: Optional[int] = Field(None, description="Trial end timestamp")

class InteractionLimitsResult(BaseModel):
    """Standardized result type for interaction limits checking"""
    response: Union[InteractionLimitsErrorResponse, InteractionLimitsWelcomeResponse, None]
    status_code: int
    is_error: bool

    @classmethod
    def success(cls, welcome_response: Optional[InteractionLimitsWelcomeResponse] = None) -> "InteractionLimitsResult":
        """Create a success result"""
        return cls(
            response=welcome_response,
            status_code=200,
            is_error=False
        )

    @classmethod
    def error(cls, error_response: InteractionLimitsErrorResponse, status_code: int = 403) -> "InteractionLimitsResult":
        """Create an error result"""
        return cls(
            response=error_response,
            status_code=status_code,
            is_error=True
        )

    def to_tuple(self) -> Optional[Tuple[Dict[str, Any], int, bool]]:
        """Convert to the legacy tuple format for backward compatibility"""
        if self.response is None:
            return None
        
        response_dict = self.response.model_dump()
        return response_dict, self.status_code, self.is_error

    @classmethod
    def from_tuple(cls, tuple_result: Optional[Tuple[Dict[str, Any], int, bool]]) -> "InteractionLimitsResult":
        """Create from legacy tuple format"""
        if tuple_result is None:
            return cls.success()
        
        response_dict, status_code, is_error = tuple_result
        
        if is_error:
            error_response = InteractionLimitsErrorResponse(**response_dict)
            return cls.error(error_response, status_code)
        else:
            welcome_response = InteractionLimitsWelcomeResponse(**response_dict)
            return cls.success(welcome_response) 