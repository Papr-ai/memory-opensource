"""
Global exception handlers for the FastAPI application.
Catches database connection errors and returns 500 status instead of crashing.
"""

import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from pymongo.errors import (
    AutoReconnect, 
    ServerSelectionTimeoutError, 
    ConnectionFailure, 
    OperationFailure
)
from neo4j.exceptions import (
    ServiceUnavailable,
    AuthError,
    ClientError,
    TransientError
)
from qdrant_client.http.exceptions import UnexpectedResponse
import time

logger = logging.getLogger(__name__)

# Track error rates to prevent spam
_error_counts = {}
_error_window = 60  # seconds
_max_errors_per_window = 10

def is_error_rate_limited(error_type: str) -> bool:
    """Check if we're logging too many errors of this type"""
    current_time = time.time()
    
    if error_type not in _error_counts:
        _error_counts[error_type] = {"count": 0, "first_error": current_time}
        return False
    
    error_info = _error_counts[error_type]
    
    # Reset if outside window
    if current_time - error_info["first_error"] > _error_window:
        error_info["count"] = 0
        error_info["first_error"] = current_time
        return False
    
    error_info["count"] += 1
    return error_info["count"] > _max_errors_per_window

async def mongodb_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle MongoDB connection errors gracefully"""
    error_type = type(exc).__name__
    
    if not is_error_rate_limited(f"mongodb_{error_type}"):
        logger.error(f"MongoDB error in {request.url.path}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "status": "error",
            "error": "Database connection temporarily unavailable",
            "details": {
                "error_type": "database_connection_error",
                "message": "The service is experiencing database connectivity issues. Please try again later."
            }
        }
    )

async def neo4j_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Neo4j connection errors gracefully"""
    error_type = type(exc).__name__
    
    if not is_error_rate_limited(f"neo4j_{error_type}"):
        logger.error(f"Neo4j error in {request.url.path}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "status": "error",
            "error": "Graph database temporarily unavailable",
            "details": {
                "error_type": "database_connection_error",
                "message": "The service is experiencing graph database connectivity issues. Please try again later."
            }
        }
    )

async def qdrant_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Qdrant connection errors gracefully"""
    error_type = type(exc).__name__
    
    if not is_error_rate_limited(f"qdrant_{error_type}"):
        logger.error(f"Qdrant error in {request.url.path}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "status": "error",
            "error": "Vector database temporarily unavailable",
            "details": {
                "error_type": "database_connection_error",
                "message": "The service is experiencing vector database connectivity issues. Please try again later."
            }
        }
    )

async def general_database_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general database errors gracefully"""
    error_type = type(exc).__name__
    
    if not is_error_rate_limited(f"general_db_{error_type}"):
        logger.error(f"Database error in {request.url.path}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "status": "error",
            "error": "Database temporarily unavailable",
            "details": {
                "error_type": "database_error",
                "message": "The service is experiencing database issues. Please try again later."
            }
        }
    )

def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app"""
    
    # MongoDB exceptions
    app.add_exception_handler(AutoReconnect, mongodb_exception_handler)
    app.add_exception_handler(ServerSelectionTimeoutError, mongodb_exception_handler)
    app.add_exception_handler(ConnectionFailure, mongodb_exception_handler)
    app.add_exception_handler(OperationFailure, mongodb_exception_handler)
    
    # Neo4j exceptions
    app.add_exception_handler(ServiceUnavailable, neo4j_exception_handler)
    app.add_exception_handler(AuthError, neo4j_exception_handler)
    app.add_exception_handler(ClientError, neo4j_exception_handler)
    app.add_exception_handler(TransientError, neo4j_exception_handler)
    
    # Qdrant exceptions
    app.add_exception_handler(UnexpectedResponse, qdrant_exception_handler)
    
    logger.info("Database exception handlers registered successfully")

    # Catch-all for any unhandled exceptions to avoid server crashes
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        error_type = type(exc).__name__
        if not is_error_rate_limited(f"general_{error_type}"):
            logger.error(f"Unhandled error in {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "status": "error",
                "error": "Internal server error",
                "details": {
                    "error_type": error_type,
                    "message": "An unexpected error occurred. The incident has been logged."
                }
            }
        )

    # Register the catch-all handler last so more specific handlers win
    app.add_exception_handler(Exception, general_exception_handler)

class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors"""
    pass

def safe_database_operation(operation_name: str):
    """Decorator to safely execute database operations"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except (AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure,
                   ServiceUnavailable, AuthError, ClientError, TransientError, UnexpectedResponse) as e:
                logger.error(f"Database error in {operation_name}: {e}")
                raise DatabaseConnectionError(f"Database operation failed: {operation_name}")
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {e}")
                raise
        return wrapper
    return decorator 