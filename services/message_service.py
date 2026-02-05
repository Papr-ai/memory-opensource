"""
Core message processing service
Handles storing messages in PostMessage class and managing processing pipeline
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx
from fastapi import HTTPException

from models.message_models import MessageRequest, MessageResponse, MessageHistoryResponse
from models.parse_server import ParsePointer
from services.logger_singleton import LoggerSingleton
from services.multi_tenant_utils import apply_multi_tenant_scoping_to_metadata
from models.shared_types import MemoryMetadata
import os

logger = LoggerSingleton.get_logger(__name__)

# Parse Server configuration
PARSE_SERVER_URL = os.getenv("PARSE_SERVER_URL") or os.getenv("PARSE_SERVER_URL")
PARSE_APPLICATION_ID = os.getenv("PARSE_APPLICATION_ID") 
PARSE_MASTER_KEY = os.getenv("PARSE_MASTER_KEY")

# Ensure PARSE_SERVER_URL has protocol
if PARSE_SERVER_URL and not PARSE_SERVER_URL.startswith(('http://', 'https://')):
    PARSE_SERVER_URL = f"https://{PARSE_SERVER_URL}"

HEADERS = {
    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
    "X-Parse-Master-Key": PARSE_MASTER_KEY,
    "Content-Type": "application/json"
}


async def store_message_in_parse(
    message_request: MessageRequest,
    user_id: str,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> MessageResponse:
    """
    Store a chat message in Parse Server PostMessage class
    
    Args:
        message_request: The message request data
        user_id: Internal user ID
        workspace_id: Optional workspace ID
        organization_id: Optional organization ID for multi-tenant
        namespace_id: Optional namespace ID for multi-tenant
        
    Returns:
        MessageResponse with stored message details
    """
    try:
        # Apply multi-tenant scoping to metadata
        auth_context = {
            'organization_id': organization_id,
            'namespace_id': namespace_id,
            'is_legacy_auth': organization_id is None,
            'auth_type': 'organization' if organization_id else 'legacy'
        }
        
        scoped_metadata = apply_multi_tenant_scoping_to_metadata(
            message_request.metadata or MemoryMetadata(),
            auth_context
        )
        
        # Get or create chat session first
        chat = await get_or_create_chat_session(message_request.sessionId, user_id, workspace_id, organization_id, namespace_id)
        if not chat:
            logger.error(f"Failed to get/create chat session for {message_request.sessionId}")
            raise HTTPException(status_code=500, detail="Failed to create chat session")

        # Handle both string and structured content
        if isinstance(message_request.content, str):
            # Simple string content goes in 'message' field
            message_field = message_request.content
            content_field = None
        else:
            # Structured content goes in 'content' field, extract text for 'message' field
            content_field = message_request.content
            # Extract text from structured content for the message field
            message_field = ""
            for item in message_request.content:
                if item.get("type") == "text" and "text" in item:
                    message_field += item["text"] + " "
            message_field = message_field.strip()
        
        # Create PostMessage data structure (matching GraphQL schema)
        post_message_data = {
            "message": message_field,  # Use 'message' field for text content
            "messageRole": message_request.role.value,  # Use 'messageRole' as per GraphQL schema
            "processingStatus": "pending" if message_request.process_messages else "stored_only",
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "chat": {  # Link to chat session
                "__type": "Pointer",
                "className": "Chat",
                "objectId": chat["objectId"]
            },
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat()
        }
        
        # Add workspace pointer if provided
        if workspace_id:
            post_message_data["workspace"] = {
                "__type": "Pointer", 
                "className": "WorkSpace",
                "objectId": workspace_id
            }

        # Add organization and namespace pointers if provided
        if organization_id:
            post_message_data["organization"] = {
                "__type": "Pointer",
                "className": "Organization",
                "objectId": organization_id
            }

        if namespace_id:
            post_message_data["namespace"] = {
                "__type": "Pointer",
                "className": "Namespace",
                "objectId": namespace_id
            }
        
        # Add structured content if present
        if content_field is not None:
            # Wrap the content array in an object structure that Parse expects
            post_message_data["content"] = {
                "type": "structured",
                "data": content_field
            }
        
        # Add metadata if present
        if scoped_metadata:
            metadata_dict = scoped_metadata.model_dump(exclude_none=True)
            post_message_data["metadata"] = json.dumps(metadata_dict)
            
            # Add ACL fields from metadata
            for acl_field in [
                "external_user_read_access", "external_user_write_access",
                "user_read_access", "user_write_access", 
                "workspace_read_access", "workspace_write_access",
                "role_read_access", "role_write_access",
                "namespace_read_access", "namespace_write_access",
                "organization_read_access", "organization_write_access"
            ]:
                if hasattr(scoped_metadata, acl_field):
                    value = getattr(scoped_metadata, acl_field)
                    if value:
                        post_message_data[acl_field] = value
        
        # Store in Parse Server
        url = f"{PARSE_SERVER_URL}/parse/classes/PostMessage"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=HEADERS, json=post_message_data)
            response.raise_for_status()
            
            result = response.json()
            object_id = result.get("objectId")
            
            if not object_id:
                raise ValueError("No objectId returned from Parse Server")
            
            # Update chat message count
            await update_chat_message_count(message_request.sessionId, user_id, increment=1)
            
            logger.info(f"Successfully stored message in PostMessage: {object_id}")
            
            return MessageResponse(
                objectId=object_id,
                sessionId=message_request.sessionId,
                role=message_request.role,
                content=message_request.content,
                createdAt=datetime.now(timezone.utc),
                processing_status=post_message_data["processingStatus"]
            )
            
    except httpx.HTTPStatusError as e:
        # Get the response body for more detailed error information
        try:
            error_body = e.response.text
            logger.error(f"Error storing message in Parse Server: {e.response.status_code} - {error_body}")
        except:
            logger.error(f"Error storing message in Parse Server: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error storing message in Parse Server: {str(e)}")
        raise


async def get_session_messages(
    session_id: str,
    user_id: str,
    limit: int = 50,
    skip: int = 0,
    workspace_id: Optional[str] = None
) -> MessageHistoryResponse:
    """
    Retrieve message history for a specific session
    
    Args:
        session_id: Session ID to retrieve messages for
        user_id: User ID for access control
        limit: Maximum number of messages to return
        skip: Number of messages to skip (for pagination)
        workspace_id: Optional workspace ID for filtering
        
    Returns:
        MessageHistoryResponse with message history
    """
    try:
        # First, find the Chat record by sessionId
        chat_query = {
            "sessionId": session_id,
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        }
        
        async with httpx.AsyncClient() as client:
            chat_response = await client.get(
                f"{PARSE_SERVER_URL}/parse/classes/Chat",
                headers={
                    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                    "X-Parse-Master-Key": PARSE_MASTER_KEY,
                    "Content-Type": "application/json"
                },
                params={"where": json.dumps(chat_query), "limit": 1}
            )
            
            if chat_response.status_code != 200:
                logger.error(f"Failed to find chat session: {chat_response.status_code} - {chat_response.text}")
                return MessageHistoryResponse(messages=[], total_count=0, has_more=False)
            
            chat_data = chat_response.json()
            if not chat_data.get("results"):
                logger.info(f"No chat session found for sessionId: {session_id}")
                return MessageHistoryResponse(sessionId=session_id, messages=[], total_count=0)
            
            chat_record = chat_data["results"][0]
            chat_object_id = chat_record["objectId"]
            
            # Extract summaries if available
            summaries_data = chat_record.get("summaries")
            summaries = None
            context_for_llm = None
            
            if summaries_data:
                from models.message_models import ConversationSummaryResponse
                summaries = ConversationSummaryResponse(
                    short_term=summaries_data.get("short_term"),
                    medium_term=summaries_data.get("medium_term"),
                    long_term=summaries_data.get("long_term"),
                    topics=summaries_data.get("topics", []),
                    last_updated=datetime.fromisoformat(summaries_data["last_updated"].replace('Z', '+00:00')) if summaries_data.get("last_updated") else None
                )
        
            # Build query filters - query by chat pointer
            where_conditions = {
                "chat": {
                    "__type": "Pointer",
                    "className": "Chat",
                    "objectId": chat_object_id
                },
                "user": {
                    "__type": "Pointer",
                    "className": "_User", 
                    "objectId": user_id
                }
            }
            
            if workspace_id:
                where_conditions["workspace"] = {
                    "__type": "Pointer",
                    "className": "WorkSpace",
                    "objectId": workspace_id
                }
            
            # Query Parse Server for PostMessages
            url = f"{PARSE_SERVER_URL}/parse/classes/PostMessage"
            params = {
                "where": json.dumps(where_conditions),
                "order": "createdAt",  # Chronological order
                "limit": limit,
                "skip": skip,
                "keys": "objectId,message,content,messageRole,createdAt,processingStatus"
            }
            
            response = await client.get(url, headers=HEADERS, params=params)
            response.raise_for_status()
            
            result = response.json()
            messages_data = result.get("results", [])
            
            # Convert to MessageResponse objects
            messages = []
            for msg_data in messages_data:
                try:
                    # Determine content - prioritize structured content over simple message
                    content_value = msg_data.get("content")  # Structured content
                    if content_value is not None:
                        # Check if it's our wrapped structured content
                        if isinstance(content_value, dict) and content_value.get("type") == "structured":
                            content_value = content_value.get("data", content_value)
                    else:
                        content_value = msg_data.get("message", "")  # Fallback to simple message
                    
                    message_response = MessageResponse(
                        objectId=msg_data["objectId"],
                        sessionId=session_id,  # Use the sessionId parameter
                        role=msg_data.get("messageRole", "unknown"),  # Use messageRole field
                        content=content_value,
                        createdAt=datetime.fromisoformat(msg_data["createdAt"].replace('Z', '+00:00')),
                        processing_status=msg_data.get("processingStatus", "unknown")
                    )
                    messages.append(message_response)
                except Exception as e:
                    logger.warning(f"Error parsing message {msg_data.get('objectId')}: {str(e)}")
                    continue
            
            # Get total count for pagination
            count_params = {
                "where": json.dumps(where_conditions),
                "count": 1,
                "limit": 0
            }
            
            count_response = await client.get(url, headers=HEADERS, params=count_params)
            count_response.raise_for_status()
            total_count = count_response.json().get("count", len(messages))
            
            # Generate context_for_llm if summaries exist
            if summaries:
                # Format recent messages (last 5-10)
                recent_messages_text = ""
                for msg in messages[-10:]:
                    role_str = msg.role if isinstance(msg.role, str) else msg.role.value
                    content_str = msg.content if isinstance(msg.content, str) else str(msg.content)[:200]
                    recent_messages_text += f"{role_str}: {content_str}\n"
                
                context_for_llm = f"""CONVERSATION SUMMARY:
Full Session: {summaries.long_term}

Recent Context (last ~100 messages): {summaries.medium_term}

Current Batch (last 15 messages): {summaries.short_term}

Key Topics: {', '.join(summaries.topics)}

LATEST MESSAGES:
{recent_messages_text}"""
            
            logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
            
            return MessageHistoryResponse(
                sessionId=session_id,
                messages=messages,
                total_count=total_count,
                summaries=summaries,
                context_for_llm=context_for_llm
            )
            
    except Exception as e:
        logger.error(f"Error retrieving session messages: {str(e)}")
        raise


async def get_session_message_count(session_id: str, user_id: str) -> int:
    """Get the count of messages in a session for a specific user"""
    try:
        # Query PostMessage for this session and user
        query_params = {
            "where": json.dumps({
                "sessionId": session_id,
                "user": {
                    "__type": "Pointer",
                    "className": "_User", 
                    "objectId": user_id
                }
            }),
            "count": 1,
            "limit": 0  # Don't return actual objects, just count
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PARSE_SERVER_URL}/parse/classes/PostMessage",
                headers=HEADERS,
                params=query_params
            )
            response.raise_for_status()
            data = response.json()
            return data.get("count", 0)
            
    except Exception as e:
        logger.error(f"Error getting session message count: {e}")
        return 0


async def get_unprocessed_messages_for_session(session_id: str, user_id: str) -> List[Dict]:
    """Get unprocessed messages from a session for batch analysis"""
    try:
        # First, get the chat session to get its objectId
        chat = await get_or_create_chat_session(session_id, user_id, None, None, None)
        if not chat:
            logger.error(f"Could not find chat session for sessionId: {session_id}")
            return []
        
        chat_object_id = chat.get("objectId")
        if not chat_object_id:
            logger.error(f"Chat session missing objectId: {chat}")
            return []
        
        # Query for messages that haven't been processed yet, using the chat pointer
        query_params = {
            "where": json.dumps({
                "chat": {
                    "__type": "Pointer",
                    "className": "Chat",
                    "objectId": chat_object_id
                },
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                },
                "$or": [
                    {"processingStatus": {"$exists": False}},
                    {"processingStatus": "pending"},
                    {"processingStatus": "stored_only"}
                ]
            }),
            "order": "createdAt",  # Process in chronological order
            "limit": 15,  # Process up to 15 messages at a time
            "keys": "objectId,message,content,messageRole,processingStatus,createdAt"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PARSE_SERVER_URL}/parse/classes/PostMessage",
                headers=HEADERS,
                params=query_params
            )
            response.raise_for_status()
            data = response.json()
            messages = data.get("results", [])
            
            logger.info(f"Found {len(messages)} unprocessed messages for session {session_id}")
            return messages
            
    except Exception as e:
        logger.error(f"Error getting unprocessed messages: {e}")
        return []


async def get_or_create_chat_session(session_id: str, user_id: str, workspace_id: Optional[str] = None, organization_id: Optional[str] = None, namespace_id: Optional[str] = None) -> Optional[Dict]:
    """Get existing chat session or create new one"""
    try:
        if not PARSE_SERVER_URL:
            logger.error("PARSE_SERVER_URL environment variable is not set.")
            return None

        # Ensure URL has protocol
        url = PARSE_SERVER_URL
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"

        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        # First, try to get existing chat
        query_params = {
            "where": json.dumps({
                "sessionId": session_id,
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                }
            }),
            "limit": 1
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{url}/parse/classes/Chat",
                headers=headers,
                params=query_params
            )

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                return results[0]

        # Create new chat session if not found
        chat_data = {
            "sessionId": session_id,
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "messageCount": 0,
            "lastProcessedMessageIndex": 0,
            "processingStatus": "active",
            "ACL": {
                user_id: {"read": True, "write": True}
            }
        }

        if workspace_id:
            chat_data["workspace"] = {
                "__type": "Pointer",
                "className": "WorkSpace",
                "objectId": workspace_id
            }

        if organization_id:
            chat_data["organization"] = {
                "__type": "Pointer",
                "className": "Organization",
                "objectId": organization_id
            }

        if namespace_id:
            chat_data["namespace"] = {
                "__type": "Pointer",
                "className": "Namespace",
                "objectId": namespace_id
            }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{url}/parse/classes/Chat",
                headers=headers,
                json=chat_data
            )

        if response.status_code == 201:
            return response.json()
        else:
            logger.error(f"Failed to create chat session: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error getting/creating chat session: {e}")
        return None


async def update_chat_message_count(session_id: str, user_id: str, increment: int = 1) -> bool:
    """Update message count for a chat session"""
    try:
        if not PARSE_SERVER_URL:
            logger.error("PARSE_SERVER_URL environment variable is not set.")
            return False

        # Ensure URL has protocol
        url = PARSE_SERVER_URL
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"

        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        # Get chat session
        chat = await get_or_create_chat_session(session_id, user_id, None, None, None)
        if not chat:
            return False

        # Update message count
        update_data = {
            "messageCount": {"__op": "Increment", "amount": increment}
        }

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{url}/parse/classes/Chat/{chat['objectId']}",
                headers=headers,
                json=update_data
            )

        return response.status_code == 200

    except Exception as e:
        logger.error(f"Error updating chat message count: {e}")
        return False


async def get_previous_chat_needing_processing(user_id: str, current_session_id: str) -> Optional[Dict]:
    """
    Get the most recent chat (excluding current session) that needs processing
    This implements the correct logic: when starting a new session, check the previous session
    """
    try:
        if not PARSE_SERVER_URL:
            logger.error("PARSE_SERVER_URL environment variable is not set.")
            return None

        # Ensure URL has protocol
        url = PARSE_SERVER_URL
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"

        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        # Query for the most recent chat (excluding current session)
        query_params = {
            "where": json.dumps({
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                },
                "sessionId": {"$ne": current_session_id},  # Exclude current session
                "processingStatus": {"$ne": "archived"}
            }),
            "order": "-updatedAt",  # Most recent first
            "limit": 1  # Only get the previous session
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{url}/parse/classes/Chat",
                headers=headers,
                params=query_params
            )

        if response.status_code == 200:
            data = response.json()
            chats = data.get("results", [])
            
            if chats:
                chat = chats[0]
                message_count = chat.get("messageCount", 0)
                last_processed = chat.get("lastProcessedMessageIndex", 0)
                unprocessed_count = message_count - last_processed
                
                # Only return if it has unprocessed messages (any amount, not just 15+)
                if unprocessed_count > 0:
                    logger.info(f"Found previous chat {chat.get('sessionId')} with {unprocessed_count} unprocessed messages")
                    return chat
            
            logger.info("No previous chat needs processing")
            return None
        else:
            logger.error(f"Failed to fetch previous chat: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error fetching previous chat needing processing: {e}")
        return None


async def should_trigger_analysis(session_id: str, user_id: str, is_new_session: bool = False) -> bool:
    """
    Correct analysis trigger logic:
    1. Process current session when it hits 15+ unprocessed messages
    2. When starting a new session, check if the previous session has unprocessed messages
    """
    try:
        # Get the current chat session
        chat = await get_or_create_chat_session(session_id, user_id, None, None, None)
        if not chat:
            return False
        
        message_count = chat.get("messageCount", 0)
        last_processed = chat.get("lastProcessedMessageIndex", 0)
        unprocessed_count = message_count - last_processed
        
        # Trigger if current session has 15+ unprocessed messages
        if unprocessed_count >= 15:
            logger.info(f"Current session {session_id} has {unprocessed_count} unprocessed messages - triggering analysis")
            return True
        
        # If this is a new session, check if the previous session has unprocessed messages
        if is_new_session:
            previous_chat = await get_previous_chat_needing_processing(user_id, session_id)
            if previous_chat:
                prev_session_id = previous_chat.get("sessionId")
                prev_unprocessed = previous_chat.get("messageCount", 0) - previous_chat.get("lastProcessedMessageIndex", 0)
                logger.info(f"New session {session_id} started - found previous session {prev_session_id} with {prev_unprocessed} unprocessed messages")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error determining analysis trigger: {e}")
        return False


async def update_message_processing_status(
    message_id: str,
    status: str,
    error_message: Optional[str] = None
) -> bool:
    """
    Update the processing status of a message
    
    Args:
        message_id: PostMessage objectId
        status: New processing status
        error_message: Optional error message if status is 'failed'
        
    Returns:
        True if successful, False otherwise
    """
    try:
        update_data = {
            "processingStatus": status
        }
        
        if error_message:
            update_data["processingError"] = error_message
        
        # Note: Parse Server automatically updates 'updatedAt' field
        
        url = f"{PARSE_SERVER_URL}/parse/classes/PostMessage/{message_id}"
        
        async with httpx.AsyncClient() as client:
            response = await client.put(url, headers=HEADERS, json=update_data)
            response.raise_for_status()
            
            logger.info(f"Updated message {message_id} status to {status}")
            return True
            
    except Exception as e:
        logger.error(f"Error updating message status: {str(e)}")
        return False


async def update_chat_summaries(
    session_id: str,
    user_id: str,
    summaries: Dict[str, Any],
    workspace_id: Optional[str] = None
) -> bool:
    """
    Update Chat.summaries field in Parse Server with hierarchical conversation summaries.
    
    Args:
        session_id: Session identifier
        user_id: User ID for access control
        summaries: Dict with short_term, medium_term, long_term, topics keys
        workspace_id: Optional workspace ID for filtering
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Find Chat by sessionId
        chat_query = {
            "sessionId": session_id,
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        }
        
        async with httpx.AsyncClient() as client:
            chat_response = await client.get(
                f"{PARSE_SERVER_URL}/parse/classes/Chat",
                headers=HEADERS,
                params={"where": json.dumps(chat_query), "limit": 1}
            )
            
            if chat_response.status_code != 200:
                logger.error(f"Failed to find chat session: {chat_response.status_code} - {chat_response.text}")
                return False
            
            chat_data = chat_response.json()
            if not chat_data.get("results"):
                logger.warning(f"No chat session found for sessionId: {session_id}")
                return False
            
            chat_object_id = chat_data["results"][0]["objectId"]
            
            # Update Chat with summaries
            update_data = {
                "summaries": {
                    "short_term": summaries.get("short_term", ""),
                    "medium_term": summaries.get("medium_term", ""),
                    "long_term": summaries.get("long_term", ""),
                    "topics": summaries.get("topics", []),
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            }
            
            update_response = await client.put(
                f"{PARSE_SERVER_URL}/parse/classes/Chat/{chat_object_id}",
                headers=HEADERS,
                json=update_data
            )
            
            if update_response.status_code in [200, 201]:
                logger.info(f"✅ Updated Chat summaries for session {session_id}")
                return True
            else:
                logger.error(f"Failed to update Chat summaries: {update_response.status_code} - {update_response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Error updating chat summaries: {str(e)}")
        return False
