from fastapi import APIRouter, HTTPException, Query, Header, Depends, Response
from typing import Optional, List, Dict, Any
from models.user_models import (
    CreateUserRequest, 
    UpdateUserRequest, 
    UserResponse, 
    UserListResponse, 
    DeleteUserResponse,
    BatchUserCreateRequest
)
import httpx
from os import environ as env
from services.url_utils import clean_url
from uuid import uuid4
from services.logger_singleton import LoggerSingleton
from fastapi.responses import JSONResponse
from services.auth_utils import get_user_from_token
import hashlib
import json
import asyncio
from services.user_utils import User
from pydantic import BaseModel
from services.utils import log_amplitude_event, serialize_datetime
from models.shared_types import MemoryMetadata
import re
from fastapi import APIRouter, HTTPException, Query, Header, Depends, Response
from fastapi.security import HTTPBearer, APIKeyHeader

# Get logger instance
logger = LoggerSingleton.get_logger(__name__)

# Security schemes
bearer_auth = HTTPBearer(scheme_name="Bearer", bearerFormat="JWT", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
session_token_header = APIKeyHeader(name="X-Session-Token", auto_error=False)

router = APIRouter(prefix="/user", tags=["User"])

# Initialize Parse client settings
PARSE_SERVER_URL = clean_url(env.get("PARSE_SERVER_URL"))
PARSE_APPLICATION_ID = clean_url(env.get("PARSE_APPLICATION_ID"))
PARSE_MASTER_KEY = clean_url(env.get("PARSE_MASTER_KEY"))

# Headers for Parse Server requests
PARSE_HEADERS = {
    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
    "X-Parse-Master-Key": PARSE_MASTER_KEY,
    "Content-Type": "application/json"
}

def generate_user_password(developer_id: str, identifier: str) -> str:
    """Generate a deterministic password based on developer ID and user identifier"""
    # Combine developer_id with the identifier (email or anonymous username)
    combined = f"{developer_id}:{identifier}:{PARSE_MASTER_KEY}"  # Add master key as salt
    # Create SHA-256 hash
    hashed = hashlib.sha256(combined.encode()).hexdigest()
    return hashed[:32]  # Return first 32 chars of hash

async def validate_developer_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> Dict[str, Optional[str]]:
    """Validate the developer's API key and return developer ID, organization ID, and namespace ID"""
    try:
        # Create auth header for API key
        auth_header = f"APIKey {x_api_key}"
        logger.info(f"auth_header: {auth_header}")
        
        # Use get_user_from_token to validate and get user info
        user_id, _, user_info, _ = await get_user_from_token(auth_header, "papr_plugin")
        
        # Fetch developer's organization and namespace for multi-tenant isolation
        organization_id = None
        namespace_id = None
        try:
            async with httpx.AsyncClient() as client:
                developer_url = f"{PARSE_SERVER_URL}/parse/classes/_User/{user_id}"
                dev_resp = await client.get(developer_url, headers=PARSE_HEADERS)
                dev_resp.raise_for_status()
                developer_data = dev_resp.json()
                organization_id = developer_data.get("organization_id")
                
                # Get namespace from organization's default_namespace
                if organization_id:
                    org_url = f"{PARSE_SERVER_URL}/parse/classes/Organization/{organization_id}"
                    org_resp = await client.get(org_url, headers=PARSE_HEADERS)
                    org_resp.raise_for_status()
                    org_data = org_resp.json()
                    namespace_id = org_data.get("default_namespace_id")
                    if not namespace_id and org_data.get("default_namespace"):
                        namespace_id = org_data["default_namespace"].get("objectId")
        except Exception as e:
            logger.warning(f"Could not fetch developer organization/namespace during auth: {e}")
        
        # Return developer info as a dict
        return {
            "developer_id": user_id,
            "organization_id": organization_id,
            "namespace_id": namespace_id
        }
        
    except ValueError as e:
        raise HTTPException(status_code=401, detail="Invalid API key")
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid API key")

async def create_developer_user(email: Optional[str] = None, developer_id: str = None) -> str:
    """Get or create a Parse User with deterministic password"""
    logger.info(f"create_developer_user called with email={email}, developer_id={developer_id}, email_type={type(email)}, developer_id_type={type(developer_id)}")
    if not developer_id:
        raise HTTPException(status_code=500, detail="Developer ID is required")
        
    try:
        # Always create a new anonymous user
        username = f"anon_{uuid4().hex[:10]}"
        url = f"{PARSE_SERVER_URL}/parse/users"
        # Generate password based on username
        password = generate_user_password(str(developer_id), username)
        
        data = {
            "username": username,
            "password": password,
            "fullname": username,
            "type": "developerUser",
            "ACL": {
                str(developer_id): {  
                    "read": True,
                    "write": True
                }
            }
        }
        
        if email:
            if not isinstance(email, str):
                logger.error(f"Email is not a string: {email} (type: {type(email)})")
                email = str(email) if email is not None else ""
            unique_email = hashlib.sha256(f"{email}:{developer_id}".encode()).hexdigest()[:16] + "@anon.papr.ai"
            data["email"] = unique_email
        else:
            unique_email = hashlib.sha256(f"{developer_id}:{uuid4().hex}".encode()).hexdigest()[:16] + "@anon.papr.ai"
            data["email"] = unique_email
            
        logger.info(f"Creating new user: {data}")
        
        # Create the user
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=PARSE_HEADERS, json=data)
            if response.status_code == 201:
                object_id = response.json()["objectId"]
                # Fetch the user to ensure it's saved
                user_url = f"{PARSE_SERVER_URL}/parse/users/{object_id}"
                user_response = await client.get(user_url, headers=PARSE_HEADERS)
                logger.info(f"Fetched user after creation: url={user_url}, status_code={user_response.status_code}, response={user_response.text}")
                if user_response.status_code == 200:
                    return object_id
                else:
                    logger.error(f"User fetch failed after creation: {user_response.status_code} - {user_response.text}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"User fetch failed after creation: {user_response.text}"
                    )
            else:
                # Log the error response
                logger.error(f"Error creating user: {response.status_code} - {response.text}")
                
                # Parse the error response to determine appropriate status code
                try:
                    error_data = response.json()
                    error_code = error_data.get("code")
                    
                    # Map Parse Server error codes to appropriate HTTP status codes
                    if error_code == 203:  # Account already exists
                        status_code = 409  # Conflict
                        detail = "User with this email already exists"
                    elif error_code == 202:  # Username taken
                        status_code = 409  # Conflict
                        detail = "Username already taken"
                    elif error_code == 142:  # Invalid email
                        status_code = 400  # Bad Request
                        detail = "Invalid email format"
                    else:
                        # Default to 400 for client errors, 500 for server errors
                        status_code = 400 if response.status_code < 500 else 500
                        detail = f"Failed to create user: {response.text}"
                        
                except (ValueError, KeyError):
                    # If we can't parse the error response, use the original status code
                    status_code = 400 if response.status_code < 500 else 500
                    detail = f"Failed to create user: {response.text}"
                
                raise HTTPException(
                    status_code=status_code,
                    detail=detail
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_or_create_user: {str(e)} | email={email}, developer_id={developer_id}, email_type={type(email)}, developer_id_type={type(developer_id)}")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

async def create_user_core(request: CreateUserRequest, developer_id: str, x_api_key: str, organization_id=None, namespace_id=None) -> UserResponse:
    """Core logic for creating a user, DeveloperUser, and adding to workspace."""
    try:
        # Get or create Parse User
        user_id = await create_developer_user(request.email, developer_id)
        # --- Parallel: Get developer's workspace and prepare addPeopleToWorkspace payload ---
        logger.info(f"get_selected_workspace_id_async called with developer_id={developer_id}, x_api_key={x_api_key}")
        workspace_id = await User.get_selected_workspace_id_async(developer_id, None, api_key=x_api_key)
        logger.info(f"workspace_id: {workspace_id}")
        logger.info(f"user_id: {user_id}")
        logger.info(f"developer_id: {developer_id}")
        logger.info(f"organization_id: {organization_id}")
        logger.info(f"namespace_id: {namespace_id}")
        add_people_payload = None
        if workspace_id:
            add_people_payload = {
                "userWhoAddedPeople": {
                    "id": developer_id
                },
                "workspace": workspace_id,
                "usersToAdd": [
                    {
                        "id": user_id
                    }
                ]
            }
        # --- Define coroutines for parallel execution ---
        async def create_developer_user_record():
            url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser"
            data = {
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                },
                "developer": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": developer_id
                },
                "external_id": request.external_id,
                "metadata": request.metadata,
                "ACL": {
                    developer_id: {
                        "read": True,
                        "write": True
                    }
                }
            }
            
            # Add organization pointer (Parse Server will auto-generate _p_organization)
            if organization_id:
                data["organization"] = {
                    "__type": "Pointer",
                    "className": "Organization",
                    "objectId": organization_id
                }
                logger.info(f"Added organization pointer to DeveloperUser: {organization_id}")
            
            # Add namespace pointer (Parse Server will auto-generate _p_namespace)
            if namespace_id:
                data["namespace"] = {
                    "__type": "Pointer",
                    "className": "Namespace",
                    "objectId": namespace_id
                }
                logger.info(f"Added namespace pointer to DeveloperUser: {namespace_id}")
            
            if request.email:
                data["email"] = request.email
            timeout = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.post(url, headers=PARSE_HEADERS, json=data)
        async def add_user_to_workspace():
            if not add_people_payload:
                return None
            logger.info(f"add_people_payload: {add_people_payload}")
            
            # Check if we're in open-source edition - use Python implementation instead of cloud function
            import os
            papr_edition = os.getenv("PAPR_EDITION", "").lower()
            is_opensource = papr_edition == "opensource"
            
            if is_opensource:
                # Use Python implementation for open-source
                from services.workspace_utils import add_people_to_workspace_opensource
                try:
                    user_who_added_id = add_people_payload.get("userWhoAddedPeople", {}).get("id")
                    workspace_id_param = add_people_payload.get("workspace")
                    users_to_add = add_people_payload.get("usersToAdd", [])
                    
                    user_who_added_follower, new_followers = await add_people_to_workspace_opensource(
                        user_who_added_id,
                        workspace_id_param,
                        users_to_add
                    )
                    
                    # Return response in same format as cloud function
                    # Cloud function returns: {"result": [userWhoAddedPeopleWorkspaceFollower, newWorkspaceFollowers]}
                    result_array = []
                    if user_who_added_follower:
                        result_array.append(user_who_added_follower)
                    else:
                        result_array.append(None)
                    result_array.append(new_followers)
                    
                    # Create a mock response object that mimics httpx.Response
                    class MockResponse:
                        def __init__(self, status_code, json_data):
                            self.status_code = status_code
                            self._json_data = json_data
                        
                        def json(self):
                            return self._json_data
                        
                        @property
                        def text(self):
                            return json.dumps(self._json_data)
                    
                    return MockResponse(200, {"result": result_array})
                except Exception as e:
                    logger.error(f"add_people_to_workspace_opensource error: {e}", exc_info=True)
                    return None
            else:
                # Use cloud function for cloud edition
                cloud_func_url = f"{PARSE_SERVER_URL}/parse/functions/addPeopleToDeveloperWorkspace"
                timeout = httpx.Timeout(connect=5.0, read=25.0, write=10.0, pool=5.0)
                # Robust retry for transient slowdowns
                for attempt in range(3):
                    try:
                        async with httpx.AsyncClient(timeout=timeout) as client:
                            return await client.post(cloud_func_url, headers=PARSE_HEADERS, json=add_people_payload)
                    except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                        logger.warning(f"addPeopleToDeveloperWorkspace timeout (attempt {attempt+1}/3): {e}")
                        if attempt < 2:
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                    except Exception as e:
                        logger.error(f"addPeopleToDeveloperWorkspace error: {e}")
                    # Fallback verification: check if workspace_follower already exists and set pointer
                    try:
                        query = {
                            "where": {
                                "user": {"__type": "Pointer", "className": "_User", "objectId": user_id},
                                "workspace": {"__type": "Pointer", "className": "WorkSpace", "objectId": workspace_id}
                            },
                            "limit": 1,
                            "keys": "objectId"
                        }
                        import json as _json
                        wf_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower?{httpx.QueryParams({'where': _json.dumps(query['where']), 'limit': 1, 'keys': 'objectId'})}"
                        async with httpx.AsyncClient(timeout=timeout) as client:
                            wf_resp = await client.get(wf_url, headers=PARSE_HEADERS)
                            if wf_resp.status_code == 200:
                                wf_data = wf_resp.json()
                                results = wf_data.get('results', [])
                                if results:
                                    follower_id = results[0].get('objectId')
                                    if follower_id:
                                        user_update_data = {
                                            "isSelectedWorkspaceFollower": {
                                                "__type": "Pointer",
                                                "className": "workspace_follower",
                                                "objectId": follower_id
                                            }
                                        }
                                        user_update_url = f"{PARSE_SERVER_URL}/parse/users/{user_id}"
                                        update_user_response = await client.put(
                                            user_update_url,
                                            headers=PARSE_HEADERS,
                                            json=user_update_data
                                        )
                                        if update_user_response.status_code == 200:
                                            logger.info(f"✅ Set isSelectedWorkspaceFollower on user {user_id} to {follower_id} via fallback lookup")
                                        else:
                                            logger.warning(f"Fallback pointer set failed: {update_user_response.status_code} - {update_user_response.text}")
                    except Exception as ve:
                        logger.warning(f"Fallback verification failed: {ve}")
                    return None
        # --- Run both in parallel ---
        dev_user_response, add_people_response = await asyncio.gather(
            create_developer_user_record(),
            add_user_to_workspace()
        )
        # --- Handle results ---
        if dev_user_response.status_code != 201:
            logger.error(f"DeveloperUser creation failed: {dev_user_response.status_code} - {dev_user_response.text}")
            return UserResponse.failure(
                error=f"Failed to create developer user: {dev_user_response.text}",
                code=500
            )
        if add_people_response:
            if add_people_response.status_code != 200:
                logger.error(f"addPeopleToWorkspace failed: {add_people_response.status_code} - {add_people_response.text}")
            else:
                cloud_response = add_people_response.json()
                logger.info(f"addPeopleToDeveloperWorkspace succeeded: {cloud_response}")
                
                # *** FIX: Extract workspace_follower from cloud function response ***
                # The addPeopleToDeveloperWorkspace returns: [developer_workspace_follower, new_user_workspace_followers]
                # We need the second element (index 1) which contains the newly created workspace_followers
                try:
                    result_array = cloud_response.get("result", [])
                    if len(result_array) >= 2:
                        # Index 1 contains the newly created workspace_followers for the added users
                        new_workspace_followers = result_array[1]
                        
                        if new_workspace_followers and len(new_workspace_followers) > 0:
                            # Get the first (and should be only) workspace_follower for our new user
                            workspace_follower = new_workspace_followers[0]
                            follower_id = workspace_follower.get("objectId")
                            
                            if follower_id:
                                logger.info(f"Found workspace_follower from cloud function response: {follower_id}")
                                
                                # Set isSelectedWorkspaceFollower on the user
                                async with httpx.AsyncClient() as client:
                                    user_update_data = {
                                        "isSelectedWorkspaceFollower": {
                                            "__type": "Pointer",
                                            "className": "workspace_follower", 
                                            "objectId": follower_id
                                        }
                                    }
                                    
                                    user_update_url = f"{PARSE_SERVER_URL}/parse/users/{user_id}"
                                    update_user_response = await client.put(
                                        user_update_url,
                                        headers=PARSE_HEADERS,
                                        json=user_update_data
                                    )
                                    
                                    if update_user_response.status_code == 200:
                                        logger.info(f"✅ Successfully set isSelectedWorkspaceFollower on user {user_id} to {follower_id}")
                                    else:
                                        logger.error(f"Failed to set isSelectedWorkspaceFollower on user: {update_user_response.status_code} - {update_user_response.text}")
                            else:
                                logger.error(f"No objectId found in workspace_follower: {workspace_follower}")
                        else:
                            logger.error(f"No new workspace_followers found in cloud function response")
                    else:
                        logger.error(f"Invalid cloud function response format - expected array with 2 elements, got: {len(result_array)}")
                            
                except Exception as e:
                    logger.error(f"Error extracting workspace_follower from cloud response: {str(e)}")
        else:
            logger.warning(f"No workspace found for developer {developer_id}, skipping addPeopleToWorkspace.")
        # --- Fetch user details and return response as before ---
        async with httpx.AsyncClient() as client:
            dev_user_data = dev_user_response.json()
            user_url = f"{PARSE_SERVER_URL}/parse/classes/_User/{user_id}"
            user_response_data = await client.get(user_url, headers=PARSE_HEADERS)
            user_data = user_response_data.json()
            return UserResponse.success(
                user_id=user_id,
                email=request.email if request.email else None,
                external_id=request.external_id,
                metadata=request.metadata,
                created_at=dev_user_data.get("createdAt"),
                updated_at=user_data.get("updatedAt")
            )
    except HTTPException as http_ex:
        # Convert HTTPException to UserResponse format while preserving status code
        logger.error(f"HTTPException in create_user_core: {http_ex.status_code} - {http_ex.detail}")
        return UserResponse.failure(
            error=http_ex.detail,
            code=http_ex.status_code
        )
    except Exception as e:
        import traceback
        logger.error(f"Error creating user: {str(e)}\n{traceback.format_exc()}")
        return UserResponse.failure(
            error=str(e),
            code=500
        )

@router.post("", response_model=UserResponse,
            openapi_extra={"x-operation-name": "create_user",
                          "x-operation-description": "Create a new user or link existing user to developer",
                          "x-operation-tags": ["User"],
                          "operationId": "create_user"})
async def create_user(
    request: CreateUserRequest,
    response: Response,
    dev_info: Dict[str, Optional[str]] = Depends(validate_developer_api_key),
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """Create a new user or link existing user to developer"""
    try:
        developer_id = dev_info["developer_id"]
        organization_id = dev_info["organization_id"]
        namespace_id = dev_info["namespace_id"]
        
        user_response = await create_user_core(request, developer_id, x_api_key, organization_id, namespace_id)
        response.status_code = user_response.code
        return user_response
    except HTTPException as http_ex:
        # Convert HTTPException to UserResponse format while preserving status code
        logger.error(f"HTTPException in create_user endpoint: {http_ex.status_code} - {http_ex.detail}")
        user_response = UserResponse.failure(error=http_ex.detail, code=http_ex.status_code)
        response.status_code = user_response.code
        return user_response
    except Exception as e:
        logger.error(f"Error in create_user endpoint: {str(e)}")
        user_response = UserResponse.failure(error=str(e), code=500)
        response.status_code = user_response.code
        return user_response

@router.get("/{user_id}", response_model=UserResponse,
            openapi_extra={"x-operation-name": "get_user",
                          "x-operation-description": "Get user details by user_id (_User.objectId) and developer association",
                          "x-operation-tags": ["User"],
                          "operationId": "get_user"})
async def get_user(
    user_id: str,
    response: Response,
    dev_info: Dict[str, Optional[str]] = Depends(validate_developer_api_key)
):
    """Get user details by user_id (_User.objectId) and developer association"""
    try:
        developer_id = dev_info["developer_id"]
        url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser"
        params = {
            "where": json.dumps({
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                },
                "developer": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": developer_id
                }
            })
        }
        async with httpx.AsyncClient() as client:
            parse_response = await client.get(url, headers=PARSE_HEADERS, params=params)
            if parse_response.status_code != 200:
                user_response = UserResponse.failure(
                    error="Failed to find DeveloperUser record",
                    code=500,
                    details={"parse_status_code": parse_response.status_code, "parse_body": parse_response.text}
                )
                response.status_code = user_response.code
                return user_response
            dev_user_results = parse_response.json().get("results", [])
            if not dev_user_results:
                user_response = UserResponse.failure(
                    error="User not found",
                    code=404
                )
                response.status_code = user_response.code
                return user_response
            dev_user_data = dev_user_results[0]
            user_response = UserResponse.success(
                user_id=user_id,
                email=dev_user_data.get("email") if dev_user_data.get("email") else None,
                external_id=dev_user_data.get("external_id"),
                metadata=dev_user_data.get("metadata"),
                created_at=dev_user_data.get("createdAt"),
                updated_at=dev_user_data.get("updatedAt")
            )
            response.status_code = user_response.code
            return user_response
    except Exception as e:
        logger.error(f"Error getting user: {str(e)}")
        user_response = UserResponse.failure(
            error=str(e),
            code=500
        )
        response.status_code = user_response.code
        return user_response

@router.put("/{user_id}", response_model=UserResponse,
            openapi_extra={"x-operation-name": "update_user",
                          "x-operation-description": "Update user details by user_id (_User.objectId) and developer association",
                          "x-operation-tags": ["User"],
                          "operationId": "update_user"})

async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    response: Response,
    dev_info: Dict[str, Optional[str]] = Depends(validate_developer_api_key)
):
    """Update user details by user_id (_User.objectId) and developer association"""
    try:
        developer_id = dev_info["developer_id"]
        # Find the DeveloperUser record for this user and developer
        url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser"
        params = {
            "where": json.dumps({
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                },
                "developer": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": developer_id
                }
            })
        }
        async with httpx.AsyncClient() as client:
            dev_user_response = await client.get(url, headers=PARSE_HEADERS, params=params)
            if dev_user_response.status_code != 200:
                user_response = UserResponse.failure(
                    error="Failed to find DeveloperUser record",
                    code=500,
                    details={"parse_status_code": dev_user_response.status_code, "parse_body": dev_user_response.text}
                )
                response.status_code = user_response.code
                return user_response
            dev_user_results = dev_user_response.json().get("results", [])
            if not dev_user_results:
                user_response = UserResponse.failure(
                    error="User not found",
                    code=404
                )
                response.status_code = user_response.code
                return user_response
            dev_user_data = dev_user_results[0]
            dev_user_id = dev_user_data["objectId"]

            # Update DeveloperUser record
            update_data = {}
            if request.external_id is not None:
                update_data["external_id"] = request.external_id
            if request.metadata is not None:
                update_data["metadata"] = request.metadata
            if request.type is not None:
                update_data["type"] = request.type.value
            if request.email is not None:
                update_data["email"] = request.email

            if update_data:
                dev_user_url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser/{dev_user_id}"
                update_response = await client.put(dev_user_url, headers=PARSE_HEADERS, json=update_data)
                if update_response.status_code != 200:
                    user_response = UserResponse.failure(
                        error="Failed to update DeveloperUser",
                        code=500,
                        details={"parse_status_code": update_response.status_code, "parse_body": update_response.text}
                    )
                    response.status_code = user_response.code
                    return user_response
                # Fetch the full DeveloperUser record again
                get_dev_user_response = await client.get(dev_user_url, headers=PARSE_HEADERS)
                if get_dev_user_response.status_code != 200:
                    user_response = UserResponse.failure(
                        error="Failed to fetch updated DeveloperUser",
                        code=500,
                        details={"parse_status_code": get_dev_user_response.status_code, "parse_body": get_dev_user_response.text}
                    )
                    response.status_code = user_response.code
                    return user_response
                dev_user_data = get_dev_user_response.json()

            # Update _User record if email provided
            if request.email is not None:
                user_url = f"{PARSE_SERVER_URL}/parse/classes/_User/{user_id}"
                user_update = {"email": request.email}
                user_update_response = await client.put(user_url, headers=PARSE_HEADERS, json=user_update)
                if user_update_response.status_code != 200:
                    user_response = UserResponse.failure(
                        error="Failed to update user email",
                        code=500,
                        details={"parse_status_code": user_update_response.status_code, "parse_body": user_update_response.text}
                    )
                    response.status_code = user_response.code
                    return user_response
                # Fetch the full _User record again
                user_response_data = await client.get(user_url, headers=PARSE_HEADERS)
                user_data = user_response_data.json()
            else:
                user_url = f"{PARSE_SERVER_URL}/parse/classes/_User/{user_id}"
                user_response_data = await client.get(user_url, headers=PARSE_HEADERS)
                user_data = user_response_data.json()

            user_response = UserResponse.success(
                user_id=user_id,
                email=dev_user_data.get("email") if dev_user_data.get("email") else None,
                external_id=dev_user_data.get("external_id"),
                metadata=dev_user_data.get("metadata"),
                created_at=dev_user_data.get("createdAt"),
                updated_at=dev_user_data.get("updatedAt")
            )
            response.status_code = user_response.code
            return user_response
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        user_response = UserResponse.failure(
            error=str(e),
            code=500
        )
        response.status_code = user_response.code
        return user_response

@router.delete("/{user_id}", response_model=DeleteUserResponse, 
               openapi_extra={"x-operation-name": "delete_user",
                             "x-operation-description": "Delete user association with developer and the user itself by , assume external user_id is provided, and resolve to internal user_id (_User.objectId)",
                             "x-operation-tags": ["User"],
                             "operationId": "delete_user"})
async def delete_user(
    user_id: str,
    response: Response,
    dev_info: Dict[str, Optional[str]] = Depends(validate_developer_api_key),
    x_api_key: str = Header(..., alias="X-API-Key"),
    is_external: bool = Query(False, description="Is this an external user ID?")
):
    """Delete user association with developer and the user itself by , assume external user_id is provided, and resolve to internal user_id (_User.objectId)"""
    try:
        developer_id = dev_info["developer_id"]
        # Only resolve if is_external is True
        if is_external:
            metadata = MemoryMetadata(external_user_id=user_id)
            async with httpx.AsyncClient() as client:
                user_service = User(developer_id)
                updated_metadata, _, _ = await user_service.resolve_external_user_ids_to_internal(
                    developer_id=developer_id,
                    metadata=metadata,
                    httpx_client=client,
                    x_api_key=x_api_key
                )
                user_id = updated_metadata.user_id
        url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser"
        async with httpx.AsyncClient() as client:
            params = {
                "where": json.dumps({
                    "user": {
                        "__type": "Pointer",
                        "className": "_User",
                        "objectId": user_id
                    },
                    "developer": {
                        "__type": "Pointer",
                        "className": "_User",
                        "objectId": developer_id
                    }
                })
            }
            response_dev = await client.get(url, headers=PARSE_HEADERS, params=params)
            if response_dev.status_code != 200:
                result = DeleteUserResponse.failure(
                    user_id=user_id,
                    error="Failed to find DeveloperUser record(s)",
                    code=500,
                    details={"parse_status_code": response_dev.status_code, "parse_body": response_dev.text}
                )
                response.status_code = result.code
                return result
            dev_user_results = response_dev.json().get("results", [])
            # Delete all matching DeveloperUser records
            for dev_user in dev_user_results:
                dev_user_id = dev_user["objectId"]
                del_url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser/{dev_user_id}"
                del_response = await client.delete(del_url, headers=PARSE_HEADERS)
                if del_response.status_code != 200:
                    result = DeleteUserResponse.failure(
                        user_id=user_id,
                        error="Failed to delete DeveloperUser record",
                        code=500,
                        details={"parse_status_code": del_response.status_code, "parse_body": del_response.text}
                    )
                    response.status_code = result.code
                    return result
            # Delete the _User record itself using the correct Parse endpoint
            user_url = f"{PARSE_SERVER_URL}/parse/users/{user_id}"
            user_response = await client.delete(user_url, headers=PARSE_HEADERS)
            if user_response.status_code != 200:
                result = DeleteUserResponse.failure(
                    user_id=user_id,
                    error="Failed to delete user",
                    code=500,
                    details={"parse_status_code": user_response.status_code, "parse_body": user_response.text}
                )
                response.status_code = result.code
                return result
            result = DeleteUserResponse.success(user_id=user_id)
            response.status_code = result.code
            return result
    except Exception as e:
        logger.error(f"Error deleting user: {str(e)}")
        result = DeleteUserResponse.failure(
            user_id=user_id,
            error=str(e),
            code=500
        )
        response.status_code = result.code
        return result

@router.get("", response_model=UserListResponse,
             openapi_extra={"x-operation-name": "list_users",
                             "x-operation-description": "List users for a developer",
                             "x-operation-tags": ["User"],
                             "operationId": "list_users"})
async def list_users(
    response: Response,
    dev_info: Dict[str, Optional[str]] = Depends(validate_developer_api_key),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    external_id: Optional[str] = None,
    email: Optional[str] = None
):
    """List users for a developer"""
    try:
        developer_id = dev_info["developer_id"]
        skip = (page - 1) * page_size
        where = {
            "developer": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": developer_id
            }
        }
        if external_id:
            where["external_id"] = external_id

        url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser"
        params = {
            "where": json.dumps(where),
            "skip": skip,
            "limit": page_size,
            "include": "user",
            "order": "-createdAt"
        }

        async with httpx.AsyncClient() as client:
            response_users = await client.get(url, headers=PARSE_HEADERS, params=params)
            if response_users.status_code != 200:
                result = UserListResponse.failure(
                    error="Failed to list users",
                    code=500,
                    details={"parse_status_code": response_users.status_code, "parse_body": response_users.text}
                )
                response.status_code = result.code
                return result

            data = response_users.json()
            users = []
            for item in data["results"]:
                user_data = item.get("user", {})
                if email and user_data.get("email") != email:
                    continue
                users.append(UserResponse(
                    user_id=user_data.get("objectId"),
                    email=user_data.get("email") if user_data.get("email") else None,
                    external_id=item.get("external_id"),
                    metadata=item.get("metadata"),
                    created_at=item["createdAt"],
                    updated_at=item["updatedAt"],
                    code=200,
                    status="success"
                ))

            # Get total count
            count_params = {"where": json.dumps(where), "count": 1, "limit": 0, "order": "-createdAt"}
            count_response = await client.get(url, headers=PARSE_HEADERS, params=count_params)
            total = count_response.json().get("count", 0)

            result = UserListResponse.success(
                data=users,
                total=total,
                page=page,
                page_size=page_size
            )
            response.status_code = result.code
            return result

    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        result = UserListResponse.failure(
            error=str(e),
            code=500
        )
        response.status_code = result.code
        return result

@router.post("/batch", response_model=UserListResponse,
             openapi_extra={"x-operation-name": "create_user_batch",
                             "x-operation-description": "Create multiple users or link existing users to developer, and add each to the developer's workspace (if one exists).",
                             "x-operation-tags": ["User"],
                             "operationId": "create_user_batch"})
async def create_user_batch(
    requests: BatchUserCreateRequest,
    response: Response,
    dev_info: Dict[str, Optional[str]] = Depends(validate_developer_api_key),
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """Create multiple users or link existing users to developer, and add each to the developer's workspace (if one exists)."""
    user_requests = requests.users
    try:
        developer_id = dev_info["developer_id"]
        organization_id = dev_info["organization_id"]
        namespace_id = dev_info["namespace_id"]
        
        user_responses = await asyncio.gather(*(create_user_core(req, developer_id, x_api_key, organization_id, namespace_id) for req in user_requests))
        all_success = all(u.code == 200 for u in user_responses)
        response.status_code = 200 if all_success else 207
        return UserListResponse.success(
            data=user_responses,
            total=len(user_responses),
            page=1,
            page_size=len(user_responses)
        )
    except Exception as e:
        logger.error(f"Error in create_user_batch: {str(e)}")
        response.status_code = 500
        return UserListResponse.failure(error=str(e), code=500) 