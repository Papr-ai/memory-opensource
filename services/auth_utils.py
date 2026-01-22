from services.user_utils import User
import logging
import asyncio
from auth0.authentication import GetToken
from os import environ as env
from services.logging_config import get_logger
from typing import Dict, Tuple, Optional, Any, List, Awaitable
from authlib.integrations.flask_client import OAuth
import httpx
import json
from services.url_utils import clean_url
import time
from models.memory_models import SearchRequest, AddMemoryRequest, BatchMemoryRequest, UpdateMemoryRequest, OptimizedAuthResponse, SyncTiersRequest
from models.feedback_models import FeedbackRequest
from services.user_utils import PARSE_APPLICATION_ID, PARSE_MASTER_KEY, PARSE_SERVER_URL, MemoryMetadata
from functools import lru_cache
from datetime import datetime, timedelta
import threading
from services.cache_utils import api_key_cache, session_token_cache, auth_optimized_cache, api_key_to_user_id_cache
from services.cache_utils import enhanced_api_key_cache
from services.logger_singleton import LoggerSingleton

# Import MemoryGraph for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from memory.memory_graph import MemoryGraph

logger = LoggerSingleton.get_logger(__name__)

PARSE_SERVER_URL = clean_url(env.get("PARSE_SERVER_URL"))

PARSE_HEADERS = {
    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
    "X-Parse-Master-Key": PARSE_MASTER_KEY,
    "Content-Type": "application/json"
}

# All cache instances are now imported from cache_utils.py for consistency

# ============================================================================
# User ID Validation Functions
# ============================================================================
# These functions help prevent common errors where developers use external IDs
# (like UUIDs, emails, or custom prefixes) in the user_id field instead of
# external_user_id.

import re
from dataclasses import dataclass

# Common patterns for external user identifiers
UUID_PATTERN = re.compile(
    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
)
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
EXTERNAL_PREFIXES = ('user_', 'ext_', 'external_', 'usr_', 'u_', 'customer_', 'cust_', 'cus_', 'client_', 'acct_', 'sub_', 'org_')


def looks_like_external_id(user_id: str) -> bool:
    """
    Heuristic to detect if a user_id looks like an external identifier.

    External IDs typically:
    - Are UUIDs
    - Are email addresses
    - Have common prefixes like 'user_', 'ext_', etc.
    - Contain hyphens or underscores with alphanumeric segments
    - Are longer than typical Parse ObjectIds (10 alphanumeric chars)

    Parse Server internal IDs are typically:
    - 10 alphanumeric characters (e.g., 'mkcNHhG5KP')
    - No special characters like hyphens, underscores, or @ symbols

    Args:
        user_id: The user ID string to check

    Returns:
        True if the ID looks like an external identifier, False otherwise
    """
    if not user_id or not isinstance(user_id, str):
        return False

    # Check for UUID format (very common for external IDs)
    if UUID_PATTERN.match(user_id):
        logger.debug(f"user_id '{user_id[:20]}...' matches UUID pattern")
        return True

    # Check for email format
    if EMAIL_PATTERN.match(user_id):
        logger.debug(f"user_id '{user_id[:20]}...' matches email pattern")
        return True

    # Check for common external prefixes (case-insensitive)
    user_id_lower = user_id.lower()
    for prefix in EXTERNAL_PREFIXES:
        if user_id_lower.startswith(prefix):
            logger.debug(f"user_id '{user_id[:20]}...' has external prefix '{prefix}'")
            return True

    # Check for hyphenated IDs (common in external systems, not in Parse)
    # Parse IDs don't contain hyphens
    if '-' in user_id and len(user_id) > 10:
        logger.debug(f"user_id '{user_id[:20]}...' contains hyphens (likely external)")
        return True

    # Check for IDs that are too long to be Parse ObjectIds
    # Parse ObjectIds are exactly 10 alphanumeric characters
    # Allow some tolerance for other ID formats
    if len(user_id) > 20 and not user_id.isalnum():
        logger.debug(f"user_id '{user_id[:20]}...' is too long and not alphanumeric")
        return True

    # Parse ObjectIds are exactly 10 alphanumeric characters
    # If it matches this pattern, it's likely a valid Parse ID
    if len(user_id) == 10 and user_id.isalnum():
        return False

    return False


@dataclass
class UserIdValidationError:
    """Structured error for user ID validation failures."""
    code: int
    error: str
    field: str
    provided_value: str
    reason: str
    suggestion: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "code": self.code,
            "error": self.error,
            "details": {
                "field": self.field,
                "provided_value": self.provided_value,
                "reason": self.reason,
                "suggestion": self.suggestion
            }
        }


async def validate_user_identification(
    request: Any,
    memory_graph: Optional["MemoryGraph"] = None,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> Optional[UserIdValidationError]:
    """
    Validate user IDs in a request to prevent common errors.

    This function checks if:
    1. A user_id looks like an external identifier (should use external_user_id instead)
    2. A user_id, if provided, corresponds to a valid Parse user

    Args:
        request: The API request (AddMemoryRequest, SearchRequest, etc.)
        memory_graph: Optional MemoryGraph for user validation
        httpx_client: Optional httpx client for API calls

    Returns:
        UserIdValidationError if validation fails, None if validation passes
    """
    # Extract user_id from request and metadata
    request_user_id = getattr(request, 'user_id', None)
    metadata = getattr(request, 'metadata', None)
    metadata_user_id = getattr(metadata, 'user_id', None) if metadata else None

    # Use the first non-None user_id
    user_id = request_user_id or metadata_user_id

    if not user_id:
        return None  # No user_id provided, validation passes

    # Check if user_id looks like an external identifier
    if looks_like_external_id(user_id):
        logger.warning(
            f"user_id '{user_id[:30]}...' looks like an external identifier. "
            "Developers should use 'external_user_id' instead."
        )
        return UserIdValidationError(
            code=400,
            error="Invalid user_id format",
            field="user_id",
            provided_value=user_id[:50] + ("..." if len(user_id) > 50 else ""),
            reason="This looks like an external user identifier (UUID, email, or custom format). "
                   "Did you mean to use 'external_user_id' instead?",
            suggestion="Use 'external_user_id' for your application's user identifiers. "
                      "'user_id' is reserved for Papr internal user IDs (10 alphanumeric characters)."
        )

    # If memory_graph is provided, validate the Parse user exists
    # Note: This is optional validation - we don't want to slow down every request
    if memory_graph and len(user_id) == 10 and user_id.isalnum():
        try:
            # Check if Parse user exists
            parse_user = await _fetch_parse_user_for_validation(user_id, httpx_client)
            if not parse_user:
                logger.warning(f"user_id '{user_id}' does not correspond to a valid Parse user")
                return UserIdValidationError(
                    code=400,
                    error="Invalid user_id",
                    field="user_id",
                    provided_value=user_id,
                    reason="No Papr user found with this ID. "
                           "If this is your application's user identifier, use 'external_user_id' instead.",
                    suggestion="Use 'external_user_id' for your application's user identifiers. "
                              "Papr will automatically resolve or create internal users as needed."
                )
        except Exception as e:
            # Don't fail the request if validation lookup fails
            logger.debug(f"Could not validate Parse user (non-fatal): {e}")

    return None  # Validation passed


async def _fetch_parse_user_for_validation(
    user_id: str,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch a Parse user by ID for validation purposes.

    Args:
        user_id: The Parse user objectId
        httpx_client: Optional httpx client

    Returns:
        User dict if found, None otherwise
    """
    try:
        url = f"{PARSE_SERVER_URL}/parse/classes/_User/{user_id}"
        params = {"keys": "objectId"}

        if httpx_client:
            response = await httpx_client.get(url, headers=PARSE_HEADERS, params=params)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=PARSE_HEADERS, params=params)

        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch Parse user for validation: {e}")
        return None


def log_deprecation_warning(old_field: str, new_field: str, context: str = ""):
    """
    Log a deprecation warning for API field changes.

    Args:
        old_field: The deprecated field name
        new_field: The recommended field name
        context: Additional context about where this was used
    """
    warning_msg = (
        f"DEPRECATION WARNING: '{old_field}' is deprecated, use '{new_field}' instead. "
        f"{context}"
    )
    logger.warning(warning_msg)


def get_oauth_client(oauth: OAuth, client_type: str) -> Any:
    """Get OAuth client based on client type.

    Args:
        oauth (OAuth): The OAuth instance
        client_type (str): Type of client ('browser_extension' or 'papr_plugin')

    Returns:
        Any: The OAuth client instance

    Raises:
        ValueError: If client_type is not valid
    """
    if client_type == 'browser_extension':
        client = oauth.create_client("auth0_browser_extension")
    elif client_type == 'papr_plugin':
        client = oauth.create_client("auth0_papr_plugin")
    else:
        raise ValueError("Invalid client type")

    logger.info(f"Retrieved OAuth client for {client_type} with client_id: {client.client_id}")
    return client

def determine_client_type(redirect_uri: Optional[str]) -> str:
    """Determine the client type based on the redirect URI.

    Args:
        redirect_uri (Optional[str]): The redirect URI from the request

    Returns:
        str: The determined client type ('browser_extension' or 'papr_plugin')
    """
    if redirect_uri is None:
        # Handle the None case appropriately
        return "papr_plugin"  # or raise an exception

    if 'chromiumapp.org' in redirect_uri:
        return 'browser_extension'
    elif 'chat.openai.com' in redirect_uri:
        return 'papr_plugin'
    return 'papr_plugin'

class CustomGetToken(GetToken):
    def __init__(self, domain: str) -> None:
        # Pass a placeholder or default client_id to the base class constructor
        default_client_id = "default_client_id"
        super().__init__(domain, default_client_id)

    async def authorization_code(self, client_type: str, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Perform the authorization code flow.

        Args:
            client_type (str): Type of client ('browser_extension' or 'papr_plugin')
            code (str): The authorization code you received.
            redirect_uri (str): The redirect URI that you specified when you initiated the authorization code flow.

        Returns:
            Dict[str, Any]: A dictionary containing the access token and ID token.
        """
        # Choose the right client credentials based on client_type
        client_id, client_secret = self.get_client_credentials(client_type)
        # Log inputs
        logger.info(f"Received client_id: {client_id}")
        logger.info(f"Received client_secret: {client_secret[:5]}...")  # Do not log full client_secret!
        logger.info(f"Received code: {code[:5]}...")  # Do not log full refresh_token!
        logger.info(f"Received redirect_uri: {redirect_uri}")  # Do not log full refresh_token!
        url = f"{self.protocol}://{self.domain}/oauth/token"
        data = {
            'grant_type': 'authorization_code',
            'client_id': client_id,
            'client_secret': client_secret,
            'code': code,
            'redirect_uri': redirect_uri,
        }

        # Log the request being made
        logger.info(f"Making POST request to {url} with payload: {data}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data)
            response.raise_for_status()
            return response.json()

    async def refresh_token(self, client_type: str, refresh_token: str) -> Dict[str, Any]:
        """Perform the refresh token flow.

        Args:
            client_type (str): Type of client ('browser_extension' or 'papr_plugin')
            refresh_token (str): The refresh token you received.

        Returns:
            Dict[str, Any]: A dictionary containing the new access token and ID token.
        """
        # Choose the right client credentials based on client_type
        client_id, client_secret = self.get_client_credentials(client_type)
        # Log inputs
        logger.info(f"Received client_id: {client_id}")
        logger.info(f"Received client_secret: {client_secret[:5]}...")  # Do not log full client_secret!
        logger.info(f"Received refresh_token: {refresh_token[:5]}...")  # Do not log full refresh_token!

        url = f"{self.protocol}://{self.domain}/oauth/token"
        payload = {
            'grant_type': 'refresh_token',
            'client_id': client_id,
            'client_secret': client_secret,
            'refresh_token': refresh_token,
        }
        
        # Log the request being made
        logger.info(f"Making POST request to {url} with payload: {payload}...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=payload)
            response.raise_for_status()
            return response.json()

    def get_client_credentials(self, client_type: str) -> Tuple[str, str]:
        """Get client credentials based on client type.

        Args:
            client_type (str): Type of client ('browser_extension' or 'papr_plugin')

        Returns:
            Tuple[str, str]: A tuple containing (client_id, client_secret)

        Raises:
            ValueError: If client_type is not valid
        """
        if client_type == 'browser_extension':
            return env.get("AUTH0_CLIENT_ID_BROWSER", ""), env.get("AUTH0_CLIENT_SECRET_BROWSER", "")
        elif client_type == 'papr_plugin':
            return env.get("AUTH0_CLIENT_ID_PAPR", ""), env.get("AUTH0_CLIENT_SECRET_PAPR", "")
        else:
            raise ValueError("Invalid client type")

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Retrieve user information using the access token.

        Args:
            access_token (str): The access token of the user.

        Returns:
            Dict[str, Any]: A dictionary containing user information.
        """
        url = f"{self.protocol}://{self.domain}/userinfo"
        headers = {'Authorization': f'Bearer {access_token}'}

        # Log the request being made
        logger.info(f"Making GET request to {url} with access token.")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            user_info = response.json()
            logger.info(f"Received user info: {user_info}...")
            return user_info

    async def client_credentials(self, client_id: str, client_secret: str) -> Dict[str, Any]:
        """Perform the client credentials flow for API clients.

        Args:
            client_id (str): The API client ID
            client_secret (str): The API client secret

        Returns:
            Dict[str, Any]: A dictionary containing the access token
        """
        url = f"{self.protocol}://{self.domain}/oauth/token"
        data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
            'audience': f"{self.protocol}://{self.domain}/api/v2/"
        }

        logger.info(f"Making client credentials request for client_id: {client_id}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data)
            response.raise_for_status()
            return response.json()

async def get_user_from_token(
    auth_header: str, 
    client_type: str,
    api_key: Optional[str] = None
) -> Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]:
    """
    Asynchronously extracts user information from either Bearer or Session token.

    Args:
        auth_header (str): The Authorization header from the request
        client_type (str): Type of client ('browser_extension' or 'papr_plugin')
        api_key (Optional[str]): The API key from the request
    Returns:
        Tuple[str, str, Optional[Dict[str, Any]]]: A tuple containing (user_id, sessionToken, user_info)

    Raises:
        ValueError: If the authorization header is invalid or the token is invalid
        AuthenticationError: If token verification fails
    """
    try:
        logger.info(f"Received auth_header: {auth_header}")
        logger.info(f"Received client_type: {client_type}")

        if not auth_header:
            logger.error("No Authorization header provided")
            raise ValueError("Invalid Authorization header")

        if 'Bearer ' not in auth_header and 'Session ' not in auth_header and 'APIKey ' not in auth_header:
            logger.error("Authorization header does not contain 'Bearer ' or 'Session '")
            raise ValueError("Invalid Authorization header")

        if 'Bearer ' in auth_header:
            token = auth_header.split('Bearer ')[1]
            logger.info(f"Got the access_token: {token[:5]}...")

            user_info = await User.verify_access_token(token, client_type)
            if not user_info:
                logger.error("Invalid access token")
                raise ValueError("Invalid access token")

            user_id = user_info['https://papr.scope.com/objectId']
            sessionToken = user_info['https://papr.scope.com/sessionToken']
            return user_id, sessionToken, user_info, None

        elif 'Session ' in auth_header:
            # Session token
            sessionToken = auth_header.split('Session ')[1]
            logger.info(f"Got the session_token: {sessionToken[:5]}...")

            parse_user = await User.verify_session_token(sessionToken)
            if not parse_user:
                logger.error("Invalid session token")
                raise ValueError("Invalid session token")

            # Get user_id from ParseUserPointer model
            user_id = parse_user.objectId
            logger.info(f"Retrieved user_id: {user_id}")
            
            return user_id, sessionToken, None, None

        elif 'APIKey ' in auth_header:
            # API key
            api_key = auth_header.split('APIKey ')[1]
            logger.info(f"Got the api_key: {api_key[:5]}...")

            user_info = await User.verify_api_key(api_key)
            if not user_info:
                logger.error("Invalid API key")
                raise ValueError("Invalid API key")
            # Get user_id from ParseUserPointer model
            user_id = user_info['objectId']
            logger.info(f"Retrieved user_id: {user_id}")
            
            return user_id, None, user_info, api_key
    except ValueError as e:
        logger.error(f"Validation error in get_user_from_token_async: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_user_from_token_async: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise ValueError(f"Authentication failed: {str(e)}")

def flush_auth_caches():
    """Flush all authentication caches"""
    try:
        logger.info("Flushing all authentication caches...")
        api_key_cache.clear()
        session_token_cache.clear()
        auth_optimized_cache.clear()
        api_key_to_user_id_cache.clear()
        enhanced_api_key_cache.clear()  # Include enhanced API key cache
        logger.info("All authentication caches cleared successfully")
    except Exception as e:
        logger.error(f"Error flushing caches: {e}")

def flush_auth_cache_for_incomplete_data():
    """Flush only the auth_optimized_cache to clear any cached incomplete data"""
    try:
        logger.info("Flushing auth_optimized_cache to clear incomplete data...")
        auth_optimized_cache.clear()
        logger.info("Auth optimized cache cleared successfully")
        print("Auth optimized cache flushed - incomplete data cleared")
    except Exception as e:
        logger.error(f"Error flushing auth_optimized_cache: {e}")

def log_cache_stats():
    """Log statistics for all authentication caches"""
    try:
        logger.info("Cache Statistics:")
        
        # API Key Cache
        api_key_stats = api_key_cache.get_stats() if hasattr(api_key_cache, 'get_stats') else {
            'hits': getattr(api_key_cache, 'hits', 0),
            'misses': getattr(api_key_cache, 'misses', 0),
            'sets': getattr(api_key_cache, 'sets', 0),
            'hit_rate': f"{(getattr(api_key_cache, 'hits', 0) / max(1, getattr(api_key_cache, 'hits', 0) + getattr(api_key_cache, 'misses', 0))) * 100:.1f}%",
            'size': len(api_key_cache),
            'maxsize': getattr(api_key_cache, 'maxsize', 'N/A'),
            'ttl_seconds': getattr(api_key_cache, 'ttl', 'N/A')
        }
        logger.info(f"  API Key Cache: {api_key_stats}")
        
        # Session Token Cache
        session_stats = session_token_cache.get_stats() if hasattr(session_token_cache, 'get_stats') else {
            'hits': getattr(session_token_cache, 'hits', 0),
            'misses': getattr(session_token_cache, 'misses', 0),
            'sets': getattr(session_token_cache, 'sets', 0),
            'hit_rate': f"{(getattr(session_token_cache, 'hits', 0) / max(1, getattr(session_token_cache, 'hits', 0) + getattr(session_token_cache, 'misses', 0))) * 100:.1f}%",
            'size': len(session_token_cache),
            'maxsize': getattr(session_token_cache, 'maxsize', 'N/A'),
            'ttl_seconds': getattr(session_token_cache, 'ttl', 'N/A')
        }
        logger.info(f"  Session Token Cache: {session_stats}")
        
        # Auth Optimized Cache
        auth_opt_stats = auth_optimized_cache.get_stats() if hasattr(auth_optimized_cache, 'get_stats') else {
            'hits': getattr(auth_optimized_cache, 'hits', 0),
            'misses': getattr(auth_optimized_cache, 'misses', 0),
            'sets': getattr(auth_optimized_cache, 'sets', 0),
            'hit_rate': f"{(getattr(auth_optimized_cache, 'hits', 0) / max(1, getattr(auth_optimized_cache, 'hits', 0) + getattr(auth_optimized_cache, 'misses', 0))) * 100:.1f}%",
            'size': len(auth_optimized_cache),
            'maxsize': getattr(auth_optimized_cache, 'maxsize', 'N/A'),
            'ttl_seconds': getattr(auth_optimized_cache, 'ttl', 'N/A')
        }
        logger.info(f"  Auth Optimized Cache: {auth_opt_stats}")
        
        # API Key to User ID Cache
        api_key_user_stats = api_key_to_user_id_cache.get_stats() if hasattr(api_key_to_user_id_cache, 'get_stats') else {
            'hits': getattr(api_key_to_user_id_cache, 'hits', 0),
            'misses': getattr(api_key_to_user_id_cache, 'misses', 0),
            'sets': getattr(api_key_to_user_id_cache, 'sets', 0),
            'hit_rate': f"{(getattr(api_key_to_user_id_cache, 'hits', 0) / max(1, getattr(api_key_to_user_id_cache, 'hits', 0) + getattr(api_key_to_user_id_cache, 'misses', 0))) * 100:.1f}%",
            'size': len(api_key_to_user_id_cache),
            'maxsize': getattr(api_key_to_user_id_cache, 'maxsize', 'N/A'),
            'ttl_seconds': getattr(api_key_to_user_id_cache, 'ttl', 'N/A')
        }
        logger.info(f"  API Key to User ID Cache: {api_key_user_stats}")
        
    except Exception as e:
        logger.error(f"Error logging cache statistics: {e}")

async def get_user_from_token_optimized(
    auth_header: str, 
    client_type: str,
    memory_graph: "MemoryGraph",
    api_key: Optional[str] = None,
    search_request: Optional["SearchRequest"] = None,
    memory_request: Optional["AddMemoryRequest"] = None,
    batch_request: Optional["BatchMemoryRequest"] = None,
    update_request: Optional["UpdateMemoryRequest"] = None,
    feedback_request: Optional["FeedbackRequest"] = None,
    sync_tiers_request: Optional["SyncTiersRequest"] = None,
    httpx_client: Optional[httpx.AsyncClient] = None,
    include_schemas: bool = False,
    url_enable_agentic_graph: Optional[bool] = None
) -> OptimizedAuthResponse:
    """
    Optimized authentication with clear separation between developer and resolved user.
    
    Three main cases:
    1. Developer is also the end user (no user_id/external_user_id provided)
    2. Developer has end user with user_id (direct mapping)
    3. Developer has end user using external_user_id (needs resolution to user_id)
    
    Args:
        auth_header: The authorization header string
        client_type: The client type ('browser_extension' or 'papr_plugin')
        api_key: Optional API key
        search_request: Optional SearchRequest for user resolution
        memory_request: Optional AddMemoryRequest for user resolution
        batch_request: Optional BatchMemoryRequest for user resolution
        update_request: Optional UpdateMemoryRequest for user resolution
        feedback_request: Optional FeedbackRequest for user resolution
        sync_tiers_request: Optional SyncTiersRequest for user resolution
        httpx_client: Optional httpx client for reuse
    
    Returns:
        OptimizedAuthResponse containing all authentication and user resolution data
    """
    try:
        start_time = time.time()
        
        # Create cache key based on raw input parameters (determines what resolution to do)
        cache_key = _generate_auth_cache_key(
            auth_header=auth_header,
            client_type=client_type,
            api_key=api_key,
            search_request=search_request,
            memory_request=memory_request,
            batch_request=batch_request,
            update_request=update_request,
            feedback_request=feedback_request,
            sync_tiers_request=sync_tiers_request
        )
        
        # Check cache first
        cached_result = auth_optimized_cache.get(cache_key)
        if cached_result:
            logger.info(f"Auth optimized cache HIT for key: {cache_key[:70]}{'...' if len(cache_key) > 70 else ''}")
            logger.info(f"Cached result details: developer_id={getattr(cached_result, 'developer_id', 'N/A')}, end_user_id={getattr(cached_result, 'end_user_id', 'N/A')}, workspace_id={getattr(cached_result, 'workspace_id', 'N/A')}")
            log_cache_stats()
            return cached_result
        
        logger.info(f"Auth optimized cache MISS for key: {cache_key[:70]}{'...' if len(cache_key) > 70 else ''}")
        logger.info(f"Starting optimized authentication...")
        
        # Step 0.5: Extract enhanced API key info first (needed for comprehensive user info)
        enhanced_api_key_info = None
        if 'APIKey ' in auth_header:
            extracted_api_key = auth_header.split('APIKey ')[1]
            logger.info(f"Pre-extracting enhanced API key info: {extracted_api_key[:10]}...")
            enhanced_api_key_info = await get_enhanced_api_key_info(extracted_api_key, memory_graph, httpx_client=httpx_client)
        elif auth_header.startswith('APIKey'):
            extracted_api_key = auth_header.replace('APIKey', '').strip()
            logger.info(f"Pre-extracting enhanced API key info (no space): {extracted_api_key[:10]}...")
            enhanced_api_key_info = await get_enhanced_api_key_info(extracted_api_key, memory_graph, httpx_client=httpx_client)
        elif api_key:
            logger.info(f"Pre-extracting enhanced API key info from parameter: {api_key[:10]}...")
            enhanced_api_key_info = await get_enhanced_api_key_info(api_key, memory_graph, httpx_client=httpx_client)
        
        # Step 1: Get developer user info (always needed)
        # This gets the full developer user information including roles, workspace, etc.
        developer_user_info_task = _get_comprehensive_user_info_parallel_mongo(
            auth_header, client_type, memory_graph, api_key, httpx_client=httpx_client, enhanced_api_key_info=enhanced_api_key_info
        )
        
        # Step 1.5: Get cached schema patterns in parallel (for search optimization)
        # This will be started after we get user info to avoid re-authentication
        schema_cache_task = None
        
        # Step 2: Extract developer user ID for resolution purposes
        developer_user_id = None
        
        # Extract developer_user_id and multi-tenant context from the enhanced API key info we already obtained
        organization_id = None
        namespace_id = None
        if enhanced_api_key_info:
            developer_user_id = enhanced_api_key_info.get('user_id')
            organization_id = enhanced_api_key_info.get('organization_id')
            namespace_id = enhanced_api_key_info.get('namespace_id')
            logger.info(f"Got developer_user_id from pre-extracted enhanced API key: {developer_user_id}")
            logger.info(f"Got multi-tenant context - org_id: {organization_id}, namespace_id: {namespace_id}")
        else:
            logger.info("No enhanced API key info available, will extract from auth header")
        
        # Initialize user_info to preserve it from Bearer token verification
        user_info = {}
        
        if not developer_user_id:
            # Fallback: extract from auth header (for Bearer/Session tokens)
            if 'Bearer ' in auth_header:
                token = auth_header.split('Bearer ')[1]
                try:
                    bearer_user_info = await User.verify_access_token(token, client_type)
                    if bearer_user_info:
                        developer_user_id = bearer_user_info['https://papr.scope.com/objectId']
                        user_info = bearer_user_info  # Preserve user_info for final result
                        logger.info(f"Got developer_user_id from Auth0 verification: {developer_user_id}")
                    else:
                        logger.error("Invalid access token")
                except Exception as e:
                    logger.error(f"Failed to verify access token: {e}")
                    pass
            elif 'Session ' in auth_header:
                # For session tokens, we'll get it from the comprehensive info
                pass
        
        # Step 3: Determine if we need user resolution
        needs_user_resolution = search_request is not None or memory_request is not None or batch_request is not None or update_request is not None or feedback_request is not None
        
        # Debug logging removed - batch resolution is working
        
        # Step 4: Run developer info and resolution in parallel
        if needs_user_resolution and developer_user_id:
            # Debug logging removed - batch resolution is working
            
            # Run resolution in parallel with developer info
            if search_request:
                logger.info(f"CRITICAL DEBUG: Creating search resolution task")
                resolution_task = _resolve_user_for_search_parallel_v2(
                    developer_user_id, search_request, api_key, httpx_client=httpx_client,
                    organization_id=organization_id, namespace_id=namespace_id
                )
            elif memory_request:
                logger.info(f"CRITICAL DEBUG: Creating memory resolution task")
                resolution_task = _resolve_user_for_memory_parallel_v2(
                    developer_user_id, memory_request, api_key, httpx_client=httpx_client,
                    organization_id=organization_id, namespace_id=namespace_id
                )
            elif batch_request:
                logger.info(f"CRITICAL DEBUG: Creating batch resolution task")
                resolution_task = _resolve_user_for_batch_parallel_v2(
                    developer_user_id, batch_request, api_key, httpx_client=httpx_client,
                    organization_id=organization_id, namespace_id=namespace_id
                )
            elif update_request:
                logger.info(f"CRITICAL DEBUG: Creating update resolution task")
                resolution_task = _resolve_user_for_update_parallel_v2(
                    developer_user_id, update_request, api_key, httpx_client=httpx_client,
                    organization_id=organization_id, namespace_id=namespace_id
                )
            elif feedback_request:
                logger.info(f"CRITICAL DEBUG: Creating feedback resolution task")
                resolution_task = _resolve_user_for_feedback_parallel_v2(
                    developer_user_id, feedback_request, api_key, httpx_client=httpx_client,
                    organization_id=organization_id, namespace_id=namespace_id
                )
            else:
                resolution_task = None
            
            if resolution_task:
                # Run both tasks in parallel
                developer_info_result, resolution_result = await asyncio.gather(
                    developer_user_info_task,
                    resolution_task,
                    return_exceptions=True
                )
            else:
                developer_info_result = await developer_user_info_task
                resolution_result = None
        else:
            # Only get developer info
            developer_info_result = await developer_user_info_task
            resolution_result = None
        
        # Step 5: Handle developer info result
        if isinstance(developer_info_result, Exception):
            logger.error(f"Failed to get developer user info: {developer_info_result}")
            raise ValueError(f"Authentication failed: {developer_info_result}")
        
        developer_workspace_id, developer_is_qwen_route, developer_user_roles, developer_user_workspace_ids = developer_info_result
        logger.info(f"🔍 TRACE AUTH STEP 0 - DEVELOPER INFO: developer_workspace_id={developer_workspace_id}")
        
        # Extract additional info from auth (session token, API key, etc.)
        session_token = None
        final_api_key = api_key
        # Track whether the API key came explicitly from the client (header/param),
        # as opposed to being auto-fetched from a Bearer user. We only mark
        # developer users when the client explicitly supplied the key.
        api_key_client_supplied = bool(final_api_key)
        
        if 'Session ' in auth_header:
            session_token = auth_header.split('Session ')[1]
        elif 'APIKey ' in auth_header:
            final_api_key = auth_header.split('APIKey ')[1]
            api_key_client_supplied = True
        elif 'Bearer ' in auth_header:
            # For Bearer tokens, handle API key based on whether developer provided one
            if final_api_key:
                # Developer provided their own API key - use it for Parse Server operations
                logger.info(f"Using developer-provided API key for Bearer token authentication: {final_api_key[:10]}...")
            elif developer_user_id:
                # No developer API key provided - get API key from Bearer token user's userAPIkey field
                try:
                    final_api_key = await get_api_key_from_user_id(developer_user_id, memory_graph, httpx_client=httpx_client)
                    if final_api_key:
                        logger.info(f"Got API key from Bearer token user_id: {final_api_key[:10]}...")
                        # IMPORTANT: Do NOT set api_key_client_supplied here; this key was auto-fetched.
                    else:
                        logger.warning(f"Could not get API key for Bearer token user_id: {developer_user_id}")
                except Exception as e:
                    logger.warning(f"Failed to get API key for Bearer token: {e}")
            # Extract session token from user_info if available
            if user_info and 'https://papr.scope.com/sessionToken' in user_info:
                session_token = user_info['https://papr.scope.com/sessionToken']
        
        # If we don't have developer_user_id yet, extract it from the comprehensive info
        if not developer_user_id:
            if 'Session ' in auth_header:
                # For session tokens, we need to verify to get user ID
                sessionToken = auth_header.split('Session ')[1]
                try:
                    parse_user = await User.verify_session_token(sessionToken, httpx_client=httpx_client)
                    if parse_user:
                        developer_user_id = parse_user.objectId
                except Exception as e:
                    logger.error(f"Session token verification failed: {e}")
                    raise ValueError(f"Invalid session token: {e}")
            elif 'APIKey ' in auth_header or auth_header.startswith('APIKey'):
                # For API keys, try to verify to get user ID
                extracted_api_key = auth_header.split('APIKey ')[1] if 'APIKey ' in auth_header else auth_header.replace('APIKey', '').strip()
                try:
                    user_info = await User.verify_api_key(extracted_api_key, httpx_client=httpx_client)
                    if user_info and isinstance(user_info, dict) and 'objectId' in user_info:
                        developer_user_id = user_info['objectId']
                        logger.info(f"Extracted developer_user_id from API key verification: {developer_user_id}")
                except Exception as e:
                    logger.error(f"API key verification failed for developer_user_id extraction: {e}")
                    raise ValueError(f"Invalid API key: {e}")
        
        # Step 6: Handle resolution result
        updated_metadata = None
        updated_batch_request = None
        resolved_user_id = developer_user_id  # Default: developer is the resolved user
        resolved_is_qwen_route = developer_is_qwen_route  # Default: use developer's setting
        resolved_user_roles = developer_user_roles  # Default: use developer's roles
        resolved_user_workspace_ids = developer_user_workspace_ids  # Default: use developer's workspaces
        resolved_workspace_id = developer_workspace_id  # Default: use developer's workspace
        
        if needs_user_resolution and resolution_result and not isinstance(resolution_result, Exception):
            if batch_request:
                updated_batch_request, resolved_user_id, resolved_is_qwen_route, resolved_user_roles, resolved_user_workspace_ids, resolved_workspace_id = resolution_result
            else:
                updated_metadata, resolved_user_id, resolved_is_qwen_route, resolved_user_roles, resolved_user_workspace_ids, resolved_workspace_id = resolution_result
            
            logger.info(f"User resolution successful - resolved_user_id: {resolved_user_id}")
            logger.info(f"Resolved isQwenRoute: {resolved_is_qwen_route}")
        elif needs_user_resolution and isinstance(resolution_result, Exception):
            logger.warning(f"User resolution failed: {resolution_result}")
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Optimized authentication timing: {total_time:.2f}ms")
        
        # Final validation: ensure we have a valid developer_user_id
        if not developer_user_id:
            logger.error(f"Failed to extract developer_user_id from auth_header: {auth_header[:30]}...")
            raise ValueError("Failed to authenticate: could not extract developer user ID")
        
        if not resolved_user_id:
            logger.warning(f"resolved_user_id is None, using developer_user_id: {developer_user_id}")
            resolved_user_id = developer_user_id
        
        # Apply fallback logic: use developer values when resolved values are None
        logger.info(f"🔍 TRACE AUTH STEP 1 - BEFORE FALLBACK: resolved_workspace_id={resolved_workspace_id}, developer_workspace_id={developer_workspace_id}")
        final_workspace_id = resolved_workspace_id if resolved_workspace_id is not None else developer_workspace_id
        final_is_qwen_route = resolved_is_qwen_route if resolved_is_qwen_route is not None else (developer_is_qwen_route if developer_is_qwen_route is not None else False)
        final_user_roles = resolved_user_roles if resolved_user_roles else developer_user_roles
        final_user_workspace_ids = resolved_user_workspace_ids if resolved_user_workspace_ids else developer_user_workspace_ids
        logger.info(f"🔍 TRACE AUTH STEP 2 - AFTER FALLBACK: final_workspace_id={final_workspace_id}")
        
        # Start cached schema lookup now that we have user_id and workspace_id
        # Only fetch schema cache for search requests with agentic graph enabled
        # URL parameter takes precedence over JSON body parameter
        # Use getattr to safely access enable_agentic_graph (not all request types have this attribute)
        final_enable_agentic_graph = url_enable_agentic_graph if url_enable_agentic_graph is not None else (getattr(search_request, 'enable_agentic_graph', False) if search_request else False)
        
        # DEBUG: Log all auth conditions for enhanced schema cache
        logger.info(f"🔍 AUTH DEBUG: search_request={bool(search_request)}, resolved_user_id={resolved_user_id}, final_workspace_id={final_workspace_id}, final_enable_agentic_graph={final_enable_agentic_graph}")
        
        # Note: Schema fetching will be done after multi-tenant fields are extracted (lines 728-737)
        
        # Log fallback usage for debugging
        if final_workspace_id != resolved_workspace_id:
            logger.info(f"Using developer workspace_id fallback: {final_workspace_id} (resolved was {resolved_workspace_id})")
        if final_is_qwen_route != resolved_is_qwen_route:
            logger.info(f"Using developer is_qwen_route fallback: {final_is_qwen_route} (resolved was {resolved_is_qwen_route})")
        if final_user_roles != resolved_user_roles:
            logger.info(f"Using developer user_roles fallback: {len(final_user_roles)} roles (resolved had {len(resolved_user_roles) if resolved_user_roles else 0})")
        if final_user_workspace_ids != resolved_user_workspace_ids:
            logger.info(f"Using developer user_workspace_ids fallback: {len(final_user_workspace_ids)} workspaces (resolved had {len(resolved_user_workspace_ids) if resolved_user_workspace_ids else 0})")
        
        # Mark user as developer ONLY when the API key was explicitly provided by the client
        # (APIKey header or X-API-Key param). Skip for Bearer-only flows (e.g., ChatGPT plugin),
        # where we auto-fetch the user's own API key.
        if final_api_key and developer_user_id and api_key_client_supplied:
            # Use background task to avoid blocking the response
            # Don't pass httpx_client to background task as it may be closed
            try:
                asyncio.create_task(mark_user_as_developer_if_needed(
                    developer_user_id, final_api_key, memory_graph, None  # Let function create its own client
                ))
            except Exception as e:
                logger.warning(f"Failed to create developer marking task: {e}")
        else:
            if final_api_key and developer_user_id and not api_key_client_supplied:
                logger.info("Skipping developer marking: API key was auto-fetched via Bearer flow (not client-supplied)")
        
        # Determine multi-tenant fields from enhanced API key info (already extracted earlier)
        # organization_id and namespace_id were already extracted at line 476-481
        is_legacy_auth = True  # Default to legacy
        auth_type = "legacy"   # Default to legacy
        api_key_info = None

        if enhanced_api_key_info:
            # Extract remaining multi-tenant information from enhanced API key
            is_legacy_auth = enhanced_api_key_info.get('is_legacy_auth', True)
            auth_type = "organization" if not is_legacy_auth else "legacy"
            api_key_info = enhanced_api_key_info

            logger.info(f"Multi-tenant auth detected - org_id: {organization_id}, namespace_id: {namespace_id}, auth_type: {auth_type}")

        # Now that multi-tenant fields are set, fetch schema cache if needed for agentic search
        if (search_request and resolved_user_id and final_workspace_id and 
            final_enable_agentic_graph):  # Only fetch if agentic graph is enabled
            logger.info(f"🚀 ENHANCED AGENTIC CACHE: Starting parallel fetch of ActiveNodeRel + UserGraphSchema")
            logger.info(
                "Agentic schema cache context: workspace_id=%s, org_id=%s, namespace_id=%s",
                final_workspace_id,
                organization_id,
                namespace_id,
            )
            
            # Start both tasks in parallel
            active_patterns_task = _get_cached_schema_patterns_direct(
                user_object_id=resolved_user_id,
                workspace_object_id=final_workspace_id,
                httpx_client=httpx_client,
                skip_registration=True,  # SEARCH OPTIMIZATION: Skip schema registration for search operations
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            # NEW: Also fetch UserGraphSchema for property definitions
            # Pass multi-tenant context for flexible ACL (matches ADD operation logic)
            user_schemas_task = _get_user_schemas_for_agentic_search(
                user_object_id=resolved_user_id,
                workspace_object_id=final_workspace_id,
                httpx_client=httpx_client,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            schema_cache_task = _combine_schema_cache_results(
                active_patterns_task,
                user_schemas_task
            )
        elif search_request and not final_enable_agentic_graph:
            # Safely get enable_agentic_graph - only SearchRequest has this attribute
            request_enable_agentic = getattr(search_request, 'enable_agentic_graph', None)
            logger.info(f"🚀 CONDITIONAL SCHEMA CACHE: Skipping schema cache task (url_param={url_enable_agentic_graph}, json_body={request_enable_agentic}, final={final_enable_agentic_graph})")
        else:
            logger.info(
                "🚀 CONDITIONAL SCHEMA CACHE: Skipping schema cache task (search_request=%s, resolved_user_id=%s, workspace_id=%s)",
                bool(search_request),
                resolved_user_id,
                final_workspace_id,
            )

        # Fetch user schemas in parallel if requested
        user_schemas = []
        if include_schemas and resolved_user_id:
            try:
                logger.info(f"Fetching schemas in parallel for user {resolved_user_id}")
                schema_fetch_start = time.time()
                
                # Fetch schemas using existing httpx_client for efficiency
                from services.schema_service import SchemaService
                schema_service = SchemaService()
                
                # Extract multi-tenant context for schema fetching
                organization_id = api_key_info.get('organization_id') if api_key_info else None
                namespace_id = api_key_info.get('namespace_id') if api_key_info else None
                
                logger.info(f"Fetching schemas with multi-tenant context: org_id={organization_id}, namespace_id={namespace_id}")
                user_schemas = await schema_service.get_active_schemas(
                    resolved_user_id, 
                    final_workspace_id,
                    organization_id,
                    namespace_id
                )
                
                schema_fetch_time = (time.time() - schema_fetch_start) * 1000
                logger.info(f"Schema fetch completed in {schema_fetch_time:.2f}ms - found {len(user_schemas)} active schemas")
                
            except Exception as e:
                logger.warning(f"Failed to fetch schemas in parallel: {e}")
                user_schemas = []
        
        # Get cached schema from parallel task if available
        cached_schema = None
        logger.info(f"🔧 DEBUG AUTH: schema_cache_task is_none={schema_cache_task is None}, type={type(schema_cache_task)}")
        if schema_cache_task:
            try:
                cached_schema = await schema_cache_task
                logger.info(f"Schema cache task completed: {type(cached_schema)}, keys={list(cached_schema.keys()) if isinstance(cached_schema, dict) else 'not_a_dict'}")
            except Exception as e:
                logger.warning(f"Schema cache task failed: {e}")
                cached_schema = None
        
        # Create the response object with multi-tenant support
        result = OptimizedAuthResponse(
            developer_id=developer_user_id,
            end_user_id=resolved_user_id,
            session_token=session_token,
            user_info=user_info,
            api_key=final_api_key,
            workspace_id=final_workspace_id,
            is_qwen_route=final_is_qwen_route,
            user_roles=final_user_roles,
            user_workspace_ids=final_user_workspace_ids,
            updated_metadata=updated_metadata,
            updated_batch_request=updated_batch_request,
            user_schemas=user_schemas,
            cached_schema=cached_schema,
            # Multi-tenant fields
            organization_id=organization_id,
            namespace_id=namespace_id,
            is_legacy_auth=is_legacy_auth,
            auth_type=auth_type,
            api_key_info=api_key_info
        )
        
        # Log the final result
        logger.info(f"Final auth result:")
        logger.info(f"  developer_id: {result.developer_id}")
        logger.info(f"  end_user_id: {result.end_user_id}")
        logger.info(f"  workspace_id: {result.workspace_id}")
        logger.info(f"  is_qwen_route: {result.is_qwen_route}")
        logger.info(
            f"  organization_id: {result.organization_id}, namespace_id: {result.namespace_id}, auth_type: {result.auth_type}"
        )
        
        # Cache the result using our custom TTLCache API (only if workspace_id is not None)
        if result.workspace_id is not None:
            auth_optimized_cache.set(cache_key, result)
            logger.info(f"Stored result details: developer_id={result.developer_id}, end_user_id={result.end_user_id}, workspace_id={result.workspace_id}")
        else:
            logger.warning(f"NOT caching auth result due to workspace_id=None - developer_id={result.developer_id}, end_user_id={result.end_user_id}")
            logger.warning("This indicates incomplete authentication data that should not be cached")
        
        log_cache_stats()
        return result
        
    except ValueError as e:
        logger.error(f"Validation error in get_user_from_token_optimized: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_user_from_token_optimized: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise ValueError(f"Authentication failed: {str(e)}")

async def _verify_auth_and_get_user_id(
    auth_header: str, 
    client_type: str,
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]:
    """Verify authentication and get basic user info"""
    auth_start = time.time()
    user_id = None  # Always define at the top
    sessionToken = None
    user_info = None
    final_api_key = None
    
    logger.info(f"_verify_auth_and_get_user_id - auth_header: {auth_header}")
    logger.info(f"_verify_auth_and_get_user_id - client_type: {client_type}")
    logger.info(f"_verify_auth_and_get_user_id - api_key: {api_key}")
    
    if 'Bearer ' in auth_header:
        token = auth_header.split('Bearer ')[1]
        logger.info(f"Optimized auth - Got the access_token: {token[:5]}...")

        try:
            user_info = await User.verify_access_token(token, client_type)
            if not user_info:
                logger.error("Invalid access token")
                raise ValueError("Invalid access token")

            user_id = user_info['https://papr.scope.com/objectId']
            sessionToken = user_info['https://papr.scope.com/sessionToken']
            logger.info(f"Bearer token verification successful - user_id: {user_id}")
        except Exception as e:
            logger.error(f"Bearer token verification failed: {e}")
            raise ValueError(f"Invalid access token: {e}")
            
    elif 'Session ' in auth_header:
        sessionToken = auth_header.split('Session ')[1]
        logger.info(f"Optimized auth - Got the sessionToken: {sessionToken[:5]}...")

        try:
            user_info = await User.verify_session_token(sessionToken)
            if not user_info:
                logger.error("Invalid session token")
                raise ValueError("Invalid session token")

            user_id = user_info.objectId
            logger.info(f"Session token verification successful - user_id: {user_id}")
        except Exception as e:
            logger.error(f"Session token verification failed: {e}")
            raise ValueError(f"Invalid session token: {e}")
            
    elif 'APIKey ' in auth_header:
        final_api_key = auth_header.split('APIKey ')[1]
        logger.info(f"Optimized auth - Got the api_key: {final_api_key[:5]}...")

        try:
            user_info = await User.verify_api_key(final_api_key)
            if not user_info or not isinstance(user_info, dict) or 'objectId' not in user_info:
                logger.error("Invalid API key")
                raise ValueError("Invalid API key")

            user_id = user_info['objectId']
            logger.info(f"API key verification successful - user_id: {user_id}")
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            raise ValueError(f"Invalid API key: {e}")
            
    elif auth_header.startswith('APIKey'):
        # Handle case where there's no space after APIKey
        final_api_key = auth_header.replace('APIKey', '').strip()
        logger.info(f"Optimized auth - Got the api_key (no space): {final_api_key[:5]}...")

        try:
            user_info = await User.verify_api_key(final_api_key)
            if not user_info or not isinstance(user_info, dict) or 'objectId' not in user_info:
                logger.error("Invalid API key (no space)")
                raise ValueError("Invalid API key")

            user_id = user_info['objectId']
            logger.info(f"API key verification successful (no space) - user_id: {user_id}")
        except Exception as api_e:
            logger.error(f"API key verification failed (no space): {api_e}")
            raise ValueError(f"Invalid API key: {api_e}")
    else:
        logger.error(f"Unknown auth header format: {auth_header}")
        raise ValueError("Invalid Authorization header format")
    
    if not user_id:
        logger.error("user_id is None after authentication")
        raise ValueError("Could not extract user_id from authentication")
    
    auth_time = (time.time() - auth_start) * 1000
    logger.info(f"Bearer token verification took: {auth_time:.2f}ms")
    logger.info(f"Final result - user_id: {user_id}, sessionToken: {sessionToken[:10] if sessionToken else None}, final_api_key: {final_api_key[:10] if final_api_key else None}")
    
    return user_id, sessionToken, user_info, final_api_key

async def _get_comprehensive_user_info_parallel(
    auth_header: str,
    client_type: str,
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> Tuple[Optional[str], Optional[bool], List[str], List[str]]:
    """Get comprehensive user info in parallel with authentication - SUPER OPTIMIZED VERSION"""
    try:
        start_time = time.time()
        logger.info(f"Starting SUPER OPTIMIZED comprehensive user info...")
        
        user_id = None
        user_obj = None
        
        # Step 1: Fast auth verification and user_id extraction
        step1_start = time.time()
        if 'Bearer ' in auth_header:
            token = auth_header.split('Bearer ')[1]
            user_info = await User.verify_access_token(token, client_type, httpx_client=httpx_client)
            user_id = user_info.get('https://papr.scope.com/objectId')
            if not user_id:
                logger.warning("Could not extract user_id from Auth0 token")
                return None, None, [], []
                
        elif 'Session ' in auth_header:
            sessionToken = auth_header.split('Session ')[1]
            parse_user = await User.verify_session_token(sessionToken, httpx_client=httpx_client)
            if parse_user:
                user_id = parse_user.objectId
            else:
                logger.warning("Could not extract user_id from session token")
                return None, None, [], []
                
        elif 'APIKey ' in auth_header or auth_header.startswith('APIKey'):
            # For API keys, we can get everything in one optimized call
            extracted_api_key = auth_header.split('APIKey ')[1] if 'APIKey ' in auth_header else auth_header.replace('APIKey', '').strip()
            
            # Use the already optimized verify_api_key that includes all needed fields
            user_obj = await User.verify_api_key(extracted_api_key, httpx_client=httpx_client)
            if user_obj:
                user_id = user_obj.get('objectId')
                logger.info(f"Got user_id from API key verification: {user_id}")
            else:
                logger.warning("Could not extract user_id from API key")
                return None, None, [], []
        else:
            logger.warning("Could not extract user_id for user info")
            return None, None, [], []
        
        step1_time = (time.time() - step1_start) * 1000
        logger.info(f"Step 1 (auth verification & user_id extraction) took: {step1_time:.2f}ms")
        
        # Step 2: If we don't have full user_obj yet, get it with all needed includes in ONE call
        step2_start = time.time()
        if not user_obj:
            headers = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "Content-Type": "application/json",
                "X-Parse-Master-Key": PARSE_MASTER_KEY
            }
            user_url = f"{PARSE_SERVER_URL}/parse/classes/_User/{user_id}"
            user_params = {
                "include": "isSelectedWorkspaceFollower,isSelectedWorkspaceFollower.workspace",
                "keys": "objectId,username,email,isQwenRoute,isSelectedWorkspaceFollower,isSelectedWorkspaceFollower.workspace"
            }
            
            if httpx_client:
                response = await httpx_client.get(user_url, headers=headers, params=user_params)
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(user_url, headers=headers, params=user_params)
            
            response.raise_for_status()
            user_obj = response.json()
        
        step2_time = (time.time() - step2_start) * 1000
        logger.info(f"Step 2 (get user object with includes) took: {step2_time:.2f}ms")
        
        # Step 3: Extract basic user info
        step3_start = time.time()
        is_qwen_route = user_obj.get("isQwenRoute", None) if user_obj else None
        logger.info(f"isQwenRoute for user_id {user_id}: {is_qwen_route}")
        
        # Extract workspace_id from included workspace follower
        # Handle both MongoDB format (_p_isSelectedWorkspaceFollower) and Parse format (isSelectedWorkspaceFollower)
        workspace_id = None
        selected_follower = None
        if user_obj:
            # Try Parse format first (for Parse Server responses)
            selected_follower = user_obj.get('isSelectedWorkspaceFollower')
            # If not found, try MongoDB format (for MongoDB responses)  
            if not selected_follower:
                selected_follower = user_obj.get('_p_isSelectedWorkspaceFollower')
        
        if selected_follower and isinstance(selected_follower, dict):
            workspace = selected_follower.get('workspace')
            if workspace and isinstance(workspace, dict):
                workspace_id = workspace.get('objectId')
        
        step3_time = (time.time() - step3_start) * 1000
        logger.info(f"Step 3 (extract basic user info) took: {step3_time:.2f}ms")
        
        # Step 4: Get roles and workspaces in parallel (these are the only remaining calls)
        step4_start = time.time()
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "Content-Type": "application/json",
            "X-Parse-Master-Key": PARSE_MASTER_KEY
        }
        
        # Roles query
        roles_url = f"{PARSE_SERVER_URL}/parse/classes/_Join:users:_Role"
        roles_params = {
            "where": json.dumps({"relatedId": user_id}),
            "include": "owningId"
        }
        
        # Workspaces query  
        workspaces_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower"
        workspaces_params = {
            "where": json.dumps({
                "user": {
                    "__type": "Pointer",
                    "className": "_User", 
                    "objectId": user_id
                }
            }),
            "include": "workspace",
            "keys": "workspace"
        }
        
        parallel_start = time.time()
        # Execute both queries in parallel
        if httpx_client:
            roles_task = httpx_client.get(roles_url, headers=headers, params=roles_params)
            workspaces_task = httpx_client.get(workspaces_url, headers=headers, params=workspaces_params)
            roles_response, workspaces_response = await asyncio.gather(roles_task, workspaces_task, return_exceptions=True)
        else:
            async with httpx.AsyncClient() as client:
                roles_task = client.get(roles_url, headers=headers, params=roles_params)
                workspaces_task = client.get(workspaces_url, headers=headers, params=workspaces_params)
                roles_response, workspaces_response = await asyncio.gather(roles_task, workspaces_task, return_exceptions=True)
        
        parallel_time = (time.time() - parallel_start) * 1000
        logger.info(f"Step 4a (parallel roles & workspaces queries) took: {parallel_time:.2f}ms")
        
        # Process roles result
        roles_processing_start = time.time()
        user_roles = []
        if not isinstance(roles_response, Exception):
            try:
                roles_response.raise_for_status()
                roles_data = roles_response.json()
                role_ids = [link["owningId"] for link in roles_data.get("results", [])]
                
                if role_ids:
                    # Get role names in batch
                    roles_names_start = time.time()
                    roles_names_url = f"{PARSE_SERVER_URL}/parse/classes/_Role"
                    roles_names_params = {
                        "where": json.dumps({"objectId": {"$in": role_ids}}),
                        "keys": "name"
                    }
                    if httpx_client:
                        roles_names_response = await httpx_client.get(roles_names_url, headers=headers, params=roles_names_params)
                    else:
                        async with httpx.AsyncClient() as client:
                            roles_names_response = await client.get(roles_names_url, headers=headers, params=roles_names_params)
                    
                    roles_names_response.raise_for_status()
                    roles_names_data = roles_names_response.json()
                    user_roles = [role.get("name") for role in roles_names_data.get("results", []) if role.get("name")]
                    
                    roles_names_time = (time.time() - roles_names_start) * 1000
                    logger.info(f"Step 4b (get role names) took: {roles_names_time:.2f}ms")
                    
            except Exception as e:
                logger.warning(f"Failed to get roles for user {user_id}: {e}")
        else:
            logger.warning(f"Failed to get roles for user {user_id}: {roles_response}")
        
        roles_processing_time = (time.time() - roles_processing_start) * 1000
        logger.info(f"Step 4b (process roles) took: {roles_processing_time:.2f}ms")
        
        # Process workspaces result
        workspaces_processing_start = time.time()
        user_workspace_ids = []
        if not isinstance(workspaces_response, Exception):
            try:
                workspaces_response.raise_for_status()
                workspaces_data = workspaces_response.json()
                user_workspace_ids = []
                for follower in workspaces_data.get("results", []):
                    workspace = follower.get("workspace")
                    if workspace and isinstance(workspace, dict):
                        workspace_obj_id = workspace.get("objectId")
                        if workspace_obj_id:
                            user_workspace_ids.append(workspace_obj_id)
                            
            except Exception as e:
                logger.warning(f"Failed to get workspaces for user {user_id}: {e}")
        else:
            logger.warning(f"Failed to get workspaces for user {user_id}: {workspaces_response}")
        
        workspaces_processing_time = (time.time() - workspaces_processing_start) * 1000
        logger.info(f"Step 4c (process workspaces) took: {workspaces_processing_time:.2f}ms")
        
        step4_time = (time.time() - step4_start) * 1000
        logger.info(f"Step 4 (total roles & workspaces) took: {step4_time:.2f}ms")
        
        user_info_time = (time.time() - start_time) * 1000
        logger.info(f"SUPER OPTIMIZED comprehensive user info took: {user_info_time:.2f}ms")
        
        return workspace_id, is_qwen_route, user_roles, user_workspace_ids
        
    except Exception as e:
        logger.error(f"Error getting comprehensive user info: {e}")
        return None, None, [], []

async def _get_user_data_mongo(db, user_id: str) -> Dict[str, Any]:
    """Get user data directly from MongoDB"""
    try:
        user_doc = db["_User"].find_one({"_id": user_id})
        if not user_doc:
            return {"workspace_id": None, "is_qwen_route": None}

        logger.info(f"Found user document for {user_id}: {user_doc}")

        # Get selected workspace follower (Parse pointer field)
        selected_follower = user_doc.get('_p_isSelectedWorkspaceFollower')
        workspace_id = None
        
        logger.info(f"_p_isSelectedWorkspaceFollower value for user {user_id}: {selected_follower}")
        
        if selected_follower:
            # Handle Parse pointer field format: "workspace_follower$mGqLmvPhYY"
            follower_id_to_lookup = selected_follower
            if selected_follower.startswith("workspace_follower$"):
                # Strip the "workspace_follower$" prefix to get the actual document ID
                follower_id_to_lookup = selected_follower.split("$", 1)[1]
            
            # Follow the pointer to get workspace
            follower_doc = db["workspace_follower"].find_one({"_id": follower_id_to_lookup})
            logger.info(f"Found workspace_follower document: {follower_doc}")
            
            if follower_doc:
                workspace_pointer = follower_doc.get('_p_workspace')
                if workspace_pointer and workspace_pointer.startswith('WorkSpace$'):
                    workspace_id = workspace_pointer.split('$', 1)[1]
                    logger.info(f"🔍 TRACE AUTH MONGO STEP 3 - Found workspace_id via isSelectedWorkspaceFollower: {workspace_id}")
        else:
            logger.warning(f"User {user_id} has no isSelectedWorkspaceFollower set - this may be a data integrity issue")
            
            # Let's check if there are any workspace_follower records for this user for debugging
            debug_followers = list(db["workspace_follower"].find(
                {"_p_user": f"_User${user_id}"},
                {"_id": 1, "_p_workspace": 1}
            ).limit(5))
            logger.warning(f"Available workspace_follower records for user {user_id}: {debug_followers}")

        return {
            "workspace_id": workspace_id,
            "is_qwen_route": user_doc.get("isQwenRoute")
        }
    except Exception as e:
        logger.error(f"Error getting user data from MongoDB: {e}")
        return {"workspace_id": None, "is_qwen_route": None}

async def _get_user_roles_mongo(db, user_id: str) -> List[str]:
    """Get user roles directly from MongoDB"""
    try:
        # Find role links for this user
        role_links = list(db["_Join:users:_Role"].find({"relatedId": user_id}))
        role_ids = [link["owningId"] for link in role_links]
        
        # Get role names
        roles = list(db["_Role"].find({"_id": {"$in": role_ids}}))
        role_names = [role['name'] for role in roles]
        
        return role_names
    except Exception as e:
        logger.error(f"Error getting user roles from MongoDB: {e}")
        return []

async def _get_user_workspaces_mongo(db, user_id: str) -> List[str]:
    """Get user workspaces directly from MongoDB"""
    try:
        pointer_value = f"_User${user_id}"
        workspaces_follower = list(db["workspace_follower"].find({"_p_user": pointer_value}))
        
        workspace_ids = []
        for follower in workspaces_follower:
            pointer = follower.get('_p_workspace')
            if pointer and pointer.startswith('WorkSpace$'):
                workspace_id = pointer.split('$', 1)[1]
                workspace_ids.append(workspace_id)
        
        return workspace_ids
    except Exception as e:
        logger.error(f"Error getting user workspaces from MongoDB: {e}")
        return []

async def _verify_api_key_mongo(db, api_key: str, memory_graph=None) -> Optional[Dict[str, Any]]:
    """Verify API key directly via MongoDB - SUPER OPTIMIZED VERSION
    
    Args:
        db: MongoDB database instance
        api_key: API key to verify
        memory_graph: MemoryGraph instance for reconnection if needed
    """
    try:
        logger.info(f"_verify_api_key_mongo called with API key: {api_key[:10]}... and db: {type(db)}")
        start_time = time.time()
        
        # STEP 1: Fast user lookup by API key (optimized for DocumentDB)
        user_query_start = time.time()
        # Use with_options for DocumentDB optimization with hint for index usage
        from pymongo import ReadPreference
        
        max_retry_attempts = 2  # Try up to 2 times to reconnect if connection is closed
        collection = None
        for attempt in range(max_retry_attempts + 1):
            try:
                collection = db["_User"].with_options(read_preference=ReadPreference.PRIMARY_PREFERRED)
                # Test that the collection is actually usable by trying a simple operation
                _ = collection.estimated_document_count()
                break  # Success, exit retry loop
            except Exception as conn_err:
                # Handle "Cannot use MongoClient after close" and other connection errors
                conn_err_str = str(conn_err).lower()
                if "after close" in conn_err_str or "closed" in conn_err_str or "connection" in conn_err_str:
                    if attempt < max_retry_attempts and memory_graph:
                        logger.warning(f"MongoDB connection is closed/unhealthy, attempting to reconnect (attempt {attempt + 1}/{max_retry_attempts + 1})")
                        # Try to get a fresh database reference from the singleton
                        from services.mongo_client import get_mongo_db
                        fresh_db = get_mongo_db()
                        if fresh_db is not None:
                            logger.info("Got fresh MongoDB connection from singleton")
                            db = fresh_db
                            memory_graph.db = fresh_db  # Update memory_graph's reference
                            continue  # Retry the operation
                        else:
                            # Singleton couldn't reconnect, try memory_graph reconnect
                            reconnected = await asyncio.to_thread(memory_graph.reconnect_mongodb)
                            if reconnected:
                                logger.info("MongoDB reconnection successful via memory_graph")
                                db = memory_graph.db  # Use the new db instance
                                continue  # Retry the operation
                            else:
                                logger.error("MongoDB reconnection failed on all attempts")
                                return None
                    else:
                        logger.error(f"MongoDB connection is closed and cannot reconnect after {max_retry_attempts} attempts: {conn_err}")
                        return None
                else:
                    # Different error, raise it
                    raise
        
        # If we couldn't get a working collection after all retries, give up
        if collection is None:
            logger.error("Failed to obtain a working MongoDB collection after retries")
            return None
        
        # Primary query approach - try standard equality first
        query = {"userAPIkey": api_key}
        logger.info(f"Executing MongoDB query: {query}")
        
        try:
            # Use hint only if the exact index exists; avoids planner errors on DocumentDB
            has_user_apikey_index = False
            try:
                for ix in collection.list_indexes():
                    try:
                        # ix.key is an SON mapping of index keys
                        if list(ix.key.items()) == [("userAPIkey", 1)]:
                            has_user_apikey_index = True
                            break
                    except Exception:
                        continue
            except Exception as idx_err:
                logger.info(f"Could not enumerate indexes (ok to ignore on managed services): {idx_err}")

            if has_user_apikey_index:
                cursor = collection.find(query).hint([("userAPIkey", 1)]).limit(1)
                user_doc = next(cursor, None)
                logger.info(f"Query with hint result: {user_doc is not None}")
            else:
                user_doc = collection.find_one(query)
                logger.info("Query without hint (index missing) executed")
        except Exception as e:
            logger.warning(f"Index hint check/usage failed, falling back without hint: {e}")
            user_doc = collection.find_one(query)
            logger.info(f"Query without hint result: {user_doc is not None}")
        
        # If standard query fails, use regex approach (which we know works)
        if not user_doc:
            logger.info("Standard query failed, using regex approach...")
            import re
            user_doc = collection.find_one({"userAPIkey": {"$regex": f"^{re.escape(api_key)}$"}})
            logger.info(f"Regex query result: {user_doc is not None}")
            
            # If still no result, there might be a real issue
            if not user_doc:
                logger.warning("Even regex query failed - this suggests a deeper issue")
        user_query_time = (time.time() - user_query_start) * 1000
        logger.info(f"MongoDB user lookup by API key took: {user_query_time:.2f}ms")
        
        if not user_doc:
            logger.warning(f"No user document found for API key: {api_key[:10]}...")
            return None
        
        user_id = user_doc["_id"]
        logger.info(f"Found user in MongoDB for API key verification")
        
        # STEP 2: Get workspace_id using _p_isSelectedWorkspaceFollower
        workspace_id = None
        
        # Check if _p_isSelectedWorkspaceFollower exists and is not None (Parse pointer field)
        selected_follower_id = user_doc.get("_p_isSelectedWorkspaceFollower")
        logger.info(f"_p_isSelectedWorkspaceFollower value for user {user_id}: {selected_follower_id}")
        
        if selected_follower_id:
            workspace_query_start = time.time()
            
            # Handle Parse pointer field format: "workspace_follower$mGqLmvPhYY"
            follower_id_to_lookup = selected_follower_id
            if selected_follower_id.startswith("workspace_follower$"):
                # Strip the "workspace_follower$" prefix to get the actual document ID
                follower_id_to_lookup = selected_follower_id.split("$", 1)[1]
            
            logger.info(f"Looking up workspace_follower with _id: {follower_id_to_lookup}")
            follower_doc = db["workspace_follower"].find_one(
                {"_id": follower_id_to_lookup},
                {"_p_workspace": 1}
            )
            workspace_query_time = (time.time() - workspace_query_start) * 1000
            logger.info(f"MongoDB workspace lookup (via isSelectedWorkspaceFollower) took: {workspace_query_time:.2f}ms")
            logger.info(f"Found workspace_follower document: {follower_doc}")
            
            if not follower_doc:
                logger.warning(f"No workspace_follower document found with _id: {follower_id_to_lookup}")
                # Let's see what workspace_follower documents exist for this user
                debug_query = {"_p_user": f"_User${user_id}"}
                debug_followers = list(db["workspace_follower"].find(debug_query).limit(5))
                logger.warning(f"Available workspace_follower documents for user {user_id}: {debug_followers}")
            
            if follower_doc and follower_doc.get("_p_workspace"):
                workspace_pointer = follower_doc["_p_workspace"]
                if workspace_pointer and workspace_pointer.startswith("WorkSpace$"):
                    workspace_id = workspace_pointer[10:]  # Remove "WorkSpace$" prefix
                    logger.info(f"Found workspace_id via isSelectedWorkspaceFollower: {workspace_id}")
        else:
            logger.warning(f"User {user_id} has no _p_isSelectedWorkspaceFollower set - this may be a data integrity issue")
            logger.warning(f"Full user document fields: {list(user_doc.keys())}")
            
            # Let's check if there are any workspace_follower records for this user for debugging
            debug_followers = list(db["workspace_follower"].find(
                {"_p_user": f"_User${user_id}"},
                {"_id": 1, "_p_workspace": 1}
            ).limit(5))
            logger.warning(f"Available workspace_follower records for user {user_id}: {debug_followers}")
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"MongoDB API key verification (optimized) took: {total_time:.2f}ms for workspace_id: {workspace_id}")
        
        return {
            "objectId": user_doc["_id"],
            "username": user_doc.get("username"),
            "email": user_doc.get("email"), 
            "isQwenRoute": user_doc.get("isQwenRoute"),
            "workspace_id": workspace_id
        }
        
    except Exception as e:
        logger.error(f"Error verifying API key via MongoDB: {e}")
        return None

async def _get_comprehensive_user_info_parallel_mongo(
    auth_header: str,
    client_type: str,
    memory_graph: "MemoryGraph",
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None,
    enhanced_api_key_info: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Optional[bool], List[str], List[str], Optional[str], Optional[str]]:
    """Get comprehensive user info in parallel with authentication - MONGO SUPER OPTIMIZED VERSION"""
    try:
        start_time = time.time()
        logger.info(f"Starting MONGO SUPER OPTIMIZED comprehensive user info...")
        
        if not memory_graph.mongo_client:
            logger.warning("MongoDB not available, falling back to Parse Server")
            return await _get_comprehensive_user_info_parallel(auth_header, client_type, api_key, httpx_client)
        
        db = memory_graph.db
        logger.info(f"MongoDB client available: {memory_graph.mongo_client is not None}, db: {type(db)}")
        
        user_id = None
        user_obj = None
        
        # Step 1: Fast auth verification and user_id extraction via MongoDB
        step1_start = time.time()
        
        # Prioritize API key authentication if available (more reliable)
        if api_key:
            # Use enhanced API key info if provided, otherwise get it
            if not enhanced_api_key_info:
                enhanced_api_key_info = await get_enhanced_api_key_info(api_key, memory_graph, httpx_client=httpx_client)
            
            if enhanced_api_key_info:
                user_id = enhanced_api_key_info.get('user_id')
                logger.info(f"🔍 TRACE AUTH MONGO STEP 0 - Got user_id from enhanced API key: {user_id}")
            else:
                # Fallback to legacy MongoDB verification
                user_obj = await _verify_api_key_mongo(db, api_key, memory_graph)
                if user_obj and isinstance(user_obj, dict) and 'objectId' in user_obj:
                    user_id = user_obj['objectId']
                    logger.info(f"🔍 TRACE AUTH MONGO STEP 0 - Got user_id from legacy API key: {user_id}")
                else:
                    logger.warning(f"MongoDB API key verification failed for key: {api_key[:10]}...")
                    return None, None, [], []
        elif 'Bearer ' in auth_header:
            token = auth_header.split('Bearer ')[1]
            from services.user_utils import User
            user_info = await User.verify_access_token(token, client_type, httpx_client=httpx_client)
            user_id = user_info.get('https://papr.scope.com/objectId')
            if not user_id:
                logger.warning("Could not get user_id from Bearer token")
                return None, None, [], []
                
        elif 'Session ' in auth_header:
            sessionToken = auth_header.split('Session ')[1]
            from services.user_utils import User
            parse_user = await User.verify_session_token(sessionToken, httpx_client=httpx_client)
            if parse_user:
                user_id = parse_user.objectId
            else:
                logger.warning("Could not verify session token")
                return None, None, [], []
                
        elif 'APIKey ' in auth_header or auth_header.startswith('APIKey'):
            # Extract API key
            if 'APIKey ' in auth_header:
                extracted_api_key = auth_header.split('APIKey ')[1]
            else:
                extracted_api_key = auth_header.replace('APIKey', '').strip()
            
            # Check if we have enhanced API key info for this key (organization-based API keys)
            if enhanced_api_key_info and enhanced_api_key_info.get('user_id'):
                user_id = enhanced_api_key_info.get('user_id')
                logger.info(f"🔍 TRACE AUTH MONGO STEP 0 - Got user_id from enhanced API key: {user_id}")
            else:
                # Fallback: Use MongoDB for legacy API key verification
                user_obj = await _verify_api_key_mongo(db, extracted_api_key, memory_graph)
                if user_obj and isinstance(user_obj, dict) and 'objectId' in user_obj:
                    user_id = user_obj['objectId']
                    logger.info(f"Got user_id from MongoDB API key verification: {user_id}")
                else:
                    logger.warning(f"MongoDB API key verification failed for key: {extracted_api_key[:10]}...")
                    # Fallback to Parse Server for API key verification
                    logger.info("Falling back to Parse Server for API key verification...")
                    # Ensure User is imported (may not be in scope due to local imports)
                    from services.user_utils import User as UserClass
                    parse_user = await UserClass.verify_api_key(extracted_api_key, httpx_client=httpx_client)
                    if parse_user:
                        user_id = parse_user.objectId
                        logger.info(f"Got user_id from Parse Server API key verification: {user_id}")
                    else:
                        logger.warning("Parse Server API key verification also failed")
                        return None, None, [], []
        else:
            logger.warning(f"Unsupported auth header format: {auth_header[:20]}...")
            return None, None, [], []
        
        step1_time = (time.time() - step1_start) * 1000
        logger.info(f"Step 1 (MongoDB auth verification & user_id extraction) took: {step1_time:.2f}ms")
        
        # Step 2: Extract user data (workspace_id now comes from the single optimized query!)
        step2_start = time.time()
        
        # Check if we have enhanced API key info with workspace (organization-based API key)
        if enhanced_api_key_info and enhanced_api_key_info.get('workspace_id'):
            workspace_id = enhanced_api_key_info.get('workspace_id')
            logger.info(f"🔍 TRACE AUTH MONGO STEP 0.5 - Got workspace_id from organization API key: {workspace_id}")
            # Still need to get isQwenRoute from user data
            if not user_obj:
                user_data = await _get_user_data_mongo(db, user_id)
                is_qwen_route = user_data.get("is_qwen_route")
            else:
                is_qwen_route = user_obj.get("isQwenRoute")
        elif not user_obj:
            user_data = await _get_user_data_mongo(db, user_id)
            is_qwen_route = user_data.get("is_qwen_route")
            workspace_id = user_data.get("workspace_id")
            logger.info(f"🔍 TRACE AUTH MONGO STEP 1 - Got workspace_id from single query not user_obj: {workspace_id}")
        else:
            # Extract from user_obj that we got from optimized API key verification
            is_qwen_route = user_obj.get("isQwenRoute")
            workspace_id = user_obj.get("workspace_id")  # Already included in the single query!
            logger.info(f"🔍 TRACE AUTH MONGO STEP 2 - Got workspace_id from single query: {workspace_id}")
        
        step2_time = (time.time() - step2_start) * 1000
        logger.info(f"Step 2 (extract user data - no extra queries!) took: {step2_time:.2f}ms")
        
        # Step 3: Get roles and workspaces in parallel via MongoDB - SUPER FAST!
        step3_start = time.time()
        #roles_task = _get_user_roles_mongo(db, user_id)
        #workspaces_task = _get_user_workspaces_mongo(db, user_id)
        
        #user_roles, user_workspace_ids = await asyncio.gather(roles_task, workspaces_task, return_exceptions=True)
        step3_time = (time.time() - step3_start) * 1000
        logger.info(f"Step 3 (parallel MongoDB roles & workspaces) took: {step3_time:.2f}ms")
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"MONGO SUPER OPTIMIZED comprehensive user info took: {total_time:.2f}ms")

        user_roles = []
        user_workspace_ids = []
        
        return workspace_id, is_qwen_route, user_roles, user_workspace_ids
        
    except Exception as e:
        logger.error(f"Error in MongoDB comprehensive user info: {e}")
        logger.warning("Falling back to Parse Server version")
        # Ensure User is imported for the fallback function
        try:
            return await _get_comprehensive_user_info_parallel(auth_header, client_type, api_key, httpx_client)
        except NameError as ne:
            if "User" in str(ne):
                # User import issue, import it explicitly
                from services.user_utils import User
                return await _get_comprehensive_user_info_parallel(auth_header, client_type, api_key, httpx_client)
            raise

async def _get_comprehensive_user_info_async(
    user_id: str,
    api_key: Optional[str] = None
) -> Tuple[Optional[str], Optional[bool], List[str], List[str]]:
    """Get comprehensive user info including workspace, roles, and isQwenRoute in a single Parse call"""
    try:
        start_time = time.time()
        logger.info(f"Starting comprehensive user info for user_id: {user_id}")
        
        # Now get comprehensive user info with includes in a single call
        result = await User.get_comprehensive_user_info_async(user_id, api_key)
        
        user_info_time = (time.time() - start_time) * 1000
        logger.info(f"Comprehensive user info took: {user_info_time:.2f}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting comprehensive user info: {e}")
        return None, None, [], []

async def _resolve_user_for_search_parallel(
    auth_header: str,
    search_request: "SearchRequest",
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> Tuple["MemoryMetadata", str, Optional[bool]]:
    """Resolve user for search in parallel with authentication - extracts user_id from auth_header"""
    user_id = None  # Always define at the top
    try:
        start_time = time.time()
        logger.info(f"Starting parallel user resolution")
        
        # Extract user_id from auth_header with proper verification
        if 'Bearer ' in auth_header:
            token = auth_header.split('Bearer ')[1]
            try:
                user_info = await User.verify_access_token(token, 'papr_plugin')  # Use default client_type
                if user_info:
                    user_id = user_info['https://papr.scope.com/objectId']
                    logger.info(f"Extracted user_id from Bearer token: {user_id}")
                else:
                    logger.warning("Invalid access token")
            except Exception as e:
                logger.warning(f"Access token verification failed: {e}")
                
        elif 'Session ' in auth_header:
            sessionToken = auth_header.split('Session ')[1]
            try:
                parse_user = await User.verify_session_token(sessionToken)
                if parse_user:
                    user_id = parse_user.objectId
                    logger.info(f"Extracted user_id from Session token: {user_id}")
            except Exception as session_e:
                logger.warning(f"Session token verification failed: {session_e}")
                
        elif 'APIKey ' in auth_header:
            api_key = auth_header.split('APIKey ')[1]
            try:
                user_info = await User.verify_api_key(api_key)
                if user_info and isinstance(user_info, dict) and 'objectId' in user_info:
                    user_id = user_info['objectId']
                    logger.info(f"Extracted user_id from API key: {user_id}")
                else:
                    logger.warning(f"API key verification returned invalid result: {user_info}")
            except Exception as api_e:
                logger.warning(f"API key verification failed: {api_e}")
                
        elif auth_header.startswith('APIKey'):
            api_key = auth_header.replace('APIKey', '').strip()
            try:
                user_info = await User.verify_api_key(api_key)
                if user_info and isinstance(user_info, dict) and 'objectId' in user_info:
                    user_id = user_info['objectId']
                    logger.info(f"Extracted user_id from API key (no space): {user_id}")
                else:
                    logger.warning(f"API key verification returned invalid result (no space): {user_info}")
            except Exception as api_e:
                logger.warning(f"API key verification failed (no space): {api_e}")
        
        if not user_id:
            logger.warning("Could not extract user_id for parallel user resolution")
            # Return a default result instead of raising an exception
            from models.memory_models import MemoryMetadata
            metadata = search_request.metadata or MemoryMetadata()
            return metadata, "unknown_user", None
        
        # Build metadata from search request
        from models.memory_models import MemoryMetadata
        
        metadata = search_request.metadata or MemoryMetadata()
        if getattr(search_request, "user_id", None) is not None:
            metadata.user_id = search_request.user_id
        if getattr(search_request, "external_user_id", None) is not None:
            metadata.external_user_id = search_request.external_user_id
        if getattr(metadata, "user_id", None) is None and getattr(metadata, "external_user_id", None) is None:
            metadata.user_id = user_id
        
        # Remove any None fields from metadata
        metadata_dict = metadata.model_dump(exclude_none=True)
        metadata = MemoryMetadata(**metadata_dict)
        
        # Use httpx client for the resolution
        async with httpx.AsyncClient() as httpx_client:
            result = await User.resolve_end_user_id(
                developer_id=user_id,
                metadata=metadata,
                authenticated_user_id=user_id,
                httpx_client=httpx_client,
                api_key=api_key
            )
            
            user_resolution_time = (time.time() - start_time) * 1000
            logger.info(f"Parallel user resolution took: {user_resolution_time:.2f}ms")
            
            return result
        
    except Exception as e:
        logger.error(f"Error in parallel user resolution: {e}")
        # Return a default result instead of raising an exception
        from models.memory_models import MemoryMetadata
        metadata = search_request.metadata or MemoryMetadata()
        return metadata, "unknown_user", None

async def _resolve_user_for_search(
    user_id: str,
    search_request: "SearchRequest",
    api_key: Optional[str] = None
) -> Tuple["MemoryMetadata", str, Optional[bool]]:
    """Resolve user for search in parallel with authentication"""
    try:
        start_time = time.time()
        logger.info(f"Starting user resolution for user_id: {user_id}")
        
        from models.memory_models import MemoryMetadata
        
        # Build metadata from search request
        metadata = search_request.metadata or MemoryMetadata()
        if getattr(search_request, "user_id", None) is not None:
            metadata.user_id = search_request.user_id
        if getattr(search_request, "external_user_id", None) is not None:
            metadata.external_user_id = search_request.external_user_id
        if getattr(metadata, "user_id", None) is None and getattr(metadata, "external_user_id", None) is None:
            metadata.user_id = user_id
        
        # Remove any None fields from metadata
        metadata_dict = metadata.model_dump(exclude_none=True)
        metadata = MemoryMetadata(**metadata_dict)
        
        # Use httpx client for the resolution
        async with httpx.AsyncClient() as httpx_client:
            result = await User.resolve_end_user_id(
                developer_id=user_id,
                metadata=metadata,
                authenticated_user_id=user_id,
                httpx_client=httpx_client,
                api_key=api_key
            )
            
            user_resolution_time = (time.time() - start_time) * 1000
            logger.info(f"User resolution took: {user_resolution_time:.2f}ms")
            
            return result
        
    except Exception as e:
        logger.error(f"Error resolving user for search: {e}")
        raise

async def verify_api_key(api_key: str) -> Tuple[str, Dict[str, Any]]:
    """Verify API key and return user info with caching"""
    try:
        # Check cache first
        cache_key = f"api_key:{api_key}"
        cached_result = api_key_cache.get(cache_key)
        if cached_result:
            logger.info(f"API key cache HIT for {api_key[:5]}...")
            return cached_result
        
        logger.info(f"API key cache MISS for {api_key[:5]}...")
        
        # Query Parse for user with matching API key
        params = {
            "where": json.dumps({"userAPIkey": api_key})
        }
        url = f"{PARSE_SERVER_URL}/parse/classes/_User"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=PARSE_HEADERS, params=params)
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    user = results[0]
                    result = (user["objectId"], user)
                    # Cache the result
                    api_key_cache.set(cache_key, result)
                    logger.info(f"Cached API key result for {api_key[:5]}...")
                    return result
                    
            raise ValueError("Invalid API key")
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error occurred while verifying API key: {str(e)}")
        raise ValueError("Invalid API key")
    except Exception as e:
        logger.error(f"Error verifying API key: {str(e)}")
        raise ValueError("Invalid API key")

async def _resolve_user_for_memory_parallel(
    auth_header: str,
    memory_request: "AddMemoryRequest",
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> Tuple["MemoryMetadata", str, Optional[bool]]:
    """Resolve user for memory operations in parallel with authentication - extracts user_id from auth_header"""
    user_id = None  # Always define at the top
    try:
        start_time = time.time()
        logger.info(f"Starting parallel user resolution for memory")
        
        # Extract user_id from auth_header with proper verification
        if 'Bearer ' in auth_header:
            token = auth_header.split('Bearer ')[1]
            try:
                user_info = await User.verify_access_token(token, 'papr_plugin')  # Use default client_type
                if user_info:
                    user_id = user_info['https://papr.scope.com/objectId']
                    logger.info(f"Extracted user_id from Bearer token: {user_id}")
                else:
                    logger.warning("Invalid access token")
            except Exception as e:
                logger.warning(f"Access token verification failed: {e}")
                
        elif 'Session ' in auth_header:
            sessionToken = auth_header.split('Session ')[1]
            try:
                parse_user = await User.verify_session_token(sessionToken)
                if parse_user:
                    user_id = parse_user.objectId
                    logger.info(f"Extracted user_id from Session token: {user_id}")
            except Exception as session_e:
                logger.warning(f"Session token verification failed: {session_e}")
                
        elif 'APIKey ' in auth_header:
            api_key = auth_header.split('APIKey ')[1]
            try:
                user_info = await User.verify_api_key(api_key)
                if user_info and isinstance(user_info, dict) and 'objectId' in user_info:
                    user_id = user_info['objectId']
                    logger.info(f"Extracted user_id from API key: {user_id}")
                else:
                    logger.warning(f"API key verification returned invalid result: {user_info}")
            except Exception as api_e:
                logger.warning(f"API key verification failed: {api_e}")
                
        elif auth_header.startswith('APIKey'):
            api_key = auth_header.replace('APIKey', '').strip()
            try:
                user_info = await User.verify_api_key(api_key)
                if user_info and isinstance(user_info, dict) and 'objectId' in user_info:
                    user_id = user_info['objectId']
                    logger.info(f"Extracted user_id from API key (no space): {user_id}")
                else:
                    logger.warning(f"API key verification returned invalid result (no space): {user_info}")
            except Exception as api_e:
                logger.warning(f"API key verification failed (no space): {api_e}")
        
        if not user_id:
            logger.warning("Could not extract user_id for parallel user resolution")
            # Return a default result instead of raising an exception
            from models.memory_models import MemoryMetadata
            metadata = memory_request.metadata or MemoryMetadata()
            return metadata, "unknown_user", None
        
        # Use the memory request's metadata
        metadata = memory_request.metadata or MemoryMetadata()
        
        # Use httpx client for the resolution
        async with httpx.AsyncClient() as httpx_client:
            result = await User.resolve_end_user_id(
                developer_id=user_id,
                metadata=metadata,
                authenticated_user_id=user_id,
                httpx_client=httpx_client,
                api_key=api_key
            )
            
            user_resolution_time = (time.time() - start_time) * 1000
            logger.info(f"Parallel user resolution for memory took: {user_resolution_time:.2f}ms")
            
            return result
        
    except Exception as e:
        logger.error(f"Error in parallel user resolution for memory: {e}")
        # Return a default result instead of raising an exception
        from models.memory_models import MemoryMetadata
        metadata = memory_request.metadata or MemoryMetadata()
        return metadata, "unknown_user", None

async def _resolve_user_for_batch_parallel(
    auth_header: str,
    batch_request: "BatchMemoryRequest",
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> Tuple["MemoryMetadata", str, Optional[bool]]:
    """Resolve user for batch operations in parallel with authentication - extracts user_id from auth_header"""
    user_id = None  # Always define at the top
    try:
        start_time = time.time()
        logger.info(f"Starting parallel user resolution for batch")
        
        # Extract user_id from auth_header with proper verification
        if 'Bearer ' in auth_header:
            token = auth_header.split('Bearer ')[1]
            try:
                user_info = await User.verify_access_token(token, 'papr_plugin')  # Use default client_type
                if user_info:
                    user_id = user_info['https://papr.scope.com/objectId']
                    logger.info(f"Extracted user_id from Bearer token: {user_id}")
                else:
                    logger.warning("Invalid access token")
            except Exception as e:
                logger.warning(f"Access token verification failed: {e}")
                
        elif 'Session ' in auth_header:
            sessionToken = auth_header.split('Session ')[1]
            try:
                parse_user = await User.verify_session_token(sessionToken)
                if parse_user:
                    user_id = parse_user.objectId
                    logger.info(f"Extracted user_id from Session token: {user_id}")
            except Exception as session_e:
                logger.warning(f"Session token verification failed: {session_e}")
                
        elif 'APIKey ' in auth_header:
            api_key = auth_header.split('APIKey ')[1]
            try:
                user_info = await User.verify_api_key(api_key)
                if user_info and isinstance(user_info, dict) and 'objectId' in user_info:
                    user_id = user_info['objectId']
                    logger.info(f"Extracted user_id from API key: {user_id}")
                else:
                    logger.warning(f"API key verification returned invalid result: {user_info}")
            except Exception as api_e:
                logger.warning(f"API key verification failed: {api_e}")
                
        elif auth_header.startswith('APIKey'):
            api_key = auth_header.replace('APIKey', '').strip()
            try:
                user_info = await User.verify_api_key(api_key)
                if user_info and isinstance(user_info, dict) and 'objectId' in user_info:
                    user_id = user_info['objectId']
                    logger.info(f"Extracted user_id from API key (no space): {user_id}")
                else:
                    logger.warning(f"API key verification returned invalid result (no space): {user_info}")
            except Exception as api_e:
                logger.warning(f"API key verification failed (no space): {api_e}")
        
        if not user_id:
            logger.warning("Could not extract user_id for parallel user resolution")
            # Return a default result instead of raising an exception
            from models.memory_models import MemoryMetadata
            metadata = MemoryMetadata()
            return metadata, "unknown_user", None
        
        # Use patch_and_resolve_user_ids_and_acls for batch requests
        # This method handles all the user resolution and ACL patching for batch requests
        try:
            # Use the provided httpx_client or create a new one
            if not httpx_client:
                async with httpx.AsyncClient() as httpx_client:
                    updated_batch_request, resolved_end_user_id, is_qwen_route = await User.patch_and_resolve_user_ids_and_acls(
                        developer_id=user_id,
                        batch_request=batch_request,
                        httpx_client=httpx_client,
                        x_api_key=api_key
                    )
            else:
                updated_batch_request, resolved_end_user_id, is_qwen_route = await User.patch_and_resolve_user_ids_and_acls(
                    developer_id=user_id,
                    batch_request=batch_request,
                    httpx_client=httpx_client,
                    x_api_key=api_key
                )
            
            user_resolution_time = (time.time() - start_time) * 1000
            logger.info(f"Parallel user resolution for batch took: {user_resolution_time:.2f}ms")
            logger.info(f"Resolved end_user_id: {resolved_end_user_id}")
            
            # Return the updated batch request, resolved end_user_id, and isQwenRoute
            return updated_batch_request, resolved_end_user_id, is_qwen_route
            
        except Exception as e:
            logger.error(f"Error in patch_and_resolve_user_ids_and_acls: {e}")
            # Fallback to simple resolution
            from models.memory_models import MemoryMetadata
            metadata = MemoryMetadata()
            if getattr(batch_request, "user_id", None) is not None:
                metadata.user_id = batch_request.user_id
            if getattr(batch_request, "external_user_id", None) is not None:
                metadata.external_user_id = batch_request.external_user_id
            if getattr(metadata, "user_id", None) is None and getattr(metadata, "external_user_id", None) is None:
                metadata.user_id = user_id
            
            return metadata, user_id, None
        
    except Exception as e:
        logger.error(f"Error in parallel user resolution for batch: {e}")
        # Return a default result instead of raising an exception
        from models.memory_models import MemoryMetadata
        metadata = MemoryMetadata()
        return metadata, "unknown_user", None

async def _resolve_user_for_update_parallel(
    auth_header: str,
    update_request: "UpdateMemoryRequest",
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> Tuple["MemoryMetadata", str, Optional[bool]]:
    """Resolve user for update operations in parallel with authentication - extracts user_id from auth_header"""
    user_id = None  # Always define at the top
    try:
        start_time = time.time()
        logger.info(f"Starting parallel user resolution for update")
        
        # Extract user_id from auth_header with proper verification
        if 'Bearer ' in auth_header:
            token = auth_header.split('Bearer ')[1]
            try:
                user_info = await User.verify_access_token(token, 'papr_plugin')  # Use default client_type
                if user_info:
                    user_id = user_info['https://papr.scope.com/objectId']
                    logger.info(f"Extracted user_id from Bearer token: {user_id}")
                else:
                    logger.warning("Invalid access token")
            except Exception as e:
                logger.warning(f"Access token verification failed: {e}")
                
        elif 'Session ' in auth_header:
            sessionToken = auth_header.split('Session ')[1]
            try:
                parse_user = await User.verify_session_token(sessionToken)
                if parse_user:
                    user_id = parse_user.objectId
                    logger.info(f"Extracted user_id from Session token: {user_id}")
            except Exception as session_e:
                logger.warning(f"Session token verification failed: {session_e}")
                
        elif 'APIKey ' in auth_header:
            api_key = auth_header.split('APIKey ')[1]
            try:
                user_info = await User.verify_api_key(api_key)
                if user_info and isinstance(user_info, dict) and 'objectId' in user_info:
                    user_id = user_info['objectId']
                    logger.info(f"Extracted user_id from API key: {user_id}")
                else:
                    logger.warning(f"API key verification returned invalid result: {user_info}")
            except Exception as api_e:
                logger.warning(f"API key verification failed: {api_e}")
        
        if not user_id:
            logger.warning("Could not extract user_id for parallel user resolution")
            return None, "unknown_user", None
        
        # Use httpx client for the resolution
        async with httpx.AsyncClient() as httpx_client:
            result = await _resolve_user_for_update(user_id, update_request, api_key)
            
            user_resolution_time = (time.time() - start_time) * 1000
            logger.info(f"Update user resolution took: {user_resolution_time:.2f}ms")
            
            return result
        
    except Exception as e:
        logger.error(f"Error resolving user for update: {e}")
        return None, "unknown_user", None

async def _resolve_user_for_feedback_parallel(
    auth_header: str,
    feedback_request: "FeedbackRequest",
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> Tuple["MemoryMetadata", str, Optional[bool]]:
    """Resolve user for feedback operations in parallel with authentication - extracts user_id from auth_header"""
    user_id = None  # Always define at the top
    try:
        start_time = time.time()
        logger.info(f"Starting parallel user resolution for feedback")
        
        # Extract user_id from auth_header with proper verification
        if 'Bearer ' in auth_header:
            token = auth_header.split('Bearer ')[1]
            try:
                user_info = await User.verify_access_token(token, 'papr_plugin')  # Use default client_type
                if user_info:
                    user_id = user_info['https://papr.scope.com/objectId']
                    logger.info(f"Extracted user_id from Bearer token: {user_id}")
                else:
                    logger.warning("Invalid access token")
            except Exception as e:
                logger.warning(f"Access token verification failed: {e}")
                
        elif 'Session ' in auth_header:
            sessionToken = auth_header.split('Session ')[1]
            try:
                parse_user = await User.verify_session_token(sessionToken)
                if parse_user:
                    user_id = parse_user.objectId
                    logger.info(f"Extracted user_id from Session token: {user_id}")
            except Exception as session_e:
                logger.warning(f"Session token verification failed: {session_e}")
                
        elif 'APIKey ' in auth_header:
            api_key = auth_header.split('APIKey ')[1]
            try:
                user_info = await User.verify_api_key(api_key)
                if user_info and isinstance(user_info, dict) and 'objectId' in user_info:
                    user_id = user_info['objectId']
                    logger.info(f"Extracted user_id from API key: {user_id}")
                else:
                    logger.warning(f"API key verification returned invalid result: {user_info}")
            except Exception as api_e:
                logger.warning(f"API key verification failed: {api_e}")
        
        if not user_id:
            logger.warning("Could not extract user_id for parallel user resolution")
            return None, "unknown_user", None
        
        # Use httpx client for the resolution
        async with httpx.AsyncClient() as httpx_client:
            result = await _resolve_user_for_feedback(user_id, feedback_request, api_key)
            
            user_resolution_time = (time.time() - start_time) * 1000
            logger.info(f"Feedback user resolution took: {user_resolution_time:.2f}ms")
            
            return result
        
    except Exception as e:
        logger.error(f"Error resolving user for feedback: {e}")
        return None, "unknown_user", None

def _generate_auth_cache_key(
    auth_header: str,
    client_type: str,
    api_key: Optional[str] = None,
    search_request: Optional["SearchRequest"] = None,
    memory_request: Optional["AddMemoryRequest"] = None,
    batch_request: Optional["BatchMemoryRequest"] = None,
    update_request: Optional["UpdateMemoryRequest"] = None,
    feedback_request: Optional["FeedbackRequest"] = None,
    sync_tiers_request: Optional["SyncTiersRequest"] = None
) -> str:
    """
    Generate a cache key based on authentication inputs and user resolution parameters.
    Cache key uniquely identifies the input parameters that determine resolution behavior.
    
    Args:
        auth_header: The authorization header string
        client_type: The client type ('browser_extension' or 'papr_plugin')
        api_key: Optional API key
        search_request: Optional SearchRequest for user resolution
        memory_request: Optional AddMemoryRequest for user resolution
        batch_request: Optional BatchMemoryRequest for user resolution
        update_request: Optional UpdateMemoryRequest for user resolution
        feedback_request: Optional FeedbackRequest for user resolution
        sync_tiers_request: Optional SyncTiersRequest for user resolution
    
    Returns:
        A cache key string that uniquely identifies the authentication context
    """
    # Normalize auth inputs for consistent cache keys
    normalized_auth_header = auth_header.strip()
    normalized_client_type = client_type.strip()
    
    # Handle cases where api_key might be redundant with auth_header
    # If auth_header already contains the api_key, don't duplicate it in cache key
    cache_key = f"auth_optimized:{normalized_auth_header[:50]}:{normalized_client_type}"
    logger.info(f"Optimized auth - base cache_key: {cache_key}")
    
    # Only add api_key to cache key if it's different from what's in auth_header
    if api_key and api_key not in normalized_auth_header:
        cache_key += f":{api_key[:20]}"
    
    # Extract original request parameters to differentiate resolution paths
    # These raw parameters determine what resolution will happen and should be in cache key
    user_id_from_request = None
    external_user_id_from_request = None
    
    if search_request:
        user_id_from_request = getattr(search_request, "user_id", None)
        external_user_id_from_request = getattr(search_request, "external_user_id", None)
        # Also check metadata for search requests
        if search_request.metadata:
            if not user_id_from_request:
                user_id_from_request = getattr(search_request.metadata, "user_id", None)
            if not external_user_id_from_request:
                external_user_id_from_request = getattr(search_request.metadata, "external_user_id", None)
    elif sync_tiers_request:
        user_id_from_request = getattr(sync_tiers_request, "user_id", None)
        external_user_id_from_request = getattr(sync_tiers_request, "external_user_id", None)
    elif memory_request:
        if memory_request.metadata:
            user_id_from_request = getattr(memory_request.metadata, "user_id", None)
            external_user_id_from_request = getattr(memory_request.metadata, "external_user_id", None)
    elif batch_request:
        user_id_from_request = getattr(batch_request, "user_id", None)
        external_user_id_from_request = getattr(batch_request, "external_user_id", None)
    elif update_request:
        if update_request.metadata:
            user_id_from_request = getattr(update_request.metadata, "user_id", None)
            external_user_id_from_request = getattr(update_request.metadata, "external_user_id", None)
    elif feedback_request:
        user_id_from_request = getattr(feedback_request, "user_id", None)
        external_user_id_from_request = getattr(feedback_request, "external_user_id", None)
    
    # Add raw request user_id to cache key if present (what was requested)
    if user_id_from_request:
        cache_key += f":req_user:{user_id_from_request[:20]}"
        logger.info(f"Added requested user_id to cache key: {user_id_from_request[:20]}")
    
    # Add external_user_id to cache key if present (original external identifier)
    if external_user_id_from_request:
        cache_key += f":ext_user:{external_user_id_from_request[:20]}"
        logger.info(f"Added external_user_id to cache key: {external_user_id_from_request[:20]}")
    
    # Indicate whether this is a "no-resolution" case (Case 1: developer == end user)
    if not user_id_from_request and not external_user_id_from_request:
        cache_key += f":case1"
        logger.info(f"Added case1 flag to cache key (developer == end user)")
    
    # Add ACL fields to cache key for update requests to ensure proper sharing behavior
    if update_request and update_request.metadata:
        acl_fields = []
        if hasattr(update_request.metadata, 'user_read_access') and update_request.metadata.user_read_access:
            acl_fields.append(f"read:{','.join(update_request.metadata.user_read_access[:3])}")
        if hasattr(update_request.metadata, 'user_write_access') and update_request.metadata.user_write_access:
            acl_fields.append(f"write:{','.join(update_request.metadata.user_write_access[:3])}")
        if acl_fields:
            cache_key += f":acl:{','.join(acl_fields)}"
            logger.info(f"Added ACL fields to cache key: {acl_fields}")
    
    # Log the final complete cache key
    logger.info(f"Final complete cache key: {cache_key}")
    return cache_key

async def _resolve_user_for_update(
    user_id: str,
    update_request: "UpdateMemoryRequest",
    api_key: Optional[str] = None
) -> Tuple["MemoryMetadata", str, Optional[bool]]:
    """Resolve user for update operations"""
    try:
        start_time = time.time()
        logger.info(f"Starting user resolution for user_id: {user_id}")
        
        from models.memory_models import MemoryMetadata
        
        # Build metadata from update request
        metadata = update_request.metadata or MemoryMetadata()
        if getattr(update_request, "user_id", None) is not None:
            metadata.user_id = update_request.user_id
        if getattr(update_request, "external_user_id", None) is not None:
            metadata.external_user_id = update_request.external_user_id
        if getattr(metadata, "user_id", None) is None and getattr(metadata, "external_user_id", None) is None:
            metadata.user_id = user_id
        
        # Remove any None fields from metadata
        metadata_dict = metadata.model_dump(exclude_none=True)
        metadata = MemoryMetadata(**metadata_dict)
        
        # Use httpx client for the resolution
        async with httpx.AsyncClient() as httpx_client:
            result = await User.resolve_end_user_id(
                developer_id=user_id,
                metadata=metadata,
                authenticated_user_id=user_id,
                httpx_client=httpx_client,
                api_key=api_key
            )
            
            user_resolution_time = (time.time() - start_time) * 1000
            logger.info(f"User resolution took: {user_resolution_time:.2f}ms")
            
            return result
        
    except Exception as e:
        logger.error(f"Error resolving user for update: {e}")
        raise

async def _resolve_user_for_feedback(
    user_id: str,
    feedback_request: "FeedbackRequest",
    api_key: Optional[str] = None
) -> Tuple["MemoryMetadata", str, Optional[bool]]:
    """Resolve user for feedback operations"""
    try:
        start_time = time.time()
        logger.info(f"Starting user resolution for feedback user_id: {user_id}")
        
        from models.memory_models import MemoryMetadata
        
        # Build metadata from feedback request
        metadata = MemoryMetadata()
        if getattr(feedback_request, "external_user_id", None) is not None:
            metadata.external_user_id = feedback_request.external_user_id
        if getattr(metadata, "user_id", None) is None and getattr(metadata, "external_user_id", None) is None:
            metadata.user_id = user_id
        
        # Remove any None fields from metadata
        metadata_dict = metadata.model_dump(exclude_none=True)
        metadata = MemoryMetadata(**metadata_dict)
        
        # Use httpx client for the resolution
        async with httpx.AsyncClient() as httpx_client:
            result = await User.resolve_end_user_id(
                developer_id=user_id,
                metadata=metadata,
                authenticated_user_id=user_id,
                httpx_client=httpx_client,
                api_key=api_key
            )
            
            user_resolution_time = (time.time() - start_time) * 1000
            logger.info(f"Feedback user resolution took: {user_resolution_time:.2f}ms")
            
            return result
        
    except Exception as e:
        logger.error(f"Error resolving user for feedback: {e}")
        raise

async def get_enhanced_api_key_info(api_key: str, memory_graph: "MemoryGraph", httpx_client: Optional[httpx.AsyncClient] = None) -> Optional[Dict[str, Any]]:
    """Get enhanced API key information including organization and namespace data"""
    if not api_key:
        return None

    # Check cache first - use a different cache key for enhanced info
    cache_key = f"enhanced_api_key_{api_key}"
    cached_result = api_key_cache.get(cache_key)
    if cached_result:
        logger.info(f"Enhanced API key cache HIT for {api_key[:10]}...")
        return cached_result

    try:
        # Try to find organization-based API key first
        if memory_graph.mongo_client:
            # Check if MongoDB connection is healthy, reconnect if needed
            try:
                # Test the connection with a simple operation
                memory_graph.db.admin.command('ping')
            except Exception as conn_err:
                conn_err_str = str(conn_err).lower()
                if "after close" in conn_err_str or "closed" in conn_err_str or "connection" in conn_err_str:
                    logger.warning(f"MongoDB connection is closed in get_enhanced_api_key_info, attempting to reconnect...")
                    # Try to get a fresh database reference from the singleton
                    from services.mongo_client import get_mongo_db
                    fresh_db = get_mongo_db()
                    if fresh_db is not None:
                        logger.info("Got fresh MongoDB connection from singleton in get_enhanced_api_key_info")
                        memory_graph.db = fresh_db
                        memory_graph.mongo_client = fresh_db.client
                    else:
                        # Singleton couldn't reconnect, try memory_graph reconnect
                        try:
                            reconnected = await asyncio.to_thread(memory_graph.reconnect_mongodb)
                            if reconnected:
                                logger.info("MongoDB reconnection successful via memory_graph in get_enhanced_api_key_info")
                            else:
                                logger.error("MongoDB reconnection failed in get_enhanced_api_key_info")
                                memory_graph.mongo_client = None  # Mark as unavailable
                        except Exception as reconnect_err:
                            logger.error(f"MongoDB reconnection error in get_enhanced_api_key_info: {reconnect_err}")
                            memory_graph.mongo_client = None  # Mark as unavailable
            
            # Only proceed if MongoDB is still available after reconnection attempt
            api_key_doc = None
            if memory_graph.mongo_client:
                try:
                    # Look for new organization-based API keys
                    api_key_doc = memory_graph.db["APIKey"].find_one({"key": api_key})
                except Exception as db_err:
                    # Connection might have closed between check and use
                    db_err_str = str(db_err).lower()
                    if "after close" in db_err_str or "closed" in db_err_str:
                        logger.warning(f"MongoDB connection closed during get_enhanced_api_key_info query: {db_err}")
                        # Try one more time to get fresh connection
                        from services.mongo_client import get_mongo_db
                        fresh_db = get_mongo_db()
                        if fresh_db is not None:
                            memory_graph.db = fresh_db
                            memory_graph.mongo_client = fresh_db.client
                            try:
                                api_key_doc = memory_graph.db["APIKey"].find_one({"key": api_key})
                            except Exception:
                                logger.error("Failed to query MongoDB even after reconnection")
                                memory_graph.mongo_client = None
                    else:
                        raise  # Re-raise non-connection errors
            
            if api_key_doc:
                logger.info(f"Found organization-based API key for {api_key[:10]}...")
                if "_id" in api_key_doc and not isinstance(api_key_doc["_id"], str):
                    api_key_doc["_id"] = str(api_key_doc["_id"])

                # Extract organization and namespace info from pointers
                organization_id = None
                namespace_id = None

                if "organization" in api_key_doc and isinstance(api_key_doc["organization"], dict):
                    organization_id = api_key_doc["organization"].get("objectId")
                elif "_p_organization" in api_key_doc:
                    # Parse pointer format: "Organization$abc123"
                    org_pointer = api_key_doc["_p_organization"]
                    if org_pointer.startswith("Organization$"):
                        organization_id = org_pointer.replace("Organization$", "")

                if "namespace" in api_key_doc and isinstance(api_key_doc["namespace"], dict):
                    namespace_id = api_key_doc["namespace"].get("objectId")
                elif "_p_namespace" in api_key_doc:
                    # Parse pointer format: "Namespace$abc123"
                    ns_pointer = api_key_doc["_p_namespace"]
                    if ns_pointer.startswith("Namespace$"):
                        namespace_id = ns_pointer.replace("Namespace$", "")

                if organization_id and namespace_id:
                    # Get organization to find owner
                    org_doc = memory_graph.db["Organization"].find_one({"_id": organization_id})
                    if org_doc:
                        # Extract owner user ID
                        owner_user_id = None
                        if "owner" in org_doc and isinstance(org_doc["owner"], dict):
                            owner_user_id = org_doc["owner"].get("objectId")
                        elif "_p_owner" in org_doc:
                            # Parse pointer format: "_User$abc123"
                            owner_pointer = org_doc["_p_owner"]
                            if owner_pointer.startswith("_User$"):
                                owner_user_id = owner_pointer.replace("_User$", "")

                        # Extract workspace_id from organization
                        workspace_id = None
                        if "workspace" in org_doc and isinstance(org_doc["workspace"], dict):
                            workspace_id = org_doc["workspace"].get("objectId")
                        elif "_p_workspace" in org_doc:
                            # Parse pointer format: "WorkSpace$abc123"
                            workspace_pointer = org_doc["_p_workspace"]
                            if workspace_pointer.startswith("WorkSpace$"):
                                workspace_id = workspace_pointer.replace("WorkSpace$", "")
                        
                        logger.info(f"Extracted workspace_id from organization: {workspace_id}")

                        if owner_user_id:
                            enhanced_info = {
                                "user_id": owner_user_id,
                                "organization_id": organization_id,
                                "namespace_id": namespace_id,
                                "workspace_id": workspace_id,
                                "is_organization_api_key": True,
                                "api_key_doc": api_key_doc,
                                "organization_doc": org_doc
                            }

                            # Cache the result
                            api_key_cache.set(cache_key, enhanced_info)
                            logger.info(f"Enhanced API key cache SET (organization) for {api_key[:10]}... -> org: {organization_id}, ns: {namespace_id}, user: {owner_user_id}, workspace: {workspace_id}")
                            return enhanced_info

        # Fallback: Try legacy API key lookup (API key stored in _User.userAPIkey field)
        # For legacy keys, we need to get the developer's workspace and extract organization/namespace
        from services.user_utils import User
        user_data = await User.verify_api_key(api_key, httpx_client)

        if user_data:
            legacy_user_id = user_data.get('objectId')

            # Extract organization_id, namespace_id, and workspace_id from the developer's workspace
            # Use the developer's isSelectedWorkspaceFollower, not Organization.workspace (which may be stale)
            organization_id = None
            namespace_id = None
            workspace_id = None

            workspace_follower = user_data.get('isSelectedWorkspaceFollower')
            if workspace_follower and isinstance(workspace_follower, dict):
                workspace = workspace_follower.get('workspace')
                if workspace and isinstance(workspace, dict):
                    # Get workspace_id from the user's selected workspace (most accurate)
                    workspace_id = workspace.get('objectId')

                    # Extract organization
                    organization = workspace.get('organization')
                    if organization and isinstance(organization, dict):
                        organization_id = organization.get('objectId')

                    # Extract namespace
                    namespace = workspace.get('namespace')
                    if namespace and isinstance(namespace, dict):
                        namespace_id = namespace.get('objectId')

            logger.info(f"Legacy API key: extracted org_id={organization_id}, ns_id={namespace_id}, workspace_id={workspace_id} from developer's isSelectedWorkspaceFollower")

            legacy_info = {
                "user_id": legacy_user_id,
                "organization_id": organization_id,
                "namespace_id": namespace_id,
                "workspace_id": workspace_id,
                "is_organization_api_key": False,
                "api_key_doc": None,
                "organization_doc": None
            }

            # Attempt to fetch the API key document so we have objectId available for downstream usage
            api_key_doc = None
            try:
                if memory_graph and memory_graph.mongo_client:
                    api_key_doc = memory_graph.db["APIKey"].find_one({"key": api_key})
                    if api_key_doc and "_id" in api_key_doc and not isinstance(api_key_doc["_id"], str):
                        api_key_doc["_id"] = str(api_key_doc["_id"])
            except Exception as e:
                logger.warning(f"Mongo lookup for API key doc failed: {e}")

            if not api_key_doc:
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        params = {"where": json.dumps({"key": api_key})}
                        response = await client.get(
                            f"{PARSE_SERVER_URL}/parse/classes/APIKey",
                            headers=PARSE_HEADERS,
                            params=params
                        )
                        if response.status_code == 200:
                            results = response.json().get("results", [])
                            if results:
                                api_key_doc = results[0]
                        else:
                            logger.warning(f"Failed to fetch API key doc from Parse: {response.text}")
                except Exception as e:
                    logger.warning(f"HTTP error fetching API key doc: {e}")

            if api_key_doc:
                legacy_info["api_key_doc"] = api_key_doc

            # Cache the result
            api_key_cache.set(cache_key, legacy_info)
            logger.info(f"Enhanced API key cache SET (legacy) for {api_key[:10]}... -> user: {legacy_user_id}, org: {organization_id}, ns: {namespace_id}")
            return legacy_info

        logger.warning(f"API key not found in organization or legacy systems: {api_key[:10]}...")
        return None

    except Exception as e:
        logger.error(f"Error getting enhanced API key info: {e}")
        return None

async def get_user_id_from_api_key_legacy(api_key: str, memory_graph: "MemoryGraph", httpx_client: Optional[httpx.AsyncClient] = None) -> Optional[str]:
    """Legacy API key resolution - for backward compatibility"""
    if not api_key:
        return None

    # Check cache first
    cached_user_id = api_key_to_user_id_cache.get(api_key)
    if cached_user_id:
        logger.info(f"Legacy API key cache HIT for {api_key[:10]}...")
        return cached_user_id

    # Cache miss - verify API key
    try:
        # Try MongoDB first for super fast lookup
        if memory_graph.mongo_client:
            user_info = await _verify_api_key_mongo(memory_graph.db, api_key, memory_graph)
            if user_info and isinstance(user_info, dict) and 'objectId' in user_info:
                user_id = user_info['objectId']
                # Cache the result using custom cache API
                api_key_to_user_id_cache.set(api_key, user_id)
                logger.info(f"Legacy API key cache SET (MongoDB) for {api_key[:10]}... -> {user_id}")
                return user_id

        # Fallback to Parse Server if MongoDB not available
        from services.user_utils import User
        user_info = await User.verify_api_key(api_key, httpx_client=httpx_client)
        if user_info and isinstance(user_info, dict) and 'objectId' in user_info:
            user_id = user_info['objectId']
            # Cache the result using custom cache API
            api_key_to_user_id_cache.set(api_key, user_id)
            logger.info(f"Legacy API key cache SET (Parse fallback) for {api_key[:10]}... -> {user_id}")
            return user_id
    except Exception as e:
        logger.error(f"Error verifying legacy API key: {e}")
        return None

    return None

async def get_user_id_from_api_key(api_key: str, memory_graph: "MemoryGraph", httpx_client: Optional[httpx.AsyncClient] = None) -> Optional[str]:
    """Enhanced API key resolution with organization support - maintains backward compatibility"""
    enhanced_info = await get_enhanced_api_key_info(api_key, memory_graph, httpx_client)
    return enhanced_info["user_id"] if enhanced_info else None

async def get_api_key_from_user_id(user_id: str, memory_graph: "MemoryGraph", httpx_client: Optional[httpx.AsyncClient] = None) -> Optional[str]:
    """Get API key from user ID - for Bearer token authentication that needs API key for Parse Server"""
    try:
        # Check cache first (reverse lookup)
        cached_api_key = api_key_to_user_id_cache.find_value(user_id)
        if cached_api_key:
            logger.info(f"Found API key in cache for user_id {user_id}")
            return cached_api_key
        
        # Try MongoDB first for super fast lookup
        if memory_graph.mongo_client:
            try:
                user_doc = memory_graph.db['_User'].find_one(
                    {"_id": user_id},
                    {"userAPIkey": 1}
                )
                if user_doc and user_doc.get("userAPIkey"):
                    api_key = user_doc["userAPIkey"]
                    # Cache the result for future lookups
                    api_key_to_user_id_cache.set(api_key, user_id)
                    logger.info(f"Got API key from MongoDB for user_id {user_id}")
                    return api_key
            except Exception as e:
                logger.warning(f"MongoDB lookup failed for user_id {user_id}: {e}")
        
        # Fallback to Parse Server if MongoDB not available
        from services.user_utils import User
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }
        url = f"{PARSE_SERVER_URL}/parse/classes/_User/{user_id}"
        params = {"keys": "userAPIkey"}
        
        if httpx_client:
            response = await httpx_client.get(url, headers=headers, params=params)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            user_data = response.json()
            api_key = user_data.get("userAPIkey")
            if api_key:
                # Cache the result for future lookups
                api_key_to_user_id_cache.set(api_key, user_id)
                logger.info(f"Got API key from Parse Server for user_id {user_id}")
                return api_key
    except Exception as e:
        logger.error(f"Failed to get API key from user_id {user_id}: {e}")
    
    return None

async def _resolve_user_for_search_parallel_v2(
    developer_user_id: str,
    search_request: "SearchRequest",
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> Tuple["MemoryMetadata", str, Optional[bool], List[str], List[str], Optional[str]]:
    """
    V2: Resolve user for search with developer_user_id already known.
    Returns: (metadata, resolved_user_id, is_qwen_route, user_roles, user_workspace_ids, workspace_id)
    """
    try:
        start_time = time.time()
        logger.info(f"Starting V2 user resolution for search with developer_user_id: {developer_user_id}")
        
        from models.memory_models import MemoryMetadata
        
        # Build metadata from search request
        metadata = search_request.metadata or MemoryMetadata()
        if getattr(search_request, "user_id", None) is not None:
            metadata.user_id = search_request.user_id
        if getattr(search_request, "external_user_id", None) is not None:
            metadata.external_user_id = search_request.external_user_id
        
        # Case 1: Developer is the end user (no user_id/external_user_id in original request)
        # Check the original search request, not the metadata which may have been modified
        original_user_id = getattr(search_request, "user_id", None)
        original_external_user_id = getattr(search_request, "external_user_id", None)
        original_metadata_user_id = getattr(search_request.metadata, "user_id", None) if search_request.metadata else None
        original_metadata_external_user_id = getattr(search_request.metadata, "external_user_id", None) if search_request.metadata else None
        
        if (not original_user_id and not original_external_user_id and 
            not original_metadata_user_id and not original_metadata_external_user_id):
            metadata.user_id = developer_user_id
            resolved_user_id = developer_user_id
            
            # No need to fetch roles/workspace here - _get_comprehensive_user_info_parallel already does this
            # Return empty lists since the main auth function will use the comprehensive info
            logger.info(f"Case 1: Developer is end user - resolved_user_id: {resolved_user_id}")
            
            user_resolution_time = (time.time() - start_time) * 1000
            logger.info(f"V2 user resolution took: {user_resolution_time:.2f}ms")
            
            return metadata, resolved_user_id, None, [], [], None
            
        else:
            # Case 2 & 3: Need to resolve to different end user
            if metadata.user_id is None and metadata.external_user_id is None:
                metadata.user_id = developer_user_id
            
            # Use the existing resolution logic with multi-tenant context
            result = await User.resolve_end_user_id(
                developer_id=developer_user_id,
                metadata=metadata,
                authenticated_user_id=developer_user_id,
                httpx_client=httpx_client,
                api_key=api_key,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            updated_metadata, resolved_user_id, is_qwen_route = result
            metadata = updated_metadata
            
            # Get resolved user's roles and workspace info only if different from developer
            if resolved_user_id != developer_user_id:
                resolved_user = User.get(resolved_user_id)
                roles_task = resolved_user.get_roles_async() if resolved_user else asyncio.sleep(0, [])
                workspaces_task = User.get_workspaces_for_user_async(resolved_user_id)
                
                roles_result, workspaces_result = await asyncio.gather(
                    roles_task, workspaces_task, return_exceptions=True
                )
                
                user_roles = roles_result if not isinstance(roles_result, Exception) else []
                user_workspace_ids = workspaces_result if not isinstance(workspaces_result, Exception) else []
                
                # Get resolved user's workspace
                headers = {
                    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                    "Content-Type": "application/json",
                    "X-Parse-Master-Key": PARSE_MASTER_KEY
                }
                user_url = f"{PARSE_SERVER_URL}/parse/classes/_User/{resolved_user_id}"
                user_params = {
                    "include": "isSelectedWorkspaceFollower,isSelectedWorkspaceFollower.workspace",
                    "keys": "objectId,isSelectedWorkspaceFollower"
                }
                
                if httpx_client:
                    response = await httpx_client.get(user_url, headers=headers, params=user_params)
                else:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(user_url, headers=headers, params=user_params)
                
                user_obj = response.json()
                workspace_id = None
                selected_follower = user_obj.get('isSelectedWorkspaceFollower')
                if selected_follower and isinstance(selected_follower, dict):
                    workspace = selected_follower.get('workspace')
                    if workspace and isinstance(workspace, dict):
                        workspace_id = workspace.get('objectId')
                        
                logger.info(f"Case 2/3: Resolved to different user - resolved_user_id: {resolved_user_id}, workspace_id: {workspace_id}")
            else:
                # Same as developer, return empty lists since main function has comprehensive info
                # The main auth function already has developer's workspace from _get_comprehensive_user_info_parallel
                user_roles = []
                user_workspace_ids = []
                workspace_id = None  # Main auth function will use developer_workspace_id
                logger.info(f"Case 1: Developer is end user - returning None, main auth will use developer_workspace_id")
        
        user_resolution_time = (time.time() - start_time) * 1000
        logger.info(f"V2 user resolution took: {user_resolution_time:.2f}ms")
        
        return metadata, resolved_user_id, is_qwen_route, user_roles, user_workspace_ids, workspace_id
        
    except Exception as e:
        logger.error(f"Error in V2 user resolution for search: {e}")
        from models.memory_models import MemoryMetadata
        metadata = search_request.metadata or MemoryMetadata()
        return metadata, developer_user_id, None, [], [], None


async def _resolve_user_for_memory_parallel_v2(
    developer_user_id: str,
    memory_request: "AddMemoryRequest",
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> Tuple["MemoryMetadata", str, Optional[bool], List[str], List[str], Optional[str]]:
    """V2: Resolve user for memory with developer_user_id already known."""
    try:
        logger.info(f"Starting V2 user resolution for memory with developer_user_id: {developer_user_id}")
        
        from models.memory_models import MemoryMetadata
        metadata = memory_request.metadata or MemoryMetadata()
        
        # Same logic as search but for memory request - check original request
        original_metadata_user_id = getattr(memory_request.metadata, "user_id", None) if memory_request.metadata else None
        original_metadata_external_user_id = getattr(memory_request.metadata, "external_user_id", None) if memory_request.metadata else None
        
        if (not original_metadata_user_id and not original_metadata_external_user_id):
            metadata.user_id = developer_user_id
            return metadata, developer_user_id, None, [], [], None
        
        # Use existing resolution with multi-tenant context
        result = await User.resolve_end_user_id(
            developer_id=developer_user_id,
            metadata=metadata,
            authenticated_user_id=developer_user_id,
            httpx_client=httpx_client,
            api_key=api_key,
            organization_id=organization_id,
            namespace_id=namespace_id
        )
        
        updated_metadata, resolved_user_id, is_qwen_route = result
        return updated_metadata, resolved_user_id, is_qwen_route, [], [], None
        
    except Exception as e:
        logger.error(f"Error in V2 user resolution for memory: {e}")
        from models.memory_models import MemoryMetadata
        metadata = memory_request.metadata or MemoryMetadata()
        return metadata, developer_user_id, None, [], [], None


async def _resolve_user_for_batch_parallel_v2(
    developer_user_id: str,
    batch_request: "BatchMemoryRequest",
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> Tuple["BatchMemoryRequest", str, Optional[bool], List[str], List[str], Optional[str]]:
    """V2: Resolve user for batch with developer_user_id already known."""
    try:
        logger.info(f"Starting V2 user resolution for batch with developer_user_id: {developer_user_id}")
        logger.info(f"batch_request.external_user_id: {getattr(batch_request, 'external_user_id', None)}")
        
        # Use existing batch resolution
        logger.info(f"CRITICAL DEBUG: About to call User.patch_and_resolve_user_ids_and_acls")
        logger.info(f"CRITICAL DEBUG: batch_request before patch: {batch_request}")
        updated_batch_request, resolved_end_user_id, is_qwen_route = await User.patch_and_resolve_user_ids_and_acls(
            developer_id=developer_user_id,
            batch_request=batch_request,
            httpx_client=httpx_client,
            x_api_key=api_key
        )
        logger.info(f"CRITICAL DEBUG: batch_request after patch: {updated_batch_request}")
        logger.info(f"CRITICAL DEBUG: resolved_end_user_id: {resolved_end_user_id}")
        
        # Get resolved user's workspace if different from developer
        workspace_id = None
        if resolved_end_user_id != developer_user_id:
            logger.info(f"Batch Case 2/3: Resolved to different user - fetching workspace for {resolved_end_user_id}")
            headers = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "Content-Type": "application/json",
                "X-Parse-Master-Key": PARSE_MASTER_KEY
            }
            user_url = f"{PARSE_SERVER_URL}/parse/classes/_User/{resolved_end_user_id}"
            user_params = {
                "include": "isSelectedWorkspaceFollower,isSelectedWorkspaceFollower.workspace",
                "keys": "objectId,isSelectedWorkspaceFollower"
            }
            
            if httpx_client:
                response = await httpx_client.get(user_url, headers=headers, params=user_params)
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(user_url, headers=headers, params=user_params)
            
            user_obj = response.json()
            selected_follower = user_obj.get('isSelectedWorkspaceFollower')
            if selected_follower and isinstance(selected_follower, dict):
                workspace = selected_follower.get('workspace')
                if workspace and isinstance(workspace, dict):
                    workspace_id = workspace.get('objectId')
                    logger.info(f"Batch Case 2/3: Found workspace_id for resolved user: {workspace_id}")
        else:
            logger.info(f"Batch Case 1: Developer is end user - workspace_id will be filled by main auth function")
        
        return updated_batch_request, resolved_end_user_id, is_qwen_route, [], [], workspace_id
        
    except Exception as e:
        logger.error(f"Error in V2 user resolution for batch: {e}")
        return batch_request, developer_user_id, None, [], [], None


async def _resolve_user_for_update_parallel_v2(
    developer_user_id: str,
    update_request: "UpdateMemoryRequest",
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> Tuple["MemoryMetadata", str, Optional[bool], List[str], List[str], Optional[str]]:
    """V2: Resolve user for update with developer_user_id already known."""
    try:
        logger.info(f"Starting V2 user resolution for update with developer_user_id: {developer_user_id}")
        
        from models.memory_models import MemoryMetadata
        metadata = update_request.metadata or MemoryMetadata()
        
        # Check original request metadata to determine if this is Case 1
        original_metadata_user_id = getattr(update_request.metadata, "user_id", None) if update_request.metadata else None
        original_metadata_external_user_id = getattr(update_request.metadata, "external_user_id", None) if update_request.metadata else None
        
        if (not original_metadata_user_id and not original_metadata_external_user_id):
            metadata.user_id = developer_user_id
            return metadata, developer_user_id, None, [], [], None
        
        result = await User.resolve_end_user_id(
            developer_id=developer_user_id,
            metadata=metadata,
            authenticated_user_id=developer_user_id,
            httpx_client=httpx_client,
            api_key=api_key,
            organization_id=organization_id,
            namespace_id=namespace_id
        )
        
        updated_metadata, resolved_user_id, is_qwen_route = result
        return updated_metadata, resolved_user_id, is_qwen_route, [], [], None
        
    except Exception as e:
        logger.error(f"Error in V2 user resolution for update: {e}")
        from models.memory_models import MemoryMetadata
        metadata = update_request.metadata or MemoryMetadata()
        return metadata, developer_user_id, None, [], [], None


async def _resolve_user_for_feedback_parallel_v2(
    developer_user_id: str,
    feedback_request: "FeedbackRequest",
    api_key: Optional[str] = None,
    httpx_client: Optional[httpx.AsyncClient] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> Tuple[None, str, Optional[bool], List[str], List[str], Optional[str]]:
    """V2: Resolve user for feedback with developer_user_id already known."""
    try:
        logger.info(f"Starting V2 user resolution for feedback with developer_user_id: {developer_user_id}")
        
        # For feedback, typically no resolution needed - use developer
        return None, developer_user_id, None, [], [], None
        
    except Exception as e:
        logger.error(f"Error in V2 user resolution for feedback: {e}")
        return None, developer_user_id, None, [], [], None

async def mark_user_as_developer_if_needed(user_id: str, api_key: str, memory_graph: "MemoryGraph", httpx_client: Optional[httpx.AsyncClient] = None) -> None:
    """
    Mark a user as developer if they own an API key and haven't been checked before.
    This runs only once per user to avoid unnecessary database writes.
    """
    try:
        # Try MongoDB first for performance
        if memory_graph.mongo_client:
            try:
                user_doc = memory_graph.db['_User'].find_one(
                    {"_id": user_id},
                    {"isDeveloper": 1, "isDeveloperChecked": 1, "userAPIkey": 1}
                )
                
                if user_doc:
                    # Skip if already checked
                    if user_doc.get("isDeveloperChecked"):
                        return
                    
                    # Verify this user owns the API key
                    if user_doc.get("userAPIkey") == api_key:
                        # Mark as developer
                        memory_graph.db['_User'].update_one(
                            {"_id": user_id},
                            {
                                "$set": {
                                    "isDeveloper": True,
                                    "isDeveloperChecked": True,
                                    "developerOnboardedAt": datetime.now()
                                }
                            }
                        )
                        logger.info(f"Marked user {user_id} as developer (MongoDB)")
                        return
            except Exception as e:
                logger.warning(f"MongoDB developer marking failed, falling back to Parse: {e}")
        
        # Fallback to Parse Server
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }
        
        # Get user data to check if already processed
        user_url = f"{PARSE_SERVER_URL}/parse/classes/_User/{user_id}"
        params = {"keys": "isDeveloper,isDeveloperChecked,userAPIkey"}
        
        if httpx_client:
            response = await httpx_client.get(user_url, headers=headers, params=params)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(user_url, headers=headers, params=params)
        
        if response.status_code == 200:
            user_data = response.json()
            
            # Skip if already checked
            if user_data.get("isDeveloperChecked"):
                return
            
            # Verify this user owns the API key
            if user_data.get("userAPIkey") == api_key:
                # Update user to mark as developer
                update_data = {
                    "isDeveloper": True,
                    "isDeveloperChecked": True,
                    "developerOnboardedAt": {
                        "__type": "Date",
                        "iso": datetime.now().isoformat()
                    }
                }
                
                if httpx_client:
                    update_response = await httpx_client.put(user_url, headers=headers, json=update_data)
                else:
                    async with httpx.AsyncClient() as client:
                        update_response = await client.put(user_url, headers=headers, json=update_data)
                
                if update_response.status_code == 200:
                    logger.info(f"Marked user {user_id} as developer (Parse Server)")
                else:
                    logger.warning(f"Failed to mark user as developer: {update_response.status_code}")
                    
    except Exception as e:
        logger.error(f"Error marking user as developer: {e}")

async def _get_cached_schema_patterns_direct(
    user_object_id: str,
    workspace_object_id: str,
    httpx_client: Optional[httpx.AsyncClient] = None,
    skip_registration: bool = False,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get cached schema patterns directly using user_id and workspace_id.
    This avoids re-authentication and uses Parse Master Key directly.

    Args:
        user_object_id: Parse User objectId
        workspace_object_id: Parse Workspace objectId
        httpx_client: Optional HTTP client for reuse
        skip_registration: If True, skip dynamic property registration (for search operations)
        organization_id: Optional organization context
        namespace_id: Optional namespace context
    """
    try:
        logger.info(
            f"🚀 DIRECT SCHEMA CACHE: Starting cached schema lookup for user {user_object_id}, "
            f"workspace {workspace_object_id}, org_id={organization_id}, namespace_id={namespace_id}"
        )
        
        from services.active_node_rel_service import get_active_node_rel_service
        cache_service = get_active_node_rel_service()
        
        # Start both tasks in parallel: get cached schema AND register dynamic properties
        async def get_schema_task():
            if httpx_client:
                return await cache_service.get_cached_schema(
                    user_object_id=user_object_id,
                    workspace_object_id=workspace_object_id,
                    httpx_client=httpx_client
                )
            else:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    return await cache_service.get_cached_schema(
                        user_object_id=user_object_id,
                        workspace_object_id=workspace_object_id,
                        httpx_client=client
                    )
        
        async def register_dynamic_properties_task():
            """Register user's custom properties in parallel"""
            try:
                from services.schema_service import SchemaService
                from models.cipher_ast import register_user_custom_properties
                
                schema_service = SchemaService()
                logger.info(
                    f"🔍 REGISTER PROPS: Fetching schemas with org_id={organization_id}, namespace_id={namespace_id}"
                )
                # Get user's active schemas
                user_schemas = await schema_service.get_active_schemas(
                    user_object_id, workspace_object_id, organization_id, namespace_id
                )
                
                # Dynamically register custom properties (with caching)
                register_user_custom_properties(user_schemas)
                logger.info(f"🚀 PARALLEL REGISTRATION: Processed {len(user_schemas)} user schemas for custom properties")
                return True
                
            except Exception as e:
                logger.warning(f"🚀 PARALLEL REGISTRATION FAILED: {e}")
                return False
        
        # Run schema task (and optionally registration task in parallel)
        import asyncio

        if skip_registration:
            # SEARCH OPTIMIZATION: Skip registration for search operations
            logger.info(f"🚀 SEARCH MODE: Skipping dynamic property registration")
            schema_result = await get_schema_task()
            registration_result = None

            if isinstance(schema_result, Exception):
                logger.warning(f"🚀 SCHEMA CACHE ERROR: {schema_result}")
                cached_schema = None
            else:
                cached_schema = schema_result
        else:
            # ADD MODE: Run both tasks in parallel for full schema + registration
            logger.info(f"🚀 ADD MODE: Running schema fetch + dynamic registration in parallel")
            schema_result, registration_result = await asyncio.gather(
                get_schema_task(),
                register_dynamic_properties_task(),
                return_exceptions=True
            )

            # Handle schema result
            if isinstance(schema_result, Exception):
                logger.warning(f"🚀 SCHEMA CACHE ERROR: {schema_result}")
                cached_schema = None
            else:
                cached_schema = schema_result

            # Handle registration result
            if isinstance(registration_result, Exception):
                logger.warning(f"🚀 REGISTRATION ERROR: {registration_result}")
            elif registration_result:
                logger.info(f"🚀 PARALLEL OPTIMIZATION: Dynamic registration completed successfully")
        
        if cached_schema:
            logger.info(f"🚀 DIRECT SCHEMA CACHE HIT: Found cached schema with {len(cached_schema.get('nodes', []))} nodes and {len(cached_schema.get('patterns', []))} patterns")
            logger.info(f"🔍 ACTIVE PATTERNS DEBUG: cached_schema keys={list(cached_schema.keys())}, patterns_count={len(cached_schema.get('patterns', []))}")
            if cached_schema.get('patterns'):
                logger.info(f"🔍 ACTIVE PATTERNS DEBUG: First 3 patterns={cached_schema.get('patterns', [])[:3]}")
            # Add registration status to cached schema
            cached_schema['dynamic_registration_completed'] = bool(registration_result)
            return cached_schema
        else:
            logger.info(f"🚀 DIRECT SCHEMA CACHE MISS: No cached schema found")
            return None
            
    except Exception as e:
        logger.warning(f"🚀 DIRECT SCHEMA CACHE ERROR: Failed to retrieve cached schema: {e}")
        return None

async def _get_cached_schema_patterns_parallel(
    auth_header: str,
    client_type: str,
    memory_graph: "MemoryGraph",
    httpx_client: Optional[httpx.AsyncClient] = None
) -> Optional[Dict[str, Any]]:
    """
    Get cached schema patterns from Parse in parallel with authentication.
    This optimizes search performance by pre-fetching schema patterns.
    """
    try:
        logger.info(f"🚀 PARALLEL SCHEMA CACHE: Starting cached schema lookup")
        
        # First, get user_id and workspace_id from auth
        user_id = None
        workspace_id = None
        
        # Extract user_id from auth header
        if 'Bearer ' in auth_header:
            token = auth_header.split('Bearer ')[1]
            user_info = await User.verify_access_token(token, client_type, httpx_client=httpx_client)
            user_id = user_info.get('https://papr.scope.com/objectId')
        elif 'Session ' in auth_header:
            sessionToken = auth_header.split('Session ')[1]
            parse_user = await User.verify_session_token(sessionToken, httpx_client=httpx_client)
            if parse_user:
                user_id = parse_user.objectId
        elif 'APIKey ' in auth_header:
            api_key = auth_header.split('APIKey ')[1]
            if memory_graph.mongo_client:
                user_obj = await _verify_api_key_mongo(memory_graph.db, api_key, memory_graph)
                if user_obj:
                    user_id = user_obj.get('_id')
                    workspace_id = user_obj.get('workspace_id')
        
        if not user_id:
            logger.warning(f"🚀 PARALLEL SCHEMA CACHE: Could not extract user_id from auth")
            return None
        
        # Get workspace_id if not already available
        if not workspace_id:
            user_instance = User(user_id)
            workspace_data, _ = await user_instance._get_workspace_and_subscription_fast(memory_graph)
            if workspace_data:
                workspace_id = workspace_data.get('objectId')
        
        # Now get cached schema patterns
        from services.active_node_rel_service import get_active_node_rel_service
        cache_service = get_active_node_rel_service()
        
        # Use provided httpx_client or create a new one
        if httpx_client:
            cached_schema = await cache_service.get_cached_schema(
                user_object_id=user_id,
                workspace_object_id=workspace_id,
                httpx_client=httpx_client
            )
        else:
            async with httpx.AsyncClient(timeout=5.0) as client:
                cached_schema = await cache_service.get_cached_schema(
                    user_object_id=user_id,
                    workspace_object_id=workspace_id,
                    httpx_client=client
                )
        
        if cached_schema:
            logger.info(f"🚀 PARALLEL SCHEMA CACHE HIT: Found cached schema with {len(cached_schema.get('nodes', []))} nodes and {len(cached_schema.get('patterns', []))} patterns")
            return cached_schema
        else:
            logger.info(f"🚀 PARALLEL SCHEMA CACHE MISS: No cached schema found")
            return None
            
    except Exception as e:
        logger.warning(f"🚀 PARALLEL SCHEMA CACHE ERROR: Failed to retrieve cached schema: {e}")
        return None


async def _get_user_schemas_for_agentic_search(
    user_object_id: str,
    workspace_object_id: str,
    httpx_client: Optional[httpx.AsyncClient] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> List[Any]:
    """Fetch UserGraphSchema for agentic search property enhancement using flexible multi-tenant query"""
    
    try:
        logger.info(f"🚀 USER SCHEMA FETCH: Getting UserGraphSchema for user {user_object_id}, workspace {workspace_object_id}")
        logger.info(f"🔍 USER SCHEMA DEBUG: Multi-tenant context - org_id={organization_id}, namespace_id={namespace_id}")
        
        from services.schema_service import SchemaService
        schema_service = SchemaService()
        
        # FIXED: Use get_active_schemas (calls list_schemas) with flexible multi-tenant ACL
        # This matches the same logic used during ADD operations
        user_schemas = await schema_service.get_active_schemas(
            user_id=user_object_id,
            workspace_id=workspace_object_id,
            organization_id=organization_id,
            namespace_id=namespace_id
        )
        
        logger.info(f"🚀 USER SCHEMA FETCH: Found {len(user_schemas)} active schemas using flexible ACL")
        logger.info(f"🔍 USER SCHEMA DEBUG: Schema types={[type(s).__name__ for s in user_schemas]}")
        return user_schemas
        
    except Exception as e:
        logger.warning(f"🚀 USER SCHEMA FETCH ERROR: {e}")
        logger.error(f"🔍 USER SCHEMA DEBUG: Exception details", exc_info=True)
        return []


async def _combine_schema_cache_results(
    active_patterns_task: Awaitable,
    user_schemas_task: Awaitable
) -> Optional[Dict[str, Any]]:
    """Combine ActiveNodeRel patterns with UserGraphSchema definitions"""
    
    try:
        logger.info(f"🔍 COMBINE DEBUG: Starting to combine ActiveNodeRel + UserGraphSchema results")
        
        # Wait for both tasks to complete
        active_patterns, user_schemas = await asyncio.gather(
            active_patterns_task,
            user_schemas_task,
            return_exceptions=True
        )
        
        logger.info(f"🔍 COMBINE DEBUG: Gather completed - active_patterns type={type(active_patterns)}, user_schemas type={type(user_schemas)}")
        
        # Handle exceptions
        if isinstance(active_patterns, Exception):
            logger.warning(f"ActiveNodeRel fetch failed: {active_patterns}")
            active_patterns = None
            
        if isinstance(user_schemas, Exception):
            logger.warning(f"UserGraphSchema fetch failed: {user_schemas}")
            user_schemas = []
        
        # Build combined schema cache - BACKWARD COMPATIBLE
        combined_cache = {}
        
        # CRITICAL: Always include ActiveNodeRel data if available (for backward compatibility)
        if active_patterns:
            combined_cache.update(active_patterns)
            logger.info(f"🚀 COMBINED CACHE: ActiveNodeRel data included with {len(active_patterns.get('patterns', []))} patterns")
        else:
            logger.warning(f"🚀 COMBINED CACHE: No ActiveNodeRel patterns available")
            
        # OPTIONAL: Add UserGraphSchema data for property enhancement (don't fail if missing)
        if user_schemas:
            combined_cache['user_schemas'] = user_schemas
            combined_cache['indexable_properties'] = _build_indexable_properties_map(user_schemas)
            logger.info(f"🚀 COMBINED CACHE: UserGraphSchema enhancement added with {len(user_schemas)} schemas")
        else:
            logger.info(f"🚀 COMBINED CACHE: No UserGraphSchema data - continuing with ActiveNodeRel only")
            
        logger.info(f"🚀 COMBINED CACHE RESULT: ActiveNodeRel={bool(active_patterns)}, UserSchemas={len(user_schemas) if user_schemas else 0}, Total patterns={len(combined_cache.get('patterns', []))}")
        
        # BACKWARD COMPATIBLE: Return ActiveNodeRel data even if UserGraphSchema fails
        return combined_cache if active_patterns else None
        
    except Exception as e:
        logger.error(f"🚀 COMBINED CACHE ERROR: {e}")
        return None


def _build_indexable_properties_map(user_schemas: List[Any]) -> Dict[str, List[Dict]]:
    """Build comprehensive map of indexable properties from user schemas"""
    
    indexable_properties = {}
    
    # Add system schema properties (Memory, Person, Organization, etc.)
    from models.cipher_ast import NODE_PROPERTY_MAP
    for node_label, property_class in NODE_PROPERTY_MAP.items():
        node_type = node_label.value if hasattr(node_label, 'value') else str(node_label)
        
        if hasattr(property_class, 'model_fields'):
            for field_name, field_info in property_class.model_fields.items():
                if _is_indexable_system_property(field_name, field_info):
                    prop_key = f"{node_type}.{field_name}"
                    if prop_key not in indexable_properties:
                        indexable_properties[prop_key] = []
                    
                    indexable_properties[prop_key].append({
                        'schema_id': None,  # System schema
                        'schema_name': 'System',
                        'is_required': field_info.is_required(),
                        'property_type': 'string',
                        'has_enum': False
                    })
    
    # Add user schema properties
    for schema in user_schemas:
        schema_id = getattr(schema, 'id', None) or getattr(schema, 'objectId', None)
        schema_name = getattr(schema, 'name', 'Unknown')
        
        if hasattr(schema, 'node_types'):
            for node_type_name, node_def in schema.node_types.items():
                if hasattr(node_def, 'properties'):
                    required_props = getattr(node_def, 'required_properties', [])
                    
                    for prop_name, prop_def in node_def.properties.items():
                        if _is_indexable_user_property(prop_name, prop_def, required_props):
                            prop_key = f"{node_type_name}.{prop_name}"
                            if prop_key not in indexable_properties:
                                indexable_properties[prop_key] = []
                            
                            indexable_properties[prop_key].append({
                                'schema_id': schema_id,
                                'schema_name': schema_name,
                                'is_required': True,  # We only index required properties
                                'property_type': prop_def.type.value if hasattr(prop_def.type, 'value') else str(prop_def.type),
                                'has_enum': bool(getattr(prop_def, 'enum_values', None))
                            })
    
    logger.info(f"🔧 INDEXABLE PROPERTIES: Built map with {len(indexable_properties)} indexable properties")
    return indexable_properties


def _is_indexable_system_property(field_name: str, field_info: Any) -> bool:
    """Check if system property should be indexed"""
    
    # Skip system metadata fields
    if field_name in ['id', 'user_id', 'workspace_id', 'createdAt', 'updatedAt']:
        return False
        
    # Only index required string fields
    if not field_info.is_required():
        return False
        
    # Check if it's a string type (simplified check)
    annotation = getattr(field_info, 'annotation', None)
    if annotation and annotation != str:
        return False
        
    return True


def _is_indexable_user_property(prop_name: str, prop_def: Any, required_props: List[str]) -> bool:
    """Check if user schema property should be indexed"""
    
    # Must be required
    if prop_name not in required_props:
        return False
        
    # Must be string type  
    from models.user_schemas import PropertyType
    if not hasattr(prop_def, 'type') or prop_def.type != PropertyType.STRING:
        return False
        
    # Skip if has enum values (deterministic)
    if getattr(prop_def, 'enum_values', None):
        return False
        
    # Skip system properties
    if prop_name in ['id', 'user_id', 'workspace_id', 'createdAt', 'updatedAt']:
        return False
        
    return True
