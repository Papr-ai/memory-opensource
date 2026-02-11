# user.py
from flask_login import UserMixin
from jose import jwt  # Add this import
import requests  # Add this import
from auth0.authentication import Users  # Add this import
from dotenv import find_dotenv, load_dotenv
from os import environ as env  # Change this line
import json
from services.logger_singleton import LoggerSingleton
from services.url_utils import clean_url, get_parse_server_url
import asyncio
from datetime import datetime, timezone, UTC
from services.stripe_service import stripe_service, stripe  # Import both the service and the module
from uuid import uuid4  # Add this import
from threading import Thread  # Add this import
import httpx  # Import at the top of the file in practice
import math
from typing import Dict, Tuple, Optional, Any, List, Union, TypedDict, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from models.operation_types import MemoryOperationType
from models.parse_server import ParseUserPointer, InteractionLimits, TierLimits
from models.memory_models import MemoryMetadata, ContextItem, BatchMemoryRequest
from models.user_models import CreateUserRequest
from services.cache_utils import session_token_cache, access_token_cache, customer_tier_cache, workspace_subscription_cache

# Get logger instance
logger = LoggerSingleton.get_logger(__name__)


# Load environment variables (conditionally based on USE_DOTENV)
use_dotenv = env.get("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)

# Initialize Parse client
PARSE_SERVER_URL = clean_url(env.get("PARSE_SERVER_URL"))
logger.info(f"PARSE_SERVER_URL: {PARSE_SERVER_URL}")
PARSE_APPLICATION_ID = clean_url(env.get("PARSE_APPLICATION_ID"))
logger.info(f"PARSE_APPLICATION_ID: {PARSE_APPLICATION_ID}")
PARSE_MASTER_KEY = clean_url(env.get("PARSE_MASTER_KEY"))
logger.info(f"PARSE_MASTER_KEY: {PARSE_MASTER_KEY}")

# Update HEADERS to only use required fields
HEADERS = {
    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
    "X-Parse-Master-Key": PARSE_MASTER_KEY,
    "Content-Type": "application/json"
}

TierType = Literal['pro', 'business_plus', 'enterprise', 'free_trial', 'developer', 'starter', 'growth']
InteractionType = Literal['mini', 'premium']

# Add these constants at the top of the file with other constants
PRICE_IDS = {
    'pro': {
        'monthly': 'price_1QYkbKLvxLkj9c6vSascP5yn',
        'yearly': 'price_1QYkafLvxLkj9c6vTXtTry9W',
    },
    'businessPlus': {
        'monthly': 'price_1QYkncLvxLkj9c6vZffzw8JS',
        'yearly': 'price_1QYkoKLvxLkj9c6vOPKvQ89Y',
    },
    'developer': {
        'monthly': 'price_1RaumyLvxLkj9c6vLA6ZzCW8',
        'yearly': 'price_1RaumyLvxLkj9c6vLA6ZzCW8',
    },
    'starter': {
        'monthly': 'price_1RaTTDLvxLkj9c6vO6NfLFem',
        'yearly': 'price_1RaTUwLvxLkj9c6vMqQRpElV',
    },
    'growth': {
        'monthly': 'price_1Rac1GLvxLkj9c6vfXC5NlQc',
        'yearly': 'price_1Rac1GLvxLkj9c6v7rV5Vf5D',
    },
    'enterprise': {
        'monthly': 'price_1QYl3wLvxLkj9c6vZ9LhwEnu',
        'yearly': 'price_1QYl3wLvxLkj9c6vZ9LhwEnu',
    },
}

from enum import Enum

class StripeSubscriptionStatus(Enum):
    INCOMPLETE = 'incomplete'  # Initial payment failed, awaiting payment
    INCOMPLETE_EXPIRED = 'incomplete_expired'  # Initial payment failed and expired
    TRIALING = 'trialing'  # In trial period
    ACTIVE = 'active'  # Subscription is active
    PAST_DUE = 'past_due'  # Payment failed but retrying
    CANCELED = 'canceled'  # Subscription canceled
    UNPAID = 'unpaid'  # Payment failed and no more attempts
    PAUSED = 'paused'  # Trial ended without payment method


class UserEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, User):
            return obj.to_dict()  # Use the to_dict method to serialize
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

class User(UserMixin):
    def __init__(self, id):
        self.objectId = id
        self.id = id  # Keep this for UserMixin compatibility
        
    def to_dict(self):
        # Convert to a dictionary representation
        return {'objectId': self.objectId, 'id': self.id}

    @staticmethod
    def get(user_id):
        # Here you should fetch the user from your Parse Server DB using the user_id
        # For now, let's just return a User object
        return User(user_id)

    @staticmethod
    async def get_user_async(user_id: str):
        """
        Asynchronously fetch user information from Parse Server
        Returns user's display name or full name
        """
        url = f"{PARSE_SERVER_URL}/parse/users/{user_id}"
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                user_data = response.json()
                return {
                    'name': user_data.get('displayName') or user_data.get('fullName') or 'Unknown User'
                }
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch user data: {str(e)}")
                raise Exception(f"Failed to fetch user data: {str(e)}")

    @staticmethod
    async def get_company_async(user_id: str, workspace_id: str = None, session_token: str = None, api_key: Optional[str] = None):
        """
        Asynchronously fetch company information from Parse Server
        First gets workspace info, then fetches the linked company's display name
        """
        if not workspace_id:
            if not session_token:
                session_token = await User.lookup_user_token(user_id)
            workspace_id = await User.get_selected_workspace_id_async(user_id, session_token, api_key)
            if not workspace_id:
                return None

        # First, get the workspace to find the company pointer
        workspace_url = f"{PARSE_SERVER_URL}/parse/classes/WorkSpace/{workspace_id}"
        params = {
            "include": "company"  # This will include the company object in the response
        }
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,            
            "Content-Type": "application/json"
        }
        if api_key is not None:
            headers["X-Parse-Master-Key"] = PARSE_MASTER_KEY
        else:
            headers["X-Parse-Session-Token"] = session_token

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(workspace_url, headers=headers, params=params)
                response.raise_for_status()
                workspace_data = response.json()
                company_data = workspace_data.get('company')
                if company_data:
                    return company_data.get('displayName', 'Unknown Company')
                return None
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch workspace/company data: {str(e)}")
                raise Exception(f"Failed to fetch workspace/company data: {str(e)}")
        
    @staticmethod
    def get_workspaces_for_user(user_id: str):
        # Prepare the URL to query workspace_follower class
        url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower"

        # Prepare the query to filter by user
        query = {
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        }

        # Prepare the headers
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        # Send the request
        response = requests.get(url, headers=headers, params={"where": json.dumps(query)})

        # Check the response
        if response.status_code == 200:
            data = response.json()
            workspace_ids = [follower['workspace']['objectId'] for follower in data['results']]
            return workspace_ids
        else:
            raise Exception(f"Failed to fetch workspaces for user: {response.text}")
        
    async def get_roles_async_mongo(self, MongoClient) -> List[str]:
        """
        Asynchronously fetch roles from the _User class using httpx
        
        Returns:
            List[str]: List of role names for the user
        """
        if not MongoClient:
            logger.error("MongoClient is not initialized")
            return []
        db = MongoClient.get_default_database()
        logger.info(f"[get_roles_async_mongo] user_id={self.id}")
        # 1. Find all role IDs for this user
        role_links = list(db["_Join:users:_Role"].find({"relatedId": self.id}))
        logger.info(f"[get_roles_async_mongo] role_links={role_links}")
        role_ids = [link["owningId"] for link in role_links]
        logger.info(f"[get_roles_async_mongo] role_ids={role_ids}")
        # 2. Get the role documents
        roles = list(db["_Role"].find({"_id": {"$in": role_ids}}))
        logger.info(f"[get_roles_async_mongo] roles={roles}")
        role_names = [role['name'] for role in roles]
        logger.info(f"[get_roles_async_mongo] role_names={role_names}")
        return role_names
    
    async def get_roles_async(self) -> List[str]:
        """
        Asynchronously fetch roles from the _User class using httpx
        
        Returns:
            List[str]: List of role names for the user
        """
        # Use runtime function to get URL (applies localhost override for open-source local testing)
        parse_url = get_parse_server_url()
        url = f"{parse_url}/parse/classes/_Role"
        
        params = {
            "where": json.dumps({
                "users": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": self.id
                }
            })
        }
        
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                
                roles_data = response.json()
                return [role['name'] for role in roles_data.get('results', [])]
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred while fetching roles: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching roles: {e}")
            return []

    
    async def get_workspaces_for_user_async_mongodb(self, MongoClient, user_id: str) -> List[str]:
        """
        Asynchronously fetch workspaces for a user using httpx
        """
        if not MongoClient:
            logger.error("MongoClient is not initialized")
            return []
        db = MongoClient.get_default_database()
        pointer_value = f"_User${user_id}"
        logger.info(f"[get_workspaces_for_user_async_mongodb] user_id={user_id}, pointer_value={pointer_value}")
        workspaces_follower = list(db["workspace_follower"].find({"_p_user": pointer_value}))
        logger.info(f"[get_workspaces_for_user_async_mongodb] workspaces_follower={workspaces_follower}")
        workspace_ids = []
        for follower in workspaces_follower:
            pointer = follower.get('_p_workspace')
            if pointer and pointer.startswith('WorkSpace$'):
                workspace_id = pointer.split('$', 1)[1]
                workspace_ids.append(workspace_id)
        logger.info(f"[get_workspaces_for_user_async_mongodb] workspace_ids={workspace_ids}")
        return workspace_ids
        
    @staticmethod
    async def get_workspaces_for_user_async(user_id: str) -> List[str]:
        """
        Asynchronously fetch workspaces for a user using httpx
        
        Args:
            user_id (str): The ID of the user to fetch workspaces for
            
        Returns:
            List[str]: List of workspace IDs the user has access to
            
        Raises:
            HTTPError: If the request fails
        """
        # Use runtime function to get URL (applies localhost override for open-source local testing)
        parse_url = get_parse_server_url()
        url = f"{parse_url}/parse/classes/workspace_follower"

        query = {
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        }

        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        params = {
            "where": json.dumps(query)
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:  
                response = await client.get(
                    url,
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                
                data = response.json()
                workspace_ids = [
                    follower['workspace']['objectId'] 
                    for follower in data.get('results', [])
                    if 'workspace' in follower and 'objectId' in follower['workspace']
                ]
                return workspace_ids

        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred while fetching workspaces: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching workspaces for user {user_id}: {e}")
            raise
    @staticmethod
    async def get_user_info_enhanced(user_id: str, api_key: str = None, httpx_client: Optional[httpx.AsyncClient] = None) -> Optional[Dict[str, Any]]:
        """
        Get enhanced user info including isQwenRoute in a single call.
        """
        # Use runtime function to get URL (applies localhost override for open-source local testing)
        parse_url = get_parse_server_url()
        url = f"{parse_url}/parse/classes/_User/{user_id}"
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,            
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                user_data = response.json()
                return user_data
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred while getting user info: {e}")
                raise
            except Exception as e:
                logger.error(f"Error getting user info: {e}")
                raise

    @staticmethod
    async def verify_api_key(api_key: str, httpx_client: Optional[httpx.AsyncClient] = None) -> Optional[dict]:
        """
        Asynchronously verify an API key and return the full user object with all needed fields (isQwenRoute, isSelectedWorkspaceFollower, etc.).
        """
        # Use runtime URL resolution only for open-source edition (for localhost conversion in local tests)
        # Cloud edition uses static PARSE_SERVER_URL to maintain backward compatibility
        from config.features import get_features
        features = get_features()
        
        if not features.is_cloud:
            # Open-source edition: use runtime URL resolution for localhost conversion
            from services.url_utils import get_parse_server_url
            parse_server_url = get_parse_server_url()
            if not parse_server_url:
                logger.error("PARSE_SERVER_URL is not set")
                raise ValueError("PARSE_SERVER_URL is not configured")
        else:
            # Cloud edition: use static PARSE_SERVER_URL (backward compatible)
            parse_server_url = PARSE_SERVER_URL
            if not parse_server_url:
                logger.error("PARSE_SERVER_URL is not set")
                raise ValueError("PARSE_SERVER_URL is not configured")
        
        url = f"{parse_server_url}/parse/users"
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,            
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }
        
        # Query parameters to find user with matching API key, and include all needed fields including organization and namespace
        params = {
            "where": json.dumps({
                "userAPIkey": api_key
            }),
            "include": "isSelectedWorkspaceFollower,isSelectedWorkspaceFollower.workspace,isSelectedWorkspaceFollower.workspace.organization,isSelectedWorkspaceFollower.workspace.namespace",
            "keys": "objectId,username,email,isQwenRoute,isSelectedWorkspaceFollower,userAPIkey"
        }
        
        client = httpx_client or httpx.AsyncClient()
        try:
            if httpx_client:
                response = await client.get(url, headers=headers, params=params)
            else:
                async with client:
                    response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            user_data = response.json()
            
            # Check if any users were found
            if user_data.get('results') and len(user_data['results']) > 0:
                return user_data['results'][0]  # Return the first matching user (full object)
            return None  # Return None if no user found with this API key
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred while verifying API key: {e}")
            raise
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            raise
    
    @staticmethod
    async def verify_session_token(session_token: str, httpx_client: Optional[httpx.AsyncClient] = None) -> Optional[ParseUserPointer]:
        """
        Asynchronously verify a session token and return the associated user with caching.
        
        Args:
            session_token (str): The session token to verify
            
        Returns:
            Optional[User]: User instance if token is valid, None otherwise
        """
        # Check cache first
        cache_key = f"session_token:{session_token}"
        cached_result = session_token_cache.get(cache_key)
        if cached_result:
            logger.info(f"Session token cache HIT for {session_token[:5]}...")
            return cached_result
        
        logger.info(f"Session token cache MISS for {session_token[:5]}...")
        
        # Use runtime URL resolution to handle localhost conversion for local tests
        # get_parse_server_url() only applies localhost override when:
        # - PAPR_EDITION=opensource AND running locally (pytest or not in Docker)
        # This ensures cloud deployments are not affected
        from services.url_utils import get_parse_server_url
        parse_server_url = get_parse_server_url()
        logger.info(f"parse server URL: {parse_server_url}")
        logger.info(f"Verifying session token: {session_token[:5]}...")
        logger.info(f"PARSE_APPLICATION_ID: {env.get('PARSE_APPLICATION_ID')}")
        
        # Construct URL correctly - get_parse_server_url() returns base URL (e.g., http://localhost:1337 for local tests)
        # For cloud, it returns the original PARSE_SERVER_URL unchanged
        # Parse Server endpoint is /parse/users/me
        base_url = parse_server_url.rstrip('/')
        url = f"{base_url}/parse/users/me"
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Session-Token": session_token  # Session token is sufficient for authentication
        }

        # Include workspace, organization, and namespace for multi-tenant context
        params = {
            "include": "isSelectedWorkspaceFollower,isSelectedWorkspaceFollower.workspace,isSelectedWorkspaceFollower.workspace.organization,isSelectedWorkspaceFollower.workspace.namespace",
            "keys": "objectId,username,email,isQwenRoute,isSelectedWorkspaceFollower,userAPIkey"
        }

        logger.info(f"Making request to {url}")

        client = httpx_client or httpx.AsyncClient()
        try:
            if httpx_client:
                response = await client.get(url, headers=headers, params=params)
            else:
                async with client:
                    response = await client.get(url, headers=headers, params=params)
            
            logger.info(f"Response status code: {response.status_code}")
            logger.debug(f"Response body: {response.text}")
            
            response.raise_for_status()
            
            user_data = response.json()
            logger.debug(f"User data: {user_data}")
            result = User(user_data['objectId'])
            
            # Cache the result
            session_token_cache.set(cache_key, result)
            logger.info(f"Cached session token result for {session_token[:5]}...")
            
            return result
                
        except httpx.HTTPError as e:
            logger.error(f"Invalid session token. Error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error verifying session token: {str(e)}")
            return None

    @staticmethod
    async def verify_access_token(token: str, client_type: str, httpx_client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
        """
        Asynchronously verify an access token and retrieve user information from Auth0 with caching.
        
        Args:
            token (str): The access token to verify
            client_type (str): The type of client ('browser_extension' or 'papr_plugin')
            
        Returns:
            Dict[str, Any]: User information from Auth0
            
        Raises:
            ValueError: If client_type is invalid
        """
        # Check cache first
        cache_key = f"access_token:{token}"
        cached_result = access_token_cache.get(cache_key)
        if cached_result:
            logger.info(f"Access token cache HIT for {token[:5]}...")
            return cached_result
        
        logger.info(f"Access token cache MISS for {token[:5]}...")
        
        if client_type == 'browser_extension':
            client_id = clean_url(env.get("AUTH0_CLIENT_ID_BROWSER"))
        elif client_type == 'papr_plugin':
            client_id = clean_url(env.get("AUTH0_CLIENT_ID_PAPR"))
        else:
            raise ValueError("Invalid client type")

        auth0_domain = clean_url(env.get("AUTH0_DOMAIN"))
        url = f"https://{auth0_domain}/userinfo"
        headers = {'Authorization': f'Bearer {token}'}

        try:
            client = httpx_client or httpx.AsyncClient()
            if httpx_client:
                response = await client.get(url, headers=headers)
            else:
                async with client:
                    response = await client.get(url, headers=headers)
            
            response.raise_for_status()
            user_info = response.json()
            logger.info(f"Received user_info: {user_info}")
            
            # Cache the result
            access_token_cache.set(cache_key, user_info)
            logger.info(f"Cached access token result for {token[:5]}...")
            
            return user_info
        except httpx.HTTPError as e:
            logger.error(f"Failed to verify access token: {str(e)}")
            raise ValueError("Failed to verify access token") from e

    
    @staticmethod
    async def store_auth_token(access_token: str, user_id: str, session_token: str, api_key: Optional[str] = None):
        """Store the access token for a user in Parse Server.

        Args:
            access_token (str): The access token to store
            user_id (str): The user's ID
            session_token (str): The session token for authentication

        Raises:
            Exception: If the update fails
        """
        # Prepare the URL
        url = f"{PARSE_SERVER_URL}/parse/users/{user_id}"
        logger.info(f"url for parseServer: {url}")

        # Prepare the headers
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,            
            "Content-Type": "application/json"
        }
        if api_key is not None:
            headers["X-Parse-Master-Key"] = PARSE_MASTER_KEY
        else:
            headers["X-Parse-Session-Token"] = session_token

        # Prepare the data
        data = {
            "access_token": access_token
        }

        # Send the request
        async with httpx.AsyncClient() as client:
            response = await client.put(url, headers=headers, json=data)
            logger.info(f"added data to Parse Server: {str(response)}")

            # Check the response
            if response.status_code != 200:
                raise Exception(f"Failed to update user: {response.text}")

    @staticmethod
    async def lookup_access_token(user_id: str, session_token: str = None):
        """Look up the access token for a user from Parse Server.

        Args:
            user_id (str): The user's ID
            session_token (str, optional): The session token for authentication. Defaults to None.

        Returns:
            str: The access token if found

        Raises:
            Exception: If the retrieval fails
        """
        # Prepare the URL
        url = f"{PARSE_SERVER_URL}/parse/users/{user_id}"

        # Prepare the headers
        if session_token:
            headers = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-Session-Token": session_token,
                "Content-Type": "application/json"
            }
        else:
            headers = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-Master-Key": PARSE_MASTER_KEY,
                "Content-Type": "application/json"
            }

        # Send the request
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            logger.info(f"query to get user_id from Parse Server: {str(response)}")

            # Check the response
            if response.status_code != 200:
                raise Exception(f"Failed to retrieve user: {response.text}")

            # Parse the response
            data = response.json()
            logger.info(f"response from parse parsed: {str(data)}")

            # Return the access token
            return data.get("access_token")

    @staticmethod
    async def lookup_user_token(user_id: str) -> str:
        """
        Asynchronously lookup the most recent session token for a user.
        
        Args:
            user_id (str): The user's ID to lookup
            
        Returns:
            str: The session token if found
            
        Raises:
            Exception: If no session is found or if the request fails
        """
        # Prepare the URL
        url = f"{PARSE_SERVER_URL}/parse/classes/_Session"

        # Prepare the query parameters
        query_params = {
            "where": json.dumps({
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                }
            }),
            "order": "-createdAt",
            "limit": 1
        }

        # Prepare the headers
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }
        logger.info(f"lookup_user_token headers: {headers} params: {query_params}")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, params=query_params)
                response.raise_for_status()
                
                logger.info(f"Query to get user session from Parse Server: {response.url}")
                
                # Parse the response
                data = response.json()
                logger.info(f"Response from Parse Server: {data}")

                if data.get("results") and len(data["results"]) > 0:
                    return data["results"][0].get("sessionToken")
                else:
                    return None  # Instead of raising Exception
                    
            except httpx.HTTPError as e:
                error_msg = f"Failed to retrieve user session: {str(e)}"
                logger.error(error_msg)
                return None

    @staticmethod
    def save_get_memory_request(query: str, user_id: str, context: Optional[List[ContextItem]], relation_type: str, metadata, result, neoQuery=None, memory_source=None):
        # Prepare the URL
        url = f"{PARSE_SERVER_URL}/parse/classes/GetMemoryRequests"


        # Prepare the headers
        headers = HEADERS

        # Handle None or empty object for context
        if context is None or context == {}:
            context = []  # Set to an empty array

        # Prepare the data
        data = {
            "query": query,
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "context": context,
            "relation_type": relation_type,
            "metadata": metadata,
            "result": result,
            "neoQuery": neoQuery,
            "memorySource": memory_source            
        }

        # Send the request
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Check the response
        if response.status_code != 201:  # 201 Created
            raise Exception(f"Failed to save get_memory request: {response.text}")

        # Log the successful addition
        logger.info(f"Successfully added get_memory request to Parse Server: {response.json()}")

        # Return the result
        return response.json()
    
    def get_roles(self):
        # Fetch roles from the _User class
        url = f"{PARSE_SERVER_URL}/parse/classes/_Role?where={{\"users\":{{\"__type\":\"Pointer\",\"className\":\"_User\",\"objectId\":\"{self.id}\"}}}}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            roles_data = response.json()
            return [role['name'] for role in roles_data['results']]
        return []

    @staticmethod
    def get_user_session_by_tenant(subtenant_id: str):
        # Fetch the session token directly using the subtenant_id (which is the user_id)
        session_url = f"{PARSE_SERVER_URL}/parse/classes/_Session?where={{\"user\":{{\"__type\":\"Pointer\",\"className\":\"_User\",\"objectId\":\"{subtenant_id}\"}}}}"
        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }
        session_response = requests.get(session_url, headers=HEADERS)
        logger.info(f"session response: {session_response}")

        if session_response.status_code == 200:
            session_data = session_response.json()
            if session_data['results']:
                logger.info(f"session_data: {session_data}")
                # Find session with longest time before expiring
                session = None
                latest_expiry = None
                for result in session_data['results']:
                    expires_at = result.get('expiresAt')
                    if expires_at and isinstance(expires_at, dict):
                        iso_date = expires_at.get('iso')
                        if iso_date:
                            if not latest_expiry or iso_date > latest_expiry:
                                latest_expiry = iso_date
                                session = result
                if not session and session_data['results']:
                    session = session_data['results'][0]  # Fallback if no expiry times found
                logger.info(f"session: {session}")
                session_token = session.get('sessionToken')
                if session_token:
                    return session_token
                else:
                    logger.error("Session token not found in the response")
                    return None, None
        else:
            logger.error(f"Failed to fetch session token: {session_response.text}")
        return None, None

    @staticmethod
    async def get_acl_for_workspace(
        workspace_id: Optional[str] = None, 
        tenant_id: Optional[str] = None, 
        user_id: Optional[str] = None, 
        additional_user_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, List[str]]]:
        """
        Asynchronously retrieves and transforms the ACL for a given workspace.

        Args:
            workspace_id (Optional[str]): The objectId of the workspace
            tenant_id (Optional[str]): The tenantId to query the workspace
            user_id (Optional[str]): The userId to include if no specific access is set
            additional_user_ids (Optional[List[str]]): List of additional userIds to grant access

        Returns:
            Optional[Dict[str, List[str]]]: Transformed ACL dictionary containing:
                - user_read_access: List[str]
                - user_write_access: List[str]
                - workspace_read_access: List[str]
                - workspace_write_access: List[str]
                - role_read_access: List[str]
                - role_write_access: List[str]
                Returns None if workspace is not found or on error
        """
        if workspace_id:
            where_clause = {"objectId": workspace_id}
        elif tenant_id:
            where_clause = {"tenantId": tenant_id}
        else:
            return None

        url = f"{PARSE_SERVER_URL}/parse/classes/WorkSpace"
        params = {
            "where": json.dumps(where_clause)
        }
        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        user_acl = {user_id: {"read": True, "write": True}} if user_id else {}
        logger.info(f"user_acl: {user_acl}")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=HEADERS, params=params)
                response.raise_for_status()
                
                workspace_data = response.json()
                if workspace_data['results']:
                    workspace = workspace_data['results'][0]
                    acl = workspace.get('ACL')
                    
                    if acl is not None:
                        return User.transform_acl(acl, workspace_id, user_id, additional_user_ids)
                    else:
                        # Handle the case where ACL is not set (public access)
                        logger.info(f"No ACL found for workspace_id: {workspace_id}. Assuming public access.")
                        return User.transform_acl(
                            {"*": {"read": True, "write": True}}, 
                            workspace_id, 
                            user_id, 
                            additional_user_ids
                        )
                else:
                    logger.warning(f"No workspace found for query: {where_clause}")
                    return None
                    
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch workspace: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching workspace: {str(e)}")
                return None
    
    @staticmethod
    async def get_acl_for_postMessage(
        workspace_id: Optional[str] = None, 
        post_message_id: Optional[str] = None, 
        user_id: Optional[str] = None, 
        additional_user_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, List[str]]]:
        """
        Asynchronously retrieves and transforms the ACL for a given PostMessage.

        Args:
            workspace_id (Optional[str]): The objectId of the workspace
            post_message_id (Optional[str]): The objectId of the PostMessage
            user_id (Optional[str]): The userId to include
            additional_user_ids (Optional[List[str]]): List of additional userIds to grant access

        Returns:
            Optional[Dict[str, List[str]]]: Transformed ACL dictionary containing:
                - user_read_access: List[str]
                - user_write_access: List[str]
                - workspace_read_access: List[str]
                - workspace_write_access: List[str]
                - role_read_access: List[str]
                - role_write_access: List[str]
                Returns None if PostMessage is not found or on error
        """
        if not post_message_id:
            return None

        url = f"{PARSE_SERVER_URL}/parse/classes/PostMessage"
        params = {
            "where": json.dumps({"objectId": post_message_id})
        }
        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=HEADERS, params=params)
                response.raise_for_status()
                
                post_message_data = response.json()
                if post_message_data['results']:
                    post_message = post_message_data['results'][0]
                    acl = post_message.get('ACL')
                    
                    if acl is not None:
                        return User.transform_acl(acl, workspace_id, user_id, additional_user_ids)
                    else:
                        # Handle the case where ACL is not set (public access)
                        logger.info(f"No ACL found for post_message_id: {post_message_id}. Assuming public access.")
                        return User.transform_acl(
                            {"*": {"read": True, "write": True}}, 
                            workspace_id, 
                            user_id, 
                            additional_user_ids
                        )
                else:
                    logger.warning(f"No PostMessage found for id: {post_message_id}")
                    return None
                    
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch PostMessage: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching PostMessage: {str(e)}")
                return None
    
    @staticmethod
    async def get_acl_for_post(
        workspace_id: Optional[str] = None, 
        post_id: Optional[str] = None, 
        user_id: Optional[str] = None, 
        additional_user_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, List[str]]]:
        """
        Asynchronously retrieves and transforms the ACL for a given Post.

        Args:
            workspace_id (Optional[str]): The objectId of the workspace
            post_id (Optional[str]): The objectId of the Post
            user_id (Optional[str]): The userId to include
            additional_user_ids (Optional[List[str]]): List of additional userIds to grant access

        Returns:
            Optional[Dict[str, List[str]]]: Transformed ACL dictionary containing:
                - user_read_access: List[str]
                - user_write_access: List[str]
                - workspace_read_access: List[str]
                - workspace_write_access: List[str]
                - role_read_access: List[str]
                - role_write_access: List[str]
                Returns None if Post is not found or on error
        """
        if not post_id:
            return None

        url = f"{PARSE_SERVER_URL}/parse/classes/Post"
        params = {
            "where": json.dumps({"objectId": post_id})
        }
        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=HEADERS, params=params)
                response.raise_for_status()
                
                post_data = response.json()
                if post_data['results']:
                    post = post_data['results'][0]
                    acl = post.get('ACL')
                    
                    if acl is not None:
                        return User.transform_acl(acl, workspace_id, user_id, additional_user_ids)
                    else:
                        # Handle the case where ACL is not set (public access)
                        logger.info(f"No ACL found for post_id: {post_id}. Assuming public access.")
                        return User.transform_acl(
                            {"*": {"read": True, "write": True}}, 
                            workspace_id, 
                            user_id, 
                            additional_user_ids
                        )
                else:
                    logger.warning(f"No Post found for id: {post_id}")
                    return None
                    
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch Post: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching Post: {str(e)}")
                return None

    
    @staticmethod
    def transform_acl(acl: dict, workspace_id: str = None, user_id: str = None, additional_user_ids: list = None):
        """
        Transforms the ACL from the Parse Server into a structured dictionary.

        Parameters:
            acl (dict): The ACL dictionary from Parse Server.
            workspace_id (str): The objectId of the workspace.
            user_id (str, optional): The userId to include if no specific access is set.
            additional_user_ids (list, optional): List of additional userIds to grant access.

        Returns:
            dict: Transformed ACL with access lists.
        """
        user_read_access = []
        user_write_access = []
        workspace_read_access = []
        workspace_write_access = []
        role_read_access = []
        role_write_access = []
        namespace_read_access = []
        namespace_write_access = []
        organization_read_access = []
        organization_write_access = []

        # If ACL is provided, process it normally
        for key, value in acl.items():
            if (key == "*" and workspace_id):
                if value.get("read", True):
                    workspace_read_access.append(workspace_id)
                if value.get("write", True):
                    workspace_write_access.append(workspace_id)
            elif key.startswith("role:"):
                role_id = key.split(":")[-1]  # Extract the role ID after the last colon
                if value.get("read", True):
                    role_read_access.append(role_id)
                if value.get("write", True):
                    role_write_access.append(role_id)
            elif key.startswith("namespace:"):
                namespace_id = key.split(":")[-1]  # Extract the namespace ID after the last colon
                if value.get("read", True):
                    namespace_read_access.append(namespace_id)
                if value.get("write", True):
                    namespace_write_access.append(namespace_id)
            elif key.startswith("organization:"):
                organization_id = key.split(":")[-1]  # Extract the organization ID after the last colon
                if value.get("read", True):
                    organization_read_access.append(organization_id)
                if value.get("write", True):
                    organization_write_access.append(organization_id)
            else:
                user_id = key
                if value.get("read", True):
                    user_read_access.append(user_id)
                if value.get("write", True):
                    user_write_access.append(user_id)

        # If user_id is provided and no specific user access is set, add it to user access
        if user_id and not user_read_access and not user_write_access:
            user_read_access.append(user_id)
            user_write_access.append(user_id)

        # Add additional_user_ids to both read and write access if provided
        if additional_user_ids:
            user_read_access.extend(additional_user_ids)
            user_write_access.extend(additional_user_ids)

        # Remove duplicates
        user_read_access = list(set(user_read_access))
        user_write_access = list(set(user_write_access))
        workspace_read_access = list(set(workspace_read_access))
        workspace_write_access = list(set(workspace_write_access))
        role_read_access = list(set(role_read_access))
        role_write_access = list(set(role_write_access))
        namespace_read_access = list(set(namespace_read_access))
        namespace_write_access = list(set(namespace_write_access))
        organization_read_access = list(set(organization_read_access))
        organization_write_access = list(set(organization_write_access))

        return {
            "user_read_access": user_read_access,
            "user_write_access": user_write_access,
            "workspace_read_access": workspace_read_access,
            "workspace_write_access": workspace_write_access,
            "role_read_access": role_read_access,
            "role_write_access": role_write_access,
            "namespace_read_access": namespace_read_access,
            "namespace_write_access": namespace_write_access,
            "organization_read_access": organization_read_access,
            "organization_write_access": organization_write_access
        }

    @staticmethod
    async def get_selected_workspace_id_async(user_id: str, session_token: Optional[str] = None, api_key: Optional[str] = None):
        """Get the workspace ID from the user's selected workspace follower asynchronously"""
        url = f"{PARSE_SERVER_URL}/parse/users/{user_id}"
        
        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,             
            "Content-Type": "application/json"
        }
        if api_key is not None:
            HEADERS["X-Parse-Master-Key"] = PARSE_MASTER_KEY
        elif session_token is not None:
            HEADERS["X-Parse-Session-Token"] = session_token
        
        logger.info(f"get_selected_workspace_id_async called with user_id={user_id}, session_token={session_token}, api_key={api_key}")
        logger.info(f"HEADERS: {HEADERS}")
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url, headers=HEADERS)
                response.raise_for_status()
                user_data = response.json()
                selected_follower = user_data.get('isSelectedWorkspaceFollower')
                
                if not selected_follower:
                    logger.warning(f"No selected workspace follower for user {user_id}")
                    return None

                # Get workspace follower data
                follower_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower/{selected_follower['objectId']}"
                follower_response = await client.get(follower_url, headers=HEADERS)
                follower_response.raise_for_status()
                follower_data = follower_response.json()
                workspace = follower_data.get('workspace')
                
                if workspace and 'objectId' in workspace:
                    return workspace['objectId']
                
                return None
            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred: {str(e)}")
                return None

    async def check_memory_limits(
        self,
        increment_count: bool = True,
        memory_size_mb: float = 0.0,
        namespace_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        organization_id: Optional[str] = None
    ) -> Optional[Tuple[Dict[str, Any], int]]:
        """
        Asynchronously checks if the user has exceeded their memory limits based on their subscription.
        If increment_count is True, it also increments the memory counter and sends a meter event to Stripe.

        This check is ONLY enabled in cloud edition. Open source has no limits.

        Args:
            increment_count (bool): Whether to increment the memory count (default: True)
            memory_size_mb (float): Size of the memory being added in MB (default: 0.0)
            namespace_id (Optional[str]): Namespace ID if using multi-tenant isolation
            api_key_id (Optional[str]): API Key ID for tracking usage per key
            workspace_id (Optional[str]): Workspace ID (if provided, skips workspace lookup via Parse REST API)
            organization_id (Optional[str]): Organization ID for multi-tenant support (if provided, updates organization counts directly)

        Returns:
            Optional[Tuple[Dict[str, Any], int]]: Tuple containing response and status code if limit exceeded,
                                                or None if ok
        """
        # Import here to avoid circular dependency
        from config.features import get_features
        features = get_features()

        def _to_int(value: Any) -> int:
            try:
                if value is None:
                    return 0
                if isinstance(value, (int, float)):
                    return int(value)
                if isinstance(value, str):
                    return int(float(value))
            except (ValueError, TypeError):
                pass
            return 0
        
        def _bytes_to_mb(value: int) -> float:
            return value / (1024 * 1024) if value else 0.0
        
        # Skip checks entirely in open source edition
        if not features.is_cloud:
            logger.info("Open source edition detected - skipping memory limits check")
            return None
        
        # Skip if subscription enforcement is disabled
        if not features.is_enabled("subscription_enforcement"):
            logger.info("Subscription enforcement disabled - skipping memory limits check")
            return None
        
        try:
            # Get workspace data - either from provided workspace_id or from user's selected workspace
            if workspace_id:
                # Use provided workspace_id directly - query workspace and find workspace_follower
                logger.info(f"Using provided workspace_id: {workspace_id}")

                # Query workspace directly
                workspace_url = f"{PARSE_SERVER_URL}/parse/classes/WorkSpace/{workspace_id}"
                params = {
                    "include": "subscription,organization"  # Include subscription and organization
                }
                HEADERS = {
                    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                    "X-Parse-Master-Key": PARSE_MASTER_KEY,
                    "Content-Type": "application/json"
                }

                # 30 second timeout to handle slow network conditions (e.g., coffee shop WiFi, MongoDB connection issues)
                async with httpx.AsyncClient(timeout=30.0) as client:
                    workspace_response = await client.get(workspace_url, headers=HEADERS, params=params)
                    used_fallback = False

                    if workspace_response.status_code != 200:
                        logger.warning(f"Failed to get workspace {workspace_id}: {workspace_response.text}")
                        logger.info(f"Falling back to user's workspace_follower for workspace")

                        # Fallback: Query workspace_follower collection directly for this user's selected workspace
                        follower_query_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower"
                        follower_params = {
                            "where": json.dumps({
                                "user": {"__type": "Pointer", "className": "_User", "objectId": self.id},
                                "isSelected": True
                            }),
                            "include": "workspace,workspace.subscription,workspace.organization",
                            "limit": 1
                        }
                        fallback_response = await client.get(follower_query_url, headers=HEADERS, params=follower_params)
                        if fallback_response.status_code != 200:
                            logger.error(f"Failed to get workspace follower for user {self.id}: {fallback_response.text}")
                            return {"error": "Unable to verify workspace"}, 500

                        fallback_results = fallback_response.json().get('results', [])
                        if not fallback_results:
                            logger.error(f"No workspace follower found for user {self.id}")
                            return {"error": "Unable to determine workspace"}, 400

                        workspace_data = fallback_results[0]
                        workspace = workspace_data.get('workspace')
                        if not workspace:
                            return {"error": "Unable to determine workspace"}, 400
                        used_fallback = True
                        logger.info(f"Successfully fell back to user's selected workspace: {workspace.get('objectId')}")
                    else:
                        workspace = workspace_response.json()

                    # Only need to find the workspace_follower if we didn't use the fallback
                    # (fallback already gives us workspace_data)
                    if not used_fallback:
                        follower_query_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower"
                        follower_params = {
                            "where": json.dumps({
                                "user": {"__type": "Pointer", "className": "_User", "objectId": self.id},
                                "workspace": {"__type": "Pointer", "className": "WorkSpace", "objectId": workspace_id}
                            })
                        }
                        follower_response = await client.get(follower_query_url, headers=HEADERS, params=follower_params)
                        if follower_response.status_code == 200:
                            follower_results = follower_response.json().get('results', [])
                            workspace_data = follower_results[0] if follower_results else {}
                        else:
                            logger.warning(f"Could not find workspace_follower for user {self.id} and workspace {workspace_id}")
                            workspace_data = {}
            else:
                # Use existing logic - get selected workspace follower from user
                selected_follower_id = await self.get_selected_workspace_follower()
                if not selected_follower_id:
                    return {"error": "Unable to determine workspace"}, 400

                # Get workspace follower details
                workspace_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower/{selected_follower_id}"
                params = {
                    "include": "workspace,workspace.subscription,workspace.organization"  # Include workspace, its subscription, and organization
                }
                HEADERS = {
                    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                    "X-Parse-Master-Key": PARSE_MASTER_KEY,
                    "Content-Type": "application/json"
                }

                # 30 second timeout to handle slow network conditions (e.g., coffee shop WiFi, MongoDB connection issues)
                async with httpx.AsyncClient(timeout=30.0) as client:
                    workspace_response = await client.get(workspace_url, headers=HEADERS, params=params)
                    if workspace_response.status_code != 200:
                        logger.error(f"Failed to get workspace follower: {workspace_response.text}")
                        return {"error": "Unable to verify workspace"}, 500

                    workspace_data = workspace_response.json()
                    workspace = workspace_data.get('workspace')
                    if not workspace:
                        return {"error": "Unable to determine workspace"}, 400

            # Common processing for both paths - extract subscription
            # Ensure workspace is a dict before accessing its attributes
            if not workspace or not isinstance(workspace, dict):
                logger.error(f"Invalid workspace object: {workspace}")
                return {"error": "Unable to verify workspace"}, 500
            
            # Ensure workspace_data is initialized - if it's None, initialize it as empty dict
            if workspace_data is None:
                logger.warning("workspace_data was None, initializing as empty dict")
                workspace_data = {}
            
            # Ensure workspace_data is a dict before accessing its attributes
            if not isinstance(workspace_data, dict):
                logger.error(f"Invalid workspace_data object type: {type(workspace_data)}, value: {workspace_data}")
                return {"error": "Unable to verify workspace data"}, 500
            
            subscription = workspace.get('subscription')
            if not subscription:
                return {
                    "error": "No subscription found for workspace please go to https://app.papr.ai to sign-up for a subscription"
                }, 400
            
            # Ensure subscription is a dict before accessing its attributes
            if not isinstance(subscription, dict):
                logger.error(f"Invalid subscription object type: {type(subscription)}, value: {subscription}")
                return {"error": "Invalid subscription data"}, 500

            stripe_customer_id = subscription.get('stripeCustomerId')
            is_metered_billing_on = subscription.get('isMeteredBillingOn', False)

            # Get current counts from workspace (organization level)
            workspace_obj_id = workspace.get('objectId')
            workspace_memories_count = workspace.get('memoriesCount', 0) or 0
            workspace_storage_bytes = _to_int(workspace.get('storageCount', 0))

            # Also get workspace_follower counts (for individual tracking)
            follower_memories_count = workspace_data.get('memoriesCount', 0) or 0
            follower_storage_bytes = _to_int(workspace_data.get('storageCount', 0))

            memory_size_bytes = int(math.ceil(memory_size_mb * 1024 * 1024)) if memory_size_mb else 0

            logger.info(
                f"workspace_memories_count: {workspace_memories_count}, "
                f"workspace_storage_bytes: {workspace_storage_bytes} ({_bytes_to_mb(workspace_storage_bytes):.6f} MB)"
            )
            logger.info(
                f"follower_memories_count: {follower_memories_count}, "
                f"follower_storage_bytes: {follower_storage_bytes} ({_bytes_to_mb(follower_storage_bytes):.6f} MB)"
            )
            logger.info(f"subscription: {subscription}")
            logger.info(f"api_key_id: {api_key_id}")
            logger.info(f"organization_id: {organization_id}")
            logger.info(f"namespace_id: {namespace_id}")

            # Increment counts in Parse Server if requested
            if increment_count and workspace_data:
                # Create new httpx client for increment operations with longer timeout
                # 30 second timeout to handle slow network conditions (e.g., coffee shop WiFi, MongoDB connection issues)
                async with httpx.AsyncClient(timeout=30.0) as increment_client:
                    # Get follower_id - either from selected_follower_id or workspace_data
                    follower_id = workspace_data.get('objectId') if workspace_data else None
                    if follower_id:
                        # Update workspace_follower counts
                        update_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower/{follower_id}"
                        update_data = {
                                "memoriesCount": {"__op": "Increment", "amount": 1},
                                "storageCount": {"__op": "Increment", "amount": memory_size_bytes}
                        }
                        update_response = await increment_client.put(update_url, headers=HEADERS, json=update_data)

                        if update_response.status_code != 200:
                            logger.error(f"Failed to update workspace_follower counts: {update_response.text}")
                        else:
                                follower_memories_count += 1
                                follower_storage_bytes += memory_size_bytes
                                logger.info(
                                    "Successfully incremented workspace_follower: "
                                    f"memories={follower_memories_count}, "
                                    f"storage_bytes={follower_storage_bytes} ({_bytes_to_mb(follower_storage_bytes):.6f} MB)"
                                )

                        # Also update workspace-level counts (organization aggregate)
                        workspace_update_url = f"{PARSE_SERVER_URL}/parse/classes/WorkSpace/{workspace_obj_id}"
                        workspace_update_data = {
                                "memoriesCount": {"__op": "Increment", "amount": 1},
                                "storageCount": {"__op": "Increment", "amount": memory_size_bytes}
                        }
                        workspace_update_response = await increment_client.put(workspace_update_url, headers=HEADERS, json=workspace_update_data)

                        if workspace_update_response.status_code != 200:
                            logger.error(f"Failed to update workspace counts: {workspace_update_response.text}")
                        else:
                                workspace_memories_count += 1
                                workspace_storage_bytes += memory_size_bytes
                                logger.info(
                                    "Successfully incremented workspace: "
                                    f"memories={workspace_memories_count}, "
                                    f"storage_bytes={workspace_storage_bytes} ({_bytes_to_mb(workspace_storage_bytes):.6f} MB)"
                                )

                        # Update organization counts (workspace = organization in our model)
                        # Use provided organization_id or fall back to extracting from workspace
                        org_id_to_use = organization_id
                        if not org_id_to_use:
                            # Fallback: Get organization_id from workspace (organization is a field on WorkSpace, not Subscription)
                            logger.info(f"DEBUG: workspace type: {type(workspace)}, workspace keys: {workspace.keys() if isinstance(workspace, dict) else 'Not a dict'}")
                            org = workspace.get('organization') if isinstance(workspace, dict) else None
                            logger.info(f"DEBUG: org type: {type(org)}, org value: {org}")
                            org_id_to_use = org.get('objectId') if isinstance(org, dict) else None
                            logger.info(f"organization_id extracted from workspace: {org_id_to_use}")
                        else:
                            logger.info(f"Using provided organization_id: {org_id_to_use}")
                        
                        if org_id_to_use:
                            org_update_url = f"{PARSE_SERVER_URL}/parse/classes/Organization/{org_id_to_use}"
                            org_update_data = {
                                "memoriesCount": {"__op": "Increment", "amount": 1},
                                "storageCount": {"__op": "Increment", "amount": memory_size_bytes}
                            }
                            org_update_response = await increment_client.put(org_update_url, headers=HEADERS, json=org_update_data)
                            
                            if org_update_response.status_code != 200:
                                logger.error(f"Failed to update organization counts: {org_update_response.text}")
                            else:
                                logger.info(
                                    f"Successfully incremented organization {org_id_to_use}: "
                                    f"+1 memory, +{memory_size_bytes} bytes"
                                )
                        else:
                            logger.warning("No organization_id provided or found in subscription to sync counts")

                        # Update namespace counts if namespace_id provided
                        if namespace_id:
                            namespace_update_url = f"{PARSE_SERVER_URL}/parse/classes/Namespace/{namespace_id}"
                            # Use increment operation for atomic updates
                            namespace_update_data = {
                                "memoriesCount": {"__op": "Increment", "amount": 1},
                                "storageCount": {"__op": "Increment", "amount": memory_size_bytes}
                            }
                            namespace_update_response = await increment_client.put(namespace_update_url, headers=HEADERS, json=namespace_update_data)
                            if namespace_update_response.status_code != 200:
                                logger.error(f"Failed to update namespace counts: {namespace_update_response.text}")
                            else:
                                logger.info(
                                    f"Successfully incremented namespace {namespace_id}: "
                                    f"+1 memory, +{memory_size_bytes} bytes"
                                )

                        # Update API key counts if api_key_id provided
                        if api_key_id:
                            apikey_update_url = f"{PARSE_SERVER_URL}/parse/classes/APIKey/{api_key_id}"
                            apikey_update_data = {
                                "memoriesCount": {"__op": "Increment", "amount": 1},
                                "storageCount": {"__op": "Increment", "amount": memory_size_bytes},
                                "last_used_at": {"__type": "Date", "iso": datetime.now(timezone.utc).isoformat()}
                            }
                            apikey_update_response = await increment_client.put(apikey_update_url, headers=HEADERS, json=apikey_update_data)
                            if apikey_update_response.status_code != 200:
                                logger.error(f"Failed to update API key counts: {apikey_update_response.text}")
                            else:
                                logger.info(
                                    f"Successfully incremented API key {api_key_id}: "
                                    f"+1 memory, +{memory_size_bytes} bytes"
                                )
                    
                    # Always send meter event to Stripe when incrementing the count
                    # regardless of whether metered billing is enabled
                    async def send_meter_event():
                        try:
                            meter_response = await stripe_service.send_meter_event(
                                event_name="papr_memories",
                                value=1,
                                stripe_customer_id=stripe_customer_id,
                            )
                            if meter_response is None:
                                logger.warning("Failed to send memory meter event to Stripe, but continuing")
                        except Exception as e:
                            logger.error(f"Error sending memory meter event: {str(e)}")

                    # Create a background task for the meter event
                    asyncio.create_task(send_meter_event())

            # Get customer tier from Stripe (regardless of metered billing status)
            logger.info(f"🔍 DEBUG: Getting customer tier from Stripe for customer_id: {stripe_customer_id}")
            # Use cached version to avoid blocking on Stripe API calls
            customer_tier = await self._get_customer_tier_fast(stripe_customer_id)
            logger.info(f"🔍 DEBUG: Stripe returned customer_tier: {customer_tier}")
            
            # Check if in trial period
            try:
                # Use stripe directly instead of the client
                subscriptions = await asyncio.to_thread(
                    stripe.Subscription.list,
                    customer=stripe_customer_id,
                    status='active',
                    limit=1
                )
                
                if subscriptions.data:
                    stripe_subscription = subscriptions.data[0]
                    is_trial = stripe_subscription.status == 'trialing'
                    logger.info(f"🔍 DEBUG: Subscription status from Stripe: {stripe_subscription.status}, is_trial: {is_trial}")
                    
                    # Update subscription status in Parse if needed
                    if is_trial != (subscription.get('status') == 'trial'):
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            update_url = f"{PARSE_SERVER_URL}/parse/classes/Subscription/{subscription['objectId']}"
                            update_data = {"status": "trial" if is_trial else "active"}
                            await client.put(update_url, headers=HEADERS, json=update_data)
                else:
                    is_trial = False
                    logger.info(f"🔍 DEBUG: No active subscriptions found, is_trial: False")

            except stripe.error.StripeError as e:
                logger.error(f"Stripe API error: {str(e)}")
                created_at = subscription.get('createdAt')
                if created_at:
                    created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                    is_trial = (datetime.now() - created_date).days <= 7
                else:
                    is_trial = False
                logger.info(f"🔍 DEBUG: Stripe error fallback, is_trial: {is_trial}")

            # Determine effective tier
            effective_tier = 'free_trial' if is_trial else customer_tier
            logger.info(f"🔍 DEBUG: effective_tier: {effective_tier}")
            
            # Get tier limits from configuration
            tier_limits = features.get_tier_limits(effective_tier)
            logger.info(f"🔍 DEBUG: tier_limits from config: {tier_limits}")
            
            # If metered billing is on, no need to check tier/limits (customer pays per usage)
            if is_metered_billing_on:
                logger.info(f"✅ Metered billing enabled - bypassing tier/subscription checks (customer pays per usage)")
                return None
            
            # Check if customer has no tier (no active subscription)
            if effective_tier is None or effective_tier == 'None':
                logger.warning(f"❌ No active subscription found for customer {stripe_customer_id}")
                return {
                    "error": "No active subscription",
                    "message": (
                        "You don't have an active subscription. "
                        "Please visit https://dashboard.papr.ai to activate a subscription and start adding memories."
                    ),
                    "tier": None,
                    "is_trial": False
                }, 403
            
            if not tier_limits:
                # No limits (unlimited tier or enterprise)
                logger.info(f"✅ No limits for tier '{effective_tier}' - allowing operation")
                return None
            
            # Extract limits (with fallbacks for legacy tiers)
            memory_count_limit = tier_limits.get('max_active_memories', float('inf'))
            storage_gb_limit = tier_limits.get('max_storage_gb', float('inf'))
            
            # Convert storage from bytes to GB for comparison
            workspace_storage_gb = workspace_storage_bytes / (1024 ** 3) if workspace_storage_bytes else 0.0
            
            logger.info(f"📊 TIER LIMITS CHECK:")
            logger.info(f"  - Customer: {stripe_customer_id}")
            logger.info(f"  - Tier: {effective_tier}")
            logger.info(f"  - Is Trial: {is_trial}")
            logger.info(f"  - Metered Billing Enabled: {is_metered_billing_on}")
            logger.info(f"  - Memory Limit: {memory_count_limit:,} memories")
            logger.info(f"  - Storage Limit: {storage_gb_limit}GB")
            logger.info(f"  - Current Memories: {workspace_memories_count:,} / {memory_count_limit:,}")
            logger.info(f"  - Current Storage: {workspace_storage_gb:.2f}GB / {storage_gb_limit}GB")
            
            # Check if EITHER limit is exceeded
            memory_limit_exceeded = memory_count_limit and workspace_memories_count >= memory_count_limit
            storage_limit_exceeded = storage_gb_limit and workspace_storage_gb >= storage_gb_limit
            
            if memory_limit_exceeded or storage_limit_exceeded:
                # Determine which limit was exceeded
                limit_type = []
                if memory_limit_exceeded:
                    limit_type.append(f"memory count ({workspace_memories_count:,}/{memory_count_limit:,})")
                if storage_limit_exceeded:
                    limit_type.append(f"storage ({workspace_storage_gb:.2f}GB/{storage_gb_limit}GB)")
                limit_description = " and ".join(limit_type)
                
                logger.info(f"❌ LIMIT EXCEEDED: {limit_description}")
                
                # Generate tier-specific error message
                tier_name = customer_tier.replace('_', ' ').title()
                
                if customer_tier in ['developer', 'free_trial']:
                    error_message = (
                        f"You've reached the {limit_description} limit for your {tier_name} plan. "
                        "To continue adding memories, upgrade to Starter ($100/mo) or Growth ($500/mo) plan.\n"
                        "Visit https://dashboard.papr.ai to manage your subscription."
                    )
                elif customer_tier == 'starter':
                    error_message = (
                        f"You've reached the {limit_description} limit for your Starter plan. "
                        "To continue, you can either:\n"
                        "1. Enable metered billing in your current plan, or\n"
                        "2. Upgrade to Growth plan for higher limits\n"
                        "Visit https://dashboard.papr.ai to manage your subscription."
                    )
                elif customer_tier == 'growth':
                    error_message = (
                        f"You've reached the {limit_description} limit for your Growth plan. "
                        "To continue, you can either:\n"
                        "1. Enable metered billing in your current plan, or\n"
                        "2. Contact us for Enterprise plan with unlimited resources\n"
                        "Visit https://dashboard.papr.ai to manage your subscription."
                    )
                elif customer_tier == 'pro':
                    error_message = (
                        f"You've reached the {limit_description} limit for your Pro plan. "
                        "To continue, enable metered billing or upgrade to a higher tier.\n"
                        "Visit https://app.papr.ai to manage your subscription."
                    )
                elif customer_tier == 'business_plus':
                    error_message = (
                        f"You've reached the {limit_description} limit for your Business Plus plan. "
                        "To continue, enable metered billing or contact us for Enterprise.\n"
                        "Visit https://app.papr.ai to manage your subscription."
                    )
                elif customer_tier == 'enterprise':
                    error_message = (
                        f"You've reached the {limit_description} limit for your Enterprise plan. "
                        "To continue, enable metered billing or contact us for Enterprise.\n"
                        "Visit https://dashboard.papr.ai to manage your subscription."
                    )
                else:
                    error_message = (
                        f"You've reached the {limit_description} limit for your {tier_name} plan. "
                        "Please visit https://dashboard.papr.ai to upgrade your subscription."
                    )
                
                return {
                    "error": "Limit reached",
                    "message": error_message,
                    "limits_exceeded": {
                        "memory_count": memory_limit_exceeded,
                        "storage": storage_limit_exceeded
                    },
                    "current": {
                        "memory_count": workspace_memories_count,
                        "storage_gb": round(workspace_storage_gb, 2)
                    },
                    "limits": {
                        "memory_count": memory_count_limit if memory_count_limit != float('inf') else None,
                        "storage_gb": storage_gb_limit if storage_gb_limit != float('inf') else None
                    },
                    "tier": customer_tier,
                    "is_trial": is_trial
                }, 403

            logger.info(f"✅ Within tier limits - allowing operation")
            return None  # No limits exceeded

        except httpx.ReadTimeout as e:
            logger.warning(f"Timeout checking memory limits (possibly slow network or MongoDB issues): {str(e)}")
            logger.warning("Allowing operation to continue despite timeout - limits check will be retried on next request")
            # Don't block the operation due to transient network issues
            # The memory counter will be updated on the next successful check
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error checking memory limits: {e.response.status_code} - {e.response.text}")
            logger.error("Full traceback:", exc_info=True)
            return {"error": "Unable to verify memory limits please visit https://dashboard.papr.ai to manage subscription."}, 500
        except Exception as e:
            logger.error(f"Error checking memory limits: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return {"error": "Unable to verify memory limits please visit https://dashboard.papr.ai to manage subscription."}, 500

    async def get_selected_workspace_follower(self) -> Optional[str]:
        """
        Get the workspace follower ID from the user's selected workspace follower asynchronously.

        Returns:
            Optional[str]: The workspace follower ID if found, None otherwise
        """
        url = f"{PARSE_SERVER_URL}/parse/users/{self.id}"

        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        # Include the isSelectedWorkspaceFollower field
        params = {
            "keys": "isSelectedWorkspaceFollower"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=HEADERS, params=params)
                response.raise_for_status()
                
                if response.status_code != 200:
                    logger.error(f"Failed to get user data: {response.text}")
                    return None

                user_data = response.json()
                selected_follower = user_data.get('isSelectedWorkspaceFollower')
                
                follower_id = selected_follower.get('objectId') if selected_follower else None
                logger.info(f"Retrieved workspace follower ID: {follower_id}")
                
                return follower_id

            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred while getting workspace follower: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error getting workspace follower: {str(e)}")
                logger.error("Full traceback:", exc_info=True)
                return None
    

    async def check_interaction_limits(
        self, 
        interaction_type: str = 'mini'
    ) -> Optional[Tuple[Dict[str, Any], int, bool]]:
        """
        Checks if the user has exceeded their interaction limits based on their subscription.
        Also ensures an Interaction object exists for the current month and updates counts.
        
        This check is ONLY enabled in cloud edition. Open source has no limits.
        
        Args:
            interaction_type (str): either 'mini' or 'premium'
            
       Returns:
            Optional[Tuple[Dict[str, Any], int, bool]]: Tuple containing (response_dict, status_code, is_error)
                                                or None if ok

        """
        # Import here to avoid circular dependency
        from config.features import get_features
        features = get_features()
        
        # Skip checks entirely in open source edition
        if not features.is_cloud:
            logger.info("Open source edition detected - skipping interaction limits check")
            return None
        
        # Skip if subscription enforcement is disabled
        if not features.is_enabled("subscription_enforcement"):
            logger.info("Subscription enforcement disabled - skipping interaction limits check")
            return None
        
        try:
            # Get workspace follower with included workspace and subscription
            selected_follower_id = await self.get_selected_workspace_follower()
            if not selected_follower_id:
                return {
                    "error": "No workspace found",
                    "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
                }, 403, True

            # Get workspace follower details with included pointers
            workspace_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower/{selected_follower_id}"
            params = {
                "include": "workspace,workspace.subscription,workspace.company"
            }
            HEADERS = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-Master-Key": PARSE_MASTER_KEY,
                "Content-Type": "application/json"
            }

            logger.info(f"workspace_url: {workspace_url}")

            async with httpx.AsyncClient() as client:
                workspace_response = await client.get(workspace_url, headers=HEADERS, params=params)
                if workspace_response.status_code != 200:
                    logger.error(f"Failed to get workspace follower: {workspace_response.text}")
                    return {
                        "error": "No workspace access",
                        "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
                    }, 403, True

                workspace_data = workspace_response.json()
                workspace = workspace_data.get('workspace')
                logger.info(f"workspace: {workspace}")
                if not workspace:
                    return {
                        "error": "No workspace found",
                        "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
                    }, 403, True

                subscription = workspace.get('subscription')
                logger.info(f"subscription: {subscription}")
                if not subscription:
                    return {
                        "error": "No active subscription",
                        "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
                    }, 403, True

                stripe_customer_id = subscription.get('stripeCustomerId')
                logger.info(f"check interaction limits stripe_customer_id: {stripe_customer_id}")
                is_metered_billing_on = subscription.get('isMeteredBillingOn', False)
                logger.info(f"check interaction limits is_metered_billing_on: {is_metered_billing_on}")

                # Check and update subscription trial status if needed
                try:
                    # First check all subscriptions (including canceled ones)
                    subscriptions = await asyncio.to_thread(
                        stripe.Subscription.list,
                        customer=stripe_customer_id,
                        status='all',  # Get all subscriptions including canceled
                        limit=1,
                        expand=['data.latest_invoice']  # Get invoice info to check payment status
                    )
                    logger.debug(f"check interaction limits subscriptions: {subscriptions}")

                    if not subscriptions.data:
                        logger.info("No subscription found - creating new trial subscription")
                        # Create a new subscription with trial
                        try:
                            # Use the monthly pro price ID from our constants
                            price_id = PRICE_IDS['pro']['monthly']
                            logger.info(f"Creating trial subscription with price_id: {price_id}")
                            
                            # Create the subscription with trial settings
                            new_subscription = await asyncio.to_thread(
                                stripe.Subscription.create,
                                customer=stripe_customer_id,
                                items=[{'price': price_id}],
                                trial_period_days=21,
                                trial_settings={
                                    'end_behavior': {
                                        'missing_payment_method': 'cancel'
                                    }
                                },
                                payment_settings={
                                    'payment_method_types': ['card'],
                                    'save_default_payment_method': 'on_subscription'
                                },
                                collection_method='charge_automatically'
                            )
                            logger.info(f"Created new trial subscription: {new_subscription.id}")
                            
                            # Update Parse subscription record
                            update_url = f"{PARSE_SERVER_URL}/parse/classes/Subscription/{subscription['objectId']}"
                            update_data = {
                                "status": "trial",
                                "tier": "pro",
                                "trialEndsAt": datetime.fromtimestamp(new_subscription.trial_end).isoformat()
                            }
                            await client.put(update_url, headers=HEADERS, json=update_data)
                            
                            # Store welcome message but continue execution
                            welcome_message = {
                                "message": "Welcome to Papr! You've been enrolled in a 21-day Pro trial. Visit https://app.papr.ai to add your payment method and continue using Papr after the trial.",
                                "trial_started": True,
                                "days_remaining": 21,
                                "trial_end": new_subscription.trial_end
                            }
                            
                            # Continue with rest of the code for metered events...
                            
                        except stripe.error.StripeError as e:
                            logger.error(f"Failed to create trial subscription: {str(e)}")
                            return {
                                "error": "Subscription setup failed",
                                "message": "Failed to set up your trial subscription. Please visit https://app.papr.ai to try again."
                            }, 403, True
                    
                    if subscriptions.data:
                        existing_subscription = subscriptions.data[0]
                        status = StripeSubscriptionStatus(existing_subscription.status)
                        logger.info(f"check memory limits status: {status}")
                        # Case 1: User has active or past_due subscription
                        if status in [StripeSubscriptionStatus.ACTIVE, StripeSubscriptionStatus.TRIALING]:
                            logger.info(f"check memory limits active or past due")
                            is_trial = False
                                       
                        # Case 3: Subscription needs attention
                        elif status in [
                            StripeSubscriptionStatus.CANCELED,
                            StripeSubscriptionStatus.INCOMPLETE,
                            StripeSubscriptionStatus.INCOMPLETE_EXPIRED,
                            StripeSubscriptionStatus.UNPAID,
                            StripeSubscriptionStatus.PAUSED,
                            StripeSubscriptionStatus.PAST_DUE
                        ]:
                            status_messages = {
                                StripeSubscriptionStatus.CANCELED: "Your subscription has been canceled",
                                StripeSubscriptionStatus.INCOMPLETE: "Your subscription setup was not completed",
                                StripeSubscriptionStatus.INCOMPLETE_EXPIRED: "Your initial subscription setup was not completed",
                                StripeSubscriptionStatus.UNPAID: "Your subscription has unpaid invoices",
                                StripeSubscriptionStatus.PAUSED: "Your subscription is paused",
                                StripeSubscriptionStatus.PAST_DUE: "Your subscription is past due"
                            }
                            return {
                                "error": "Subscription required",
                                "message": f"{status_messages[status]}. Please visit https://dashboard.papr.ai to reactivate your subscription and continue using Papr.",
                                "subscription_status": status.value
                            }, 403, True
                        
                        # Case 4: New user needs trial setup
                        else:
                            logger.info(f"check memory limits new user needs trial setup")
                            # Update subscription with trial settings
                            updated_subscription = await asyncio.to_thread(
                                stripe.Subscription.modify,
                                existing_subscription.id,
                                trial_period_days=21,
                                trial_settings={
                                    'end_behavior': {
                                        'missing_payment_method': 'cancel'
                                    }
                                },
                                payment_settings={
                                    'payment_method_types': ['card'],
                                    'save_default_payment_method': 'on_subscription'
                                },
                                collection_method='charge_automatically'
                            )
                            logger.info(f"check memory limits updated_subscription: {updated_subscription}")
                            
                            # Update Parse subscription record
                            update_url = f"{PARSE_SERVER_URL}/parse/classes/Subscription/{subscription['objectId']}"
                            update_data = {
                                "status": "trial",
                                "tier": "pro",
                                "trialEndsAt": datetime.fromtimestamp(updated_subscription.trial_end).isoformat()
                            }
                            await client.put(update_url, headers=HEADERS, json=update_data)
                            
                            # Store welcome message but don't return yet
                            welcome_message = {
                                "message": "Welcome to Papr! You've been enrolled in a 21-day Pro trial. Visit https://app.papr.ai to add your payment method and continue using Papr after the trial.",
                                "trial_started": True,
                                "days_remaining": 21,
                                "trial_end": updated_subscription.trial_end
                            }
                            # Continue execution...

                except stripe.error.StripeError as e:
                    logger.error(f"Stripe API error: {str(e)}")
                    created_at = subscription.get('createdAt')
                    if created_at:
                        created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                        is_trial = (datetime.now() - created_date).days <= 7
                    else:
                        is_trial = False

                # Get current month's usage and ensure Interaction object exists
                current_date = datetime.now()
                current_month = current_date.month
                current_year = current_date.year

                # Query current month's interaction
                interaction_url = f"{PARSE_SERVER_URL}/parse/classes/Interaction"
                query = {
                    "where": json.dumps({
                        "user": {
                            "__type": "Pointer",
                            "className": "_User",
                            "objectId": self.id
                        },
                        "workspace": {
                            "__type": "Pointer",
                            "className": "WorkSpace",
                            "objectId": workspace.get('objectId')
                        },
                        "type": interaction_type,
                        "month": current_month,
                        "year": current_year
                    })
                }
                
                interaction_response = await client.get(interaction_url, headers=HEADERS, params=query)
                if interaction_response.status_code != 200:
                    logger.error(f"Failed to get interaction count: {interaction_response.text}")
                    return {"error": "Unable to verify usage limits"}, 500, True

                interactions = interaction_response.json().get('results', [])
                logger.info(f"interactions: {interactions}")
                logger.info(f"mini interaction count: {interactions[0].get('count', 0) if interactions else 0}")
                
                if not interactions:
                    # Create new interaction record if none exists
                    new_interaction = {
                        "workspace": {
                            "__type": "Pointer",
                            "className": "WorkSpace",
                            "objectId": workspace.get('objectId')
                        },
                        "user": {
                            "__type": "Pointer",
                            "className": "_User",
                            "objectId": self.id
                        },
                        "type": interaction_type,
                        "month": current_month,
                        "year": current_year,
                        "count": 1
                    }
                    # Only add company pointer if it exists in workspace
                    if workspace.get('company'):
                        new_interaction["company"] = {
                            "__type": "Pointer",
                            "className": "Company",
                            "objectId": workspace['company']['objectId']
                        }
                    if subscription.get('objectId'):
                        new_interaction["subscription"] = {
                            "__type": "Pointer",
                            "className": "Subscription",
                            "objectId": subscription.get('objectId')
                        }
                    
                    create_response = await client.post(interaction_url, headers=HEADERS, json=new_interaction)
                    if create_response.status_code != 201:
                        logger.error(f"Failed to create interaction record: {create_response.text}")
                        return {"error": "Unable to create usage record"}, 500, True
                    current_count = 1
                else:
                    # Increment the existing interaction count
                    interaction = interactions[0]
                    current_count = interaction.get('count', 0) + 1
                    
                    # Update the interaction record
                    update_url = f"{interaction_url}/{interaction['objectId']}"
                    update_data = {"count": current_count}
                    update_response = await client.put(update_url, headers=HEADERS, json=update_data)
                    
                    if update_response.status_code != 200:
                        logger.error(f"Failed to update interaction count: {update_response.text}")
                        return {"error": "Unable to update usage record"}, 500, True

                # Always send meter event to Stripe regardless of whether metered billing is enabled
                async def send_meter_event():
                    try:
                        meter_response = await stripe_service.send_meter_event(
                            event_name=f"papr_{interaction_type}_interactions",
                            value=1,
                            stripe_customer_id=stripe_customer_id,
                        )
                        if meter_response is None:
                            logger.warning("Failed to send meter event to Stripe, but continuing")
                    except Exception as e:
                        logger.error(f"Error sending meter event: {str(e)}")

                # Create a background task for the meter event
                asyncio.create_task(send_meter_event())

                # If metered billing is on, no need to check limits
                if is_metered_billing_on:
                    # At the very end of the method
                    if 'welcome_message' in locals():
                        return welcome_message, 200, False
                    return None, 200, False  # No limits exceeded

                # Get customer tier from Stripe (cached)
                customer_tier = await self._get_customer_tier_fast(stripe_customer_id)

                # Check subscription status from Stripe
                try:
                    customer = await asyncio.to_thread(
                        stripe.Customer.retrieve,
                        stripe_customer_id
                    )
                    subscriptions = await asyncio.to_thread(
                        stripe.Subscription.list,
                        customer=stripe_customer_id,
                        limit=1
                    )
                    
                    if subscriptions.data:
                        stripe_subscription = subscriptions.data[0]
                        is_trial = stripe_subscription.status == 'trialing'
                        
                        # Update subscription status in Parse if needed
                        if is_trial != (subscription.get('status') == 'trial'):
                            update_url = f"{PARSE_SERVER_URL}/parse/classes/Subscription/{subscription['objectId']}"
                            update_data = {
                                "status": "trial" if is_trial else "active",
                                "tier": customer_tier
                            }
                            await client.put(update_url, headers=HEADERS, json=update_data)
                    else:
                        is_trial = False
                except stripe.error.StripeError as e:
                    logger.error(f"Stripe API error: {str(e)}")
                    created_at = subscription.get('createdAt')
                    if created_at:
                        created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                        is_trial = (datetime.now() - created_date).days <= 7
                    else:
                        is_trial = False

                # Get tier limits from configuration
                effective_tier = 'free_trial' if is_trial else customer_tier
                tier_config = features.get_tier_limits(effective_tier)
                
                if not tier_config:
                    # No limits (unlimited tier or enterprise)
                    logger.info(f"No interaction limits for tier {effective_tier}")
                    if 'welcome_message' in locals():
                        return welcome_message, 200, False
                    return None
                
                # Extract interaction limits
                mini_limit = tier_config.get('max_mini_interactions_per_month', float('inf'))
                premium_limit = tier_config.get('max_premium_interactions_per_month', float('inf'))
                
                # Map to old format for compatibility
                tier_limits = {
                    'mini': mini_limit if mini_limit != float('inf') else None,
                    'premium': premium_limit if premium_limit != float('inf') else None
                }
                
                limit_value = tier_limits.get(interaction_type)
                
                logger.info(f"Tier: {effective_tier}, {interaction_type} limit: {limit_value}")
                logger.info(f"Current count: {current_count}")

                if limit_value and current_count >= limit_value:
                    tier_name = customer_tier.replace('_', ' ').title()
                    
                    if customer_tier in ['developer', 'free_trial']:
                        error_message = (
                            f"You've reached the {limit_value:,} {interaction_type} interactions limit for your {tier_name} plan. "
                            "To continue, upgrade to Starter ($100/mo) or Growth ($500/mo) plan.\n"
                            "Visit https://dashboard.papr.ai to manage your subscription."
                        )
                    elif customer_tier == 'starter':
                        error_message = (
                            f"You've reached the {limit_value:,} {interaction_type} interactions limit for your Starter plan. "
                            "To continue, you can either:\n"
                            "1. Enable metered billing in your current plan, or\n"
                            "2. Upgrade to Growth plan for higher limits\n"
                            "Visit https://dashboard.papr.ai to manage your subscription."
                        )
                    elif customer_tier == 'growth':
                        error_message = (
                            f"You've reached the {limit_value:,} {interaction_type} interactions limit for your Growth plan. "
                            "To continue, you can either:\n"
                            "1. Enable metered billing in your current plan, or\n"
                            "2. Contact us for Enterprise plan with unlimited resources\n"
                            "Visit https://dashboard.papr.ai to manage your subscription."
                        )
                    elif customer_tier == 'pro':
                        error_message = (
                            f"You've reached the {limit_value:,} {interaction_type} interactions limit for your Pro plan. "
                            "To continue, enable metered billing or upgrade to a higher tier.\n"
                            "Visit https://app.papr.ai to manage your subscription."
                        )
                    elif customer_tier == 'business_plus':
                        error_message = (
                            f"You've reached the {limit_value:,} {interaction_type} interactions limit for your Business Plus plan. "
                            "To continue, enable metered billing or contact us for Enterprise.\n"
                            "Visit https://app.papr.ai to manage your subscription."
                        )
                    elif customer_tier == 'enterprise':
                        error_message = (
                            f"You've reached the {limit_value:,} {interaction_type} interactions limit for your Enterprise plan. "
                            "To continue, enable metered billing or contact us for Enterprise.\n"
                            "Visit https://dashboard.papr.ai to manage your subscription."
                        )
                    else:
                        error_message = (
                            f"You've reached the {limit_value:,} {interaction_type} interactions limit for your {tier_name} plan. "
                            "Please visit https://dashboard.papr.ai to upgrade your subscription."
                        )
                    
                    return {
                        "error": "Interaction limit reached",
                        "message": error_message,
                        "current_count": current_count,
                        "limit": limit_value,
                        "tier": customer_tier,
                        "is_trial": is_trial
                    }, 403, True

                 # At the end of the method, after metered events
                if 'welcome_message' in locals():
                    return welcome_message, 200, False
                return None, 200, False  # No limits exceeded and not an error

        except Exception as e:
            logger.error(f"Error checking interaction limits: {str(e)}")
            return {
                "error": "Subscription required",
                "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
            }, 403, True

    async def check_interaction_limits_fast(
        self, 
        interaction_type: str = 'mini',
        memory_graph = None,
        operation: Optional['MemoryOperationType'] = None,
        batch_size: Optional[int] = None,
        enable_agentic_graph: bool = False,
        enable_rank_results: bool = False,
        api_key_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None,
        defer_usage_tracking: bool = False
    ) -> Optional[Tuple[Dict[str, Any], int, bool]]:
        """
        ULTRA-FAST version of check_interaction_limits using MongoDB and aggressive caching.
        Maintains exact same logic and output format but targets <200ms execution time.
        
        This check is ONLY enabled in cloud edition. Open source has no limits.
        
        Args:
            interaction_type (str): either 'mini' or 'premium'
            memory_graph (Optional[MemoryGraph]): MongoDB connection for direct queries
            operation (Optional[MemoryOperationType]): The specific API operation being performed
            batch_size (Optional[int]): For batch operations, the number of items
            enable_agentic_graph (bool): For search operations, whether agentic graph is enabled
            enable_rank_results (bool): For search operations, whether result ranking is enabled
            api_key_id (Optional[str]): API Key ID for updating last_used_at timestamp
            organization_id (Optional[str]): Organization ID for tracking interaction at organization level
            namespace_id (Optional[str]): Namespace ID for tracking interaction at namespace level
            
        Returns:
            Optional[Tuple[Dict[str, Any], int, bool]]: Tuple containing (response_dict, status_code, is_error)
                                                or None if ok
        """
        # Import here to avoid circular dependency
        from config.features import get_features
        features = get_features()
        
        # Skip checks entirely in open source edition
        if not features.is_cloud:
            logger.debug("Open source edition detected - skipping fast interaction limits check")
            return None
        
        # Skip if subscription enforcement is disabled
        if not features.is_enabled("subscription_enforcement"):
            logger.debug("Subscription enforcement disabled - skipping fast interaction limits check")
            return None
        
        import time
        start_time = time.time()
        
        # Calculate actual interaction cost based on operation type
        if operation:
            from models.operation_types import get_operation_cost
            interaction_cost = get_operation_cost(
                operation=operation,
                batch_size=batch_size,
                enable_agentic_graph=enable_agentic_graph,
                enable_rank_results=enable_rank_results
            )
            logger.info(f"Operation {operation.value} cost: {interaction_cost} mini interactions")
            
            # Early return for zero-cost operations
            if interaction_cost == 0:
                logger.info(f"Zero-cost operation {operation.value} - skipping limits check")
                return None
        else:
            # Legacy mode - default to 1 interaction
            interaction_cost = 1
            logger.warning("No operation type provided - defaulting to 1 mini interaction")
        
        logger.info(f"check_interaction_limits_fast START - user: {self.id[:10]}, type: {interaction_type}, cost: {interaction_cost}")
        logger.info(f"memory_graph provided: {memory_graph is not None}")
        if memory_graph:
            logger.info(f"memory_graph.mongo_client: {memory_graph.mongo_client is not None}")
            if memory_graph.mongo_client:
                logger.info(f"MongoDB connection string ends with: ...{str(memory_graph.mongo_client.address)}")
        
        try:
            # Phase 1: Fast workspace lookup with caching (target: 20-50ms)
            phase1_start = time.time()
            workspace_data, subscription_data = await self._get_workspace_and_subscription_fast(memory_graph)
            phase1_time = (time.time() - phase1_start) * 1000
            logger.info(f"Phase 1 (workspace lookup) took: {phase1_time:.2f}ms")
            
            if not workspace_data or not subscription_data:
                logger.warning(f"Phase 1 failed - workspace_data: {workspace_data is not None}, subscription_data: {subscription_data is not None}")
                return {
                    "error": "No workspace found",
                    "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
                }, 403, True

            stripe_customer_id = subscription_data.get('stripeCustomerId')
            is_metered_billing_on = subscription_data.get('isMeteredBillingOn', False)
            
            logger.info(f"Fast check - stripe_customer_id: {stripe_customer_id}, metered: {is_metered_billing_on}")
            logger.info(f"workspace_data keys: {list(workspace_data.keys()) if workspace_data else 'None'}")
            logger.info(f"subscription_data keys: {list(subscription_data.keys()) if subscription_data else 'None'}")

            # Set current date for interaction tracking
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year

            # Fast path: metered billing enabled, skip tier/subscription checks
            if is_metered_billing_on:
                logger.info("Metered billing enabled - skipping tier/subscription checks")

                async def _safe_task(coro, label: str):
                    try:
                        await coro
                    except Exception as e:
                        logger.warning(f"{label} failed (non-critical): {e}")

                # Always update interaction counts; optionally defer to background
                operation_name = operation.value if operation else "unknown"
                interaction_coro = self._update_interaction_count_fast(
                    workspace_data.get('objectId'),
                    interaction_type,
                    current_month,
                    current_year,
                    subscription_data.get('objectId'),
                    workspace_data.get('company', {}).get('objectId') if workspace_data.get('company') else None,
                    memory_graph,
                    increment_by=interaction_cost,
                    operation_type=operation_name,
                    organization_id=organization_id,
                    namespace_id=namespace_id,
                    api_key_id=api_key_id
                )

                if defer_usage_tracking:
                    asyncio.create_task(_safe_task(interaction_coro, "Interaction update"))
                    if api_key_id:
                        asyncio.create_task(_safe_task(self._update_api_key_last_used(api_key_id), "API key update"))
                    logger.info("Deferred usage tracking for metered billing")
                    return None, 200, False

                # Synchronous update for metered billing
                interaction_result = await interaction_coro
                if isinstance(interaction_result, Exception):
                    logger.error(f"Interaction update failed: {interaction_result}", exc_info=True)
                    return {"error": "Unable to update usage record"}, 500, True
                if api_key_id:
                    await _safe_task(self._update_api_key_last_used(api_key_id), "API key update")
                return None, 200, False

            # Phase 2: Parallel operations (target: 50-100ms)
            # Run Stripe checks, interaction updates, and tier lookup in parallel
            # current_date/current_month/current_year already set above

            # Create all parallel tasks
            tasks = []
            
            # Task 1: Update interaction count (MongoDB or Parse)
            operation_name = operation.value if operation else "unknown"
            interaction_task = self._update_interaction_count_fast(
                workspace_data.get('objectId'),
                interaction_type,
                current_month,
                current_year,
                subscription_data.get('objectId'),
                workspace_data.get('company', {}).get('objectId') if workspace_data.get('company') else None,
                memory_graph,
                increment_by=interaction_cost,  # Increment by actual operation cost
                operation_type=operation_name,  # Track which operation consumed the interactions
                organization_id=organization_id,  # Track organization
                namespace_id=namespace_id,  # Track namespace
                api_key_id=api_key_id  # Track API key
            )
            tasks.append(interaction_task)
            
            # Task 2: Get customer tier (cached Stripe call)
            tier_task = self._get_customer_tier_fast(stripe_customer_id)
            tasks.append(tier_task)
            
            # Task 3: Check subscription status (lightweight Stripe call)
            subscription_task = self._check_subscription_status_fast(stripe_customer_id, subscription_data)
            tasks.append(subscription_task)
            
            # Task 4: Update API key last_used_at if api_key_id provided
            if api_key_id:
                api_key_update_task = self._update_api_key_last_used(api_key_id)
                tasks.append(api_key_update_task)
            
            # Execute all tasks in parallel
            phase2_start = time.time()
            logger.info(f"Phase 2 - Starting {len(tasks)} parallel tasks...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Unpack results based on whether we updated API key or not
            if api_key_id:
                interaction_result, customer_tier, subscription_status, api_key_update_result = results
                if isinstance(api_key_update_result, Exception):
                    logger.warning(f"API key update failed (non-critical): {api_key_update_result}")
            else:
                interaction_result, customer_tier, subscription_status = results
            phase2_time = (time.time() - phase2_start) * 1000
            logger.info(f"Phase 2 (parallel tasks) took: {phase2_time:.2f}ms")

            # Handle any exceptions from parallel tasks
            logger.info(f"Task results - interaction_result type: {type(interaction_result)}, customer_tier: {customer_tier}, subscription_status type: {type(subscription_status)}")
            
            if isinstance(interaction_result, Exception):
                logger.error(f"Interaction update failed: {interaction_result}", exc_info=True)
                return {"error": "Unable to update usage record"}, 500, True
                
            if isinstance(customer_tier, Exception):
                logger.warning(f"Customer tier lookup failed: {customer_tier}")
                customer_tier = 'free_trial'  # Fallback
                
            if isinstance(subscription_status, Exception):
                logger.warning(f"Subscription status check failed: {subscription_status}")
                subscription_status = {'is_trial': False, 'needs_attention': False}

            current_count, welcome_message = interaction_result
            
            # Phase 3: Background tasks and early returns (target: 10-20ms)
            # Send meter event as background task (don't wait for it)
            asyncio.create_task(self._send_meter_event_fast(
                interaction_type, stripe_customer_id, value=interaction_cost
            ))

            # Early return for metered billing
            if is_metered_billing_on:
                logger.info(f"Fast check completed in {(time.time() - start_time) * 1000:.2f}ms (metered billing)")
                if welcome_message:
                    return welcome_message, 200, False
                return None

            # Handle subscription issues
            if subscription_status.get('needs_attention'):
                return subscription_status.get('error_response', {
                    "error": "Subscription required",
                    "message": "Subscription requires attention"
                }), 403, True

            # Phase 4: Limit checking (target: 10-20ms)
            is_trial = subscription_status.get('is_trial', False)
            effective_tier = 'free_trial' if is_trial else customer_tier
            
            tier_config = features.get_tier_limits(effective_tier)
            
            if not tier_config:
                # No limits (unlimited tier or enterprise)
                logger.info(f"Fast check: No limits for tier {effective_tier}")
                if welcome_message:
                    return welcome_message, 200, False
                return None
            
            # Extract interaction limits
            mini_limit = tier_config.get('max_mini_interactions_per_month', float('inf'))
            premium_limit = tier_config.get('max_premium_interactions_per_month', float('inf'))
            
            # Map to format for compatibility
            tier_limits = {
                'mini': mini_limit if mini_limit != float('inf') else None,
                'premium': premium_limit if premium_limit != float('inf') else None
            }
            
            limit_value = tier_limits.get(interaction_type)
            
            if limit_value and current_count >= limit_value:
                # Generate appropriate error message based on tier
                error_message = self._generate_limit_error_message(
                    customer_tier, limit_value, interaction_type
                )
                
                total_time = (time.time() - start_time) * 1000
                logger.warning(f"Fast check completed in {total_time:.2f}ms (limit exceeded)")
                logger.warning(f"Operation cost: {interaction_cost} mini interactions, current_count: {current_count}, limit: {limit_value}")
                
                return {
                    "error": "Interaction limit reached, please go to https://dashboard.papr.ai to upgrade your plan to be able to use Papr.",
                    "message": error_message,
                    "current_count": current_count,
                    "limit": limit_value,
                    "operation_cost": interaction_cost,
                    "operation": operation.value if operation else "unknown",
                    "tier": customer_tier,
                    "is_trial": is_trial
                }, 403, True

            total_time = (time.time() - start_time) * 1000
            logger.info(f"Fast check completed in {total_time:.2f}ms (success)")
            
            # Return welcome message or success
            if welcome_message:
                return welcome_message, 200, False
            return None

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error(f"Fast check failed in {total_time:.2f}ms: {str(e)}")
            return {
                "error": "Subscription required",
                "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
            }, 403, True

    async def _get_workspace_and_subscription_fast(self, memory_graph=None):
        """Get workspace and subscription data with aggressive caching"""
        try:
            cache_key = f"workspace_sub_{self.id}"
            
            # Check cache first
            cached_data = workspace_subscription_cache.get(cache_key)
            if cached_data:
                logger.info(f"Workspace subscription cache HIT for user {self.id[:10]}...")
                return cached_data
            
            logger.info(f"Workspace subscription cache MISS for user {self.id[:10]}...")
            
            # Get fresh data
            result = None
            # Check if MongoDB is available, and if not, try to fix it
            mongo_available = False
            logger.info(f"DEBUG: memory_graph exists: {memory_graph is not None}")
            logger.info(f"DEBUG: memory_graph type: {type(memory_graph)}")
            
            # CRITICAL FIX: Use "is not None" instead of truthiness check
            # For some reason, bool(memory_graph) returns False even when memory_graph is not None
            if memory_graph is not None:
                logger.info(f"DEBUG: Entered if memory_graph block")
                logger.info(f"DEBUG: memory_graph.mongo_client before check: {memory_graph.mongo_client}")
                logger.info(f"DEBUG: memory_graph.db before check: {memory_graph.db}")
                
                if memory_graph.mongo_client:
                    mongo_available = True
                    logger.info("DEBUG: MongoDB client already available")
                else:
                    logger.warning("DEBUG: MongoDB client is None, attempting to fix...")
                    # Try to fix missing mongo_client by getting it from shared client
                    from services.mongo_client import get_mongo_db
                    shared_db = get_mongo_db()
                    logger.info(f"DEBUG: get_mongo_db() returned: {shared_db is not None}")
                    if shared_db is not None:
                        logger.info(f"DEBUG: Assigning shared_db.client to memory_graph.mongo_client")
                        memory_graph.mongo_client = shared_db.client
                        memory_graph.db = shared_db
                        logger.info(f"DEBUG: After assignment - mongo_client: {memory_graph.mongo_client is not None}, db: {memory_graph.db.name if memory_graph.db else None}")
                        mongo_available = True
                        logger.info(f"✅ Fixed missing mongo_client for MemoryGraph instance")
                    else:
                        logger.error(f"❌ Could not fix missing mongo_client - get_mongo_db() returned None")
                        # Let's try to understand why
                        import os
                        logger.error(f"DEBUG: MONGO_URI set: {bool(os.getenv('MONGO_URI'))}")
                        logger.error(f"DEBUG: DATABASE_URI set: {bool(os.getenv('DATABASE_URI'))}")
            else:
                logger.error("DEBUG: memory_graph is None!")

            logger.info(f"DEBUG: Final mongo_available: {mongo_available}")
            if mongo_available:
                # MongoDB path (fastest)
                logger.info(f"Attempting MongoDB workspace lookup for user {self.id[:10]}...")
                logger.info(f"MongoDB database: {memory_graph.db.name if hasattr(memory_graph.db, 'name') else 'unknown'}")
                result = await self._get_workspace_subscription_mongo(memory_graph.db)
                logger.info(f"MongoDB workspace lookup result: {result}")
            else:
                logger.info(f"MongoDB not available, using Parse Server fallback for user {self.id[:10]}...")
                logger.info(f"memory_graph: {memory_graph}, mongo_client: {memory_graph.mongo_client if memory_graph else 'None'}")
            
            # If MongoDB failed or not available, try Parse Server fallback
            if not result or not result[0] or not result[1]:
                logger.warning(f"MongoDB failed for user {self.id[:10]}, trying Parse Server fallback...")
                result = await self._get_workspace_subscription_parse()
                logger.info(f"Parse Server workspace lookup result: {result}")
            
            # Cache the result if we got valid data
            if result and result[0] and result[1]:
                workspace_subscription_cache.set(cache_key, result)
                logger.info(f"Cached workspace subscription data for user {self.id[:10]}...")
            else:
                logger.warning(f"Not caching result for user {self.id[:10]} - result: {result}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error getting workspace and subscription: {e}", exc_info=True)
            return None, None

    async def _get_workspace_subscription_mongo(self, db):
        """Get workspace and subscription data directly from MongoDB"""
        try:
            logger.info(f"DEBUG MONGO: _get_workspace_subscription_mongo called")
            logger.info(f"DEBUG MONGO: db={db}, db.name={db.name if hasattr(db, 'name') else 'N/A'}")
            logger.info(f"DEBUG MONGO: self.id={self.id}")
            logger.info(f"MongoDB workspace lookup starting for user: {self.id[:10]}...")
            
            # Get user's selected workspace follower
            logger.info(f"DEBUG MONGO: About to execute find_one on _User collection")
            user_doc = db["_User"].find_one(
                {"_id": self.id},
                {"_p_isSelectedWorkspaceFollower": 1}
            )
            
            logger.info(f"DEBUG MONGO: Query executed successfully")
            logger.info(f"User doc result: {user_doc}")
            
            if not user_doc or not user_doc.get('_p_isSelectedWorkspaceFollower'):
                logger.warning(f"No selected workspace follower found for user {self.id[:10]}...")
                return None, None
            
            # Extract follower ID from the Parse pointer structure
            follower_pointer = user_doc['_p_isSelectedWorkspaceFollower']
            logger.info(f"Follower pointer structure: {follower_pointer}")
            
            # Handle different possible structures:
            # 1. Parse pointer format: {"__type": "Pointer", "className": "workspace_follower", "objectId": "..."}
            # 2. Simple objectId reference: "workspace_follower$objectId"
            # 3. Direct objectId: "objectId"
            
            follower_id = None
            if isinstance(follower_pointer, dict):
                if follower_pointer.get('__type') == 'Pointer' and follower_pointer.get('className') == 'workspace_follower':
                    follower_id = follower_pointer.get('objectId')
                elif 'objectId' in follower_pointer:
                    follower_id = follower_pointer['objectId']
            elif isinstance(follower_pointer, str):
                if follower_pointer.startswith('workspace_follower$'):
                    follower_id = follower_pointer[19:]  # Remove "workspace_follower$" prefix
                else:
                    follower_id = follower_pointer
            
            if not follower_id:
                logger.warning(f"Could not extract follower ID from pointer: {follower_pointer}")
                return None, None
                
            logger.info(f"Extracted follower_id: {follower_id}")
            
            # Get workspace follower with workspace reference
            follower_doc = db["workspace_follower"].find_one(
                {"_id": follower_id},
                {"_p_workspace": 1}
            )
            
            logger.info(f"Follower doc result: {follower_doc}")
            
            if not follower_doc or not follower_doc.get('_p_workspace'):
                logger.warning(f"No workspace reference found for follower {follower_id}")
                return None, None
                
            # Extract workspace ID from Parse pointer
            workspace_pointer = follower_doc['_p_workspace']
            logger.info(f"Workspace pointer: {workspace_pointer}")
            
            if workspace_pointer.startswith('WorkSpace$'):
                workspace_id = workspace_pointer[10:]  # Remove "WorkSpace$" prefix
                logger.info(f"Extracted workspace_id: {workspace_id}")
            else:
                logger.warning(f"Invalid workspace pointer format: {workspace_pointer}")
                return None, None
            
            # Get workspace with subscription reference
            workspace_doc = db["WorkSpace"].find_one(
                {"_id": workspace_id},
                {"_id": 1, "_p_subscription": 1, "_p_company": 1}
            )
            
            logger.info(f"Workspace doc result: {workspace_doc}")
            
            if not workspace_doc:
                logger.warning(f"No workspace found with id: {workspace_id}")
                return None, None
                
            # Get subscription data
            subscription_pointer = workspace_doc.get('_p_subscription')
            logger.info(f"Subscription pointer: {subscription_pointer}")
            
            if not subscription_pointer or not subscription_pointer.startswith('Subscription$'):
                logger.warning(f"No or invalid subscription pointer: {subscription_pointer}")
                return None, None
                
            subscription_id = subscription_pointer[13:]  # Remove "Subscription$" prefix
            logger.info(f"Extracted subscription_id: {subscription_id}")
            
            subscription_doc = db["Subscription"].find_one(
                {"_id": subscription_id},
                {"stripeCustomerId": 1, "isMeteredBillingOn": 1, "status": 1, "tier": 1, "createdAt": 1}
            )
            
            logger.info(f"Subscription doc result: {subscription_doc}")
            
            if not subscription_doc:
                logger.warning(f"No subscription found with id: {subscription_id}")
                return None, None
                
            # Format data similar to Parse Server response
            workspace_data = {
                "objectId": workspace_id,
                "company": {"objectId": workspace_doc.get('_p_company', '').replace('Company$', '')} if workspace_doc.get('_p_company') else None
            }
            
            subscription_data = {
                "objectId": subscription_id,
                "stripeCustomerId": subscription_doc.get('stripeCustomerId'),
                "isMeteredBillingOn": subscription_doc.get('isMeteredBillingOn', False),
                "status": subscription_doc.get('status'),
                "tier": subscription_doc.get('tier'),
                "createdAt": subscription_doc.get('createdAt')
            }
            
            logger.info(f"MongoDB lookup successful - workspace: {workspace_data}, subscription: {subscription_data}")
            return workspace_data, subscription_data
            
        except Exception as e:
            logger.error(f"MongoDB workspace lookup failed: {e}", exc_info=True)
            return None, None

    async def _get_workspace_subscription_parse(self):
        """Fallback to Parse Server for workspace and subscription data"""
        try:
            logger.info(f"Parse Server workspace lookup starting for user: {self.id[:10]}...")
            
            # Get workspace follower ID
            selected_follower_id = await self.get_selected_workspace_follower()
            logger.info(f"Selected follower ID from get_selected_workspace_follower: {selected_follower_id}")
            
            if not selected_follower_id:
                logger.warning(f"No selected workspace follower ID returned for user {self.id[:10]}")
                return None, None

            # Get workspace follower with includes
            workspace_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower/{selected_follower_id}"
            params = {"include": "workspace,workspace.subscription,workspace.company"}
            
            logger.info(f"Making Parse Server request to: {workspace_url}")
            logger.info(f"Request params: {params}")
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(workspace_url, headers=HEADERS, params=params)
                logger.info(f"Parse Server response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.warning(f"Parse Server returned non-200 status: {response.status_code}")
                    logger.warning(f"Response text: {response.text}")
                    return None, None
                    
                data = response.json()
                logger.info(f"Parse Server response data: {data}")
                
                workspace = data.get('workspace')
                logger.info(f"Extracted workspace: {workspace}")
                
                if not workspace:
                    logger.warning(f"No workspace found in Parse Server response")
                    return None, None
                    
                subscription = workspace.get('subscription')
                logger.info(f"Extracted subscription: {subscription}")
                
                if not subscription:
                    logger.warning(f"No subscription found in workspace data")
                    return None, None
                    
                logger.info(f"Parse Server lookup successful - workspace: {workspace}, subscription: {subscription}")
                return workspace, subscription
                
        except Exception as e:
            logger.error(f"Parse workspace lookup failed: {e}", exc_info=True)
            return None, None

    async def _update_interaction_count_fast(self, workspace_id, interaction_type, current_month, current_year, subscription_id, company_id, memory_graph, increment_by=1, operation_type="unknown", organization_id=None, namespace_id=None, api_key_id=None):
        """Update interaction count with MongoDB optimization"""
        try:
            logger.info(f"_update_interaction_count_fast - workspace_id: {workspace_id}, type: {interaction_type}, increment_by: {increment_by}, operation: {operation_type}, org_id: {organization_id}, ns_id: {namespace_id}, api_key_id: {api_key_id}")
            logger.info(f"MongoDB available: {memory_graph and memory_graph.mongo_client is not None}")
            
            if memory_graph and memory_graph.mongo_client:
                logger.info(f"Using MongoDB path for interaction update")
                try:
                    return await self._update_interaction_count_mongo(
                        memory_graph.db, workspace_id, interaction_type, current_month, current_year, subscription_id, company_id, increment_by, operation_type, organization_id, namespace_id, api_key_id
                    )
                except Exception as mongo_error:
                    logger.error(f"MongoDB path failed, falling back to Parse Server: {mongo_error}")
                    return await self._update_interaction_count_parse(
                        workspace_id, interaction_type, current_month, current_year, subscription_id, company_id, increment_by, operation_type, organization_id, namespace_id, api_key_id
                    )
            else:
                logger.info(f"Using Parse Server path for interaction update")
                return await self._update_interaction_count_parse(
                    workspace_id, interaction_type, current_month, current_year, subscription_id, company_id, increment_by, operation_type, organization_id, namespace_id, api_key_id
                )
        except Exception as e:
            logger.error(f"Error updating interaction count: {e}", exc_info=True)
            # Final fallback to Parse Server
            logger.info("Final fallback to Parse Server for interaction update")
            return await self._update_interaction_count_parse(
                workspace_id, interaction_type, current_month, current_year, subscription_id, company_id, increment_by, operation_type, organization_id, namespace_id, api_key_id
            )

    async def _update_interaction_count_mongo(self, db, workspace_id, interaction_type, current_month, current_year, subscription_id, company_id, increment_by=1, operation_type="unknown", organization_id=None, namespace_id=None, api_key_id=None):
        """Update interaction count using MongoDB atomic operations with Parse Server compatibility"""
        try:
            logger.info(f"_update_interaction_count_mongo START - db: {type(db)}")
            logger.info(f"Parameters - workspace_id: {workspace_id}, user_id: {self.id}, type: {interaction_type}, increment_by: {increment_by}, operation: {operation_type}, org_id: {organization_id}, ns_id: {namespace_id}, api_key_id: {api_key_id}")
            
            # Use MongoDB's findOneAndUpdate with upsert for atomic increment
            filter_query = {
                "_p_user": f"_User${self.id}",
                "_p_workspace": f"WorkSpace${workspace_id}",
                "type": interaction_type,
                "month": current_month,
                "year": current_year
            }
            logger.info(f"MongoDB filter query: {filter_query}")
            
            # First, try to find existing document
            existing_doc = db["Interaction"].find_one(filter_query)
            
            if existing_doc:
                # Update existing document
                logger.info(f"Found existing interaction document: {existing_doc.get('_id')}")
                current_count = existing_doc.get('count', 0) + increment_by
                
                # Build update data
                update_data = {
                    "count": current_count,
                    "operation_type": operation_type,  # Track which operation consumed interactions
                    "_updated_at": datetime.now()
                }
                
                # Add pointers if provided
                if organization_id:
                    update_data["_p_organization"] = f"Organization${organization_id}"
                if namespace_id:
                    update_data["_p_namespace"] = f"Namespace${namespace_id}"
                if api_key_id:
                    update_data["_p_apiKey"] = f"APIKey${api_key_id}"
                
                update_result = db["Interaction"].update_one(
                    {"_id": existing_doc["_id"]},
                    {"$set": update_data}
                )
                
                if update_result.modified_count > 0:
                    logger.info(f"MongoDB interaction count updated to: {current_count} (incremented by {increment_by})")
                    return current_count, None
                else:
                    logger.error("Failed to update existing interaction document")
                    raise Exception("Failed to update existing interaction document")
            else:
                # Document doesn't exist - fall back to Parse Server for creation
                logger.info("No existing interaction document found - falling back to Parse Server for proper creation")
                return await self._update_interaction_count_parse(
                    workspace_id, interaction_type, current_month, current_year, subscription_id, company_id, increment_by, operation_type, organization_id, namespace_id, api_key_id
                )
            
        except Exception as e:
            logger.error(f"MongoDB interaction update failed: {e}", exc_info=True)
            logger.info("Falling back to Parse Server for interaction update")
            # Fall back to Parse Server if MongoDB fails
            return await self._update_interaction_count_parse(
                workspace_id, interaction_type, current_month, current_year, subscription_id, company_id, increment_by, operation_type, organization_id, namespace_id, api_key_id
            )

    async def _update_interaction_count_parse(self, workspace_id, interaction_type, current_month, current_year, subscription_id, company_id, increment_by=1, operation_type="unknown", organization_id=None, namespace_id=None, api_key_id=None):
        """Fallback Parse Server interaction count update"""
        try:
            interaction_url = f"{PARSE_SERVER_URL}/parse/classes/Interaction"
            query = {
                "where": json.dumps({
                    "user": {"__type": "Pointer", "className": "_User", "objectId": self.id},
                    "workspace": {"__type": "Pointer", "className": "WorkSpace", "objectId": workspace_id},
                    "type": interaction_type,
                    "month": current_month,
                    "year": current_year
                })
            }
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(interaction_url, headers=HEADERS, params=query)
                if response.status_code != 200:
                    raise Exception(f"Failed to get interaction: {response.text}")
                
                interactions = response.json().get('results', [])
                logger.info(f"Parse interaction query returned {len(interactions)} results: {interactions}")
                
                if not interactions:
                    # Create new interaction
                    logger.info(f"Creating new interaction for user {self.id}, workspace {workspace_id}, increment_by: {increment_by}, operation: {operation_type}, org_id: {organization_id}, ns_id: {namespace_id}, api_key_id: {api_key_id}")
                    new_interaction = {
                        "workspace": {"__type": "Pointer", "className": "WorkSpace", "objectId": workspace_id},
                        "user": {"__type": "Pointer", "className": "_User", "objectId": self.id},
                        "type": interaction_type,
                        "month": current_month,
                        "year": current_year,
                        "count": increment_by,  # Use increment_by for initial count
                        "operation_type": operation_type  # Track which operation consumed interactions
                    }
                    
                    if company_id:
                        new_interaction["company"] = {"__type": "Pointer", "className": "Company", "objectId": company_id}
                    if subscription_id:
                        new_interaction["subscription"] = {"__type": "Pointer", "className": "Subscription", "objectId": subscription_id}
                    if organization_id:
                        new_interaction["organization"] = {"__type": "Pointer", "className": "Organization", "objectId": organization_id}
                    if namespace_id:
                        new_interaction["namespace"] = {"__type": "Pointer", "className": "Namespace", "objectId": namespace_id}
                    if api_key_id:
                        new_interaction["apiKey"] = {"__type": "Pointer", "className": "APIKey", "objectId": api_key_id}
                    
                    logger.info(f"Creating interaction with data: {new_interaction}")
                    create_response = await client.post(interaction_url, headers=HEADERS, json=new_interaction)
                    logger.info(f"Create response status: {create_response.status_code}, body: {create_response.text}")
                    
                    if create_response.status_code != 201:
                        raise Exception(f"Failed to create interaction: {create_response.text}")
                    
                    return increment_by, None
                else:
                    # Update existing interaction
                    interaction = interactions[0]
                    current_count = interaction.get('count', 0) + increment_by
                    logger.info(f"Updating existing interaction {interaction['objectId']} from count {interaction.get('count', 0)} to {current_count} (increment_by: {increment_by}), operation: {operation_type}, org_id: {organization_id}, ns_id: {namespace_id}, api_key_id: {api_key_id}")
                    
                    update_url = f"{interaction_url}/{interaction['objectId']}"
                    update_data = {
                        "count": current_count,
                        "operation_type": operation_type  # Track which operation consumed interactions
                    }
                    
                    # Add pointers if provided
                    if organization_id:
                        update_data["organization"] = {"__type": "Pointer", "className": "Organization", "objectId": organization_id}
                    if namespace_id:
                        update_data["namespace"] = {"__type": "Pointer", "className": "Namespace", "objectId": namespace_id}
                    if api_key_id:
                        update_data["apiKey"] = {"__type": "Pointer", "className": "APIKey", "objectId": api_key_id}
                    
                    update_response = await client.put(update_url, headers=HEADERS, json=update_data)
                    logger.info(f"Update response status: {update_response.status_code}, body: {update_response.text}")
                    
                    if update_response.status_code != 200:
                        raise Exception(f"Failed to update interaction: {update_response.text}")
                    
                    return current_count, None
                    
        except Exception as e:
            logger.error(f"Parse interaction update failed: {e}")
            raise

    async def _update_api_key_last_used(self, api_key_id: str) -> bool:
        """Update API key's last_used_at timestamp"""
        try:
            logger.info(f"Updating API key last_used_at for key ID: {api_key_id}")
            async with httpx.AsyncClient() as client:
                apikey_update_url = f"{PARSE_SERVER_URL}/parse/classes/APIKey/{api_key_id}"
                apikey_update_data = {
                    "last_used_at": {"__type": "Date", "iso": datetime.now(timezone.utc).isoformat()}
                }
                
                HEADERS = {
                    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                    "X-Parse-Master-Key": PARSE_MASTER_KEY,
                    "Content-Type": "application/json"
                }
                
                apikey_update_response = await client.put(apikey_update_url, headers=HEADERS, json=apikey_update_data)
                
                if apikey_update_response.status_code != 200:
                    logger.error(f"Failed to update API key last_used_at: {apikey_update_response.text}")
                    return False
                
                logger.info(f"Successfully updated API key last_used_at for key ID: {api_key_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating API key last_used_at: {e}", exc_info=True)
            return False

    async def _get_customer_tier_fast(self, stripe_customer_id):
        """Get customer tier with caching"""
        try:
            cache_key = f"customer_tier_{stripe_customer_id}"
            
            # Check cache first
            cached_tier = customer_tier_cache.get(cache_key)
            if cached_tier:
                logger.info(f"Customer tier cache HIT for {stripe_customer_id[:10]}...")
                return cached_tier
            
            logger.info(f"Customer tier cache MISS for {stripe_customer_id[:10]}...")
            
            # Get tier from Stripe
            tier = await stripe_service.get_customer_tier(stripe_customer_id)
            
            # Cache the result
            customer_tier_cache.set(cache_key, tier)
            logger.info(f"Cached customer tier '{tier}' for {stripe_customer_id[:10]}...")
            
            return tier
        except Exception as e:
            logger.warning(f"Customer tier lookup failed: {e}")
            return 'free_trial'  # Safe fallback

    async def _check_subscription_status_fast(self, stripe_customer_id, subscription_data):
        """Lightweight subscription status check"""
        try:
            # Get minimal subscription info from Stripe
            subscriptions = await asyncio.to_thread(
                stripe.Subscription.list,
                customer=stripe_customer_id,
                limit=1
            )
            
            if not subscriptions.data:
                # No subscription - check if we need to create trial
                return {
                    'is_trial': False,
                    'needs_attention': True,
                    'error_response': {
                        "error": "No active subscription",
                        "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
                    }
                }
            
            subscription = subscriptions.data[0]
            status = subscription.status
            
            # Check for problematic statuses
            problematic_statuses = ['canceled', 'incomplete', 'incomplete_expired', 'unpaid', 'paused', 'past_due']
            if status in problematic_statuses:
                status_messages = {
                    'canceled': "Your subscription has been canceled",
                    'incomplete': "Your subscription setup was not completed",
                    'incomplete_expired': "Your initial subscription setup was not completed",
                    'unpaid': "Your subscription has unpaid invoices",
                    'paused': "Your subscription is paused",
                    'past_due': "Your subscription is past due"
                }
                
                return {
                    'is_trial': False,
                    'needs_attention': True,
                    'error_response': {
                        "error": "Subscription required",
                        "message": f"{status_messages.get(status, 'Subscription issue')}. Please visit https://dashboard.papr.ai to reactivate your subscription and continue using Papr.",
                        "subscription_status": status
                    }
                }
            
            # Active or trialing
            return {
                'is_trial': status == 'trialing',
                'needs_attention': False
            }
            
        except Exception as e:
            logger.warning(f"Subscription status check failed: {e}")
            # Fallback based on subscription data
            created_at = subscription_data.get('createdAt')
            if created_at:
                try:
                    created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                    is_trial = (datetime.now() - created_date).days <= 7
                except:
                    is_trial = False
            else:
                is_trial = False
                
            return {'is_trial': is_trial, 'needs_attention': False}

    async def _send_meter_event_fast(self, interaction_type, stripe_customer_id, value=1):
        """Send meter event as background task (non-blocking)"""
        try:
            await stripe_service.send_meter_event(
                event_name=f"papr_{interaction_type}_interactions",
                value=value,
                stripe_customer_id=stripe_customer_id,
            )
        except Exception as e:
            logger.warning(f"Background meter event failed (non-critical): {e}")

    def _generate_limit_error_message(self, customer_tier: str, limit: int, interaction_type: str) -> str:
        """Generate appropriate error message based on customer tier and limit"""
        tier_name = customer_tier.replace('_', ' ').title()
        
        if customer_tier in ['developer', 'free_trial']:
            return (
                f"You've reached the {limit:,} {interaction_type} interactions limit for your {tier_name} plan. "
                "To continue, upgrade to Starter ($100/mo) or Growth ($500/mo) plan.\n"
                "Visit https://dashboard.papr.ai to manage your subscription."
            )
        elif customer_tier == 'starter':
            return (
                f"You've reached the {limit:,} {interaction_type} interactions limit for your Starter plan. "
                "To continue, you can either:\n"
                "1. Enable metered billing in your current plan, or\n"
                "2. Upgrade to Growth plan for higher limits\n"
                "Visit https://dashboard.papr.ai to manage your subscription."
            )
        elif customer_tier == 'growth':
            return (
                f"You've reached the {limit:,} {interaction_type} interactions limit for your Growth plan. "
                "To continue, you can either:\n"
                "1. Enable metered billing in your current plan, or\n"
                "2. Contact us for Enterprise plan with unlimited resources\n"
                "Visit https://dashboard.papr.ai to manage your subscription."
            )
        elif customer_tier == 'pro':
            return (
                f"You've reached the {limit:,} {interaction_type} interactions limit for your Pro plan. "
                "To continue, enable metered billing or upgrade to a higher tier.\n"
                "Visit https://app.papr.ai to manage your subscription."
            )
        elif customer_tier == 'business_plus':
            return (
                f"You've reached the {limit:,} {interaction_type} interactions limit for your Business Plus plan. "
                "To continue, enable metered billing or contact us for Enterprise.\n"
                "Visit https://app.papr.ai to manage your subscription."
            )
        elif customer_tier == 'enterprise':
            return (
                f"You've reached the {limit:,} {interaction_type} interactions limit for your Enterprise plan. "
                "To continue, enable metered billing or contact us for Enterprise.\n"
                "Visit https://dashboard.papr.ai to manage your subscription."
            )
        else:
            return (
                f"You've reached the {limit:,} {interaction_type} interactions limit for your {tier_name} plan. "
                "Please visit https://dashboard.papr.ai to upgrade your subscription."
            )

    @staticmethod
    async def _fetch_is_qwen_route(user_id: str, httpx_client: "httpx.AsyncClient") -> Optional[bool]:
        """Fetch isQwenRoute for a user_id with error handling"""
        try:
            url = f"{PARSE_SERVER_URL}/parse/classes/_User/{user_id}"
            headers = get_parse_headers()
            resp = await httpx_client.get(url, headers=headers)
            resp.raise_for_status()
            user_obj = resp.json()
            is_qwen_route = user_obj.get("isQwenRoute", None)
            logger.info(f"Fetched isQwenRoute for user_id {user_id}: {is_qwen_route}")
            return is_qwen_route
        except Exception as e:
            logger.warning(f"Failed to fetch isQwenRoute for user_id {user_id}: {e}")
            return None

    @staticmethod
    async def resolve_end_user_id(
        developer_id: str,
        metadata: Optional["MemoryMetadata"],
        authenticated_user_id: str,
        httpx_client: "httpx.AsyncClient",
        is_update: bool = False,
        api_key: str = None,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None
    ) -> Tuple["MemoryMetadata", str, Optional[bool]]:
        """
        Resolves the end_user_id for memory operations and ensures ACL fields are set correctly.
        Returns a tuple: (possibly updated metadata, resolved internal user_id (Parse _User objectId), isQwenRoute flag).
        
        Args:
            organization_id: Optional organization ID to avoid fetching it (reduces latency)
            namespace_id: Optional namespace ID to avoid fetching it (reduces latency)
        """

        if not metadata:
            return None, authenticated_user_id, None

        user_id = getattr(metadata, 'user_id', None)
        external_user_id = getattr(metadata, 'external_user_id', None)
        ext_read = getattr(metadata, 'external_user_read_access', None) or []
        ext_write = getattr(metadata, 'external_user_write_access', None) or []
        user_read = getattr(metadata, 'user_read_access', None) or []
        user_write = getattr(metadata, 'user_write_access', None) or []
        
        logger.info(f"resolve_end_user_id - Input metadata: user_id={user_id}, external_user_id={external_user_id}")
        logger.info(f"resolve_end_user_id - Input metadata: ext_read={ext_read}, ext_write={ext_write}")
        logger.info(f"resolve_end_user_id - Input metadata: user_read={user_read}, user_write={user_write}")

        # Set multi-tenant context in metadata if provided (critical for tenant isolation)
        if organization_id:
            metadata.organization_id = organization_id
            logger.info(f"resolve_end_user_id - Set organization_id in metadata: {organization_id}")
        if namespace_id:
            metadata.namespace_id = namespace_id
            logger.info(f"resolve_end_user_id - Set namespace_id in metadata: {namespace_id}")

        if is_update:
            # On update, never change user_id (creator)
            # If user_read_access or user_write_access are present, trust them and do not resolve
            if user_read or user_write:
                logger.info("Update: user_read_access or user_write_access present, skipping resolution.")
                return metadata, user_id or authenticated_user_id, None
            # If only external_user_read_access/write_access are present, resolve them
            if ext_read or ext_write:
                user_service = User(developer_id)
                updated_metadata, _, _ = await user_service.resolve_external_user_ids_to_internal(
                    developer_id=developer_id,
                    metadata=metadata,
                    httpx_client=httpx_client,
                    x_api_key=api_key,
                    organization_id=organization_id,
                    namespace_id=namespace_id
                )
                metadata = updated_metadata
                # After resolution, user_read_access/user_write_access should be set
                return metadata, getattr(metadata, 'user_id', None) or authenticated_user_id, None
            # Fallback: nothing to resolve, return creator
            return metadata, user_id or authenticated_user_id, None

        # --- ADD logic (not update) ---
        # Both user_id and external_user_id set
        if user_id and external_user_id:
            logger.warning("Both user_id and external_user_id are set in metadata. Preferring user_id for ACL and end_user_id.")
            metadata.user_read_access = [user_id]
            metadata.user_write_access = [user_id]
            if not getattr(metadata, 'external_user_read_access', None):
                metadata.external_user_read_access = [external_user_id]
            if not getattr(metadata, 'external_user_write_access', None):
                metadata.external_user_write_access = [external_user_id]
            return metadata, user_id, None

        # Only user_id set
        if user_id:
            logger.info(f"resolve_end_user_id - user_id is set: {user_id}, setting ACL fields")
            metadata.user_read_access = [user_id]
            metadata.user_write_access = [user_id]
            # Fetch isQwenRoute from _User (optimized single call)
            is_qwen_route = await User._fetch_is_qwen_route(user_id, httpx_client)
            logger.info(f"resolve_end_user_id - Returning metadata with user_id: {user_id}")
            return metadata, user_id, is_qwen_route

        # Only external_user_id set
        if external_user_id:
            user_service = User(developer_id)
            updated_metadata, id_map, is_qwen_route = await user_service.resolve_external_user_ids_to_internal(
                developer_id=developer_id,
                metadata=metadata,
                httpx_client=httpx_client,
                x_api_key=api_key,
                fetch_is_qwen_route=True,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            metadata = updated_metadata
            logger.info(f"resolved metadata after resolve_external_user_ids_to_internal: {metadata}")
            internal_user_id = getattr(metadata, 'user_id', None)
            if internal_user_id:
                metadata.user_read_access = [internal_user_id]
                metadata.user_write_access = [internal_user_id]
                if not getattr(metadata, 'external_user_read_access', None):
                    metadata.external_user_read_access = [external_user_id]
                if not getattr(metadata, 'external_user_write_access', None):
                    metadata.external_user_write_access = [external_user_id]
                metadata.external_user_id = external_user_id
                return metadata, internal_user_id, is_qwen_route

        # Neither set: fallback to authenticated_user_id
        metadata.user_id = authenticated_user_id
        metadata.user_read_access = [authenticated_user_id]
        metadata.user_write_access = [authenticated_user_id]
        # Fetch isQwenRoute from _User (optimized single call)
        is_qwen_route = await User._fetch_is_qwen_route(authenticated_user_id, httpx_client)
        return metadata, authenticated_user_id, is_qwen_route

    async def resolve_external_user_ids_to_internal(
        self,
        developer_id: str,
        metadata: MemoryMetadata,
        httpx_client: httpx.AsyncClient,
        x_api_key: str,
        fetch_is_qwen_route: bool = False,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None
    ) -> Tuple[MemoryMetadata, dict, Optional[bool]]:
        """
        Resolves external_user_id and external_user_read_access/write_access to internal user IDs and DeveloperUser objectIds.
        Returns a tuple: (possibly updated metadata, id_map, isQwenRoute flag if fetch_is_qwen_route is True).
        
        Args:
            organization_id: Optional organization ID to avoid fetching it (reduces latency)
            namespace_id: Optional namespace ID to avoid fetching it (reduces latency)
        """
        external_ids = set()
        if metadata.external_user_id:
            external_ids.add(metadata.external_user_id)
        for eid in (metadata.external_user_read_access or []):
            external_ids.add(eid)
        for eid in (metadata.external_user_write_access or []):
            external_ids.add(eid)
        external_ids = list(external_ids)

        # Get developer's organization and namespace for multi-tenant isolation
        # Only fetch if not provided (for backward compatibility and latency reduction)
        developer_org_id = organization_id
        developer_namespace_id = namespace_id
        
        if not developer_org_id or not developer_namespace_id:
            try:
                developer_url = f"{PARSE_SERVER_URL}/parse/classes/_User/{developer_id}"
                headers = {
                    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                    "X-Parse-Master-Key": PARSE_MASTER_KEY,
                    "Content-Type": "application/json"
                }
                dev_resp = await httpx_client.get(developer_url, headers=headers)
                dev_resp.raise_for_status()
                developer_data = dev_resp.json()
                if not developer_org_id:
                    developer_org_id = developer_data.get("organization_id")
                
                # Get namespace from organization's default_namespace
                if developer_org_id and not developer_namespace_id:
                    org_url = f"{PARSE_SERVER_URL}/parse/classes/Organization/{developer_org_id}"
                    org_resp = await httpx_client.get(org_url, headers=headers)
                    org_resp.raise_for_status()
                    org_data = org_resp.json()
                    # Try multiple ways to get namespace
                    developer_namespace_id = org_data.get("default_namespace_id")
                    if not developer_namespace_id and org_data.get("default_namespace"):
                        developer_namespace_id = org_data["default_namespace"].get("objectId")
                    
                    logger.info(f"🔍 Developer {developer_id} -> Organization {developer_org_id} -> Namespace {developer_namespace_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch developer organization/namespace: {e}")

        id_map: Dict[str, dict] = {}
        missing_external_ids = []
        if external_ids:
            url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser"
            
            # Build where query with namespace for multi-tenant isolation
            where_query = {
                "developer": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": developer_id
                },
                "external_id": {"$in": external_ids}
            }
            
            # Add namespace filter for multi-tenant isolation
            if developer_namespace_id:
                where_query["namespace"] = {
                    "__type": "Pointer",
                    "className": "Namespace",
                    "objectId": developer_namespace_id
                }
                logger.info(f"🔍 Added namespace filter to query: {developer_namespace_id}")
            
            params = {
                "where": json.dumps(where_query),
                "limit": len(external_ids),
                "include": "user"
            }
            headers = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-Master-Key": PARSE_MASTER_KEY,
                "Content-Type": "application/json"
            }
            logger.info(f"🔍 DEBUG: Querying DeveloperUser with developer_id={developer_id}, external_ids={external_ids}")
            logger.info(f"🔍 DEBUG: Query URL: {url}")
            logger.info(f"🔍 DEBUG: Query params: {params}")
            resp = await httpx_client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            response_data = resp.json()
            results = response_data.get("results", [])
            logger.info(f"🔍 DEBUG: DeveloperUser query returned {len(results)} results")
            for r in results:
                logger.info(f"🔍 DEBUG: DeveloperUser record: objectId={r.get('objectId')}, external_id={r.get('external_id')}, user={r.get('user')}")
                if r.get("user") and r.get("objectId"):
                    id_map[r["external_id"]] = {
                        "user_objectId": r["user"]["objectId"],
                        "developerUser_objectId": r["objectId"],
                        "user_obj": r["user"]  # Include the full user object
                    }
                    logger.info(f"🔍 DEBUG: Added to id_map: {r['external_id']} -> developerUser_objectId={r['objectId']}")
                else:
                    logger.warning(f"🔍 DEBUG: Skipped record - user={r.get('user')}, objectId={r.get('objectId')}")
            missing_external_ids = [eid for eid in external_ids if eid not in id_map]
            logger.info(f"🔍 DEBUG: id_map after query: {id_map}")
            logger.info(f"🔍 DEBUG: missing_external_ids: {missing_external_ids}")

        # If any missing, create them in batch using batch_create_users_core
        if missing_external_ids:
            create_requests = [CreateUserRequest(external_id=eid) for eid in missing_external_ids]
            # Pass organization and namespace info for DeveloperUser creation
            created_users = await batch_create_users_core(
                create_requests, 
                developer_id, 
                x_api_key,
                organization_id=developer_org_id,
                namespace_id=developer_namespace_id
            )
            for user_resp in created_users:
                if user_resp.user_id and user_resp.external_id:
                    # After creation, fetch DeveloperUser objectId
                    url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser"
                    
                    # Build where query with namespace for multi-tenant isolation
                    fetch_where_query = {
                        "developer": {
                            "__type": "Pointer",
                            "className": "_User",
                            "objectId": developer_id
                        },
                        "external_id": user_resp.external_id
                    }
                    
                    # Add namespace filter for multi-tenant isolation
                    if developer_namespace_id:
                        fetch_where_query["namespace"] = {
                            "__type": "Pointer",
                            "className": "Namespace",
                            "objectId": developer_namespace_id
                        }
                    
                    params = {
                        "where": json.dumps(fetch_where_query),
                        "limit": 1,
                        "include": "user"
                    }
                    resp = await httpx_client.get(url, headers=headers, params=params)
                    resp.raise_for_status()
                    results = resp.json().get("results", [])
                    if results:
                        dev_user_obj = results[0]
                        id_map[user_resp.external_id] = {
                            "user_objectId": user_resp.user_id,
                            "developerUser_objectId": dev_user_obj["objectId"],
                            "user_obj": dev_user_obj.get("user", {})  # Include the user object if available
                        }

        # Set user_id (internal) and external_user_id (external string) in metadata
        is_qwen_route = None
        if metadata.external_user_id and metadata.external_user_id in id_map:
            resolved_user_id = id_map[metadata.external_user_id]["user_objectId"]
            metadata.user_id = resolved_user_id
            metadata.external_user_id = metadata.external_user_id  # keep as string
            if fetch_is_qwen_route and "user_obj" in id_map[metadata.external_user_id]:
                # Use the included user object instead of making another API call
                user_obj = id_map[metadata.external_user_id]["user_obj"]
                is_qwen_route = user_obj.get("isQwenRoute", None)
                logger.info(f"Got isQwenRoute from included user object for resolve_external_user_ids_to_internal: {resolved_user_id}: {is_qwen_route}")
        # Do NOT add user_objectId or developerUser_objectId to metadata

        # Resolve all external_user_read_access to internal user ObjectIds
        resolved_user_read_access = []
        for eid in (metadata.external_user_read_access or []):
            if eid in id_map and id_map[eid]["user_objectId"]:
                resolved_user_read_access.append(id_map[eid]["user_objectId"])
        # Deduplicate and remove None
        metadata.user_read_access = list({uid for uid in resolved_user_read_access if uid})

        # Resolve all external_user_write_access to internal user ObjectIds
        resolved_user_write_access = []
        for eid in (metadata.external_user_write_access or []):
            if eid in id_map and id_map[eid]["user_objectId"]:
                resolved_user_write_access.append(id_map[eid]["user_objectId"])
        metadata.user_write_access = list({uid for uid in resolved_user_write_access if uid})

        # Return developerUser_objectId for pointer fields if needed
        if metadata.external_user_id and metadata.external_user_id in id_map:
            return metadata, {"developerUser_objectId": id_map[metadata.external_user_id]["developerUser_objectId"]}, is_qwen_route
        return metadata, {}, is_qwen_route

    @staticmethod
    async def batch_resolve_external_user_ids_to_internal(
        developer_id: str,
        external_ids: list,
        httpx_client: httpx.AsyncClient,
        x_api_key: str
    ) -> dict:
        """
        Batch resolve external user IDs to internal user IDs and DeveloperUser objectIds.
        Returns a mapping: {external_id: {"user_objectId": ..., "developerUser_objectId": ...}}
        """

        url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser"
        params = {
            "where": json.dumps({
                "developer": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": developer_id
                },
                "external_id": {"$in": external_ids}
            }),
            "limit": len(external_ids)
        }
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }
        resp = await httpx_client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        id_map = {}
        for r in results:
            if r.get("user") and r.get("objectId"):
                id_map[r["external_id"]] = {
                    "user_objectId": r["user"]["objectId"],
                    "developerUser_objectId": r["objectId"]
                }
        missing_external_ids = [eid for eid in external_ids if eid not in id_map]

        # 2. Batch create any missing users using batch_create_users_core
        if missing_external_ids:
            create_requests = [CreateUserRequest(external_id=eid) for eid in missing_external_ids]
            created_users = await batch_create_users_core(create_requests, developer_id, x_api_key)
            for user_resp in created_users:
                if user_resp.user_id and user_resp.external_id:
                    # After creation, fetch DeveloperUser objectId
                    url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser"
                    params = {
                        "where": json.dumps({
                            "developer": {
                                "__type": "Pointer",
                                "className": "_User",
                                "objectId": developer_id
                            },
                            "external_id": user_resp.external_id
                        }),
                        "limit": 1
                    }
                    resp = await httpx_client.get(url, headers=headers, params=params)
                    resp.raise_for_status()
                    results = resp.json().get("results", [])
                    if results:
                        dev_user_obj = results[0]
                        id_map[user_resp.external_id] = {
                            "user_objectId": user_resp.user_id,
                            "developerUser_objectId": dev_user_obj["objectId"]
                        }
        return id_map

    @staticmethod
    async def patch_and_resolve_user_ids_and_acls(
        developer_id: str,
        batch_request: BatchMemoryRequest,
        httpx_client: 'httpx.AsyncClient',
        x_api_key: str = None
    ) -> Tuple[BatchMemoryRequest, str, Optional[bool]]:
        """
        Resolves top-level external_user_id to user_id if needed.
        Patches each memory's metadata with user_id/external_user_id if missing.
        Resolves all ACL external user IDs to internal user IDs and patches ACL fields.
        Also patches batch_request.user_id and batch_request.external_user_id to the resolved values.
        Returns a tuple: (updated batch_request, resolved internal user_id (end_user_id), isQwenRoute flag).
        """
        logger.info(f"=== patch_and_resolve_user_ids_and_acls called ===")
        logger.info(f"developer_id: {developer_id}")
        logger.info(f"batch_request.external_user_id: {getattr(batch_request, 'external_user_id', None)}")
        logger.info(f"batch_request.memories count: {len(batch_request.memories) if batch_request.memories else 0}")
        

        external_user_id = getattr(batch_request, 'external_user_id', None)
        memories = batch_request.memories
        end_user_id = getattr(batch_request, 'user_id', None)

        # --- FIX: If both are missing, set to developer_id ---
        if not end_user_id and not external_user_id:
            end_user_id = developer_id
            batch_request.user_id = developer_id
            batch_request.external_user_id = None

        # 1. Resolve top-level external_user_id to user_id if needed
        is_qwen_route = None
        if not end_user_id and external_user_id:
            url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser"
            params = {
                "where": json.dumps({
                    "developer": {
                        "__type": "Pointer",
                        "className": "_User",
                        "objectId": developer_id
                    },
                    "external_id": external_user_id
                }),
                "limit": 1,
                "include": "user"
            }
            headers = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-Master-Key": PARSE_MASTER_KEY,
                "Content-Type": "application/json"
            }
            resp = await httpx_client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results and results[0].get("user"):
                end_user_id = results[0]["user"]["objectId"]
                is_qwen_route = results[0]["user"].get("isQwenRoute", None)
                logger.info(f"Resolved external_user_id {external_user_id} to end_user_id {end_user_id} with isQwenRoute: {is_qwen_route}")
            else:
                create_req = CreateUserRequest(external_id=external_user_id)
                try:
                    created = await batch_create_users_core([create_req], developer_id, x_api_key)
                    if created and created[0].user_id:
                        end_user_id = created[0].user_id
                        # Fetch isQwenRoute for newly created user
                        try:
                            url = f"{PARSE_SERVER_URL}/parse/classes/_User/{end_user_id}"
                            resp = await httpx_client.get(url, headers=headers)
                            resp.raise_for_status()
                            user_obj = resp.json()
                            is_qwen_route = user_obj.get("isQwenRoute", None)
                            logger.info(f"Created user {end_user_id} with isQwenRoute: {is_qwen_route}")
                        except Exception as e:
                            logger.warning(f"Failed to fetch isQwenRoute for newly created user {end_user_id}: {e}")
                    else:
                        logger.error(f"Failed to create user for external_user_id: {external_user_id}")
                        raise Exception(f"Failed to resolve or create user for external_user_id: {external_user_id}")
                except Exception as e:
                    logger.error(f"Error creating user for external_user_id {external_user_id}: {e}")
                    raise Exception(f"Failed to resolve or create user for external_user_id: {external_user_id}") from e

        # Patch top-level fields
        batch_request.user_id = end_user_id
        batch_request.external_user_id = external_user_id

        # 2. Patch each memory's metadata with user_id/external_user_id if missing
        for memory in memories:
            if memory.metadata is None:
                memory.metadata = MemoryMetadata()
            if not getattr(memory.metadata, 'user_id', None):
                memory.metadata.user_id = end_user_id
            
            # CRITICAL: Always set external_user_id from batch request if provided
            if external_user_id:
                print(f"CRITICAL: Setting external_user_id={external_user_id} on memory metadata")
                memory.metadata.external_user_id = external_user_id
                print(f"CRITICAL: After setting, memory.metadata.external_user_id={getattr(memory.metadata, 'external_user_id', None)}")

            # Ensure external_user_id is in external_user_read_access and external_user_write_access
            if external_user_id:
                if not getattr(memory.metadata, 'external_user_read_access', None):
                    memory.metadata.external_user_read_access = []
                if not getattr(memory.metadata, 'external_user_write_access', None):
                    memory.metadata.external_user_write_access = []
                if external_user_id not in memory.metadata.external_user_read_access:
                    memory.metadata.external_user_read_access.append(external_user_id)
                if external_user_id not in memory.metadata.external_user_write_access:
                    memory.metadata.external_user_write_access.append(external_user_id)
                logger.info(f"After setting ACLs: external_user_read_access={memory.metadata.external_user_read_access}, external_user_write_access={memory.metadata.external_user_write_access}")
            
            logger.info(f"Final memory metadata: external_user_id={getattr(memory.metadata, 'external_user_id', None)}")

        # 3. Collect all unique external user IDs from ACL fields across all memories
        all_external_acl_ids = set()
        for memory in memories:
            meta = memory.metadata
            if meta:
                for eid in getattr(meta, 'external_user_read_access', []) or []:
                    all_external_acl_ids.add(eid)
                for eid in getattr(meta, 'external_user_write_access', []) or []:
                    all_external_acl_ids.add(eid)
        if external_user_id in all_external_acl_ids:
            all_external_acl_ids.remove(external_user_id)

        # 4. Batch resolve all these external IDs to internal user IDs
        id_map = {}
        if all_external_acl_ids:
            id_map = await User.batch_resolve_external_user_ids_to_internal(
                developer_id=developer_id,
                external_ids=list(all_external_acl_ids),
                httpx_client=httpx_client,
                x_api_key=x_api_key
            )

        # 5. No ACL patching needed here - ACLs are set correctly upstream
        # The external_user_id is already in external_user_read_access/external_user_write_access (lines 3050-3059)
        # The user_id (if provided) should already be in the metadata from upstream
        # We keep them separate and don't merge/resolve between them

        # If we still don't have isQwenRoute and we have end_user_id, fetch it
        if is_qwen_route is None and end_user_id:
            try:
                url = f"{PARSE_SERVER_URL}/parse/classes/_User/{end_user_id}"
                headers = {
                    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                    "X-Parse-Master-Key": PARSE_MASTER_KEY,
                    "Content-Type": "application/json"
                }
                resp = await httpx_client.get(url, headers=headers)
                resp.raise_for_status()
                user_obj = resp.json()
                is_qwen_route = user_obj.get("isQwenRoute", None)
                logger.info(f"Fetched isQwenRoute for end_user_id {end_user_id}: {is_qwen_route}")
            except Exception as e:
                logger.warning(f"Failed to fetch isQwenRoute for end_user_id {end_user_id}: {e}")

        return batch_request, end_user_id, is_qwen_route

    @staticmethod
    async def get_comprehensive_user_info_async(user_id: str, api_key: Optional[str] = None) -> Tuple[Optional[str], Optional[bool], List[str], List[str]]:
        """
        Get comprehensive user info including workspace, roles, and isQwenRoute in a single Parse call.
        
        Returns:
            Tuple containing (workspace_id, is_qwen_route, user_roles, user_workspace_ids)
        """
        try:
            # Build headers
            headers = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "Content-Type": "application/json"
            }
            if api_key:
                headers["X-Parse-Master-Key"] = PARSE_MASTER_KEY
            else:
                headers["X-Parse-Master-Key"] = PARSE_MASTER_KEY  # Use master key for includes
            
            logger.info(f"Getting comprehensive user info for user_id={user_id}")
            
            # Run ALL Parse calls in parallel
            async with httpx.AsyncClient() as client:
                # Task 1: Get user data with includes
                user_url = f"{PARSE_SERVER_URL}/parse/classes/_User/{user_id}"
                user_params = {
                    "include": "isSelectedWorkspaceFollower,isSelectedWorkspaceFollower.workspace"
                }
                user_task = client.get(user_url, headers=headers, params=user_params)
                
                # Task 2: Get user roles
                user_instance = User.get(user_id)
                roles_task = user_instance.get_roles_async() if user_instance else asyncio.sleep(0, [])
                
                # Task 3: Get user workspaces
                workspaces_task = User.get_workspaces_for_user_async(user_id)
                
                # Run all tasks in parallel
                user_response, roles_result, workspaces_result = await asyncio.gather(
                    user_task,
                    roles_task,
                    workspaces_task,
                    return_exceptions=True
                )
                
                # Handle user data result
                if isinstance(user_response, Exception):
                    logger.warning(f"Failed to get user data for {user_id}: {user_response}")
                    workspace_id = None
                    is_qwen_route = None
                else:
                    user_data = user_response.json()
                    
                    # Extract isQwenRoute
                    is_qwen_route = user_data.get("isQwenRoute", None)
                    
                    # Extract workspace_id from included workspace follower
                    workspace_id = None
                    selected_follower = user_data.get('isSelectedWorkspaceFollower')
                    if selected_follower and isinstance(selected_follower, dict):
                        workspace = selected_follower.get('workspace')
                        if workspace and isinstance(workspace, dict):
                            workspace_id = workspace.get('objectId')
                
                # Handle roles result
                if isinstance(roles_result, Exception):
                    logger.warning(f"Failed to get roles for user {user_id}: {roles_result}")
                    user_roles = []
                else:
                    user_roles = roles_result
                
                # Handle workspaces result
                if isinstance(workspaces_result, Exception):
                    logger.warning(f"Failed to get workspaces for user {user_id}: {workspaces_result}")
                    user_workspace_ids = []
                else:
                    user_workspace_ids = workspaces_result
                
                logger.info(f"Comprehensive user info - workspace_id: {workspace_id}, is_qwen_route: {is_qwen_route}, roles: {len(user_roles)}, workspaces: {len(user_workspace_ids)}")
                
                return workspace_id, is_qwen_route, user_roles, user_workspace_ids
                
        except Exception as e:
            logger.error(f"Error getting comprehensive user info: {e}")
            return None, None, [], []

    async def _get_cached_schema_patterns_fast(self, user_object_id: str, workspace_object_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Fast lookup of cached schema patterns from Parse ActiveNodeRel class.
        This runs in parallel with other authentication tasks for optimal performance.
        """
        try:
            logger.info(f"🚀 FAST CACHE LOOKUP: Checking Parse cache for user {user_object_id}, workspace {workspace_object_id}")
            
            from services.active_node_rel_service import get_active_node_rel_service
            cache_service = get_active_node_rel_service()
            
            # Use httpx client for Parse operations
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:  # Short timeout for fast lookup
                cached_schema = await cache_service.get_cached_schema(
                    user_object_id=user_object_id,
                    workspace_object_id=workspace_object_id,
                    httpx_client=client
                )
            
            if cached_schema:
                logger.info(f"🚀 FAST CACHE HIT: Found cached schema with {len(cached_schema.get('nodes', []))} nodes and {len(cached_schema.get('patterns', []))} patterns")
                return cached_schema
            else:
                logger.info(f"🚀 FAST CACHE MISS: No cached schema found")
                return None
                
        except Exception as e:
            logger.warning(f"🚀 FAST CACHE ERROR: Failed to retrieve cached schema: {e}")
            return None

def get_parse_headers(session_token=None, api_key=None):
    """Build Parse headers, always using master key; keep params for compatibility."""
    headers = {
        "X-Parse-Application-Id": env.get("PARSE_APPLICATION_ID"),
        "X-Parse-Master-Key": env.get("PARSE_MASTER_KEY"),  # always use master key
        "Content-Type": "application/json",
    }

    # Optional API key header (ignored by some deployments)
    if api_key:
        headers["X-API-Key"] = api_key

    return headers

# If you want a batch utility for user creation, define it as follows:

async def batch_create_users_core(users, developer_id, x_api_key, organization_id=None, namespace_id=None):
    from routers.v1.user_routes import create_user_core  # Avoid circular import
    return await asyncio.gather(*(create_user_core(user, developer_id, x_api_key, organization_id, namespace_id) for user in users))

# Now, anywhere you need batch user creation, use batch_create_users_core(users, developer_id, x_api_key)
# Ensure there are no references to the old batch_create_users method.
