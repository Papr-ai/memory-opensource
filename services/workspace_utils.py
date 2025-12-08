"""
Workspace utilities for open-source edition.
Implements addPeopleToWorkspace functionality that would normally be handled by Parse Server cloud functions.
"""
import httpx
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from os import environ as env
from services.logger_singleton import LoggerSingleton
from services.url_utils import clean_url

logger = LoggerSingleton.get_logger(__name__)

PARSE_SERVER_URL = clean_url(env.get("PARSE_SERVER_URL"))
PARSE_APPLICATION_ID = clean_url(env.get("PARSE_APPLICATION_ID"))
PARSE_MASTER_KEY = clean_url(env.get("PARSE_MASTER_KEY"))

PARSE_HEADERS = {
    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
    "X-Parse-Master-Key": PARSE_MASTER_KEY,
    "Content-Type": "application/json"
}


async def add_people_to_workspace_opensource(
    user_who_added_people_id: str,
    workspace_id: str,
    users_to_add: List[Dict[str, str]]
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Python implementation of addPeopleToWorkspace for open-source edition.
    
    This function:
    1. Gets or creates workspace_follower records for each user
    2. Sets isFollower=true and isMember=true
    3. Sets isSelectedWorkspaceFollower on the user (for the first user added)
    4. Returns [userWhoAddedPeopleWorkspaceFollower, newWorkspaceFollowers]
    
    Args:
        user_who_added_people_id: The user ID who is adding people
        workspace_id: The workspace ID
        users_to_add: List of dicts with 'id' key containing user IDs to add
        
    Returns:
        Tuple of (userWhoAddedPeopleWorkspaceFollower, list of new workspace_followers)
    """
    logger.info(f"add_people_to_workspace_opensource: workspace={workspace_id}, users={[u.get('id') for u in users_to_add]}")
    
    timeout = httpx.Timeout(connect=5.0, read=25.0, write=10.0, pool=5.0)
    
    # Get userWhoAddedPeople's workspace_follower
    user_who_added_workspace_follower = None
    try:
        query_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower"
        query_params = {
            "where": json.dumps({
                "user": {"__type": "Pointer", "className": "_User", "objectId": user_who_added_people_id},
                "workspace": {"__type": "Pointer", "className": "WorkSpace", "objectId": workspace_id}
            }),
            "limit": 1
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(query_url, headers=PARSE_HEADERS, params=query_params)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results:
                    user_who_added_workspace_follower = results[0]
                    logger.info(f"Found existing workspace_follower for userWhoAddedPeople: {user_who_added_workspace_follower.get('objectId')}")
    except Exception as e:
        logger.warning(f"Error getting userWhoAddedPeople workspace_follower: {e}")
    
    # Get workspace to fetch post/postMessage
    workspace_post = None
    workspace_post_message = None
    try:
        workspace_url = f"{PARSE_SERVER_URL}/parse/classes/WorkSpace/{workspace_id}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(workspace_url, headers=PARSE_HEADERS)
            if response.status_code == 200:
                workspace_data = response.json()
                # Note: In open-source, we skip fetching post/postMessage for simplicity
                # The cloud version fetches these, but they're optional
                logger.debug(f"Workspace fetched: {workspace_id}")
    except Exception as e:
        logger.warning(f"Error fetching workspace: {e}")
    
    # Process each user to add
    new_workspace_followers = []
    user_ids_to_add = [u.get('id') or u.get('objectId') for u in users_to_add]
    
    # First, check which users already have workspace_followers
    existing_followers = {}
    try:
        query_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower"
        query_params = {
            "where": json.dumps({
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "$in": [{"__type": "Pointer", "className": "_User", "objectId": uid} for uid in user_ids_to_add]
                },
                "workspace": {"__type": "Pointer", "className": "WorkSpace", "objectId": workspace_id}
            }),
            "limit": 1000
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(query_url, headers=PARSE_HEADERS, params=query_params)
            if response.status_code == 200:
                data = response.json()
                for follower in data.get('results', []):
                    user_ptr = follower.get('user')
                    if isinstance(user_ptr, dict):
                        user_id = user_ptr.get('objectId')
                    else:
                        user_id = str(user_ptr)
                    existing_followers[user_id] = follower
                logger.info(f"Found {len(existing_followers)} existing workspace_followers")
    except Exception as e:
        logger.warning(f"Error checking existing workspace_followers: {e}")
    
    # Update existing or create new workspace_followers
    for user_id in user_ids_to_add:
        if user_id in existing_followers:
            # Update existing workspace_follower
            follower_id = existing_followers[user_id].get('objectId')
            update_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower/{follower_id}"
            update_data = {
                "isFollower": True,
                "isMember": True
            }
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.put(update_url, headers=PARSE_HEADERS, json=update_data)
                    if response.status_code == 200:
                        updated_follower = response.json()
                        new_workspace_followers.append(updated_follower)
                        logger.info(f"Updated existing workspace_follower for user {user_id}: {follower_id}")
                    else:
                        logger.error(f"Failed to update workspace_follower {follower_id}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error updating workspace_follower for user {user_id}: {e}")
        else:
            # Create new workspace_follower
            create_data = {
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                },
                "workspace": {
                    "__type": "Pointer",
                    "className": "WorkSpace",
                    "objectId": workspace_id
                },
                "isFollower": True,
                "isMember": True,
                "isSelected": False,  # Will be set below for first user
                "archive": False,
                "notificationCount": 0,
                "isNotified": False,
                "isUnRead": False,
                "ACL": {
                    user_id: {"read": True, "write": True}
                }
            }
            
            try:
                create_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower"
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(create_url, headers=PARSE_HEADERS, json=create_data)
                    if response.status_code in [200, 201]:
                        new_follower = response.json()
                        new_workspace_followers.append(new_follower)
                        logger.info(f"Created new workspace_follower for user {user_id}: {new_follower.get('objectId')}")
                    else:
                        logger.error(f"Failed to create workspace_follower for user {user_id}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error creating workspace_follower for user {user_id}: {e}")
    
    # Set isSelectedWorkspaceFollower on the first user (if any were added)
    if new_workspace_followers:
        first_follower = new_workspace_followers[0]
        first_user_id = None
        user_ptr = first_follower.get('user')
        if isinstance(user_ptr, dict):
            first_user_id = user_ptr.get('objectId')
        else:
            first_user_id = str(user_ptr)
        
        if first_user_id:
            # Set isSelected=true on the workspace_follower
            follower_id = first_follower.get('objectId')
            try:
                update_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower/{follower_id}"
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.put(update_url, headers=PARSE_HEADERS, json={"isSelected": True})
                    if response.status_code == 200:
                        logger.info(f"Set isSelected=true on workspace_follower {follower_id}")
            except Exception as e:
                logger.warning(f"Error setting isSelected on workspace_follower: {e}")
            
            # Set isSelectedWorkspaceFollower pointer on the user
            try:
                user_update_url = f"{PARSE_SERVER_URL}/parse/users/{first_user_id}"
                user_update_data = {
                    "isSelectedWorkspaceFollower": {
                        "__type": "Pointer",
                        "className": "workspace_follower",
                        "objectId": follower_id
                    }
                }
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.put(user_update_url, headers=PARSE_HEADERS, json=user_update_data)
                    if response.status_code == 200:
                        logger.info(f"✅ Set isSelectedWorkspaceFollower on user {first_user_id} to {follower_id}")
                    else:
                        logger.warning(f"Failed to set isSelectedWorkspaceFollower: {response.status_code} - {response.text}")
            except Exception as e:
                logger.warning(f"Error setting isSelectedWorkspaceFollower on user: {e}")
    
    # Return in the same format as cloud function: [userWhoAddedPeopleWorkspaceFollower, newWorkspaceFollowers]
    return user_who_added_workspace_follower, new_workspace_followers

