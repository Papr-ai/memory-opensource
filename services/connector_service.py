from abc import ABC, abstractmethod
from services.memory_management import find_user_by_connector_ids, get_workspaceId_using_tenantId, store_connector_user_id
from services.logging_config import get_logger
from typing import Optional

from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)


# Remove ts fields
def remove_ts_fields(d):
    if isinstance(d, dict):
        return {k: remove_ts_fields(v) for k, v in d.items() if 'ts' not in k}
    elif isinstance(d, list):
        return [remove_ts_fields(i) for i in d]
    else:
        return d

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def clean_body_data(body_data):
    # Function to clean bodyData from JSON to plain text
    import json
    try:
        body = json.loads(body_data)
        return ' '.join([content['text'] for paragraph in body['content'] for content in paragraph['content'] if 'text' in content])
    except (json.JSONDecodeError, KeyError):
        return body_data

def process_metadata(metadata: dict, connector: str, excluded_keys: list = None) -> dict:
    """
    Cleans and formats the metadata dictionary.

    Args:
        metadata (dict): The original metadata.
        connector (str): The connector name (e.g., 'slack', 'linear').
        excluded_keys (list, optional): Keys to exclude from metadata. Defaults to None.

    Returns:
        dict: The cleaned and formatted metadata.
    """
    if excluded_keys is None:
        excluded_keys = ['sessionToken', 'text', 'blocks', 'messages']

    logger.info(f"Original metadata: {metadata}")

    # Rename 'id' to '<connector>_id'
    if 'id' in metadata:
        metadata[f"{connector}_id"] = metadata.pop("id")

    if connector: 
        metadata["connector"] = connector

    # Remove keys with None values
    metadata = {k: v for k, v in metadata.items() if v is not None}
    logger.info(f"Metadata after removing None: {metadata}")

    # Flatten nested dictionaries
    flattened_metadata = flatten_dict(metadata)
    logger.info(f"Flattened metadata: {flattened_metadata}")

    # Initialize supported_metadata
    supported_metadata = {}

    for k, v in flattened_metadata.items():
        if k in excluded_keys:
            continue  # Skip excluded fields
        elif isinstance(v, (str, int, float, bool)):
            supported_metadata[k] = v
        elif isinstance(v, list):
            if all(isinstance(i, str) for i in v):
                supported_metadata[k] = v
            else:
                supported_metadata[k] = str(v)
        else:
            supported_metadata[k] = str(v)

    # Remove keys with None values
    supported_metadata = {k: v for k, v in supported_metadata.items() if v is not None}
    logger.info(f"Supported metadata: {supported_metadata}")

    return supported_metadata


async def transpose_data_to_memory(input_data: dict, url: str, session_token: str, parse_user_id: str, tenant_id: str, update=False, api_key: Optional[str] = None):
    """
    Transforms input data into a memory item suitable for storage.

    Args:
        input_data (dict): The raw input data.
        url (str): The URL indicating the data source.
        session_token (str): The session token for authentication.
        parse_user_id (str): The user ID from Parse Server.
        update (bool, optional): Flag indicating if this is an update. Defaults to False.
        tenant_id (str): The tenant ID from Parse Server.
    Returns:
        MemoryItem as a dictionary
    """
    # Extract connector from URL
    connector = url.split('/')[3]

    # Initialize variables
    memory_type = "text"
    hierarchical_structures = ""
    authed_user_id = input_data.get('authed_user_id')
    is_private = input_data.get('is_private', True)  # Default to True if not provided
    sourceType = ""
    content = ""
    context = input_data.get('context')
    acl_object_ids = []
    source_urls = []
    workspace_id = await get_workspaceId_using_tenantId(session_token, tenant_id)
    logger.info(f"workspace_id: {workspace_id}")
    # Delimiters
    content_delimiter = ' ||| '

    members_raw = input_data.get('members', [])

    if not is_private:
        # For public memories, ACL will be handled later by setting workspace access
        logger.info("Memory is public. ACL will be set to workspace-wide access.")
    

    # Ensure 'members' is a list
    if isinstance(members_raw, list):
        members = members_raw
    elif isinstance(members_raw, str):
        # If 'members' is a comma-separated string, split it into a list
        members = members_raw.split(',')
    else:
        # Fallback to an empty list if 'members' is neither list nor string
        members = []

    logger.info(f"Processed members: {members} (Type: {type(members)})")

    logger.info(f"Input data for Slack message: {input_data}")

    # Initialize variables for metadata
    specific_metadata = {}
    

    # Determine type based on URL
    if "/linear/issues" in url:
        memory_type = "issue"
        sourceType = 'linear'
        hierarchical_structures = f"{input_data.get('project', {}).get('name', '')}, Issues"
        content = input_data["title"]
    elif "/linear/projects" in url:
        memory_type = "text"
        sourceType = 'linear'
        hierarchical_structures = f"{input_data.get('name', '')}, Projects"
        milestones = input_data.get("projectMilestones")
        content = f"Project: {input_data['name']}\nMilestones: {milestones}"
        input_data["projectMilestones"] = milestones  # Keep milestones in metadata
    elif "/linear/comments" in url:
        memory_type = "text"
        sourceType = 'linear'
        hierarchical_structures = "Comments"
        content = clean_body_data(input_data.get("bodyData", input_data.get("body", "")))
    elif "/linear/users" in url:
        memory_type = "text"
        sourceType = 'linear'
        hierarchical_structures = "Users"
        content = f"User: {input_data['displayName']} ({input_data['email']})"

    elif "/slack/messages" in url:
        memory_type = "text"
        hierarchical_structures = "Slack Message"
        sourceType = 'slack'

        # Add debug logging
        logger.info(f"Processing Slack message with input_data: {input_data}")

        # If the connector is Slack, retrieve Papr user objectIds for members
        if connector == "slack" and members:
            acl_object_ids = await find_user_by_connector_ids(session_token, connector, members)
            logger.info(f"ACL Object IDs: {acl_object_ids}")
            
            # Remove duplicates
            acl_object_ids = list(set(acl_object_ids))

        # Handle 'messageList' type
        if input_data.get('type') == 'messageList':
            logger.info(f"input_data: {input_data}")

            messages = input_data.get('messages', [])
            logger.info(f"messages: {messages}")

            # Lists to collect per-message data
            message_texts = []
            user_ids = []
            client_msg_ids = []

            for message in messages:
                # Collect message text
                text = message.get('text', '')
                if text:
                    message_texts.append(text)

                # Collect per-message metadata
                user_id = message.get('user', '')
                client_msg_id = message.get('client_msg_id', '')
                source_url = message.get('sourceUrl', '')

                user_ids.append(user_id)
                client_msg_ids.append(client_msg_id)
                source_urls.append(source_url)

            # Combine all message texts into content using the content delimiter
            content = content_delimiter.join(message_texts)
            logger.info(f"content: {content}")

            logger.info(f"user_ids: {user_ids}")
            logger.info(f"client_msg_ids: {client_msg_ids}")
            logger.info(f"source_urls: {source_urls}")

            # Prepare specific metadata for 'messageList'
            specific_metadata = {
                'type': input_data.get('type'),
                'members': members,
                'authed_user_id': authed_user_id,
                'user_ids': user_ids,
                'client_msg_ids': client_msg_ids,
                'source_urls': source_urls,
                'sourceUrl': source_urls[0] if source_urls else '',
                'workspace_id': workspace_id,
            }
            logger.info(f"Prepare specific metadata for 'messageList': {specific_metadata}")

            # Clean the specific metadata
            specific_metadata = process_metadata(specific_metadata, connector)
       
        # handle 'message' type
        else:
            source_urls.append(input_data.get('sourceUrl', ''))
            logger.info(f"source_urls: {source_urls}")

            if update:
                content = f"{input_data['message']['text']} (edited)"
                if "blocks" in input_data['message'] and len(input_data['message']["blocks"]) > 0:
                    input_data['message']["block_id"] = input_data['message']["blocks"][0].get("block_id")
                    del input_data['message']
                if "blocks" in input_data['previous_message'] and len(input_data['previous_message']["blocks"]) > 0:
                    del input_data['previous_message']["blocks"]
                
                # Ensure client_msg_id remains in metadata
                client_msg_id = input_data.get('message', {}).get('client_msg_id')
                if client_msg_id:
                    input_data['client_msg_id'] = client_msg_id
                        
            else:
                # Get the text content
                content = input_data.get('text')
                if not content:
                    raise ValueError("No text content found in Slack message")
                    
                logger.info(f"Extracted content from Slack message: {content}")
                
                if "blocks" in input_data and len(input_data["blocks"]) > 0:
                    input_data["block_id"] = input_data["blocks"][0].get("block_id")
                    del input_data["blocks"]
            
             # Prepare specific metadata for 'message'
            specific_metadata = {
                'client_msg_id': input_data.get('client_msg_id'),
                'sourceUrl': source_urls[0] if source_urls else '',
                'workspace_id': workspace_id,
            }
            logger.info(f"Prepare specific metadata for 'message': {specific_metadata}")

            # Clean the specific metadata
            specific_metadata = process_metadata(specific_metadata, connector)
    else:
        memory_type = "text"
        content = input_data.get('content')
        hierarchical_structures = input_data.get('project', {}).get('name', '')

    input_data = remove_ts_fields(input_data)

    logger.info(f"content transpose_data_to_memory: {content}")

    # Prepare general metadata by excluding specific keys within process_metadata
    excluded_keys = ["sessionToken", "text", "blocks", "messages", "members"]
    logger.info(f"General metadata before processing: {input_data}")

    general_metadata = process_metadata(input_data, connector, excluded_keys=excluded_keys)


    logger.info(f"General metadata after processing: {general_metadata}")

    # Merge specific metadata with general metadata
    combined_metadata = {**general_metadata, **specific_metadata}
    logger.info(f"Combined metadata: {combined_metadata}")

    # At this point, combined_metadata is already processed and flattened by process_metadata
    # No need to flatten or process again

    # Initialize supported_metadata directly from combined_metadata
    supported_metadata = combined_metadata  # Already processed

    logger.info(f"Final supported metadata: {supported_metadata}")

    if authed_user_id: 
        stored = await store_connector_user_id(session_token, parse_user_id, connector, authed_user_id, api_key=api_key)
        logger.info(f"stored connector user id: {authed_user_id} for parse user id: {parse_user_id} in connector: {connector} with result: {stored}")
   
        
    final_metadata = {
        "hierarchical structures": hierarchical_structures,
        "sourceType": sourceType,
        "sourceUrl": source_urls[0] if source_urls else '',
        "workspace_id": workspace_id,
        'acl_object_ids': acl_object_ids,
        "is_private": is_private,
        **supported_metadata        
    }

    memory_item_dict = {
        "content": content,
        "type": memory_type,
        "metadata": final_metadata
    }

    

    logger.info(f"transpose_data_to_memory memory_item_dict: {memory_item_dict}")

    return memory_item_dict
