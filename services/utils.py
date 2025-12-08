import asyncio
from datetime import datetime, date
from typing import Optional, Any, Dict, List
from fastapi import Request
from memory.memory_graph import MemoryGraph
from enum import Enum

# Legacy Amplitude imports (no longer used - kept for reference)
# from amplitude import Identify, BaseEvent, EventOptions

def serialize_datetime(obj):
    """Helper function to serialize datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def _to_json_safe(value: Any) -> Any:
    """Recursively convert values to JSON-serializable types.

    - Enum -> its value
    - datetime/date -> ISO string
    - set/tuple -> list
    - dict/list -> recurse
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    return value

def to_json_safe_mapping(mapping: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not mapping:
        return {}
    return {str(k): _to_json_safe(v) for k, v in mapping.items()}

# Dependency to provide the singleton MemoryGraph
def get_memory_graph(request: Request) -> MemoryGraph:
    return request.app.state.memory_graph

async def log_amplitude_event(
    event_type: str,
    user_info: dict,
    client_type: str,
    amplitude_client,  # Deprecated: kept for backward compatibility
    logger,
    extra_properties: Optional[dict] = None,
    end_user_id: Optional[str] = None
) -> bool:
    """
    Helper function to log events using TelemetryService.
    
    This function has been updated to use the new telemetry system:
    - Cloud edition: Full tracking with developer_id + end_user_id (Amplitude)
    - OSS edition: Anonymous tracking (PostHog)
    
    The amplitude_client parameter is kept for backward compatibility but is no longer used.
    
    Returns True if event was successfully logged, False otherwise.
    """
    try:
        import os
        from core.services.telemetry import get_telemetry
        
        # Extract developer_id (API key owner / workspace owner)
        developer_id = None
        if user_info:
            developer_id = user_info.get('developer_id')
        if not developer_id and extra_properties:
            developer_id = extra_properties.get('developer_id')
        
        # Extract end_user_id (the developer's customer)
        user_id = None
        if end_user_id:
            user_id = str(end_user_id)
        elif user_info:
            user_id = (
                user_info.get('https://papr.scope.com/objectId') or
                user_info.get('objectId') or
                user_info.get('user_id')
            )
        
        # Fallback: if no developer_id, use user_id as developer
        if not developer_id and user_id:
            developer_id = user_id
        
        # Check if we're in cloud mode
        edition = os.getenv("PAPR_EDITION", "opensource").lower()
        is_cloud = edition == "cloud"
        
        # Build event properties
        event_properties: Dict[str, Any] = {
            'client_type': client_type,
        }
        
        # For cloud mode, include additional analytics data
        if is_cloud and user_info:
            # Include email and geo info for cloud analytics
            email_address = user_info.get('email')
            geo_info = user_info.get('geoip', {})
            
            if email_address:
                event_properties['email'] = email_address
            
            # Add geo info for cloud analytics
            for key in ['country_code', 'country_name', 'city_name', 'time_zone', 
                       'continent_code', 'subdivision_code', 'subdivision_name']:
                if key in geo_info and geo_info[key] is not None:
                    event_properties[key] = geo_info[key]
            
            for key in ['latitude', 'longitude']:
                if key in geo_info and geo_info[key] is not None:
                    try:
                        event_properties[key] = float(geo_info[key])
                    except ValueError:
                        logger.warning(f"Omitting {key} due to invalid format: {geo_info[key]}")
        
        # Add extra properties
        if extra_properties:
            event_properties.update(extra_properties)
        
        # Ensure all event properties are JSON-safe
        event_properties = to_json_safe_mapping(event_properties)
        
        # Track event using telemetry service
        # In cloud: tracks by developer_id, includes end_user_id in properties
        # In OSS: anonymizes everything
        telemetry = get_telemetry()
        await telemetry.track(
            event_type, 
            event_properties, 
            user_id=user_id,
            developer_id=developer_id
        )
        
        return True  # Success!
    except Exception as e:
        logger.error(f"Error logging event to telemetry: {e}")
        return False  # Failure
