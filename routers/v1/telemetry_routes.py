"""
Telemetry Proxy Route

This endpoint allows OSS users to send anonymous telemetry data to Papr.
The endpoint acts as a proxy - it receives telemetry events and forwards them
to Amplitude using Papr's API key (which never leaves the server).

This prevents abuse by keeping the API key secure on Papr's infrastructure.
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import httpx
import os
from os import environ as env
from services.logger_singleton import LoggerSingleton
from core.services.telemetry import TelemetryProvider

logger = LoggerSingleton.get_logger(__name__)

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


# Pydantic models for request/response
class TelemetryEvent(BaseModel):
    """Single telemetry event"""
    event_name: str = Field(..., description="Event name (e.g., 'memory_created', 'search_performed')")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Event properties (will be anonymized)")
    user_id: Optional[str] = Field(default=None, description="Anonymous user ID (hashed)")
    timestamp: Optional[int] = Field(default=None, description="Event timestamp (Unix epoch in milliseconds)")


class TelemetryRequest(BaseModel):
    """Request body for telemetry endpoint"""
    events: List[TelemetryEvent] = Field(..., description="List of telemetry events to track")
    anonymous_id: Optional[str] = Field(default=None, description="Anonymous session ID")


class TelemetryResponse(BaseModel):
    """Response from telemetry endpoint"""
    success: bool = Field(..., description="Whether the events were successfully processed")
    events_received: int = Field(..., description="Number of events received")
    events_processed: int = Field(..., description="Number of events successfully processed")
    message: Optional[str] = Field(default=None, description="Optional message")


@router.post(
    "/events",
    response_model=TelemetryResponse,
    description="""
    Telemetry proxy endpoint for anonymous OSS adoption tracking.
    
    This endpoint receives telemetry events from OSS installations and forwards them
    to Amplitude using Papr's API key (which stays secure on the server).
    
    **Privacy**:
    - All user IDs are hashed/anonymized
    - No PII is collected
    - Data is used only for understanding OSS adoption patterns
    
    **Opt-in**: Users must explicitly enable telemetry in their OSS installation.
    
    **Request Body**:
    ```json
    {
      "events": [
        {
          "event_name": "memory_created",
          "properties": {
            "type": "text",
            "has_metadata": true
          },
          "user_id": "hashed_user_id",
          "timestamp": 1234567890000
        }
      ],
      "anonymous_id": "session_id"
    }
    ```
    """,
    openapi_extra={
        "operationId": "telemetry_events_v1",
        "x-openai-isConsequential": False
    }
)
async def telemetry_events(
    request: TelemetryRequest,
    http_request: Request
):
    """
    Receive telemetry events from OSS installations and forward to Amplitude.
    
    This endpoint:
    1. Validates incoming telemetry events
    2. Anonymizes any remaining PII
    3. Forwards to Amplitude using Papr's secure API key
    4. Returns success/failure status
    """
    try:
        # Get Amplitude API key from environment (server-side only)
        amplitude_api_key = env.get("AMPLITUDE_API_KEY")
        
        if not amplitude_api_key:
            logger.warning("Amplitude API key not configured - telemetry events will not be forwarded")
            return TelemetryResponse(
                success=False,
                events_received=len(request.events),
                events_processed=0,
                message="Telemetry service not configured"
            )
        
        # Validate and process events
        events_received = len(request.events)
        events_processed = 0
        
        # Anonymize properties and prepare events for Amplitude
        from core.services.telemetry import get_telemetry
        telemetry_service = get_telemetry()
        
        # Process each event
        for event in request.events:
            try:
                # Use anonymous_id from request or generate one
                user_id = event.user_id or request.anonymous_id or telemetry_service.anonymous_id
                
                # Anonymize properties (remove PII)
                safe_properties = telemetry_service._anonymize_properties(event.properties or {})
                safe_properties['edition'] = 'opensource'
                safe_properties['is_oss'] = True
                safe_properties['telemetry_destination'] = 'papr_proxy'
                
                # Add technical context
                safe_properties.update(telemetry_service._get_context())
                
                # Forward to Amplitude via HTTP API (not SDK, to keep it simple)
                await _send_to_amplitude(
                    amplitude_api_key=amplitude_api_key,
                    event_name=event.event_name,
                    user_id=user_id,
                    properties=safe_properties,
                    timestamp=event.timestamp
                )
                
                events_processed += 1
                
            except Exception as e:
                logger.warning(f"Failed to process telemetry event {event.event_name}: {e}")
                # Continue processing other events even if one fails
        
        logger.info(f"Processed {events_processed}/{events_received} telemetry events")
        
        return TelemetryResponse(
            success=events_processed > 0,
            events_received=events_received,
            events_processed=events_processed,
            message=f"Processed {events_processed} of {events_received} events" if events_processed < events_received else None
        )
        
    except Exception as e:
        logger.error(f"Error processing telemetry request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process telemetry events: {str(e)}"
        )


async def _send_to_amplitude(
    amplitude_api_key: str,
    event_name: str,
    user_id: str,
    properties: Dict[str, Any],
    timestamp: Optional[int] = None
):
    """
    Send event to Amplitude via HTTP API.
    
    Uses Amplitude's HTTP API instead of SDK to keep dependencies minimal.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Amplitude HTTP API endpoint
            url = "https://api2.amplitude.com/2/httpapi"
            
            # Build request body
            event_data = {
                "event_type": event_name,
                "user_id": user_id,
                "event_properties": properties,
            }
            
            if timestamp:
                event_data["time"] = timestamp
            
            payload = {
                "api_key": amplitude_api_key,
                "events": [event_data]
            }
            
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.warning(f"Amplitude API returned status {response.status_code}: {response.text}")
            else:
                logger.debug(f"Successfully sent event {event_name} to Amplitude")
                
    except httpx.TimeoutException:
        logger.warning(f"Timeout sending event {event_name} to Amplitude")
    except Exception as e:
        logger.warning(f"Error sending event {event_name} to Amplitude: {e}")
        # Don't raise - we want telemetry to fail silently

