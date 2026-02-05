"""
Multi-tenant utilities for PAPR Memory system.

This module provides utilities for handling multi-tenant authentication and scoping
across the PAPR Memory API. It supports both legacy authentication (for backward
compatibility with Papr chat app) and organization-based authentication (for
developer dashboard).

Key concepts:
- Legacy Auth: Traditional API keys tied directly to users (backward compatible)
- Organization Auth: API keys scoped to Organization → Namespace → APIKey hierarchy
- Automatic scoping: Memories are automatically scoped to their organization/namespace context
"""

from typing import Dict, Any, Optional, Union
from models.memory_models import MemoryMetadata, SearchRequest, AddMemoryRequest, BatchMemoryRequest, UpdateMemoryRequest
from models.feedback_models import FeedbackRequest
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)

def extract_multi_tenant_context(auth_response) -> Dict[str, Any]:
    """
    Extract multi-tenant context from authentication response.

    Args:
        auth_response: OptimizedAuthResponse object containing authentication data

    Returns:
        Dict containing:
        - organization_id: Organization ID from auth context (None for legacy)
        - namespace_id: Namespace ID from auth context (None for legacy)
        - is_legacy_auth: Whether this is legacy authentication
        - auth_type: "legacy" or "organization"
    """
    context = {
        'organization_id': getattr(auth_response, 'organization_id', None),
        'namespace_id': getattr(auth_response, 'namespace_id', None),
        'is_legacy_auth': getattr(auth_response, 'is_legacy_auth', True),
        'auth_type': getattr(auth_response, 'auth_type', 'legacy')
    }

    # Log multi-tenant context for debugging
    if context['organization_id'] or context['namespace_id']:
        logger.info(f"Multi-tenant context extracted - org_id: {context['organization_id']}, "
                   f"namespace_id: {context['namespace_id']}, auth_type: {context['auth_type']}")
    else:
        logger.info(f"Legacy authentication context - auth_type: {context['auth_type']}")

    return context

def apply_multi_tenant_scoping_to_metadata(
    metadata: Optional[MemoryMetadata],
    auth_context: Dict[str, Any],
    request_context: Optional[Dict[str, Any]] = None
) -> MemoryMetadata:
    """
    Apply multi-tenant scoping to metadata object.

    This function ensures that memories are properly scoped to their organization
    and namespace context based on the authenticated user's context.

    Args:
        metadata: Existing MemoryMetadata object (can be None)
        auth_context: Multi-tenant context from authentication
        request_context: Optional explicit organization/namespace from request

    Returns:
        MemoryMetadata object with multi-tenant scoping applied
    """
    # Initialize metadata if None
    if metadata is None:
        metadata = MemoryMetadata()

    def _log_custom_metadata_state(stage: str) -> None:
        """Helper to log whether customMetadata already carries org/ns filters."""
        custom_meta = getattr(metadata, "customMetadata", None) or {}
        if not custom_meta:
            logger.info(f"{stage}: metadata.customMetadata is empty")
            return
        logger.info(f"{stage}: metadata.customMetadata keys = {list(custom_meta.keys())}")
        for key in ("organization_id", "namespace_id"):
            if key in custom_meta:
                logger.warning(
                    f"{stage}: metadata.customMetadata already contains '{key}' "
                    "which will force a strict filter downstream"
                )

    _log_custom_metadata_state("Before multi-tenant scoping")

    # Apply organization scoping from auth context (takes precedence)
    if auth_context.get('organization_id') is not None:
        # Set typed field for compatibility
        try:
            setattr(metadata, 'organization_id', auth_context['organization_id'])
        except Exception:
            pass
        logger.info(f"Applied organization scoping from auth: {auth_context['organization_id']}")

    # Apply namespace scoping from auth context (takes precedence)
    if auth_context.get('namespace_id') is not None:
        # Set typed field for compatibility
        try:
            setattr(metadata, 'namespace_id', auth_context['namespace_id'])
        except Exception:
            pass
        logger.info(f"Applied namespace scoping from auth: {auth_context['namespace_id']}")

    # Apply explicit organization/namespace from request only when auth context is absent
    if request_context:
        req_org = request_context.get('organization_id')
        req_ns = request_context.get('namespace_id')
        auth_org = auth_context.get('organization_id')
        auth_ns = auth_context.get('namespace_id')

        if req_org is not None:
            if auth_org is None:
                try:
                    setattr(metadata, 'organization_id', req_org)
                except Exception:
                    pass
                logger.info(f"Applied explicit organization from request: {req_org}")
            elif req_org != auth_org:
                logger.warning(
                    "Ignoring explicit organization_id from request (%s) due to auth context (%s)",
                    req_org,
                    auth_org,
                )

        if req_ns is not None:
            if auth_ns is None:
                try:
                    setattr(metadata, 'namespace_id', req_ns)
                except Exception:
                    pass
                logger.info(f"Applied explicit namespace from request: {req_ns}")
            elif req_ns != auth_ns:
                logger.warning(
                    "Ignoring explicit namespace_id from request (%s) due to auth context (%s)",
                    req_ns,
                    auth_ns,
                )

    logger.info(
        "After multi-tenant scoping: org_id=%s, namespace_id=%s",
        getattr(metadata, 'organization_id', None),
        getattr(metadata, 'namespace_id', None),
    )
    _log_custom_metadata_state("After multi-tenant scoping")

    return metadata

def apply_multi_tenant_scoping_to_search_request(
    search_request: SearchRequest,
    auth_context: Dict[str, Any]
) -> MemoryMetadata:
    """
    Apply multi-tenant scoping to search request metadata.

    Args:
        search_request: SearchRequest object
        auth_context: Multi-tenant context from authentication

    Returns:
        MemoryMetadata with applied scoping
    """
    # Start with search request metadata
    metadata = search_request.metadata or MemoryMetadata()

    # Extract request context
    request_context = {
        'organization_id': getattr(search_request, 'organization_id', None),
        'namespace_id': getattr(search_request, 'namespace_id', None)
    }

    # Keep SearchRequest aligned with auth context when present
    auth_org = auth_context.get('organization_id')
    auth_ns = auth_context.get('namespace_id')
    if auth_org is not None:
        if request_context['organization_id'] not in (None, auth_org):
            logger.warning(
                "Overriding SearchRequest.organization_id (%s) with auth context (%s)",
                request_context['organization_id'],
                auth_org,
            )
        setattr(search_request, 'organization_id', auth_org)
    if auth_ns is not None:
        if request_context['namespace_id'] not in (None, auth_ns):
            logger.warning(
                "Overriding SearchRequest.namespace_id (%s) with auth context (%s)",
                request_context['namespace_id'],
                auth_ns,
            )
        setattr(search_request, 'namespace_id', auth_ns)

    return apply_multi_tenant_scoping_to_metadata(metadata, auth_context, request_context)

def apply_multi_tenant_scoping_to_memory_request(
    memory_request: AddMemoryRequest,
    auth_context: Dict[str, Any]
) -> None:
    """
    Apply multi-tenant scoping to memory request (modifies in place).

    Args:
        memory_request: AddMemoryRequest object to modify
        auth_context: Multi-tenant context from authentication
    """
    # Set organization_id and namespace_id directly on the memory request
    # Auth context takes precedence, but allow existing values to remain if auth context is None
    if auth_context.get('organization_id') is not None:
        memory_request.organization_id = auth_context['organization_id']
        logger.info(f"Applied organization scoping from auth: {auth_context['organization_id']}")

    if auth_context.get('namespace_id') is not None:
        memory_request.namespace_id = auth_context['namespace_id']
        logger.info(f"Applied namespace scoping from auth: {auth_context['namespace_id']}")

    # Extract request context
    request_context = {
        'organization_id': getattr(memory_request, 'organization_id', None),
        'namespace_id': getattr(memory_request, 'namespace_id', None)
    }

    # Apply scoping to metadata
    memory_request.metadata = apply_multi_tenant_scoping_to_metadata(
        memory_request.metadata,
        auth_context,
        request_context
    )

def apply_multi_tenant_scoping_to_batch_request(
    batch_request: BatchMemoryRequest,
    auth_context: Dict[str, Any]
) -> None:
    """
    Apply multi-tenant scoping to batch memory request (modifies in place).

    Args:
        batch_request: BatchMemoryRequest object to modify
        auth_context: Multi-tenant context from authentication
    """
    # Set organization_id and namespace_id directly on the batch request
    # Auth context takes precedence, but allow existing values to remain if auth context is None
    if auth_context.get('organization_id') is not None:
        batch_request.organization_id = auth_context['organization_id']
        logger.info(f"Applied organization scoping from auth: {auth_context['organization_id']}")

    if auth_context.get('namespace_id') is not None:
        batch_request.namespace_id = auth_context['namespace_id']
        logger.info(f"Applied namespace scoping from auth: {auth_context['namespace_id']}")

    # Apply scoping to each memory in the batch
    for memory_request in batch_request.memories:
        # Use the utility function to apply scoping to each individual memory
        apply_multi_tenant_scoping_to_memory_request(memory_request, auth_context)

def apply_multi_tenant_scoping_to_update_request(
    update_request: UpdateMemoryRequest,
    auth_context: Dict[str, Any]
) -> None:
    """
    Apply multi-tenant scoping to update memory request (modifies in place).

    Args:
        update_request: UpdateMemoryRequest object to modify
        auth_context: Multi-tenant context from authentication
    """
    # Extract request context
    request_context = {
        'organization_id': getattr(update_request, 'organization_id', None),
        'namespace_id': getattr(update_request, 'namespace_id', None)
    }

    # Apply scoping to metadata
    update_request.metadata = apply_multi_tenant_scoping_to_metadata(
        update_request.metadata,
        auth_context,
        request_context
    )

def get_scoping_context_for_feedback(
    feedback_request: FeedbackRequest,
    auth_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get scoping context for feedback requests.

    Args:
        feedback_request: FeedbackRequest object
        auth_context: Multi-tenant context from authentication

    Returns:
        Dict with organization_id and namespace_id for scoping
    """
    # For feedback, we use auth context but allow request override
    scoping_context = {
        'organization_id': auth_context.get('organization_id'),
        'namespace_id': auth_context.get('namespace_id')
    }

    # Allow explicit override from request
    if getattr(feedback_request, 'organization_id', None) is not None:
        scoping_context['organization_id'] = feedback_request.organization_id
        logger.info(f"Feedback scoping - explicit org from request: {feedback_request.organization_id}")

    if getattr(feedback_request, 'namespace_id', None) is not None:
        scoping_context['namespace_id'] = feedback_request.namespace_id
        logger.info(f"Feedback scoping - explicit namespace from request: {feedback_request.namespace_id}")

    return scoping_context

def is_organization_based_auth(auth_context: Dict[str, Any]) -> bool:
    """
    Check if the authentication is organization-based.

    Args:
        auth_context: Multi-tenant context from authentication

    Returns:
        True if organization-based, False if legacy
    """
    return not auth_context.get('is_legacy_auth', True)

def get_auth_scoping_summary(auth_context: Dict[str, Any]) -> str:
    """
    Get a human-readable summary of the authentication scoping.

    Args:
        auth_context: Multi-tenant context from authentication

    Returns:
        String summary of the scoping context
    """
    if is_organization_based_auth(auth_context):
        org_id = auth_context.get('organization_id', 'unknown')
        ns_id = auth_context.get('namespace_id', 'unknown')
        return f"Organization Auth (org: {org_id}, namespace: {ns_id})"
    else:
        return "Legacy Auth (backward compatible)"