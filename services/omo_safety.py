"""
OMO (Open Memory Object) Safety Standards Implementation.

This module implements the Open Memory Object standard for:
- Consent tracking and enforcement
- Risk assessment and filtering
- ACL propagation from memory to nodes
- Audit trail creation for compliance

The OMO safety pipeline integrates at the graph extraction stage,
ensuring all memories (structured and unstructured) go through
the same safety standards.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


# ============================================================================
# Consent Enforcement
# ============================================================================

async def enforce_consent_standard(
    memory_content: str,
    memory_id: str,
    consent: str,
    nodes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Skip or annotate nodes based on consent level.

    Args:
        memory_content: The memory content (for logging)
        memory_id: The memory ID
        consent: Consent level ('explicit', 'implicit', 'terms', 'none')
        nodes: List of extracted nodes to process

    Returns:
        Processed nodes (may be empty if consent=none)
    """
    if consent == "none":
        logger.warning(
            f"Memory {memory_id} has consent='none' - skipping graph extraction. "
            "Memories with no consent should not have graph nodes created."
        )
        return []  # Don't extract nodes without consent

    # Annotate all nodes with consent provenance
    for node in nodes:
        if "properties" not in node:
            node["properties"] = {}
        node["properties"]["_omo_consent"] = consent
        node["properties"]["_omo_source_memory_id"] = memory_id

    logger.debug(
        f"Applied consent standard to {len(nodes)} nodes from memory {memory_id}: consent={consent}"
    )
    return nodes


# ============================================================================
# Risk Assessment
# ============================================================================

async def enforce_risk_standard(
    memory_id: str,
    risk: str,
    external_user_id: Optional[str],
    nodes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Apply stricter constraints for high-risk content.

    For 'flagged' content:
    - Mark nodes as requiring review
    - Restrict ACL to only the memory owner

    For 'sensitive' content:
    - Mark nodes as sensitive
    - No additional restrictions (normal ACL applies)

    Args:
        memory_id: The memory ID
        risk: Risk level ('none', 'sensitive', 'flagged')
        external_user_id: The external user ID (for ACL restriction)
        nodes: List of nodes to process

    Returns:
        Processed nodes with risk annotations
    """
    if risk == "flagged":
        logger.warning(
            f"Memory {memory_id} has risk='flagged' - restricting ACL to owner only"
        )
        for node in nodes:
            if "properties" not in node:
                node["properties"] = {}
            node["properties"]["_omo_risk"] = "flagged"
            node["properties"]["_omo_requires_review"] = True

            # Restrict ACL to only the memory owner
            if external_user_id:
                node["acl"] = {
                    "read": [external_user_id],
                    "write": [external_user_id]
                }

    elif risk == "sensitive":
        logger.info(
            f"Memory {memory_id} has risk='sensitive' - marking nodes as sensitive"
        )
        for node in nodes:
            if "properties" not in node:
                node["properties"] = {}
            node["properties"]["_omo_risk"] = "sensitive"

    else:
        # risk == "none" - no special handling needed
        for node in nodes:
            if "properties" not in node:
                node["properties"] = {}
            node["properties"]["_omo_risk"] = "none"

    return nodes


# ============================================================================
# ACL Propagation
# ============================================================================

async def propagate_acl(
    memory_id: str,
    acl: Optional[Dict[str, List[str]]],
    external_user_id: Optional[str],
    developer_user_id: Optional[str],
    nodes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Propagate ACL from memory to extracted nodes.

    If acl is provided, use it directly.
    Otherwise, create default ACL based on external_user_id and developer_user_id.

    Args:
        memory_id: The memory ID
        acl: Explicit ACL configuration {'read': [...], 'write': [...]}
        external_user_id: The external user ID
        developer_user_id: The developer's user ID
        nodes: List of nodes to process

    Returns:
        Nodes with ACL propagated
    """
    # Skip if nodes already have restricted ACL (from risk enforcement)
    nodes_with_acl = [n for n in nodes if "acl" in n]
    if nodes_with_acl:
        logger.debug(
            f"Skipping ACL propagation for {len(nodes_with_acl)} nodes that already have ACL set"
        )
        return nodes

    # Determine ACL to apply
    if acl:
        acl = acl
        logger.debug(f"Using explicit acl for memory {memory_id}: {acl}")
    else:
        # Default ACL: developer + external_user can read/write
        read_access = []
        write_access = []

        if external_user_id:
            read_access.append(external_user_id)
            write_access.append(external_user_id)

        if developer_user_id and developer_user_id not in read_access:
            read_access.append(developer_user_id)
            write_access.append(developer_user_id)

        acl = {"read": read_access, "write": write_access}
        logger.debug(f"Using default ACL for memory {memory_id}: {acl}")

    # Apply ACL to all nodes
    for node in nodes:
        node["acl"] = acl

    return nodes


# ============================================================================
# Audit Trail
# ============================================================================

async def create_audit_trail(
    memory_id: str,
    consent: str,
    risk: str,
    extraction_method: str,
    nodes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Add audit trail to nodes for compliance tracking.

    Args:
        memory_id: The memory ID
        consent: Consent level used
        risk: Risk level assessed
        extraction_method: 'llm' or 'manual' (structured)
        nodes: List of nodes to process

    Returns:
        Nodes with audit trail added
    """
    audit_timestamp = datetime.now(timezone.utc).isoformat()

    for node in nodes:
        if "properties" not in node:
            node["properties"] = {}

        node["properties"]["_omo_audit"] = {
            "source_memory_id": memory_id,
            "extracted_at": audit_timestamp,
            "consent": consent,
            "risk": risk,
            "extraction_method": extraction_method
        }

    logger.debug(
        f"Created audit trail for {len(nodes)} nodes from memory {memory_id}"
    )
    return nodes


# ============================================================================
# Main Pipeline
# ============================================================================

async def process_memory_with_omo(
    memory_id: str,
    memory_content: str,
    extracted_nodes: List[Dict[str, Any]],
    consent: str = "implicit",
    risk: str = "none",
    acl: Optional[Dict[str, List[str]]] = None,
    external_user_id: Optional[str] = None,
    developer_user_id: Optional[str] = None,
    extraction_method: str = "llm"
) -> List[Dict[str, Any]]:
    """
    Apply OMO safety standards pipeline to extracted nodes.

    This is the main entry point for the OMO safety pipeline.
    Call this after graph extraction, before storing nodes.

    Args:
        memory_id: The memory ID
        memory_content: The memory content (for logging)
        extracted_nodes: Nodes extracted from the memory
        consent: Consent level ('explicit', 'implicit', 'terms', 'none')
        risk: Risk level ('none', 'sensitive', 'flagged')
        acl: Optional explicit ACL configuration
        external_user_id: The external user ID
        developer_user_id: The developer's user ID
        extraction_method: 'llm' (auto mode) or 'manual' (manual mode)

    Returns:
        Processed nodes with OMO annotations, or empty list if consent='none'
    """
    logger.info(
        f"Processing memory {memory_id} with OMO safety pipeline: "
        f"consent={consent}, risk={risk}, nodes={len(extracted_nodes)}"
    )

    # 1. Consent check (may skip extraction)
    nodes = await enforce_consent_standard(
        memory_content, memory_id, consent, extracted_nodes
    )
    if not nodes:
        return []

    # 2. Risk assessment (may restrict ACL)
    nodes = await enforce_risk_standard(
        memory_id, risk, external_user_id, nodes
    )

    # 3. ACL propagation (memory ACL → node ACL)
    nodes = await propagate_acl(
        memory_id, acl, external_user_id, developer_user_id, nodes
    )

    # 4. Audit trail (compliance tracking)
    nodes = await create_audit_trail(
        memory_id, consent, risk, extraction_method, nodes
    )

    logger.info(
        f"OMO safety pipeline complete for memory {memory_id}: "
        f"{len(nodes)} nodes processed"
    )

    return nodes


# ============================================================================
# Utility Functions
# ============================================================================

def validate_consent_level(consent: str) -> bool:
    """Validate that consent level is a valid OMO value."""
    valid_levels = {"explicit", "implicit", "terms", "none"}
    return consent in valid_levels


def validate_risk_level(risk: str) -> bool:
    """Validate that risk level is a valid OMO value."""
    valid_levels = {"none", "sensitive", "flagged"}
    return risk in valid_levels


def get_extraction_method_from_policy_mode(mode: str) -> str:
    """
    Convert PolicyMode to extraction method for audit trail.

    Args:
        mode: PolicyMode value ('auto', 'manual'). Deprecated: 'structured' → 'manual', 'hybrid' → 'auto'

    Returns:
        'manual' for manual/structured mode, 'llm' for auto/hybrid mode
    """
    return "manual" if mode in ("manual", "structured") else "llm"
