"""
Memory Policy Resolver Service.

This module handles resolution and merging of memory policies from different sources:
1. Schema-level policies (defaults for all memories using a schema)
2. Memory-level policies (overrides for specific memories)

The precedence is: Memory-level > Schema-level > System defaults
"""

from typing import Any, Dict, List, Optional
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


# System defaults for OMO safety
DEFAULT_CONSENT = "implicit"
DEFAULT_RISK = "none"
DEFAULT_MODE = "auto"


def merge_memory_policies(
    schema_policy: Optional[Dict[str, Any]],
    memory_policy: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge schema-level and memory-level policies.

    Memory-level policy takes precedence over schema-level.
    For node_constraints, memory-level constraints are merged on top of schema-level.

    Args:
        schema_policy: Policy defined at schema level (defaults)
        memory_policy: Policy defined at memory level (overrides)

    Returns:
        Merged policy dictionary
    """
    # Start with system defaults
    merged = {
        "mode": DEFAULT_MODE,
        "consent": DEFAULT_CONSENT,
        "risk": DEFAULT_RISK,
        "node_constraints": [],
        "nodes": None,
        "relationships": None,
        "acl": None
    }

    # Apply schema-level policy
    if schema_policy:
        logger.debug(f"Applying schema-level policy: {schema_policy}")
        _apply_policy_layer(merged, schema_policy)

    # Apply memory-level policy (overrides schema)
    if memory_policy:
        logger.debug(f"Applying memory-level policy (override): {memory_policy}")
        _apply_policy_layer(merged, memory_policy)

    logger.info(f"Resolved memory policy: mode={merged['mode']}, consent={merged['consent']}, risk={merged['risk']}")
    return merged


def _apply_policy_layer(base: Dict[str, Any], layer: Dict[str, Any]) -> None:
    """
    Apply a policy layer on top of base policy (mutates base).

    Args:
        base: Base policy to modify
        layer: Policy layer to apply
    """
    # Mode: direct override
    if layer.get("mode"):
        base["mode"] = layer["mode"]

    # OMO safety fields: direct override
    if layer.get("consent"):
        base["consent"] = layer["consent"]
    if layer.get("risk"):
        base["risk"] = layer["risk"]
    if layer.get("acl"):
        base["acl"] = layer["acl"]

    # Structured mode fields: direct override
    if layer.get("nodes") is not None:
        base["nodes"] = layer["nodes"]
    if layer.get("relationships") is not None:
        base["relationships"] = layer["relationships"]

    # Node constraints: merge (memory constraints added to schema constraints)
    if layer.get("node_constraints"):
        schema_constraints = base.get("node_constraints", [])
        memory_constraints = layer["node_constraints"]
        base["node_constraints"] = _merge_node_constraints(schema_constraints, memory_constraints)


def _merge_node_constraints(
    schema_constraints: List[Dict[str, Any]],
    memory_constraints: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge node constraints from schema and memory levels.

    Memory-level constraints for the same node_type override schema-level.
    Schema-level constraints for node_types not in memory-level are preserved.

    Args:
        schema_constraints: Constraints from schema
        memory_constraints: Constraints from memory (take precedence)

    Returns:
        Merged list of node constraints
    """
    # Build lookup of memory-level constraints by node_type
    memory_by_type = {}
    for constraint in memory_constraints:
        node_type = constraint.get("node_type")
        if node_type:
            memory_by_type[node_type] = constraint

    # Start with schema constraints, but skip those overridden by memory
    merged = []
    schema_node_types = set()

    for constraint in schema_constraints:
        node_type = constraint.get("node_type")
        schema_node_types.add(node_type)

        if node_type in memory_by_type:
            # Memory-level overrides schema-level for this node_type
            merged.append(memory_by_type[node_type])
        else:
            # Keep schema-level constraint
            merged.append(constraint)

    # Add memory-level constraints for node_types not in schema
    for node_type, constraint in memory_by_type.items():
        if node_type not in schema_node_types:
            merged.append(constraint)

    return merged


async def resolve_memory_policy_from_schema(
    memory_graph,
    schema_id: Optional[str],
    memory_policy: Optional[Dict[str, Any]],
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Resolve the effective memory policy by fetching schema-level policy if needed.

    Args:
        memory_graph: MemoryGraph instance for fetching schema
        schema_id: Optional schema ID to fetch policy from
        memory_policy: Optional memory-level policy (overrides schema)
        user_id: User ID for schema access
        workspace_id: Workspace ID for schema access
        organization_id: Organization ID for schema access
        namespace_id: Namespace ID for schema access
        api_key: API key for authentication

    Returns:
        Resolved memory policy dictionary
    """
    schema_policy = None

    # Fetch schema-level policy if schema_id is provided
    if schema_id:
        try:
            schema = await memory_graph.get_user_schema_async(
                schema_id=schema_id,
                user_id=user_id,
                workspace_id=workspace_id,
                organization_id=organization_id,
                namespace_id=namespace_id,
                api_key=api_key
            )

            if schema and hasattr(schema, 'memory_policy') and schema.memory_policy:
                schema_policy = schema.memory_policy
                logger.info(f"Loaded schema-level memory_policy from schema {schema_id}")
            elif schema and isinstance(schema, dict) and schema.get('memory_policy'):
                schema_policy = schema['memory_policy']
                logger.info(f"Loaded schema-level memory_policy from schema {schema_id}")
            else:
                logger.debug(f"Schema {schema_id} has no memory_policy defined")

        except Exception as e:
            logger.warning(f"Failed to fetch schema {schema_id} for policy resolution: {e}")

    # Merge schema and memory policies
    return merge_memory_policies(schema_policy, memory_policy)


def extract_omo_fields_from_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract OMO safety fields from a resolved policy.

    Args:
        policy: Resolved memory policy

    Returns:
        Dictionary with consent, risk, and acl fields
    """
    return {
        "consent": policy.get("consent", DEFAULT_CONSENT),
        "risk": policy.get("risk", DEFAULT_RISK),
        "acl": policy.get("acl")
    }


def should_skip_graph_extraction(policy: Dict[str, Any]) -> bool:
    """
    Check if graph extraction should be skipped based on policy.

    Args:
        policy: Resolved memory policy

    Returns:
        True if extraction should be skipped (consent='none')
    """
    return policy.get("consent") == "none"
