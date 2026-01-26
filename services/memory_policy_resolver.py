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
    For node_constraints and edge_constraints, memory-level constraints are merged on top of schema-level.

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
        "edge_constraints": [],
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

    logger.info(f"Resolved memory policy: mode={merged['mode']}, consent={merged['consent']}, risk={merged['risk']}, "
                f"node_constraints={len(merged.get('node_constraints', []))}, edge_constraints={len(merged.get('edge_constraints', []))}")
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

    # Edge constraints: merge (memory constraints added to schema constraints)
    if layer.get("edge_constraints"):
        schema_edge_constraints = base.get("edge_constraints", [])
        memory_edge_constraints = layer["edge_constraints"]
        base["edge_constraints"] = _merge_edge_constraints(schema_edge_constraints, memory_edge_constraints)


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


def _merge_edge_constraints(
    schema_constraints: List[Dict[str, Any]],
    memory_constraints: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge edge constraints from schema and memory levels.

    Memory-level constraints for the same edge_type override schema-level.
    Schema-level constraints for edge_types not in memory-level are preserved.

    For constraints with the same edge_type but different source_type/target_type filters,
    we use a composite key: (edge_type, source_type, target_type).

    Args:
        schema_constraints: Constraints from schema
        memory_constraints: Constraints from memory (take precedence)

    Returns:
        Merged list of edge constraints
    """
    def _constraint_key(constraint: Dict[str, Any]) -> tuple:
        """Generate a unique key for an edge constraint."""
        return (
            constraint.get("edge_type"),
            constraint.get("source_type"),
            constraint.get("target_type")
        )

    # Build lookup of memory-level constraints by composite key
    memory_by_key = {}
    for constraint in memory_constraints:
        key = _constraint_key(constraint)
        memory_by_key[key] = constraint

    # Start with schema constraints, but skip those overridden by memory
    merged = []
    schema_keys = set()

    for constraint in schema_constraints:
        key = _constraint_key(constraint)
        schema_keys.add(key)

        if key in memory_by_key:
            # Memory-level overrides schema-level for this edge type + filters
            merged.append(memory_by_key[key])
        else:
            # Keep schema-level constraint
            merged.append(constraint)

    # Add memory-level constraints for edge types not in schema
    for key, constraint in memory_by_key.items():
        if key not in schema_keys:
            merged.append(constraint)

    return merged


def extract_type_level_constraints(schema: Any) -> Dict[str, Any]:
    """
    Extract node and edge constraints from schema type definitions.

    UserNodeType and UserRelationshipType can have optional `constraint` fields
    that define default behavior for those types. This function extracts them
    and converts them to policy-compatible constraint lists.

    Args:
        schema: Schema object or dict with node_types and relationship_types

    Returns:
        Dict with 'node_constraints' and 'edge_constraints' lists
    """
    node_constraints = []
    edge_constraints = []

    # Handle both object and dict forms
    if hasattr(schema, 'node_types'):
        node_types = schema.node_types
    elif isinstance(schema, dict):
        node_types = schema.get('node_types', {})
    else:
        node_types = {}

    if hasattr(schema, 'relationship_types'):
        rel_types = schema.relationship_types
    elif isinstance(schema, dict):
        rel_types = schema.get('relationship_types', {})
    else:
        rel_types = {}

    # Extract node constraints from UserNodeType.constraint
    for node_type_name, node_type_def in (node_types.items() if isinstance(node_types, dict) else []):
        constraint = None
        if hasattr(node_type_def, 'constraint') and node_type_def.constraint:
            constraint = node_type_def.constraint
        elif isinstance(node_type_def, dict) and node_type_def.get('constraint'):
            constraint = node_type_def['constraint']

        if constraint:
            # Convert to dict if it's a Pydantic model
            if hasattr(constraint, 'model_dump'):
                constraint_dict = constraint.model_dump(exclude_none=True)
            elif hasattr(constraint, 'dict'):
                constraint_dict = constraint.dict(exclude_none=True)
            else:
                constraint_dict = dict(constraint) if not isinstance(constraint, dict) else constraint

            # Ensure node_type is set (schema-level gets it from the key)
            if 'node_type' not in constraint_dict:
                constraint_dict['node_type'] = node_type_name

            node_constraints.append(constraint_dict)
            logger.debug(f"Extracted node constraint from {node_type_name}: {constraint_dict}")

    # Extract edge constraints from UserRelationshipType.constraint
    for rel_type_name, rel_type_def in (rel_types.items() if isinstance(rel_types, dict) else []):
        constraint = None
        if hasattr(rel_type_def, 'constraint') and rel_type_def.constraint:
            constraint = rel_type_def.constraint
        elif isinstance(rel_type_def, dict) and rel_type_def.get('constraint'):
            constraint = rel_type_def['constraint']

        if constraint:
            # Convert to dict if it's a Pydantic model
            if hasattr(constraint, 'model_dump'):
                constraint_dict = constraint.model_dump(exclude_none=True)
            elif hasattr(constraint, 'dict'):
                constraint_dict = constraint.dict(exclude_none=True)
            else:
                constraint_dict = dict(constraint) if not isinstance(constraint, dict) else constraint

            # Ensure edge_type is set (schema-level gets it from the key)
            if 'edge_type' not in constraint_dict:
                constraint_dict['edge_type'] = rel_type_name

            # Also extract source/target type info from the relationship type definition
            if 'source_type' not in constraint_dict:
                allowed_sources = None
                if hasattr(rel_type_def, 'allowed_source_types'):
                    allowed_sources = rel_type_def.allowed_source_types
                elif isinstance(rel_type_def, dict):
                    allowed_sources = rel_type_def.get('allowed_source_types')
                # If there's exactly one allowed source, use it
                if allowed_sources and len(allowed_sources) == 1:
                    constraint_dict['source_type'] = allowed_sources[0]

            if 'target_type' not in constraint_dict:
                allowed_targets = None
                if hasattr(rel_type_def, 'allowed_target_types'):
                    allowed_targets = rel_type_def.allowed_target_types
                elif isinstance(rel_type_def, dict):
                    allowed_targets = rel_type_def.get('allowed_target_types')
                # If there's exactly one allowed target, use it
                if allowed_targets and len(allowed_targets) == 1:
                    constraint_dict['target_type'] = allowed_targets[0]

            edge_constraints.append(constraint_dict)
            logger.debug(f"Extracted edge constraint from {rel_type_name}: {constraint_dict}")

    if node_constraints or edge_constraints:
        logger.info(f"Extracted {len(node_constraints)} node constraints, {len(edge_constraints)} edge constraints from schema types")

    return {
        'node_constraints': node_constraints,
        'edge_constraints': edge_constraints
    }


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

    Policy precedence (later overrides earlier):
    1. System defaults
    2. Type-level constraints (from UserNodeType.constraint, UserRelationshipType.constraint)
    3. Schema-level memory_policy
    4. Memory-level memory_policy (passed as parameter)

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
    type_level_constraints = None

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

            if schema:
                # Extract type-level constraints from node_types and relationship_types
                type_level_constraints = extract_type_level_constraints(schema)

                # Get schema-level memory_policy
                if hasattr(schema, 'memory_policy') and schema.memory_policy:
                    schema_policy = schema.memory_policy
                    logger.info(f"Loaded schema-level memory_policy from schema {schema_id}")
                elif isinstance(schema, dict) and schema.get('memory_policy'):
                    schema_policy = schema['memory_policy']
                    logger.info(f"Loaded schema-level memory_policy from schema {schema_id}")
                else:
                    logger.debug(f"Schema {schema_id} has no memory_policy defined")

        except Exception as e:
            logger.warning(f"Failed to fetch schema {schema_id} for policy resolution: {e}")

    # Build combined schema policy: type-level constraints + schema-level memory_policy
    # Type-level constraints are the base, schema memory_policy overrides them
    combined_schema_policy = None
    if type_level_constraints and (type_level_constraints.get('node_constraints') or type_level_constraints.get('edge_constraints')):
        combined_schema_policy = type_level_constraints
        if schema_policy:
            # Merge schema_policy on top of type-level constraints
            combined_schema_policy = merge_memory_policies(type_level_constraints, schema_policy)
    elif schema_policy:
        combined_schema_policy = schema_policy

    # Final merge: combined schema policy + memory-level policy
    return merge_memory_policies(combined_schema_policy, memory_policy)


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
