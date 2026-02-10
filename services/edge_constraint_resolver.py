"""
Edge Constraint Resolver Service.

This module handles the application of edge constraints during graph generation.
It determines whether edges should be created and what properties they should have
based on the configured EdgeConstraint policies.

The edge constraint system mirrors the node constraint system, allowing developers to:
1. Control edge creation (auto vs. never - controlled vocabulary)
2. Define how to find existing target nodes (via search config)
3. Set edge properties (exact values or auto-extracted)
4. Apply constraints conditionally (via when clause)
5. Filter by source/target node types
"""

from typing import Any, Dict, List, Optional, Tuple
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


async def apply_edge_constraints(
    source_node: Dict[str, Any],
    target_node: Dict[str, Any],
    edge_type: str,
    edge_constraints: List[Dict[str, Any]],
    memory_graph: Any,
    extracted_edge_properties: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Apply edge constraints to determine if an edge should be created and with what properties.

    This function evaluates applicable edge constraints and determines:
    1. Whether the edge should be created at all (based on create policy)
    2. What target node to link to (existing vs. new)
    3. What properties the edge should have

    Args:
        source_node: The source node of the edge (dict with 'type', 'properties', etc.)
        target_node: The proposed target node (may be replaced if existing found)
        edge_type: The type of edge being created (e.g., 'MITIGATES', 'ASSIGNED_TO')
        edge_constraints: List of EdgeConstraint dicts from resolved memory policy
        memory_graph: MemoryGraph instance for searching existing nodes
        extracted_edge_properties: Properties extracted by LLM for this edge
        context: Additional context for condition evaluation (metadata, etc.)

    Returns:
        Tuple of:
        - should_create (bool): Whether the edge should be created
        - final_target (Optional[Dict]): The final target node (may be existing node or None)
        - final_properties (Optional[Dict]): The final edge properties to use
    """
    if not edge_constraints:
        # No constraints - allow edge creation with extracted properties
        return True, target_node, extracted_edge_properties

    # Find applicable constraint for this edge
    constraint = _find_applicable_constraint(
        edge_type=edge_type,
        source_type=source_node.get("type"),
        target_type=target_node.get("type"),
        edge_constraints=edge_constraints,
        edge_properties=extracted_edge_properties,
        context=context
    )

    if not constraint:
        # No matching constraint - allow edge creation with extracted properties
        logger.debug(f"No constraint found for edge {edge_type}, allowing creation")
        return True, target_node, extracted_edge_properties

    logger.info(f"Applying edge constraint for {edge_type}: create={constraint.get('create', 'upsert')}")

    # Evaluate 'when' condition if present
    if constraint.get("when"):
        condition_met = _evaluate_when_condition(
            condition=constraint["when"],
            edge_properties=extracted_edge_properties,
            source_node=source_node,
            target_node=target_node,
            context=context
        )
        if not condition_met:
            logger.debug(f"Edge constraint 'when' condition not met for {edge_type}")
            return True, target_node, extracted_edge_properties

    # Apply search to find existing target node
    final_target = target_node
    if constraint.get("search"):
        existing_target = await _search_for_target(
            search_config=constraint["search"],
            target_node=target_node,
            memory_graph=memory_graph,
            context=context
        )
        if existing_target:
            logger.info(f"Found existing target node for {edge_type}")
            final_target = existing_target

    # Apply create policy with backwards compatibility
    create_policy = constraint.get("create", "upsert")

    # Backwards compatibility: map old values to new
    if create_policy == "auto":
        create_policy = "upsert"
    elif create_policy == "never":
        create_policy = "lookup"

    # Handle on_miss if specified (overrides create policy behavior)
    on_miss = constraint.get("on_miss")

    if create_policy == "lookup":
        if final_target == target_node:
            # No existing target found and create='lookup' - check on_miss behavior
            if on_miss == "error":
                raise ValueError(f"Target node for edge {edge_type} not found and on_miss='error'. "
                               f"No existing node matched the search criteria.")
            # Default lookup behavior: skip edge creation
            logger.info(f"Edge {edge_type} skipped: create='lookup' and no existing target found")
            return False, None, None
        # else: existing target found - proceed with edge creation
    elif on_miss == "error" and final_target == target_node:
        # upsert with on_miss='error' should fail if target not found
        raise ValueError(f"Target node for edge {edge_type} not found and on_miss='error'. "
                        f"No existing node matched the search criteria.")

    # Apply 'set' values to edge properties
    final_properties = _apply_set_values(
        constraint=constraint,
        extracted_properties=extracted_edge_properties
    )

    return True, final_target, final_properties


def _find_applicable_constraint(
    edge_type: str,
    source_type: Optional[str],
    target_type: Optional[str],
    edge_constraints: List[Dict[str, Any]],
    edge_properties: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Find the most applicable edge constraint for the given edge.

    Priority:
    1. Exact match on edge_type + source_type + target_type
    2. Match on edge_type + source_type (any target)
    3. Match on edge_type + target_type (any source)
    4. Match on edge_type only

    Args:
        edge_type: The edge type to match
        source_type: The source node type
        target_type: The target node type
        edge_constraints: List of constraints to search
        edge_properties: Edge properties for condition evaluation
        context: Additional context

    Returns:
        The most applicable constraint or None
    """
    candidates = []

    for constraint in edge_constraints:
        constraint_edge_type = constraint.get("edge_type")
        constraint_source_type = constraint.get("source_type")
        constraint_target_type = constraint.get("target_type")

        # Check edge_type match
        if constraint_edge_type and constraint_edge_type != edge_type:
            continue

        # Calculate specificity score
        specificity = 0
        if constraint_edge_type == edge_type:
            specificity += 1

        # Check source_type filter
        if constraint_source_type:
            if source_type and constraint_source_type == source_type:
                specificity += 2
            elif source_type:
                continue  # Source type specified but doesn't match

        # Check target_type filter
        if constraint_target_type:
            if target_type and constraint_target_type == target_type:
                specificity += 2
            elif target_type:
                continue  # Target type specified but doesn't match

        candidates.append((specificity, constraint))

    if not candidates:
        return None

    # Return the most specific constraint
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _evaluate_when_condition(
    condition: Dict[str, Any],
    edge_properties: Optional[Dict[str, Any]],
    source_node: Dict[str, Any],
    target_node: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Evaluate a 'when' condition to determine if constraint should apply.

    Supports logical operators: _and, _or, _not

    Args:
        condition: The condition dict (may include _and, _or, _not operators)
        edge_properties: Properties of the edge being evaluated
        source_node: Source node for context
        target_node: Target node for context
        context: Additional context

    Returns:
        True if condition is met, False otherwise
    """
    if not condition:
        return True

    props = edge_properties or {}

    # Handle logical operators
    if "_and" in condition:
        return all(
            _evaluate_when_condition(sub, edge_properties, source_node, target_node, context)
            for sub in condition["_and"]
        )

    if "_or" in condition:
        return any(
            _evaluate_when_condition(sub, edge_properties, source_node, target_node, context)
            for sub in condition["_or"]
        )

    if "_not" in condition:
        return not _evaluate_when_condition(
            condition["_not"], edge_properties, source_node, target_node, context
        )

    # Simple property matching
    for key, expected_value in condition.items():
        if key.startswith("_"):
            continue  # Skip operators

        actual_value = props.get(key)
        if actual_value != expected_value:
            return False

    return True


async def _search_for_target(
    search_config: Dict[str, Any],
    target_node: Dict[str, Any],
    memory_graph: Any,
    context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Search for an existing target node using the search configuration.

    Args:
        search_config: SearchConfig dict with properties and mode
        target_node: The proposed target node (used for extracting search values)
        memory_graph: MemoryGraph instance for searching
        context: Additional context

    Returns:
        Existing node dict if found, None otherwise
    """
    properties = search_config.get("properties", [])
    target_props = target_node.get("properties", {})
    target_type = target_node.get("type")

    for prop_match in properties:
        if isinstance(prop_match, str):
            # String shorthand - exact match
            prop_name = prop_match
            mode = "exact"
            threshold = None
            value = prop_match.get("value") if isinstance(prop_match, dict) else target_props.get(prop_name)
        else:
            prop_name = prop_match.get("name")
            mode = prop_match.get("mode", "exact")
            threshold = prop_match.get("threshold", 0.85)
            value = prop_match.get("value") or target_props.get(prop_name)

        if not value:
            continue

        try:
            if mode == "exact":
                # Exact property match
                existing = await memory_graph.find_node_by_property(
                    node_type=target_type,
                    property_name=prop_name,
                    property_value=value,
                    context=context
                )
            elif mode == "semantic":
                # Semantic similarity search
                existing = await memory_graph.find_node_by_semantic_match(
                    node_type=target_type,
                    property_name=prop_name,
                    query_text=str(value),
                    threshold=threshold,
                    context=context
                )
            elif mode == "fuzzy":
                # Fuzzy string match
                existing = await memory_graph.find_node_by_fuzzy_match(
                    node_type=target_type,
                    property_name=prop_name,
                    query_text=str(value),
                    threshold=threshold,
                    context=context
                )
            else:
                continue

            if existing:
                logger.debug(f"Found existing target via {mode} match on {prop_name}")
                return existing

        except Exception as e:
            logger.warning(f"Error searching for target node: {e}")
            continue

    return None


def _apply_set_values(
    constraint: Dict[str, Any],
    extracted_properties: Optional[Dict[str, Any]],
    existing_edge_properties: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply 'set' values from constraint to edge properties.

    Supports three text modes for auto-extract values:
    - replace (default): Overwrite existing value with extracted value
    - append: Add extracted value to end of existing value (for text fields)
    - merge: Intelligently combine extracted value with existing (for summaries)

    Args:
        constraint: The edge constraint with optional 'set' field
        extracted_properties: Properties extracted by LLM
        existing_edge_properties: Properties from existing edge (for append/merge)

    Returns:
        Final merged properties
    """
    final_props = dict(extracted_properties or {})
    set_values = constraint.get("set")
    existing_props = existing_edge_properties or {}

    if not set_values:
        return final_props

    for prop_name, prop_value in set_values.items():
        if isinstance(prop_value, dict):
            mode = prop_value.get("mode")
            text_mode = prop_value.get("text_mode", "replace")

            if mode == "auto":
                # Auto-extract: use LLM-extracted value if available
                extracted_value = final_props.get(prop_name)
                existing_value = existing_props.get(prop_name)

                if extracted_value is not None:
                    if text_mode == "append" and existing_value is not None:
                        # Append: Add to end of existing text
                        if isinstance(existing_value, str) and isinstance(extracted_value, str):
                            final_props[prop_name] = f"{existing_value}\n{extracted_value}"
                        elif isinstance(existing_value, list):
                            final_props[prop_name] = existing_value + [extracted_value]
                        else:
                            final_props[prop_name] = extracted_value
                    elif text_mode == "merge" and existing_value is not None:
                        # Merge: Intelligently combine (for summaries, descriptions)
                        # Mark for LLM merge during graph generation
                        final_props[prop_name] = {
                            "_merge": True,
                            "existing": existing_value,
                            "new": extracted_value
                        }
                    else:
                        # Replace (default): Use extracted value as-is
                        final_props[prop_name] = extracted_value
                elif existing_value is not None and text_mode in ("append", "merge"):
                    # No new value but keep existing for append/merge modes
                    final_props[prop_name] = existing_value
                # If neither extracted nor existing, leave unset
        else:
            # Exact value override
            final_props[prop_name] = prop_value

    return final_props


def get_edge_constraints_for_type(
    edge_type: str,
    edge_constraints: List[Dict[str, Any]],
    source_type: Optional[str] = None,
    target_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all edge constraints that could apply to a given edge type.

    Useful for preprocessing or validation.

    Args:
        edge_type: The edge type to match
        edge_constraints: All edge constraints from policy
        source_type: Optional source type filter
        target_type: Optional target type filter

    Returns:
        List of potentially applicable constraints
    """
    applicable = []

    for constraint in edge_constraints:
        constraint_edge_type = constraint.get("edge_type")
        constraint_source_type = constraint.get("source_type")
        constraint_target_type = constraint.get("target_type")

        # Edge type must match (or be None for wildcard)
        if constraint_edge_type and constraint_edge_type != edge_type:
            continue

        # Source type filter
        if constraint_source_type and source_type:
            if constraint_source_type != source_type:
                continue

        # Target type filter
        if constraint_target_type and target_type:
            if constraint_target_type != target_type:
                continue

        applicable.append(constraint)

    return applicable


def validate_edge_constraints(edge_constraints: List[Dict[str, Any]]) -> List[str]:
    """
    Validate a list of edge constraints and return any errors.

    Args:
        edge_constraints: List of edge constraint dicts

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for i, constraint in enumerate(edge_constraints):
        prefix = f"edge_constraints[{i}]"

        # Check required field at memory level
        if not constraint.get("edge_type"):
            errors.append(f"{prefix}: edge_type is required at memory level")

        # Validate create value (including backwards-compatible values)
        create_value = constraint.get("create", "upsert")
        valid_create_values = ("upsert", "lookup", "auto", "never")
        if create_value not in valid_create_values:
            errors.append(f"{prefix}: create must be 'upsert' or 'lookup' (or deprecated 'auto'/'never'), got '{create_value}'")

        # Validate direction value
        direction = constraint.get("direction", "outgoing")
        if direction not in ("outgoing", "incoming", "both"):
            errors.append(f"{prefix}: direction must be 'outgoing', 'incoming', or 'both'")

        # Validate search config if present
        search = constraint.get("search")
        if search:
            if not isinstance(search, dict):
                errors.append(f"{prefix}.search: must be a dictionary")
            elif search.get("properties") and not isinstance(search["properties"], list):
                errors.append(f"{prefix}.search.properties: must be a list")

        # Validate when clause if present
        when = constraint.get("when")
        if when:
            when_errors = _validate_when_clause(when, prefix)
            errors.extend(when_errors)

    return errors


def _validate_when_clause(when: Any, prefix: str) -> List[str]:
    """Validate a 'when' clause recursively."""
    errors = []

    if not isinstance(when, dict):
        errors.append(f"{prefix}.when: must be a dictionary")
        return errors

    valid_operators = {"_and", "_or", "_not"}

    for key in when.keys():
        if key.startswith("_") and key not in valid_operators:
            errors.append(f"{prefix}.when: unknown operator '{key}'")

    if "_and" in when:
        if not isinstance(when["_and"], list):
            errors.append(f"{prefix}.when._and: must be a list")
        else:
            for j, sub in enumerate(when["_and"]):
                errors.extend(_validate_when_clause(sub, f"{prefix}.when._and[{j}]"))

    if "_or" in when:
        if not isinstance(when["_or"], list):
            errors.append(f"{prefix}.when._or: must be a list")
        else:
            for j, sub in enumerate(when["_or"]):
                errors.extend(_validate_when_clause(sub, f"{prefix}.when._or[{j}]"))

    if "_not" in when:
        if not isinstance(when["_not"], dict):
            errors.append(f"{prefix}.when._not: must be a dictionary")
        else:
            errors.extend(_validate_when_clause(when["_not"], f"{prefix}.when._not"))

    return errors
