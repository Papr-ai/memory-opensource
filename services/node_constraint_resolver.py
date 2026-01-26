"""
Node Constraint Resolver Service.

This module handles the application of node constraints during graph generation.
It determines whether nodes should be created and what properties they should have
based on the configured NodeConstraint policies.

The node constraint system allows developers to:
1. Control node creation (auto vs. never - controlled vocabulary)
2. Define how to find existing nodes (via search config)
3. Set node properties (exact values or auto-extracted)
4. Apply constraints conditionally (via when clause)
5. Filter by node type
"""

from typing import Any, Dict, List, Optional, Tuple
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


async def apply_node_constraints(
    node: Dict[str, Any],
    node_type: str,
    node_constraints: List[Dict[str, Any]],
    memory_graph: Any,
    extracted_node_properties: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Apply node constraints to determine if a node should be created and with what properties.

    This function evaluates applicable node constraints and determines:
    1. Whether the node should be created at all (based on create policy)
    2. What existing node to use (if search finds one)
    3. What properties the node should have

    Args:
        node: The proposed node (dict with 'type', 'properties', etc.)
        node_type: The type of node being created (e.g., 'TacticDef', 'Person')
        node_constraints: List of NodeConstraint dicts from resolved memory policy
        memory_graph: MemoryGraph instance for searching existing nodes
        extracted_node_properties: Properties extracted by LLM for this node
        context: Additional context for condition evaluation (metadata, etc.)

    Returns:
        Tuple of:
        - should_create (bool): Whether a new node should be created
        - existing_node (Optional[Dict]): The existing node if found (None if creating new)
        - final_properties (Optional[Dict]): The final node properties to use
    """
    if not node_constraints:
        # No constraints - allow node creation with extracted properties
        return True, None, extracted_node_properties

    # Find applicable constraint for this node
    constraint = _find_applicable_constraint(
        node_type=node_type,
        node_constraints=node_constraints,
        node_properties=extracted_node_properties,
        context=context
    )

    if not constraint:
        # No matching constraint - allow node creation with extracted properties
        logger.debug(f"No constraint found for node {node_type}, allowing creation")
        return True, None, extracted_node_properties

    logger.info(f"Applying node constraint for {node_type}: create={constraint.get('create', 'auto')}")

    # Evaluate 'when' condition if present
    if constraint.get("when"):
        condition_met = _evaluate_when_condition(
            condition=constraint["when"],
            node_properties=extracted_node_properties,
            context=context
        )
        if not condition_met:
            logger.debug(f"Node constraint 'when' condition not met for {node_type}")
            return True, None, extracted_node_properties

    # Apply search to find existing node
    existing_node = None
    if constraint.get("search"):
        existing_node = await _search_for_existing_node(
            search_config=constraint["search"],
            node=node,
            node_type=node_type,
            memory_graph=memory_graph,
            context=context
        )
        if existing_node:
            logger.info(f"Found existing node for {node_type}")

    # Apply create policy
    create_policy = constraint.get("create", "auto")
    if create_policy == "never":
        if not existing_node:
            # No existing node found and create='never' - skip node creation
            logger.info(f"Node {node_type} skipped: create='never' and no existing node found")
            return False, None, None
        # Existing node found - use it instead of creating
        return False, existing_node, None

    # Apply 'set' values to node properties
    final_properties = _apply_set_values(
        constraint=constraint,
        extracted_properties=extracted_node_properties
    )

    # If we found an existing node but create='auto', we can merge/update
    if existing_node:
        return False, existing_node, final_properties

    return True, None, final_properties


def _find_applicable_constraint(
    node_type: str,
    node_constraints: List[Dict[str, Any]],
    node_properties: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Find the most applicable node constraint for the given node.

    Args:
        node_type: The node type to match
        node_constraints: List of constraints to search
        node_properties: Node properties for condition evaluation
        context: Additional context

    Returns:
        The most applicable constraint or None
    """
    for constraint in node_constraints:
        constraint_node_type = constraint.get("node_type")

        # Check node_type match
        if constraint_node_type and constraint_node_type != node_type:
            continue

        # Found a match
        return constraint

    return None


def _evaluate_when_condition(
    condition: Dict[str, Any],
    node_properties: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Evaluate a 'when' condition to determine if constraint should apply.

    Supports logical operators: _and, _or, _not

    Args:
        condition: The condition dict (may include _and, _or, _not operators)
        node_properties: Properties of the node being evaluated
        context: Additional context

    Returns:
        True if condition is met, False otherwise
    """
    if not condition:
        return True

    props = node_properties or {}

    # Handle logical operators
    if "_and" in condition:
        return all(
            _evaluate_when_condition(sub, node_properties, context)
            for sub in condition["_and"]
        )

    if "_or" in condition:
        return any(
            _evaluate_when_condition(sub, node_properties, context)
            for sub in condition["_or"]
        )

    if "_not" in condition:
        return not _evaluate_when_condition(
            condition["_not"], node_properties, context
        )

    # Simple property matching
    for key, expected_value in condition.items():
        if key.startswith("_"):
            continue  # Skip operators

        actual_value = props.get(key)
        if actual_value != expected_value:
            return False

    return True


async def _search_for_existing_node(
    search_config: Dict[str, Any],
    node: Dict[str, Any],
    node_type: str,
    memory_graph: Any,
    context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Search for an existing node using the search configuration.

    Args:
        search_config: SearchConfig dict with properties and mode
        node: The proposed node (used for extracting search values)
        node_type: The node type to search for
        memory_graph: MemoryGraph instance for searching
        context: Additional context

    Returns:
        Existing node dict if found, None otherwise
    """
    properties = search_config.get("properties", [])
    node_props = node.get("properties", {})

    for prop_match in properties:
        if isinstance(prop_match, str):
            # String shorthand - exact match
            prop_name = prop_match
            mode = "exact"
            threshold = None
            value = node_props.get(prop_name)
        else:
            prop_name = prop_match.get("name")
            mode = prop_match.get("mode", "exact")
            threshold = prop_match.get("threshold", 0.85)
            value = prop_match.get("value") or node_props.get(prop_name)

        if not value:
            continue

        try:
            if mode == "exact":
                # Exact property match
                existing = await memory_graph.find_node_by_property(
                    node_type=node_type,
                    property_name=prop_name,
                    property_value=value,
                    context=context
                )
            elif mode == "semantic":
                # Semantic similarity search
                existing = await memory_graph.find_node_by_semantic_match(
                    node_type=node_type,
                    property_name=prop_name,
                    query_text=str(value),
                    threshold=threshold,
                    context=context
                )
            elif mode == "fuzzy":
                # Fuzzy string match
                existing = await memory_graph.find_node_by_fuzzy_match(
                    node_type=node_type,
                    property_name=prop_name,
                    query_text=str(value),
                    threshold=threshold,
                    context=context
                )
            else:
                continue

            if existing:
                logger.debug(f"Found existing node via {mode} match on {prop_name}")
                return existing

        except Exception as e:
            logger.warning(f"Error searching for existing node: {e}")
            continue

    # Also try via_relationship if configured
    via_relationships = search_config.get("via_relationship", [])
    for via_rel in via_relationships:
        try:
            existing = await _search_via_relationship(
                via_config=via_rel,
                node=node,
                node_type=node_type,
                memory_graph=memory_graph,
                context=context
            )
            if existing:
                return existing
        except Exception as e:
            logger.warning(f"Error searching via relationship: {e}")
            continue

    return None


async def _search_via_relationship(
    via_config: Dict[str, Any],
    node: Dict[str, Any],
    node_type: str,
    memory_graph: Any,
    context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Search for an existing node via its relationships.

    Args:
        via_config: RelationshipMatch config
        node: The proposed node
        node_type: The node type
        memory_graph: MemoryGraph instance
        context: Additional context

    Returns:
        Existing node if found, None otherwise
    """
    edge_type = via_config.get("edge_type")
    target_type = via_config.get("target_type")
    target_search = via_config.get("target_search", {})
    direction = via_config.get("direction", "outgoing")

    if not edge_type or not target_type:
        return None

    try:
        # Search for target node first using target_search config
        target_properties = target_search.get("properties", [])
        if not target_properties:
            return None

        # Find the target node
        target_node = await _search_for_existing_node(
            search_config=target_search,
            node={"properties": node.get("properties", {})},
            node_type=target_type,
            memory_graph=memory_graph,
            context=context
        )

        if not target_node:
            return None

        # Now find a node of our type connected to this target via the edge
        existing = await memory_graph.find_node_via_relationship(
            node_type=node_type,
            edge_type=edge_type,
            target_node_id=target_node.get("id"),
            direction=direction,
            context=context
        )

        if existing:
            logger.debug(f"Found existing {node_type} via {edge_type}->{target_type}")
            return existing

    except Exception as e:
        logger.warning(f"Error in via_relationship search: {e}")

    return None


def _apply_set_values(
    constraint: Dict[str, Any],
    extracted_properties: Optional[Dict[str, Any]],
    existing_node_properties: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply 'set' values from constraint to node properties.

    Supports three text modes for auto-extract values:
    - replace (default): Overwrite existing value with extracted value
    - append: Add extracted value to end of existing value (for text fields)
    - merge: Intelligently combine extracted value with existing (for summaries)

    Args:
        constraint: The node constraint with optional 'set' field
        extracted_properties: Properties extracted by LLM
        existing_node_properties: Properties from existing node (for append/merge)

    Returns:
        Final merged properties
    """
    final_props = dict(extracted_properties or {})
    set_values = constraint.get("set")
    existing_props = existing_node_properties or {}

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


def get_node_constraints_for_type(
    node_type: str,
    node_constraints: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Get all node constraints that could apply to a given node type.

    Useful for preprocessing or validation.

    Args:
        node_type: The node type to match
        node_constraints: All node constraints from policy

    Returns:
        List of potentially applicable constraints
    """
    applicable = []

    for constraint in node_constraints:
        constraint_node_type = constraint.get("node_type")

        # Node type must match (or be None for wildcard)
        if constraint_node_type and constraint_node_type != node_type:
            continue

        applicable.append(constraint)

    return applicable


def validate_node_constraints(node_constraints: List[Dict[str, Any]]) -> List[str]:
    """
    Validate a list of node constraints and return any errors.

    Args:
        node_constraints: List of node constraint dicts

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for i, constraint in enumerate(node_constraints):
        prefix = f"node_constraints[{i}]"

        # Check required field at memory level
        if not constraint.get("node_type"):
            errors.append(f"{prefix}: node_type is required at memory level")

        # Validate create value
        create_value = constraint.get("create", "auto")
        if create_value not in ("auto", "never"):
            errors.append(f"{prefix}: create must be 'auto' or 'never', got '{create_value}'")

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
