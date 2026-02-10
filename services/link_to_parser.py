"""
Link-To DSL Parser Service.

Parses the `link_to` shorthand DSL into full `NodeConstraint` and `EdgeConstraint` objects.

## DSL Syntax

### Node Constraints (Direct Node Matching)

| Syntax | Description | PropertyMatch |
|--------|-------------|---------------|
| `Type:property` | Semantic match on property | `mode="semantic"` |
| `Type:property=value` | Exact match with value | `mode="exact", value="value"` |
| `Type:property~value` | Semantic match with value | `mode="semantic", value="value"` |
| `Type:property~@0.9` | Semantic match with custom threshold | `mode="semantic", threshold=0.9` |
| `Type:property~@0.9~value` | Semantic with threshold and value | `mode="semantic", threshold=0.9, value="value"` |
| `Type:property1,property2` | Multiple properties | Multiple PropertyMatch |

### Node with Graph Traversal (Via Relationship)

| Syntax | Description |
|--------|-------------|
| `Type.via(EDGE->Target:prop)` | Find Type via EDGE relationship to Target |
| `Type.via(EDGE->Target:prop=value)` | Via with exact value filter |

### Edge Constraints (Arrow Syntax)

| Syntax | Description |
|--------|-------------|
| `Source->EDGE_TYPE->Target:property` | Full edge path with target search |
| `Source->EDGE_TYPE:property` | Edge with implicit target (from schema) |
| `->EDGE_TYPE->Target:property` | Edge from any source type |
| `Source->EDGE_TYPE` | Edge using schema defaults for search |

### Special References

| Reference | Description |
|-----------|-------------|
| `$this` | The Memory node being created |
| `$previous` | User's most recent memory |
| `$context:N` | Last N memories in conversation |

### Value Options (Dict Values)

When using dict form, values can include:
- `set`: Properties to set (exact value or `{"mode": "auto"}`)
- `when`: Condition for constraint to apply
- `create`: Override creation policy (`"auto"` or `"never"`)

## Examples

```python
# String form (single entity)
link_to = "Task:title"

# With custom threshold
link_to = "Task:title~@0.85"

# List form (multiple entities)
link_to = ["Task:title", "Person:email"]

# Via relationship (find Task via ASSIGNED_TO to Person)
link_to = "Task.via(ASSIGNED_TO->Person:email=john@example.com)"

# Dict form (with options)
link_to = {
    "Task:title": {"set": {"status": "completed"}},
    "Person:email=john@example.com": {"create": "never"},
    "SecurityBehavior->MITIGATES->TacticDef:name": {"create": "never"}
}
```
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from services.logger_singleton import LoggerSingleton

# Import Pydantic types for proper type safety
# These are imported lazily to avoid circular imports
# Use Dict[str, Any] for internal storage, but expose proper types in the API

logger = LoggerSingleton.get_logger(__name__)


# Regular expressions for parsing
NODE_PATTERN = re.compile(
    r'^(?P<node_type>[A-Za-z][A-Za-z0-9_]*)'  # Node type (e.g., Task, Person)
    r':(?P<properties>.+)$'                    # Properties after colon
)

# Via relationship pattern: Type.via(EDGE->Target:prop)
VIA_PATTERN = re.compile(
    r'^(?P<node_type>[A-Za-z][A-Za-z0-9_]*)'   # Source node type
    r'\.via\('                                  # .via(
    r'(?P<edge_type>[A-Z][A-Z0-9_]*)'          # Edge type (UPPER_CASE)
    r'->'                                       # Arrow
    r'(?P<target_type>[A-Za-z][A-Za-z0-9_]*)'  # Target type
    r'(?::(?P<target_props>.+))?'              # Optional target properties
    r'\)$'                                      # Closing paren
)

EDGE_ARROW_PATTERN = re.compile(
    r'^(?P<source>[A-Za-z][A-Za-z0-9_]*)?'     # Optional source type
    r'->'                                       # Arrow
    r'(?P<edge_type>[A-Z][A-Z0-9_]*)'          # Edge type (UPPER_CASE)
    r'(?:->'                                   # Optional target part
    r'(?P<target>[A-Za-z][A-Za-z0-9_]*))?'     # Target type
    r'(?::(?P<properties>.+))?$'               # Optional properties
)

# Property pattern with threshold support
# Examples: title, title=value, title~value, title~@0.9, title~@0.9~value
PROPERTY_PATTERN = re.compile(
    r'^(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)'       # Property name
    r'(?:'                                      # Optional match group
    r'(?P<operator>[=~])'                       # Operator (= or ~)
    r'(?:'                                      # Value options
    r'@(?P<threshold>\d+\.?\d*)'               # Threshold: @0.9
    r'(?:~(?P<threshold_value>.+))?'           # Optional value after threshold: ~value
    r'|'                                        # OR
    r'(?P<value>.+)'                           # Regular value
    r')'
    r')?$'
)

SPECIAL_REFS = {'$this', '$previous'}
CONTEXT_REF_PATTERN = re.compile(r'^\$context:(\d+)$')


class LinkToParseResult(BaseModel):
    """
    Result of parsing link_to DSL.

    Contains parsed node constraints, edge constraints, special references,
    and any parsing errors encountered.

    The constraints are stored as dicts internally for flexibility, but they
    conform to the NodeConstraint and EdgeConstraint schemas from shared_types.py.

    Example:
        >>> result = parse_link_to("Task:title")
        >>> result.has_errors()
        False
        >>> result.node_constraints[0]["node_type"]
        'Task'
    """
    node_constraints: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Parsed node constraints (conform to NodeConstraint schema)"
    )
    edge_constraints: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Parsed edge constraints (conform to EdgeConstraint schema)"
    )
    special_refs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Parsed special references ($this, $previous, $context:N)"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Parsing errors encountered"
    )

    def has_errors(self) -> bool:
        """Check if any parsing errors occurred."""
        return len(self.errors) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for merging with memory_policy."""
        return {
            "node_constraints": self.node_constraints,
            "edge_constraints": self.edge_constraints,
            "special_refs": self.special_refs
        }


def parse_link_to(
    link_to: Union[str, List[str], Dict[str, Any]],
    schema: Optional[Dict[str, Any]] = None
) -> LinkToParseResult:
    """
    Parse link_to DSL into NodeConstraint and EdgeConstraint dicts.

    Args:
        link_to: The link_to value in one of three forms:
            - String: Single constraint (e.g., "Task:title")
            - List: Multiple constraints (e.g., ["Task:title", "Person:email"])
            - Dict: Constraints with options (e.g., {"Task:title": {"set": {...}}})
        schema: Optional schema for inferring target types from edge definitions

    Returns:
        LinkToParseResult with parsed constraints and any errors
    """
    result = LinkToParseResult()

    if isinstance(link_to, str):
        # Single string form
        _parse_single_key(link_to, {}, result, schema)

    elif isinstance(link_to, list):
        # List form - multiple keys, no options
        for key in link_to:
            if isinstance(key, str):
                _parse_single_key(key, {}, result, schema)
            else:
                result.errors.append(f"List items must be strings, got {type(key).__name__}")

    elif isinstance(link_to, dict):
        # Dict form - keys with options
        for key, options in link_to.items():
            if not isinstance(key, str):
                result.errors.append(f"Dict keys must be strings, got {type(key).__name__}")
                continue
            if options is None:
                options = {}
            if not isinstance(options, dict):
                result.errors.append(f"Options for '{key}' must be a dict, got {type(options).__name__}")
                continue
            _parse_single_key(key, options, result, schema)

    else:
        result.errors.append(f"link_to must be str, list, or dict, got {type(link_to).__name__}")

    return result


def _parse_single_key(
    key: str,
    options: Dict[str, Any],
    result: LinkToParseResult,
    schema: Optional[Dict[str, Any]] = None
) -> None:
    """
    Parse a single link_to key and add to result.

    Args:
        key: The DSL key (e.g., "Task:title", "Source->EDGE->Target:prop")
        options: The options dict (set, when, create)
        result: Result object to append to
        schema: Optional schema for type inference
    """
    key = key.strip()

    # Check for special references
    if key in SPECIAL_REFS:
        result.special_refs.append({
            "ref": key,
            **options
        })
        return

    context_match = CONTEXT_REF_PATTERN.match(key)
    if context_match:
        result.special_refs.append({
            "ref": "$context",
            "count": int(context_match.group(1)),
            **options
        })
        return

    # Check for via pattern: Type.via(EDGE->Target:prop)
    via_match = VIA_PATTERN.match(key)
    if via_match:
        _parse_via_key(via_match, options, result, schema)
        return

    # Check if it's an edge (arrow syntax)
    if '->' in key:
        _parse_edge_key(key, options, result, schema)
    else:
        # Node syntax
        _parse_node_key(key, options, result)


def _parse_node_key(
    key: str,
    options: Dict[str, Any],
    result: LinkToParseResult
) -> None:
    """
    Parse a node DSL key into NodeConstraint.

    Syntax: Type:property or Type:property=value or Type:property~value
    """
    match = NODE_PATTERN.match(key)
    if not match:
        result.errors.append(f"Invalid node syntax: '{key}'. Expected 'Type:property'")
        return

    node_type = match.group('node_type')
    properties_str = match.group('properties')

    # Parse properties
    property_matches = _parse_properties(properties_str, result, key)
    if not property_matches:
        return

    # Build NodeConstraint
    constraint = {
        "node_type": node_type,
        "search": {
            "properties": property_matches
        }
    }

    # Add options
    if "set" in options:
        constraint["set"] = options["set"]
    if "when" in options:
        constraint["when"] = options["when"]
    if "create" in options:
        constraint["create"] = options["create"]

    result.node_constraints.append(constraint)
    logger.debug(f"Parsed node constraint: {node_type} with {len(property_matches)} properties")


def _parse_edge_key(
    key: str,
    options: Dict[str, Any],
    result: LinkToParseResult,
    schema: Optional[Dict[str, Any]] = None
) -> None:
    """
    Parse an edge DSL key (arrow syntax) into EdgeConstraint.

    Syntax:
    - Source->EDGE_TYPE->Target:property
    - Source->EDGE_TYPE:property (implicit target)
    - ->EDGE_TYPE->Target:property (any source)
    """
    match = EDGE_ARROW_PATTERN.match(key)
    if not match:
        result.errors.append(f"Invalid edge syntax: '{key}'. Expected 'Source->EDGE->Target:property'")
        return

    source_type = match.group('source') or None
    edge_type = match.group('edge_type')
    target_type = match.group('target') or None
    properties_str = match.group('properties')

    # If no target type specified, try to infer from schema
    if not target_type and schema:
        target_type = _infer_target_type(edge_type, source_type, schema)
        if target_type:
            logger.debug(f"Inferred target type '{target_type}' for edge '{edge_type}'")

    # Parse properties if present
    property_matches = []
    if properties_str:
        property_matches = _parse_properties(properties_str, result, key)

    # Build EdgeConstraint
    constraint = {
        "edge_type": edge_type
    }

    if source_type:
        constraint["source_type"] = source_type
    if target_type:
        constraint["target_type"] = target_type

    if property_matches:
        constraint["search"] = {
            "properties": property_matches
        }

    # Add options
    if "set" in options:
        constraint["set"] = options["set"]
    if "when" in options:
        constraint["when"] = options["when"]
    if "create" in options:
        constraint["create"] = options["create"]

    result.edge_constraints.append(constraint)
    logger.debug(f"Parsed edge constraint: {source_type or '*'}->{edge_type}->{target_type or '*'}")


def _parse_via_key(
    via_match: re.Match,
    options: Dict[str, Any],
    result: LinkToParseResult,
    schema: Optional[Dict[str, Any]] = None
) -> None:
    """
    Parse a via pattern into NodeConstraint with via_relationship.

    Syntax: Type.via(EDGE->Target:prop) or Type.via(EDGE->Target:prop=value)

    This creates a NodeConstraint that searches for nodes via their relationships.
    """
    node_type = via_match.group('node_type')
    edge_type = via_match.group('edge_type')
    target_type = via_match.group('target_type')
    target_props_str = via_match.group('target_props')

    # Parse target properties if present
    target_property_matches = []
    if target_props_str:
        # Create a temporary result to collect any errors
        temp_result = LinkToParseResult()
        target_property_matches = _parse_properties(target_props_str, temp_result, f"{node_type}.via(...)")
        result.errors.extend(temp_result.errors)

    # Build via_relationship
    via_relationship = {
        "edge_type": edge_type,
        "target_type": target_type,
        "direction": "outgoing"
    }

    if target_property_matches:
        via_relationship["target_search"] = {
            "properties": target_property_matches
        }

    # Build NodeConstraint with via_relationship in search
    constraint = {
        "node_type": node_type,
        "search": {
            "via_relationship": [via_relationship]
        }
    }

    # Add options
    if "set" in options:
        constraint["set"] = options["set"]
    if "when" in options:
        constraint["when"] = options["when"]
    if "create" in options:
        constraint["create"] = options["create"]

    result.node_constraints.append(constraint)
    logger.debug(f"Parsed via constraint: {node_type}.via({edge_type}->{target_type})")


def _parse_properties(
    properties_str: str,
    result: LinkToParseResult,
    original_key: str
) -> List[Dict[str, Any]]:
    """
    Parse properties string into PropertyMatch dicts.

    Supports:
    - Single: "title"
    - Multiple: "title,description"
    - With operator: "id=value" (exact), "title~value" (semantic)
    - With threshold: "title~@0.9" (semantic with threshold)
    - With threshold and value: "title~@0.9~auth bug" (semantic with both)
    """
    property_matches = []

    # Split by comma for multiple properties
    property_parts = [p.strip() for p in properties_str.split(',')]

    for prop_str in property_parts:
        match = PROPERTY_PATTERN.match(prop_str)
        if not match:
            result.errors.append(f"Invalid property syntax in '{original_key}': '{prop_str}'")
            continue

        prop_name = match.group('name')
        operator = match.group('operator')
        threshold = match.group('threshold')
        threshold_value = match.group('threshold_value')
        value = match.group('value')

        prop_match = {"name": prop_name}

        if operator == '=':
            # Exact match
            prop_match["mode"] = "exact"
            if value:
                prop_match["value"] = value
        elif operator == '~':
            # Semantic match
            prop_match["mode"] = "semantic"
            if threshold:
                # Threshold specified: ~@0.9 or ~@0.9~value
                prop_match["threshold"] = float(threshold)
                if threshold_value:
                    prop_match["value"] = threshold_value
            elif value:
                # Regular value: ~value
                prop_match["value"] = value
        else:
            # No operator - default to semantic
            prop_match["mode"] = "semantic"

        property_matches.append(prop_match)

    return property_matches


def _infer_target_type(
    edge_type: str,
    source_type: Optional[str],
    schema: Dict[str, Any]
) -> Optional[str]:
    """
    Infer target type from schema's relationship_types.

    Args:
        edge_type: The edge type (e.g., "MITIGATES")
        source_type: Optional source type filter
        schema: Schema dict with relationship_types

    Returns:
        Inferred target type or None
    """
    rel_types = schema.get("relationship_types", {})
    rel_def = rel_types.get(edge_type)

    if not rel_def:
        return None

    allowed_targets = rel_def.get("allowed_target_types", [])
    allowed_sources = rel_def.get("allowed_source_types", [])

    # If source_type provided, check if it's valid
    if source_type and allowed_sources and source_type not in allowed_sources:
        return None

    # Return single target type if only one allowed
    if len(allowed_targets) == 1:
        return allowed_targets[0]

    return None


def expand_link_to_to_policy(
    link_to: Union[str, List[str], Dict[str, Any]],
    existing_policy: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Expand link_to DSL to full memory_policy dict.

    This is the main entry point for route handlers.

    Args:
        link_to: The link_to value
        existing_policy: Existing memory_policy to merge with
        schema: Optional schema for type inference

    Returns:
        Expanded memory_policy dict

    Raises:
        ValueError: If parsing fails
    """
    result = parse_link_to(link_to, schema)

    if result.has_errors():
        raise ValueError(f"link_to parsing errors: {'; '.join(result.errors)}")

    # Start with existing policy or empty
    policy = dict(existing_policy or {})

    # Merge node_constraints (ensure never None)
    existing_node = policy.get("node_constraints") or []
    if result.node_constraints:
        policy["node_constraints"] = existing_node + result.node_constraints
    elif not existing_node:
        policy["node_constraints"] = []

    # Merge edge_constraints (ensure never None)
    existing_edge = policy.get("edge_constraints") or []
    if result.edge_constraints:
        policy["edge_constraints"] = existing_edge + result.edge_constraints
    elif not existing_edge:
        policy["edge_constraints"] = []

    # Handle special refs (convert to relationships)
    if result.special_refs:
        existing_rels = policy.get("relationships", [])
        for ref in result.special_refs:
            if ref["ref"] == "$previous":
                existing_rels.append({
                    "source": "$this",
                    "target": "$previous",
                    "type": ref.get("type", "FOLLOWS")
                })
            elif ref["ref"] == "$context":
                # Context linking handled separately
                policy["context_depth"] = ref.get("count", 3)
            elif ref["ref"] == "$this":
                # Self-reference - might be used for self-loops
                pass
        if existing_rels:
            policy["relationships"] = existing_rels

    logger.info(
        f"Expanded link_to: {len(result.node_constraints)} node constraints, "
        f"{len(result.edge_constraints)} edge constraints, "
        f"{len(result.special_refs)} special refs"
    )

    return policy


def validate_link_to(
    link_to: Union[str, List[str], Dict[str, Any]],
    schema: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Validate link_to DSL without expanding.

    Returns list of error messages (empty if valid).
    """
    result = parse_link_to(link_to, schema)
    return result.errors


# DSL syntax helpers for documentation and testing
DSL_EXAMPLES = {
    # Node constraints
    "Task:title": "Semantic match on Task.title",
    "Task:id=TASK-123": "Exact match Task.id = 'TASK-123'",
    "Task:title~auth bug": "Semantic search for 'auth bug' on Task.title",
    "Person:email,name": "Match on Person.email OR Person.name",

    # Edge constraints
    "SecurityBehavior->MITIGATES->TacticDef:name": "Full edge path with target search",
    "SecurityBehavior->MITIGATES:name": "Edge with implicit target (from schema)",
    "->MITIGATES->TacticDef:name": "Edge from any source type",
    "Task->ASSIGNED_TO->Person:email=john@example.com": "Edge with exact value",

    # Special refs
    "$previous": "Link to user's previous memory",
    "$context:5": "Link to last 5 conversation messages",
    "$this": "Reference to the memory being created",
}
