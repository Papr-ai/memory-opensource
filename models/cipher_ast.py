from enum import Enum
from typing import List, Union, Optional, Dict, Any, Literal, get_args, get_origin
from pydantic import BaseModel, Field, field_validator, ConfigDict
from models.structured_outputs import (
    InsightProperties, MeetingProperties, 
    TaskProperties, OpportunityProperties,
    CodeProperties, MemoryProperties, 
    PersonProperties, CompanyProperties,
    ProjectProperties
)
from models.shared_types import NodeLabel, RelationshipType
import json
import logging

logger = logging.getLogger(__name__)


class NodeAlias(str, Enum):
    SOURCE = 'm'
    TARGET = 'n'
    RELATIONSHIP = 'r'

class Direction(str, Enum):
    BOTH = "-"

class ComparisonOperator(str, Enum):
    # Equality
    EQUALS = "="
    NOT_EQUALS = "<>"
    
    # Numeric comparisons
    GREATER_THAN = ">"
    GREATER_THAN_EQUALS = ">="
    LESS_THAN = "<"
    LESS_THAN_EQUALS = "<="
    
    # Pattern matching
    CONTAINS = "CONTAINS"
    STARTS_WITH = "STARTS WITH"
    ENDS_WITH = "ENDS WITH"
    
    # Collection operations
    IN = "IN"
    NOT_IN = "NOT IN"
    
    # Null checks
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    
    # Regular expression
    MATCHES = "=~"

# Map NodeLabel to their corresponding property types
NODE_PROPERTY_MAP = {
    NodeLabel.Insight: InsightProperties,
    NodeLabel.Meeting: MeetingProperties,
    NodeLabel.Task: TaskProperties,
    NodeLabel.Opportunity: OpportunityProperties,
    NodeLabel.Code: CodeProperties,
    NodeLabel.Memory: MemoryProperties,
    NodeLabel.Person: PersonProperties,
    NodeLabel.Company: CompanyProperties,
    NodeLabel.Project: ProjectProperties
}

def create_dynamic_property_class(node_name: str, properties: Dict[str, Any]) -> type:
    """Create a dynamic Pydantic property class from user schema"""
    from pydantic import create_model
    
    # Build field definitions for Pydantic
    field_definitions = {}
    required_fields = []
    
    # Always include 'id' field
    field_definitions['id'] = (str, ...)
    required_fields.append('id')
    
    # Add user-defined properties
    for prop_name, prop_def in properties.items():
        if hasattr(prop_def, 'type'):
            # Handle PropertyDefinition objects
            prop_type = prop_def.type
            is_required = getattr(prop_def, 'required', False)
            default_value = getattr(prop_def, 'default', None)
        elif isinstance(prop_def, dict):
            # Handle dictionary format
            prop_type = prop_def.get('type', 'string')
            is_required = prop_def.get('required', False)
            default_value = prop_def.get('default', None)
        else:
            # Fallback
            prop_type = 'string'
            is_required = False
            default_value = None
        
        # Map property types to Python types
        type_mapping = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'array': list,
            'datetime': str,  # Keep as string for simplicity
            'object': dict
        }
        
        python_type = type_mapping.get(prop_type, str)
        
        if is_required:
            field_definitions[prop_name] = (python_type, ...)
            required_fields.append(prop_name)
        else:
            field_definitions[prop_name] = (python_type, default_value)
    
    # Create the dynamic model
    model_config = {
        'extra': 'forbid',
        'json_schema_extra': {
            'required': required_fields,
            'additionalProperties': False
        }
    }
    
    dynamic_class = create_model(
        f'{node_name}Properties',
        **field_definitions,
        __config__=model_config
    )
    
    logger.info(f"🔧 DYNAMIC: Created {node_name}Properties with fields: {list(field_definitions.keys())}")
    return dynamic_class

# Cache for registered schemas to avoid re-registration
_registered_schemas_cache = set()

def register_user_custom_properties(user_schemas: List[Any]):
    """Dynamically register custom node properties from user schemas"""
    if not user_schemas:
        return
    
    # Check cache to avoid re-registering the same schemas
    schema_ids = set()
    for schema in user_schemas:
        # Create stable cache key from schema content, not object identity
        if hasattr(schema, 'objectId') and schema.objectId:
            schema_id = schema.objectId
        else:
            # Create stable hash from schema content (node types and properties)
            content_key = ""
            if hasattr(schema, 'node_types'):
                node_types = schema.node_types
                content_key = str(sorted(node_types.keys())) if node_types else ""
            elif isinstance(schema, dict) and 'node_types' in schema:
                node_types = schema['node_types']
                content_key = str(sorted(node_types.keys())) if node_types else ""
            schema_id = f"schema_{hash(content_key)}"
        schema_ids.add(schema_id)
    
    # Skip if all schemas are already registered
    if schema_ids.issubset(_registered_schemas_cache):
        logger.info(f"🚀 CACHE HIT: All {len(user_schemas)} schemas already registered, skipping dynamic registration")
        return
    
    # Filter to only new schemas using the same stable ID logic
    new_schemas = []
    for schema in user_schemas:
        if hasattr(schema, 'objectId') and schema.objectId:
            schema_id = schema.objectId
        else:
            content_key = ""
            if hasattr(schema, 'node_types'):
                node_types = schema.node_types
                content_key = str(sorted(node_types.keys())) if node_types else ""
            elif isinstance(schema, dict) and 'node_types' in schema:
                node_types = schema['node_types']
                content_key = str(sorted(node_types.keys())) if node_types else ""
            schema_id = f"schema_{hash(content_key)}"
        
        if schema_id not in _registered_schemas_cache:
            new_schemas.append(schema)
    
    logger.info(f"🔧 DYNAMIC REGISTRATION: Processing {len(new_schemas)} new schemas (skipped {len(user_schemas) - len(new_schemas)} cached)")
    
    for schema in new_schemas:
        try:
            # Handle both Pydantic objects and dictionaries
            if hasattr(schema, 'node_types'):
                node_types = schema.node_types
            elif isinstance(schema, dict) and 'node_types' in schema:
                node_types = schema['node_types']
            else:
                continue
            
            # Register each custom node type
            for node_name, node_def in node_types.items():
                try:
                    # Skip system node types
                    system_types = ['Memory', 'Person', 'Company', 'Project', 'Task', 'Insight', 'Meeting', 'Opportunity', 'Code']
                    if node_name in system_types:
                        continue
                    
                    # Get properties from node definition
                    if hasattr(node_def, 'properties'):
                        properties = node_def.properties
                    elif isinstance(node_def, dict) and 'properties' in node_def:
                        properties = node_def['properties']
                    else:
                        properties = {}
                    
                    # Create dynamic property class
                    property_class = create_dynamic_property_class(node_name, properties)
                    
                    # Create/get the dynamic NodeLabel and register
                    custom_label = NodeLabel(node_name)
                    NODE_PROPERTY_MAP[custom_label] = property_class
                    
                    logger.info(f"🔧 REGISTERED DYNAMIC: {node_name} -> {property_class.__name__}")
                    
                except Exception as e:
                    logger.warning(f"Failed to register custom node {node_name}: {e}")
            
            # Mark schema as registered using the same stable ID logic
            if hasattr(schema, 'objectId') and schema.objectId:
                schema_id = schema.objectId
            else:
                content_key = ""
                if hasattr(schema, 'node_types'):
                    node_types = schema.node_types
                    content_key = str(sorted(node_types.keys())) if node_types else ""
                elif isinstance(schema, dict) and 'node_types' in schema:
                    node_types = schema['node_types']
                    content_key = str(sorted(node_types.keys())) if node_types else ""
                schema_id = f"schema_{hash(content_key)}"
            _registered_schemas_cache.add(schema_id)
                    
        except Exception as e:
            logger.warning(f"Failed to process user schema: {e}")

# First, collect all property names at module level
ALL_NODE_PROPERTIES = tuple(
    prop for node_type in NODE_PROPERTY_MAP.values() 
    for prop in node_type.model_fields.keys()
)

class WhereCondition(BaseModel):
    """Represents a WHERE condition in Cypher"""
    # Create a Union of all possible property names from all node types
    property: Union[Literal[*ALL_NODE_PROPERTIES]]
    operator: ComparisonOperator
    value: Union[str, int, float, bool, list]
    and_operator: Union[bool, None] = True
    node_label: Optional[NodeLabel] = None

    @field_validator('value')
    @classmethod
    def validate_value(cls, v: Any) -> Any:
        """Validate that string values don't contain special characters"""
        if isinstance(v, str):
            if any(char in v for char in '{},[]'):
                raise ValueError(f"String values cannot contain special characters like {{}}, [], or ,")
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str) and any(char in item for char in '{},[]'):
                    raise ValueError("List string values cannot contain special characters like {}, [], or ,")
        return v

    @field_validator('property')
    @classmethod
    def validate_property_exists(cls, v: str, info) -> str:
        # Get the node label from context if available
        node_label = info.data.get('node_label')
        if node_label and node_label in NODE_PROPERTY_MAP:
            property_model = NODE_PROPERTY_MAP[node_label]
            valid_properties = property_model.model_fields.keys()
            if v not in valid_properties:
                raise ValueError(
                    f"Property '{v}' is not valid for node type {node_label}. "
                    f"Valid properties are: {valid_properties}"
                )
        return v

    def to_cypher(self) -> str:
        """Convert condition to Cypher string with proper quoting"""
        # Get the operator value directly
        operator_value = self.operator.value
        
        # Handle different value types
        if isinstance(self.value, str):
            value_str = f"'{self.value}'"
        elif isinstance(self.value, list):
            value_items = [f"'{item}'" if isinstance(item, str) else str(item) for item in self.value]
            value_str = f"[{', '.join(value_items)}]"
        elif self.operator in [ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL]:
            value_str = ""
        else:
            value_str = str(self.value)
        
        # Handle special cases for IS NULL/IS NOT NULL
        if self.operator in [ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL]:
            return f"{self.property} {operator_value}"
        
        return f"{self.property} {operator_value} {value_str}"

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "property": {
                    "anyOf": [
                        {
                            "type": "string",
                            "enum": list(model.model_fields.keys())
                        } for model in NODE_PROPERTY_MAP.values()
                    ]
                },
                "operator": {
                    "type": "string",
                    "enum": [op.value for op in ComparisonOperator]
                },
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"},
                        {"type": "array"}
                    ]
                },
                "and_operator": {
                    "type": ["boolean", "null"],
                    "default": True
                },
                "node_label": {
                    "type": ["string", "null"],
                    "enum": [label.value for label in NodeLabel]
                }
            },
            "required": ["property", "operator", "value", "and_operator"]
        }
    )

class CipherNode(BaseModel):
    alias: str
    label: NodeLabel
    conditions: Optional[List[WhereCondition]] = None

    @classmethod
    def model_json_schema(cls, by_alias: bool = True, ref_template: str = '#/$defs/{model}') -> Dict[str, Any]:
        """Custom schema generation that includes dynamic NodeLabel values"""
        schema = super().model_json_schema(by_alias=by_alias, ref_template=ref_template)
        
        # Get all available NodeLabel values (including dynamic ones)
        all_labels = []
        
        # Add system labels
        for label in NodeLabel:
            if hasattr(label, 'value'):
                all_labels.append(label.value)
        
        # Add custom labels from the registry
        if hasattr(NodeLabel, '_custom_labels'):
            all_labels.extend(NodeLabel._custom_labels)
        
        # Update the label enum in the schema
        if 'properties' in schema and 'label' in schema['properties']:
            schema['properties']['label']['enum'] = sorted(set(all_labels))
        
        logger.info(f"🔧 CipherNode schema updated with {len(all_labels)} labels: {all_labels}")
        return schema

    @field_validator('conditions')
    @classmethod
    def validate_conditions(cls, v, values):
        if v is None:
            return v
        
        # Get the node label
        node_label = values.data.get('label')
        if node_label and node_label in NODE_PROPERTY_MAP:
            property_model = NODE_PROPERTY_MAP[node_label]
            valid_properties = property_model.model_fields.keys()
            
            for condition in v:
                if condition.property not in valid_properties:
                    raise ValueError(
                        f"Property '{condition.property}' in condition is not valid for node type {node_label}. "
                        f"Valid properties are: {valid_properties}"
                    )
        return v

    def to_cypher(self) -> str:
        """Convert node to Cypher syntax"""
        node_str = f"({self.alias}:{self.label.value})"
        if self.conditions:
            where_conditions = [cond.to_cypher() for cond in self.conditions]
            node_str += f" WHERE {' AND '.join(where_conditions)}"
        return node_str

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "alias": {
                    "type": "string",
                    "enum": [e.value for e in NodeAlias]
                },
                "label": {
                    "type": "string",
                    "enum": [l.value for l in NodeLabel]
                },
                "conditions": {
                    "type": ["array", "null"],
                    "default": None,
                    "items": {
                        "type": "object",
                        "properties": {
                            "property": {
                                "anyOf": [
                                    {
                                        "type": "string",
                                        "enum": list(model.model_fields.keys())
                                    } for model in NODE_PROPERTY_MAP.values()
                                ]
                            },
                            "operator": {
                                "type": "string",
                                "enum": [op.value for op in ComparisonOperator]
                            },
                            "value": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "number"},
                                    {"type": "boolean"},
                                    {"type": "null"},
                                    {"type": "array"}
                                ]
                            }
                        },
                        "required": ["property", "operator", "value"]
                    }
                }
            },
            "required": ["alias", "label"]
        }
    )

class Edge(BaseModel):
    relationship: RelationshipType
    direction: Direction
    conditions: Optional[List[WhereCondition]] = None
    alias: str

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "relationship": {
                    "type": "string",
                    "enum": [r.value for r in RelationshipType]
                },
                "direction": {
                    "type": "string",
                    "enum": [d.value for d in Direction]
                },
                "conditions": {
                    "type": ["array", "null"],
                    "default": None,
                    "items": {
                        "type": "object",
                        "properties": {
                            "property": {"type": "string"},
                            "operator": {
                                "type": "string",
                                "enum": [op.value for op in ComparisonOperator]
                            },
                            "value": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "number"},
                                    {"type": "boolean"},
                                    {"type": "array"}
                                ]
                            }
                        },
                        "required": ["property", "operator", "value"]
                    }
                },
                "alias": {
                    "type": "string",
                    "enum": [NodeAlias.RELATIONSHIP.value]
                }
            },
            "required": ["relationship", "direction", "alias"]
        }
    )

class PatternElement(BaseModel):
    left_node: CipherNode
    relationship: Edge
    right_node: CipherNode

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "left_node": {
                    "$ref": "#/$defs/CipherNode"
                },
                "relationship": {
                    "$ref": "#/$defs/Edge"
                },
                "right_node": {
                    "$ref": "#/$defs/CipherNode"
                }
            },
            "required": ["left_node", "relationship", "right_node"],
            "additionalProperties": False,
            "$defs": {
                "CipherNode": {
                    "type": "object",
                    "properties": {
                        "alias": {
                            "type": "string",
                            "enum": [e.value for e in NodeAlias]
                        },
                        "label": {
                            "type": "string",
                            "enum": [l.value for l in NodeLabel]
                        },
                        "conditions": {
                            "type": "array",
                            "items": {
                                "$ref": "#/$defs/WhereCondition"
                            }
                        }
                    },
                    "required": ["alias", "label"]
                },
                "Edge": {
                    "$ref": "#/model_config/json_schema_extra/$defs/Edge"
                },
                "WhereCondition": {
                    "$ref": "#/model_config/json_schema_extra/$defs/WhereCondition"
                }
            }
        }
    )
    
    def to_cypher(self) -> str:
        # Build the base pattern
        left_part = f"({self.left_node.alias}:{self.left_node.label.value})"
        rel_part = f"[{self.relationship.alias}:{self.relationship.relationship.value}]"
        right_part = f"({self.right_node.alias}:{self.right_node.label.value})"
        
        pattern = f"{left_part}-{rel_part}-{right_part}"
        
        # Add WHERE conditions if they exist
        where_conditions = []
        
        # Process conditions for both nodes
        for node, conditions in [
            (self.right_node, self.right_node.conditions),
            (self.left_node, self.left_node.conditions)
        ]:
            if conditions:
                for i, cond in enumerate(conditions):
                    if isinstance(cond.value, str):
                        value = f"'{cond.value}'"
                    elif isinstance(cond.value, list):
                        value_items = [f"'{item}'" if isinstance(item, str) else str(item) for item in cond.value]
                        value = f"[{', '.join(value_items)}]"
                    else:
                        value = str(cond.value)
                    
                    condition = f"{node.alias}.{cond.property} {cond.operator.value} {value}"
                    
                    # Only add condition if it has a value
                    if cond.value is not None:
                        # First condition doesn't need an operator
                        if not where_conditions:
                            where_conditions.append(condition)
                        else:
                            # Subsequent conditions must have an operator (AND/OR)
                            if cond.and_operator is True:
                                where_conditions.append("AND")
                            elif cond.and_operator is False:
                                where_conditions.append("OR")
                            else:
                                # If no operator specified, skip this condition
                                continue
                            where_conditions.append(condition)

        # Combine pattern with WHERE clause if conditions exist
        if where_conditions:
            conditions_str = ' '.join(where_conditions)
            return f"{pattern}\nWHERE {conditions_str}"
        return pattern

class MatchClause(BaseModel):
    pattern: PatternElement

    def to_cypher(self) -> str:
        """Convert match clause to Cypher string"""
        return f"MATCH {self.pattern.to_cypher()}"

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "pattern": {
                    "$ref": "#/$defs/PatternElement"
                }
            },
            "required": ["pattern"],
            "additionalProperties": False,
            "$defs": {
                "PatternElement": {
                    "$ref": "#/model_config/json_schema_extra/$defs/PatternElement"
                },
                "CipherNode": {
                    "$ref": "#/model_config/json_schema_extra/$defs/CipherNode"
                },
                "Edge": {
                    "$ref": "#/model_config/json_schema_extra/$defs/Edge"
                },
                "WhereCondition": {
                    "$ref": "#/model_config/json_schema_extra/$defs/WhereCondition"
                }
            }
        }
    )

class ReturnClause(BaseModel):
    expressions: List[NodeAlias]
    order_by: Union[str, None]
    aggregation: Union[str, None]

    @field_validator('expressions')
    @classmethod
    def validate_expressions(cls, v):
        required_aliases = [NodeAlias.SOURCE, NodeAlias.RELATIONSHIP, NodeAlias.TARGET]
        
        # Check if all required aliases are present
        missing_aliases = [alias for alias in required_aliases if alias not in v]
        if missing_aliases:
            raise ValueError(f"Return clause must include all variables: {required_aliases}. Missing: {missing_aliases}")
        
        # Check if there are any invalid aliases
        invalid_aliases = [expr for expr in v if expr not in required_aliases]
        if invalid_aliases:
            raise ValueError(f"Invalid return expressions: {invalid_aliases}. Must use: {required_aliases}")
        
        # Ensure the order is always m, r, n
        return required_aliases

    def to_cypher(self) -> str:
        """Convert return clause to Cypher with enforced spacing."""
        return_expr = [alias.value for alias in self.expressions]
        if self.aggregation:
            return_expr = [f"{self.aggregation}({expr})" for expr in return_expr]
        return "RETURN " + ", ".join(return_expr)  # Enforced space after RETURN

# Helper to get enum values for a property
def get_property_schema(model):
    schemas = []
    for name, field in model.model_fields.items():
        schema = {"type": "string"}
        # Check for Literal
        if get_origin(field.annotation) is Literal:
            schema["enum"] = list(get_args(field.annotation))
        # Check for Enum
        elif isinstance(field.annotation, type) and issubclass(field.annotation, Enum):
            schema["enum"] = [e.value for e in field.annotation]
        schema["title"] = name
        schemas.append((name, schema))
    return schemas

class CypherQuery(BaseModel):
    match: MatchClause

    @field_validator('match')
    @classmethod
    def validate_pattern(cls, v: MatchClause) -> MatchClause:
        """Validate the single pattern and its conditions"""
        # Validate string values in conditions for both nodes
        for node in [v.pattern.left_node, v.pattern.right_node]:
            if node.conditions:
                for condition in node.conditions:
                    if isinstance(condition.value, str):
                        if any(char in condition.value for char in '{},[]'):
                            raise ValueError(f"Invalid characters in condition value: {condition.value}")
        return v
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "match": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "object",
                            "properties": {
                                "left_node": {
                                    "type": "object",
                                    "properties": {
                                        "alias": {
                                            "type": "string",
                                            "enum": [str(e.value) for e in NodeAlias]
                                        },
                                        "label": {
                                            "type": "string",
                                            "enum": [str(l.value) for l in NodeLabel]
                                        },
                                        "conditions": {
                                            "type": ["array", "null"],
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "property": {
                                                        "anyOf": [
                                                            {
                                                                **schema,
                                                                "description": f"Property '{name}' for {model.__name__}"
                                                            } for model in NODE_PROPERTY_MAP.values() for name, schema in get_property_schema(model)
                                                        ]
                                                    },
                                                    "operator": {
                                                        "type": "string",
                                                        "enum": [str(op.value) for op in ComparisonOperator]
                                                    },
                                                    "value": {
                                                        "anyOf": [
                                                            {"type": "string"},
                                                            {"type": "number"},
                                                            {"type": "boolean"},
                                                            {"type": "null"},
                                                            {
                                                                "type": "array",
                                                                "items": {
                                                                    "anyOf": [
                                                                        {"type": "string"},
                                                                        {"type": "number"},
                                                                        {"type": "boolean"}
                                                                    ]
                                                                }
                                                            }
                                                        ]
                                                    },
                                                    "and_operator": {
                                                        "type": ["boolean", "null"]
                                                    }
                                                },
                                                "required": ["property", "operator", "value", "and_operator"]
                                            }
                                        }
                                    },
                                    "required": ["alias", "label", "conditions"]
                                },
                                "relationship": {
                                    "type": "object",
                                    "properties": {
                                        "relationship": {
                                            "type": "string",
                                            "enum": [str(r.value) for r in RelationshipType]
                                        },
                                        "direction": {
                                            "type": "string",
                                            "enum": [str(d.value) for d in Direction]
                                        },
                                        "alias": {
                                            "type": "string",
                                            "enum": [str(NodeAlias.RELATIONSHIP.value)]
                                        }
                                    },
                                    "required": ["relationship", "direction", "alias"]
                                },
                                "right_node": {
                                    "type": "object",
                                    "properties": {
                                        "alias": {
                                            "type": "string",
                                            "enum": [str(e.value) for e in NodeAlias]
                                        },
                                        "label": {
                                            "type": "string",
                                            "enum": [str(l.value) for l in NodeLabel]
                                        },
                                        "conditions": {
                                            "type": ["array", "null"],
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "property": {
                                                        "anyOf": [
                                                            {
                                                                **schema,
                                                                "description": f"Property '{name}' for {model.__name__}"
                                                            } for model in NODE_PROPERTY_MAP.values() for name, schema in get_property_schema(model)
                                                        ]
                                                    },
                                                    "operator": {
                                                        "type": "string",
                                                        "enum": [str(op.value) for op in ComparisonOperator]
                                                    },
                                                    "value": {
                                                        "anyOf": [
                                                            {"type": "string"},
                                                            {"type": "number"},
                                                            {"type": "boolean"},
                                                            {"type": "null"},
                                                            {
                                                                "type": "array",
                                                                "items": {
                                                                    "anyOf": [
                                                                        {"type": "string"},
                                                                        {"type": "number"},
                                                                        {"type": "boolean"}
                                                                    ]
                                                                }
                                                            }
                                                        ]
                                                    }
                                                },
                                                "required": ["property", "operator", "value"]
                                            }
                                        }
                                    },
                                    "required": ["alias", "label", "conditions"]
                                }
                            },
                            "required": ["left_node", "relationship", "right_node"]
                        }
                    },
                    "required": ["pattern"]
                }
            },
            "required": ["match"]
        }
    )
    
    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        """Override to add logging for schema generation"""
        schema = super().model_json_schema()
        logger.debug(f"Generated JSON Schema for CypherQuery: {json.dumps(schema, indent=2)}")
        return schema

    def to_cypher(self) -> str:
        """Convert the entire query to Cypher string"""
        # Get the pattern from match clause
        pattern = self.match.pattern.to_cypher()
        
        # Build ACL conditions for both source and target nodes
        source_alias = self.match.pattern.left_node.alias
        target_alias = self.match.pattern.right_node.alias
        
        source_acl_conditions = [
            f"{source_alias}.user_id = $user_id",
            f"any(x IN coalesce({source_alias}.user_read_access, []) WHERE x IN $user_read_access)",
            f"any(x IN coalesce({source_alias}.workspace_read_access, []) WHERE x IN $workspace_read_access)",
            f"any(x IN coalesce({source_alias}.role_read_access, []) WHERE x IN $role_read_access)",
            f"any(x IN coalesce({source_alias}.organization_read_access, []) WHERE x IN $organization_read_access)",
            f"any(x IN coalesce({source_alias}.namespace_read_access, []) WHERE x IN $namespace_read_access)"
        ]
        
        target_acl_conditions = [
            f"{target_alias}.user_id = $user_id",
            f"any(x IN coalesce({target_alias}.user_read_access, []) WHERE x IN $user_read_access)",
            f"any(x IN coalesce({target_alias}.workspace_read_access, []) WHERE x IN $workspace_read_access)",
            f"any(x IN coalesce({target_alias}.role_read_access, []) WHERE x IN $role_read_access)",
            f"any(x IN coalesce({target_alias}.organization_read_access, []) WHERE x IN $organization_read_access)",
            f"any(x IN coalesce({target_alias}.namespace_read_access, []) WHERE x IN $namespace_read_access)"
        ]
        
        # Combine ACL conditions
        combined_acl_condition = f"({' OR '.join(source_acl_conditions)}) AND ({' OR '.join(target_acl_conditions)})"
        
        # Build complete query with path assignment, ACL conditions, and structured return
        query_parts = [
            f"MATCH path = {pattern}",
            f"WHERE {combined_acl_condition}",
            "WITH DISTINCT path",
            """RETURN {
                path: path,
                nodes: [n IN nodes(path) | { id: n.id, labels: labels(n), properties: properties(n) }],
                relationships: [r IN relationships(path) | {
                    type: type(r), properties: properties(r),
                    startNode: startNode(r).id, endNode: endNode(r).id
                }]
            } AS result"""
        ]
        
        return "\n".join(query_parts)