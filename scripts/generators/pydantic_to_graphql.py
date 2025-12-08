#!/usr/bin/env python3
"""
Convert Pydantic models to Neo4j GraphQL type definitions.

This script automatically generates GraphQL schema from Pydantic models,
ensuring they stay in sync.

Usage:
    # From memory repo root:
    python scripts/pydantic_to_graphql.py

    # Or import as module:
    from scripts.pydantic_to_graphql import pydantic_to_graphql_node
"""
import sys
from pathlib import Path
from typing import get_origin, get_args, Union
import inspect

# Add parent directory to path to import models
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import will happen in main() to handle missing dependencies gracefully
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    print("Error: pydantic not installed")
    print("Run: pip install pydantic")
    sys.exit(1)


def python_type_to_graphql(python_type, field_info) -> str:
    """
    Convert Python type annotation to GraphQL type.

    Args:
        python_type: The Python type from Pydantic field
        field_info: Pydantic FieldInfo object

    Returns:
        GraphQL type string (e.g., "String!", "[String!]!", "String")
    """
    from pydantic_core import PydanticUndefined

    origin = get_origin(python_type)
    args = get_args(python_type)
    is_optional = False

    # Handle Optional types (Union with None)
    if origin is Union or (args and type(None) in args):
        is_optional = True
        # Extract non-None type
        non_none_types = [t for t in args if t is not type(None)]
        if non_none_types:
            inner_type = non_none_types[0]
            # Recursively convert but mark as optional
            result = python_type_to_graphql_inner(inner_type, field_info)
            return result  # Don't add ! for optional fields

    # Handle List types
    if origin is list:
        item_type = args[0] if args else str
        gql_item_type = _base_type_to_graphql(item_type)
        # Check if field is required (no default value)
        is_required = field_info.default is PydanticUndefined and field_info.default_factory is None
        if is_required:
            return f"[{gql_item_type}!]!"  # Required list
        return f"[{gql_item_type}!]"  # Optional list

    # Base types
    # Field is required if it has no default and is not Optional
    is_required = (field_info.default is PydanticUndefined and
                   field_info.default_factory is None and
                   not is_optional)

    if is_required:
        return _base_type_to_graphql(python_type) + "!"  # Required
    return _base_type_to_graphql(python_type)  # Optional


def python_type_to_graphql_inner(python_type, field_info) -> str:
    """Helper for nested type conversion"""
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle List types
    if origin is list:
        item_type = args[0] if args else str
        gql_item_type = _base_type_to_graphql(item_type)
        return f"[{gql_item_type}!]"

    # Base types (no ! suffix for inner optional types)
    return _base_type_to_graphql(python_type)


def _base_type_to_graphql(python_type) -> str:
    """Map Python base types to GraphQL scalar types"""
    type_map = {
        str: "String",
        int: "Int",
        float: "Float",
        bool: "Boolean",
    }

    # Handle string type names
    if isinstance(python_type, str):
        if 'str' in python_type.lower():
            return "String"
        if 'int' in python_type.lower():
            return "Int"
        if 'float' in python_type.lower():
            return "Float"
        if 'bool' in python_type.lower():
            return "Boolean"
        return "String"  # Default

    return type_map.get(python_type, "String")


def pydantic_to_graphql_node(
    model: type[BaseModel],
    node_label: str = None,
    add_authorization: bool = True
) -> str:
    """
    Convert a Pydantic model to a GraphQL @node type definition.

    Args:
        model: Pydantic model class
        node_label: Override the node type name (defaults to model class name)
        add_authorization: Whether to add @authorization directive

    Returns:
        GraphQL type definition as string
    """
    type_name = node_label or model.__name__.replace("Properties", "").replace("Node", "")

    # Start building the GraphQL type
    lines = []

    # Add authorization directive if requested
    if add_authorization:
        lines.append(f"type {type_name} @node")
        lines.append("  @authorization(filter: [")
        lines.append("    {")
        lines.append("      operations: [READ, AGGREGATE],")
        lines.append("      where: {")
        lines.append("        node: {")
        lines.append("          AND: [")
        lines.append("            # Multi-tenant workspace isolation")
        lines.append("            { workspace_id: \"$jwt.workspace_id\" }")
        lines.append("            ")
        lines.append("            # User must own OR have read access")
        lines.append("            { OR: [")
        lines.append("                { user_id: \"$jwt.user_id\" }")
        lines.append("                { user_read_access_INCLUDES: \"$jwt.user_id\" }")
        lines.append("                { workspace_read_access_INCLUDES: \"$jwt.workspace_id\" }")
        lines.append("              ]")
        lines.append("            }")
        lines.append("          ]")
        lines.append("        }")
        lines.append("      }")
        lines.append("    },")
        lines.append("    {")
        lines.append("      operations: [UPDATE, DELETE],")
        lines.append("      where: {")
        lines.append("        node: {")
        lines.append("          AND: [")
        lines.append("            { workspace_id: \"$jwt.workspace_id\" }")
        lines.append("            { OR: [")
        lines.append("                { user_id: \"$jwt.user_id\" }")
        lines.append("                { user_write_access_INCLUDES: \"$jwt.user_id\" }")
        lines.append("                { workspace_write_access_INCLUDES: \"$jwt.workspace_id\" }")
        lines.append("              ]")
        lines.append("            }")
        lines.append("          ]")
        lines.append("        }")
        lines.append("      }")
        lines.append("    }")
        lines.append("  ])")
    else:
        lines.append(f"type {type_name} @node")

    lines.append("{")

    # Add fields from Pydantic model
    for field_name, field_info in model.model_fields.items():
        # Get the field type
        field_type = field_info.annotation

        # Convert to GraphQL type
        gql_type = python_type_to_graphql(field_type, field_info)

        # Add field description as comment if available
        if field_info.description:
            lines.append(f"  # {field_info.description}")

        lines.append(f"  {field_name}: {gql_type}")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_all_types():
    """Generate GraphQL types for all supported models"""
    from models.memory_models import (
        MemoryNodeProperties,
        PersonNodeProperties,
        CompanyNodeProperties,
        ProjectNodeProperties,
        TaskNodeProperties,
        InsightNodeProperties,
        MeetingNodeProperties,
        OpportunityNodeProperties,
        CodeNodeProperties
    )

    types = []

    # Define all node types with their models
    node_definitions = [
        ("Memory", MemoryNodeProperties),
        ("Person", PersonNodeProperties),
        ("Company", CompanyNodeProperties),
        ("Project", ProjectNodeProperties),
        ("Task", TaskNodeProperties),
        ("Insight", InsightNodeProperties),
        ("Meeting", MeetingNodeProperties),
        ("Opportunity", OpportunityNodeProperties),
        ("Code", CodeNodeProperties),
    ]

    # Generate GraphQL type for each node
    for node_label, model_class in node_definitions:
        graphql_type = pydantic_to_graphql_node(
            model_class,
            node_label=node_label,
            add_authorization=True
        )
        types.append((node_label, graphql_type))

    return types


def main():
    """Generate GraphQL types from Pydantic models"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Neo4j GraphQL schema from Pydantic models"
    )
    parser.add_argument(
        "--model",
        choices=["memory", "person", "company", "project", "task", "insight", "meeting", "opportunity", "code", "all"],
        default="all",
        help="Which model(s) to generate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable @authorization directives"
    )

    args = parser.parse_args()

    # Generate header
    output = []
    output.append("# Auto-generated GraphQL schema from Pydantic models")
    output.append("# Generated by: scripts/pydantic_to_graphql.py")
    output.append(f"# Source: models/memory_models.py")
    output.append("")
    output.append("# ⚠️  WARNING: Do not edit this file directly!")
    output.append("# Edit the Pydantic models in models/memory_models.py instead,")
    output.append("# then regenerate this file using: python scripts/pydantic_to_graphql.py")
    output.append("")

    # Generate types
    types = generate_all_types()

    for name, graphql_type in types:
        if args.model == "all" or args.model == name.lower():
            output.append(graphql_type)

    output.append("\n# Usage:")
    output.append("# 1. Review the generated types above")
    output.append("# 2. Copy to models/graphql/papr_graphql_default_schema.graphql")
    output.append("# 3. Upload to Neo4j GraphQL (via Neo4j Aura Console or API)")
    output.append("# 4. Add relationships manually (not auto-generated)")

    # Output
    result = "\n".join(output)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result)
        print(f"✅ Generated GraphQL schema: {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
