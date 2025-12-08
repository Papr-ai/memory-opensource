#!/usr/bin/env python3
"""
Add relationship definitions to GraphQL node types based on relationship mapping.
"""
from pathlib import Path
import re

# Relationship mapping based on user's specification
RELATIONSHIPS = {
    "CREATED_BY": {
        "properties": "CreatedByProperties",
        "sources": ["Memory", "Project", "Task", "Insight", "Code"],
        "targets": ["Person"],
        "cardinality": "many-to-one",  # Many entities can be created by one person
    },
    "WORKS_AT": {
        "properties": "WorksAtProperties",
        "sources": ["Person"],
        "targets": ["Company"],
        "cardinality": "many-to-one",
    },
    "ASSIGNED_TO": {
        "properties": "AssignedToProperties",
        "sources": ["Task"],
        "targets": ["Person"],
        "cardinality": "many-to-one",
    },
    "MANAGED_BY": {
        "properties": "ManagedByProperties",
        "sources": ["Project"],
        "targets": ["Person"],
        "cardinality": "many-to-one",
    },
    "CONTAINS": {
        "properties": "ContainsProperties",
        "sources": ["Project", "Meeting"],
        "targets": ["Task", "Insight", "Memory"],
        "cardinality": "one-to-many",
    },
    "PARTICIPATED_IN": {
        "properties": "ParticipatedInProperties",
        "sources": ["Person"],
        "targets": ["Meeting", "Project"],
        "cardinality": "many-to-many",
    },
    "BELONGS_TO": {
        "properties": "BelongsToProperties",
        "sources": ["Task", "Insight", "Opportunity"],
        "targets": ["Project", "Company"],
        "cardinality": "many-to-one",
    },
    "RELATED_TO": {
        "properties": "RelatedToProperties",
        "sources": ["Insight", "Memory", "Task", "Opportunity"],
        "targets": ["Insight", "Memory", "Task", "Opportunity"],
        "cardinality": "many-to-many",
    },
    "REFERENCES": {
        "properties": "ReferencesProperties",
        "sources": ["Insight", "Memory", "Code"],
        "targets": ["Project", "Task", "Code"],
        "cardinality": "many-to-many",
    },
}


def generate_relationship_field(rel_name: str, rel_info: dict, node_type: str, direction: str) -> list[str]:
    """
    Generate GraphQL relationship field definitions.

    Args:
        rel_name: Relationship type name (e.g., "CREATED_BY")
        rel_info: Relationship metadata
        node_type: Current node type
        direction: "OUT" or "IN"

    Returns:
        List of field definition lines
    """
    fields = []

    if direction == "OUT":
        # This node is the source
        target_types = rel_info["targets"]

        for target in target_types:
            # Generate field name: createdByPerson, worksAtCompany, etc.
            field_name = _to_camel_case(rel_name.lower()) + target

            # Determine if array based on cardinality
            is_array = rel_info["cardinality"] in ["one-to-many", "many-to-many"]

            if is_array:
                type_def = f"[{target}!]!"
            else:
                type_def = f"[{target}!]!"  # Still array but usually one item

            # Add comment
            fields.append(f"  # {rel_name}: {node_type} -> {target}")

            # Add relationship field
            fields.append(
                f'  {field_name}: {type_def} @relationship(type: "{rel_name}", direction: OUT, properties: "{rel_info["properties"]}")'
            )

    elif direction == "IN":
        # This node is the target
        source_types = rel_info["sources"]

        for source in source_types:
            # Generate inverse field name
            inverse_name = _get_inverse_name(rel_name)
            field_name = _to_camel_case(inverse_name.lower()) + source

            # Always array for incoming relationships
            type_def = f"[{source}!]!"

            # Add comment
            fields.append(f"  # {rel_name} (inverse): {source} -> {node_type}")

            # Add relationship field
            fields.append(
                f'  {field_name}: {type_def} @relationship(type: "{rel_name}", direction: IN, properties: "{rel_info["properties"]}")'
            )

    return fields


def _to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def _get_inverse_name(rel_name: str) -> str:
    """Get inverse relationship name for display"""
    inverses = {
        "CREATED_BY": "created",
        "WORKS_AT": "employees",
        "ASSIGNED_TO": "assignedTasks",
        "MANAGED_BY": "managedProjects",
        "CONTAINS": "containedIn",
        "PARTICIPATED_IN": "participants",
        "BELONGS_TO": "has",
        "RELATED_TO": "relatedTo",
        "REFERENCES": "referencedBy",
    }
    return inverses.get(rel_name, rel_name.lower())


def add_relationships_to_schema(input_file: Path, output_file: Path):
    """Add relationship definitions to each node type"""

    # Read the schema
    content = input_file.read_text()

    # Split by node type definitions
    lines = content.split('\n')

    # Track current node type
    output_lines = []
    current_node = None
    in_node_fields = False  # Track if we're in the fields section (after opening {)
    seen_opening_brace = False
    node_fields_added = set()

    for i, line in enumerate(lines):
        # Check if starting a new node type
        if line.startswith('type ') and '@node' in line:
            # Extract node name
            match = re.match(r'type (\w+) @node', line)
            if match:
                current_node = match.group(1)
                in_node_fields = False
                seen_opening_brace = False
                node_fields_added = set()

        # Check for opening brace that starts the fields section
        if current_node and not seen_opening_brace and line.strip() == '{':
            seen_opening_brace = True
            in_node_fields = True

        # Check if this is the closing brace of a node definition (fields section)
        if in_node_fields and seen_opening_brace and line.strip() == '}' and current_node:
            relationship_lines = []
            relationship_lines.append("")
            relationship_lines.append("  # ========================================")
            relationship_lines.append(f"  # Relationships for {current_node}")
            relationship_lines.append("  # ========================================")

            # Add outgoing relationships (this node is source)
            for rel_name, rel_info in RELATIONSHIPS.items():
                if current_node in rel_info["sources"]:
                    rel_fields = generate_relationship_field(rel_name, rel_info, current_node, "OUT")
                    relationship_lines.extend(rel_fields)

            # Add incoming relationships (this node is target)
            for rel_name, rel_info in RELATIONSHIPS.items():
                if current_node in rel_info["targets"]:
                    rel_fields = generate_relationship_field(rel_name, rel_info, current_node, "IN")
                    relationship_lines.extend(rel_fields)

            # Add the relationship lines before closing brace
            output_lines.extend(relationship_lines)

            # Reset state
            in_node_fields = False
            seen_opening_brace = False
            current_node = None

        # Add the original line
        output_lines.append(line)

    # Write output
    output_file.write_text('\n'.join(output_lines))
    print(f"✅ Added relationships to {output_file}")
    print(f"   Processed relationships: {', '.join(RELATIONSHIPS.keys())}")


if __name__ == "__main__":
    input_file = Path("models/graphql/all_nodes_generated.graphql")
    output_file = Path("models/graphql/schema_with_relationships.graphql")

    if not input_file.exists():
        print(f"❌ Error: {input_file} not found")
        exit(1)

    add_relationships_to_schema(input_file, output_file)

    print(f"\n📝 Next steps:")
    print(f"   1. Review: {output_file}")
    print(f"   2. Copy relationship properties from papr_graphql.graphql")
    print(f"   3. Upload to Neo4j Aura Console")
