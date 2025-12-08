#!/usr/bin/env python3
"""
Add relationship definitions to GraphQL node types - Version 2
Uses simpler logic: find the LAST } before the next 'type' keyword
"""
from pathlib import Path
import re

# Same relationship mapping as before
RELATIONSHIPS = {
    "CREATED_BY": {
        "properties": "CreatedByProperties",
        "sources": ["Memory", "Project", "Task", "Insight", "Code"],
        "targets": ["Person"],
    },
    "WORKS_AT": {
        "properties": "WorksAtProperties",
        "sources": ["Person"],
        "targets": ["Company"],
    },
    "ASSIGNED_TO": {
        "properties": "AssignedToProperties",
        "sources": ["Task"],
        "targets": ["Person"],
    },
    "MANAGED_BY": {
        "properties": "ManagedByProperties",
        "sources": ["Project"],
        "targets": ["Person"],
    },
    "CONTAINS": {
        "properties": "ContainsProperties",
        "sources": ["Project", "Meeting"],
        "targets": ["Task", "Insight", "Memory"],
    },
    "PARTICIPATED_IN": {
        "properties": "ParticipatedInProperties",
        "sources": ["Person"],
        "targets": ["Meeting", "Project"],
    },
    "BELONGS_TO": {
        "properties": "BelongsToProperties",
        "sources": ["Task", "Insight", "Opportunity"],
        "targets": ["Project", "Company"],
    },
    "RELATED_TO": {
        "properties": "RelatedToProperties",
        "sources": ["Insight", "Memory", "Task", "Opportunity"],
        "targets": ["Insight", "Memory", "Task", "Opportunity"],
    },
    "REFERENCES": {
        "properties": "ReferencesProperties",
        "sources": ["Insight", "Memory", "Code"],
        "targets": ["Project", "Task", "Code"],
    },
}


def generate_relationships_for_node(node_type: str) -> list[str]:
    """Generate all relationship fields for a given node type"""
    lines = []
    lines.append("")
    lines.append("  # ========================================")
    lines.append(f"  # Relationships for {node_type}")
    lines.append("  # ========================================")

    # Outgoing relationships
    for rel_name, rel_info in RELATIONSHIPS.items():
        if node_type in rel_info["sources"]:
            for target in rel_info["targets"]:
                field_name = _to_camel_case(rel_name.lower()) + target
                lines.append(f"  # {rel_name}: {node_type} -> {target}")
                lines.append(
                    f'  {field_name}: [{target}!]! @relationship(type: "{rel_name}", direction: OUT, properties: "{rel_info["properties"]}")'
                )

    # Incoming relationships
    for rel_name, rel_info in RELATIONSHIPS.items():
        if node_type in rel_info["targets"]:
            inverse_name = _get_inverse_name(rel_name)
            for source in rel_info["sources"]:
                field_name = _to_camel_case(inverse_name.lower()) + source
                lines.append(f"  # {rel_name} (inverse): {source} -> {node_type}")
                lines.append(
                    f'  {field_name}: [{source}!]! @relationship(type: "{rel_name}", direction: IN, properties: "{rel_info["properties"]}")'
                )

    return lines


def _to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def _get_inverse_name(rel_name: str) -> str:
    """Get inverse relationship name"""
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
    """Add relationship definitions using regex to find node boundaries"""
    content = input_file.read_text()

    # Pattern to match a complete node definition
    # Matches from "type NodeName @node" to the closing "}" before the next "type"
    pattern = r'(type\s+(\w+)\s+@node.*?)\n\}(?=\n\ntype\s|\n\n#\s|$)'

    def replace_node(match):
        """Replace function that adds relationships before the closing brace"""
        node_def = match.group(1)
        node_name = match.group(2)

        # Generate relationships for this node
        relationships = generate_relationships_for_node(node_name)

        # Add relationships before the closing brace
        return node_def + '\n' + '\n'.join(relationships) + '\n}'

    # Replace all node definitions
    result = re.sub(pattern, replace_node, content, flags=re.DOTALL)

    # Write output
    output_file.write_text(result)
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
    print(f"   2. Add relationship property types from papr_graphql.graphql")
    print(f"   3. Upload to Neo4j Aura Console")
