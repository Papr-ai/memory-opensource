"""
ActiveNodeRel Parse model for caching Neo4j schema patterns to optimize search performance.

This model stores the available node types, relationship types, and their combinations
along with counts, eliminating the need to run expensive schema discovery queries during search.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class NodeRelationshipPattern(BaseModel):
    """Represents a specific node-relationship-node pattern with metadata"""
    source_label: str
    relationship_type: str
    target_label: str
    count: int
    source_properties: List[str] = Field(default_factory=list)
    target_properties: List[str] = Field(default_factory=list)


class ActiveNodeRel(BaseModel):
    """
    Parse model for caching Neo4j schema patterns to optimize search performance.
    
    This eliminates the expensive schema discovery query during search by pre-computing
    and caching the available node/relationship combinations after each memory add.
    """
    
    # Parse required fields
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    
    # User/workspace identification
    user_id: str
    workspace_id: Optional[str] = None
    
    # Cached schema data
    available_nodes: List[str] = Field(default_factory=list)
    available_relationships: List[str] = Field(default_factory=list)
    relationship_patterns: List[NodeRelationshipPattern] = Field(default_factory=list)
    
    # Metadata
    total_patterns: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    schema_version: str = "1.0"  # For future schema evolution
    
    class Config:
        # Parse configuration
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ActiveNodeRelService:
    """Service for managing ActiveNodeRel cache in Parse"""
    
    @staticmethod
    def create_from_neo4j_patterns(
        user_id: str, 
        workspace_id: Optional[str],
        patterns: List[Dict[str, Any]]
    ) -> ActiveNodeRel:
        """
        Create an ActiveNodeRel instance from Neo4j schema discovery results.
        
        Args:
            user_id: The user ID
            workspace_id: Optional workspace ID
            patterns: List of pattern dictionaries from Neo4j schema discovery
            
        Returns:
            ActiveNodeRel instance ready for Parse storage
        """
        # Extract unique nodes and relationships
        nodes = set()
        relationships = set()
        relationship_patterns = []
        
        for pattern in patterns:
            # Handle both formats: Neo4j discovery uses 'source'/'target'/'relationship', Parse cache uses 'source_label'/'target_label'/'relationship_type'
            source_label = pattern.get('source') or pattern.get('source_label')
            relationship_type = pattern.get('relationship') or pattern.get('relationship_type')  
            target_label = pattern.get('target') or pattern.get('target_label')
            count = pattern.get('count', 0)
            source_properties = pattern.get('source_properties', [])
            target_properties = pattern.get('target_properties', [])
            
            if source_label and relationship_type and target_label:
                # Add to sets
                nodes.add(source_label)
                nodes.add(target_label)
                relationships.add(relationship_type)
                
                # Create pattern
                relationship_patterns.append(
                    NodeRelationshipPattern(
                        source_label=source_label,
                        relationship_type=relationship_type,
                        target_label=target_label,
                        count=count,
                        source_properties=source_properties if isinstance(source_properties, list) else [],
                        target_properties=target_properties if isinstance(target_properties, list) else []
                    )
                )
        
        return ActiveNodeRel(
            user_id=user_id,
            workspace_id=workspace_id,
            available_nodes=sorted(list(nodes)),
            available_relationships=sorted(list(relationships)),
            relationship_patterns=relationship_patterns,
            total_patterns=len(relationship_patterns),
            last_updated=datetime.utcnow()
        )
    
    @staticmethod
    def to_search_format(active_node_rel: ActiveNodeRel) -> Dict[str, Any]:
        """
        Convert ActiveNodeRel to the format expected by search functions.
        
        Returns:
            Dictionary with 'nodes', 'relationships', and 'patterns' keys
        """
        patterns = []
        for pattern in active_node_rel.relationship_patterns:
            patterns.append({
                'source': pattern.source_label,
                'relationship': pattern.relationship_type,
                'target': pattern.target_label,
                'count': pattern.count,
                'source_properties': pattern.source_properties,
                'target_properties': pattern.target_properties
            })
        
        return {
            'nodes': active_node_rel.available_nodes,
            'relationships': active_node_rel.available_relationships,
            'patterns': patterns
        }


# Parse server configuration for ActiveNodeRel
ACTIVE_NODE_REL_PARSE_CONFIG = {
    "className": "ActiveNodeRel",
    "fields": {
        "user_id": {"type": "String", "required": True},
        "workspace_id": {"type": "String"},
        "available_nodes": {"type": "Array"},
        "available_relationships": {"type": "Array"},
        "relationship_patterns": {"type": "Array"},
        "total_patterns": {"type": "Number"},
        "last_updated": {"type": "Date"},
        "schema_version": {"type": "String"}
    },
    "indexes": {
        "user_workspace": {"user_id": 1, "workspace_id": 1}
    },
    "classLevelPermissions": {
        "find": {"*": True},
        "get": {"*": True},
        "create": {"*": True},
        "update": {"*": True},
        "delete": {"*": True}
    }
}





