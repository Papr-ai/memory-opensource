from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from models.shared_types import RelationshipType
from models.memory_models import (
    NeoMemoryNode, NeoPersonNode, NeoCompanyNode, NeoProjectNode, 
    NeoTaskNode, NeoInsightNode, NeoMeetingNode, NeoOpportunityNode, NeoCodeNode,
    FlexibleCustomNode
)

class PathSegment(BaseModel):
    """Represents a segment in a Neo4j path with Neo node types"""
    start_node: Union[
        NeoMemoryNode,
        NeoPersonNode,
        NeoCompanyNode,
        NeoProjectNode,
        NeoTaskNode,
        NeoInsightNode,
        NeoMeetingNode,
        NeoOpportunityNode,
        NeoCodeNode,
        FlexibleCustomNode  # Support for custom node types
    ]
    relationship: RelationshipType
    end_node: Union[
        NeoMemoryNode,
        NeoPersonNode,
        NeoCompanyNode,
        NeoProjectNode,
        NeoTaskNode,
        NeoInsightNode,
        NeoMeetingNode,
        NeoOpportunityNode,
        NeoCodeNode,
        FlexibleCustomNode  # Support for custom node types
    ]
    
    def to_context_string(self) -> str:
        """Convert path segment to human readable context"""
        context = ""
        
        # Handle Task relationships
        if isinstance(self.start_node, NeoTaskNode):
            context = f"Task: {self.start_node.title} ({self.start_node.status})"
            if self.relationship == "BELONGS_TO" and isinstance(self.end_node, NeoTaskNode):
                context += f" belongs to task {self.end_node.title}"
            elif self.relationship == "CREATED_BY" and isinstance(self.end_node, NeoPersonNode):
                context += f" created by {self.end_node.name}"
                
        # Handle Person relationships
        elif isinstance(self.start_node, NeoPersonNode):
            if self.relationship == "CREATED_BY":
                if isinstance(self.end_node, NeoTaskNode):
                    context = f"{self.start_node.name} ({self.start_node.role}) created task: {self.end_node.title}"
            elif self.relationship == "WORKS_AT" and isinstance(self.end_node, NeoCompanyNode):
                context = f"{self.start_node.name} ({self.start_node.role}) works at {self.end_node.name}"
                
        # Handle Memory relationships
        elif isinstance(self.start_node, NeoMemoryNode):
            context = f"Memory: {self.start_node.content[:100]}..."  # Truncate long content
            if self.relationship == "RELATED_TO" and isinstance(self.end_node, NeoTaskNode):
                context += f" related to task {self.end_node.title}"
            elif self.relationship == "CREATED_BY" and isinstance(self.end_node, NeoPersonNode):
                context += f" created by {self.end_node.name}"
                
        # Handle Project relationships
        elif isinstance(self.start_node, NeoProjectNode):
            context = f"Project: {self.start_node.name} ({self.start_node.type})"
            if self.relationship == "MANAGED_BY" and isinstance(self.end_node, NeoPersonNode):
                context += f" managed by {self.end_node.name}"
            elif self.relationship == "BELONGS_TO" and isinstance(self.end_node, NeoCompanyNode):
                context += f" belongs to {self.end_node.name}"
                
        # Handle Company relationships
        elif isinstance(self.start_node, NeoCompanyNode):
            context = f"Company: {self.start_node.name}"
            if self.relationship == "HAS" and isinstance(self.end_node, NeoProjectNode):
                context += f" has project {self.end_node.name}"
            elif self.relationship == "ASSOCIATED_WITH" and isinstance(self.end_node, NeoCompanyNode):
                context += f" associated with {self.end_node.name}"
                
        # Handle Insight relationships
        elif isinstance(self.start_node, NeoInsightNode):
            context = f"Insight: {self.start_node.title} ({self.start_node.type})"
            if self.relationship == "RELATED_TO" and isinstance(self.end_node, NeoProjectNode):
                context += f" related to project {self.end_node.name}"
            elif self.relationship == "REPORTED_BY" and isinstance(self.end_node, NeoPersonNode):
                context += f" reported by {self.end_node.name}"
                
        # Handle Meeting relationships
        elif isinstance(self.start_node, NeoMeetingNode):
            context = f"Meeting: {self.start_node.title} ({self.start_node.date})"
            if self.relationship == "PARTICIPATED_IN" and isinstance(self.end_node, NeoPersonNode):
                context += f" with participant {self.end_node.name}"
            elif self.relationship == "RELATED_TO" and isinstance(self.end_node, NeoProjectNode):
                context += f" related to project {self.end_node.name}"
                
        # Handle Opportunity relationships
        elif isinstance(self.start_node, NeoOpportunityNode):
            context = f"Opportunity: {self.start_node.title} (${self.start_node.value:,.2f}, {self.start_node.stage})"
            if self.relationship == "MANAGED_BY" and isinstance(self.end_node, NeoPersonNode):
                context += f" managed by {self.end_node.name}"
            elif self.relationship == "BELONGS_TO" and isinstance(self.end_node, NeoCompanyNode):
                context += f" belongs to {self.end_node.name}"
                
        # Handle Code relationships
        elif isinstance(self.start_node, NeoCodeNode):
            context = f"Code: {self.start_node.name} ({self.start_node.language})"
            if self.relationship == "CREATED_BY" and isinstance(self.end_node, NeoPersonNode):
                context += f" created by {self.end_node.name}"
            elif self.relationship == "BELONGS_TO" and isinstance(self.end_node, NeoProjectNode):
                context += f" belongs to project {self.end_node.name}"
                
        # Handle Custom Node relationships
        elif isinstance(self.start_node, FlexibleCustomNode):
            # Get a meaningful identifier from the custom node
            node_name = getattr(self.start_node, 'name', None) or getattr(self.start_node, 'title', None) or self.start_node.id
            node_type = getattr(self.start_node, 'label', 'CustomNode')  # Use label if available
            context = f"{node_type}: {node_name}"
            
            # Handle common relationship patterns for custom nodes
            if isinstance(self.end_node, NeoMemoryNode):
                context += f" {self.relationship.lower()} memory: {self.end_node.content[:50]}..."
            elif isinstance(self.end_node, FlexibleCustomNode):
                end_name = getattr(self.end_node, 'name', None) or getattr(self.end_node, 'title', None) or self.end_node.id
                end_type = getattr(self.end_node, 'label', 'CustomNode')
                context += f" {self.relationship.lower()} {end_type.lower()}: {end_name}"
            else:
                context += f" {self.relationship.lower()} {type(self.end_node).__name__.replace('Neo', '').replace('Node', '')}"
                
        return context

class GraphPath(BaseModel):
    """Represents a complete path in the graph"""
    segments: List[PathSegment]
    length: int
    
    def to_context_string(self) -> str:
        """Convert entire path to human readable context"""
        # Get context for each segment
        context_parts = []
        for segment in self.segments:
            segment_context = segment.to_context_string()
            if segment_context:  # Only add non-empty contexts
                context_parts.append(segment_context)
        
        # Join with separator and filter out empty strings
        return " | ".join(filter(None, context_parts))

class QueryResult(BaseModel):
    """The final query result with paths and context"""
    paths: List[GraphPath]
    query: Optional[str] = None
    
    def get_related_context(self) -> str:
        """Get all path contexts combined"""
        contexts = [path.to_context_string() for path in self.paths]
        return "\n".join(filter(None, contexts))

# Example usage:
"""
# Converting Neo4j JSON to context:
{
    "path": {
        "segments": [
            {
                "start": {"labels": ["Person"], "properties": {"name": "Shawkat", "role": "CEO"}},
                "relationship": {"type": "WORKS_AT"},
                "end": {"labels": ["Company"], "properties": {"name": "Papr"}}
            },
            {
                "start": {"labels": ["Task"], "properties": {"title": "Update pitch deck", "status": "In Progress"}},
                "relationship": {"type": "ASSIGNED_TO"},
                "end": {"labels": ["Person"], "properties": {"name": "Shawkat"}}
            }
        ]
    }
}

# Would become:
"Shawkat (CEO) works at Papr | Task: Update pitch deck (In Progress) assigned to Shawkat"
"""