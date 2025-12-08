from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from typing import List, Literal, Dict, Any, Union, TypedDict, Optional
import logging
from enum import Enum
from typing_extensions import Annotated
from models.parse_server import ParseStoredMemory
from models.shared_types import NodeLabel, RelationshipType

logger = logging.getLogger(__name__)

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Goal(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["title", "description", "status", "key_results"],
            "additionalProperties": False
        }
    )
    title: str = Field(description="The title of the goal")
    description: str = Field(description="The description of the goal")
    status: Literal["new", "existing"] = Field(description="Whether this is a new or existing goal")
    key_results: List[str] = Field(description="List of key results for this goal")


class UseCase(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["name", "description", "status"],
            "additionalProperties": False
        }
    )
    name: str = Field(description="The name of the use case")
    description: str = Field(description="The description of the use case")
    status: Literal["new", "existing"] = Field(description="Whether this is a new or existing use case")


class UseCaseMemoryItem(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["goals", "use_cases"],
            "additionalProperties": False
        }
    )
    goals: List[Goal] = Field(description="List of goals extracted from the memory item")
    use_cases: List[UseCase] = Field(description="List of use cases extracted from the memory item")


# Properties models for each node type
class MemoryProperties(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["id", "content", "type", "topics", "emotion_tags", "steps", "current_step"],
            "additionalProperties": False
        }
    )
    id: str
    content: str
    type: str
    topics: List[str]
    emotion_tags: List[str]
    steps: List[str]
    current_step: str
    
class PersonProperties(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["id", "name", "role", "description"],
            "additionalProperties": False
        }
    )
    id: str
    name: str
    role: str
    description: str

    @model_validator(mode="before")
    @classmethod
    def accept_content_as_description(cls, data):
        # Accept 'content' as 'description' if present
        if isinstance(data, dict):
            if 'description' not in data and 'content' in data:
                data['description'] = data['content']
        return data
    

class CompanyProperties(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["id", "name", "description"],
            "additionalProperties": False
        }
    )
    id: str
    name: str
    description: str

    @model_validator(mode="before")
    @classmethod
    def accept_content_as_description(cls, data):
        # Accept 'content' as 'description' if present
        if isinstance(data, dict):
            if 'description' not in data and 'content' in data:
                data['description'] = data['content']
        return data
    

class CustomerProperties(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["id", "name", "status", "description"],
            "additionalProperties": False
        }
    )
    id: str
    name: str
    status: Literal["prospect", "active", "churn_risk", "lost", "up_for_renewal"]
    description: str

    @model_validator(mode="before")
    @classmethod
    def accept_content_as_description(cls, data):
        # Accept 'content' as 'description' if present
        if isinstance(data, dict):
            if 'description' not in data and 'content' in data:
                data['description'] = data['content']
        return data
    

class ProjectProperties(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["id", "name", "type", "description"],
            "additionalProperties": False
        }
    )
    id: str
    name: str
    type: str
    description: str

    @model_validator(mode="before")
    @classmethod
    def accept_content_as_description(cls, data):
        # Accept 'content' as 'description' if present
        if isinstance(data, dict):
            if 'description' not in data and 'content' in data:
                data['description'] = data['content']
        return data
    

class TaskProperties(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["id", "title", "description", "status", "type"],
            "additionalProperties": False
        }
    )
    id: str
    title: str
    description: str
    status: Literal["new", "in_progress", "completed"]
    type: Literal["task", "subtask", "bug", "feature_request", "epic", "support_ticket"]
    priority: Optional[TaskPriority] = TaskPriority.MEDIUM
    date: Optional[str] = Field(
        default=None,
        description="Due date for the task in ISO 8601 format",
        json_schema_extra={"format": "date-time"}
    )

    @model_validator(mode="before")
    @classmethod
    def accept_content_as_description(cls, data):
        # Accept 'content' as 'description' if present
        if isinstance(data, dict):
            if 'description' not in data and 'content' in data:
                data['description'] = data['content']
        return data

    @model_validator(mode="before")
    @classmethod
    def accept_name_as_title(cls, data):
        # Accept 'name' as synonym for 'title' if 'title' is missing
        if isinstance(data, dict):
            if 'title' not in data and 'name' in data:
                data['title'] = data['name']
        return data

    @model_validator(mode="before")
    @classmethod
    def accept_date_or_due_date(cls, data):
        # Accept both 'date' and 'due_date', prefer 'due_date'
        if isinstance(data, dict):
            if 'date' not in data and 'due_date' in data:
                data['date'] = data['due_date']
        return data




class InsightProperties(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def accept_content_as_description(cls, data):
        # Accept 'content' as 'description' if present
        if isinstance(data, dict):
            if 'description' not in data and 'content' in data:
                data['description'] = data['content']
        return data
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["id", "title", "description", "source", "type"],
            "additionalProperties": False
        }
    )
    id: str
    title: str
    description: str
    source: str
    type: Literal["customer_insight", "product_insight", "market_insight", "competitive_insight", "technical_insight", "other"]
    

class MeetingProperties(BaseModel):
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["id", "title", "agenda", "type", "participants", "date", "time", "summary", "outcome", "action_items"],
            "additionalProperties": False
        }
    )
    id: str
    title: str
    agenda: str
    type: str
    participants: List[str]
    date: str
    time: str
    summary: str
    outcome: str
    action_items: List[str]


class OpportunityProperties(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["id", "title", "description", "value", "stage", "close_date", "probability", "next_steps"],
            "additionalProperties": False
        }
    )
    id: str
    title: str
    description: str
    value: float
    stage: Literal["prospect", "lead", "opportunity", "won", "lost"]
    close_date: str
    probability: float
    next_steps: List[str]
    
class CodeProperties(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["id", "name", "language", "author"],
            "additionalProperties": False
        }
    )
    id: str
    name: str
    language: str
    author: str


# Now define separate node models for each label using a discriminated union
class BaseNode(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label"],
            "additionalProperties": False
        }
    )
    label: NodeLabel

class MemoryNode(BaseNode):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label", "properties"],
            "additionalProperties": False
        }
    )
    # Don't set a default value; use Field(...) to mark it as required
    label: Literal[NodeLabel.Memory] = Field(..., description="The label for a Memory node")
    properties: MemoryProperties = Field(..., description="Properties of the memory node")


class PersonNode(BaseNode):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label", "properties"],
            "additionalProperties": False
        }
    )
    label: Literal[NodeLabel.Person] = Field(..., description="The label for a Person node")
    properties: PersonProperties = Field(..., description="Properties of the person node")


class CompanyNode(BaseNode):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label", "properties"],
            "additionalProperties": False
        }
    )
    label: Literal[NodeLabel.Company] = Field(..., description="The label for a Company node")
    properties: CompanyProperties = Field(..., description="Properties of the company node")

class CustomerNode(BaseNode):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label", "properties"],
            "additionalProperties": False
        }
    )
    label: Literal[NodeLabel.Company] = Field(..., description="The label for a Customer node")
    properties: CustomerProperties = Field(..., description="Properties of the customer node")

class ProjectNode(BaseNode):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label", "properties"],
            "additionalProperties": False
        }
    )
    label: Literal[NodeLabel.Project] = Field(..., description="The label for a Project node")
    properties: ProjectProperties = Field(..., description="Properties of the project node")

class TaskNode(BaseNode):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label", "properties"],
            "additionalProperties": False
        }
    )
    label: Literal[NodeLabel.Task] = Field(..., description="The label for a Task node")
    properties: TaskProperties = Field(..., description="Properties of the task node")

class InsightNode(BaseNode):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label", "properties"],
            "additionalProperties": False
        }
    )
    label: Literal[NodeLabel.Insight] = Field(..., description="The label for an Insight node")
    properties: InsightProperties = Field(..., description="Properties of the insight node")

class MeetingNode(BaseNode):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label", "properties"],
            "additionalProperties": False
        }
    )
    label: Literal[NodeLabel.Meeting] = Field(..., description="The label for a Meeting node")
    properties: MeetingProperties = Field(..., description="Properties of the meeting node")

class CodeNode(BaseNode):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label", "properties"],
            "additionalProperties": False
        }
    )
    label: Literal[NodeLabel.Code] = Field(..., description="The label for a Code node")
    properties: CodeProperties = Field(..., description="Properties of the code node")

class OpportunityNode(BaseNode):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label", "properties"],
            "additionalProperties": False
        }
    )
    label: Literal[NodeLabel.Opportunity] = Field(..., description="The label for an Opportunity node")
    properties: OpportunityProperties = Field(..., description="Properties of the opportunity node")    

# Define a Union that Pydantic will render as a oneOf in the JSON schema
NodeUnion = Annotated[
    Union[
        MemoryNode,
        PersonNode,
        CompanyNode,
        ProjectNode,
        TaskNode,
        InsightNode,
        MeetingNode,
        CodeNode,
        OpportunityNode 
    ],
    Field(discriminator='label')
]

# A generic node reference containing label and id
class NodeReference(BaseModel):
    label: str = Field(..., description="The label of the node")
    id: str = Field(..., description="The ID of the node")

    @field_validator('label')
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate that label is either a system label or custom label"""
        if NodeLabel.is_system_label(v):
            return v
        # Accept custom labels - validation happens later with context
        return v

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "required": ["label", "id"],
            "additionalProperties": False
        }
    )

class LLMGraphNode(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra= {
            "anyOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": { "type": "string", "enum": ["Memory"] },
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": { "type": "string" },
                                                "content": { "type": "string" },
                                                "type": { "type": "string" },
                                                "topics": { "type": "array", "items": { "type": "string" } },
                                                "emotion_tags": { "type": "array", "items": { "type": "string" } },
                                                "steps": { "type": "array", "items": { "type": "string" } },
                                                "current_step": { "type": "string" }
                                            },
                                            "required": ["id", "content", "type", "topics", "emotion_tags", "steps", "current_step"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Person"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "role": {
                                                    "type": "string"
                                                },
                                                "description": {"type": "string"}
                                            },
                                            "required": ["id", "name", "role", "description"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Company"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "description": {"type": "string"}
                                            },
                                            "required": ["id", "name", "description"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                            
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": { "type": "string", "enum": ["Project"] },
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": { "type": "string" },
                                                "type": { "type": "string" },
                                                "description": { "type": "string" }
                                            },
                                            "required": ["id", "name", "type", "description"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Task"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "title": {"type": "string"},
                                                "description": {"type": "string"},
                                                "status": {"type": "string", "enum": ["new", "in_progress", "completed"]},
                                                "type": {"type": "string", "enum": ["task", "subtask", "bug", "feature_request", "epic", "support_ticket"]},
                                                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"], "default": "medium"},
                                                "date": {"type": "string", "format": "date-time"}
                                            },
                                            "required": ["id", "title", "description", "status", "type", "date", "priority"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Insight"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "title": {"type": "string"},
                                                "description": {"type": "string"},
                                                "source": {
                                                    "type": "string"
                                                },
                                                "type": {"type": "string", "enum": ["customer_insight", "product_insight", "market_insight", "competitive_insight", "technical_insight", "other"]}
                                            },
                                            "required": ["id", "title", "description", "source", "type"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Meeting"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "title": {"type": "string"},
                                                "agenda": {
                                                    "type": "string"
                                                },
                                                "type": {
                                                    "type": "string"
                                                },
                                                "participants": {
                                                    "type": "array", 
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                },
                                                "date": {
                                                    "type": "string"
                                                },
                                                "time": {
                                                    "type": "string"
                                                },
                                                "summary": {"type": "string"},
                                                "outcome": {"type": "string"},
                                                "action_items": {"type": "array", "items": {"type": "string"}}
                                            },
                                            "required": ["id", "title", "agenda", "type", "participants", "date", "time", "summary", "outcome", "action_items"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Opportunity"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "title": {"type": "string"},
                                                "description": {"type": "string"},
                                                "value": {
                                                    "type": "number"
                                                },
                                                "stage": {"type": "string", "enum": ["prospect", "lead", "opportunity", "won", "lost"]},
                                                "close_date": {
                                                    "type": "string"
                                                },
                                                "probability": {"type": "number"},
                                                "next_steps": {"type": "array", "items": {"type": "string"}}
                                            },
                                            "required": ["id", "title", "description", "value", "stage", "close_date", "probability", "next_steps"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Code"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "language": {"type": "string"},
                                                "author": {
                                                    "type": "string"
                                                }
                                            },
                                            "required": ["id", "name", "language", "author"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                }
                            ]
        }
    )
    label: str = Field(..., description="The label of the node")
    properties: Dict[str, Any] = Field(..., description="Properties of the node")

    @field_validator('label')
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate that label is either a system label or matches custom labels from context"""
        # For now, we'll accept any string and let the downstream validation handle it
        # The actual validation happens in generate_node_ids() with custom_labels
        if NodeLabel.is_system_label(v):
            return v
        # If it's not a system label, assume it's a custom label and let it pass
        # The validation with custom_labels happens later in the pipeline
        return v

    @classmethod    
    def get_fixed_json_schema(cls) -> Dict[str, Any]:
        """
        Return the oneOf-based JSON schema you provided. This ensures that 
        the OpenAI structured outputs must conform exactly to this schema.
        """
        return {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "label": { "type": "string", "enum": ["Memory"] },
                                    "properties": {
                                        "type": "object",
                                        "properties": {
                                            "id": { "type": "string" },
                                            "content": { "type": "string" },
                                            "type": { "type": "string" },
                                            "topics": { "type": "array", "items": { "type": "string" } },
                                            "emotion_tags": { "type": "array", "items": { "type": "string" } },
                                            "steps": { "type": "array", "items": { "type": "string" } },
                                            "current_step": { "type": "string" }
                                        },
                                        "required": ["id", "content", "type", "topics", "emotion_tags", "steps", "current_step"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["label", "properties"],
                                "additionalProperties": False
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "enum": ["Person"]},
                                    "properties": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "name": {"type": "string"},
                                            "role": {
                                                "type": "string"
                                            },
                                            "description": {"type": "string"}
                                        },
                                        "required": ["id", "name", "role", "description"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["label", "properties"],
                                "additionalProperties": False
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "enum": ["Company"]},
                                    "properties": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "name": {"type": "string"},
                                            "description": {"type": "string"}
                                        },
                                        "required": ["id", "name", "description"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["label", "properties"],
                                "additionalProperties": False
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "label": { "type": "string", "enum": ["Project"] },
                                    "properties": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "name": { "type": "string" },
                                            "type": { "type": "string" },
                                            "description": { "type": "string" }
                                        },
                                        "required": ["id", "name", "type", "description"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["label", "properties"],
                                "additionalProperties": False
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "enum": ["Task"]},
                                    "properties": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "title": {"type": "string"},
                                            "description": {"type": "string"},
                                            "status": {"type": "string", "enum": ["new", "in_progress", "completed"]},
                                            "type": {"type": "string", "enum": ["task", "subtask", "bug", "feature_request", "epic", "support_ticket"]},
                                            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"], "default": "medium"},
                                            "date": {"type": "string", "format": "date-time"}
                                        },
                                        "required": ["id", "title", "description", "status", "type", "date", "priority"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["label", "properties"],
                                "additionalProperties": False
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "enum": ["Insight"]},
                                    "properties": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "title": {"type": "string"},
                                            "description": {"type": "string"},
                                            "source": {
                                                "type": "string"
                                            },
                                            "type": {"type": "string", "enum": ["customer_insight", "product_insight", "market_insight", "competitive_insight", "technical_insight", "other"]}
                                        },
                                        "required": ["id", "title", "description", "source", "type"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["label", "properties"],
                                "additionalProperties": False
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "enum": ["Meeting"]},
                                    "properties": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "title": {"type": "string"},
                                            "agenda": {
                                                "type": "string"
                                            },
                                            "type": {
                                                "type": "string"
                                            },
                                            "participants": {
                                                "type": "array", 
                                                "items": {
                                                    "type": "string"
                                                }
                                            },
                                            "date": {
                                                "type": "string"
                                            },
                                            "time": {
                                                "type": "string"
                                            },
                                            "summary": {"type": "string"},
                                            "outcome": {"type": "string"},
                                            "action_items": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["id", "title", "agenda", "type", "participants", "date", "time", "summary", "outcome", "action_items"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["label", "properties"],
                                "additionalProperties": False
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "enum": ["Opportunity"]},
                                    "properties": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "title": {"type": "string"},
                                            "description": {"type": "string"},
                                            "value": {
                                                "type": "number"
                                            },
                                            "stage": {"type": "string", "enum": ["prospect", "lead", "opportunity", "won", "lost"]},
                                            "close_date": {
                                                "type": "string"
                                            },
                                            "probability": {"type": "number"},
                                            "next_steps": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["id", "title", "description", "value", "stage", "close_date", "probability", "next_steps"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["label", "properties"],
                                "additionalProperties": False
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "enum": ["Code"]},
                                    "properties": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "name": {"type": "string"},
                                            "language": {"type": "string"},
                                            "author": {
                                                "type": "string"
                                            }
                                        },
                                        "required": ["id", "name", "language", "author"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["label", "properties"],
                                "additionalProperties": False
                            }
                        ]
                    }
                },
            },
            "required": ["nodes"],
            "additionalProperties": False
            }
    
# Relationship model with object references for source and target
class LLMGraphRelationship(BaseModel):
    type: str = Field(..., description="The type of relationship")
    direction: Literal["->", "<-"] = Field(..., description="Direction of the relationship")
    source: NodeReference = Field(..., description="Source node reference")
    target: NodeReference = Field(..., description="Target node reference")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that relationship type is either a system type or custom type"""
        if RelationshipType.is_system_relationship(v):
            return v
        # Accept custom relationships - validation happens later with context
        return v
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [r.value for r in RelationshipType]
                },
                "direction": { "type": "string", "enum": ["->", "<-"] },
                "source": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": [l.value for l in NodeLabel]
                        },
                        "id": {"type": "string"}
                    },
                    "required": ["label", "id"],
                    "additionalProperties": False
                },
                "target": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": [l.value for l in NodeLabel]
                        },
                        "id": {"type": "string"}
                    },
                    "required": ["label", "id"],
                    "additionalProperties": False
                }
            },
            "required": ["type", "direction", "source", "target"],
            "additionalProperties": False
        }
    )

    @classmethod    
    def get_fixed_json_schema(cls) -> Dict[str, Any]:
        """
        Return the oneOf-based JSON schema you provided. This ensures that 
        the OpenAI structured outputs must conform exactly to this schema.
        """
        return {
            "type": "object",
            "properties": {
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                            "CREATED_BY", "WORKS_AT", "ASSOCIATED_WITH",
                            "CONTAINS", "ASSIGNED_TO", "MANAGED_BY",
                            "RELATED_TO", "HAS", "IS_A", "PARTICIPATED_IN", "BELONGS_TO", "REPORTED_BY", "REFERENCES"
                            ]
                        },
                        "direction": { "type": "string", "enum": ["->", "<-"] },
                        "source": {
                            "type": "object",
                            "properties": {
                            "label": {
                                "type": "string",
                                "enum": ["Memory", "Person", "Company", "Project", "Task", "Insight", "Meeting", "Opportunity", "Code"]

                            },
                            "id": {"type": "string"}
                            },
                            "required": ["label", "id"],
                            "additionalProperties": False
                        },
                        "target": {
                            "type": "object",
                            "properties": {
                            "label": {
                                "type": "string",
                                "enum": ["Memory", "Person", "Company", "Project", "Task", "Insight", "Meeting", "Opportunity", "Code"]
                            },
                            "id": {"type": "string"}
                            },
                            "required": ["label", "id"],
                            "additionalProperties": False
                        }
                        },
                        "required": ["type", "direction", "source", "target"],
                        "additionalProperties": False
                    }
                    }
                },
                "required": ["relationships"],
                "additionalProperties": False
        }
    
# Finally, define the MemoryGraphSchema using these unions
class MemoryGraphSchema(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": { 
                        "$ref": "#/$defs/LLMGraphNode"
                    }
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "$ref": "#/$defs/LLMGraphRelationship"
                    }
                }
            },
            "required": ["nodes", "relationships"],
            "additionalProperties": False
        }
    )
    nodes: List[LLMGraphNode] = Field(..., description="List of nodes in the memory graph")
    relationships: List[LLMGraphRelationship] = Field(..., description="List of relationships between nodes")

    # The get_memory_only_schema and get_json_schema could remain as static methods
    # if you still want to provide them as custom schemas.
    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        # Return the schema generated by Pydantic for the entire model
        return cls.model_json_schema()
    @classmethod
    def get_memory_only_schema(cls) -> Dict[str, Any]:
        """
        Return a simplified JSON schema that only includes Memory nodes
        while maintaining all relationship types.
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "MemoryOnlyGraphSchema",
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": { "type": "string", "enum": ["Memory"] },
                            "properties": {
                                "type": "object",
                                "properties": {
                                  
                                    "content": { "type": "string" },
                                   
                                   
                                    "type": { "type": "string" },
                                    "topics": {
                                        "type": "array",
                                        "items": { "type": "string" }
                                    },
                                    "emotion_tags": {
                                        "type": "array",
                                        "items": { "type": "string" }
                                    },
                                    "steps": {
                                        "type": "array",
                                        "items": { "type": "string" }
                                    },
                                    "current_step": { "type": "string" }
                                },
                                "required": [ "content", "type", "topics", "emotion_tags", "steps", "current_step"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["label", "properties"],
                        "additionalProperties": False
                    }
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "CREATED_BY",
                                    "WORKS_AT",
                                    "ASSOCIATED_WITH",
                                    "CONTAINS",
                                    "ASSIGNED_TO",
                                    "MANAGED_BY",
                                    "RELATED_TO"
                                ]
                            },
                            "direction": { "type": "string", "enum": ["->", "<-"] },
                            "source": { "type": "string" },
                            "target": { "type": "string" },
                        },
                        "required": ["type", "direction", "source", "target"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["nodes", "relationships"],
            "additionalProperties": False
        }
    @classmethod    
    def get_fixed_json_schema(cls) -> Dict[str, Any]:
        """
        Return the oneOf-based JSON schema you provided. This ensures that 
        the OpenAI structured outputs must conform exactly to this schema.
        """
        return {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": { "type": "string", "enum": ["Memory"] },
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": { "type": "string" },
                                                "content": { "type": "string" },
                                                "type": { "type": "string" },
                                                "topics": { "type": "array", "items": { "type": "string" } },
                                                "emotion_tags": { "type": "array", "items": { "type": "string" } },
                                                "steps": { "type": "array", "items": { "type": "string" } },
                                                "current_step": { "type": "string" }
                                            },
                                            "required": ["id", "content", "type", "topics", "emotion_tags", "steps", "current_step"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Person"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "role": {
                                                    "type": "string"
                                                },
                                                "description": {"type": "string"}
                                            },
                                            "required": ["id", "name", "role", "description"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Company"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "description": {"type": "string"}
                                            },
                                            "required": ["id", "name", "description"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": { "type": "string", "enum": ["Project"] },
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": { "type": "string" },
                                                "type": { "type": "string" },
                                                "description": { "type": "string" }
                                            },
                                            "required": ["id", "name", "type", "description"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Task"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "title": {"type": "string"},
                                                "description": {"type": "string"},
                                                "status": {"type": "string", "enum": ["new", "in_progress", "completed"]},
                                                "type": {"type": "string", "enum": ["task", "subtask", "bug", "feature_request", "epic", "support_ticket"]},
                                                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"], "default": "medium"},
                                                "date": {"type": "string", "format": "date-time"}
                                            },
                                            "required": ["id", "title", "description", "status", "type", "date", "priority"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Insight"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "title": {"type": "string"},
                                                "description": {"type": "string"},
                                                "source": {
                                                    "type": "string"
                                                },
                                                "type": {"type": "string", "enum": ["customer_insight", "product_insight", "market_insight", "competitive_insight", "technical_insight", "other"]}
                                            },
                                            "required": ["id", "title", "description", "source", "type"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Meeting"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "title": {"type": "string"},
                                                "agenda": {
                                                    "type": "string"
                                                },
                                                "type": {
                                                    "type": "string"
                                                },
                                                "participants": {
                                                    "type": "array", 
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                },
                                                "date": {
                                                    "type": "string"
                                                },
                                                "time": {
                                                    "type": "string"
                                                },
                                                "summary": {"type": "string"},
                                                "outcome": {"type": "string"},
                                                "action_items": {"type": "array", "items": {"type": "string"}}
                                            },
                                            "required": ["id", "title", "agenda", "type", "participants", "date", "time", "summary", "outcome", "action_items"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Opportunity"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "title": {"type": "string"},
                                                "description": {"type": "string"},
                                                "value": {
                                                    "type": "number"
                                                },
                                                "stage": {"type": "string", "enum": ["prospect", "lead", "opportunity", "won", "lost"]},
                                                "close_date": {
                                                    "type": "string"
                                                },
                                                "probability": {"type": "number"},
                                                "next_steps": {"type": "array", "items": {"type": "string"}}
                                            },
                                            "required": ["id", "title", "description", "value", "stage", "close_date", "probability", "next_steps"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string", "enum": ["Code"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "language": {"type": "string"},
                                                "author": {
                                                    "type": "string"
                                                }
                                            },
                                            "required": ["id", "name", "language", "author"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["label", "properties"],
                                    "additionalProperties": False
                                }
                            ]
                        }
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                "CREATED_BY", "WORKS_AT", "ASSOCIATED_WITH",
                                "CONTAINS", "ASSIGNED_TO", "MANAGED_BY",
                                "RELATED_TO", "HAS", "IS_A", "PARTICIPATED_IN", "BELONGS_TO", "REPORTED_BY", "REFERENCES"
                                ]
                            },
                            "direction": { "type": "string", "enum": ["->", "<-"] },
                            "source": {
                                "type": "object",
                                "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["Memory", "Person", "Company", "Project", "Task", "Insight", "Meeting", "Opportunity", "Code"]

                                },
                                "id": {"type": "string"}
                                },
                                "required": ["type", "id"],
                                "additionalProperties": False
                            },
                            "target": {
                                "type": "object",
                                "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["Memory", "Person", "Company", "Project", "Task", "Insight", "Meeting", "Opportunity", "Code"]
                                },
                                "id": {"type": "string"}
                                },
                                "required": ["type", "id"],
                                "additionalProperties": False
                            }
                            },
                            "required": ["type", "direction", "source", "target"],
                            "additionalProperties": False
                        }
                        }
                    },
                    "required": ["nodes", "relationships"],
                    "additionalProperties": False
                }

class MemoryMetrics(TypedDict):
    total_cost: float
    token_size: int
    storage_size: int
    operation_costs: Dict[str, float]

class SchemaResponse(TypedDict):
    data: MemoryGraphSchema
    metrics: Dict[str, float]

class ProcessMemoryResponse(TypedDict):
    status_code: int
    success: bool
    error: Optional[str]
    data: Optional[Dict[str, Any]]  # Will contain the ProcessMemoryData when success is True

class ProcessMemoryData(TypedDict):
    goal_usecases: Dict[str, List[Dict[str, Any]]]  # Contains goals and use cases
    memory_graph: MemoryGraphSchema
    related_memories: List[ParseStoredMemory]
    related_memories_relationships: List[LLMGraphRelationship]
    metrics: MemoryMetrics


class ModelFindRealtedmemmories(BaseModel):
    """
    Model for structured output when finding related memories.
    Generated from findrelatedmemory.json schema.
    """
    queries: Optional[List[str]] = Field(
        ...,
        description='List of queries that we need to run to get related memories for the user. Up to 3 queries',
    )