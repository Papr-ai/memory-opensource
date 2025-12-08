"""
Pydantic models for Manufacturing Floor Knowledge Graph schema.

This module defines type-safe models for the Manufacturing Floor Knowledge Graph,
representing facilities, production lines, machines, sensors, alarms, maintenance,
scheduling, and operational knowledge.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# Enum Types
# ============================================================================

class ProductionLineStatus(str, Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    PAUSED = "paused"
    DOWN = "down"
    RETIRED = "retired"


class MachineStatus(str, Enum):
    UP = "up"
    IDLE = "idle"
    SETUP = "setup"
    DOWN = "down"
    MAINTENANCE = "maintenance"
    RETIRED = "retired"


class AlarmSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    TRIP = "trip"


class DocumentType(str, Enum):
    MANUAL = "manual"
    DATASHEET = "datasheet"
    SERVICE_BULLETIN = "service_bulletin"
    WIRING_DIAGRAM = "wiring_diagram"
    SOP = "sop"


class ProcedureCategory(str, Enum):
    SETUP = "setup"
    OPERATION = "operation"
    MAINTENANCE = "maintenance"
    SAFETY = "safety"
    TROUBLESHOOTING = "troubleshooting"


class FailureModeSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WorkOrderStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"


class WorkOrderPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ProductionOrderStatus(str, Enum):
    PLANNED = "planned"
    IN_PROCESS = "in_process"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


class ScheduleSlotType(str, Enum):
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"
    SETUP = "setup"
    BUFFER = "buffer"


class ScheduleSlotStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class OperatorRole(str, Enum):
    OPERATOR = "operator"
    TECHNICIAN = "technician"
    ENGINEER = "engineer"
    SUPERVISOR = "supervisor"
    LINE_LEADER = "line_leader"


class KnowledgeNoteType(str, Enum):
    OBSERVATION = "observation"
    FIX = "fix"
    TIP = "tip"
    CAUTION = "caution"
    ROOT_CAUSE = "root_cause"


class KnowledgeSource(str, Enum):
    OPERATOR = "operator"
    AGENT = "agent"
    SYSTEM = "system"


class KnowledgeVisibility(str, Enum):
    PRIVATE = "private"
    LINE = "line"
    FACILITY = "facility"
    WORKSPACE = "workspace"


class KnowledgeStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


class AgentChannel(str, Enum):
    VOICE = "voice"
    CHAT = "chat"
    TERMINAL = "terminal"


class AgentPurpose(str, Enum):
    TRIAGE = "triage"
    NOTIFICATION = "notification"
    HOW_TO = "how_to"
    STATUS = "status"


# ============================================================================
# Node Type Models
# ============================================================================

class Facility(BaseModel):
    """A manufacturing plant or site."""
    model_config = ConfigDict(extra='forbid')
    
    id: str = Field(..., description="Stable facility identifier (UUID or code)")
    name: str = Field(..., min_length=1, max_length=200)
    location: Optional[str] = Field(None, description="City/Region/Country or geo URI")
    timezone: Optional[str] = Field(None, description="IANA timezone, e.g., America/Los_Angeles")
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class ProductionLine(BaseModel):
    """A logical line or area executing a sequence of process steps."""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    name: str = Field(..., max_length=200)
    code: Optional[str] = Field(None, description="Line code used in MES/ERP")
    status: Optional[ProductionLineStatus] = ProductionLineStatus.ACTIVE
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class Machine(BaseModel):
    """A production asset (e.g., compressor dryer, press, CNC)."""
    model_config = ConfigDict(extra='forbid')
    
    id: str = Field(..., description="Stable machine ID")
    name: str
    model: Optional[str] = None
    serial_number: Optional[str] = None
    vendor: Optional[str] = None
    status: Optional[MachineStatus] = MachineStatus.UP
    rated_pressure_psi: Optional[float] = Field(None, ge=0)
    rated_temperature_c: Optional[float] = None
    commissioned_at: Optional[datetime] = None
    last_service_at: Optional[datetime] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None
    external_id: Optional[str] = Field(None, description="Asset tag in CMMS/ERP")


class Sensor(BaseModel):
    """Telemetry source mounted on a machine or in a line area."""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    name: str
    type: Optional[str] = Field(None, description="e.g., temperature, vibration, dewpoint, pressure")
    unit: Optional[str] = None
    sampling_hz: Optional[float] = Field(None, ge=0)
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class ManualDocument(BaseModel):
    """Machine manual or service bulletin for troubleshooting and maintenance."""
    model_config = ConfigDict(extra='forbid')
    
    id: str = Field(..., description="Document ID or URI")
    title: str = Field(..., max_length=300)
    publisher: Optional[str] = None
    published_at: Optional[datetime] = None
    language: Optional[str] = None
    document_type: Optional[DocumentType] = None
    doc_url: Optional[str] = Field(None, description="Link to the file in storage")
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class Procedure(BaseModel):
    """SOP for setup, troubleshooting, or maintenance."""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    name: str
    category: Optional[ProcedureCategory] = None
    revision: Optional[str] = None
    effective_at: Optional[datetime] = None
    estimated_duration_min: Optional[int] = Field(None, ge=0)
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class FailureMode(BaseModel):
    """Canonical failure mode (FMEA-style) or symptom."""
    model_config = ConfigDict(extra='forbid')
    
    id: str = Field(..., description="Stable failure code, e.g., FM-0001")
    name: str
    severity: Optional[FailureModeSeverity] = None
    description: Optional[str] = Field(None, max_length=500)
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class AlarmEvent(BaseModel):
    """Discrete alarm or event raised by a controller/PLC/HMI."""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    code: str = Field(..., description="Controller code or mnemonic, e.g., H2, L2")
    message: Optional[str] = Field(None, max_length=500)
    severity: Optional[AlarmSeverity] = AlarmSeverity.ALARM
    occurred_at: datetime
    cleared_at: Optional[datetime] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class WorkOrder(BaseModel):
    """Maintenance or corrective work instruction."""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    title: str
    status: WorkOrderStatus
    priority: Optional[WorkOrderPriority] = WorkOrderPriority.MEDIUM
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None
    external_id: Optional[str] = Field(None, description="CMMS work order number")


class ProductionOrder(BaseModel):
    """A customer/ERP order to produce a quantity of a SKU."""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    order_number: str
    sku: Optional[str] = None
    quantity: Optional[int] = Field(None, ge=0)
    due_at: Optional[datetime] = None
    status: Optional[ProductionOrderStatus] = ProductionOrderStatus.PLANNED
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class ProcessStep(BaseModel):
    """A step in a routing or line sequence."""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    name: str
    sequence: Optional[int] = Field(None, ge=0, description="Order within a line routing")
    cycle_time_sec: Optional[int] = Field(None, ge=0)
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class ScheduleSlot(BaseModel):
    """A scheduled production or maintenance window."""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    start_at: datetime
    end_at: datetime
    slot_type: Optional[ScheduleSlotType] = None
    status: Optional[ScheduleSlotStatus] = ScheduleSlotStatus.PLANNED
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class Operator(BaseModel):
    """A person operating or maintaining equipment."""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    name: str
    role: Optional[OperatorRole] = None
    external_user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class KnowledgeNote(BaseModel):
    """Human- or agent-contributed observation, fix, tip, or root-cause note captured on the floor."""
    model_config = ConfigDict(extra='forbid')
    
    id: str = Field(..., description="Stable note ID")
    title: str = Field(..., max_length=200)
    body: str
    note_type: KnowledgeNoteType
    source: KnowledgeSource = Field(..., description="Declares who originated the note")
    created_at: datetime
    updated_at: Optional[datetime] = None
    visibility: Optional[KnowledgeVisibility] = KnowledgeVisibility.LINE
    status: Optional[KnowledgeStatus] = KnowledgeStatus.APPROVED
    tags: Optional[List[str]] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class AgentSession(BaseModel):
    """A conversational session between the AI agent and a user (e.g., line leader) for triage, guidance, and notifications."""
    model_config = ConfigDict(extra='forbid')
    
    id: str = Field(..., description="Stable session ID")
    started_at: datetime
    ended_at: Optional[datetime] = None
    channel: Optional[AgentChannel] = None
    purpose: Optional[AgentPurpose] = None
    summary: Optional[str] = Field(None, max_length=500)
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


# ============================================================================
# Relationship Property Models
# ============================================================================

class HasStepProperties(BaseModel):
    """Properties for HAS_STEP relationship."""
    position: Optional[int] = Field(None, ge=0)


class AlarmMatchesFailureProperties(BaseModel):
    """Properties for ALARM_MATCHES_FAILURE relationship."""
    plausibility: int = Field(..., ge=0, le=3, description="0=none,1=low,2=medium,3=high")
    rationale: Optional[str] = Field(None, max_length=500)
    source: Optional[str] = Field(None, description="Model, rule, or document reference")


class ImpactsScheduleProperties(BaseModel):
    """Properties for IMPACTS_SCHEDULE relationship."""
    predicted_delay_min: Optional[int] = Field(None, ge=0)
    impact_score: int = Field(..., ge=0, le=3, description="0=none,1=low,2=medium,3=high")


class KnowledgeAppliesToProperties(BaseModel):
    """Properties for KNOWLEDGE_APPLIES_TO relationship."""
    relevance: int = Field(..., ge=0, le=3, description="0=not relevant, 1=low, 2=medium, 3=high")
    scope_note: Optional[str] = Field(None, max_length=300)


class KnowledgeReferencesProperties(BaseModel):
    """Properties for KNOWLEDGE_REFERENCES relationship."""
    confidence: int = Field(..., ge=0, le=3, description="0=anecdotal,3=strong evidential support")
    rationale: Optional[str] = Field(None, max_length=500)


# ============================================================================
# Union Type for All Node Types
# ============================================================================

ManufacturingNode = (
    Facility | ProductionLine | Machine | Sensor | ManualDocument |
    Procedure | FailureMode | AlarmEvent | WorkOrder | ProductionOrder |
    ProcessStep | ScheduleSlot | Operator | KnowledgeNote | AgentSession
)
