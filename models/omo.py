"""
Open Memory Object (OMO) Standard Implementation.

This module implements the OMO v1 schema from the open-memory-object repository.
See: https://github.com/papr-ai/open-memory-object/blob/main/schema/omo-v1.schema.json

Architecture (Option 1 + 3: API Format Option):
- Existing Papr storage (Parse/Qdrant/Neo4j) remains UNCHANGED
- OMO format is available as an API response format via ?format=omo
- to_omo() / from_omo() conversion methods for portability
- Export/import endpoints for bulk .omo.json operations

This allows:
- Zero change for existing Papr users
- OMO-aware users can request portable format
- Platform migration via .omo.json export/import
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum
import json

if TYPE_CHECKING:
    from models.shared_types import MemoryMetadata, MemoryPolicy


# =============================================================================
# OMO v1 Core Types (from open-memory-object/schema/omo-v1.schema.json)
# =============================================================================

class OMOType(str, Enum):
    """Primary media type of the memory (OMO v1 standard)."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    CODE = "code"


class OMOConsent(str, Enum):
    """
    How the data owner allowed this memory to be stored/used.
    OMO v1 standard - REQUIRED field.
    """
    EXPLICIT = "explicit"   # User explicitly agreed
    IMPLICIT = "implicit"   # Inferred from context
    TERMS = "terms"         # Covered by Terms of Service
    NONE = "none"           # No consent recorded


class OMORisk(str, Enum):
    """
    Post-ingest safety assessment.
    OMO v1 standard - optional, defaults to 'none'.
    """
    NONE = "none"           # Safe content
    SENSITIVE = "sensitive" # Contains PII, financial, health info
    FLAGGED = "flagged"     # Requires review before retrieval


class OMOACL(BaseModel):
    """Access Control List (OMO v1 standard)."""
    read: List[str] = Field(default_factory=list, description="User IDs with read access")
    write: List[str] = Field(default_factory=list, description="User IDs with write access")

    model_config = ConfigDict(extra='forbid')


class OpenMemoryObject(BaseModel):
    """
    Open Memory Object (OMO) v1 Schema.

    This is the core portable format defined by the OMO standard.
    All platform-specific fields go in the 'ext' namespace.

    Schema: https://github.com/papr-ai/open-memory-object/blob/main/schema/omo-v1.schema.json
    """

    # Required fields
    id: str = Field(..., description="Global URI or UUID")
    createdAt: datetime = Field(..., description="ISO datetime when created")
    type: OMOType = Field(..., description="Primary media type of the memory")
    content: str = Field(..., description="UTF-8 (or base64) body")
    consent: OMOConsent = Field(..., description="How data owner allowed storage/use")

    # Optional fields with defaults
    risk: OMORisk = Field(default=OMORisk.NONE, description="Post-ingest safety assessment")
    topics: List[str] = Field(default_factory=list, description="Topic tags")
    sourceUrl: Optional[str] = Field(default=None, description="Original source URL")
    acl: Optional[OMOACL] = Field(default=None, description="Access control list")

    # Extension namespace for vendor-specific fields
    ext: Dict[str, Any] = Field(
        default_factory=dict,
        description="Namespaced extension fields (e.g., 'papr:memory_policy')"
    )

    model_config = ConfigDict(extra='forbid')

    def model_dump_json_omo(self, **kwargs) -> str:
        """Serialize to OMO-compliant JSON with proper datetime handling."""
        data = self.model_dump(mode='json', **kwargs)
        # Ensure createdAt is ISO format string
        if isinstance(data.get('createdAt'), str):
            pass  # Already string
        return json.dumps(data, indent=2, default=str)


# =============================================================================
# Conversion Functions (Papr Internal ↔ OMO Standard)
# =============================================================================

def to_omo(
    memory_id: str,
    content: str,
    memory_type: str = "text",
    consent: str = "implicit",
    risk: str = "none",
    created_at: Optional[datetime] = None,
    topics: Optional[List[str]] = None,
    source_url: Optional[str] = None,
    acl: Optional[Dict[str, List[str]]] = None,
    memory_policy: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> OpenMemoryObject:
    """
    Convert Papr internal format to OMO standard format.

    Papr-specific fields are stored in ext.papr:* namespace.

    Args:
        memory_id: Unique memory identifier
        content: Memory content
        memory_type: Type (text, image, audio, video, file, code)
        consent: OMO consent level
        risk: OMO risk level
        created_at: Creation timestamp
        topics: Topic tags
        source_url: Source URL
        acl: Access control list {read: [], write: []}
        memory_policy: Papr memory policy (graph generation, etc.)
        metadata: Papr-specific metadata fields

    Returns:
        OpenMemoryObject in OMO v1 format
    """
    # Build extension namespace for Papr-specific fields
    ext = {}

    if memory_policy:
        ext["papr:memory_policy"] = memory_policy

    if metadata:
        ext["papr:metadata"] = metadata

    # Convert ACL format
    acl = None
    if acl:
        acl = OMOACL(
            read=acl.get("read", []),
            write=acl.get("write", [])
        )

    # Map memory type to OMO type
    omo_type = OMOType.TEXT
    if memory_type:
        type_lower = memory_type.lower()
        if type_lower in [t.value for t in OMOType]:
            omo_type = OMOType(type_lower)

    # Map consent string to enum
    omo_consent = OMOConsent.IMPLICIT
    if consent:
        consent_lower = consent.lower()
        if consent_lower in [c.value for c in OMOConsent]:
            omo_consent = OMOConsent(consent_lower)

    # Map risk string to enum
    omo_risk = OMORisk.NONE
    if risk:
        risk_lower = risk.lower()
        if risk_lower in [r.value for r in OMORisk]:
            omo_risk = OMORisk(risk_lower)

    return OpenMemoryObject(
        id=memory_id,
        createdAt=created_at or datetime.now(timezone.utc),
        type=omo_type,
        content=content,
        consent=omo_consent,
        risk=omo_risk,
        topics=topics or [],
        sourceUrl=source_url,
        acl=acl,
        ext=ext
    )


def from_omo(omo: OpenMemoryObject) -> Dict[str, Any]:
    """
    Convert OMO standard format to Papr internal format.

    Extracts Papr-specific fields from ext.papr:* namespace.

    Args:
        omo: OpenMemoryObject in OMO v1 format

    Returns:
        Dictionary with Papr internal fields ready for AddMemoryRequest
    """
    result = {
        "content": omo.content,
        "type": omo.type.value,
    }

    # Build metadata from OMO fields
    metadata = {
        "createdAt": omo.createdAt.isoformat() if omo.createdAt else None,
        "consent": omo.consent.value,
        "risk": omo.risk.value,
        "topics": omo.topics if omo.topics else None,
        "sourceUrl": omo.sourceUrl,
    }

    # Extract ACL
    if omo.acl:
        metadata["acl"] = {
            "read": omo.acl.read,
            "write": omo.acl.write
        }

    # Extract Papr extensions
    memory_policy = None
    if "papr:memory_policy" in omo.ext:
        memory_policy = omo.ext["papr:memory_policy"]
        result["memory_policy"] = memory_policy

    if "papr:metadata" in omo.ext:
        papr_meta = omo.ext["papr:metadata"]
        # Merge Papr-specific metadata
        metadata.update({
            "external_user_id": papr_meta.get("external_user_id"),
            "user_id": papr_meta.get("user_id"),
            "workspace_id": papr_meta.get("workspace_id"),
            "organization_id": papr_meta.get("organization_id"),
            "namespace_id": papr_meta.get("namespace_id"),
            "role": papr_meta.get("role"),
            "category": papr_meta.get("category"),
            "conversationId": papr_meta.get("conversationId"),
            "sessionId": papr_meta.get("sessionId"),
        })
        # Copy ACL fields
        for acl_field in [
            "external_user_read_access", "external_user_write_access",
            "user_read_access", "user_write_access",
            "workspace_read_access", "workspace_write_access",
        ]:
            if papr_meta.get(acl_field):
                metadata[acl_field] = papr_meta[acl_field]

    # Clean None values from metadata
    metadata = {k: v for k, v in metadata.items() if v is not None}
    result["metadata"] = metadata

    return result


def memory_to_omo(
    memory_id: str,
    content: str,
    memory_type: str = "text",
    metadata: Optional["MemoryMetadata"] = None,
    memory_policy: Optional["MemoryPolicy"] = None,
) -> OpenMemoryObject:
    """
    Convert a Papr memory (with MemoryMetadata) to OMO format.

    This is the main conversion function used by the API.

    Args:
        memory_id: Unique memory identifier
        content: Memory content
        memory_type: Type (text, image, audio, video, file, code)
        metadata: MemoryMetadata instance
        memory_policy: MemoryPolicy instance

    Returns:
        OpenMemoryObject in OMO v1 format
    """
    # Extract values from metadata if provided
    consent = "implicit"
    risk = "none"
    topics = None
    source_url = None
    acl = None
    created_at = None
    papr_metadata = {}

    if metadata:
        # OMO core fields
        consent = getattr(metadata, 'consent', None) or "implicit"
        risk = getattr(metadata, 'risk', None) or "none"
        topics = getattr(metadata, 'topics', None)
        source_url = getattr(metadata, 'sourceUrl', None)

        # Parse createdAt
        created_at_str = getattr(metadata, 'createdAt', None)
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                created_at = None

        # OMO ACL from metadata
        acl = getattr(metadata, 'acl', None)
        if acl:
            acl = acl

        # Papr-specific metadata (goes into ext.papr:metadata)
        papr_metadata = {
            "external_user_id": getattr(metadata, 'external_user_id', None),
            "user_id": getattr(metadata, 'user_id', None),
            "workspace_id": getattr(metadata, 'workspace_id', None),
            "organization_id": getattr(metadata, 'organization_id', None),
            "namespace_id": getattr(metadata, 'namespace_id', None),
            "role": getattr(metadata, 'role', None),
            "category": getattr(metadata, 'category', None),
            "conversationId": getattr(metadata, 'conversationId', None),
            "sessionId": getattr(metadata, 'sessionId', None),
            "location": getattr(metadata, 'location', None),
            "emoji_tags": getattr(metadata, 'emoji_tags', None),
            "emotion_tags": getattr(metadata, 'emotion_tags', None),
            # ACL fields
            "external_user_read_access": getattr(metadata, 'external_user_read_access', None),
            "external_user_write_access": getattr(metadata, 'external_user_write_access', None),
            "user_read_access": getattr(metadata, 'user_read_access', None),
            "user_write_access": getattr(metadata, 'user_write_access', None),
            "workspace_read_access": getattr(metadata, 'workspace_read_access', None),
            "workspace_write_access": getattr(metadata, 'workspace_write_access', None),
        }
        # Clean None values
        papr_metadata = {k: v for k, v in papr_metadata.items() if v is not None}

    # Convert memory_policy to dict for ext namespace
    policy_dict = None
    if memory_policy:
        if hasattr(memory_policy, 'model_dump'):
            policy_dict = memory_policy.model_dump(exclude_none=True)
        elif isinstance(memory_policy, dict):
            policy_dict = memory_policy

    return to_omo(
        memory_id=memory_id,
        content=content,
        memory_type=memory_type,
        consent=consent,
        risk=risk,
        created_at=created_at,
        topics=topics,
        source_url=source_url,
        acl=acl,
        memory_policy=policy_dict,
        metadata=papr_metadata if papr_metadata else None,
    )


# =============================================================================
# Export/Import Functions (.omo.json format)
# =============================================================================

def export_omo_json(memories: List[OpenMemoryObject]) -> str:
    """
    Export memories to OMO JSON format (.omo.json).

    Per OMO spec, .omo.json files contain an array of OMO objects.

    Args:
        memories: List of OpenMemoryObject instances

    Returns:
        JSON string in OMO export format
    """
    return json.dumps(
        [m.model_dump(mode='json') for m in memories],
        indent=2,
        default=str
    )


def import_omo_json(json_str: str) -> List[OpenMemoryObject]:
    """
    Import memories from OMO JSON format (.omo.json).

    Args:
        json_str: JSON string in OMO export format

    Returns:
        List of OpenMemoryObject instances
    """
    data = json.loads(json_str)
    if not isinstance(data, list):
        data = [data]
    return [OpenMemoryObject(**item) for item in data]


# =============================================================================
# Response Format Helper
# =============================================================================

class OMOResponseFormat(str, Enum):
    """Response format options for API endpoints."""
    DEFAULT = "default"  # Normal Papr format
    OMO = "omo"          # OMO standard format


def should_return_omo_format(format_param: Optional[str]) -> bool:
    """Check if response should be in OMO format based on query parameter."""
    return format_param and format_param.lower() == "omo"
