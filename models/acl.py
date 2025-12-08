from pydantic import BaseModel
from typing import Optional

class ACLCondition(BaseModel):
    user_id: dict[str, str] | None = None
    user_read_access: dict[str, list[str]] | None = None
    workspace_read_access: dict[str, list[str]] | None = None
    role_read_access: dict[str, list[str]] | None = None
    organization_id: dict[str, str] | None = None
    organization_read_access: dict[str, list[str]] | None = None
    namespace_id: dict[str, str] | None = None
    namespace_read_access: dict[str, list[str]] | None = None

class ACLFilter(BaseModel):
    or_: list[ACLCondition] = []

