import json
from typing import Optional, List
from models.shared_types import ContextItem

def parse_context(context_value) -> Optional[List[ContextItem]]:
    if context_value in (None, '', 'null', 'undefined'):
        return []
    if isinstance(context_value, list):
        return context_value
    if isinstance(context_value, str):
        try:
            parsed = json.loads(context_value)
            if isinstance(parsed, list):
                return parsed
            else:
                return []
        except Exception:
            return []
    return []