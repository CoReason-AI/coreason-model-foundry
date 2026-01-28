from typing import Any, Dict, List

from pydantic import BaseModel

from .types import SecretStr


class UserContext(BaseModel):
    user_id: SecretStr
    roles: List[str]
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True
