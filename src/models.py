from pydantic import BaseModel, ConfigDict
from typing import Any

class UserSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    session_id: str
    conversation_manager: Any
