from typing import Optional
from pydantic import BaseModel

class Prompt(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None

    @property
    def is_dual(self) -> bool:
        return self.system_prompt is not None and self.user_prompt is not None

    @property
    def single_prompt(self) -> Optional[str]:
        if self.user_prompt:
            return self.user_prompt
        return self.system_prompt