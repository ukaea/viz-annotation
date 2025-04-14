from datetime import datetime
from typing import List
from pydantic import BaseModel

class Shot(BaseModel):
    created: str = datetime.now().isoformat()
    shot_id: int
    validated: bool = False