from typing import Tuple, List
from pydantic import BaseModel

class Filters(BaseModel):
    sample_low: int = None
    sample_high: int = None
    sample_list: List[int] = None
    
class AnnotationFilters(Filters):
    validated: bool = None