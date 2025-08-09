from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class IngestItem(BaseModel):
    id: Optional[str] = None
    doc: str
    section: str
    anchor_id: str
    text: str = Field(min_length=5)
    product: str = "domains"
    lang: str = "en"

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    product: Optional[str] = None
    lang: Optional[str] = None
