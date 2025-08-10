from pydantic import BaseModel, Field, constr
from typing import List, Literal, Optional, Dict, Any

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

class TicketRequest(BaseModel):
    ticket_text: constr(strip_whitespace=True, min_length=5)
    top_k: int = 8

class TicketResponse(BaseModel):
    answer: constr(strip_whitespace=True, min_length=5)
    references: List[constr(strip_whitespace=True, min_length=3)]
    action_required: Literal[
        "no_escalation_needed",
        "request_more_information",
        "escalate_to_abuse_team",
        "escalate_to_billing_team",
        "escalate_to_support_level_2",
    ]
