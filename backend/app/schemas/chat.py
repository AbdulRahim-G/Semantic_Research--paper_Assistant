from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class MessageBase(BaseModel):
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")

class MessageCreate(MessageBase):
    pass

class CitationSchema(BaseModel):
    title: str
    authors: str
    year: int
    page_number: int
    text_snippet: str

class MessageResponse(MessageBase):
    id: str
    session_id: str
    citations: List[Dict[str, Any]] = []
    created_at: datetime

    class Config:
        from_attributes = True

class SessionResponse(BaseModel):
    id: str
    created_at: datetime
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True

class QueryRequest(BaseModel):
    session_id: str
    question: str
    paper_ids: Optional[List[str]] = Field(default=None, description="Optional paper IDs to restrict search to")
