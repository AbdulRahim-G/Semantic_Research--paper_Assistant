from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class PaperBase(BaseModel):
    title: Optional[str] = None
    authors: List[str] = []
    publication_year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None

class PaperCreate(PaperBase):
    file_path: str

class PaperUpdate(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    publication_year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    status: Optional[str] = None

class PaperResponse(PaperBase):
    id: str
    status: str
    file_path: str
    uploaded_at: datetime
    references: List[str] = []

    class Config:
        from_attributes = True

class PaperCompareRequest(BaseModel):
    paper_ids: List[str] = Field(..., min_items=2, description="List of paper IDs to compare")
