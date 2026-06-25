from sqlalchemy import Column, String, Integer, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.core.database import Base
import uuid

class Paper(Base):
    __tablename__ = "papers"

    # UUID columns, fallback to String for SQLite compatibility during local testing
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(512), nullable=True)
    authors = Column(JSON, default=[])  # Store list of authors
    publication_year = Column(Integer, nullable=True)
    journal = Column(String(256), nullable=True)
    doi = Column(String(100), nullable=True)
    file_path = Column(String(512), nullable=False)
    status = Column(String(50), default="processing")  # 'processing', 'completed', 'failed'
    references = Column(JSON, default=[])  # Store list of bibliography references
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
