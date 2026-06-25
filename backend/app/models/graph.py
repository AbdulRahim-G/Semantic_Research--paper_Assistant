# pyrefly: ignore [missing-import]
from sqlalchemy import Column, String, ForeignKey
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid

class KGEntity(Base):
    __tablename__ = "kg_entities"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    paper_id = Column(String(36), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(256), nullable=False)
    category = Column(String(100), nullable=True)  # 'Model', 'Dataset', 'Method', 'Metric'
    description = Column(String(1024), nullable=True)

    # Relationships
    paper = relationship("Paper")

class KGRelation(Base):
    __tablename__ = "kg_relations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    paper_id = Column(String(36), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    source_entity_id = Column(String(36), ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False)
    target_entity_id = Column(String(36), ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False)
    relation_type = Column(String(100), nullable=False)  # 'uses', 'outperforms', 'evaluates_on'

    # Relationships
    paper = relationship("Paper")
    source_entity = relationship("KGEntity", foreign_keys=[source_entity_id])
    target_entity = relationship("KGEntity", foreign_keys=[target_entity_id])
