from pydantic import BaseModel, Field
from typing import List, Optional

class EntityBase(BaseModel):
    name: str
    category: str = Field(..., description="Entity type: e.g. Model, Dataset, Method, Metric")
    description: Optional[str] = None

class RelationBase(BaseModel):
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation_type: str = Field(..., description="Relation type: e.g. uses, outperforms, trains_on")

class NodeSchema(BaseModel):
    id: str
    label: str
    title: Optional[str] = None
    group: Optional[str] = None  # Group determines visual coloring

class EdgeSchema(BaseModel):
    source: str
    target: str
    label: str

class GraphResponse(BaseModel):
    nodes: List[NodeSchema]
    edges: List[EdgeSchema]
