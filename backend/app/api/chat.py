from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import logging

from app.core.database import get_db
from app.models.chat import ChatSession, ChatMessage
from app.schemas.chat import SessionResponse, QueryRequest, MessageResponse
from app.services.rag_pipeline import RAGPipeline
from app.services.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize RAG Pipeline with the shared Vector Store
vector_manager = VectorStoreManager()
rag_pipeline = RAGPipeline(vector_manager)

@router.post("/session", response_model=SessionResponse)
def create_session(db: Session = Depends(get_db)):
    """
    Creates a new chat session.
    """
    session = ChatSession()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.get("/session/{id}/history", response_model=SessionResponse)
def get_session_history(id: str, db: Session = Depends(get_db)):
    """
    Retrieves conversational message history for a session.
    """
    session = db.query(ChatSession).filter(ChatSession.id == id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.post("/query", response_model=MessageResponse)
def submit_query(payload: QueryRequest, db: Session = Depends(get_db)):
    """
    Submits a user query, processes it through RAG (retrieval, prompt constraints, and citation mapping),
    and records the interaction in the database.
    """
    session = db.query(ChatSession).filter(ChatSession.id == payload.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 1. Retrieve session history as serializable list
    raw_history = []
    for msg in session.messages:
        raw_history.append({
            "role": msg.role,
            "content": msg.content
        })

    # 2. Run RAG Pipeline
    logger.info(f"Processing query for session {payload.session_id}: '{payload.question}'")
    rag_result = rag_pipeline.query(
        question=payload.question,
        session_history=raw_history,
        paper_ids=payload.paper_ids
    )

    # 3. Save User Message
    user_msg = ChatMessage(
        session_id=payload.session_id,
        role="user",
        content=payload.question
    )
    db.add(user_msg)

    # 4. Save Assistant Message (with source citation metadata)
    assistant_msg = ChatMessage(
        session_id=payload.session_id,
        role="assistant",
        content=rag_result["answer"],
        citations=rag_result["citations"]
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return assistant_msg
