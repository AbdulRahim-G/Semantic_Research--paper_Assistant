from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import os
import uuid
import shutil
import logging

from app.core.database import get_db
from app.core.config import settings
from app.models.paper import Paper
from app.models.graph import KGEntity, KGRelation
from app.schemas.paper import PaperResponse, PaperCompareRequest
from app.services.pdf_processor import PDFProcessor
from app.services.text_splitter import DocumentChunker
from app.services.vector_store import VectorStoreManager
from app.services.summarizer import SummarizationEngine
from app.services.citation_generator import CitationGenerator
from app.services.graph_extractor import GraphExtractor

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize managers
vector_manager = VectorStoreManager()
chunker = DocumentChunker()
summarizer = SummarizationEngine()
graph_extractor = GraphExtractor()

def process_uploaded_pdf_task(paper_id: str, file_path: str, filename: str, db_session_factory):
    """
    Background worker task to extract text, chunk it, embed chunks, and run KG entity extraction.
    """
    db = db_session_factory()
    try:
        logger.info(f"Background processing started for paper ID {paper_id}")
        
        # Read file bytes
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # 1. Parse text and metadata
        parsed_data = PDFProcessor.process_pdf(file_bytes, filename)
        meta = parsed_data["metadata"]
        
        # 2. Chunk text
        chunks = chunker.split_pages(parsed_data["pages"], paper_id, meta)
        
        # 3. Add to Vector DB
        vector_manager.add_chunks(chunks)

        # 4. Extract and save Knowledge Graph entities
        # Use first 3 pages of text for KG extraction to keep it concise and relevant
        intro_text = "\n\n".join([p["text"] for p in parsed_data["pages"][:3]])
        extracted_kg = graph_extractor.extract_graph(intro_text)
        
        # Save entities
        entity_map = {}
        for ent in extracted_kg.get("entities", []):
            db_ent = KGEntity(
                paper_id=paper_id,
                name=ent["name"],
                category=ent["category"],
                description=ent["description"]
            )
            db.add(db_ent)
            db.flush() # Populate generated IDs
            entity_map[ent["name"].lower()] = db_ent.id
            
        # Save relations
        for rel in extracted_kg.get("relations", []):
            src_name = rel["source"].lower()
            tgt_name = rel["target"].lower()
            if src_name in entity_map and tgt_name in entity_map:
                db_rel = KGRelation(
                    paper_id=paper_id,
                    source_entity_id=entity_map[src_name],
                    target_entity_id=entity_map[tgt_name],
                    relation_type=rel["relation_type"]
                )
                db.add(db_rel)

        # 5. Update Paper database record
        paper = db.query(Paper).filter(Paper.id == paper_id).first()
        if paper:
            paper.title = meta["title"]
            paper.authors = meta["authors"]
            paper.publication_year = meta["year"]
            paper.journal = meta["journal"]
            paper.references = parsed_data["references"]
            paper.status = "completed"
            db.commit()
            
        logger.info(f"Background processing successfully finished for paper ID {paper_id}")
        
    except Exception as e:
        logger.error(f"Failed background processing for paper {paper_id}: {e}", exc_info=True)
        db.rollback()
        paper = db.query(Paper).filter(Paper.id == paper_id).first()
        if paper:
            paper.status = "failed"
            db.commit()
    finally:
        db.close()

@router.post("/upload", response_model=PaperResponse)
def upload_paper(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """
    Uploads a single PDF paper and launches the background processing pipeline.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Unique file storage path
    paper_id = str(uuid.uuid4())
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, f"{paper_id}.pdf")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Initial database entry
    db_paper = Paper(
        id=paper_id,
        title=file.filename.replace(".pdf", ""),
        file_path=file_path,
        status="processing"
    )
    db.add(db_paper)
    db.commit()
    db.refresh(db_paper)

    # Create thread-safe session factory reference for background worker
    from app.core.database import SessionLocal
    background_tasks.add_task(
        process_uploaded_pdf_task, 
        paper_id, 
        file_path, 
        file.filename, 
        SessionLocal
    )

    return db_paper

@router.get("/", response_model=List[PaperResponse])
def list_papers(db: Session = Depends(get_db)):
    """
    Lists all uploaded papers in the system.
    """
    return db.query(Paper).all()

@router.get("/{id}", response_model=PaperResponse)
def get_paper(id: str, db: Session = Depends(get_db)):
    """
    Gets details of a single paper.
    """
    paper = db.query(Paper).filter(Paper.id == id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

@router.delete("/{id}")
def delete_paper(id: str, db: Session = Depends(get_db)):
    """
    Deletes a paper from the storage directory, Postgres metadata, and ChromaDB vectors.
    """
    paper = db.query(Paper).filter(Paper.id == id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    # 1. Delete actual PDF file
    if os.path.exists(paper.file_path):
        os.remove(paper.file_path)

    # 2. Delete vectors from ChromaDB
    try:
        vector_manager.delete_paper_vectors(id)
    except Exception as e:
        logger.error(f"Error removing vectors for paper {id}: {e}")

    # 3. Relational delete cascade handles KG entities, relations, and paper
    db.delete(paper)
    db.commit()
    
    return {"message": "Paper and all related records successfully deleted."}

@router.post("/{id}/summarize")
def summarize_paper_endpoint(id: str, db: Session = Depends(get_db)):
    """
    Generates a structured summary for the research paper.
    """
    paper = db.query(Paper).filter(Paper.id == id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
        
    if paper.status != "completed":
        raise HTTPException(status_code=400, detail=f"Paper is currently in status: {paper.status}")

    # Re-extract pages using PDFProcessor to fetch the text segments
    try:
        with open(paper.file_path, "rb") as f:
            file_bytes = f.read()
        parsed_data = PDFProcessor.process_pdf(file_bytes, os.path.basename(paper.file_path))
        summary = summarizer.summarize_paper(parsed_data["pages"])
        return summary
    except Exception as e:
        logger.error(f"Failed to generate summary for paper {id}: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@router.get("/{id}/citation")
def get_paper_citations(id: str, db: Session = Depends(get_db)):
    """
    Retrieves generated citations in APA, MLA, and IEEE formats.
    """
    paper = db.query(Paper).filter(Paper.id == id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    citations = CitationGenerator.generate_citations(
        title=paper.title,
        authors_list=paper.authors,
        year=paper.publication_year,
        journal=paper.journal
    )
    return citations
