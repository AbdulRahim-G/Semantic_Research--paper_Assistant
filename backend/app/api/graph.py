# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from app.core.database import get_db
from app.models.paper import Paper
from app.models.graph import KGEntity, KGRelation
from app.schemas.graph import GraphResponse, NodeSchema, EdgeSchema
from app.schemas.paper import PaperCompareRequest
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate

# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import JsonOutputParser

# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

def _init_compare_llm():
    if settings.GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=settings.GOOGLE_API_KEY)
    elif settings.OPENAI_API_KEY:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=settings.OPENAI_API_KEY)
    else:
        # pyrefly: ignore [missing-import]
        from langchain_core.runnables import Runnable
        class MockLLM(Runnable):
            def invoke(self, input, config=None, **kwargs):
                # pyrefly: ignore [missing-import]
                from langchain_core.messages import AIMessage
                import json
                return AIMessage(content=json.dumps({
                    "methodologies": "Mock comparison of methodologies. Please configure API keys.",
                    "datasets": "Mock datasets comparison.",
                    "models": "Mock models comparison.",
                    "results": "Mock results comparison.",
                    "limitations": "Mock limitations comparison."
                }))
        return MockLLM()

@router.get("/", response_model=GraphResponse)
def get_knowledge_graph(paper_ids: Optional[List[str]] = None, db: Session = Depends(get_db)):
    """
    Retrieves nodes (entities) and edges (relations) to construct the interactive knowledge graph.
    """
    entity_query = db.query(KGEntity)
    relation_query = db.query(KGRelation)

    if paper_ids:
        entity_query = entity_query.filter(KGEntity.paper_id.in_(paper_ids))
        relation_query = relation_query.filter(KGRelation.paper_id.in_(paper_ids))

    entities = entity_query.all()
    relations = relation_query.all()

    # Map to NodeSchema format
    nodes = []
    seen_entities = set()
    for ent in entities:
        if ent.name.lower() not in seen_entities:
            seen_entities.add(ent.name.lower())
            nodes.append(NodeSchema(
                id=ent.name.lower(),
                label=ent.name,
                title=f"Category: {ent.category}\nDesc: {ent.description or ''}",
                group=ent.category
            ))

    # Map to EdgeSchema format
    edges = []
    for rel in relations:
        edges.append(EdgeSchema(
            source=rel.source_entity.name.lower(),
            target=rel.target_entity.name.lower(),
            label=rel.relation_type
        ))

    return GraphResponse(nodes=nodes, edges=edges)

def _generate_accurate_local_comparison(papers: List[Paper]) -> Dict[str, str]:
    # pyrefly: ignore [missing-import]
    import chromadb
    import re
    
    # Categories and keywords
    categories_keywords = {
        "methodologies": ["method", "proposed framework", "architecture", "system design", "implementation", "approach", "algorithm", "technique"],
        "datasets": ["dataset", "data split", "corpus", "training set", "test set", "validation", "sample size", "token"],
        "models": ["model", "parameter", "weight", "hyperparameter", "neural network", "attention", "layer", "dimension"],
        "results": ["result", "metric", "accuracy", "performance", "f1-score", "bleu", "precision", "outperform", "score"],
        "limitations": ["limitation", "weakness", "bias", "future work", "constraint", "error analysis", "challenge"]
    }
    
    # Initialize the Chroma client directly (without loading HuggingFace embeddings model)
    try:
        client = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
        collection = client.get_collection("research_papers")
    except Exception as e:
        logger.error(f"Failed to load Chroma collection for comparison: {e}")
        collection = None

    comparison = {cat: "" for cat in categories_keywords}
    
    for cat, keywords in categories_keywords.items():
        cat_texts = []
        for paper in papers:
            paper_sentences = []
            
            # Retrieve chunks for this paper from ChromaDB
            chunks = []
            if collection:
                try:
                    res = collection.get(where={"paper_id": str(paper.id)})
                    chunks = res.get("documents", []) or []
                except Exception as e:
                    logger.error(f"Error fetching chunks for paper {paper.id}: {e}")
            
            # Extract sentences from chunks
            for chunk in chunks:
                # Split chunk into sentences
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                for sent in sentences:
                    sent = sent.strip()
                    if 35 < len(sent) < 300: # Reasonable sentence length
                        # Calculate score based on unique keyword matches
                        score = sum(1 for kw in keywords if kw in sent.lower())
                        if score > 0:
                            paper_sentences.append((score, sent))
            
            # Sort sentences by score descending, remove duplicates, and take top 3
            paper_sentences.sort(key=lambda x: x[0], reverse=True)
            seen_sentences = set()
            unique_sentences = []
            for score, sent in paper_sentences:
                # Basic duplicate check
                sent_clean = sent.lower().replace(" ", "")
                if sent_clean not in seen_sentences:
                    seen_sentences.add(sent_clean)
                    unique_sentences.append(sent)
                    if len(unique_sentences) >= 3:
                        break
            
            # Fallback if no matching sentences found
            if not unique_sentences:
                # Just take the first 2 sentences from the first chunk (abstract)
                if chunks:
                    fallback_sents = re.split(r'(?<=[.!?])\s+', chunks[0])[:2]
                    unique_sentences = [s.strip() for s in fallback_sents if len(s.strip()) > 20]
                else:
                    unique_sentences = ["No details available for extraction."]
            
            # Format markdown list for this paper
            paper_markdown = f"**📄 {paper.title}** ({paper.publication_year or 'N/A'})\n"
            paper_markdown += "\n".join([f"- {s}" for s in unique_sentences])
            cat_texts.append(paper_markdown)
            
        # Join papers comparison for this category
        comparison[cat] = "\n\n---\n\n".join(cat_texts)
        
    return comparison

@router.post("/compare")
def compare_papers(payload: PaperCompareRequest, db: Session = Depends(get_db)):
    """
    Executes a semantic comparison matrix across multiple user-selected papers.
    """
    papers = db.query(Paper).filter(Paper.id.in_(payload.paper_ids)).all()
    if len(papers) < 2:
        raise HTTPException(status_code=400, detail="Must provide at least 2 valid paper IDs for comparison.")

    # Check if API keys are configured; run local semantic extractor if keys are missing
    if not (settings.GOOGLE_API_KEY or settings.OPENAI_API_KEY):
        logger.info("No API keys provided. Running local semantic comparison extractor.")
        return _generate_accurate_local_comparison(papers)

    # Aggregate summaries or metadata details of each paper
    comparison_context = []
    for paper in papers:
        comparison_context.append(
            f"Paper ID: {paper.id}\n"
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}\n"
            f"Year: {paper.publication_year or 'N/A'}\n"
            f"Journal: {paper.journal or 'N/A'}\n"
        )
    
    context_str = "\n\n===\n\n".join(comparison_context)

    # Use LLM to perform structural matrix comparison
    prompt = ChatPromptTemplate.from_template("""
You are a senior research scientist comparing multiple scientific papers.
Generate a structural comparison matrix in raw JSON format (do not wrap in markdown blocks like ```json).

Adhere strictly to this JSON format:
{{
    "methodologies": "Detailed comparison of the research methodologies, architectures, and theoretical approaches.",
    "datasets": "Detailed comparison of data collections, pipelines, sizes, and splits used.",
    "models": "Detailed comparison of the model architectures, parameters, or weights evaluated.",
    "results": "Detailed comparison of the benchmarks, metrics, and quantitative outcomes achieved.",
    "limitations": "Detailed comparison of the weaknesses, biases, or constraints of each study."
}}

---
PAPERS TO COMPARE:
{papers_context}

---
JSON OUTPUT:
""")
    
    llm = _init_compare_llm()
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    try:
        comparison_result = chain.invoke({"papers_context": context_str})
        return comparison_result
    except Exception as e:
        logger.error(f"Error comparing papers: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
