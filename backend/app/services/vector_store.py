from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from app.core.config import settings
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        """
        Initializes the HuggingFace BGE Embeddings and sets up the Chroma client.
        We default to 'BAAI/bge-small-en-v1.5' for optimal CPU execution speed and 
        high semantic retrieval accuracy (outperforms standard Sentence-Transformers).
        """
        model_name = "BAAI/bge-small-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True}  # Vital for cosine similarity
        
        logger.info(f"Loading HuggingFace Embeddings model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Can be changed to 'cuda' for GPU acceleration
            encode_kwargs=encode_kwargs
        )
        
        logger.info(f"Initializing ChromaDB client at persistent directory: {settings.VECTOR_DB_DIR}")
        self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
        
        self.db = Chroma(
            client=self.client,
            collection_name="research_papers",
            embedding_function=self.embeddings
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Inserts document chunks into ChromaDB with unique IDs and corresponding metadata.
        """
        if not chunks:
            return
            
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # Construct deterministic unique keys for chunks
        ids = [
            f"{c['metadata']['paper_id']}_c{c['metadata']['global_chunk_index']}_p{c['metadata']['page_number']}" 
            for c in chunks
        ]
        
        logger.info(f"Ingesting {len(chunks)} chunks into ChromaDB...")
        self.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info("ChromaDB ingestion completed.")

    def search(self, query: str, k: int = 5, paper_ids: Optional[List[str]] = None) -> List[Any]:
        """
        Performs semantic similarity search. Optionally filters results to specific papers.
        """
        filter_dict = None
        if paper_ids:
            if len(paper_ids) == 1:
                filter_dict = {"paper_id": paper_ids[0]}
            else:
                # Chroma metadata filter syntax for OR matches
                filter_dict = {"$or": [{"paper_id": pid} for pid in paper_ids]}

        logger.info(f"Running vector search for query: '{query}' (k={k}, filter={filter_dict})")
        docs = self.db.similarity_search(query, k=k, filter=filter_dict)
        return docs

    def delete_paper_vectors(self, paper_id: str) -> None:
        """
        Removes all vectors belonging to a specific paper.
        """
        logger.info(f"Deleting vectors for paper ID: {paper_id}")
        # ChromaDB as_retriever filter syntax can be run via client or direct delete
        # Chroma collection delete by metadata filter
        collection = self.client.get_collection("research_papers")
        collection.delete(where={"paper_id": str(paper_id)})
