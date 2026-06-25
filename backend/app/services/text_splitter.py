
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import uuid

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the text chunker.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlapping characters between adjacent chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_pages(self, pages: List[Dict[str, Any]], paper_id: uuid.UUID, paper_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits pages into semantic chunks and attaches reference metadata to each chunk.
        
        Args:
            pages: List of dicts, each with 'page_number' and 'text'.
            paper_id: UUID of the parent paper in the database.
            paper_metadata: Dictionary with 'title', 'authors', and 'year'.
            
        Returns:
            List of dicts representing chunks, formatted for vector db ingestion.
        """
        chunks = []
        global_chunk_idx = 0
        
        # Serialize authors list to string for metadata compatibility
        authors_raw = paper_metadata.get("authors", [])
        if isinstance(authors_raw, list):
            authors_str = ", ".join(authors_raw)
        else:
            authors_str = str(authors_raw)

        for page in pages:
            page_num = page["page_number"]
            page_text = page["text"]
            
            if not page_text.strip():
                continue
                
            # Perform text split on current page content
            split_texts = self.splitter.split_text(page_text)
            
            for page_chunk_idx, text in enumerate(split_texts):
                chunks.append({
                    "text": text,
                    "metadata": {
                        "paper_id": str(paper_id),
                        "title": paper_metadata.get("title", "Unknown"),
                        "authors": authors_str,
                        "year": paper_metadata.get("year") or 0,
                        "page_number": page_num,
                        "page_chunk_index": page_chunk_idx,
                        "global_chunk_index": global_chunk_idx
                    }
                })
                global_chunk_idx += 1
                
        return chunks
