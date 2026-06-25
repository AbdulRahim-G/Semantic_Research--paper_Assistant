from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.services.vector_store import VectorStoreManager
from typing import List, Dict, Any, Tuple
import os
import logging

logger = logging.getLogger(__name__)

def _generate_local_rag_fallback(question: str, retrieved_docs: List[Any], citations: List[Dict[str, Any]]) -> Dict[str, Any]:
    import re
    if not retrieved_docs:
        return {
            "answer": "I cannot find this information in the provided research paper.",
            "citations": []
        }
    
    # Heuristic word matching score
    question_words = set(re.findall(r'\w+', question.lower()))
    
    scored_sentences = []
    for idx, doc in enumerate(retrieved_docs):
        text = doc.page_content
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent_clean = sent.strip()
            if len(sent_clean) > 20:
                words = re.findall(r'\w+', sent_clean.lower())
                overlap = len(question_words.intersection(words))
                if overlap > 0:
                    scored_sentences.append((overlap, idx + 1, sent_clean))
                    
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    seen = set()
    best_sentences = []
    for score, source_idx, sent in scored_sentences:
        sent_lower = sent.lower().replace(" ", "")
        if sent_lower not in seen:
            seen.add(sent_lower)
            best_sentences.append(f"{sent} [Source #{source_idx}]")
            if len(best_sentences) >= 3:
                break
                
    if best_sentences:
        answer = "Based on the retrieved segments from the paper:\n\n" + "\n\n".join([f"- {s}" for s in best_sentences])
    else:
        first_doc = retrieved_docs[0]
        snippet = first_doc.page_content[:400]
        answer = f"Here is the most relevant segment found in the paper [Source #1]:\n\n> {snippet}..."
        
    return {
        "answer": answer,
        "citations": citations
    }

class RAGPipeline:
    def __init__(self, vector_manager: VectorStoreManager):
        self.vector_manager = vector_manager
        self.llm = self._init_llm()

    def _init_llm(self):
        """
        Initializes the LLM provider based on environment keys.
        Prioritizes Google Gemini, then OpenAI, and raises error if neither is configured.
        """
        if settings.GOOGLE_API_KEY:
            logger.info("Initializing ChatGoogleGenerativeAI (gemini-1.5-flash)...")
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.0,
                google_api_key=settings.GOOGLE_API_KEY
            )
        elif settings.OPENAI_API_KEY:
            logger.info("Initializing ChatOpenAI (gpt-4o-mini)...")
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                api_key=settings.OPENAI_API_KEY
            )
        else:
            logger.warning("No LLM API keys provided! Defaulting to MockLLM. Please configure Google or OpenAI keys.")
            # Simple mock LLM for testing without keys
            from langchain_core.runnables import Runnable
            class MockLLM(Runnable):
                def invoke(self, input, config=None, **kwargs):
                    from langchain_core.messages import AIMessage
                    return AIMessage(content="[MOCK RESPONSE] Please configure GOOGLE_API_KEY or OPENAI_API_KEY in your env.")
            return MockLLM()

    def query(self, question: str, session_history: List[Dict[str, Any]], paper_ids: List[str] = None) -> Dict[str, Any]:
        """
        Executes a RAG query: retrieves relevant chunks, compiles prompts, calls the LLM, 
        and extracts source references.
        """
        # Retrieve chunks (k=5)
        retrieved_docs = self.vector_manager.search(question, k=5, paper_ids=paper_ids)
        
        # Format the context block
        context_blocks = []
        citations = []
        for idx, doc in enumerate(retrieved_docs):
            meta = doc.metadata
            context_blocks.append(
                f"[Source #{idx+1}]\n"
                f"Paper: {meta.get('title', 'Unknown')}\n"
                f"Authors: {meta.get('authors', 'Unknown')}\n"
                f"Page: {meta.get('page_number', 'N/A')}\n"
                f"Text: {doc.page_content}"
            )
            citations.append({
                "source_index": idx + 1,
                "title": meta.get("title", "Unknown"),
                "authors": meta.get("authors", "Unknown"),
                "year": meta.get("year", 0),
                "page_number": meta.get("page_number", 0),
                "text_snippet": doc.page_content
            })
            
        context_str = "\n\n".join(context_blocks)

        # If API keys are missing, run local semantic RAG fallback
        if not (settings.GOOGLE_API_KEY or settings.OPENAI_API_KEY):
            logger.info("No API keys provided. Running local semantic RAG fallback.")
            return _generate_local_rag_fallback(question, retrieved_docs, citations)

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly analytical Research Assistant AI.
Answer the question based ONLY on the provided research context. 
If the context does not contain the answer, state: "I cannot find this information in the provided research paper."
Do not speculate, assume, or make up facts. 

Whenever you state a fact from a specific source, append the source index at the end of the sentence, for example: "The paper uses a Transformer architecture [Source #1]." or "BGE is an embedding model [Source #2]."

---
RESEARCH CONTEXT:
{context}
"""),
            # Placeholder for chat history
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        # Format historical messages for LangChain
        from langchain_core.messages import HumanMessage, AIMessage
        history_messages = []
        # Keep only the last 6 messages to avoid context overflow
        for msg in session_history[-6:]:
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            else:
                history_messages.append(AIMessage(content=msg["content"]))

        # Build chain and run
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response_text = chain.invoke({
                "context": context_str,
                "history": history_messages,
                "question": question
            })
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            response_text = f"An error occurred while calling the language model: {str(e)}"
            
        return {
            "answer": response_text,
            "citations": citations
        }
