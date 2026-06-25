from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from app.core.config import settings
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ExtractedEntity(BaseModel):
    name: str = Field(description="Name of the scientific entity (e.g., BERT, ResNet, Adam Optimizer, ImageNet, BLEU). Keep it short (1-3 words).")
    category: str = Field(description="One of: 'Model', 'Dataset', 'Method', 'Metric', 'Concept'")
    description: str = Field(description="A brief description of how this entity is used in the paper.")

class ExtractedRelation(BaseModel):
    source: str = Field(description="Name of the source entity (must match a name in the entities list).")
    target: str = Field(description="Name of the target entity (must match a name in the entities list).")
    relation_type: str = Field(description="Type of connection. Choose from: 'uses', 'outperforms', 'trains_on', 'implements', 'evaluates_on', 'extends'")

class ExtractedGraphSchema(BaseModel):
    entities: List[ExtractedEntity] = Field(description="List of extracted entities.")
    relations: List[ExtractedRelation] = Field(description="List of extracted relations between the entities.")

class GraphExtractor:
    def __init__(self):
        self.llm = self._init_llm()
        self.parser = JsonOutputParser(pydantic_object=ExtractedGraphSchema)

    def _init_llm(self):
        if settings.GOOGLE_API_KEY:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.0,
                google_api_key=settings.GOOGLE_API_KEY
            )
        elif settings.OPENAI_API_KEY:
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                api_key=settings.OPENAI_API_KEY
            )
        else:
            from langchain_core.runnables import Runnable
            class MockLLM(Runnable):
                def invoke(self, input, config=None, **kwargs):
                    from langchain_core.messages import AIMessage
                    import json
                    mock_data = {
                        "entities": [
                            {"name": "Transformer", "category": "Model", "description": "Core attention architecture"},
                            {"name": "WMT 2014", "category": "Dataset", "description": "Translation dataset"}
                        ],
                        "relations": [
                            {"source": "Transformer", "target": "WMT 2014", "relation_type": "evaluates_on"}
                        ]
                    }
                    return AIMessage(content=json.dumps(mock_data))
            return MockLLM()

    def extract_graph(self, text_sample: str) -> Dict[str, Any]:
        """
        Parses text and extracts entities and connections.
        """
        prompt = ChatPromptTemplate.from_template("""
Analyze the following text from a research paper. Identify and extract key scientific entities and the relations between them.
Return a single JSON object matching the requested schema. Return raw JSON (do not place inside markdown blocks like ```json).

Schema details:
{format_instructions}

---
PAPER TEXT:
{text_content}

---
JSON OUTPUT:
""")
        chain = prompt | self.llm | self.parser
        
        try:
            # We pass the intro/abstract parts as they contain the dense connections
            result = chain.invoke({
                "text_content": text_sample[:8000],  # Clip size to avoid model context bounds
                "format_instructions": self.parser.get_format_instructions()
            })
            return result
        except Exception as e:
            logger.error(f"Error extracting graph details: {e}")
            return {"entities": [], "relations": []}
