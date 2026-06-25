from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import engine, Base
from app.api import papers, chat, graph
import logging

# Configure system logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize database schemas
try:
    logger.info("Initializing relational database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing database tables: {e}", exc_info=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Policy configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to streamlit host in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach API routers
app.include_router(papers.router, prefix=f"{settings.API_V1_STR}/papers", tags=["papers"])
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])
app.include_router(graph.router, prefix=f"{settings.API_V1_STR}/graph", tags=["graph"])

@app.get("/health", tags=["health"])
def health_check():
    """
    Service health check endpoint.
    """
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "database": "connected" if settings.DATABASE_URL else "not configured"
    }
