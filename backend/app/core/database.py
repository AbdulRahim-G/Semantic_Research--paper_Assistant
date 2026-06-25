from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import Session
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Determine if using SQLite to apply special thread-safety flags
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

try:
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args=connect_args,
        pool_pre_ping=True
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.error(f"Failed to connect to database at {settings.DATABASE_URL}: {e}")
    raise e

Base = declarative_base()

def get_db() -> Session:
    """
    FastAPI dependency yielding a database session context.
    Ensures rollback on exceptions and finalizes session closing.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
