import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Semantic Research Paper Assistant"
    API_V1_STR: str = "/api/v1"
    
    # API Credentials
    GOOGLE_API_KEY: str = Field(default="")
    OPENAI_API_KEY: str = Field(default="")
    
    # Database Configuration
    DATABASE_URL: str = Field(default="sqlite:///./local_db.sqlite")
    
    # Storage Configuration
    VECTOR_DB_DIR: str = Field(default="./chroma_db")
    UPLOAD_DIR: str = Field(default="./uploaded_papers")
    
    # Security Config
    JWT_SECRET: str = Field(default="super_secret_session_token_key_123!@#")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)
    
    # Allow loading configuration from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure critical directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)
