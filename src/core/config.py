"""
Core configuration and environment settings.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "").lstrip("y") if os.getenv("TAVILY_API_KEY", "").startswith("ytvly") else os.getenv("TAVILY_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "").lstrip("y") if os.getenv("QDRANT_URL", "").startswith("yhttp") else os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    CODE_COLLECTION: str = os.getenv("QDRANT_CODE_COLLECTION", "codebase")
    DOCS_COLLECTION: str = os.getenv("QDRANT_DOCS_COLLECTION", "guidelines")
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "adaptive_rag")


settings = Settings()

# Set env variables for LangChain integrations
os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
