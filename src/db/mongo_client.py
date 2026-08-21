"""
MongoDB client initialization.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from src.core.config import settings

MONGO_URL = settings.MONGODB_URL or "mongodb://localhost:27017"
DB_NAME = settings.MONGODB_DB_NAME or "adaptive_rag"

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
