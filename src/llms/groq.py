"""
Groq LLM initialization and configuration.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.core.config import settings

load_dotenv()

primary_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=settings.GROQ_API_KEY,
    max_retries=0
)

fallback_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=settings.GROQ_API_KEY,
    max_retries=2
)

llm = primary_llm.with_fallbacks([fallback_llm])
