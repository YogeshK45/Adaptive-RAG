"""
Backward-compatible LLM import redirecting to Groq.
"""

from src.llms.groq import llm

__all__ = ["llm"]