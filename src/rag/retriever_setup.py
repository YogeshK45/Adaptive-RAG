"""
Retriever setup and vector store configuration.
"""

import os

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.core.config import settings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Global variable to store the Qdrant vectorstore instance
_qdrant_vectorstore = None


def _get_qdrant_client() -> QdrantClient:
    """Helper to initialize QdrantClient with configured credentials."""
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )


def retriever_chain(chunks: list[Document]) -> bool:
    """
    Initialize and store documents in Qdrant vector database.

    Args:
        chunks: List of document chunks to store.

    Returns:
        Boolean indicating success of the operation.
    """
    global _qdrant_vectorstore

    try:
        vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name=settings.CODE_COLLECTION,
        )

        _qdrant_vectorstore = vectorstore

        print("Qdrant vector store initialized with documents")
        print(f"Vectorstore contains {len(chunks)} document chunks")
        return True
    except Exception as e:
        print(f"Error storing documents in Qdrant: {e}")
        return False


def get_retriever():
    """
    Get a retriever tool connected to the Qdrant vector store.

    Returns the retriever tool that can search documents stored by retriever_chain().
    If no documents have been uploaded yet, connects to existing collection or creates
    an initialization state.

    Returns:
        A LangChain retriever tool configured for the vector store.

    Raises:
        Exception: If vector store initialization fails.
    """
    global _qdrant_vectorstore

    try:
        if _qdrant_vectorstore is not None:
            retriever = _qdrant_vectorstore.as_retriever()
            print("Using existing in-memory Qdrant vectorstore handle")
        else:
            client = _get_qdrant_client()
            collection_exists = client.collection_exists(settings.CODE_COLLECTION)

            if collection_exists:
                vectorstore = QdrantVectorStore.from_existing_collection(
                    embedding=embeddings,
                    collection_name=settings.CODE_COLLECTION,
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                )
                _qdrant_vectorstore = vectorstore
                retriever = vectorstore.as_retriever()
                print("Connected to existing Qdrant collection")
            else:
                print("No collection found in Qdrant, creating initial collection")
                dummy_doc = Document(
                    page_content="No documents have been uploaded yet. Please upload a document first.",
                    metadata={"source": "initialization"}
                )
                vectorstore = QdrantVectorStore.from_documents(
                    documents=[dummy_doc],
                    embedding=embeddings,
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    collection_name=settings.CODE_COLLECTION,
                )
                _qdrant_vectorstore = vectorstore
                retriever = vectorstore.as_retriever()

        # Load document description
        if os.path.exists("description.txt"):
            with open("description.txt", "r", encoding="utf-8") as f:
                description = f.read().strip()
        else:
            description = "uploaded documents and guidelines"

        retriever_tool = create_retriever_tool(
            retriever,
            "retriever_customer_uploaded_documents",
            f"Use this tool **only** to answer questions about: {description}\n"
            "Don't use this tool to answer anything else."
        )

        return retriever_tool

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise Exception(e)
