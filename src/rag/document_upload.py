"""
Document upload and processing module.
"""

import os
import tempfile

from fastapi import UploadFile, File
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.retriever_setup import retriever_chain
from src.tools.common_tools import enhance_description_with_llm


def documents(description: str, file: UploadFile = File(...)):
    """
    Process and upload a document for RAG.

    Validates file type, loads content, enhances description, chunks documents,
    and stores them in the vector database.

    Args:
        description: User-provided document description.
        file: The uploaded file (PDF or TXT).

    Returns:
        Boolean indicating success of the upload process.

    Raises:
        HTTPException: If file type is not supported or loading fails.
    """
    filename = file.filename
    print(filename)
    if not filename.endswith(".pdf") and not filename.endswith(".txt"):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported"
        )

    file_bytes = file.file.read()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(filename)[1]
    ) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")

    try:
        docs = loader.load()
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Error loading file: {e}"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Validate that extracted text exists
    total_text = "".join([d.page_content.strip() for d in docs])
    if not total_text:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Document contains no extractable text."
        )

    # Attach document metadata to every document page
    for doc in docs:
        if not hasattr(doc, "metadata") or doc.metadata is None:
            doc.metadata = {}
        doc.metadata["filename"] = filename
        doc.metadata["source"] = filename

    # Enhance description using LLM
    description_llm = enhance_description_with_llm(description)

    # Save enhanced description
    with open("description.txt", "w", encoding="utf-8") as f:
        f.write(description_llm)

    with open("description.txt", "r", encoding="utf-8") as f:
        print("Document description from storage:")
        print(f.read())

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    # Invalidate cached vectorstore handle so retriever picks up new data
    import src.rag.retriever_setup as rs
    rs._qdrant_vectorstore = None

    return retriever_chain(chunks)




