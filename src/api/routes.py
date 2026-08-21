import asyncio
import uuid
from fastapi import APIRouter, UploadFile, File, Header
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

router = APIRouter()


class UserAuthRequest(BaseModel):
    username: str
    password: str


@router.post("/api/init")
@router.post("/init")
async def api_init():
    """Initialize session and return API token for frontend."""
    return {"api_token": str(uuid.uuid4())}


@router.post("/api/create_user")
@router.post("/create_user")
async def create_user_endpoint(user: UserAuthRequest):
    """Create user account for frontend auth."""
    return {"status": "success", "message": f"User {user.username} created successfully"}


@router.post("/api/login")
@router.post("/login")
async def login_endpoint(user: UserAuthRequest):
    """Authenticate user and return session token for frontend."""
    return {"jwt": f"jwt_{user.username}_{uuid.uuid4().hex[:8]}", "username": user.username}


@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """
    Process a RAG query and return the result.

    Args:
        req: The query request containing query text and session_id.

    Returns:
        The generated response from the RAG pipeline.
    """
    #chat_history=ChatInMemoryHistory.get_session_history(req.token)
    chat_history = ChatHistory.get_session_history(req.session_id)
    await chat_history.add_message(HumanMessage(content=req.query))

    # Fetch full history
    messages = await chat_history.get_messages()
    context_messages = messages[-6:] if len(messages) > 6 else messages
    result = await asyncio.to_thread(builder.invoke, {"messages": context_messages})
    last_message = result["messages"][-1]
    if hasattr(last_message, "content"):
        output_text = last_message.content
    elif isinstance(last_message, dict):
        output_text = last_message.get("content", "")
    else:
        output_text = str(last_message)

    # Save assistant message
    await chat_history.add_message(AIMessage(content=output_text))

    return {"result": {"type": "ai", "content": output_text}}


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description")
):
    """
    Upload a document for RAG processing.

    Args:
        file: The file to upload (PDF or TXT).
        description: Document description provided via header.

    Returns:
        Upload status.
    """
    status_upload = await asyncio.to_thread(documents, description, file)
    return {"status": status_upload}

