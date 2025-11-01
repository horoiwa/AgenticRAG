
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from enum import Enum

# --- データモデルの定義 ---

class DocumentStatus(str, Enum):
    COMPLETED = "completed"
    PROCESSING = "processing"
    FAILED = "failed"

class Source(BaseModel):
    document_id: str
    document_name: str
    snippet: str

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

class DocumentMetadata(BaseModel):
    document_id: str
    document_name: str
    uploaded_at: datetime
    status: DocumentStatus

# --- FastAPIアプリケーションのインスタンス化 ---

app = FastAPI(
    title="Agentic RAG API",
    description="An API for Retrieval-Augmented Generation with agentic capabilities.",
    version="0.1.0",
)

# --- エンドポイントの定義 ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Receives a query, performs RAG, and returns an answer with sources.
    """
    # (仮のレスポンス)
    dummy_source = Source(
        document_id="doc_123",
        document_name="Example Document",
        snippet="This is a snippet from the example document."
    )
    dummy_answer = f"This is a dummy answer to your query: '{request.query}'"

    return ChatResponse(
        answer=dummy_answer,
        sources=[dummy_source]
    )

@app.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    """
    Lists metadata of all uploaded documents.
    """
    # (仮のレスポンス)
    dummy_doc = DocumentMetadata(
        document_id="doc_456",
        document_name="Another Example.txt",
        uploaded_at=datetime.now(),
        status=DocumentStatus.COMPLETED
    )
    return [dummy_doc]
