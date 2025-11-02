from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from enum import Enum


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

class SearchResponse(BaseModel):
    results: List[Source]

class DocumentMetadata(BaseModel):
    document_id: str
    document_name: str
    uploaded_at: datetime
    status: DocumentStatus
