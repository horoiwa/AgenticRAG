from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from enum import Enum


class Source(BaseModel):
    filepath: str
    filename: str
    content: str
    full_text: str
    chunk_id: int


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
