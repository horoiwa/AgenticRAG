from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum


class Source(BaseModel):
    filepath: str
    filename: str
    content: str
    full_text: str
    chunk_id: int


class ChatContext(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str
    context: List[ChatContext] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
