
from fastapi import FastAPI, HTTPException
from typing import List
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from src.es_search import get_es_client
from src import settings
from src.schemas import (
    ChatRequest,
    ChatResponse,
    SearchResponse,
    DocumentMetadata,
    Source,
    DocumentStatus,
)

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- アプリケーションライフサイクルイベント ---
@asynccontextmanager
async def lifespan_event_handler(app: FastAPI):
    logger.info("Starting up application...")
    async with get_es_client() as es_client:
        if not await es_client.ping():
            logger.error("Failed to connect to Elasticsearch. Exiting.")
            raise HTTPException(status_code=500, detail="Failed to connect to Elasticsearch")

        # RAG用インデックスの作成（存在しない場合）
        mappings = {
            "properties": {
                "document_id": {"type": "keyword"},
                "document_name": {"type": "keyword"},
                "content": {"type": "text"},
                "uploaded_at": {"type": "date"},
                "status": {"type": "keyword"},
            }
        }
        if not await es_client.create_index(settings.INDEX_NAME, mappings):
            logger.error(f"Failed to create or ensure index \'{settings.INDEX_NAME}\'. Exiting.")
            raise HTTPException(status_code=500, detail=f"Failed to create or ensure index \'{settings.INDEX_NAME}\'")
        logger.info(f"Elasticsearch index \'{settings.INDEX_NAME}\' is ready.")
    yield
    logger.info("Shutting down application...")

# --- FastAPIアプリケーションのインスタンス化 ---

app = FastAPI(
    title="Agentic RAG API",
    description="An API for Retrieval-Augmented Generation with agentic capabilities.",
    version="0.1.0",
    lifespan=lifespan_event_handler
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

@app.get("/search", response_model=SearchResponse)
async def search(query: str):
    """
    Performs a simple keyword search based on the user's query and returns the results.
    """
    es_client = await get_es_client()
    if not await es_client.ping():
        raise HTTPException(status_code=500, detail="Failed to connect to Elasticsearch")

    search_results = await es_client.search(
        index_name=settings.INDEX_NAME,
        query=query,
        fields=["content", "document_name"]
    )

    sources = []
    for hit in search_results:
        source_data = hit["source"]
        sources.append(Source(
            document_id=source_data.get("document_id", "unknown"),
            document_name=source_data.get("document_name", "unknown"),
            snippet=source_data.get("content", "")
        ))

    return SearchResponse(results=sources)

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
