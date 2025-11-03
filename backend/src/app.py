from fastapi import FastAPI, HTTPException
from typing import List
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from src.es_search import get_es_client
from src import settings
from src.schemas import (
    ChatRequest,
    ChatResponse,
    Source,
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
            raise HTTPException(
                status_code=500, detail="Failed to connect to Elasticsearch"
            )

        # RAG用インデックスの作成（存在しない場合）
        if not await es_client.create_index(settings.DEFAULT_INDEX_NAME):
            logger.error(
                f"Failed to create or ensure index '{settings.DEFAULT_INDEX_NAME}'. Exiting."
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create or ensure index '{settings.DEFAULT_INDEX_NAME}'",
            )
        logger.info(f"Elasticsearch index '{settings.DEFAULT_INDEX_NAME}' is ready.")
    yield
    logger.info("Shutting down application...")


# --- FastAPIアプリケーションのインスタンス化 ---

app = FastAPI(
    title="Agentic RAG API",
    description="An API for Retrieval-Augmented Generation with agentic capabilities.",
    version="0.1.0",
    lifespan=lifespan_event_handler,
)


# --- エンドポイントの定義 ---


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Receives a query, performs RAG, and returns an answer with sources.
    """
    raise NotImplementedError()


@app.get("/search", response_model=List[Source])
async def search(query: str, index_name: str = settings.DEFAULT_INDEX_NAME):
    """
    Performs a hybrid search based on the user's query and returns the results.
    """
    async with get_es_client() as es_client:
        results = await es_client.hybrid_search(query=query, index_name=index_name)
    return results


@app.get("/documents", response_model=List[Path])
async def list_documents(index_name: str = settings.DEFAULT_INDEX_NAME):
    """
    Returns a list of unique file paths of documents stored in Elasticsearch.
    """
    async with get_es_client() as es_client:
        documents = await es_client.get_document_list(index_name=index_name)
    return documents
