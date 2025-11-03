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
        if not await es_client.create_index(settings.INDEX_NAME):
            logger.error(
                f"Failed to create or ensure index '{settings.INDEX_NAME}'. Exiting."
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create or ensure index '{settings.INDEX_NAME}'",
            )
        logger.info(f"Elasticsearch index '{settings.INDEX_NAME}' is ready.")
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
async def search(query: str):
    """
    Performs a simple keyword search based on the user's query and returns the results.
    """
    raise NotImplementedError()


@app.get("/documents", response_model=List[Path])
async def list_documents():
    raise NotImplementedError()
