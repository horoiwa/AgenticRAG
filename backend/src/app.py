from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import tempfile

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from src.agent import root_agent
from src.es_search import get_es_client
from src import settings
from src.schemas import (
    ChatRequest,
    ChatResponse,
    Source,
)


# --- ADK Runnerの初期化 ---
session_service = InMemorySessionService()
runner = Runner(
    app_name="agentic_rag", agent=root_agent, session_service=session_service
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
    session_id = request.session_id or await session_service.create_session(
        app_name="agentic_rag", user_id="user"
    )

    final_answer = ""
    async for event in runner.run_async(
        user_id="user",
        session_id=session_id,
        new_message=genai_types.Content(
            role="user", parts=[genai_types.Part.from_text(text=request.query)]
        ),
    ):
        if event.is_final_response():
            final_answer = event.content.parts[0].text

    return ChatResponse(answer=final_answer, sources=[])


@app.get("/search", response_model=List[Source])
async def search(query: str, index_name: str = settings.DEFAULT_INDEX_NAME):
    """
    Performs a hybrid search based on the user's query and returns the results.
    """
    async with get_es_client() as es_client:
        results = await es_client.hybrid_search(query=query, index_name=index_name)
    return results


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    prefix: str = "",
    index_name: str = settings.DEFAULT_INDEX_NAME,
):
    """
    Uploads a new document, vectorizes it, and stores it in Elasticsearch.
    同時に元ファイルもローカル保管
    """
    try:
        prefix = prefix.replace("\\", "/").strip("/") if prefix else ""
        file_path = settings.DATA_DIR / prefix / file.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        async with get_es_client() as es_client:
            await es_client.index_document(file_path=file_path, index_name=index_name)

        return {"message": "Document indexed successfully"}
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing document: {e}")


@app.get("/documents", response_model=List[Path])
async def list_documents(index_name: str = settings.DEFAULT_INDEX_NAME):
    """
    Returns a list of unique file paths of documents stored in Elasticsearch.
    """
    async with get_es_client() as es_client:
        documents = await es_client.get_document_list(index_name=index_name)
    return documents


@app.delete("/documents")
async def delete_document(
    file_path: Path, index_name: str = settings.DEFAULT_INDEX_NAME
):
    """
    Deletes a document from Elasticsearch based on its file_id.
    """
    raw_file_path = settings.DATA_DIR / str(file_path).replace("\\", "/").lstrip("/")
    if raw_file_path.exists():
        raw_file_path.unlink()

    async with get_es_client() as es_client:
        try:
            await es_client.delete_document(
                file_path=raw_file_path, index_name=index_name
            )
            return {"message": "Document deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")
