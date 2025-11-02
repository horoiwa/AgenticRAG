from typing import List, Dict, Any, Optional, AsyncGenerator
import traceback
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import functools
from dataclasses import dataclass

from elasticsearch import AsyncElasticsearch, NotFoundError
from sentence_transformers import SentenceTransformer
from markitdown import MarkItDown
from datetime import datetime
import uuid

from src.settings import (
    ELASTIC_SEARCH_HOST,
    USE_DEVICE,
    INDEX_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CONTEXT_SIZE,
)

# ロガーの設定
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

FIELD_MAPPINGS = {
    "mappings": {
        "properties": {
            "filename": {"type": "text"},
            "content": {"type": "text"},
            "full_text": {"type": "text"},
            "content_vector": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIM,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
}


class ElasticsearchClient:
    def __init__(self, host: str):
        self.client = AsyncElasticsearch(
            hosts=[host],
            verify_certs=False,  # SSL証明書を検証しない
            ssl_show_warn=False,  # SSL/TLSが無効であることの警告を非表示にする
        )
        self.host = host
        logger.info(f"Elasticsearch client initialized for host: {host}")

    async def close(self):
        await self.client.close()

    async def ping(self) -> bool:
        """Elasticsearchサーバーへの接続を確認します。"""
        try:
            res: bool = await self.client.ping()
            return res
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            return False

    async def create_index(
        self, index_name: str, mappings: Dict[str, Any] = FIELD_MAPPINGS
    ) -> bool:
        """
        指定されたインデックス名とマッピングでインデックスを作成します。
        既に存在する場合は何もしません。
        """
        if not await self.client.indices.exists(index=index_name):
            try:
                await self.client.indices.create(index=index_name, mappings=mappings)
                logger.info(f"Index '{index_name}' created successfully.")
                return True
            except Exception as e:
                logger.error(f"Error creating index '{index_name}': {e}")
                return False
        else:
            logger.info(f"Index '{index_name}' already exists.")
            return True

    async def delete_index(self, index_name: str) -> bool:
        """指定されたインデックスを削除します。"""
        try:
            await self.client.indices.delete(index=index_name)
            logger.info(f"Index '{index_name}' deleted successfully.")
            return True
        except NotFoundError:
            logger.warning(f"Index '{index_name}' not found, skipping deletion.")
            return True  # 存在しない場合は削除成功とみなす
        except Exception as e:
            logger.error(f"Error deleting index '{index_name}': {e}")
            return False

    async def index_document(
        self, file_path: Path, index_name: str = INDEX_NAME
    ) -> list[str]:
        """
        ドキュメントを解析、チャンク化し、Elasticsearchにインデックスします。
        成功した場合はドキュメントIDのリストを返します。
        """
        try:
            logger.info(f"Starting to index document: {file_path}")
            markdown_converter = MarkItDown()
            markdown_text = markdown_converter.convert(file_path)
            if not markdown_text:
                logger.warning(f"Could not extract text from {file_path}")
                return []

            chunks = self._chunk_text(markdown_text)
            doc_ids = []
            document_id = str(uuid.uuid4())

            for i, chunk in enumerate(chunks):
                doc = {
                    "document_id": document_id,
                    "document_name": file_path.name,
                    "content": chunk,
                    "uploaded_at": datetime.now(),
                    "status": "completed",
                }
                # Elasticsearchにインデックス
                res = await self.client.index(index=index_name, document=doc)
                doc_ids.append(res["_id"])

            logger.info(
                f"Successfully indexed {len(chunks)} chunks for document: {file_path.name} with document_id: {document_id}"
            )
            return doc_ids

        except Exception as e:
            logger.error(f"Failed to index document {file_path}: {e}")
            logger.error(traceback.format_exc())
            return []

    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
        """テキストを指定されたチャンクサイズに分割します。"""
        # 簡単な実装例：指定文字数で分割
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def search(
        self, index_name: str, query: str, fields: List[str], size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        指定されたインデックスとフィールドに対して全文検索を実行します。
        """
        try:
            response = await self.client.search(
                index=index_name,
                query={"multi_match": {"query": query, "fields": fields}},
                size=size,
            )
            hits = []
            for hit in response["hits"]["hits"]:
                hits.append(
                    {"id": hit["_id"], "score": hit["_score"], "source": hit["_source"]}
                )
            logger.info(
                f"Search for '{query}' in '{index_name}' returned {len(hits)} hits."
            )
            return hits
        except Exception as e:
            logger.error(
                f"Error during search in '{index_name}' for query '{query}': {e}"
            )
            return []

    async def get_document(self, index_name: str, id: str) -> Optional[Dict[str, Any]]:
        """ドキュメントIDでドキュメントを取得します。"""
        try:
            response = await self.client.get(index=index_name, id=id)
            logger.info(f"Document '{id}' retrieved from '{index_name}'.")
            return response["_source"]
        except NotFoundError:
            logger.warning(f"Document '{id}' not found in '{index_name}'.")
            return None
        except Exception as e:
            logger.error(f"Error getting document '{id}' from '{index_name}': {e}")
            return None


@asynccontextmanager
async def get_es_client(
    host: str = ELASTIC_SEARCH_HOST,
) -> AsyncGenerator[ElasticsearchClient, None]:
    client = ElasticsearchClient(host=host)
    try:
        yield client
    except:
        logger.error(traceback.print_exc())
    finally:
        await client.close()


@functools.cache
def get_embedding_model():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=USE_DEVICE)
    return embedding_model


def embed(sentences: str | list[str]) -> list[list[float]]:
    if isinstance(sentences, str):
        sentences = [sentences]
    model = get_embedding_model()
    embeddings = model.encode(
        sentences, convert_to_tensor=False, normalize_embeddings=True
    )
    return [e.tolist() for e in embeddings]


async def _debug():
    sentence = ["hello world"]
    res = embed(sentence)
    import pdb; pdb.set_trace()  # fmt: skip


if __name__ == "__main__":
    import asyncio

    asyncio.run(_debug())
