from typing import List, Dict, Any, Optional, AsyncGenerator
import traceback
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from elasticsearch import AsyncElasticsearch, NotFoundError
from sentence_transformers import SentenceTransformer

from src.settings import ELASTIC_SEARCH_HOST, USE_DEVICE

# ロガーの設定
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    def __init__(self, host: str):
        self.client = AsyncElasticsearch(
            hosts=[host],
            verify_certs=False,   # SSL証明書を検証しない
            ssl_show_warn=False, # SSL/TLSが無効であることの警告を非表示にする
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

    async def create_index(self, index_name: str, mappings: Dict[str, Any]) -> bool:
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
            return True # 存在しない場合は削除成功とみなす
        except Exception as e:
            logger.error(f"Error deleting index '{index_name}': {e}")
            return False

    async def index_document(self, index_name: str, document: Dict[str, Any], id: Optional[str] = None) -> Optional[str]:
        """
        ドキュメントをインデックスに追加します。
        成功した場合はドキュメントIDを返します。
        """
        try:
            response = await self.client.index(index=index_name, document=document, id=id)
            logger.info(f"Document indexed successfully in '{index_name}' with ID: {response['_id']}")
            return response['_id']
        except Exception as e:
            logger.error(f"Error indexing document in '{index_name}': {e}")
            return None

    async def search(self, index_name: str, query: str, fields: List[str], size: int = 5) -> List[Dict[str, Any]]:
        """
        指定されたインデックスとフィールドに対して全文検索を実行します。
        """
        try:
            response = await self.client.search(
                index=index_name,
                query={
                    "multi_match": {
                        "query": query,
                        "fields": fields
                    }
                },
                size=size
            )
            hits = []
            for hit in response['hits']['hits']:
                hits.append({
                    "id": hit['_id'],
                    "score": hit['_score'],
                    "source": hit['_source']
                })
            logger.info(f"Search for '{query}' in '{index_name}' returned {len(hits)} hits.")
            return hits
        except Exception as e:
            logger.error(f"Error during search in '{index_name}' for query '{query}': {e}")
            return []

    async def get_document(self, index_name: str, id: str) -> Optional[Dict[str, Any]]:
        """ドキュメントIDでドキュメントを取得します。"""
        try:
            response = await self.client.get(index=index_name, id=id)
            logger.info(f"Document '{id}' retrieved from '{index_name}'.")
            return response['_source']
        except NotFoundError:
            logger.warning(f"Document '{id}' not found in '{index_name}'.")
            return None
        except Exception as e:
            logger.error(f"Error getting document '{id}' from '{index_name}': {e}")
            return None


@asynccontextmanager
async def get_es_client(host: str = ELASTIC_SEARCH_HOST) -> AsyncGenerator[ElasticsearchClient, None]:
    client = ElasticsearchClient(host=host)
    try:
        yield client
    except:
        logger.error(traceback.print_exc())
    finally:
        await client.close()


embedding_model = SentenceTransformer("cl-nagoya/ruri-v3-30m", device=USE_DEVICE)


def embed(sentences: str | list[str]) -> list[list[float]]:
    if isinstance(sentences, str):
        sentences = [sentences]
    embeddings = embedding_model.encode(sentences, convert_to_tensor=False)
    return [e.tolist() for e in embeddings]


async def debug():
    res = embed("hello, this is a test")
    import pdb; pdb.set_trace()  # fmt: skip


if __name__ == '__main__':
    import asyncio
    asyncio.run(debug())


