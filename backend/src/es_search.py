from typing import List, Dict, Any, Optional, AsyncGenerator
import traceback
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import functools
import collections
import asyncio
import time

from elasticsearch import AsyncElasticsearch, NotFoundError
from sentence_transformers import SentenceTransformer

from src.settings import (
    ELASTIC_SEARCH_HOST,
    USE_DEVICE,
    DEFAULT_INDEX_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    NUM_CONTEXT_CHUNKS,
    RRF_RANK_CONST,
    RRF_TOP_K,
)
from src import utils
from src.schemas import Source

# ロガーの設定
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

FIELD_MAPPINGS = {
    "properties": {
        "filepath": {"type": "keyword"},
        "filename": {"type": "text", "analyzer": "kuromoji"},
        "chunk_id": {"type": "integer"},
        "content": {"type": "text", "analyzer": "kuromoji"},
        "full_text": {"type": "text", "analyzer": "kuromoji"},
        "content_vector": {
            "type": "dense_vector",
            "dims": EMBEDDING_DIM,
            "index": True,
            "similarity": "cosine",
        },
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
                await self.client.indices.put_settings(
                    index=index_name, body={"index": {"number_of_replicas": 0}}
                )
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
        self, file_path: Path, index_name: str = DEFAULT_INDEX_NAME
    ) -> bool:
        """
        ドキュメントを解析、チャンク化し、Elasticsearchにインデックスします。
        成功した場合はドキュメントIDのリストを返します。
        """
        try:
            logger.info(f"Starting to index document: {file_path}")
            file_path = file_path.resolve() if file_path.is_absolute() else file_path
            markdown_text = utils.to_markdown(file_path)
            chunks: list[str] = [
                markdown_text[i : i + CHUNK_SIZE]
                for i in range(0, len(markdown_text), CHUNK_SIZE)
            ]
            embeddings: list[list[float]] = embed(chunks)

            docs = []
            for chunk_id, (chunk, embedding) in enumerate(
                zip(chunks, embeddings, strict=True)
            ):
                # fulltextは周辺の最大±2チャンクを連結した文字列
                prev: list[str] = chunks[
                    max(0, chunk_id - NUM_CONTEXT_CHUNKS) : chunk_id
                ]
                post: list[str] = chunks[
                    chunk_id : min(len(chunks), chunk_id + NUM_CONTEXT_CHUNKS + 1)
                ]
                full_text: str = "".join(prev + post)
                doc = {
                    "filepath": str(file_path),
                    "filename": file_path.name,
                    "chunk_id": chunk_id,
                    "content": chunk,
                    "full_text": full_text,
                    "content_vector": embedding,
                }
                docs.append(doc)

            # 1. filepath==str(file_path)の既存レコードをすべて削除
            await self.delete_document(file_path=file_path, index_name=index_name)

            # 2. docsをindexに登録
            operations = []
            for doc in docs:
                operations.append({"index": {"_index": index_name}})
                operations.append(doc)
            await self.client.bulk(operations=operations, refresh=True)

            logger.info(
                f"Successfully indexed {len(chunks)} chunks for document: {file_path.name}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to index document {file_path}: {e}")
            logger.error(traceback.format_exc())
            return False

    async def delete_document(
        self, file_path: Path, index_name: str = DEFAULT_INDEX_NAME
    ):
        await self.client.delete_by_query(
            index=index_name,
            body={"query": {"term": {"filepath": str(file_path)}}},
            refresh=True,
        )

    async def search(
        self, index_name: str, query: str, fields: List[str], size: int = 5
    ) -> List[Source]:
        """
        指定されたインデックスとフィールドに対して全文検索を実行します。
        """
        try:
            response = await self.client.search(
                index=index_name,
                query={"multi_match": {"query": query, "fields": fields}},
                size=size,
            )
            hits = response["hits"]["hits"]
            search_response: List[Source] = format_search_results(hits)
            logger.info(
                f"Search for '{query}' in '{index_name}' returned {len(search_response)} hits."
            )
            return search_response
        except Exception as e:
            logger.error(
                f"Error during search in '{index_name}' for query '{query}': {e}"
            )
            return []

    async def hybrid_search(
        self, query: str, size: int = 5, index_name: str = DEFAULT_INDEX_NAME
    ) -> List[Source]:
        """ハイブリッド検索を実行. RRFは有償版限定なので自力実装"""

        try:
            # --- 1. キーワード検索 (BM25) ---
            bm25_response = asyncio.create_task(
                self.client.search(
                    index=index_name,
                    query={
                        "multi_match": {
                            "query": query,
                            "fields": ["filename", "content"],
                        }
                    },
                    size=RRF_TOP_K,
                )
            )

            # --- 2. ベクトル検索 (kNN) ---
            query_vector = embed(query)[0]
            knn_response = asyncio.create_task(
                self.client.search(
                    index=index_name,
                    knn={
                        "field": "content_vector",
                        "query_vector": query_vector,
                        "k": RRF_TOP_K,
                        "num_candidates": 100,
                    },
                    _source=[
                        "filepath",
                        "filename",
                        "content",
                        "full_text",
                        "chunk_id",
                    ],  # kNNはデフォルトで_sourceを返さないため明示
                    size=RRF_TOP_K,
                )
            )

            # --- 3. 自力RRF ---
            bm25_hits = (await bm25_response)["hits"]["hits"]
            knn_hits = (await knn_response)["hits"]["hits"]
            fused_scores = collections.defaultdict(lambda: 0.0)

            for rank, hit in enumerate(bm25_hits):
                doc_id = hit["_id"]
                score = 1.0 / (RRF_RANK_CONST + rank + 1)
                fused_scores[doc_id] += score

            for rank, hit in enumerate(knn_hits):
                doc_id = hit["_id"]
                score = 1.0 / (RRF_RANK_CONST + rank + 1)
                fused_scores[doc_id] += score

            sorted_results = sorted(
                fused_scores.items(), key=lambda item: item[1], reverse=True
            )
            all_hits = {hit["_id"]: hit for hit in bm25_hits + knn_hits}
            final_hits = [all_hits[doc_id] for doc_id, score in sorted_results[:size]]

            # 4. 結果の整形
            search_response: List[Source] = format_search_results(final_hits)
            logger.info(
                f"Hybrid search for '{query}' in '{index_name}' returned {len(search_response)} hits."
            )
            return search_response
        except Exception as e:
            logger.error(
                f"Error during hybrid search in '{index_name}' for query '{query}': {e}"
            )
            logger.error(traceback.format_exc())
            return []

    async def get_document_list(
        self, index_name: str = DEFAULT_INDEX_NAME
    ) -> list[Path]:
        """ユニークなファイルパスのリストを取得します。"""
        try:
            query = {
                "size": 0,
                "aggs": {
                    "unique_filepaths": {
                        "terms": {
                            "field": "filepath",
                            "size": 10000,  # Get up to 10,000 unique file paths
                        }
                    }
                },
            }
            response = await self.client.search(index=index_name, body=query)
            filepaths = [
                Path(bucket["key"])
                for bucket in response["aggregations"]["unique_filepaths"]["buckets"]
            ]
            logger.info(
                f"Found {len(filepaths)} unique documents in index '{index_name}'."
            )
            return filepaths
        except Exception as e:
            logger.error(f"Error getting document list from index '{index_name}': {e}")
            return []


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


@utils.timer
def embed(sentences: str | list[str]) -> list[list[float]]:
    if isinstance(sentences, str):
        sentences = [sentences]
    model = get_embedding_model()
    embeddings = model.encode(
        sentences, convert_to_tensor=False, normalize_embeddings=True
    )
    return [e.tolist() for e in embeddings]


def format_search_results(hits: list[dict]) -> List[Source]:
    results = []
    for hit in hits:
        source_data = hit["_source"]
        source = Source(
            filepath=source_data["filepath"],
            filename=source_data["filename"],
            content=source_data["content"],
            full_text=source_data["full_text"],
            chunk_id=source_data["chunk_id"],
        )
        results.append(source)
    return results


async def _debug_1():
    for path in Path(
        "C:\\Users\\horoi\\Desktop\\AgenticRAG\\backend\\tests\\test_data"
    ).glob("*.pdf"):
        async with get_es_client() as es_client:
            ret = await es_client.index_document(path)


async def _debug_2():
    async with get_es_client() as es_client:
        ret = await es_client.search(
            query="ロシアの状況", fields=["content"], index_name=DEFAULT_INDEX_NAME
        )
    import pdb; pdb.set_trace()  # fmt: skip


async def _debug_3():
    async with get_es_client() as es_client:
        ret = await es_client.hybrid_search(query="ロシアの状況")
    import pdb; pdb.set_trace()  # fmt: skip


if __name__ == "__main__":
    import asyncio

    asyncio.run(_debug_3())
