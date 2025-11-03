import pytest
import pytest_asyncio
import asyncio
from typing import AsyncGenerator
from pathlib import Path

from src.es_search import get_es_client, ElasticsearchClient
from src import schemas

INDEX_NAME = "test_index"


@pytest_asyncio.fixture
async def es_client() -> AsyncGenerator[ElasticsearchClient, None]:
    """Elasticsearchクライアントのフィクスチャ"""
    async with get_es_client() as client:
        yield client


@pytest.mark.asyncio
async def test_create_and_delete_index(es_client: ElasticsearchClient):
    """
    インデックスの作成と削除をテストします。
    """
    # テスト用のインデックス名
    test_index_name = "test_create_delete_index"

    try:
        # 最初にインデックスが存在しないことを確認
        if await es_client.client.indices.exists(index=test_index_name):
            await es_client.delete_index(test_index_name)

        # インデックスを作成
        created = await es_client.create_index(test_index_name)
        assert created

        # インデックスが存在することを確認
        exists = await es_client.client.indices.exists(index=test_index_name)
        assert exists

    finally:
        # テスト後にインデックスを削除
        deleted = await es_client.delete_index(test_index_name)
        assert deleted

        # インデックスが削除されたことを確認
        exists_after_delete = await es_client.client.indices.exists(
            index=test_index_name
        )
        assert not exists_after_delete


@pytest.mark.asyncio
async def test_index_document(es_client: ElasticsearchClient):
    """
    ドキュメントのインデックスをテストします。
    """
    test_index_name = "test_index_document"
    try:
        await es_client.create_index(test_index_name)

        # テスト用PDFファイル
        pdf_path = Path(
            "C:/Users/horoi/Desktop/AgenticRAG/backend/tests/test_data/1-1-1.pdf"
        )

        # ドキュメントをインデックス
        indexed = await es_client.index_document(
            file_path=pdf_path, index_name=test_index_name
        )
        assert indexed

        # インデックスされたドキュメントを検索
        await asyncio.sleep(1)  # refreshされるまで少し待つ
        res: schemas.SearchResponse = await es_client.search(
            index_name=test_index_name, query="ロシア", fields=["content"]
        )
        assert len(res.results) > 0

    finally:
        # テスト後にインデックスを削除
        await es_client.delete_index(test_index_name)
