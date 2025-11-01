import pytest
import httpx
import asyncio
import os
from src.elasticsearch_client import ElasticsearchClient

# FastAPIアプリケーションのベースURL
# 環境変数から取得するか、デフォルト値を使用
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
# Elasticsearchのホスト
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")

@pytest.mark.asyncio
async def test_fastapi_connection():
    """
    バックエンドAPIへの接続テスト。
    /documentsエンドポイントにGETリクエストを送信し、200 OKが返されることを確認します。
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{FASTAPI_BASE_URL}/documents")
            response.raise_for_status()  # HTTPエラーがあれば例外を発生させる
            assert response.status_code == 200
            assert isinstance(response.json(), list)
            print(f"FastAPI connection successful. Status: {response.status_code}")
    except httpx.ConnectError as e:
        pytest.fail(f"FastAPI connection failed: {e}. Ensure the FastAPI application is running at {FASTAPI_BASE_URL}")
    except Exception as e:
        pytest.fail(f"FastAPI test failed: {e}")

@pytest.mark.asyncio
async def test_elasticsearch_connection():
    """
    Elasticsearchサーバーへの接続テスト。
    ElasticsearchClientのpingメソッドを使用し、接続が成功することを確認します。
    """
    try:
        es_client = ElasticsearchClient(host=ELASTICSEARCH_HOST)
        is_connected = await es_client.ping()
        assert is_connected is True
        print(f"Elasticsearch connection successful to {ELASTICSEARCH_HOST}")
    except Exception as e:
        pytest.fail(f"Elasticsearch connection failed: {e}. Ensure Elasticsearch is running and accessible at {ELASTICSEARCH_HOST}")
