import pytest
from fastapi.testclient import TestClient
from src.app import app
import asyncio
import os
from src.elasticsearch_client import ElasticsearchClient

# Elasticsearchのホスト
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")

client = TestClient(app)

def test_fastapi_connection():
    """
    バックエンドAPIへの接続テスト。
    /documentsエンドポイントにGETリクエストを送信し、200 OKが返されることを確認します。
    """
    response = client.get("/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    print(f"FastAPI connection successful. Status: {response.status_code}")

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
