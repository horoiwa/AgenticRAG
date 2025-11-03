import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import asyncio

from src.app import app
from src.es_search import get_es_client
from src import settings

# Use a separate index for testing
TEST_INDEX_NAME = "test_api_search_endpoint_sync"


@pytest.fixture(scope="module")
def client():
    """
    Provides a TestClient for making requests to the app.
    """

    # Setup: create index and index a document
    async def setup():
        async with get_es_client() as es_client:
            if await es_client.client.indices.exists(index=TEST_INDEX_NAME):
                await es_client.delete_index(TEST_INDEX_NAME)
            await es_client.create_index(TEST_INDEX_NAME)
            pdf_path = Path(__file__).resolve().parent / "test_data" / "1-1-1.pdf"
            await es_client.index_document(
                file_path=pdf_path, index_name=TEST_INDEX_NAME
            )
            await asyncio.sleep(1)

    asyncio.run(setup())

    with TestClient(app) as c:
        yield c

    # Teardown: delete index
    async def teardown():
        async with get_es_client() as es_client:
            await es_client.delete_index(TEST_INDEX_NAME)

    asyncio.run(teardown())


def test_search_endpoint(client: TestClient, monkeypatch):
    """
    Tests the /search endpoint.
    """
    response = client.get(
        "/search", params={"query": "ロシア", "index_name": TEST_INDEX_NAME}
    )

    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) > 0
    # Check the content of the results
    for item in results:
        assert "filepath" in item
        assert "filename" in item
        assert "content" in item
        assert "full_text" in item
        assert "chunk_id" in item
        # Check if the filename matches the indexed document
        assert item["filename"] == "1-1-1.pdf"
