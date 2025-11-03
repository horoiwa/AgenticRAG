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


def test_documents_endpoint(client: TestClient):
    """
    Tests the /documents endpoint.
    """
    # Test file path
    pdf_path = Path(__file__).resolve().parent / "test_data" / "1-2-1.pdf"

    # 1. POST /documents (Upload)
    with open(pdf_path, "rb") as f:
        response = client.post(
            "/documents",
            files={"file": (pdf_path.name, f, "application/pdf")},
            params={"index_name": TEST_INDEX_NAME},
        )

    assert response.status_code == 200
    assert response.json()["message"] == "Document indexed successfully"

    # Allow time for Elasticsearch to index the document
    async def wait_for_indexing():
        await asyncio.sleep(1)

    asyncio.run(wait_for_indexing())

    # 2. GET /documents (List)
    response = client.get("/documents", params={"index_name": TEST_INDEX_NAME})
    assert response.status_code == 200
    documents = response.json()
    assert isinstance(documents, list)
    assert any(doc["filename"] == pdf_path.name for doc in documents)

    # 3. DELETE /documents/{document_id}
    # Find the document_id (file_id) from the GET /documents response
    document_to_delete = next(
        (doc for doc in documents if doc["filename"] == pdf_path.name), None
    )
    assert document_to_delete is not None
    document_id = document_to_delete["file_id"]

    response = client.delete(
        f"/documents/{document_id}", params={"index_name": TEST_INDEX_NAME}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Document deleted successfully"

    # Allow time for Elasticsearch to delete the document
    asyncio.run(wait_for_indexing())

    # 4. GET /documents (Verify deletion)
    response = client.get("/documents", params={"index_name": TEST_INDEX_NAME})
    assert response.status_code == 200
    documents_after_delete = response.json()
    assert not any(
        doc["filename"] == pdf_path.name for doc in documents_after_delete
    )
