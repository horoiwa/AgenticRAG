import pytest
import asyncio

from src.agent import AdvancedRAGAgent
from src.schemas import ChatRequest, ChatResponse, Source

@pytest.fixture
def advanced_rag_agent():
    return AdvancedRAGAgent()

@pytest.mark.asyncio
async def test_advanced_rag_agent_chat_returns_chat_response(advanced_rag_agent):
    request = ChatRequest(query="What is the capital of France?")
    response = await advanced_rag_agent.chat(request)

    assert isinstance(response, ChatResponse)
    assert isinstance(response.answer, str)
    assert isinstance(response.sources, list)

@pytest.mark.asyncio
async def test_advanced_rag_agent_chat_with_history_returns_chat_response(advanced_rag_agent):
    request = ChatRequest(
        query="What about its population?",
        context=[
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
    )
    response = await advanced_rag_agent.chat(request)

    assert isinstance(response, ChatResponse)
    assert isinstance(response.answer, str)
    assert isinstance(response.sources, list)