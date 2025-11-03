from typing import List, Optional
from pydantic import BaseModel
from google.adk.agents import Agent
from src.schemas import ChatRequest, ChatResponse, Source, ChatContext
from src.es_search import ElasticsearchClient, get_es_client
import asyncio

# Define the LLM model to be used by the agents
LLM = 'gemini-1.5-flash' # Changed to string as per ADK example

class AdvancedRAGAgent:
    def __init__(self):
        self.query_rewriter_agent_instance = Agent(
            model=LLM,
            name="query_rewriter_agent",
            instruction="Rewrites the user's query based on chat history to improve search relevance.",
        )
        self.answer_agent_instance = Agent(
            model=LLM,
            name="answer_agent",
            instruction="Generates an answer draft based on a specific search result and the original query.",
        )
        self.synthesizer_agent_instance = Agent(
            model=LLM,
            name="synthesizer_agent",
            instruction="Synthesizes multiple answer drafts into a final, coherent response, citing sources and resolving contradictions.",
        )

    async def query_rewriter_agent(self, original_query: str, chat_history: List[ChatContext]) -> str:
        prompt = f"""Given the original query and chat history, rewrite the query to be more effective for a keyword and semantic search.
        Original Query: {original_query}
        Chat History: {chat_history}
        Rewritten Query:"""
        # The Agent class's generate_text method does not take AgentContext directly.
        # It's implicitly handled or not needed for simple text generation.
        rewritten_query = await self.query_rewriter_agent_instance(prompt=prompt)
        return rewritten_query

    async def answer_agent(self, original_query: str, chat_history: List[ChatContext], search_result: Source) -> str:
        prompt = f"""Given the original query, chat history, and the following search result, generate a concise answer draft.
        Original Query: {original_query}
        Chat History: {chat_history}
        Search Result Content: {search_result.full_text}
        Answer Draft:"""
        answer_draft = await self.answer_agent_instance(prompt=prompt)
        return answer_draft

    async def synthesizer_agent(self, original_query: str, chat_history: List[ChatContext], answer_drafts: List[str], sources: List[Source]) -> ChatResponse:
        prompt = f"""Given the original query, chat history, and several answer drafts, synthesize a final, comprehensive answer.
        Cite the sources for each piece of information. If there are contradictions, address them and provide the most likely correct information.

        Original Query: {original_query}
        Chat History: {chat_history}
        Answer Drafts:
        {chr(10).join([f"- {draft}" for draft in answer_drafts])}

        Sources:
        {chr(10).join([f"- {s.filename} (chunk {s.chunk_id})" for s in sources])}

        Final Answer:"""
        final_answer_text = await self.synthesizer_agent_instance.generate_text(prompt=prompt)
        return ChatResponse(answer=final_answer_text, sources=sources)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        original_query = request.query
        chat_history = request.context if request.context else []

        # Step 1: Query Expansion
        rewritten_query = await self.query_rewriter_agent(
            original_query=original_query, chat_history=chat_history
        )

        # Step 2: ElasticSearchからのハイブリッド検索
        search_results: List[Source] = []
        async with get_es_client() as es_client:
            search_results = await es_client.hybrid_search(query=rewritten_query, size=5)

        if not search_results:
            # Fallback: If no relevant search results are found
            fallback_answer = "I couldn't find specific information related to your query in the documents. Here's a general response based on my knowledge."
            return ChatResponse(answer=fallback_answer, sources=[])

        # Step 3: 個別回答の生成 (並列実行)
        answer_draft_tasks = [
            self.answer_agent(
                original_query=original_query,
                chat_history=chat_history,
                search_result=result,
            )
            for result in search_results
        ]
        answer_drafts = await asyncio.gather(*answer_draft_tasks)

        # Step 4: 最終回答の生成
        final_response = await self.synthesizer_agent(
            original_query=original_query,
            chat_history=chat_history,
            answer_drafts=answer_drafts,
            sources=search_results,
        )

        return final_response
