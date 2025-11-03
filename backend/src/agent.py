import copy

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from typing import AsyncGenerator


from src.es_search import get_es_client
from src.schemas import Source
from src import settings


# ステップ 1: クエリ拡張
query_rewriter_agent = LlmAgent(
    name="query_rewriter_agent",
    model=settings.LLM,
    instruction="""ユーザーの質問と会話履歴を考慮して、Elasticsearchでの検索に最適な検索クエリを生成してください。
    生成するクエリは、具体的で、検索に適したキーワードを複数含むようにしてください。
    応答は検索クエリの文字列だけにしてください。

    会話履歴：
    {history}
    """,
    output_key="search_query",
)


# ステップ 2: ElasticSearchからのハイブリッド検索
class SearchAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        query = ctx.session.state.get("search_query")
        if not query:
            yield Event(author=self.name, content="検索クエリが見つかりませんでした。")
            return

        async with get_es_client() as es_client:
            search_results: list[Source] = await es_client.hybrid_search(query=query)

        for i, search_result in enumerate(search_results, start=1):
            ctx.session.state[f"filename_{i}"] = search_result.filename
            ctx.session.state[f"content_{i}"] = search_result.content

        yield Event(
            author=self.name,
        )


search_agent = SearchAgent(name="search_agent")

# ステップ 3: 個別回答の生成
answer_agent = LlmAgent(
    name="answer_agent",
    model=settings.LLM,
    instruction="""ユーザーの質問と会話履歴を考慮して、参考資料に基づく回答を生成してください。

    参考資料:
    [1]{filename_1}
    {content_1}
    [2]{filename_2}
    {content_2}
    [3]{filename_3}
    {content_3}
    [4]{filename_4}
    {content_4}
    [5]{filename_5}
    {content_5}
    """,
    output_key="current_answer",
)


AdvancedRAGAgent = SequentialAgent(
    name="AdvancedRAGWorkflow",
    sub_agents=[
        query_rewriter_agent,
        search_agent,
        answer_agent,
    ],
)


from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role="user", parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        # You can uncomment the line below to see *all* events during execution
        print(
            f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}"
        )

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif (
                event.actions and event.actions.escalate
            ):  # Handle potential errors/escalations
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )
            # Add more checks here if needed (e.g., specific error codes)
            # break  # Stop processing events once the final response is found

    print(f"<<< Agent Response: {final_response_text}")


async def debug_1():
    APP_NAME = "AdvancedRAGApp"
    USER_ID = "User1"
    SESSION_ID = "session_id_1"

    session_service = InMemorySessionService()
    history = [
        {"role": "user", "content": "農業情報が気になります"},
        {
            "role": "assistant",
            "content": "承知しました、どの国について調べましょうか？",
        },
    ]
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={"history": history},
    )

    runner = Runner(
        agent=AdvancedRAGAgent,  # The agent we want to run
        app_name=APP_NAME,  # Associates runs with our app
        session_service=session_service,  # Uses our session manager
    )
    query = "ロシアの状況はどのようになっている？"
    await call_agent_async(query, runner, USER_ID, SESSION_ID)


if __name__ == "__main__":
    import asyncio

    asyncio.run(debug_1())
