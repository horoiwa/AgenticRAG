# This file has been modified by the Gemini CLI.

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from typing import AsyncGenerator

from src.es_search import get_es_client
from src.schemas import Source


# ステップ 1: クエリ拡張
query_rewriter_agent = LlmAgent(
    name="query_rewriter_agent",
    model="gemini-1.5-flash",
    instruction="""ユーザーの質問と会話履歴を考慮して、Elasticsearchでの検索に最適な検索クエリを生成してください。
    生成するクエリは、具体的で、検索に適したキーワードを複数含むようにしてください。
    応答は検索クエリの文字列だけにしてください。
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
            search_results = await es_client.hybrid_search(query=query)

        ctx.session.state["search_results"] = search_results
        yield Event(
            author=self.name,
            content=f"{len(search_results)}件の検索結果を取得しました。",
        )


search_agent = SearchAgent(name="search_agent")

# ステップ 3: 個別回答の生成
answer_agent = LlmAgent(
    name="answer_agent",
    model="gemini-1.5-flash",
    instruction="""ユーザーの質問と会話履歴、そして以下の検索結果を考慮して、回答を生成してください。

    検索結果:
    {search_result}
    """,
    output_key="current_answer",
)


class AnswerLoopAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        search_results = ctx.session.state.get("search_results", [])
        answer_proposals = []

        for result in search_results:
            ctx.session.state["search_result"] = result.content
            async for event in answer_agent.run_async(ctx):
                if event.is_final_response():
                    answer_proposals.append(ctx.session.state.get("current_answer"))

        ctx.session.state["answer_proposals"] = answer_proposals
        yield Event(
            author=self.name,
            content=f"{len(answer_proposals)}件の回答案を生成しました。",
        )


answer_loop_agent = AnswerLoopAgent(name="answer_loop_agent")

# ステップ 4: 最終回答の生成
synthesizer_agent = LlmAgent(
    name="synthesizer_agent",
    model="gemini-1.5-pro",
    instruction="""ユーザーの質問と会話履歴、そして以下の複数の回答案を考慮して、最終的な回答を生成してください。
    各回答案から情報を取捨選択し、より洗練された「最終回答」を生成します。
    その際、どの情報源（検索結果）から引用したかを明記します。
    また、回答案の中に矛盾する情報が含まれる場合は、その点を考慮して最適な回答を構築します。

    回答案:
    {answer_proposals}
    """,
)

# 補足：適切な検索結果が得られなかった場合
fallback_agent = LlmAgent(
    name="fallback_agent",
    model="gemini-1.5-flash",
    instruction="""検索結果が見つからなかったため、LLMが持つ一般的な知識に基づいて回答を生成してください。""",
)


class OrchestratorAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        search_results = ctx.session.state.get("search_results", [])

        rag_pipeline = self.sub_agents[0]
        fallback_agent = self.sub_agents[1]

        if search_results:
            async for event in rag_pipeline.run_async(ctx):
                yield event
        else:
            async for event in fallback_agent.run_async(ctx):
                yield event


rag_pipeline = SequentialAgent(
    name="rag_pipeline", sub_agents=[answer_loop_agent, synthesizer_agent]
)

orchestrator_agent = OrchestratorAgent(
    name="orchestrator_agent", sub_agents=[rag_pipeline, fallback_agent]
)


AdvancedRAGAgent = SequentialAgent(
    name="AdvancedRAGAgent",
    sub_agents=[
        query_rewriter_agent,
        search_agent,
        orchestrator_agent,
    ],
)


root_agent = AdvancedRAGAgent
