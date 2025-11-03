# Google Agent Development Kit (ADK) Python Implementation Tutorial

## 1. はじめに
Google Agent Development Kit (ADK) は、LLM（大規模言語モデル）を活用した、複雑で自律的なエージェントアプリケーションを効率的に構築するためのフレームワークです。ADKは、エージェントの思考プロセス、外部ツールとの連携、複数のエージェントによる協調作業などを構造化し、開発を容易にします。

このチュートリアルでは、Pythonを使用してADKエージェントを実装する方法を、RAG (Retrieval-Augmented Generation) エージェントの例を通して、基本的な実装から応用的なトピックまで具体的に学びます。

## 2. ADKの主要コンセプト
ADKは主に3つのコンポーネントで構成されます。

*   **Tool (ツール)**: エージェントが外部の世界と対話するための具体的な手段です。APIの呼び出し、データベースへのクエリ、ファイル操作など、特定の機能を持つ関数やクラスとして実装されます。
*   **Agent (エージェント)**: 与えられたタスクを達成するために、思考し、ツールを使いこなす主体です。エージェントは、LLMを思考エンジンとして利用し、どのツールをどの順番で使うかを自律的に判断します。
*   **Workflow (ワークフロー)**: 複数のエージェントやツールを組み合わせ、より複雑なタスクを達成するための処理の流れを定義します。各ステップの入出力を定義し、エージェント間の連携をオーケストレーションします。

## 3. 開発環境のセットアップ
ADKを利用するための環境を準備します。

1.  **Pythonのインストール**: Python 3.9以上がインストールされていることを確認してください。
2.  **仮想環境の作成とアクティベート**:
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate
    # macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **ADKライブラリのインストール**:
    ```bash
    pip install google-generativeai-adk "google-generativeai[gemini]" python-dotenv
    ```
    `python-dotenv`は、APIキーなどの設定情報を管理するために追加します。

4.  **APIキーの設定**:
    プロジェクトのルートディレクトリに `.env` ファイルを作成し、Google AIのAPIキーを記述します。
    ```
    # .env
    GOOGLE_API_KEY="YOUR_API_KEY"
    ```

    コード内でこのキーを安全に読み込むために、`Gemini`モデルの初期化部分を修正します。

    ```python
    # src/agent.py
    import os
    from dotenv import load_dotenv
    from adk.llm import Gemini

    load_dotenv()
    llm = Gemini(api_key=os.getenv("GOOGLE_API_KEY"))
    ```

## 4. RAGエージェントの実装
`GEMINI.md`の戦略に沿って、具体的な実装を進めます。

### 4.1. ツール (Tools) の実装
`@tool`デコレータは、関数のdocstringを読み取り、LLMがツールの機能を理解するための説明として利用します。**詳細で分かりやすいdocstringを記述することが、エージェントの性能向上に繋がります。**

```python
# src/tools.py
from adk.tool import tool
from typing import List, Dict

# ... (search_documents関数の定義) ...

@tool
def search_tool(query: str) -> List[Dict[str, str]]:
    """
    ユーザーの質問に関連するドキュメントをナレッジベースから検索します。
    Args:
        query (str): 検索キーワードや質問文。
    Returns:
        List[Dict[str, str]]: 見つかったドキュメントのリスト。各ドキュメントはsourceとcontentを持つ辞書です。
    """
    return search_documents(query)
```

### 4.2. エージェント (Agents) の実装
プロンプトを工夫することで、エージェントの性能をさらに引き出すことができます。

#### 発展: Few-Shotプロンプティング
プロンプト内に具体的な応答例（ショット）をいくつか含めることで、LLMがタスクをより正確に理解し、期待する形式で出力するよう促すことができます。

```python
# src/agent.py (QueryRewriterAgentの例)
class QueryRewriterAgent(Agent):
    def __init__(self):
        super().__init__(
            llm=llm,
            prompt=Prompt(
                "ユーザーの質問と会話履歴を考慮して、情報検索に最適な検索クエリを生成してください。\n\n"
                "--- 例1 ---\n"
                "会話履歴: 「こんにちは」\n"
                "ユーザーの質問: 「ADKについて教えて」\n"
                "検索クエリ: 「Google Agent Development Kit」\n"
                "--- 例2 ---\n"
                "会話履歴: 「ADKはエージェント開発を支援するんですね。」\n"
                "ユーザーの質問: 「RAGとの関連は？」\n"
                "検索クエリ: 「ADK RAG エージェント」\n"
                "--- 本番 ---\n"
                "会話履歴: {{chat_history}}\n"
                "ユーザーの質問: {{query}}\n"
                "検索クエリ:"
            ),
        )
    # ... executeメソッドは同じ ...
```

### 4.3. ワークフロー (Workflow) の構築
基本的なワークフローの定義は前のセクションと同じです。ここでは、より実践的な側面を次のセクションで掘り下げます。

## 5. 応用的なトピック
### 5.1. エラーハンドリング
ワークフローの途中でエラーが発生した場合（例: 外部APIの呼び出し失敗）、ワークフロー全体が停止してしまいます。`Step`に`on_error`コールバックを登録することで、エラー発生時の代替処理を定義できます。

```python
# src/workflow.py
def handle_search_error(error: Exception, inputs: Dict) -> Dict:
    """検索ステップでエラーが発生した際の処理"""
    print(f"Search failed: {error}. Returning empty results.")
    return {"search_results": []}

# search_stepの定義を修正
search_step = Step(
    agent=search_tool,
    name="SearchDocuments",
    input_params={"query": str},
    output_params={"search_results": List[Dict[str, str]]},
    on_error=handle_search_error, # エラーハンドラを登録
)
```

### 5.2. 回答のストリーミング
LLMの生成が完了するのを待つのではなく、生成されたトークンを順次クライアントに送信することで、体感速度を大幅に向上させることができます。

1.  **LLMのストリーミング設定**: `Gemini`モデルでストリーミングを有効にします。
    ```python
    # src/agent.py (SynthesizerAgentのexecuteメソッド)
    def execute(self, query: str, answer_drafts: List[str]):
        """最終回答をストリームで生成します。"""
        return self.llm.stream( # .infer()の代わりに.stream()を使用
            prompt=self.prompt,
            query=query,
            answer_drafts=str(answer_drafts),
        )
    ```

2.  **FastAPIでのストリーミング応答**: `StreamingResponse`を使用して、ジェネレータからの出力をストリーミングします。
    ```python
    # src/app.py
    from fastapi.responses import StreamingResponse

    async def stream_generator(response_stream):
        """ADKのストリームをFastAPIのレスポンスに変換するジェネレータ"""
        for chunk in response_stream:
            yield chunk.get("output", "")

    @app.post("/chat-stream")
    async def chat_stream(request: ChatRequest):
        """RAGワークフローを実行し、回答をストリーミングします。"""
        workflow_input = {"query": request.query, "chat_history": request.chat_history}

        # SynthesizerAgentのexecuteが返すのはジェネレータ
        response_stream = rag_workflow.execute(workflow_input).get("final_answer")

        return StreamingResponse(stream_generator(response_stream), media_type="text/plain")
    ```

### 5.3. 会話履歴の管理（応用）

ADKはデフォルトで会話の文脈を自動でLLMに渡しますが、会話が長くなるとトークン量の増大と思考精度の低下を招く可能性があります。そのため、開発者が会話履歴を明示的に管理し、LLMに渡す情報を最適化することが推奨されます。これにより、トークン量を節約し、エージェントの応答精度を向上させることができます。

#### パターンA: クエリ書き換え (Query Rewriting)

長い会話履歴全体をLLMに渡す代わりに、会話の文脈を理解する専門のエージェント（`QueryRewriterAgent`）を用意するパターンです。このエージェントは、過去のやり取りとユーザーの最新の質問を基に、次のアクション（検索など）に最も適した、簡潔なクエリを新たに生成します。

```python
# src/agent.py (QueryRewriterAgentの例)

query_rewriter_prompt = """
ユーザーの質問とこれまでの会話履歴を考慮して、情報検索に最適な検索クエリを一つ生成してください。

--- 例 ---
会話履歴: 「ADKについて教えて」\nAI: 「ADKはGoogleが開発したエージェント開発キットです。」
ユーザーの質問: 「RAGとの関連は？」

検索クエリ: 「ADK RAG エージェント 連携」
--- 本番 ---
会話履歴: {chat_history}
ユーザーの質問: {query}

検索クエリ:"""

query_rewriter_agent = Agent(
    name="QueryRewriterAgent",
    model="gemini-1.5-flash",
    instruction=query_rewriter_prompt,
    output_key="rewritten_query" # 生成したクエリをstateに保存
)

# ワークフローの最初のステップとしてこのエージェントを組み込む
# rag_workflow = SequentialAgent(sub_agents=[query_rewriter_agent, ...])
```

#### パターンB: 手動での履歴整形と注入

アプリケーション側（例: FastAPIサーバー）で、会話履歴を特定のフォーマットの文字列に整形し、ワークフローの初期入力として`session.state`に注入する方法です。プロンプト内で履歴をどのように見せるかを完全に制御できます。

```python
# src/app.py (FastAPIなどのアプリケーション層での処理)

# (セッションIDに基づいて永続化層から履歴を取得するロジック)
history = get_history_for_session(session_id)

# 履歴を特定のフォーマットの文字列に整形
# 例: 直近3回のやり取りのみを対象にする
chat_history_str = "\n".join(
    f"User: {h['q']}\nAI: {h['a']}" for h in history[-3:]
)

# ワークフローの入力として、整形した履歴を渡す
workflow_input = {
    "query": "ユーザーの現在の質問",
    "chat_history": chat_history_str, # stateに格納される
}

# ワークフローを実行
result = rag_workflow.execute(workflow_input)

# ...

# エージェントのプロンプト側では {chat_history} で履歴を受け取る
# instruction="... 過去の会話: {chat_history} ..."
```

#### 永続化について

これらのパターンを本番環境で運用するには、会話履歴の永続化が不可欠です。開発初期段階ではインメモリの辞書で問題ありませんが、ユーザーがセッションをまたいで会話を継続できるようにするためには、`Session`オブジェクト（会話履歴を含む）をデータベースに保存する必要があります。ADKは、本番環境向けに`VertexAiSessionService`や`DatabaseSessionService`といった永続化のための仕組みを提供しています。


## 6. デバッグとロギング
ワークフローの各ステップの動作を理解するために、ADKはロギング機能を提供しています。

```python
import logging

# ログレベルを設定してADKの内部ログを表示
logging.basicConfig(level=logging.INFO)
logging.getLogger("adk").setLevel(logging.DEBUG)

# この設定の後、workflow.execute()を実行すると、
# 各ステップの入出力やLLMとのやり取りがコンソールに出力されるようになります。
```

## 7. テスト
前のセクションで示したユニットテストに加えて、ワークフロー全体の振る舞いを確認する結合テストも重要です。

```python
# tests/test_workflow.py
from src.workflow import create_rag_workflow
from adk.testing import mock_llm_stream, mock_tool_call

def test_rag_workflow():
    """RAGワークフロー全体の結合テスト"""
    rag_workflow = create_rag_workflow()

    # LLMのストリーミング応答とツールの呼び出しをモック
    with mock_llm_stream([{"output": "rewritten query"}]), \
         mock_tool_call("search_tool", [{"source": "mock", "content": "mock content"}]), \
         mock_llm_stream([{"output": "draft answer"}]), \
         mock_llm_stream([{"output": "final answer"}]):

        result = rag_workflow.execute({
            "query": "test query",
            "chat_history": ""
        })

    assert result.get("final_answer") is not None
```

## 8. まとめ
このチュートリアルでは、ADKの基本から、エラーハンドリング、ストリーミング、状態管理といった応用的なトピックまでをカバーしました。ADKは、LLMエージェント開発における複雑さを整理し、堅牢でスケーラブルなアプリケーションを構築するための強力な基盤を提供します。

より詳細な情報や高度な使い方については、公式ドキュメントやサンプルリポジトリを参照してください。
*   **ADK Samples**: https://github.com/google/adk-samples

## 9. ADK実装パターン集 (`adk-samples`より)

`adk-samples`リポジトリには、このチュートリアルで構築したRAGエージェント以外にも、様々な実装パターンを示すサンプルが含まれています。ここでは、その中からいくつかの代表的な例を紹介します。これらのサンプルは、より複雑なエージェントを構築する上での実践的な参考資料となります。

### 9.1. ReAct (Reason + Act) パターン

**概要:**
ReActは、エージェントが「思考（Reason）」と「行動（Act）」を交互に繰り返すことで、複雑なタスクを解決するための基本的な思考パターンです。LLMは次の行動計画を立て、ツールを実行し、その結果を観察して、次の思考へと繋げます。これはADKにおける単一の`LlmAgent`の基本的な動作原理です。

**主要コンポーネント:**
*   `LlmAgent`: 思考と推論の中心。
*   `instruction`: エージェントの役割、タスク解決のための思考プロセス、ツールの使い方を定義するプロンプト。
*   `tools`: エージェントが利用可能な関数のリスト。

**実装例 (`LlmAgent`の基本的な定義):**
```python
from google.adk.agents import Agent

# ツールとして使用する関数を定義
def get_current_time(city: str) -> dict:
    """指定された都市の現在時刻を返します。"""
    # ... (実際の時刻取得処理) ...
    if city.lower() == "new york":
        return {"status": "success", "time": "10:30 AM EST"}
    return {"status": "error", "message": f"Time for {city} not available."}

# ReActパターンを実装するエージェント
time_teller_agent = Agent(
    name="time_teller_agent",
    model="gemini-1.5-flash",
    instruction="あなたは都市の現在時刻を教えるアシスタントです。'get_current_time'ツールを使ってください。",
    description="指定された都市の現在時刻を教えます。",
    tools=[get_current_time]
)
```
このエージェントは、ユーザーから「ニューヨークの時間は？」と尋ねられると、`instruction`に従って思考し、`get_current_time`ツールを呼び出すという「行動」を起こします。

### 9.2. 階層型タスク委任 (マルチエージェント・コラボレーション)

**概要:**
より複雑なタスクを解決するために、1つの上位エージェント（コーディネーター）が、タスクをより小さなサブタスクに分解し、それぞれを専門とする下位エージェント（サブエージェント）に委任するパターンです。ADKでは、`LlmAgent`の`sub_agents`プロパティや`AgentTool`を利用してこの階層構造を構築します。

**主要コンポーネント:**
*   **コーディネーターエージェント (`LlmAgent`)**: ユーザーからのリクエストを受け取り、どのサブエージェントに処理を任せるかを判断します。`instruction`で委任のロジックを定義します。
*   **サブエージェント (`LlmAgent`, `SequentialAgent`など)**: 特定の専門領域のタスクを実行します。サブエージェントは、自身の`description`プロパティを明確に定義することが重要です。コーディネーターは、この`description`を読んで委任先を判断します。
*   `AgentTool`: あるエージェントを、別のエージェントの「ツール」として機能させるためのラッパー。これにより、エージェントが他のエージェントを直接呼び出せます。

**実装例 (研究アシスタント):**
```python
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools.agent_tool import AgentTool

# --- サブエージェントの定義 ---
# 1. リサーチ計画を生成するエージェント
plan_generator = LlmAgent(
    name="plan_generator",
    model="gemini-1.5-flash",
    description="行動指向のリサーチ計画を4-5行で生成します。",
    # ... instruction ...
)

# 2. 実際の調査とレポート作成を行うワークフローエージェント
research_pipeline = SequentialAgent(
    name="research_pipeline",
    description="承認されたリサーチ計画を実行し、最終的なレポートを作成します。",
    sub_agents=[
        # ... (調査、評価、レポート作成を行うエージェント群) ...
    ],
)

# --- コーディネーターエージェントの定義 ---
# ユーザーと対話し、サブエージェントにタスクを委任する
interactive_planner_agent = LlmAgent(
    name="interactive_planner_agent",
    model="gemini-1.5-flash",
    description="リサーチ計画を作成し、承認後に実行を委任する主要アシスタント。",
    instruction="""
    あなたの仕事は以下の通りです:
    1. `plan_generator`ツールを使い、リサーチ計画の草案を作成します。
    2. ユーザーの承認が得られたら、`research_pipeline`エージェントに実行を委任します。
    あなた自身は調査を行いません。計画、修正、委任に徹してください。
    """,
    sub_agents=[research_pipeline], # 委任可能なサブエージェント
    tools=[AgentTool(plan_generator)], # ツールとして使用するサブエージェント
)

# アプリケーションのルートエージェント
root_agent = interactive_planner_agent
```
この例では、`interactive_planner_agent`がユーザーとの対話を担当し、計画作成は`plan_generator`に、計画の実行は`research_pipeline`に、それぞれ委任しています。

### 9.3. 反復的改善 (Iterative Refinement) パターン

**概要:**
生成(Generate)と評価(Critique)のサイクルを繰り返すことで、成果物の品質を段階的に向上させるパターンです。例えば、ブログ記事の草稿を生成し、それを評価エージェントがレビューして修正点を指摘、その指摘に基づいて草稿を修正する、というプロセスを繰り返します。このパターンは`LoopAgent`を使って実装するのが一般的です。

**主要コンポーネント:**
*   `LoopAgent`: 指定された回数または特定の条件が満たされるまで、サブエージェント群を繰り返し実行します。
*   **生成エージェント (`LlmAgent`)**: 成果物（文章、コードなど）を作成します。
*   **評価エージェント (`LlmAgent`)**: 成果物を評価し、フィードバック（修正点やスコア）を生成します。`output_schema`を使って評価結果をJSON形式で出力させると、後続の処理が容易になります。
*   **終了条件判定エージェント (`BaseAgent`)**: 評価結果に基づき、ループを継続するか終了するかを判断します。`EventActions(escalate=True)`を持つ`Event`をyieldすることで、ループを停止できます。

**実装例 (リサーチ結果の品質チェック):**
```python
from google.adk.agents import LoopAgent, Agent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator

# 評価結果を定義するPydanticモデル
class Feedback(BaseModel):
    grade: Literal["pass", "fail"]
    comment: str

# 1. リサーチ結果を評価し、Feedbackモデルに従ったJSONを出力するエージェント
research_evaluator = Agent(
    name="research_evaluator",
    model="gemini-1.5-pro",
    instruction="リサーチ結果を厳しく評価し、合否とコメントをJSONで出力せよ。",
    output_schema=Feedback,
    output_key="research_evaluation", # 結果をstateに保存
)

# 2. 評価結果(state)を読み取り、ループを停止するか判断するエージェント
class EscalationChecker(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        evaluation = ctx.session.state.get("research_evaluation")
        if evaluation and evaluation.get("grade") == "pass":
            # gradeが'pass'なら、escalate=Trueでループを抜ける
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            # それ以外はループを継続
            yield Event(author=self.name)

# 3. 指摘に基づきリサーチをやり直すエージェント
enhanced_search_executor = Agent(
    name="enhanced_search_executor",
    # ... instruction ...
)

# 上記エージェントをループで実行
iterative_refinement_loop = LoopAgent(
    name="IterativeRefinementLoop",
    sub_agents=[
        research_evaluator,       # ステップ1: 評価
        EscalationChecker(name="EscalationChecker"), # ステップ2: 終了判定
        enhanced_search_executor, # ステップ3: 修正（ループが継続する場合のみ実行）
    ],
    max_iterations=5, # 無限ループを防ぐための最大回数
)
```

### 9.4. ツールセットによる機能拡張

**概要:**
関連する複数のツールを一つの「ツールセット」としてまとめることで、エージェントの能力を体系的に拡張するパターンです。例えば、データベースを操作するためのツール群（`list_tables`, `execute_sql`など）を`DatabaseToolset`として定義し、エージェントに提供します。これにより、エージェントはより複雑なデータベース関連のタスクをこなせるようになります。

**主要コンポーネント:**
*   `FunctionTool`: 個々のツールを定義します。
*   `OpenAPIToolset`: OpenAPI仕様書から自動的にツールセットを生成します。
*   `BigQueryToolset`, `ApplicationIntegrationToolset`など、Google Cloudサービスと連携するための組み込みツールセット。

**実装例 (BigQueryツールセットの利用):**
```python
from google.adk.agents import Agent
from google.adk.tools.bigquery import BigQueryToolset

# BigQueryのプロジェクトIDとデータセットIDを指定
bq_toolset = BigQueryToolset(
    project_id="your-gcp-project-id", 
    dataset_id="your_bq_dataset_id"
)

data_analyst_agent = Agent(
    name="data_analyst_agent",
    model="gemini-1.5-pro",
    instruction="あなたはBigQueryを操作してデータ分析を行う専門家です。",
    # BigQueryToolsetをツールとしてエージェントに渡す
    # これにより、list_tables, get_schema, execute_sqlなどのツールが利用可能になる
    tools=bq_toolset.tools(), 
)
```

### 9.5. ダイナミック・プランニング

**概要:**
ユーザーからの曖昧または複雑なリクエストに対して、エージェントが自らタスクを複数のステップに分解し、動的に実行計画（プラン）を立てるパターンです。ADKでは`Planner`オブジェクトをエージェントに設定することで、この機能を有効化できます。モデルが持つ多段階の推論能力を最大限に引き出します。

**主要コンポーネント:**
*   `Planner`: エージェントに計画立案能力を付与します。
    *   `BuiltInPlanner`: Geminiのような、モデルに組み込まれた計画・思考機能を利用します。
    *   `PlanReActPlanner`: モデルがPlan-Reason-Act形式で思考するように促します。
*   `LlmAgent`: `Planner`を搭載し、計画に基づいてツールを実行します。

**実装例 (BuiltInPlannerの利用):**
```python
from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.genai.types import ThinkingConfig
from google.adk.tools import google_search

# 思考プロセスを有効にする設定
thinking_config = ThinkingConfig(include_thoughts=True)

# プランナーを搭載したエージェント
planning_agent = Agent(
    name="planning_agent",
    model="gemini-1.5-pro",
    instruction="与えられたタスクを解決するための計画を立て、実行してください。",
    tools=[google_search], # 計画の実行に必要なツール
    planner=BuiltInPlanner(
        thinking_config=thinking_config
    ),
)
```
このエージェントに「来週の東京の天気と、それに基づいた服装の提案をして」といった複雑なリクエストを投げると、エージェントは内部的に「1. 東京の来週の天気を調べる」「2. 天気情報から服装を提案する」といった計画を立て、ステップバイステップでタスクを実行します。

### 9.6. セーフティ・ガードレール

**概要:**
エージェントが安全でない振る舞い（不適切なコンテンツの生成、意図しないツールの実行など）をしないように、多層的な防御を実装するパターンです。ADKでは、コールバック関数を使ってモデルやツールの入出力を監視・制御することで、ガードレールを設けることができます。

**主要コンポーネント:**
*   **コールバック関数**: エージェントのライフサイクルの特定のポイント（モデル呼び出し前後、ツール実行前後など）で実行される関数。
    *   `before_model_callback`: LLMにリクエストを送信する直前に呼び出され、プロンプトの内容を検証・修正できます。
    *   `before_tool_callback`: ツールが実行される直前に呼び出され、ツール名や引数を検証・修正できます。
*   **`BasePlugin`**: アプリケーション全体にまたがる横断的な関心事（ロギング、ポリシー適用など）を実装するための仕組み。コールバックよりも広範囲な制御が可能です。

**実装例 (入力内容の事前チェック):**
```python
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.genai.types import LlmRequest, LlmResponse, Content, Part
from typing import Optional

# 不適切な単語のリスト
FORBIDDEN_WORDS = ["dangerous_phrase", "inappropriate_topic"]

def input_safety_checker(context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """LLMへのリクエストに不適切な単語が含まれていないかチェックするコールバック"""
    user_input = llm_request.contents[-1].parts[0].text
    if any(word in user_input.lower() for word in FORBIDDEN_WORDS):
        # 不適切な単語が見つかった場合、LLMへのリクエストを中断し、
        # 固定の応答を返すことで、エージェントの実行を安全に停止する。
        return LlmResponse(
            content=Content(parts=[Part(text="申し訳ありませんが、そのリクエストにはお答えできません。")])
        )
    # 問題がなければNoneを返し、通常の処理を続行する
    return None

safe_agent = Agent(
    name="safe_agent",
    model="gemini-1.5-flash",
    instruction="ユーザーをサポートする安全なアシスタントです。",
    # モデル呼び出しの前にセーフティチェックを実行する
    before_model_callback=input_safety_checker,
)
```

### 9.7. リアルタイム対話エージェント

**概要:**
テキストだけでなく、音声や映像をリアルタイムで処理し、低遅延で双方向の対話を実現する最先端のパターンです。ユーザーが話している途中で割り込んだり、エージェントが思考しながら音声で応答したりといった、より自然なインタラクションが可能になります。

**主要コンポーネント:**
*   **`Runner.run_live()`**: ストリーミングセッションを開始するためのエントリーポイント。
*   **`LiveRequestQueue`**: クライアントからエージェントへ、音声チャンクなどのデータを送信するためのキュー。
*   **ストリーミング対応ツール**: `async def`かつ戻り値が`AsyncGenerator`である関数。処理の途中経過をyieldで返すことができます。
*   **マルチモーダル対応モデル**: `gemini-1.5-flash-exp`など、テキスト、音声、画像を処理できるモデル。

**実装例 (概念コード):**
```python
from typing import AsyncGenerator

# 非同期ジェネレータとして定義されたストリーミングツール
async def monitor_stock_price(symbol: str) -> AsyncGenerator[str, None]:
    """株価の更新情報を継続的にyieldで返すストリーミングツール"""
    while True:
        price = await get_live_price(symbol) # 外部APIから最新価格を取得
        yield f"{symbol}の現在の価格: ${price}"
        await asyncio.sleep(5)

# ... Runnerのセットアップ ...

# RunConfigで応答モダリティ（テキストや音声）を指定
run_config = RunConfig(
    response_modalities=['TEXT', 'AUDIO']
)

# run_live()でリアルタイムセッションを開始
async for event in runner.run_live(
    user_id="user_123",
    session_id="session_456",
    run_config=run_config,
):
    # サーバーからクライアントへイベントをリアルタイムに送信
    # (例: WebSocket経由で音声データやテキストを送信)
    if event.content:
        # ...
```
このパターンは、カスタマーサポートの音声ボットや、リアルタイムの言語翻訳、インタラクティブなゲームなど、高度な応用分野で利用されます。

### 9.8. 構造化データ出力 (Structured Data Output)

**概要:**
LLMの出力を自然言語のテキストではなく、厳密なスキーマに準拠したJSON形式に強制するパターンです。これにより、後続のプログラムでの処理、API呼び出し、データベースへの保存が容易かつ確実になります。ADKでは`output_schema`にPydanticモデルを指定することで実現します。

**注意点:** `output_schema`を使用すると、エージェントはJSONを出力することに専念するため、ツール呼び出しや他のエージェントへの委任といった能力が無効になります。

**主要コンポーネント:**
*   `pydantic.BaseModel`: 出力させたいJSONの構造、データ型、そして各フィールドの説明を定義します。この説明文はLLMへの指示として利用されるため、非常に重要です。
*   `LlmAgent.output_schema`: エージェントの出力形式としてPydanticモデルを指定します。
*   `LlmAgent.output_key`: 生成されたJSONオブジェクトを`session.state`に保存するためのキーを指定します。

**実装例 (リサーチ品質の評価):**
```python
from pydantic import BaseModel, Field
from typing import Literal
from google.adk.agents import LlmAgent

# Pydanticモデルで出力スキーマを定義
class Feedback(BaseModel):
    """リサーチ品質の評価フィードバックを表すモデル"""
    grade: Literal["pass", "fail"] = Field(
        description="評価結果。リサーチが十分なら'pass'、不十分なら'fail'。"
    )
    comment: str = Field(
        description="評価の詳細な説明。リサーチの長所や短所を具体的に指摘する。"
    )

# output_schemaを指定した評価エージェント
research_evaluator = LlmAgent(
    name="research_evaluator",
    model="gemini-1.5-pro",
    instruction="""あなたは品質保証アナリストです。与えられたリサーチ結果を厳しく評価し、'Feedback'スキーマに準拠したJSONオブジェクトを一つだけ出力してください。""",
    output_schema=Feedback, # 出力スキーマを指定
    output_key="research_evaluation", # 結果をstateに保存するキー
    disallow_transfer_to_peers=True, # 他エージェントへの委任を禁止
)
```

### 9.9. 状態管理とコンテキスト注入 (State Management & Context Injection)

**概要:**
エージェント間の情報の受け渡しや、会話の文脈維持に`session.state`を利用するパターンです。あるエージェントの出力(`output_key`経由)やツールの実行結果を`state`に保存し、別のエージェントのプロンプトに`{state_key}`の形式で埋め込むことで、ワークフロー全体で情報を共有します。

**主要コンポーネント:**
*   `session.state`: 同一セッション内で共有される揮発性のメモリ領域（Pythonの辞書）。
*   `LlmAgent.output_key`: エージェントの最終的なテキスト出力を`state`の指定されたキーに自動的に保存します。
*   プロンプト内のプレースホルダー: `instruction`や`prompt`内の`{state_key}`という記述は、実行時に`state['state_key']`の値で置き換えられます。

**実装例 (要約と質問生成):**
```python
from google.adk.agents import SequentialAgent, Agent

# エージェント1: ドキュメントを要約し、結果を state['document_summary'] に保存
summarizer = Agent(
    name="DocumentSummarizer",
    model="gemini-1.5-flash",
    instruction="提供されたドキュメントを3文で要約してください。",
    output_key="document_summary" # 出力をstateに保存
)

# エージェント2: stateから要約を読み込み、それに基づいて質問を生成
question_generator = Agent(
    name="QuestionGenerator",
    model="gemini-1.5-flash",
    # stateの'document_summary'キーの値をプロンプトに注入
    instruction="この要約に基づいて、理解度を確認するための質問を3つ生成してください: {document_summary}",
)

# 上記エージェントを順番に実行するワークフロー
document_pipeline = SequentialAgent(
    name="SummaryQuestionPipeline",
    sub_agents=[summarizer, question_generator], # 実行順が重要
)
```

### 9.10. 評価とテスト (Evaluation & Testing)

**概要:**
エージェントが期待通りに振る舞うかを確認し、品質を維持するためのパターンです。ADKは、`adk eval`コマンドや`pytest`と統合可能な評価フレームワークを提供しており、定義されたテストケースに基づいてエージェントの応答やツール呼び出しの正確性を自動評価します。

**主要コンポーネント:**
*   **Evalset (`.evalset.json`)**: 評価シナリオを定義したJSONファイル。各テストケースには、ユーザー入力、期待される最終応答、期待される中間データ（ツール呼び出しなど）が含まれます。
*   `adk eval`コマンド: Evalsetファイルを使ってエージェントの評価を実行するCLIツール。
*   `AgentEvaluator.evaluate()`: `pytest`などのテストコード内でプログラム的に評価を実行するためのメソッド。

**実装例 (Evalsetの断片):**
```json
{
  "eval_set_id": "weather_bot_eval",
  "eval_cases": [
    {
      "eval_id": "london_weather_query",
      "conversation": [
        {
          "user_content": {"parts": [{"text": "ロンドンの天気は？"}]},
          "final_response": {"parts": [{"text": "ロンドンの天気は曇りです..."}]},
          "intermediate_data": {
            "tool_uses": [{"name": "get_weather", "args": {"city": "London"}}]
          }
        }
      ]
    }
  ]
}
```
このEvalsetは、「ロンドンの天気は？」という質問に対し、エージェントが`get_weather(city="London")`というツールコールを行い、「ロンドンの天気は曇りです...」という応答を返すことを期待しています。`adk eval`コマンドは、実際のエージェントの動作がこの期待と一致するかを評価します。

### 9.11. 決定論的なツール呼び出し

**概要:**
`LlmAgent`によるツール呼び出しは、LLMの判断に依存するため本質的に「非決定的」です。しかし、ワークフローのある段階で「必ず特定のツールを、特定の引数で実行したい」という要件は頻繁に発生します。この「決定的」なツール呼び出しは、カスタムの`BaseAgent`を作成し、その中でツール関数を直接呼び出すことで実現できます。

**実装方法:**
`@tool`でデコレートされた関数も、実体は単なるPython関数です。LLMを介さずに、`BaseAgent`を継承したカスタムエージェントの実行ロジック（`_run_async_impl`）内で直接呼び出します。

#### ステップ1: ツールを定義する (`src/tools.py`)

まず、通常通りツールを定義します。
```python
# src/tools.py
from adk.tool import tool
from typing import List, Dict

@tool
def search_tool(query: str) -> List[Dict[str, str]]:
    """
    ユーザーの質問に関連するドキュメントをナレッジベースから検索します。
    """
    print(f"「{query}」で検索を実行しました。")
    # ...検索ロジック...
    return [{"source": "doc1.txt", "content": f"「{query}」に関する情報です。"}]
```

#### ステップ2: ツールを直接呼び出すカスタムエージェントを作成する

`BaseAgent`を継承し、`_run_async_impl`メソッド内で`search_tool`を直接呼び出します。
```python
# src/agent.py
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator
from .tools import search_tool

class DeterministicSearchAgent(BaseAgent):
    """受け取ったクエリを使い、必ずsearch_toolを呼び出すエージェント。"""
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        query_to_search = ctx.session.state.get("query")
        if not query_to_search:
            yield Event(author=self.name, content={"error": "検索クエリが見つかりません。"})
            return

        # LLMの判断を介さず、ツールを直接呼び出す
        search_results = search_tool(query=query_to_search)

        # 結果を後続のエージェントが使えるようにstateに保存
        ctx.session.state["search_results"] = search_results
        
        yield Event(
            author=self.name,
            content={"status": "success", "message": f"検索を実行し、{len(search_results)}件の結果をstateに保存しました。"}
        )
```

#### ステップ3: ワークフローに組み込む

作成したカスタムエージェントを`SequentialAgent`に組み込むことで、処理の順序を保証します。
```python
# src/workflow.py
from google.adk.agents import SequentialAgent, LlmAgent
from .agent import DeterministicSearchAgent

# 決定論的な検索エージェント
deterministic_search = DeterministicSearchAgent(name="DeterministicSearch")

# 検索結果を要約するLLMエージェント
summarizer_agent = LlmAgent(
    name="Summarizer",
    model="gemini-1.5-flash",
    instruction="提供された検索結果を要約してください: {search_results}"
)

# 必ず「検索」→「要約」の順で実行されるワークフロー
workflow = SequentialAgent(
    name="DeterministicSearchWorkflow",
    sub_agents=[
        deterministic_search,
        summarizer_agent
    ]
)
```
このように、LLMの自律的な判断（非決定論）と開発者による厳密な制御（決定論）を組み合わせることで、柔軟かつ信頼性の高いエージェントアプリケーションを構築できます。
