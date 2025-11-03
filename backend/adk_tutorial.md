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

### 5.3. 会話履歴の管理
セッションベースで会話履歴を管理する簡単な例です。本番環境では、Redisやデータベースなど、より永続的なストレージを検討してください。

```python
# src/app.py
from collections import defaultdict

# セッションごとの会話履歴を保持するシンプルなインメモリ辞書
chat_sessions = defaultdict(list)

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or "default_session"

    # 過去の履歴を取得
    history = chat_sessions[session_id]
    chat_history_str = "\n".join(f"User: {h['q']}\nAI: {h['a']}" for h in history)

    workflow_input = {
        "query": request.query,
        "chat_history": chat_history_str,
    }

    result = rag_workflow.execute(workflow_input)
    final_answer = result.get("final_answer")

    # 今回のやり取りを履歴に追加
    history.append({"q": request.query, "a": final_answer})

    return {"answer": final_answer, "session_id": session_id}
```

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
