# Agentic RAG API 仕様書

## 概要

FastAPIを利用して構築する、AgenticなRAG (Retrieval-Augmented Generation) APIの仕様。


## 開発ルール

- 重要な実装前に、このファイルに仕様を追記してください
- 仕様と実装が一貫しているか確認してください
- 実装完了後は必ず `tests/` にunittestを実装してください
- 実装完了後は必ず `uv run pytest` コマンドでテスト実行してください


## APIエンドポイント

### 1. チャットエンドポイント

- **エンドポイント:** `POST /chat`
- **機能:** ユーザーからの質問を受け取り、RAGを実行して回答を生成します。会話履歴を考慮した応答も可能です。
- **リクエスト:**
    - `query`: ユーザーからの質問文字列
    - `session_id` (任意): 会話セッションを識別するためのID
- **レスポンス:**
    - `answer`: 生成された回答
    - `sources`: 回答の根拠となった情報源（ドキュメント名、抜粋など）のリスト

### 2. 検索エンドポイント

- **エンドポイント:** `GET /search`
    - **機能:** ユーザーからのクエリに基づいて単純なキーワード検索を実行し、結果を直接返却します。
    - **リクエスト:**
        - `query`: ユーザーの検索クエリ文字列
    - **レスポンス:**
        - `results`: 検索結果のリスト（例: ドキュメントチャンクの内容とメタデータ）

### 3. ドキュメント管理エンドポイント

RAGの知識源となるドキュメントを管理します。

- **エンドポイント:** `POST /documents`
    - **機能:** 新しいドキュメントをアップロードし、ベクトル化して保存します。
    - **リクエスト:** ファイルデータ or ドキュメントのテキスト
- **エンドポイント:** `GET /documents`
    - **機能:** 登録されているドキュメントの一覧を取得します。
- **エンドポイント:** `DELETE /documents/{document_id}`
    - **機能:** 指定されたドキュメントを削除します。

### 4. ドキュメントのインデックス作成の仕様

- **ドキュメントのテキスト化**:
  - `Markitdown`ライブラリを利用して、アップロードされたドキュメントをMarkdown形式のテキストに変換します。
  - 対応ファイル形式: PDF (`.pdf`), Word (`.docx`), PowerPoint (`.pptx`)
- **チャンク分割**:
  - 抽出したテキストを、最大2048文字（`CHUNK_SIZE`）のチャンクに分割します。これはElasticsearchの`mappings`における`content`フィールドに格納されます。
- **コンテキストの追加**:
  - 各チャンクに対して、その周辺の最大5チャンク分のテキストを`full_text`として追加で格納します。これにより、検索結果の文脈理解を助けます。
- **ベクトル埋め込み**:
  - `content`フィールドの内容は、ベクトル化（Embedding）を行い、セマンティック検索を可能にします。

- **Elasticsearch Mappings**:
  - `file_id`: ドキュメントのファイル名のハッシュ文字列(`text`)
  - `chunk_id`: 何番目のチャンクか (`str(int)`)
  - `filename`: ドキュメントのファイル名 (`text`)
  - `content`: 分割されたチャンクのテキスト (`text`)
  - `full_text`: 周辺のチャンクを含むコンテキストテキスト (`text`)
  - `content_vector`: `content`のベクトル表現 (`dense_vector`)
    - `dims`: `EMBEDDING_DIM` (設定値)
    - `index`: `True`
    - `similarity`: `cosine`


## 検索エンジンの仕様 (Retriever)

本APIにおける検索エンジン（Retriever）は、ユーザーのクエリに対して、関連性の高い情報を知識ベースから効率的に取得する役割を担います。AgenticなRAGの要件を満たすため、柔軟な検索戦略と知識ベースの管理を考慮します。

### 1. 目的

*   ユーザーの質問に関連するドキュメントチャンクや情報を知識ベースから取得し、LLMへの入力として提供する。
*   取得した情報の信頼性や関連性を評価し、必要に応じて検索戦略を調整する。

### 2. 主要機能

*   **ドキュメントの取り込みと前処理:**
    *   アップロードされたドキュメントの解析、チャンク分割。
    *   各チャンクのベクトル埋め込み（Embedding）生成。（初期バージョンではベクトル埋め込みは行いません）
    *   チャンクとメタデータの知識ベースへの保存。
*   **クエリ処理:**
    *   ユーザーのクエリのベクトル埋め込み生成。（初期バージョンではベクトル埋め込みは行いません）
    *   必要に応じて、クエリの書き換えや拡張。
*   **関連情報検索:**
    *   クエリと知識ベース内のチャンクとの関連度を計算し、上位N件のチャンクを取得。
    *   セッション履歴やコンテキストを考慮した検索。
*   **検索結果の後処理:**
    *   取得したチャンクのランキング、フィルタリング、重複排除。
    *   LLMへの入力に適した形式での情報提供。


## AgenticRAG戦略

Google Agent Development Kit (`adk`) を利用してAgentワークフローを実装します。このワークフローを管理するメインエージェントは`AdvancedRAGAgent`として実装します。


### **ステップ 1: クエリ拡張**
ユーザーの「元の質問」「これまでの会話履歴」を `query_rewriter_agent` に渡し、「検索クエリ」を生成します。

### **ステップ 2: ElasticSearchからのハイブリッド検索**
「検索クエリ」でElastic Searchからハイブリッド検索を行い、上位5件の検索結果(「検索結果1-5」)を取得します。

### **ステップ 3: 個別回答の生成**
「元の質問」「これまでの会話履歴」と「ぞれぞれの検索結果」を 並列で`answer_agent` に渡し、「回答案1」-「回答案5」を生成します。すなわち`answer_agent`は5並列で回答を生成します。

### **ステップ 4: 最終回答の生成**
「元の質問」「これまでの会話履歴」と「回答案1」-「回答案5」を`synthesizer_agent`に渡します。`synthesizer_agent`は各回答案から情報を取捨選択し、より洗練された「最終回答」を生成します。その際、**どの情報源（検索結果）から引用したかを明記**します。また、**回答案の中に矛盾する情報が含まれる場合は、その点を考慮して最適な回答を構築**します。

### **ステップ 5: プロセスの完了**
「最終回答」をユーザーへ返却しプロセスを終了します。

### **補足：適切な検索結果が得られなかった場合**
ステップ2でElasticSearchから適切な検索結果が得られなかった場合、その旨をユーザーに伝えた上で、LLMが持つ一般的な知識に基づいて回答を生成します。

### そのほかの参考資料
会話履歴:
stateに突っ込もう

並列化
- https://google.github.io/adk-docs/agents/workflow-agents/parallel-agents/#full-example-parallel-web-research

outputを番号指定の複数にしてパラレルって感じかな
state["res1"] = "検索結果1"

"answer1"...

受け渡しは以下で
https://google.github.io/adk-docs/agents/custom-agents/#implementing-custom-logic

参考文献を受け取りたい
- artifactを使えばいい

プロンプトは以下を参考にできる
https://github.com/google/adk-samples/blob/5401edac5a51fd3367377d62d19d1f0fa23c407b/python/agents/RAG/rag/prompts.py

## 開発環境セットアップ

### Elasticsearchの起動

本プロジェクトでは、知識ベースとしてElasticsearchを使用します。開発環境ではDockerコンテナとしてローカルで起動することを推奨します。
`kuromoji`プラグインを導入するため、`Dockerfile`を使用してカスタムイメージをビルドします。

1. **Dockerイメージのビルド**

   `backend`ディレクトリに移動し、以下のコマンドを実行してDockerイメージをビルドします。

   ```bash
   docker build -t elasticsearch-kuromoji -f docker/Dockerfile .
   ```

2. **Elasticsearchコンテナの起動**

   ビルドしたイメージを使用して、Elasticsearchコンテナを起動します。

   ```bash
   docker run -d
     --name elasticsearch
     -p 9200:9200 -p 9300:9300
     -e "discovery.type=single-node"
     -e "xpack.security.enabled=false"
     -e "xpack.security.http.ssl.enabled=false"
     -e "xpack.security.enrollment.enabled=false"
     elasticsearch-kuromoji:latest
   ```

コンテナが起動したら、`http://localhost:9200` でElasticsearchにアクセスできることを確認してください。


### Backend APIの起動方法

開発サーバーを起動するには、プロジェクトのルートディレクトリで以下のコマンドを実行します。

```bash
uv run uvicorn src.app:app --reload
```

サーバーは `http://127.0.0.1:8000` で利用可能になります。
変更が保存されると自動的にリロードされます。

### テスト実行

```
uv run pytest
```


### Elastic Searchのインデックス確認

```
curl "http://localhost:9200/_cat/indices?v"
```
