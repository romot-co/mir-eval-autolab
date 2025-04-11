# MCP サーバー & AI 自動改善システム (AutoImprover) 詳細 README

## 1. 概要

このドキュメントは、MIRアルゴリズムの評価・改善プラットフォームにおける **MCP (Machine Communication Protocol) サーバー** コンポーネントと、それを利用するクライアント（**統合CLI `mirai`**）の詳細について説明します。

MCPサーバーは、プロジェクトのコア機能（MIRアルゴリズム評価、データ処理など）やAI連携機能（LLM呼び出し）へのアクセスを提供する**インターフェース**としての役割を担います。MCP標準に準拠することで、AIエージェント（例: Claude Desktop）や他の対応アプリケーションとの連携を容易にします。

**注意:** AI駆動の自動改善機能 (`mirai improve start` や関連するLLMツール) は実験的なものであり、安定性や結果の品質は保証されません。

**対象読者:**
*   MCPサーバーを運用・カスタマイズする開発者
*   AI自動改善システムの詳細な仕組みや使い方を知りたいユーザー
*   MCPクライアントを開発する開発者

**前提:**
*   基本的なPython環境とプロジェクトのセットアップが完了していること。セットアップ方法は [`README.md`](README.md) を参照してください。

## 2. MCPサーバーの起動と運用

### 2.1. 起動方法

仮想環境を有効化した後、以下のコマンドでMCPサーバーを起動します。

```bash
python -m src.cli.mcp_server
```

デフォルトでは `http://localhost:5002` でリクエストを受け付けます。ポート番号などは `config.yaml` または環境変数で変更可能です。

**オプション:**
*   `--config`: 設定ファイルのパスを指定 (デフォルト: `config.yaml`)
*   `--host`: バインドするホストアドレス (デフォルト: `127.0.0.1`)
*   `--port`: 使用するポート番号 (デフォルト: `5002`)
*   `--log-level`: ログレベル (デフォルト: `INFO`)
*   `--reload`: (開発用) コード変更時にサーバーを自動リロード (Uvicornオプション)

```bash
# 例: 異なるポートとデバッグログレベルで起動
python -m src.cli.mcp_server --port 8000 --log-level DEBUG --reload
```

### 2.2. 停止方法

サーバーを起動したターミナルで `Ctrl+C` を押して停止します。

### 2.3. ログ確認

サーバーログは標準出力に表示されます。
ログレベルは起動時オプション (`--log-level`) または設定ファイル (`config.yaml` の `server.log_level`) で変更できます。

### 2.4. データベース

サーバーの状態（ジョブ、セッション履歴など）はSQLiteデータベースに保存されます。
デフォルトのパスは `./.mcp_server_data/db/mcp_server_state.db` です (`config.yaml` の `paths.db` で変更可能)。

## 3. システムアーキテクチャとフロー

### 3.1. コンポーネント図

```mermaid
graph LR
    subgraph Core MIR Logic [src/* (Independent)]
        Detectors[src/detectors]
        Evaluation[src/evaluation]
        Utils[src/utils]
        DataLoader[src/data_loader.py]
    end

    subgraph MCP Server
        MCPEntry(src/cli/mcp_server.py) --- ServerLogic[src/mcp_server_logic]
        ServerLogic --- DB[(mcp_server_state.db)]
        ServerLogic -- Uses --> Core MIR Logic
        ServerLogic -- Interacts --> LLM[LLM API (Claude/OpenAI)]
    end

    subgraph MCP Clients
        MiraiCLI(src/cli/mirai.py) -- Uses Core (Standalone) --> Core MIR Logic
        MiraiCLI -- MCP (Server Mode) --> MCPEntry
        ClaudeDesktop[Claude Desktop (Host)] -- MCP --> MCPEntry
    end

    style MiraiCLI fill:#ccf,stroke:#333,stroke-width:2px
    style ClaudeDesktop fill:#ccf,stroke:#333,stroke-width:2px

    style MCPEntry fill:#f9f,stroke:#333,stroke-width:2px
    style ServerLogic fill:#f9f,stroke:#333,stroke-width:2px
    style DB fill:#ccc,stroke:#333,stroke-width:2px

    style Core MIR Logic fill:#9cf,stroke:#333,stroke-width:2px
```

### 3.2. サーバー経由の評価フロー (`mirai evaluate run --server`)

1.  **(ユーザー)** `mirai evaluate run --server ...` を実行。
2.  **(クライアント `mirai`)** MCPサーバーに `run_evaluation` ツールを呼び出し。
3.  **(サーバー)** リクエストを受け付け、非同期評価タスクをジョブキューに追加。ジョブIDを生成・登録し、クライアントに返す。
4.  **(クライアント `mirai`)** ジョブIDを使って `get_job_status` を定期的に呼び出し、完了をポーリング。
5.  **(サーバー: Worker)** ジョブワーカーがキューから評価タスクを取得し実行。DBステータスを `running` に更新。
6.  **(サーバー: Worker)** コア評価関数を別スレッドで実行。
7.  **(サーバー: Worker)** 評価完了後、結果を整形し、DBのジョブステータスを `completed` に更新、結果を保存。
8.  **(クライアント `mirai`)** ポーリングにより `completed` ステータスを検知。結果を取得し表示。

### 3.3. (実験的) 自動改善フロー (`mirai improve start`)

**重要な概念:** 自動改善ループの状態（現在のサイクル数、ベストスコア、停滞カウントなど）は**サーバー側で一元管理**されます (`sessions` テーブルの `cycle_state` フィールドなど)。`mirai improve start` コマンドは、このサーバー主導のループを開始/再開し、進捗を監視・報告する役割を担います。

1.  **(ユーザー)** `mirai improve start --server ...` を実行。
2.  **(クライアント `mirai`)** サーバーの `start_session` ツールを呼び出し、改善セッションを開始または再開。
3.  **(クライアント `mirai`)** セッションIDを受け取る。
4.  **(クライアント: ループ開始)**
5.  **(クライアント `mirai`)** サーバーの `get_session_info` を呼び出し、現在のセッション状態（ステータス、サイクル数など）を取得。
6.  **(クライアント `mirai`)** セッション状態に基づき、ループを継続するか終了するか判断（最大サイクル数、タイムアウト、手動停止など）。
7.  **(クライアント `mirai`)** ループ継続の場合、サーバーの **`advance_improvement_cycle`** ツール（※将来的な実装想定）を呼び出す。このツールがサーバー側で「評価→分析→提案→実行」の1サイクルを管理するジョブを開始する。
8.  **(クライアント `mirai`)** `advance_improvement_cycle` ツールから返されたジョブIDを使い、`poll_job_status` でサイクルジョブの完了を待機。
9.  **(クライアント: ループ継続)** 完了後、ステップ5に戻る。

**現在の実装:**
現状、`advance_improvement_cycle` のようなサーバー主導の統合ツールは実装されていません。代わりに、クライアント (`mirai improve start` のループ内) が `run_evaluation`, `analyze_evaluation`, `improve_code`, `save_code` などの**個別のツール**を順次呼び出し、結果をハンドリングして次のステップに進む形式になっています。このため、ループ制御の一部がクライアント側に存在します。

将来的には、ループ制御をよりサーバー側に寄せ、クライアントはジョブの起動と監視に徹するアーキテクチャを目指すことが考えられます。

### 3.4. 非同期アーキテクチャ (MCPサーバー)
MCPサーバーは `asyncio` をベースに構築されています。

*   **メインプロセス:** FastAPI/Uvicorn がHTTPリクエストを非同期に処理。
*   **ツール実行:** MCPツール関数 (`async def`) が呼び出される。
*   **ジョブ投入:** 時間のかかる処理は `job_manager.start_async_job` 経由で `asyncio.Queue` に投入され、即座にジョブIDが返される。
*   **ジョブワーカー:** 複数の `job_manager.job_worker` タスクがジョブキューを監視し、タスクを取り出して実行。
*   **同期処理の委譲:** コア機能の同期関数やブロッキングI/Oは `loop.run_in_executor(None, ...)` で別スレッドで実行。
*   **DBアクセス:** `aiosqlite` を使用して非同期でDBにアクセス (`db_utils.py`)。

## 4. MCP サーバーの詳細アーキテクチャ

### 4.1. サーバー中心の状態管理
MCPサーバーは、ジョブの状態、改善セッションの履歴や進捗状況などを、SQLiteデータベース (`.mcp_server_data/db/mcp_server_state.db`) に一元管理します。

*   **Stateless クライアント:** `mirai` CLI などのクライアントは、原則として状態を持ちません。
*   **状態更新:** 状態変更は特定のツール経由でサーバーに依頼され、サーバー側でアトミックに DB が更新されます。
*   **非同期ジョブ:** 時間のかかる処理は非同期ジョブとして実行されます。

**主要なデータベーステーブル:**
*   **`sessions`:** 改善セッション情報を管理 (ID, status, history, config, cycle_state, metrics, best_code_version など)。
*   **`jobs`:** 非同期ジョブ情報を管理 (ID, session_id, tool_name, status, timestamps, result, error_details など)。

詳細は `src/mcp_server_logic/db_utils.py` の `initialize_db` 関数を参照してください。

### 4.2. 非同期処理とジョブ管理
`job_manager` モジュールが非同期ジョブを管理します (`asyncio.Queue` と `job_worker` タスクを使用)。

### 4.3. スキーマ定義 (`schemas.py`)
Pydanticモデル (`schemas.py`) がツール間のデータ交換やDB構造の定義に利用されます。

### 4.4. 設定管理 (`config.yaml`, 環境変数)
サーバーの挙動は `config.yaml` で設定され、デフォルト値、YAMLファイル、環境変数 (`MCP__<SECTION>__<KEY>`) の順でマージされます。
パス設定 (`paths.*`) は、環境変数 `MIREX_WORKSPACE` (デフォルト: プロジェクトルート) を基準とする絶対パスに解決されます。環境変数 `MIREX_OUTPUT_BASE` で出力ディレクトリの基準パスを指定することも可能です。
詳細は `src/mcp_server_logic/core.py` の `load_config` 関数および `src/utils/path_utils.py` を参照してください。

## 5. MCP サーバーツール一覧

MCPサーバーは以下のツールを提供します (`src/mcp_server_logic/schemas.py` で定義されたスキーマを使用)。ツール関数は対応するモジュール (`session_manager.py`, `evaluation_tools.py` など) に実装されています。

**セッション管理 (`session_manager.py`)**
*   `start_session`: 新しい改善セッションを開始。入力: `StartSessionInput`, 出力: `SessionInfoResponse`。
*   `get_session_info`: 指定セッションIDの情報を取得。入力: `GetSessionInfoInput`, 出力: `SessionInfoResponse`。
*   **(内部用) `add_session_history`:** 履歴イベントを追加。
*   **(将来実装) `advance_improvement_cycle`:** 改善サイクルを1ステップ進める非同期ジョブを開始。入力: `session_id`, 出力: `JobStartResponse`。

**ジョブ管理 (`job_manager.py`)**
*   `get_job_status`: 指定ジョブIDの状態と結果を取得。入力: `GetJobStatusInput`, 出力: `JobInfo`。

**評価ツール (`evaluation_tools.py`)**
*   `run_evaluation`: 評価を実行（非同期ジョブ）。入力: `RunEvaluationInput`, 出力: `JobStartResponse`, ジョブ結果: `EvaluationResultData`。
*   `run_grid_search`: グリッドサーチを実行（非同期ジョブ）。入力: `RunGridSearchInput`, 出力: `JobStartResponse`, ジョブ結果: `GridSearchResultData`。

**コードツール (`code_tools.py`)**
*   `get_code`: コードを取得（非同期ジョブ）。入力: `GetCodeInput`, 出力: `JobStartResponse`, ジョブ結果: `GetCodeResultData`。
*   `save_code`: コードを保存（非同期ジョブ）。入力: `SaveCodeInput`, 出力: `JobStartResponse`, ジョブ結果: `CodeSaveResultData`。

**LLMツール (`llm_tools.py`)**
*   `improve_code`: コード改善提案を生成（非同期ジョブ）。入力: `ImproveCodeInput`, 出力: `JobStartResponse`, ジョブ結果: `ImproveCodeResultData`。
*   `suggest_parameters`: パラメータ調整案を生成（非同期ジョブ）。入力: `SuggestParametersInput`, 出力: `JobStartResponse`, ジョブ結果: `SuggestParametersResultData`。
*   `analyze_evaluation_results`: 評価結果を分析（非同期ジョブ）。入力: `AnalyzeEvaluationInput`, 出力: `JobStartResponse`, ジョブ結果: `AnalyzeEvaluationResultData`。
*   `generate_hypotheses`: 改善仮説を生成（非同期ジョブ）。入力: `GenerateHypothesesInput`, 出力: `JobStartResponse`, ジョブ結果: `GenerateHypothesesResultData`。
*   `suggest_exploration_strategy`: 次の行動戦略を提案（非同期ジョブ）。入力: `SuggestExplorationStrategyInput`, 出力: `JobStartResponse`, ジョブ結果: `SuggestStrategyResultData`。

**注意:** 各ツールの具体的な入出力スキーマは `src/mcp_server_logic/schemas.py` を参照してください。

## 6. (実験的) AI自動改善システム (AutoImprover)

### 6.1. 目的

LLM（大規模言語モデル）を活用し、MIRアルゴリズムの評価結果に基づいて、コードやパラメータの改善案を自動生成し、改善サイクルを回すことを目的とした実験的なシステムです。

### 6.2. 使い方 (`mirai improve start`)

1.  **MCPサーバーを起動します。**
2.  **LLM APIキーを設定します。** (`.env` ファイルまたは環境変数 `MCP__LLM__API_KEY` など)。
3.  **改善対象とデータセットを指定して `mirai improve start` を実行します。**
    ```bash
    python -m src.cli.mirai improve start \\
      --server http://localhost:5002 \\
      --detector YourDetectorClassName \\
      --dataset your_dataset_name \\
      --goal "Improve F1-score for note detection" \\
      --max-cycles 10
    ```
    *   `--goal`: (オプション) LLMに伝える改善目標。
    *   `--session-id`: (オプション) 既存のセッションを再開。
    *   `--max-cycles`: (オプション) サーバー側での最大サイクル数。
    *   `--threshold`, `--max-stagnation`: (オプション) 改善停止条件。
    *   詳細は `python -m src.cli.mirai improve start --help` を参照。
4.  **クライアント (`mirai`) がサーバーの状態を監視し、ループを進めます。**
    ログに進捗が表示されます。`Ctrl+C` でクライアント側の監視を停止できます（サーバー側のセッションは停止しません）。
5.  **セッションの状態確認・停止:**
    ```bash
    # 状態確認
    python -m src.cli.mirai improve status --server http://localhost:5002 <session_id>
    # 手動停止
    python -m src.cli.mirai improve stop --server http://localhost:5002 <session_id>
    ```

### 6.3. 内部ロジック (改善サイクル)

`mirai improve start` がサーバーと連携して実行する主なステップは以下の通りです（現状はクライアント主導でツールを順次呼び出す形式）。

1.  **初期評価:** 対象アルゴリズムのベースライン性能を評価 (`run_evaluation`)。
2.  **分析:** 評価結果をLLMで分析 (`analyze_evaluation_results`)。
3.  **戦略提案:** 分析結果に基づき、次に取るべきアクション（コード改善、パラメータ調整など）をLLMが提案 (`suggest_exploration_strategy`)。
4.  **アクション実行:**
    *   **コード改善の場合:** `improve_code` でLLMにコード改善案を生成させ、`save_code` で新バージョンとして保存。
    *   **パラメータ調整の場合:** `suggest_parameters` でLLMにパラメータ範囲を提案させ、`run_grid_search` を実行（現状はLLMによる直接的なパラメータ値提案ではなく、探索範囲の提案）。
5.  **再評価:** 改善/調整後のアルゴリズムを評価 (`run_evaluation`)。
6.  **履歴記録:** 各ステップの結果を `add_session_history` でサーバーに記録。
7.  **状態更新:** サーバー側で `cycle_state` (サイクル数、停滞カウント、ベストスコアなど) を更新。
8.  ステップ2に戻り、ループを繰り返す。

**停止条件:**
*   最大サイクル数に到達 (`--max-cycles`)。
*   一定期間改善が見られない (`--max-stagnation`, `--threshold`)。
*   セッションタイムアウト (`config.yaml` の `server.session_timeout_seconds`)。
*   手動停止 (`mirai improve stop`)。
*   致命的なエラー発生。

### 6.4. プロンプトエンジニアリング

LLMツール (`llm_tools.py`) では、`prompts/` ディレクトリにあるJinja2テンプレートを使用してLLMへのプロンプトを生成しています。
改善の質を高めるには、これらのプロンプトの調整が重要になります。

### 6.5. 制限事項と今後の課題

*   **実験的機能:** 安定性や結果の品質は保証されません。
*   **プロンプト依存:** LLMの性能はプロンプトに大きく依存します。
*   **探索戦略:** 現在の戦略提案 (`suggest_exploration_strategy`) は比較的単純です。
*   **Grid Search連携:** LLMによるパラメータ提案とGrid Searchの連携が限定的です。
*   **エラーハンドリング:** 複雑なエラーケースへの対応が不十分な場合があります。

## 7. Claude Desktop 連携 (オプション)

Claude DesktopアプリケーションをMCPクライアントとして使用するための設定が可能です。詳細は `src/cli/setup_claude_integration.py --help` を参照してください。

## 8. トラブルシューティング

*   **サーバーが起動しない:** ポートが既に使用されていないか、依存関係が正しくインストールされているか確認してください。
*   **クライアントが接続できない:** サーバーURLとポート番号が正しいか、サーバーが起動しているか確認してください。
*   **ジョブが失敗する:** サーバーログ (`stdout`) を確認し、エラーメッセージやトレースバックを調査してください。入力データや設定に問題がある可能性があります。
*   **改善ループが意図通りに動作しない:** サーバーログ、セッション履歴 (`get_session_info`)、LLMツールのプロンプト (`prompts/`) を確認してください。