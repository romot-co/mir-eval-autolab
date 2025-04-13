# MCP サーバー & AI 自動改善システム (AutoImprover) 詳細 README

## 1. 概要

このドキュメントは、MIRアルゴリズムの評価・改善プラットフォームにおける **MCP (Machine Communication Protocol) サーバー** コンポーネントと、それを利用するクライアント（**統合CLI `mirai`**）の詳細について説明します。

MCPサーバーは、プロジェクトのコア機能（MIRアルゴリズム評価、データ処理など）やAI連携機能（プロンプト生成）へのアクセスを提供する**インターフェース**としての役割を担います。MCP標準に準拠することで、AIエージェント（例: Claude Desktop）や他の対応アプリケーションとの連携を容易にします。

**注意:** AI駆動の自動改善機能 (`mirai improve start`) は実験的なもので、安定性や結果の品質は保証されません。

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
*   `--host`: バインドするホストアドレス (デフォルト: `0.0.0.0`)
*   `--port`: 使用するポート番号 (デフォルト: `5002`、環境変数 `MCP_PORT` でも可)
*   `--log-level`: ログレベル (デフォルト: `INFO`、環境変数 `MCP_LOG_LEVEL` でも可)
*   `--reload`: (開発用) コード変更時にサーバーを自動リロード (Uvicornオプション)

```bash
# 例: 異なるポートとデバッグログレベルで起動
python -m src.cli.mcp_server --port 8000 --log-level DEBUG --reload
```

### 2.2. 停止方法

サーバーを起動したターミナルで `Ctrl+C` を押して停止します。

### 2.3. ログ確認

サーバーログは標準出力に表示されます。
ログレベルは起動時オプション (`--log-level`) または設定ファイル (`config.yaml` の `server.log_level`、環境変数 `MCP_LOG_LEVEL`) で変更できます。

### 2.4. データベースとワークスペース

*   **データベース:** サーバーの状態（ジョブ、セッション履歴など）はSQLiteデータベースに保存されます。デフォルトのパスは `./.mcp_server_data/db/mcp_server_state.db` です。パスは環境変数 `MIREX_WORKSPACE` または `config.yaml` の `paths.db` で変更可能です（`path_utils.py` により解決）。
*   **ワークスペース:** AIが生成したコード (`improved_versions`) やその他の作業ファイルはワークスペースディレクトリ (デフォルト: `./.mcp_server_data`) 以下に保存されます。これも環境変数 `MIREX_WORKSPACE` で変更可能です。
*   **出力:** 評価結果やグリッドサーチ結果などは出力ベースディレクトリ (デフォルト: `./output`) 以下に保存されます。環境変数 `MIREX_OUTPUT_BASE` で変更可能です。

## 3. システムアーキテクチャとフロー

### 3.1. コンポーネント図

```mermaid
graph LR
    subgraph Core MIR Logic [src/* (Independent)]
        Detectors[src/detectors]
        Evaluation[src/evaluation]
        Utils[src/utils]
        # DataLoader removed, handled within tools
    end

    subgraph MCP Server [mcp_server.py + mcp_server_logic/*]
        MCPEntry(src/cli/mcp_server.py) -- Manages --> ServerLogic[mcp_server_logic Modules]
        ServerLogic -- Interacts --> JobMgr[job_manager: Async Jobs]
        ServerLogic -- Interacts --> SessionMgr[session_manager: State & History]
        SessionMgr -- Uses --> DBUtils[db_utils: SQLite Access]
        JobMgr -- Uses --> DBUtils
        DBUtils -- Accesses --> DB[(mcp_server_state.db)]
        ServerLogic -- Uses --> PromptTools[llm_tools: Prompt Generation]
        ServerLogic -- Uses --> EvalTools[evaluation_tools: Evaluation Core]
        ServerLogic -- Uses --> CodeTools[code_tools: Code & Git]
        EvalTools -- Uses --> Core MIR Logic
    end

    subgraph MCP Clients
        MiraiCLI(src/cli/mirai.py) -- Standalone Uses --> Core MIR Logic
        MiraiCLI -- Server Mode --> MCPEntry
        MiraiCLI -- Uses --> LLMClient[src/cli/llm_client.py]
        LLMClient -- Interacts --> LLM_API[LLM API (Claude/OpenAI)]
        ClaudeDesktop[Claude Desktop (Host)] -- MCP --> MCPEntry
    end

    style MiraiCLI fill:#ccf,stroke:#333,stroke-width:2px
    style LLMClient fill:#ccf,stroke:#333,stroke-width:2px
    style ClaudeDesktop fill:#ccf,stroke:#333,stroke-width:2px

    style MCPEntry fill:#f9f,stroke:#333,stroke-width:2px
    style ServerLogic fill:#f9f,stroke:#333,stroke-width:2px
    style DB fill:#ccc,stroke:#333,stroke-width:2px

    style Core MIR Logic fill:#9cf,stroke:#333,stroke-width:2px
```

### 3.2. サーバー経由の評価フロー (`mirai evaluate run --server`)

1.  **(ユーザー)** `mirai evaluate run --server ...` を実行。
2.  **(クライアント `mirai`)** MCPサーバーに `run_evaluation` ツールを呼び出し。
3.  **(サーバー)** リクエストを受け付け、`evaluation_tools.run_evaluation_tool` が実行される。
4.  **(サーバー)** `job_manager.start_async_job` を呼び出し、非同期評価タスク (`evaluation_tools._run_evaluate_async`) をジョブキューに追加。ジョブIDを生成・登録し、クライアントに `JobStartResponse` を返す。
5.  **(クライアント `mirai`)** ジョブIDを使って `get_job_status` を定期的に呼び出し (`poll_job_status`)、完了をポーリング。
6.  **(サーバー: Worker)** `job_manager.job_worker` がキューから評価タスクを取得し実行。DBステータスを `running` に更新。
7.  **(サーバー: Worker)** `_run_evaluate_async` ラッパーが実行され、内部で `evaluation_runner.run_evaluation_core` (同期) を別スレッド (`run_in_executor`) で実行。
8.  **(サーバー: Worker)** 評価完了後、`_run_evaluate_async` が結果 (スキーマ `EvaluationResultData`) を整形。
9.  **(サーバー: Worker)** `job_worker` がDBのジョブステータスを `completed` に更新し、結果をJSONとして保存。セッション履歴 (`evaluation_complete`) も追加。
10. **(クライアント `mirai`)** ポーリングにより `completed` ステータスを検知。`get_job_status` から結果を取得し表示。

### 3.3. 自動改善フロー (`mirai improve start`) - 新アーキテクチャ

**重要な概念:** 自動改善ループの状態（サイクル数、ベストスコア、停滞カウント、最後に実行したアクション、分析結果、仮説など）は**サーバー側でセッションとして管理**されます（`sessions` テーブルの `cycle_state` フィールドなど）。`mirai improve start` コマンドは、**CLIホスト主導**で改善ループを制御し、LLM APIと直接通信して改善処理を行います。

1.  **(ユーザー)** `mirai improve start --server ...` を実行。
2.  **(クライアント `mirai`)** サーバーの `start_session` ツールを呼び出し、改善セッションを開始または再開。セッションID (`active_session_id`) を受け取る。
3.  **(クライアント: 初期化)** AnthropicなどのLLMクライアントを初期化（APIキーは環境変数から取得）。
4.  **(クライアント: ループ開始)** クライアント側で改善ループ (`while cycle_count < max_cycles`) を開始。
5.  **(クライアント `mirai`)** サーバーの `get_session_info` を呼び出し、現在のセッション状態（ステータス、サイクル数など）を取得・表示。
6.  **(クライアント `mirai`)** セッション状態が終了状態 (`completed`, `failed`, `stopped`, `timed_out`, `stalled`) であれば、クライアント側のループを終了。
7.  **(クライアント `mirai`)** 次のアクションを決定するため、サーバーから戦略提案プロンプトを取得 (`get_suggest_exploration_strategy_prompt`)し、LLMに送信して次のアクションを決定。
8.  **(クライアント `mirai`)** 決定されたアクションに応じた処理を実行:
    
    a. **評価分析 (analyze_evaluation):** 
       - プロンプト取得 (`get_analyze_evaluation_prompt`)
       - LLMに送信して分析結果を取得
       - 分析結果をセッション履歴に記録

    b. **仮説生成 (generate_hypotheses):**
       - プロンプト取得 (`get_generate_hypotheses_prompt`)  
       - LLMに送信して仮説を生成
       - 仮説をセッション履歴に記録

    c. **コード改善 (improve_code):**
       - コード取得 (`get_code`)
       - プロンプト取得 (`get_improve_code_prompt`)
       - LLMに送信して改善コードを取得
       - コードの差分をユーザーに表示し、確認を求める
       - 確認後、改善コードを保存 (`save_code`)
    
    d. **評価実行 (run_evaluation):**
       - サーバーの評価ツールを呼び出し
       - 完了を待機し、結果を表示
    
    e. **パラメータ最適化 (optimize_parameters):**
       - プロンプト取得 (`get_suggest_parameters_prompt`)
       - LLMに送信してパラメータ調整案を取得
       - グリッドサーチを実行 (`run_grid_search`)
       - 最適パラメータをコードに適用し保存

9.  **(クライアント `mirai`)** ループの次のサイクルへ進む前に、ユーザーに継続確認を求め (オプション)。
10. **(クライアント `mirai`)** 全サイクル完了後、最終的なセッション情報を取得して表示。

**メリット:**
- LLMAPIキー管理をクライアント側で行うため、サーバー側では機密情報を保持しない
- ユーザーによるコード変更確認など、より対話的な操作が可能
- サーバー側の実装をシンプル化（プロンプト生成のみ）
- マルチユーザー環境でも安全（各ユーザーが自身のAPIキーで操作）

### 3.4. 非同期アーキテクチャ (MCPサーバー)
MCPサーバーは `asyncio` をベースに構築されています。

*   **メインプロセス:** FastAPI/Uvicorn がHTTPリクエストを非同期に処理。
*   **ツール実行:** MCPツール関数 (`async def`) が呼び出される。
*   **ジョブ投入:** 時間のかかる処理 (評価、グリッドサーチ、プロンプト生成) は `job_manager.start_async_job` 経由で `asyncio.Queue` に投入され、即座にジョブIDが返される。
*   **ジョブワーカー:** 複数の `job_manager.job_worker` タスク (バックグラウンド実行) がジョブキューを監視し、タスクを取り出して非同期に実行 (`await task_coro_func(...)`)。
*   **同期処理の委譲:** コア機能の同期関数 (例: `evaluation_runner.run_evaluation_core`) やブロッキングI/Oは、ツールラッパー (`_run_*_async`) 内で `loop.run_in_executor(None, ...)` により別スレッドで実行される。
*   **DBアクセス:** `aiosqlite` を使用し、`db_utils.py` 内のヘルパー関数 (`db_execute_commit_async` 等) を介して非同期でDBにアクセス。書き込み時には `asyncio.Lock` による排他制御を行う。

## 4. MCP サーバーの詳細アーキテクチャ

### 4.1. サーバー中心の状態管理
MCPサーバーは、ジョブの状態、改善セッションの履歴や進捗状況などを、SQLiteデータベース (`.mcp_server_data/db/mcp_server_state.db`) に一元管理します。

*   **Stateless クライアント:** `mirai` CLI などのクライアントは、原則として状態を持ちません。セッションIDやジョブIDを保持するのみです。
*   **状態更新:** 状態変更は特定のツール経由でサーバーに依頼され、サーバー側 (主に `session_manager.add_session_history`) でアトミックに DB が更新されます。
*   **非同期ジョブ:** 時間のかかる処理は非同期ジョブとして実行されます。

**主要なデータベーステーブル:**
*   **`sessions`:** 改善セッション情報を管理 (ID, status, history (JSON List), config (JSON), cycle_state (JSON), current_metrics (JSON), best_metrics (JSON), best_code_version など)。
*   **`jobs`:** 非同期ジョブ情報を管理 (ID, session_id, tool_name, status, timestamps, result (JSON), error_details (JSON), task_args (JSON) など)。

詳細は `src/mcp_server_logic/db_utils.py` の `initialize_db` 関数を参照してください。

### 4.2. 非同期処理とジョブ管理
`job_manager` モジュールが非同期ジョブを管理します (`asyncio.Queue` と `job_worker` タスクを使用)。

### 4.3. スキーマ定義 (`schemas.py`)
Pydanticモデル (`schemas.py`) がツール間のデータ交換、DB構造の定義、履歴イベントの構造化に幅広く利用されます。

### 4.4. 設定管理 (`config.yaml`, 環境変数)
サーバーの挙動は `config.yaml` で設定され、デフォルト値、YAMLファイル、環境変数 (`MCP__<SECTION>__<KEY>`) の順でマージされます。
パス設定 (`paths.*`) は、環境変数 `MIREX_WORKSPACE` (デフォルト: プロジェクトルート/.mcp_server_data) や `MIREX_OUTPUT_BASE` (デフォルト: プロジェクトルート/output) を基準とする絶対パスに解決されます。
詳細は `src/mcp_server_logic/core.py` の `load_config` 関数および `src/utils/path_utils.py` を参照してください。

## 5. MCP サーバーツール一覧

MCPサーバーは以下のツールを提供します (`src/mcp_server_logic/schemas.py` で定義されたスキーマを使用)。

**セッション管理 (`session_manager.py`)**
*   `start_session`: 新しい改善セッションを開始。入力: `StartSessionInput`, 出力: `SessionInfoResponse` (同期処理、ジョブではない)。
*   `get_session_info`: 指定セッションIDの情報を取得。入力: `GetSessionInfoInput`, 出力: `SessionInfoResponse` (同期処理、ジョブではない)。
*   `add_session_history`: 履歴イベントを追加。入力: `AddSessionHistoryInput`, 出力: なし (同期処理、ジョブではない)。

**ジョブ管理 (`job_manager.py`)**
*   `get_job_status`: 指定ジョブIDの状態と結果を取得。入力: `GetJobStatusInput`, 出力: `JobInfo` (同期処理、ジョブではない)。

**評価ツール (`evaluation_tools.py`)**
*   `run_evaluation`: 評価を実行（非同期ジョブ）。入力: `RunEvaluationInput`, 出力: `JobStartResponse`, ジョブ結果: `EvaluationResultData`。
*   `run_grid_search`: グリッドサーチを実行（非同期ジョブ）。入力: `RunGridSearchInput`, 出力: `JobStartResponse`, ジョブ結果: `GridSearchResultData`。

**コードツール (`code_tools.py`)**
*   `get_code`: コードを取得（同期処理）。入力: `GetCodeInput`, 出力: `GetCodeResultData`。
*   `save_code`: コードを保存（非同期ジョブ、Git連携含む）。入力: `SaveCodeInput`, 出力: `JobStartResponse`, ジョブ結果: `CodeSaveResultData`。

**プロンプト生成ツール (`llm_tools.py`)**
*   `get_improve_code_prompt`: コード改善用プロンプトを生成（非同期ジョブ）。入力: `GetImproveCodePromptInput`, 出力: `JobStartResponse`, ジョブ結果: `PromptData`。
*   `get_suggest_parameters_prompt`: パラメータ提案用プロンプトを生成（非同期ジョブ）。入力: `GetSuggestParametersPromptInput`, 出力: `JobStartResponse`, ジョブ結果: `PromptData`。
*   `get_analyze_evaluation_prompt`: 評価分析用プロンプトを生成（非同期ジョブ）。入力: `GetAnalyzeEvaluationPromptInput`, 出力: `JobStartResponse`, ジョブ結果: `PromptData`。
*   `get_generate_hypotheses_prompt`: 仮説生成用プロンプトを生成（非同期ジョブ）。入力: `GetGenerateHypothesesPromptInput`, 出力: `JobStartResponse`, ジョブ結果: `PromptData`。
*   `get_suggest_exploration_strategy_prompt`: 戦略提案用プロンプトを生成（非同期ジョブ）。入力: `GetSuggestExplorationStrategyPromptInput`, 出力: `JobStartResponse`, ジョブ結果: `PromptData`。
*   `get_assess_improvement_prompt`: 改善評価用プロンプトを生成（非同期ジョブ）。入力: `GetAssessImprovementPromptInput`, 出力: `JobStartResponse`, ジョブ結果: `PromptData`。

## 6. AI自動改善システム (AutoImprover)

### 6.1. 目的

LLM（大規模言語モデル）を活用し、MIRアルゴリズムの評価結果に基づいて、改善仮説の生成、コードやパラメータの改善案の生成・実行、改善効果の評価といった一連のサイクルを自動化し、アルゴリズム性能を向上させることを目的としたシステムです。

### 6.2. 使い方 (`mirai improve start`)

1.  **MCPサーバーを起動します。** (`python -m src.cli.mcp_server`)
2.  **LLM APIキーを設定します。** (`.env` ファイルまたは環境変数 `ANTHROPIC_API_KEY` など)。
3.  **改善対象とデータセットを指定して `mirai improve start` を実行します。**
    ```bash
    python -m src.cli.mirai improve start \
      --server http://localhost:5002 \
      --detector YourDetectorClassName \
      --dataset your_dataset_name \
      --goal "Improve F1-score for note detection" \
      --max-cycles 10
    ```
    *   `--server`: MCPサーバーのURL (必須)。
    *   `--detector`: 改善対象の検出器クラス名 (必須)。
    *   `--dataset`: 評価に使用するデータセット名 (必須、`config.yaml` で定義)。
    *   `--goal`: (オプション) LLMに伝える改善目標。プロンプト生成に使用されます。
    *   `--session-id`: (オプション) 既存のセッションIDを指定して再開。
    *   `--max-cycles`: (オプション) 実行する最大サイクル数。
    *   `--llm-model`: (オプション) 使用するLLMモデル（デフォルト: claude-3-opus-20240229）。
    *   `--api-key`: (オプション) LLM APIキー（省略時は環境変数から取得）。
    *   `--always-confirm`: (オプション) コード変更前に常に確認するかどうか。
    *   詳細は `python -m src.cli.mirai improve start --help` を参照。
4.  **改善ループの進行:**
    *   CLIが非同期にプロンプト生成リクエストをサーバーに送信し、生成されたプロンプトを使用してLLMに直接問い合わせます。
    *   各アクション（評価分析、仮説生成、コード改善など）の進捗がCLI上に表示されます。
    *   コード変更時にはユーザー確認が要求されます（`--always-confirm=false` で無効化可能）。
    *   `Ctrl+C` で改善ループを停止できます。

### 6.3. アーキテクチャと内部フロー

自動改善システムは、CLIホスト主導のアーキテクチャを採用しています：

1. **セッション管理:** サーバー側でセッション状態（履歴、メトリクス、cycle_state）を管理します。
2. **プロンプト生成:** サーバーはプロンプトを生成し、CLIがそれを受け取ります。
3. **LLM通信:** CLIホストが直接LLM APIと通信し、APIキーを管理します。
4. **アクション実行:** CLIはLLMの応答に基づいてアクションを実行し、結果をサーバーに記録します。

**主なアクションとフロー:**
1. **戦略決定:** 現在の状態に基づき、次のアクションを決定します。
2. **評価分析:** 評価結果をLLMで分析し、強み/弱み/傾向を特定します。
3. **仮説生成:** 問題点や改善のための仮説をLLMに生成させます。
4. **コード改善:** 仮説に基づいて、LLMにコード改善案を提案させます。
5. **パラメータ最適化:** 性能向上のためのパラメータ調整をLLMに提案させ、グリッドサーチで検証します。
6. **評価実行:** 改善されたコードやパラメータの性能を評価します。

**停止条件:**
*   最大サイクル数に到達 (`--max-cycles`)。
*   改善停滞回数が上限に到達。
*   戦略提案により `stop` アクションが選択された。
*   ユーザーによる手動停止（確認プロンプトでのキャンセル）。

### 6.4. プロンプトエンジニアリング

サーバー側の `llm_tools.py` では、`prompts/` ディレクトリにあるJinja2テンプレートを使用してLLMへのプロンプトを生成しています。改善の質を高めるには、これらのプロンプト（`analyze_evaluation_results.j2`, `improve_code.j2`, `suggest_parameters.j2`, `generate_hypotheses.j2`, `suggest_exploration_strategy.j2`, `assess_improvement.j2` など）の調整が重要になります。

### 6.5. 制限事項と今後の課題

*   **実験的機能:** 安定性や結果の品質は保証されません。特にLLMの応答は予測不可能な場合があります。
*   **プロンプト依存:** LLMの性能はプロンプトに大きく依存します。継続的な改善が必要です。
*   **テストカバレッジ:** 現在、自動改善システムのテストカバレッジは限定的です。
*   **パラメータ最適化:** 現在はLLM提案+GridSearchですが、より高度な最適化手法（Bayesian Optimizationなど）の統合も考えられます。
*   **エラー回復:** エラー発生時のより洗練された回復メカニズムの実装が課題です。
*   **計算コスト:** LLM呼び出しや多数の評価/グリッドサーチ実行は、時間と計算リソース（場合によってはAPIコスト）を消費します。

## 7. Claude Desktop 連携 (オプション)

Claude DesktopアプリケーションをMCPクライアントとして使用するための設定が可能です。

```bash
python -m src.cli.setup_claude_integration --help
```
このスクリプトを実行すると、Claude Desktop の設定ファイル (`config.json`) に、ローカルで起動したMCPサーバーへの接続設定や、関連するプロンプトテンプレートが追加されます。詳細はスクリプトのヘルプを参照してください。

## 8. トラブルシューティング

*   **サーバーが起動しない:**
    *   ポート (`5002` など) が既に使用されていないか確認 (`netstat` や `lsof` コマンド)。
    *   依存関係 (`uv pip sync requirements-lock.txt`) が正しくインストールされているか確認。
    *   Pythonのバージョンが `pyproject.toml` の `requires-python` の範囲内か確認。
    *   設定ファイル (`config.yaml`) のYAML構文が正しいか確認。
*   **クライアント (`mirai`) が接続できない:**
    *   サーバーURL (`--server`) とポート番号が正しいか確認。
    *   MCPサーバーが起動しているか確認。
    *   ネットワーク接続やファイアウォール設定を確認。
*   **ジョブが失敗する (`failed` ステータス):**
    *   サーバーの標準出力ログを確認し、エラーメッセージやトレースバックを調査。
    *   `mirai improve status <session_id>` を実行し、`last_error` フィールドを確認。
    *   DB (`.mcp_server_data/db/mcp_server_state.db`) の `jobs` テーブルで該当ジョブの `error_details` カラムを確認 (SQLiteクライアントなどを使用)。
    *   入力データ（データセットパスなど）や設定 (`config.yaml`) に問題がないか確認。
    *   LLM APIキーが正しく設定されているか確認。
*   **改善ループが意図通りに動作しない/停止する:**
    *   サーバーログを確認し、プロンプト生成ジョブの実行状況やエラーを追跡。
    *   `mirai improve status <session_id>` でセッション状態（ステータス、サイクル数、停滞カウント）を確認。
    *   DBの `sessions` テーブルで `status`, `cycle_state`, `history` を確認。
    *   プロンプトテンプレート (`prompts/`) を確認。
    *   クライアントの環境変数 (`ANTHROPIC_API_KEY` など) が正しく設定されているか確認。
*   **LLMクライアントのエラー:**
    *   APIキーが正しいことを確認。
    *   ネットワーク接続を確認。
    *   レート制限やクォータ超過など、サービス側の制限を確認。
    *   LLMクライアントが正しいエンドポイントと通信しているか確認。
