# MCP サーバー & AI 自動改善システム (AutoImprover) 詳細README

## 1. 概要

このドキュメントは、MIRアルゴリズムのAI駆動自動改善システム、特に **MCP (Music Creation Platform) サーバー** と **AutoImprover クライアント** の詳細について説明します。これは実験的な機能であり、アルゴリズムの性能向上プロセスを自動化する可能性を探ることを目的としています。

**注意:** このシステムは開発途上であり、安定性や結果の品質は保証されません。

**主なコンポーネント:**

*   **MCP サーバー (`src/cli/mcp_server.py`):**
    *   FastAPI と Uvicorn をベースにしたWebサーバー。
    *   アルゴリズム評価、グリッドサーチ、コード操作、可視化などの機能を**ツール**として提供します。
    *   LLM (現在 Anthropic Claude を想定) と連携し、プロンプトに基づいてツールを呼び出します。
*   **AutoImprover クライアント (`src/cli/auto_improver.py`):**
    *   MCPサーバーに対して、アルゴリズムの自動改善サイクル（評価 → LLMによる改善提案 → 改善適用 → 再評価）を開始するCLIツール。
    *   改善の目標や使用する検出器、データセットなどを指定します。
*   **Claude Desktop 連携 (オプション, `src/cli/setup_claude_integration.py`):**
    *   Claude Desktop アプリケーションとの連携を設定し、よりインタラクティブな改善プロセスを可能にします (実験的)。

## 2. システムの動作フロー

1.  **MCPサーバーの起動:** `python -m src.cli.mcp_server` でサーバーを起動します。設定 (`config.yaml`, `.env`) を読み込み、データベース接続、ジョブワーカーを初期化します。
2.  **(ユーザー) AutoImprover の実行:** `python -m src.cli.auto_improver` で改善プロセスを開始します。ターゲット検出器、データセット、目標などを指定します。
3.  **(AutoImprover) 改善セッション開始:** AutoImprover は MCPサーバーに「改善セッションの開始」をリクエストします。
4.  **(MCPサーバー) 初期評価:** サーバーは指定された検出器とデータセットで初期評価を実行します (非同期ジョブ)。
5.  **(MCPサーバー) LLM への改善依頼:** 初期評価結果に基づき、LLM (Claude) に改善戦略の提案と具体的なコード変更を依頼します。この際、サーバーが提供するツール (コード取得、評価実行など) を LLM が利用できるようプロンプトを構成します。
6.  **(LLM) 改善提案とツール利用:** LLM は改善案を考え、必要に応じてMCPサーバーのツール (例: `get_code`, `run_evaluation`) を呼び出して情報を収集・検証します。
7.  **(MCPサーバー) ツール実行:** LLM からのリクエストに応じて、対応するツール (関数) を実行し、結果をLLMに返します (評価などは非同期ジョブとして実行)。
8.  **(LLM) 改善コード生成:** LLM は最終的な改善コードを生成します。
9.  **(MCPサーバー) 改善コード保存:** MCPサーバーはLLMから受け取った改善コードを `mcp_workspace/improved_versions/` 配下に保存します。
10. **(MCPサーバー) 改善後評価:** 保存された改善コードを使って再度評価を実行します (非同期ジョブ)。
11. **(MCPサーバー) 結果報告:** 改善前後の評価結果を比較し、AutoImprover (または直接のAPI呼び出し元) に報告します。
12. **(繰り返し):** AutoImprover の設定に基づき、改善が見られる場合はステップ 5-11 を繰り返します。

## 3. セットアップ (AI 自動改善機能向け)

基本的なセットアップは `README.md` の「3. セットアップ」に従ってください。以下はAI機能に特有の追加設定です。

1.  **依存関係の確認:**
    *   `pyproject.toml` の `[tool.uv.sources]` または `[project.optional-dependencies]` に `mcp` や `anthropic` 関連の依存関係が含まれていることを確認します。
    *   `uv pip install -e .[dev,numba,crepe,mcp]` や `uv pip sync requirements-lock.txt` (推奨、事前に `uv pip compile ... --all-extras`) などで関連パッケージ (`fastapi`, `uvicorn`, `anthropic`, `aiosqlite` 等) がインストールされていることを確認します。

2.  **環境変数と設定 (`.env`, `config.yaml`):**
    `.env` ファイルまたは環境変数で以下を設定します。
    *   `ANTHROPIC_API_KEY`: Anthropic Claude API キー。
    *   `MIREX_WORKSPACE`: MCPサーバーが作業ファイル (DB、改善コード、結果、ログ) を保存する**絶対パス**。デフォルトはプロジェクトルート下の `mcp_workspace` ですが、明示的な絶対パス指定を推奨します。
    *   **(オプション) `MCP__SERVER__HOST`, `MCP__SERVER__PORT`:** サーバーのホストとポート (デフォルト: `localhost`, `5002`)。
    *   **(オプション) `MCP__DATABASE__PATH`:** データベースファイルのパス。デフォルトは `<MIREX_WORKSPACE>/db/mcp_server_state.db`。
    *   **(オプション) `MCP__IMPROVED_CODE__DIR`:** 改善コード保存ディレクトリ。デフォルトは `<MIREX_WORKSPACE>/improved_versions`。
    *   **(オプション) `MCP__RESULTS__DIR`:** 評価結果保存ディレクトリ。デフォルトは `<MIREX_WORKSPACE>/results`。
    *   **(オプション) `MCP__LOG__DIR`:** ログ保存ディレクトリ。デフォルトは `<MIREX_WORKSPACE>/logs`。
    *   **(オプション) `DATASETS_BASE_DIR`:** データセットディレクトリのベースパス。デフォルトはプロジェクトルート下の `datasets`。
    *   **(オプション) `ANTHROPIC_API_MODEL`:** 使用する Claude モデル名 (デフォルト: `claude-3-opus-20240229`)。

    **設定の優先順位:** 環境変数 > `.env` ファイル > `config.yaml`
    詳細は `src/mcp_server_logic/core.py` の `load_config` 関数を参照してください。

3.  **ワークスペースディレクトリの作成:**
    `MIREX_WORKSPACE` で指定したディレクトリが存在しない場合は作成してください。
    ```bash
    mkdir -p <MIREX_WORKSPACEで指定したパス>
    ```
    サーバーは起動時にサブディレクトリ (`db`, `improved_versions`, `results`, `logs`) を自動生成しようとします。

## 4. 使用方法

### 4.1. MCP サーバーの起動

ターミナルで以下を実行します。

```bash
python -m src.cli.mcp_server --host 0.0.0.0 --port 5002
```

*   `--host`: サーバーがリッスンするホストアドレス。コンテナ環境や外部からのアクセスを許可する場合は `0.0.0.0` を指定します。
*   `--port`: サーバーがリッスンするポート番号。
*   設定は `config.yaml`, `.env`, 環境変数からも読み込まれます。

サーバーは起動ログ（設定、DBパス、利用可能なツールなど）を出力し、リクエスト待機状態になります。**Ctrl+C で停止**できます。

### 4.2. AutoImprover の実行 (改善サイクルの開始)

別のターミナルで以下を実行します。

```bash
python -m src.cli.auto_improver \
    --detector PZSTDDetector \
    --dataset synthesized_v1 \
    --metric note.f_measure \
    --goal "Increase F-measure by improving onset accuracy" \
    --max-iterations 3 \
    --mcp-server-url http://localhost:5002
```

*   `--detector`: 改善対象の検出器クラス名。
*   `--dataset`: 評価に使用するデータセット名 (`config.yaml` で定義)。
*   `--metric`: 改善の主要評価指標 (例: `note.f_measure`, `frame.accuracy`)。
*   `--goal`: LLM に伝える改善目標 (自由記述)。
*   `--max-iterations`: 最大試行回数。
*   `--mcp-server-url`: 実行中の MCP サーバーの URL。
*   **(オプション) `--session-id`:** 既存のセッションを再開する場合に指定。
*   **(オプション) `--output-dir`:** 結果サマリの保存先。**注意: これはクライアント側のパスです。**

AutoImprover は MCP サーバーと通信し、改善プロセスを開始します。進捗はターミナルに出力されます。

### 4.3. 評価とグリッドサーチ (CLI経由)

`README.md` の「4. 基本的な使用方法」で説明されている `evaluate` および `grid-search` コマンドは、内部的に実行中の MCP サーバーの API を呼び出します。

**例:**

```bash
# MCPサーバーが起動している前提
python -m src.cli.improver_cli evaluate --detector PZSTDDetector --dataset synthesized_v1 --output-dir mcp_workspace/results/my_eval

python -m src.cli.improver_cli grid-search grid_config.yaml --output-dir mcp_workspace/results/my_grid --best-metric note.f_measure
```

*   これらのコマンドは MCP サーバーに**非同期ジョブ**をリクエストします。
*   実行後、ジョブIDが返されます。
*   結果はサーバー側の `mcp_workspace/results` (または設定による) 配下の指定されたディレクトリに保存されます。
*   ジョブのステータス確認や結果取得は、現時点では MCP サーバーの API を直接叩くか、データベース (`mcp_server_state.db`) を確認する必要があります (専用のCLIツールは未実装)。

### 4.4. (実験的) Claude Desktop 連携

`src/cli/setup_claude_integration.py` を使用して設定します。詳細はスクリプト内のコメントや `--help` を参照してください。これにより、Claude Desktop から直接 MCP サーバーのツールを呼び出すことが可能になる場合があります (Anthropic の機能に依存)。

## 5. MCP サーバーのツール (LLM が利用)

MCP サーバーは、LLM がアルゴリズム改善のために利用できる様々なツール (Python関数) を公開しています。これらのツールは FastAPI ルートとして公開され、LLM (Anthropic API の Tool Use 機能) から呼び出されます。

主なツールカテゴリ:

*   **評価ツール (`mcp_server_logic/evaluation_tools.py`):**
    *   `run_evaluation`: 指定された検出器、データセット、パラメータで評価を実行 (非同期ジョブ)。
    *   `run_grid_search`: グリッドサーチを実行 (非同期ジョブ)。
*   **コードツール (`mcp_server_logic/code_tools.py`):**
    *   `get_code`: 指定された検出器クラスのソースコードを取得。
    *   `save_improved_code`: LLM が生成した改善コードを保存。
    *   `list_available_detectors`: 利用可能な検出器クラスの一覧を取得。
*   **改善ループツール (`mcp_server_logic/improvement_loop.py`):**
    *   `propose_improvement_strategy`: 現在の評価結果に基づき、LLM に改善戦略を提案させる。
    *   `implement_code_changes`: LLM に具体的なコード変更を実装させる。
*   **可視化ツール (`mcp_server_logic/visualization_tools.py`):**
    *   `generate_comparison_plot`: 評価結果の比較プロットを生成 (非同期ジョブ)。
*   **ジョブ管理ツール (`mcp_server_logic/job_manager.py` API経由):**
    *   `get_job_status`: 指定されたジョブIDの状態（実行中, 完了, エラー）を取得。
    *   `get_job_result`: 完了したジョブの結果を取得。
*   **セッション管理ツール (`mcp_server_logic/session_manager.py` API経由):**
    *   `start_improvement_session`: 新しい改善セッションを開始。
    *   `get_session_history`: セッションの履歴 (実行されたツール、結果など) を取得。
*   **(オプション) 拡張ツール (`mcp_server_logic/mcp_server_extensions.py`):**
    *   プロジェクト固有のカスタムツールを追加できます。

これらのツールは `src/cli/mcp_server.py` で FastAPI アプリケーションに登録され、Anthropic API との連携部分 (`mcp_server_logic/llm_tools.py`) で利用されます。

## 6. データベース (`mcp_server_state.db`)

MCP サーバーは状態管理のために SQLite データベースを使用します。デフォルトでは `mcp_workspace/db/mcp_server_state.db` に作成されます。`aiosqlite` を使用して**非同期**でアクセスされます。

主なテーブル:

*   `jobs`: 非同期ジョブの情報を格納 (job_id, task_name, status, created_at, result など)。
*   `improvement_sessions`: 自動改善セッションの情報を格納 (session_id, detector, dataset, goal, status, history など)。

データベーススキーマの詳細は `mcp_server_logic/db_utils.py` の `initialize_database` 関数を参照してください。

## 7. トラブルシューティング

*   **サーバーが起動しない:**
    *   ポートが既に使用されていないか確認 (`netstat` や `lsof` コマンド)。
    *   必要な依存関係 (`fastapi`, `uvicorn`, `aiosqlite` 等) がインストールされているか確認。
    *   設定ファイル (`config.yaml`, `.env`) のパスや内容が正しいか確認。
    *   `MIREX_WORKSPACE` ディレクトリの書き込み権限があるか確認。
*   **AutoImprover がサーバーに接続できない:**
    *   サーバーが正しく起動しており、指定した URL (`--mcp-server-url`) が正しいか確認。
    *   ファイアウォール設定を確認。
*   **LLM API エラー:**
    *   `ANTHROPIC_API_KEY` が正しく設定されているか確認。
    *   APIキーの利用制限や支払い状況を確認。
    *   Anthropic のサービスステータスを確認。
*   **ジョブが `pending` または `running` のまま進まない:**
    *   サーバーログ (`mcp_workspace/logs/mcp_server.log`) でエラーが発生していないか確認。
    *   サーバープロセスのリソース使用状況（CPU, メモリ）を確認。
    *   評価やグリッドサーチ自体に時間がかかっている可能性。
    *   `aiosqlite` の非同期処理に問題が発生していないか確認 (デバッグログ有効化など)。
*   **ファイルパス関連のエラー:**
    *   クライアント (AutoImprover, improver_cli) から渡されるパス (データセット、グリッド設定、出力先) が、サーバー側の設定 (`config.yaml` の `allowed_paths` や `MIREX_WORKSPACE`, `DATASETS_BASE_DIR`) で許可された範囲内にあるか確認。
    *   特に `--output-dir` はサーバー側の `mcp_workspace/results` 配下に解決される点に注意。
*   **データベースエラー:**
    *   DBファイル (`mcp_server_state.db`) が破損していないか確認。
    *   同時に複数のサーバープロセスが同じDBファイルに書き込もうとしていないか確認 (`aiosqlite` は非同期アクセスを扱いますが、プロセスレベルの競合は問題になる可能性)。

## 8. Dockerでの実行 (再掲)

`README.md` と同様に Docker を使用できます。

1.  **イメージのビルド:** (`uv` とロックファイル推奨)
    ```bash
    uv pip compile pyproject.toml --all-extras -o requirements-lock.txt
    docker build -t mirex-auto-improver .
    ```

2.  **コンテナの実行 (MCPサーバー):**
    ```bash
    docker run -p 5002:5002 \
           -e ANTHROPIC_API_KEY="<あなたのAPIキー>" \
           -e MIREX_WORKSPACE="/app/mcp_workspace" \
           -e MCP__SERVER__HOST="0.0.0.0" \
           -v "$(pwd)/mcp_workspace":/app/mcp_workspace \
           -v "$(pwd)/datasets":/app/datasets \
           -v "$(pwd)/config.yaml":/app/config.yaml \
           -v "$(pwd)/.env":/app/.env \
           --rm -it mirex-auto-improver \
           python -m src.cli.mcp_server --host 0.0.0.0 --port 5002
    ```
    *   **ホストは `0.0.0.0` を指定** してコンテナ外部からのアクセスを許可します。
    *   必要な環境変数 (`ANTHROPIC_API_KEY`, `MIREX_WORKSPACE` など) を `-e` で渡します。コンテナ内のパス (`/app/mcp_workspace`) を指定します。
    *   必要なディレクトリや設定ファイルを `-v` でマウントします。

3.  **コンテナの実行 (AutoImprover クライアント):**
    別のターミナルから、実行中のサーバーコンテナに対して AutoImprover を実行する場合:
    ```bash
    # ホストからコンテナ内のサーバー (localhost:5002) に接続する場合
    # (ホスト側で仮想環境が有効になっている前提)
    python -m src.cli.auto_improver --mcp-server-url http://localhost:5002 [その他の引数...]

    # または、別のコンテナからサーバーコンテナに接続する場合 (Dockerネットワーク設定が必要)
    # 例: docker exec を使う場合
    docker exec -it <サーバーコンテナ名またはID> \
        python -m src.cli.auto_improver --mcp-server-url http://localhost:5002 [その他の引数...]
    ```