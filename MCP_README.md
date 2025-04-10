# MIR アルゴリズム自動改善 (MCP/AutoImprover 詳細)

**注意:** ここで説明する機能は**実験的**であり、活発な開発下にあります。安定性や結果の品質は保証されません。基本的な評価フレームワークの使い方については、メインの `README.md` を参照してください。

## 1. システム概要 (AI駆動自動改善)

本システムは、音楽情報検索 (MIR) アルゴリズムの開発、評価、改善のプロセスにAI (大規模言語モデル、LLM) を組み込み、**アルゴリズム自体の自動的な発見と進化**を目指す実験的な研究プラットフォームです。

**コアコンポーネント:**

*   **MCPサーバー (`mcp_server.py`):** ModelContextProtocol (MCP) に準拠したAPIサーバー。アルゴリズムのコード操作、評価実行、パラメータ探索、LLM連携などの「ツール」を提供します。非同期ジョブ管理、セッション管理（DB使用）も行います。
*   **AutoImprover クライアント (`auto_improver.py`):** MCPサーバーのツールを利用して、AI駆動の自動改善サイクル（評価→分析→戦略決定→コード改善/パラメータ調整→評価…）を実行するクライアントアプリケーション。
*   **LLM (例: Anthropic Claude):** サーバー側のツールから呼び出され、コード改善案の生成、評価結果の分析、次の改善戦略の提案など、インテリジェントなタスクを担当します。
*   **Claude Desktop (オプション):** MCPサーバーのツールを対話的に呼び出し、改善プロセスをステップバイステップで実行・監視するためのインターフェース。

**目指す動作フロー:**

1.  ユーザーがベースアルゴリズムと目標を指定して `auto_improver.py` を起動。
2.  AutoImproverがMCPサーバーでセッションを開始。
3.  AutoImproverが初期評価 (`run_evaluation`) を実行。
4.  AutoImproverが評価結果を分析 (`analyze_evaluation_results` - 内部でLLM呼び出し) させる。
5.  AutoImproverが分析結果と履歴から次の戦略を提案 (`suggest_exploration_strategy` - 内部でLLM呼び出し) させる。
6.  提案された戦略に基づき、AutoImproverが対応するツール（`improve_code`, `optimize_parameters` など）を呼び出す。`improve_code` の場合は分析結果や仮説もプロンプトに含める。
7.  改善/最適化されたアルゴリズムをAutoImproverが評価 (`run_evaluation`)。
8.  結果を比較し、最良版を更新。
9.  ループを指定回数または収束するまで繰り返す。

## 2. セットアップ手順 (AI自動改善機能向け)

基本的なセットアップ (Python環境、`uv` の導入、仮想環境作成) はメインの `README.md` の「3. セットアップ (`uv` 使用)」を参照してください。AI自動改善機能には追加のステップが必要です。

### 2.1 依存ライブラリの確認

`uv` を使用して依存関係をインストールする際に、必要なライブラリが `pyproject.toml` に含まれていることを確認してください。

*   `mcp-fastmcp`: MCPサーバー/クライアントの基盤ライブラリ。
*   `fastapi`, `uvicorn`: MCPサーバーのWebフレームワークとASGIサーバー。
*   `requests`, `httpx`: クライアントからのHTTPリクエスト用。
*   `pyyaml`: 設定ファイル (`config.yaml`) の読み込み用。
*   `numpy`, `pandas`: データ処理用。
*   `tenacity`: リトライ処理用 (LLM API呼び出しなど)。
*   `python-dotenv`: `.env` ファイルからの環境変数読み込み用。
*   必要なLLMクライアントライブラリ (例: `anthropic`, `openai`)。

`uv pip install -e .[dev,numba,crepe]` またはロックファイル (`requirements-lock.txt` など) を使用した `uv pip sync` でインストールされます。

### 2.2 ワークスペースと設定

*   **ワークスペース:** スクリプト実行時に `mcp_workspace` ディレクトリが自動生成されます (場所は環境変数 `MIREX_WORKSPACE` または `config.yaml` で指定可能)。この中にセッション状態DB (`mcp_server_state.db`)、改善版コード、ログなどが保存されます。
*   **設定ファイル (`config.yaml`):** プロジェクトルートの `config.yaml` で、サーバー設定、パス、リソース制限、クリーンアップ設定、LLMモデル、データセットなどを構成できます。構造は機能ごとに整理されています。
*   **環境変数 (`.env`):** プロジェクトルートに `.env` ファイルを作成 (または `.env.example` をコピー) し、以下の**必須**またはオプションの変数を設定します。**環境変数は `config.yaml` の値を上書きします。**
    *   `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`: (条件付き必須) 利用するLLMのAPIキー。
    *   `MIREX_WORKSPACE`: (推奨) ワークスペースディレクトリのパス。指定しない場合は `./mcp_workspace` が試みられますが、書き込み権限などの問題が発生する可能性があります。
    *   `MCP_SERVER_URL`: (任意) クライアントが接続するMCPサーバーのURL (デフォルト: `http://localhost:5002`)。
    *   `MCP_PORT`: (任意) MCPサーバーがリッスンするポート (デフォルト: 5002)。
    *   その他、`config.yaml` の値を上書きするための環境変数。
        *   命名規則: `MCP__SECTION__KEY=value` (例: `MCP__LLM__MODEL=gpt-4`, `MCP__SERVER__LOG_LEVEL=DEBUG`)
        *   詳細は `mcp_server_logic/core.py` の `load_config` 関数の Docstring を参照してください。

### 2.3 Claude Desktop用設定 (オプション)

Claude DesktopアプリからMCPサーバーのツールを直接操作したい場合、以下のコマンドで連携設定を行います。

```bash
python setup_claude_integration.py --open-claude
```

*   `--open-claude`: 設定適用後にClaude Desktopアプリを起動します。
*   `--config-path`: (オプション) Claude Desktopの設定ファイルパスを指定します。

このコマンドは `~/Library/Application Support/Claude/storage/config.json` (macOSの場合) を更新し、MCPサーバー情報、ツール定義、プロンプトテンプレートなどを追加します。

## 3. 利用方法 (AI自動改善)

### 3.1 MCPサーバーの起動 (必須)

AI自動改善機能を利用するには、まずMCPサーバーが起動している必要があります。

```bash
# 仮想環境を有効化していることを確認
python mcp_server.py [--port <ポート番号>] [--log-level <レベル>] [--config <設定ファイル>] [--workspace <パス>] [--num-workers <数>]
```

*   サーバーはAPIリクエストを待ち受け、非同期ジョブを実行します。
*   `--config`: デフォルト (`config.yaml`) 以外の設定ファイルを指定する場合。
*   `--workspace`: `.env` や `config.yaml` の設定を上書きする場合。
*   サーバー起動時にデータベース (`mcp_workspace/db/mcp_server_state.db`) が初期化されます。
*   サーバー起動時にパス検証用の許可ディレクトリリストが `config.yaml` から読み込まれます。

### 3.2 自動改善サイクルの実行 (`auto_improver.py`)

コマンドラインから自動改善プロセス全体を開始します。

```bash
# 仮想環境を有効化していることを確認
python auto_improver.py --detector <検出器名> [--goal <目標>] [--iterations <回数>] [--server-url <URL>] [--config-file <設定ファイル>]
```

*   `--detector`: 改善対象のベースとなる検出器名。
*   他オプションは以前と同様。
*   `--config-file`: (New!) AutoImprover 固有の設定を記述したYAMLファイルのパス。

実行すると、`auto_improver.py` がクライアントとして動作し、MCPサーバーの各種ツールを呼び出しながら改善サイクルを進めます。進捗はログに出力され、セッション状態はサーバーのDBに記録されます。

### 3.3 Claude Desktopでの対話的操作 (オプション)

`setup_claude_integration.py` で設定後、Claude DesktopからMCPツールを個別に呼び出して、対話的に改善プロセスを進めることも可能です。

1.  Claude Desktopを起動します。
2.  ハンマーアイコン (🔨) をクリックし、「mirex-auto-improver」サーバーを選択してツールを有効化します。
3.  チャット入力欄で `@tool <ツール名>(引数=値, ...)` の形式でツールを呼び出します。
    ```
    # 例: セッション開始
    @tool start_session(base_algorithm="PZSTDDetector")

    # 例: 評価実行 (ジョブIDが返る)
    @tool run_evaluation(detector_name="PZSTDDetector", dataset_name="synthesized_v1")

    # 例: ジョブステータス確認
    @tool get_job_status(job_id="返ってきたジョブID")
    ```
4.  ワークフローについては `@prompt workflow-guide` でガイドを表示できます。

### 3.4 利用可能なMCPツール一覧 (`mcp_server.py` 提供)

*   `health_check`: サーバー動作確認。
*   `start_session`: 改善セッション開始。
*   `get_session_info`: セッション情報取得。
*   `add_session_history`: セッション履歴追加（主に内部/クライアントが使用）。
*   `get_job_status`: 非同期ジョブ状態確認。
*   `get_code`: 検出器コード取得。
*   `save_code`: 改善コード保存。
*   `improve_code`: (LLM) コード改善依頼 (非同期ジョブ)。
*   `run_evaluation`: 検出器評価実行 (非同期ジョブ)。
*   `run_grid_search`: 設定ファイルに基づくグリッドサーチ実行 (非同期ジョブ)。
*   `analyze_evaluation_results`: (LLM) 評価結果分析 (非同期ジョブ)。
*   `generate_hypotheses`: (LLM) 科学的仮説生成 (非同期ジョブ)。
*   `analyze_code_segment`: (LLM) コード断片分析 (非同期ジョブ)。
*   `suggest_parameters`: (LLM) パラメータ調整範囲提案 (非同期ジョブ)。
*   `optimize_parameters`: (LLM+GridSearch) パラメータ自動提案と最適化 (非同期ジョブ)。
*   `suggest_exploration_strategy`: (LLM) 次の改善戦略提案 (非同期ジョブ)。
*   `visualize_improvement_trajectory`: 改善軌跡グラフ生成 (非同期ジョブ)。
*   `create_thumbnail`: 画像サムネイル生成。
*   `visualize_spectrogram`: スペクトログラム画像生成。
*   `process_audio_files`: 複数音声ファイル処理 (Context使用例)。
*   **(拡張機能)** `visualize_code_impact`, `generate_performance_heatmap`, `generate_scientific_outputs`: 可視化・科学的成果物生成ツール (`mcp_server_extensions.py` が存在する場合)。
*   **注意:** パスを引数に取るツール (`run_evaluation`, `run_grid_search`, `get_code`, `save_code` など) は、入力されたパスがサーバー側で許可されたディレクトリ内にあるか検証します。

## 4. 自動改善プロセスの詳細 (AutoImprover)

`auto_improver.py` 内の `run_improvement_cycle` メソッドが中心となり、以下のステップを繰り返します。

1.  **戦略決定:** `suggest_exploration_strategy` を呼び出し、現在の状況（パフォーマンス、停滞具合、履歴）に基づいて次のアクション（`improve_code`, `optimize_parameters`, `generate_hypothesis` など）を決定します。
2.  **アクション実行:** 決定されたアクションに対応するMCPツールを呼び出します。
    *   **コード改善の場合:** `analyze_evaluation_results` で得られた弱点や、`generate_hypotheses` で得られた仮説をプロンプトに含めて `improve_code` を呼び出します。
    *   **パラメータ最適化の場合:** `optimize_parameters` を呼び出します。
    *   **仮説生成の場合:** `generate_hypotheses` を呼び出し、得られた仮説を次の改善の指針とします。
3.  **評価:** コード改善やパラメータ最適化が行われた場合、`run_evaluation` を呼び出して新しいバージョンの性能を評価します。
4.  **状態更新:** 評価結果を比較し、性能が向上していれば最良バージョン情報を更新します。改善履歴や戦略履歴も記録します。
5.  **状態保存:** 現在のサイクル状態 (`self.cycle_state`) をファイルに保存し、中断・再開を可能にします。

## 5. トラブルシューティング

*   **サーバーが起動しない:**
    *   ポート競合、依存関係不足 (`uv pip sync` または `uv pip install` を確認)、Pythonバージョンを確認。
    *   設定ファイル (`config.yaml`) のパスや内容が正しいか確認。
    *   ワークスペースディレクトリの権限を確認。
    *   ログ (`mcp_server.log` 等) を確認。
*   **LLM APIエラー:**
    *   `.env` のAPIキー設定、インターネット接続、サービス障害、使用量制限を確認。
*   **ジョブが進まない (`pending`/`queued`):**
    *   サーバーログを確認 (ワーカーエラー)。
    *   `config.yaml` の `resource_limits.max_concurrent_jobs` 設定を確認。
    *   DB (`mcp_workspace/db/mcp_server_state.db`) の状態を確認。
*   **ファイル/パス関連エラー:**
    *   サーバープロセスがワークスペースやデータセットディレクトリへの読み書き権限を持っているか確認。
    *   クライアントから指定したパスが、サーバーの `config.yaml` で許可されたベースディレクトリ内にあるか確認 (`FileError: 指定されたパスは許可されたディレクトリ内にありません` エラー)。
    *   指定したファイルが存在するか、ファイル/ディレクトリ種別が正しいか確認 (`FileError: 指定されたパスが存在しません` / `...ファイルではありません` / `...ディレクトリではありません` エラー)。
*   **`ValueError` / `FileError` / `ConfigError` (ツール呼び出し時):**
    *   MCPツールに渡した引数（特にパス）が不正であるか、サーバー側の検証でエラーになった可能性があります。サーバーログとツールのDocstringを確認してください。

**ログファイルの場所:**

*   **MCPサーバーログ:** 通常、サーバーを起動したディレクトリに `mcp_server.log` (または設定による) が生成されます。
*   **AutoImproverログ:** `auto_improver.py` の実行時にコンソールに出力されます (ファイル出力は未実装)。
*   **Claude Desktop連携ログ:** `~/Library/Logs/Claude/mcp*.log` (macOSの場合)。

## 6. まとめ

このプロジェクトは、MIRアルゴリズムの評価基盤と、AIを活用した自動改善のための実験的プラットフォームを提供します。基本的な評価機能は安定していますが、AI自動改善機能は開発途上です。フィードバックを歓迎します。

## インストール

```bash
# 1. リポジトリをクローン
git clone <this_repository_url>
cd <repository_directory>

# 2. (推奨) 仮想環境を作成・有効化
python -m venv venv
source venv/bin/activate # Linux/macOS
# venv\Scripts\activate # Windows

# 3. 必要なライブラリをインストール
pip install -r requirements.txt
# poetry を使用する場合:
# poetry install

# 4. CREPE のインストール (ピッチ推定評価で使用する場合)
pip install crepe
# 注意: TensorFlow のバージョン互換性により問題が発生する場合があります。
# 詳細は CREPE のドキュメントを参照してください: https://github.com/marl/crepe

# 5. (オプション) 開発用ツールをインストール
pip install -r requirements-dev.txt
# poetry を使用する場合:
# poetry install --with dev
```

## 設定