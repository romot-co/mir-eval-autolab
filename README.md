# MIR アルゴリズム評価・自動改善プラットフォーム

このリポジトリ自体の作成にVibeコーディングが可能かも含めた実験環境であり、実用を考慮したものではないことに注意してください。

## 実際どんなのができる？

- **AIの提案:** [`algos/`](algos/) ディレクトリに、AIが提案した10の音楽信号処理アルゴリズム（ONDE、PZSTD、ECHOE、FAZEなど）仕様が保存されています。
- **AIの実装:** [`src/detectors/`](src/detectors/) ディレクトリに、AIが提案した10のアルゴリズムの実装があります。
- 提案されたアルゴリズムのうち、実際に実用に耐えるのは **PZSTD** と **ONDE/ONDE-FIVE** のみです。
- この取り組みの詳細な結果と分析は[Zenn記事「AIは新しいアルゴリズム発見の夢を見るのか」](https://zenn.dev/romot/articles/11edc86fb6ab11)で公開しています。

## 1. プロジェクト概要

本プロジェクトは、音楽情報検索 (MIR) アルゴリズム（音符検出、オンセット検出、ピッチ推定など）の性能評価、比較、そしてAIを活用した改善を支援するためのプラットフォームです。

**コア機能:**

*   **評価:** `mir_eval` ライブラリに基づき、客観的な評価指標（Precision, Recall, F-measure など）を算出します。
*   **データ処理:** 評価に必要な音声ファイルや参照ラベル（CSV形式など）をロードします。
*   **検出器インターフェース:** 新しい検出アルゴリズムを容易に追加・評価できる共通インターフェースを提供します (`src/detectors/`)。
*   **パラメータ最適化:** グリッドサーチにより、検出器の最適なパラメータ設定を探索できます (`src/evaluation/grid_search/`)。

**主要コンポーネント:**

*   **コアMIR機能 (`src/` 配下):** 上記の評価、データ処理、検出器などの基本的な機能を提供します。これらは他のコンポーネントから独立して利用可能です。
*   **統合CLI (`src/cli/mirai.py`):** コア機能の直接利用（スタンドアロン評価）と、MCPサーバー経由での評価、グリッドサーチ、自動改善ループ実行など、本プロジェクトの主要な操作を単一のコマンドラインインターフェースから行います。
*   **MCPサーバー (`src/cli/mcp_server.py` と `src/mcp_server_logic/`):** コア機能やAI連携機能（LLM呼び出し）を外部に公開するためのインターフェースです。MCP (Model Context Protocol) に準拠し、非同期ジョブ管理、**サーバー中心の状態管理**（DB、改善セッション履歴、サイクル状態など）、AIエージェント（例: Claude Desktop）との連携を担います。

**AI駆動自動改善システム (実験的機能):**

本プロジェクトには、MCPサーバーと `mirai improve` コマンドを用いて、MIRアルゴリズムの改善サイクルを自動化する**実験的な機能**が含まれています。`mirai improve start`コマンドは、**CLIホスト主導**で評価、分析、仮説生成、コード改善、パラメータ最適化といった一連のステップを管理・実行します。これはLLM APIを直接呼び出し、AIによるアルゴリズム改善の可能性を探る試みであり、安定性や結果の品質は保証されません。

**AI駆動自動改善システムの詳細なセットアップ、利用方法、トラブルシューティングについては、[`MCP_README.md`](MCP_README.md) を参照してください。**

## 2. ディレクトリ構成 (主要部分)

```
.
├── src/                     # MIR関連機能・サーバーロジックのソースコード
│   ├── detectors/           # 検出アルゴリズム実装 (.py)
│   ├── evaluation/          # 評価関連モジュール
│   │   └── grid_search/     # グリッドサーチ関連
│   ├── data_generation/     # 合成データ生成 (オプション)
│   ├── utils/               # 共通ユーティリティ (パス, 例外, JSON, Logging etc.)
│   ├── mcp_server_logic/    # MCPサーバーの中核ロジック
│   │   ├── core.py              # 設定読み込み、クリーンアップ等
│   │   ├── db_utils.py          # 非同期DB操作
│   │   ├── job_manager.py       # 非同期ジョブ管理
│   │   ├── session_manager.py   # 改善セッション管理 (履歴記録含む)
│   │   ├── llm_tools.py         # LLMプロンプト生成ツール (分析, 改善, 仮説, 戦略提案, 評価)
│   │   ├── evaluation_tools.py  # 評価/グリッドサーチ実行ツール (サーバー側)
│   │   ├── code_tools.py        # コード取得/保存ツール (Git連携含む)
│   │   ├── schemas.py           # Pydantic スキーマ定義
│   │   └── prompts/             # LLM プロンプトテンプレート (.j2)
│   └── cli/                 # コマンドラインインターフェーススクリプト
│       ├── mirai.py             # ★ 統合CLIツール (評価、グリッドサーチ、改善) - MCPホスト機能を実装
│       ├── llm_client.py        # ★ LLM API呼び出しクライアント (Anthropic Claude等)
│       ├── mcp_server.py        # MCPサーバー起動スクリプト
│       └── setup_claude_integration.py # (オプション) Claude Desktop連携設定
├── tests/                   # テストコード
├── datasets/                # データファイル (設定ファイルや .env でパス指定)
├── output/                  # サーバー経由/スタンドアロンの出力ベースディレクトリ (設定可能)
├── .mcp_server_data/        # MCPサーバーのデフォルト作業ディレクトリ (DB、改善コードなど。設定可能)
├── pyproject.toml           # プロジェクト設定、依存関係 (uv が使用)
├── config.yaml              # プロジェクト全体の設定ファイル
├── requirements-lock.txt    # (推奨) 生成された依存関係ロックファイル
├── .env.example             # 環境変数設定のテンプレート
├── Dockerfile               # Dockerイメージビルド用 (uv 使用)
├── MCP_README.md            # MCPサーバー/AutoImprover向け詳細README
└── README.md                # このファイル
```

## 3. セットアップ (`uv` 使用)

本プロジェクトでは、高速な Python パッケージインストーラおよびリゾルバである `uv` の使用を推奨します。

1.  **リポジトリのクローン:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **uv のインストール (未導入の場合):**
    ```bash
    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Windows (詳細は https://github.com/astral-sh/uv を参照)
    # pip install uv
    ```

3.  **仮想環境の作成と有効化:**
    ```bash
    # 仮想環境を作成 (例: .venv)
    uv venv .venv
    # 仮想環境を有効化
    source .venv/bin/activate  # Linux/macOS
    # .venv\\Scripts\\activate    # Windows (Cmd/Powershell)
    ```
    `.gitignore` に `.venv/` を追加してください。

4.  **依存関係のインストール:**
    `pyproject.toml` に基づいて依存関係をインストールします。**非同期DBアクセスに必要な `aiosqlite`、Webサーバー `FastAPI` `uvicorn`、LLM連携 `httpx` `jinja2`、MCPプロトコル `mcp` などが含まれます。**

    *   **(推奨) ロックファイルを使用する場合:**
        事前に推奨されるロックファイル (`requirements-lock.txt` など) を生成しておきます。
        ```bash
        # すべての依存関係をロックファイルにコンパイル (初回または更新時)
        uv pip compile pyproject.toml --all-extras -o requirements-lock.txt
        # ロックファイルに基づいて環境を同期
        uv pip sync requirements-lock.txt
        ```
        生成されたロックファイル (`requirements-lock.txt`) を Git にコミットしてください。

    *   **ロックファイルを使用しない場合:**
        `pyproject.toml` から直接インストールします。
        ```bash
        # 編集可能モード (-e) で、dev, numba, crepe オプションを含む依存関係をインストール
        uv pip install -e .[dev,numba,crepe]
        ```
        必要なオプション (`numba`, `crepe`) は適宜調整してください。

5.  **環境変数の設定:**
    `.env.example` をコピーして `.env` を作成し、必要な設定（特に**LLM APIキー (例: `ANTHROPIC_API_KEY`)**、ワークスペースパス `MIREX_WORKSPACE`、出力パス `MIREX_OUTPUT_BASE` など）を行います。`.env` ファイルは Git にコミットしないでください。
    ```bash
    cp .env.example .env
    # nano .env や vim .env などで編集
    ```
    環境変数は `config.yaml` の値を上書きします。詳細は `MCP_README.md` または `src/mcp_server_logic/core.py` の `load_config` 関数の Docstring を参照してください。

6.  **合成データ生成 (評価に必要):**
    ```bash
    python -m src.data_generation.generate_all
    ```
    これにより、`datasets/synthesized/` (または `config.yaml` や環境変数で指定されたパス) 以下に評価用の音声とラベルが生成されます。

## 4. 基本的な使用方法

本プロジェクトは、主に統合CLIツール `mirai` を通じて使用します。

```bash
python -m src.cli.mirai --help
```

### 4.1. スタンドアロン評価 (`mirai evaluate run`)
MCPサーバーを起動せずに、単一の音声ファイルと参照ファイルに対して、指定した検出器の評価を直接実行します。

```bash
python -m src.cli.mirai evaluate run \\
  --detector YourDetectorClassName \\
  --audio path/to/audio.wav \\
  --ref path/to/reference.csv \\
  --output results/standalone_eval \\
  --params '{"threshold": 0.6}' \\
  --save-plot
```

*   `--detector`: 評価する検出器のクラス名。
*   `--audio`, `--ref`: 評価対象の音声と参照ファイルのパス。
*   `--output`: (オプション) 結果（JSONやプロット）の保存先ディレクトリ。
*   `--params`: (オプション) 検出器のパラメータをJSON文字列で指定。
*   `--save-plot`: (オプション) 結果のプロットを保存。

詳細は `python -m src.cli.mirai evaluate run --help` を参照してください。

### 4.2. サーバー経由の評価・グリッドサーチ (`mirai evaluate run --server`, `mirai grid-search run --server`)
データセット全体に対する評価やパラメータのグリッドサーチを行う場合は、MCPサーバーを利用します。これにより、処理をバックグラウンドで実行し、複数のリクエストを管理できます。

**注意:** この方法を使用するには、事前に**MCPサーバー (`src/cli/mcp_server.py`) を起動しておく必要があります**。サーバーの起動方法については [`MCP_README.md`](MCP_README.md) を参照してください。

**評価の実行:**
```bash
# MCPサーバーが別ターミナルなどで起動している前提
python -m src.cli.mirai evaluate run \\
  --server http://localhost:5002 \\
  --detector YourDetectorClassName \\
  --dataset your_dataset_name \\
  --version optional_version_tag # 評価するコードのバージョン (省略時は最新)
```
*   コマンドはMCPサーバーに評価ジョブをリクエストし、完了を待ちます。
*   詳細は `python -m src.cli.mirai evaluate run --help` を参照してください。

**グリッドサーチの実行:**
```bash
# MCPサーバーが別ターミナルなどで起動している前提
# 1. グリッド設定ファイル (例: grid_config.yaml) を準備
# 2. コマンド実行
python -m src.cli.mirai grid-search run \\
  --server http://localhost:5002 \\
  --config path/to/grid_config.yaml
```
*   コマンドはMCPサーバーにグリッドサーチジョブをリクエストし、完了を待ちます。
*   詳細は `python -m src.cli.mirai grid-search run --help` を参照してください。

### 4.3. (実験的) AI駆動自動改善 (`mirai improve start --server`)
**注意:** これは実験的な機能です。セットアップ、詳細な使い方、内部ロジックについては **[`MCP_README.md`](MCP_README.md)** を参照してください。

MCPサーバーを利用して、CLIホスト側でLLM APIを直接呼び出し、評価、分析、仮説生成、コード改善/パラメータ最適化のサイクルを自動的に実行します。

```bash
# MCPサーバーが別ターミナルなどで起動している前提
# かつ .env または環境変数にLLM APIキー(ANTHROPIC_API_KEY)が設定されていること
python -m src.cli.mirai improve start \
  --server http://localhost:5002 \
  --detector YourDetectorClassName \
  --dataset your_dataset_name \
  --max-cycles 10 # 実行する最大サイクル数
```
*   `improve start` コマンドは、MCPアーキテクチャに準拠したCLIホスト主導の設計を採用しています：
    1. サーバーからプロンプトを取得
    2. ホスト側でLLM APIを直接呼び出し（APIキーはクライアント側で管理）
    3. 結果に基づいたアクション（コード改善、評価実行など）を実行
    4. アクション実行結果をセッション履歴として記録
*   コード変更時にはユーザー確認が必須で、変更内容を確認してから適用するため安全に利用できます
*   詳細は `python -m src.cli.mirai improve start --help` を参照してください。

## 5. 合成データセット詳細

### 5.1. 概要
アルゴリズム評価のために、様々な音響的特徴を持つ合成音声ファイル (`.wav`) と、それに対応する正解ラベルファイル (`.csv`) を生成します。

### 5.2. ラベルフォーマット (`.csv`)
各ラベルファイルはCSV形式で、以下の3列を持ちます（ヘッダー行なし）。
1.  **Onset (秒):** ノートの開始時間
2.  **Offset (秒):** ノートの終了時間
3.  **Frequency (Hz):** ノートの基本周波数 (f0)。ピッチを持たない音（パーカッション等）や無音区間は `0.0` で表現されます。
ポリフォニーは時間的に重複するノートとして、それぞれ別の行に記録されます。

### 5.3. アノテーションガイドライン
*   **Onset/Offset:** 音が知覚的に開始/終了する時間。
*   **Frequency:** ノート期間中の基本周波数 (f0)。`0.0` はピッチなし。
*   **評価許容誤差:** `mir_eval` のデフォルト値 (Onset: 50ms, Pitch: 50 cents, Offset: max(50ms, 20% duration)) が適用されます。`config.yaml` で変更可能です。

### 5.4. 生成ファイル一覧
多様なテストケースが含まれます。詳細は `src/data_generation/synthesizer.py` のコードを参照してください。(サイン波、倍音、ダイナミクス、ピッチ変調、ノイズ、リバーブ、和音、ポリフォニー、パーカッションなど)

## 6. 注意事項と既知の問題

*   **CREPE依存:** 一部検出器は `crepe` のインストールが必要です (`uv pip install .[crepe]`)。
*   **パフォーマンス:** 大規模評価やグリッドサーチ、AI改善ループは時間がかかることがあります。
*   **MCPサーバー依存:** `mirai` CLIの `--server` オプションを使用する機能はMCPサーバーの起動が前提です。スタンドアロン評価はサーバー不要です。
*   **パス検証:** クライアントからサーバーへ渡されるパスは、サーバー設定で許可された範囲内にあるか検証されます。
*   **(実験的機能):** AI駆動自動改善システムは開発途上であり、LLMの応答や戦略決定ロジックによって予期せぬ動作をする可能性があります。利用する場合は `MCP_README.md` を参照してください。

## 7. Dockerでの実行
`Dockerfile` を使用してコンテナイメージをビルド・実行できます。
```bash
# ビルド
docker build -t mirai_app .

# 実行 (例: サーバー起動 - 環境変数を .env から読み込む)
# .env ファイルに APIキーなどを設定しておく
docker run -it --rm -p 5002:5002 --env-file .env -v $(pwd)/output:/app/output -v $(pwd)/datasets:/app/datasets mirai_app python -m src.cli.mcp_server --host 0.0.0.0
# (ボリュームマウントは必要に応じて調整してください)
```
