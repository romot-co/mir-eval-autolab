# MIR アルゴリズム評価・自動改善プラットフォーム試験

## 1. プロジェクト概要

本プロジェクトは、音楽情報検索 (MIR) アルゴリズム、特に音符検出、オンセット検出、ピッチ推定などの性能評価と比較を容易にするためのフレームワークを提供します。

**8割Vibeコーディングで生成されています。**

**主な機能:**

*   **評価:** `mir_eval` ライブラリに基づいた客観的な評価指標（Precision, Recall, F-measure）を算出します。ノートベース評価（ポリフォニー対応）とフレームベース評価の両方をサポートします。
*   **合成データ生成:** 多様な合成音声データセット（ノイズ、リバーブ、ポリフォニー、パーカッションなど）を生成できます。
*   **評価実行:** 単一ファイルまたはディレクトリ全体、単一または複数の検出器、カスタムパラメータでの評価が可能です。
*   **結果の可視化:** 検出結果と参照データを比較するプロットを生成できます。
*   **パラメータ最適化:** グリッドサーチにより、検出器の最適なパラメータ設定を探索できます。

### 1.1 実験的機能：AI駆動自動改善システム (AutoImprover)

**注意:** 本プロジェクトには、AIを活用してMIRアルゴリズムの改善サイクルを自動化する**実験的な機能**が含まれています。これは、AIがアルゴリズム自体を発見・改善させる可能性を探求する試みであり、安定性や結果の品質は保証されません。
基本的にはLLM自体で手動で試すよりも品質が低く安定しないとお考えください。

**AI駆動自動改善システムの詳細なセットアップ、利用方法、トラブルシューティングについては、[`MCP_README.md`](MCP_README.md) を参照してください。**

## 2. ディレクトリ構成 (主要部分)

```
.
├── src/                     # MIR関連機能のソースコード
│   ├── detectors/           # 検出アルゴリズム実装 (.py)
│   ├── evaluation/          # 評価関連モジュール
│   │   └── grid_search/     # グリッドサーチ関連
│   ├── data_generation/     # 合成データ生成
│   ├── utils/               # 共通ユーティリティ (パス, 例外, etc.)
│   ├── visualization/       # 可視化関数
│   ├── science_automation/  # (実験的) 科学自動化戦略など
│   └── cli/                 # コマンドラインインターフェーススクリプト
│       ├── mcp_server.py        # (実験的) MCPサーバー起動スクリプト
│       ├── improver_cli.py      # 評価・グリッドサーチ用CLI
│       ├── auto_improver.py     # (実験的) 自動改善クライアント
│       └── setup_claude_integration.py # (実験的) Claude Desktop連携設定
├── mcp_server_logic/        # (実験的) MCPサーバーの中核ロジック
│   ├── core.py              # 設定読み込み、非同期ジョブワーカー、クリーンアップ等
│   ├── db_utils.py          # 非同期DB操作 (aiosqlite)
│   ├── job_manager.py       # 非同期ジョブキュー・状態管理
│   ├── session_manager.py   # 改善セッション管理
│   ├── llm_tools.py         # LLM連携ツール
│   ├── evaluation_tools.py  # 評価/グリッドサーチ実行ツール
│   ├── code_tools.py        # コード取得/保存ツール
│   ├── improvement_loop.py  # 改善ループ/戦略提案ツール
│   ├── visualization_tools.py # 可視化ツール
│   ├── serialization_utils.py # シリアライズ関連ユーティリティ
│   └── mcp_server_extensions.py # (オプション) 拡張ツール
├── tests/                   # テストコード
├── datasets/                # データファイル (設定ファイルや .env でパス指定)
├── templates/               # (実験的) プロンプトテンプレート等
├── mcp_workspace/           # MCPサーバーのデフォルト作業ディレクトリ
│   ├── db/                  # データベースファイル (mcp_server_state.db)
│   ├── improved_versions/   # AIが改善したコード
│   ├── results/             # 評価・グリッドサーチ結果
│   └── logs/                # MCPサーバーログなど
├── pyproject.toml           # プロジェクト設定、依存関係 (uv が使用)
├── config.yaml              # プロジェクト全体の設定ファイル
├── requirements-lock.txt    # (推奨) 生成された依存関係ロックファイル
├── .env.example             # 環境変数設定のテンプレート
├── Dockerfile               # Dockerイメージビルド用 (uv 使用)
├── MCP_README.md            # (実験的) MCPサーバー/AutoImprover向け詳細README
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
    # .venv\Scripts\activate    # Windows (Cmd/Powershell)
    ```
    `.gitignore` に `.venv/` を追加してください。

4.  **依存関係のインストール:**
    `pyproject.toml` に基づいて依存関係をインストールします。**非同期DBアクセスに必要な `aiosqlite` なども含まれます。**

    *   **(推奨) ロックファイルを使用する場合:**
        事前に推奨されるロックファイル (`requirements-lock.txt` など) を生成しておきます。
        ```bash
        # すべての依存関係をロックファイルにコンパイル (初回または更新時)
        uv pip compile pyproject.toml --all-extras -o requirements-lock.txt
        # ロックファイルに基づいて環境を同期
        uv pip sync requirements-lock.txt
        ```
        または、基本依存と開発依存を分けて管理する場合:
        ```bash
        uv pip compile pyproject.toml -o requirements.txt
        uv pip compile pyproject.toml --extra dev -o requirements-dev.txt
        uv pip sync requirements.txt requirements-dev.txt
        ```
        生成されたロックファイル (`requirements*.txt`) を Git にコミットしてください。

    *   **ロックファイルを使用しない場合:**
        `pyproject.toml` から直接インストールします。
        ```bash
        # 編集可能モード (-e) で、dev, numba, crepe オプションを含む依存関係をインストール
        uv pip install -e .[dev,numba,crepe]
        ```
        必要なオプション (`numba`, `crepe`) は適宜調整してください。

5.  **環境変数の設定:**
    `.env.example` をコピーして `.env` を作成し、必要な設定（特にAPIキー、ワークスペースパスなど）を行います。`.env` ファイルは Git にコミットしないでください。
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

## 4. 基本的な使用方法 (評価とグリッドサーチ)

コマンドラインからの操作は `src/cli/improver_cli.py` を使用します。
**注意:** 以下のコマンド (`evaluate`, `grid-search`) は、バックグラウンドで動作する**非同期MCPサーバー (`src/cli/mcp_server.py`)** のAPIツールを呼び出します。そのため、これらのコマンドを実行する前に **MCPサーバーが起動している必要があります** (詳細は `MCP_README.md` 参照)。サーバーはリクエストを受け付けると、評価やグリッドサーチを**非同期ジョブ**として実行します。

### 4.1. アルゴリズム評価 (`evaluate` コマンド)

指定した検出器の性能を評価します。

**基本的な使い方:**

```bash
# MCPサーバーが別ターミナルなどで起動している前提
python -m src.cli.improver_cli evaluate --detector PZSTDDetector --dataset synthesized_v1 --output-dir mcp_workspace/results/pzstd_eval
```

*   `--detector`: 評価する検出器のクラス名を指定します。
*   `--dataset`: `config.yaml` で定義されたデータセット名を指定します。
*   `--output-dir`: 結果（JSONやプロット）の保存先ディレクトリを指定します。**注意: このパスはサーバー側の `mcp_workspace/results` (または設定による) 配下に解釈されます。** クライアント側で絶対パスを指定しても、サーバー側で検証・調整される可能性があります。
*   **(オプション) 個別ファイル指定:** `--audio-path` と `--ref-path` でファイルを直接指定します (`--dataset` と排他)。指定されたパスはサーバー側で検証されます。
*   **(オプション) パラメータ指定 (JSON形式):** `--detector-params` でパラメータを上書きします。
*   **(オプション) 保存設定:** `--plot/--no-plot`, `--json/--no-json`。
*   **(オプション) 評価指標の選択:** `--note/--no-note`, `--pitch/--no-pitch`, `--frame/--no-frame`。

詳細は `python -m src.cli.improver_cli evaluate --help` を参照してください。

### 4.2. パラメータグリッドサーチ (`grid-search` コマンド)

検出器の最適なパラメータ組み合わせを探索します。

**ステップ 1: グリッド設定ファイルの作成 (YAML形式)**

例: `grid_config.yaml`
```yaml
detector_name: PZSTDDetector
param_grid:
  f0_score_threshold: [0.15, 0.20, 0.25, 0.30]
  # ... 他のパラメータ ...
dataset_name: synthesized_v1
```

**ステップ 2: グリッドサーチの実行**

```bash
# MCPサーバーが別ターミナルなどで起動している前提
python -m src.cli.improver_cli grid-search grid_config.yaml --output-dir results/pzstd_gridsearch --best-metric note.f_measure
```

*   最初の引数にグリッド設定ファイルのパスを指定します。パスはサーバー側で検証されます。
*   `--output-dir`: 結果の保存先。パスはサーバー側で検証されます。
*   `--best-metric`: 最適化基準とする指標。
*   `--num-procs`: (オプション) 並列実行プロセス数。
*   `--skip-existing`: (オプション) 既存結果のスキップ。

詳細は `python -m src.cli.improver_cli grid-search --help` を参照してください。

## 5. 合成データセット詳細

### 5.1. 概要

アルゴリズム評価のために、様々な音響的特徴を持つ合成音声ファイル (`.wav`) と、それに対応する正解ラベルファイル (`.csv`) を生成します。

### 5.2. ラベルフォーマット (`.csv`)

各ラベルファイルはCSV形式で、以下の3列を持ちます（ヘッダー行なし）。

1.  **Onset (秒):** ノートの開始時間
2.  **Offset (秒):** ノートの終了時間
3.  **Frequency (Hz):** ノートの基本周波数 (f0)。ピッチを持たない音（パーカッション等）や無音区間は `0.0` で表現されます。

**ポリフォニー:** 時間的に重複するノートは、それぞれ別の行として記録されます。

### 5.3. アノテーションガイドライン

*   **Onset:** 音が知覚的に開始される時間。
*   **Offset:** 音が知覚的に終了する時間（リバーブ含まず、Release成分は一部含む）。
*   **Frequency:** ノート期間中の基本周波数 (f0)。ピッチが変動する場合は代表値を記録。`0.0` はピッチなし。
*   **評価許容誤差:** `mir_eval` のデフォルト値 (Onset: 50ms, Pitch: 50 cents, Offset: max(50ms, 20% duration)) が適用されます。

### 5.4. 生成ファイル一覧

多様なテストケースが含まれます。詳細は `src/data_generation/synthesizer.py` のコードを参照してください。(基本的なサイン波、倍音、ダイナミクス変化、ピッチ変調、ノイズ、リバーブ、和音、ポリフォニー、パーカッション、スタッカート、レガートなど)

## 6. 注意事項と既知の問題

*   **CREPE依存:** `CriteriaDetector` など一部検出器は `crepe` のインストールが必要です。
*   **パフォーマンス:** 大規模評価やグリッドサーチは時間がかかることがあります。
*   **MCPサーバー依存:** `improver_cli.py` の `evaluate`, `grid-search` コマンドはMCPサーバーの起動を前提とします。
*   **パス検証:** クライアントからサーバーへ渡されるファイルパスは、サーバーの `config.yaml` で許可されたディレクトリ内にあるか検証されます。
*   **実験的機能:** AI駆動自動改善システム (MCPサーバー, AutoImprover) は開発途上です。利用する場合は `MCP_README.md` を参照してください。

## 7. Dockerでの実行

`Dockerfile` を使用して、コンテナ内でアプリケーションを実行できます。

1.  **イメージのビルド:**
    `uv` とロックファイル (`requirements-lock.txt`) を使用してビルドします。
    ```bash
    # 事前にロックファイルを生成
    uv pip compile pyproject.toml --all-extras -o requirements-lock.txt
    # Docker イメージをビルド
    docker build -t mirex-auto-improver .
    ```

2.  **コンテナの実行 (例: MCPサーバー):**
    ```bash
    docker run -p 5002:5002 \
           -e ANTHROPIC_API_KEY="<あなたのAPIキー>" \
           -v $(pwd)/mcp_workspace:/app/mcp_workspace \
           mirex-auto-improver \
           python -m src.cli.mcp_server
    ```
    *   `-p`: ポートをホストにマッピングします。
    *   `-e`: APIキーなどの環境変数を渡します。
    *   `-v`: ホストのワークスペースディレクトリをコンテナにマウントします。

(CLIツールの実行など、他のユースケースについては `Dockerfile` を参照してください)
