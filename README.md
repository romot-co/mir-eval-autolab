はい、承知いたしました。
`README.md` はプロジェクト全体の概要と基本的な評価機能に焦点を当て、`MCP_README.md` は実験的なAI自動改善システムの詳細に特化するように、それぞれの内容を整理し、省略せずに完全な形で出力します。

---

**`README.md` (修正版)**

```markdown
# MIR アルゴリズム評価・自動改善プラットフォーム試験

## 1. プロジェクト概要

本プロジェクトは、音楽情報検索 (MIR) アルゴリズム、特に音符検出、オンセット検出、ピッチ推定などの性能評価と比較を容易にするためのフレームワークを提供します。

**主な機能:**

*   **標準化された評価:** `mir_eval` ライブラリに基づいた客観的な評価指標（Precision, Recall, F-measure）を算出します。ノートベース評価（ポリフォニー対応）とフレームベース評価の両方をサポートします。
*   **合成データ生成:** アルゴリズムの頑健性をテストするための多様な合成音声データセット（ノイズ、リバーブ、ポリフォニー、パーカッションなど）を簡単に生成できます。
*   **柔軟な評価実行:** 単一ファイルまたはディレクトリ全体、単一または複数の検出器、カスタムパラメータでの評価が可能です。
*   **結果の可視化:** 検出結果と参照データを比較するプロットを生成できます。
*   **パラメータ最適化:** グリッドサーチにより、検出器の最適なパラメータ設定を探索できます。

### 1.1 実験的機能：AI駆動自動改善システム (AutoImprover)

**注意:** 本プロジェクトには、AIを活用してMIRアルゴリズムの改善サイクルを自動化する**実験的な機能**が含まれています。これは、AIがアルゴリズム自体を発見・進化させる可能性を探求する試みであり、活発な開発下にあります。安定性や結果の品質は保証されません。

**AI駆動自動改善システムの詳細なセットアップ、利用方法、トラブルシューティングについては、[`MCP_README.md`](MCP_README.md) を参照してください。**

## 2. ディレクトリ構成

```
.
├── src/                     # ソースコード
│   ├── detectors/           # 検出アルゴリズム実装 (.py)
│   │   ├── base_detector.py # 全検出器の基底クラス
│   │   └── improved_versions/ # (実験的) AIが生成した改善版コード
│   ├── evaluation/          # 評価関連モジュール
│   │   └── grid_search/     # グリッドサーチ関連
│   ├── data_generation/     # 合成データ生成
│   ├── science_automation/  # (実験的) 仮説生成、戦略提案など
│   ├── utils/               # 共通ユーティリティ
│   ├── visualization/       # 可視化関数
│   └── structures/          # データ構造 (Noteなど)
├── datasets/                # データファイル
│   ├── synthesized/         # 生成された合成データ
│   │   ├── audio/           # 音声ファイル (.wav)
│   │   └── labels/          # 正解ラベル (.csv)
│   └── mcp_state.db         # (実験的) MCPサーバーの状態DB
├── evaluation_results/      # 評価結果の出力先
├── grid_search_results/     # グリッドサーチ結果の出力先
├── mcp_workspace/           # (実験的) MCP/AutoImproverの作業ディレクトリ
│   ├── improvement_states/  # (実験的) AutoImproverのセッション状態
│   └── ...
├── scientific_output/       # (実験的) 生成された科学的成果物 (仮説、論文案)
├── templates/               # (実験的) 論文・レポートテンプレート
├── improver_cli.py          # メインのコマンドラインインターフェース
├── mcp_server.py            # (実験的) MCP/AI連携用APIサーバー
├── auto_improver.py         # (実験的) 自動改善サイクル実行クライアント
├── run_evaluate.py          # (旧) 評価実行スクリプト (improver_cli.py evaluate を使用推奨)
├── run_grid_search.py       # (旧) グリッドサーチ実行スクリプト (improver_cli.py grid-search を使用推奨)
├── setup_claude_integration.py # (実験的) Claude Desktop連携設定ツール
├── requirements.txt         # Python依存ライブラリ
├── config.yaml              # プロジェクト全体の設定ファイル
├── MCP_README.md            # (実験的) MCPサーバー/AutoImprover向けREADME
└── README.md                # このファイル
```

## 3. セットアップ

1.  **リポジトリのクローン:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **環境設定:**
    *   `.env.example` をコピーして `.env` を作成し、必要に応じてAPIキーなどを設定します。
        ```bash
        cp .env.example .env
        ```

3.  **依存関係のインストール:**
    *   Poetry を使用している場合:
        ```bash
        poetry install --with dev
        ```
    *   pip を使用している場合:
        ```bash
        pip install -r requirements.txt
        pip install -r requirements-dev.txt # 開発用依存関係
        ```
    *   **CREPE に関する注意:** 評価等で CREPE を使用する場合、別途インストールが必要です。
        ```bash
        pip install crepe
        ```
        TensorFlowのバージョン互換性により問題が発生する場合があります。詳細は [CREPEのドキュメント](https://github.com/marl/crepe) を参照してください。

4.  **データベースの初期化:**

5.  **合成データ生成 (評価に必要):**
    ```bash
    python src/data_generation/generate_all.py
    ```
    これにより、`datasets/synthesized/` 以下に評価用の音声とラベルが生成されます。

## 4. 基本的な使用方法 (評価とグリッドサーチ)

コマンドラインからの操作は `improver_cli.py` を使用します。

### 4.1. アルゴリズム評価 (`evaluate` コマンド)

指定した検出器の性能を評価します。

**基本的な使い方:**

```bash
python improver_cli.py evaluate --detector PZSTDDetector --dataset synthesized_v1 --output-dir results/pzstd_eval
```

*   `--detector`: 評価する検出器のクラス名を指定します (`src/detectors/` 内のクラス)。
*   `--dataset`: `config.yaml` で定義されたデータセット名を指定します。`audio_dir` と `label_dir` が自動的に読み込まれます。
*   `--output-dir`: 結果（JSONやプロット）の保存先ディレクトリを指定します。
*   **(注意)** このコマンドは、バックグラウンドでMCPサーバーの `run_evaluation` ツールをAPI経由で呼び出します。サーバーが起動している必要があります。
*   **(オプション) 個別ファイル指定:** `--audio-path` と `--ref-path` で音声ファイルとラベルファイルを直接指定します。`--dataset` とは同時に使用できません。
    ```bash
    python improver_cli.py evaluate --detector PZSTDDetector --audio-path datasets/synthesized/audio/1_basic_sine.wav --ref-path datasets/synthesized/labels/1_basic_sine.csv --output-dir results/single_eval
    ```
*   **(オプション) パラメータ指定 (JSON形式):** `--detector-params` で検出器のパラメータを上書きします。
    ```bash
    python improver_cli.py evaluate --detector PZSTDDetector --dataset synthesized_v1 --output-dir results/pzstd_tuned --detector-params '{"f0_score_threshold": 0.3, "min_note_duration_sec": 0.06}'
    ```
*   **(オプション) 保存設定:** `--save-plots` フラグでプロット画像を、`--save-results-json` フラグで評価結果のJSONファイルを保存できます（デフォルトは両方オフ）。
*   **(オプション) 評価指標の選択:** `--note/--no-note`, `--pitch/--no-pitch`, `--frame/--no-frame` で計算・表示する評価指標カテゴリを選択できます（デフォルトは全て有効）。

詳細は `python improver_cli.py evaluate --help` を参照してください。

### 4.2. パラメータグリッドサーチ (`grid-search` コマンド)

検出器の最適なパラメータ組み合わせを探索します。

**ステップ 1: グリッド設定ファイルの作成 (YAML形式)**

探索したいパラメータとその候補値を記述したYAMLファイル（例: `grid_config.yaml`）を作成します。

```yaml
# grid_config.yaml の例
detector_name: PZSTDDetector     # 最適化する検出器名
param_grid:                    # 探索するパラメータと候補値のリスト
  f0_score_threshold: [0.15, 0.20, 0.25, 0.30]
  hcf_onset_peak_thresh: [0.10, 0.12, 0.15, 0.18]
  harmonic_match_tolerance_cents: [25.0, 30.0, 35.0, 40.0]
dataset_name: synthesized_v1   # グリッドサーチで使用するデータセット名 (config.yamlで定義)
# evaluator_config:             # (オプション) 評価設定の上書き
#   tolerance_onset: 0.04```

**ステップ 2: グリッドサーチの実行**

作成した設定ファイルを使ってグリッドサーチを実行します。

```bash
python improver_cli.py grid-search grid_config.yaml --output-dir results/pzstd_gridsearch --best-metric note.f_measure
```

*   最初の引数にグリッド設定ファイル (`grid_config.yaml`) のパスを指定します。
*   `--output-dir`: 全ての組み合わせの結果とサマリーが保存されるディレクトリを指定します。省略すると設定ファイル名に基づいて自動生成されます。
*   `--best-metric`: 最適化の基準とする評価指標を指定します (`カテゴリ.メトリック名` 形式、例: `note.f_measure`, `onset.recall`)。
*   **(注意)** このコマンドは、バックグラウンドでMCPサーバーの `run_grid_search` ツールをAPI経由で呼び出します。サーバーが起動している必要があります。
*   `--num-procs`: (オプション) 並列実行するプロセス数を指定します。
*   `--skip-existing`: (オプション) 既に出力ディレクトリに結果が存在する組み合わせをスキップします。

実行後、指定した出力ディレクトリに以下のファイルが生成されます。

*   `grid_search_results.csv`: 各パラメータ組み合わせの評価結果一覧。
*   `best_params.json`: 最適化指標 (`best_metric`) に基づいて最も良かったパラメータ設定。
*   `combined_evaluation_summary.json`: グリッドサーチ全体のサマリー情報。
*   各試行ごとの詳細な評価結果（サブディレクトリ内）。

詳細は `python improver_cli.py grid-search --help` を参照してください。

## 5. 合成データセット詳細 (`datasets/synthesized/`)

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

*   **CREPE依存:** `CriteriaDetector` を使用するには `crepe` ライブラリのインストールが必要です。
*   **パフォーマンス:** データセット全体の評価や広範なグリッドサーチは時間がかかることがあります。
*   **オフセット検出の限界:** `CriteriaDetector` のデフォルトのオフセット検出方法は単純なRMSベースであり、特にリバーブ環境などでは精度が低下する可能性があります。
*   **実験的機能:** AI駆動自動改善システムは開発途上です。利用する場合は `MCP_README.md` を参照してください。
