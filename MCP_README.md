# MIR アルゴリズム自動改善研究室

## 1. システム概要

本システムは音楽情報検索（MIR）アルゴリズムの開発、評価、改善を完全に自動化する「全自動改善研究室」です。Claude AIと連携し、アルゴリズムの継続的な改善サイクルを実現します。

### 1.1 主要機能

- アルゴリズムの自動読み込みと分析
- 合成音声データを使用した評価の自動実行
- AIを活用したコード改善
- パラメータの自動最適化
- 改善度合いの評価と版管理

### 1.2 動作フロー

1. 指定された検出器アルゴリズムのコードを読み込み
2. 初期評価を実行し、基準となるF値を記録
3. LLMにコードと評価結果を送信して改善案を取得
4. 改善されたコードを保存して再評価
5. パラメータ最適化を実行（改善が有望な場合）
6. 旧版と新版のF値を比較して採用/破棄を決定
7. 指定された回数だけこのサイクルを繰り返す

## 2. セットアップ手順

### 2.1 依存ライブラリのインストール

```bash
# 必須ライブラリ
pip install mcp requests pandas numpy waitress

# オプション（可視化用）
pip install matplotlib seaborn
```

### 2.2 ワークスペースの準備

本システムは以下のディレクトリ構造で動作します：

```
<workspace>/
  ├── src/detectors/              # 検出器アルゴリズム
  │   └── improved_versions/      # 改善版検出器保存先
  ├── data/                       # テストデータ
  │   ├── synthesized/            # 合成テストデータ
  │   │   ├── audio/              # 音声ファイル
  │   │   └── labels/             # 正解ラベル
  ├── evaluation_results/         # 評価結果格納先
  └── grid_search_results/        # パラメータ探索結果
```

設定に従って、自動的に必要なディレクトリが作成されます。

### 2.3 API設定（オプション）

Anthropic Claude APIを使用する場合は、環境変数に設定します：

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

API設定がない場合、システムはモック応答で動作します。

### 2.4 Claude Desktop用設定

Claude Desktopと連携するための設定を自動生成します：

```bash
python setup_claude_integration.py --open-claude
```

このコマンドは:
- Claude Desktop設定ファイルを生成/更新
- サンプル検出器ファイルを作成（存在しない場合）
- 一時作業ディレクトリを準備
- Claude Desktopアプリを起動（--open-claudeオプション指定時）

## 3. 利用方法

### 3.1 MCPサーバーの起動

直接サーバーを起動する場合（通常はClaude Desktopが自動的に起動します）：

```bash
python mcp_server.py
```

別のポートを使用する場合：

```bash
export MCP_PORT=5003
python mcp_server.py
```

### 3.2 コマンドラインからの自動改善サイクル実行

改善プロセスを完全自動化する場合：

```bash
python auto_improver.py --detector PZSTDDetector --goal "ノイズ耐性を向上させる" --iterations 3
```

オプション:
- `--detector`: 改善対象の検出器名（`src/detectors/`内のファイル名から.pyを除いたもの）
- `--goal`: 改善目標を自然言語で記述
- `--iterations`: 改善サイクルの繰り返し回数（デフォルト: 3）
- `--server-url`: MCPサーバーのURL（デフォルト: http://localhost:5002）

### 3.3 Claude Desktopでの対話的操作

1. Claude Desktopを起動
2. ハンマーアイコン(🔨)をクリックして利用可能なツールを確認
3. 以下のようなプロンプトを入力：

```
PZSTDDetectorのリバーブ環境での精度を向上させてください。
以下のステップで改善を行ってください：
1. まずstart_sessionでセッションを開始し
2. run_evaluationで現状の性能を評価し
3. improve_codeでコードを改善して
4. 再度評価を行って改善度を確認してください
```

### 3.4 利用可能なMCPツール一覧

MCPサーバーは以下のツールを提供します：

1. `health_check`: サーバー動作確認
2. `start_session`: 改善セッション開始（検出器名指定）
3. `get_session_info`: セッション情報取得
4. `add_session_history`: セッション履歴追加
5. `get_job_status`: 非同期ジョブ状態確認
6. `improve_code`: コード改善依頼
7. `run_evaluation`: 検出器評価実行
8. `run_grid_search`: パラメータ探索実行 (F値向上時に自動実行)
9. `analyze_code_segment`: コード分析
10. `suggest_parameters`: パラメータ提案

## 4. 新機能：堅牢化と高度な自動化

### 4.1 パス整合性の強化

異なる環境でも正しく動作するよう、パス管理が強化されました：

- プロジェクトルートを絶対パスで特定するロジックの追加
- 作業ディレクトリの柔軟な設定（環境変数、ホームディレクトリ、一時ディレクトリの順で試行）
- 権限エラー時の自動フォールバック機能

### 4.2 動的ロードの堅牢性向上

検出器モジュールのロード機能が強化されました：

- `DETECTOR_CLASS_NAME` メタデータによる明示的なクラス指定機能追加
- 改良コード生成時にメタデータを自動挿入
- 複数クラス存在時の曖昧さ解消

### 4.3 自動パラメータ最適化

改善サイクルに自動グリッドサーチが統合されました：

- F値が5%以上向上した場合に自動実行
- 主要パラメータの最適化を実行
- 最適化結果を自動的に評価し採用

### 4.4 セッション管理とエラーハンドリング

安定性と継続性を高めるため、セッション管理が強化されました：

- 古いセッションとジョブの自動クリーンアップ
- タイムアウト設定による資源の効率的管理
- 連続失敗検出による無限ループ防止
- 詳細なエラーログ記録

## 5. 自動改善プロセスの詳細

### 5.1 検出器コードの読み込み

```python
detector_path = f"src/detectors/{detector_name}.py"
with open(detector_path, "r") as f:
    original_code = f.read()
```

対象となる検出器のコードを`src/detectors/`ディレクトリから読み込みます。

### 5.2 初期評価の実行

```python
base_eval = self.run_evaluation(detector_name)
if base_eval:
    best_f_measure = base_eval.get("overall_metrics", {}).get("note", {}).get("f_measure", 0)
    logger.info(f"ベースライン F-measure: {best_f_measure:.4f}")
```

`run_evaluate.py`を内部的に呼び出し、検出器の初期性能を評価します。ここでは:
- 検出器クラスをインポート
- 合成テストデータに対して検出処理実行
- F値、適合率、再現率などの評価指標を算出
- 問題があるファイルの分析

### 5.3 LLMによるコード改善

```python
improved_code = self.request_code_improvement(current_code, improvement_goal)
improved_file = self.save_improved_code(improved_code, detector_name, version)
```

1. 現在のコードと改善目標をLLMに送信
2. LLMがコード分析と改善提案を生成
3. 改善されたコードを取得
4. メタデータを追加して保存

### 5.4 改善コードの評価と自動グリッドサーチ

```python
eval_result = self.run_evaluation(detector_name)
current_f_measure = eval_result.get("overall_metrics", {}).get("note", {}).get("f_measure", 0)

# 改善が一定以上の場合、自動的にグリッドサーチを実行
if current_f_measure > best_f_measure:
    improvement_percentage = ((current_f_measure - best_f_measure) / best_f_measure * 100)
    if improvement_percentage >= 5.0:
        logger.info(f"有望な改善を検出 (+{improvement_percentage:.2f}%) - グリッドサーチを実行中...")
        grid_search_result = self.run_grid_search(detector_name, json.dumps(default_grid_params))
```

### 5.5 最終判断と保存

```python
if current_f_measure > best_f_measure:
    logger.info(f"改善を検出: {best_f_measure:.4f} -> {current_f_measure:.4f}")
    best_f_measure = current_f_measure
    best_code = improved_code
else:
    logger.info(f"改善なし: {best_f_measure:.4f} -> {current_f_measure:.4f}")
    current_code = best_code  # 最良のコードを維持
```

## 6. 評価データセット

`data/synthesized/` ディレクトリには以下のテストデータが含まれています：

### 6.1 音声ファイル (`audio/`)

様々な音響特性を持つ合成波形：
- 基本波形（サイン波など）
- ノイズ付き音声（SNR比の異なるもの）
- リバーブ付き音声（短い/長いリバーブ）
- 和音、ポリフォニー音声
- パーカッション音声
- ビブラート、ポルタメントなどの表現技法

### 6.2 正解ラベル (`labels/`)

各音声ファイルに対応するCSV形式の正解データ：
- 1列目: オンセット（音符開始時間、秒単位）
- 2列目: オフセット（音符終了時間、秒単位）
- 3列目: 周波数（Hz単位、パーカッションなどピッチなしの場合は0）

例：
```
0.500,1.000,440.00
1.200,1.700,523.25
2.000,2.500,587.33
```

## 7. 実装詳細

### 7.1 MCPサーバー (`mcp_server.py`)

- Flask/MCPベースのRESTful APIサーバー
- 非同期ジョブ管理機能
- セッション管理機能とクリーンアップ
- パス整合性を保証する機能
- Claude API連携（コード改善）
- 評価スクリプト実行

### 7.2 自動改善ツール (`auto_improver.py`)

- 改善サイクル全体の制御
- 自動パラメータ最適化の実行
- MCPサーバーAPI呼び出し
- 結果比較ロジック
- エラー検出と再試行機能
- 履歴管理

### 7.3 Claude Desktop設定ツール (`setup_claude_integration.py`)

- Claude Desktop用設定ファイルの生成
- サンプル検出器の作成
- 一時作業ディレクトリの設定
- 環境変数設定

## 8. トラブルシューティング

### 8.1 一般的な問題と解決策

#### ファイルシステム権限エラー
```
OSError: [Errno 30] Read-only file system: 'mcp_workspace'
```
**解決策**: システムは自動的に書き込み可能なディレクトリ（ホームディレクトリまたは一時ディレクトリ）に切り替わります。ログを確認してください。

#### ポート競合
```
OSError: Address already in use
```
**解決策**: 環境変数で別のポートを指定します。
```bash
export MCP_PORT=5003
```

#### LLM API呼び出しエラー
```
LLM API request error
```
**解決策**: 
- APIキーが正しく設定されているか確認
- インターネット接続を確認
- API制限に達していないか確認

### 8.2 ログ確認方法

```bash
# MCPサーバーログ
cat mcp_server.log

# Claude Desktop連携ログ
tail -n 20 -f ~/Library/Logs/Claude/mcp*.log

# 設定ツールログ
cat claude_setup.log
```

## 9. 実際の使用例

### 9.1 ノイズ耐性改善

```bash
python auto_improver.py --detector PZSTDDetector --goal "ノイズが存在する環境でも正確に音符を検出できるよう改善する。特に5_noisy系列のファイルでの性能向上を目指す" --iterations 3
```

### 9.2 リバーブ対応強化

```bash
python auto_improver.py --detector PZSTDDetector --goal "リバーブが長い環境での検出精度を高める。特に6_reverb_long.wavでの性能を優先的に向上させる" --iterations 3
```

### 9.3 和音検出改善

```bash
python auto_improver.py --detector PZSTDDetector --goal "和音検出の精度を向上させる。特に7_chords.wavと8_polyphony.wavでの検出漏れを減らす" --iterations 3
```

## 10. ファイル名と実行順序

システムのファイル構成は以下の通りです：

1. `setup_claude_integration.py`（旧: run_auto_improver.py）
   - Claude Desktop設定を準備
   - サンプル検出器を作成
   - 実行順序: 最初に実行

2. `mcp_server.py`
   - MCP対応APIサーバー
   - Claude Desktopからは自動起動
   - 手動実行する場合は独立して起動

3. `auto_improver.py`
   - 自動改善サイクルを実行
   - コマンドラインから直接実行

4. `src/detectors/__init__.py`
   - 検出器の動的読み込みと登録
   - システムの一部として自動的に使用

5. `run_evaluate.py` と `run_grid_search.py`
   - 評価とパラメータ探索を実行
   - MCPサーバーから呼び出される

## 11. まとめ

本システムは「全自動改善研究室」として以下を実現します：

1. ✅ **改善対象アルゴリズムの読み込みと分析**
   - 検出器コードを自動的に読み込み
   - コード構造と目的を分析

2. ✅ **旧版と評価結果の保存**
   - 初期バージョンを保存
   - 評価指標のベースラインを記録

3. ✅ **LLMによる実際のコード修正と実行**
   - Claude APIでコード改善を取得
   - メタデータを追加して実行可能なコードを生成
   - 実際に動作確認を行う

4. ✅ **有望な場合の自動パラメータサーチ**
   - 改善が5%以上の場合に自動実行
   - 最適化されたパラメータを特定

5. ✅ **旧版と比較して改善判断**
   - F値による定量的比較
   - 改善した場合は採用、さもなければ破棄
   - 最終的な最良バージョンを選定

これで「全自動改善研究室」として必要な全機能が強化された堅牢なシステムが実現されています。
