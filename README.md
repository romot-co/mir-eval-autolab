# MIR Algorithm Automatic Improvement Experience

## 1. プロジェクト概要

このプロジェクトは、音楽情報検索（MIR）アルゴリズム、特に音符検出（Note Detection）、オンセット検出（Onset Detection）、ピッチ推定（Pitch Estimation）などの性能を評価するためのフレームワークを提供します。

合成データセットの生成機能と、複数の検出アルゴリズムを一括で評価するスクリプトが含まれています。

### 1.1 実験的機能：AI駆動自動改善システム

**注意: この機能は実験的段階にあり、安定性は保証されていません。**

本プロジェクトには、Claude AIと連携してアルゴリズムの継続的な改善を試みる「全自動改善研究室」機能が含まれています。この実験的機能は以下を提供します：

- アルゴリズムの自動読み込みと分析
- 合成音声データを使用した評価の自動実行
- AIを活用したコード改善
- パラメータの自動最適化
- 改善度合いの評価と版管理

## 2. ディレクトリ構成

- `src/`: 主要なソースコード
  - `detectors/`: 各種検出アルゴリズムの実装
    - `improved_versions/`: AI改善による検出器の保存先（実験的）
  - `evaluation/`: 評価ロジック（`mir_eval` を利用）
  - `data_generation/`: 合成データ生成スクリプト
  - `utils/`: 共通ユーティリティ関数
  - `cli/`: コマンドラインインターフェース関連
  - `visualization/`: 結果可視化用プロット関数
- `data/`: データファイル
  - `synthesized/`: 生成された合成音声 (`audio/`) とラベル (`labels/`)
- `configs/`: 検出器のパラメータ設定ファイル (`.yaml`)
- `evaluation_results/`: 評価スクリプトの出力結果（JSON、プロット画像）
- `grid_search_results/`: パラメータグリッドサーチ結果
- `mcp_workspace/`: AI自動改善用ワークスペース（実験的）

## 3. セットアップと実行方法

### 3.1. セットアップ

必要なPythonライブラリをインストールします。

```bash
pip install -r requirements.txt
```

### 3.2. 合成データの生成

評価に使用する合成データセットを生成します。

```bash
python src/data_generation/generate_all.py
```

生成された音声ファイルは `data/synthesized/audio/` に、対応するラベルファイルは `data/synthesized/labels/` に保存されます。

### 3.3. 評価の実行

指定した検出器で評価を実行します。

```bash
# 例: PZSTDDetector で合成データを評価
python run_evaluate.py --audio-dir data/synthesized/audio \
                       --reference-dir data/synthesized/labels \
                       --reference-pattern "*.csv" \
                       --detectors PZSTDDetector \
                       --output-dir evaluation_results/eval_objective \ 
                       --save-plots --save-results-json
```

- `--detectors`: 評価したい検出器名をカンマ区切りで指定します（クラス名）。
- `--audio-dir`, `--reference-dir`: 評価対象の音声とラベルのディレクトリを指定します。
- `--reference-pattern`: ラベルファイルのパターンを指定します。
- `--save-plots`: 結果のプロット画像を保存します。
- `--save-results-json`: 評価結果をJSONファイルに保存します。

詳細は `python run_evaluate.py --help` を参照してください。

## 4. 合成データセット (`data/synthesized/`)

### 4.1. 概要

アルゴリズム評価のために、様々な音響的特徴を持つ合成音声ファイル (`.wav`) と、それに対応する正解ラベルファイル (`.csv`) を生成します。

### 4.2. ラベルフォーマット (`.csv`)

各ラベルファイルはCSV形式で、以下の列を持ちます（ヘッダー行は存在しない場合があります）。

1.  **Onset (秒):** ノートの開始時間
2.  **Offset (秒):** ノートの終了時間
3.  **Frequency (Hz):** ノートの基本周波数 (f0)

**重要:** ポリフォニー（複数ノートの同時発音）は、時間的に重複する区間を持つ複数の行として表現されます。
例えば、和音の場合は同じ時間区間（同じOnset-Offset）に異なるFrequencyを持つ複数の行が存在します。
また、独立した複数のメロディラインの場合は、時間的に重複する異なるOnset-Offsetの区間が存在することがあります。

### 4.3. アノテーションガイドライン

ラベル付けの基準は以下の通りです。

- **オンセット (Onset):**
  - 音が知覚的に開始されると想定される時間（秒）です。
  - 現行のADSRエンベロープに基づく合成音では、Attackフェーズの開始時間に相当します。
  - ポリフォニーの場合、各ノートの実際の開始時間を個別に記録します。
- **オフセット (Offset):**
  - 音が**知覚的に終了する**と想定される時間（秒）です。これは通常、ADSRエンベロープの**Releaseフェーズがほぼ終了し、音が聞こえなくなる時点**に相当します。リバーブ成分は含みません。
  - 現在の実装では、Sustain終了時刻にRelease時間の80%を加えた時刻をオフセットとしています。
  - **重要:** リバーブ（残響）が付加されている場合でも、リバーブ成分は含まないオフセットを記録します。これは楽譜上の音符の長さに近い概念ではありますが、Release成分も含めて聴覚的に音が消えると判断される時点を指します。
  - ポリフォニーの場合、各ノートの実際の終了時間を個別に記録します。
- **周波数 (Frequency):**
  - ノート期間中の基本周波数 (f0) をHz単位で示します。
  - ピッチが一定でない場合（ビブラート、ポルタメントなど）、その区間の代表的な周波数、または区間開始時の周波数を記録します。詳細は各ファイル生成ロジックのコメントを参照してください。
  - **ピッチ = 0.000 Hz:** これはパーカッションのような明確なピッチを持たない非楽音成分、または意図的な無音区間を示します。評価アルゴリズムはこれをピッチ情報なしとして扱うことを想定しています。
  - ポリフォニーの場合、各ノートの周波数を個別の行として記録します。
- **評価時の許容誤差:**
  - `mir_eval` を用いた評価では、以下のデフォルト許容誤差が適用されます。
    - オンセット許容誤差: 50ミリ秒 (`onset_tolerance=0.05`)
    - ピッチ許容誤差: 50セント (`pitch_tolerance=50.0`)
    - オフセット許容誤差: ノート長の20% または 50ミリ秒 の大きい方 (`offset_ratio=0.2`, `offset_min_tolerance=0.05`)
  - ポリフォニーの評価では、`mir_eval.transcription` の関数群を使用し、参照ノートリストと推定ノートリスト間の最適なマッチングを行います。

### 4.4. 生成されるファイル一覧

- `1_basic_sine.wav`: 基本的なサイン波シーケンス。
- `2_harmonic_tone.wav`: 複数の倍音を含む音のシーケンス。
- `3_dynamics_change.wav`: 音量のクレッシェンド/デクレッシェンドを含む単一ノート。
- `4_pitch_mod.wav`: ビブラートとポルタメント（現状はステップ状）を含む単一ノート。
- `5_noisy_*.wav`: 様々なSNRのホワイトノイズが付加されたシーケンス。
- `6_reverb_*.wav`: 短い/長いリバーブが付加されたシーケンス。
- `7_chords.wav`: 和音シーケンス。
- `8_polyphony.wav`: 複数の独立したメロディライン（ポリフォニー）。
- `9_10_percussion_only.wav`: キック、スネア、ハイハットのみのシーケンス（ピッチは0）。
- `11_mixed_melody_perc.wav`: メロディとパーカッションのミックス。
- `12_complex_mix.wav`: ポリフォニー、パーカッション、ノイズ、リバーブを含む複雑なミックス。
- `13_smooth_vibrato.wav`: 滑らかなサイン波ビブラート（周期的なピッチ変動）を持つ単一ノート。
- `14_portamento.wav`: あるピッチから別のピッチへ滑らかに線形移行する（ポルタメント）単一ノート。
- `15_slow_attack.wav`: 非常に遅いアタックを持つ単一ノート（ストリングス風）。
- `16_staccato.wav`: 非常に短い発音長のノート（スタッカート）のシーケンス。
- `17_pianissimo.wav`: 非常に小さい音量のノートシーケンス（正規化なし）。
- `18_octave_errors.wav`: オクターブ違いの同じ音名が連続するシーケンス（オクターブエラー誘発）。
- `19_inharmonicity.wav`: 倍音周波数がわずかにずれる単一ノート（ピアノのインハーモニシティ模倣）。
- `20_legato.wav`: ノート間がわずかに重なり滑らかに繋がる（レガート/クロスフェード）シーケンス。
- `21_vocal_imitation.wav`: ピッチの揺らぎと簡易的なフォルマントを持つボーカル模倣音。
- `22_clicks.wav`: 基本的なメロディにランダムなクリックノイズが付加されたシーケンス。

## 5. パラメータグリッドサーチ

検出器のパラメータを最適化するためのグリッドサーチツールを提供します。

### 5.1. グリッド設定ファイルの生成

`run_grid_search.py create-config` コマンドを使用して、パラメータ候補値を指定するYAMLファイル（`grid_config.yaml`）を生成します。

**必須引数:**

*   `--detector`: 最適化対象の検出器名（例：`PZSTDDetector`）
*   `--audio-dir`: 評価用音声ファイルのディレクトリ（例：`data/synthesized/audio`）
*   `--reference-dir`: 正解ラベルファイルのディレクトリ（例：`data/synthesized/labels`）
*   `--output`: 生成するグリッド設定ファイルのパス（例：`grid_config.yaml`）
*   `--param`: パラメータ名とその候補値（複数回指定可能）

**例:**

```bash
python run_grid_search.py create-config \
--detector PZSTDDetector \
--audio-dir data/synthesized/audio \
--reference-dir data/synthesized/labels \
--output grid_config.yaml \
--param f0_score_threshold 0.20 0.25 0.30 \
--param HCF_ONSET_PEAK_THRESH 0.12 0.16 \
--param HCF_COHERENCE_OFFSET_THRESH 0.07 0.13 \
--param max_pitch_std_dev_cents 40.0 50.0 \
--param harmonic_match_tolerance_cents 30.0 40.0
```

### 5.2. グリッドサーチの実行

`run_grid_search.py run` コマンドを使用して、生成した設定に基づきグリッドサーチを実行します。

**必須引数:**

*   `--config`: 基本設定ファイルのパス（例：`config.yaml`）
*   `--grid-config`: ステップ1で生成したグリッド設定ファイルのパス（例：`grid_config.yaml`）
*   `--output-dir`: 結果を保存するディレクトリのパス（例：`results/grid_search`）

**オプション引数:**

*   `--best-metric`: 最適化する指標（デフォルト：`note.f_measure`）
*   `--save-plots`: 各パラメータ組み合わせの評価プロットを保存

**例:**

```bash
python run_grid_search.py run \
--config config.yaml \
--grid-config grid_config.yaml \
--output-dir results/grid_search
```

各パラメータ組み合わせの評価指標と最適なパラメータが指定された出力ディレクトリに保存されます。

## 6. 実験的機能：AI駆動自動改善システム

**注意: この機能は実験的段階にあり、安定性は保証されていません。**

### 6.1. セットアップ（実験的機能）

Claude AIとの連携を設定します：

```bash
# 必須ライブラリ
pip install mcp requests pandas numpy waitress

# オプション（可視化用）
pip install matplotlib seaborn

# Claude Desktop設定
python setup_claude_integration.py --open-claude
```

### 6.2. 自動改善の実行（実験的機能）

コマンドラインから改善プロセスを実行：

```bash
python auto_improver.py --detector PZSTDDetector --goal "ノイズ耐性を向上させる" --iterations 3
```

オプション:
- `--detector`: 改善対象の検出器名
- `--goal`: 改善目標を自然言語で記述
- `--iterations`: 改善サイクルの繰り返し回数（デフォルト: 3）

### 6.3. 実験的機能のトラブルシューティング

#### ファイルシステム権限エラー
```
OSError: [Errno 30] Read-only file system: 'mcp_workspace'
```
**解決策**: システムは自動的に書き込み可能なディレクトリに切り替わります。

#### ポート競合
```
OSError: Address already in use
```
**解決策**: 
```bash
export MCP_PORT=5003
```

#### LLM API呼び出しエラー
```
LLM API request error
```
**解決策**: 
- APIキーの設定を確認
- インターネット接続を確認
- API制限を確認

### 6.4. 実験的機能の使用例

```bash
# ノイズ耐性改善
python auto_improver.py --detector PZSTDDetector --goal "ノイズが存在する環境でも正確に音符を検出できるよう改善する。特に5_noisy系列のファイルでの性能向上を目指す" --iterations 3

# リバーブ対応強化
python auto_improver.py --detector PZSTDDetector --goal "リバーブが長い環境での検出精度を高める。特に6_reverb_long.wavでの性能を優先的に向上させる" --iterations 3

# 和音検出改善
python auto_improver.py --detector PZSTDDetector --goal "和音検出の精度を向上させる。特に7_chords.wavと8_polyphony.wavでの検出漏れを減らす" --iterations 3
```

