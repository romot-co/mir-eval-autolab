
# アルゴリズム仕様書
**Pinna（ピンナ）検出器**  
――蝸牛フィルタバンクと神経スパイクモデルを用いた生理学的アプローチ――

---

## 1. アルゴリズム名称
**Pinna（ピンナ）**  
（ヒトの耳介 “Pinna” および内耳（蝸牛）をモチーフとしたネーミング）

---

## 2. 技術分野
本アルゴリズムは、**音声信号処理**、特に音楽音響や楽器音、音声などを対象とした **オンセット・オフセット検出**および**ピッチ推定**に関する技術分野に属する。人間の聴覚機構を簡易的に模倣したフィルタバンクと神経スパイクモデルを用い、リアルタイム性と高い検出精度を両立させる。

---

## 3. 従来技術および背景

### 3.1 既存手法
- 音響信号に対するオンセット検出は、しばしば **エネルギー差分** や **スペクトルフラックス** を使う方法が提案されている。しかし、打撃音など急峻な立ち上がりが強い場合や、滑らかなアタック音（ストリングス、ウィンド系楽器など）を扱う場合、誤検出・検出遅延が発生しやすい。
- ピッチ推定においては、**自己相関**や**ケプストラム**、あるいは**ディープラーニング**を用いる手法が多い。しかし、倍音構造が複雑またはノイズ多めの音源では精度が落ちることがある。

### 3.2 聴覚生理学の応用
- ヒトの聴覚系（外耳 -> 中耳 -> 内耳）では、まず **蝸牛フィルタ**（周波数分解機構）を経て、さらに有毛細胞の電気信号化を経由し、聴神経線維が **スパイク列**（インパルス列）を生成する。
- 音響信号が急激に変化すると、神経スパイクの発火率に大きな変化が現れる（立ち上がりでスパイク頻度が急増するなど）。
- 本アルゴリズム “Pinna” では、この**蝸牛フィルタバンク** + **スパイクモデル**の組み合わせを簡易的に再現し、オンセットやオフセットを捉えると同時に、スパイク列の**インターバル**（インタースパイク間隔; ISI）からピッチ（基本周波数）を推定する。

---

## 4. アルゴリズム概要
Pinna 検出器の大まかな処理フローは以下のとおりである。

1. **フィルタバンク処理**  
   - Gammatone あるいは類似のバンドパスフィルタを複数用い、周波数帯を分割する。  
   - 各帯域信号を取り出し、聴覚の周波数分解を簡易模擬。

2. **半波整流 + 圧縮**  
   - 各帯域出力を正方向のみ取り出し（負値は0にする）+ ルート圧縮などで**非線形変換**し、有毛細胞の振動-電気変換を模擬。

3. **スパイク発火モデル**  
   - 整流・圧縮された信号をフレーム単位またはサンプル単位で蓄積し、**一定閾値を超えたらスパイクを発火**→ 蓄積値リセット。  
   - これにより、**スパイク列（時刻リスト）**を各帯域ごとに取得する。

4. **オンセット・オフセット検出**  
   - 各帯域スパイクの発火率（短時間内のスパイク数）を計算し、フレーム列にマッピング。  
   - 発火率の急増をオンセットの指標、急減をオフセットの指標とし、バンドごとの寄与を合計あるいは二乗合計して**オンセット関数**・**オフセット関数**を得る。

5. **動的閾値処理**  
   - オンセット関数とオフセット関数、さらに音量（発火率合計のログなど）・ピッチ安定度などの指標を組み合わせて状態変数 \( Z(t) \) を更新。  
   - \( Z(t) \) の時間微分を閾値比較するなどの方法で、最終的なオンセットフレーム・オフセットフレームを決定する。

6. **ピッチ推定（インタースパイク間隔解析）**  
   - ピッチ帯域（例: 50〜2,000 Hz）に絞り込み、バンドごとのスパイク間隔をヒストグラム化。  
   - ヒストグラムの最頻値から周期を推定し、逆数をとって基本周波数（ピッチ）とする。  
   - 音が鳴っている区間に対し、この手法で**各ノートのピッチ**を推定。

7. **音量推定**  
   - 各帯域の発火率を合計することで、音量（エネルギー）を簡易的に算出。  
   - ログスケール化することで人間の聴感上のダイナミクスに近い値を得る。

このように、ヒト聴覚を模倣した **バンド分割・スパイク化** という発想により、音の立ち上がり・立下がり（オンセット・オフセット）を明確に捕捉し、倍音構造が混在していてもスパイク列から周期構造を抽出することで、**ピッチ推定**を同時に行う点が特徴的である。

---

## 5. 独創的特徴および利点

1. **生理学的発想**:  
   Gammatone型フィルタバンクやスパイクモデルは聴覚研究における生理学的知見をベースとしており、急激な立ち上がりや音の消失に敏感である。

2. **オンセット・オフセットの鋭敏な捉え**:  
   立ち上がり時のスパイクが集中発火しやすいため、既存のエネルギー差分ベースを上回る検出精度が期待できる。音止みも同様に、スパイクが停止することで正確に捉えられる。

3. **ISI（インタースパイク間隔）によるピッチ推定**:  
   倍音が多数存在する音でも、複数バンドのスパイク間隔に基音周期が反映されやすい。自己相関・ケプストラムとは別の視点からのアプローチだが、等価または補完的な精度が得られる。

4. **パラメータの少数化・リアルタイム性**:  
   - フィルタバンクバンド数・スパイク閾値・動的閾値の平滑化係数など、数個のパラメータでアルゴリズム全体を制御可能。  
   - 処理の中心はバンドパスフィルタと簡単な発火判定、カウント作業であるため軽量化が図れる。

5. **音量推定と組み合わせ**:  
   スパイク発火率合計が自然とエネルギー指標となるので、**オンセット・オフセット・ピッチ・音量**の4要素を一貫した枠組みで同時推定できる。

---

## 6. アルゴリズム構成要素

### 6.1 フィルタバンク設計
- **Gammatoneフィルタ**(または近似IIR)を周波数軸に対数分割して複数生成。  
- 各フィルタは中心周波数 \( \mathrm{cf}_b \) とバンド幅 \( \mathrm{bw}_b \) を持ち、サンプリングレート \( sr \) に合わせて設計。  
- 本実装例では `signal.butter` などで代用しているが、厳密なGammatone特性を用いる場合は専用設計も可。

### 6.2 半波整流 + 圧縮
- フィルタ出力 \( y_b(t) \) を
  \[
    x_b(t) = \sqrt{\max(0,\,y_b(t))}
  \]
  とする。  
- 他にも log圧縮などがあるが、\(\sqrt{\cdot}\) は単純かつ適度な動的範囲を確保できる。

### 6.3 スパイク発火モデル
1. **蓄積変数**: 各バンド \(b\) に対し、\(\mathrm{accum}_b\) を定義し、フレームごと/サンプルごとに
   \[
     \mathrm{accum}_b \;\leftarrow\; \mathrm{accum}_b + x_b(t) \times \mathrm{accumulation\_rate}.
   \]
2. **閾値判定**: \(\mathrm{accum}_b > \mathrm{spike\_threshold}\) であればスパイク発火 → \(\mathrm{accum}_b\)をリセット。
3. 発火時刻 \( t_{b,k} \) を記録し、各バンドごとに**スパイク時系列リスト** \(\{t_{b,k}\}\)を得る。

### 6.4 スパイク発火率およびオンセット関数
1. 発火率 \( R_b(n) \) は、フレーム \(n\) 周辺の短い時間窓内に生じたスパイク数をカウントして正規化する。  
2. フレーム間差分の正部分 \(\max(0,\,R_b(n) - R_b(n-1))\) を**オンセット候補**とみなし、二乗和などで**オンセット関数**に変換。  
3. 同様に負部分 \(\max(0,\,R_b(n-1) - R_b(n))\) を **オフセット関数** として扱う。

### 6.5 動的閾値によるオンセット・オフセット判定
- オンセット関数とオフセット関数、さらに音量変化やピッチ安定度を考慮した状態変数 \( Z(n) \) を更新。  
- \( Z(n) \) の時間微分 \( Z'(n) \) を用い、過去の正値平均・負値平均などを参照した閾値と比較し、**ピークが規定量を越えたところ**をオンセット/オフセットフレームとして確定する。

### 6.6 ピッチ推定（インタースパイク間隔）
1. ピッチ帯域（例: 50〜2,000 Hz）に限り、バンド \( b \) のスパイク列 \(\{t_{b,k}\}\) からスパイク間隔 \(\Delta_{b,k} = t_{b,k+1} - t_{b,k}\) を取得。  
2. \(\Delta_{b,k}\) をヒストグラム化し、最も頻度の高い間隔を \(\Delta^*\) とする（ただし \(\Delta^*\) が妥当な範囲内）。  
3. **基本周波数** \( f_0 = \frac{sr}{\Delta^*}\) としてピッチを求める。  
   - バンド毎にヒストグラムを合算するなどで**倍音情報**を統合し、精度を向上させる。

### 6.7 音量（発火率合計）
- 各フレームで \(\sum_{b} R_b(n)\) を取り、さらに \(\log\bigl(\epsilon + \sum_{b}R_b(n)\bigr)\) として音量指標とする。

---

## 7. 実施例（プログラム構造）

### 7.1 概要

- **pinia_detector.py** 内で、`PinnaDetector` クラスを定義し、`BaseDetector` を継承。  
- **主なメソッド**:
  1. `detect(...)`: 入力音声ファイル or 時系列データからオンセット・オフセット・ピッチを包括的に検出  
  2. `detect_notes(...)`: スパイク発火列を生成し、フレーム単位のオンセット・オフセット検出およびピッチ推定を行い、**音符情報 (intervals, pitches)** を返す  
  3. `detect_onsets(...)`: オンセット時刻のみを返す  
  4. `detect_offsets(...)`: オフセット時刻のみを返す  
  5. `detect_pitches(...)`: ピッチ推定のみを行う（オンセット・オフセットを外部から指定することも可）  

### 7.2 パラメータ例
- `num_bands`: フィルタバンド数（初期値 32）  
- `min_freq, max_freq`: フィルタバンドの周波数帯域下限・上限  
- `spike_threshold`: スパイク発火閾値（連続蓄積値がこれを超えたらスパイク）  
- `accumulation_rate`: 蓄積率（入力信号のスケールに応じた調整）  
- `frame_length, hop_length`: フレームサイズとホップ  
- `onset_threshold, offset_threshold`: 動的閾値計算で用いる基準  
- `pitch_min, pitch_max`: ピッチ推定における最低〜最高周波数  
- `window_size`: 発火率を計算する際の時間窓（秒）

### 7.3 コア関数

1. **`_create_gammatone_filterbank(sr)`**  
   - 周波数を対数スケールで分割し、各バンドごとにバターワースフィルタなどで近似したGammatoneフィルタを生成・リスト化する。

2. **`_apply_filterbank(audio, sr)`**  
   - 入力信号に対し各バンドフィルタを畳み込み（あるいは `lsim` 等）を適用し、バンド出力を得る。

3. **`_half_wave_rectify_compress(signals)`**  
   - 各バンド出力に対し半波整流 + ルート圧縮を施し、聴覚近似の非線形処理をモデル化。

4. **`_generate_spikes(signals, sr)`**  
   - 整形後の波形をサンプル単位で蓄積し、閾値超過時にスパイク発火。スパイク時刻(サンプルインデックス)をリスト化して返す。

5. **`_calculate_spike_rates(spikes, signal_length, sr)`**  
   - スパイク列から短時間内（`window_size`秒程度）に含まれるスパイク数をカウントし、発火率を算出してフレーム構造にする。

6. **`_calculate_onset_offset_functions(spike_rates)`**  
   - フレームごとにオンセット関数、オフセット関数を計算する。  
   - 「発火率のフレーム間差分の正部分/負部分の二乗合計」を平滑化して取得。

7. **`_dynamic_thresholding(...)`**  
   - オンセット・オフセット関数とエネルギー変化、およびピッチ安定度を組み合わせて状態変数 \( z \) を更新。  
   - \( z \) の時間微分に移動平均ベースの閾値を適用してオンセットフレーム / オフセットフレームを検出。

8. **`_analyze_interspike_intervals(spikes, onsets, offsets, sr)`**  
   - オンセット・オフセットで区切られた区間ごとに、ピッチ帯域のスパイク列からISIヒストグラムを求め、ピッチを推定。

---

## 8. 実行結果の形式

- `detect(...)` の戻り値：  
  ```python
  {
      'onsets': np.ndarray,       # オンセット時刻(秒)
      'offsets': np.ndarray,      # オフセット時刻(秒)
      'intervals': np.ndarray,    # shape (N_notes, 2) の [開始, 終了] 時刻
      'pitches': np.ndarray,      # shape (N_notes,) の各ノートピッチ(Hz)
      'detector_time': float,     # 処理時間(秒)
      'name': str                 # 検出器名 (PinnaDetector)
  }
  ```
- `detect_notes(...)` の場合：  
  ```python
  {
      'intervals': np.ndarray,  # [start_time, end_time]
      'pitches': np.ndarray     # 各ノートの推定ピッチ(Hz)
  }
  ```
- その他 `detect_onsets`, `detect_offsets`, `detect_pitches` でも上記情報の一部を出力。

---

## 9. 効果・応用範囲

1. **精度と汎用性**  
   - 打楽器やパーカッシブな音から、弦楽器や管楽器のようにアタックが緩やかな音まで、幅広く適応。  
   - ポリフォニックな音源でも、各バンドのスパイク列から**部分的に基音周期が浮かび上がる**場合があり、有用性が期待される。

2. **リアルタイム実装のしやすさ**  
   - フィルタリング + スパイク生成 + 短時間内カウントを順次処理でき、フレーム遅延も比較的少ない。  
   - パラメータ調整が明快で、デバイス上のリソース制限に合わせてバンド数を減らし軽量化するなどの工夫が可能。

3. **総合的な音情報抽出**  
   - オンセット検出 + ピッチ推定 + 音量推定 + オフセット検出を1つの枠組みで扱う。  
   - 自動採譜システム、演奏解析、録音編集支援、MIDI変換アプリケーションなどへの応用が広い。

---

## 10. 特許請求（サンプル）

**請求項1**  
音声信号を複数のバンドパスフィルタにより周波数帯域ごとに分割する工程と、各帯域信号を半波整流および圧縮処理する工程と、前記処理後の各帯域信号について閾値超過によるスパイク発火モデルを適用してスパイク列を生成する工程と、該スパイク列から帯域ごとの発火率を算出し、フレーム間差分を用いてオンセット及びオフセットを検出する工程と、前記スパイク列のインタースパイク間隔に基づいて基本周波数を推定する工程とを含むことを特徴とする音声信号のオンセット・オフセット・ピッチ同時検出方法。

**請求項2**  
請求項1において、前記オンセット及びオフセットの検出に際し、発火率の増減分を二乗合計した関数を平滑化して動的閾値処理を行うことを特徴とする音声信号のオンセット・オフセット検出方法。

**請求項3**  
請求項1または2において、前記インタースパイク間隔のヒストグラムを複数帯域で合算し、最頻値に対応する逆数を基本周波数とすることを特徴とするピッチ推定方法。

---

## 11. まとめ
**Pinna Detector**は、生理学的にインスパイアされたアプローチによって、**オンセット・オフセット検出**と**ピッチ推定**、さらには**音量推定**を一元的に実現するアルゴリズムである。本アルゴリズムのコア要素は「蝸牛フィルタバンク」と「神経スパイクモデル」であり、音の変化がスパイク列に即座に反映されるため、トランジェントや音の消失を敏感に捉える。

- 従来のエネルギー差分・自己相関手法に対して、**倍音構造が複雑な楽音**や**アタック特性が多様な音源**に対しても高精度で動作可能。  
- 軽量実装が容易で、リアルタイム処理にも適している。  
- オンセット・オフセット・ピッチ・音量を同一フレームワークで推定できるため、**自動採譜**や**演奏解析ツール**など多方面での応用が期待される。

本仕様書で示した実装例（`pinia_detector.py`）はあくまで一例であり、フィルタデザインや発火モデルの詳細は多様なバリエーションがあり得る。今後はフィルタ特性の改良やISI 解析の高精度化などを行うことで、更なる性能向上や適応範囲の拡大が期待できる。
