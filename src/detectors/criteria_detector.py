#!/usr/bin/env python3
"""
基準ノート検出器 (CriteriaDetector)

このモジュールは、音楽情報検索タスクにおける基準（ベースライン）となる
ノート検出器 `CriteriaDetector` を実装します。以下の広く知られた手法を組み合わせています。

- **ピッチ検出:** CREPE (Convolutional REpresentation for Pitch Estimation) モデルを使用。
                 深層学習に基づき、比較的高精度なフレーム単位のピッチ推定を行います。
- **オンセット検出:** librosa ライブラリのスペクトル変化に基づくオンセット検出機能を使用。
                   音楽信号における音の立ち上がり時刻を推定します。
- **オフセット検出:** librosa ライブラリの RMS (Root Mean Square) エネルギーに基づき、
                   各オンセット後のエネルギー減衰点を探索して音の終了時刻を推定します。
                   (この方法は単純であり、リバーブ等に影響される可能性があります)
"""

import os
import time
import numpy as np
import librosa
import argparse
# import soundfile as sf # 現在は直接使用していない
from typing import Dict, Any, Optional, Tuple, List
import logging
import traceback # エラーログ用

# 絶対インポート (src が PYTHONPATH にある前提)
try:
    from src.detectors.base_detector import BaseDetector
    from src.detectors import register_detector # __init__.py から register_detector をインポート
except ImportError as e:
    # __init__.py が先に実行されることを期待しているが、単体実行時などのフォールバック
    logging.error(f"Failed to import BaseDetector or register_detector: {e}. Ensure 'src' is in PYTHONPATH.")
    # BaseDetectorがないと動作しないため、エラーを再送出
    raise e

# CREPE ライブラリのインポートと存在確認
try:
    import crepe
except ImportError:
    logging.error("CREPE library not found. CriteriaDetector requires CREPE.")
    logging.error("Please install it: pip install crepe")
    # crepe がなければ CriteriaDetector は機能しない
    crepe = None

# ロガー設定
logger = logging.getLogger(__name__)

# register_detector デコレータを使用してクラスを登録
@register_detector(
    name="CriteriaDetector",
    description="Baseline note detector using CREPE (pitch) and librosa (onset/offset).",
    version="1.2" # バージョン更新
)
class CriteriaDetector(BaseDetector):
    """
    CREPE (ピッチ) と librosa (オンセット/オフセット) を使用する基準ノート検出器。

    高精度なピッチ検出モデル CREPE と、広く使われている音声処理ライブラリ
    librosa の機能を組み合わせて、ノートの開始(Onset)、終了(Offset)、
    およびピッチ(基本周波数)を推定します。
    主にモノフォニック（単旋律）な信号を対象としていますが、
    限定的なポリフォニー（和音など）に対しても動作します。
    ただし、オフセット検出は単純なRMSベースのため、精度には限界があります。
    """

    # デフォルトパラメータ: 外部から設定可能
    default_params = {
        # --- Onset Detection (librosa.onset.onset_detect) ---
        "onset_threshold": 0.085,        # オンセット検出の閾値 (delta)。中間的な値に設定。
        "onset_hop_length": 512,        # オンセット強度計算時のホップ長 (samples)。
        "onset_wait": 3,                # 検出されるオンセット間の最小待機フレーム数。

        # --- Offset Detection (RMS based) ---
        "offset_threshold_ratio": 0.1,  # オフセット検出時、RMSピーク値に対する相対的なエネルギー閾値。
        "offset_hop_length": 512,       # RMS計算時のホップ長 (samples)。
        "offset_frame_length": 2048,    # RMS計算時のフレーム長 (samples)。

        # --- Pitch Detection (CREPE) ---
        "pitch_confidence_threshold": 0.2, # CREPEが出力するピッチ信頼度の閾値。
        "crepe_model_capacity": "full", # CREPEモデルのサイズ。
        "crepe_step_size": 10,          # CREPEの分析フレーム間隔 (ミリ秒)。
        "crepe_viterbi": True,          # CREPEでViterbiアルゴリズムを使用するか。

        # --- Note Filtering ---
        "min_note_duration": 0.05,      # 検出されるノートの最小持続時間 (秒)。
        "max_note_duration": 3.0,       # 検出されるノートの最大持続時間 (秒)。
        "default_note_duration": 0.3,   # RMSベースのオフセット検出が失敗した場合のデフォルトノート持続時間 (秒)。
    }
    version = "1.2" # 明示的にバージョン属性を追加

    def __init__(self, **kwargs: Any):
        """
        CriteriaDetectorを初期化します。
        
        Parameters
        ----------
        **kwargs : Any
            設定したいパラメータをキーワード引数で指定します。
            指定されなかったパラメータは `default_params` の値が使用されます。
            例: `CriteriaDetector(onset_threshold=0.05, crepe_model_capacity='medium')`
        """
        super().__init__(**kwargs) # BaseDetectorの__init__を呼び出し、self.params を設定

        # crepeが利用可能かチェック (インポート時に確認済みだが念のため)
        if crepe is None:
            raise ImportError("CREPE library is not available. Cannot initialize CriteriaDetector.")

        # パラメータを個別の属性としても保持 (コード内でアクセスしやすくするため)
        # self.params 辞書から値を取得 (kwargsで渡された値 > default_params)
        for key, default_value in self.default_params.items():
            setattr(self, key, self.params.get(key, default_value))

        logger.info(f"{self.__class__.__name__} (v{self.version}) initialized with effective parameters:")
        # self.params を表示 (kwargsで上書きされた値を含む)
        for key, value in self.params.items():
             # default_params にないパラメータがkwargsで渡された場合も表示される
             logger.info(f"  {key}: {value}")
    
    def detect(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        与えられた音声データからノート情報を検出し、結果を辞書で返します。
        `BaseDetector` のインターフェース仕様に従います。

        Parameters
        ----------
        audio_data : np.ndarray
            モノラルの音声波形データ (1次元 NumPy 配列)。
        sr : int
            音声データのサンプリングレート (Hz)。

        Returns
        -------
        Dict[str, Any]
            検出結果を含む辞書。主なキーは以下の通り:
            - 'intervals': (N, 2) ndarray, [onset, offset] (秒)
            - 'note_pitches': (N,) ndarray, 各ノートのピッチ (Hz), 0Hzはピッチなし
            - 'frame_times': (M,) ndarray, CREPEフレーム時刻 (秒)
            - 'frame_frequencies': (M,) ndarray, CREPEフレームピッチ (Hz), NaNはピッチなし
            - 'frame_confidences': (M,) ndarray, CREPEフレーム信頼度
            - 'detector_name': str, この検出器の名前
            - 'detection_time': float, 処理時間 (秒)
            - 'params': dict, 使用されたパラメータ
        """
        start_time = time.time()
        
        # --- 入力検証 ---
        if audio_data is None or audio_data.ndim != 1 or audio_data.size == 0:
            logger.error("Invalid audio data: Must be a non-empty 1D numpy array.")
            # エラーを示す空の結果を返すか、例外を発生させるか選択
            # return self._empty_result(start_time, error="Invalid audio data")
            raise ValueError("Audio data must be a non-empty 1D numpy array.")
        if not isinstance(sr, int) or sr <= 0:
            logger.error(f"Invalid sample rate: {sr}. Must be a positive integer.")
            # return self._empty_result(start_time, error="Invalid sample rate")
            raise ValueError("Sample rate must be a positive integer.")
        if crepe is None: # 再度チェック (万が一 __init__ を迂回した場合など)
             logger.error("CREPE library is not available.")
             raise RuntimeError("CREPE library is required but not available.")

        logger.info(f"Starting detection on audio with duration={len(audio_data)/sr:.2f}s, sr={sr}Hz...")

        # --- 1. オンセット検出 ---
        onsets = self._detect_onsets(audio_data, sr)
        if len(onsets) == 0:
             logger.warning("No onsets detected. Returning empty result.")
             return self._empty_result(start_time)

        # --- 2. ピッチ検出 (フレームベース by CREPE) ---
        try:
             logger.debug(f"Running CREPE pitch detection (model: {self.crepe_model_capacity}, step: {self.crepe_step_size}ms)...")
             # crepe.predict は (times, frequencies, confidence, activation) を返す
             frame_times, frame_frequencies, frame_confidences, _ = crepe.predict(
                 audio_data, sr,
                 model_capacity=self.crepe_model_capacity,
                 step_size=self.crepe_step_size,
                 viterbi=self.crepe_viterbi,
                 center=True, # フレーム時刻はフレームの中央
                 verbose=0 # CREPE自体のログ出力を抑制 (0 or 1 or 2)
             )
             logger.debug(f"CREPE finished, found {len(frame_times)} frames.")

             # CREPEが出力する周波数 0Hz や負値はピッチなし (NaN) として扱う
             invalid_pitch_mask = (frame_frequencies <= 0) | np.isnan(frame_frequencies)
             if np.any(invalid_pitch_mask):
                  logger.debug(f"Found {np.sum(invalid_pitch_mask)} invalid pitch values (<=0 or NaN) from CREPE, setting to NaN.")
                  frame_frequencies[invalid_pitch_mask] = np.nan

        except Exception as e:
             logger.error(f"CREPE pitch detection failed: {e}")
             logger.error(traceback.format_exc())
             # CREPEが失敗した場合、フレーム情報は空とし、処理を続行するか選択
             frame_times, frame_frequencies, frame_confidences = np.array([]), np.array([]), np.array([])
             # ここで空の結果を返しても良い
             # return self._empty_result(start_time, error="CREPE detection failed")


        # --- 3. オフセット検出 (RMSベース) ---
        offsets = self._detect_offsets(audio_data, sr, onsets)
        if len(offsets) == 0: # 通常は onsets と同じ長さのはずだが念のため
             logger.warning("Offset detection resulted in an empty array. Using default durations.")
             # オンセット + デフォルト長で仮のオフセットを作成
             offsets = onsets + self.default_note_duration


        # --- 4. ノートのペアリング、フィルタリング、ピッチ抽出 ---
        logger.debug("Pairing onsets/offsets and filtering notes...")
        valid_intervals = []
        valid_onsets_for_pitch = [] # ピッチ抽出に使うオンセット時刻

        if len(onsets) > 0:
             # オンセット数に基づいてオフセットをペアリング (通常は同数のはず)
             num_notes_to_pair = min(len(onsets), len(offsets))
             total_duration = len(audio_data) / sr

             for i in range(num_notes_to_pair):
                 onset_time = onsets[i]
                 # 候補となるオフセット時刻
                 candidate_offset_time = offsets[i]

                 # 次のオンセット時刻 (なければ曲の終わり)
                 # ノートが重ならないように、次のオンセットのわずかに手前を上限とする
                 next_onset_boundary = onsets[i+1] - 1e-3 if i + 1 < len(onsets) else total_duration

                 # 最大ノート長による上限
                 max_offset_boundary = onset_time + self.max_note_duration

                 # オフセット時刻を決定: 候補オフセット、次のオンセット境界、最大長境界のうち最も早いもの
                 offset_time = min(candidate_offset_time, next_onset_boundary, max_offset_boundary)

                 # 最小ノート長の制約を適用
                 # オフセットはオンセットより後で、かつ最小持続時間を満たす必要がある
                 if offset_time > onset_time and (offset_time - onset_time) >= self.min_note_duration:
                      valid_intervals.append([onset_time, offset_time])
                      valid_onsets_for_pitch.append(onset_time) # 対応するオンセット時刻
                 else:
                      logger.debug(f"Note {i+1} (onset {onset_time:.3f}s) discarded due to duration constraints "
                                   f"(candidate offset: {candidate_offset_time:.3f}, final offset: {offset_time:.3f}).")


        intervals = np.array(valid_intervals) if valid_intervals else np.array([]).reshape(0, 2)
        logger.info(f"Filtered notes based on duration and overlap. Kept {len(intervals)} notes.")

        # --- 5. 各有効ノートのピッチを抽出 ---
        note_pitches = np.array([])
        if intervals.shape[0] > 0 and frame_times.shape[0] > 0:
             logger.debug("Extracting pitch for each valid note interval...")
             note_pitches = self._extract_note_pitches(intervals, frame_times, frame_frequencies, frame_confidences)
             logger.debug("Pitch extraction complete.")
        elif intervals.shape[0] > 0:
             logger.warning("Valid note intervals found, but no frame data available for pitch extraction. Returning 0Hz pitches.")
             note_pitches = np.zeros(intervals.shape[0]) # フレーム情報がない場合は 0 Hz とする


        # --- 6. 結果の集計 ---
        detection_time = time.time() - start_time
        logger.info(f"Detection process finished in {detection_time:.4f} seconds.")
        
        result = {
            'intervals': intervals,             # (N, 2) onset, offset times
            'note_pitches': note_pitches,       # (N,) pitch in Hz (0 for no pitch)
            'frame_times': frame_times,         # (M,) frame times from CREPE
            'frame_frequencies': frame_frequencies, # (M,) frame pitch in Hz (NaN for no pitch)
            'frame_confidences': frame_confidences, # (M,) frame pitch confidence
            'detector_name': self.__class__.__name__, # クラス名
            'detection_time': detection_time,   # 処理時間
            'params': self.params.copy()      # 使用したパラメータのコピー
        }
        return result
    
    def _detect_onsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """librosa を使用して音声信号からオンセット時刻を検出します。"""
        logger.debug("Detecting onsets using librosa.onset.onset_detect...")
        try:
            # 1. オンセット強度関数を計算 (標準的なスペクトル変化量)
            onset_env = librosa.onset.onset_strength(
                y=audio, sr=sr,
                hop_length=self.onset_hop_length,
                # aggregate=np.median # -> デフォルト(平均)に戻す
            )

            # 2. オンセット強度関数からピーク（オンセット候補）を検出
            #    delta: ピークの閾値 (平均値からの差)
            #    wait: ピーク間の最小距離 (フレーム単位)
            #    ピークピッキングパラメータ(pre/post_max/avg)は削除し、librosaの標準ロジックに任せる
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=self.onset_hop_length,
                units='frames',
                # pre_max=self.onset_pre_max, # 削除
                # post_max=self.onset_post_max, # 削除
                # pre_avg=self.onset_pre_avg, # 削除
                # post_avg=self.onset_post_avg, # 削除
                delta=self.onset_threshold, # 閾値パラメータ
                wait=self.onset_wait         # 待機フレーム数パラメータ
            )

            # 3. フレームインデックスを時間に変換
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.onset_hop_length)

            # 稀に負の値になる場合があるため補正
            onset_times = np.maximum(0, onset_times)

            logger.debug(f"Detected {len(onset_times)} raw onset times.")
            return onset_times

        except Exception as e:
            logger.error(f"Onset detection failed: {e}")
            logger.error(traceback.format_exc())
            return np.array([]) # エラー時は空配列を返す

    def _detect_offsets(self, audio: np.ndarray, sr: int, onsets: np.ndarray) -> np.ndarray:
        """
        RMSエネルギーに基づいて各オンセットに対応するオフセット時刻を推定します。

        注意: この方法は単純であり、以下のような限界があります。
        - リバーブ（残響）が長い場合、実際のノート終了より遅いオフセットを検出しやすい。
        - 音量が小さいノートや、減衰が非常に速い/遅い音では不安定になることがある。
        - ポリフォニックな部分では、他の音のエネルギーに影響される。
        より高度なオフセット検出には、スペクトル情報や機械学習モデルが必要になる場合があります。
        """
        if len(onsets) == 0:
            return np.array([])

        logger.debug("Detecting offsets based on RMS energy decay...")
        try:
            # 1. RMSエネルギーを計算
            rms = librosa.feature.rms(
                y=audio,
                frame_length=self.offset_frame_length,
                hop_length=self.offset_hop_length
            )[0] # 結果は (1, num_frames) なので [0] で1次元配列に

            # 2. RMSフレームに対応する時刻を計算
            rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.offset_hop_length)
            total_duration = len(audio) / sr # 曲全体の長さ

            offsets = np.zeros_like(onsets) # 結果を格納する配列

            # 3. 各オンセット区間に対してオフセットを探す
            for i, onset_time in enumerate(onsets):
                # 次のオンセット時刻（なければ曲の終わり）を区間の終端とする
                next_onset_time = onsets[i + 1] if i + 1 < len(onsets) else total_duration

                # 現在のノート区間 [onset_time, next_onset_time) に対応するRMSフレームの範囲を探す
                start_rms_idx = np.searchsorted(rms_times, onset_time, side='left')
                # 次のオンセットの直前のRMSフレームまでを探索範囲とする
                end_rms_idx = np.searchsorted(rms_times, next_onset_time, side='left')

                offset_found_for_this_note = False
                if start_rms_idx < end_rms_idx: # 区間内にRMSフレームが1つ以上存在する場合
                    # 区間内のRMS値と時刻を取得
                    rms_segment = rms[start_rms_idx:end_rms_idx]
                    times_segment = rms_times[start_rms_idx:end_rms_idx]
                
                if len(rms_segment) > 0:
                        # 区間内のRMSエネルギーのピークを探す
                        peak_idx_in_segment = np.argmax(rms_segment)
                        peak_rms = rms_segment[peak_idx_in_segment]

                        # オフセット検出のためのエネルギー閾値 (ピーク値に対する相対値)
                        offset_rms_threshold = peak_rms * self.offset_threshold_ratio

                        # ピーク位置以降で、RMSが閾値を最初に下回るフレームを探す
                        decay_indices = np.where(rms_segment[peak_idx_in_segment:] < offset_rms_threshold)[0]

                        if len(decay_indices) > 0:
                            # 閾値を下回るフレームが見つかった場合
                            offset_idx_in_segment = peak_idx_in_segment + decay_indices[0]
                            offset_time = times_segment[offset_idx_in_segment]
                            # オフセット時刻は次のオンセット時刻を超えないようにクリップ
                            offsets[i] = min(offset_time, next_onset_time - 1e-6) # 微小値を引いて境界一致を防ぐ
                            offset_found_for_this_note = True
                            # logger.debug(f"Offset found for onset {i+1} at {offsets[i]:.3f}s (RMS method)")


                # オフセットがRMSベースで見つからなかった場合
                if not offset_found_for_this_note:
                    # デフォルト長を加えた時刻と、次のオンセット時刻の小さい方をとる
                    default_offset_candidate = onset_time + self.default_note_duration
                    offsets[i] = min(default_offset_candidate, next_onset_time - 1e-6) # 微小値を引いて境界一致を防ぐ
                    logger.debug(f"Offset not found via RMS decay for note {i+1} (onset {onset_time:.3f}s). Using calculated default offset: {offsets[i]:.3f}s")


            # 念のため、オフセット時刻が対応するオンセット時刻より前にならないようにクリップ
            offsets = np.maximum(onsets + 1e-6, offsets) # 非常に小さい値を加えて同値にならないように

            logger.debug(f"Calculated {len(offsets)} raw offset times.")
            return offsets

        except Exception as e:
            logger.error(f"Offset detection failed: {e}")
            logger.error(traceback.format_exc())
            # エラー時はデフォルト長で返す (ただし次のオンセットは超えない)
            logger.warning("Offset detection error. Falling back to default durations limited by next onset.")
            fallback_offsets = np.zeros_like(onsets)
            for i, onset_time in enumerate(onsets):
                 next_onset_time = onsets[i + 1] if i + 1 < len(onsets) else total_duration
                 fallback_offsets[i] = min(onset_time + self.default_note_duration, next_onset_time - 1e-6)
            return np.maximum(onsets + 1e-6, fallback_offsets)


    def _extract_note_pitches(self, intervals: np.ndarray, frame_times: np.ndarray,
                              frame_frequencies: np.ndarray, frame_confidences: np.ndarray) -> np.ndarray:
        """
        各ノート区間 (`intervals`) 内のフレーム情報を用いて、代表ピッチを抽出します。
        信頼度が閾値以上のフレームピッチを、信頼度で重み付け平均して計算します。
        有効なピッチがない場合は 0 Hz を返します。
        """
        num_notes = len(intervals)
        note_pitches = np.zeros(num_notes) # 結果配列 (デフォルト 0 Hz)

        if frame_times.shape[0] == 0: # フレーム情報がなければピッチは計算不可
            logger.warning("No frame times available for pitch extraction.")
            return note_pitches # 0 Hz 配列を返す

        # 各ノート区間について処理
        for i, (onset, offset) in enumerate(intervals):
            # 1. ノート区間 [onset, offset] 内のフレームのインデックスを取得
            #    frame_times はソート済みと仮定
            #    onset <= frame_time < offset の範囲のフレームを選択
            start_idx = np.searchsorted(frame_times, onset, side='left')
            end_idx = np.searchsorted(frame_times, offset, side='left') # offset は含まない

            # 2. 区間内のピッチと信頼度を取得
            interval_freqs = frame_frequencies[start_idx:end_idx]
            interval_confs = frame_confidences[start_idx:end_idx]

            # 3. 有効なピッチフレームを選択
            #    - ピッチが NaN でない
            #    - 信頼度が閾値以上
            valid_mask = ~np.isnan(interval_freqs) & (interval_confs >= self.pitch_confidence_threshold)

            # 4. 有効なフレームが存在する場合、ピッチを計算
            if np.any(valid_mask):
                valid_pitches = interval_freqs[valid_mask]
                valid_confs = interval_confs[valid_mask]
                    
                    # 信頼度で重み付けした平均ピッチを計算
                # ゼロ除算を避けるため、信頼度の合計を確認
                sum_confs = np.sum(valid_confs)
                if sum_confs > 1e-6: # 小さな値より大きいか確認
                    weighted_pitch = np.sum(valid_pitches * valid_confs) / sum_confs
                    # 計算結果が NaN や inf になる可能性も考慮 (通常はならないはずだが)
                    if np.isfinite(weighted_pitch) and weighted_pitch > 0:
                        note_pitches[i] = weighted_pitch
                    else:
                        logger.warning(f"Weighted pitch calculation resulted in invalid value ({weighted_pitch}) for note {i+1}. Using median.")
                        # フォールバックとして単純な中央値を使う
                        note_pitches[i] = np.nanmedian(valid_pitches)

                else:
                    # 信頼度の合計が非常に小さい場合 (有効フレームはあるが信頼度がほぼ0)
                    # 単純な中央値（または平均値）を使用
                    logger.debug(f"Sum of confidences is very low ({sum_confs}) for note {i+1}. Using median pitch.")
                    note_pitches[i] = np.nanmedian(valid_pitches)

            else:
                # 区間内に閾値以上の信頼度を持つ有効なピッチフレームがなかった場合
                # 閾値未満でもピッチが存在すれば、その中央値を採用することも検討できるが、
                # ここでは「信頼できるピッチなし」として 0 Hz のままにする
                logger.debug(f"No frames with sufficient confidence found for note {i+1} ({onset:.3f}-{offset:.3f}s). Pitch set to 0 Hz.")
                note_pitches[i] = 0.0 # 0 Hz を維持

        # 最終的なピッチが NaN や負値になっていないか確認し、0 Hz に修正
        note_pitches[~np.isfinite(note_pitches) | (note_pitches < 0)] = 0.0
        logger.debug(f"Extracted {len(note_pitches)} note pitches (0 Hz indicates no reliable pitch).")
        return note_pitches
    
    def _empty_result(self, start_time: float, error: Optional[str] = None) -> Dict[str, Any]:
        """エラー発生時や検出結果がない場合に返す空の結果辞書を生成します"""
        detection_time = time.time() - start_time
        result = {
            'intervals': np.array([]).reshape(0, 2),
            'note_pitches': np.array([]),
            'frame_times': np.array([]),
            'frame_frequencies': np.array([]),
            'frame_confidences': np.array([]),
            'detector_name': self.__class__.__name__,
            'detection_time': detection_time,
            'params': self.params.copy(),
        }
        if error:
            result['error'] = error # エラーメッセージを追加 (オプション)
            logger.error(f"Returning empty result due to error: {error}")
        else:
             logger.info("Returning empty result (no notes detected or error occurred).")
        return result

# --- コマンドライン実行用ブロック ---
if __name__ == "__main__":
    # このブロックは、スクリプトが直接実行された場合にのみ動作します。
    # 依存ライブラリがインストールされているか、PYTHONPATHが設定されている環境で実行してください。

    # 基本的なロギング設定 (コマンドライン実行時用)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description='CriteriaDetector: Detect musical notes from an audio file using CREPE and librosa.'
                    ' Outputs detected notes (onset, offset, pitch) to the console.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # デフォルト値をヘルプに表示
    )
    parser.add_argument('audio_file', type=str, help='Path to the input audio file (e.g., WAV, MP3).')

    # 主要なパラメータをコマンドライン引数として公開
    param_group = parser.add_argument_group('Detection Parameters')
    for param, default_value in CriteriaDetector.default_params.items():
        arg_name = f'--{param.replace("_", "-")}' # アンダースコアをハイフンに
        param_type = type(default_value) if default_value is not None else str # 型を推定
        if param_type == bool:
             # bool型は store_true / store_false で扱うことが多いが、ここでは type=bool
             # (ただし、コマンドラインで False を指定するのが少し面倒になる)
             # 代わりに choices=[True, False] などを使う方法もある
             param_group.add_argument(arg_name, type=lambda x: (str(x).lower() == 'true'), default=default_value,
                                      help=f'{param} (bool)')
        else:
             param_group.add_argument(arg_name, type=param_type, default=default_value,
                                      help=f'{param}')
    
    args = parser.parse_args()
    
    # 引数からパラメータ辞書を作成 (default_params に存在するキーのみ)
    params_from_args = {key: getattr(args, key) for key in CriteriaDetector.default_params if hasattr(args, key)}

    try:
        # --- 検出器の準備 ---
        logger.info("Initializing CriteriaDetector...")
        # 引数で指定されたパラメータでインスタンス化
        detector = CriteriaDetector(**params_from_args)

        # --- 音声ファイルのロード ---
        logger.info(f"Loading audio file: {args.audio_file}")
        # librosa.load は多くのフォーマットに対応、sr=NoneでネイティブSR維持、mono=Trueでモノラル化
        audio, sr = librosa.load(args.audio_file, sr=None, mono=True)
        duration = len(audio) / sr
        logger.info(f"Audio loaded: Duration={duration:.2f}s, Sample Rate={sr}Hz")

        # --- 検出の実行 ---
        logger.info("Starting note detection...")
        results = detector.detect(audio_data=audio, sr=sr)
        logger.info("Detection complete.")

        # --- 結果の表示 ---
        print("\n--- Detection Results ---")
        print(f"Detector: {results.get('detector_name', 'N/A')} (v{getattr(detector, 'version', 'N/A')})")
        print(f"Audio File: {os.path.basename(args.audio_file)}")
        print(f"Detection Time: {results.get('detection_time', -1):.4f} seconds")

        intervals = results.get('intervals', np.array([]))
        note_pitches = results.get('note_pitches', np.array([]))
        num_detected_notes = len(intervals)
        print(f"\nDetected Notes: {num_detected_notes}")

        if num_detected_notes > 0:
            print("----------------------------------------------------------")
            print("  # | Onset (s) | Offset (s)| Duration(s)| Pitch (Hz)")
            print("----------------------------------------------------------")
            # 表示するノート数を制限 (例: 50件)
            max_notes_to_display = 50
            for i in range(min(num_detected_notes, max_notes_to_display)):
                onset = intervals[i, 0]
                offset = intervals[i, 1]
                pitch = note_pitches[i]
                duration = offset - onset
                # ピッチが0Hzの場合は 'N/A' または '-' と表示しても良い
                pitch_str = f"{pitch:8.2f}" if pitch > 0 else "   N/A  "
                print(f"{i+1:>3} | {onset:9.3f} | {offset:9.3f} | {duration:10.3f} | {pitch_str}")
            print("----------------------------------------------------------")
            if num_detected_notes > max_notes_to_display:
                print(f"  ... (displaying first {max_notes_to_display} notes of {num_detected_notes})")
        else:
            print("  No notes were detected that meet the criteria.")

        # 使用されたパラメータを表示 (オプション)
        print("\nUsed Parameters:")
        used_params = results.get('params', {})
        for key, value in used_params.items():
             print(f"  {key}: {value}")

        print("\n--- End of Results ---")

    except FileNotFoundError:
        logger.error(f"Error: Audio file not found at '{args.audio_file}'")
        print(f"\nエラー: 音声ファイルが見つかりません: {args.audio_file}")
    except ImportError as e:
         logger.error(f"Import error: {e}. Please ensure all dependencies (crepe, librosa, numpy) are installed.")
         print(f"\nインポートエラー: {e}。\n依存ライブラリ (crepe, librosa, numpy) が正しくインストールされているか確認してください。")
    except Exception as e:
        logger.error(f"An unexpected error occurred during detection: {e}")
        logger.error(traceback.format_exc()) # 詳細なエラー情報をログに出力
        print(f"\n予期せぬエラーが発生しました: {e}")
        print("詳細についてはログを確認してください。")