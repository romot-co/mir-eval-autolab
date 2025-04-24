"""
ONDEアルゴリズムに基づいた音楽イベント検出器

本実装は単一パラメータΘによるZスコア閾値を用いたオンセット/オフセット判定、
動的な背景モデル(平均・分散)更新、パーカッシブ判定、ピッチ検出、和音検出などを行います。
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np

from src.detectors.base_detector import BaseDetector


class ONDEDetector(BaseDetector):
    """
    ONDEアルゴリズムの検出器実装

    単一パラメータThetaに基づき:
      - 閾値 T = c_T * Theta
      - 背景モデル更新率 alpha_base = exp(-c_alpha * dt / Theta)
      - イベント最小長などをスケール
    パーカッシブ検出、ピッチ検出、和音検出はモジュール的にON/OFF可能。

    Attributes:
        Theta (float):          単一パラメータΘ（感度・時間スケールを制御）
        enable_percussive_detection (bool): パーカッシブ判定を有効にするか
        enable_pitch_detection (bool):      ピッチ検出を有効にするか
        enable_chord_detection (bool):      和音検出を有効にするか

        # CQTパラメータ
        hop_length (int):       CQTとフレーム分析のホップ長
        frame_length (int):     分析フレームの長さ
        f_min (float):          最低周波数 (Hz)
        f_max (float):          最高周波数 (Hz)
        bins_per_octave (int):  オクターブあたりのビン数

        # 内部定数（Thetaに基づいて計算される係数）
        c_T (float):            閾値係数
        c_alpha (float):        背景モデル更新係数
        c_min (float):          最小イベント長係数 (秒)

        # 代表値計算
        lambda_weight (float):  代表値計算での上位ビン重み
        top_k (int):            代表値計算に使用する周波数ビン数

        # オフセット検出
        offset_hysteresis (int):          オフセット判定のヒステリシスフレーム数

        # パーカッシブ判定関連
        spectral_flatness_threshold (float)
        high_freq_ratio_threshold (float)

        # ピッチ検出関連
        confidence_threshold (float)
        min_peak_ratio (float)

        # デバッグフラグ
        debug (bool): デバッグ出力を行うか
    """

    def __init__(
        self,
        sr: int = 44100,
        hop_length: int = 512,
        onset_threshold: float = 1.5,
        offset_threshold: float = 0.5,
        min_note_length: float = 0.1,
        max_note_length: float = 4.0,
        confidence_threshold: float = 2.0,
        enable_pitch_detection: bool = True,
        enable_percussive_detection: bool = True,
        enable_chord_detection: bool = False,
        cqt_bins_per_octave: int = 36,
        cqt_n_bins: int = 7 * 36,
        cqt_fmin: Optional[float] = None,
        cqt_fmax: Optional[float] = None,
        # 単一パラメータThetaとその他の追加パラメータ
        Theta: float = 1.0,
        frame_length: int = 1024,
        f_min: float = 65.4,
        f_max: float = 8000.0,
        bins_per_octave: int = 36,
        c_T: float = 1.4,
        c_alpha: float = 1.0,
        c_eta: float = 0.08,
        c_min: float = 0.03,
        offset_hysteresis: int = 3,
        offset_threshold_ratio: float = 0.5,
        event_slow_factor: float = 10.0,
        lambda_weight: float = 0.6,
        top_k: int = 5,
        min_peak_ratio: float = 0.5,
        gamma_enhancement: float = 0.5,
        high_dynamic_threshold: float = 40.0,
        low_dynamic_threshold: float = 15.0,
    ):
        """
        ONDEDetector初期化

        Args:
            sr: サンプリングレート
            hop_length: ホップサイズ
            onset_threshold: オンセット閾値（高いほど減少）
            offset_threshold: オフセット閾値（高いほど減少）
            min_note_length: 最小音符長（秒）
            max_note_length: 最大音符長（秒）
            confidence_threshold: ピッチ信頼度閾値
            enable_pitch_detection: ピッチ検出を有効にするか
            enable_percussive_detection: 打楽器検出を有効にするか
            enable_chord_detection: 和音検出を有効にするか
            cqt_bins_per_octave: CQTのオクターブあたりのビン数
            cqt_n_bins: CQTの総ビン数
            cqt_fmin: 最低周波数（Noneの場合はC1に設定）
            cqt_fmax: 最高周波数（Noneの場合は8000Hzに設定）
            Theta: ONDEアルゴリズムの単一パラメータ
            enable_stochastic_resonance: 確率共鳴を有効にするか
            enable_gamma_distribution: ガンマ分布モデルを有効にするか
            frame_length: 分析フレームの長さ
            f_min: 最低周波数（Hz）
            f_max: 最高周波数（Hz）
            bins_per_octave: オクターブあたりのビン数
            c_T: 閾値係数
            c_alpha: 背景モデル更新係数
            c_eta: SRノイズ強度係数
            c_min: 最小イベント長係数（秒）
            offset_hysteresis: オフセット判定フレーム数
            offset_threshold_ratio: オフセット判定閾値比率
            event_slow_factor: イベント中の更新抑制係数
            lambda_weight: 代表値計算の上位ビン重み
            top_k: 代表値計算に使用するビン数
            min_peak_ratio: ピーク検出の閾値
            gamma_enhancement: 弱オンセット強調係数
            high_dynamic_threshold: 高ダイナミックレンジの閾値（dB）
            low_dynamic_threshold: 低ダイナミックレンジの閾値（dB）
        """
        # 基本パラメータ
        self.sr = sr
        self.hop_length = hop_length
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_note_length = min_note_length
        self.max_note_length = max_note_length

        # ピッチ検出パラメータ
        self.confidence_threshold = confidence_threshold
        self.enable_pitch_detection = enable_pitch_detection
        self.enable_percussive_detection = enable_percussive_detection
        self.enable_chord_detection = enable_chord_detection

        # CQTパラメータ
        self.bins_per_octave = (
            bins_per_octave if bins_per_octave is not None else cqt_bins_per_octave
        )
        self.n_bins = cqt_n_bins
        self.f_min = (
            f_min if f_min is not None else (cqt_fmin if cqt_fmin is not None else 65.4)
        )
        self.f_max = (
            f_max
            if f_max is not None
            else (cqt_fmax if cqt_fmax is not None else 8000.0)
        )

        self.eps = 1e-8
        self.min_peak_ratio = min_peak_ratio

        # 単一パラメータ
        self.Theta = Theta

        # 内部定数
        self.c_T = c_T
        self.c_alpha = c_alpha
        self.c_min = c_min

        # 代表値計算パラメータ
        self.lambda_weight = lambda_weight
        self.top_k = top_k

        # オフセット判定
        self.offset_hysteresis = offset_hysteresis
        self.event_slow_factor = event_slow_factor

        # パーカッシブ判定
        self.spectral_flatness_threshold = 0.2
        self.high_freq_ratio_threshold = 0.25

        # デバッグ
        self.debug = False

    def _compute_cqt(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        オーディオデータのCQT振幅スペクトログラムを計算

        Args:
            audio: 音声データ
            sr: サンプリングレート

        Returns:
            np.ndarray: CQT振幅スペクトログラム
        """
        n_bins = int(np.ceil(np.log2(self.f_max / self.f_min) * self.bins_per_octave))

        # librosa.cqtがNoneを返す場合の対策として、ウィンドウサイズを小さくする
        window = "hann"
        if len(audio) < 2048:
            window = "hamming"  # 短い音声用に異なるウィンドウを使用

        C = librosa.cqt(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
            fmin=self.f_min,
            n_bins=n_bins,
            bins_per_octave=self.bins_per_octave,
            window=window,
        )

        # スペクトログラムのパワー計算（CQTの振幅）
        A = np.abs(C)

        # ノイズフロア処理 (音量が小さすぎる部分を排除)
        A = np.maximum(A, 1e-10)

        return A

    def _get_top_k_indices(self, array: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(array))
        return np.argpartition(array, -k)[-k:]

    def _compute_representative_value(self, spec_frame: np.ndarray) -> float:
        """
        スペクトルフレームの代表値 r_n を計算:
          r_n = sum(A^2) + lambda_weight * mean_of_topK
        """
        energy = np.sum(spec_frame**2)
        top_indices = self._get_top_k_indices(spec_frame, self.top_k)
        top_mean = np.mean(spec_frame[top_indices])
        return energy + self.lambda_weight * top_mean

    def _compute_spectral_flatness(self, spec_frame: np.ndarray) -> float:
        safe_spec = spec_frame + self.eps
        geo_mean = np.exp(np.mean(np.log(safe_spec)))
        arith_mean = np.mean(safe_spec)
        return geo_mean / (arith_mean + self.eps)

    def _compute_high_frequency_ratio(self, spec_frame: np.ndarray) -> float:
        n_bins = len(spec_frame)
        split_idx = n_bins // 2
        low_energy = np.sum(spec_frame[:split_idx] ** 2)
        high_energy = np.sum(spec_frame[split_idx:] ** 2)
        total_energy = low_energy + high_energy
        if total_energy < self.eps:
            return 0.0
        return high_energy / total_energy

    def _is_percussive(self, spec_frame: np.ndarray) -> bool:
        """
        フレームがパーカッシブかどうかを判定
        （enable_percussive_detectionがFalseなら常にFalse）
        """
        if not self.enable_percussive_detection:
            return False

        sf = self._compute_spectral_flatness(spec_frame)
        hfr = self._compute_high_frequency_ratio(spec_frame)

        # Θが大きいほど判定が少し緩くなる例
        sf_threshold = self.spectral_flatness_threshold * (1.0 / self.Theta)
        hfr_threshold = self.high_freq_ratio_threshold * (1.0 / self.Theta)

        return (sf > sf_threshold) or (hfr > hfr_threshold)

    def _detect_events(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        音声データからイベント（ノート）を検出します。
        """
        # CQTの計算
        A = self._compute_cqt(audio, sr)
        M, N_frames = A.shape

        # フレーム時間の計算
        dt = self.hop_length / sr

        # 閾値T、忘却係数alpha_base
        T = self.c_T * self.Theta
        alpha_base = np.exp(-self.c_alpha * dt / self.Theta)

        # 最小イベント長（フレーム数）
        L_min = int(np.ceil(self.c_min * self.Theta / dt))
        # 最小ノート長による制約も考慮
        min_note_frames = int(self.min_note_length / dt)
        L_min = max(L_min, min_note_frames)

        # 背景モデル
        mu = 0.0
        sigma_sq = 1.0

        # 結果配列
        mask = np.zeros(N_frames, dtype=int)
        d_vals = np.zeros(N_frames)
        r_vals = np.zeros(N_frames)

        in_event = False
        onset_frames = []
        offset_frames = []
        event_start = None
        zeros_count = 0
        percussive_flags = []
        event_specs = []

        for n in range(N_frames):
            frame = A[:, n]
            # 1) 代表値
            r_n = self._compute_representative_value(frame)
            r_vals[n] = r_n

            # 2) 背景更新
            diff = r_n - mu
            if in_event:
                # イベント中は更新をさらに抑える
                alpha = 1.0 - (1.0 - alpha_base) / self.event_slow_factor
            else:
                alpha = alpha_base

            mu = alpha * mu + (1 - alpha) * r_n
            sigma_sq = alpha * sigma_sq + (1 - alpha) * (diff * diff)
            sigma = max(np.sqrt(sigma_sq), self.eps)

            # 3) Zスコア (SRなし => そのまま)
            d_n = (r_n - mu) / sigma
            d_vals[n] = d_n

            # 4) 閾値判定
            mask[n] = 1 if d_n > T else 0

            # 5) オンセット/オフセット管理
            if not in_event and mask[n] == 1:
                in_event = True
                event_start = n
                zeros_count = 0
                event_specs = [frame]
            elif in_event:
                if mask[n] == 1:
                    zeros_count = 0
                    event_specs.append(frame)
                else:
                    zeros_count += 1

                if zeros_count >= self.offset_hysteresis:
                    length = (n - self.offset_hysteresis) - event_start
                    if length >= L_min:
                        # 最小ノート長より長いイベントのみを記録（ノイズ除去）
                        onset_frames.append(event_start)
                        offset_frames.append(n - self.offset_hysteresis)

                        # 平均スペクトルでパーカッシブ判定
                        avg_spec = np.mean(np.array(event_specs), axis=0)
                        percussive_flags.append(self._is_percussive(avg_spec))

                    in_event = False
                    event_specs = []

        # 終端処理: イベント中の場合
        if in_event:
            length = (N_frames - 1) - event_start
            if length >= L_min:
                # イベント長が最小ノート長以上の場合のみ記録（終端処理）
                onset_frames.append(event_start)
                offset_frames.append(N_frames - 1)
                avg_spec = np.mean(np.array(event_specs), axis=0)
                percussive_flags.append(self._is_percussive(avg_spec))

        onset_frames = np.array(onset_frames, dtype=int)
        offset_frames = np.array(offset_frames, dtype=int)
        percussive_flags = np.array(percussive_flags, dtype=bool)

        return {
            "mask": mask,
            "d_values": d_vals,
            "r_values": r_vals,
            "onset_frames": onset_frames,
            "offset_frames": offset_frames,
            "percussive_flags": percussive_flags,
        }

    def _frames_to_time(self, frames: np.ndarray, sr: int) -> np.ndarray:
        return frames * self.hop_length / sr

    def _detect_single_pitch(self, avg_spec: np.ndarray) -> Tuple[float, float]:
        peak_idx = np.argmax(avg_spec)
        peak_val = avg_spec[peak_idx]

        bin_width = 2 ** (1 / self.bins_per_octave)
        pitch_hz = self.f_min * (bin_width**peak_idx)

        mean_val = np.mean(avg_spec)
        if mean_val < self.eps:
            confidence = 0.0
        else:
            confidence = peak_val / mean_val

        # 倍音ブースト
        harmonic_boost = self._check_harmonics(avg_spec, peak_idx)
        confidence *= harmonic_boost

        # 判定
        if confidence < self.confidence_threshold:
            return -1.0, 0.0
        return pitch_hz, confidence

    def _check_harmonics(self, spectrum: np.ndarray, fundamental_idx: int) -> float:
        num_bins = len(spectrum)
        harmonic_indices = []

        for h in range(2, 5):
            harmonic_idx = fundamental_idx + int(np.log2(h) * self.bins_per_octave)
            if harmonic_idx < num_bins:
                harmonic_indices.append(harmonic_idx)

        if not harmonic_indices:
            return 1.0

        fundamental_val = spectrum[fundamental_idx]
        harmonic_vals = [spectrum[idx] for idx in harmonic_indices]
        harmonic_sum = sum(harmonic_vals)

        if fundamental_val < self.eps:
            return 1.0

        harmonic_boost = 1.0 + 0.5 * (harmonic_sum / fundamental_val)
        return harmonic_boost

    def _check_if_harmonic_series(self, peak_indices: List[int]) -> bool:
        if len(peak_indices) < 2:
            return False
        fundamental_idx = min(peak_indices)

        for idx in peak_indices:
            if idx == fundamental_idx:
                continue
            freq_ratio = 2 ** ((idx - fundamental_idx) / self.bins_per_octave)
            nearest_harm = round(freq_ratio)
            if nearest_harm >= 2 and abs(freq_ratio - nearest_harm) < 0.1:
                continue
            else:
                return False
        return True

    def _detect_chord(self, avg_spec: np.ndarray) -> Tuple[List[float], float]:
        if not self.enable_chord_detection:
            return [], 0.0

        peak_indices = []
        for i in range(1, len(avg_spec) - 1):
            if avg_spec[i] > avg_spec[i - 1] and avg_spec[i] > avg_spec[i + 1]:
                if avg_spec[i] > self.min_peak_ratio * np.max(avg_spec):
                    peak_indices.append(i)

        if len(peak_indices) < 2:
            return [], 0.0

        bin_width = 2 ** (1 / self.bins_per_octave)
        peak_freqs = [self.f_min * (bin_width**idx) for idx in peak_indices]
        peak_vals = [avg_spec[idx] for idx in peak_indices]

        # 倍音直列かどうか
        if self._check_if_harmonic_series(peak_indices):
            return [], 0.0

        mean_val = np.mean(avg_spec)
        if mean_val < self.eps:
            confidence = 0.0
        else:
            confidence = sum(peak_vals) / (mean_val * len(peak_vals))

        if confidence < self.confidence_threshold:
            return [], 0.0
        return peak_freqs, confidence

    def _detect_pitch(
        self, audio: np.ndarray, sr: int, event_results: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        オンセット/オフセットからのピッチ検出

        Parameters
        ----------
        audio : np.ndarray
            音声信号
        sr : int
            サンプリングレート
        event_results : Dict[str, np.ndarray]
            イベント検出結果

        Returns
        -------
        Dict[str, np.ndarray]
            ピッチ検出結果
            - pitches: ピッチ周波数（Hz）
            - confidences: 信頼度
            - pitch_frames: フレーム単位のピッチ値
            - pitch_times: ピッチフレームの時間
            - pitch_values: フレーム単位のピッチ値（Hz）
        """
        onset_frames = event_results["onset_frames"]
        offset_frames = event_results["offset_frames"]
        percussive_flags = event_results["percussive_flags"]

        if not self.enable_pitch_detection:
            # ピッチ検出が無効の場合は空データを返す
            pitches = np.full(len(onset_frames), -1.0)
            confidences = np.zeros(len(onset_frames))
            return {
                "pitches": pitches,
                "confidences": confidences,
                "pitch_frames": np.array([]),
                "pitch_times": np.array([]),
                "pitch_values": np.array([]),
            }

        # FFTとCQTのケース検討
        S = self._compute_cqt(audio, sr)

        # 結果保存用
        pitches = np.zeros(len(onset_frames))
        confidences = np.zeros(len(onset_frames))
        chord_info = []

        # ピッチフレームデータのシミュレーション用
        # CQT計算で使用したホップ長と合わせる
        hop_length = self.hop_length
        # 総フレーム数の計算
        total_frames = len(audio) // hop_length + 1
        # ピッチフレームデータの初期化
        pitch_frames = np.zeros(total_frames)
        pitch_times = np.arange(total_frames) * hop_length / sr

        # 各音符のピッチを検出
        for i, (onset, offset) in enumerate(zip(onset_frames, offset_frames)):
            if percussive_flags[i]:
                # 打楽器音の場合はピッチなし
                pitches[i] = -1
                confidences[i] = 0.0
                continue

            # CQTから該当区間の平均スペクトルを取得
            avg_spec = np.mean(S[:, onset : offset + 1], axis=1)

            # 和音検出を試みる（複数の候補がある場合）
            chord_pitches, conf = self._detect_chord(avg_spec)
            if chord_pitches:
                pitches[i] = chord_pitches[0]
                confidences[i] = conf
                chord_info.append(chord_pitches)

                # ピッチフレームデータに音符のピッチを割り当て
                if chord_pitches[0] > 10.0:  # 有効なピッチ
                    frame_start = max(0, onset)
                    frame_end = min(total_frames, offset + 1)
                    pitch_frames[frame_start:frame_end] = chord_pitches[0]
                continue

            # 単音ピッチ検出
            pitch, conf = self._detect_single_pitch(avg_spec)
            pitches[i] = pitch
            confidences[i] = conf

            # ピッチフレームデータに音符のピッチを割り当て
            if pitch > 10.0:  # 有効なピッチ (10Hz以上)
                frame_start = max(0, onset)
                frame_end = min(total_frames, offset + 1)
                pitch_frames[frame_start:frame_end] = pitch

        return {
            "pitches": pitches,
            "confidences": confidences,
            "pitch_frames": pitch_frames,
            "pitch_times": pitch_times,
            "pitch_values": pitch_frames,  # ピッチフレームをピッチ値としても使用
        }

    def detect(self, audio_data: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        音声データから音符を検出する内部メソッド

        Args:
            audio_data: 音声データ
            sr: サンプリングレート

        Returns:
            Dict[str, np.ndarray]: 検出結果
                - intervals: オンセット・オフセット時間のペア配列 (N, 2) [onset, offset]
                - note_pitches: ピッチ周波数（Hz）
                - frame_times: ピッチフレームの時間
                - frame_frequencies: フレーム単位のピッチ値（Hz）
                - detection_time: 検出処理時間（秒）
                - detector_name: 検出器の名前
        """
        start_time = time.time()

        # 音声データの検証
        if audio_data is None:
            raise ValueError("音声データが指定されていません")
        if len(audio_data) == 0:
            raise ValueError("音声データが空です")
        if sr <= 0:
            raise ValueError(f"不正なサンプリングレート: {sr}Hz")

        # イベント検出（オンセット/オフセット）
        logging.info(
            f"イベント検出を開始します (audio.shape={audio_data.shape}, sr={sr}Hz)"
        )
        event_results = self._detect_events(audio_data, sr)

        # ピッチ検出（設定で無効化可能）
        pitch_frames = None
        pitch_times = None
        pitch_values = None

        pitches = np.array([])
        confidences = np.array([])

        if self.enable_pitch_detection:
            if event_results["onset_frames"].size > 0:
                logging.info(
                    f"{len(event_results['onset_frames'])}個のイベントに対してピッチ検出を開始します"
                )
                pitch_results = self._detect_pitch(audio_data, sr, event_results)
                pitches = pitch_results["pitches"]
                confidences = pitch_results["confidences"]
                pitch_frames = pitch_results.get("pitch_frames")
                pitch_times = pitch_results.get("pitch_times")
                pitch_values = pitch_results.get("pitch_values")

                # 結果の整合性チェック
                if len(event_results["onset_frames"]) != len(pitches):
                    logging.warning(
                        f"検出結果の長さが一致しません: onsets({len(event_results['onset_frames'])}) vs pitches({len(pitches)})"
                    )
                    # 長さを揃える（小さい方に合わせる）
                    min_len = min(len(event_results["onset_frames"]), len(pitches))
                    event_results["onset_frames"] = event_results["onset_frames"][
                        :min_len
                    ]
                    event_results["offset_frames"] = event_results["offset_frames"][
                        :min_len
                    ]
                    event_results["percussive_flags"] = event_results[
                        "percussive_flags"
                    ][:min_len]
                    pitches = pitches[:min_len]
                    confidences = confidences[:min_len]
            else:
                logging.info("検出されたイベントがないため、ピッチ検出をスキップします")
        else:
            logging.info("ピッチ検出が無効化されています")
            # ピッチ検出が無効でも、検出された音符数に合わせてダミーピッチを生成
            if event_results["onset_frames"].size > 0:
                pitches = np.zeros(len(event_results["onset_frames"]))
                confidences = np.zeros(len(event_results["onset_frames"]))

        # 検出時間を計算
        detection_time = time.time() - start_time

        # オンセット/オフセット時間に変換（フレームからの変換）
        onsets = event_results["onset_frames"] * self.hop_length / sr
        offsets = event_results["offset_frames"] * self.hop_length / sr

        # インターバル配列を作成 (N,2) [onset, offset]
        intervals = (
            np.column_stack((onsets, offsets))
            if len(onsets) > 0
            else np.array([]).reshape(0, 2)
        )

        # 有効なピッチのマスクを生成（-1はピッチなしとする）
        if len(pitches) > 0:
            valid_pitch_mask = pitches > 0
            # 無効なピッチを持つ音符も含め、すべての音符のインターバルを保持
            # ただし、pitchesが0以下の場合は1.0Hzに修正して評価に含める
            pitches_for_eval = pitches.copy()
            pitches_for_eval[~valid_pitch_mask] = 1.0  # 評価用に1Hzに設定（ほぼ無音）
        else:
            valid_pitch_mask = np.array([], dtype=bool)
            pitches_for_eval = np.array([])

        # 検出完了ログ
        logging.info(
            f"音符検出が完了しました: {len(onsets)}個の音符を検出 (処理時間: {detection_time:.3f}秒)"
        )

        # 結果を返す
        return {
            # 評価システムで必須の情報
            "intervals": intervals,
            "note_pitches": pitches_for_eval,  # 評価用に調整したピッチ
            "frame_times": pitch_times if pitch_times is not None else np.array([]),
            "frame_frequencies": (
                pitch_values if pitch_values is not None else np.array([])
            ),
            # 共通メタ情報
            "detector_name": self.__class__.__name__,
            "detection_time": detection_time,
        }

    def detect_onsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """オンセットのみを検出する簡略版メソッド"""
        event_results = self._detect_events(audio, sr)
        return self._frames_to_time(event_results["onset_frames"], sr)

    def detect_offsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """オフセットのみを検出する簡略版メソッド"""
        event_results = self._detect_events(audio, sr)
        return self._frames_to_time(event_results["offset_frames"], sr)

    def detect_pitch_from_events(
        self,
        audio: np.ndarray,
        sr: int,
        onset_frames: np.ndarray,
        offset_frames: np.ndarray,
        percussive_flags: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        既知のイベント位置からピッチを検出する

        Args:
            audio: 音声データ
            sr: サンプリングレート
            onset_frames: オンセットフレーム
            offset_frames: オフセットフレーム
            percussive_flags: 打楽器フラグ

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: ピッチ, 信頼度, 打楽器フラグ
        """

        pitch_results = self._detect_pitch(
            audio,
            sr,
            {
                "onset_frames": onset_frames,
                "offset_frames": offset_frames,
                "percussive_flags": percussive_flags,
            },
        )
        return pitch_results["pitches"], pitch_results["confidences"], percussive_flags

    def detect_notes(
        self,
        audio: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        audio_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        音声データまたは音声ファイルから音符を検出する

        Args:
            audio: 音声データ（オプション）
            sr: サンプリングレート（オプション）
            audio_path: 音声ファイルのパス（オプション）

        Returns:
            Dict[str, np.ndarray]: 検出結果
                - intervals: オンセット・オフセット時間のペア配列 (N, 2) [onset, offset]
                - note_pitches: ピッチ周波数（Hz）
                - times: ピッチフレームの時間
                - freqs: フレーム単位のピッチ値（Hz）
                - detection_time: 検出処理時間（秒）
                - detector_name: 検出器の名前
                - pitch_frames: フレーム単位のピッチ値
                - onset_frames: オンセットフレーム位置
                - offset_frames: オフセットフレーム位置
                - chord_info: 和音情報 (該当する場合)
                - percussive_flags: パーカッシブフラグ
                - mask: マスク情報 (該当する場合)
        """
        start_time = time.time()

        try:
            # audio_pathが指定された場合は音声ファイルを読み込む
            if audio_path is not None:
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(
                        f"音声ファイルが見つかりません: {audio_path}"
                    )
                try:
                    audio, sr = librosa.load(audio_path, sr=44100, mono=True)
                    logging.info(
                        f"音声ファイルを読み込みました: {audio_path} (sr={sr}Hz, length={len(audio)}samples)"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"音声ファイルの読み込みに失敗しました: {str(e)}"
                    )
            elif audio is None or sr is None:
                raise ValueError(
                    "audio_pathまたはaudioとsrの両方を指定する必要があります"
                )

            # 音声データの基本的な検証
            if len(audio) == 0:
                raise ValueError("音声データが空です")
            if sr <= 0:
                raise ValueError(f"不正なサンプリングレート: {sr}Hz")

            # detectメソッドを呼び出し
            logging.info(f"音符検出を開始します (audio.shape={audio.shape}, sr={sr}Hz)")
            results = self.detect(audio_data=audio, sr=sr)

            # 結果の検証
            if not isinstance(results, dict):
                raise RuntimeError(
                    f"detectメソッドが不正な型を返しました: {type(results)}"
                )

            # 検出完了ログ
            logging.info(
                f"音符検出が完了しました: {len(results.get('intervals', []))}個の音符を検出 (処理時間: {time.time() - start_time:.3f}秒)"
            )

            return results

        except Exception as e:
            # エラー時は空の結果を返す
            logging.error(f"音符検出中にエラーが発生しました: {str(e)}")

            # 空の結果を返す
            empty_result = {
                "intervals": np.array([]).reshape(0, 2),
                "note_pitches": np.array([]),
                "frame_times": np.array([]),
                "frame_frequencies": np.array([]),
                "detection_time": time.time() - start_time,
                "detector_name": self.__class__.__name__,
            }

            return empty_result

    # パラメータ感度分析用メソッド
    def analyze_parameter_sensitivity(
        self, audio_data: np.ndarray, sr: int, param_name: str, values: List[float]
    ) -> Dict[str, List]:
        """
        指定されたパラメータの感度分析を行います

        Args:
            audio_data: 音声データ
            sr: サンプリングレート
            param_name: 分析するパラメータ名
            values: テストする値のリスト

        Returns:
            Dict[str, List]: パラメータごとの検出結果
        """
        results = {
            "param_values": values,
            "event_counts": [],
            "detection_times": [],
            "details": [],
        }

        # 元の値を保存
        original_value = getattr(self, param_name, None)
        if original_value is None:
            raise ValueError(f"パラメータ '{param_name}' は存在しません")

        try:
            for value in values:
                # パラメータを一時的に変更
                setattr(self, param_name, value)

                # 検出を実行
                detection_result = self.detect(audio_data, sr)

                # 結果を保存
                event_count = len(detection_result["intervals"])
                results["event_counts"].append(event_count)
                results["detection_times"].append(detection_result["detection_time"])
                results["details"].append(
                    {
                        "value": value,
                        "events": event_count,
                        "time": detection_result["detection_time"],
                    }
                )

                logging.info(
                    f"パラメータ {param_name}={value} の場合: {event_count}個のイベントを検出 (処理時間: {detection_result['detection_time']:.3f}秒)"
                )

        finally:
            # 元の値に戻す
            setattr(self, param_name, original_value)

        return results

    def evaluate_with_multiple_parameters(
        self, audio_data: np.ndarray, sr: int
    ) -> Dict[str, Any]:
        """
        複数のパラメータ組み合わせでONDEDetectorを評価します

        Args:
            audio_data: 音声データ
            sr: サンプリングレート

        Returns:
            Dict[str, Any]: 各パラメータ設定の評価結果
        """
        # パラメータ組み合わせのリスト
        parameter_sets = [
            {"Theta": 0.8, "c_T": 1.2, "c_min": 0.03},
            {"Theta": 1.0, "c_T": 1.4, "c_min": 0.03},
            {"Theta": 1.2, "c_T": 1.6, "c_min": 0.04},
            {"Theta": 0.8, "c_T": 1.6, "c_min": 0.02},
            {"Theta": 1.2, "c_T": 1.2, "c_min": 0.04},
        ]

        results = []

        # 元のパラメータを保存
        original_params = {"Theta": self.Theta, "c_T": self.c_T, "c_min": self.c_min}

        try:
            for params in parameter_sets:
                # パラメータを変更
                self.Theta = params["Theta"]
                self.c_T = params["c_T"]
                self.c_min = params["c_min"]

                # 検出を実行
                start_time = time.time()
                detection_result = self.detect(audio_data, sr)
                detection_time = time.time() - start_time

                # 結果をまとめる
                result = {
                    "parameters": params.copy(),
                    "event_count": len(detection_result["intervals"]),
                    "detection_time": detection_time,
                    "intervals": detection_result["intervals"],
                }

                results.append(result)

                logging.info(
                    f"パラメータ: Theta={params['Theta']}, c_T={params['c_T']}, c_min={params['c_min']} で {result['event_count']}個のイベントを検出 (処理時間: {detection_time:.3f}秒)"
                )

        finally:
            # 元のパラメータに戻す
            self.Theta = original_params["Theta"]
            self.c_T = original_params["c_T"]
            self.c_min = original_params["c_min"]

        return {"parameter_sets": parameter_sets, "results": results}
