"""
Nuonde (新音検出) - 高度な時間周波数統合による音声検出アルゴリズム

このモジュールは高度な時間-周波数表現と適応的背景モデリングを用いた
音符検出・ピッチ推定アルゴリズムを実装しています。
"""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import scipy.signal

from src.detectors.base_detector import BaseDetector


class NuondeDetector(BaseDetector):
    """
    Nuonde (新音検出) - 高度な時間周波数統合による音声検出アルゴリズム

    このクラスは以下の特徴を持つアルゴリズムを実装しています：
    - 適応的時間-周波数表現（非定常Gabor変換ベース）
    - 動的背景モデリング
    - 構造的一貫性係数を用いた周期性評価
    - 調和-ピッチ統合による堅牢な検出
    - 状態遷移検出による正確なオンセット・オフセット推定
    """

    def __init__(
        self,
        # 時間-周波数表現パラメータ
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 65.4,  # C2の周波数
        f_max: float = 2093.0,  # C7の周波数
        # 検出パラメータ
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.3,
        min_note_length: float = 0.1,  # 秒
        median_filter_width: int = 3,
        # 背景モデルパラメータ
        background_adaptation_rate: float = 0.05,
        background_threshold: float = 2.0,
        # 周期性評価パラメータ
        synergy_threshold: float = 0.6,
        consistency_threshold: float = 0.7,
        # その他のパラメータ
        use_adaptive_window: bool = True,
        pitch_correction: bool = True,
        **kwargs,
    ):
        """
        Nuondeアルゴリズムの初期化

        Parameters
        ----------
        n_fft : int
            FFTのサイズ
        hop_length : int
            フレーム間のサンプル数
        f_min : float
            分析する最低周波数 (Hz)
        f_max : float
            分析する最高周波数 (Hz)
        onset_threshold : float
            オンセット検出の閾値
        offset_threshold : float
            オフセット検出の閾値
        min_note_length : float
            最小ノート長 (秒)
        median_filter_width : int
            メディアンフィルタの幅
        background_adaptation_rate : float
            背景モデルの適応速度
        background_threshold : float
            背景モデルからの逸脱判定閾値
        synergy_threshold : float
            相乗効果指標の閾値
        consistency_threshold : float
            構造的一貫性の閾値
        use_adaptive_window : bool
            適応窓を使用するかどうか
        pitch_correction : bool
            ピッチ補正を行うかどうか
        **kwargs : dict
            その他のパラメータ
        """
        super().__init__(**kwargs)

        # 基本パラメータの設定
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max

        # 検出パラメータの設定
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_note_length = min_note_length
        self.median_filter_width = median_filter_width

        # 背景モデルパラメータの設定
        self.background_adaptation_rate = background_adaptation_rate
        self.background_threshold = background_threshold

        # 周期性評価パラメータの設定
        self.synergy_threshold = synergy_threshold
        self.consistency_threshold = consistency_threshold

        # その他のパラメータ設定
        self.use_adaptive_window = use_adaptive_window
        self.pitch_correction = pitch_correction

        # ロガーの設定
        self.logger = logging.getLogger(__name__)

    def compute_adaptive_tfr(
        self, audio_data: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        適応的時間-周波数表現（Adaptive Time-Frequency Representation）を計算します。

        非定常Gabor変換をベースとし、周波数に応じて適応的な窓幅を使用します。

        Parameters
        ----------
        audio_data : np.ndarray
            音声データ
        sr : int
            サンプリングレート

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (tfr, freqs, times)
            - tfr: 時間-周波数表現（振幅）
            - freqs: 周波数軸
            - times: 時間軸
        """
        # 前処理：信号の正規化
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)

        # 周波数軸の設定
        n_bins = int(
            np.ceil(np.log2(self.f_max / self.f_min) * 12)
        )  # オクターブあたり12ビン
        freqs = self.f_min * 2 ** (np.arange(n_bins) / 12)
        freqs = freqs[freqs <= self.f_max]

        # 時間フレームの計算
        n_frames = 1 + int((len(audio_data) - self.n_fft) / self.hop_length)

        # 一般的なSTFTを計算
        stft = librosa.stft(
            audio_data,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window="hann",
            center=True,
        )

        # 実際の時間フレーム数に基づいて時間軸を計算
        actual_n_frames = stft.shape[1]
        times = librosa.frames_to_time(
            np.arange(actual_n_frames), sr=sr, hop_length=self.hop_length
        )

        # 時間-周波数表現を格納する配列を初期化
        tfr = np.zeros((len(freqs), len(times)), dtype=np.float32)

        if self.use_adaptive_window:
            # 周波数に応じた適応窓を使用
            for i, freq in enumerate(freqs):
                # 周波数に応じて窓幅を計算（低周波では広い窓、高周波では狭い窓）
                window_size = int(min(self.n_fft, sr * 8 / freq))
                window_size = max(32, window_size)  # 最小窓サイズを保証
                window_size = min(self.n_fft, window_size)  # 最大窓サイズを保証
                window_size = 2 ** int(np.log2(window_size))  # 2の累乗に切り捨て

                # ハミング窓を作成
                window = np.hamming(window_size)

                # 周波数のビン位置を計算
                bin_idx = int(freq * self.n_fft / sr)
                bin_idx = min(bin_idx, self.n_fft // 2 - 1)

                # 適応窓でのSTFTを計算（center=Trueを使用）
                adaptive_stft = librosa.stft(
                    audio_data,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=window_size,
                    window=window,
                    center=True,
                )

                # 特定の周波数帯域のマグニチュードを抽出
                # 実際のフレーム数と一致するように調整
                band_mag = np.abs(adaptive_stft[bin_idx, : len(times)])

                # 時間-周波数表現に格納
                tfr[i, :] = band_mag
        else:
            # 通常のSTFTを使用（すでに計算済み）
            for i, freq in enumerate(freqs):
                bin_idx = int(freq * self.n_fft / sr)
                bin_idx = min(bin_idx, self.n_fft // 2 - 1)
                tfr[i, :] = np.abs(stft[bin_idx, :])

        # 対数スケールの振幅に変換（騒音と信号の差を強調）
        tfr = np.log1p(tfr)

        return tfr, freqs, times

    def compute_optimal_tfr(
        self, audio_data: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        情報量最大化原理に基づく最適時間-周波数表現を計算します。

        微分幾何学的観点から信号の内在的構造を保存する表現を構築します。

        Parameters
        ----------
        audio_data : np.ndarray
            音声データ
        sr : int
            サンプリングレート

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (tfr, freqs, times)
            - tfr: 時間-周波数表現（振幅）
            - freqs: 周波数軸
            - times: 時間軸
        """
        # 前処理：信号の正規化（L2ノルム）
        audio_data = audio_data / (np.linalg.norm(audio_data) + 1e-10)

        # 解析的信号への変換（ヒルベルト変換による複素表現）
        analytic_signal = scipy.signal.hilbert(audio_data)

        # 瞬時振幅と瞬時位相の抽出
        inst_amplitude = np.abs(analytic_signal)
        inst_phase = np.unwrap(np.angle(analytic_signal))
        inst_frequency = np.diff(inst_phase, prepend=inst_phase[0]) / (2.0 * np.pi) * sr

        # 最適な周波数分布の構築（情報エントロピー最大化）
        f_min, f_max = self.f_min, self.f_max

        # 音楽的に意味のある周波数分布（対数間隔に基づく最適化）
        beta = 12  # 1オクターブあたりのビン数（音楽理論に基づく）
        n_bins = int(np.ceil(beta * np.log2(f_max / f_min)))

        # 周波数ビンを音楽理論に基づいて配置
        central_freqs = f_min * 2 ** (np.arange(n_bins) / beta)
        central_freqs = central_freqs[central_freqs <= f_max]

        # 複素変調辞書構築（ガボールウェーブレット）
        tfr_complex = np.zeros((len(central_freqs), len(audio_data)), dtype=complex)

        for i, cf in enumerate(central_freqs):
            # 周波数に応じた最適帯域幅（不確定性原理に基づく）
            bandwidth = cf * np.sqrt(2) / beta

            # 時間解像度と周波数解像度の最適バランス
            sigma_t = np.sqrt(1 / (2 * np.pi * bandwidth))

            # ガボールウェーブレットの構築
            t = np.arange(-int(4 * sigma_t * sr), int(4 * sigma_t * sr) + 1) / sr
            if len(t) > 1:
                window = np.exp(-0.5 * (t / sigma_t) ** 2)
                # 複素正弦波による変調
                wavelet = window * np.exp(2j * np.pi * cf * t)

                # 畳み込みによる時間-周波数表現の計算
                tfr_complex[i] = scipy.signal.fftconvolve(
                    audio_data, wavelet, mode="same"
                )

        # 適応的時間ステップの計算
        adaptive_hop = int(sr * 0.01)  # 10ms
        time_indices = np.arange(0, len(audio_data), adaptive_hop)
        times = time_indices / sr

        # 最終的な時間-周波数表現
        tfr = np.abs(tfr_complex[:, time_indices])

        # 対数スケール変換（知覚的に意味のある表現）
        tfr = np.log1p(tfr)

        return tfr, central_freqs, times

    def compute_structural_consistency(
        self, tfr: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """
        構造的一貫性係数（Structural Consistency Coefficient）を計算します。

        この係数は、時間-周波数表現における局所的な構造の一貫性を評価します。
        高い値は安定した周期的パターンの存在を示し、低い値はノイズや非定常部分を示します。

        Parameters
        ----------
        tfr : np.ndarray
            時間-周波数表現
        window_size : int
            局所的な分析に使用する窓サイズ

        Returns
        -------
        np.ndarray
            構造的一貫性係数
        """
        n_freqs, n_times = tfr.shape
        consistency = np.zeros((n_freqs, n_times))

        # パディングを行ったTFRを作成
        pad_width = window_size // 2
        tfr_padded = np.pad(tfr, ((0, 0), (pad_width, pad_width)), mode="edge")

        for i in range(n_freqs):
            for j in range(n_times):
                # 局所的な窓を取得
                local_window = tfr_padded[i, j : j + window_size]

                # 窓内の変動係数を計算（標準偏差/平均）
                mean_val = np.mean(local_window)
                if mean_val > 0:
                    std_val = np.std(local_window)
                    # 変動係数の逆数を一貫性係数として使用
                    variation_coef = std_val / mean_val
                    consistency[i, j] = 1.0 / (1.0 + variation_coef)
                else:
                    consistency[i, j] = 0.0

        return consistency

    def compute_structural_coherence(
        self, tfr: np.ndarray, freqs: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """
        構造的一貫性を位相空間幾何学の観点から再定義します。

        局所的位相空間の曲率として一貫性を定式化します。

        Parameters
        ----------
        tfr : np.ndarray
            時間-周波数表現
        freqs : np.ndarray
            周波数軸
        times : np.ndarray
            時間軸

        Returns
        -------
        np.ndarray
            構造的一貫性係数
        """
        n_freqs, n_times = tfr.shape
        coherence = np.zeros((n_freqs, n_times))

        # リーマン計量テンソルの近似計算
        grad_f = np.gradient(tfr, axis=0)  # 周波数方向の勾配
        grad_t = np.gradient(tfr, axis=1)  # 時間方向の勾配

        # ヘッセ行列の近似計算
        hess_ff = np.gradient(grad_f, axis=0)
        hess_ft = np.gradient(grad_f, axis=1)
        hess_tt = np.gradient(grad_t, axis=1)

        # 局所的構造のリッチ曲率近似値を計算
        for i in range(n_freqs):
            for j in range(n_times):
                if i > 0 and i < n_freqs - 1 and j > 0 and j < n_times - 1:
                    # 局所的な曲率の計算
                    det_hessian = hess_ff[i, j] * hess_tt[i, j] - hess_ft[i, j] ** 2
                    trace_hessian = hess_ff[i, j] + hess_tt[i, j]

                    # リッチ曲率に基づく一貫性指標
                    if trace_hessian != 0:
                        ricci_curvature = det_hessian / trace_hessian

                        # 一貫性は局所的な曲率の滑らかさとして定義
                        coherence[i, j] = 1.0 / (1.0 + np.abs(ricci_curvature))

                        # 局所的な変動の方向性を考慮
                        anisotropy = np.abs(hess_ff[i, j] - hess_tt[i, j]) / (
                            np.abs(hess_ff[i, j]) + np.abs(hess_tt[i, j]) + 1e-10
                        )
                        coherence[i, j] *= 1.0 - 0.5 * anisotropy  # 等方的な変化を優先

        # 窓関数による平滑化（スペクトル漏れの軽減）
        coherence = scipy.signal.convolve2d(
            coherence,
            np.outer(
                scipy.signal.windows.gaussian(5, 1.0),
                scipy.signal.windows.gaussian(5, 1.0),
            ),
            mode="same",
            boundary="symm",
        )

        return coherence

    def compute_dynamic_background(self, tfr: np.ndarray) -> np.ndarray:
        """
        動的背景モデル（Dynamic Background Model）を計算します。

        このモデルは、時間-周波数表現における各周波数ビンの局所的な背景レベルを推定します。
        適応的な方法で背景レベルを更新し、ノイズや環境音から有意な音符成分を分離します。

        Parameters
        ----------
        tfr : np.ndarray
            時間-周波数表現

        Returns
        -------
        np.ndarray
            動的背景モデル
        """
        n_freqs, n_times = tfr.shape
        background = np.zeros_like(tfr)

        # 最初のフレームを初期背景として設定
        if n_times > 0:
            background[:, 0] = tfr[:, 0]

        # 時間方向に背景モデルを更新
        for t in range(1, n_times):
            # 適応速度（レート）を使って背景モデルを更新
            mask = tfr[:, t] <= background[:, t - 1] * (1 + self.background_threshold)

            # 閾値以下の部分は背景と見なし、ゆっくり適応
            background[mask, t] = (1 - self.background_adaptation_rate) * background[
                mask, t - 1
            ] + self.background_adaptation_rate * tfr[mask, t]

            # 閾値を超える部分は前フレームの値を保持（音符部分が背景に入らないように）
            background[~mask, t] = background[~mask, t - 1]

        return background

    def compute_synergy_index(
        self, tfr: np.ndarray, background: np.ndarray, consistency: np.ndarray
    ) -> np.ndarray:
        """
        相乗効果指標（Synergy Index）を計算します。

        この指標は、時間-周波数表現、背景モデル、構造的一貫性係数を統合し、
        音符の存在確率を表す総合的な指標を生成します。

        Parameters
        ----------
        tfr : np.ndarray
            時間-周波数表現
        background : np.ndarray
            動的背景モデル
        consistency : np.ndarray
            構造的一貫性係数

        Returns
        -------
        np.ndarray
            相乗効果指標
        """
        # 背景からの逸脱を計算（信号と背景の比）
        deviation = np.zeros_like(tfr)
        mask = background > 0
        deviation[mask] = tfr[mask] / background[mask]

        # 背景からの逸脱を非線形変換して強調
        deviation = 1.0 - 1.0 / (1.0 + deviation)

        # 構造的一貫性と背景からの逸脱を組み合わせた相乗効果指標
        synergy = deviation * consistency

        # メディアンフィルタで平滑化（ノイズ除去）
        if self.median_filter_width > 1:
            for i in range(synergy.shape[0]):
                synergy[i, :] = scipy.signal.medfilt(
                    synergy[i, :], self.median_filter_width
                )

        return synergy

    def compute_synergy_index_advanced(
        self,
        tfr: np.ndarray,
        background: np.ndarray,
        coherence: np.ndarray,
        freqs: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """
        相乗効果指標を情報理論の枠組みで再定義します。

        相互情報量と転移エントロピーの概念を導入します。

        Parameters
        ----------
        tfr : np.ndarray
            時間-周波数表現
        background : np.ndarray
            動的背景モデル
        coherence : np.ndarray
            構造的一貫性係数
        freqs : np.ndarray
            周波数軸
        times : np.ndarray
            時間軸

        Returns
        -------
        np.ndarray
            拡張相乗効果指標
        """
        n_freqs, n_times = tfr.shape

        # 背景からの情報利得（KLダイバージェンス）
        info_gain = np.zeros_like(tfr)
        mask = background > 0
        if np.any(mask):
            # 正規化エネルギー分布
            p_signal = tfr / (np.sum(tfr) + 1e-10)
            p_background = background / (np.sum(background) + 1e-10)

            # KLダイバージェンスの近似計算
            ratio = np.ones_like(p_signal)
            ratio[mask] = p_signal[mask] / (p_background[mask] + 1e-10)
            info_gain = p_signal * np.log1p(ratio)

        # 構造的一貫性による重み付け
        weighted_info = info_gain * coherence

        # 周波数的連続性の評価（時間方向の自己相関）
        continuity = np.zeros_like(tfr)
        for i in range(n_freqs):
            if n_times > 1:
                # 自己相関に基づく連続性指標
                signal = tfr[i, :]
                acorr = np.correlate(signal, signal, mode="full")[len(signal) - 1 :]
                if len(acorr) > 1:
                    acorr = acorr / (acorr[0] + 1e-10)
                    decay_rate = -np.polyfit(
                        np.arange(min(5, len(acorr))),
                        np.log(acorr[: min(5, len(acorr))] + 1e-10),
                        1,
                    )[0]

                    # 減衰率に基づく連続性スコア
                    continuity[i, :] = np.exp(-decay_rate)

        # 隣接周波数の相互情報量（周波数軸に沿った情報の伝搬）
        mutual_info = np.zeros_like(tfr)
        for i in range(1, n_freqs - 1):
            for j in range(n_times):
                # 3点の局所的相互情報量
                triplet = tfr[i - 1 : i + 2, j]
                if np.all(triplet > 0):
                    # 標準化
                    triplet = triplet / np.sum(triplet)

                    # エントロピー計算
                    entropy = -np.sum(triplet * np.log2(triplet + 1e-10))
                    max_entropy = np.log2(3)  # 最大可能エントロピー

                    # 情報量に基づく相互関連度
                    mutual_info[i, j] = 1.0 - entropy / max_entropy

        # 総合的な相乗効果指標（多角的評価の統合）
        synergy = weighted_info + 0.3 * continuity + 0.2 * mutual_info

        # 非線形強調（重要なパターンの浮き彫り）
        synergy = np.tanh(2.0 * synergy)

        return synergy

    def detect_pitch(
        self, synergy: np.ndarray, freqs: np.ndarray, times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        相乗効果指標からフレームごとのピッチを検出します。

        各時間フレームにおいて、最大の相乗効果指標を持つ周波数を選択し、
        それをそのフレームのピッチとして推定します。

        Parameters
        ----------
        synergy : np.ndarray
            相乗効果指標
        freqs : np.ndarray
            周波数軸
        times : np.ndarray
            時間軸

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (frame_frequencies, frame_confidences)
            - frame_frequencies: フレームごとのピッチ（Hz）
            - frame_confidences: フレームごとの信頼度
        """
        n_freqs, n_times = synergy.shape
        frame_frequencies = np.zeros(n_times)
        frame_confidences = np.zeros(n_times)

        for t in range(n_times):
            # このフレームの最大相乗効果指標を持つ周波数を見つける
            max_idx = np.argmax(synergy[:, t])
            max_val = synergy[max_idx, t]

            # 閾値以上の場合にのみピッチとして採用
            if max_val >= self.synergy_threshold:
                frame_frequencies[t] = freqs[max_idx]
                frame_confidences[t] = max_val

        # ピッチ補正（オプション）
        if self.pitch_correction:
            self._correct_pitch_trajectory(frame_frequencies, frame_confidences)

        return frame_frequencies, frame_confidences

    def _correct_pitch_trajectory(
        self,
        frame_frequencies: np.ndarray,
        frame_confidences: np.ndarray,
        max_jump_ratio: float = 0.1,
        min_segment_length: int = 3,
    ) -> None:
        """
        ピッチ軌跡を補正して滑らかにします。

        Parameters
        ----------
        frame_frequencies : np.ndarray
            フレームごとのピッチ
        frame_confidences : np.ndarray
            フレームごとの信頼度
        max_jump_ratio : float
            許容される最大の周波数ジャンプ比率
        min_segment_length : int
            最小のセグメント長
        """
        n_frames = len(frame_frequencies)

        # 短すぎる有声セグメントを無声化
        i = 0
        while i < n_frames:
            if frame_frequencies[i] > 0:
                segment_start = i
                while i < n_frames and frame_frequencies[i] > 0:
                    i += 1
                segment_end = i
                segment_length = segment_end - segment_start

                if segment_length < min_segment_length:
                    frame_frequencies[segment_start:segment_end] = 0
                    frame_confidences[segment_start:segment_end] = 0
            else:
                i += 1

        # 急激なピッチ変化を補正
        for i in range(1, n_frames):
            if frame_frequencies[i] > 0 and frame_frequencies[i - 1] > 0:
                freq_ratio = frame_frequencies[i] / frame_frequencies[i - 1]

                # 比率が大きすぎる/小さすぎる場合は異常と判断
                if freq_ratio > 1 + max_jump_ratio or freq_ratio < 1 / (
                    1 + max_jump_ratio
                ):
                    # 信頼度が低い方を採用
                    if frame_confidences[i] < frame_confidences[i - 1]:
                        frame_frequencies[i] = frame_frequencies[i - 1]
                    else:
                        frame_frequencies[i - 1] = frame_frequencies[i]

    def compute_harmonic_profile(
        self, tfr: np.ndarray, freqs: np.ndarray, f0: float, t: int
    ) -> np.ndarray:
        """
        楽器非依存の調和音プロファイルを計算します。

        Parameters
        ----------
        tfr : np.ndarray
            時間-周波数表現
        freqs : np.ndarray
            周波数軸
        f0 : float
            基本周波数
        t : int
            時間フレームインデックス

        Returns
        -------
        np.ndarray
            調和音プロファイル（各倍音の強度）
        """
        # 最大倍音次数（周波数範囲と基本周波数に依存）
        h_max = min(10, int(freqs[-1] / f0))
        harmonic_profile = np.zeros(h_max + 1)

        # 基本周波数のエネルギー
        f0_idx = np.argmin(np.abs(freqs - f0))
        f0_energy = tfr[f0_idx, t]
        harmonic_profile[1] = 1.0  # 基準

        # 幅広い楽器に対応する中程度の減衰係数
        beta = 1.3

        # 倍音エネルギーの計算と正規化
        for h in range(2, h_max + 1):
            h_freq = f0 * h
            if h_freq <= freqs[-1]:
                h_idx = np.argmin(np.abs(freqs - h_freq))
                # 高次倍音は自然に減衰する重み付け
                harmonic_profile[h] = (tfr[h_idx, t] / (f0_energy + 1e-10)) * (h**-beta)

        return harmonic_profile

    def analyze_harmonic_structure(
        self, tfr: np.ndarray, freqs: np.ndarray, f0_candidates: np.ndarray, t: int
    ) -> np.ndarray:
        """
        楽器非依存の汎用的なハーモニック構造分析

        Parameters
        ----------
        tfr : np.ndarray
            時間-周波数表現（スペクトログラム）
        freqs : np.ndarray
            周波数軸
        f0_candidates : np.ndarray
            基本周波数候補の配列
        t : int
            分析する時間フレーム

        Returns
        -------
        np.ndarray
            各候補のハーモニックスコア
        """
        harmonic_scores = np.zeros_like(f0_candidates)

        # 周波数解像度の計算
        freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1

        # スペクトル全体のエネルギー
        total_frame_energy = np.sum(np.abs(tfr[:, t]))
        if total_frame_energy < 1e-8:
            return harmonic_scores  # 無音フレームは処理しない

        # 各候補を評価
        for i, f0 in enumerate(f0_candidates):
            if f0 <= 0:
                continue

            # 基本性能を高めるため最大倍音数を適応的に設定
            # 高音では少ない倍音数、低音ではより多くの倍音を考慮
            max_harmonic = min(20, int(np.floor(freqs[-1] / f0)))
            if max_harmonic < 3:  # 最低でも3倍音は必要
                continue

            # 倍音エネルギーと周波数幅を保存
            harmonic_energies = np.zeros(max_harmonic)
            harmonic_bandwidth = np.zeros(max_harmonic)

            # 基本周波数と倍音のエネルギーを抽出
            for h in range(1, max_harmonic + 1):
                h_freq = f0 * h
                if h_freq >= freqs[-1]:
                    break

                # 周波数ビンのインデックスを計算
                center_idx = np.argmin(np.abs(freqs - h_freq))

                # 倍音の周波数幅は基本周波数に比例
                bandwidth = max(3, int(np.ceil(h * f0 * 0.05 / freq_resolution)))
                start_idx = max(0, center_idx - bandwidth)
                end_idx = min(len(freqs) - 1, center_idx + bandwidth + 1)

                # 倍音エネルギーを抽出（窓内の最大値）
                harmonic_region = np.abs(tfr[start_idx:end_idx, t])
                if len(harmonic_region) > 0:
                    harmonic_energies[h - 1] = np.max(harmonic_region)

                    # ピークの幅も記録（倍音の形状特性）
                    peak_indices = np.where(
                        harmonic_region > 0.5 * np.max(harmonic_region)
                    )[0]
                    if len(peak_indices) > 0:
                        harmonic_bandwidth[h - 1] = (
                            peak_indices[-1] - peak_indices[0] + 1
                        ) * freq_resolution

            # 基本周波数のエネルギーを基準にした相対的な強度
            fundamental_energy = harmonic_energies[0]
            if fundamental_energy > 0:
                relative_energies = harmonic_energies / fundamental_energy
            else:
                continue  # 基本周波数のエネルギーがない場合はスキップ

            # 評価基準1: 倍音構造の一貫性
            # 実際の楽器音では倍音はある程度予測可能なパターンで減衰する
            expected_decay = np.exp(-0.5 * np.arange(max_harmonic))
            decay_consistency = 1.0 - np.mean(
                np.abs(relative_energies - expected_decay) / (expected_decay + 0.1)
            )
            decay_consistency = max(0, decay_consistency)  # 負の値を防止

            # 評価基準2: 倍音の存在比率
            # 実際の音ではある程度の倍音が存在するはず
            harmonic_presence = (
                np.sum(harmonic_energies > 0.1 * fundamental_energy) / max_harmonic
            )

            # 評価基準3: 倍音エネルギーの総和と基本周波数の比率
            energy_ratio = np.sum(harmonic_energies) / (total_frame_energy + 1e-8)
            energy_concentration = min(1.0, energy_ratio * 2)  # 正規化

            # 評価基準4: 倍音の周波数整合性
            # 実際の倍音はきれいな整数倍に近いはず
            freq_coherence = 0.0
            for h in range(2, max_harmonic + 1):
                if harmonic_energies[h - 1] > 0.05 * fundamental_energy:
                    # h倍の周波数からのずれを評価
                    expected_freq = f0 * h
                    nearest_peak_idx = start_idx + np.argmax(
                        np.abs(tfr[start_idx:end_idx, t])
                    )
                    if start_idx <= nearest_peak_idx < end_idx:
                        actual_freq = freqs[nearest_peak_idx]
                        # 相対的な周波数誤差
                        rel_error = abs(actual_freq - expected_freq) / expected_freq
                        freq_coherence += np.exp(
                            -10 * rel_error
                        )  # 誤差が小さいほど高スコア

            # 有効な倍音数で正規化
            valid_harmonics = np.sum(harmonic_energies > 0.05 * fundamental_energy)
            if valid_harmonics > 1:
                freq_coherence /= valid_harmonics - 1  # 基本周波数を除く
            else:
                freq_coherence = 0.0

            # 総合スコアの計算 - 各評価基準の重み付け和
            score = (
                0.3 * decay_consistency
                + 0.3 * harmonic_presence
                + 0.2 * energy_concentration
                + 0.2 * freq_coherence
            )

            harmonic_scores[i] = score

        return harmonic_scores

    def detect_harmonics(
        self,
        tfr: np.ndarray,
        freqs: np.ndarray,
        frame_frequencies: np.ndarray,
        frame_confidences: np.ndarray,
    ) -> np.ndarray:
        """
        調和音成分を検出し、ピッチ推定の信頼度を向上させます。

        Parameters
        ----------
        tfr : np.ndarray
            時間-周波数表現
        freqs : np.ndarray
            周波数軸
        frame_frequencies : np.ndarray
            フレームごとのピッチ
        frame_confidences : np.ndarray
            フレームごとの信頼度

        Returns
        -------
        np.ndarray
            調和性を考慮した修正済み信頼度
        """
        n_freqs, n_frames = tfr.shape
        harmonic_confidences = np.copy(frame_confidences)

        for t in range(n_frames):
            if frame_frequencies[t] > 0:
                f0 = frame_frequencies[t]

                # 調和音プロファイルを計算
                harmonic_profile = self.compute_harmonic_profile(tfr, freqs, f0, t)

                # 調和性スコアを計算（基本音を除く倍音の強度合計）
                harmonic_score = np.sum(harmonic_profile[2:])

                # 調和音の存在に基づいて信頼度を調整
                # 強い調和構造があるほど信頼度を向上
                harmonic_weight = 1.0 / (1.0 + np.exp(-(harmonic_score - 1.0) * 2.0))

                # 調和性を考慮して信頼度を修正（最大50%の向上）
                harmonic_confidences[t] = frame_confidences[t] * (
                    1.0 + 0.5 * harmonic_weight
                )

                # 亜調波チェック（基本周波数の1/2, 1/3など）
                sub_harmonic_detected = False
                for div in [2, 3, 4]:
                    sub_f = f0 / div
                    if sub_f >= freqs[0]:
                        sub_idx = np.argmin(np.abs(freqs - sub_f))
                        sub_energy = tfr[sub_idx, t]
                        f0_idx = np.argmin(np.abs(freqs - f0))
                        f0_energy = tfr[f0_idx, t]
                        # 亜調波のエネルギーが基本音より著しく大きい場合、信頼度を下げる
                        if sub_energy > f0_energy * 1.5:
                            harmonic_confidences[t] *= 0.7
                            sub_harmonic_detected = True
                            break

        return harmonic_confidences

    def debug_detection(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        検出プロセスの中間結果を返します。デバッグ・評価用です。

        Parameters
        ----------
        audio_data : np.ndarray
            音声データ
        sr : int
            サンプリングレート

        Returns
        -------
        Dict[str, Any]
            中間結果を含む詳細な検出結果
        """
        # 通常の検出プロセスを実行
        result = self.detect(audio_data, sr)

        try:
            # 追加の中間結果を計算
            tfr, freqs, times = self.compute_adaptive_tfr(audio_data, sr)
            consistency = self.compute_structural_consistency(tfr)
            background = self.compute_dynamic_background(tfr)
            synergy = self.compute_synergy_index(tfr, background, consistency)

            # 調和性分析（オプション）
            if "frame_frequencies" in result and "frame_confidences" in result:
                harmonic_confidences = self.detect_harmonics(
                    tfr, freqs, result["frame_frequencies"], result["frame_confidences"]
                )

                # デバッグ情報を結果に追加
                result.update(
                    {
                        "tfr": tfr,
                        "freqs": freqs,
                        "times": times,
                        "consistency": consistency,
                        "background": background,
                        "synergy": synergy,
                        "harmonic_confidences": harmonic_confidences,
                    }
                )

        except Exception as e:
            self.logger.error(f"デバッグ情報の生成中にエラーが発生しました: {str(e)}")

        return result

    def detect_notes(
        self,
        frame_frequencies: np.ndarray,
        frame_confidences: np.ndarray,
        times: np.ndarray,
        sr: int = 44100,
    ) -> List[Tuple[float, float, float, float]]:
        """
        フレームピッチから音符を検出します。

        Parameters
        ----------
        frame_frequencies : np.ndarray
            フレームごとのピッチ
        frame_confidences : np.ndarray
            フレームごとの信頼度
        times : np.ndarray
            時間軸
        sr : int
            サンプリングレート

        Returns
        -------
        List[Tuple[float, float, float, float]]
            音符リスト (開始時間, 終了時間, ピッチ, 信頼度)
        """
        # 閾値を下げて感度を高める
        min_note_duration = 0.05  # 50ミリ秒以上の音符を検出
        min_confidence = max(
            0.01, np.mean(frame_confidences[frame_confidences > 0]) * 0.5
        )

        # 有声フレームの検出
        voiced_frames = frame_frequencies > 0

        # 連続した有声フレームをグループ化して音符を検出
        notes = []
        in_note = False
        note_start_idx = 0
        note_start_time = 0

        self.logger.info(
            f"フレーム数: {len(frame_frequencies)}, 有声フレーム数: {np.sum(voiced_frames)}"
        )

        for i in range(len(frame_frequencies)):
            # 音符の開始
            if voiced_frames[i] and not in_note:
                in_note = True
                note_start_idx = i
                note_start_time = times[i]

            # 音符の終了または音符の途中でピッチが大幅に変化
            elif in_note and (
                not voiced_frames[i]
                or (
                    i > 0
                    and voiced_frames[i - 1]
                    and abs(frame_frequencies[i] / frame_frequencies[i - 1] - 1.0) > 0.2
                )
            ):
                note_end_time = times[i - 1]
                note_duration = note_end_time - note_start_time

                # 短すぎる音符は除外
                if note_duration >= min_note_duration:
                    # ノートのフレーム範囲
                    note_frames = frame_frequencies[note_start_idx:i]
                    note_confs = frame_confidences[note_start_idx:i]

                    # フレームの中央値をノートピッチとして使用
                    note_pitch = np.median(note_frames)
                    note_conf = np.mean(note_confs)

                    # 最小信頼度を満たす場合のみ追加
                    if note_conf >= min_confidence:
                        notes.append(
                            (note_start_time, note_end_time, note_pitch, note_conf)
                        )

                in_note = False

        # 最後の音符の処理
        if in_note:
            note_end_time = times[-1]
            note_duration = note_end_time - note_start_time

            if note_duration >= min_note_duration:
                note_frames = frame_frequencies[note_start_idx:]
                note_confs = frame_confidences[note_start_idx:]

                note_pitch = np.median(note_frames)
                note_conf = np.mean(note_confs)

                if note_conf >= min_confidence:
                    notes.append(
                        (note_start_time, note_end_time, note_pitch, note_conf)
                    )

        self.logger.info(f"検出された音符数: {len(notes)}")
        return notes

    def detect_note_boundaries(
        self,
        pitch_trajectories: np.ndarray,
        pitch_confidences: np.ndarray,
        times: np.ndarray,
        tfr: np.ndarray,
        synergy: np.ndarray,
    ) -> List[Tuple]:
        """
        複合信号解析に基づく音符境界検出

        Parameters
        ----------
        pitch_trajectories : np.ndarray
            フレームごとのピッチ軌跡
        pitch_confidences : np.ndarray
            フレームごとの信頼度
        times : np.ndarray
            時間軸
        tfr : np.ndarray
            時間-周波数表現
        synergy : np.ndarray
            相乗効果指標

        Returns
        -------
        List[Tuple]
            音符のリスト（開始時間、終了時間、ピッチ、信頼度）
        """
        notes = []
        n_frames = len(times)
        n_freqs = tfr.shape[0]

        if n_frames < 2:
            return notes

        # ピッチ状態変化の検出
        pitch_state = pitch_trajectories > 0

        # 時間-周波数表現からの特徴量

        # 1. 瞬時エネルギー変化（全帯域）
        energy_profile = np.sum(np.abs(tfr), axis=0)
        energy_derivative = np.zeros_like(energy_profile)
        energy_derivative[1:] = energy_profile[1:] - energy_profile[:-1]

        # ノイズフロア推定
        noise_floor = np.percentile(energy_profile, 15)
        energy_threshold = noise_floor * 2.5

        # 2. 周波数帯域別エネルギー変化（低域/中域/高域）
        freq_bands = [
            (0, int(n_freqs * 0.2)),  # 低域 (0-20%)
            (int(n_freqs * 0.2), int(n_freqs * 0.6)),  # 中域 (20-60%)
            (int(n_freqs * 0.6), n_freqs),  # 高域 (60-100%)
        ]

        band_energy_profiles = []
        band_derivatives = []

        for start_idx, end_idx in freq_bands:
            band_energy = np.sum(np.abs(tfr[start_idx:end_idx, :]), axis=0)
            band_derivative = np.zeros_like(band_energy)
            band_derivative[1:] = band_energy[1:] - band_energy[:-1]

            band_energy_profiles.append(band_energy)
            band_derivatives.append(band_derivative)

        # 3. スペクトル新規性（spectral novelty）
        novelty_function = np.zeros(n_frames)
        for t in range(1, n_frames):
            # 現在フレームと前フレームの相関が低いほど新規性が高い
            if t > 0:
                correlation = np.corrcoef(np.abs(tfr[:, t]), np.abs(tfr[:, t - 1]))[
                    0, 1
                ]
                novelty_function[t] = 1.0 - max(0, correlation)

        # 4. 相乗効果指標の導関数
        synergy_derivative = np.zeros_like(synergy)
        synergy_derivative[:, 1:] = synergy[:, 1:] - synergy[:, :-1]

        # 複合オンセット関数の構築
        onset_function = np.zeros(n_frames)

        # 全帯域のエネルギー変化（正の変化のみ）
        normalized_energy = energy_profile / (np.max(energy_profile) + 1e-8)
        energy_weight = np.minimum(
            1.0, normalized_energy * 3
        )  # エネルギーが高いほど重み大
        onset_function += np.maximum(0, energy_derivative) * energy_weight

        # 周波数帯域別の変化（各帯域の重要度を調整）
        band_weights = [0.5, 1.0, 0.7]  # 低域/中域/高域の重み
        for i, (derivative, weight) in enumerate(zip(band_derivatives, band_weights)):
            onset_function += np.maximum(0, derivative) * weight * energy_weight

        # スペクトル新規性の寄与
        onset_function += novelty_function * 0.5

        # 相乗効果変化の寄与
        for i in range(n_freqs):
            onset_function += np.maximum(0, synergy_derivative[i, :]) * 0.3

        # ピッチ信頼度の急激な上昇
        confidence_derivative = np.zeros_like(pitch_confidences)
        confidence_derivative[1:] = pitch_confidences[1:] - pitch_confidences[:-1]
        onset_function += np.maximum(0, confidence_derivative) * 0.5

        # オンセット関数の正規化
        if np.max(onset_function) > 0:
            onset_function /= np.max(onset_function)

        # オフセット関数の構築（オンセットと同様の手法だが符号を反転）
        offset_function = np.zeros(n_frames)

        # エネルギー減少
        offset_function += np.maximum(0, -energy_derivative) * energy_weight

        # 帯域別エネルギー減少
        for i, (derivative, weight) in enumerate(zip(band_derivatives, band_weights)):
            offset_function += np.maximum(0, -derivative) * weight * energy_weight

        # 相乗効果の減少
        for i in range(n_freqs):
            offset_function += np.maximum(0, -synergy_derivative[i, :]) * 0.3

        # ピッチ信頼度の低下
        offset_function += np.maximum(0, -confidence_derivative) * 0.5

        # オフセット関数の正規化
        if np.max(offset_function) > 0:
            offset_function /= np.max(offset_function)

        # 適応的閾値の計算
        onset_values = onset_function[onset_function > 0.05]
        if len(onset_values) > 0:
            onset_threshold = (
                0.15 * np.mean(onset_values) + 0.1 * np.median(onset_values) + 0.05
            )
        else:
            onset_threshold = 0.15

        offset_values = offset_function[offset_function > 0.05]
        if len(offset_values) > 0:
            offset_threshold = (
                0.15 * np.mean(offset_values) + 0.1 * np.median(offset_values) + 0.03
            )
        else:
            offset_threshold = 0.1

        # 時間間隔の計算
        time_interval = times[1] - times[0] if len(times) > 1 else 0.01

        # ピーク検出窓サイズの設定（約30msのウィンドウ）
        peak_window = max(2, int(0.03 / time_interval))

        # オンセットピーク検出（適応的閾値と最小セパレーション）
        onset_peaks = []
        for t in range(peak_window, n_frames - peak_window):
            # エネルギーが最小閾値を超え、ピークが十分に顕著であること
            if (
                energy_profile[t] > energy_threshold
                and onset_function[t] > onset_threshold
            ):
                # 局所的最大値の確認
                if onset_function[t] == np.max(
                    onset_function[t - peak_window : t + peak_window + 1]
                ):
                    # 前後のピークとの最小間隔を確保
                    if not onset_peaks or t - onset_peaks[-1] >= peak_window:
                        onset_peaks.append(t)

        # オフセットピーク検出
        offset_peaks = []
        for t in range(peak_window, n_frames - peak_window):
            if offset_function[t] > offset_threshold:
                # 局所的最大値の確認
                if offset_function[t] == np.max(
                    offset_function[t - peak_window : t + peak_window + 1]
                ):
                    # 前後のピークとの最小間隔を確保
                    if not offset_peaks or t - offset_peaks[-1] >= peak_window:
                        offset_peaks.append(t)

        # ピッチ軌跡に基づく音符検出
        note_starts = []
        note_ends = []

        # 1. ピッチ状態の変化点を追加
        for t in range(1, n_frames):
            if pitch_state[t] and not pitch_state[t - 1]:  # 無声→有声の遷移
                note_starts.append(t)
            elif not pitch_state[t] and pitch_state[t - 1]:  # 有声→無声の遷移
                note_ends.append(t)

        # 2. オンセットピークを追加（既存の開始点と重複しないようにする）
        for peak in onset_peaks:
            # 既存の開始点から十分離れているか確認
            if all(abs(peak - start) > peak_window for start in note_starts):
                note_starts.append(peak)

        # 3. オフセットピークを追加（既存の終了点と重複しないようにする）
        for peak in offset_peaks:
            # 既存の終了点から十分離れているか確認
            if all(abs(peak - end) > peak_window for end in note_ends):
                note_ends.append(peak)

        # オンセット・オフセットのペアを形成して音符を作成
        note_starts.sort()
        note_ends.sort()

        # 各オンセットに対して次のオフセットを見つける
        for start_idx in note_starts:
            # このオンセット後の最初のオフセットを探す
            valid_offsets = [end_idx for end_idx in note_ends if end_idx > start_idx]

            if valid_offsets:
                end_idx = min(valid_offsets)  # 最も近いオフセット

                # 音符の最小長さを確認（30ms以上）
                min_note_length = max(3, int(0.03 / time_interval))

                if end_idx - start_idx >= min_note_length:
                    # この区間のピッチを決定
                    segment_pitches = pitch_trajectories[start_idx:end_idx]
                    segment_confidences = pitch_confidences[start_idx:end_idx]

                    # 有効なピッチ値のみを考慮
                    valid_indices = segment_pitches > 0
                    if np.any(valid_indices):
                        # 信頼度で重み付けした平均ピッチ
                        sum_confidences = np.sum(segment_confidences[valid_indices])
                        if sum_confidences > 0:
                            weighted_pitch = np.sum(
                                segment_pitches[valid_indices]
                                * segment_confidences[valid_indices]
                            )
                            weighted_pitch /= sum_confidences
                        else:
                            # 信頼度の合計がゼロの場合、単純平均を使用
                            weighted_pitch = np.mean(segment_pitches[valid_indices])

                        # 平均信頼度
                        mean_confidence = np.mean(segment_confidences[valid_indices])

                        # 音符の追加
                        note_start_time = times[start_idx]
                        note_end_time = times[end_idx]

                        notes.append(
                            (
                                note_start_time,
                                note_end_time,
                                weighted_pitch,
                                mean_confidence,
                            )
                        )

        return notes

    def detect(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        音声データからノートを検出し、検出結果を返します。

        Parameters
        ----------
        audio_data : np.ndarray
            音声データ
        sr : int
            サンプリングレート

        Returns
        -------
        Dict[str, Any]
            検出結果の辞書
        """
        # 処理時間の計測開始
        start_time = time.time()

        # サンプリングレートを保存（後続の処理で使用）
        self.sr = sr

        try:
            # ステレオをモノラルに変換
            if audio_data.ndim > 1 and audio_data.shape[0] > 1:
                audio_data = np.mean(audio_data, axis=0)
            elif audio_data.ndim > 1:
                audio_data = audio_data[0]

            # 1. 時間-周波数表現の計算
            tfr, freqs, times = self.compute_adaptive_tfr(audio_data, sr)

            # 2. 動的背景ノイズモデルの計算
            background = self.compute_dynamic_background(tfr)

            # 3. 構造的一貫性の計算（改善版）
            consistency = self.compute_enhanced_consistency(tfr, freqs, times)

            # 4. 相乗効果指標の計算（改善版）
            synergy = self.compute_synergy_index_advanced(
                tfr, background, consistency, freqs, times
            )

            # 5. ピッチ軌跡抽出（改善版）
            pitch_trajectories, pitch_confidences = self.extract_pitch_trajectories(
                synergy, freqs, times, tfr
            )

            # 6. 音符境界検出（改善版）
            notes = self.detect_note_boundaries(
                pitch_trajectories, pitch_confidences, times, tfr, synergy
            )

            # 結果を辞書形式にまとめる
            if notes and len(notes) > 0:
                intervals = np.array([[note[0], note[1]] for note in notes])
                note_pitches = np.array([note[2] for note in notes])
            else:
                # 音符が検出されなかった場合は空の配列を返す
                intervals = np.array([]).reshape(0, 2)
                note_pitches = np.array([])

            result = {
                "intervals": intervals,
                "note_pitches": note_pitches,
                "frame_times": np.array(times),
                "frame_frequencies": pitch_trajectories,  # 改良されたピッチ軌跡を使用
                "detector_name": self.__class__.__name__,
                "detection_time": time.time() - start_time,
                # 追加の結果（必須ではない）
                "frame_confidences": pitch_confidences,
                "synergy": synergy,
            }

            return result

        except Exception as e:
            # エラー処理：エラーをログに記録し、空の結果を返す
            self.logger.error(f"検出中にエラーが発生しました: {str(e)}")
            traceback.print_exc()

            # 処理時間の計測終了
            detection_time = time.time() - start_time

            # エラー時は空の結果を返す
            result = {
                "intervals": np.array([]).reshape(0, 2),
                "note_pitches": np.array([]),
                "frame_times": np.array([]),
                "frame_frequencies": np.array([]),
                "detector_name": self.__class__.__name__,
                "detection_time": detection_time,
            }

            return result

    def detect_pitch_enhanced(
        self, synergy: np.ndarray, freqs: np.ndarray, times: np.ndarray, tfr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        調和構造を考慮した拡張ピッチ検出アルゴリズム

        Parameters
        ----------
        synergy : np.ndarray
            相乗効果指標
        freqs : np.ndarray
            周波数軸
        times : np.ndarray
            時間軸
        tfr : np.ndarray
            時間-周波数表現

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (frame_frequencies, frame_confidences)
        """
        n_freqs, n_times = synergy.shape
        frame_frequencies = np.zeros(n_times)
        frame_confidences = np.zeros(n_times)

        # 閾値を下げて感度を高める
        synergy_threshold = max(0.01, np.mean(synergy) + 0.5 * np.std(synergy))
        self.logger.info(f"適応的相乗効果閾値: {synergy_threshold:.4f}")

        # 1. 初期ピッチ候補の検出
        for t in range(n_times):
            # ピークの検出（局所的な極大値）
            peaks = []
            for i in range(1, n_freqs - 1):
                if (
                    synergy[i, t] > synergy_threshold
                    and synergy[i, t] > synergy[i - 1, t]
                    and synergy[i, t] > synergy[i + 1, t]
                ):
                    peaks.append((i, synergy[i, t]))

            # ピークがない場合
            if not peaks:
                continue

            # ピークを信頼度でソート
            peaks.sort(key=lambda x: x[1], reverse=True)

            # 最も信頼度の高いピーク
            best_peak_idx, best_peak_val = peaks[0]
            candidate_freq = freqs[best_peak_idx]

            # 2. 調和音構造の検証
            # 最大ピークが高調波である可能性をチェック
            for div in [2, 3, 4, 5]:  # 基本音の倍音をチェック
                potential_f0 = candidate_freq / div
                if potential_f0 >= freqs[0]:
                    f0_idx = np.argmin(np.abs(freqs - potential_f0))

                    # 潜在的な基本音のエネルギーを分析
                    if (
                        synergy[f0_idx, t] > synergy_threshold * 0.5
                    ):  # 閾値を下げて感度を高める
                        # 倍音構造を評価
                        harmonic_profile = self.compute_harmonic_profile(
                            tfr, freqs, potential_f0, t
                        )
                        harmonic_score = np.sum(harmonic_profile[2:])

                        # 強い調和構造があれば、基本音に調整
                        if harmonic_score > 1.0:  # 閾値を下げて感度を高める
                            best_peak_idx = f0_idx
                            best_peak_val = synergy[f0_idx, t] * (
                                1 + 0.3 * harmonic_score
                            )
                            candidate_freq = potential_f0
                            break

            # 3. 最終的なピッチと信頼度を設定
            frame_frequencies[t] = candidate_freq
            frame_confidences[t] = best_peak_val

        # 4. ピッチ軌跡の補正
        self._correct_pitch_trajectory_enhanced(frame_frequencies, frame_confidences)

        return frame_frequencies, frame_confidences

    def _correct_pitch_trajectory_enhanced(
        self,
        frame_frequencies: np.ndarray,
        frame_confidences: np.ndarray,
        max_jump_ratio: float = 0.1,
        min_segment_length: int = 3,
    ) -> None:
        """
        ピッチ軌跡を強化アルゴリズムで補正します。

        Parameters
        ----------
        frame_frequencies : np.ndarray
            フレームごとのピッチ
        frame_confidences : np.ndarray
            フレームごとの信頼度
        max_jump_ratio : float
            許容される最大の周波数ジャンプ比率
        min_segment_length : int
            最小のセグメント長
        """
        n_frames = len(frame_frequencies)

        # 1. 短すぎる有声セグメントの除去
        i = 0
        while i < n_frames:
            if frame_frequencies[i] > 0:
                segment_start = i
                while i < n_frames and frame_frequencies[i] > 0:
                    i += 1
                segment_end = i
                segment_length = segment_end - segment_start

                if segment_length < min_segment_length:
                    frame_frequencies[segment_start:segment_end] = 0
                    frame_confidences[segment_start:segment_end] = 0
            else:
                i += 1

        # 2. メディアンフィルタによる外れ値の除去
        for i in range(n_frames):
            if frame_frequencies[i] > 0:
                # 現在のピッチを中心とした周辺フレームを収集
                window_size = 5
                half_window = window_size // 2
                start_idx = max(0, i - half_window)
                end_idx = min(n_frames, i + half_window + 1)

                window_freqs = [
                    frame_frequencies[j]
                    for j in range(start_idx, end_idx)
                    if frame_frequencies[j] > 0
                ]

                # 有声フレームが十分ある場合にのみメディアンを適用
                if len(window_freqs) >= 3:
                    median_freq = np.median(window_freqs)
                    # 現在のピッチが中央値から大幅に外れている場合
                    if abs(frame_frequencies[i] / median_freq - 1.0) > max_jump_ratio:
                        # 信頼度が低ければメディアン値に置き換え
                        if frame_confidences[i] < np.mean(
                            [
                                frame_confidences[j]
                                for j in range(start_idx, end_idx)
                                if frame_frequencies[j] > 0
                            ]
                        ):
                            frame_frequencies[i] = median_freq

        # 3. オクターブ誤りの補正
        for i in range(1, n_frames):
            if frame_frequencies[i] > 0 and frame_frequencies[i - 1] > 0:
                freq_ratio = frame_frequencies[i] / frame_frequencies[i - 1]

                # オクターブジャンプの可能性
                if 1.8 < freq_ratio < 2.2:  # 約2倍（オクターブ上）
                    # 周辺を見て一貫性を判断
                    if i < n_frames - 1 and frame_frequencies[i + 1] > 0:
                        next_ratio = frame_frequencies[i + 1] / frame_frequencies[i]
                        # 次のフレームも高いままなら維持、そうでなければ補正
                        if next_ratio < 0.7 or next_ratio > 1.4:
                            frame_frequencies[i] = frame_frequencies[i - 1]
                elif 0.45 < freq_ratio < 0.55:  # 約1/2（オクターブ下）
                    # 周辺を見て一貫性を判断
                    if i < n_frames - 1 and frame_frequencies[i + 1] > 0:
                        next_ratio = frame_frequencies[i + 1] / frame_frequencies[i]
                        # 次のフレームも低いままなら維持、そうでなければ補正
                        if next_ratio < 0.7 or next_ratio > 1.4:
                            frame_frequencies[i] = frame_frequencies[i - 1]

    def extract_pitch_trajectories(
        self, synergy: np.ndarray, freqs: np.ndarray, times: np.ndarray, tfr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        複合情報源を活用した高精度ピッチ軌跡抽出

        Parameters
        ----------
        synergy : np.ndarray
            相乗効果指標
        freqs : np.ndarray
            周波数軸
        times : np.ndarray
            時間軸
        tfr : np.ndarray
            時間-周波数表現

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ピッチ軌跡と信頼度
        """
        n_frames = len(times)
        n_freqs = len(freqs)

        # 時間-周波数表現からの追加特徴量抽出
        # 1. スペクトル流量（spectral flux）- 急激な変化を検出
        spectral_flux = np.zeros(n_frames)
        for t in range(1, n_frames):
            spectral_flux[t] = np.sum(
                np.maximum(0, np.abs(tfr[:, t]) - np.abs(tfr[:, t - 1]))
            )

        # 2. スペクトル重心（spectral centroid）- 周波数の分布重心
        spectral_centroid = np.zeros(n_frames)
        for t in range(n_frames):
            if np.sum(np.abs(tfr[:, t])) > 0:
                spectral_centroid[t] = np.sum(freqs * np.abs(tfr[:, t])) / np.sum(
                    np.abs(tfr[:, t])
                )

        # 3. スペクトル平坦度（spectral flatness）- ノイズ性vs調和性
        spectral_flatness = np.zeros(n_frames)
        for t in range(n_frames):
            magnitude = np.abs(tfr[:, t])
            if np.sum(magnitude) > 0:
                geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
                arithmetic_mean = np.mean(magnitude)
                if arithmetic_mean > 0:
                    spectral_flatness[t] = geometric_mean / arithmetic_mean

        # ピッチ確率分布の初期化
        pitch_probabilities = np.zeros((n_freqs, n_frames))

        # 各フレームでの処理
        for t in range(n_frames):
            # 低エネルギーフレームはスキップ
            frame_energy = np.sum(np.abs(tfr[:, t]))
            if frame_energy < 1e-5:
                continue

            # 相乗効果指標からのピーク検出
            peaks = []
            for i in range(1, n_freqs - 1):
                if (
                    synergy[i, t] > synergy[i - 1, t]
                    and synergy[i, t] > synergy[i + 1, t]
                    and synergy[i, t] > 0.2 * np.max(synergy[:, t])
                ):

                    # ピーク位置の精密化（放物線補間）
                    y0, y1, y2 = synergy[i - 1, t], synergy[i, t], synergy[i + 1, t]
                    peak_pos = i + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-10)
                    peak_pos = np.clip(peak_pos, i - 0.5, i + 0.5)

                    # 周波数値に変換
                    freq_pos = freqs[int(np.floor(peak_pos))] + (
                        peak_pos - np.floor(peak_pos)
                    ) * (
                        freqs[min(n_freqs - 1, int(np.floor(peak_pos)) + 1)]
                        - freqs[int(np.floor(peak_pos))]
                    )

                    # ハーモニック関係を考慮したF0候補
                    # 観測された周波数がF0の場合と、倍音である場合の両方を考慮
                    f0_candidates = []

                    # 観測周波数がF0の場合
                    f0_candidates.append(freq_pos)

                    # 観測周波数が第2〜5倍音の場合
                    for h in range(2, 6):
                        f0_subharmonic = freq_pos / h
                        if f0_subharmonic >= freqs[0]:
                            f0_candidates.append(f0_subharmonic)

                    # 候補をnumpy配列に変換
                    f0_candidates = np.array(f0_candidates)

                    # TFRを利用したハーモニック構造分析
                    harmonic_scores = self.analyze_harmonic_structure(
                        tfr, freqs, f0_candidates, t
                    )

                    # 最良の候補を選択
                    best_idx = np.argmax(harmonic_scores)
                    best_f0 = f0_candidates[best_idx]
                    best_score = harmonic_scores[best_idx]

                    # スペクトル平坦度による調整（調和性の高い音ほど信頼度を上げる）
                    flatness_factor = 1.0 - min(1.0, spectral_flatness[t] * 5)

                    # ピークの信頼度スコア計算
                    confidence = synergy[i, t] * best_score * flatness_factor

                    # スコアが閾値以上の場合のみ追加
                    if confidence > 0.1:
                        peaks.append((best_f0, confidence))

            # ピークがない場合は次のフレームへ
            if not peaks:
                continue

            # 複数のピークがある場合はスコアの高い順にソート
            peaks.sort(key=lambda x: x[1], reverse=True)

            # 最も信頼度の高いピークを選択
            best_pitch, best_confidence = peaks[0]

            # ピッチ確率分布に反映
            for freq, confidence in peaks:
                # 周波数ビンのインデックスを計算
                idx = np.argmin(np.abs(freqs - freq))

                # ガウス窓で確率を割り当て
                spread = max(1, int(n_freqs * 0.01))  # 周波数の1%程度の広がり
                for j in range(
                    max(0, idx - spread * 2), min(n_freqs, idx + spread * 2)
                ):
                    dist = abs(j - idx)
                    weight = np.exp(-0.5 * (dist / spread) ** 2)
                    pitch_probabilities[j, t] = max(
                        pitch_probabilities[j, t], confidence * weight
                    )

        # 時間連続性を考慮した軌跡抽出（動的計画法）
        pitch_trajectory = np.zeros(n_frames)
        pitch_confidence = np.zeros(n_frames)

        # 状態遷移コスト行列
        transition_costs = np.zeros((n_freqs, n_freqs))
        for i in range(n_freqs):
            for j in range(n_freqs):
                # 周波数比に基づくコスト（オクターブ関係は低コスト）
                f_i, f_j = freqs[i], freqs[j]
                if f_i > 0 and f_j > 0:
                    ratio = max(f_i / f_j, f_j / f_i)

                    # オクターブ関係（比が2に近い）の場合は低コスト
                    octave_factor = min(
                        abs(ratio - 2), abs(ratio - 1), abs(ratio - 0.5)
                    )
                    if octave_factor < 0.05:  # オクターブに非常に近い
                        transition_costs[i, j] = 0.5
                    else:
                        # それ以外は周波数変化に比例したコスト
                        transition_costs[i, j] = min(
                            5.0, abs(f_i - f_j) / min(f_i, f_j)
                        )
                else:
                    transition_costs[i, j] = 5.0  # デフォルトの高コスト

        # 累積コスト行列と経路記録
        cumulative_costs = np.zeros((n_freqs, n_frames)) + float("inf")
        backpointers = np.zeros((n_freqs, n_frames), dtype=int)

        # 初期化（最初のフレーム）
        cumulative_costs[:, 0] = -np.log(pitch_probabilities[:, 0] + 1e-10)

        # 動的計画法による最適経路探索
        for t in range(1, n_frames):
            for i in range(n_freqs):
                for j in range(n_freqs):
                    # 前のフレームからのコスト + 遷移コスト + 現在の確率コスト
                    cost = cumulative_costs[j, t - 1] + transition_costs[j, i]

                    if cost < cumulative_costs[i, t]:
                        cumulative_costs[i, t] = cost
                        backpointers[i, t] = j

                # 現在の確率に基づくコストを加算
                cumulative_costs[i, t] += -np.log(pitch_probabilities[i, t] + 1e-10)

        # バックトレースによる最適経路の抽出
        if n_frames > 0:
            # 最終フレームでの最小コスト状態を選択
            current_state = np.argmin(cumulative_costs[:, -1])
            pitch_trajectory[-1] = freqs[current_state]
            pitch_confidence[-1] = pitch_probabilities[current_state, -1]

            # 残りのフレームをバックトレース
            for t in range(n_frames - 2, -1, -1):
                current_state = backpointers[current_state, t + 1]
                pitch_trajectory[t] = freqs[current_state]
                pitch_confidence[t] = pitch_probabilities[current_state, t]

        # 後処理：信頼度閾値と最小セグメント長によるフィルタリング
        confidence_threshold = 0.15
        min_segment_length = 3  # フレーム数

        voiced = pitch_confidence > confidence_threshold

        # 短いセグメントを除去
        segment_start = None
        for t in range(n_frames + 1):  # 最後に強制終了用の+1
            if t < n_frames and voiced[t]:
                if segment_start is None:
                    segment_start = t
            elif segment_start is not None:  # セグメント終了
                segment_length = t - segment_start
                if segment_length < min_segment_length:
                    # 短いセグメントを無声化
                    pitch_trajectory[segment_start:t] = 0
                    pitch_confidence[segment_start:t] = 0
                segment_start = None if t >= n_frames or not voiced[t] else t

        return pitch_trajectory, pitch_confidence

    def _refine_pitch_trajectory(
        self, pitch_trajectories: np.ndarray, pitch_confidences: np.ndarray
    ) -> None:
        """
        ピッチ軌跡を洗練し、滑らかにします。

        Parameters
        ----------
        pitch_trajectories : np.ndarray
            ピッチ軌跡
        pitch_confidences : np.ndarray
            ピッチ信頼度
        """
        # メディアンフィルタと移動平均の統合による平滑化
        n_frames = len(pitch_trajectories)

        if n_frames < 5:
            return

        # メディアンフィルタによる外れ値除去
        for i in range(2, n_frames - 2):
            if pitch_trajectories[i] > 0:
                window = [
                    pitch_trajectories[j]
                    for j in range(i - 2, i + 3)
                    if pitch_trajectories[j] > 0
                ]
                if len(window) >= 3:
                    median = np.median(window)
                    # 現在のピッチが中央値から大きく外れている場合
                    if abs(pitch_trajectories[i] / median - 1.0) > 0.1:
                        pitch_trajectories[i] = median

        # ギャップの補間（短期間の無声部分）
        i = 0
        while i < n_frames - 2:
            if (
                pitch_trajectories[i] > 0
                and pitch_trajectories[i + 1] == 0
                and pitch_trajectories[i + 2] > 0
            ):
                # 1フレームのギャップ
                freq_ratio = pitch_trajectories[i + 2] / pitch_trajectories[i]
                # 近い周波数なら内挿
                if 0.8 < freq_ratio < 1.25:
                    pitch_trajectories[i + 1] = np.sqrt(
                        pitch_trajectories[i] * pitch_trajectories[i + 2]
                    )
                    pitch_confidences[i + 1] = (
                        min(pitch_confidences[i], pitch_confidences[i + 2]) * 0.8
                    )
            i += 1

    def compute_enhanced_consistency(
        self, tfr: np.ndarray, freqs: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """
        方向性と周波数の連続性を考慮した拡張された構造的一貫性係数を計算します。
        最適化バージョン。

        Parameters
        ----------
        tfr : np.ndarray
            時間-周波数表現
        freqs : np.ndarray
            周波数軸
        times : np.ndarray
            時間軸

        Returns
        -------
        np.ndarray
            拡張された構造的一貫性係数
        """
        n_freqs, n_times = tfr.shape
        consistency = np.zeros((n_freqs, n_times))

        # 計算を最適化するためのサブサンプリング
        # 全てのピクセルではなく、間引いたピクセルで計算
        freq_stride = max(1, n_freqs // 50)  # 周波数方向のサブサンプリング
        time_stride = max(1, n_times // 50)  # 時間方向のサブサンプリング

        # スケールパラメータ（少なくする）
        time_scales = [1, 2]  # マルチスケール分析のための時間スケール
        freq_scales = [1, 2]  # マルチスケール分析のための周波数スケール

        # 水平方向と垂直方向の差分を一度に計算（高速化）
        h_diff = np.abs(np.diff(tfr, axis=1))
        v_diff = np.abs(np.diff(tfr, axis=0))

        # 平均変動計算の高速化
        h_variation_map = np.zeros((n_freqs, n_times))
        h_variation_map[:, :-1] = h_diff
        h_variation_map[:, -1] = h_variation_map[:, -2]  # 最後の列を補完

        v_variation_map = np.zeros((n_freqs, n_times))
        v_variation_map[:-1, :] = v_diff
        v_variation_map[-1, :] = v_variation_map[-2, :]  # 最後の行を補完

        # マルチスケール分析の並列化
        for t_scale in time_scales:
            for f_scale in freq_scales:
                # スケールに応じた重み
                scale_weight = 1.0 / (t_scale * f_scale)

                # 効率的なウィンドウ計算のための畳み込みカーネル
                # 異なるスケールごとに局所的な変動を効率的に計算
                h_kernel = np.ones((2 * f_scale + 1, 2 * t_scale + 1)) / (
                    (2 * f_scale + 1) * (2 * t_scale + 1)
                )
                v_kernel = h_kernel

                # 畳み込みの代わりに、より効率的なスライディングウィンドウの平均計算
                for i in range(0, n_freqs, freq_stride):
                    f_min = max(0, i - f_scale)
                    f_max = min(n_freqs, i + f_scale + 1)

                    for j in range(0, n_times, time_stride):
                        t_min = max(0, j - t_scale)
                        t_max = min(n_times, j + t_scale + 1)

                        if t_max <= t_min or f_max <= f_min:
                            continue

                        # 局所領域の平均変動
                        local_h_var = np.mean(h_variation_map[f_min:f_max, t_min:t_max])
                        local_v_var = np.mean(v_variation_map[f_min:f_max, t_min:t_max])

                        # 周波数連続性の評価（簡略化）
                        freq_continuity = 0.0
                        if tfr[i, j] > 0:
                            local_window = tfr[f_min:f_max, t_min:t_max]
                            local_window = local_window[local_window > 0]
                            if len(local_window) > 0:
                                ratios = np.abs(local_window / tfr[i, j] - 1.0)
                                freq_continuity = 1.0 / (1.0 + np.mean(ratios))

                        # 一貫性スコア計算
                        variation_score = 1.0 / (
                            1.0 + 0.5 * local_h_var + 0.3 * local_v_var
                        )
                        consistency_score = (
                            variation_score
                            * (0.7 + 0.3 * freq_continuity)
                            * scale_weight
                        )

                        # サブサンプリングしたポイントの結果を近傍に拡張
                        for fi in range(i, min(i + freq_stride, n_freqs)):
                            for tj in range(j, min(j + time_stride, n_times)):
                                consistency[fi, tj] += consistency_score

        # 正規化（スケール数で割る）
        consistency /= len(time_scales) * len(freq_scales)

        # 閾値を適用して強調
        consistency = np.power(consistency, 1.5)  # 高い一貫性値を強調

        return consistency
