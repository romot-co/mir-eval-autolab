"""
Pinna検出器モジュール

この検出器は、ヒト内耳（蝸牛）を模したフィルタバンクと神経スパイク（インパルス）に着想を得た
アルゴリズムを実装しています。オンセット・オフセット・ピッチ・音量を同時に捉える生理学的モデルです。
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import scipy
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from src.detectors.base_detector import BaseDetector


class PinnaDetector(BaseDetector):
    """
    Pinna (ピンナ) 検出器

    生理学的・聴覚的な視点を取り入れ、ヒト内耳（蝸牛）を模したフィルタバンクと
    神経スパイク（インパルス）に着想を得たアルゴリズムを実装しています。
    スパイク発火率の変化からオンセット・オフセットを検出し、
    インタースパイク間隔 (ISI) からピッチを推定します。

    Parameters
    ----------
    num_bands : int
        フィルタバンクのバンド数
    min_freq : float
        最低周波数 (Hz)
    max_freq : float
        最高周波数 (Hz)
    spike_threshold : float
        スパイク発火の閾値
    reset_constant : float
        蓄積値リセット後の初期値（0〜1）
    onset_threshold : float
        オンセット検出の閾値
    offset_threshold : float
        オフセット検出の閾値
    frame_length : int
        フレーム長
    hop_length : int
        ホップ長
    time_constant : float
        時定数（動的閾値の平滑化係数）
    pitch_min : float
        ピッチ検出の最低周波数 (Hz)
    pitch_max : float
        ピッチ検出の最高周波数 (Hz)
    """

    def __init__(self, **kwargs):
        """
        パラメータで初期化します
        """
        super().__init__(**kwargs)

        # デフォルトパラメータ
        self.num_bands = kwargs.get("num_bands", 32)  # フィルタバンク数
        self.min_freq = kwargs.get("min_freq", 50.0)  # 最低周波数 (Hz)
        self.max_freq = kwargs.get("max_freq", 8000.0)  # 最高周波数 (Hz)
        self.spike_threshold = kwargs.get("spike_threshold", 0.1)  # スパイク発火閾値
        self.reset_constant = kwargs.get("reset_constant", 0.1)  # リセット後の初期値
        self.onset_threshold = kwargs.get("onset_threshold", 0.5)  # オンセット閾値
        self.offset_threshold = kwargs.get("offset_threshold", 0.5)  # オフセット閾値
        self.frame_length = kwargs.get("frame_length", 1024)  # フレーム長
        self.hop_length = kwargs.get("hop_length", 256)  # ホップ長
        self.time_constant = kwargs.get("time_constant", 0.9)  # 時定数
        self.pitch_min = kwargs.get("pitch_min", 60.0)  # ピッチ検出最低周波数
        self.pitch_max = kwargs.get("pitch_max", 2000.0)  # ピッチ検出最高周波数
        self.window_size = kwargs.get("window_size", 0.05)  # 分析窓サイズ（秒）

        # 動的閾値のパラメータ
        self.alpha = kwargs.get("alpha", 0.9)  # 動的閾値の平滑化係数
        self.omega1 = kwargs.get("omega1", 1.0)  # 新規性重み
        self.omega2 = kwargs.get("omega2", 0.5)  # 音量変化重み
        self.omega3 = kwargs.get("omega3", 0.3)  # ピッチ安定度重み

        # スパイク発火モデルのパラメータ
        self.accumulation_rate = kwargs.get("accumulation_rate", 0.1)  # 蓄積率

    def detect(
        self,
        audio_file: Optional[str] = None,
        audio: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        音声ファイルまたは音声データからオンセット、オフセット、ピッチを検出します

        Parameters
        ----------
        audio_file : Optional[str], optional
            分析する音声ファイルのパス, by default None
        audio : Optional[np.ndarray], optional
            分析する音声データ, by default None
        sr : Optional[int], optional
            サンプリングレート（Hz）, by default None

        Returns
        -------
        Dict[str, Any]
            検出結果を含む辞書
        """
        start_time = time.time()

        # 音声データを読み込み
        if audio_file is not None:
            audio, sr = librosa.load(audio_file, sr=sr, mono=True)
        elif audio is None or sr is None:
            raise ValueError("audio_fileまたはaudio+srを指定してください")

        # モノラルに変換
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 音声データを正規化
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # オンセット・オフセット・ピッチを検出
        notes = self.detect_notes(audio, sr)

        onsets = notes["intervals"][:, 0]
        offsets = notes["intervals"][:, 1]
        intervals = notes["intervals"]
        pitches = notes["pitches"]

        # 結果を返す
        results = {
            "onsets": onsets,
            "offsets": offsets,
            "intervals": intervals,
            "pitches": pitches,
            "detector_time": time.time() - start_time,
            "name": self.__class__.__name__,
        }

        return results

    def _create_gammatone_filterbank(self, sr: int) -> List[scipy.signal.lti]:
        """
        Gammatoneフィルタバンクを作成します

        Parameters
        ----------
        sr : int
            サンプリングレート（Hz）

        Returns
        -------
        List[scipy.signal.lti]
            Gammatoneフィルタのリスト
        """
        # 対数スケールで周波数帯域を分割
        frequencies = np.logspace(
            np.log10(self.min_freq), np.log10(self.max_freq), self.num_bands
        )

        # フィルタのパラメータ
        order = 4  # ガンマトーンフィルタの次数
        width = 1.0  # バンド幅の係数

        filters = []
        for cf in frequencies:
            # ERB（等価長方形帯域幅）を計算
            erb = 24.7 * (4.37 * cf / 1000 + 1)

            # バンド幅
            b = width * 2 * np.pi * erb

            # ガンマトーンフィルタの近似（IIR）
            # ここでは簡易的な実装のためバターワースフィルタで代用
            nyq = sr / 2.0
            low = max(0.001, (cf - erb / 2) / nyq)  # 0より大きい値にする
            high = min(0.999, (cf + erb / 2) / nyq)  # 1より小さい値にする

            # low < high であることを確認
            if low >= high:
                # 周波数が近すぎる場合は調整
                center = (low + high) / 2
                low = max(0.001, center - 0.001)
                high = min(0.999, center + 0.001)

            b, a = signal.butter(order, [low, high], btype="band")
            filters.append(signal.lti(b, a))

        return filters

    def _apply_filterbank(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        フィルタバンクを適用します

        Parameters
        ----------
        audio : np.ndarray
            音声データ
        sr : int
            サンプリングレート（Hz）

        Returns
        -------
        List[np.ndarray]
            各バンドの出力
        """
        filters = self._create_gammatone_filterbank(sr)
        filtered_outputs = []

        for filt in filters:
            # フィルタを適用
            t, filtered, _ = signal.lsim(filt, audio, np.arange(len(audio)) / sr)
            filtered_outputs.append(filtered)

        return filtered_outputs

    def _half_wave_rectify_compress(
        self, signals: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        半波整流と信号圧縮を適用します

        Parameters
        ----------
        signals : List[np.ndarray]
            各バンドのフィルタ出力

        Returns
        -------
        List[np.ndarray]
            整流・圧縮された信号
        """
        rectified = []

        for signal in signals:
            # 半波整流（負の部分をカット）
            rectified_signal = np.maximum(0, signal)

            # ルート圧縮
            compressed_signal = np.sqrt(rectified_signal)

            rectified.append(compressed_signal)

        return rectified

    def _generate_spikes(self, signals: List[np.ndarray], sr: int) -> List[List[int]]:
        """
        信号からスパイク（発火時点）を生成します

        Parameters
        ----------
        signals : List[np.ndarray]
            整流・圧縮された信号
        sr : int
            サンプリングレート（Hz）

        Returns
        -------
        List[List[int]]
            各バンドのスパイク時点のリスト
        """
        spikes = []

        for signal in signals:
            band_spikes = []
            # 蓄積値の初期化
            accumulation = self.reset_constant

            for i, sample in enumerate(signal):
                # 信号値を蓄積
                accumulation += sample * self.accumulation_rate

                # 閾値を超えたらスパイク発火
                if accumulation > self.spike_threshold:
                    band_spikes.append(i)
                    # 蓄積値をリセット
                    accumulation = self.reset_constant

            spikes.append(band_spikes)

        return spikes

    def _calculate_spike_rates(
        self, spikes: List[List[int]], signal_length: int, sr: int
    ) -> np.ndarray:
        """
        スパイク発火率を計算します

        Parameters
        ----------
        spikes : List[List[int]]
            各バンドのスパイク時点のリスト
        signal_length : int
            信号の長さ（サンプル数）
        sr : int
            サンプリングレート（Hz）

        Returns
        -------
        np.ndarray
            スパイク発火率の時系列 [バンド数 x フレーム数]
        """
        # フレームの総数を計算
        n_frames = 1 + (signal_length - self.frame_length) // self.hop_length

        # スパイク発火率の初期化
        spike_rates = np.zeros((self.num_bands, n_frames))

        # 各フレームでのウィンドウの範囲（サンプル）
        window_samples = int(self.window_size * sr)

        for b, band_spikes in enumerate(spikes):
            for i in range(n_frames):
                # フレームの中心時点
                frame_center = i * self.hop_length + self.frame_length // 2

                # ウィンドウの範囲
                window_start = max(0, frame_center - window_samples // 2)
                window_end = min(signal_length, frame_center + window_samples // 2)

                # ウィンドウ内のスパイク数をカウント
                count = sum(
                    1 for spike in band_spikes if window_start <= spike < window_end
                )

                # 発火率に変換（単位時間あたりのスパイク数）
                spike_rates[b, i] = count / (self.window_size + 1e-8)

        return spike_rates

    def _calculate_onset_offset_functions(
        self, spike_rates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        スパイク発火率からオンセット関数とオフセット関数を計算します

        Parameters
        ----------
        spike_rates : np.ndarray
            スパイク発火率の時系列 [バンド数 x フレーム数]

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            オンセット関数とオフセット関数
        """
        n_frames = spike_rates.shape[1]

        # オンセット関数とオフセット関数を初期化
        onset_function = np.zeros(n_frames)
        offset_function = np.zeros(n_frames)

        # 前フレームのスパイク発火率
        prev_rates = np.zeros(spike_rates.shape)
        prev_rates[:, 1:] = spike_rates[:, :-1]

        # 発火率の増加分（正の部分のみ）- オンセット用
        positive_diff = np.maximum(0, spike_rates - prev_rates)

        # 発火率の減少分（正の部分のみ）- オフセット用
        negative_diff = np.maximum(0, prev_rates - spike_rates)

        # 全バンドの寄与を合計
        for i in range(n_frames):
            # オンセット関数 = 発火率の急増の二乗和
            onset_function[i] = np.sum(positive_diff[:, i] ** 2)

            # オフセット関数 = 発火率の急減の二乗和
            offset_function[i] = np.sum(negative_diff[:, i] ** 2)

        # 関数を平滑化
        onset_function = gaussian_filter1d(onset_function, sigma=1.0)
        offset_function = gaussian_filter1d(offset_function, sigma=1.0)

        return onset_function, offset_function

    def _dynamic_thresholding(
        self,
        onset_function: np.ndarray,
        offset_function: np.ndarray,
        energy_function: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        動的閾値処理を適用してオンセットとオフセットを検出します

        Parameters
        ----------
        onset_function : np.ndarray
            オンセット関数
        offset_function : np.ndarray
            オフセット関数
        energy_function : np.ndarray
            エネルギー関数（音量の代理）

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            検出されたオンセットとオフセットのインデックス
        """
        n_frames = len(onset_function)

        # 状態変数を初期化
        z = np.zeros(n_frames)

        # エネルギーの変化率を計算
        energy_diff = np.zeros(n_frames)
        energy_diff[1:] = energy_function[1:] - energy_function[:-1]

        # ピッチ安定度の代理として一定値を使用（実際はISIヒストグラムの尖り度などを使用）
        pitch_stability = np.ones(n_frames) * 0.5

        # 状態変数を更新
        for i in range(1, n_frames):
            # 状態更新式
            input_term = (
                self.omega1 * (onset_function[i] - offset_function[i])
                + self.omega2 * energy_diff[i]
                + self.omega3 * pitch_stability[i]
            )

            z[i] = (1 - self.alpha) * z[i - 1] + self.alpha * np.tanh(input_term)

        # 状態変数の時間微分
        z_diff = np.zeros(n_frames)
        z_diff[1:] = z[1:] - z[:-1]

        # 動的閾値の計算
        onset_threshold = np.zeros(n_frames)
        offset_threshold = np.zeros(n_frames)

        # 移動平均で平滑化した閾値
        window_size = 10  # 閾値計算の窓サイズ
        for i in range(window_size, n_frames):
            # オンセット閾値は過去の陽性値の平均x係数
            onset_threshold[i] = (
                self.onset_threshold
                * np.mean(z_diff[i - window_size : i][z_diff[i - window_size : i] > 0])
                if np.any(z_diff[i - window_size : i] > 0)
                else self.onset_threshold * 0.01
            )

            # オフセット閾値は過去の陰性値の平均x係数
            offset_threshold[i] = (
                self.offset_threshold
                * np.mean(
                    np.abs(z_diff[i - window_size : i][z_diff[i - window_size : i] < 0])
                )
                if np.any(z_diff[i - window_size : i] < 0)
                else self.offset_threshold * 0.01
            )

        # オンセットとオフセットの検出
        onset_peaks = []
        offset_peaks = []

        # ピーク検出
        for i in range(1, n_frames - 1):
            # オンセットピーク: z_diffが閾値を超え、前後のフレームより大きい
            if (
                z_diff[i] > onset_threshold[i]
                and z_diff[i] > z_diff[i - 1]
                and z_diff[i] >= z_diff[i + 1]
            ):
                onset_peaks.append(i)

            # オフセットピーク: z_diffが閾値より小さく、前後のフレームより小さい
            if (
                z_diff[i] < -offset_threshold[i]
                and z_diff[i] < z_diff[i - 1]
                and z_diff[i] <= z_diff[i + 1]
            ):
                offset_peaks.append(i)

        return np.array(onset_peaks), np.array(offset_peaks)

    def _analyze_interspike_intervals(
        self, spikes: List[List[int]], onsets: np.ndarray, offsets: np.ndarray, sr: int
    ) -> np.ndarray:
        """
        インタースパイク間隔（ISI）からピッチを推定します

        Parameters
        ----------
        spikes : List[List[int]]
            各バンドのスパイク時点のリスト
        onsets : np.ndarray
            オンセットのフレームインデックス
        offsets : np.ndarray
            オフセットのフレームインデックス
        sr : int
            サンプリングレート（Hz）

        Returns
        -------
        np.ndarray
            各音符のピッチ（Hz）
        """
        hop_samples = self.hop_length
        frame_samples = self.frame_length

        pitches = []

        for i in range(len(onsets)):
            onset_sample = onsets[i] * hop_samples

            # オフセットがない場合は次のオンセットまで、または最後まで
            if i < len(offsets) and offsets[i] > onsets[i]:
                offset_sample = offsets[i] * hop_samples
            else:
                if i + 1 < len(onsets):
                    offset_sample = onsets[i + 1] * hop_samples
                else:
                    offset_sample = None  # 最後の音符

            # 区間を少し短くして、安定した部分のみを分析
            if offset_sample is not None:
                duration = offset_sample - onset_sample
                analysis_start = onset_sample + min(
                    int(duration * 0.1), 1000
                )  # 10%後から
                analysis_end = offset_sample - min(int(duration * 0.1), 1000)  # 90%まで

                if analysis_end <= analysis_start:
                    analysis_start = onset_sample
                    analysis_end = offset_sample
            else:
                analysis_start = onset_sample
                analysis_end = None

            # ピッチ帯域にあるバンドのみを使用
            pitch_band_indices = []
            for b in range(self.num_bands):
                freq = np.logspace(
                    np.log10(self.min_freq), np.log10(self.max_freq), self.num_bands
                )[b]

                if self.pitch_min <= freq <= self.pitch_max:
                    pitch_band_indices.append(b)

            # 各バンドのISIヒストグラムを計算
            isi_histograms = []
            for b in pitch_band_indices:
                # 分析区間内のスパイクを抽出
                spikes_in_range = []
                for spike in spikes[b]:
                    if (analysis_start <= spike) and (
                        analysis_end is None or spike < analysis_end
                    ):
                        spikes_in_range.append(spike)

                # スパイクが少なすぎる場合はスキップ
                if len(spikes_in_range) < 2:
                    continue

                # ISIを計算
                intervals = np.diff(spikes_in_range)

                # ISIのヒストグラムを計算
                max_period = int(sr / self.pitch_min)  # 最低周波数に対応する最大周期
                min_period = int(sr / self.pitch_max)  # 最高周波数に対応する最小周期

                # 周期範囲内のISIのみを考慮
                valid_intervals = intervals[
                    (intervals >= min_period) & (intervals <= max_period)
                ]

                if len(valid_intervals) > 0:
                    # ヒストグラムのビン
                    bins = np.arange(min_period, max_period + 1)
                    hist, _ = np.histogram(valid_intervals, bins=bins)
                    isi_histograms.append(hist)

            # 全バンドのISIヒストグラムを合算
            if isi_histograms:
                combined_hist = np.sum(isi_histograms, axis=0)

                # 最頻値を探す
                if np.any(combined_hist > 0):
                    peak_idx = np.argmax(combined_hist)
                    period = min_period + peak_idx

                    # 周期からピッチ（Hz）に変換
                    pitch = sr / period

                    # 有効な周波数範囲内かチェック
                    if self.pitch_min <= pitch <= self.pitch_max:
                        pitches.append(pitch)
                    else:
                        pitches.append(0.0)  # 無音または不明
                else:
                    pitches.append(0.0)  # 無音または不明
            else:
                pitches.append(0.0)  # 無音または不明

        return np.array(pitches)

    def detect_notes(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        音声データから音符（オンセット、オフセット、ピッチの組み合わせ）を検出します

        Parameters
        ----------
        audio : np.ndarray
            音声データ
        sr : int
            サンプリングレート（Hz）

        Returns
        -------
        Dict[str, np.ndarray]
            検出された音符の情報を含む辞書
        """
        # 1. フィルタバンクを適用
        filtered_signals = self._apply_filterbank(audio, sr)

        # 2. 半波整流と圧縮
        rectified_signals = self._half_wave_rectify_compress(filtered_signals)

        # 3. スパイク生成
        spikes = self._generate_spikes(rectified_signals, sr)

        # 4. スパイク発火率の計算
        spike_rates = self._calculate_spike_rates(spikes, len(audio), sr)

        # 5. オンセット関数とオフセット関数の計算
        onset_function, offset_function = self._calculate_onset_offset_functions(
            spike_rates
        )

        # 6. 音量（エネルギー）関数の計算
        energy_function = np.sum(spike_rates, axis=0)
        energy_function = np.log(1e-8 + energy_function)  # 対数スケール

        # 7. 動的閾値処理によるオンセットとオフセットの検出
        onset_frames, offset_frames = self._dynamic_thresholding(
            onset_function, offset_function, energy_function
        )

        # フレームインデックスを時間（秒）に変換
        onset_times = librosa.frames_to_time(
            onset_frames, sr=sr, hop_length=self.hop_length, n_fft=self.frame_length
        )
        offset_times = librosa.frames_to_time(
            offset_frames, sr=sr, hop_length=self.hop_length, n_fft=self.frame_length
        )

        # オンセットとオフセットの数を調整
        if len(onset_times) == 0:
            return {"intervals": np.array([]).reshape(0, 2), "pitches": np.array([])}

        # オンセットとオフセットの数が一致しない場合の処理
        if len(onset_times) > len(offset_times):
            # 最後のオンセットに対応するオフセットがない場合、音声の終わりを使用
            offset_times = np.append(offset_times, len(audio) / sr)
        elif len(onset_times) < len(offset_times):
            # 最初のオフセットに対応するオンセットがない場合、不要なオフセットを削除
            offset_times = offset_times[-len(onset_times) :]

        # オンセットとオフセットの数が一致しているか確認
        if len(onset_times) != len(offset_times):
            # 数が一致しない場合は、小さい方に合わせる
            min_len = min(len(onset_times), len(offset_times))
            onset_times = onset_times[:min_len]
            offset_times = offset_times[:min_len]

        # 音符の区間を作成
        intervals = np.column_stack((onset_times, offset_times))

        # 8. ピッチ推定
        # オンセットフレームとオフセットフレームの長さを一致させる
        min_frames = min(len(onset_frames), len(offset_frames))
        onset_frames = onset_frames[:min_frames]
        offset_frames = offset_frames[:min_frames]

        pitches = self._analyze_interspike_intervals(
            spikes, onset_frames, offset_frames, sr
        )

        return {"intervals": intervals, "pitches": pitches}

    def detect_onsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        音声データからオンセット（音の始まり）を検出します

        Parameters
        ----------
        audio : np.ndarray
            音声データ
        sr : int
            サンプリングレート（Hz）

        Returns
        -------
        np.ndarray
            検出されたオンセット時間（秒）の配列
        """
        notes = self.detect_notes(audio, sr)
        return notes["intervals"][:, 0]

    def detect_offsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        音声データからオフセット（音の終わり）を検出します

        Parameters
        ----------
        audio : np.ndarray
            音声データ
        sr : int
            サンプリングレート（Hz）

        Returns
        -------
        np.ndarray
            検出されたオフセット時間（秒）の配列
        """
        notes = self.detect_notes(audio, sr)
        return notes["intervals"][:, 1]

    def detect_pitches(
        self,
        audio: np.ndarray,
        sr: int,
        onsets: Optional[np.ndarray] = None,
        offsets: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        音声データからピッチ（音の高さ）を検出します

        Parameters
        ----------
        audio : np.ndarray
            音声データ
        sr : int
            サンプリングレート（Hz）
        onsets : Optional[np.ndarray], optional
            オンセット時間（秒）の配列, by default None
        offsets : Optional[np.ndarray], optional
            オフセット時間（秒）の配列, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
            (音符ピッチ配列, ピッチフレーム時間配列, ピッチフレーム周波数配列)のタプル
        """
        if onsets is None or offsets is None:
            notes = self.detect_notes(audio, sr)
            pitches = notes["pitches"]
            intervals = notes["intervals"]
            onsets = intervals[:, 0]
            offsets = intervals[:, 1]
        else:
            # オンセットとオフセットが与えられた場合、それらを使用してピッチを検出
            # フレームに変換
            onset_frames = librosa.time_to_frames(
                onsets, sr=sr, hop_length=self.hop_length, n_fft=self.frame_length
            )
            offset_frames = librosa.time_to_frames(
                offsets, sr=sr, hop_length=self.hop_length, n_fft=self.frame_length
            )

            # フィルタバンクを適用
            filtered_signals = self._apply_filterbank(audio, sr)

            # 半波整流と圧縮
            rectified_signals = self._half_wave_rectify_compress(filtered_signals)

            # スパイク生成
            spikes = self._generate_spikes(rectified_signals, sr)

            # ピッチ推定
            pitches = self._analyze_interspike_intervals(
                spikes, onset_frames, offset_frames, sr
            )

        # ピッチフレーム時間とピッチフレーム周波数を生成
        # （このシンプル実装では、各音符のピッチを一定とみなす）
        pitch_times = np.array([])
        pitch_freqs = np.array([])

        for i, (start, end) in enumerate(zip(onsets, offsets)):
            duration = end - start
            num_frames = max(1, int(duration * sr / self.hop_length))

            # 各フレームの時間
            times = np.linspace(start, end, num_frames)

            # 各フレームの周波数（一定）
            freqs = np.ones(num_frames) * pitches[i]

            # 結合
            pitch_times = np.append(pitch_times, times)
            pitch_freqs = np.append(pitch_freqs, freqs)

        return pitches, pitch_times, pitch_freqs
