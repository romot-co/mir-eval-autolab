"""
定Q変換 (CQT) フィルターバンクの実装
"""

from typing import Dict, Optional

import librosa
import numpy as np

from src.detectors.modules.filterbank.base_filterbank import FilterBank


class CQTFilterBank(FilterBank):
    """
    定Q変換 (CQT) フィルターバンク

    周波数軸が対数スケールであり、低域では周波数分解能が高く高域では時間分解能が高い
    変換を提供します。
    """

    def _initialize(self, **kwargs):
        """
        CQTフィルターバンクの初期化

        Parameters
        ----------
        n_bins : int
            周波数ビン数
        bins_per_octave : int
            オクターブあたりのビン数
        fmin : float
            最低周波数 (Hz)
        hop_length : int
            ホップ長
        """
        self.n_bins = kwargs.get("n_bins", 84)
        self.bins_per_octave = kwargs.get("bins_per_octave", 12)
        self.fmin = kwargs.get("fmin", 32.7)  # C1の周波数
        self.hop_length = kwargs.get("hop_length", 512)

        # 周波数軸の計算
        self.frequencies = librosa.cqt_frequencies(
            n_bins=self.n_bins, fmin=self.fmin, bins_per_octave=self.bins_per_octave
        )

    def process(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        音声信号を処理してCQT表現を返す

        Parameters
        ----------
        audio : np.ndarray
            入力音声信号

        Returns
        -------
        Dict[str, np.ndarray]
            CQT表現（振幅とスペクトログラム）
        """
        # 定Q変換の計算
        C = librosa.cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )

        # 振幅スペクトログラムの計算
        magnitude = np.abs(C)

        # 位相スペクトログラムの計算
        phase = np.angle(C)

        # 時間軸の計算
        times = librosa.times_like(
            magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )

        return {
            "magnitude": magnitude,
            "phase": phase,
            "frequencies": self.frequencies,
            "times": times,
            "C": C,
        }

    def get_frequencies(self) -> np.ndarray:
        """
        フィルターバンクの中心周波数を返す

        Returns
        -------
        np.ndarray
            中心周波数の配列
        """
        return self.frequencies
