"""
短時間フーリエ変換 (STFT) フィルターバンクの実装
"""

from typing import Dict, Optional

import librosa
import numpy as np

from src.detectors.modules.filterbank.base_filterbank import FilterBank


class STFTFilterBank(FilterBank):
    """
    短時間フーリエ変換 (STFT) フィルターバンク

    均一な周波数分解能を持つ時間-周波数表現を提供します。
    """

    def _initialize(self, **kwargs):
        """
        STFTフィルターバンクの初期化

        Parameters
        ----------
        n_fft : int
            FFTのサイズ
        hop_length : int
            ホップ長
        win_length : int
            窓関数の長さ
        window : str
            窓関数の種類
        """
        self.n_fft = kwargs.get("n_fft", 2048)
        self.hop_length = kwargs.get("hop_length", 512)
        self.win_length = kwargs.get("win_length", None)
        self.window = kwargs.get("window", "hann")

        # 周波数軸の計算
        self.frequencies = librosa.fft_frequencies(
            sr=self.sample_rate, n_fft=self.n_fft
        )

    def process(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        音声信号を処理してSTFT表現を返す

        Parameters
        ----------
        audio : np.ndarray
            入力音声信号

        Returns
        -------
        Dict[str, np.ndarray]
            STFT表現（振幅とスペクトログラム）
        """
        # 短時間フーリエ変換
        D = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )

        # 振幅スペクトログラムの計算
        magnitude = np.abs(D)

        # 位相スペクトログラムの計算
        phase = np.angle(D)

        # 時間軸の計算
        times = librosa.times_like(
            magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )

        return {
            "magnitude": magnitude,
            "phase": phase,
            "frequencies": self.frequencies,
            "times": times,
            "D": D,
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
