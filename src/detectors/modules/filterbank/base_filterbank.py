"""
フィルターバンクの基底クラス
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class FilterBank(ABC):
    """
    フィルターバンクの基底クラス

    すべてのフィルターバンク実装はこのクラスを継承する必要があります。
    """

    def __init__(self, sample_rate: int = 44100, **kwargs):
        """
        フィルターバンクの初期化

        Parameters
        ----------
        sample_rate : int
            サンプリングレート
        **kwargs
            追加のパラメータ
        """
        self.sample_rate = sample_rate
        self._initialize(**kwargs)

    @abstractmethod
    def _initialize(self, **kwargs):
        """
        サブクラス固有の初期化を行う

        Parameters
        ----------
        **kwargs
            追加のパラメータ
        """
        pass

    @abstractmethod
    def process(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        音声信号を処理してフィルターバンク出力を返す

        Parameters
        ----------
        audio : np.ndarray
            入力音声信号

        Returns
        -------
        Dict[str, np.ndarray]
            フィルターバンク出力（時間-周波数表現など）
        """
        pass

    @abstractmethod
    def get_frequencies(self) -> np.ndarray:
        """
        フィルターバンクの中心周波数を返す

        Returns
        -------
        np.ndarray
            中心周波数の配列
        """
        pass
