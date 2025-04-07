"""
音楽ノート検出の基本クラス
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

class BaseDetector(ABC):
    """
    音楽ノート検出の基本クラス
    すべての検出器はこのクラスを継承する必要があります。
    """
    
    def __init__(self, **kwargs):
        """
        検出器の初期化

        Parameters
        ----------
        **kwargs : dict
            検出器のパラメータ
        """
        self.params = kwargs
    
    @abstractmethod
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
            検出結果の辞書。以下のキーを含む必要があります：
            {
                'intervals': np.ndarray,  # 形状 (N, 2), 各行は [onset, offset] (秒)
                'note_pitches': np.ndarray,  # 形状 (N,), 各要素はHz単位のピッチ
                'frame_times': np.ndarray,  # 形状 (M,), フレーム時刻 (秒)
                'frame_frequencies': np.ndarray,  # 形状 (M,), フレーム周波数 (Hz)
                'detector_name': str,  # 検出器名
                'detection_time': float  # 検出処理時間 (秒)
            }
            
            注意:
            - フレーム評価ができない検出器は 'frame_times' と 'frame_frequencies' に空配列 np.array([]) を設定してください
            - ノートが検出されなかった場合は 'intervals' は shape=(0,2)、'note_pitches' は shape=(0,) の空配列を返してください
            - 無音/無声フレームの周波数は 0.0 Hz で表現してください
            - ポリフォニー（複数ノートの同時発音）は、'intervals' 配列内で時間的に重複する複数の区間と、
              それに対応する 'note_pitches' の要素によって表現されます。例えば、和音の場合は同じ時間区間に
              複数のピッチが存在することになります。
            - フレームベースの結果 ('frame_frequencies') は、各時点で最も顕著なピッチのみを返すことを
              想定しています。複数のピッチを返す場合は、追加のキー 'frame_multi_frequencies' (shape=(M,K))
              を使用してください。
        """
        raise NotImplementedError("detect メソッドを実装してください")
    
    def __str__(self) -> str:
        return self.__class__.__name__ 