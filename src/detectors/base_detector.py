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
    def detect(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """
        オーディオデータから音楽情報を検出します。

        サブクラスはこのメソッドを実装する必要があります。

        Parameters
        ----------
        audio_data : np.ndarray
            入力オーディオデータ。形状は (サンプル数,) または (チャンネル数, サンプル数)。
            通常はモノラル (サンプル数,) を想定します。
        sample_rate : int
            オーディオデータのサンプルレート (Hz)。

        Returns
        -------
        Dict[str, np.ndarray]
            検出結果を含む辞書。以下のキーを含む必要があります:

            - 'intervals': np.ndarray
                形状: (n_notes, 2)
                各行が [開始時間 (秒), 終了時間 (秒)] を表すノートのインターバル。
                ノートが検出されなかった場合は空の配列 (形状 (0, 2)) を返します。
            - 'note_pitches': np.ndarray
                形状: (n_notes,)
                各ノートに対応するピッチ (MIDIノート番号、または Hz)。
                'intervals' の各行に対応します。
                ノートが検出されなかった場合は空の配列 (形状 (0,)) を返します。

            以下のキーはオプションですが、提供されることが推奨されます:

            - 'frame_times': np.ndarray (オプション)
                形状: (n_frames,)
                フレームごとの解析を行う場合、各フレームの中心時間 (秒)。
                提供されない場合は空の配列 (形状 (0,)) またはキー自体が存在しない場合があります。
            - 'frame_frequencies': np.ndarray (オプション)
                形状: (n_frames,)
                各フレームで検出された主要な周波数 (Hz)。
                'frame_times' の各フレームに対応します。
                提供されない場合、またはフレーム解析を行わない場合は空の配列 (形状 (0,)) またはキー自体が存在しない場合があります。
                多声音楽の場合、この値の解釈は限定的になる可能性があります。

            サブクラスは、追加の情報を独自のキーで返すことも可能です。
            返される配列のデータ型は通常 float または int です。
        """
        raise NotImplementedError("サブクラスはこのメソッドを実装する必要があります。")
    
    def __str__(self) -> str:
        return self.__class__.__name__ 