"""
検出器の基本機能をテストするモジュール

このモジュールでは、BaseDetectorを継承した検出器クラスが
正しく動作することを確認するためのテストを提供します。
"""

import pytest
import numpy as np
import os
import pandas as pd
from typing import Dict, Any

from src.detectors.base_detector import BaseDetector
from src.detectors.criteria_detector import CriteriaDetector
from src.detectors.pzstd_detector import PZSTDDetector
from src.utils.detection_result import DetectionResult
from src.utils.detector_utils import get_detector_class


def test_base_detector_interface():
    """BaseDetectorのインターフェースが正しく定義されていることを確認"""
    # BaseDetectorは抽象クラスなのでインスタンス化できないことを確認
    with pytest.raises(TypeError):
        detector = BaseDetector()
    
    # 抽象メソッドが存在することを確認
    assert hasattr(BaseDetector, 'detect')

def test_get_detector_class():
    """get_detector_class関数が正しく動作することを確認"""
    # 標準検出器の取得
    criteria_cls = get_detector_class('CriteriaDetector')
    assert criteria_cls.__name__ == 'CriteriaDetector'

    # PZSTDDetectorの取得
    pzstd_cls = get_detector_class('PZSTDDetector')
    assert pzstd_cls.__name__ == 'PZSTDDetector'

    # 存在しない検出器名でのエラー
    with pytest.raises(ImportError):
        get_detector_class('NonExistentDetector')


@pytest.fixture
def dummy_audio_data():
    """
    テスト用のダミー音声データを生成するフィクスチャ
    
    Returns
    -------
    Tuple[np.ndarray, int]
        (音声データ, サンプリングレート)のタプル
    """
    # 4秒のサイン波（440Hz, 587Hz, 659Hz, 880Hz）
    sr = 22050  # サンプリングレート
    duration = 4.0  # 秒
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 各ノートのデータ生成
    note1 = 0.5 * np.sin(2 * np.pi * 440 * t[:int(sr * 1.0)])  # A4, 1秒
    note2 = 0.5 * np.sin(2 * np.pi * 587 * t[int(sr * 1.0):int(sr * 2.0)])  # D5, 1秒
    note3 = 0.5 * np.sin(2 * np.pi * 659 * t[int(sr * 2.0):int(sr * 3.0)])  # E5, 1秒
    note4 = 0.5 * np.sin(2 * np.pi * 880 * t[int(sr * 3.0):int(sr * 4.0)])  # A5, 1秒
    
    # ノートを結合して音声データを作成
    audio = np.concatenate((note1, note2, note3, note4))
    
    return audio, sr


def test_detector_unified_interface(dummy_audio_data):
    """検出器の統一インターフェースが正しく動作することを確認"""
    audio, sr = dummy_audio_data
    
    # 複数の検出器でテスト
    detectors = [
        CriteriaDetector(),
        PZSTDDetector()
    ]
    
    for detector in detectors:
        # detectメソッドを呼び出し
        result = detector.detect(audio_data=audio, sr=sr)
        
        # 必須キーが存在することを確認
        assert 'intervals' in result, f"{detector.__class__.__name__}からintervalsがありません"
        assert 'note_pitches' in result, f"{detector.__class__.__name__}からnote_pitchesがありません"
        assert 'detector_name' in result, f"{detector.__class__.__name__}からdetector_nameがありません"
        
        # フレームベース出力が存在することを確認
        assert 'frame_times' in result, f"{detector.__class__.__name__}からframe_timesがありません"
        assert 'frame_frequencies' in result, f"{detector.__class__.__name__}からframe_frequenciesがありません"
        
        # 旧形式のキーが存在しないことを確認
        assert 'onsets' not in result, f"{detector.__class__.__name__}が旧形式のキー 'onsets' を使用しています"
        assert 'offsets' not in result, f"{detector.__class__.__name__}が旧形式のキー 'offsets' を使用しています"
        assert 'pitch_times' not in result, f"{detector.__class__.__name__}が旧形式のキー 'pitch_times' を使用しています"
        assert 'pitch_values' not in result, f"{detector.__class__.__name__}が旧形式のキー 'pitch_values' を使用しています"
        assert 'times' not in result, f"{detector.__class__.__name__}が旧形式のキー 'times' を使用しています"
        assert 'freqs' not in result, f"{detector.__class__.__name__}が旧形式のキー 'freqs' を使用しています"
        
        # データの整合性を確認
        intervals = result['intervals']
        pitches = result['note_pitches']
        
        if len(intervals) > 0:
            # intervalsとnote_pitchesの長さが同じであることを確認
            assert len(intervals) == len(pitches), \
                f"{detector.__class__.__name__}: intervalsとnote_pitchesの長さが異なります"
            
            # intervalsの形状が正しいことを確認
            assert intervals.shape[1] == 2, \
                f"{detector.__class__.__name__}: intervalsの列数が2ではありません"
            
            # frame_timesとframe_frequenciesの長さが一致することを確認
            if len(result['frame_times']) > 0:
                assert len(result['frame_times']) == len(result['frame_frequencies']), \
                    f"{detector.__class__.__name__}: frame_timesとframe_frequenciesの長さが一致しません"
        else:
            # データがない場合は空の配列になっていることを確認
            assert len(pitches) == 0, f"{detector.__class__.__name__}: データがないのにnote_pitchesが空でありません" 