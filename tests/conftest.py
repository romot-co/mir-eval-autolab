"""
テスト共通のフィクスチャや設定を提供するモジュール

このモジュールは、テスト実行時に共通して使用されるフィクスチャや設定を定義します。
pytest によって自動的に読み込まれ、各テストから利用可能になります。
"""

import pytest
import numpy as np
import os
import tempfile
import json
import yaml
import shutil
from typing import Dict, Any, Tuple, List
from src.utils.detection_result import DetectionResult
from src.detectors.base_detector import BaseDetector

# ヘルパー関数
def create_temp_output_dir():
    """一時的な出力ディレクトリを作成します"""
    temp_dir = tempfile.mkdtemp(prefix="mirex_test_")
    return temp_dir

def cleanup_temp_dir(temp_dir):
    """一時ディレクトリを削除します"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def create_test_config_file(output_dir, filename, evaluator_params=None):
    """テスト用の設定ファイルを作成します"""
    config = {
        'random_seed': 42,
    }
    
    if evaluator_params:
        config['evaluation'] = evaluator_params
    
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w') as f:
        yaml.dump(config, f)
    
    return file_path, config

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


@pytest.fixture
def dummy_reference_data():
    """
    テスト用のダミー参照データを生成するフィクスチャ
    
    Returns
    -------
    Dict[str, np.ndarray]
        参照データの辞書
    """
    # 4つのノートの情報（オンセット, オフセット, ピッチ）
    intervals = np.array([
        [0.0, 1.0],  # A4
        [1.0, 2.0],  # D5
        [2.0, 3.0],  # E5
        [3.0, 4.0]   # A5
    ])
    
    pitches = np.array([440.0, 587.0, 659.0, 880.0])
    
    return {
        'intervals': intervals,
        'pitches': pitches
    }


@pytest.fixture
def dummy_detection_result():
    """
    検出結果のダミーデータを生成します。
    
    Returns
    -------
    DetectionResult
        検出結果オブジェクト
    """
    # オンセット（やや遅れがある）
    onsets = np.array([0.05, 1.05, 2.05, 3.05])
    
    # オフセット（やや早い終わり）
    offsets = np.array([0.95, 1.95, 2.95, 3.95])
    
    # 音符区間の作成
    intervals = np.column_stack((onsets, offsets))
    
    # ピッチ（やや不正確）
    pitches = np.array([442.0, 585.0, 661.0, 878.0])
    
    # フレームデータも作成
    frame_times = np.linspace(0, 4, 400)
    frame_frequencies = np.zeros_like(frame_times)
    
    # 各音符のフレームデータを設定
    for i, (onset, offset) in enumerate(intervals):
        start_idx = int(onset * 100)
        end_idx = int(offset * 100)
        if end_idx < len(frame_frequencies):
            frame_frequencies[start_idx:end_idx] = pitches[i]
    
    return DetectionResult(
        intervals=intervals,
        note_pitches=pitches,
        frame_times=frame_times,
        frame_frequencies=frame_frequencies,
        detector_name="DummyDetector",
        detection_time=0.1
    )


@pytest.fixture
def dummy_evaluation_result():
    """
    テスト用のダミー評価結果を生成するフィクスチャ
    
    Returns
    -------
    Dict[str, Any]
        評価結果の辞書
    """
    return {
        'onset_precision': 0.9,
        'onset_recall': 0.85,
        'onset_f_measure': 0.874,
        
        'offset_precision': 0.8,
        'offset_recall': 0.75,
        'offset_f_measure': 0.774,
        
        'pitch_precision': 0.95,
        'pitch_recall': 0.9,
        'pitch_f_measure': 0.924,
        
        'note_precision': 0.85,
        'note_recall': 0.8,
        'note_f_measure': 0.824,
        
        'overall_f_measure': 0.85
    }


@pytest.fixture
def temp_config_file():
    """
    テスト用の一時設定ファイルを作成するフィクスチャ
    
    Yields
    ------
    str
        一時ファイルのパス
    """
    # ヘルパー関数を使用して設定ファイルを作成
    evaluator_params = {
        'tolerance_onset': 0.05,
        'tolerance_offset': 0.05,
        'tolerance_pitch': 0.5,
        'offset_ratio': 0.2,
        'use_pitch_chroma': False
    }
    
    config = {
        'random_seed': 42,
        'evaluation': evaluator_params
    }
    
    temp_dir = create_temp_output_dir()
    file_path, _ = create_test_config_file(temp_dir, "config.yaml", evaluator_params=evaluator_params)
    
    yield file_path
    
    # テスト終了後にディレクトリを削除
    cleanup_temp_dir(temp_dir)


@pytest.fixture
def temp_grid_config_file():
    """
    テスト用の一時グリッドサーチ設定ファイルを作成するフィクスチャ
    
    Yields
    ------
    str
        一時ファイルのパス
    """
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as temp_file:
        temp_path = temp_file.name
        
        # グリッドサーチ設定を書き込み
        config = {
            'detector': 'MockDetector',
            'param_grid': {
                'onset_threshold': [0.3, 0.4],
                'offset_threshold': [0.1, 0.2]
            },
            'audio_dir': 'datasets/synthesized/audio',
            'reference_dir': 'datasets/synthesized/labels',
            'save_plots': True
        }
        
        yaml.dump(config, temp_file)
    
    yield temp_path
    
    # テスト終了後にファイルを削除
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_output_dir():
    """
    テスト用の一時出力ディレクトリを作成するフィクスチャ
    
    Yields
    ------
    str
        一時ディレクトリのパス
    """
    temp_dir = create_temp_output_dir()
    yield temp_dir
    cleanup_temp_dir(temp_dir)


@pytest.fixture
def temp_audio_file(dummy_audio_data):
    """
    テスト用の一時音声ファイルを作成するフィクスチャ
    
    Parameters
    ----------
    dummy_audio_data : Tuple[np.ndarray, int]
        (音声データ, サンプリングレート)のタプル
        
    Yields
    ------
    str
        一時ファイルのパス
    """
    import soundfile as sf
    
    temp_dir = create_temp_output_dir()
    temp_path = os.path.join(temp_dir, "audio.wav")
        
    # ファイルに音声データを書き込み
    audio, sr = dummy_audio_data
    sf.write(temp_path, audio, sr)
    
    yield temp_path
    
    # テスト終了後にディレクトリを削除
    cleanup_temp_dir(temp_dir)


@pytest.fixture
def temp_reference_file(dummy_reference_data):
    """
    テスト用の一時参照ファイルを作成するフィクスチャ
    
    Parameters
    ----------
    dummy_reference_data : Dict[str, np.ndarray]
        参照データの辞書
        
    Yields
    ------
    str
        一時ファイルのパス
    """
    temp_dir = create_temp_output_dir()
    
    # JSON形式で参照データを保存
    file_path = os.path.join(temp_dir, "reference.json")
    with open(file_path, 'w') as f:
        json.dump({
            'intervals': dummy_reference_data['intervals'].tolist(),
            'pitches': dummy_reference_data['pitches'].tolist()
        }, f, indent=2)
    
    yield file_path
    
    # テスト終了後にディレクトリを削除
    cleanup_temp_dir(temp_dir)


@pytest.fixture
def mock_detector():
    """
    テスト用のモック検出器を作成するフィクスチャ
    
    Returns
    -------
    MockDetector
        モック検出器のインスタンス
    """
    return MockDetector()


@pytest.fixture
def custom_mock_detector():
    """
    テスト用のカスタマイズ可能なモック検出器を作成するフィクスチャ
    
    Returns
    -------
    CustomizableMockDetector
        カスタマイズ可能なモック検出器のインスタンス
    """
    return CustomizableMockDetector()


@pytest.fixture
def error_mock_detector():
    """
    エラーを発生させるモック検出器を作成するフィクスチャ
    
    Returns
    -------
    ErrorMockDetector
        エラーを発生させるモック検出器のインスタンス
    """
    return ErrorMockDetector()

# モック検出器クラス
class MockDetector(BaseDetector):
    """テスト用の固定結果を返すモック検出器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def detect(self, audio_data, sr):
        # 固定の検出結果を返す
        intervals = np.array([[0.05, 0.95], [1.05, 1.95], [2.05, 2.95], [3.05, 3.95]])
        pitches = np.array([442.0, 585.0, 661.0, 878.0])
        
        frame_times = np.linspace(0, 4, 400)
        frame_frequencies = np.zeros_like(frame_times)
        
        # 検出結果を返す
        return {
            'intervals': intervals,
            'note_pitches': pitches,
            'frame_times': frame_times,
            'frame_frequencies': frame_frequencies,
            'detector_name': 'MockDetector',
            'detection_time': 0.1
        }

class CustomizableMockDetector(BaseDetector):
    """カスタマイズ可能なモック検出器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.intervals = np.array([[0.05, 0.95], [1.05, 1.95], [2.05, 2.95], [3.05, 3.95]])
        self.pitches = np.array([442.0, 585.0, 661.0, 878.0])
        
    def set_intervals(self, intervals):
        self.intervals = np.array(intervals)
        
    def set_pitches(self, pitches):
        self.pitches = np.array(pitches)
        
    def detect(self, audio_data, sr):
        # カスタマイズされた検出結果を返す
        frame_times = np.linspace(0, 4, 400)
        frame_frequencies = np.zeros_like(frame_times)
        
        return {
            'intervals': self.intervals,
            'note_pitches': self.pitches,
            'frame_times': frame_times,
            'frame_frequencies': frame_frequencies,
            'detector_name': 'CustomizableMockDetector',
            'detection_time': 0.1
        }

class ErrorMockDetector(BaseDetector):
    """エラーを発生させるモック検出器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def detect(self, audio_data, sr):
        # 常にエラーを発生させる
        raise RuntimeError("テスト用のエラー: このエラーは意図的に発生させています") 