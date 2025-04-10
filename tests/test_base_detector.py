"""
BaseDetectorクラスのテスト
"""
import pytest
import numpy as np
from abc import abstractmethod
from typing import Dict, Any

# テスト対象のモジュールをインポート
# PYTHONPATHが設定されていれば、通常はこれで動作するはず
# 必要に応じて sys.path の調整が必要になる場合があります
try:
    from src.detectors.base_detector import BaseDetector
except ImportError:
    # プロジェクトルートからの相対パスでインポートを試みる
    # (pytest実行時のカレントディレクトリに依存)
    import sys
    import os
    # 現在のファイル (__file__) のディレクトリを取得し、その親 (tests) の親 (プロジェクトルート) をパスに追加
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.detectors.base_detector import BaseDetector


# --- テスト用の具象クラス ---
class ConcreteDetector(BaseDetector):
    """テスト用の具象検出器クラス"""
    def detect(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """具体的な detect メソッドの実装"""
        # ダミーの検出結果を返す
        return {
            'intervals': np.array([[0.1, 0.5], [1.0, 1.5]]),
            'note_pitches': np.array([440.0, 880.0]),
            'frame_times': np.array([0.0, 0.1, 0.2]),
            'frame_frequencies': np.array([0.0, 440.0, 0.0]),
            'detector_name': self.__class__.__name__,
            'detection_time': 0.1
        }

# --- テスト用の抽象メソッド未実装クラス ---
class AbstractDetectorMissingDetect(BaseDetector):
    """detect メソッドを実装していないテスト用クラス"""
    # abstractmethodである detect を実装しない
    pass

# --- テスト関数 ---

def test_base_detector_init():
    """BaseDetectorの初期化テスト"""
    params = {'param1': 10, 'param2': 'test'}
    detector = ConcreteDetector(**params)
    assert detector.params == params

def test_base_detector_init_no_params():
    """パラメータなしでの初期化テスト"""
    detector = ConcreteDetector()
    assert detector.params == {}

def test_base_detector_str():
    """BaseDetectorの __str__ メソッドのテスト"""
    detector = ConcreteDetector()
    assert str(detector) == "ConcreteDetector"

def test_base_detector_abstract_method_enforcement():
    """抽象メソッド(detect)の実装を強制するかのテスト"""
    # detectを実装していないクラスをインスタンス化しようとするとTypeErrorが発生することを確認
    with pytest.raises(TypeError) as excinfo:
        AbstractDetectorMissingDetect()
    # エラーメッセージに 'abstract methods detect' が含まれているか確認（より厳密なチェック）
    # Pythonのバージョンによってメッセージが若干異なる可能性があるため、緩めにチェック
    assert "abstract method" in str(excinfo.value)
    assert "detect" in str(excinfo.value)

def test_concrete_detector_instantiation():
    """具象クラスが問題なくインスタンス化できるかのテスト"""
    try:
        ConcreteDetector(param1='value')
    except Exception as e:
        pytest.fail(f"ConcreteDetectorのインスタンス化に失敗しました: {e}")

def test_concrete_detector_detect_method():
    """具象クラスのdetectメソッドが期待通り動作するかのテスト"""
    detector = ConcreteDetector()
    dummy_audio = np.zeros(16000)
    sr = 16000
    result = detector.detect(dummy_audio, sr)

    # 戻り値の型と主要なキーの存在を確認
    assert isinstance(result, dict)
    required_keys = ['intervals', 'note_pitches', 'frame_times', 'frame_frequencies', 'detector_name', 'detection_time']
    for key in required_keys:
        assert key in result

    # 各キーの値の型を確認 (より詳細なチェック)
    assert isinstance(result['intervals'], np.ndarray)
    assert isinstance(result['note_pitches'], np.ndarray)
    assert isinstance(result['frame_times'], np.ndarray)
    assert isinstance(result['frame_frequencies'], np.ndarray)
    assert isinstance(result['detector_name'], str)
    assert isinstance(result['detection_time'], float)

    # detector_nameが正しいか確認
    assert result['detector_name'] == "ConcreteDetector" 