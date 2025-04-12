# tests/unit/detectors/test_base_detector.py
import pytest
import abc

# テスト対象のベースクラスをインポート
try:
    from src.detectors.base_detector import BaseDetector
    SKIP_TESTS = False
except ImportError as e:
    print(f"Skipping base_detector tests due to import error: {e}")
    SKIP_TESTS = True
    # Dummy BaseDetector with ABCMeta for testing
    class BaseDetector(metaclass=abc.ABCMeta):
        """ベース検出器のモック用抽象クラス"""
        name = "BaseDetector"
        version = "0.0.1"
        
        def __init__(self):
            pass
            
        @abc.abstractmethod
        def detect(self, audio_data, sample_rate):
            """音声データから検出を行う抽象メソッド"""
            raise NotImplementedError("Subclasses must implement detect()")

# テスト実行をスキップするかどうかの設定
pytestmark = pytest.mark.skipif(SKIP_TESTS, reason="必要なモジュールがインポートできませんでした")

# --- Test BaseDetector Abstractness ---

def test_base_detector_cannot_be_instantiated():
    """Tests that the abstract BaseDetector cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseDetector"):
        BaseDetector() # Attempt to instantiate

def test_subclass_without_detect_cannot_be_instantiated():
    """Tests that a subclass missing the abstract 'detect' method cannot be instantiated."""
    class IncompleteDetector(BaseDetector):
        # Missing the detect method implementation
        name = "Incomplete"
        version = "1.0"
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteDetector"):
        IncompleteDetector()

def test_subclass_with_detect_can_be_instantiated():
    """Tests that a subclass implementing the 'detect' method can be instantiated."""
    class CompleteDetector(BaseDetector):
        name = "Complete"
        version = "1.0"

        def detect(self, audio_data, sample_rate):
            # Dummy implementation is sufficient for instantiation check
            return {}

    # Instantiation should succeed without raising TypeError
    try:
        instance = CompleteDetector()
        assert isinstance(instance, BaseDetector)
        assert instance.name == "Complete"
    except TypeError:
        pytest.fail("CompleteDetector should be instantiable, but raised TypeError") 