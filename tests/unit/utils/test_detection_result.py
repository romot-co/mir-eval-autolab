import numpy as np
import pytest

from src.utils.detection_result import DetectionResult

# --- DetectionResult Tests ---

@pytest.fixture
def sample_detection_data():
    """テスト用のサンプル検出データを作成するフィクスチャ"""
    return {
        "intervals": np.array([[0.1, 0.5], [0.6, 1.0]]),
        "note_pitches": np.array([60.0, 72.0]),
        "frame_times": np.arange(0.0, 1.1, 0.1),
        "frame_frequencies": np.array([440.0] * 5 + [880.0] * 6),
        "metadata": {"detector_name": "TestDetector", "param": 1},
        "extra_data": np.array([1, 2, 3]) # テスト用にNumpy配列を含む
    }

def test_detection_result_creation(sample_detection_data):
    """DetectionResultインスタンスが正しく作成されることを確認する"""
    result = DetectionResult(**sample_detection_data)

    np.testing.assert_array_equal(result.intervals, sample_detection_data["intervals"])
    np.testing.assert_array_equal(result.note_pitches, sample_detection_data["note_pitches"])
    np.testing.assert_array_equal(result.frame_times, sample_detection_data["frame_times"])
    np.testing.assert_array_equal(result.frame_frequencies, sample_detection_data["frame_frequencies"])
    assert result.metadata == sample_detection_data["metadata"]
    np.testing.assert_array_equal(result.extra_data, sample_detection_data["extra_data"])

def test_detection_result_from_dict(sample_detection_data):
    """from_dictメソッドが正しく動作することを確認する"""
    # to_dictで一度シリアライズ可能な形式にする必要がある
    serializable_data = DetectionResult(**sample_detection_data).to_dict()
    result = DetectionResult.from_dict(serializable_data)

    # Numpy配列の比較
    np.testing.assert_array_equal(result.intervals, sample_detection_data["intervals"])
    np.testing.assert_array_equal(result.note_pitches, sample_detection_data["note_pitches"])
    np.testing.assert_array_equal(result.frame_times, sample_detection_data["frame_times"])
    np.testing.assert_array_equal(result.frame_frequencies, sample_detection_data["frame_frequencies"])
    assert result.metadata == sample_detection_data["metadata"]
    # extra_data は to_dict でリストに変換されるので、再度配列に変換して比較
    np.testing.assert_array_equal(np.array(result.extra_data), sample_detection_data["extra_data"])

def test_detection_result_to_dict(sample_detection_data):
    """to_dictメソッドが正しく動作することを確認する (JSONシリアライズ可能な形式になるか)"""
    result = DetectionResult(**sample_detection_data)
    result_dict = result.to_dict()

    # Numpy配列がリストに変換されていることを確認
    assert isinstance(result_dict["intervals"], list)
    assert isinstance(result_dict["note_pitches"], list)
    assert isinstance(result_dict["frame_times"], list)
    assert isinstance(result_dict["frame_frequencies"], list)
    assert isinstance(result_dict["extra_data"], list) # Numpy配列もリストになる

    # 内容が一致しているか確認 (リストに変換して比較)
    np.testing.assert_array_equal(np.array(result_dict["intervals"]), sample_detection_data["intervals"])
    np.testing.assert_array_equal(np.array(result_dict["note_pitches"]), sample_detection_data["note_pitches"])
    np.testing.assert_array_equal(np.array(result_dict["frame_times"]), sample_detection_data["frame_times"])
    np.testing.assert_array_equal(np.array(result_dict["frame_frequencies"]), sample_detection_data["frame_frequencies"])
    assert result_dict["metadata"] == sample_detection_data["metadata"]
    np.testing.assert_array_equal(np.array(result_dict["extra_data"]), sample_detection_data["extra_data"])

def test_detection_result_missing_optional_fields():
    """オプショナルなフィールドがなくてもインスタンス化できることを確認する"""
    minimal_data = {
        "intervals": np.array([[0.1, 0.5]]),
        "note_pitches": np.array([60.0]),
        "frame_times": np.array([0.0, 0.1]),
        "frame_frequencies": np.array([440.0, 440.0]),
    }
    result = DetectionResult(**minimal_data)
    assert result.intervals is not None
    assert result.note_pitches is not None
    assert result.frame_times is not None
    assert result.frame_frequencies is not None
    assert result.metadata == {} # デフォルトは空辞書
    assert result.extra_data is None # デフォルトはNone

    # to_dict/from_dict の動作も確認
    result_dict = result.to_dict()
    reloaded_result = DetectionResult.from_dict(result_dict)
    assert reloaded_result.metadata == {}
    assert reloaded_result.extra_data is None

def test_detection_result_empty_arrays():
    """空のNumpy配列でインスタンス化できることを確認する"""
    empty_data = {
        "intervals": np.array([]).reshape(0, 2), # shape を合わせる
        "note_pitches": np.array([]),
        "frame_times": np.array([]),
        "frame_frequencies": np.array([]),
    }
    result = DetectionResult(**empty_data)
    assert result.intervals.shape == (0, 2)
    assert len(result.note_pitches) == 0
    assert len(result.frame_times) == 0
    assert len(result.frame_frequencies) == 0

    # to_dict/from_dict の動作も確認
    result_dict = result.to_dict()
    reloaded_result = DetectionResult.from_dict(result_dict)
    assert reloaded_result.intervals.shape == (0, 2) # from_dict で shape が復元される
    assert len(reloaded_result.note_pitches) == 0
    assert len(reloaded_result.frame_times) == 0
    assert len(reloaded_result.frame_frequencies) == 0 