import numpy as np
import pytest
from unittest.mock import patch
import logging

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
        "detector_name": "TestDetector",
        "detection_time": 0.5,
        "param": 1,
        "extra_array": np.array([1, 2, 3]),
    }


# ロギング関数をモック化して警告をスキップ
@patch("logging.debug")
@patch("logging.warning")
@patch("logging.error")
def test_detection_result_creation(
    mock_error, mock_warning, mock_debug, sample_detection_data
):
    """DetectionResultインスタンスが正しく作成されることを確認する"""
    result = DetectionResult.from_dict(sample_detection_data)

    np.testing.assert_array_equal(result.intervals, sample_detection_data["intervals"])
    np.testing.assert_array_equal(
        result.note_pitches, sample_detection_data["note_pitches"]
    )
    np.testing.assert_array_equal(
        result.frame_times, sample_detection_data["frame_times"]
    )
    np.testing.assert_array_equal(
        result.frame_frequencies, sample_detection_data["frame_frequencies"]
    )
    assert result.detector_name == sample_detection_data["detector_name"]
    assert result.detection_time == sample_detection_data["detection_time"]

    # オプショナルフィールド以外はadditional_dataに入るはず
    result_dict = result.to_dict()
    assert "param" in result_dict
    assert result_dict["param"] == 1
    assert "extra_array" in result_dict
    np.testing.assert_array_equal(
        np.array(result_dict["extra_array"]), sample_detection_data["extra_array"]
    )


@patch("logging.debug")
@patch("logging.warning")
@patch("logging.error")
def test_detection_result_from_dict(
    mock_error, mock_warning, mock_debug, sample_detection_data
):
    """from_dictメソッドが正しく動作することを確認する"""
    result = DetectionResult.from_dict(sample_detection_data)

    # Numpy配列の比較
    np.testing.assert_array_equal(result.intervals, sample_detection_data["intervals"])
    np.testing.assert_array_equal(
        result.note_pitches, sample_detection_data["note_pitches"]
    )
    np.testing.assert_array_equal(
        result.frame_times, sample_detection_data["frame_times"]
    )
    np.testing.assert_array_equal(
        result.frame_frequencies, sample_detection_data["frame_frequencies"]
    )
    assert result.detector_name == sample_detection_data["detector_name"]
    assert result.detection_time == sample_detection_data["detection_time"]

    # to_dict でオプショナルフィールド以外が含まれていることを確認
    result_dict = result.to_dict()
    assert "param" in result_dict
    assert result_dict["param"] == 1


@patch("logging.debug")
@patch("logging.warning")
@patch("logging.error")
def test_detection_result_to_dict(
    mock_error, mock_warning, mock_debug, sample_detection_data
):
    """to_dictメソッドが正しく動作することを確認する"""
    result = DetectionResult.from_dict(sample_detection_data)
    result_dict = result.to_dict()

    # データが含まれているか確認
    assert "intervals" in result_dict
    assert "note_pitches" in result_dict
    assert "frame_times" in result_dict
    assert "frame_frequencies" in result_dict
    assert "detector_name" in result_dict
    assert "detection_time" in result_dict

    # 内容が一致しているか確認
    np.testing.assert_array_equal(
        np.array(result_dict["intervals"]), sample_detection_data["intervals"]
    )
    np.testing.assert_array_equal(
        np.array(result_dict["note_pitches"]), sample_detection_data["note_pitches"]
    )
    np.testing.assert_array_equal(
        np.array(result_dict["frame_times"]), sample_detection_data["frame_times"]
    )
    np.testing.assert_array_equal(
        np.array(result_dict["frame_frequencies"]),
        sample_detection_data["frame_frequencies"],
    )
    assert result_dict["detector_name"] == sample_detection_data["detector_name"]
    assert result_dict["detection_time"] == sample_detection_data["detection_time"]
    assert "param" in result_dict
    assert result_dict["param"] == 1


@patch("logging.debug")
@patch("logging.warning")
@patch("logging.error")
def test_detection_result_missing_optional_fields(mock_error, mock_warning, mock_debug):
    """オプショナルなフィールドがなくてもインスタンス化できることを確認する"""
    minimal_data = {
        "intervals": np.array([[0.1, 0.5]]),
        "note_pitches": np.array([60.0]),
        "frame_times": np.array([0.0, 0.1]),
        "frame_frequencies": np.array([440.0, 440.0]),
    }
    result = DetectionResult.from_dict(minimal_data)
    assert result.intervals is not None
    assert result.note_pitches is not None
    assert result.frame_times is not None
    assert result.frame_frequencies is not None
    assert result.detector_name == "Unknown"  # デフォルト値
    assert result.detection_time == 0.0  # デフォルト値
    assert isinstance(result.additional_data, dict)  # デフォルトは空辞書

    # to_dict の動作も確認
    result_dict = result.to_dict()
    assert result_dict["detector_name"] == "Unknown"
    assert result_dict["detection_time"] == 0.0


@patch("logging.debug")
@patch("logging.warning")
@patch("logging.error")
def test_detection_result_empty_arrays(mock_error, mock_warning, mock_debug):
    """空のNumpy配列でインスタンス化できることを確認する"""
    empty_data = {
        "intervals": np.array([]).reshape(0, 2),  # shape を合わせる
        "note_pitches": np.array([]),
        "frame_times": np.array([]),
        "frame_frequencies": np.array([]),
    }
    result = DetectionResult.from_dict(empty_data)
    assert result.intervals.shape == (0, 2)
    assert len(result.note_pitches) == 0
    assert len(result.frame_times) == 0
    assert len(result.frame_frequencies) == 0

    # to_dict の動作も確認
    result_dict = result.to_dict()
    assert len(result_dict["intervals"]) == 0
    assert len(result_dict["note_pitches"]) == 0
    assert len(result_dict["frame_times"]) == 0
    assert len(result_dict["frame_frequencies"]) == 0


@patch("logging.debug")
@patch("logging.warning")
@patch("logging.error")
def test_to_note_list(mock_error, mock_warning, mock_debug):
    """to_note_listメソッドが正しく動作することを確認する"""
    detection_data = {
        "intervals": np.array([[0.1, 0.5], [0.6, 1.0]]),
        "note_pitches": np.array([60.0, 72.0]),
        "frame_times": np.array([0.0, 0.1]),
        "frame_frequencies": np.array([440.0, 440.0]),
    }
    result = DetectionResult.from_dict(detection_data)
    note_list = result.to_note_list()

    assert len(note_list) == 2
    assert note_list[0]["onset"] == 0.1
    assert note_list[0]["offset"] == 0.5
    assert note_list[0]["pitch"] == 60.0
    assert note_list[1]["onset"] == 0.6
    assert note_list[1]["offset"] == 1.0
    assert note_list[1]["pitch"] == 72.0


@patch("logging.debug")
@patch("logging.warning")
def test_to_note_list_missing_pitches(mock_warning, mock_debug, caplog):
    """intervalsとnote_pitchesの長さが一致しない場合、ValueErrorが発生し、エラーログが記録されることを確認する"""
    detection_data = {
        "intervals": np.array([[0.1, 0.5], [0.6, 1.0], [1.2, 1.5]]),
        "note_pitches": np.array([60.0, 72.0]),  # 3つ目のピッチが欠けている
        "frame_times": np.array([0.0, 0.1]),
        "frame_frequencies": np.array([440.0, 440.0]),
    }

    # __post_init__でValueErrorが発生することを確認
    # caplogを使ってエラーログをキャプチャ
    with caplog.at_level(logging.ERROR):
        with pytest.raises(
            ValueError, match="intervals と note_pitches の長さが一致しません。"
        ):
            DetectionResult.from_dict(detection_data)

    # エラーレベルのログが1件以上記録されていることを確認
    assert len(caplog.records) >= 1
    error_log_found = False
    for record in caplog.records:
        if record.levelno == logging.ERROR and "長さが一致しません" in record.message:
            error_log_found = True
            break
    assert error_log_found, "期待されるエラーログが見つかりませんでした。"


@patch("logging.debug")
@patch("logging.warning")
@patch("logging.error")
def test_detection_result_missing_essential_keys(mock_error, mock_warning, mock_debug):
    """必須キーが欠けている場合に適切なエラーが発生することを確認する"""
    incomplete_data = {
        "intervals": np.array([[0.1, 0.5]]),
        "note_pitches": np.array([60.0]),
        # "frame_times" が欠けている
        "frame_frequencies": np.array([440.0, 440.0]),
    }

    with pytest.raises(ValueError) as excinfo:
        DetectionResult.from_dict(incomplete_data)

    # エラーメッセージに欠けているキーが含まれていることを確認
    assert "frame_times" in str(excinfo.value)
    mock_error.assert_called_once()


@patch("logging.debug")
@patch("logging.warning")
@patch("logging.error")
def test_to_note_list_empty(mock_error, mock_warning, mock_debug):
    """空の検出結果に対してto_note_listが空のリストを返すことを確認する"""
    empty_data = {
        "intervals": np.array([]).reshape(0, 2),
        "note_pitches": np.array([]),
        "frame_times": np.array([]),
        "frame_frequencies": np.array([]),
    }
    result = DetectionResult.from_dict(empty_data)
    note_list = result.to_note_list()

    assert isinstance(note_list, list)
    assert len(note_list) == 0


# extract_note_data関数のテスト
from src.utils.detection_result import extract_note_data


def test_extract_note_data_from_dict():
    """辞書から音符データを抽出できることを確認する"""
    note_data = {
        "intervals": np.array([[0.1, 0.5], [0.6, 1.0]]),
        "note_pitches": np.array([60.0, 72.0]),
    }

    intervals, pitches = extract_note_data(note_data)

    np.testing.assert_array_equal(intervals, note_data["intervals"])
    np.testing.assert_array_equal(pitches, note_data["note_pitches"])


def test_extract_note_data_from_dict_with_pitches():
    """pitchesキーを持つ辞書からデータを抽出できることを確認する"""
    ref_data = {
        "intervals": np.array([[0.1, 0.5], [0.6, 1.0]]),
        "pitches": np.array([60.0, 72.0]),  # 参照データはpitchesキーを使用
    }

    intervals, pitches = extract_note_data(ref_data)

    np.testing.assert_array_equal(intervals, ref_data["intervals"])
    np.testing.assert_array_equal(pitches, ref_data["pitches"])


def test_extract_note_data_from_object():
    """get_note_dataメソッドを持つオブジェクトからデータを抽出できることを確認する"""

    class MockObject:
        def get_note_data(self):
            return (np.array([[0.1, 0.5]]), np.array([60.0]))

    mock_obj = MockObject()
    intervals, pitches = extract_note_data(mock_obj)

    np.testing.assert_array_equal(intervals, np.array([[0.1, 0.5]]))
    np.testing.assert_array_equal(pitches, np.array([60.0]))


def test_extract_note_data_from_other():
    """その他のオブジェクトからデータを抽出できることを確認する"""
    # ここでは単純なタプルではなく、実際に必要な形式（intervals, pitches）を渡す
    # extract_note_data関数は、辞書でなく、かつget_note_dataメソッドを持たないオブジェクトについては、
    # そのまま結果として返す
    data = (np.array([[0.1, 0.5]]), np.array([60.0]))
    result = extract_note_data(data)

    # 関数の実装ではdictでないオブジェクト（get_note_dataメソッドを持たない）はそのまま返される
    assert result is data

    # アンパックを避け、テストのためだけに結果を取り出す
    if isinstance(result, tuple) and len(result) == 2:
        intervals, pitches = result
        np.testing.assert_array_equal(intervals, np.array([[0.1, 0.5]]))
        np.testing.assert_array_equal(pitches, np.array([60.0]))


@patch("src.utils.detection_result.logger.warning")
def test_extract_note_data_missing_data(mock_warning):
    """データが見つからない場合に空の配列が返されることを確認する"""
    empty_dict = {}
    intervals, pitches = extract_note_data(empty_dict)

    assert intervals.shape == (0, 2)
    assert pitches.shape == (0,)
    mock_warning.assert_called_once()
