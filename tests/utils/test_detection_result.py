import pytest
import numpy as np
from src.utils.detection_result import DetectionResult, extract_note_data
import logging

class TestDetectionResult:
    """DetectionResultクラスのテスト"""
    
    def test_empty_constructor(self):
        """空のコンストラクタでDetectionResultが正しく初期化されることを確認"""
        result = DetectionResult()
        
        # 必須フィールドが空の配列として初期化されていることを確認
        assert result.intervals.shape == (0, 2)
        assert result.note_pitches.shape == (0,)
        assert result.frame_times.shape == (0,)
        assert result.frame_frequencies.shape == (0,)
        
        # オプションフィールドがデフォルト値で初期化されていることを確認
        assert result.detector_name == "Unknown"
        assert result.detection_time == 0.0
        assert result.additional_data == {}
        
    def test_constructor_with_args(self):
        """引数付きのコンストラクタでDetectionResultが正しく初期化されることを確認"""
        intervals = np.array([[0.1, 0.5], [0.7, 1.2]])
        note_pitches = np.array([440.0, 880.0])
        frame_times = np.array([0.1, 0.2, 0.3])
        frame_frequencies = np.array([440.0, 440.0, 880.0])
        
        result = DetectionResult(
            intervals=intervals,
            note_pitches=note_pitches,
            frame_times=frame_times,
            frame_frequencies=frame_frequencies,
            detector_name="TestDetector",
            detection_time=0.5
        )
        
        # 各フィールドが正しく設定されていることを確認
        np.testing.assert_array_equal(result.intervals, intervals)
        np.testing.assert_array_equal(result.note_pitches, note_pitches)
        np.testing.assert_array_equal(result.frame_times, frame_times)
        np.testing.assert_array_equal(result.frame_frequencies, frame_frequencies)
        assert result.detector_name == "TestDetector"
        assert result.detection_time == 0.5
        assert result.additional_data == {}
        
    def test_from_dict_valid(self):
        """有効な辞書からDetectionResultが正しく作成されることを確認"""
        detection_dict = {
            'intervals': [[0.1, 0.5], [0.7, 1.2]],
            'note_pitches': [440.0, 880.0],
            'frame_times': [0.1, 0.2, 0.3],
            'frame_frequencies': [440.0, 440.0, 880.0],
            'detector_name': "TestDetector",
            'detection_time': 0.5
        }
        
        result = DetectionResult.from_dict(detection_dict)
        
        # 各フィールドが正しく設定されていることを確認
        np.testing.assert_array_equal(result.intervals, np.array(detection_dict['intervals']))
        np.testing.assert_array_equal(result.note_pitches, np.array(detection_dict['note_pitches']))
        np.testing.assert_array_equal(result.frame_times, np.array(detection_dict['frame_times']))
        np.testing.assert_array_equal(result.frame_frequencies, np.array(detection_dict['frame_frequencies']))
        assert result.detector_name == "TestDetector"
        assert result.detection_time == 0.5
        
    def test_from_dict_missing_keys(self):
        """必須キーが不足している辞書からDetectionResultを作成しようとするとエラーが発生することを確認"""
        # 必須キー'intervals'が不足している辞書
        detection_dict = {
            'note_pitches': [440.0, 880.0],
            'frame_times': [0.1, 0.2, 0.3],
            'frame_frequencies': [440.0, 440.0, 880.0]
        }
        
        with pytest.raises(ValueError) as excinfo:
            DetectionResult.from_dict(detection_dict)
            
        assert "必須キーが不足しています" in str(excinfo.value)
        
    def test_from_dict_length_mismatch(self, caplog):
        """intervalsとnote_pitchesの長さが一致しない場合、警告が出力されることを確認"""
        caplog.set_level(logging.WARNING)
        
        # intervalsとnote_pitchesの長さが一致しない辞書
        detection_dict = {
            'intervals': [[0.1, 0.5], [0.7, 1.2]],
            'note_pitches': [440.0],  # note_pitchesの長さが不足
            'frame_times': [0.1, 0.2, 0.3],
            'frame_frequencies': [440.0, 440.0, 880.0]
        }
        
        result = DetectionResult.from_dict(detection_dict)
        
        # 警告が出力されていることを確認
        assert "検出結果の長さが一致しません" in caplog.text
        
        # 結果は作成されていることを確認
        assert result.intervals.shape == (2, 2)
        assert result.note_pitches.shape == (1,)
        
    def test_to_dict(self):
        """to_dictメソッドが正しく動作することを確認"""
        intervals = np.array([[0.1, 0.5], [0.7, 1.2]])
        note_pitches = np.array([440.0, 880.0])
        frame_times = np.array([0.1, 0.2, 0.3])
        frame_frequencies = np.array([440.0, 440.0, 880.0])
        
        result = DetectionResult(
            intervals=intervals,
            note_pitches=note_pitches,
            frame_times=frame_times,
            frame_frequencies=frame_frequencies,
            detector_name="TestDetector",
            detection_time=0.5
        )
        
        # 追加データを設定
        result.additional_data = {'confidence': [0.9, 0.8]}
        
        # 辞書に変換
        result_dict = result.to_dict()
        
        # 各キーが存在し、正しい値が設定されていることを確認
        np.testing.assert_array_equal(result_dict['intervals'], intervals)
        np.testing.assert_array_equal(result_dict['note_pitches'], note_pitches)
        np.testing.assert_array_equal(result_dict['frame_times'], frame_times)
        np.testing.assert_array_equal(result_dict['frame_frequencies'], frame_frequencies)
        assert result_dict['detector_name'] == "TestDetector"
        assert result_dict['detection_time'] == 0.5
        assert result_dict['confidence'] == [0.9, 0.8]
        
    def test_to_note_list(self):
        """to_note_listメソッドが正しく動作することを確認"""
        intervals = np.array([[0.1, 0.5], [0.7, 1.2]])
        note_pitches = np.array([440.0, 880.0])
        
        result = DetectionResult(
            intervals=intervals,
            note_pitches=note_pitches
        )
        
        note_list = result.to_note_list()
        
        assert len(note_list) == 2
        assert note_list[0]['onset'] == 0.1
        assert note_list[0]['offset'] == 0.5
        assert note_list[0]['pitch'] == 440.0
        assert note_list[1]['onset'] == 0.7
        assert note_list[1]['offset'] == 1.2
        assert note_list[1]['pitch'] == 880.0
        
    def test_to_note_list_empty(self):
        """空のDetectionResultからto_note_listメソッドが空のリストを返すことを確認"""
        result = DetectionResult()
        note_list = result.to_note_list()
        
        assert len(note_list) == 0
        
    def test_to_note_list_pitch_mismatch(self):
        """note_pitchesの長さがintervalsより少ない場合のto_note_listの挙動を確認"""
        intervals = np.array([[0.1, 0.5], [0.7, 1.2]])
        note_pitches = np.array([440.0])  # 1つしかない
        
        result = DetectionResult(
            intervals=intervals,
            note_pitches=note_pitches
        )
        
        note_list = result.to_note_list()
        
        assert len(note_list) == 2
        assert note_list[0]['pitch'] == 440.0
        assert note_list[1]['pitch'] == 0.0  # ピッチが不足している場合は0.0が設定される

    def test_from_dict_debug_logging(self, caplog):
        """from_dictメソッドがデバッグログを正しく出力することを確認"""
        caplog.set_level(logging.DEBUG)
        
        # 様々な型のデータを含む辞書を作成
        detection_dict = {
            'intervals': np.array([[0.1, 0.5], [0.7, 1.2]]),
            'note_pitches': np.array([440.0, 880.0]),
            'frame_times': np.array([0.1, 0.2, 0.3]),
            'frame_frequencies': np.array([440.0, 440.0, 880.0]),
            'detector_name': "TestDetector",
            'detection_time': 0.5,
            'empty_array': np.array([]), 
            'string_data': "テストデータ"
        }
        
        # caplogを使ってログ出力を捕捉
        DetectionResult.from_dict(detection_dict)
        
        # デバッグログの出力を確認
        assert "==== 検出器から受け取った結果のキー ====" in caplog.text
        assert "キー 'intervals': 形状=(2, 2), 長さ=2" in caplog.text
        assert "キー 'note_pitches': 形状=(2,), 長さ=2" in caplog.text
        assert "キー 'frame_times': 形状=(3,), 長さ=3" in caplog.text
        assert "キー 'frame_frequencies': 形状=(3,), 長さ=3" in caplog.text
        assert "キー 'detector_name': タイプ=<class 'str'>" in caplog.text
        assert "キー 'detection_time': タイプ=<class 'float'>" in caplog.text
        assert "キー 'empty_array': 形状=(0,), 長さ=0" in caplog.text
        assert "内容: 空の配列" in caplog.text
        assert "==== DetectionResultに変換後のデータ ====" in caplog.text
        assert "intervals: 形状=(2, 2), 長さ=2" in caplog.text
        assert "note_pitches: 形状=(2,), 長さ=2" in caplog.text
    
    def test_from_dict_debug_logging_small_array(self, caplog):
        """from_dictメソッドが小さな配列の内容を詳細にデバッグログに出力することを確認"""
        caplog.clear()  # 前のテストのログをクリア
        caplog.set_level(logging.DEBUG)
        
        # 要素が少ない配列を含む辞書
        detection_dict = {
            'intervals': np.array([[0.1, 0.2]]),  # 1つの間隔のみ
            'note_pitches': np.array([440.0]),    # 1つのピッチのみ
            'frame_times': np.array([0.1, 0.2]),
            'frame_frequencies': np.array([440.0, 440.0])
        }
        
        DetectionResult.from_dict(detection_dict)
        
        # 小さな配列の場合、内容が出力されることを確認
        assert "キー 'intervals': 形状=(1, 2), 長さ=1" in caplog.text
        assert "内容: [[0.1 0.2]]" in caplog.text
        assert "キー 'note_pitches': 形状=(1,), 長さ=1" in caplog.text
        assert "内容: [440." in caplog.text  # 浮動小数点の表示形式が環境によって異なる可能性があるため部分一致

    def test_from_dict_debug_logging_with_array_values(self, caplog):
        """from_dictメソッドが様々なタイプの配列のデバッグログを正しく出力することを確認"""
        caplog.clear()  # 前のテストのログをクリア
        caplog.set_level(logging.DEBUG)
        
        # 様々なタイプと大きさの配列を含む辞書
        detection_dict = {
            'intervals': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),  # 標準的な配列
            'note_pitches': np.array([440.0, 880.0, 660.0]),
            'frame_times': np.array([]),  # 空の配列
            'frame_frequencies': np.array([440.0] * 5),  # 5要素の配列
            'detector_name': "TestDetector",
            'detection_time': 0.5
        }
        
        DetectionResult.from_dict(detection_dict)
        
        # 標準的な配列のログ出力を確認
        assert "キー 'intervals': 形状=(3, 2), 長さ=3" in caplog.text
        assert "キー 'note_pitches': 形状=(3,), 長さ=3" in caplog.text
        
        # 空の配列のログ出力を確認
        assert "キー 'frame_times': 形状=(0,), 長さ=0" in caplog.text
        assert "内容: 空の配列" in caplog.text
        
        # 小さな配列は内容も表示されることを確認
        assert "内容: [440. 440. 440. 440. 440.]" in caplog.text
        
        # DebugResultへの変換後のログも確認
        assert "==== DetectionResultに変換後のデータ ====" in caplog.text
        assert "intervals: 形状=(3, 2), 長さ=3" in caplog.text
        assert "note_pitches: 形状=(3,), 長さ=3" in caplog.text


class TestExtractNoteData:
    """extract_note_data関数のテスト"""
    
    def test_extract_from_dict_standard_keys(self):
        """標準的なキーを持つ辞書からノートデータを抽出できることを確認"""
        data = {
            'intervals': [[0.1, 0.5], [0.7, 1.2]],
            'note_pitches': [440.0, 880.0]
        }
        
        intervals, pitches = extract_note_data(data)
        
        np.testing.assert_array_equal(intervals, np.array(data['intervals']))
        np.testing.assert_array_equal(pitches, np.array(data['note_pitches']))
        
    def test_extract_from_dict_alternative_keys(self):
        """代替キー'pitches'を持つ辞書からノートデータを抽出できることを確認"""
        data = {
            'intervals': [[0.1, 0.5], [0.7, 1.2]],
            'pitches': [440.0, 880.0]  # 'note_pitches'ではなく'pitches'
        }
        
        intervals, pitches = extract_note_data(data)
        
        np.testing.assert_array_equal(intervals, np.array(data['intervals']))
        np.testing.assert_array_equal(pitches, np.array(data['pitches']))
        
    def test_extract_from_detection_result(self):
        """DetectionResultオブジェクトからノートデータを抽出できることを確認"""
        result = DetectionResult(
            intervals=np.array([[0.1, 0.5], [0.7, 1.2]]),
            note_pitches=np.array([440.0, 880.0])
        )
        
        intervals, pitches = extract_note_data(result.to_dict())
        
        np.testing.assert_array_equal(intervals, result.intervals)
        np.testing.assert_array_equal(pitches, result.note_pitches)
        
    def test_extract_from_dict_missing_data(self, caplog):
        """ノートデータが不足している辞書から抽出を試みると警告が出力されることを確認"""
        caplog.set_level(logging.WARNING)
        
        data = {'some_other_key': 'value'}  # ノートデータが不足
        
        intervals, pitches = extract_note_data(data)
        
        # 警告が出力されていることを確認
        assert "ノートデータが見つかりません" in caplog.text
        
        # 空の配列が返されることを確認
        assert intervals.shape == (0, 2)
        assert pitches.shape == (0,)
        
    def test_extract_from_non_dict(self):
        """辞書以外のオブジェクトが渡された場合はそのまま返されることを確認"""
        data = "not a dict"
        
        result = extract_note_data(data)
        
        assert result == "not a dict"
        
    def test_extract_from_object_with_get_note_data(self):
        """get_note_dataメソッドを持つオブジェクトからノートデータを抽出できることを確認"""
        class MockObject:
            def get_note_data(self):
                return np.array([[0.1, 0.5]]), np.array([440.0])
                
        mock_obj = MockObject()
        
        intervals, pitches = extract_note_data(mock_obj)
        
        np.testing.assert_array_equal(intervals, np.array([[0.1, 0.5]]))
        np.testing.assert_array_equal(pitches, np.array([440.0])) 