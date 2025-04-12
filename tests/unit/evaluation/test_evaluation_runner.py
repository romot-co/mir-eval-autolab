import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, ANY, call, Mock

# システムモジュールが見つからない場合のスキップフラグ
SKIP_TESTS = False

try:
    from src.evaluation.evaluation_runner import (
        EvaluationRunner,
        evaluate_detection_result,
        evaluate_detector,
        _calculate_evaluation_summary,
        run_evaluation
    )
    from src.utils.detection_result import DetectionResult
    from src.detectors.base_detector import BaseDetector
    from src.utils.exception_utils import MirexError, EvaluationError
    from src.structures.note import NoteData
except ImportError as e:
    import sys
    print(f"必要なモジュールをインポートできません: {e}")
    SKIP_TESTS = True
    
    # テスト用のダミー実装
    class NoteData:
        """音符データを表すダミークラス"""
        def __init__(self, start_time=0, end_time=0, midi_note=0):
            self.start_time = start_time
            self.end_time = end_time
            self.midi_note = midi_note
    
    class BaseDetector:
        """検出器の基本クラスのダミー実装"""
        def __init__(self, **kwargs):
            self.name = kwargs.get('name', 'DummyDetector')
            self.version = kwargs.get('version', '0.1.0')
            
        async def detect(self, audio_data, sample_rate):
            """ダミーの検出メソッド"""
            # Return a dummy detection result
            return {
                'intervals': np.array([[0.1, 0.5]]),
                'note_pitches': np.array([60]),
                'frame_times': np.linspace(0, 1, 100),
                'frame_frequencies': np.full(100, 440.0)
            }
    
    class EvaluationRunner:
        """評価ランナーのダミークラス"""
        def __init__(self, config=None):
            self.config = config or {}
            
        async def evaluate_audio_file(self, audio_path, detector, reference_path=None, output_dir=None):
            """オーディオファイル評価のダミーメソッド"""
            return {'valid': True, 'detector_name': detector.name, 'metrics': {'note.f_measure': 0.85}}
            
        async def evaluate_directory(self, audio_dir, detector, reference_dir=None, output_dir=None, parallel=False):
            """ディレクトリ評価のダミーメソッド"""
            return [{'valid': True, 'detector_name': detector.name, 'metrics': {'note.f_measure': 0.85}}]
    
    class EvaluationError(Exception):
        """評価エラーを表すダミー例外クラス"""
        pass
    
    class MirexError(Exception):
        """Mirexエラーを表すダミー例外クラス"""
        pass
    
    class DetectionResult:
        """検出結果を表すダミークラス"""
        def __init__(self, intervals=None, note_pitches=None, frame_times=None, frame_frequencies=None):
            self.intervals = intervals if intervals is not None else np.empty((0, 2))
            self.note_pitches = note_pitches if note_pitches is not None else np.empty(0)
            self.frame_times = frame_times if frame_times is not None else np.empty(0)
            self.frame_frequencies = frame_frequencies if frame_frequencies is not None else np.empty(0)
    
    def evaluate_reference_estimate_pair(reference_notes, estimated_notes, **kwargs):
        """リファレンスと推定値のペアを評価するダミー関数"""
        return {"note.f_measure": 0.85}
    
    def evaluate_datasets(reference_dir, results_dir, detectors, **kwargs):
        """データセットを評価するダミー関数"""
        return [{"note.f_measure": 0.85}]
    
    def compute_metrics_from_frames(ref_freq, est_freq, **kwargs):
        """フレームからメトリクスを計算するダミー関数"""
        return {"frame.Overall_Accuracy": 0.9}
    
    def evaluate_detection_result(detection_result, reference_intervals, reference_pitches, **kwargs):
        """検出結果を評価するダミー関数"""
        return {"note.f_measure": 0.85, "frame.Overall_Accuracy": 0.9}
    
    async def evaluate_detector(detector, audio_path, reference_path, output_dir=None):
        """検出器を評価するダミー関数"""
        return {'valid': True, 'detector_name': detector.name, 'metrics': {'note.f_measure': 0.85}}
    
    async def run_evaluation(detector_name, audio_dir, reference_dir, output_dir, **kwargs):
        """評価を実行するダミー関数"""
        return {'summary': {'note.f_measure': 0.85}, 'results': []}
    
    def _calculate_evaluation_summary(results):
        """評価サマリーを計算するダミー関数"""
        return {'note.f_measure (mean)': 0.85}

# テストをスキップするかどうかのマーカー
pytestmark = pytest.mark.skipif(
    SKIP_TESTS, 
    reason="必要なモジュールをインポートできません。実装が完了してから再実行してください。"
)

# mir_eval モジュールをモック
class mir_eval:
    """mir_evalモジュールのモック"""
    class transcription:
        @staticmethod
        def precision_recall_f1_overlap(*args, **kwargs): 
            """精度、リコール、F1を計算するモック関数"""
            return {
                'Precision': 0.9, 'Recall': 0.8, 'F-measure': 0.85,
                'Average_Overlap_Ratio': 0.7
            }
    
    class melody:
        @staticmethod
        def evaluate(*args, **kwargs): 
            """メロディー評価のモック関数"""
            return {
                'Voicing Recall': 0.9, 'Voicing False Alarm': 0.1,
                'Raw Pitch Accuracy': 0.85, 'Raw Chroma Accuracy': 0.9,
                'Overall Accuracy': 0.8
            }

@pytest.fixture
def mock_mir_eval():
    """mir_evalモジュールをモックする"""
    with patch('src.evaluation.evaluation_runner.mir_eval', new=mir_eval) as mock:
        yield mock

@pytest.fixture
def mock_evaluation_io():
    """evaluation_ioモジュールをモックする"""
    mock = MagicMock()
    # モックの関数を設定
    mock.save_evaluation_result = MagicMock(return_value=True)
    mock.load_evaluation_result = MagicMock(return_value={})
    mock.save_detection_plot = MagicMock(return_value=True)
    
    with patch('src.evaluation.evaluation_runner.evaluation_io', mock):
        yield mock

@pytest.fixture
def mock_audio_utils():
    """audio_utilsモジュールをモックする"""
    mock = MagicMock()
    # モックの関数を設定
    mock.load_audio_file = MagicMock(return_value=(np.zeros(1000), 44100))
    mock.load_reference_data = MagicMock(return_value=(np.array([[0.1, 0.5], [0.6, 1.0]]), np.array([60, 62])))
    
    with patch('src.evaluation.evaluation_runner.audio_utils', mock):
        yield mock

@pytest.fixture
def mock_detector_utils():
    """detector_utilsモジュールをモックする"""
    mock = MagicMock()
    # モックの関数を設定
    # Dummy normalized result
    mock.normalize_detection_result = MagicMock(return_value=DetectionResult(
        intervals=np.array([[0.1, 0.48]]),
        note_pitches=np.array([60.5]),
        frame_times=np.linspace(0, 1, 100),
        frame_frequencies=np.full(100, 220.0)
    ))
    mock.create_detector = MagicMock(return_value=MagicMock(spec=BaseDetector))
    
    with patch('src.evaluation.evaluation_runner.detector_utils', mock):
        yield mock

@pytest.fixture
def mock_detector_instance():
    """検出器インスタンスのモックを提供する"""
    mock = MagicMock(spec=BaseDetector)
    # Dummy detection result dictionary before normalization
    mock.detect = AsyncMock(return_value={
        'intervals': np.array([[0.1, 0.48]]),
        'note_pitches': np.array([60.5]),
        'frame_times': np.linspace(0, 1, 100),
        'frame_frequencies': np.full(100, 220.0)
    })
    # Add other necessary attributes/methods if BaseDetector defines them
    mock.name = "MockDetector"
    mock.version = "0.1.0"
    return mock

@pytest.fixture
def mock_path_utils_fixture():
    """path_utilsモジュールをモックする"""
    mock = MagicMock()
    mock.ensure_dir = MagicMock(return_value=Path("/fake/output/dir"))
    mock.find_files = MagicMock(return_value=[])
    mock.get_dataset_paths = MagicMock(return_value=([], []))
    
    with patch('src.evaluation.evaluation_runner.path_utils', mock):
        yield mock

@pytest.fixture
def mock_multiprocessing():
    """multiprocessingモジュールをモックする"""
    mock_mp = MagicMock()
    mock_pool = MagicMock()
    # Simulate map returning results directly for simplicity
    mock_pool.map.side_effect = lambda func, iterable: [func(item) for item in iterable]
    mock_mp.Pool.return_value.__enter__.return_value = mock_pool
    
    with patch('src.evaluation.evaluation_runner.multiprocessing', mock_mp):
        yield mock_mp


# --- Test evaluate_detection_result ---

def test_evaluate_detection_result_basic(mock_mir_eval):
    """検出結果評価の基本的なテスト"""
    ref_intervals = np.array([[0.1, 0.5], [0.6, 1.0]])
    ref_pitches = np.array([60, 62])
    est_intervals = np.array([[0.12, 0.48], [0.65, 0.95]])
    est_pitches = np.array([60, 63]) # One correct, one wrong pitch
    frame_times = np.linspace(0, 1, 100)
    ref_freq = np.full(100, 220.0) # 440Hz = A4 = MIDI 69
    ref_voicing = ref_freq > 0
    est_freq = np.full(100, 225.0) # Slightly off
    est_voicing = est_freq > 0

    detection_result = DetectionResult(
        intervals=est_intervals,
        note_pitches=est_pitches,
        frame_times=frame_times,
        frame_frequencies=est_freq
    )

    result = evaluate_detection_result(
        detection_result=detection_result,
        reference_intervals=ref_intervals,
        reference_pitches=ref_pitches
    )

    # Check if mir_eval functions were called with correct *estimated* data
    mock_mir_eval.transcription.precision_recall_f1_overlap.assert_called_once_with(
        ref_intervals, ref_pitches, detection_result.intervals, detection_result.note_pitches,
        offset_ratio=ANY, pitch_tolerance=ANY, onset_tolerance=ANY, offset_min_tolerance=ANY
    )

    # Check if expected keys are in the result (values are from mock)
    assert 'note.f_measure' in result
    assert result['note.f_measure'] == 0.85 # From mock_mir_eval fixture


# --- Test evaluate_detector ---

@pytest.mark.asyncio
async def test_evaluate_detector_success(mock_audio_utils, mock_detector_utils, mock_detector_instance):
    """evaluate_detectorの成功パスのテスト"""
    audio_path = Path("dummy/audio.wav")
    ref_path = Path("dummy/ref.csv")

    # Define dummy ref data returned by mock_audio_utils.load_reference_data
    dummy_ref_intervals = np.array([[0.1, 0.5]])
    dummy_ref_pitches = np.array([60])
    mock_audio_utils.load_reference_data.return_value = (dummy_ref_intervals, dummy_ref_pitches)

    # Define dummy normalized detection result returned by mock_detector_utils
    dummy_norm_result = DetectionResult(
        intervals=np.array([[0.12, 0.48]]), note_pitches=np.array([60]),
        frame_times=np.linspace(0, 1, 100), frame_frequencies=np.full(100, 220.0)
    )
    mock_detector_utils.normalize_detection_result.return_value = dummy_norm_result

    # evaluate_detection_result関数をモック
    with patch('src.evaluation.evaluation_runner.evaluate_detection_result') as mock_evaluate_result:
        mock_evaluate_result.return_value = {"note.f_measure": 0.9, "frame.Overall_Accuracy": 0.8}

        result_dict = await evaluate_detector(
            detector=mock_detector_instance,
            audio_path=audio_path,
            reference_path=ref_path,
            output_dir=None # No plotting or saving details
        )

        # モックが期待通りに呼ばれたことを確認
        mock_audio_utils.load_audio_file.assert_called_once_with(audio_path)
        mock_audio_utils.load_reference_data.assert_called_once_with(ref_path)
        # detectがオーディオデータとサンプルレートを取る
        mock_detector_instance.detect.assert_called_once_with(
            mock_audio_utils.load_audio_file.return_value[0], # audio_data
            mock_audio_utils.load_audio_file.return_value[1]  # sample_rate
        )
        mock_detector_utils.normalize_detection_result.assert_called_once_with(
            mock_detector_instance.detect.return_value, # 検出後の生の結果
            mock_audio_utils.load_reference_data.return_value[0] # フレーム整列のための参照区間
        )
        mock_evaluate_result.assert_called_once()
        # evaluate_detection_resultに渡された引数を確認
        call_args, call_kwargs = mock_evaluate_result.call_args
        assert call_kwargs['detection_result'] == dummy_norm_result
        np.testing.assert_array_equal(call_kwargs['reference_intervals'], dummy_ref_intervals)
        np.testing.assert_array_equal(call_kwargs['reference_pitches'], dummy_ref_pitches)

        assert result_dict['valid'] is True
        assert result_dict['audio_path'] == str(audio_path)
        assert result_dict['detector_name'] == mock_detector_instance.name
        assert 'metrics' in result_dict
        assert result_dict['metrics']['note.f_measure'] == 0.9
        assert result_dict['metrics']['frame.Overall_Accuracy'] == 0.8
        assert 'error_message' not in result_dict

@pytest.mark.asyncio
async def test_evaluate_detector_load_audio_error(mock_audio_utils, mock_detector_instance):
    """オーディオロード失敗時のevaluate_detectorのテスト"""
    error_msg = "Cannot load audio"
    mock_audio_utils.load_audio_file.side_effect = MirexError(error_msg)

    audio_path = Path("dummy/audio_fail.wav")
    ref_path = Path("dummy/ref_fail.csv")

    result_dict = await evaluate_detector(
        detector=mock_detector_instance,
        audio_path=audio_path,
        reference_path=ref_path,
        output_dir=None
    )

    assert result_dict['valid'] is False
    assert error_msg in result_dict['error_message']
    assert result_dict['audio_path'] == str(audio_path)
    assert result_dict['detector_name'] == mock_detector_instance.name
    assert 'metrics' not in result_dict # 失敗時はメトリクスなし

    # 後続の関数が呼ばれていないことを確認
    mock_audio_utils.load_reference_data.assert_not_called()
    mock_detector_instance.detect.assert_not_called()

@pytest.mark.asyncio
async def test_evaluate_detector_detection_error(mock_audio_utils, mock_detector_instance, mock_detector_utils):
    """detector.detect失敗時のevaluate_detectorのテスト"""
    error_msg = "Detection failed!"
    mock_detector_instance.detect.side_effect = MirexError(error_msg)

    audio_path = Path("dummy/audio.wav")
    ref_path = Path("dummy/ref.csv")

    result_dict = await evaluate_detector(
        detector=mock_detector_instance,
        audio_path=audio_path,
        reference_path=ref_path,
        output_dir=None
    )

    assert result_dict['valid'] is False
    assert error_msg in result_dict['error_message']
    assert result_dict['audio_path'] == str(audio_path)
    assert result_dict['detector_name'] == mock_detector_instance.name
    assert 'metrics' not in result_dict

    # 適切なモックが呼ばれたことを確認
    mock_audio_utils.load_audio_file.assert_called_once()
    mock_audio_utils.load_reference_data.assert_called_once()
    mock_detector_instance.detect.assert_called_once()
    # 正規化とその後の処理は呼ばれていないはず
    mock_detector_utils.normalize_detection_result.assert_not_called()

@pytest.mark.asyncio
async def test_run_evaluation_basic_flow(mock_detector_instance, tmp_path, mock_evaluation_io, 
                                        mock_path_utils_fixture, mock_multiprocessing, mock_detector_utils):
    """run_evaluationの基本的なフローのテスト"""
    # モックの準備
    mock_detector_utils.create_detector.return_value = mock_detector_instance
    
    audio_paths = [Path("dummy/audio1.wav"), Path("dummy/audio2.wav")]
    ref_paths = [Path("dummy/ref1.csv"), Path("dummy/ref2.csv")]
    mock_path_utils_fixture.find_files.return_value = audio_paths
    mock_path_utils_fixture.get_dataset_paths.return_value = (audio_paths, ref_paths)
    
    # evaluate_detectorの結果をモック
    eval_results = [
        {'valid': True, 'audio_path': str(audio_paths[0]), 'detector_name': 'TestDet', 'metrics': {'note.f_measure': 0.9}},
        {'valid': True, 'audio_path': str(audio_paths[1]), 'detector_name': 'TestDet', 'metrics': {'note.f_measure': 0.8}}
    ]
    
    # evaluate_detector関数をモック
    with patch('src.evaluation.evaluation_runner.evaluate_detector', new_callable=AsyncMock) as mock_eval_detector:
        mock_eval_detector.side_effect = eval_results
        
        # _calculate_evaluation_summary関数をモック
        with patch('src.evaluation.evaluation_runner._calculate_evaluation_summary') as mock_calc_summary:
            summary = {
                'note.f_measure (mean)': 0.85,
                'note.f_measure (std)': 0.05,
                'files_count': 2,
                'success_rate': 1.0
            }
            mock_calc_summary.return_value = summary
            
            # 関数を実行
            result = await run_evaluation(
                detector_name='TestDet',
                audio_dir=str(tmp_path / "audio"),
                reference_dir=str(tmp_path / "ref"),
                output_dir=str(tmp_path / "output"),
                config={'param1': 'value1'},
                parallel=False
            )
            
            # 結果の検証
            assert result['summary'] == summary
            assert result['results'] == eval_results
            assert result['success'] is True
            assert 'detector_params' in result
            assert result['detector_params']['param1'] == 'value1'
            
            # 適切な関数が呼ばれたことを確認
            mock_detector_utils.create_detector.assert_called_once_with('TestDet', param1='value1')
            mock_path_utils_fixture.get_dataset_paths.assert_called_once()
            assert mock_eval_detector.call_count == 2  # 2つのオーディオファイル
            mock_calc_summary.assert_called_once_with(eval_results)
            mock_evaluation_io.save_evaluation_result.assert_called()

@pytest.mark.asyncio
async def test_run_evaluation_parallel(mock_detector_instance, tmp_path, mock_evaluation_io, 
                                     mock_path_utils_fixture, mock_multiprocessing, mock_detector_utils):
    """並列処理でのrun_evaluationのテスト"""
    # モックの準備
    mock_detector_utils.create_detector.return_value = mock_detector_instance
    
    audio_paths = [Path("dummy/audio1.wav"), Path("dummy/audio2.wav"), Path("dummy/audio3.wav")]
    ref_paths = [Path("dummy/ref1.csv"), Path("dummy/ref2.csv"), Path("dummy/ref3.csv")]
    mock_path_utils_fixture.get_dataset_paths.return_value = (audio_paths, ref_paths)
    
    # evaluate_detectorの結果をモック
    async def evaluate_side_effect(detector, audio_path, ref_path, output_dir):
        # オーディオパスに基づいて異なる結果を生成
        if "audio1" in str(audio_path):
            return {'valid': True, 'audio_path': str(audio_path), 'detector_name': 'TestDet', 'metrics': {'note.f_measure': 0.9}}
        elif "audio2" in str(audio_path):
            return {'valid': True, 'audio_path': str(audio_path), 'detector_name': 'TestDet', 'metrics': {'note.f_measure': 0.8}}
        else:
            return {'valid': False, 'audio_path': str(audio_path), 'detector_name': 'TestDet', 'error_message': 'Failed'}
            
    # evaluate_detector関数をモック
    with patch('src.evaluation.evaluation_runner.evaluate_detector', new_callable=AsyncMock) as mock_eval_detector:
        mock_eval_detector.side_effect = evaluate_side_effect
        
        # _calculate_evaluation_summary関数をモック
        with patch('src.evaluation.evaluation_runner._calculate_evaluation_summary') as mock_calc_summary:
            summary = {'note.f_measure (mean)': 0.85, 'success_rate': 0.67}
            mock_calc_summary.return_value = summary
            
            # 関数を実行
            result = await run_evaluation(
                detector_name='TestDet',
                audio_dir=str(tmp_path / "audio"),
                reference_dir=str(tmp_path / "ref"),
                output_dir=str(tmp_path / "output"),
                parallel=True
            )
            
            # 結果の検証
            assert result['summary'] == summary
            assert len(result['results']) == 3
            assert result['success'] is True
            
            # 並列処理が使用されたことを確認
            mock_multiprocessing.Pool.assert_called_once()
            mock_multiprocessing.Pool.return_value.__enter__.assert_called_once()
            pool_instance = mock_multiprocessing.Pool.return_value.__enter__.return_value
            assert pool_instance.map.called
            
            # Calculate summaryが呼ばれたことを確認
            mock_calc_summary.assert_called_once()

@pytest.mark.asyncio
async def test_run_evaluation_missing_detector_class(tmp_path, mock_detector_utils):
    """検出器クラスが見つからない場合のrun_evaluationのテスト"""
    # 検出器作成時にエラーが発生するようにモック
    mock_detector_utils.create_detector.side_effect = ImportError("Detector class not found")
    
    # 関数を実行
    result = await run_evaluation(
        detector_name='MissingDetector',
        audio_dir=str(tmp_path / "audio"),
        reference_dir=str(tmp_path / "ref"),
        output_dir=str(tmp_path / "output")
    )
    
    # 結果の検証
    assert result['success'] is False
    assert "Detector class not found" in result['error_message']
    assert result['results'] == []
    assert 'summary' not in result

def test_calculate_evaluation_summary_basic():
    """基本的なサマリー計算のテスト"""
    results_list = [
        {'valid': True, 'detector_name': 'DetA', 'audio_path': 'a1', 'metrics': {'note.f_measure': 0.8, 'other': 1}},
        {'valid': True, 'detector_name': 'DetA', 'audio_path': 'a2', 'metrics': {'note.f_measure': 0.6, 'other': 2}},
        {'valid': False, 'detector_name': 'DetA', 'audio_path': 'a3', 'error_message': 'Failed'},
    ]
    
    summary = _calculate_evaluation_summary(results_list)
    
    # 期待される要素を確認
    assert 'note.f_measure (mean)' in summary
    assert summary['note.f_measure (mean)'] == 0.7  # (0.8 + 0.6) / 2
    assert 'note.f_measure (std)' in summary
    assert summary['files_count'] == 3
    assert summary['success_count'] == 2
    assert summary['success_rate'] == 2/3

def test_calculate_evaluation_summary_no_success():
    """成功結果がない場合のサマリー計算のテスト"""
    results_list = [
        {'valid': False, 'detector_name': 'DetA', 'audio_path': 'a1', 'error_message': 'Fail1'},
        {'valid': False, 'detector_name': 'DetA', 'audio_path': 'a2', 'error_message': 'Fail2'},
    ]
    
    summary = _calculate_evaluation_summary(results_list)
    
    # 期待される要素を確認
    assert 'files_count' in summary
    assert summary['files_count'] == 2
    assert summary['success_count'] == 0
    assert summary['success_rate'] == 0
    # メトリクスの平均や標準偏差は含まれないはず
    assert 'note.f_measure (mean)' not in summary 