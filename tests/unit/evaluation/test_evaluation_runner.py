# tests/unit/evaluation/test_evaluation_runner.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, ANY, call, AsyncMock
import pandas as pd
from pathlib import Path

# テスト対象モジュールと依存モジュールをインポート
try:
    from src.evaluation import evaluation_runner
    from src.evaluation.evaluation_runner import (
        evaluate_detection_result,
        evaluate_detector,
        run_evaluation,
        _evaluate_file_for_detector, # Private function, consider if direct test needed
        _calculate_evaluation_summary # Private function, consider if direct test needed
    )
    from src.structures.note import Note
    from src.utils.detection_result import DetectionResult
    from src.utils.exception_utils import MiraiError, ConfigError
    from src.detectors.base_detector import BaseDetector # For typing/mocking
    import mir_eval # Mock target
except ImportError:
    pytest.skip("Skipping evaluation_runner tests due to missing src modules", allow_module_level=True)
    # Define dummy classes/functions if needed for static analysis, but tests will be skipped
    class MiraiError(Exception): pass
    class ConfigError(Exception): pass
    class BaseDetector: pass
    class Note: pass
    class DetectionResult: pass
    class pd: # Dummy pandas
        @staticmethod
        def DataFrame(*args, **kwargs): return MagicMock()
    class mir_eval: # Dummy mir_eval
        class transcription:
            @staticmethod
            def precision_recall_f1_overlap(*args, **kwargs): return (0.0, 0.0, 0.0, 0.0) # P, R, F1, Avg Overlap
        class melody:
            @staticmethod
            def evaluate(*args, **kwargs): return {'Voicing Recall': 0.0, 'Overall Accuracy': 0.0} # Dummy metrics


# --- Fixtures ---

pytestmark = pytest.mark.asyncio # Assuming some functions might become async later

@pytest.fixture
def mock_mir_eval():
    """Mocks the mir_eval library functions."""
    with patch('src.evaluation.evaluation_runner.mir_eval', autospec=True) as mock:
        # Configure default return values for mocked functions
        mock.transcription.precision_recall_f1_overlap.return_value = (0.8, 0.9, 0.85, 0.7) # P, R, F1, Overlap
        mock.melody.evaluate.return_value = {
            'Voicing Recall': 0.95,
            'Voicing False Alarm': 0.1,
            'Raw Pitch Accuracy': 0.88,
            'Raw Chroma Accuracy': 0.92,
            'Overall Accuracy': 0.85
        }
        yield mock

@pytest.fixture
def mock_evaluation_io():
    """Mocks functions from evaluation_io."""
    with patch('src.evaluation.evaluation_runner.evaluation_io', autospec=True) as mock:
        yield mock

@pytest.fixture
def mock_audio_utils():
    """Mocks functions from audio_utils."""
    with patch('src.evaluation.evaluation_runner.audio_utils', autospec=True) as mock:
        mock.load_audio_file.return_value = (np.zeros(1000), 44100) # Dummy audio, sr
        mock.load_reference_data.return_value = (np.array([0.1, 0.5]), np.array([0.5, 1.0]), np.array([60, 62])) # Dummy ref intervals, pitches
        yield mock

@pytest.fixture
def mock_detector_utils():
    """Mocks functions from detector_utils."""
    with patch('src.evaluation.evaluation_runner.detector_utils', autospec=True) as mock:
        # Dummy normalized result
        mock.normalize_detection_result.return_value = DetectionResult(
            intervals=np.array([[0.1, 0.48]]),
            note_pitches=np.array([60.5]),
            frame_times=np.linspace(0, 1, 100),
            frame_frequencies=np.full(100, 220.0)
        )
        yield mock

@pytest.fixture
def mock_detector_instance():
    """Provides a mocked detector instance."""
    mock = MagicMock(spec=BaseDetector)
    # Dummy detection result dictionary before normalization
    mock.detect.return_value = {
        'intervals': np.array([[0.1, 0.48]]),
        'note_pitches': np.array([60.5]),
        'frame_times': np.linspace(0, 1, 100),
        'frame_frequencies': np.full(100, 220.0)
    }
    # Add other necessary attributes/methods if BaseDetector defines them
    mock.name = "MockDetector"
    mock.version = "0.1.0"
    return mock

@pytest.fixture
def mock_path_utils():
    """Mocks functions from path_utils."""
    with patch('src.evaluation.evaluation_runner.path_utils', autospec=True) as mock:
        mock.ensure_dir.return_value = Path("/fake/output/dir")
        yield mock

@pytest.fixture
def mock_multiprocessing():
    """Mocks the multiprocessing module, simplifying parallel execution testing."""
    with patch('src.evaluation.evaluation_runner.multiprocessing', autospec=True) as mock_mp:
        mock_pool = MagicMock()
        # Simulate map returning results directly for simplicity
        mock_pool.map.side_effect = lambda func, iterable: [func(item) for item in iterable]
        mock_mp.Pool.return_value.__enter__.return_value = mock_pool
        yield mock_mp


# --- Test evaluate_detection_result ---

def test_evaluate_detection_result_basic(mock_mir_eval):
    """Tests basic call to evaluate_detection_result with mocked mir_eval."""
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

    result = evaluation_runner.evaluate_detection_result(
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches,
        ref_frame_times=frame_times, # Assume same times for simplicity
        ref_frame_frequencies=ref_freq,
        detection_result=detection_result
    )

    # Check if mir_eval functions were called with correct *estimated* data
    mock_mir_eval.transcription.precision_recall_f1_overlap.assert_called_once_with(
        ref_intervals, ref_pitches, est_intervals, est_pitches, # Correct args
        offset_ratio=None, pitch_tolerance=ANY, onset_tolerance=ANY, offset_min_tolerance=ANY
    )
    # We need estimated voicing for melody evaluation
    est_voicing_arg = est_freq > 0
    mock_mir_eval.melody.evaluate.assert_called_once_with(
        ref_time=frame_times, ref_freq=ref_freq, est_time=frame_times, est_freq=est_freq
    )

    # Check if expected keys are in the result (values are from mock)
    assert 'note.f_measure' in result
    assert 'frame.Overall_Accuracy' in result
    assert result['note.f_measure'] == 0.85 # From mock_mir_eval fixture
    assert result['frame.Overall_Accuracy'] == 0.85 # From mock_mir_eval fixture
    assert result['note.precision'] == 0.8
    assert result['note.recall'] == 0.9
    assert result['frame.Voicing_Recall'] == 0.95

def test_evaluate_detection_result_empty_estimation(mock_mir_eval):
    """Tests evaluate_detection_result with empty estimation."""
    # Similar setup as basic, but empty estimation
    ref_intervals = np.array([[0.1, 0.5], [0.6, 1.0]])
    ref_pitches = np.array([60, 62])
    frame_times = np.linspace(0, 1, 100)
    ref_freq = np.full(100, 220.0)
    ref_voicing = ref_freq > 0
    empty_est_freq = np.zeros(100) # No estimated frequency

    detection_result = DetectionResult(
        intervals=np.empty((0, 2)),
        note_pitches=np.empty(0),
        frame_times=frame_times,
        frame_frequencies=empty_est_freq
    )

    result = evaluation_runner.evaluate_detection_result(
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches,
        ref_frame_times=frame_times,
        ref_frame_frequencies=ref_freq,
        detection_result=detection_result
    )

    # Check calls with empty estimated data
    mock_mir_eval.transcription.precision_recall_f1_overlap.assert_called_once_with(
        ref_intervals, ref_pitches, np.empty((0, 2)), np.empty(0), # Empty arrays passed
        offset_ratio=ANY, pitch_tolerance=ANY, onset_tolerance=ANY, offset_min_tolerance=ANY
    )
    mock_mir_eval.melody.evaluate.assert_called_once_with(
        ref_time=frame_times, ref_freq=ref_freq, est_time=frame_times, est_freq=empty_est_freq
    )
    # Basic check: metrics should reflect performance based on mocks (mir_eval would return 0s usually)
    assert result['note.f_measure'] == 0.85 # Mock value returned
    assert result['frame.Overall_Accuracy'] == 0.85 # Mock value returned


# --- Test evaluate_detector ---

@pytest.mark.asyncio # If detector.detect is async
async def test_evaluate_detector_success(mock_audio_utils, mock_detector_utils, mock_detector_instance):
    """Tests the success path of evaluate_detector."""
    audio_path = Path("dummy/audio.wav")
    ref_path = Path("dummy/ref.csv")
    # Mock detector.detect to be async if needed
    if not isinstance(mock_detector_instance.detect, AsyncMock):
        mock_detector_instance.detect = AsyncMock(return_value=mock_detector_instance.detect.return_value)

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

    # Mock the evaluate_detection_result function itself for this test
    with patch('src.evaluation.evaluation_runner.evaluate_detection_result') as mock_evaluate_result:
        mock_evaluate_result.return_value = {"note.f_measure": 0.9, "frame.Overall_Accuracy": 0.8}

        result_dict = await evaluation_runner.evaluate_detector(
            detector=mock_detector_instance,
            audio_path=audio_path,
            reference_path=ref_path,
            output_dir=None # No plotting or saving details
        )

        # Verify mocks were called in expected order/with expected args
        mock_audio_utils.load_audio_file.assert_called_once_with(audio_path)
        mock_audio_utils.load_reference_data.assert_called_once_with(ref_path)
        # Assume detect takes audio data and sample rate
        mock_detector_instance.detect.assert_called_once_with(
            mock_audio_utils.load_audio_file.return_value[0], # audio_data
            mock_audio_utils.load_audio_file.return_value[1]  # sample_rate
        )
        mock_detector_utils.normalize_detection_result.assert_called_once_with(
            mock_detector_instance.detect.return_value, # Raw result from detect
            mock_audio_utils.load_reference_data.return_value[0] # Ref intervals for frame alignment
        )
        mock_evaluate_result.assert_called_once()
        # Check args passed to evaluate_detection_result more specifically
        call_args, call_kwargs = mock_evaluate_result.call_args
        np.testing.assert_array_equal(call_kwargs['ref_intervals'], dummy_ref_intervals)
        np.testing.assert_array_equal(call_kwargs['ref_pitches'], dummy_ref_pitches)
        assert call_kwargs['detection_result'] == dummy_norm_result
        # Need ref_frame data if frame eval is active
        # assert 'ref_frame_times' in call_kwargs
        # assert 'ref_frame_frequencies' in call_kwargs

        assert result_dict['valid'] is True
        assert result_dict['audio_path'] == str(audio_path)
        assert result_dict['detector_name'] == mock_detector_instance.name
        assert 'metrics' in result_dict
        assert result_dict['metrics']['note.f_measure'] == 0.9
        assert result_dict['metrics']['frame.Overall_Accuracy'] == 0.8
        assert 'error_message' not in result_dict

@pytest.mark.asyncio
async def test_evaluate_detector_load_audio_error(mock_audio_utils, mock_detector_instance):
    """Tests evaluate_detector when load_audio_file fails."""
    error_msg = "Cannot load audio"
    mock_audio_utils.load_audio_file.side_effect = MiraiError(error_msg)
    # Ensure detect is async if needed
    if not isinstance(mock_detector_instance.detect, AsyncMock):
        mock_detector_instance.detect = AsyncMock()

    audio_path = Path("dummy/audio_fail.wav")
    ref_path = Path("dummy/ref_fail.csv")

    result_dict = await evaluation_runner.evaluate_detector(
        detector=mock_detector_instance,
        audio_path=audio_path,
        reference_path=ref_path,
        output_dir=None
    )

    assert result_dict['valid'] is False
    assert error_msg in result_dict['error_message']
    assert result_dict['audio_path'] == str(audio_path)
    assert result_dict['detector_name'] == mock_detector_instance.name
    assert 'metrics' not in result_dict # No metrics on failure

    # Check that subsequent functions were not called
    mock_audio_utils.load_reference_data.assert_not_called()
    mock_detector_instance.detect.assert_not_called()

@pytest.mark.asyncio
async def test_evaluate_detector_detection_error(mock_audio_utils, mock_detector_instance, mock_detector_utils):
    """Tests evaluate_detector when detector.detect fails."""
    error_msg = "Detection failed internally"
    # Ensure detect is async and raises error
    mock_detector_instance.detect = AsyncMock(side_effect=MiraiError(error_msg))

    audio_path = Path("dummy/detect_fail.wav")
    ref_path = Path("dummy/detect_fail.csv")

    result_dict = await evaluation_runner.evaluate_detector(
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

    # We should still attempt to load data before detection
    mock_audio_utils.load_audio_file.assert_called_once_with(audio_path)
    mock_audio_utils.load_reference_data.assert_called_once_with(ref_path)
    mock_detector_instance.detect.assert_called_once() # It was called and raised error

    # Normalization and evaluation should not happen after detect error
    mock_detector_utils.normalize_detection_result.assert_not_called()
    # Need to patch evaluate_detection_result to check it wasn't called
    with patch('src.evaluation.evaluation_runner.evaluate_detection_result') as mock_evaluate_result:
        await evaluation_runner.evaluate_detector(detector=mock_detector_instance, audio_path=audio_path, reference_path=ref_path, output_dir=None) # Re-run to check mock
        mock_evaluate_result.assert_not_called() # Verify it's not called on error


# --- Test run_evaluation ---

# Note: Testing run_evaluation thoroughly requires more setup (mocking file systems, etc.)
# These are basic tests focusing on flow and mocking dependencies.

@pytest.mark.asyncio
async def test_run_evaluation_basic_flow(mock_detector_instance, tmp_path, mock_evaluation_io, mock_path_utils, mock_multiprocessing, mock_detector_utils):
    """Tests the basic flow of run_evaluation with mocked file lists and evaluation (num_procs=1)."""
    # Setup dummy dataset/paths
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_base_dir = tmp_path / "output"
    # Specific output dir for this detector
    expected_output_dir = output_base_dir / mock_detector_instance.name / "results"
    mock_path_utils.ensure_dir.return_value = expected_output_dir # Mock specific dir

    audio1 = dataset_dir / "track1.wav"
    ref1 = dataset_dir / "track1.csv"
    audio1.touch()
    ref1.touch()

    # Mock detector loading
    mock_detector_utils.get_detector_class.return_value = MagicMock() # Dummy class
    mock_detector_utils.create_detector.return_value = mock_detector_instance

    # Mock _evaluate_file_for_detector (which wraps evaluate_detector)
    dummy_result = {
        'valid': True,
        'audio_path': str(audio1),
        'detector_name': mock_detector_instance.name,
        'metrics': {'note.f_measure': 0.8}
    }
    with patch('src.evaluation.evaluation_runner._evaluate_file_for_detector', new_callable=AsyncMock) as mock_evaluate_file:
        mock_evaluate_file.return_value = dummy_result

        # Mock summary calculation
        with patch('src.evaluation.evaluation_runner._calculate_evaluation_summary') as mock_calculate_summary:
            mock_calculate_summary.return_value = {"summary": "data"}

            detector_params = {"class_name": mock_detector_instance.name, "params": {"param1": "value1"}}
            await evaluation_runner.run_evaluation(
                detector_params_list=[detector_params],
                dataset_dir=str(dataset_dir),
                output_base_dir=str(output_base_dir),
                num_procs=1, # Test single process first
                cross_validation_folds=None,
                file_limit=None,
                save_plots=False,
                save_individual_results=True # Need this to check saving
            )

            # Check detector loading was called
            mock_detector_utils.get_detector_class.assert_called_once_with(detector_params['class_name'])
            mock_detector_utils.create_detector.assert_called_once_with(ANY, detector_params['params'])

            # Check that ensure_dir was called for the specific output dir
            mock_path_utils.ensure_dir.assert_called_once_with(expected_output_dir)

            # Check that multiprocessing Pool was NOT used
            mock_multiprocessing.Pool.assert_not_called()

            # Check that evaluation was called for the file pair directly
            assert mock_evaluate_file.call_count == 1
            call_args, call_kwargs = mock_evaluate_file.call_args
            assert call_args[0] == mock_detector_instance
            assert call_args[1] == audio1
            assert call_args[2] == ref1
            assert call_args[3] == expected_output_dir # output dir passed correctly

            # Check that summary was called with the results list
            mock_calculate_summary.assert_called_once_with([dummy_result], mock_detector_instance.name, detector_params['params'])

            # Check that saving was called for individual and summary results
            assert mock_evaluation_io.save_evaluation_result.call_count == 2
            # Call 1: Individual result
            mock_evaluation_io.save_evaluation_result.assert_any_call(
                dummy_result,
                expected_output_dir / f"{audio1.stem}.json"
            )
            # Call 2: Summary result
            mock_evaluation_io.save_evaluation_result.assert_any_call(
                mock_calculate_summary.return_value,
                expected_output_dir / "summary.json"
            )


@pytest.mark.asyncio
async def test_run_evaluation_parallel(mock_detector_instance, tmp_path, mock_evaluation_io, mock_path_utils, mock_multiprocessing, mock_detector_utils):
    """Tests if multiprocessing Pool is used when num_procs > 1."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_base_dir = tmp_path / "output"
    expected_output_dir = output_base_dir / mock_detector_instance.name / "results"
    mock_path_utils.ensure_dir.return_value = expected_output_dir

    audio1 = dataset_dir / "track1.wav"
    ref1 = dataset_dir / "track1.csv"
    audio2 = dataset_dir / "track2.wav"
    ref2 = dataset_dir / "track2.csv"
    audio1.touch()
    ref1.touch()
    audio2.touch()
    ref2.touch()

    mock_detector_utils.get_detector_class.return_value = MagicMock()
    mock_detector_utils.create_detector.return_value = mock_detector_instance

    # Simulate _evaluate_file_for_detector returning different results for different files
    result1 = {'valid': True, 'audio_path': str(audio1), 'detector_name': mock_detector_instance.name, 'metrics': {'note.f_measure': 0.7}}
    result2 = {'valid': True, 'audio_path': str(audio2), 'detector_name': mock_detector_instance.name, 'metrics': {'note.f_measure': 0.9}}

    # The mocked Pool.map will call the passed function directly
    # We need to patch the *actual* function that Pool.map will call
    # Assume run_evaluation creates partials or uses a helper that wraps _evaluate_file_for_detector
    # Patching _evaluate_file_for_detector is simpler if it's directly callable or passed
    with patch('src.evaluation.evaluation_runner._evaluate_file_for_detector', new_callable=AsyncMock) as mock_evaluate_file:
        # Side effect to return different results based on input
        async def evaluate_side_effect(detector, audio_path, ref_path, output_dir):
            if audio_path == audio1:
                return result1
            elif audio_path == audio2:
                return result2
            return None # Should not happen in this test
        mock_evaluate_file.side_effect = evaluate_side_effect

        with patch('src.evaluation.evaluation_runner._calculate_evaluation_summary') as mock_calculate_summary:
            mock_calculate_summary.return_value = {"summary": "parallel data"}

            detector_params = {"class_name": mock_detector_instance.name, "params": {}}
            await evaluation_runner.run_evaluation(
                detector_params_list=[detector_params],
                dataset_dir=str(dataset_dir),
                output_base_dir=str(output_base_dir),
                num_procs=2, # Use parallel processing
                save_individual_results=False # Test without individual saving
            )

            # Check detector loading
            mock_detector_utils.get_detector_class.assert_called_once()
            mock_detector_utils.create_detector.assert_called_once()

            # Check dir creation
            mock_path_utils.ensure_dir.assert_called_once_with(expected_output_dir)

            # Check that multiprocessing Pool WAS used
            mock_multiprocessing.Pool.assert_called_once_with(processes=2)
            mock_pool = mock_multiprocessing.Pool.return_value.__enter__.return_value
            # Check Pool.map was called (or similar parallel execution method)
            assert mock_pool.map.call_count == 1

            # Verify the arguments passed to Pool.map's function implicitly contain the right data
            # Since our mock map calls the function directly, check the calls to _evaluate_file_for_detector
            assert mock_evaluate_file.call_count == 2
            mock_evaluate_file.assert_any_call(mock_detector_instance, audio1, ref1, expected_output_dir)
            mock_evaluate_file.assert_any_call(mock_detector_instance, audio2, ref2, expected_output_dir)

            # Check summary calculation with both results
            # Order might vary depending on parallel execution, use set/list comparison
            mock_calculate_summary.assert_called_once()
            call_args, call_kwargs = mock_calculate_summary.call_args
            assert set(r['audio_path'] for r in call_args[0]) == {str(audio1), str(audio2)}
            assert call_args[1] == mock_detector_instance.name
            assert call_args[2] == detector_params['params']

            # Check that only summary result was saved
            mock_evaluation_io.save_evaluation_result.assert_called_once_with(
                mock_calculate_summary.return_value,
                expected_output_dir / "summary.json"
            )


@pytest.mark.asyncio # run_evaluation is likely async
async def test_run_evaluation_missing_detector_class(tmp_path, mock_detector_utils):
    """Tests error handling when detector class cannot be loaded."""
    # Mock detector loading to raise ImportError
    error_msg = "Cannot find detector class UnknownDetector"
    mock_detector_utils.get_detector_class.side_effect = ImportError(error_msg)

    detector_params = {"class_name": "UnknownDetector"}
    with pytest.raises(ConfigError, match=f"Failed to load detector class {detector_params['class_name']}"):
        await evaluation_runner.run_evaluation(
            detector_params_list=[detector_params],
            dataset_dir=str(tmp_path), # Needs some dummy path
            output_base_dir=str(tmp_path / "out"),
            num_procs=1
        )

    # Ensure create_detector was not called
    mock_detector_utils.create_detector.assert_not_called()


# --- Test _calculate_evaluation_summary --- (Example)

def test_calculate_evaluation_summary_basic():
    """Tests basic summary calculation."""
    results_list = [
        {'valid': True, 'detector_name': 'DetA', 'audio_path': 'a1', 'metrics': {'note.f_measure': 0.8, 'other': 1}},
        {'valid': True, 'detector_name': 'DetA', 'audio_path': 'a2', 'metrics': {'note.f_measure': 0.6, 'other': 2}},
        {'valid': False, 'detector_name': 'DetA', 'audio_path': 'a3', 'error_message': 'Failed'},
    ]
    summary = _calculate_evaluation_summary(results_list, 'DetA', {}) # Pass empty detector params

    assert 'detector_name' in summary
    assert summary['detector_name'] == 'DetA'
    assert summary['num_files_total'] == 3
    assert summary['num_files_success'] == 2
    assert summary['num_files_failed'] == 1
    assert 'metrics_mean' in summary
    assert 'metrics_std' in summary
    assert 'failed_files' in summary
    assert np.isclose(summary['metrics_mean']['note.f_measure'], 0.7)
    assert np.isclose(summary['metrics_std']['note.f_measure'], 0.1)
    assert len(summary['failed_files']) == 1
    assert summary['failed_files'][0]['audio_path'] == 'a3'

def test_calculate_evaluation_summary_no_success():
    """Tests summary calculation with no successful results."""
    results_list = [
        {'valid': False, 'detector_name': 'DetA', 'audio_path': 'a1', 'error_message': 'Fail1'},
        {'valid': False, 'detector_name': 'DetA', 'audio_path': 'a2', 'error_message': 'Fail2'},
    ]
    summary = _calculate_evaluation_summary(results_list, 'DetA', {})

    assert summary['num_files_success'] == 0
    assert summary['num_files_failed'] == 2
    assert 'metrics_mean' not in summary # Or should be NaNs/empty dict
    assert 'metrics_std' not in summary
    assert len(summary['failed_files']) == 2

# Add more tests for edge cases, different evaluation metrics, etc. 