# tests/unit/evaluation/test_evaluation_io.py
import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

# テスト対象モジュールをインポート
try:
    from src.evaluation import evaluation_io
    from src.evaluation.evaluation_io import (
        save_evaluation_result,
        load_evaluation_result,
        load_multiple_evaluation_results,
        create_summary_dataframe,
        save_detection_plot
    )
    # Assuming NumpyEncoder is defined here or in json_utils
    from src.utils.json_utils import NumpyEncoder
    # Need DetectionResult if save_detection_plot uses it
    from src.utils.detection_result import DetectionResult
except ImportError:
    pytest.skip("Skipping evaluation_io tests due to missing src modules", allow_module_level=True)
    # Dummy classes/functions
    class NumpyEncoder(json.JSONEncoder): pass
    class DetectionResult: pass
    class pd:
        @staticmethod
        def DataFrame(*args, **kwargs): return MagicMock()

# --- Fixtures ---

@pytest.fixture
def mock_open():
    """ Mocks the built-in open function for file I/O checks. """
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
         mock_file_handle = MagicMock()
         mock_open.return_value.__enter__.return_value = mock_file_handle
         yield mock_open, mock_file_handle

@pytest.fixture
def mock_json():
    """Mocks the json module."""
    with patch('src.evaluation.evaluation_io.json', autospec=True) as mock:
        yield mock

@pytest.fixture
def mock_pathlib():
    """Mocks relevant pathlib.Path methods."""
    # Wrap multiple patch context managers in parentheses
    with (
        patch('pathlib.Path.glob') as mock_glob,
        patch('pathlib.Path.exists') as mock_exists,
        patch('pathlib.Path.is_file') as mock_is_file
    ):
        yield {"glob": mock_glob, "exists": mock_exists, "is_file": mock_is_file}

@pytest.fixture
def mock_matplotlib():
    """Mocks matplotlib pyplot."""
    # Try multiple common import paths for pyplot
    try:
        with patch('src.evaluation.evaluation_io.plt', autospec=True) as mock_plt:
            yield mock_plt
    except (ImportError, AttributeError):
         try:
             with patch('matplotlib.pyplot', autospec=True) as mock_plt:
                 yield mock_plt
         except (ImportError, AttributeError):
             # Create a dummy mock if matplotlib is not installed or patch target is wrong
             yield MagicMock()


# --- Test save/load_evaluation_result ---

def test_save_evaluation_result(mock_open, mock_json):
    """Tests saving an evaluation result dictionary to JSON."""
    mock_open_func, mock_file_handle = mock_open
    result_dict = {
        'detector_name': 'TestDet',
        'metrics': {'f_measure': 0.85, 'precision': np.float32(0.8)},
        'params': {'threshold': 0.5}
    }
    output_path = Path("/fake/output/result.json")

    save_evaluation_result(result_dict, output_path)

    # Check that open was called correctly
    mock_open_func.assert_called_once_with(output_path, 'w')
    # Check that json.dump was called with the correct arguments
    mock_json.dump.assert_called_once_with(
        result_dict,
        mock_file_handle, # The file handle from mock_open
        cls=NumpyEncoder, # Check if the custom encoder is used
        indent=4
    )

def test_load_evaluation_result(mock_open, mock_json):
    """Tests loading an evaluation result dictionary from JSON."""
    mock_open_func, mock_file_handle = mock_open
    input_path = Path("/fake/input/result.json")
    expected_dict = {"detector_name": "LoadedDet", "metrics": {"f_measure": 0.7}}

    # Configure mock_json.load to return the expected dictionary
    mock_json.load.return_value = expected_dict

    loaded_dict = load_evaluation_result(input_path)

    # Check open and json.load calls
    mock_open_func.assert_called_once_with(input_path, 'r')
    mock_json.load.assert_called_once_with(mock_file_handle)
    assert loaded_dict == expected_dict

def test_load_evaluation_result_file_not_found(mock_open):
    """Tests loading when the JSON file does not exist."""
    mock_open_func, _ = mock_open
    mock_open_func.side_effect = FileNotFoundError
    input_path = Path("/fake/nonexistent.json")

    with pytest.raises(FileNotFoundError):
        load_evaluation_result(input_path)

# --- Test load_multiple_evaluation_results ---

def test_load_multiple_evaluation_results(mock_pathlib, mock_open):
    """Tests loading multiple JSON result files from a directory."""
    mock_open_func, mock_file_handle = mock_open
    results_dir = Path("/fake/results")
    file1_path = results_dir / "result1.json"
    file2_path = results_dir / "result2.json"
    result1_data = {"id": 1, "metric": 0.8}
    result2_data = {"id": 2, "metric": 0.9}

    # Configure mocks
    mock_pathlib["exists"].return_value = True
    mock_pathlib["glob"].return_value = [file1_path, file2_path]
    # Simulate reading different files
    def open_side_effect(path, mode):
        mock_fh = MagicMock()
        if path == file1_path:
            mock_fh.read.return_value = json.dumps(result1_data)
        elif path == file2_path:
            mock_fh.read.return_value = json.dumps(result2_data)
        else:
            raise FileNotFoundError
        # Need to return the context manager interface
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_fh
        mock_cm.__exit__.return_value = None
        return mock_cm
    mock_open_func.side_effect = open_side_effect

    # Patch json.load within the function's scope if necessary, or rely on reading text
    # For simplicity, assume load_multiple uses open().read() and json.loads()
    with patch('src.evaluation.evaluation_io.json.loads') as mock_json_loads:
         mock_json_loads.side_effect = [result1_data, result2_data]

         loaded_results = load_multiple_evaluation_results(results_dir)

    # Assertions
    mock_pathlib["exists"].assert_called_once_with()
    mock_pathlib["glob"].assert_called_once_with("*.json")
    assert mock_open_func.call_count == 2
    mock_open_func.assert_has_calls([call(file1_path, 'r'), call(file2_path, 'r')], any_order=True)
    assert mock_json_loads.call_count == 2
    assert loaded_results == [result1_data, result2_data]

def test_load_multiple_evaluation_results_dir_not_found(mock_pathlib):
    """Tests loading from a directory that does not exist."""
    mock_pathlib["exists"].return_value = False
    results_dir = Path("/fake/nonexistent_dir")

    loaded_results = load_multiple_evaluation_results(results_dir)

    mock_pathlib["exists"].assert_called_once_with()
    mock_pathlib["glob"].assert_not_called()
    assert loaded_results == []

# --- Test create_summary_dataframe ---

def test_create_summary_dataframe_basic():
    """Tests creating a basic summary DataFrame from a list of result dicts."""
    results_list = [
        {'detector_name': 'A', 'audio_path': 'f1', 'params': {'p1': 1}, 'metrics': {'f1': 0.8, 'p': 0.7}},
        {'detector_name': 'A', 'audio_path': 'f2', 'params': {'p1': 1}, 'metrics': {'f1': 0.6, 'p': 0.5}},
        {'detector_name': 'B', 'audio_path': 'f3', 'params': {'p2': 2}, 'metrics': {'f1': 0.9, 'p': 0.9}},
    ]
    # Mock pandas DataFrame to verify its creation
    with patch('src.evaluation.evaluation_io.pd.DataFrame') as mock_pd_dataframe:
        mock_df_instance = MagicMock()
        mock_pd_dataframe.return_value = mock_df_instance

        df = create_summary_dataframe(results_list)

        # Check that DataFrame was called
        mock_pd_dataframe.assert_called_once()
        # Check the data passed to DataFrame constructor (might need more complex checks)
        call_args, call_kwargs = mock_pd_dataframe.call_args
        assert len(call_args[0]) == len(results_list) # Check number of rows
        # Check if columns look reasonable (e.g., flattened keys)
        assert 'detector_name' in call_args[0][0]
        assert 'audio_path' in call_args[0][0]
        assert 'params.p1' in call_args[0][0]
        assert 'metrics.f1' in call_args[0][0]

        assert df == mock_df_instance # Returned the mocked DataFrame

def test_create_summary_dataframe_empty():
    """Tests creating a summary DataFrame with an empty input list."""
    with patch('src.evaluation.evaluation_io.pd.DataFrame') as mock_pd_dataframe:
        df = create_summary_dataframe([])
        # Should likely call pd.DataFrame with an empty list or similar
        mock_pd_dataframe.assert_called_once_with([])

# --- Test save_detection_plot ---

@patch('src.evaluation.evaluation_io.plot_utils.plot_detection_result') # Assuming plot func is in plot_utils
def test_save_detection_plot_calls_plot(mock_plot_detection, mock_matplotlib, tmp_path):
    """Tests that save_detection_plot calls the underlying plotting function."""
    output_path = tmp_path / "plot.png"
    # Create dummy data/results needed by plot_detection_result
    ref_intervals = np.array([[0.1, 0.5]])
    ref_pitches = np.array([60])
    detection_result = DetectionResult(intervals=np.array([[0.1, 0.4]]), note_pitches=np.array([61]))
    sample_rate = 16000
    audio_data = np.zeros(sample_rate)

    save_detection_plot(
        output_path=output_path,
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches,
        detection_result=detection_result,
        sample_rate=sample_rate,
        audio_data=audio_data
    )

    # Check that the plotting function was called with correct args
    mock_plot_detection.assert_called_once_with(
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches,
        detection_result=detection_result,
        sample_rate=sample_rate,
        audio_data=audio_data,
        ax=ANY # Check that an axes object was passed
    )

    # Check that matplotlib savefig and close were called
    mock_matplotlib.savefig.assert_called_once_with(output_path)
    mock_matplotlib.close.assert_called_once()

@patch('src.evaluation.evaluation_io.plot_utils.plot_detection_result')
def test_save_detection_plot_handles_plot_error(mock_plot_detection, mock_matplotlib, tmp_path):
    """Tests error handling if the plotting function fails."""
    mock_plot_detection.side_effect = ValueError("Plotting failed")
    output_path = tmp_path / "plot_error.png"

    # Expecting the error to be caught and logged (check log or ensure no crash)
    try:
        save_detection_plot(
            output_path=output_path,
            ref_intervals=np.array([]), ref_pitches=np.array([]),
            detection_result=DetectionResult(intervals=np.array([]), note_pitches=np.array([])),
            sample_rate=16000, audio_data=None
        )
        # Check logs if logging is implemented
    except Exception as e:
        pytest.fail(f"save_detection_plot should handle plotting errors gracefully, but raised {e}")

    # Ensure savefig/close are not called if plotting failed
    mock_matplotlib.savefig.assert_not_called()
    # Close might still be called in a finally block, depends on implementation
    # mock_matplotlib.close.assert_called_once() # Or assert_not_called() 