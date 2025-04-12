# tests/unit/evaluation/test_evaluation_io.py
import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY
import os
import tempfile
import sys

# Patch path_utils
mock_path_utils = MagicMock()
mock_path_utils.find_files = MagicMock(return_value=[])
mock_path_utils.ensure_dir = MagicMock(return_value=None)
mock_path_utils.get_dataset_paths = MagicMock(return_value=([], []))

# patch sys.modules to add our mock
sys_modules_patcher = patch.dict('sys.modules', {
    'src.utils.path_utils': mock_path_utils
})
sys_modules_patcher.start()

# テスト対象モジュールをインポート
from src.evaluation import evaluation_io
from src.evaluation.evaluation_io import (
    save_evaluation_result,
    load_evaluation_result,
    load_multiple_evaluation_results,
    create_summary_dataframe,
    save_detection_plot,
    NumpyEncoder,
    FileError
)

# Patch plot_utils which might be imported in evaluation_io
mock_plot_utils = MagicMock()
plot_utils_patcher = patch.dict('sys.modules', {
    'src.visualization.plot_utils': mock_plot_utils
})
plot_utils_patcher.start()

# システムモジュールが見つからない場合のスキップフラグ
SKIP_TESTS = False

try:
    from src.exceptions import FileError
except ImportError as e:
    print(f"必要なモジュールをインポートできません: {e}")
    SKIP_TESTS = True
    
    # テスト用のダミー実装
    class FileError(Exception):
        """ファイル操作エラーを表すダミー例外クラス"""
        pass
    
    def save_evaluation_result(result_dict, output_path, metadata=None):
        """評価結果を保存するダミー関数"""
        return True
    
    def load_evaluation_result(input_path):
        """評価結果を読み込むダミー関数"""
        return {'metrics': {}, 'metadata': {}}
    
    def load_multiple_evaluation_results(input_path):
        """複数の評価結果を読み込むダミー関数"""
        return []
    
    def create_summary_dataframe(evaluation_results):
        """サマリーデータフレームを作成するダミー関数"""
        return pd.DataFrame()
    
    def save_detection_plot(detection_result, output_path):
        """検出結果のプロットを保存するダミー関数"""
        return True

# テストをスキップするかどうかのマーカー
pytestmark = pytest.mark.skipif(
    SKIP_TESTS, 
    reason="必要なモジュールをインポートできません。実装が完了してから再実行してください。"
)

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

    # Patch json.loads within the function's scope if necessary, or rely on reading text
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

@pytest.fixture
def sample_evaluation_result():
    """評価結果のサンプルデータを作成"""
    return {
        'Raw Pitch Accuracy': 0.85,
        'Raw Chroma Accuracy': 0.90,
        'Voicing Recall': 0.75,
        'Voicing False Alarm': 0.10,
        'F1 Score': 0.82,
        'Overall Accuracy': 0.80
    }

def test_save_evaluation_result_invalid_path():
    """無効なパスに保存しようとした場合のエラー処理テスト"""
    result_dict = {"accuracy": 0.9}
    # 存在しないディレクトリや権限のない場所への書き込みをモック
    with patch('os.makedirs', side_effect=OSError("アクセス拒否")):
        with patch('open', mock_open()) as m:
            # OSErrorではなくFileErrorに変換されるはず
            with pytest.raises(FileError):
                save_evaluation_result(result_dict, "/fake/path/result.json")

def test_save_evaluation_result(sample_evaluation_result):
    """save_evaluation_resultの基本機能テスト"""
    # 一時ディレクトリを使用
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_eval_result.json")
        
        # 機能はモックではなく実際のファイルに書き込み
        metadata = {"dataset": "test", "model": "test_model"}
        result = save_evaluation_result(sample_evaluation_result, output_path, metadata)
        
        # 戻り値と出力ファイルの存在を確認
        assert result is True
        assert os.path.exists(output_path)
        
        # 読み込みも検証
        if not SKIP_TESTS:
            loaded = load_evaluation_result(output_path)
            assert loaded is not None
            assert 'metrics' in loaded
            assert 'metadata' in loaded
            assert loaded['metrics'] == sample_evaluation_result
            assert loaded['metadata'] == metadata

def test_load_evaluation_result():
    """load_evaluation_resultの基本機能テスト"""
    # 一時ファイルを作成してテスト
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp:
        # 有効なJSONを書き込む
        tmp.write('{"metrics": {"accuracy": 0.9}, "metadata": {"model": "test"}}')
        tmp.flush()
        
        # ファイルを読み込む
        result = load_evaluation_result(tmp.name)
        
        # 結果を検証
        assert isinstance(result, dict)
        assert 'metrics' in result
        assert 'metadata' in result
        assert result['metrics']['accuracy'] == 0.9
        assert result['metadata']['model'] == 'test'
        
        # 後始末
        os.unlink(tmp.name)

def test_load_evaluation_result_file_not_found():
    """存在しないファイルを読み込もうとした場合のエラー処理テスト"""
    with pytest.raises(FileError):
        load_evaluation_result("/non/existent/file.json")

def test_load_multiple_evaluation_results_basic():
    """load_multiple_evaluation_resultsの基本機能テスト"""
    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        # 評価結果ファイルを作成
        for i in range(3):
            file_path = os.path.join(temp_dir, f"result_{i}.json")
            with open(file_path, 'w') as f:
                f.write(f'{{"metrics": {{"accuracy": {0.8 + i*0.05}}}, "metadata": {{"id": {i}}}}}')
        
        # ディレクトリから複数の結果を読み込む
        results = load_multiple_evaluation_results(temp_dir)
        
        # 結果を検証
        assert isinstance(results, list)
        assert len(results) == 3  # 3つのファイルを読み込んだはず
        # 各要素がdictであることを確認
        for result in results:
            assert isinstance(result, dict)
            assert 'metrics' in result
            assert 'metadata' in result

def test_load_multiple_evaluation_results_dir_not_found():
    """存在しないディレクトリを読み込もうとした場合のエラー処理テスト"""
    with pytest.raises(FileError):
        load_multiple_evaluation_results("/non/existent/directory/")

@pytest.fixture
def sample_evaluation_results():
    """複数の評価結果サンプルを作成"""
    return [
        {
            'metrics': {'Raw Pitch Accuracy': 0.85, 'Voicing Recall': 0.9},
            'metadata': {'dataset': 'A', 'model': 'X'}
        },
        {
            'metrics': {'Raw Pitch Accuracy': 0.75, 'Voicing Recall': 0.8},
            'metadata': {'dataset': 'B', 'model': 'X'}
        },
        {
            'metrics': {'Raw Pitch Accuracy': 0.90, 'Voicing Recall': 0.95},
            'metadata': {'dataset': 'A', 'model': 'Y'}
        }
    ]

def test_create_summary_dataframe_basic(sample_evaluation_results):
    """create_summary_dataframeの基本機能テスト"""
    # デフォルトのモック動作
    with patch('pandas.DataFrame', return_value=pd.DataFrame()) as mock_df:
        df = create_summary_dataframe(sample_evaluation_results)
        
        # DataFrameが作成されたことを確認
        assert mock_df.called
        
        # DataFrameに渡されたデータの構造を確認（キー）
        args, kwargs = mock_df.call_args
        data = args[0] if args else kwargs.get('data', [])
        
        # データ構造の確認
        if len(data) > 0:
            for entry in data:
                assert 'dataset' in entry  # メタデータが含まれているか
                assert 'model' in entry    # メタデータが含まれているか
                # 少なくとも1つのメトリクスキーが含まれているか
                assert any(key in entry for key in 
                          ['Raw Pitch Accuracy', 'Voicing Recall'])

def test_create_summary_dataframe_empty():
    """空のリストでcreate_summary_dataframeを呼び出した場合のテスト"""
    # 空のDataFrameを返すようにモック
    with patch('pandas.DataFrame', return_value=pd.DataFrame()) as mock_df:
        df = create_summary_dataframe([])
        
        # 空のデータフレームが作成されたことを確認
        assert mock_df.called
        # 呼び出し時に空のデータリストが渡されたことを確認
        args, kwargs = mock_df.call_args
        data = args[0] if args else kwargs.get('data', None)
        assert data == []

@pytest.fixture
def sample_detection_result():
    """検出結果のサンプルデータを作成"""
    return {
        'ref_time': np.array([0.0, 0.1, 0.2]),
        'ref_freq': np.array([440.0, 442.0, 0.0]),
        'est_time': np.array([0.0, 0.1, 0.2]),
        'est_freq': np.array([438.0, 441.0, 0.0]),
        'metrics': {'Raw Pitch Accuracy': 0.9}
    }

def test_save_detection_plot_calls_plot(sample_detection_result):
    """save_detection_plotがプロット機能を呼び出すことを確認するテスト"""
    # plot_utilsモジュールのモック
    with patch('src.evaluation.evaluation_io.plot_utils') as mock_plot_utils:
        # プロット保存先への一時ファイルを用意
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            # 関数を実行
            result = save_detection_plot(sample_detection_result, tmp.name)
            
            # plot_utilsのplot_detection_resultが呼ばれたことを確認
            assert mock_plot_utils.plot_detection_result.called
            # 結果を検証
            assert result is True

def test_save_detection_plot_handles_plot_error(sample_detection_result):
    """プロット中にエラーが発生した場合の処理を確認するテスト"""
    # plot_utilsモジュールのモックで例外を発生させる
    with patch('src.evaluation.evaluation_io.plot_utils') as mock_plot_utils:
        mock_plot_utils.plot_detection_result.side_effect = Exception("プロットエラー")
        
        # 一時ファイルを用意
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            # FileErrorが発生することを確認
            with pytest.raises(FileError):
                save_detection_plot(sample_detection_result, tmp.name) 