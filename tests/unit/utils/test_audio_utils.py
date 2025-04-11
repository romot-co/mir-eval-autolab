import pytest
from unittest.mock import patch, mock_open, MagicMock
import numpy as np
import soundfile as sf # soundfileはモックされるが、インポート自体は必要
import mir_eval.io # 同上

from src.utils.audio_utils import load_audio_file, load_reference_data

# --- load_audio_file Tests ---

# soundfile.read をモックするためのパッチ
@patch('soundfile.read')
def test_load_audio_file_success(mock_sf_read):
    """音声ファイルの読み込み成功ケース"""
    dummy_sr = 44100
    dummy_audio_data = np.array([0.1, 0.2, 0.3])
    mock_sf_read.return_value = (dummy_audio_data, dummy_sr)
    
    file_path = "tests/data/audio/dummy_audio.wav" # 存在はするが中身はモックされる
    
    audio_data, sample_rate = load_audio_file(file_path)
    
    mock_sf_read.assert_called_once_with(file_path)
    np.testing.assert_array_equal(audio_data, dummy_audio_data)
    assert sample_rate == dummy_sr

# soundfile.read が例外を発生させるケース
@patch('soundfile.read', side_effect=sf.SoundFileError("Mocked SoundFileError"))
def test_load_audio_file_soundfile_error(mock_sf_read):
    """音声ファイルの読み込み時にsoundfileエラーが発生するケース"""
    file_path = "tests/data/audio/dummy_audio.wav"
    
    with pytest.raises(sf.SoundFileError):
        load_audio_file(file_path)
    
    mock_sf_read.assert_called_once_with(file_path)

# ファイルが存在しないケース (FileNotFoundError)
@patch('soundfile.read', side_effect=FileNotFoundError("Mocked FileNotFoundError"))
def test_load_audio_file_not_found(mock_sf_read):
    """音声ファイルが存在しないケース"""
    file_path = "/non/existent/file.wav"
    
    with pytest.raises(FileNotFoundError):
        load_audio_file(file_path)
        
    mock_sf_read.assert_called_once_with(file_path)


# --- load_reference_data Tests ---

# mir_eval.io.load_valued_intervals をモックするためのパッチ
@patch('mir_eval.io.load_valued_intervals')
def test_load_reference_data_success(mock_load_intervals):
    """参照データの読み込み成功ケース"""
    dummy_intervals = np.array([[0.0, 1.0], [1.5, 2.0]])
    dummy_values = np.array([60.0, 72.0]) # 例としてMIDIノート
    mock_load_intervals.return_value = (dummy_intervals, dummy_values)
    
    file_path = "tests/data/reference/dummy_ref.csv"
    
    intervals, values = load_reference_data(file_path)
    
    mock_load_intervals.assert_called_once_with(file_path)
    np.testing.assert_array_equal(intervals, dummy_intervals)
    np.testing.assert_array_equal(values, dummy_values)

# mir_eval.io.load_valued_intervals が例外を発生させるケース
@patch('mir_eval.io.load_valued_intervals', side_effect=IOError("Mocked IOError"))
def test_load_reference_data_io_error(mock_load_intervals):
    """参照データの読み込み時にIOErrorが発生するケース"""
    file_path = "tests/data/reference/dummy_ref.csv"
    
    with pytest.raises(IOError):
        load_reference_data(file_path)
        
    mock_load_intervals.assert_called_once_with(file_path)

# ファイルが存在しないケース (FileNotFoundError - load_valued_intervalsが内部で発生させることを想定)
@patch('mir_eval.io.load_valued_intervals', side_effect=FileNotFoundError("Mocked FileNotFoundError"))
def test_load_reference_data_not_found(mock_load_intervals):
    """参照ファイルが存在しないケース"""
    file_path = "/non/existent/ref.csv"
    
    with pytest.raises(FileNotFoundError):
        load_reference_data(file_path)
        
    mock_load_intervals.assert_called_once_with(file_path) 