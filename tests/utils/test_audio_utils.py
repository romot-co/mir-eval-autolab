"""
音声ユーティリティ (`src/utils/audio_utils.py`) のテスト
"""
import pytest
import numpy as np
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path
import shutil
import json

# テスト対象のモジュールをインポート
try:
    from src.utils import audio_utils
except ImportError:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.utils import audio_utils


# --- テストデータディレクトリ --- #
TEST_DATA_DIR = Path(__file__).parent.parent / "data"

# --- フィクスチャ: テスト用の一時ディレクトリとダミーファイル --- #
@pytest.fixture(scope="function")
def temp_ref_dir(tmp_path): # pytest組み込みのtmp_pathフィクスチャを利用
    """各テスト用に一時ディレクトリを作成し、ダミー参照ファイルをコピーする"""
    # data_dir = tmp_path / "ref_data"
    # data_dir.mkdir()
    # pytest の tmp_path はテスト実行毎にユニークなディレクトリを提供
    
    # ダミーファイルを一時ディレクトリにコピー
    dummy_files = [
        "ref_header.csv",
        "ref_no_header.csv",
        "ref_valid.json",
        "ref_bad_cols.csv",
        "ref_invalid_data.csv",
        "ref_empty.csv",
        "ref_missing_key.json",
        "ref_partial_header.csv",
        "ref_error.json"
    ]
    for filename in dummy_files:
        source_path = TEST_DATA_DIR / filename
        if source_path.exists():
            shutil.copy(source_path, tmp_path / filename)
        else:
            # テストデータが見つからない場合はエラーにするか、ファイルを作成する
            # ここではエラーにする
            pytest.fail(f"Test data file not found: {source_path}")
            
    # # デバッグ用に一時ディレクトリの内容を確認 (任意)
    # print(f"Temp dir contents ({tmp_path}): {list(tmp_path.iterdir())}")
    
    yield tmp_path # 一時ディレクトリのパスをテスト関数に渡す
    # tmp_pathフィクスチャが自動でクリーンアップしてくれる


# --- load_audio_file のテスト --- #

@patch('src.utils.audio_utils.librosa.load')
def test_load_audio_file_success(mock_librosa_load):
    """load_audio_file: 正常系のテスト (モック使用)"""
    dummy_audio = np.array([0.1, 0.2, 0.3])
    dummy_sr = 44100
    mock_librosa_load.return_value = (dummy_audio, dummy_sr)
    audio_path = "/fake/path/to/audio.wav"

    # ロガーもモック化して、ログ出力を確認できるようにする
    mock_logger = MagicMock(spec=logging.Logger)

    audio, sr = audio_utils.load_audio_file(audio_path, logger=mock_logger)

    # librosa.load が正しい引数で呼ばれたか確認
    mock_librosa_load.assert_called_once_with(audio_path, sr=44100, mono=True)
    # 返り値が正しいか確認
    assert np.array_equal(audio, dummy_audio)
    assert sr == dummy_sr
    # ログが INFO レベルで出力されたか確認
    mock_logger.info.assert_called_once()
    assert audio_path in mock_logger.info.call_args[0][0]


@patch('src.utils.audio_utils.librosa.load')
@patch('src.utils.audio_utils.log_exception') # log_exception もモック化
def test_load_audio_file_failure(mock_log_exception, mock_librosa_load):
    """load_audio_file: librosa.load が例外を発生させる場合のテスト (モック使用)"""
    test_exception = IOError("Fake IO error")
    mock_librosa_load.side_effect = test_exception
    audio_path = "/fake/path/to/bad_audio.wav"
    mock_logger = MagicMock(spec=logging.Logger)

    audio, sr = audio_utils.load_audio_file(audio_path, logger=mock_logger)

    # librosa.load が呼ばれたか確認
    mock_librosa_load.assert_called_once_with(audio_path, sr=44100, mono=True)
    # 返り値が (None, None) であることを確認
    assert audio is None
    assert sr is None
    # log_exception が呼ばれたか確認
    mock_log_exception.assert_called_once()
    # log_exception に正しい引数が渡されたか（より詳細なチェック）
    call_args, call_kwargs = mock_log_exception.call_args
    assert call_args[0] is mock_logger
    assert call_args[1] is test_exception
    assert audio_path in call_args[2]

@patch('src.utils.audio_utils.librosa.load')
@patch('builtins.print') # print関数をモック化
def test_load_audio_file_failure_no_logger(mock_print, mock_librosa_load):
    """load_audio_file: ロガーなしで失敗した場合に print が呼ばれるか"""
    test_exception = ValueError("Fake value error")
    mock_librosa_load.side_effect = test_exception
    audio_path = "/fake/path/no_logger.wav"

    audio, sr = audio_utils.load_audio_file(audio_path, logger=None) # logger を None にする

    mock_librosa_load.assert_called_once_with(audio_path, sr=44100, mono=True)
    assert audio is None
    assert sr is None
    # print が呼ばれたか確認
    mock_print.assert_called_once()
    assert audio_path in mock_print.call_args[0][0]

# --- load_reference_data のテスト --- #

def test_load_reference_data_csv_header(temp_ref_dir):
    """load_reference_data: ヘッダー付きCSVの正常読み込み"""
    ref_path = temp_ref_dir / "ref_header.csv"
    result = audio_utils.load_reference_data(str(ref_path))
    
    expected_intervals = np.array([[0.1, 0.5], [1.0, 1.5], [2.1, 2.8]])
    expected_pitches = np.array([440.0, 880.0, 659.25])
    
    assert "intervals" in result
    assert "note_pitches" in result
    np.testing.assert_allclose(result["intervals"], expected_intervals)
    np.testing.assert_allclose(result["note_pitches"], expected_pitches)

def test_load_reference_data_csv_no_header(temp_ref_dir):
    """load_reference_data: ヘッダーなしCSVの正常読み込み"""
    ref_path = temp_ref_dir / "ref_no_header.csv"
    result = audio_utils.load_reference_data(str(ref_path))
    
    expected_intervals = np.array([[0.2, 0.6], [1.1, 1.6], [2.2, 2.9]])
    expected_pitches = np.array([261.63, 329.63, 392.00])
    
    assert "intervals" in result
    assert "note_pitches" in result
    np.testing.assert_allclose(result["intervals"], expected_intervals)
    np.testing.assert_allclose(result["note_pitches"], expected_pitches)

def test_load_reference_data_json_valid(temp_ref_dir):
    """load_reference_data: 正常なJSONの読み込み"""
    ref_path = temp_ref_dir / "ref_valid.json"
    result = audio_utils.load_reference_data(str(ref_path))
    
    expected_intervals = np.array([[0.3, 0.7], [1.2, 1.7]])
    expected_pitches = np.array([523.25, 698.46])
    
    assert "intervals" in result
    assert "note_pitches" in result
    np.testing.assert_allclose(result["intervals"], expected_intervals)
    np.testing.assert_allclose(result["note_pitches"], expected_pitches)

@patch('src.utils.audio_utils.logger.error') # logger.error をモック化
def test_load_reference_data_csv_bad_cols(mock_logger_error, temp_ref_dir):
    """load_reference_data: 列不足CSVの場合"""
    ref_path = temp_ref_dir / "ref_bad_cols.csv"
    result = audio_utils.load_reference_data(str(ref_path))
    
    # 結果は空の配列になるはず
    assert "intervals" in result
    assert "note_pitches" in result
    assert result["intervals"].shape == (0, 2)
    assert result["note_pitches"].shape == (0,)
    # エラーログが出力されることを確認 (列数不足は読み込み失敗として処理される)
    mock_logger_error.assert_called()
    assert "読み込みに失敗しました" in mock_logger_error.call_args[0][0]
    assert "列数が3未満" in mock_logger_error.call_args[0][0]

@patch('src.utils.audio_utils.logger.warning') # logger.warning をモック化
def test_load_reference_data_csv_invalid_data(mock_logger_warning, temp_ref_dir):
    """load_reference_data: 無効データを含むCSVの場合"""
    ref_path = temp_ref_dir / "ref_invalid_data.csv"
    result = audio_utils.load_reference_data(str(ref_path))
    
    # 結果は空の配列になるはず
    assert "intervals" in result
    assert "note_pitches" in result
    assert result["intervals"].shape == (0, 2)
    assert result["note_pitches"].shape == (0,)
    # 警告ログが出力されることを確認
    mock_logger_warning.assert_called()
    assert "無効なノートを除外" in mock_logger_warning.call_args[0][0]

@patch('src.utils.audio_utils.logger.error')
def test_load_reference_data_empty_csv(mock_logger_error, temp_ref_dir):
    """load_reference_data: 空のCSVファイルの場合"""
    ref_path = temp_ref_dir / "ref_empty.csv"
    result = audio_utils.load_reference_data(str(ref_path))
    
    # 結果は空の配列になるはず
    assert "intervals" in result
    assert "note_pitches" in result
    assert result["intervals"].shape == (0, 2)
    assert result["note_pitches"].shape == (0,)
    # 警告ログが出力されることを確認
    mock_logger_error.assert_called()
    assert "読み込みに失敗しました" in mock_logger_error.call_args[0][0]

@patch('src.utils.audio_utils.logger.error') # logger.error をモック化
def test_load_reference_data_missing_key_json(mock_logger_error, temp_ref_dir):
    """load_reference_data: 必要なキーがないJSONの場合"""
    ref_path = temp_ref_dir / "ref_missing_key.json"
    result = audio_utils.load_reference_data(str(ref_path))
    
    # 結果は空の配列になるはず
    assert "intervals" in result
    assert "note_pitches" in result
    assert result["intervals"].shape == (0, 2)
    assert result["note_pitches"].shape == (0,)
    # エラーログが出力されることを確認
    mock_logger_error.assert_called()
    assert "必要なキーがありません" in mock_logger_error.call_args[0][0]

@patch('src.utils.audio_utils.logger.error') # logger.error をモック化
def test_load_reference_data_unsupported_ext(mock_logger_error, temp_ref_dir):
    """load_reference_data: 未対応の拡張子の場合"""
    # ダミーのテキストファイルを作成
    ref_path = temp_ref_dir / "ref_unsupported.txt"
    ref_path.touch()
    
    result = audio_utils.load_reference_data(str(ref_path))
    
    # 結果は空の配列になるはず
    assert "intervals" in result
    assert "note_pitches" in result
    assert result["intervals"].shape == (0, 2)
    assert result["note_pitches"].shape == (0,)
    # エラーログが出力されることを確認
    mock_logger_error.assert_called()
    assert "未対応のファイル形式" in mock_logger_error.call_args[0][0]

@patch('src.utils.audio_utils.logger.warning') # logger.warning をモック化
@patch('src.utils.audio_utils.logger.error') # logger.error もモック化
def test_load_reference_data_csv_partial_header(mock_logger_error, mock_logger_warning, temp_ref_dir):
    """load_reference_data: 一部異なる列名を持つCSVの場合"""
    ref_path = temp_ref_dir / "ref_partial_header.csv"
    
    # エラーが発生するので空の配列になることを期待
    result = audio_utils.load_reference_data(str(ref_path))
    
    # エラーが発生して空の配列になるはず
    assert "intervals" in result
    assert "note_pitches" in result
    assert result["intervals"].shape == (0, 2)
    assert result["note_pitches"].shape == (0,)
    
    # エラーログが出力されることを確認
    mock_logger_error.assert_called()

@patch('src.utils.audio_utils.logger.error') # logger.error をモック化
def test_load_reference_data_json_error(mock_logger_error, temp_ref_dir):
    """load_reference_data: JSONパースエラーの場合"""
    ref_path = temp_ref_dir / "ref_error.json"
    
    # JSONパースエラーをシミュレート
    with patch('src.utils.audio_utils.json.load', side_effect=json.JSONDecodeError("Invalid JSON", "doc", 1)):
        result = audio_utils.load_reference_data(str(ref_path))
    
    # 結果は空の配列になるはず
    assert "intervals" in result
    assert "note_pitches" in result
    assert result["intervals"].shape == (0, 2)
    assert result["note_pitches"].shape == (0,)
    
    # エラーログが出力されることを確認
    mock_logger_error.assert_called()
    
    # 任意のエラーメッセージチェック（トレースバックも含まれる可能性があるので部分一致で確認）
    # error_message = mock_logger_error.call_args[0][0]
    # assert "参照データの読み込み中にエラーが発生しました" in error_message

@patch('src.utils.audio_utils.logger.warning')
def test_load_audio_and_reference_null_ref_path(mock_logger_warning):
    """load_audio_and_reference: ref_pathがNoneの場合"""
    # モックの設定
    dummy_audio = np.array([0.1])
    dummy_sr = 44100
    
    with patch('src.utils.audio_utils.load_audio_file', return_value=(dummy_audio, dummy_sr)):
        audio_path = "/fake/audio.wav"
        ref_path = None  # ref_pathをNoneに設定
        mock_logger = MagicMock()
        
        audio, ref_data, sr = audio_utils.load_audio_and_reference(audio_path, ref_path, logger=mock_logger)
        
        assert np.array_equal(audio, dummy_audio)
        assert ref_data is None  # ref_dataはNoneのまま
        assert sr == dummy_sr
        mock_logger.warning.assert_called_once()  # 警告が出るはず
        assert "見つからないか指定されていません" in mock_logger.warning.call_args[0][0]

@patch('src.utils.audio_utils.os.path')
def test_make_output_path_with_results_suffix(mock_os_path):
    """make_output_path: 入力ファイル名に_resultsが含まれる場合"""
    # モックの設定
    mock_os_path.basename.return_value = "track_results.wav"
    mock_os_path.splitext.return_value = ("track_results", ".wav")
    mock_os_path.join.side_effect = lambda *args: '/'.join(args)
    
    output_dir = "results"
    audio_file = "data/audio/track_results.wav"
    detector_name = "TestDetector"
    extension = "json"
    
    # 関数を呼び出し
    result = audio_utils.make_output_path(output_dir, audio_file, detector_name, extension)
    
    # モックが正しく呼ばれたか確認
    mock_os_path.basename.assert_called_once_with(audio_file)
    mock_os_path.splitext.assert_called_once_with("track_results.wav")
    mock_os_path.join.assert_called_once_with(output_dir, "track_results_TestDetector.json")
    
    # 現在の実装では _results は削除されない
    expected = "results/track_results_TestDetector.json"
    assert result == expected

# --- 完全カバレッジのためのテストケース --- #

@patch('src.utils.audio_utils.pd.read_csv')
@patch('src.utils.audio_utils.logger.error')
def test_load_reference_data_csv_read_error(mock_logger_error, mock_read_csv, temp_ref_dir):
    """load_reference_data: CSVの読み込みエラーのテスト"""
    # モックの設定
    mock_read_csv.side_effect = Exception("CSV read error")
    ref_path = temp_ref_dir / "ref_header.csv"
    
    result = audio_utils.load_reference_data(str(ref_path))
    
    # 結果は空の配列になるはず
    assert "intervals" in result
    assert "note_pitches" in result
    assert result["intervals"].shape == (0, 2)
    assert result["note_pitches"].shape == (0,)
    
    # エラーログが出力されることを確認
    mock_logger_error.assert_called()
    assert "CSVファイルの読み込みに失敗" in mock_logger_error.call_args[0][0]

@patch('src.utils.audio_utils.logger.warning')
@patch('src.utils.audio_utils.logger.error')
def test_load_reference_data_data_length_mismatch(mock_logger_error, mock_logger_warning):
    """load_reference_data: CSVのデータ長が不一致の場合"""
    # モックのDataFrameを作成
    mock_df = MagicMock()
    mock_df.empty = False
    
    # pandas.Series.tolist()メソッドをシミュレートするために必要な設定
    columns_mock = MagicMock()
    columns_mock.tolist.return_value = ['onset', 'offset', 'pitch']
    mock_df.columns = columns_mock
    
    # 長さの異なる配列を返すように設定
    onsets = np.array([0.1, 0.5, 1.0])
    offsets = np.array([0.2, 0.6])  # 要素数が少ない
    pitches = np.array([440.0, 550.0, 660.0, 770.0])  # 要素数が多い
    
    # pandas.Series.values属性をシミュレート
    onset_mock = MagicMock()
    onset_mock.values = onsets
    offset_mock = MagicMock()
    offset_mock.values = offsets
    pitch_mock = MagicMock()
    pitch_mock.values = pitches
    
    mock_df.__getitem__.side_effect = lambda x: {
        'onset': onset_mock,
        'offset': offset_mock,
        'pitch': pitch_mock
    }[x]
    
    with patch('src.utils.audio_utils.pd.read_csv', return_value=mock_df), \
         patch('src.utils.audio_utils.os.path.splitext', return_value=('', '.csv')):
        result = audio_utils.load_reference_data('/fake/path.csv')
        
        # 異なる長さのデータが適切に処理されるか確認
        # 現在の実装では何らかのエラーが発生し、空の配列が返る
        assert result['intervals'].shape == (0, 2)
        assert result['note_pitches'].shape == (0,)
        
        # エラーログが出力されることを確認
        mock_logger_error.assert_called()
        # エラー内容をチェック
        error_message = mock_logger_error.call_args[0][0]
        assert "CSVファイルの読み込みに失敗" in error_message or "参照データの読み込み中にエラーが発生しました" in error_message

# --- load_audio_and_reference のテスト --- #

@patch('src.utils.audio_utils.load_reference_data')
@patch('src.utils.audio_utils.load_audio_file')
def test_load_audio_and_reference_success(mock_load_audio, mock_load_ref):
    """load_audio_and_reference: 正常系のテスト (モック使用)"""
    dummy_audio = np.array([0.1])
    dummy_sr = 44100
    dummy_ref_data = {'intervals': np.array([[0.1, 0.5]]), 'note_pitches': np.array([440.0])}
    mock_load_audio.return_value = (dummy_audio, dummy_sr)
    mock_load_ref.return_value = dummy_ref_data
    
    audio_path = "/fake/audio.wav"
    ref_path = "/fake/ref.csv"
    mock_logger = MagicMock()

    # os.path.exists もモック化する必要がある
    with patch('src.utils.audio_utils.os.path.exists', return_value=True):
        audio, ref_data, sr = audio_utils.load_audio_and_reference(audio_path, ref_path, logger=mock_logger)

    mock_load_audio.assert_called_once_with(audio_path, mock_logger)
    mock_load_ref.assert_called_once_with(ref_path)
    assert np.array_equal(audio, dummy_audio)
    assert ref_data is dummy_ref_data
    assert sr == dummy_sr
    mock_logger.warning.assert_not_called() # 警告は出ないはず

@patch('src.utils.audio_utils.load_reference_data')
@patch('src.utils.audio_utils.load_audio_file')
@patch('src.utils.audio_utils.os.path.exists') # os.path.exists をモック化
def test_load_audio_and_reference_ref_not_exist(mock_exists, mock_load_audio, mock_load_ref):
    """load_audio_and_reference: 参照ファイルが存在しない場合"""
    dummy_audio = np.array([0.1])
    dummy_sr = 44100
    mock_load_audio.return_value = (dummy_audio, dummy_sr)
    mock_exists.return_value = False # ファイルが存在しないように設定
    
    audio_path = "/fake/audio.wav"
    ref_path = "/fake/non_existent_ref.csv"
    mock_logger = MagicMock()

    audio, ref_data, sr = audio_utils.load_audio_and_reference(audio_path, ref_path, logger=mock_logger)

    mock_load_audio.assert_called_once_with(audio_path, mock_logger)
    mock_exists.assert_called_once_with(ref_path)
    mock_load_ref.assert_not_called() # refがないので呼ばれない
    assert np.array_equal(audio, dummy_audio)
    assert ref_data is None # ref_dataはNoneになるはず
    assert sr == dummy_sr
    mock_logger.warning.assert_called_once() # 警告が出るはず
    assert ref_path in mock_logger.warning.call_args[0][0]

# --- make_output_path のテスト --- #

def test_make_output_path():
    """make_output_path: パス生成のテスト"""
    output_dir = "results"
    audio_file = "data/audio/song.wav"
    detector_name = "MyDetector"
    
    expected_json = "results/song_MyDetector.json"
    expected_csv = "results/song_MyDetector.csv"
    expected_png = "results/song_MyDetector.png"
    
    assert audio_utils.make_output_path(output_dir, audio_file, detector_name, extension="json") == expected_json
    assert audio_utils.make_output_path(output_dir, audio_file, detector_name, extension="csv") == expected_csv
    assert audio_utils.make_output_path(output_dir, audio_file, detector_name, extension=".png") == expected_png # 先頭のドットは除去されるはず
    assert audio_utils.make_output_path(output_dir, audio_file, detector_name) == expected_json # デフォルト拡張子はjson

def test_make_output_path_absolute():
    """make_output_path: 絶対パス入力のテスト"""
    output_dir = "/abs/path/results"
    # audio_file は basename が使われるので相対でも絶対でも結果は同じはず
    audio_file = "/another/path/track.mp3"
    detector_name = "AnotherDet"
    
    expected = "/abs/path/results/track_AnotherDet.txt"
    assert audio_utils.make_output_path(output_dir, audio_file, detector_name, extension="txt") == expected 