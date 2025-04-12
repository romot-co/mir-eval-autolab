import pytest
from unittest.mock import patch, mock_open, MagicMock
import numpy as np
import os
import pandas as pd
import json
import logging

from src.utils.audio_utils import load_audio_file, load_reference_data, load_audio_and_reference, make_output_path

# 共通のダミーデータ作成フィクスチャ
@pytest.fixture
def dummy_audio_data():
    """ダミーの音声データと サンプルレートを返す"""
    return np.array([0.1, 0.2, 0.3]), 44100

@pytest.fixture
def dummy_ref_data():
    """ダミーの参照データを辞書形式で返す"""
    return {
        'intervals': np.array([[0.1, 0.5], [0.6, 1.0]]),
        'note_pitches': np.array([60.0, 72.0])
    }

@pytest.fixture
def dummy_ref_df():
    """ダミーの参照データをDataFrame形式で返す"""
    return pd.DataFrame({
        'onset': [0.1, 0.6],
        'offset': [0.5, 1.0],
        'pitch': [60.0, 72.0]
    })

@pytest.fixture
def dummy_ref_json():
    """ダミーの参照データをJSON互換の辞書形式で返す"""
    return {
        'intervals': [[0.1, 0.5], [0.6, 1.0]],
        'note_pitches': [60.0, 72.0]
    }

@pytest.fixture
def mock_logger():
    """モックロガーオブジェクトを返す"""
    return MagicMock(spec=logging.Logger)

# --- load_audio_file Tests ---
def test_load_audio_file_success(dummy_audio_data, mock_logger):
    """音声ファイルの読み込み成功ケース"""
    dummy_audio, dummy_sr = dummy_audio_data
    
    with patch('librosa.load', return_value=dummy_audio_data) as mock_librosa_load:
        with patch('os.path.exists', return_value=True):
            file_path = "dummy_audio.wav"
            audio_data, sample_rate = load_audio_file(file_path, mock_logger)
            
            mock_librosa_load.assert_called_once_with(file_path, sr=44100, mono=True)
            np.testing.assert_array_equal(audio_data, dummy_audio)
            assert sample_rate == dummy_sr
            mock_logger.info.assert_called_once()

def test_load_audio_file_error(mock_logger):
    """音声ファイルの読み込み時にエラーが発生するケース"""
    
    with patch('librosa.load', side_effect=Exception("Mocked Exception")) as mock_librosa_load:
        with patch('os.path.exists', return_value=True):
            file_path = "dummy_audio.wav"
            audio_data, sample_rate = load_audio_file(file_path, mock_logger)
            
            mock_librosa_load.assert_called_once_with(file_path, sr=44100, mono=True)
            assert audio_data is None
            assert sample_rate is None
            # エラーログが記録されていることを確認
            mock_logger.error.assert_called_once()

def test_load_audio_file_nonexistent_file(mock_logger):
    """存在しない音声ファイルを読み込もうとする場合のテスト"""
    # ファイルが存在しないことを確認するパッチ
    with patch('os.path.exists', return_value=False):
        file_path = "nonexistent_audio.wav"
        audio_data, sample_rate = load_audio_file(file_path, mock_logger)
        
        # 結果がNoneであることを確認
        assert audio_data is None
        assert sample_rate is None
        # エラーログが記録されていることを確認
        mock_logger.error.assert_called_once()

# --- load_reference_data Tests ---
def test_load_reference_data_csv_success(dummy_ref_df):
    """CSVファイルからの参照データ読み込み成功ケース"""
    with patch('pandas.read_csv', return_value=dummy_ref_df) as mock_read_csv:
        with patch('os.path.exists', return_value=True):
            with patch('os.path.splitext', return_value=('dummy_ref', '.csv')):
                with patch('logging.getLogger', return_value=MagicMock()) as mock_get_logger:
                    file_path = "dummy_ref.csv"
                    result = load_reference_data(file_path)
                    
                    mock_read_csv.assert_called_once_with(file_path, header=0)
                    
                    assert 'intervals' in result
                    assert 'note_pitches' in result
                    np.testing.assert_array_equal(result['intervals'], np.array([[0.1, 0.5], [0.6, 1.0]]))
                    np.testing.assert_array_equal(result['note_pitches'], np.array([60.0, 72.0]))

def test_load_reference_data_json_success(dummy_ref_json):
    """JSONファイルからの参照データ読み込み成功ケース"""
    m = mock_open()
    
    with patch('os.path.splitext', return_value=('dummy_ref', '.json')):
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', m):
                with patch('json.load', return_value=dummy_ref_json) as mock_json_load:
                    with patch('logging.getLogger', return_value=MagicMock()) as mock_get_logger:
                        file_path = "dummy_ref.json"
                        result = load_reference_data(file_path)
                        
                        m.assert_called_once_with(file_path, 'r')
                        mock_json_load.assert_called_once()
                        
                        assert 'intervals' in result
                        assert 'note_pitches' in result
                        np.testing.assert_array_equal(result['intervals'], np.array([[0.1, 0.5], [0.6, 1.0]]))
                        np.testing.assert_array_equal(result['note_pitches'], np.array([60.0, 72.0]))

def test_load_reference_data_csv_error():
    """CSVファイルからの参照データ読み込み失敗ケース"""
    with patch('os.path.splitext', return_value=('dummy_ref', '.csv')):
        with patch('os.path.exists', return_value=True):
            with patch('pandas.read_csv', side_effect=Exception("Mocked CSV Error")) as mock_read_csv:
                with patch('logging.getLogger', return_value=MagicMock()) as mock_get_logger:
                    file_path = "dummy_ref.csv"
                    result = load_reference_data(file_path)
                    
                    # 実装ではヘッダーありとなしの両方で試すため2回呼ばれる
                    assert mock_read_csv.call_count == 2
                    
                    # エラー時は空の配列が返されることを確認
                    assert 'intervals' in result
                    assert 'note_pitches' in result
                    assert result['intervals'].shape == (0, 2)
                    assert result['note_pitches'].shape == (0,)

def test_load_reference_data_unsupported_format():
    """未対応のファイル形式からの参照データ読み込みケース"""
    with patch('os.path.splitext', return_value=('dummy_ref', '.txt')):
        with patch('os.path.exists', return_value=True):
            with patch('logging.getLogger', return_value=MagicMock()) as mock_get_logger:
                file_path = "dummy_ref.txt"
                result = load_reference_data(file_path)
                
                # 未対応の形式は空の配列が返されることを確認
                assert 'intervals' in result
                assert 'note_pitches' in result
                assert result['intervals'].shape == (0, 2)
                assert result['note_pitches'].shape == (0,)

def test_load_reference_data_csv_headers_missing(tmp_path, caplog):
    """CSVファイルにヘッダーがない場合のテスト"""
    # ヘッダーなしのCSVファイルを作成
    csv_path = tmp_path / "no_header.csv"
    with open(csv_path, 'w') as f:
        f.write("0.1,0.5,60.0\n")
        f.write("0.6,1.0,72.0\n")
    
    # 必要なモジュールをパッチしてテスト
    with patch('os.path.exists', return_value=True):
        result = load_reference_data(str(csv_path))
        
        # ヘッダーなしでも正しく読み込まれることを確認
        assert 'intervals' in result
        assert 'note_pitches' in result
        np.testing.assert_array_equal(result['intervals'], np.array([[0.1, 0.5], [0.6, 1.0]]))
        np.testing.assert_array_equal(result['note_pitches'], np.array([60.0, 72.0]))
        # 警告ログに「位置でデータを判断」というメッセージが含まれていることを確認
        assert "位置でデータを判断" in caplog.text

def test_load_reference_data_csv_invalid_data(tmp_path, caplog):
    """CSVファイルに無効なデータが含まれる場合のテスト"""
    # 無効なデータを含むCSVファイルを作成（負のオンセット時間）
    csv_path = tmp_path / "invalid_data.csv"
    with open(csv_path, 'w') as f:
        f.write("onset,offset,pitch\n")
        f.write("-0.1,0.5,60.0\n")  # 負のオンセット時間
        f.write("0.6,1.0,72.0\n")
    
    # 必要なモジュールをパッチしてテスト
    with patch('os.path.exists', return_value=True):
        result = load_reference_data(str(csv_path))
        
        # 無効なデータを含む場合、空の配列が返されることを確認
        assert 'intervals' in result
        assert 'note_pitches' in result
        assert result['intervals'].shape == (0, 2)
        assert result['note_pitches'].shape == (0,)
        # 警告ログに「無効なノートを除外」というメッセージが含まれていることを確認
        assert "無効なノート" in caplog.text

def test_load_reference_data_json_missing_keys(tmp_path, caplog):
    """JSONファイルに必要なキーがない場合のテスト"""
    # 必要なキーがないJSONファイルを作成
    json_path = tmp_path / "missing_keys.json"
    with open(json_path, 'w') as f:
        f.write('{"some_key": "some_value"}')
    
    # 必要なモジュールをパッチしてテスト
    with patch('os.path.exists', return_value=True):
        result = load_reference_data(str(json_path))
        
        # 必要なキーがない場合、空の配列が返されることを確認
        assert 'intervals' in result
        assert 'note_pitches' in result
        assert result['intervals'].shape == (0, 2)
        assert result['note_pitches'].shape == (0,)
        # エラーログに「必要なキーがありません」というメッセージが含まれていることを確認
        assert "必要なキー" in caplog.text

def test_load_reference_data_csv_insufficient_columns(tmp_path, caplog):
    """CSVファイルの列数が不足している場合のテスト"""
    # 列数が不足しているCSVファイルを作成
    csv_path = tmp_path / "insufficient_columns.csv"
    with open(csv_path, 'w') as f:
        f.write("0.1,0.5\n")  # ピッチ列がない
        f.write("0.6,1.0\n")
    
    # 必要なモジュールをパッチしてテスト
    with patch('os.path.exists', return_value=True):
        result = load_reference_data(str(csv_path))
        
        # 列数が不足している場合、空の配列が返されることを確認
        assert 'intervals' in result
        assert 'note_pitches' in result
        assert result['intervals'].shape == (0, 2)
        assert result['note_pitches'].shape == (0,)
        # エラーログに「列数不足」というメッセージが含まれていることを確認
        assert "列数不足" in caplog.text or "列数が3未満" in caplog.text

def test_load_reference_data_permission_error(tmp_path, caplog):
    """ファイルの読み取り権限がない場合のテスト"""
    # CSVパスを設定
    csv_path = tmp_path / "permission_error.csv"
    
    # open関数がPermissionErrorを送出するようにモック
    mock_open = MagicMock(side_effect=PermissionError("権限がありません"))
    
    # 必要なモジュールをパッチしてテスト
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open):
            with patch('os.path.splitext', return_value=('dummy_ref', '.csv')):
                with patch('pandas.read_csv', side_effect=PermissionError("権限がありません")):
                    result = load_reference_data(str(csv_path))
                    
                    # 権限エラーの場合、空の配列が返されることを確認
                    assert 'intervals' in result
                    assert 'note_pitches' in result
                    assert result['intervals'].shape == (0, 2)
                    assert result['note_pitches'].shape == (0,)
                    # エラーログにエラーメッセージが含まれていることを確認
                    assert "CSVファイルの読み込みに失敗しました" in caplog.text
                    assert "権限がありません" in caplog.text

# --- load_audio_and_reference Tests ---
def test_load_audio_and_reference_success(dummy_audio_data, dummy_ref_data, mock_logger):
    """音声と参照データの読み込み成功ケース"""
    
    with patch('src.utils.audio_utils.load_audio_file', return_value=dummy_audio_data) as mock_load_audio:
        with patch('src.utils.audio_utils.load_reference_data', return_value=dummy_ref_data) as mock_load_reference:
            with patch('os.path.exists', return_value=True):
                audio_path = "dummy_audio.wav"
                ref_path = "dummy_ref.csv"
                
                audio, ref_data, sr = load_audio_and_reference(audio_path, ref_path, mock_logger)
                
                mock_load_audio.assert_called_once_with(audio_path, mock_logger)
                mock_load_reference.assert_called_once_with(ref_path)
                
                np.testing.assert_array_equal(audio, dummy_audio_data[0])
                assert sr == dummy_audio_data[1]
                assert ref_data == dummy_ref_data

def test_load_audio_and_reference_missing_reference(dummy_audio_data, mock_logger):
    """参照データがない場合のテスト"""
    
    with patch('src.utils.audio_utils.load_audio_file', return_value=dummy_audio_data) as mock_load_audio:
        with patch('os.path.exists', return_value=False):
            audio_path = "dummy_audio.wav"
            ref_path = "nonexistent_ref.csv"
            
            audio, ref_data, sr = load_audio_and_reference(audio_path, ref_path, mock_logger)
            
            mock_load_audio.assert_called_once_with(audio_path, mock_logger)
            # 参照データファイルが存在しない場合の警告ログを確認
            mock_logger.warning.assert_called_once()
            
            np.testing.assert_array_equal(audio, dummy_audio_data[0])
            assert sr == dummy_audio_data[1]
            assert ref_data is None

def test_load_audio_and_reference_no_ref_path(dummy_audio_data, mock_logger):
    """参照データパスが指定されていない場合のテスト"""
    
    with patch('src.utils.audio_utils.load_audio_file', return_value=dummy_audio_data) as mock_load_audio:
        audio_path = "dummy_audio.wav"
        ref_path = None
        
        audio, ref_data, sr = load_audio_and_reference(audio_path, ref_path, mock_logger)
        
        mock_load_audio.assert_called_once_with(audio_path, mock_logger)
        # 参照データファイルが指定されていない場合の警告ログを確認
        mock_logger.warning.assert_called_once()
        
        np.testing.assert_array_equal(audio, dummy_audio_data[0])
        assert sr == dummy_audio_data[1]
        assert ref_data is None

def test_load_audio_and_reference_audio_error(dummy_ref_data, mock_logger):
    """音声ファイルの読み込みに失敗した場合のテスト"""
    
    with patch('src.utils.audio_utils.load_audio_file', return_value=(None, None)) as mock_load_audio:
        with patch('src.utils.audio_utils.load_reference_data', return_value=dummy_ref_data) as mock_load_reference:
            with patch('os.path.exists', return_value=True):
                audio_path = "error_audio.wav"
                ref_path = "dummy_ref.csv"
                
                audio, ref_data, sr = load_audio_and_reference(audio_path, ref_path, mock_logger)
                
                mock_load_audio.assert_called_once_with(audio_path, mock_logger)
                mock_load_reference.assert_called_once_with(ref_path)
                
                assert audio is None
                assert sr is None
                assert ref_data == dummy_ref_data
                # エラー時の警告ログが記録されていないことを確認（load_audio_file内でログ出力済み）
                mock_logger.warning.assert_not_called()

def test_make_output_path_relative_path():
    """相対パスでの出力パス生成のテスト"""
    output_dir = "output"
    audio_file = "audio/sample.wav"
    detector_name = "TestDetector"
    
    result = make_output_path(output_dir, audio_file, detector_name)
    
    # 正しい出力パスが生成されるか確認
    expected = os.path.join(output_dir, f"sample_{detector_name}.json")
    assert result == expected

def test_make_output_path_with_extension():
    """カスタム拡張子での出力パス生成のテスト"""
    output_dir = "output"
    audio_file = "audio/sample.wav"
    detector_name = "TestDetector"
    extension = "csv"
    
    result = make_output_path(output_dir, audio_file, detector_name, extension)
    
    # 正しい出力パスが生成されるか確認
    expected = os.path.join(output_dir, f"sample_{detector_name}.csv")
    assert result == expected

def test_make_output_path_absolute_path():
    """絶対パスでの出力パス生成のテスト"""
    output_dir = "/absolute/path/output"
    audio_file = "/data/audio/sample.wav"
    detector_name = "TestDetector"
    
    result = make_output_path(output_dir, audio_file, detector_name)
    
    # 正しい出力パスが生成されるか確認
    expected = os.path.join(output_dir, f"sample_{detector_name}.json")
    assert result == expected

def test_make_output_path_removes_results_suffix():
    """ファイル名から_resultsサフィックスが削除されることを確認するテスト"""
    output_dir = "output"
    audio_file = "audio/sample_results.wav"
    detector_name = "TestDetector"
    
    result = make_output_path(output_dir, audio_file, detector_name)
    
    # _resultsが削除され、正しい出力パスが生成されるか確認
    expected = os.path.join(output_dir, f"sample_{detector_name}.json")
    assert result == expected
    
    # 別のファイル名パターンでも確認
    audio_file2 = "/path/to/test_file_results.mp3"
    result2 = make_output_path(output_dir, audio_file2, detector_name)
    expected2 = os.path.join(output_dir, f"test_file_{detector_name}.json")
    assert result2 == expected2
    
    # 中間に_resultsを含むファイル名の場合（末尾のみが削除対象）
    audio_file3 = "audio/sample_results_test.wav"
    result3 = make_output_path(output_dir, audio_file3, detector_name)
    expected3 = os.path.join(output_dir, f"sample_results_test_{detector_name}.json")
    assert result3 == expected3
    
    # 複数の_resultsを持つケース（最後のみ削除される）
    audio_file4 = "audio/sample_results_results.wav"
    result4 = make_output_path(output_dir, audio_file4, detector_name)
    expected4 = os.path.join(output_dir, f"sample_results_{detector_name}.json")
    assert result4 == expected4
    
    # _resultsで始まるファイル名の場合
    audio_file5 = "_results_sample.wav"
    result5 = make_output_path(output_dir, audio_file5, detector_name)
    expected5 = os.path.join(output_dir, f"_results_sample_{detector_name}.json")
    assert result5 == expected5
    
    # _resultsのみのファイル名の場合
    audio_file6 = "_results.wav"
    result6 = make_output_path(output_dir, audio_file6, detector_name)
    expected6 = os.path.join(output_dir, f"_{detector_name}.json")
    assert result6 == expected6 