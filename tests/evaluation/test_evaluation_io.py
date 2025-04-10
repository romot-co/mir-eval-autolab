"""
評価結果のIO操作モジュールのテスト

このモジュールは、src/evaluation/evaluation_io.pyの関数をテストします。
特に、評価結果の保存と読み込み、サマリー統計の生成と表示をテストします。
"""

import pytest
import os
import json
import numpy as np
import pandas as pd
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import matplotlib
matplotlib.use('Agg') # 非対話的なバックエンドを指定

from src.evaluation.evaluation_io import (
    save_evaluation_result,
    load_evaluation_result,
    load_multiple_evaluation_results,
    print_summary_statistics,
    create_summary_dataframe,
    save_detection_plot,
    save_detection_and_evaluation_results
)
from src.utils.exception_utils import FileError
from src.utils.json_utils import NumpyEncoder

# テスト用の評価結果データ
@pytest.fixture
def sample_evaluation_result():
    """テスト用の評価結果を提供します"""
    return {
        "detector_name": "TestDetector",
        "audio_file": "test_audio.wav",
        "ref_file": "test_ref.csv",
        "detection_time": 0.5,
        "evaluation": {
            "onset": {"precision": 0.8, "recall": 0.7, "f_measure": 0.75},
            "offset": {"precision": 0.75, "recall": 0.65, "f_measure": 0.7},
            "pitch": {"precision": 0.85, "recall": 0.8, "f_measure": 0.825},
            "note": {"precision": 0.82, "recall": 0.78, "f_measure": 0.8},
            "frame_pitch": {
                "voicing_recall": 0.7,
                "voicing_false_alarm": 0.2,
                "raw_pitch_accuracy": 0.75,
                "raw_chroma_accuracy": 0.8,
                "overall_accuracy": 0.7
            }
        },
        "valid": True,
        "error": None
    }

# テスト用の検出結果データ
@pytest.fixture
def sample_detection_result():
    """テスト用の検出結果を提供します"""
    return {
        "intervals": np.array([[0.1, 0.5], [0.7, 1.2]]),
        "note_pitches": np.array([440.0, 880.0]),
        "frame_times": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
        "frame_frequencies": np.array([440.0, 440.0, 440.0, 440.0, 440.0, 0.0, 880.0, 880.0, 880.0, 880.0, 880.0, 880.0])
    }

@pytest.fixture
def sample_reference_data():
    """テスト用の参照データを提供します"""
    return {
        "intervals": np.array([[0.15, 0.55], [0.75, 1.25]]),
        "note_pitches": np.array([440.0, 880.0]),
        "frame_times": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
        "frame_frequencies": np.array([440.0, 440.0, 440.0, 440.0, 440.0, 0.0, 880.0, 880.0, 880.0, 880.0, 880.0, 880.0])
    }

# save_evaluation_resultのテスト
def test_save_evaluation_result(sample_evaluation_result):
    """save_evaluation_result関数が正しく動作することを確認します"""
    # テスト用の一時ファイルを使用
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_output.json")
        
        # 関数の実行
        save_evaluation_result(sample_evaluation_result, output_path)
        
        # ファイルが作成されたことを確認
        assert os.path.exists(output_path)
        
        # ファイルの内容を検証
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded_result = json.load(f)
            
        # ロードしたデータが元のデータと一致することを確認
        assert loaded_result["detector_name"] == sample_evaluation_result["detector_name"]
        assert loaded_result["audio_file"] == sample_evaluation_result["audio_file"]
        assert loaded_result["evaluation"]["onset"]["precision"] == sample_evaluation_result["evaluation"]["onset"]["precision"]

# load_evaluation_resultのテスト
def test_load_evaluation_result(sample_evaluation_result):
    """load_evaluation_result関数が正しく動作することを確認します"""
    # テスト用の一時ファイルを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_output.json")
        
        # JSONファイルを作成
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_evaluation_result, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
        
        # 関数の実行
        loaded_result = load_evaluation_result(output_path)
        
        # ロードしたデータが元のデータと一致することを確認
        assert loaded_result["detector_name"] == sample_evaluation_result["detector_name"]
        assert loaded_result["audio_file"] == sample_evaluation_result["audio_file"]
        assert loaded_result["evaluation"]["onset"]["precision"] == sample_evaluation_result["evaluation"]["onset"]["precision"]

# load_multiple_evaluation_resultsのテスト
def test_load_multiple_evaluation_results(sample_evaluation_result):
    """load_multiple_evaluation_results関数が正しく動作することを確認します"""
    # テスト用の一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        # 複数のJSONファイルを作成
        expected_audio_files = []
        for i in range(3):
            result_copy = sample_evaluation_result.copy()
            # 元のaudio_fileを変更して順序を認識できるようにする
            audio_file = f"test_audio_{i}.wav"
            result_copy["audio_file"] = audio_file
            expected_audio_files.append(audio_file)
            
            output_path = os.path.join(temp_dir, f"test_output_{i}_evaluation.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_copy, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
        
        # 関数の実行
        loaded_results = load_multiple_evaluation_results(temp_dir)
        
        # 結果の検証
        assert len(loaded_results) == 3
        
        # ファイル名でソートしてから検証する
        loaded_audio_files = [result["audio_file"] for result in loaded_results]
        loaded_audio_files.sort()
        expected_audio_files.sort()
        
        for i in range(3):
            assert loaded_audio_files[i] == expected_audio_files[i]
            assert any(result["detector_name"] == sample_evaluation_result["detector_name"] for result in loaded_results)

# load_multiple_evaluation_resultsのエラーケース
def test_load_multiple_evaluation_results_directory_not_found():
    """存在しないディレクトリに対するload_multiple_evaluation_resultsのテスト"""
    # 存在しないディレクトリパス
    non_existent_dir = "/path/that/does/not/exist"
    
    # FileErrorが発生することを確認
    with pytest.raises(FileError):
        load_multiple_evaluation_results(non_existent_dir)

# load_multiple_evaluation_resultsで一部ファイルに問題がある場合
def test_load_multiple_evaluation_results_partial_errors(sample_evaluation_result, caplog):
    """一部のファイルに問題がある場合のload_multiple_evaluation_resultsのテスト"""
    # テスト用の一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        # 正常なJSONファイル
        valid_path = os.path.join(temp_dir, "valid_evaluation.json")
        with open(valid_path, 'w', encoding='utf-8') as f:
            json.dump(sample_evaluation_result, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
        
        # 不正なJSONファイル
        invalid_path = os.path.join(temp_dir, "invalid_evaluation.json")
        with open(invalid_path, 'w', encoding='utf-8') as f:
            f.write("{invalid json")
        
        # 関数の実行
        loaded_results = load_multiple_evaluation_results(temp_dir)
        
        # 結果の検証: 正常なファイルだけが読み込まれていることを確認
        assert len(loaded_results) == 1
        assert loaded_results[0]["detector_name"] == sample_evaluation_result["detector_name"]
        
        # ログに警告メッセージが含まれていることを確認
        assert "JSONファイルの解析に失敗しました" in caplog.text

# print_summary_statisticsのテスト
def test_print_summary_statistics():
    """print_summary_statistics関数が正しく動作することを確認します"""
    # テスト用のDataFrameを作成
    summary_data = {
        'detector_name': ['TestDetector'],
        'files_count': [10],
        'detection_time_mean': [0.5],
        'onset_precision': [0.8],
        'onset_recall': [0.7],
        'onset_f_measure': [0.75],
        'note_precision': [0.82],
        'note_recall': [0.78],
        'note_f_measure': [0.8]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # モックロガーを作成
    mock_logger = MagicMock()
    
    # 関数の実行
    print_summary_statistics(summary_df, logger=mock_logger)
    
    # ロガーが適切に呼び出されたことを確認
    assert mock_logger.info.call_count >= 4  # 少なくとも4回は呼び出されるはず
    
    # 重要な指標が出力されていることを確認
    mock_logger.info.assert_any_call("\n評価結果のサマリー:")
    # 検出器名の確認
    assert any("TestDetector" in str(call) for call in mock_logger.info.call_args_list)
    # 評価ファイル数
    assert any("10" in str(call) for call in mock_logger.info.call_args_list)

# create_summary_dataframeのテスト
def test_create_summary_dataframe(sample_evaluation_result):
    """create_summary_dataframe関数が正しく動作することを確認します"""
    # テスト用の結果リストを作成
    results = []
    for i in range(3):
        result_copy = sample_evaluation_result.copy()
        result_copy["audio_path"] = f"test_audio_{i}.wav"
        result_copy["status"] = "success"  # statusフィールドを追加
        results.append(result_copy)
    
    # 関数の実行
    df = create_summary_dataframe(results)
    
    # 結果の検証
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # 3つの結果に対応する行がある
    assert "detector_name" in df.columns
    assert "detection_time" in df.columns
    assert df["detector_name"].iloc[0] == "TestDetector"

# save_detection_plotのテスト（matplotlib依存のためMockを使用）
@patch('matplotlib.pyplot.close') # plt.close をモック
@patch('matplotlib.pyplot.figure') # plt.figure をモック
@patch('src.visualization.plots.plot_detection_results')
def test_save_detection_plot(mock_plot_results, mock_figure, mock_close, sample_detection_result, sample_reference_data):
    """save_detection_plot関数が正しく動作することを確認します"""
    # テスト用の一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_plot.png")
        
        # 関数の実行
        save_detection_plot(
            sample_detection_result,
            sample_reference_data,
            output_path
        )
        
        # 適切なモックメソッドが呼び出されたことを確認
        mock_figure.assert_called_once()
        mock_close.assert_called_once()
        mock_plot_results.assert_called_once()
        
        # 出力パスが正しく渡されたことを確認
        assert mock_plot_results.call_args[1]['save_path'] == output_path

# save_detection_and_evaluation_resultsのテスト
def test_save_detection_and_evaluation_results(sample_detection_result, sample_evaluation_result):
    """save_detection_and_evaluation_results関数が正しく動作することを確認します"""
    # テスト用の一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        base_name = "test_audio"
        detector_name = "TestDetector"
        
        # 関数の実行
        result_paths = save_detection_and_evaluation_results(
            sample_detection_result,
            sample_evaluation_result,
            temp_dir,
            base_name,
            detector_name
        )
        
        # 結果の検証
        assert 'detection_result' in result_paths
        assert 'evaluation_result' in result_paths
        
        # ファイルが作成されたことを確認
        detection_path = result_paths['detection_result']
        evaluation_path = result_paths['evaluation_result']
        
        assert os.path.exists(detection_path)
        assert os.path.exists(evaluation_path)
        
        # ファイルの内容を検証
        with open(detection_path, 'r', encoding='utf-8') as f:
            loaded_detection = json.load(f)
            # NumPy配列はリストとして保存される
            assert "intervals" in loaded_detection
            assert "note_pitches" in loaded_detection
            
        with open(evaluation_path, 'r', encoding='utf-8') as f:
            loaded_evaluation = json.load(f)
            assert loaded_evaluation["detector_name"] == sample_evaluation_result["detector_name"] 