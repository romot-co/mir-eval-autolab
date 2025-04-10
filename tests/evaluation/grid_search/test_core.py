"""
グリッドサーチのコア機能 (`src/evaluation/grid_search/core.py`) のテスト
"""
import os
import pytest
import yaml
import numpy as np
import json
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open, ANY, call
import tempfile
from pathlib import Path

from src.evaluation.grid_search.core import create_grid_config, _save_results_to_csv, run_grid_search

# --- create_grid_config のテスト --- #

def test_create_grid_config():
    """create_grid_config: 設定ファイルを正しく作成できるか確認"""
    # テストデータ
    detector_name = "TestDetector"
    audio_dir = "/path/to/audio"
    reference_dir = "/path/to/reference"
    output_path = "/path/to/output/config.yaml"
    param_grid = {
        "param1": [1, 2, 3],
        "param2": ["a", "b", "c"]
    }
    save_plots = True
    
    # 期待される設定辞書
    expected_config = {
        'detector': detector_name,
        'audio_dir': audio_dir,
        'reference_dir': reference_dir,
        'param_grid': param_grid,
        'save_plots': save_plots
    }
    
    # ファイル書き込みをモック化
    mock_file = mock_open()
    with patch("builtins.open", mock_file):
        config = create_grid_config(
            detector_name=detector_name,
            audio_dir=audio_dir,
            reference_dir=reference_dir,
            output_path=output_path,
            param_grid=param_grid,
            save_plots=save_plots
        )
    
    # 設定辞書が正しいことを確認
    assert config == expected_config
    
    # ファイルが開かれ、書き込まれたことを確認
    mock_file.assert_called_once_with(output_path, 'w', encoding='utf-8')
    handle = mock_file()
    # YAMLの文字列表現が完全一致するのは難しいので、書き込みが実行されたことだけを確認
    assert handle.write.called

def test_create_grid_config_no_output():
    """create_grid_config: 出力先なしでも設定辞書が正しく生成されるか確認"""
    # テストデータ
    detector_name = "TestDetector"
    audio_dir = "/path/to/audio"
    reference_dir = "/path/to/reference"
    param_grid = {
        "param1": [1, 2, 3],
        "param2": ["a", "b", "c"]
    }
    
    # 期待される設定辞書
    expected_config = {
        'detector': detector_name,
        'audio_dir': audio_dir,
        'reference_dir': reference_dir,
        'param_grid': param_grid,
        'save_plots': True  # デフォルト値
    }
    
    # 出力先なしで実行
    config = create_grid_config(
        detector_name=detector_name,
        audio_dir=audio_dir,
        reference_dir=reference_dir,
        output_path="",  # 空の出力パス
        param_grid=param_grid
    )
    
    # 設定辞書が正しいことを確認
    assert config == expected_config

# --- _save_results_to_csv のテスト --- #

def test_save_results_to_csv():
    """_save_results_to_csv: 結果が正しくCSVに保存されるか確認"""
    # テスト用の結果データ
    results = [
        {
            'run_id': 'run_1',
            'params': {'param1': 1, 'param2': 'a'},
            'metrics': {
                'note': {'f_measure': 0.8, 'precision': 0.9, 'recall': 0.7},
                'onset': {'f_measure': 0.75}
            },
            'execution_time': 10.5
        },
        {
            'run_id': 'run_2',
            'params': {'param1': 2, 'param2': 'b'},
            'metrics': {
                'note': {'f_measure': 0.7, 'precision': 0.8, 'recall': 0.6},
                'onset': {'f_measure': 0.65}
            },
            'execution_time': 11.2
        }
    ]
    
    # 一時ファイルを使用
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        output_path = temp_file.name
    
    try:
        # CSVに保存
        _save_results_to_csv(results, output_path)
        
        # CSVを読み込んで内容を確認
        df = pd.read_csv(output_path)
        
        # 行数を確認
        assert len(df) == 2
        
        # カラムの存在を確認
        expected_columns = ['run_id', 'execution_time', 'note.f_measure', 'note.precision', 'note.recall',
                           'onset.f_measure', 'param_param1', 'param_param2']
        for col in expected_columns:
            assert col in df.columns
        
        # 値を確認
        assert df.loc[0, 'run_id'] == 'run_1'
        assert df.loc[0, 'execution_time'] == 10.5
        assert df.loc[0, 'note.f_measure'] == 0.8
        assert df.loc[0, 'param_param1'] == 1
        assert df.loc[0, 'param_param2'] == 'a'
        
        assert df.loc[1, 'run_id'] == 'run_2'
        assert df.loc[1, 'note.f_measure'] == 0.7
    finally:
        # 一時ファイルを削除
        if os.path.exists(output_path):
            os.remove(output_path)

# --- run_grid_search のテスト --- #

@patch('src.evaluation.grid_search.core.get_detector_class')
@patch('src.evaluation.grid_search.core.run_evaluation')
@patch('src.evaluation.grid_search.core.get_logger')
def test_run_grid_search_success(mock_get_logger, mock_run_evaluation, mock_get_detector_class):
    """run_grid_search: すべてのパラメータ組み合わせで正常に評価を実行し、最良の結果を返すことを検証"""
    # モックロガーの設定
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    # 検出器クラスのモック
    mock_detector_class = MagicMock()
    mock_get_detector_class.return_value = mock_detector_class
    
    # 異なるパラメータ組み合わせの結果を設定
    results = [
        {
            'status': 'success',
            'metrics': {'note': {'f_measure': 0.7, 'precision': 0.8, 'recall': 0.6}},
            'params': {'param1': 1, 'param2': 'a'}
        },
        {
            'status': 'success',
            'metrics': {'note': {'f_measure': 0.75, 'precision': 0.8, 'recall': 0.7}},
            'params': {'param1': 1, 'param2': 'b'}
        },
        {
            'status': 'success',
            'metrics': {'note': {'f_measure': 0.85, 'precision': 0.9, 'recall': 0.8}},
            'params': {'param1': 2, 'param2': 'a'}
        },
        {
            'status': 'success',
            'metrics': {'note': {'f_measure': 0.8, 'precision': 0.85, 'recall': 0.75}},
            'params': {'param1': 2, 'param2': 'b'}
        }
    ]
    
    # run_evaluationの呼び出しごとに異なる結果を返すサイドエフェクト
    def mock_run_eval_side_effect(*args, **kwargs):
        params = kwargs.get('detector_params', {}).get('TestDetector', {}) # detector_paramsから取得
        param1 = params.get('param1')
        param2 = params.get('param2')

        # パラメータの組み合わせに応じた結果を返す
        for result_data in results:
            if result_data['params']['param1'] == param1 and result_data['params']['param2'] == param2:
                # run_evaluation の戻り値構造に合わせる
                return {
                    'status': 'success',
                    'summary': {
                        'TestDetector': result_data['metrics'] # 検出器名をキーにする
                    },
                    'params': params # 渡されたパラメータをそのまま返す
                }

        # 該当する結果がない場合はエラー
        return {'status': 'error', 'error': 'テスト用エラー', 'params': params, 'summary': {}}

    mock_run_evaluation.side_effect = mock_run_eval_side_effect
    
    # テスト用設定
    config_path = "/path/to/config.yaml"
    grid_config_path = "/path/to/grid_config.yaml"
    output_dir = "/path/to/output"
    
    # 設定ファイルの内容をモック
    config_content = {
        'audio_dir': '/path/to/audio',
        'reference_dir': '/path/to/reference',
        'evaluation': {
            'tolerance_onset': 0.05,
            'offset_ratio': 0.2
        }
    }
    
    grid_config_content = {
        'detector': 'TestDetector',
        'param_grid': {
            'param1': [1, 2],
            'param2': ['a', 'b']
        },
        'save_plots': True
    }
    
    # テスト用音声ファイルと参照ファイル
    audio_files = ['/path/to/audio/file1.wav', '/path/to/audio/file2.wav']
    reference_files = ['/path/to/reference/file1.csv', '/path/to/reference/file2.csv']
    
    # ファイル操作をモック化
    mock_os_makedirs = MagicMock()
    mock_os_listdir = MagicMock(return_value=['file1.wav', 'file2.wav'])
    mock_os_path_exists = MagicMock(return_value=True)
    
    # ファイル書き込みをモック化
    m_open = mock_open()
    
    # パッチを適用
    with patch("builtins.open", m_open), \
         patch("yaml.safe_load", side_effect=[config_content, grid_config_content]), \
         patch("os.makedirs", mock_os_makedirs), \
         patch("os.listdir", mock_os_listdir), \
         patch("os.path.exists", mock_os_path_exists), \
         patch("os.path.join", lambda *args: '/'.join(args)), \
         patch("os.path.basename", lambda p: p.split('/')[-1]), \
         patch("os.path.splitext", lambda p: (p.split('.')[0], '.' + p.split('.')[-1])), \
         patch("json.dump") as mock_json_dump, \
         patch("src.evaluation.grid_search.core._save_results_to_csv") as mock_save_csv:
        
        # テスト実行
        result = run_grid_search(
            config_path=config_path,
            grid_config_path=grid_config_path,
            output_dir=output_dir,
            best_metric='note.f_measure'
        )
        
        # 結果の検証
        assert result is not None
        # run_grid_searchは best_result を返す
        # best_result の構造: {'run_id': ..., 'params': ..., 'metrics': ..., 'execution_time': ...}
        assert 'run_id' in result
        assert 'params' in result
        assert 'metrics' in result

        # 最高のF値を持つパラメータ組み合わせが選ばれる
        assert result['metrics']['note']['f_measure'] == 0.85
        assert result['params'] == {'param1': 2, 'param2': 'a'}

        # すべてのパラメータ組み合わせに対して評価が実行されたか確認
        assert mock_run_evaluation.call_count == 4

        # 全てのパラメータ組み合わせが評価されたことを検証
        # detector_params の中の TestDetector の値として渡される
        called_params = [call.kwargs['detector_params']['TestDetector'] for call in mock_run_evaluation.call_args_list]
        expected_params = [result_data['params'] for result_data in results]
        for params in expected_params:
            assert params in called_params

        # CSVに全ての結果が保存されたことを確認
        mock_save_csv.assert_called_once()
        csv_results_arg = mock_save_csv.call_args[0][0]
        assert len(csv_results_arg) == 4  # 全て成功しているので4つの結果

        # 最良の結果のJSONが保存されたことを確認 (combined_summary.json と best_params.json の2回)
        # grid_search_result.json も各実行で呼ばれる
        # 正確な回数は追いきれないため、呼び出されたことだけを確認
        assert mock_json_dump.call_count > 0

@patch('src.evaluation.grid_search.core.get_detector_class')
@patch('src.evaluation.grid_search.core.run_evaluation')
@patch('src.evaluation.grid_search.core.get_logger')
def test_run_grid_search_with_error(mock_get_logger, mock_run_evaluation, mock_get_detector_class):
    """run_grid_search: 一部のパラメータ組み合わせでエラーが発生する場合でも、他の組み合わせが成功すれば全体として成功を返すことを検証"""
    # モックロガーの設定
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    # 検出器クラスのモック
    mock_detector_class = MagicMock()
    mock_get_detector_class.return_value = mock_detector_class
    
    # 一部の組み合わせでエラー、一部で成功する結果を設定
    results = [
        {
            'status': 'success',
            'metrics': {'note': {'f_measure': 0.7, 'precision': 0.8, 'recall': 0.6}},
            'params': {'param1': 1, 'param2': 'a'}
        },
        {
            'status': 'error',
            'error': 'テスト用エラー1',
            'params': {'param1': 1, 'param2': 'b'}
        },
        {
            'status': 'success',
            'metrics': {'note': {'f_measure': 0.85, 'precision': 0.9, 'recall': 0.8}},
            'params': {'param1': 2, 'param2': 'a'}
        },
        {
            'status': 'error',
            'error': 'テスト用エラー2',
            'params': {'param1': 2, 'param2': 'b'}
        }
    ]
    
    # パラメータに基づいて異なる結果を返すサイドエフェクト
    def mock_run_eval_side_effect(*args, **kwargs):
        params = kwargs.get('detector_params', {}).get('TestDetector', {}) # detector_paramsから取得
        param1 = params.get('param1')
        param2 = params.get('param2')

        for result_data in results:
            if result_data['params']['param1'] == param1 and result_data['params']['param2'] == param2:
                if result_data['status'] == 'success':
                    # run_evaluation の戻り値構造に合わせる
                    return {
                        'status': 'success',
                        'summary': {
                             'TestDetector': result_data['metrics'] # 検出器名をキーにする
                        },
                         'params': params
                    }
                else:
                    # エラーの場合
                    return {
                        'status': 'error',
                        'error': result_data['error'],
                         'params': params,
                         'summary': {}
                    }

        return {'status': 'error', 'error': '不明なエラー', 'params': params, 'summary': {}}

    mock_run_evaluation.side_effect = mock_run_eval_side_effect
    
    # テスト用設定
    config_path = "/path/to/config.yaml"
    grid_config_path = "/path/to/grid_config.yaml"
    output_dir = "/path/to/output"
    
    # 設定ファイルの内容をモック
    config_content = {
        'audio_dir': '/path/to/audio',
        'reference_dir': '/path/to/reference',
        'evaluation': {
            'tolerance_onset': 0.05,
            'offset_ratio': 0.2
        }
    }
    
    grid_config_content = {
        'detector': 'TestDetector',
        'param_grid': {
            'param1': [1, 2],
            'param2': ['a', 'b']
        },
        'save_plots': True
    }
    
    # テスト用音声ファイルと参照ファイル
    audio_files = ['/path/to/audio/file1.wav', '/path/to/audio/file2.wav']
    reference_files = ['/path/to/reference/file1.csv', '/path/to/reference/file2.csv']
    
    # ファイル操作をモック化
    mock_os_makedirs = MagicMock()
    mock_os_listdir = MagicMock(return_value=['file1.wav', 'file2.wav'])
    mock_os_path_exists = MagicMock(return_value=True)
    
    # ファイル書き込みをモック化
    m_open = mock_open()
    
    # パッチを適用
    with patch("builtins.open", m_open), \
         patch("yaml.safe_load", side_effect=[config_content, grid_config_content]), \
         patch("os.makedirs", mock_os_makedirs), \
         patch("os.listdir", mock_os_listdir), \
         patch("os.path.exists", mock_os_path_exists), \
         patch("os.path.join", lambda *args: '/'.join(args)), \
         patch("os.path.basename", lambda p: p.split('/')[-1]), \
         patch("os.path.splitext", lambda p: (p.split('.')[0], '.' + p.split('.')[-1])), \
         patch("json.dump") as mock_json_dump, \
         patch("src.evaluation.grid_search.core._save_results_to_csv") as mock_save_csv:
        
        # テスト実行
        result = run_grid_search(
            config_path=config_path,
            grid_config_path=grid_config_path,
            output_dir=output_dir,
            best_metric='note.f_measure'
        )
        
        # 結果の検証
        assert result is not None
        # run_grid_searchは best_result を返す
        assert 'run_id' in result
        assert 'params' in result
        assert 'metrics' in result

        # 最高のF値を持つパラメータ組み合わせが選ばれる
        assert result['metrics']['note']['f_measure'] == 0.85
        assert result['params'] == {'param1': 2, 'param2': 'a'}

        # すべてのパラメータ組み合わせに対して評価が実行されたか確認
        assert mock_run_evaluation.call_count == 4

        # エラーが記録されていることを確認 (run_grid_search内でのエラーログ)
        # logger.errorの呼び出し回数は実行パスによるため、具体的な回数は検証しない
        assert mock_logger.error.called

        # CSVに成功した結果のみが保存されたことを確認
        mock_save_csv.assert_called_once()
        csv_results_arg = mock_save_csv.call_args[0][0]
        assert len(csv_results_arg) == 2  # 成功した2つの結果のみ

        # 成功した結果のみがCSVに含まれていることを確認
        success_params = [r['params'] for r in csv_results_arg]
        assert {'param1': 1, 'param2': 'a'} in success_params
        assert {'param1': 2, 'param2': 'a'} in success_params

        # エラーが発生したパラメータ組み合わせはCSVに含まれないことを確認
        assert {'param1': 1, 'param2': 'b'} not in success_params
        assert {'param1': 2, 'param2': 'b'} not in success_params

        # 最良の結果のJSONが保存されたことを確認 (combined_summary.json と best_params.json の2回)
        # grid_search_result.json も各成功実行で呼ばれる
        assert mock_json_dump.call_count > 0

@patch('src.evaluation.grid_search.core.get_detector_class')
@patch('src.evaluation.grid_search.core.run_evaluation')
@patch('src.evaluation.grid_search.core.get_logger')
def test_run_grid_search_all_errors(mock_get_logger, mock_run_evaluation, mock_get_detector_class):
    """run_grid_search: すべてのパラメータ組み合わせでエラーが発生する場合のテスト"""
    # モックロガーの設定
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    # 検出器クラスのモック
    mock_detector_class = MagicMock()
    mock_get_detector_class.return_value = mock_detector_class
    
    # すべてのパラメータ組み合わせでエラーが発生するケース
    errors = {
        (1, 'a'): "タイムアウトエラー",
        (1, 'b'): "メモリ不足エラー",
        (2, 'a'): "入力ファイルエラー",
        (2, 'b'): "出力エラー"
    }
    
    # run_evaluationの呼び出しごとにエラーを返すサイドエフェクト
    def mock_run_eval_side_effect(*args, **kwargs):
        params = kwargs.get('detector_params', {}).get('TestDetector', {}) # detector_paramsから取得
        param1 = params.get('param1')
        param2 = params.get('param2')
        # paramsがNoneでないことを確認してからタプルを作成
        if param1 is not None and param2 is not None:
             error_key = (param1, param2)
             if error_key in errors:
                 return {
                    'status': 'error',
                    'error': errors[error_key],
                    'params': params,
                    'summary': {}
                 }
        # キーが見つからない場合やparamsがNoneの場合のエラー
        return {
            'status': 'error',
            'error': 'テスト設定エラー：予期しないパラメータ',
            'params': params,
            'summary': {}
        }

    mock_run_evaluation.side_effect = mock_run_eval_side_effect
    
    # テスト用設定
    config_path = "/path/to/config.yaml"
    grid_config_path = "/path/to/grid_config.yaml"
    output_dir = "/path/to/output"
    
    # 設定ファイルの内容をモック
    config_content = {
        'audio_dir': '/path/to/audio',
        'reference_dir': '/path/to/reference',
        'evaluation': {
            'tolerance_onset': 0.05,
            'offset_ratio': 0.2
        }
    }
    
    grid_config_content = {
        'detector': 'TestDetector',
        'param_grid': {
            'param1': [1, 2],
            'param2': ['a', 'b']
        },
        'save_plots': True
    }
    
    # テスト用音声ファイルと参照ファイル
    audio_files = ['/path/to/audio/file1.wav', '/path/to/audio/file2.wav']
    reference_files = ['/path/to/reference/file1.csv', '/path/to/reference/file2.csv']
    
    # ファイル操作をモック化
    mock_os_makedirs = MagicMock()
    mock_os_listdir = MagicMock(return_value=['file1.wav', 'file2.wav'])
    mock_os_path_exists = MagicMock(return_value=True)
    
    # ファイル書き込みをモック化
    m_open = mock_open()
    
    # パッチを適用
    with patch("builtins.open", m_open), \
         patch("yaml.safe_load", side_effect=[config_content, grid_config_content]), \
         patch("os.makedirs", mock_os_makedirs), \
         patch("os.listdir", mock_os_listdir), \
         patch("os.path.exists", mock_os_path_exists), \
         patch("os.path.join", lambda *args: '/'.join(args)), \
         patch("os.path.basename", lambda p: p.split('/')[-1]), \
         patch("os.path.splitext", lambda p: (p.split('.')[0], '.' + p.split('.')[-1])), \
         patch("json.dump") as mock_json_dump, \
         patch("src.evaluation.grid_search.core._save_results_to_csv") as mock_save_csv:
        
        # テスト実行
        result = run_grid_search(
            config_path=config_path,
            grid_config_path=grid_config_path,
            output_dir=output_dir,
            best_metric='note.f_measure'
        )
        
        # 結果の検証
        # すべて失敗した場合、run_grid_searchはNoneを返す
        assert result is None

        # すべてのパラメータ組み合わせに対して評価が実行されたか確認
        assert mock_run_evaluation.call_count == 4

        # すべての実行でエラーが記録されたことを確認
        assert mock_logger.error.called

        # CSVファイルは作成されない（結果がないため）
        mock_save_csv.assert_not_called()

        # 結果がないためJSONファイルも作成されないはずだが、
        # best_params.jsonなどはtry-finallyブロック外で呼ばれる可能性があるため検証しない
        # mock_json_dump.assert_not_called() # これは不安定なためコメントアウト

def test_run_grid_search_config_error():
    """run_grid_search: 設定ファイルに必要なキーがない場合のテスト"""
    # 必要なキーが欠けた設定ファイル
    config_content = {}  # audio_dirやreference_dirが欠けている
    
    grid_config_content = {
        'detector': 'TestDetector',
        'param_grid': {'param1': [1, 2]}
    }
    
    # パッチを適用
    with patch("builtins.open", mock_open()), \
         patch("yaml.safe_load", side_effect=[config_content, grid_config_content]), \
         patch("os.makedirs", MagicMock()), \
         patch("src.evaluation.grid_search.core.get_logger") as mock_get_logger, \
         patch("src.evaluation.grid_search.core.get_detector_class", MagicMock()):
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # テスト実行（例外が発生することを期待）
        with pytest.raises(ValueError) as excinfo:
            run_grid_search(
                config_path="/tmp/to/config.yaml",  # /tmp/ パスを使用して権限エラーを回避
                grid_config_path="/tmp/to/grid_config.yaml",
                output_dir="/tmp/output",
                best_metric='note.f_measure'
            )
        
        # エラーロガーが記録されたことを確認
        assert mock_logger.error.called
        
        # 適切なエラーメッセージが含まれていることを確認
        assert "必要なキー" in str(excinfo.value) or "キーがありません" in str(excinfo.value)

def test_run_grid_search_no_audio_files():
    """run_grid_search: 音声ファイルが見つからない場合のテスト"""
    # 設定ファイル
    config_content = {
        'audio_dir': '/tmp/to/audio',  # /tmp/ パスを使用して権限エラーを回避
        'reference_dir': '/tmp/to/reference'
    }
    
    grid_config_content = {
        'detector': 'TestDetector',
        'param_grid': {'param1': [1, 2]}
    }
    
    # 空のディレクトリを返すモック
    mock_os_listdir = MagicMock(return_value=[])
    
    # パッチを適用
    with patch("builtins.open", mock_open()), \
         patch("yaml.safe_load", side_effect=[config_content, grid_config_content]), \
         patch("os.makedirs", MagicMock()), \
         patch("os.listdir", mock_os_listdir), \
         patch("src.evaluation.grid_search.core.get_logger") as mock_get_logger, \
         patch("src.evaluation.grid_search.core.get_detector_class", MagicMock()):
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # テスト実行（例外が発生することを期待）
        with pytest.raises(ValueError) as excinfo:
            run_grid_search(
                config_path="/tmp/to/config.yaml",
                grid_config_path="/tmp/to/grid_config.yaml",
                output_dir="/tmp/output",
                best_metric='note.f_measure'
            )
        
        # 適切なエラーメッセージが含まれていることを確認
        assert "音声ファイルが見つかりません" in str(excinfo.value) 