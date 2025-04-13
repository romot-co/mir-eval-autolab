# tests/unit/evaluation/test_grid_search_core.py
import pytest
import numpy as np
import pandas as pd
import os
import yaml
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call, mock_open, ANY
import datetime

# テスト対象モジュールをインポート
from src.evaluation.grid_search.core import (
    create_grid_config,
    _save_results_to_csv,
    run_grid_search,
)

# 依存する可能性のある例外等もインポート
from src.utils.exception_utils import ConfigError, FileError

# --- Fixtures ---


@pytest.fixture
def temp_dir():
    """一時ディレクトリを提供するフィクスチャ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_dependencies():
    """grid_search.core の依存関係をモックするフィクスチャ"""
    with patch(
        "src.evaluation.grid_search.core.yaml.safe_load"
    ) as mock_yaml_load, patch(
        "src.evaluation.grid_search.core.yaml.dump"
    ) as mock_yaml_dump, patch(
        "src.evaluation.grid_search.core.os.makedirs"
    ) as mock_makedirs, patch(
        "src.evaluation.grid_search.core.os.listdir"
    ) as mock_listdir, patch(
        "src.evaluation.grid_search.core.os.path.exists"
    ) as mock_exists, patch(
        "src.evaluation.grid_search.core.get_detector_class"
    ) as mock_get_detector, patch(
        "src.evaluation.grid_search.core.run_evaluation"
    ) as mock_run_eval, patch(
        "src.evaluation.grid_search.core.pd.DataFrame"
    ) as mock_pd_dataframe, patch(
        "src.evaluation.grid_search.core.get_logger"
    ) as mock_get_logger, patch(
        "builtins.open", new_callable=mock_open
    ) as mock_file_open:  # open もモック

        # デフォルトのモック動作設定
        mock_listdir.return_value = ["audio1.wav", "audio2.flac"]
        mock_exists.return_value = True  # ファイルは存在すると仮定
        mock_get_logger.return_value = MagicMock()  # ダミーロガー
        mock_run_eval.return_value = {  # run_evaluation のデフォルト戻り値
            "summary": {
                "TestDetector": {"note": {"f_measure": 0.8}}
            },  # detector 名キーの下にメトリクス
            "all_results": [],  # 個別結果は省略
            "status": "success",  # 成功ステータス
        }
        mock_df_instance = MagicMock()
        mock_pd_dataframe.return_value = mock_df_instance

        yield {
            "yaml_load": mock_yaml_load,
            "yaml_dump": mock_yaml_dump,
            "makedirs": mock_makedirs,
            "listdir": mock_listdir,
            "exists": mock_exists,
            "get_detector": mock_get_detector,
            "run_eval": mock_run_eval,
            "pd_dataframe": mock_pd_dataframe,
            "df_instance": mock_df_instance,
            "get_logger": mock_get_logger,
            "logger": mock_get_logger.return_value,
            "open": mock_file_open,
        }


# --- Test create_grid_config ---


def test_create_grid_config_basic(mock_dependencies, temp_dir):
    """create_grid_configが設定辞書を作成し、YAMLファイルに保存するテスト"""
    detector_name = "MyDetector"
    audio_dir = str(temp_dir / "audio")
    reference_dir = str(temp_dir / "ref")
    output_path = str(temp_dir / "grid_conf.yaml")
    param_grid = {"thresh": [0.5, 0.6], "win": [1024]}
    save_plots = False

    # 関数実行
    config_dict = create_grid_config(
        detector_name, audio_dir, reference_dir, output_path, param_grid, save_plots
    )

    # 戻り値の辞書内容を確認
    assert config_dict["detector"] == detector_name
    assert config_dict["audio_dir"] == audio_dir
    assert config_dict["reference_dir"] == reference_dir
    assert config_dict["param_grid"] == param_grid
    assert config_dict["save_plots"] is False

    # yaml.dump が呼ばれたか確認
    mock_dependencies["open"].assert_called_once_with(
        output_path, "w", encoding="utf-8"
    )
    mock_dependencies["yaml_dump"].assert_called_once()
    # dump の第一引数（データ）と第二引数（ファイルハンドル）を検証
    dump_args, dump_kwargs = mock_dependencies["yaml_dump"].call_args
    assert dump_args[0] == config_dict
    assert dump_args[1] == mock_dependencies["open"].return_value  # ファイルハンドル
    assert dump_kwargs.get("allow_unicode") is True
    assert dump_kwargs.get("default_flow_style") is False


def test_create_grid_config_no_output_path(mock_dependencies):
    """create_grid_configでoutput_pathがNoneの場合、ファイル書き込みが行われないテスト"""
    config_dict = create_grid_config("Det", "a", "r", None, {"p": [1]})
    assert config_dict is not None
    mock_dependencies["open"].assert_not_called()
    mock_dependencies["yaml_dump"].assert_not_called()


# --- Test _save_results_to_csv ---


def test_save_results_to_csv_basic(mock_dependencies, temp_dir):
    """_save_results_to_csvが結果リストをCSVに正しく保存するテスト"""
    results = [
        {
            "run_id": "r1",
            "execution_time": 1.1,
            "params": {"p1": 0.1, "p2": "a"},
            "metrics": {
                "note": {"f_measure": 0.8, "recall": 0.7},
                "onset": {"precision": 0.9},
            },
        },
        {
            "run_id": "r2",
            "execution_time": 1.2,
            "params": {"p1": 0.2, "p2": "b"},
            "metrics": {
                "note": {"f_measure": 0.85, "recall": 0.75},
                "onset": {"precision": 0.95},
            },
        },
        {
            "run_id": "r3",
            "execution_time": 1.3,
            "params": {"p1": 0.3, "p2": "c"},
            "metrics": {"note": {"f_measure": 0.7}},
        },  # onset がないケース
    ]
    output_csv_path = str(temp_dir / "summary.csv")

    # 関数実行
    _save_results_to_csv(results, output_csv_path)

    # pandas.DataFrame が呼ばれたか確認
    mock_dependencies["pd_dataframe"].assert_called_once()
    # DataFrame に渡されたデータを確認 (データ構造がフラット化されているか)
    call_args, _ = mock_dependencies["pd_dataframe"].call_args
    passed_data = call_args[0]
    assert len(passed_data) == 3
    # 1行目のデータを確認
    assert passed_data[0]["run_id"] == "r1"
    assert passed_data[0]["execution_time"] == 1.1
    assert passed_data[0]["param_p1"] == 0.1
    assert passed_data[0]["param_p2"] == "a"
    assert passed_data[0]["note.f_measure"] == 0.8
    assert passed_data[0]["note.recall"] == 0.7
    assert passed_data[0]["onset.precision"] == 0.9
    # 3行目のデータを確認 (onset.precision が欠損しているはず)
    assert passed_data[2]["run_id"] == "r3"
    assert passed_data[2]["note.f_measure"] == 0.7
    assert "onset.precision" not in passed_data[2] or pd.isna(
        passed_data[2]["onset.precision"]
    )

    # DataFrame.to_csv が呼ばれたか確認
    mock_dependencies["df_instance"].to_csv.assert_called_once_with(
        output_csv_path, index=False
    )


def test_save_results_to_csv_empty(mock_dependencies, temp_dir):
    """_save_results_to_csvが空の結果リストを処理するテスト"""
    output_csv_path = str(temp_dir / "empty.csv")
    _save_results_to_csv([], output_csv_path)
    mock_dependencies["pd_dataframe"].assert_called_once_with(
        []
    )  # 空リストで DataFrame が呼ばれる
    mock_dependencies["df_instance"].to_csv.assert_called_once_with(
        output_csv_path, index=False
    )


# --- Test run_grid_search ---


def test_run_grid_search_basic(mock_dependencies, temp_dir):
    """run_grid_searchの基本フロー（成功ケース）をテストします"""
    config_path = str(temp_dir / "config.yaml")
    grid_config_path = str(temp_dir / "grid.yaml")
    output_dir = str(temp_dir / "output")
    best_metric = "note.f_measure"

    # モックの設定
    config_data = {
        "audio_dir": str(temp_dir / "a"),
        "reference_dir": str(temp_dir / "r"),
        "evaluation": {"tolerance_onset": 0.06},
    }
    grid_config_data = {
        "detector": "TestDetector",
        "param_grid": {"p1": [1, 2], "p2": ["x"]},
    }
    mock_dependencies["yaml_load"].side_effect = [config_data, grid_config_data]
    mock_dependencies["get_detector"].return_value = MagicMock()  # ダミー検出器クラス

    # run_evaluation の戻り値をパラメータごとに変える
    run_eval_results = [
        {
            "summary": {"TestDetector": {"note": {"f_measure": 0.8}}},
            "status": "success",
        },  # p1=1, p2='x'
        {
            "summary": {"TestDetector": {"note": {"f_measure": 0.9}}},
            "status": "success",
        },  # p1=2, p2='x'
    ]
    mock_dependencies["run_eval"].side_effect = run_eval_results

    # 関数実行
    result_data = run_grid_search(
        config_path, grid_config_path, output_dir, best_metric
    )

    # 検証
    # ディレクトリ作成、設定読み込み
    mock_dependencies["makedirs"].assert_called_once_with(output_dir, exist_ok=True)
    assert mock_dependencies["yaml_load"].call_count == 2
    # ファイルリスト取得（listdir, exists）
    mock_dependencies["listdir"].assert_called_once_with(config_data["audio_dir"])
    assert mock_dependencies["exists"].call_count > 0  # ref ファイルの存在確認
    # get_detector_class 呼び出し
    mock_dependencies["get_detector"].assert_called_once_with(
        grid_config_data["detector"]
    )
    # run_evaluation 呼び出し
    assert mock_dependencies["run_eval"].call_count == 2  # 2つのパラメータ組み合わせ
    calls = mock_dependencies["run_eval"].call_args_list
    # 呼び出し引数を確認 (detector_params)
    assert calls[0][1]["detector_names"] == ["TestDetector"]
    assert calls[0][1]["detector_params"] == {"TestDetector": {"p1": 1, "p2": "x"}}
    assert (
        calls[0][1]["evaluator_config"] == config_data["evaluation"]
    )  # 設定が渡されるか
    assert calls[1][1]["detector_params"] == {"TestDetector": {"p1": 2, "p2": "x"}}
    # 結果の保存
    assert mock_dependencies["open"].call_count >= 2  # JSON と CSV
    assert (
        mock_dependencies["yaml_dump"].call_count == 0
    )  # YAML dump は create_grid_config
    assert mock_dependencies["pd_dataframe"].call_count == 1  # CSV 用
    assert mock_dependencies["df_instance"].to_csv.call_count == 1
    # JSON dump (NumpyEncoder 付き) の呼び出しを確認 - テスト構造に合わせて修正
    # json_dump_call = next(c for c in mock_dependencies['open'].mock_calls if c[1][0].endswith('.json')) # JSON ファイルへの open を探す
    # write_call_args = mock_dependencies['open'].return_value.write.call_args_list  # write の引数を取得

    # 戻り値の確認
    assert "best_result" in result_data
    assert result_data["best_result"]["params"] == {
        "p1": 2,
        "p2": "x",
    }  # f_measure が高い方
    assert result_data["best_result"]["metrics"]["note"]["f_measure"] == 0.9
    assert "all_results" in result_data
    assert len(result_data["all_results"]) == 2
    # ロガーの呼び出し確認
    mock_dependencies["logger"].info.assert_any_call(
        "グリッドサーチを開始: 2個のパラメータ組み合わせを評価"
    )
    mock_dependencies["logger"].info.assert_any_call(
        "最適なパラメータが見つかりました (Metric: note.f_measure): {'p1': 2, 'p2': 'x'} (Score: 0.9000)"
    )


def test_run_grid_search_config_load_error(mock_dependencies, temp_dir):
    """run_grid_searchで設定ファイルの読み込みに失敗した場合のテスト"""
    mock_dependencies["yaml_load"].side_effect = yaml.YAMLError("Cannot parse YAML")
    with pytest.raises(yaml.YAMLError):
        run_grid_search(
            str(temp_dir / "c.yaml"), str(temp_dir / "g.yaml"), str(temp_dir / "o")
        )
    mock_dependencies["logger"].error.assert_called()


def test_run_grid_search_missing_keys_in_config(mock_dependencies, temp_dir):
    """run_grid_searchで設定ファイルに必要なキーがない場合のテスト"""
    config_data = {"reference_dir": "r"}  # audio_dir がない
    grid_config_data = {"detector": "D", "param_grid": {"p": [1]}}
    mock_dependencies["yaml_load"].side_effect = [config_data, grid_config_data]
    with pytest.raises(ValueError, match="基本設定ファイルに必要なキーがありません"):
        run_grid_search(
            str(temp_dir / "c.yaml"), str(temp_dir / "g.yaml"), str(temp_dir / "o")
        )
    mock_dependencies["logger"].error.assert_called()


def test_run_grid_search_no_audio_files_found(mock_dependencies, temp_dir):
    """run_grid_searchで評価対象の音声ファイルが見つからない場合のテスト"""
    config_data = {
        "audio_dir": str(temp_dir / "a"),
        "reference_dir": str(temp_dir / "r"),
    }
    grid_config_data = {"detector": "D", "param_grid": {"p": [1]}}
    mock_dependencies["yaml_load"].side_effect = [config_data, grid_config_data]
    mock_dependencies["listdir"].return_value = []  # 空のファイルリスト
    mock_dependencies["get_detector"].return_value = MagicMock()

    with pytest.raises(ValueError, match="評価対象の音声ファイルが見つかりません"):
        run_grid_search(
            str(temp_dir / "c.yaml"), str(temp_dir / "g.yaml"), str(temp_dir / "o")
        )
    mock_dependencies[
        "logger"
    ].warning.assert_not_called()  # エラーになるので Warning は出ないはず


def test_run_grid_search_no_ref_files_found(mock_dependencies, temp_dir):
    """run_grid_searchで参照ファイルが見つからない場合のテスト"""
    config_data = {
        "audio_dir": str(temp_dir / "a"),
        "reference_dir": str(temp_dir / "r"),
    }
    grid_config_data = {"detector": "D", "param_grid": {"p": [1]}}
    mock_dependencies["yaml_load"].side_effect = [config_data, grid_config_data]
    mock_dependencies["listdir"].return_value = ["audio1.wav"]
    mock_dependencies["exists"].return_value = False  # 参照ファイルが存在しない
    mock_dependencies["get_detector"].return_value = MagicMock()

    # このケースでは ValueError が発生するはず (audio_files が空になるため)
    with pytest.raises(ValueError, match="評価対象の音声ファイルが見つかりません"):
        run_grid_search(
            str(temp_dir / "c.yaml"), str(temp_dir / "g.yaml"), str(temp_dir / "o")
        )

    # 参照ファイルが見つからない旨の Warning ログが出るはず
    mock_dependencies["logger"].warning.assert_any_call(
        "参照ファイルが見つかりません: audio1.*"
    )


def test_run_grid_search_run_evaluation_fails(mock_dependencies, temp_dir):
    """run_grid_searchでrun_evaluationがエラーを返す場合のテスト"""
    config_data = {
        "audio_dir": str(temp_dir / "a"),
        "reference_dir": str(temp_dir / "r"),
    }
    grid_config_data = {"detector": "D", "param_grid": {"p": [1]}}
    mock_dependencies["yaml_load"].side_effect = [config_data, grid_config_data]
    mock_dependencies["get_detector"].return_value = MagicMock()
    # run_evaluation がエラーを返す
    mock_dependencies["run_eval"].return_value = {
        "status": "error",
        "error": "Eval failed miserably",
    }

    result_data = run_grid_search(
        str(temp_dir / "c.yaml"), str(temp_dir / "g.yaml"), str(temp_dir / "o")
    )

    assert mock_dependencies["run_eval"].call_count == 1
    assert len(result_data["all_results"]) == 1
    assert result_data["all_results"][0]["error"] is True
    assert result_data["all_results"][0]["metrics"]["error"] == "Eval failed miserably"
    # 最適な結果は見つからないはず
    assert result_data["best_result"] is None
    mock_dependencies["logger"].error.assert_any_call(
        "実行 run_1 で評価エラーが発生しました: Eval failed miserably"
    )
    mock_dependencies["logger"].warning.assert_any_call(
        "有効な評価結果が見つからなかったため、最適なパラメータを決定できませんでした。"
    )
    # 結果の保存は行われるはず
    assert mock_dependencies["open"].call_count >= 2  # JSON & CSV


@patch("traceback.format_exc", return_value="Traceback...")  # traceback もモック
def test_run_grid_search_unexpected_exception(
    mock_traceback, mock_dependencies, temp_dir
):
    """run_grid_searchのループ内で予期せぬ例外が発生した場合のテスト"""
    config_data = {
        "audio_dir": str(temp_dir / "a"),
        "reference_dir": str(temp_dir / "r"),
    }
    grid_config_data = {"detector": "D", "param_grid": {"p": [1, 2]}}
    mock_dependencies["yaml_load"].side_effect = [config_data, grid_config_data]
    mock_dependencies["get_detector"].return_value = MagicMock()
    # 2回目の run_evaluation で例外を発生させる
    mock_dependencies["run_eval"].side_effect = [
        {
            "summary": {"TestDetector": {"note": {"f_measure": 0.8}}},
            "status": "success",
        },
        RuntimeError("Unexpected crash"),
    ]

    result_data = run_grid_search(
        str(temp_dir / "c.yaml"), str(temp_dir / "g.yaml"), str(temp_dir / "o")
    )

    assert mock_dependencies["run_eval"].call_count == 2
    assert len(result_data["all_results"]) == 2
    # 1回目の結果は正常
    assert "error" not in result_data["all_results"][0]
    # 2回目の結果はエラー
    assert result_data["all_results"][1]["error"] is True
    assert "Unexpected crash" in result_data["all_results"][1]["metrics"]["error"]
    assert result_data["all_results"][1]["traceback"] == "Traceback..."
    # 最適な結果は1回目のものになる
    assert result_data["best_result"] is not None
    assert result_data["best_result"]["params"] == {"p": 1}
    mock_dependencies["logger"].error.assert_any_call(
        "実行 run_2中に予期せぬエラーが発生しました: Unexpected crash\nTraceback..."
    )


def test_run_grid_search_invalid_best_metric_format(mock_dependencies, temp_dir):
    """run_grid_searchでbest_metricの形式が不正な場合のテスト"""
    config_data = {
        "audio_dir": str(temp_dir / "a"),
        "reference_dir": str(temp_dir / "r"),
    }
    grid_config_data = {"detector": "D", "param_grid": {"p": [1]}}
    mock_dependencies["yaml_load"].side_effect = [config_data, grid_config_data]
    mock_dependencies["get_detector"].return_value = MagicMock()
    # run_evaluation の戻り値に note.f_measure を含める
    mock_dependencies["run_eval"].return_value = {
        "summary": {"TestDetector": {"note": {"f_measure": 0.9}}},
        "status": "success",
    }

    result_data = run_grid_search(
        str(temp_dir / "c.yaml"),
        str(temp_dir / "g.yaml"),
        str(temp_dir / "o"),
        best_metric="invalid-format",  # 不正な形式
    )

    mock_dependencies["logger"].warning.assert_any_call(
        "不正な best_metric 形式: invalid-format。'category.metric' 形式で指定してください。デフォルトの 'note.f_measure' を使用します。"
    )
    # デフォルトの 'note.f_measure' で最適値が選ばれるはず
    assert result_data["best_result"]["metrics"]["note"]["f_measure"] == 0.9
    assert result_data["best_metric"] == "note.f_measure"  # 使用されたメトリックも確認


def test_run_grid_search_metric_not_found_in_results(mock_dependencies, temp_dir):
    """run_grid_searchで指定されたbest_metricが結果に含まれない場合のテスト"""
    config_data = {
        "audio_dir": str(temp_dir / "a"),
        "reference_dir": str(temp_dir / "r"),
    }
    grid_config_data = {"detector": "D", "param_grid": {"p": [1]}}
    mock_dependencies["yaml_load"].side_effect = [config_data, grid_config_data]
    mock_dependencies["get_detector"].return_value = MagicMock()
    # run_evaluation の戻り値に指定したメトリックが含まれない
    mock_dependencies["run_eval"].return_value = {
        "summary": {"TestDetector": {"onset": {"precision": 0.9}}},
        "status": "success",
    }

    result_data = run_grid_search(
        str(temp_dir / "c.yaml"),
        str(temp_dir / "g.yaml"),
        str(temp_dir / "o"),
        best_metric="note.f_measure",  # 結果に含まれないメトリック
    )

    # 最適な結果は見つからないはず
    assert result_data["best_result"] is None
    mock_dependencies["logger"].warning.assert_any_call(
        "有効な評価結果が見つからなかったため、最適なパラメータを決定できませんでした。"
    )


def test_run_grid_search_save_results_error(mock_dependencies, temp_dir):
    """run_grid_searchの結果保存中にエラーが発生した場合のテスト"""
    config_data = {
        "audio_dir": str(temp_dir / "a"),
        "reference_dir": str(temp_dir / "r"),
    }
    grid_config_data = {"detector": "D", "param_grid": {"p": [1]}}
    mock_dependencies["yaml_load"].side_effect = [config_data, grid_config_data]
    mock_dependencies["get_detector"].return_value = MagicMock()
    mock_dependencies["run_eval"].return_value = {
        "summary": {"TestDetector": {"note": {"f_measure": 0.9}}},
        "status": "success",
    }

    # open を条件付きでモック: 設定ファイル読み込み時は正常に動作し、
    # 結果保存時（jsonやcsvファイル）のみエラーを発生させる
    original_open = open

    def side_effect_open(*args, **kwargs):
        if args and isinstance(args[0], str):
            # 設定ファイル読み込みは許可
            if args[0].endswith((".yaml", ".yml")):
                return mock_open().return_value
            # 結果保存ファイルはエラー
            elif args[0].endswith((".json", ".csv")) or args[1] == "w":
                raise IOError("Cannot write results")
        return mock_open().return_value

    mock_dependencies["open"].side_effect = side_effect_open

    # run_grid_search を実行（エラーはキャッチされるはず）
    result_data = run_grid_search(
        str(temp_dir / "c.yaml"), str(temp_dir / "g.yaml"), str(temp_dir / "o")
    )

    # 戻り値は存在するはず
    assert result_data is not None
    assert "best_result" in result_data  # 最適値の計算は行われる

    # エラーログが出力されるはず
    mock_dependencies["logger"].error.assert_any_call(
        "結果ファイルの保存中にエラーが発生しました: Cannot write results"
    )
