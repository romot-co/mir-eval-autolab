# tests/unit/evaluation/test_evaluation_io.py
import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call, mock_open, ANY
import logging
import tempfile
import os
import builtins  # builtins をインポート

# テスト対象のモジュールをインポート
from src.evaluation.evaluation_io import (
    save_evaluation_result,
    load_evaluation_result,
    load_multiple_evaluation_results,
    print_summary_statistics,
    create_summary_dataframe,
    save_detection_plot,
    save_detection_and_evaluation_results,
    NumpyEncoder,  # JSONエンコーダもテスト対象にする可能性あり
    FileError,  # 例外クラスもインポート
)

# DetectionResult は evaluation_runner 側で使われるため、ここでは不要かもしれないが、
# save_detection_plot の引数型として使われるためインポート
from src.utils.detection_result import DetectionResult

# --- Fixtures ---


@pytest.fixture
def temp_dir():
    """一時ディレクトリを提供するフィクスチャ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_eval_result():
    """サンプル評価結果データを返すフィクスチャ"""
    return {
        "detector_name": "TestDetector",
        "audio_path": "/path/to/audio.wav",
        "status": "success",
        "evaluation": {
            "note": {"precision": 0.8, "recall": 0.7, "f_measure": np.float32(0.75)},
            "onset": {"precision": 0.9, "recall": 0.8, "f_measure": 0.85},
            "frame_pitch": {"raw_pitch_accuracy": 0.95, "voicing_recall": 0.9},
        },
        "detection_time": 1.234,
        "ref_note_count": 10,
        "est_note_count": 9,
    }


@pytest.fixture
def sample_detection_result_dict():
    """サンプルの検出結果辞書（Numpy配列を含む）を返すフィクスチャ"""
    return {
        "detector_name": "TestDetector",
        "intervals": np.array([[0.1, 0.5], [0.6, 1.0]]),
        "note_pitches": np.array([440.0, 880.0]),
        "frame_times": np.linspace(0, 1, 10, dtype=np.float32),
        "frame_frequencies": np.random.rand(10).astype(np.float64),
        "detection_time": 0.55,
    }


# --- Test NumpyEncoder ---


def test_numpy_encoder_serialization():
    """NumpyEncoderがNumPyの型を正しくJSONシリアライズ可能かテストします"""
    data_with_numpy = {
        "int32": np.int32(10),
        "int64": np.int64(20),
        "float32": np.float32(0.5),
        "float64": np.float64(0.123),
        "bool": np.bool_(True),
        "array": np.array([1, 2, 3]),
        "nested": {"inner_array": np.arange(5)},
        "python_int": 100,
        "python_float": 0.99,
        "python_list": [1, 2, 3],
    }
    expected_json_str = '{"int32": 10, "int64": 20, "float32": 0.5, "float64": 0.123, "bool": true, "array": [1, 2, 3], "nested": {"inner_array": [0, 1, 2, 3, 4]}, "python_int": 100, "python_float": 0.99, "python_list": [1, 2, 3]}'

    # NumpyEncoderを使用してJSON文字列に変換
    json_output = json.dumps(data_with_numpy, cls=NumpyEncoder, sort_keys=True)

    # 期待されるJSON文字列と比較 (順序を無視するため辞書にロードし直して比較)
    assert json.loads(json_output) == json.loads(expected_json_str)


# def test_numpy_encoder_unsupported_type():
#     """NumpyEncoderがサポートされていない型に対してTypeErrorを発生させるかテストします"""
#     # Note: Current NumpyEncoder *does* handle complex numbers.
#     # Test if it correctly serializes complex numbers instead.
#     unsupported_data = {'complex': np.complex128(1+2j)}
#     expected_json = '{"complex": {"real": 1.0, "imag": 2.0}}'
#     json_output = json.dumps(unsupported_data, cls=NumpyEncoder, sort_keys=True)
#     assert json.loads(json_output) == json.loads(expected_json)

# --- Test save_evaluation_result ---


def test_save_evaluation_result_basic(sample_eval_result, temp_dir):
    """save_evaluation_resultがJSONファイルを正しく作成・書き込みするかテストします"""
    output_path = temp_dir / "result.json"
    save_evaluation_result(sample_eval_result, str(output_path))

    # ファイルが存在することを確認
    assert output_path.exists()

    # ファイルの内容を確認
    with open(output_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    # 保存されたデータが元のデータと一致することを確認 (Numpy型は変換されている)
    # np.float32 が float に変換されていることを確認
    assert isinstance(loaded_data["evaluation"]["note"]["f_measure"], float)
    # 元のデータと比較（変換後の型で）
    assert loaded_data["detector_name"] == sample_eval_result["detector_name"]
    assert (
        loaded_data["evaluation"]["note"]["precision"]
        == sample_eval_result["evaluation"]["note"]["precision"]
    )
    assert np.isclose(
        loaded_data["evaluation"]["note"]["f_measure"],
        sample_eval_result["evaluation"]["note"]["f_measure"],
    )


def test_save_evaluation_result_creates_directory(sample_eval_result, temp_dir):
    """save_evaluation_resultが出力ディレクトリが存在しない場合に作成するかテストします"""
    output_path = temp_dir / "new_subdir" / "result.json"
    save_evaluation_result(sample_eval_result, str(output_path))

    assert output_path.exists()
    assert output_path.parent.is_dir()


def test_save_evaluation_result_permission_error(sample_eval_result, temp_dir):
    """save_evaluation_resultで書き込み権限がない場合の動作をテストします"""
    # 読み取り専用ディレクトリを作成
    read_only_dir = temp_dir / "read_only"
    read_only_dir.mkdir(mode=0o555)  # 読み取り + 実行権限のみ
    output_path = read_only_dir / "result.json"

    # PermissionErrorが発生することを期待
    with pytest.raises(PermissionError):
        # try-except で囲むことで、os.makedirs が失敗した場合もテスト可能にする
        try:
            save_evaluation_result(sample_eval_result, str(output_path))
        except PermissionError:
            # ディレクトリ作成で失敗した場合も PermissionError を期待
            raise
        except Exception as e:
            pytest.fail(f"Expected PermissionError, but got {type(e)}")


# --- Test load_evaluation_result ---


def test_load_evaluation_result_basic(sample_eval_result, temp_dir):
    """load_evaluation_resultがJSONファイルを正しく読み込むかテストします"""
    input_path = temp_dir / "result.json"
    # まずテストデータを保存
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(sample_eval_result, f, cls=NumpyEncoder)

    # 関数を実行
    loaded_data = load_evaluation_result(str(input_path))

    # 読み込んだデータが元のデータと一致することを確認
    assert loaded_data["detector_name"] == sample_eval_result["detector_name"]
    assert (
        loaded_data["evaluation"]["note"]["precision"]
        == sample_eval_result["evaluation"]["note"]["precision"]
    )
    assert np.isclose(
        loaded_data["evaluation"]["note"]["f_measure"],
        sample_eval_result["evaluation"]["note"]["f_measure"],
    )


def test_load_evaluation_result_file_not_found(temp_dir):
    """load_evaluation_resultでファイルが存在しない場合の動作をテストします"""
    non_existent_path = temp_dir / "non_existent.json"
    with pytest.raises(FileNotFoundError):
        load_evaluation_result(str(non_existent_path))


def test_load_evaluation_result_invalid_json(temp_dir):
    """load_evaluation_resultでJSON形式が無効な場合の動作をテストします"""
    invalid_json_path = temp_dir / "invalid.json"
    with open(invalid_json_path, "w", encoding="utf-8") as f:
        f.write("this is not json")

    with pytest.raises(json.JSONDecodeError):
        load_evaluation_result(str(invalid_json_path))


# --- Test load_multiple_evaluation_results ---


def test_load_multiple_evaluation_results_basic(sample_eval_result, temp_dir):
    """load_multiple_evaluation_resultsが複数のJSONファイルを正しく読み込むかテストします"""
    results_dir = temp_dir / "results"
    results_dir.mkdir()

    # 複数の評価結果ファイルを作成
    result1_path = results_dir / "file1_evaluation.json"
    result2_path = results_dir / "file2_evaluation.json"
    other_path = results_dir / "other_file.txt"  # 対象外ファイル

    result1_data = sample_eval_result.copy()
    result1_data["audio_path"] = "file1.wav"
    result2_data = sample_eval_result.copy()
    result2_data["audio_path"] = "file2.wav"
    result2_data["evaluation"]["note"]["f_measure"] = 0.9

    save_evaluation_result(result1_data, str(result1_path))
    save_evaluation_result(result2_data, str(result2_path))
    other_path.touch()

    # 関数を実行
    loaded_results = load_multiple_evaluation_results(str(results_dir))

    # 検証
    assert len(loaded_results) == 2
    # 読み込まれた結果のリストに、保存したデータが含まれているか確認（順序は不定）
    loaded_paths = sorted([res["audio_path"] for res in loaded_results])
    assert loaded_paths == ["file1.wav", "file2.wav"]


def test_load_multiple_evaluation_results_custom_pattern(sample_eval_result, temp_dir):
    """load_multiple_evaluation_resultsでカスタムパターンを使用するテスト"""
    results_dir = temp_dir / "results"
    results_dir.mkdir()

    # 異なる名前のファイルを作成
    path1 = results_dir / "result_abc.res"
    path2 = results_dir / "result_xyz.res"
    path3 = results_dir / "result_123.json"  # パターン外

    save_evaluation_result(sample_eval_result, str(path1))
    save_evaluation_result(sample_eval_result, str(path2))
    save_evaluation_result(sample_eval_result, str(path3))

    # カスタムパターンで実行
    loaded_results = load_multiple_evaluation_results(str(results_dir), pattern="*.res")

    # 検証
    assert len(loaded_results) == 2


def test_load_multiple_evaluation_results_empty_dir(temp_dir):
    """load_multiple_evaluation_resultsでディレクトリが空の場合のテスト"""
    empty_dir = temp_dir / "empty_results"
    empty_dir.mkdir()

    loaded_results = load_multiple_evaluation_results(str(empty_dir))

    assert loaded_results == []


def test_load_multiple_evaluation_results_dir_not_found(temp_dir):
    """load_multiple_evaluation_resultsでディレクトリが存在しない場合のテスト"""
    non_existent_dir = temp_dir / "non_existent"

    with pytest.raises(FileError, match="入力パスがディレクトリではありません"):
        load_multiple_evaluation_results(str(non_existent_dir))


# --- Test create_summary_dataframe ---


def test_create_summary_dataframe_basic(sample_eval_result):
    """create_summary_dataframeの基本的な動作をテストします"""
    results_list = [
        sample_eval_result,
        {
            "detector_name": "DetectorB",
            "audio_path": "/path/to/audio2.wav",
            "status": "success",
            "evaluation": {
                "note": {"precision": 0.7, "recall": 0.6, "f_measure": 0.65},
                "frame_pitch": {"raw_pitch_accuracy": 0.90, "voicing_recall": 0.8},
            },
            "detection_time": 2.5,
            "ref_note_count": 15,
            "est_note_count": 12,
        },
    ]

    df = create_summary_dataframe(results_list)

    # データフレームの構造を確認
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    expected_columns = [
        "audio_path",
        "detector_name",
        "detection_time",
        "note_precision",
        "note_recall",
        "note_f_measure",
        "onset_precision",
        "onset_recall",
        "onset_f_measure",  # onsetも展開されるはず
        "frame_pitch_raw_pitch_accuracy",
        "frame_pitch_voicing_recall",
        # フレーム評価の旧互換指標も追加されるはず
        "frame_pitch_precision",
        "frame_pitch_recall",
        "frame_pitch_f_measure",
        "frame_pitch_accuracy",
        "ref_note_count",
        "est_note_count",
    ]
    for col in expected_columns:
        assert col in df.columns

    # データの内容を確認 (一部)
    assert df.loc[0, "detector_name"] == "TestDetector"
    assert df.loc[1, "detector_name"] == "DetectorB"
    assert (
        df.loc[0, "note_f_measure"]
        == sample_eval_result["evaluation"]["note"]["f_measure"]
    )
    assert df.loc[1, "note_f_measure"] == 0.65
    assert (
        df.loc[0, "frame_pitch_raw_pitch_accuracy"]
        == sample_eval_result["evaluation"]["frame_pitch"]["raw_pitch_accuracy"]
    )
    # 旧互換指標が raw_pitch_accuracy と同じ値になっているか確認
    assert (
        df.loc[0, "frame_pitch_precision"]
        == sample_eval_result["evaluation"]["frame_pitch"]["raw_pitch_accuracy"]
    )
    assert (
        df.loc[0, "frame_pitch_recall"]
        == sample_eval_result["evaluation"]["frame_pitch"]["raw_pitch_accuracy"]
    )
    assert (
        df.loc[0, "frame_pitch_f_measure"]
        == sample_eval_result["evaluation"]["frame_pitch"]["raw_pitch_accuracy"]
    )
    assert (
        df.loc[0, "frame_pitch_accuracy"]
        == sample_eval_result["evaluation"]["frame_pitch"]["raw_pitch_accuracy"]
    )


def test_create_summary_dataframe_with_failed_results(sample_eval_result):
    """create_summary_dataframeが失敗した結果を除外するかテストします"""
    results_list = [
        sample_eval_result,  # status: success
        {
            "detector_name": "DetectorB",
            "audio_path": "/path/to/audio2.wav",
            "status": "error",  # 失敗した結果
            "error_message": "Something went wrong",
            "detection_time": 0.1,
        },
    ]
    df = create_summary_dataframe(results_list)
    assert len(df) == 1  # 失敗した結果は除外される
    assert df.loc[0, "detector_name"] == "TestDetector"


def test_create_summary_dataframe_missing_evaluation(sample_eval_result):
    """create_summary_dataframeでevaluationキーがない結果を処理できるかテストします"""
    results_list = [
        sample_eval_result,
        {
            "detector_name": "DetectorC",
            "audio_path": "/path/to/audio3.wav",
            "status": "success",
            # evaluation キーがない
            "detection_time": 0.5,
        },
    ]
    df = create_summary_dataframe(results_list)
    assert len(df) == 2
    # evaluation がない行は NaN になるはず
    assert df.loc[1, "detector_name"] == "DetectorC"
    assert pd.isna(df.loc[1, "note_f_measure"])
    assert pd.isna(df.loc[1, "frame_pitch_raw_pitch_accuracy"])


def test_create_summary_dataframe_empty_input():
    """create_summary_dataframeで入力リストが空の場合のテスト"""
    df = create_summary_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# --- Test print_summary_statistics ---


def test_print_summary_statistics_basic(caplog):
    """print_summary_statisticsの基本的なログ出力をテストします"""
    summary_df = pd.DataFrame(
        {
            "detector_name": ["TestDetector"],
            "files_count": [10],
            "detection_time_mean": [1.5],
            "note_f_measure": [0.85],
            "onset_precision": [0.9],
            "frame_pitch_raw_pitch_accuracy": [0.95],
            "some_other_metric": [0.5],  # metrics_list にない指標
        }
    )
    metrics_list = [
        "note_f_measure",
        "onset_precision",
        "frame_pitch_raw_pitch_accuracy",
        "non_existent_metric",  # 存在しない指標
    ]

    # caplog を使って src.evaluation.evaluation_io ロガーの INFO レベル以上をキャプチャ
    with caplog.at_level(logging.INFO, logger="src.evaluation.evaluation_io"):
        print_summary_statistics(summary_df, metrics_list=metrics_list)

    # caplog.text でログ全体を確認
    assert "評価結果のサマリー:" in caplog.text
    assert "検出器: TestDetector" in caplog.text
    assert "評価ファイル数: 10" in caplog.text
    assert "平均検出時間: 1.5000秒" in caplog.text
    assert "note_f_measure: 0.8500" in caplog.text
    assert "onset_precision: 0.9000" in caplog.text
    assert "frame_pitch_raw_pitch_accuracy: 0.9500" in caplog.text
    assert "some_other_metric" not in caplog.text
    assert "non_existent_metric" not in caplog.text


def test_print_summary_statistics_no_metrics_list(caplog):
    """print_summary_statisticsでmetrics_listが指定されない場合のテスト"""
    summary_df = pd.DataFrame(
        {
            "detector_name": ["TestDetector"],
            "files_count": [5],
            "detection_time_mean": [0.8],
            "note_f_measure": [0.7],
            "offset_recall": [0.6],
            "frame_pitch_overall_accuracy": [0.8],
        }
    )

    # caplog を使って src.evaluation.evaluation_io ロガーの INFO レベル以上をキャプチャ
    with caplog.at_level(logging.INFO, logger="src.evaluation.evaluation_io"):
        print_summary_statistics(summary_df)  # metrics_list=None

    # caplog.text でログ全体を確認
    assert "note_f_measure: 0.7000" in caplog.text
    assert "offset_recall: 0.6000" in caplog.text
    assert "frame_pitch_overall_accuracy: 0.8000" in caplog.text


def test_print_summary_statistics_empty_df(caplog):
    """print_summary_statisticsで空のDataFrameが渡された場合のテスト"""
    empty_df = pd.DataFrame()
    # caplog を使って src.evaluation.evaluation_io ロガーの WARNING レベル以上をキャプチャ
    with caplog.at_level(logging.WARNING, logger="src.evaluation.evaluation_io"):
        print_summary_statistics(empty_df)

    # ログに警告が出力されることを確認
    assert "表示する指標がありません (DataFrame is empty)" in caplog.text


def test_print_summary_statistics_no_available_metrics(caplog):
    """print_summary_statisticsで利用可能な指標がない場合のテスト"""
    summary_df = pd.DataFrame(
        {
            "detector_name": ["TestDetector"],
            "files_count": [1],
            "detection_time_mean": [1.0],
            # metrics_listに含まれる指標がDataFrameにない
        }
    )
    metrics_list = ["note_f_measure", "onset_precision"]

    # caplog を使って src.evaluation.evaluation_io ロガーの WARNING レベル以上をキャプチャ
    with caplog.at_level(logging.WARNING, logger="src.evaluation.evaluation_io"):
        print_summary_statistics(summary_df, metrics_list=metrics_list)

    # ログに警告が出力されることを確認
    assert (
        "表示する指標がありません (No displayable metrics found in DataFrame)"
        in caplog.text
    )


# --- Test save_detection_plot ---


@pytest.mark.skip(reason="Plotting function is currently commented out or unavailable.")
@patch(
    "src.evaluation.evaluation_io.plot_module_imported", False
)  # プロットモジュールがない状態をシミュレート
def test_save_detection_plot_module_not_imported(temp_dir):
    """save_detection_plotでプロットモジュールがない場合にスキップされるかテスト"""
    dummy_audio = np.zeros(100)
    dummy_detection = DetectionResult()
    dummy_reference = DetectionResult()
    output_path = temp_dir / "plot.png"

    save_detection_plot(
        dummy_audio, 44100, dummy_detection, dummy_reference, str(output_path)
    )

    # プロットファイルが作成されないことを確認
    assert not output_path.exists()


@pytest.mark.skip(reason="Plotting function is currently commented out or unavailable.")
@patch(
    "src.evaluation.evaluation_io.plot_module_imported", True
)  # プロットモジュールがある状態
@patch(
    "src.visualization.plot_utils.plot_detection_results"
)  # プロット関数自体をモック (仮のパス)
def test_save_detection_plot_basic(mock_plot_func, temp_dir):
    """save_detection_plotの基本的な呼び出しとファイル保存をテストします"""
    dummy_audio = np.random.rand(1000)
    sr = 16000
    detection_result = DetectionResult(
        intervals=np.array([[0.1, 0.5]]), note_pitches=np.array([440])
    )
    reference = DetectionResult(
        intervals=np.array([[0.12, 0.48]]), note_pitches=np.array([441])
    )
    output_path = temp_dir / "subdir" / "my_plot.pdf"
    plot_format = "pdf"
    plot_config = {"style": "seaborn"}

    # プロット関数が呼ばれることを期待
    save_detection_plot(
        dummy_audio,
        sr,
        detection_result,
        reference,
        str(output_path),
        plot_format=plot_format,
        plot_config=plot_config,
    )

    # 出力ディレクトリが作成されたか確認
    assert output_path.parent.exists()

    # プロット関数が正しい引数で呼ばれたか確認
    # 引数は辞書形式に変換されて渡されることに注意 (evaluation_io の実装次第)
    mock_plot_func.assert_called_once_with(
        audio_data=dummy_audio,
        sr=sr,
        detection_result=detection_result.to_dict(),  # 辞書に変換して渡される想定
        reference_data=reference.to_dict(),  # 辞書に変換して渡される想定
        show=False,  # show=False が渡されるはず
        save_path=str(output_path),  # 保存パスが渡されるはず
        # plot_config は内部で展開されるかもしれないので ANY で確認するか、実装に合わせる
        # **plot_config # plot_config は直接渡されない可能性
    )


@pytest.mark.skip(reason="Plotting function is currently commented out or unavailable.")
@patch("src.evaluation.evaluation_io.plot_module_imported", True)
@patch(
    "src.visualization.plot_utils.plot_detection_results",
    side_effect=Exception("Plotting error"),
)  # プロット関数でエラー (仮のパス)
def test_save_detection_plot_error_handling(mock_plot_func, temp_dir):
    """save_detection_plotでプロット中にエラーが発生した場合のテスト"""
    dummy_audio = np.zeros(100)
    sr = 8000
    detection_result = DetectionResult()
    reference = DetectionResult()
    output_path = temp_dir / "error_plot.png"

    save_detection_plot(dummy_audio, sr, detection_result, reference, str(output_path))

    # プロットファイルは作成されない
    assert not output_path.exists()


# --- Test save_detection_and_evaluation_results ---


def test_save_detection_and_evaluation_results_basic(
    sample_detection_result_dict, sample_eval_result, temp_dir
):
    """save_detection_and_evaluation_resultsの基本的な動作をテストします"""
    output_dir = temp_dir / "combined_results"
    base_name = "my_audio"
    detector_name = "SuperDetector"

    expected_output_subdir = output_dir / base_name / detector_name
    expected_detection_path = expected_output_subdir / "detection_result.json"
    expected_evaluation_path = expected_output_subdir / "evaluation_result.json"

    # 関数を実行
    result_paths = save_detection_and_evaluation_results(
        sample_detection_result_dict,
        sample_eval_result,
        str(output_dir),
        base_name,
        detector_name,
    )

    # 返されたパスが正しいか確認
    assert result_paths["detection_result"] == str(expected_detection_path)
    assert result_paths["evaluation_result"] == str(expected_evaluation_path)

    # ディレクトリとファイルが作成されたか確認
    assert expected_output_subdir.is_dir()
    assert expected_detection_path.exists()
    assert expected_evaluation_path.exists()

    # 保存された内容を確認 (一部)
    with open(expected_detection_path, "r") as f:
        loaded_detection = json.load(f)
    assert (
        loaded_detection["detector_name"]
        == sample_detection_result_dict["detector_name"]
    )
    # Numpy 配列がリストに変換されているか確認
    assert isinstance(loaded_detection["intervals"], list)
    assert (
        loaded_detection["intervals"]
        == sample_detection_result_dict["intervals"].tolist()
    )

    with open(expected_evaluation_path, "r") as f:
        loaded_evaluation = json.load(f)
    assert loaded_evaluation["detector_name"] == sample_eval_result["detector_name"]
    assert np.isclose(
        loaded_evaluation["evaluation"]["note"]["f_measure"],
        sample_eval_result["evaluation"]["note"]["f_measure"],
    )


def test_save_detection_and_evaluation_results_io_error(
    sample_detection_result_dict, sample_eval_result, temp_dir
):
    """save_detection_and_evaluation_resultsでファイル書き込みエラーが発生する場合のテスト"""
    output_dir = temp_dir / "combined_results"
    base_name = "my_audio"
    detector_name = "SuperDetector"

    # open をモックして PermissionError を発生させる
    with patch("builtins.open", side_effect=PermissionError("Cannot write file")):
        with pytest.raises(PermissionError):
            save_detection_and_evaluation_results(
                sample_detection_result_dict,
                sample_eval_result,
                str(output_dir),
                base_name,
                detector_name,
            )

    # ディレクトリは作成されるかもしれないが、ファイルは作成されない
    expected_output_subdir = output_dir / base_name / detector_name
    assert expected_output_subdir.is_dir()  # makedirs は呼ばれるはず
    assert not (expected_output_subdir / "detection_result.json").exists()
    assert not (expected_output_subdir / "evaluation_result.json").exists()
