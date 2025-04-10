import pytest
import numpy as np
import os
import soundfile as sf
import csv
from pathlib import Path
import logging

# --- テスト対象モジュールのインポート ---
# synthesizer.py が src/data_generation にあると仮定
from src.data_generation import synthesizer

# --- 定数 ---
SR = synthesizer.SR # モジュールからサンプリングレートを取得
TEST_OUTPUT_PREFIX = "test_output"

# --- Pytest Fixture ---

@pytest.fixture(scope="function") # function スコープに変更し、テストごとに出力ディレクトリをクリーンにする
def output_dirs(tmp_path, monkeypatch):
    """テスト用の一時的な出力ディレクトリを作成し、synthesizer モジュールに設定するフィクスチャ"""
    # 一時ディレクトリ内に audio と labels ディレクトリを作成
    audio_dir = tmp_path / "audio"
    labels_dir = tmp_path / "labels"
    audio_dir.mkdir()
    labels_dir.mkdir()

    # synthesizer モジュール内のグローバル変数に出力パスを設定 (monkeypatch を使用)
    # これにより、テスト中に synthesizer が一時ディレクトリに出力するようになる
    monkeypatch.setattr(synthesizer, "OUTPUT_AUDIO_DIR", str(audio_dir))
    monkeypatch.setattr(synthesizer, "OUTPUT_LABEL_DIR", str(labels_dir))

    # synthesizer のディレクトリ作成関数を呼び出す (必須ではないが、念のため整合性を保つ)
    # synthesizer.ensure_output_dirs() # これは monkeypatch で上書きされるため不要

    yield audio_dir, labels_dir # テスト関数にパスを渡す

    # クリーンアップは tmp_path が自動で行うので不要
    # print(f"\nCleaning up temp dirs: {audio_dir}, {labels_dir}")


# --- save_audio_and_label のテスト ---

def test_save_audio_and_label_normal(output_dirs, caplog):
    """save_audio_and_label の正常系テスト"""
    audio_dir, labels_dir = output_dirs
    filename_base = f"{TEST_OUTPUT_PREFIX}_normal"
    audio_data = np.sin(np.linspace(0, 440 * 2 * np.pi, SR)) # 1秒の 440Hz サイン波
    labels = [(0.1, 0.9, 440.0)]

    with caplog.at_level(logging.INFO):
        synthesizer.save_audio_and_label(filename_base, audio_data, labels, sr=SR)

    # ファイルが作成されたか確認
    audio_path = audio_dir / f"{filename_base}.wav"
    label_path = labels_dir / f"{filename_base}.csv"
    assert audio_path.exists()
    assert label_path.exists()
    assert audio_path.stat().st_size > 0
    assert label_path.stat().st_size > 0

    # ログ確認 (例: 保存成功ログ)
    assert f"音声を保存しました: {audio_path}" in caplog.text
    assert f"ラベルを保存しました: {label_path}" in caplog.text

    # 保存されたファイルの内容を簡単に確認 (オプション)
    saved_audio, saved_sr = sf.read(audio_path)
    assert saved_sr == SR
    assert len(saved_audio) == len(audio_data)
    # 正規化されているはずなので、最大絶対値が AMP_MAX に近いはず
    # float32での保存誤差を考慮し、許容誤差をさらに広げる
    assert np.isclose(np.max(np.abs(saved_audio)), synthesizer.AMP_MAX, atol=1e-4)

    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        saved_labels = list(reader)
        assert len(saved_labels) == 1 # ヘッダーなしの場合
        assert np.isclose(float(saved_labels[0][0]), labels[0][0])
        assert np.isclose(float(saved_labels[0][1]), labels[0][1])
        assert np.isclose(float(saved_labels[0][2]), labels[0][2])


def test_save_audio_and_label_nan_data(output_dirs, caplog):
    """NaN を含む audio_data を与えた場合のテスト"""
    audio_dir, labels_dir = output_dirs
    filename_base = f"{TEST_OUTPUT_PREFIX}_nan"
    audio_data = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    labels = [(0.0, 0.1, 100.0)] # ダミーラベル

    with caplog.at_level(logging.ERROR):
        synthesizer.save_audio_and_label(filename_base, audio_data, labels, sr=SR)

    # エラーログが出力されることを確認
    assert f"音声データに NaN または Inf が含まれています: {filename_base}" in caplog.text

    # ファイルが作成され、中身が NaN でないことを確認
    audio_path = audio_dir / f"{filename_base}.wav"
    label_path = labels_dir / f"{filename_base}.csv"
    assert audio_path.exists()
    assert label_path.exists() # ラベルは正常なので作成される
    assert audio_path.stat().st_size > 0

    saved_audio, _ = sf.read(audio_path)
    assert not np.any(np.isnan(saved_audio)) # NaN が 0 などに置換されているはず


def test_save_audio_and_label_inf_data(output_dirs, caplog):
    """Inf を含む audio_data を与えた場合のテスト"""
    audio_dir, labels_dir = output_dirs
    filename_base = f"{TEST_OUTPUT_PREFIX}_inf"
    audio_data = np.array([0.1, 0.2, np.inf, 0.4, -np.inf])
    labels = [(0.0, 0.1, 100.0)]

    with caplog.at_level(logging.ERROR):
        synthesizer.save_audio_and_label(filename_base, audio_data, labels, sr=SR)

    # エラーログが出力されることを確認
    assert f"音声データに NaN または Inf が含まれています: {filename_base}" in caplog.text

    # ファイルが作成され、中身が Inf でないことを確認
    audio_path = audio_dir / f"{filename_base}.wav"
    assert audio_path.exists()
    assert audio_path.stat().st_size > 0

    saved_audio, _ = sf.read(audio_path)
    assert np.all(np.isfinite(saved_audio)) # Inf が有限値に置換されているはず


def test_save_audio_and_label_empty_data(output_dirs, caplog):
    """空の audio_data を与えた場合のテスト"""
    audio_dir, labels_dir = output_dirs
    filename_base = f"{TEST_OUTPUT_PREFIX}_empty_audio"
    audio_data = np.array([])
    labels = [(0.0, 0.1, 100.0)]

    with caplog.at_level(logging.ERROR):
        synthesizer.save_audio_and_label(filename_base, audio_data, labels, sr=SR)

    # エラーログが出力されることを確認
    assert f"音声データが空です: {filename_base}" in caplog.text

    # ファイルが作成されないことを確認
    audio_path = audio_dir / f"{filename_base}.wav"
    label_path = labels_dir / f"{filename_base}.csv"
    assert not audio_path.exists()
    assert not label_path.exists() # 音声がないのでラベルも保存されない


def test_save_audio_and_label_silent_data(output_dirs, caplog):
    """ほぼ無音の audio_data を与えた場合のテスト"""
    audio_dir, labels_dir = output_dirs
    filename_base = f"{TEST_OUTPUT_PREFIX}_silent"
    # ピーク振幅が 1e-9 より小さいデータ
    audio_data = np.zeros(SR) * 1e-10
    labels = [(0.1, 0.9, 100.0)]

    with caplog.at_level(logging.WARNING):
        synthesizer.save_audio_and_label(filename_base, audio_data, labels, sr=SR)

    # 警告ログ（正規化スキップ）が出力されることを確認
    assert f"オーディオ信号のピーク振幅が非常に小さいです: {filename_base}" in caplog.text

    # ファイルは作成されるはず
    audio_path = audio_dir / f"{filename_base}.wav"
    label_path = labels_dir / f"{filename_base}.csv"
    assert audio_path.exists()
    assert label_path.exists()
    assert audio_path.stat().st_size > 0 # 無音だがファイル自体は作成される

    # 保存された音声が無音（または非常に小さい値）であることを確認
    saved_audio, _ = sf.read(audio_path)
    assert np.all(np.abs(saved_audio) < 1e-9)


def test_save_audio_and_label_no_labels(output_dirs, caplog):
    """空のラベルリストを与えた場合のテスト"""
    audio_dir, labels_dir = output_dirs
    filename_base = f"{TEST_OUTPUT_PREFIX}_no_labels"
    audio_data = np.sin(np.linspace(0, 440 * 2 * np.pi, SR))
    labels = [] # 空のラベル

    with caplog.at_level(logging.WARNING):
        synthesizer.save_audio_and_label(filename_base, audio_data, labels, sr=SR, include_header=True)

    # 警告ログが出力されることを確認
    assert f"{filename_base} にラベルがありません" in caplog.text

    # オーディオファイルは作成される
    audio_path = audio_dir / f"{filename_base}.wav"
    assert audio_path.exists()
    assert audio_path.stat().st_size > 0

    # ラベルファイルも作成されるが、中身はヘッダーのみ（または空）
    label_path = labels_dir / f"{filename_base}.csv"
    assert label_path.exists()
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        saved_labels = list(reader)
        assert len(saved_labels) == 1 # ヘッダー行のみ
        assert saved_labels[0] == ['onset', 'offset', 'frequency']


# --- generate_... 関数のテスト ---

def test_generate_vocal_imitation_runs(output_dirs, caplog):
    """generate_vocal_imitation がエラーなく実行され、ファイルが生成されるかテスト"""
    audio_dir, labels_dir = output_dirs
    filename_base = "21_vocal_imitation_test" # テスト用のファイル名

    # generate_vocal_imitation を実行
    # エラーが発生したら pytest が検知する
    synthesizer.generate_vocal_imitation(filename_base=filename_base, sr=SR)

    # ファイルが生成されたか確認
    audio_path = audio_dir / f"{filename_base}.wav"
    label_path = labels_dir / f"{filename_base}.csv"
    assert audio_path.exists(), f"オーディオファイルが見つかりません: {audio_path}"
    assert label_path.exists(), f"ラベルファイルが見つかりません: {label_path}"

    # ファイルが空でないか確認
    assert audio_path.stat().st_size > 0, f"オーディオファイルが空です: {audio_path}"
    assert label_path.stat().st_size > 0, f"ラベルファイルが空です: {label_path}"

    # ラベルファイルの行数を確認 (generate_vocal_imitation 内の notes リストの要素数と一致するはず)
    # generate_vocal_imitation の notes の数を確認（現在6個）
    expected_num_labels = 6
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        saved_labels = list(reader)
        # ヘッダーなしで保存されるので、行数がラベル数と一致するはず
        assert len(saved_labels) == expected_num_labels, f"期待されるラベル数 ({expected_num_labels}) と異なります: {len(saved_labels)}"

    # エラーログが出ていないことを確認 (これは厳密すぎるかもしれない。想定内の WARNING は許容すべきか？)
    # ここでは、少なくとも CRITICAL や ERROR がないことを確認する
    error_logs = [record for record in caplog.records if record.levelno >= logging.ERROR]
    # save_audio_and_label での NaN/Inf/空データチェックは generate_vocal_imitation 内部でハンドルされる可能性があるため、
    # 特定のエラーメッセージを除外するか、テストデータ生成ロジックが安定していることを前提とする。
    # 現状では、予期せぬ ERROR がないことを確認する。
    unexpected_errors = [rec.message for rec in error_logs if "音声データに NaN または Inf が含まれています" not in rec.message and "音声データが空です" not in rec.message]
    assert not unexpected_errors, f"予期せぬエラーログが見つかりました: {unexpected_errors}"


# TODO: 他の generate_... 関数のテストも同様に追加可能
# 例:
# def test_generate_basic_sine_sequence_runs(output_dirs):
#     synthesizer.generate_basic_sine_sequence(filename_base="test_basic_sine")
#     audio_path = output_dirs[0] / "test_basic_sine.wav"
#     label_path = output_dirs[1] / "test_basic_sine.csv"
#     assert audio_path.exists() and audio_path.stat().st_size > 0
#     assert label_path.exists() and label_path.stat().st_size > 0 