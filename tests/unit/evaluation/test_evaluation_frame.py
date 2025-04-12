# tests/unit/evaluation/test_evaluation_frame.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, call
import logging

# テスト対象のモジュールをインポート
from src.evaluation.evaluation_frame import (
    notes_to_frames,
    evaluate_frame_pitches,
)

# --- Fixtures ---

@pytest.fixture
def mock_mir_eval_melody():
    """mir_eval.melodyモジュールをモックするフィクスチャ"""
    # spec=True を使うと、モック対象のインターフェースを検証できる
    with patch('src.evaluation.evaluation_frame.mir_eval.melody', spec=True) as mock_melody:
        # to_cent_voicing のモック: voicing 配列も返すように
        # (ref_voiced, ref_cents, est_voiced, est_cents) のタプルを返す
        def mock_to_cent_voicing_side_effect(ref_times, ref_freqs, est_times, est_freqs, base_frequency=10.0, hop=None, kind='linear'):
            # ダミーの実装: 単純に voicing (freq > 0) とセント値 (0 or 1200) を返す
            ref_voiced = ref_freqs > 0
            ref_cents = np.where(ref_voiced, 1200.0, 0.0) # ダミーのセント値
            est_voiced = est_freqs > 0
            est_cents = np.where(est_voiced, 1200.0, 0.0) # ダミーのセント値
            # mir_eval は内部で補間するため、実際にはもっと複雑
            # ここではテストに必要な構造を返すことに重点を置く
            # 時刻配列の長さが一致している必要がある場合がある
            if len(ref_times) != len(est_times):
                 # mir_eval の内部補間を模倣して同じ長さにするか、エラーを出す
                 # 簡単のため、同じ長さのデータを返す (実際のテストでは入力データに注意)
                 max_len = max(len(ref_times), len(est_times))
                 ref_voiced = np.resize(ref_voiced, max_len)
                 ref_cents = np.resize(ref_cents, max_len)
                 est_voiced = np.resize(est_voiced, max_len)
                 est_cents = np.resize(est_cents, max_len)

            return ref_voiced, ref_cents, est_voiced, est_cents

        mock_melody.to_cent_voicing.side_effect = mock_to_cent_voicing_side_effect
        # 他の mir_eval.melody 関数のモック設定
        mock_melody.voicing_measures.return_value = (0.9, 0.1)  # recall, false_alarm
        mock_melody.raw_pitch_accuracy.return_value = 0.95
        mock_melody.raw_chroma_accuracy.return_value = 0.98
        mock_melody.overall_accuracy.return_value = 0.85
        yield mock_melody

# --- Test notes_to_frames ---

# パラメータ化されたテストケース
@pytest.mark.parametrize(
    "note_intervals, note_pitches, hop_time, end_time, expected_times_len, expected_freqs_pattern",
    [
        # Case 1: Empty notes
        ([], [], 0.1, 0.3, 3, [0.0, 0.0, 0.0]),
        # Case 2: Single note
        # Note: [0.05, 0.15] in frames [0.0, 0.1, 0.2] should affect frames 0 and 1
        ([[0.05, 0.15]], [440.0], 0.1, 0.3, 3, [440.0, 440.0, 0.0]), # Corrected expected
        # Case 3: Single note covering all frames
        ([[0.0, 0.25]], [220.0], 0.1, 0.3, 3, [220.0, 220.0, 220.0]),
        # Case 4: Multiple non-overlapping notes
        # Note1: [0.0, 0.08] -> frames 0, 1 (0.0, 0.05)
        # Note2: [0.12, 0.2] -> frames 2, 3 (0.10, 0.15)
        ([[0.0, 0.08], [0.12, 0.2]], [440.0, 880.0], 0.05, 0.25, 5, [440.0, 440.0, 880.0, 880.0, 0.0]), # Corrected expected
        # Case 5: Overlapping notes (last note wins)
        # Note1: [0.0, 0.18] -> frames 0, 1, 2, 3 (0.0, 0.05, 0.10, 0.15)
        # Note2: [0.12, 0.2] -> frames 2, 3 (0.10, 0.15)
        # Overlap in frames 2, 3. Note 2 wins.
        ([[0.0, 0.18], [0.12, 0.2]], [440.0, 880.0], 0.05, 0.25, 5, [440.0, 440.0, 880.0, 880.0, 0.0]), # Corrected expected
        # Case 6: Note ends exactly on frame boundary (exclusive)
        # Note: [0.0, 0.1] -> frame 0 (0.0) only. end_idx = ceil(0.1/0.1) - 1 = 1 - 1 = 0
        ([[0.0, 0.1]], [440.0], 0.1, 0.3, 3, [440.0, 0.0, 0.0]),
        # Case 7: Note starts exactly on frame boundary (inclusive)
        # Note: [0.1, 0.2] -> frame 1 (0.1) only. start_idx = floor(0.1/0.1)=1, end_idx = ceil(0.2/0.1)-1 = 2-1=1
        ([[0.1, 0.2]], [440.0], 0.1, 0.3, 3, [0.0, 440.0, 0.0]),
        # Case 8: end_time shorter than last note offset
        ([[0.0, 0.25]], [220.0], 0.1, 0.2, 2, [220.0, 220.0]),
        # Case 9: Empty numpy arrays as input
        (np.array([]).reshape(0, 2), np.array([]), 0.1, 0.3, 3, [0.0, 0.0, 0.0]),
        # Case 10: No end_time specified (use max offset)
        ([[0.1, 0.3], [0.5, 0.7]], [440.0, 220.0], 0.1, None, 7, [0.0, 440.0, 440.0, 0.0, 0.0, 220.0, 220.0]),
        # Case 11: end_time is zero
         ([[0.1, 0.3]], [440.0], 0.1, 0.0, 0, []),
         # Case 12: hop_time > end_time
         ([[0.1, 0.3]], [440.0], 0.5, 0.3, 0, []),
    ]
)
def test_notes_to_frames(note_intervals, note_pitches, hop_time, end_time, expected_times_len, expected_freqs_pattern):
    """
    notes_to_frames関数の様々なケースをテストします。
    - 空のノート
    - 単一ノート
    - 複数ノート（オーバーラップあり・なし）
    - 境界値（フレーム境界上のノート開始・終了）
    - end_time パラメータの影響
    """
    notes_intervals_np = np.array(note_intervals)
    notes_pitches_np = np.array(note_pitches)

    # 関数を実行
    result_times, result_freqs = notes_to_frames(
        notes_intervals_np, notes_pitches_np, hop_time=hop_time, end_time=end_time
    )

    # 結果の型とサイズを確認
    assert isinstance(result_times, np.ndarray)
    assert isinstance(result_freqs, np.ndarray)
    assert result_times.ndim == 1
    assert result_freqs.ndim == 1
    assert len(result_times) == expected_times_len, f"Expected {expected_times_len} frames, got {len(result_times)}"
    assert len(result_freqs) == expected_times_len

    # 時間配列の確認 (開始時刻、終了時刻、ステップ)
    if expected_times_len > 0:
        expected_times = np.arange(expected_times_len) * hop_time
        np.testing.assert_allclose(result_times, expected_times, err_msg="Times array mismatch")
        if end_time is not None:
            assert result_times[-1] < (end_time + 1e-9) # 最後の時刻がend_timeを超えない

    # 周波数配列の内容を比較
    expected_freqs_np = np.array(expected_freqs_pattern)
    np.testing.assert_allclose(result_freqs, expected_freqs_np, rtol=1e-5, atol=1e-5, err_msg="Frequencies array mismatch")

def test_notes_to_frames_input_validation():
    """notes_to_frames 関数の入力検証 (長さ不一致など、現状実装ではチェックなし)"""
    # 現状の実装ではintervalsとpitchesの長さが異なっていてもエラーにならない
    # このテストではエラーにならないことを確認する
    note_intervals = np.array([[0.1, 0.3]])
    note_pitches = np.array([440.0, 220.0]) # 長さが違う
    
    # エラーにならずに実行できることを確認
    times, freqs = notes_to_frames(note_intervals, note_pitches)
    
    # 結果の検証（最初のピッチ値だけが使われることを確認）
    assert len(times) > 0
    assert len(freqs) > 0
    assert np.any(freqs == 440.0), "最初のピッチ値が使われていません"

# --- Test evaluate_frame_pitches ---

def test_evaluate_frame_pitches_basic(mock_mir_eval_melody, caplog):
    """evaluate_frame_pitchesの基本的な呼び出しと結果の形式をテストします。"""
    ref_time = np.array([0.0, 0.1, 0.2])
    ref_freq = np.array([0.0, 440.0, 0.0])
    est_time = np.array([0.0, 0.1, 0.2])
    est_freq = np.array([0.0, 442.0, 0.0]) # 少し高いピッチ

    # デフォルトの許容誤差で呼び出し
    metrics = evaluate_frame_pitches(ref_time, ref_freq, est_time, est_freq)

    # 結果の型と主要なキーの存在を確認
    assert isinstance(metrics, dict)
    expected_keys = [
        'voicing_recall', 'voicing_false_alarm',
        'raw_pitch_accuracy', 'raw_chroma_accuracy', 'overall_accuracy'
    ]
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float) # 値がfloatであること

    # モックが期待通りに呼び出されたか確認
    mock_mir_eval_melody.to_cent_voicing.assert_called_once()
    mock_mir_eval_melody.voicing_measures.assert_called_once()
    mock_mir_eval_melody.raw_pitch_accuracy.assert_called_once()
    mock_mir_eval_melody.raw_chroma_accuracy.assert_called_once()
    mock_mir_eval_melody.overall_accuracy.assert_called_once()

    # 警告ログが出ていないことを確認
    assert len(caplog.records) == 0

def test_evaluate_frame_pitches_empty_arrays(mock_mir_eval_melody, caplog):
    """evaluate_frame_pitchesで空の配列が入力された場合の動作をテストします。"""
    time_ok = np.array([0.0, 0.1, 0.2])
    freq_ok = np.array([0.0, 440.0, 0.0])
    empty_time = np.array([])
    empty_freq = np.array([])

    # ケース1: refが空、estが有効
    metrics_ref_empty = evaluate_frame_pitches(empty_time, empty_freq, time_ok, freq_ok)
    assert isinstance(metrics_ref_empty, dict)
    # refが空の場合、モックで設定した値が返される
    assert metrics_ref_empty['voicing_recall'] == 0.9
    assert metrics_ref_empty['raw_pitch_accuracy'] == 0.95
    assert metrics_ref_empty['overall_accuracy'] == 0.85
    # ログは出ない（空配列は内部で処理される想定）

    # ケース2: estが空、refが有効
    metrics_est_empty = evaluate_frame_pitches(time_ok, freq_ok, empty_time, empty_freq)
    assert isinstance(metrics_est_empty, dict)
    assert metrics_est_empty['voicing_recall'] == 0.9
    assert metrics_est_empty['raw_pitch_accuracy'] == 0.95
    assert metrics_est_empty['overall_accuracy'] == 0.85
    # ログは出ない

    # ケース3: 両方が空
    metrics_both_empty = evaluate_frame_pitches(empty_time, empty_freq, empty_time, empty_freq)
    assert isinstance(metrics_both_empty, dict)
    # 両方空の場合は完全一致とみなす
    assert metrics_both_empty['voicing_recall'] == 1.0
    assert metrics_both_empty['voicing_false_alarm'] == 0.0
    assert metrics_both_empty['raw_pitch_accuracy'] == 1.0
    assert metrics_both_empty['raw_chroma_accuracy'] == 1.0
    assert metrics_both_empty['overall_accuracy'] == 1.0
    assert len(caplog.records) == 0 # ログは出ない

def test_evaluate_frame_pitches_length_mismatch(mock_mir_eval_melody, caplog):
    """evaluate_frame_pitchesで時刻と周波数の配列長が不一致の場合の動作をテストします。"""
    ref_time_ok = np.array([0.0, 0.1, 0.2])
    ref_freq_ok = np.array([0.0, 440.0, 0.0])
    est_time_ok = np.array([0.0, 0.1, 0.2])
    est_freq_ok = np.array([0.0, 442.0, 0.0])

    ref_time_bad = np.array([0.0, 0.1]) # 長さ不一致
    ref_freq_bad = np.array([0.0, 440.0, 0.0])
    est_time_bad = np.array([0.0, 0.1, 0.2])
    est_freq_bad = np.array([0.0, 442.0]) # 長さ不一致

    # ケース1: refの長さ不一致
    metrics_ref_mismatch = evaluate_frame_pitches(ref_time_bad, ref_freq_bad, est_time_ok, est_freq_ok)
    assert isinstance(metrics_ref_mismatch, dict)
    # エラー時のデフォルト値を返す
    assert metrics_ref_mismatch['voicing_recall'] == 0.0
    assert metrics_ref_mismatch['voicing_false_alarm'] == 1.0
    assert metrics_ref_mismatch['raw_pitch_accuracy'] == 0.0
    mock_mir_eval_melody.to_cent_voicing.assert_not_called() # エラーなので評価関数は呼ばれない
    # 警告ログを確認
    assert any("参照データの時刻数とピッチ数が一致しません" in rec.message for rec in caplog.records)
    caplog.clear()

    # ケース2: estの長さ不一致
    metrics_est_mismatch = evaluate_frame_pitches(ref_time_ok, ref_freq_ok, est_time_bad, est_freq_bad)
    assert isinstance(metrics_est_mismatch, dict)
    assert metrics_est_mismatch['voicing_recall'] == 0.0
    assert metrics_est_mismatch['voicing_false_alarm'] == 1.0
    mock_mir_eval_melody.to_cent_voicing.assert_not_called()
    assert any("推定データの時刻数とピッチ数が一致しません" in rec.message for rec in caplog.records)
    caplog.clear()

    # ケース3: 両方の長さ不一致
    metrics_both_mismatch = evaluate_frame_pitches(ref_time_bad, ref_freq_bad, est_time_bad, est_freq_bad)
    assert isinstance(metrics_both_mismatch, dict)
    assert metrics_both_mismatch['voicing_recall'] == 0.0
    assert metrics_both_mismatch['voicing_false_alarm'] == 1.0
    mock_mir_eval_melody.to_cent_voicing.assert_not_called()
    # ref の警告が先に出るはず
    assert any("参照データの時刻数とピッチ数が一致しません" in rec.message for rec in caplog.records)

def test_evaluate_frame_pitches_mir_eval_exception(mock_mir_eval_melody, caplog):
    """evaluate_frame_pitchesで内部のmir_eval関数が例外を発生させた場合の動作をテストします。"""
    ref_time = np.array([0.0, 0.1, 0.2])
    ref_freq = np.array([0.0, 440.0, 0.0])
    est_time = np.array([0.0, 0.1, 0.2])
    est_freq = np.array([0.0, 442.0, 0.0])

    # to_cent_voicing で例外を発生させる
    mock_mir_eval_melody.to_cent_voicing.side_effect = ValueError("Test Exception from mir_eval")

    metrics = evaluate_frame_pitches(ref_time, ref_freq, est_time, est_freq)

    # 関数はクラッシュせず、デフォルトのエラー値を返す
    assert isinstance(metrics, dict)
    assert metrics['voicing_recall'] == 0.0
    assert metrics['voicing_false_alarm'] == 1.0
    assert metrics['raw_pitch_accuracy'] == 0.0

    # エラーログが出力されていることを確認
    assert any("フレーム評価中にエラーが発生しました: Test Exception from mir_eval" in rec.message for rec in caplog.records)

    # 例外発生後、他の mir_eval 関数は呼ばれない
    mock_mir_eval_melody.voicing_measures.assert_not_called()
    mock_mir_eval_melody.raw_pitch_accuracy.assert_not_called()

def test_evaluate_frame_pitches_tolerance_argument(mock_mir_eval_melody):
    """evaluate_frame_pitchesのpitch_tolerance引数がmir_evalに渡されるかテストします。"""
    ref_time = np.array([0.0, 0.1, 0.2])
    ref_freq = np.array([0.0, 440.0, 0.0])
    est_time = np.array([0.0, 0.1, 0.2])
    est_freq = np.array([0.0, 442.0, 0.0])

    # 許容誤差 0 cents で呼び出し
    evaluate_frame_pitches(ref_time, ref_freq, est_time, est_freq, pitch_tolerance=0)
    # raw_pitch_accuracy などが cent_tolerance=0 で呼ばれたか確認
    args, kwargs = mock_mir_eval_melody.raw_pitch_accuracy.call_args
    assert 'cent_tolerance' in kwargs and kwargs['cent_tolerance'] == 0
    args, kwargs = mock_mir_eval_melody.raw_chroma_accuracy.call_args
    assert 'cent_tolerance' in kwargs and kwargs['cent_tolerance'] == 0
    args, kwargs = mock_mir_eval_melody.overall_accuracy.call_args
    assert 'cent_tolerance' in kwargs and kwargs['cent_tolerance'] == 0
    mock_mir_eval_melody.reset_mock() # モックのリセット

    # 許容誤差 100 cents で呼び出し
    evaluate_frame_pitches(ref_time, ref_freq, est_time, est_freq, pitch_tolerance=100)
    args, kwargs = mock_mir_eval_melody.raw_pitch_accuracy.call_args
    assert 'cent_tolerance' in kwargs and kwargs['cent_tolerance'] == 100
    mock_mir_eval_melody.reset_mock()

    # デフォルト (50 cents) で呼び出し
    evaluate_frame_pitches(ref_time, ref_freq, est_time, est_freq)
    args, kwargs = mock_mir_eval_melody.raw_pitch_accuracy.call_args
    assert 'cent_tolerance' in kwargs and kwargs['cent_tolerance'] == 50.0 # デフォルト値を確認
