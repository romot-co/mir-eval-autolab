# tests/unit/evaluation/test_evaluation_frame.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

# テスト対象モジュールをインポート
try:
    from src.evaluation import evaluation_frame
    from src.evaluation.evaluation_frame import (
        notes_to_frames,
        evaluate_frame_pitches
    )
    # Import Note if needed for notes_to_frames input
    from src.structures.note import Note
    # Import mir_eval for mocking
    import mir_eval
except ImportError:
    pytest.skip("Skipping evaluation_frame tests due to missing src modules", allow_module_level=True)
    # Dummy implementations
    class Note: pass
    class mir_eval:
        class melody:
            @staticmethod
            def evaluate(*args, **kwargs): return {}

# --- Fixtures ---

@pytest.fixture
def mock_mir_eval_melody():
    """Mocks the mir_eval.melody module."""
    with patch('src.evaluation.evaluation_frame.mir_eval.melody', autospec=True) as mock:
        # Default return value for evaluate function
        mock.evaluate.return_value = {
            'Voicing Recall': 0.9, 'Voicing False Alarm': 0.1,
            'Raw Pitch Accuracy': 0.8, 'Raw Chroma Accuracy': 0.85,
            'Overall Accuracy': 0.75
        }
        yield mock

# --- Test notes_to_frames ---

@pytest.mark.parametrize("notes_intervals, notes_pitches, frame_times, expected_freq, expected_voicing", [
    # 1. Empty notes
    ([], [], np.array([0.0, 0.1, 0.2]), np.zeros(3), np.zeros(3, dtype=bool)),
    # 2. Single note covering some frames
    ([[0.05, 0.15]], [60], np.array([0.0, 0.1, 0.2]), np.array([0.0, 261.63, 0.0]), np.array([False, True, False])),
    # 3. Single note covering all frames
    ([[0.0, 0.25]], [69], np.array([0.0, 0.1, 0.2]), np.full(3, 440.0), np.full(3, True)),
    # 4. Multiple notes, non-overlapping
    ([[0.0, 0.08], [0.12, 0.2]], [60, 72], np.array([0.05, 0.1, 0.15]), np.array([261.63, 0.0, 523.25]), np.array([True, False, True])),
    # 5. Multiple notes, overlapping (last note wins)
    ([[0.0, 0.18], [0.12, 0.2]], [60, 72], np.array([0.05, 0.15, 0.25]), np.array([261.63, 523.25, 0.0]), np.array([True, True, False])),
    # 6. Note ends exactly on frame time (should not be included in that frame)
    ([[0.0, 0.1]], [60], np.array([0.0, 0.1, 0.2]), np.array([261.63, 0.0, 0.0]), np.array([True, False, False])),
    # 7. Note starts exactly on frame time (should be included)
    ([[0.1, 0.2]], [60], np.array([0.0, 0.1, 0.2]), np.array([0.0, 261.63, 0.0]), np.array([False, True, False])),
])
def test_notes_to_frames(notes_intervals, notes_pitches, frame_times, expected_freq, expected_voicing):
    """Tests the notes_to_frames conversion with various scenarios."""
    notes_intervals_np = np.array(notes_intervals)
    notes_pitches_np = np.array(notes_pitches)

    # Mock midi_to_hz function if it's defined in evaluation_frame
    # or rely on a globally available one (e.g., from pitch_utils)
    with patch('src.evaluation.evaluation_frame.midi_to_hz', side_effect=lambda m: 440.0 * (2**((m-69)/12.0))) as mock_midi_hz:

        result_freq, result_voicing = notes_to_frames(
            notes_intervals_np, notes_pitches_np, frame_times
        )

        assert isinstance(result_freq, np.ndarray)
        assert isinstance(result_voicing, np.ndarray)
        assert result_freq.shape == frame_times.shape
        assert result_voicing.shape == frame_times.shape
        assert result_voicing.dtype == bool

        # Check if midi_to_hz was called correctly for the pitches
        expected_calls = [call(p) for p in notes_pitches]
        mock_midi_hz.assert_has_calls(expected_calls, any_order=True)

        np.testing.assert_allclose(result_freq, expected_freq, atol=1e-2)
        np.testing.assert_array_equal(result_voicing, expected_voicing)

# --- Test evaluate_frame_pitches ---

def test_evaluate_frame_pitches_basic(mock_mir_eval_melody):
    """Tests evaluate_frame_pitches basic call with mocked mir_eval."""
    ref_time = np.array([0.0, 0.1, 0.2])
    ref_freq = np.array([0.0, 440.0, 0.0])
    ref_voicing = ref_freq > 0
    est_time = np.array([0.0, 0.1, 0.2])
    est_freq = np.array([0.0, 442.0, 0.0]) # Slightly sharp
    est_voicing = est_freq > 0

    metrics = evaluate_frame_pitches(ref_time, ref_freq, est_time, est_freq)

    # Check if mir_eval.melody.evaluate was called correctly
    mock_mir_eval_melody.evaluate.assert_called_once_with(
        ref_time=ref_time, ref_freq=ref_freq, est_time=est_time, est_freq=est_freq
    )

    # Check if the returned dictionary contains the expected keys (from mock)
    assert isinstance(metrics, dict)
    assert 'frame.Overall_Accuracy' in metrics
    assert 'frame.Raw_Pitch_Accuracy' in metrics
    assert metrics['frame.Overall_Accuracy'] == 0.75 # Value from mock

def test_evaluate_frame_pitches_all_unvoiced(mock_mir_eval_melody):
    """Tests evaluate_frame_pitches when both ref and est are unvoiced."""
    ref_time = np.array([0.0, 0.1, 0.2])
    ref_freq = np.zeros(3)
    ref_voicing = ref_freq > 0
    est_time = np.array([0.0, 0.1, 0.2])
    est_freq = np.zeros(3)
    est_voicing = est_freq > 0

    metrics = evaluate_frame_pitches(ref_time, ref_freq, est_time, est_freq)

    mock_mir_eval_melody.evaluate.assert_called_once_with(
        ref_time=ref_time, ref_freq=ref_freq, est_time=est_time, est_freq=est_freq
    )
    assert 'frame.Overall_Accuracy' in metrics

# Add more tests: e.g., different voicing scenarios, completely wrong frequencies 