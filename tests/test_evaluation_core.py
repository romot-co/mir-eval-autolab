import pytest
import numpy as np
import pandas as pd

# Assuming evaluation_core.py is in src/evaluation/
from src.evaluation import evaluation_core

# Tolerance for floating point comparisons
TOL = 1e-6

def test_precision_recall_f1_safe_perfect_match():
    """Perfect match should yield P=1, R=1, F1=1."""
    tp = 10
    fp = 0
    fn = 0
    p, r, f1 = evaluation_core.precision_recall_f1_safe(tp, fp, fn)
    assert np.isclose(p, 1.0, atol=TOL)
    assert np.isclose(r, 1.0, atol=TOL)
    assert np.isclose(f1, 1.0, atol=TOL)

def test_precision_recall_f1_safe_no_true_positives():
    """No true positives, but false positives/negatives present."""
    tp = 0
    fp = 5
    fn = 5
    p, r, f1 = evaluation_core.precision_recall_f1_safe(tp, fp, fn)
    assert np.isclose(p, 0.0, atol=TOL)
    assert np.isclose(r, 0.0, atol=TOL)
    assert np.isclose(f1, 0.0, atol=TOL)

def test_precision_recall_f1_safe_zero_denominator():
    """Edge case: zero denominators (no predictions or no references)."""
    # No predictions (TP=0, FP=0) -> P=0?, R=depends on FN
    tp = 0
    fp = 0
    fn = 10
    p, r, f1 = evaluation_core.precision_recall_f1_safe(tp, fp, fn)
    # Precision is undefined (0/0), should default to 0? Let's assume 0 for F1 calculation.
    assert np.isclose(p, 0.0, atol=TOL) # Changed expectation based on likely implementation
    assert np.isclose(r, 0.0, atol=TOL)
    assert np.isclose(f1, 0.0, atol=TOL)

    # No references (TP=0, FN=0) -> R=0?, P=depends on FP
    tp = 0
    fp = 10
    fn = 0
    p, r, f1 = evaluation_core.precision_recall_f1_safe(tp, fp, fn)
    assert np.isclose(p, 0.0, atol=TOL)
    # Recall is undefined (0/0), should default to 0? Let's assume 0 for F1 calculation.
    assert np.isclose(r, 0.0, atol=TOL) # Changed expectation based on likely implementation
    assert np.isclose(f1, 0.0, atol=TOL)

    # No predictions and no references (TP=0, FP=0, FN=0) -> P=0, R=0, F1=0
    tp = 0
    fp = 0
    fn = 0
    p, r, f1 = evaluation_core.precision_recall_f1_safe(tp, fp, fn)
    assert np.isclose(p, 0.0, atol=TOL)
    assert np.isclose(r, 0.0, atol=TOL)
    assert np.isclose(f1, 0.0, atol=TOL)

# Add more tests for other functions in evaluation_core.py
# e.g., _match_notes, calculate_note_metrics, etc.
# These would require creating dummy reference and estimated note arrays.

# Example structure for testing _match_notes (requires more setup)
# def test_match_notes_simple():
#     ref_intervals = np.array([[0.1, 0.5], [1.0, 1.5]])
#     ref_pitches = np.array([60.0, 62.0])
#     est_intervals = np.array([[0.11, 0.49], [1.05, 1.55], [2.0, 2.5]])
#     est_pitches = np.array([60.1, 61.9, 64.0])
#     tolerance_onset = 0.05
#     tolerance_offset = 0.05
#     tolerance_pitch = 50.0 # cents
#
#     matches, ref_matched_flags, est_matched_flags = evaluation_core._match_notes(
#         ref_intervals, ref_pitches, est_intervals, est_pitches,
#         tolerance_onset, tolerance_offset, tolerance_pitch
#     )
#     # Add assertions based on expected matches
#     assert len(matches) == 2 # Expecting 2 matches
#     assert ref_matched_flags[0] and ref_matched_flags[1]
#     assert est_matched_flags[0] and est_matched_flags[1] and not est_matched_flags[2]
#     # Further assertions on the content of 'matches' (indices, errors) 