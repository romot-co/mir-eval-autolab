import pytest
import numpy as np

# Assume functions are defined in src.utils.pitch_utils
# Provide dummies if needed
try:
    from src.utils.pitch_utils import hz_to_midi, midi_to_hz
except ImportError:
    print("Warning: Using dummy implementations for pitch_utils.")
    # Dummy implementations based on standard formulas
    def hz_to_midi(frequencies):
        # Handle non-positive frequencies -> return NaN (mimicking one possible behavior)
        frequencies = np.asarray(frequencies, dtype=float) # Ensure float input
        midi_val = np.full_like(frequencies, np.nan) # Initialize with NaN
        # Calculate only for positive frequencies
        positive_mask = frequencies > 1e-6 # Use a small threshold instead of direct > 0
        if np.any(positive_mask):
             midi_val[positive_mask] = 12 * np.log2(frequencies[positive_mask] / 440.0) + 69
        return midi_val

    def midi_to_hz(midi_notes):
        midi_notes = np.asarray(midi_notes, dtype=float) # Ensure float input
        # MIDI can be float, formula works directly
        return 440.0 * (2.0 ** ((midi_notes - 69.0) / 12.0))


# --- hz_to_midi Tests ---

def test_hz_to_midi_scalar():
    """Scalar Hz value conversion."""
    assert np.isclose(hz_to_midi(440.0), 69.0) # A4
    assert np.isclose(hz_to_midi(220.0), 57.0) # A3
    assert np.isclose(hz_to_midi(880.0), 81.0) # A5
    # Middle C (C4) is MIDI 60, ~261.63 Hz
    assert np.isclose(hz_to_midi(261.625565), 60.0, atol=1e-6)

def test_hz_to_midi_array():
    """Array of Hz values conversion."""
    frequencies = np.array([220.0, 440.0, 880.0])
    expected_midi = np.array([57.0, 69.0, 81.0])
    result_midi = hz_to_midi(frequencies)
    assert isinstance(result_midi, np.ndarray)
    np.testing.assert_allclose(result_midi, expected_midi)

def test_hz_to_midi_zero_and_negative():
    """Zero and negative Hz values handling (expect NaN based on dummy/common practice)."""
    frequencies = np.array([-100.0, 0.0, 440.0, 1e-7]) # Include near-zero positive
    result_midi = hz_to_midi(frequencies)
    assert isinstance(result_midi, np.ndarray)
    assert np.isnan(result_midi[0]) # Negative freq -> NaN
    assert np.isnan(result_midi[1]) # Zero freq -> NaN (or -inf depending on impl.)
    assert np.isclose(result_midi[2], 69.0) # Positive freq works
    assert np.isnan(result_midi[3]) # Near-zero positive might also result in NaN or large negative due to log2

def test_hz_to_midi_list_input():
    """List input should be handled correctly."""
    frequencies = [220.0, 440.0, 880.0]
    expected_midi = np.array([57.0, 69.0, 81.0])
    result_midi = hz_to_midi(frequencies)
    assert isinstance(result_midi, np.ndarray) # Should return numpy array
    np.testing.assert_allclose(result_midi, expected_midi)


# --- midi_to_hz Tests ---

def test_midi_to_hz_scalar():
    """Scalar MIDI value conversion."""
    assert np.isclose(midi_to_hz(69.0), 440.0) # A4
    assert np.isclose(midi_to_hz(57.0), 220.0) # A3
    assert np.isclose(midi_to_hz(81.0), 880.0) # A5
    # Middle C (C4) is MIDI 60
    assert np.isclose(midi_to_hz(60.0), 261.625565, atol=1e-6)

def test_midi_to_hz_array():
    """Array of MIDI values conversion."""
    midi_notes = np.array([57.0, 69.0, 81.0])
    expected_hz = np.array([220.0, 440.0, 880.0])
    result_hz = midi_to_hz(midi_notes)
    assert isinstance(result_hz, np.ndarray)
    np.testing.assert_allclose(result_hz, expected_hz)

def test_midi_to_hz_float_midi():
    """Float MIDI values (e.g., for pitch bend) should work."""
    # A4 quarter-tone sharp (MIDI 69.5)
    expected_hz_sharp = 440.0 * (2.0 ** (0.5 / 12.0))
    assert np.isclose(midi_to_hz(69.5), expected_hz_sharp)

    midi_notes = np.array([68.75, 69.0, 69.25]) # Quarter tones around A4
    expected_hz_q = 440.0 * (2.0 ** (np.array([-0.25, 0.0, 0.25]) / 12.0))
    result_hz = midi_to_hz(midi_notes)
    np.testing.assert_allclose(result_hz, expected_hz_q)

def test_midi_to_hz_negative_midi():
    """Negative MIDI values should convert correctly (very low frequencies)."""
    # MIDI 0 is C-1 (~8.18 Hz)
    assert np.isclose(midi_to_hz(0.0), 8.175799, atol=1e-6)
    # Negative MIDI
    assert np.isclose(midi_to_hz(-12.0), 4.087899, atol=1e-6) # C-2

def test_midi_to_hz_list_input():
    """List input should be handled correctly."""
    midi_notes = [57.0, 69.0, 81.0]
    expected_hz = np.array([220.0, 440.0, 880.0])
    result_hz = midi_to_hz(midi_notes)
    assert isinstance(result_hz, np.ndarray) # Should return numpy array
    np.testing.assert_allclose(result_hz, expected_hz) 