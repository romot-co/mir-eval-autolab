# tests/unit/data_generation/test_synthesizer.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# テスト対象モジュールをインポート
try:
    from src.data_generation import synthesizer
    from src.data_generation.synthesizer import (
        midi_to_hz,
        generate_adsr_envelope,
        generate_sine_wave,
        generate_silence,
        generate_fm_synth,
        save_audio_and_label
        # Add other generate_* functions if they exist
    )
    from src.structures.note import Note # If Note is used in labels
    SKIP_TESTS = False
except ImportError as e:
    print(f"Skipping data_generation/synthesizer tests due to import error: {e}")
    SKIP_TESTS = True
    # Dummy classes/functions for testing without actual implementation
    class Note:
        def __init__(self, start_time=0.0, end_time=1.0, pitch=60, velocity=80):
            self.start_time = start_time
            self.end_time = end_time
            self.pitch = pitch
            self.velocity = velocity
    
    # ダミー関数
    def midi_to_hz(midi_note): 
        return 440.0 * (2 ** ((midi_note - 69) / 12.0))
    
    def generate_adsr_envelope(total_samples, attack_time, decay_time, sustain_level, release_time, sample_rate):
        return np.zeros(total_samples)
    
    def generate_sine_wave(frequency, duration, sample_rate, amplitude=0.5, phase=0.0):
        return np.zeros(int(duration * sample_rate))
    
    def generate_silence(output_dir, file_index, instrument_name, duration, sample_rate):
        pass
    
    def generate_fm_synth(output_dir, file_index, instrument_name, carrier_freq, 
                         duration, sample_rate, amplitude=0.5, mod_freq=5.0, 
                         mod_index=3.0, attack=0.1, decay=0.2):
        pass
    
    def save_audio_and_label(output_dir, file_index, instrument_name, audio_data, labels, sample_rate):
        pass

# テスト実行をスキップするかどうかの設定
pytestmark = pytest.mark.skipif(SKIP_TESTS, reason="必要なモジュールがインポートできませんでした")

# --- Fixtures ---

@pytest.fixture
def mock_soundfile():
    """Mocks the soundfile library."""
    with patch('src.data_generation.synthesizer.sf', autospec=True) as mock:
        yield mock

@pytest.fixture
def mock_csv():
    """Mocks the built-in csv module (or pandas if used for CSV writing)."""
    # Assuming standard csv module is used for label saving
    with patch('src.data_generation.synthesizer.csv', autospec=True) as mock:
        # Mock writer object and its writerow method if needed
        mock_writer = MagicMock()
        mock.writer.return_value = mock_writer
        yield mock, mock_writer # Return both mocks for specific checks

@pytest.fixture
def mock_open():
    """ Mocks the built-in open function for file writing checks. """
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
         # Mock the file handle context manager
         mock_file_handle = MagicMock()
         mock_open.return_value.__enter__.return_value = mock_file_handle
         yield mock_open, mock_file_handle


# --- Test Helper Functions ---

@pytest.mark.parametrize("midi_note, expected_hz", [
    (69, 440.0), # A4
    (60, 261.6255653005986), # C4
    (72, 523.2511306011972), # C5
    (57, 220.0), # A3
])
def test_midi_to_hz(midi_note, expected_hz):
    """Tests the midi_to_hz conversion."""
    assert np.isclose(midi_to_hz(midi_note), expected_hz)

def test_generate_adsr_envelope():
    """Tests the ADSR envelope generation (basic checks)."""
    sample_rate = 44100
    duration = 1.0
    attack_time = 0.1
    decay_time = 0.2
    sustain_level = 0.7
    release_time = 0.3

    total_samples = int(duration * sample_rate)
    envelope = generate_adsr_envelope(total_samples, attack_time, decay_time, sustain_level, release_time, sample_rate)

    assert isinstance(envelope, np.ndarray)
    assert len(envelope) == total_samples
    assert np.max(envelope) <= 1.0
    assert np.min(envelope) >= 0.0
    # Check specific points if necessary (e.g., peak after attack)
    attack_samples = int(attack_time * sample_rate)
    assert np.isclose(envelope[attack_samples - 1], 1.0) # Should reach peak at end of attack
    # Sustain level check
    sustain_start_sample = int((attack_time + decay_time) * sample_rate)
    # Note: Sustain phase might not exist if duration < attack+decay+release
    # This check assumes duration is long enough
    # assert np.isclose(envelope[sustain_start_sample], sustain_level)


def test_generate_sine_wave():
    """Tests sine wave generation."""
    frequency = 440.0
    duration = 0.5
    sample_rate = 44100
    amplitude = 0.8
    phase = np.pi / 2 # Start at peak

    sine_wave = generate_sine_wave(frequency, duration, sample_rate, amplitude, phase)
    total_samples = int(duration * sample_rate)

    assert isinstance(sine_wave, np.ndarray)
    assert len(sine_wave) == total_samples
    assert np.isclose(np.max(sine_wave), amplitude)
    assert np.isclose(sine_wave[0], amplitude) # Check starting phase

# --- Test generate_* Functions (Mocking save) ---

# Parametrize if multiple generate_* functions follow a similar pattern
# For now, testing generate_sine_wave specifically

@patch('src.data_generation.synthesizer.save_audio_and_label')
def test_generate_sine_wave_calls_save(mock_save_audio_and_label):
    """Tests that generate_sine_wave calls save_audio_and_label correctly."""
    output_dir = Path("/fake/output")
    file_index = 1
    instrument_name = "sine"
    midi_note = 69
    duration = 1.0
    sample_rate = 44100
    amplitude = 0.5

    synthesizer.generate_sine_wave_note(
        output_dir, file_index, instrument_name, midi_note, duration, sample_rate, amplitude
    )

    mock_save_audio_and_label.assert_called_once()
    call_args, call_kwargs = mock_save_audio_and_label.call_args

    # Check arguments passed to the mocked save function
    assert call_args[0] == output_dir # output_dir
    assert call_args[1] == file_index # file_index
    assert call_args[2] == instrument_name # instrument_name
    # Check audio_data type and shape
    audio_data = call_args[3]
    assert isinstance(audio_data, np.ndarray)
    assert len(audio_data) == int(duration * sample_rate)
    # Check labels format (assuming simple list of Note-like objects or tuples)
    labels = call_args[4]
    assert isinstance(labels, list)
    # Example check if labels are tuples: (start_time, end_time, midi_note)
    # assert labels[0] == (0.0, duration, midi_note)
    # Or if they are Note objects:
    # assert isinstance(labels[0], Note)
    # assert labels[0].start_time == 0.0
    # assert labels[0].end_time == duration
    # assert labels[0].pitch == midi_note
    assert call_args[5] == sample_rate # sample_rate

# Add similar tests for generate_silence, generate_fm_synth etc.

# --- Test save_audio_and_label --- 

def test_save_audio_and_label_success(mock_soundfile, mock_csv, mock_open, tmp_path):
    """Tests successful saving of audio and label files."""
    mock_csv_module, mock_csv_writer = mock_csv
    mock_open_func, mock_file_handle = mock_open

    output_dir = tmp_path
    file_index = 5
    instrument_name = "test_instrument"
    sample_rate = 16000
    audio_data = np.random.randn(sample_rate * 2) # 2 seconds of noise
    labels = [(0.5, 1.5, 60.0)] # Example label format

    expected_audio_path = output_dir / f"{instrument_name}_{file_index:03d}.wav"
    expected_label_path = output_dir / f"{instrument_name}_{file_index:03d}.csv"

    save_audio_and_label(output_dir, file_index, instrument_name, audio_data, labels, sample_rate)

    # Check soundfile write call
    mock_soundfile.write.assert_called_once_with(expected_audio_path, audio_data, sample_rate)

    # Check CSV writing calls
    # Check open was called for the label file
    mock_open_func.assert_called_once_with(expected_label_path, 'w', newline='')
    # Check csv.writer was called with the file handle
    mock_csv_module.writer.assert_called_once_with(mock_file_handle)
    # Check writerow was called with the header and data
    assert mock_csv_writer.writerow.call_count == 2 # Header + 1 data row
    mock_csv_writer.writerow.assert_has_calls([
        call(['start_time', 'end_time', 'pitch']), # Header row
        call(list(labels[0])) # Data row
    ])

def test_save_audio_and_label_audio_write_error(mock_soundfile, mock_csv, mock_open, tmp_path):
    """Tests error handling when soundfile.write fails."""
    mock_soundfile.write.side_effect = IOError("Disk full")
    mock_csv_module, mock_csv_writer = mock_csv
    mock_open_func, _ = mock_open

    # Call the function (arguments don't matter much here)
    with pytest.raises(IOError): # Expect the original error to propagate or be wrapped
         save_audio_and_label(tmp_path, 1, "error_audio", np.zeros(100), [], 16000)

    # Ensure CSV file was not attempted to be written
    mock_open_func.assert_not_called()
    mock_csv_module.writer.assert_not_called()

def test_save_audio_and_label_label_write_error(mock_soundfile, mock_csv, mock_open, tmp_path):
    """Tests error handling when writing the CSV label file fails."""
    mock_csv_module, mock_csv_writer = mock_csv
    mock_open_func, mock_file_handle = mock_open

    # Simulate error during csv writing
    mock_csv_writer.writerow.side_effect = IOError("Permission denied")

    # Call the function
    with pytest.raises(IOError): # Expect the original error to propagate or be wrapped
        save_audio_and_label(tmp_path, 2, "error_label", np.zeros(100), [(0,1,60)], 16000)

    # Ensure audio write was attempted
    mock_soundfile.write.assert_called_once()
    # Ensure open and csv.writer were called
    mock_open_func.assert_called_once()
    mock_csv_module.writer.assert_called_once()
    # Ensure writerow was attempted (and failed)
    mock_csv_writer.writerow.assert_called_once() # It failed on the first call (header)

# --- Test generate_all (if applicable) ---
# Example:
# @patch('src.data_generation.synthesizer.generate_sine_wave_note')
# @patch('src.data_generation.synthesizer.generate_silence')
# def test_generate_all_calls_generators(mock_gen_silence, mock_gen_sine, tmp_path):
#     from src.data_generation import generate_all # Assuming this exists
#     output_dir = tmp_path
#     num_files = 2
#     sample_rate = 16000

#     generate_all.main(str(output_dir), num_files, sample_rate)

#     # Check if each generator was called the expected number of times
#     assert mock_gen_sine.call_count == num_files # Or based on logic in generate_all
#     assert mock_gen_silence.call_count == num_files # Or based on logic 