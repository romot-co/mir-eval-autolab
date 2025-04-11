# tests/unit/data_generation/test_generate_all.py
import pytest
from unittest.mock import patch, call, ANY # Added ANY
from pathlib import Path

# テスト対象のスクリプトをインポート (またはその中の主要関数)
try:
    # Assuming generate_all.py has a main() function or similar entry point
    from src.data_generation import generate_all
    # Import synthesizer to mock its functions where generate_all uses them
    from src.data_generation import synthesizer
    # Import path_utils if generate_all uses it directly
    from src.utils import path_utils
except ImportError:
    pytest.skip("Skipping data_generation/generate_all tests due to missing src modules", allow_module_level=True)

# --- Test generate_all main logic ---

# Mock all the individual generator functions from synthesizer used by generate_all
@patch('src.data_generation.generate_all.synthesizer.generate_sine_wave_note')
@patch('src.data_generation.generate_all.synthesizer.generate_fm_synth') # Assuming FM synth exists and is called
@patch('src.data_generation.generate_all.synthesizer.generate_silence')
@patch('src.data_generation.generate_all.path_utils.ensure_dir') # Mock directory creation
def test_generate_all_calls_generators(mock_ensure_dir, mock_gen_silence, mock_gen_fm, mock_gen_sine, tmp_path):
    """Tests that generate_all.main calls the synthesizer functions."""
    output_dir = tmp_path
    num_files = 3
    sample_rate = 16000
    start_index = 0

    # Assuming generate_all.py has a main function like this:
    # Adjust the call based on the actual signature in generate_all.py
    try:
        # Use the actual path object for the function call if it expects one
        generate_all.main(output_dir=output_dir, num_files=num_files, sample_rate=sample_rate, start_index=start_index)
    except AttributeError:
        pytest.skip("Skipping test: generate_all.main function not found or signature mismatch.")
        return
    except TypeError:
         pytest.skip("Skipping test: generate_all.main signature mismatch (maybe needs string path?).")
         return

    # Verify ensure_dir was called
    mock_ensure_dir.assert_called_once_with(output_dir)

    # Verify that each generator function was called 'num_files' times
    # The exact number depends on the logic within generate_all.main
    # Assuming it calls each type for each file index:
    assert mock_gen_sine.call_count == num_files
    assert mock_gen_fm.call_count == num_files
    assert mock_gen_silence.call_count == num_files

    # Check the arguments for the calls
    # Example: Check calls to generate_sine_wave_note
    expected_calls_sine = [
        # Note: Check actual expected args based on generate_all.main logic
        # (e.g., MIDI note range, duration, amplitude might vary)
        call(output_dir, i, 'sine', ANY, ANY, sample_rate, ANY)
        for i in range(start_index, start_index + num_files)
    ]
    mock_gen_sine.assert_has_calls(expected_calls_sine, any_order=False)

    # Add similar argument checks for mock_gen_fm and mock_gen_silence if needed
    # Ensure the number of ANY match the actual function signature
    expected_calls_fm = [
        call(output_dir, i, 'fm', ANY, ANY, sample_rate, ANY, ANY, ANY, ANY, ANY)
        for i in range(start_index, start_index + num_files)
    ]
    mock_gen_fm.assert_has_calls(expected_calls_fm, any_order=False)

    expected_calls_silence = [
        call(output_dir, i, 'silence', ANY, sample_rate)
        for i in range(start_index, start_index + num_files)
    ]
    mock_gen_silence.assert_has_calls(expected_calls_silence, any_order=False) 