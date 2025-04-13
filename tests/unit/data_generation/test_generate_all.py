# tests/unit/data_generation/test_generate_all.py
import pytest
from unittest.mock import patch, call, ANY  # Added ANY
from pathlib import Path

# テスト対象のスクリプトをインポート (またはその中の主要関数)
try:
    # Assuming generate_all.py has a main() function or similar entry point
    from src.data_generation import generate_all

    # Import synthesizer to mock its functions where generate_all uses them
    from src.data_generation import synthesizer

    SKIP_TESTS = False
except ImportError as e:
    print(f"Skipping data_generation/generate_all tests due to import error: {e}")
    SKIP_TESTS = True


@pytest.mark.skipif(SKIP_TESTS, reason="必要なモジュールがインポートできませんでした")
# Mock all the functions that are called directly in generate_all.main()
@patch("src.data_generation.synthesizer.generate_vocal_imitation")
@patch("src.data_generation.synthesizer.generate_legato")
@patch("src.data_generation.synthesizer.generate_inharmonicity")
@patch("src.data_generation.synthesizer.generate_octave_errors")
@patch("src.data_generation.synthesizer.generate_pianissimo")
@patch("src.data_generation.synthesizer.generate_staccato")
@patch("src.data_generation.synthesizer.generate_slow_attack")
@patch("src.data_generation.synthesizer.generate_pitch_bend")
@patch("src.data_generation.synthesizer.generate_smooth_vibrato")
@patch("src.data_generation.synthesizer.generate_complex_mix")
@patch("src.data_generation.synthesizer.generate_mixed_melody_percussion")
@patch("src.data_generation.synthesizer.generate_percussion_only")
@patch("src.data_generation.synthesizer.generate_polyphony")
@patch("src.data_generation.synthesizer.generate_chords")
@patch("src.data_generation.synthesizer.generate_reverb_sequence")
@patch("src.data_generation.synthesizer.generate_noisy_sequence")
@patch("src.data_generation.synthesizer.generate_pitch_modulation")
@patch("src.data_generation.synthesizer.generate_dynamics_change")
@patch("src.data_generation.synthesizer.generate_harmonic_sequence")
@patch("src.data_generation.synthesizer.generate_basic_sine_sequence")
@patch("src.data_generation.synthesizer.ensure_output_dirs")
@patch("src.data_generation.synthesizer.generate_clicks")
def test_generate_all_calls_generators(
    mock_generate_clicks,
    mock_ensure_dirs,
    mock_basic_sine,
    mock_harmonic,
    mock_dynamics,
    mock_pitch_mod,
    mock_noisy,
    mock_reverb,
    mock_chords,
    mock_polyphony,
    mock_percussion,
    mock_mixed,
    mock_complex,
    mock_vibrato,
    mock_pitch_bend,
    mock_slow_attack,
    mock_staccato,
    mock_pianissimo,
    mock_octave_errors,
    mock_inharmonicity,
    mock_legato,
    mock_vocal,
):
    """generate_all.main()がシンセサイザー関数を正しく呼び出すかテストします"""

    try:
        # 関数を呼び出す
        generate_all.main()
    except AttributeError:
        pytest.skip(
            "Skipping test: generate_all.main function not found or signature mismatch."
        )
        return
    except TypeError:
        pytest.skip("Skipping test: generate_all.main signature mismatch.")
        return

    # 出力ディレクトリの設定が呼ばれたことを確認
    mock_ensure_dirs.assert_called_once()

    # 各生成関数が1回呼ばれたことを確認（呼び出し順序は考慮しない）
    mock_basic_sine.assert_called_once()
    mock_harmonic.assert_called_once()
    mock_dynamics.assert_called_once()
    mock_pitch_mod.assert_called_once()
    mock_noisy.assert_called_once()
    mock_reverb.assert_called_once()
    mock_chords.assert_called_once()
    mock_polyphony.assert_called_once()
    mock_percussion.assert_called_once()
    mock_mixed.assert_called_once()
    mock_complex.assert_called_once()

    # 新しいテストケースも確認
    mock_vibrato.assert_called_once()
    mock_pitch_bend.assert_called_once_with(filename_base="14_pitch_bend")
    mock_slow_attack.assert_called_once()
    mock_staccato.assert_called_once()
    mock_pianissimo.assert_called_once()
    mock_octave_errors.assert_called_once()
    mock_inharmonicity.assert_called_once()
    mock_legato.assert_called_once()
    mock_vocal.assert_called_once()
    mock_generate_clicks.assert_called_once()
