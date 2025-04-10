"""
ピッチユーティリティ関数のテスト (`src/utils/pitch_utils.py`)
"""
import pytest
import numpy as np

# テスト対象のモジュールをインポート
try:
    from src.utils import pitch_utils
except ImportError:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.utils import pitch_utils

# --- テストデータ ---
HZ_A4 = 440.0
MIDI_A4 = 69.0
HZ_C0 = 16.35159783
MIDI_C0 = 12.0

# --- テスト関数 ---

# hz_to_midi のテスト
def test_hz_to_midi_scalar():
    """hz_to_midi: スカラー値のテスト"""
    assert pitch_utils.hz_to_midi(HZ_A4) == pytest.approx(MIDI_A4)
    assert pitch_utils.hz_to_midi(HZ_C0) == pytest.approx(MIDI_C0)

def test_hz_to_midi_array():
    """hz_to_midi: NumPy配列のテスト"""
    hz_array = np.array([HZ_C0, HZ_A4, HZ_A4 * 2])
    midi_expected = np.array([MIDI_C0, MIDI_A4, MIDI_A4 + 12])
    np.testing.assert_allclose(pitch_utils.hz_to_midi(hz_array), midi_expected, atol=1e-6)

def test_hz_to_midi_invalid():
    """hz_to_midi: 無効な入力 (0Hz, 負Hz) のテスト"""
    assert pitch_utils.hz_to_midi(0.0) == 0 # 0HzはMIDI 0を返す
    assert pitch_utils.hz_to_midi(-100.0) == 0 # 負HzはMIDI 0を返す
    hz_array = np.array([HZ_A4, 0.0, -10.0])
    midi_expected = np.array([MIDI_A4, 0.0, 0.0])
    np.testing.assert_allclose(pitch_utils.hz_to_midi(hz_array), midi_expected, atol=1e-6)

# midi_to_hz のテスト
def test_midi_to_hz_scalar():
    """midi_to_hz: スカラー値のテスト"""
    assert pitch_utils.midi_to_hz(MIDI_A4) == pytest.approx(HZ_A4)
    assert pitch_utils.midi_to_hz(MIDI_C0) == pytest.approx(HZ_C0)

def test_midi_to_hz_array():
    """midi_to_hz: NumPy配列のテスト"""
    midi_array = np.array([MIDI_C0, MIDI_A4, MIDI_A4 + 12])
    hz_expected = np.array([HZ_C0, HZ_A4, HZ_A4 * 2])
    np.testing.assert_allclose(pitch_utils.midi_to_hz(midi_array), hz_expected, rtol=1e-6)

def test_midi_to_hz_invalid():
    """midi_to_hz: 無効な入力 (MIDI 0, 負MIDI) のテスト"""
    assert pitch_utils.midi_to_hz(0.0) == 0.0 # MIDI 0は 0 Hz を返す
    assert pitch_utils.midi_to_hz(-10.0) == 0.0 # 負MIDIは 0 Hz を返す
    midi_array = np.array([MIDI_A4, 0.0, -5.0])
    hz_expected = np.array([HZ_A4, 0.0, 0.0])
    np.testing.assert_allclose(pitch_utils.midi_to_hz(midi_array), hz_expected, rtol=1e-6)

# cents_to_ratio のテスト
def test_cents_to_ratio():
    """cents_to_ratio のテスト"""
    assert pitch_utils.cents_to_ratio(0) == 1.0
    assert pitch_utils.cents_to_ratio(1200) == pytest.approx(2.0)
    assert pitch_utils.cents_to_ratio(-1200) == pytest.approx(0.5)
    cents_array = np.array([0, 1200, -1200])
    ratio_expected = np.array([1.0, 2.0, 0.5])
    np.testing.assert_allclose(pitch_utils.cents_to_ratio(cents_array), ratio_expected)

# ratio_to_cents のテスト
def test_ratio_to_cents():
    """ratio_to_cents のテスト"""
    assert pitch_utils.ratio_to_cents(1.0) == 0.0
    assert pitch_utils.ratio_to_cents(2.0) == pytest.approx(1200.0)
    assert pitch_utils.ratio_to_cents(0.5) == pytest.approx(-1200.0)
    ratio_array = np.array([1.0, 2.0, 0.5])
    cents_expected = np.array([0.0, 1200.0, -1200.0])
    np.testing.assert_allclose(pitch_utils.ratio_to_cents(ratio_array), cents_expected, atol=1e-6)

def test_ratio_to_cents_invalid():
    """ratio_to_cents: 無効な入力 (0, 負) のテスト"""
    assert pitch_utils.ratio_to_cents(0.0) == 0 # 比率0は 0 セントを返す
    assert pitch_utils.ratio_to_cents(-1.0) == 0 # 負の比率は 0 セントを返す
    ratio_array = np.array([2.0, 0.0, -0.5])
    cents_expected = np.array([1200.0, 0.0, 0.0])
    np.testing.assert_allclose(pitch_utils.ratio_to_cents(ratio_array), cents_expected, atol=1e-6)

# --- Numba版のテスト --- #
# Numba がインストールされていない場合はスキップ
numba = pytest.importorskip("numba")

# hz_to_cents (Numba) のテスト
def test_hz_to_cents_numba():
    """hz_to_cents (Numba版) のテスト"""
    hz_array = np.array([10.0, 20.0, 440.0])
    ref_hz = 10.0
    cents_expected = np.array([0.0, 1200.0, pitch_utils.ratio_to_cents(440.0/10.0)])
    np.testing.assert_allclose(pitch_utils.hz_to_cents(hz_array, ref_hz), cents_expected, atol=1e-6)

def test_hz_to_cents_numba_invalid_freq():
    """hz_to_cents (Numba版): 無効な周波数 (0, 負) のテスト"""
    hz_array = np.array([440.0, 0.0, -10.0])
    ref_hz = 10.0
    # Numba版では無効な周波数は -inf を返す
    cents_result = pitch_utils.hz_to_cents(hz_array, ref_hz)
    assert cents_result[0] == pytest.approx(pitch_utils.ratio_to_cents(440.0/10.0))
    assert np.isneginf(cents_result[1])
    assert np.isneginf(cents_result[2])

def test_hz_to_cents_numba_invalid_ref():
    """hz_to_cents (Numba版): 無効な参照周波数 (0, 負) のテスト"""
    hz_array = np.array([10.0, 20.0])
    # Numba版では無効な参照周波数はすべて -inf を返す
    cents_result_zero = pitch_utils.hz_to_cents(hz_array, 0.0)
    cents_result_neg = pitch_utils.hz_to_cents(hz_array, -10.0)
    assert np.all(np.isneginf(cents_result_zero))
    assert np.all(np.isneginf(cents_result_neg))

# cents_to_hz (Numba) のテスト
def test_cents_to_hz_numba():
    """cents_to_hz (Numba版) のテスト"""
    ref_hz = 10.0
    cents_array = np.array([0.0, 1200.0, -1200.0, pitch_utils.ratio_to_cents(440.0/10.0)])
    hz_expected = np.array([10.0, 20.0, 5.0, 440.0])
    np.testing.assert_allclose(pitch_utils.cents_to_hz(cents_array, ref_hz), hz_expected, rtol=1e-6)

def test_cents_to_hz_numba_invalid_ref():
    """cents_to_hz (Numba版): 無効な参照周波数 (0, 負) のテスト"""
    cents_array = np.array([0.0, 1200.0])
    # Numba版では無効な参照周波数は 0 Hz を返す
    hz_result_zero = pitch_utils.cents_to_hz(cents_array, 0.0)
    hz_result_neg = pitch_utils.cents_to_hz(cents_array, -10.0)
    np.testing.assert_allclose(hz_result_zero, np.array([0.0, 0.0]))
    np.testing.assert_allclose(hz_result_neg, np.array([0.0, 0.0])) 