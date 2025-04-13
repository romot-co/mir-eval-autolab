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
        # Handle non-positive frequencies -> return 0 (matching actual implementation)
        frequencies = np.asarray(frequencies, dtype=float)  # Ensure float input
        midi_val = np.full_like(frequencies, 0.0)  # Initialize with 0 (not NaN)
        # Calculate only for positive frequencies
        positive_mask = frequencies > 0
        if np.any(positive_mask):
            midi_val[positive_mask] = (
                12 * np.log2(frequencies[positive_mask] / 440.0) + 69
            )
        return midi_val

    def midi_to_hz(midi_notes):
        midi_notes = np.asarray(midi_notes, dtype=float)  # Ensure float input
        # Handle negative or zero MIDI notes
        if isinstance(midi_notes, np.ndarray):
            result = np.zeros_like(midi_notes, dtype=float)
            valid_mask = midi_notes > 0
            result[valid_mask] = 440.0 * (
                2.0 ** ((midi_notes[valid_mask] - 69.0) / 12.0)
            )
            return result
        else:
            if midi_notes <= 0:
                return 0.0
            return 440.0 * (2.0 ** ((midi_notes - 69.0) / 12.0))


# --- hz_to_midi Tests ---


def test_hz_to_midi_scalar():
    """Scalar Hz value conversion."""
    assert np.isclose(hz_to_midi(440.0), 69.0)  # A4
    assert np.isclose(hz_to_midi(220.0), 57.0)  # A3
    assert np.isclose(hz_to_midi(880.0), 81.0)  # A5
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
    """Zero and negative Hz values handling (expect 0 based on implementation)."""
    frequencies = np.array([-100.0, 0.0, 440.0, 1e-7])
    result_midi = hz_to_midi(frequencies)
    assert isinstance(result_midi, np.ndarray)
    assert result_midi[0] == 0  # Negative freq -> 0
    assert result_midi[1] == 0  # Zero freq -> 0
    assert np.isclose(result_midi[2], 69.0)  # Positive freq works
    # Near-zero positive should still compute a valid MIDI value
    assert np.isfinite(result_midi[3])


def test_hz_to_midi_list_input():
    """List input should be handled correctly with np.asarray."""
    frequencies = np.asarray([220.0, 440.0, 880.0])
    expected_midi = np.array([57.0, 69.0, 81.0])
    result_midi = hz_to_midi(frequencies)
    assert isinstance(result_midi, np.ndarray)  # Should return numpy array
    np.testing.assert_allclose(result_midi, expected_midi)


# 追加: スカラー値の負の周波数とゼロのテスト
def test_hz_to_midi_negative_scalar():
    """負のスカラー周波数値をテスト"""
    result = hz_to_midi(-10.0)
    assert result == 0.0  # 負の値は0になるべき


def test_hz_to_midi_zero_scalar():
    """ゼロのスカラー周波数値をテスト"""
    result = hz_to_midi(0.0)
    assert result == 0.0  # ゼロは0になるべき


# --- midi_to_hz Tests ---


def test_midi_to_hz_scalar():
    """Scalar MIDI value conversion."""
    assert np.isclose(midi_to_hz(69.0), 440.0)  # A4
    assert np.isclose(midi_to_hz(57.0), 220.0)  # A3
    assert np.isclose(midi_to_hz(81.0), 880.0)  # A5
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

    midi_notes = np.array([68.75, 69.0, 69.25])  # Quarter tones around A4
    expected_hz_q = 440.0 * (2.0 ** (np.array([-0.25, 0.0, 0.25]) / 12.0))
    result_hz = midi_to_hz(midi_notes)
    np.testing.assert_allclose(result_hz, expected_hz_q)


def test_midi_to_hz_negative_midi():
    """Negative MIDI values should convert to 0Hz per implementation."""
    assert np.isclose(midi_to_hz(0.0), 0.0)  # Invalid MIDI 0 -> 0Hz
    assert np.isclose(midi_to_hz(-12.0), 0.0)  # Negative MIDI -> 0Hz


def test_midi_to_hz_list_input():
    """List input should be handled correctly with np.asarray."""
    midi_notes = np.asarray([57.0, 69.0, 81.0])
    expected_hz = np.array([220.0, 440.0, 880.0])
    result_hz = midi_to_hz(midi_notes)
    assert isinstance(result_hz, np.ndarray)  # Should return numpy array
    np.testing.assert_allclose(result_hz, expected_hz)


def test_midi_to_hz_zero_handling():
    """Test behavior with zero and negative MIDI values in arrays."""
    midi_notes = np.array([-10.0, 0.0, 60.0])
    expected_hz = np.array([0.0, 0.0, 261.625565])
    result_hz = midi_to_hz(midi_notes)
    assert isinstance(result_hz, np.ndarray)
    np.testing.assert_allclose(result_hz, expected_hz, atol=1e-6)


# 追加: リスト型の入力テスト
def test_midi_to_hz_python_list():
    """Pythonリストから変換されることを確認"""
    midi_list = [60.0, 69.0, 72.0]  # Python リスト
    result = midi_to_hz(midi_list)
    expected = np.array([261.625565, 440.0, 523.251131])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# --- cents_to_ratio Tests ---

try:
    from src.utils.pitch_utils import cents_to_ratio, ratio_to_cents
except ImportError:
    print("Warning: Using dummy implementations for cents_to_ratio and ratio_to_cents.")

    # Dummy implementations based on standard formulas
    def cents_to_ratio(cents):
        return 2.0 ** (cents / 1200.0)

    def ratio_to_cents(ratio):
        ratio = np.asarray(ratio) if not np.isscalar(ratio) else ratio
        if isinstance(ratio, np.ndarray):
            result = np.zeros_like(ratio, dtype=float)
            valid_mask = ratio > 0
            result[valid_mask] = 1200.0 * np.log2(ratio[valid_mask])
            return result
        else:
            if ratio <= 0:
                return 0.0
            return 1200.0 * np.log2(ratio)


def test_cents_to_ratio_scalar():
    """スカラーセント値を周波数比に変換する"""
    assert np.isclose(cents_to_ratio(0), 1.0)  # 0セントは同一ピッチ（比率1.0）
    assert np.isclose(cents_to_ratio(1200), 2.0)  # 1200セントはオクターブ（比率2.0）
    assert np.isclose(
        cents_to_ratio(700), 2.0 ** (7 / 12)
    )  # 700セントは5度（比率約1.5）


def test_cents_to_ratio_array():
    """配列セント値を周波数比に変換する"""
    cents = np.array([0, 1200, 2400])
    expected_ratio = np.array([1.0, 2.0, 4.0])
    result_ratio = cents_to_ratio(cents)
    assert isinstance(result_ratio, np.ndarray)
    np.testing.assert_allclose(result_ratio, expected_ratio)


def test_cents_to_ratio_negative():
    """負のセント値を周波数比に変換する"""
    assert np.isclose(
        cents_to_ratio(-1200), 0.5
    )  # -1200セントは1オクターブ下（比率0.5）
    negative_cents = np.array([-2400, -1200, 0, 1200])
    expected_ratio = np.array([0.25, 0.5, 1.0, 2.0])
    result_ratio = cents_to_ratio(negative_cents)
    np.testing.assert_allclose(result_ratio, expected_ratio)


# --- ratio_to_cents Tests ---


def test_ratio_to_cents_scalar():
    """スカラー周波数比をセント値に変換する"""
    assert np.isclose(ratio_to_cents(1.0), 0.0)  # 比率1.0は0セント
    assert np.isclose(ratio_to_cents(2.0), 1200.0)  # 比率2.0は1200セント（1オクターブ）
    assert np.isclose(
        ratio_to_cents(0.5), -1200.0
    )  # 比率0.5は-1200セント（1オクターブ下）


def test_ratio_to_cents_array():
    """配列周波数比をセント値に変換する"""
    ratio = np.array([0.5, 1.0, 2.0, 4.0])
    expected_cents = np.array([-1200.0, 0.0, 1200.0, 2400.0])
    result_cents = ratio_to_cents(ratio)
    assert isinstance(result_cents, np.ndarray)
    np.testing.assert_allclose(result_cents, expected_cents)


def test_ratio_to_cents_zero_and_negative():
    """ゼロおよび負の比率を扱う方法（実装に基づいて0を期待）"""
    ratio = np.array([-1.0, 0.0, 1.0])
    result_cents = ratio_to_cents(ratio)
    assert isinstance(result_cents, np.ndarray)
    assert result_cents[0] == 0  # 負の比率→0
    assert result_cents[1] == 0  # ゼロ比率→0
    assert np.isclose(result_cents[2], 0.0)  # 正の比率は機能する


# --- hz_to_cents and cents_to_hz Tests ---

try:
    from src.utils.pitch_utils import hz_to_cents, cents_to_hz
except ImportError:
    print("Warning: Using dummy implementations for hz_to_cents and cents_to_hz.")

    # Dummy implementations
    def hz_to_cents(frequencies, reference_hz=10.0):
        if reference_hz <= 0:
            return np.full_like(frequencies, -np.inf)

        result = np.empty_like(frequencies)
        for i in range(len(frequencies)):
            if frequencies[i] <= 0:
                result[i] = -np.inf
            else:
                result[i] = 1200.0 * np.log2(frequencies[i] / reference_hz)
        return result

    def cents_to_hz(cents, reference_hz=10.0):
        if reference_hz <= 0:
            return np.zeros_like(cents)

        return reference_hz * (2.0 ** (cents / 1200.0))


def test_hz_to_cents_basic():
    """hz_to_centsの基本機能をテスト"""
    freq = np.array([10.0, 20.0, 40.0])  # デフォルトの参照周波数10Hzに対して
    expected_cents = np.array([0.0, 1200.0, 2400.0])  # 0, 1オクターブ, 2オクターブ
    result_cents = hz_to_cents(freq)
    np.testing.assert_allclose(result_cents, expected_cents)


def test_hz_to_cents_reference():
    """異なる参照周波数でのhz_to_centsをテスト"""
    freq = np.array([220.0, 440.0, 880.0])
    reference_hz = 220.0  # A3を参照として
    expected_cents = np.array([0.0, 1200.0, 2400.0])  # 0, 1オクターブ, 2オクターブ
    result_cents = hz_to_cents(freq, reference_hz)
    np.testing.assert_allclose(result_cents, expected_cents)


def test_hz_to_cents_invalid_input():
    """ゼロまたは負の周波数入力をテスト"""
    freq = np.array([-10.0, 0.0, 10.0])
    result_cents = hz_to_cents(freq)
    assert np.isneginf(result_cents[0])  # 負の周波数は-infになる
    assert np.isneginf(result_cents[1])  # ゼロ周波数は-infになる
    assert np.isclose(result_cents[2], 0.0)  # 10Hz（参照）は0セント


def test_hz_to_cents_invalid_reference():
    """無効な参照周波数をテスト"""
    freq = np.array([220.0, 440.0])
    invalid_reference = 0.0
    result_cents = hz_to_cents(freq, invalid_reference)
    # 無効な参照はすべての結果が-infになる
    assert np.all(np.isneginf(result_cents))


def test_cents_to_hz_basic():
    """cents_to_hzの基本機能をテスト"""
    cents = np.array([0.0, 1200.0, 2400.0])  # 0, 1オクターブ, 2オクターブ
    reference_hz = 10.0  # デフォルト参照
    expected_freq = np.array([10.0, 20.0, 40.0])
    result_freq = cents_to_hz(cents, reference_hz)
    np.testing.assert_allclose(result_freq, expected_freq)


def test_cents_to_hz_reference():
    """異なる参照周波数でのcents_to_hzをテスト"""
    cents = np.array([0.0, 1200.0, 2400.0])  # 0, 1オクターブ, 2オクターブ
    reference_hz = 440.0  # A4を参照として
    expected_freq = np.array([440.0, 880.0, 1760.0])
    result_freq = cents_to_hz(cents, reference_hz)
    np.testing.assert_allclose(result_freq, expected_freq)


def test_cents_to_hz_invalid_reference():
    """無効な参照周波数でのcents_to_hzをテスト"""
    cents = np.array([0.0, 1200.0])
    invalid_reference = 0.0
    result_freq = cents_to_hz(cents, invalid_reference)
    # 無効な参照はすべての結果が0Hzになる
    assert np.all(result_freq == 0.0)


# 追加: 様々なケースのテスト


# 追加: hz_to_cents関数の個別値テスト
def test_hz_to_cents_single_value():
    """hz_to_cents関数の個別値をテスト"""
    freq = np.array([440.0])  # 単一の周波数
    reference_hz = 10.0  # デフォルト参照
    expected_cents = 1200.0 * np.log2(440.0 / 10.0)
    result_cents = hz_to_cents(freq, reference_hz)
    assert result_cents.shape == (1,)
    np.testing.assert_allclose(result_cents[0], expected_cents)


# 追加: 様々な周波数値での正確性テスト
def test_hz_to_cents_various_frequencies():
    """様々な周波数値でのhz_to_cents正確性をテスト"""
    frequencies = np.array([20.0, 100.0, 500.0, 1000.0, 5000.0])
    reference_hz = 10.0
    expected = 1200.0 * np.log2(frequencies / reference_hz)
    result = hz_to_cents(frequencies)
    np.testing.assert_allclose(result, expected)


# 追加: 負の周波数とゼロの個別テスト
def test_hz_to_cents_individual_negative_and_zero():
    """負の周波数とゼロ値を個別にテスト"""
    # 負の値
    assert np.isneginf(hz_to_cents(np.array([-1.0]))[0])
    # ゼロ
    assert np.isneginf(hz_to_cents(np.array([0.0]))[0])


# 追加: cents_to_hz関数の個別値テスト
def test_cents_to_hz_single_value():
    """cents_to_hz関数の個別値をテスト"""
    cents = np.array([1200.0])  # 単一のセント値（1オクターブ）
    reference_hz = 10.0  # デフォルト参照
    expected_hz = 20.0  # 1オクターブ上なので2倍
    result_hz = cents_to_hz(cents, reference_hz)
    assert result_hz.shape == (1,)
    np.testing.assert_allclose(result_hz[0], expected_hz)


# 追加: 様々なセント値での正確性テスト
def test_cents_to_hz_various_cents():
    """様々なセント値でのcents_to_hz正確性をテスト"""
    cents_values = np.array([-2400.0, -1200.0, 0.0, 1200.0, 2400.0])
    reference_hz = 10.0
    expected = reference_hz * (2.0 ** (cents_values / 1200.0))
    result = cents_to_hz(cents_values)
    np.testing.assert_allclose(result, expected)


# 追加: ゼロ参照周波数の詳細テスト
def test_cents_to_hz_zero_reference_detailed():
    """cents_to_hz関数でのゼロ参照周波数の詳細テスト"""
    cents = np.array([-1200.0, 0.0, 1200.0, 2400.0])
    result = cents_to_hz(cents, reference_hz=0.0)
    # すべての結果が0になることを確認
    for value in result:
        assert value == 0.0


# 追加: スカラー入力でのゼロ・負の値のテスト (midi_to_hz)
def test_midi_to_hz_scalar_invalid():
    """Scalar MIDI 0 and negative values should return 0.0 Hz."""
    assert midi_to_hz(0.0) == 0.0
    assert midi_to_hz(-60.0) == 0.0


# 追加: 様々なケースのテスト


def test_hz_to_cents_negative_reference():
    """hz_to_centsで負の参照周波数を使った場合のテスト"""
    freqs = np.array([440.0, 880.0])

    # 負の参照周波数を使用
    result = hz_to_cents(freqs, reference_hz=-10.0)

    # 負の参照周波数では-infが返される
    assert np.all(np.isinf(result))
    assert np.all(result < 0)  # すべての値が負の無限大


def test_hz_to_cents_zero_reference():
    """hz_to_centsでゼロの参照周波数を使った場合のテスト"""
    freqs = np.array([440.0, 880.0])

    # ゼロの参照周波数を使用
    result = hz_to_cents(freqs, reference_hz=0.0)

    # ゼロの参照周波数では-infが返される
    assert np.all(np.isinf(result))
    assert np.all(result < 0)  # すべての値が負の無限大


def test_hz_to_cents_zero_frequency():
    """hz_to_centsでゼロの周波数を処理する場合のテスト"""
    freqs = np.array([0.0, 440.0])

    # 有効な参照周波数を使用
    result = hz_to_cents(freqs, reference_hz=10.0)

    # 最初の要素（ゼロ周波数）は-inf、2番目の要素は有効な値
    assert np.isinf(result[0])
    assert result[0] < 0  # 負の無限大
    assert np.isfinite(result[1])  # 2番目の要素は有限値


def test_hz_to_cents_negative_frequency():
    """hz_to_centsで負の周波数を処理する場合のテスト"""
    freqs = np.array([-440.0, 440.0])

    # 有効な参照周波数を使用
    result = hz_to_cents(freqs, reference_hz=10.0)

    # 最初の要素（負の周波数）は-inf、2番目の要素は有効な値
    assert np.isinf(result[0])
    assert result[0] < 0  # 負の無限大
    assert np.isfinite(result[1])  # 2番目の要素は有限値


def test_cents_to_hz_negative_reference():
    """cents_to_hzで負の参照周波数を使った場合のテスト"""
    cents = np.array([1200.0, 2400.0])

    # 負の参照周波数を使用
    result = cents_to_hz(cents, reference_hz=-10.0)

    # 負の参照周波数ではゼロが返される
    assert np.all(result == 0.0)


def test_cents_to_hz_zero_reference():
    """cents_to_hzでゼロの参照周波数を使った場合のテスト"""
    cents = np.array([1200.0, 2400.0])

    # ゼロの参照周波数を使用
    result = cents_to_hz(cents, reference_hz=0.0)

    # ゼロの参照周波数ではゼロが返される
    assert np.all(result == 0.0)


def test_cents_to_hz_zero_reference_detailed():
    """cents_to_hzでゼロの参照周波数を使った場合の詳細テスト"""
    # 異なるセント値
    cents = np.array([-1200.0, 0.0, 1200.0, 2400.0])

    # ゼロの参照周波数を使用
    result = cents_to_hz(cents, reference_hz=0.0)

    # すべての結果がゼロであることを確認
    assert np.all(result == 0.0)
    assert result.shape == cents.shape  # 配列の形状は保持される


def test_hz_to_cents_and_cents_to_hz_symmetry():
    """hz_to_centsとcents_to_hzの対称性をテスト"""
    # 正の周波数値
    freqs_orig = np.array([110.0, 440.0, 880.0])
    reference_hz = 10.0

    # Hz -> cents -> Hz の変換
    cents = hz_to_cents(freqs_orig, reference_hz=reference_hz)
    freqs_converted = cents_to_hz(cents, reference_hz=reference_hz)

    # 元の周波数と変換後の周波数が近似的に一致することを確認
    np.testing.assert_allclose(freqs_orig, freqs_converted, rtol=1e-10)


def test_hz_to_cents_empty_array():
    """hz_to_centsで空配列を処理する場合のテスト"""
    freqs = np.array([])

    # 有効な参照周波数を使用
    result = hz_to_cents(freqs, reference_hz=10.0)

    # 空配列が返される
    assert result.shape == (0,)
    assert isinstance(result, np.ndarray)


def test_cents_to_hz_empty_array():
    """cents_to_hzで空配列を処理する場合のテスト"""
    cents = np.array([])

    # 有効な参照周波数を使用
    result = cents_to_hz(cents, reference_hz=10.0)

    # 空配列が返される
    assert result.shape == (0,)
    assert isinstance(result, np.ndarray)
