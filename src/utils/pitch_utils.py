"""
ピッチ操作関連のユーティリティ関数モジュール

このモジュールでは、ピッチ（周波数）に関連する変換関数やユーティリティを提供します。
"""

from typing import Union

import numba
import numpy as np


def hz_to_midi(frequencies: Union[float, np.ndarray, list]) -> Union[float, np.ndarray]:
    """
    周波数（Hz）からMIDIノート番号に変換する

    Note
    ----
    この実装は特にNumbaとの互換性とエラー処理の一貫性のために維持されています。
    単純な変換のみが必要な場合はlibrosa.hz_to_midi()を使用してください。

    Parameters
    ----------
    frequencies : float or numpy.ndarray or list
        周波数の値または配列 (Hz)

    Returns
    -------
    float or numpy.ndarray
        MIDIノート番号の値または配列
    """
    # リストまたは他の配列型の入力をnumpy配列に変換
    if not isinstance(frequencies, np.ndarray) and not np.isscalar(frequencies):
        frequencies = np.asarray(frequencies, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        midi_notes = 12 * np.log2(frequencies / 440.0) + 69

    # 無効な値（0Hz以下）を0に設定
    if isinstance(midi_notes, np.ndarray):
        midi_notes[~np.isfinite(midi_notes)] = 0
    else:
        if not np.isfinite(midi_notes):
            midi_notes = 0

    return midi_notes


def midi_to_hz(midi_notes: Union[float, np.ndarray, list]) -> Union[float, np.ndarray]:
    """
    MIDIノート番号から周波数（Hz）に変換する

    Note
    ----
    この実装は特にNumbaとの互換性とエラー処理の一貫性のために維持されています。
    単純な変換のみが必要な場合はlibrosa.hz_to_midi()を使用してください。

    Parameters
    ----------
    midi_notes : float or numpy.ndarray or list
        MIDIノート番号の値または配列

    Returns
    -------
    float or numpy.ndarray
        周波数の値または配列 (Hz)
    """
    # リストまたは他の配列型の入力をnumpy配列に変換
    scalar_input = np.isscalar(midi_notes)
    if not isinstance(midi_notes, np.ndarray) and not scalar_input:
        midi_notes = np.asarray(midi_notes, dtype=float)

    # 無効な値（負または0）を0周波数にマッピング
    if isinstance(midi_notes, np.ndarray):
        # ゼロ以下の値を0にする
        midi_notes = np.copy(midi_notes)  # 元の配列を変更しないようにコピー
        zero_mask = midi_notes <= 0

        # 周波数を計算
        frequencies = 440.0 * (2.0 ** ((midi_notes - 69) / 12.0))

        # 無効なノートを0Hzに設定
        frequencies[zero_mask] = 0.0

        return frequencies
    else:
        if midi_notes <= 0:
            return 0.0
        return 440.0 * (2.0 ** ((midi_notes - 69) / 12.0))


def cents_to_ratio(cents: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    セント値を周波数比に変換する

    Parameters
    ----------
    cents : float or numpy.ndarray
        セント値

    Returns
    -------
    float or numpy.ndarray
        周波数比
    """
    return 2.0 ** (cents / 1200.0)


def ratio_to_cents(ratio: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    周波数比をセント値に変換する

    Parameters
    ----------
    ratio : float or numpy.ndarray
        周波数比

    Returns
    -------
    float or numpy.ndarray
        セント値
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(ratio)

    # 無効な値を0に設定
    if isinstance(cents, np.ndarray):
        cents[~np.isfinite(cents)] = 0
    else:
        if not np.isfinite(cents):
            cents = 0

    return cents


# --- Numba accelerated versions --- #


@numba.njit(cache=True, fastmath=True)
def hz_to_cents(frequencies: np.ndarray, reference_hz: float = 10.0) -> np.ndarray:
    """
    周波数（Hz）配列をセント値配列に変換する（参照周波数を使用, Numba用）

    Parameters
    ----------
    frequencies : numpy.ndarray
        周波数の配列 (Hz)
    reference_hz : float, optional
        セント計算の参照周波数 (デフォルト: 10.0 Hz)

    Returns
    -------
    numpy.ndarray
        セント値の配列
    """
    # Ensure reference_hz is positive
    if reference_hz <= 0:
        # Numbaはエラー送出を直接サポートしないため、np.infやnanを返す
        return np.full_like(frequencies, -np.inf)

    # isinstanceチェックを削除し、配列処理に特化
    cents_array = np.empty_like(frequencies, dtype=np.float64)
    for i in range(len(frequencies)):
        freq = frequencies[i]
        if freq <= 0:
            cents_array[i] = -np.inf  # Or another indicator for invalid input
        else:
            # np.log2 は Numba でサポートされている
            cents_array[i] = 1200.0 * np.log2(freq / reference_hz)
    return cents_array
    # Scalar handling is removed for Numba nopython compatibility


@numba.njit(cache=True, fastmath=True)
def cents_to_hz(cents: np.ndarray, reference_hz: float = 10.0) -> np.ndarray:
    """
    セント値配列を周波数（Hz）配列に変換する（参照周波数を使用, Numba用）

    Parameters
    ----------
    cents : numpy.ndarray
        セント値の配列
    reference_hz : float, optional
        セント計算の参照周波数 (デフォルト: 10.0 Hz)

    Returns
    -------
    numpy.ndarray
        周波数（Hz）の配列
    """
    if reference_hz <= 0:
        return np.zeros_like(
            cents, dtype=np.float64
        )  # Return 0 Hz for invalid reference

    # isinstanceチェックを削除し、配列処理に特化
    # Assuming valid finite cents input for simplicity here.
    # 2.0**x は Numba でサポートされている
    return reference_hz * (2.0 ** (cents / 1200.0))
    # Scalar handling is removed


# ... rest of the file remains unchanged ...
