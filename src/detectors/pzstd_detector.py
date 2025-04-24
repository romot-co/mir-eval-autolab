# -*- coding: utf-8 -*-
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numba  # Numbaをインポート
import numpy as np
from scipy.interpolate import interp1d  # 線形補間用
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.optimize import linear_sum_assignment  # For Hungarian algorithm
from scipy.signal import find_peaks

from src.detectors import register_detector  # <-- 絶対インポートに変更
from src.detectors.base_detector import BaseDetector  # BaseDetectorは別途定義されていると仮定

# --- Logging Configuration ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s') # This should be handled by the main script
logger = logging.getLogger(__name__)

# --- Constants -> Defined within the class now ---
# ... existing code ...
# --- Numba-Optimized Helper Functions ---
# NOTE: Numba 関数の内部ではロギングや複雑な型ヒントは避けるか簡略化します。


@numba.jit(nopython=True, cache=True)
def hz_to_cents_numba(freq_hz: np.ndarray, reference_hz: float = 10.0) -> np.ndarray:
    """Numba-optimized Hz to cents conversion."""
    cents = np.full_like(freq_hz, -np.inf, dtype=np.float64)
    # Use loop instead of mask for nopython mode compatibility if needed,
    # but this vectorized approach often works.
    valid_mask = freq_hz > 0
    cents[valid_mask] = 1200 * np.log2(freq_hz[valid_mask] / reference_hz)
    return cents


@numba.jit(nopython=True, cache=True)
def cents_to_hz_numba(cents: np.ndarray, reference_hz: float = 10.0) -> np.ndarray:
    """Numba-optimized Cents to Hz conversion."""
    return reference_hz * (2 ** (cents / 1200))


@numba.jit(nopython=True, cache=True)
def _parabolic_interpolation_numba(
    y_values: np.ndarray, peak_index: int
) -> Tuple[float, float]:
    """Numba-optimized parabolic interpolation."""
    n = len(y_values)
    if peak_index <= 0 or peak_index >= n - 1:
        # Ensure peak_index is valid before accessing y_values[peak_index]
        if 0 <= peak_index < n:
            return 0.0, y_values[peak_index]
        else:  # Should not happen if called correctly, but handle defensively
            return 0.0, 0.0  # Or raise an error if preferred (Numba can raise errors)

    # Direct indexing should be safe here due to the first check
    y_minus_1 = y_values[peak_index - 1]
    y_0 = y_values[peak_index]
    y_plus_1 = y_values[peak_index + 1]

    denominator = 2.0 * (y_minus_1 - 2.0 * y_0 + y_plus_1)
    if abs(denominator) < 1e-9:
        return 0.0, y_0

    delta_index = (y_minus_1 - y_plus_1) / denominator
    # Clip delta to physically meaningful range [-0.5, 0.5]
    delta_index = max(-0.5, min(delta_index, 0.5))  # Numba prefers simple max/min

    interpolated_value = y_0 - 0.25 * (y_minus_1 - y_plus_1) * delta_index
    # Ensure interpolated value is not unrealistically high
    interpolated_value = max(min(interpolated_value, y_0 * 1.1), 0.0)

    return delta_index, interpolated_value


@numba.jit(nopython=True, cache=True)
def _calculate_harmonic_freqs_numba(
    f0: float,
    max_harmonics: int,
    inharmonicity_factor: float,
    max_f_analysis: float,
    tolerance_cents: float,
) -> np.ndarray:
    """Numba-optimized harmonic frequency calculation."""
    if f0 <= 0:
        return np.zeros(0, dtype=np.float64)

    harmonics = np.arange(1, max_harmonics + 1)
    if inharmonicity_factor == 0.0:
        f_harmonics = f0 * harmonics
    else:
        inharmonicity_term = 1.0 + inharmonicity_factor * harmonics**2
        # Handle unphysical cases directly in calculation
        inharmonicity_term = np.maximum(
            0.0, inharmonicity_term
        )  # Ensure non-negative before sqrt
        f_harmonics = f0 * harmonics * np.sqrt(inharmonicity_term)

    # Filter harmonics beyond the maximum frequency we analyze + tolerance buffer
    # Calculate tolerance in Hz at each harmonic frequency approx
    # tolerance_hz_buffer = f_harmonics * (2**(tolerance_cents / 1200.0) - 1.0) # More accurate Hz buffer
    # Simplified approach for Numba: Check against max_f_analysis directly
    valid_mask = f_harmonics <= max_f_analysis
    f_harmonics_valid = f_harmonics[valid_mask]

    return f_harmonics_valid


@numba.jit(nopython=True, cache=False)  # cache=True を cache=False に変更
def _score_f0_candidate_numba(
    f0_cand_hz: float,
    peak_freqs_cents: np.ndarray,
    peak_amps: np.ndarray,
    peak_spods: np.ndarray,
    previous_f0s_cents: np.ndarray,
    # --- Scoring Parameters ---
    max_harmonics: int,
    inharmonicity_factor: float,
    max_f_analysis: float,
    harmonic_match_tolerance_cents: float,
    pitch_continuity_tolerance_cents: float,
    continuity_bonus: float,
    # <<< HCF Map Parameters 削除 >>>
    # C_map_frame: np.ndarray,
    # AvgHarmonicSPOD_map_frame: np.ndarray,
    # hcf_score_weight_c: float,
    # hcf_score_weight_spod: float,
    # <<< ------------------ >>>
    harmonic_weight_offset: float = 0.5,
    harmonic_weight_exponent: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """Numba-optimized F0 candidate scoring."""
    score = 0.0
    num_available_peaks = len(peak_freqs_cents)
    matched_peak_mask = np.zeros(num_available_peaks, dtype=np.bool_)

    if f0_cand_hz <= 0 or num_available_peaks == 0:
        return 0.0, matched_peak_mask

    # 1. Continuity Bonus
    f0_cand_cents = 1200.0 * np.log2(f0_cand_hz / 10.0)  # Use 10Hz reference directly
    if len(previous_f0s_cents) > 0:
        min_prev_diff = np.min(np.abs(f0_cand_cents - previous_f0s_cents))
        if min_prev_diff < pitch_continuity_tolerance_cents:
            score += continuity_bonus

    # 2. Harmonic Matching Score
    expected_harmonics_hz = _calculate_harmonic_freqs_numba(
        f0_cand_hz,
        max_harmonics,
        inharmonicity_factor,
        max_f_analysis,
        harmonic_match_tolerance_cents,
    )
    if len(expected_harmonics_hz) == 0:
        return score, matched_peak_mask

    expected_harmonics_cents = 1200.0 * np.log2(expected_harmonics_hz / 10.0)

    # Find matches using broadcasting (Numba supports this well)
    # Shape: (n_harmonics, n_available_peaks)
    cent_diffs = np.abs(
        expected_harmonics_cents.reshape(-1, 1) - peak_freqs_cents.reshape(1, -1)
    )

    # Iterate through harmonics to calculate score contributions
    total_matched_amplitude = 0.0
    num_matches = 0

    for h_idx in range(len(expected_harmonics_cents)):
        h = h_idx + 1
        harmonic_cent_diffs = cent_diffs[h_idx, :]  # Diffs for this harmonic
        matches_mask_h = harmonic_cent_diffs <= harmonic_match_tolerance_cents
        match_indices_h = np.where(matches_mask_h)[0]

        if len(match_indices_h) > 0:
            # Find the closest match (in cents) among the valid ones for this harmonic
            closest_idx_in_available = match_indices_h[
                np.argmin(harmonic_cent_diffs[match_indices_h])
            ]

            # Peak properties for scoring
            matched_peak_spod = peak_spods[closest_idx_in_available]
            matched_peak_amp = peak_amps[
                closest_idx_in_available
            ]  # Could use amplitude?
            min_diff_cents = harmonic_cent_diffs[closest_idx_in_available]

            # Scoring components
            # 動的パラメータを使用した重み付け計算
            harmonic_weight = (
                1.0 / (h + harmonic_weight_offset) ** harmonic_weight_exponent
            )
            peak_salience = matched_peak_spod
            # Avoid division by zero if tolerance is zero (though constructor ensures > 0)
            proximity_score = (
                max(0.0, 1.0 - (min_diff_cents / harmonic_match_tolerance_cents))
                if harmonic_match_tolerance_cents > 0
                else (1.0 if min_diff_cents == 0 else 0.0)
            )

            # --- スコア計算を元に戻す ---
            score += peak_salience * harmonic_weight * proximity_score
            # --- HCF/SPOD 重み付けロジック削除 ---
            # base_score = peak_salience * harmonic_weight * proximity_score
            # coherence_factor = 1.0 + hcf_score_weight_c * C_map_frame[f0_cand_idx]
            # spod_factor = 1.0 + hcf_score_weight_spod * AvgHarmonicSPOD_map_frame[f0_cand_idx]
            # score += base_score * max(0.0, coherence_factor) * max(0.0, spod_factor)
            # --- ------------------- ---

            matched_peak_mask[closest_idx_in_available] = True

    return score, matched_peak_mask


@numba.jit(nopython=True, cache=True)
def _calculate_cost_matrix_numba(
    active_track_last_pitches_cents: np.ndarray,
    active_track_stabilities_cents: np.ndarray,
    current_f0s_cents: np.ndarray,
    pitch_continuity_tolerance_cents: float,
    stability_bonus_threshold_cent_dev: float,
    stability_bonus_value: float,
    cost_inf: float,
) -> np.ndarray:
    """Numba-optimized cost matrix calculation for note tracking."""
    num_active_tracks = len(active_track_last_pitches_cents)
    num_current_f0s = len(current_f0s_cents)
    cost_matrix = np.full(
        (num_active_tracks, num_current_f0s), cost_inf, dtype=np.float64
    )

    if num_active_tracks == 0 or num_current_f0s == 0:
        return cost_matrix

    for r in range(num_active_tracks):
        last_pitch_cents = active_track_last_pitches_cents[r]
        stability_cents = active_track_stabilities_cents[r]
        is_stable = stability_cents < stability_bonus_threshold_cent_dev

        for c in range(num_current_f0s):
            f0_cents = current_f0s_cents[c]
            pitch_diff_cents = abs(f0_cents - last_pitch_cents)

            if pitch_diff_cents < pitch_continuity_tolerance_cents:
                cost = pitch_diff_cents
                if is_stable:
                    cost *= 1.0 - stability_bonus_value
                cost_matrix[r, c] = max(0.0, cost)

    return cost_matrix


@numba.njit(cache=True, fastmath=True)
def _interpolate_map_numba(
    map_data_frame: np.ndarray, freq_bin_float: float, n_bins: int
) -> float:
    """
    Numba-optimized map interpolation for a single frame and frequency.
    Uses linear interpolation. Handles boundary conditions.
    """
    if not (0 <= freq_bin_float < n_bins - 1):
        # Handle cases outside the valid interpolation range (e.g., return boundary value or 0)
        if freq_bin_float < 0:
            return map_data_frame[0] if n_bins > 0 else 0.0
        elif freq_bin_float >= n_bins - 1 and n_bins > 0:
            return map_data_frame[n_bins - 1]
        else:  # n_bins = 0 case
            return 0.0

    # Linear interpolation
    k_low = int(np.floor(freq_bin_float))
    k_high = k_low + 1
    frac = freq_bin_float - k_low

    val_low = map_data_frame[k_low]
    val_high = map_data_frame[k_high]

    # Check for NaN or Inf before interpolation
    if not (np.isfinite(val_low) and np.isfinite(val_high)):
        # Handle non-finite values (e.g., return nearest finite or 0)
        if np.isfinite(val_low):
            return val_low
        if np.isfinite(val_high):
            return val_high
        return 0.0  # Or some other default

    return val_low + frac * (val_high - val_low)


@numba.njit(cache=True, fastmath=True)
def _calculate_harmonic_metrics_numba(
    f0_hz: float,
    frame_idx: int,
    spod_map_frame: np.ndarray,
    A_lin_frame: np.ndarray,
    sr: int,
    n_fft: int,
    h_max: int,
    inharmonicity_factor: float,  # Added inharmonicity
    freq_resolution_hz: float,
    n_bins: int,
    harmonic_weight_offset: float = 0.5,
    harmonic_weight_exponent: float = 1.0,
) -> Tuple[float, float]:
    """
    Numba-optimized calculation of Harmonic Coherence (C) and Harmonic Amplitude Sum (AmpSum)
    for a single F0 and frame. Includes inharmonicity.
    """
    if f0_hz <= 0 or freq_resolution_hz <= 0:
        return 0.0, 0.0

    total_spod_weight = 0.0
    weighted_spod_sum = 0.0
    total_amp_weight = 0.0
    weighted_amp_sum = 0.0
    nyquist = sr / 2.0

    for h in range(1, h_max + 1):
        # Calculate harmonic frequency with inharmonicity
        harmonic_freq = h * f0_hz
        if inharmonicity_factor != 0.0:
            inh_term = np.sqrt(max(0.0, 1.0 + inharmonicity_factor * h**2))
            harmonic_freq *= inh_term

        if harmonic_freq <= 0 or harmonic_freq >= nyquist:
            continue  # Skip harmonics outside valid range

        # Convert harmonic frequency to fractional bin
        freq_bin_float = harmonic_freq / freq_resolution_hz

        # Interpolate SPOD and Amplitude using the linear interpolation helper
        # Note: Pass the frame data directly
        interp_spod = _interpolate_map_numba(spod_map_frame, freq_bin_float, n_bins)
        interp_amp = _interpolate_map_numba(A_lin_frame, freq_bin_float, n_bins)

        if np.isfinite(interp_spod) and np.isfinite(interp_amp):
            # 動的パラメータを使用した重み付け計算
            weight = 1.0 / (h + harmonic_weight_offset) ** harmonic_weight_exponent
            # Coherence calculation
            weighted_spod_sum += weight * interp_spod
            total_spod_weight += weight
            # Amplitude sum calculation
            weighted_amp_sum += weight * interp_amp
            total_amp_weight += weight

    # Calculate averages
    avg_coherence = (
        weighted_spod_sum / total_spod_weight if total_spod_weight > 0 else 0.0
    )
    avg_amp_sum = weighted_amp_sum  # AmpSum is typically the sum, not average, but let's keep the weighted sum logic for consistency for now. Or return weighted_amp_sum directly if preferred.

    # Ensure results are within valid ranges [0, 1] for coherence
    avg_coherence = max(0.0, min(1.0, avg_coherence))

    return avg_coherence, avg_amp_sum


# <<< START NEW FUNCTION >>>
@numba.njit(cache=True, fastmath=True)
def _calculate_avg_harmonic_spod_numba(
    f0_hz: float,
    frame_idx: int,  # Note: frame_idx is actually not needed if spod_map_frame is passed
    spod_map_frame: np.ndarray,
    sr: int,
    n_fft: int,
    h_max: int,
    inharmonicity_factor: float,
    freq_resolution_hz: float,
    n_bins: int,
    harmonic_weight_offset: float = 0.5,
    harmonic_weight_exponent: float = 1.0,
) -> float:
    """
    Numba-optimized calculation of weighted average harmonic SPOD stability
    for a single F0 and frame. Includes inharmonicity.
    """
    if f0_hz <= 0 or freq_resolution_hz <= 0:
        return 0.0

    total_weight = 0.0
    weighted_spod_sum = 0.0
    nyquist = sr / 2.0

    for h in range(1, h_max + 1):
        # Calculate harmonic frequency with inharmonicity
        harmonic_freq = h * f0_hz
        if inharmonicity_factor != 0.0:
            inh_term = np.sqrt(max(0.0, 1.0 + inharmonicity_factor * h**2))
            harmonic_freq *= inh_term

        if harmonic_freq <= 0 or harmonic_freq >= nyquist:
            continue  # Skip harmonics outside valid range

        # Convert harmonic frequency to fractional bin
        freq_bin_float = harmonic_freq / freq_resolution_hz

        # Interpolate SPOD using the linear interpolation helper
        interp_spod = _interpolate_map_numba(spod_map_frame, freq_bin_float, n_bins)

        if np.isfinite(interp_spod):
            # 動的パラメータを使用した重み付け計算 (Use same weights as HCF for consistency)
            weight = 1.0 / (h + harmonic_weight_offset) ** harmonic_weight_exponent
            weighted_spod_sum += weight * interp_spod
            total_weight += weight

    # Calculate weighted average
    avg_spod = weighted_spod_sum / total_weight if total_weight > 0 else 0.0

    # Ensure result is within valid range [0, 1]
    avg_spod = max(0.0, min(1.0, avg_spod))

    return avg_spod


# <<< END NEW FUNCTION >>>


# Numba function to compute maps over all frames and f0 candidates
@numba.njit(cache=True, parallel=True)  # Enable parallelism
def _calculate_hcf_maps_numba(
    spod_map: np.ndarray,
    A_lin: np.ndarray,
    F_cand_hz: np.ndarray,  # Array of F0 candidates in Hz
    sr: int,
    n_fft: int,
    hop_length: int,
    h_max: int,
    inharmonicity_factor: float,
    freq_resolution_hz: float,
    tau_sec: float,
    flux_weight_c: float,
    flux_weight_a: float,
    harmonic_weight_offset: float = 0.5,
    harmonic_weight_exponent: float = 1.0,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:  # <<< 戻り値にAvgHarmonicSPOD_map追加
    """
    Numba-optimized calculation of all HCF-related maps, including AvgHarmonicSPOD.
    Returns: C_map, AmpSum_map, Phi_C_map, Phi_A_map, Phi_comb_map, O_vec, AvgHarmonicSPOD_map
    """
    n_bins, n_frames = A_lin.shape
    n_f0_cands = len(F_cand_hz)

    # Output maps initialization
    C_map = np.zeros((n_f0_cands, n_frames), dtype=np.float64)
    AmpSum_map = np.zeros((n_f0_cands, n_frames), dtype=np.float64)
    AvgHarmonicSPOD_map = np.zeros(
        (n_f0_cands, n_frames), dtype=np.float64
    )  # <<< AvgHarmonicSPOD_map初期化
    Phi_C_map = np.zeros((n_f0_cands, n_frames), dtype=np.float64)
    Phi_A_map = np.zeros((n_f0_cands, n_frames), dtype=np.float64)
    Phi_comb_map = np.zeros((n_f0_cands, n_frames), dtype=np.float64)
    O_vec = np.zeros(n_frames, dtype=np.float64)

    # Calculate C, AmpSum, and AvgHarmonicSPOD maps first
    # Use prange for parallel loop over F0 candidates
    for i in numba.prange(n_f0_cands):
        f0 = F_cand_hz[i]
        for n in range(n_frames):
            spod_frame = spod_map[:, n]
            A_frame = A_lin[:, n]
            # Calculate C and AmpSum
            C_map[i, n], AmpSum_map[i, n] = _calculate_harmonic_metrics_numba(
                f0,
                n,
                spod_frame,
                A_frame,
                sr,
                n_fft,
                h_max,
                inharmonicity_factor,
                freq_resolution_hz,
                n_bins,
                harmonic_weight_offset,
                harmonic_weight_exponent,
            )
            # Calculate AvgHarmonicSPOD
            AvgHarmonicSPOD_map[i, n] = (
                _calculate_avg_harmonic_spod_numba(  # <<< 計算呼び出し追加
                    f0,
                    n,
                    spod_frame,
                    sr,
                    n_fft,
                    h_max,  # Use same h_max as HCF for now
                    inharmonicity_factor,
                    freq_resolution_hz,
                    n_bins,
                    harmonic_weight_offset,
                    harmonic_weight_exponent,
                )
            )

    # Calculate Flux maps
    delta_n = max(1, int(round(tau_sec * sr / hop_length)))
    if n_frames > delta_n:
        # Use prange for parallel loop over F0 candidates
        for i in numba.prange(n_f0_cands):
            # Calculate difference (Flux)
            # Note: Flux is typically defined as current - past
            Phi_C_map[i, delta_n:] = C_map[i, delta_n:] - C_map[i, :-delta_n]
            Phi_A_map[i, delta_n:] = AmpSum_map[i, delta_n:] - AmpSum_map[i, :-delta_n]

            # Calculate Combined Flux (only positive flux contributes)
            Phi_comb_map[i, :] = flux_weight_c * np.maximum(
                0.0, Phi_C_map[i, :]
            ) + flux_weight_a * np.maximum(0.0, Phi_A_map[i, :])

    # Calculate Overall Onset Function O(n) by summing Phi_comb over F0 candidates
    if n_f0_cands > 0:
        for n in range(n_frames):
            O_vec[n] = np.sum(Phi_comb_map[:, n])  # Sum over candidates for each frame

    return (
        C_map,
        AmpSum_map,
        Phi_C_map,
        Phi_A_map,
        Phi_comb_map,
        O_vec,
        AvgHarmonicSPOD_map,
    )  # <<< 戻り値に追加


# --- PZSTDDetector Class ---


@register_detector(
    name="pzstd_detector",
    description="SPOD-based multi-pitch detector with HCF enhancements and Numba optimization.",
    version="1.0.0",
)
# --- PZSTDDetector Class ---


class PZSTDDetector(BaseDetector):
    """
    Refined PZSTD-based multi-pitch detector using Numba optimization.
    Allows grid search over onset detection parameters.
    HCF Enhanced: Integrates Harmonic Coherence Flux for improved onset/offset detection.
    """

    # --- Default Parameters (Some fixed, some tunable) ---
    # Fixed (PZSTD Base) - >> Moved to __init__ defaults
    # N_FFT = 2048
    # HOP_LENGTH = 1024
    # WINDOW = 'hann'
    # ENERGY_DB_THRESHOLD = -50.0
    # MIN_FREQ_HZ = 30.0
    # MAX_FREQ_HZ = 5000.0
    # PEAK_DISTANCE_BIN = 4
    # MAX_PEAKS_PER_FRAME = 35
    # PEAK_PROMINENCE_REL = 0.03
    # SPOD_WINDOW_SEC = 0.07
    # SPOD_ALPHA = 1.1
    # MIN_PEAK_SPOD_STABILITY = 0.15
    # MAX_HARMONICS = 12 # Note: Used by F0 scoring, separate from HCF_MAX_HARMONICS
    # INHARMONICITY_FACTOR = 0.0001
    # F0_CANDIDATE_RESOLUTION_CENTS = 15.0
    # MAX_F0_ITERATIONS = 12
    # CONTINUITY_BONUS = 0.6
    # STABILITY_BONUS_THRESHOLD_CENT_DEV = 25.0
    # STABILITY_BONUS_VALUE = 0.4
    # MAX_PITCH_GAP_SEC = 0.05
    # MIN_SILENCE_DURATION_SEC = 0.04 # <<< Still configurable via kwargs
    # NOTE_MERGE_TOLERANCE_CENTS = 50.0 # <<< Still configurable via kwargs
    # MAX_PITCH_STD_DEV_CENTS = 40.0 # <<< Still configurable via kwargs

    # Defaults for Tunable Parameters (PZSTD Base & HCF)
    DEFAULT_F0_SCORE_THRESHOLD = 0.20
    DEFAULT_HARMONIC_MATCH_TOLERANCE_CENTS = 30.0
    DEFAULT_PITCH_CONTINUITY_TOLERANCE_CENTS = 50.0
    DEFAULT_MIN_NOTE_DURATION_SEC = 0.05
    DEFAULT_HCF_TAU_SEC = 0.01
    DEFAULT_HCF_MAX_HARMONICS = 10
    DEFAULT_HCF_INHARMONICITY = 0.0001
    DEFAULT_HCF_FLUX_WEIGHT_C = 0.5
    DEFAULT_HCF_FLUX_WEIGHT_A = 0.2
    DEFAULT_HCF_ONSET_SMOOTHING_SEC = 0.01
    DEFAULT_HCF_ONSET_PEAK_THRESH = 0.12
    DEFAULT_HCF_ONSET_PEAK_PROMINENCE = 0.05
    DEFAULT_HCF_COHERENCE_OFFSET_THRESH = 0.07
    DEFAULT_HCF_NOTE_START_FLUX_THRESH = (
        0.1  # グリッドサーチで最適化された値（0.06→0.1）
    )
    DEFAULT_HCF_ATTACK_TIME_SEC = 0.1
    # 倍音重み付けのデフォルトパラメータを最適値に更新
    DEFAULT_HARMONIC_WEIGHT_OFFSET = 0.0  # グリッドサーチで最適化された値（0.5→0.0）
    DEFAULT_HARMONIC_WEIGHT_EXPONENT = 1.0

    def __init__(
        self,
        # --- PZSTD Base Fixed Params (Now configurable) ---
        n_fft: int = 2048,
        hop_length: int = 1024,
        window: str = "hann",
        energy_db_threshold: float = -50.0,
        min_freq_hz: float = 30.0,
        max_freq_hz: float = 5000.0,
        peak_distance_bin: int = 4,
        max_peaks_per_frame: int = 35,
        peak_prominence_rel: float = 0.03,
        spod_window_sec: float = 0.07,
        spod_alpha: float = 1.1,
        min_peak_spod_stability: float = 0.15,
        max_harmonics: int = 12,
        inharmonicity_factor: float = 0.0001,
        f0_candidate_resolution_cents: float = 15.0,
        max_f0_iterations: int = 12,
        continuity_bonus: float = 0.6,
        stability_bonus_threshold_cent_dev: float = 25.0,
        stability_bonus_value: float = 0.4,
        max_pitch_gap_sec: float = 0.05,
        # --- PZSTD Base Tunable Params ---
        f0_score_threshold: float = DEFAULT_F0_SCORE_THRESHOLD,
        harmonic_match_tolerance_cents: float = DEFAULT_HARMONIC_MATCH_TOLERANCE_CENTS,
        pitch_continuity_tolerance_cents: float = DEFAULT_PITCH_CONTINUITY_TOLERANCE_CENTS,
        min_note_duration_sec: float = DEFAULT_MIN_NOTE_DURATION_SEC,
        # --- HCF Tunable Params ---
        hcf_max_harmonics: int = DEFAULT_HCF_MAX_HARMONICS,
        hcf_inharmonicity: float = DEFAULT_HCF_INHARMONICITY,
        hcf_tau_sec: float = DEFAULT_HCF_TAU_SEC,
        hcf_flux_weight_c: float = DEFAULT_HCF_FLUX_WEIGHT_C,
        hcf_flux_weight_a: float = DEFAULT_HCF_FLUX_WEIGHT_A,
        hcf_onset_smoothing_sec: float = DEFAULT_HCF_ONSET_SMOOTHING_SEC,
        hcf_onset_peak_thresh: float = DEFAULT_HCF_ONSET_PEAK_THRESH,
        hcf_onset_peak_prominence: float = DEFAULT_HCF_ONSET_PEAK_PROMINENCE,
        hcf_coherence_offset_thresh: float = DEFAULT_HCF_COHERENCE_OFFSET_THRESH,
        hcf_note_start_flux_thresh: float = DEFAULT_HCF_NOTE_START_FLUX_THRESH,
        # --- 倍音重み付けのパラメータを追加 ---
        harmonic_weight_offset: float = DEFAULT_HARMONIC_WEIGHT_OFFSET,
        harmonic_weight_exponent: float = DEFAULT_HARMONIC_WEIGHT_EXPONENT,
        # --- Other Params (From kwargs in prev version) ---
        min_silence_duration_sec: float = 0.04,
        note_merge_tolerance_cents: float = 50.0,
        max_pitch_std_dev_cents: float = 40.0,
        **kwargs,
    ):  # Keep kwargs for BaseDetector compatibility
        """
        Initializes the detector, including HCF parameters.
        Moved fixed PZSTD parameters to __init__ arguments for potential configuration.
        """
        super().__init__(**kwargs)

        # --- Assign Fixed PZSTD Params as Instance Variables ---
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.energy_db_threshold = energy_db_threshold
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.peak_distance_bin = peak_distance_bin
        self.max_peaks_per_frame = max_peaks_per_frame
        self.peak_prominence_rel = peak_prominence_rel
        self.spod_window_sec = spod_window_sec
        self.spod_alpha = spod_alpha
        self.min_peak_spod_stability = min_peak_spod_stability
        self.max_harmonics = max_harmonics  # F0 scoring harmonics
        self.inharmonicity_factor = inharmonicity_factor
        self.f0_candidate_resolution_cents = f0_candidate_resolution_cents
        self.max_f0_iterations = max_f0_iterations
        self.continuity_bonus = continuity_bonus
        self.stability_bonus_threshold_cent_dev = stability_bonus_threshold_cent_dev
        self.stability_bonus_value = stability_bonus_value
        self.max_pitch_gap_sec = max_pitch_gap_sec
        # ------------------------------------------------------

        # Assign PZSTD tunable parameters (validation/clipping)
        self.f0_score_threshold = max(0.0, f0_score_threshold)
        self.harmonic_match_tolerance_cents = max(
            1.0, harmonic_match_tolerance_cents
        )  # Ensure positive
        self.pitch_continuity_tolerance_cents = max(
            1.0, pitch_continuity_tolerance_cents
        )  # Ensure positive
        self.min_note_duration_sec = max(0.0, min_note_duration_sec)

        # Assign HCF tunable parameters (add validation/clipping as needed)
        self.hcf_tau_sec = max(0.001, hcf_tau_sec)
        self.hcf_max_harmonics = max(1, hcf_max_harmonics)
        self.hcf_inharmonicity = max(0.0, hcf_inharmonicity)
        self.hcf_flux_weight_c = max(0.0, hcf_flux_weight_c)
        self.hcf_flux_weight_a = max(0.0, hcf_flux_weight_a)
        self.hcf_onset_smoothing_sec = max(0.0, hcf_onset_smoothing_sec)
        self.hcf_onset_peak_thresh = max(0.0, min(1.0, hcf_onset_peak_thresh))
        self.hcf_onset_peak_prominence = max(0.0, hcf_onset_peak_prominence)
        self.hcf_coherence_offset_thresh = max(
            0.0, min(1.0, hcf_coherence_offset_thresh)
        )
        self.hcf_note_start_flux_thresh = max(0.0, hcf_note_start_flux_thresh)

        # Assign Other Params
        self.min_silence_duration_sec = max(0.0, min_silence_duration_sec)
        self.note_merge_tolerance_cents = max(0.0, note_merge_tolerance_cents)
        self.max_pitch_std_dev_cents = max(0.0, max_pitch_std_dev_cents)

        # Internal state variables (remain the same)
        self._sr: Optional[int] = None
        self._freqs: Optional[np.ndarray] = None
        self._times: Optional[np.ndarray] = None
        self._freq_resolution_hz: Optional[float] = None
        self._f0_candidates_hz: Optional[np.ndarray] = None
        self._max_f_analysis: float = 0.0  # Will be set in _initialize_run
        self._A_lin: Optional[np.ndarray] = None
        self._allowed_onset_frames: Optional[set] = None

        # HCFパラメータの設定と同様に、倍音重み付けのパラメータを設定
        self.harmonic_weight_offset = max(0.0, harmonic_weight_offset)
        self.harmonic_weight_exponent = max(0.1, harmonic_weight_exponent)

        # logger.info(f"PZSTDDetector (with HCF modifications) initialized.") # Moved to detect start

    def get_params(self) -> Dict[str, Any]:
        """Returns a dictionary of parameters including HCF ones."""
        # Start with PZSTD base parameters
        params = {
            # Fixed parameters (now instance variables)
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "window": self.window,
            "energy_db_threshold": self.energy_db_threshold,
            "min_freq_hz": self.min_freq_hz,
            "max_freq_hz": self.max_freq_hz,
            "peak_distance_bin": self.peak_distance_bin,
            "max_peaks_per_frame": self.max_peaks_per_frame,
            "peak_prominence_rel": self.peak_prominence_rel,
            "spod_window_sec": self.spod_window_sec,
            "spod_alpha": self.spod_alpha,
            "min_peak_spod_stability": self.min_peak_spod_stability,
            "max_harmonics": self.max_harmonics,
            "inharmonicity_factor": self.inharmonicity_factor,
            "f0_candidate_resolution_cents": self.f0_candidate_resolution_cents,
            "max_f0_iterations": self.max_f0_iterations,
            "continuity_bonus": self.continuity_bonus,
            "stability_bonus_threshold_cent_dev": self.stability_bonus_threshold_cent_dev,
            "stability_bonus_value": self.stability_bonus_value,
            "max_pitch_gap_sec": self.max_pitch_gap_sec,
            "min_silence_duration_sec": self.min_silence_duration_sec,
            "note_merge_tolerance_cents": self.note_merge_tolerance_cents,
            "max_pitch_std_dev_cents": self.max_pitch_std_dev_cents,
            "overlap_resolution_strategy": "merge_similar",  # Fixed
            # Tunable parameters (PZSTD Base)
            "f0_score_threshold": self.f0_score_threshold,
            "harmonic_match_tolerance_cents": self.harmonic_match_tolerance_cents,
            "pitch_continuity_tolerance_cents": self.pitch_continuity_tolerance_cents,
            "min_note_duration_sec": self.min_note_duration_sec,
            # HCF Parameters
            "hcf_tau_sec": self.hcf_tau_sec,
            "hcf_max_harmonics": self.hcf_max_harmonics,
            "hcf_inharmonicity": self.hcf_inharmonicity,
            "hcf_flux_weight_c": self.hcf_flux_weight_c,
            "hcf_flux_weight_a": self.hcf_flux_weight_a,
            "hcf_onset_smoothing_sec": self.hcf_onset_smoothing_sec,
            "hcf_onset_peak_thresh": self.hcf_onset_peak_thresh,
            "hcf_onset_peak_prominence": self.hcf_onset_peak_prominence,
            "hcf_coherence_offset_thresh": self.hcf_coherence_offset_thresh,
            "hcf_note_start_flux_thresh": self.hcf_note_start_flux_thresh,
            # 倍音重み付けのパラメータを追加
            "harmonic_weight_offset": self.harmonic_weight_offset,
            "harmonic_weight_exponent": self.harmonic_weight_exponent,
            # Onset method info (削除)
            # 'onset_method': 'hcf', # Always HCF now
        }
        return params

    def _initialize_run(self, sr: int):
        """Initializes internal state based on sample rate."""
        self._sr = sr
        self._freqs = librosa.fft_frequencies(
            sr=self._sr, n_fft=self.n_fft
        )  # Use instance var
        self._freq_resolution_hz = (
            self._freqs[1] - self._freqs[0] if len(self._freqs) > 1 else sr / self.n_fft
        )  # Use instance var
        self._max_f_analysis = self.max_freq_hz  # Use instance var
        self._A_lin = None
        self._allowed_onset_frames = None
        self._f0_candidates_hz = self._generate_f0_candidates()

    def _calculate_stft(
        self, audio_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates STFT using fixed parameters."""
        if self._sr is None:
            raise RuntimeError("Detector not initialized. Call _initialize_run first.")
        try:
            S_complex = librosa.stft(
                audio_data,
                n_fft=self.n_fft,  # Use instance var
                hop_length=self.hop_length,  # Use instance var
                window=self.window,  # Use instance var
                center=True,
            )
            S_amp = np.abs(S_complex)
            S_db = librosa.amplitude_to_db(
                S_amp, ref=np.max(S_amp) if S_amp.size > 0 else 1.0, top_db=80.0
            )
            phases = np.angle(S_complex)
            self._times = librosa.frames_to_time(
                np.arange(S_complex.shape[1]), sr=self._sr, hop_length=self.hop_length
            )  # Use instance var
            self._A_lin = S_amp
            return S_amp, S_db, phases
        except Exception as e:
            logger.error(f"Error calculating STFT: {e}", exc_info=True)
            self._times = np.array([])
            return np.array([[]]), np.array([[]]), np.array([[]])

    def _calculate_spod(self, phases: np.ndarray) -> np.ndarray:
        """Calculates the SPOD map using Gaussian filter."""
        n_bins, n_frames = phases.shape
        if n_frames <= 1 or self._sr is None:
            return np.zeros_like(phases)
        spod_window_frames = max(
            1, int(self.spod_window_sec * self._sr / self.hop_length)
        )  # Use instance vars
        phi_unwrapped = np.unwrap(phases, axis=1)
        inst_freq = np.diff(phi_unwrapped, n=1, axis=1, append=phi_unwrapped[:, -1:])
        try:
            sigma = spod_window_frames / 4.0
            kwargs = {"sigma": sigma, "axis": 1, "mode": "nearest"}
            inst_freq_sq_mean = gaussian_filter1d(inst_freq**2, **kwargs)
            inst_freq_mean = gaussian_filter1d(inst_freq, **kwargs)
            inst_freq_var = np.maximum(0, inst_freq_sq_mean - inst_freq_mean**2)
        except Exception as e:
            logger.error(
                f"Error calculating moving variance for SPOD: {e}", exc_info=True
            )
            return np.zeros_like(phases)
        spod_map = np.exp(-self.spod_alpha * inst_freq_var)  # Use instance var
        return np.clip(spod_map, 0.0, 1.0)

    def _freq_to_bin_approx(self, freq_hz: float) -> Optional[float]:
        """Finds the approximate (fractional) bin index (same as original)."""
        if (
            self._freqs is None
            or self._freq_resolution_hz is None
            or self._freq_resolution_hz <= 0
        ):
            return None
        if freq_hz < self._freqs[0] or freq_hz > self._freqs[-1]:
            return None
        return freq_hz / self._freq_resolution_hz

    def _bin_to_freq(self, bin_index: float) -> Optional[float]:
        """Converts a (potentially fractional) bin index to frequency (Hz) (same as original)."""
        if self._freqs is None or self._freq_resolution_hz is None:
            return None
        return np.clip(
            bin_index * self._freq_resolution_hz, self._freqs[0], self._freqs[-1]
        )

    def _detect_peaks_per_frame(
        self, A_lin: np.ndarray, S_db: np.ndarray, spod_map: np.ndarray
    ) -> List[List[Tuple[float, float, float]]]:
        """
        Detects spectral peaks using fixed parameters and Numba-optimized interpolation.
        Returns List[List[Tuple[interpolated_freq_hz, amplitude_at_peak, spod_at_peak_bin]]]
        """
        if self._freqs is None:
            return [[] for _ in range(A_lin.shape[1])]
        n_bins, n_frames = A_lin.shape
        peaks_per_frame: List[List[Tuple[float, float, float]]] = [
            [] for _ in range(n_frames)
        ]
        min_bin_idx = np.searchsorted(
            self._freqs, self.min_freq_hz, side="left"
        )  # Use instance var
        max_f = self.max_freq_hz  # Use instance var
        max_bin_idx = np.searchsorted(self._freqs, max_f, side="right") - 1
        if min_bin_idx > max_bin_idx:
            logger.warning("Invalid frequency range for peak detection.")
            return peaks_per_frame
        min_height_amp = librosa.db_to_amplitude(
            self.energy_db_threshold, ref=1.0
        )  # Use instance var
        for t in range(n_frames):
            amp_t_full = A_lin[:, t]
            amp_t_slice = A_lin[min_bin_idx : max_bin_idx + 1, t]
            db_t_slice = S_db[min_bin_idx : max_bin_idx + 1, t]
            spod_t_slice = spod_map[min_bin_idx : max_bin_idx + 1, t]
            if amp_t_slice.size == 0:
                continue
            energy_mask = db_t_slice > self.energy_db_threshold  # Use instance var
            spod_mask = spod_t_slice > self.min_peak_spod_stability  # Use instance var
            valid_mask = energy_mask & spod_mask
            candidate_amp = np.where(valid_mask, amp_t_slice, 0)
            if np.any(valid_mask):
                valid_amps = amp_t_slice[valid_mask]
                amp_range = (
                    np.max(valid_amps) - np.min(valid_amps)
                    if len(valid_amps) > 1
                    else np.max(valid_amps)
                )
                prominence_threshold = max(
                    1e-9, amp_range * self.peak_prominence_rel
                )  # Use instance var
            else:
                continue
            try:
                peak_indices_rel, properties = find_peaks(
                    candidate_amp,
                    distance=self.peak_distance_bin,  # Use instance var
                    height=min_height_amp,
                    prominence=prominence_threshold,
                )
                peak_indices_abs = min_bin_idx + peak_indices_rel
                if len(peak_indices_abs) > self.max_peaks_per_frame:  # Use instance var
                    prominences = properties["prominences"]
                    if len(prominences) >= self.max_peaks_per_frame:  # Use instance var
                        partition_idx = np.argpartition(
                            prominences, -self.max_peaks_per_frame
                        )[
                            -self.max_peaks_per_frame :
                        ]  # Use instance var
                        top_k_indices = partition_idx[
                            np.argsort(prominences[partition_idx])[::-1]
                        ]
                        peak_indices_abs = peak_indices_abs[top_k_indices]
                    else:
                        sorted_idx = np.argsort(prominences)[::-1]
                        peak_indices_abs = peak_indices_abs[sorted_idx]
                frame_peaks_data: List[Tuple[float, float, float]] = []
                for p_idx_abs in peak_indices_abs:
                    delta_bin, _ = _parabolic_interpolation_numba(amp_t_full, p_idx_abs)
                    interp_bin = p_idx_abs + delta_bin
                    interp_freq = self._bin_to_freq(interp_bin)
                    amp_at_peak = A_lin[p_idx_abs, t]
                    spod_at_peak = spod_map[p_idx_abs, t]
                    if (
                        interp_freq is not None
                        and self.min_freq_hz <= interp_freq <= self.max_freq_hz
                    ):  # Use instance vars
                        frame_peaks_data.append(
                            (interp_freq, amp_at_peak, spod_at_peak)
                        )
                frame_peaks_data.sort(key=lambda x: x[0])
                peaks_per_frame[t] = frame_peaks_data
            except Exception as e:
                logger.error(
                    f"Frame {t}: Error during peak detection/interpolation: {e}",
                    exc_info=False,
                )
        return peaks_per_frame

    def _generate_f0_candidates(self) -> np.ndarray:
        """Generates F0 candidates using fixed parameters."""
        if self._freqs is None:
            return np.array([])
        min_f = self.min_freq_hz  # Use instance var
        max_f = self._max_f_analysis
        if min_f >= max_f:
            logger.warning(
                f"Min F0 freq ({min_f} Hz) >= Max F0 freq ({max_f} Hz). No candidates generated."
            )
            return np.array([])
        min_cents_arr = hz_to_cents_numba(np.array([min_f]))
        max_cents_arr = hz_to_cents_numba(np.array([max_f]))
        min_cents = min_cents_arr[0]
        max_cents = max_cents_arr[0]
        if not np.isfinite(min_cents) or not np.isfinite(max_cents):
            logger.warning("Could not convert frequency bounds to cents.")
            return np.array([])
        num_steps = (
            int(np.ceil((max_cents - min_cents) / self.f0_candidate_resolution_cents))
            + 1
        )  # Use instance var
        if num_steps <= 1:
            f0_candidates_hz = (
                np.array([min_f, max_f]) if min_f != max_f else np.array([min_f])
            )
        else:
            f0_candidates_cents = np.linspace(min_cents, max_cents, num_steps)
            f0_candidates_hz = cents_to_hz_numba(f0_candidates_cents)
        logger.debug(
            f"Generated {len(f0_candidates_hz)} F0 candidates ({min_f:.1f}Hz-{max_f:.1f}Hz, {self.f0_candidate_resolution_cents} cents res)."
        )  # Use instance var
        f0_candidates_hz = f0_candidates_hz[
            (f0_candidates_hz >= min_f) & (f0_candidates_hz <= max_f)
        ]
        return f0_candidates_hz

    def _estimate_f0s_from_peaks(
        self, peaks_per_frame: List[List[Tuple[float, float, float]]]
    ) -> List[List[float]]:
        """Estimates multiple F0s per frame using tunable parameters and Numba scoring."""
        n_frames = len(peaks_per_frame)
        multi_f0_per_frame: List[List[float]] = [[] for _ in range(n_frames)]
        if self._f0_candidates_hz is None or len(self._f0_candidates_hz) == 0:
            logger.warning("No F0 candidates available for estimation.")
            return multi_f0_per_frame

        # Convert previous F0s to cents array for Numba function
        previous_f0s_cents = np.array([], dtype=np.float64)
        f0_candidates = self._f0_candidates_hz  # Use cached candidates
        max_f_analysis = self._max_f_analysis

        for t in range(n_frames):
            frame_peaks_t = peaks_per_frame[t]
            num_peaks_t = len(frame_peaks_t)
            if num_peaks_t == 0:
                previous_f0s_cents = np.array([], dtype=np.float64)
                continue

            # --- Prepare data for Numba scorer ---
            peak_data_t = np.array(frame_peaks_t, dtype=np.float64)
            peak_freqs_hz_t = peak_data_t[:, 0]
            peak_amps_t = peak_data_t[:, 1]
            peak_spods_t = peak_data_t[:, 2]
            peak_freqs_cents_t = hz_to_cents_numba(peak_freqs_hz_t)

            available_peak_indices = np.arange(num_peaks_t)
            detected_f0s_t: List[float] = []
            iteration_count = 0

            while (
                len(available_peak_indices) > 0
                and iteration_count < self.max_f0_iterations
            ):  # Use instance var
                iteration_count += 1
                num_available = len(available_peak_indices)
                if num_available == 0:
                    break

                current_peak_freqs_cents = peak_freqs_cents_t[available_peak_indices]
                current_peak_amps = peak_amps_t[available_peak_indices]
                current_peak_spods = peak_spods_t[available_peak_indices]

                scores = np.zeros(len(f0_candidates), dtype=np.float64)
                matched_masks = np.zeros(
                    (len(f0_candidates), num_available), dtype=np.bool_
                )

                for i, f0_cand in enumerate(f0_candidates):
                    # Call Numba scoring function with tunable parameters
                    score_i, matched_mask_i = _score_f0_candidate_numba(
                        f0_cand,
                        current_peak_freqs_cents,
                        current_peak_amps,
                        current_peak_spods,
                        previous_f0s_cents,
                        self.max_harmonics,  # Use instance var
                        self.inharmonicity_factor,  # Use instance var
                        max_f_analysis,
                        self.harmonic_match_tolerance_cents,
                        self.pitch_continuity_tolerance_cents,
                        self.continuity_bonus,  # Use instance var
                        harmonic_weight_offset=self.harmonic_weight_offset,
                        harmonic_weight_exponent=self.harmonic_weight_exponent,
                    )
                    scores[i] = score_i
                    matched_masks[i, :] = matched_mask_i

                best_candidate_idx = np.argmax(scores)
                best_score = scores[best_candidate_idx]

                if best_score >= self.f0_score_threshold:
                    best_f0_hz = f0_candidates[best_candidate_idx]
                    best_matched_mask = matched_masks[best_candidate_idx, :]
                    explained_relative_indices = np.where(best_matched_mask)[0]

                    if best_f0_hz > 0 and len(explained_relative_indices) > 0:
                        detected_f0s_t.append(best_f0_hz)
                        available_peak_indices = np.delete(
                            available_peak_indices, explained_relative_indices
                        )
                    else:
                        break
                else:
                    break

            multi_f0_per_frame[t] = sorted(detected_f0s_t)
            if detected_f0s_t:
                previous_f0s_cents = hz_to_cents_numba(
                    np.array(detected_f0s_t, dtype=np.float64)
                )
            else:
                previous_f0s_cents = np.array([], dtype=np.float64)

        logger.debug(
            f"F0 estimation found F0s in {sum(1 for f0s in multi_f0_per_frame if f0s)} / {n_frames} frames."
        )
        return multi_f0_per_frame

    @staticmethod
    @numba.njit(cache=True, fastmath=True)
    def _get_avg_harmonic_spod(
        f0_hz: float,
        frame_idx: int,
        spod_map: np.ndarray,
        sr: int,
        n_fft: int,  # Pass N_FFT as argument
        h_check: int = 3,  # Default h_check is 3, not parameterized currently
        harmonic_weight_offset: float = 0.5,
        harmonic_weight_exponent: float = 1.0,
    ) -> float:
        """
        Calculate the *weighted* average SPOD value for the fundamental frequency (f0)
        and its first few harmonics at a specific frame, weighting by 1/h.
        Uses Numba for optimization.
        """
        if f0_hz <= 0 or frame_idx < 0 or frame_idx >= spod_map.shape[1]:
            return 0.0

        num_freq_bins = spod_map.shape[0]
        total_weight = 0.0
        weighted_spod_sum = 0.0

        for h in range(1, h_check + 1):  # Check fundamental (h=1) and harmonics
            harmonic_freq = f0_hz * h
            # Convert frequency to bin index
            freq_bin_float = harmonic_freq * n_fft / sr  # Use argument n_fft
            freq_bin_idx = int(round(freq_bin_float))

            if 0 <= freq_bin_idx < num_freq_bins:
                spod_val = spod_map[freq_bin_idx, frame_idx]
                if np.isfinite(spod_val):  # Ensure SPOD value is valid
                    # 動的パラメータを使用した重み付け計算
                    weight = (
                        1.0 / (h + harmonic_weight_offset) ** harmonic_weight_exponent
                    )
                    weighted_spod_sum += spod_val * weight
                    total_weight += weight
            # else: # Harmonic is outside the frequency range of the SPOD map
            # logger.debug(f"Harmonic {h} ({harmonic_freq:.1f}Hz, bin {freq_bin_idx}) out of range for F0={f0_hz:.1f}Hz at frame {frame_idx}") # Debug log if needed

        # Calculate weighted average
        if total_weight > 0:
            avg_spod = weighted_spod_sum / total_weight
            return avg_spod
        else:
            # logger.warning(f"No valid SPOD values found for harmonics of F0={f0_hz:.1f}Hz at frame {frame_idx}. Returning 0.0") # Debug log if needed
            return 0.0

    def _frames_to_notes_multipitch(
        self,
        multi_f0_per_frame: List[List[float]],
        # spod_map: np.ndarray, # Removed as SPOD offset is disabled
        allowed_onset_frames: Optional[set],
        C_map: Optional[np.ndarray],
        Phi_comb_map: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Converts frame-wise F0s into notes using optimal assignment,
        HCF-based onset check, and HCF/SPOD-based offset checks.
        """
        # --- 初期化、状態チェック --- (略)
        # ...
        if (
            self._times is None
            or self._sr is None
            or self._freqs is None
            or self._freq_resolution_hz is None
        ):
            logger.error("Detector state not fully initialized for note tracking.")
            return []
        # Verify HCF maps are available if needed
        hcf_available = (
            C_map is not None
            and Phi_comb_map is not None
            and self._f0_candidates_hz is not None
        )
        if not hcf_available:
            logger.warning(
                "HCF maps not available, HCF-based onset/offset checks will be skipped."
            )
        # Ensure allowed_onset_frames is a set, even if None was passed
        if allowed_onset_frames is None:
            allowed_onset_frames = set()

        # --- Verification Counters --- (SPOD関連削除)
        onset_suppressed_count = 0
        offset_advanced_count = 0
        offset_advanced_coherence_low_count = 0

        # --- F0候補マッピング (既存のまま) ---
        # ...
        f0_cand_hz_list = (
            self._f0_candidates_hz if self._f0_candidates_hz is not None else []
        )
        f0_idx_map = {f0: i for i, f0 in enumerate(f0_cand_hz_list)}
        n_f0_cands = len(f0_cand_hz_list)

        frame_times = self._times
        num_frames = len(multi_f0_per_frame)
        if len(frame_times) != num_frames:
            logger.warning("Mismatch between frame times and F0 results length.")
            num_frames = min(len(frame_times), len(multi_f0_per_frame))
            if num_frames == 0:
                return []
            frame_times = frame_times[:num_frames]  # Use instance var hop_length
            multi_f0_per_frame = multi_f0_per_frame[:num_frames]

        notes_list: List[Dict[str, Any]] = []
        active_notes: Dict[int, Dict[str, Any]] = {}
        track_id_counter = 0

        hop_time = self.hop_length / self._sr  # Use instance vars
        if hop_time <= 0:
            hop_time = 1024 / 44100  # Safe fallback
        max_pitch_gap_frames = (
            max(1, int(np.ceil(self.max_pitch_gap_sec / hop_time)))
            if hop_time > 0
            else 2
        )  # Use instance var
        COST_INF = 1e6

        for frame_idx in range(num_frames):
            current_time = frame_times[frame_idx]
            # --- F0リスト準備、コスト行列計算 (既存のまま) ---
            current_f0s_hz = sorted([f for f in multi_f0_per_frame[frame_idx] if f > 0])
            num_current_f0s = len(current_f0s_hz)
            current_f0s_cents_np = hz_to_cents_numba(
                np.array(current_f0s_hz, dtype=np.float64)
            )

            active_track_ids = list(active_notes.keys())
            num_active_tracks = len(active_track_ids)

            # --- コスト行列計算 (_calculate_cost_matrix_numba) (既存のまま) ---
            # ... (Uses self.pitch_continuity_tolerance_cents, self.stability_bonus_threshold_cent_dev, self.stability_bonus_value)
            active_track_last_pitches_cents = np.zeros(
                num_active_tracks, dtype=np.float64
            )
            active_track_stabilities_cents = np.zeros(
                num_active_tracks, dtype=np.float64
            )
            if num_active_tracks > 0:
                for r, track_id in enumerate(active_track_ids):
                    note_info = active_notes[track_id]
                    pitch_history_cents = np.array(
                        note_info["pitches_cents"], dtype=np.float64
                    )
                    if len(pitch_history_cents) > 0:
                        active_track_last_pitches_cents[r] = pitch_history_cents[-1]
                        history_len = min(
                            len(pitch_history_cents), 5
                        )  # Look at last 5 frames
                        active_track_stabilities_cents[r] = (
                            np.std(pitch_history_cents[-history_len:])
                            if history_len >= 2
                            else 0.0
                        )
                    else:  # Handle empty history case
                        active_track_last_pitches_cents[r] = (
                            -np.inf
                        )  # Or some invalid value
                        active_track_stabilities_cents[r] = 0.0

            if num_active_tracks > 0 and num_current_f0s > 0:
                cost_matrix = _calculate_cost_matrix_numba(
                    active_track_last_pitches_cents,
                    active_track_stabilities_cents,
                    current_f0s_cents_np,
                    self.pitch_continuity_tolerance_cents,
                    self.stability_bonus_threshold_cent_dev,  # Use instance var
                    self.stability_bonus_value,  # Use instance var
                    COST_INF,
                )
            else:
                cost_matrix = np.full(
                    (num_active_tracks, num_current_f0s), COST_INF, dtype=np.float64
                )

            # --- 割り当て問題解決 (Hungarian) (既存のまま) ---
            # ...
            matched_track_indices = set()
            matched_f0_indices = set()
            if (
                num_active_tracks > 0
                and num_current_f0s > 0
                and np.any(cost_matrix < COST_INF)
            ):
                try:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    valid_assignment_mask = cost_matrix[row_ind, col_ind] < COST_INF
                    row_ind = row_ind[valid_assignment_mask]
                    col_ind = col_ind[valid_assignment_mask]
                except ValueError:
                    row_ind, col_ind = np.array([]), np.array([])

                # --- Update Matched Tracks --- (SPOD履歴更新削除)
                for r, c in zip(row_ind.astype(int), col_ind.astype(int)):
                    track_id = active_track_ids[r]
                    f0_hz = current_f0s_hz[c]
                    f0_cents = current_f0s_cents_np[c]

                    current_coherence = 0.0
                    closest_f0_cand_idx = -1
                    if hcf_available and len(f0_cand_hz_list) > 0:
                        diffs = np.abs(np.array(f0_cand_hz_list) - f0_hz)
                        closest_f0_cand_idx = np.argmin(diffs)

                    if hcf_available and 0 <= closest_f0_cand_idx < C_map.shape[0]:
                        f0_map_idx = closest_f0_cand_idx
                        current_coherence = C_map[f0_map_idx, frame_idx]

                    if "coherence_history" not in active_notes[track_id]:
                        active_notes[track_id]["coherence_history"] = []

                    active_notes[track_id]["pitches_hz"].append(f0_hz)
                    active_notes[track_id]["pitches_cents"].append(f0_cents)
                    active_notes[track_id]["last_update_frame"] = frame_idx
                    active_notes[track_id]["coherence_history"].append(
                        current_coherence
                    )
                    # --- SPOD履歴更新を削除 ---

                    matched_track_indices.add(r)
                    matched_f0_indices.add(c)

            # --- Handle Unmatched Active Tracks (Potential Note Ends) --- (SPOD判定削除)
            ended_track_ids = []
            for r, track_id in enumerate(active_track_ids):
                if r not in matched_track_indices:
                    note_info = active_notes[track_id]
                    frames_since_update = frame_idx - note_info["last_update_frame"]
                    timed_out = frames_since_update >= max_pitch_gap_frames

                    last_pitch_hz = (
                        note_info["pitches_hz"][-1] if note_info["pitches_hz"] else 0.0
                    )
                    current_coherence = 0.0
                    closest_f0_cand_idx_unmatched = -1
                    if hcf_available and len(f0_cand_hz_list) > 0 and last_pitch_hz > 0:
                        diffs = np.abs(np.array(f0_cand_hz_list) - last_pitch_hz)
                        closest_f0_cand_idx_unmatched = np.argmin(diffs)

                    if (
                        hcf_available
                        and 0 <= closest_f0_cand_idx_unmatched < C_map.shape[0]
                    ):
                        f0_map_idx = closest_f0_cand_idx_unmatched
                        current_coherence = C_map[f0_map_idx, frame_idx]

                    if "coherence_history" not in note_info:
                        note_info["coherence_history"] = []
                    note_info["coherence_history"].append(current_coherence)
                    # --- SPOD履歴更新を削除 ---

                    coherence_too_low = (
                        hcf_available
                        and current_coherence < self.hcf_coherence_offset_thresh
                    )

                    # --- SPODベースの終了条件を削除 ---

                    # --- 修正: HCF Coherence とタイムアウトのみで判定 ---
                    if timed_out or coherence_too_low:
                        # Verification Logic
                        if not timed_out:  # どの条件で早期終了したかログに残す
                            if coherence_too_low:
                                offset_advanced_coherence_low_count += 1
                                logger.debug(
                                    f"Frame {frame_idx}, Track {track_id}: Coherence low ({current_coherence:.3f} < {self.hcf_coherence_offset_thresh:.3f}) -> Advance offset."
                                )
                            # SPOD関連のログは削除
                            if coherence_too_low:
                                offset_advanced_count += 1

                        # --- 精密なオフセット時刻の補間計算 (既存のまま) ---
                        # ...
                        last_frame = min(
                            note_info["last_update_frame"], len(frame_times) - 1
                        )
                        if last_frame < 0:
                            ended_track_ids.append(track_id)
                            continue
                        precise_offset_time = frame_times[last_frame] + hop_time

                        if (
                            not timed_out
                            and coherence_too_low
                            and hcf_available
                            and 0 <= closest_f0_cand_idx_unmatched < C_map.shape[0]
                            and frame_idx > 0
                        ):
                            try:
                                f0_map_idx = closest_f0_cand_idx_unmatched
                                prev_coherence = C_map[f0_map_idx, frame_idx - 1]
                                curr_coherence = C_map[f0_map_idx, frame_idx]
                                if (
                                    prev_coherence
                                    >= self.hcf_coherence_offset_thresh
                                    > curr_coherence
                                ):
                                    coh_diff = curr_coherence - prev_coherence
                                    if abs(coh_diff) > 1e-9:
                                        frame_offset_ratio = (
                                            self.hcf_coherence_offset_thresh
                                            - prev_coherence
                                        ) / coh_diff
                                        time_offset = frame_offset_ratio * hop_time
                                        precise_offset_time = (
                                            frame_times[frame_idx - 1] + time_offset
                                        )
                            except Exception as interp_e:
                                logger.warning(
                                    f"Frame {frame_idx}, Track {track_id}: Error during precise offset time interpolation: {interp_e}",
                                    exc_info=False,
                                )

                        ended_track_ids.append(track_id)
                        # --- ノート終了処理 (uses self.min_note_duration_sec) ---
                        start_time = note_info["start_time"]
                        duration = precise_offset_time - start_time
                        pitches_hz = note_info["pitches_hz"]
                        if duration >= self.min_note_duration_sec and pitches_hz:
                            pitch_median_hz = np.median(np.array(pitches_hz))
                            notes_list.append(
                                {
                                    "start_time": start_time,
                                    "offset_time": precise_offset_time,
                                    "pitch_hz": pitch_median_hz,
                                    "duration": duration,
                                    "pitches_hz": pitches_hz,
                                }
                            )

            # Remove ended tracks (既存のまま)
            for track_id in ended_track_ids:
                if track_id in active_notes:  # Ensure key exists before deleting
                    del active_notes[track_id]

            # --- Start New Tracks for Unmatched F0s --- (SPOD履歴初期化削除)
            for c, f0_hz in enumerate(current_f0s_hz):
                if c not in matched_f0_indices:
                    is_onset = False
                    if frame_idx in allowed_onset_frames:
                        flux_value = 0.0
                        closest_f0_cand_idx_new = -1
                        if hcf_available and len(f0_cand_hz_list) > 0:
                            diffs = np.abs(np.array(f0_cand_hz_list) - f0_hz)
                            closest_f0_cand_idx_new = np.argmin(diffs)

                        if (
                            hcf_available
                            and 0 <= closest_f0_cand_idx_new < Phi_comb_map.shape[0]
                        ):
                            f0_map_idx = closest_f0_cand_idx_new
                            flux_value = Phi_comb_map[f0_map_idx, frame_idx]

                        if flux_value >= self.hcf_note_start_flux_thresh:
                            is_onset = True
                            logger.debug(
                                f"Frame {frame_idx}: HCF onset condition met for F0={f0_hz:.1f}Hz (O(n) peak AND Phi_comb={flux_value:.3f} >= {self.hcf_note_start_flux_thresh:.3f})."
                            )
                        else:
                            onset_suppressed_count += 1
                            logger.debug(
                                f"Frame {frame_idx}: Onset suppressed for F0={f0_hz:.1f}Hz (O(n) peak but Phi_comb={flux_value:.3f} < {self.hcf_note_start_flux_thresh:.3f})."
                            )
                    else:  # Not an overall onset frame according to HCF O(n)
                        onset_suppressed_count += 1

                    if is_onset:
                        # --- 精密なオンセット時刻の補間計算 (既存のまま) ---
                        # ...
                        precise_onset_time = current_time
                        if (
                            hcf_available
                            and 0 <= closest_f0_cand_idx_new < Phi_comb_map.shape[0]
                            and frame_idx > 0
                            and frame_idx < Phi_comb_map.shape[1] - 1
                        ):
                            try:
                                flux_values = Phi_comb_map[
                                    closest_f0_cand_idx_new,
                                    frame_idx - 1 : frame_idx + 2,
                                ]
                                if len(flux_values) == 3:
                                    delta_frame, _ = _parabolic_interpolation_numba(
                                        flux_values, 1
                                    )
                                    time_offset = delta_frame * hop_time
                                    precise_onset_time = current_time + time_offset
                            except Exception as interp_e:
                                logger.warning(
                                    f"Frame {frame_idx}, Track {track_id_counter}: Error during precise onset time interpolation: {interp_e}",
                                    exc_info=False,
                                )

                        # --- SPOD履歴初期化削除 ---
                        initial_coherence = 0.0
                        if (
                            hcf_available
                            and 0 <= closest_f0_cand_idx_new < C_map.shape[0]
                        ):
                            f0_map_idx = closest_f0_cand_idx_new
                            initial_coherence = C_map[f0_map_idx, frame_idx]

                        f0_cents = current_f0s_cents_np[c]
                        active_notes[track_id_counter] = {
                            "start_time": precise_onset_time,
                            "pitches_hz": [f0_hz],
                            "pitches_cents": [f0_cents],
                            "last_update_frame": frame_idx,
                            "coherence_history": [initial_coherence],
                            # 'spod_history': [initial_spod]
                        }
                        logger.debug(
                            f"Frame {frame_idx}: Started new track {track_id_counter} for F0={f0_hz:.1f}Hz (Method: HCF)."
                        )
                        track_id_counter += 1

        # --- Process Remaining Active Notes at End (uses self.min_note_duration_sec) ---
        # ... (既存のまま)
        if self._times is not None and self._sr is not None:
            frame_times = self._times
            num_frames_final = len(frame_times)
            hop_time = self.hop_length / self._sr
            if hop_time <= 0:
                hop_time = 1024 / 44100

            for track_id, note_info in active_notes.items():
                start_time = note_info["start_time"]
                last_frame = min(note_info["last_update_frame"], num_frames_final - 1)
                if last_frame < 0:
                    continue

                offset_time = frame_times[last_frame] + hop_time
                duration = offset_time - start_time
                pitches_hz = note_info["pitches_hz"]
                if duration >= self.min_note_duration_sec and pitches_hz:
                    pitch_median_hz = np.median(np.array(pitches_hz))
                    notes_list.append(
                        {
                            "start_time": start_time,
                            "offset_time": offset_time,
                            "pitch_hz": pitch_median_hz,
                            "duration": duration,
                            "pitches_hz": pitches_hz,
                        }
                    )

        # --- Verification Logging (SPOD関連削除) ---
        logger.info(
            f"HCF Check Verification: Onsets suppressed = {onset_suppressed_count}, Offsets advanced = {offset_advanced_count} (Coherence low = {offset_advanced_coherence_low_count})"
        )

        logger.debug(f"Note tracking yielded {len(notes_list)} raw notes.")
        return notes_list

    def _postprocess_notes(
        self, notes_list: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Post-processes notes using tunable parameters."""
        # ... (uses self.min_silence_duration_sec, self.note_merge_tolerance_cents, self.max_pitch_std_dev_cents, self.min_note_duration_sec)
        if not notes_list:
            return np.array([]).reshape(0, 2), np.array([])

        logger.debug(
            f"[_postprocess_notes] Start: Received {len(notes_list)} raw notes."
        )

        notes = [
            {
                "onset": n["start_time"],
                "offset": n["offset_time"],
                "pitch": n["pitch_hz"],
                "duration": n["duration"],
                "pitches_hz": n["pitches_hz"],
            }
            for n in notes_list
            if n["pitch_hz"] > 0 and n["duration"] > 1e-6
        ]
        logger.debug(
            f"[_postprocess_notes] After initial pitch/duration filter: {len(notes)} notes."
        )
        if not notes:
            return np.array([]).reshape(0, 2), np.array([])

        notes.sort(key=lambda n: n["onset"])

        # --- Merge consecutive notes (uses self.min_silence_duration_sec, self.note_merge_tolerance_cents) ---
        logger.debug(f"[_postprocess_notes] Before merging: {len(notes)} notes.")
        merged_notes: List[Dict[str, Any]] = []
        if notes:
            current_note = notes[0].copy()
            current_note_pitch_cents = hz_to_cents_numba(
                np.array([current_note["pitch"]])
            )[0]

            for i in range(1, len(notes)):
                next_note = notes[i]
                gap = next_note["onset"] - current_note["offset"]
                pitch2_cents = hz_to_cents_numba(np.array([next_note["pitch"]]))[0]
                pitch_diff_cents = (
                    abs(current_note_pitch_cents - pitch2_cents)
                    if np.isfinite(current_note_pitch_cents)
                    and np.isfinite(pitch2_cents)
                    else np.inf
                )

                if (
                    0 <= gap <= self.min_silence_duration_sec
                    and pitch_diff_cents <= self.note_merge_tolerance_cents
                ):
                    duration1 = current_note["duration"]
                    duration2 = next_note["duration"]
                    weight_duration1 = duration1 if duration1 > 1e-6 else 0.0
                    weight_duration2 = duration2 if duration2 > 1e-6 else 0.0
                    total_weight_duration = weight_duration1 + weight_duration2
                    if total_weight_duration > 1e-9:
                        current_note["pitch"] = (
                            current_note["pitch"] * weight_duration1
                            + next_note["pitch"] * weight_duration2
                        ) / total_weight_duration
                    else:
                        current_note["pitch"] = (
                            current_note["pitch"] + next_note["pitch"]
                        ) / 2.0
                    current_note["offset"] = next_note["offset"]
                    current_note["duration"] = (
                        current_note["offset"] - current_note["onset"]
                    )
                    current_note["pitches_hz"].extend(next_note["pitches_hz"])
                    current_note_pitch_cents = hz_to_cents_numba(
                        np.array([current_note["pitch"]])
                    )[0]
                else:
                    merged_notes.append(current_note)
                    current_note = next_note.copy()
                    current_note_pitch_cents = hz_to_cents_numba(
                        np.array([current_note["pitch"]])
                    )[0]

            merged_notes.append(current_note)
            notes = merged_notes
            logger.debug(
                f"[_postprocess_notes] After merging consecutive: {len(notes)} notes."
            )

        # --- Handle Overlaps (uses self.note_merge_tolerance_cents) ---
        logger.debug(
            f"[_postprocess_notes] Before overlap resolution: {len(notes)} notes."
        )
        processed_notes: List[Dict[str, Any]] = []
        if notes:
            notes.sort(key=lambda n: n["onset"])
            i = 0
            while i < len(notes):
                current_note = notes[i].copy()
                j = i + 1
                overlap_found_and_processed = False
                original_current_note_for_next_iter = current_note.copy()

                while j < len(notes) and notes[j]["onset"] < current_note["offset"]:
                    other_note = notes[j].copy()
                    pitch1_cents = hz_to_cents_numba(np.array([current_note["pitch"]]))[
                        0
                    ]
                    pitch2_cents = hz_to_cents_numba(np.array([other_note["pitch"]]))[0]
                    pitch_diff_cents = (
                        abs(pitch1_cents - pitch2_cents)
                        if np.isfinite(pitch1_cents) and np.isfinite(pitch2_cents)
                        else np.inf
                    )

                    if pitch_diff_cents <= self.note_merge_tolerance_cents:
                        duration1 = current_note["duration"]
                        duration2 = other_note["duration"]
                        weight1 = duration1
                        weight2 = duration2
                        total_weight = weight1 + weight2
                        if total_weight > 1e-9:
                            current_note["pitch"] = (
                                current_note["pitch"] * weight1
                                + other_note["pitch"] * weight2
                            ) / total_weight
                        else:
                            current_note["pitch"] = (
                                current_note["pitch"] + other_note["pitch"]
                            ) / 2.0
                        current_note["offset"] = max(
                            current_note["offset"], other_note["offset"]
                        )
                        current_note["duration"] = (
                            current_note["offset"] - current_note["onset"]
                        )
                        current_note["pitches_hz"].extend(other_note["pitches_hz"])
                        original_current_note_for_next_iter = current_note.copy()
                        notes.pop(j)
                        overlap_found_and_processed = True
                        break
                    else:
                        j += 1

                if not overlap_found_and_processed:
                    processed_notes.append(current_note)
                    i += 1
                else:
                    notes[i] = original_current_note_for_next_iter

            notes = processed_notes
            logger.debug(
                f"[_postprocess_notes] After overlap resolution: {len(notes)} notes."
            )

        # --- Pitch Stability Filter (uses self.max_pitch_std_dev_cents) ---
        logger.debug(
            f"[_postprocess_notes] Before pitch stability filter (max_std_dev={self.max_pitch_std_dev_cents:.1f} cents): {len(notes)} notes."
        )
        stable_notes: List[Dict[str, Any]] = []
        removed_unstable_count = 0
        for note in notes:
            pitches_hz = np.array(note.get("pitches_hz", []))
            if len(pitches_hz) >= 2:
                pitches_cents = hz_to_cents_numba(pitches_hz)
                valid_cents = pitches_cents[np.isfinite(pitches_cents)]
                if len(valid_cents) >= 2:
                    pitch_std_dev_cents = np.std(valid_cents)
                    if pitch_std_dev_cents <= self.max_pitch_std_dev_cents:
                        stable_notes.append(note)
                    else:
                        removed_unstable_count += 1
                else:
                    stable_notes.append(note)
            else:
                stable_notes.append(note)
        if removed_unstable_count > 0:
            logger.debug(
                f"    Removed {removed_unstable_count} notes due to pitch instability."
            )
        notes = stable_notes
        logger.debug(
            f"[_postprocess_notes] After pitch stability filter: {len(notes)} notes."
        )

        # --- Final Duration Filter (uses self.min_note_duration_sec) ---
        logger.debug(
            f"[_postprocess_notes] Before final duration filter ({self.min_note_duration_sec:.3f}s): {len(notes)} notes."
        )
        final_notes_intermediate = [
            n for n in notes if n.get("duration", 0.0) >= self.min_note_duration_sec
        ]
        logger.debug(
            f"[_postprocess_notes] Final notes after duration filter: {len(final_notes_intermediate)} notes."
        )

        if not final_notes_intermediate:
            return np.array([]).reshape(0, 2), np.array([])

        # --- 元の final_notes 作成とピッチ計算に戻す ---
        final_notes = []
        for note in notes:  # 元の 'notes' (安定性フィルタ後) を使用
            if note.get("duration", 0.0) >= self.min_note_duration_sec:
                pitches_hz = np.array(note.get("pitches_hz", []))
                if len(pitches_hz) > 0:
                    # 元のシンプルな中央値計算に戻す
                    note["pitch_hz"] = np.median(
                        pitches_hz
                    )  # 'pitch' キーではなく 'pitch_hz' に統一
                    final_notes.append(note)
                # else: ピッチ履歴がないノートは含めない（元の挙動に依存）

        logger.debug(
            f"[_postprocess_notes] Final notes after duration filter: {len(final_notes)} notes."
        )  # ログも修正

        if not final_notes:
            return np.array([]).reshape(0, 2), np.array([])

        intervals = np.array(
            [[n["onset"], n["offset"]] for n in final_notes]
        )  # final_notes を使用
        pitches = np.array([n["pitch_hz"] for n in final_notes])  # 'pitch_hz' を使用

        logger.debug(
            f"[_postprocess_notes] Returning {len(pitches)} notes (using original median pitch calculation)."
        )  # ログも修正
        return intervals, pitches

    def detect(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detects multiple pitches using the SPOD method, enhanced with HCF-based
        onset and offset detection.
        """
        start_process_time = time.time()
        # logger.info(f"Starting detection ({self.__class__.__name__}_HCF_Enhanced) with SR={sr}.") # Log at start if needed

        # --- Input Validation (既存のまま) ---
        if (
            not isinstance(audio_data, np.ndarray)
            or audio_data.ndim != 1
            or audio_data.size == 0
        ):
            logger.error("Invalid audio data provided.")
            return self._empty_result("Invalid audio data")
        if sr <= 0:
            logger.error(f"Invalid sample rate: {sr}.")
            return self._empty_result(f"Invalid sample rate: {sr}")

        try:
            # --- Initialization ---
            self._initialize_run(sr)
            if self._freq_resolution_hz is None or self._freq_resolution_hz <= 0:
                raise RuntimeError(
                    "Initialization failed, frequency resolution not set or invalid."
                )

            # --- Core Processing Steps ---
            logger.debug("1. Calculating STFT...")
            A_lin, S_db, phases = self._calculate_stft(audio_data)
            if A_lin.size == 0 or self._times is None:
                raise ValueError("STFT failed or produced empty results.")
            if self._freq_resolution_hz is None or self._freq_resolution_hz <= 0:
                raise ValueError("Frequency resolution invalid after STFT.")

            logger.debug("2. Calculating SPOD Map...")
            spod_map = self._calculate_spod(phases)
            if spod_map.shape != A_lin.shape:
                raise ValueError("SPOD map shape mismatch.")

            # --- HCF Calculation Step ---
            logger.debug("2.5. Calculating HCF Maps (C, Phi_comb, O)...")
            C_map, Phi_comb_map, O_vec = self._calculate_hcf_maps(spod_map, A_lin)
            hcf_onset_frames = None
            if O_vec is not None:
                logger.debug("Detecting onset peaks from HCF O_vec...")
                hcf_onset_peaks, _ = find_peaks(
                    O_vec,
                    height=self.hcf_onset_peak_thresh,  # Use instance var
                    prominence=self.hcf_onset_peak_prominence,  # Use instance var
                )
                hcf_onset_frames = set(hcf_onset_peaks)
                logger.debug(f"Detected {len(hcf_onset_frames)} HCF onset frames.")
            else:
                logger.warning(
                    "HCF O_vec calculation failed, cannot detect HCF onsets."
                )
                hcf_onset_frames = set()

            # --- Onset Frame Determination (常に HCF を使用) ---
            if hcf_onset_frames is not None:
                allowed_onset_frames = hcf_onset_frames
                logger.debug("Using HCF-derived onset frames.")
            # --- ハイブリッドオンセットへの分岐ロジックを削除 ---
            else:  # HCF計算失敗時のフォールバック
                logger.warning(
                    "HCF onset detection failed. Using empty set for allowed onsets."
                )
                allowed_onset_frames = set()

            self._allowed_onset_frames = allowed_onset_frames

            logger.debug("3. Detecting peaks per frame (Numba interp)...")
            peaks_per_frame = self._detect_peaks_per_frame(A_lin, S_db, spod_map)

            logger.debug("4. Estimating F0s from peaks (Numba scoring)...")
            multi_f0_per_frame = self._estimate_f0s_from_peaks(peaks_per_frame)

            logger.debug("5. Performing note segmentation (using HCF onset/offset)...")
            raw_notes_list = self._frames_to_notes_multipitch(
                multi_f0_per_frame=multi_f0_per_frame,
                # spod_map=spod_map, # Removed
                allowed_onset_frames=allowed_onset_frames,
                C_map=C_map,
                Phi_comb_map=Phi_comb_map,
            )

            logger.debug("6. Post-processing notes...")
            intervals, note_pitches = self._postprocess_notes(raw_notes_list)

            detection_time = time.time() - start_process_time
            logger.info(
                f"Detection completed in {detection_time:.4f} seconds. Found {len(note_pitches)} notes."
            )

            # --- Format Results ---
            frame_frequencies = (
                np.zeros_like(self._times, dtype=float)
                if self._times is not None
                else np.array([])
            )
            if self._times is not None and len(multi_f0_per_frame) == len(self._times):
                for idx, f0s in enumerate(multi_f0_per_frame):
                    positive_f0s = [f for f in f0s if f > 0]
                    frame_frequencies[idx] = min(positive_f0s) if positive_f0s else 0.0

            result = {
                "intervals": intervals,
                "note_pitches": note_pitches,
                "frame_times": self._times if self._times is not None else np.array([]),
                "frame_frequencies": frame_frequencies,
                "detector_name": self.__class__.__name__ + "_HCF_Enhanced",
                "detection_time": detection_time,
                "additional_data": {
                    "note_count": len(note_pitches),
                    "parameters": self.get_params(),
                    "hcf_onset_count": (
                        len(allowed_onset_frames)
                        if allowed_onset_frames is not None
                        else 0
                    ),
                },
            }
            return result

        except Exception as e:
            logger.error(f"Error during detection pipeline: {e}", exc_info=True)
            return self._empty_result(error_message=str(e))
        finally:
            # --- Clean up internal state (既存のまま) ---
            self._sr = None
            self._freqs = None
            self._times = None
            self._f0_candidates_hz = None
            self._A_lin = None
            self._allowed_onset_frames = None

    def _empty_result(
        self, error_message: str = "Detection failed or invalid input"
    ) -> Dict[str, Any]:
        """Returns a standardized empty result dictionary."""
        params = {}
        try:
            params = self.get_params()
        except Exception:
            logger.warning(
                "Could not retrieve parameters for empty result.", exc_info=False
            )

        return {
            "intervals": np.array([]).reshape(0, 2),
            "note_pitches": np.array([]),
            "frame_times": (
                self._times
                if hasattr(self, "_times") and self._times is not None
                else np.array([])
            ),
            "frame_frequencies": np.array([]),
            "detector_name": self.__class__.__name__ + "_HCF_Enhanced_Error",
            "detection_time": 0.0,
            "additional_data": {
                "error": error_message,
                "note_count": 0,
                "parameters": params,
            },
        }

    # --- HCF Helper Methods (New) ---

    def _calculate_harmonic_metrics(
        self, frame_idx: int, f0_hz: float, spod_map: np.ndarray, A_lin: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculates Harmonic Coherence (C) and Harmonic Amplitude Sum (AmpSum) for a given F0 and frame.
        Uses Numba-optimized helper.
        """
        if self._sr is None or self._freq_resolution_hz is None or self._freqs is None:
            logger.error(
                "Detector not properly initialized for harmonic metrics calculation."
            )
            return 0.0, 0.0

        n_bins = A_lin.shape[0]
        if frame_idx < 0 or frame_idx >= A_lin.shape[1]:
            logger.error(f"Invalid frame_idx {frame_idx} for map shape {A_lin.shape}")
            return 0.0, 0.0
        spod_map_frame = spod_map[:, frame_idx]
        A_lin_frame = A_lin[:, frame_idx]

        # Call Numba function
        C, AmpSum = _calculate_harmonic_metrics_numba(
            f0_hz,
            frame_idx,
            spod_map_frame,
            A_lin_frame,
            self._sr,
            self.n_fft,  # Use instance var
            self.hcf_max_harmonics,  # Use HCF param
            self.hcf_inharmonicity,  # Use HCF param
            self._freq_resolution_hz,
            n_bins,
            harmonic_weight_offset=self.harmonic_weight_offset,
            harmonic_weight_exponent=self.harmonic_weight_exponent,
        )
        return C, AmpSum

    def _calculate_hcf_maps(
        self, spod_map: np.ndarray, A_lin: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calculates HCF maps: C_map, Phi_comb_map, and O_vec.
        Returns None for maps if calculation fails.
        """
        if self._f0_candidates_hz is None or len(self._f0_candidates_hz) == 0:
            logger.warning("No F0 candidates available for HCF calculation.")
            return None, None, None
        if (
            self._sr is None
            or self._freq_resolution_hz is None
            or self._freq_resolution_hz <= 0
        ):
            logger.error("Detector not properly initialized for HCF calculation.")
            return None, None, None

        f0_candidates_hz_np = np.array(self._f0_candidates_hz, dtype=np.float64)

        try:
            logger.debug("Calculating HCF maps (Numba)...")
            (
                C_map,
                AmpSum_map,  # または _
                Phi_C_map,  # または _
                Phi_A_map,  # または _
                Phi_comb_map,
                O_vec,
                AvgHarmonicSPOD_map,  # 7番目の値を受け取る
            ) = _calculate_hcf_maps_numba(
                spod_map,
                A_lin,
                f0_candidates_hz_np,
                self._sr,
                self.n_fft,  # Use instance var
                self.hop_length,  # Use instance var
                self.hcf_max_harmonics,  # Use HCF param
                self.hcf_inharmonicity,  # Use HCF param
                self._freq_resolution_hz,
                self.hcf_tau_sec,  # Use HCF param
                self.hcf_flux_weight_c,  # Use HCF param
                self.hcf_flux_weight_a,  # Use HCF param
                self.harmonic_weight_offset,  # 倍音重み付けパラメータを追加
                self.harmonic_weight_exponent,  # 倍音重み付けパラメータを追加
            )
            logger.debug("HCF maps calculation finished.")

            # Optional: Smooth the overall onset function O_vec
            if self.hcf_onset_smoothing_sec > 0:  # Use HCF param
                smoothing_frames = max(
                    1,
                    int(
                        round(self.hcf_onset_smoothing_sec * self._sr / self.hop_length)
                    ),
                )  # Use instance vars & HCF param
                if smoothing_frames > 1 and len(O_vec) >= smoothing_frames:
                    logger.debug(
                        f"Smoothing HCF onset function O_vec with window {smoothing_frames}"
                    )
                    O_vec_writable = np.require(O_vec, requirements=["W"])
                    uniform_filter1d(
                        O_vec_writable,
                        size=smoothing_frames,
                        mode="nearest",
                        output=O_vec_writable,
                    )

            # Normalize O_vec 0-1
            max_O = np.max(O_vec)
            if max_O > 1e-9:
                O_vec /= max_O

            return C_map, Phi_comb_map, O_vec

        except Exception as e:
            logger.error(f"Error calculating HCF maps: {e}", exc_info=True)
            return None, None, None


# Helper Numba functions remain unchanged...
