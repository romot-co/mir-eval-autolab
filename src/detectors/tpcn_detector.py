# -*- coding: utf-8 -*-
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numba  # Numbaをインポート
import numpy as np
from scipy.interpolate import interp1d  # 線形補間用
from scipy.ndimage import uniform_filter1d  # インポートを追加
from scipy.optimize import linear_sum_assignment  # For Hungarian algorithm
from scipy.signal import find_peaks

from src.detectors.base_detector import BaseDetector  # BaseDetectorは別途定義されていると仮定

# Assuming Numba helper functions (hz_to_cents, cents_to_hz, _interpolate_map_numba, _parabolic_interpolation_numba) are defined elsewhere or inherited

# --- Logging Configuration ---
logger = logging.getLogger(__name__)

# --- Constants ---
HZ_REF = 10.0  # Reference frequency for cents conversion (consistent with librosa)


# --- Numba-Optimized Helper Functions for TPCN ---


@numba.njit(cache=True, fastmath=True)
def hz_to_cents_numba(freq_hz: np.ndarray, reference_hz: float = HZ_REF) -> np.ndarray:
    """Numba-optimized Hz to cents conversion."""
    cents = np.full_like(freq_hz, -np.inf, dtype=np.float64)
    valid_mask = freq_hz > 1e-9  # Avoid log(0) or log(negative)
    cents[valid_mask] = 1200 * np.log2(freq_hz[valid_mask] / reference_hz)
    return cents


@numba.njit(cache=True, fastmath=True)
def cents_to_hz_numba(cents: np.ndarray, reference_hz: float = HZ_REF) -> np.ndarray:
    """Numba-optimized Cents to Hz conversion."""
    return reference_hz * (2 ** (cents / 1200))


@numba.njit(cache=True, fastmath=True)
def _interpolate_map_numba(map_data_frame: np.ndarray, freq_bin_float: float) -> float:
    """
    Numba-optimized linear interpolation for a map value at a fractional bin index.
    Handles boundary conditions. Assumes map_data_frame is 1D for a single time frame.
    """
    n_bins = len(map_data_frame)
    if n_bins == 0:
        return 0.0

    # Boundary checks: Return nearest valid value if outside interpolation range
    if freq_bin_float < 0.0:
        return map_data_frame[0]
    if freq_bin_float >= n_bins - 1:
        # If exactly on the last bin or beyond, return the last bin's value
        return map_data_frame[n_bins - 1]

    # Linear interpolation
    k_low = int(np.floor(freq_bin_float))
    k_high = k_low + 1  # Ensured within bounds by check above
    frac = freq_bin_float - k_low

    val_low = map_data_frame[k_low]
    val_high = map_data_frame[k_high]

    # Check for NaN or Inf before interpolation
    if not (np.isfinite(val_low) and np.isfinite(val_high)):
        if np.isfinite(val_low):
            return val_low
        if np.isfinite(val_high):
            return val_high
        return 0.0  # Fallback

    return val_low + frac * (val_high - val_low)


@numba.njit(cache=True, fastmath=True)
def _parabolic_interpolation_numba(
    y_values: np.ndarray, peak_index: int
) -> Tuple[float, float]:
    """Numba-optimized parabolic interpolation for peak refinement."""
    # Simplified: assumes caller ensures peak_index is valid (not 0 or len-1)
    y_minus_1 = y_values[peak_index - 1]
    y_0 = y_values[peak_index]
    y_plus_1 = y_values[peak_index + 1]
    denominator = 2.0 * (y_minus_1 - 2.0 * y_0 + y_plus_1)
    if abs(denominator) < 1e-9:
        return 0.0, y_0
    delta_index = (y_minus_1 - y_plus_1) / denominator
    delta_index = max(-0.5, min(delta_index, 0.5))
    interpolated_value = y_0 - 0.25 * (y_minus_1 - y_plus_1) * delta_index
    interpolated_value = max(
        0.0, min(interpolated_value, y_0 * 1.1)
    )  # Ensure plausible
    return delta_index, interpolated_value


@numba.njit(cache=True, fastmath=True)
def _calculate_harmonic_frequencies_numba(
    f0: float,
    max_harmonics: int,
    inharmonicity_factor: float,
    max_f_analysis: float,  # Added max_f_analysis
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates harmonic frequencies considering inharmonicity.
    Returns harmonic_numbers, harmonic_frequencies_hz.
    Filters harmonics exceeding max_f_analysis.
    """
    if f0 <= 1e-9:  # Avoid issues with zero or negative f0
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    harmonic_numbers = np.arange(1, max_harmonics + 1, dtype=np.float64)

    if abs(inharmonicity_factor) < 1e-9:
        f_harmonics = f0 * harmonic_numbers
    else:
        # Ensure base is non-negative before sqrt
        inharmonicity_term_sq = np.maximum(
            0.0, 1.0 + inharmonicity_factor * harmonic_numbers**2
        )
        f_harmonics = f0 * harmonic_numbers * np.sqrt(inharmonicity_term_sq)

    # Filter harmonics beyond the maximum analysis frequency and invalid values
    valid_mask = (f_harmonics > 0) & (f_harmonics <= max_f_analysis)
    return harmonic_numbers[valid_mask], f_harmonics[valid_mask]


@numba.njit(cache=True, fastmath=True)
def _calculate_single_spod_metric_numba(
    f0_hz: float,
    spod_map_frame: np.ndarray,  # SPOD map for the current frame
    A_frame: np.ndarray,  # Linear amplitude spectrum for the frame
    sr: int,
    n_fft: int,
    h_max: int,
    inharmonicity_factor: float,
    freq_resolution_hz: float,
) -> Tuple[float, float]:  # Return S_peaks and total_weight used for averaging
    """
    Numba-optimized calculation of the SPOD metric (S_peaks in TPCN context)
    for a single F0 candidate and a single frame.
    Calculates weighted average SPOD at harmonic frequencies, weighted by (1/h) * Amplitude.
    [+] Corrected implementation based on TPCN paper/description.
    Returns: (S_peaks_value, total_weight)
    """
    # Basic validity check
    if f0_hz <= 0 or freq_resolution_hz <= 0:
        return 0.0, 0.0

    weighted_spod_sum = 0.0
    total_weight = 0.0  # Sum of weights (1/h * A_h)
    nyquist = sr / 2.0
    n_bins = len(spod_map_frame)  # Use SPOD map frame for n_bins
    max_f_analysis = nyquist

    # Ensure A_frame has the same number of bins
    if len(A_frame) != n_bins:
        # This case should ideally not happen if inputs are correct
        return 0.0, 0.0  # Return zero if dimensions mismatch

    # Get harmonic frequencies and numbers
    h_nums, h_freqs = _calculate_harmonic_frequencies_numba(
        f0_hz, h_max, inharmonicity_factor, max_f_analysis
    )
    num_valid_harmonics = len(h_freqs)
    if num_valid_harmonics == 0:
        return 0.0, 0.0

    # Loop through valid harmonics (h=1 to h_max)
    for h_idx in range(num_valid_harmonics):
        h = h_nums[h_idx]
        h_freq = h_freqs[h_idx]

        # Convert harmonic frequency to fractional bin index
        freq_bin_float = h_freq / freq_resolution_hz

        # Interpolate SPOD and Amplitude at the harmonic frequency
        interp_spod = 0.0
        interp_amp = 0.0
        if 0 <= freq_bin_float < n_bins - 1:  # Check bounds for interpolation
            interp_spod = _interpolate_map_numba(spod_map_frame, freq_bin_float)
            interp_amp = _interpolate_map_numba(A_frame, freq_bin_float)

        # Check if interpolated values are valid and harmonic number is positive
        if np.isfinite(interp_spod) and np.isfinite(interp_amp) and h > 0:
            # Calculate weight according to TPCN definition: (1/h) * Amplitude
            # Ensure interp_amp is non-negative
            weight = (1.0 / h) * max(0.0, interp_amp)

            if np.isfinite(weight) and weight > 0:
                # Accumulate weighted SPOD sum and total weight
                weighted_spod_sum += weight * interp_spod
                total_weight += weight

    # Calculate S_peaks: weighted average SPOD
    s_peaks = weighted_spod_sum / total_weight if total_weight > 1e-9 else 0.0

    # Clip to [0, 1]
    s_peaks = max(0.0, min(1.0, s_peaks))

    return s_peaks, total_weight  # Return the S_peaks value and the total weight used


@numba.njit(cache=True, fastmath=True)
def _calculate_single_valley_spod_metric_numba(
    f0_hz: float,
    f0_cents: float,  # F0 in cents (precomputed)
    spod_frame: np.ndarray,
    A_frame: np.ndarray,
    sr: int,
    n_fft: int,
    h_max: int,
    inharmonicity_factor: float,
    freq_resolution_hz: float,
    valley_window_cents: float,  # Search window half-width around valley center
    valley_weight_exponent: float,  # Weighting exponent for SPOD values in the window
) -> Tuple[float, float]:
    """
    Numba-optimized calculation of weighted average SPOD in harmonic valleys.
    (S_valleys in TPCN context).
    Returns: (avg_s_valleys, total_valley_weight)
    """
    # Basic validity check
    if f0_hz <= 0 or freq_resolution_hz <= 0 or not np.isfinite(f0_cents):
        return 0.0, 0.0

    weighted_spod_sum_valleys = 0.0
    total_valley_weight = 0.0
    nyquist = sr / 2.0
    n_bins = len(spod_frame)
    max_f_analysis = nyquist

    # Get valid harmonic frequencies and numbers
    h_nums, h_freqs = _calculate_harmonic_frequencies_numba(
        f0_hz,
        h_max + 1,
        inharmonicity_factor,
        max_f_analysis,  # Need h+1 for valley calculation
    )
    num_valid_harmonics = len(h_freqs)
    if num_valid_harmonics < 2:  # Need at least two harmonics to have a valley
        return 0.0, 0.0

    # Convert harmonic frequencies to cents
    h_cents = hz_to_cents_numba(h_freqs)

    # Loop through valleys between valid harmonics
    for i in range(num_valid_harmonics - 1):
        h1_cents = h_cents[i]
        h2_cents = h_cents[i + 1]

        # Check if cents are valid
        if not (np.isfinite(h1_cents) and np.isfinite(h2_cents)):
            continue

        # Calculate valley center in cents
        valley_center_cents = (h1_cents + h2_cents) / 2.0

        # --- Calculate SPOD average within the valley window ---
        # Define window boundaries in cents
        min_valley_cents = valley_center_cents - valley_window_cents / 2.0
        max_valley_cents = valley_center_cents + valley_window_cents / 2.0

        # Convert window boundaries back to Hz
        min_valley_hz = cents_to_hz_numba(np.array([min_valley_cents]))[0]
        max_valley_hz = cents_to_hz_numba(np.array([max_valley_cents]))[0]

        # Convert Hz boundaries to bin indices (approximate)
        min_valley_bin = int(np.floor(min_valley_hz / freq_resolution_hz))
        max_valley_bin = int(np.ceil(max_valley_hz / freq_resolution_hz))

        # Clamp bin indices to valid range [0, n_bins-1]
        min_valley_bin = max(0, min_valley_bin)
        max_valley_bin = min(n_bins - 1, max_valley_bin)

        # Iterate through bins within the valley window
        num_bins_in_valley = 0
        current_valley_spod_sum = 0.0
        current_valley_weight_sum = 0.0
        if min_valley_bin <= max_valley_bin:  # Ensure range is valid
            for k in range(min_valley_bin, max_valley_bin + 1):
                spod_val = spod_frame[k]
                amp_val = A_frame[k]

                # Weighting (e.g., by amplitude^exponent)
                # Spec says "振幅近傍（あるいは固定重み）". Using amplitude seems reasonable.
                # Let valley_weight_exponent control the influence of amplitude.
                # [+] Revert to amplitude weighting, exponent controlled by parameter
                if np.isfinite(spod_val) and np.isfinite(amp_val) and amp_val > 0:
                    # Calculate weight based on amplitude and exponent
                    weight = amp_val**valley_weight_exponent
                    if not np.isfinite(weight) or weight < 0:
                        weight = 0.0

                    current_valley_spod_sum += weight * spod_val
                    current_valley_weight_sum += weight
                    num_bins_in_valley += 1

        # Calculate average SPOD for this valley window
        if current_valley_weight_sum > 1e-9:
            avg_spod_this_valley = current_valley_spod_sum / current_valley_weight_sum
            # Weight the contribution of this valley (e.g., uniformly or by avg amplitude?)
            # For now, let's do a simple average over valleys
            weighted_spod_sum_valleys += avg_spod_this_valley
            total_valley_weight += 1.0  # Count number of valleys processed

    # Calculate final average SPOD across all valleys
    avg_s_valleys = (
        weighted_spod_sum_valleys / total_valley_weight
        if total_valley_weight > 0
        else 0.0
    )
    # Clip to [0, 1]
    avg_s_valleys = max(0.0, min(1.0, avg_s_valleys))

    return avg_s_valleys, total_valley_weight  # Return weight = number of valleys found


# [+] Add new Numba function for HME calculation
@numba.njit(cache=True, fastmath=True)
def _calculate_single_hme_numba(
    f0_hz: float,
    A_lin_frame: np.ndarray,  # Linear amplitude spectrum for the frame
    sr: int,
    n_fft: int,
    h_max: int,
    inharmonicity_factor: float,
    freq_resolution_hz: float,
    # [-] Remove harmonic weighting parameters, use fixed 1/h
) -> float:
    """
    Numba-optimized calculation of Harmonic Magnitude Evidence (HME)
    for a SINGLE F0 and a SINGLE frame. Uses amplitude spectrum directly.
    Includes inharmonicity and fixed 1/h harmonic weighting as per TPCN spec.
    """
    # Basic validity check
    if f0_hz <= 0 or freq_resolution_hz <= 0:
        return 0.0

    weighted_amp_sum = 0.0
    # HME is a sum, not an average, so total_weight isn't strictly needed for the final value
    # total_amp_weight = 0.0
    nyquist = sr / 2.0
    n_bins = len(A_lin_frame)
    max_f_analysis = nyquist

    # Get valid harmonic frequencies and numbers
    h_nums, h_freqs = _calculate_harmonic_frequencies_numba(
        f0_hz, h_max, inharmonicity_factor, max_f_analysis
    )
    num_valid_harmonics = len(h_freqs)
    if num_valid_harmonics == 0:
        return 0.0

    # Loop through valid harmonics
    for i in range(num_valid_harmonics):
        h = h_nums[i]
        harmonic_freq = h_freqs[i]

        # Convert harmonic frequency to fractional bin index
        freq_bin_float = harmonic_freq / freq_resolution_hz

        # Interpolate Amplitude using the linear interpolation helper
        interp_amp = 0.0
        if 0 <= freq_bin_float < n_bins - 1:
            interp_amp = _interpolate_map_numba(A_lin_frame, freq_bin_float)

        # Ensure interpolated amplitude is valid and harmonic number > 0
        if np.isfinite(interp_amp) and h > 0:
            # Calculate harmonic weight (fixed 1/h)
            weight = 1.0 / h

            # Accumulate weighted Amplitude
            weighted_amp_sum += weight * max(
                0.0, interp_amp
            )  # Ensure amplitude is non-negative
            # total_amp_weight += weight # Not needed for sum

    # HME is the weighted sum
    hme_value = weighted_amp_sum

    # Ensure HME is non-negative
    hme_value = max(0.0, hme_value)

    return hme_value


# Enable parallelism and consider cache=False due to potential size/parallelism interaction
@numba.njit(cache=False, parallel=True, fastmath=True)
def _calculate_tpcn_maps_parallel_numba(
    A_lin: np.ndarray,  # (n_bins, n_frames)
    SPOD_map: np.ndarray,  # (n_bins, n_frames)
    F_cand_hz: np.ndarray,  # (n_f0_cands,) - Array of F0 candidates in Hz
    sr: int,
    n_fft: int,
    hop_length: int,
    h_max: int,  # Max harmonics for calculations
    inharmonicity_factor: float,  # Inharmonicity factor
    freq_resolution_hz: float,  # Frequency resolution of the STFT
    tpcn_lambda: float,  # Lambda weighting factor for PC boost
    valley_window_cents: float,  # Window size for valley calculation
    valley_weight_exponent: float,  # Weight exponent for valley averaging
    # [-] Remove harmonic_weight_offset and harmonic_weight_exponent from args
    # [+] Update return type hint -> No change needed here
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized parallel calculation of TPCN maps:
    PS_lambda_map, PC_diff_map, HME_map, OS_TPCN_map.

    Returns: PS_lambda_map, PC_diff_map, HME_map, OS_TPCN_map
    """
    n_bins, n_frames = A_lin.shape
    n_f0_cands = len(F_cand_hz)

    # --- Output maps initialization ---
    PS_lambda_map = np.zeros((n_f0_cands, n_frames), dtype=np.float64)
    PC_diff_map = np.zeros((n_f0_cands, n_frames), dtype=np.float64)
    HME_map = np.zeros((n_f0_cands, n_frames), dtype=np.float64)
    OS_TPCN_map = np.zeros((n_f0_cands, n_frames), dtype=np.float64)

    # --- Precompute F0 candidates in cents for valley search ---
    F_cand_cents = np.full(n_f0_cands, -np.inf, dtype=np.float64)
    valid_f0_mask = F_cand_hz > 0
    if np.any(valid_f0_mask):
        F_cand_cents[valid_f0_mask] = 1200 * np.log2(
            F_cand_hz[valid_f0_mask] / 10.0
        )  # Ref 10 Hz

    # --- Calculate PS_lambda, PC_diff, HME maps ---
    # Parallelize the loop over F0 candidates
    for i in numba.prange(n_f0_cands):
        f0 = F_cand_hz[i]
        f0_cents = F_cand_cents[i]  # Precomputed cents value

        # Skip if F0 is invalid (e.g., <= 0)
        if f0 <= 0 or not np.isfinite(f0_cents):
            continue

        # Loop through each frame for the current F0 candidate
        for n in range(n_frames):
            # Extract the spectral data for the current frame
            A_frame = A_lin[:, n]
            SPOD_frame = SPOD_map[:, n]

            # Calculate S_peaks (weighted average SPOD at harmonics)
            # [+] Pass correct arguments, remove unused ones
            S_peaks, _ = _calculate_single_spod_metric_numba(
                f0,
                SPOD_frame,
                A_frame,  # Pass SPOD_frame and A_frame
                sr,
                n_fft,
                h_max,
                inharmonicity_factor,
                freq_resolution_hz,
                # Removed harmonic weight params
            )

            # Calculate S_valleys (weighted average SPOD in valleys)
            S_valleys, _ = _calculate_single_valley_spod_metric_numba(
                f0,
                f0_cents,
                SPOD_frame,
                A_frame,  # Pass SPOD_frame for S_valleys, A_frame for weighting
                sr,
                n_fft,
                h_max,
                inharmonicity_factor,
                freq_resolution_hz,
                valley_window_cents,
                valley_weight_exponent,
            )

            # [+] Calculate HME using the updated dedicated function (removed weight params)
            hme_value = _calculate_single_hme_numba(
                f0,
                A_frame,  # Use A_frame for HME calculation
                sr,
                n_fft,
                h_max,
                inharmonicity_factor,
                freq_resolution_hz,
                # Removed harmonic weight params
            )
            HME_map[i, n] = hme_value  # Store HME

            # Calculate Phase Contrast Difference
            pc_diff = S_peaks - S_valleys
            PC_diff_map[i, n] = pc_diff  # Store raw PC_diff

            # Calculate lambda-controlled Salience (PS_lambda)
            # Ensure phase contrast boost is non-negative
            phase_contrast_boost = max(0.0, pc_diff)
            # [+] Use hme_value calculated above
            ps_lambda = hme_value * np.exp(tpcn_lambda * phase_contrast_boost)
            PS_lambda_map[i, n] = ps_lambda

    # --- Calculate Onset Score Map (OS_TPCN) ---
    # Calculate frame-wise difference of PS_lambda_map manually for Numba compatibility
    if n_frames > 1:
        # Parallelize the loop over F0 candidates for difference calculation
        # Compute difference along time axis (axis=1) manually
        for i in numba.prange(n_f0_cands):
            for n in range(1, n_frames):  # Start from the second frame
                diff_val = PS_lambda_map[i, n] - PS_lambda_map[i, n - 1]
                OS_TPCN_map[i, n] = max(0.0, diff_val)  # Apply maximum(0, diff)

    # First frame has zero onset score by definition here
    OS_TPCN_map[:, 0] = 0.0

    # [+] Update return statement
    return PS_lambda_map, PC_diff_map, HME_map, OS_TPCN_map


# --- TPCNDetector Class ---


class TPCNDetector(BaseDetector):
    """
    TPCN-based multi-pitch detector using Phase Contrast boost.
    Inherits STFT, SPOD calculation, and basic note processing from BaseDetector or common utils.
    """

    # --- Default Parameters (copied/adapted from PZSTD where applicable) ---
    DEFAULT_N_FFT = 2048
    DEFAULT_HOP_LENGTH = 1024
    DEFAULT_WINDOW = "hann"
    DEFAULT_MIN_FREQ_HZ = 30.0
    DEFAULT_MAX_FREQ_HZ = 5000.0
    DEFAULT_F0_CANDIDATE_RESOLUTION_CENTS = 15.0  # Resolution for F0 grid
    # Parameters likely inherited from PZSTD or Base used in SPOD/HME/PC calcs:
    DEFAULT_MAX_HARMONICS = 12
    DEFAULT_INHARMONICITY_FACTOR = 0.0001
    DEFAULT_HARMONIC_WEIGHT_OFFSET = 0.0
    DEFAULT_HARMONIC_WEIGHT_EXPONENT = 1.0
    # SPOD parameters (assuming calculated similarly to PZSTD)
    DEFAULT_SPOD_WINDOW_SEC = 0.07
    DEFAULT_SPOD_ALPHA = 1.1

    # TPCN Specific Parameters
    # [+] Restore lambda to 1.0 to enable phase contrast effect
    DEFAULT_TPCN_LAMBDA = 1.0  # (λ) Weighting factor for phase contrast boost
    DEFAULT_TPCN_SALIENCE_THRESHOLD = (
        0.08  # Min PS_lambda value (Increased: 0.05 -> 0.08)
    )
    DEFAULT_TPCN_ONSET_SCORE_THRESHOLD = (
        0.1  # Min OS_TPCN value (Increased: 0.05 -> 0.1)
    )
    DEFAULT_TPCN_PC_DIFF_THRESHOLD = 0.1  # Min PC_diff value (Increased: 0.05 -> 0.1)
    DEFAULT_TPCN_VALLEY_WINDOW_CENTS = (
        50.0  # Search window size around valley centers (cents)
    )
    DEFAULT_TPCN_VALLEY_WEIGHT_EXPONENT = (
        0.5  # Weight exponent for valley SPOD averaging (Reverted: 0.0 -> 0.5)
    )

    # Note Tracking & Refinement Parameters (Some may be reused from PZSTD defaults if applicable)
    DEFAULT_TPCN_MAX_PITCH_GAP_SEC = 0.05  # Max time gap to bridge notes in tracking
    DEFAULT_TPCN_PITCH_CONTINUITY_TOLERANCE_CENTS = 50.0  # Max pitch diff for tracking
    DEFAULT_TPCN_MIN_NOTE_DURATION_SEC = 0.05  # Min duration for a valid note
    DEFAULT_TPCN_MAX_PITCH_STD_DEV_CENTS = 40.0  # Max pitch std dev within a note
    DEFAULT_TPCN_NOTE_MERGE_TOLERANCE_CENTS = (
        50.0  # Max pitch diff to merge adjacent notes
    )
    DEFAULT_TPCN_MIN_SILENCE_DURATION_SEC = (
        0.04  # Max silence duration to merge adjacent notes
    )
    # [+] Add NMS parameters
    DEFAULT_TPCN_NMS_WINDOW_CENTS = (
        50.0  # Window size for Non-Maximum Suppression on F0 candidates
    )
    # [+] Add Map Smoothing Parameters
    DEFAULT_TPCN_MAP_SMOOTH_FRAMES = (
        3  # Num frames for time smoothing (use odd numbers)
    )
    DEFAULT_TPCN_MAP_SMOOTH_CENTS = 30.0  # Cents window for frequency smoothing

    def __init__(
        self,
        # Basic STFT/Common Params
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        window: str = DEFAULT_WINDOW,
        min_freq_hz: float = DEFAULT_MIN_FREQ_HZ,
        max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
        f0_candidate_resolution_cents: float = DEFAULT_F0_CANDIDATE_RESOLUTION_CENTS,
        # Harmonic/SPOD Params (likely needed by helpers)
        max_harmonics: int = DEFAULT_MAX_HARMONICS,
        inharmonicity_factor: float = DEFAULT_INHARMONICITY_FACTOR,
        harmonic_weight_offset: float = DEFAULT_HARMONIC_WEIGHT_OFFSET,
        harmonic_weight_exponent: float = DEFAULT_HARMONIC_WEIGHT_EXPONENT,
        spod_window_sec: float = DEFAULT_SPOD_WINDOW_SEC,
        spod_alpha: float = DEFAULT_SPOD_ALPHA,
        # TPCN Specific Params
        tpcn_lambda: float = DEFAULT_TPCN_LAMBDA,
        tpcn_salience_threshold: float = DEFAULT_TPCN_SALIENCE_THRESHOLD,
        tpcn_onset_score_threshold: float = DEFAULT_TPCN_ONSET_SCORE_THRESHOLD,
        # [+] Add new parameter to constructor
        tpcn_pc_diff_threshold: float = DEFAULT_TPCN_PC_DIFF_THRESHOLD,
        tpcn_valley_window_cents: float = DEFAULT_TPCN_VALLEY_WINDOW_CENTS,
        tpcn_valley_weight_exponent: float = DEFAULT_TPCN_VALLEY_WEIGHT_EXPONENT,
        # Tracking/Refinement Params
        tpcn_max_pitch_gap_sec: float = DEFAULT_TPCN_MAX_PITCH_GAP_SEC,
        tpcn_pitch_continuity_tolerance_cents: float = DEFAULT_TPCN_PITCH_CONTINUITY_TOLERANCE_CENTS,
        tpcn_min_note_duration_sec: float = DEFAULT_TPCN_MIN_NOTE_DURATION_SEC,
        tpcn_max_pitch_std_dev_cents: float = DEFAULT_TPCN_MAX_PITCH_STD_DEV_CENTS,
        tpcn_note_merge_tolerance_cents: float = DEFAULT_TPCN_NOTE_MERGE_TOLERANCE_CENTS,
        tpcn_min_silence_duration_sec: float = DEFAULT_TPCN_MIN_SILENCE_DURATION_SEC,
        # [+] Add NMS parameter to constructor
        tpcn_nms_window_cents: float = DEFAULT_TPCN_NMS_WINDOW_CENTS,
        # [+] Add Map Smoothing parameters to constructor
        tpcn_map_smooth_frames: int = DEFAULT_TPCN_MAP_SMOOTH_FRAMES,
        tpcn_map_smooth_cents: float = DEFAULT_TPCN_MAP_SMOOTH_CENTS,
        **kwargs,
    ):
        """
        Initializes the TPCNDetector with specified parameters.
        """
        super().__init__(**kwargs)

        # --- Assign Core Parameters ---
        self.n_fft = max(128, n_fft)
        self.hop_length = max(1, hop_length)
        self.window = window
        self.min_freq_hz = max(0.0, min_freq_hz)
        self.max_freq_hz = max(self.min_freq_hz + 1, max_freq_hz)
        self.f0_candidate_resolution_cents = max(1.0, f0_candidate_resolution_cents)
        # Harmonic/SPOD Params
        self.max_harmonics = max(1, max_harmonics)
        self.inharmonicity_factor = inharmonicity_factor
        self.harmonic_weight_offset = max(0.0, harmonic_weight_offset)
        self.harmonic_weight_exponent = harmonic_weight_exponent
        self.spod_window_sec = max(0.001, spod_window_sec)
        self.spod_alpha = max(0.0, spod_alpha)

        # TPCN Specific Params
        self.tpcn_lambda = max(0.0, tpcn_lambda)
        self.tpcn_salience_threshold = max(0.0, tpcn_salience_threshold)
        self.tpcn_onset_score_threshold = max(0.0, tpcn_onset_score_threshold)
        # [+] Assign new threshold
        self.tpcn_pc_diff_threshold = max(
            -np.inf, tpcn_pc_diff_threshold
        )  # Allow negative threshold if needed? Or max(0,...)
        self.tpcn_valley_window_cents = max(1.0, tpcn_valley_window_cents)
        self.tpcn_valley_weight_exponent = tpcn_valley_weight_exponent

        # Note Tracking & Refinement Params
        self.tpcn_max_pitch_gap_sec = max(0.0, tpcn_max_pitch_gap_sec)
        self.tpcn_pitch_continuity_tolerance_cents = max(
            1.0, tpcn_pitch_continuity_tolerance_cents
        )
        self.tpcn_min_note_duration_sec = max(0.0, tpcn_min_note_duration_sec)
        self.tpcn_max_pitch_std_dev_cents = max(0.0, tpcn_max_pitch_std_dev_cents)
        self.tpcn_note_merge_tolerance_cents = max(0.0, tpcn_note_merge_tolerance_cents)
        self.tpcn_min_silence_duration_sec = max(0.0, tpcn_min_silence_duration_sec)
        # [+] Assign NMS parameter
        self.tpcn_nms_window_cents = max(1.0, tpcn_nms_window_cents)
        # [+] Assign Map Smoothing parameters
        self.tpcn_map_smooth_frames = max(1, int(tpcn_map_smooth_frames))
        self.tpcn_map_smooth_cents = max(0.0, tpcn_map_smooth_cents)

        # Internal state variables
        self._sr: Optional[int] = None
        self._times: Optional[np.ndarray] = None
        self._freqs: Optional[np.ndarray] = None
        self._freq_resolution_hz: Optional[float] = None
        self._f0_candidates_hz: Optional[np.ndarray] = None
        self._max_f_analysis: float = 0.0
        self._A_lin: Optional[np.ndarray] = None
        self._spod_map: Optional[np.ndarray] = None
        # TPCN specific maps
        self._PS_lambda_map: Optional[np.ndarray] = None
        self._PC_diff_map: Optional[np.ndarray] = None
        self._HME_map: Optional[np.ndarray] = None  # [+] Store HME map
        self._OS_TPCN_map: Optional[np.ndarray] = None

    def get_params(self) -> Dict[str, Any]:
        """Returns a dictionary of all detector parameters."""
        # Using direct listing for clarity and robustness
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "window": self.window,
            "min_freq_hz": self.min_freq_hz,
            "max_freq_hz": self.max_freq_hz,
            "f0_candidate_resolution_cents": self.f0_candidate_resolution_cents,
            "max_harmonics": self.max_harmonics,
            "inharmonicity_factor": self.inharmonicity_factor,
            "harmonic_weight_offset": self.harmonic_weight_offset,
            "harmonic_weight_exponent": self.harmonic_weight_exponent,
            "spod_window_sec": self.spod_window_sec,
            "spod_alpha": self.spod_alpha,
            "tpcn_lambda": self.tpcn_lambda,
            "tpcn_salience_threshold": self.tpcn_salience_threshold,
            "tpcn_onset_score_threshold": self.tpcn_onset_score_threshold,
            "tpcn_pc_diff_threshold": self.tpcn_pc_diff_threshold,  # [+] Include new param
            "tpcn_valley_window_cents": self.tpcn_valley_window_cents,
            "tpcn_valley_weight_exponent": self.tpcn_valley_weight_exponent,
            "tpcn_max_pitch_gap_sec": self.tpcn_max_pitch_gap_sec,
            "tpcn_pitch_continuity_tolerance_cents": self.tpcn_pitch_continuity_tolerance_cents,
            "tpcn_min_note_duration_sec": self.tpcn_min_note_duration_sec,
            "tpcn_max_pitch_std_dev_cents": self.tpcn_max_pitch_std_dev_cents,
            "tpcn_note_merge_tolerance_cents": self.tpcn_note_merge_tolerance_cents,
            "tpcn_min_silence_duration_sec": self.tpcn_min_silence_duration_sec,
            "tpcn_nms_window_cents": self.tpcn_nms_window_cents,  # [+] Include NMS param
            # [+] Include Map Smoothing params
            "tpcn_map_smooth_frames": self.tpcn_map_smooth_frames,
            "tpcn_map_smooth_cents": self.tpcn_map_smooth_cents,
        }

    def _initialize_run(self, sr: int):
        """Initializes internal state variables for a new run."""
        if sr <= 0:
            raise ValueError(f"Invalid sample rate: {sr}")
        self._sr = sr
        self._freqs = librosa.fft_frequencies(sr=self._sr, n_fft=self.n_fft)
        if len(self._freqs) < 2:
            self._freq_resolution_hz = sr / self.n_fft if self.n_fft > 0 else None
        else:
            self._freq_resolution_hz = float(self._freqs[1] - self._freqs[0])
        if self._freq_resolution_hz is None or self._freq_resolution_hz <= 0:
            raise RuntimeError("Frequency resolution calculation failed.")

        self._max_f_analysis = min(self.max_freq_hz, self._freqs[-1])
        self._f0_candidates_hz = self._generate_f0_candidates()

        self._times = None
        self._A_lin = None
        self._spod_map = None
        self._PS_lambda_map = None
        self._PC_diff_map = None
        self._HME_map = None  # [+] Reset HME map
        self._OS_TPCN_map = None
        logger.debug(
            f"Run initialized for SR={sr}. Freq resolution: {self._freq_resolution_hz:.3f} Hz. {len(self._f0_candidates_hz)} F0 candidates."
        )

    def _generate_f0_candidates(self) -> np.ndarray:
        """Generates F0 candidates in Hz on a cents scale."""
        if self._freqs is None or self._freq_resolution_hz is None:
            return np.array([])
        min_f = self.min_freq_hz
        max_f = min(self.max_freq_hz, self._freqs[-1])
        if min_f >= max_f:
            return np.array([])
        min_cents_arr = hz_to_cents_numba(np.array([min_f]))
        max_cents_arr = hz_to_cents_numba(np.array([max_f]))
        if not (np.isfinite(min_cents_arr[0]) and np.isfinite(max_cents_arr[0])):
            return np.array([])
        min_cents, max_cents = min_cents_arr[0], max_cents_arr[0]
        num_steps = (
            int(np.ceil((max_cents - min_cents) / self.f0_candidate_resolution_cents))
            + 1
        )
        if num_steps <= 1:
            f0_candidates_hz = (
                np.array([min_f, max_f]) if min_f != max_f else np.array([min_f])
            )
        else:
            f0_candidates_cents = np.linspace(min_cents, max_cents, num_steps)
            f0_candidates_hz = cents_to_hz_numba(f0_candidates_cents)
        return f0_candidates_hz[
            (f0_candidates_hz >= min_f) & (f0_candidates_hz <= max_f)
        ]

    def _calculate_stft(self, audio_data: np.ndarray) -> bool:
        """Calculates STFT and linear amplitude. Returns True on success."""
        if self._sr is None:
            return False
        try:
            S_complex = librosa.stft(
                audio_data,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                center=True,
            )
            self._A_lin = np.abs(S_complex)
            self._times = librosa.frames_to_time(
                np.arange(S_complex.shape[1]), sr=self._sr, hop_length=self.hop_length
            )
            return True
        except Exception as e:
            logger.error(f"Error calculating STFT: {e}", exc_info=True)
            return False

    def _calculate_spod(self, audio_data: np.ndarray) -> bool:
        """Calculates STFT phase and SPOD map. Requires STFT to be run first (implicitly). Returns True on success."""
        if self._sr is None:
            return False
        try:
            # Re-calculate phases here as they are needed for SPOD
            S_complex = librosa.stft(
                audio_data,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                center=True,
            )
            phases = np.angle(S_complex)

            n_bins, n_frames = phases.shape
            if n_frames <= 1:
                return False
            spod_window_frames = max(
                1, int(round(self.spod_window_sec * self._sr / self.hop_length))
            )
            phi_unwrapped = np.unwrap(phases, axis=1)
            inst_freq = np.diff(
                phi_unwrapped, n=1, axis=1, append=phi_unwrapped[:, -1:]
            )

            # Using uniform_filter1d as a simpler alternative to Gaussian for variance calculation
            inst_freq_sq_mean = uniform_filter1d(
                inst_freq**2, size=spod_window_frames, axis=1, mode="nearest"
            )
            inst_freq_mean = uniform_filter1d(
                inst_freq, size=spod_window_frames, axis=1, mode="nearest"
            )
            inst_freq_var = np.maximum(0, inst_freq_sq_mean - inst_freq_mean**2)

            self._spod_map = np.exp(-self.spod_alpha * inst_freq_var)
            self._spod_map = np.clip(self._spod_map, 0.0, 1.0)
            return True
        except Exception as e:
            logger.error(f"Error calculating SPOD: {e}", exc_info=True)
            return False

    # [+] Wrapper function (if used, needs update)
    def _calculate_tpcn_maps(self) -> bool:
        """
        Calculates TPCN maps using the optimized Numba function.
        Maps: PS_lambda, PC_diff, HME, OS_TPCN.
        Returns True on success, False on failure.
        Populates internal state variables (_PS_lambda_map, etc.).
        """
        if (
            self._A_lin is None
            or self._spod_map is None
            or self._f0_candidates_hz is None
            or self._sr is None
            or self._freq_resolution_hz is None
        ):
            logger.error("Required data for TPCN map calculation is missing.")
            return False

        start_time = time.time()
        try:
            logger.debug("Calculating TPCN maps using parallel Numba function...")
            # [+] Update call signature and return values
            ps_map, pc_map, hme_map, os_map = _calculate_tpcn_maps_parallel_numba(
                self._A_lin,
                self._spod_map,
                self._f0_candidates_hz,
                self._sr,
                self.n_fft,
                self.hop_length,
                self.max_harmonics,  # Use main harmonic setting
                self.inharmonicity_factor,
                self._freq_resolution_hz,
                self.tpcn_lambda,
                self.tpcn_valley_window_cents,
                self.tpcn_valley_weight_exponent,
            )
            # Store maps internally
            self._PS_lambda_map = ps_map
            self._PC_diff_map = pc_map
            self._HME_map = hme_map
            self._OS_TPCN_map = os_map
            calc_time = time.time() - start_time
            logger.debug(
                f"TPCN maps calculated. Shape: {ps_map.shape}. Time: {calc_time:.4f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Error calculating TPCN maps: {e}", exc_info=True)
            self._PS_lambda_map = None
            self._PC_diff_map = None
            self._HME_map = None
            self._OS_TPCN_map = None
            return False

    def _estimate_f0s_from_salience(self) -> List[List[float]]:  # Return type changed
        """
        Estimates multiple F0s per frame using TPCN salience (PS_lambda),
        Phase Contrast Difference (PC_diff), incorporating the PC+PS Peak Selection
        approach with NMS.

        [!] This method is rewritten based on the new approach.
        """
        if (
            self._PS_lambda_map is None
            or self._PC_diff_map is None
            or self._f0_candidates_hz is None
            or self._times is None
        ):
            logger.error("Required maps for F0 estimation are missing.")
            return [
                [] for _ in range(len(self._times) if self._times is not None else 0)
            ]

        n_f0_cands, n_frames = self._PS_lambda_map.shape
        multi_f0_output: List[List[float]] = [[] for _ in range(n_frames)]
        f0_candidates_hz_np = self._f0_candidates_hz
        f0_candidates_cents_np = hz_to_cents_numba(f0_candidates_hz_np)

        # --- Parameters ---
        salience_thresh = self.tpcn_salience_threshold
        pc_diff_thresh = self.tpcn_pc_diff_threshold
        nms_window_cents = self.tpcn_nms_window_cents

        # --- Process frame by frame ---
        for t in range(n_frames):
            # Get maps for the current frame
            ps_lambda_t = self._PS_lambda_map[:, t]
            pc_diff_t = self._PC_diff_map[:, t]

            # --- Step 1: Filter Candidates by Thresholds ---
            salience_mask = ps_lambda_t >= salience_thresh
            pc_diff_mask = pc_diff_t >= pc_diff_thresh
            combined_mask = salience_mask & pc_diff_mask

            # Indices of reliable candidates passing both thresholds
            reliable_indices = np.where(combined_mask)[0]

            if len(reliable_indices) == 0:
                multi_f0_output[t] = []
                continue  # No reliable candidates in this frame

            # --- Prepare Data for NMS ---
            reliable_f0s_hz = f0_candidates_hz_np[reliable_indices]
            reliable_f0s_cents = f0_candidates_cents_np[reliable_indices]
            # [+] Modify scoring: Use product of PS_lambda and max(0, PC_diff)
            reliable_ps_lambda = ps_lambda_t[reliable_indices]
            reliable_pc_diff = pc_diff_t[reliable_indices]
            reliable_scores = reliable_ps_lambda * np.maximum(0.0, reliable_pc_diff)

            # Sort candidates by PS_lambda score (descending)
            sort_order = np.argsort(reliable_scores)[::-1]
            sorted_indices = reliable_indices[sort_order]
            sorted_f0s_hz = reliable_f0s_hz[sort_order]
            sorted_f0s_cents = reliable_f0s_cents[sort_order]
            # Keep sorted PS_lambda values for potential interpolation later
            sorted_ps_lambda = reliable_ps_lambda[sort_order]  # Already sorted by this

            # --- Step 2: Apply Non-Maximum Suppression (NMS) ---
            suppressed = np.zeros(len(sorted_indices), dtype=bool)
            selected_f0s_t: List[float] = []

            for i in range(len(sorted_indices)):
                if suppressed[i]:
                    continue  # Skip if already suppressed

                current_idx = sorted_indices[i]  # Original index in f0_candidates array
                current_hz = sorted_f0s_hz[i]
                current_cents = sorted_f0s_cents[i]
                current_ps = sorted_ps_lambda[i]  # PS_lambda value for this peak

                # --- [+] Step 2.1: Parabolic Interpolation for Frequency Refinement ---
                # Refine frequency using parabolic interpolation on PS_lambda map around the peak index
                refined_hz = current_hz  # Default to original if interpolation fails
                if (
                    0 < current_idx < n_f0_cands - 1
                ):  # Check if index allows interpolation
                    # Pass the PS_lambda values around the peak index to the Numba interpolator
                    # Note: _parabolic_interpolation_numba expects y_values and the peak index *within* y_values
                    # We need the slice of ps_lambda_t around current_idx
                    y_slice = ps_lambda_t[current_idx - 1 : current_idx + 2]
                    # Ensure the slice is valid (length 3) and peak is in the middle
                    if len(y_slice) == 3 and np.isfinite(y_slice).all():
                        # The peak_index argument for the interpolator should be 1 (middle of the slice)
                        delta_index, interp_ps_val = _parabolic_interpolation_numba(
                            y_slice, 1
                        )

                        # Check if delta_index is valid before calculating refined frequency
                        if abs(delta_index) <= 0.5:  # Plausible refinement range
                            # Calculate refined index and convert back to Hz
                            refined_f0_index_float = current_idx + delta_index
                            # Interpolate Hz value using linear interpolation between candidate frequencies
                            # Need Hz values at current_idx-1, current_idx, current_idx+1
                            f_low = f0_candidates_hz_np[current_idx - 1]
                            f_mid = f0_candidates_hz_np[current_idx]
                            f_high = f0_candidates_hz_np[current_idx + 1]

                            # Simple linear interpolation for frequency refinement
                            # (Assumes roughly linear spacing in Hz locally, which might not be true for cents scale)
                            # Alternative: Interpolate on the cents scale?
                            # Let's try linear Hz interpolation first.
                            if delta_index < 0:  # Interpolate between f_low and f_mid
                                refined_hz = f_mid + delta_index * (
                                    f_mid - f_low
                                )  # delta_index is negative
                            else:  # Interpolate between f_mid and f_high
                                refined_hz = f_mid + delta_index * (f_high - f_mid)

                            # Clamp refined frequency to reasonable bounds around original peak
                            refined_hz = max(
                                f_low * 0.99, min(refined_hz, f_high * 1.01)
                            )
                            refined_hz = max(0.0, refined_hz)  # Ensure non-negative

                # Use the refined frequency for the selected peak
                selected_f0s_t.append(refined_hz)
                # --- End of Parabolic Interpolation ---

                # --- Step 2.2: Suppress Neighbors ---
                # Use the *original* CENTS value for suppression window calculation for consistency
                suppress_center_cents = current_cents

                # Suppress neighbors within the NMS window
                for j in range(i + 1, len(sorted_indices)):
                    if not suppressed[j]:
                        neighbor_cents = sorted_f0s_cents[
                            j
                        ]  # Use original cents for comparison
                        # Check if neighbor is within the window (absolute difference)
                        if (
                            abs(neighbor_cents - suppress_center_cents)
                            < nms_window_cents / 2.0
                        ):
                            suppressed[j] = True

            # Store the selected F0s for the current frame (sorted by frequency)
            multi_f0_output[t] = sorted(selected_f0s_t)

            # --- Step 3: Octave Relationship Check (Optional - Skipped for now) ---
            # ... implementation could go here ...

        logger.debug(
            f"F0 estimation (PC+PS) complete. Found F0s in {sum(1 for f0s in multi_f0_output if f0s)} / {n_frames} frames."
        )
        return multi_f0_output

    def _frames_to_notes_tpcn(
        self, multi_f0_per_frame: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """
        Converts frame-wise F0 estimations into discrete notes using Hungarian algorithm.
        Uses PS_lambda and OS_TPCN for onset/offset criteria (adapted logic).

        Args:
            multi_f0_per_frame: List (frames) of lists (F0s in Hz from PC+PS estimation).

        Returns:
            List of dictionaries, each representing a detected raw note.
        """
        if (
            self._times is None
            or self._sr is None
            or self._PS_lambda_map is None
            or self._OS_TPCN_map is None
            or self._f0_candidates_hz is None
        ):
            logger.error("Required data for note tracking is missing.")
            return []

        ps_map_orig = self._PS_lambda_map
        os_map_orig = self._OS_TPCN_map
        f0_cand_list = self._f0_candidates_hz  # For mapping back to indices
        n_f0_cands = len(f0_cand_list)

        # --- [+] Apply Smoothing to Maps (if enabled) ---
        ps_map_smoothed = ps_map_orig
        os_map_smoothed = os_map_orig
        smooth_frames = self.tpcn_map_smooth_frames
        smooth_cents = self.tpcn_map_smooth_cents

        if smooth_frames > 1 or smooth_cents > 0:
            logger.debug(
                f"Applying map smoothing: frames={smooth_frames}, cents={smooth_cents}..."
            )
            # Time smoothing (axis=1)
            if smooth_frames > 1:
                # Ensure odd window size for symmetry if desired, though uniform_filter doesn't require it
                time_window = smooth_frames
                ps_map_smoothed = uniform_filter1d(
                    ps_map_orig, size=time_window, axis=1, mode="nearest"
                )
                os_map_smoothed = uniform_filter1d(
                    os_map_orig, size=time_window, axis=1, mode="nearest"
                )
            else:  # Need a copy if only frequency smoothing is applied
                ps_map_smoothed = (
                    ps_map_orig.copy() if smooth_cents > 0 else ps_map_orig
                )
                os_map_smoothed = (
                    os_map_orig.copy() if smooth_cents > 0 else os_map_orig
                )

            # Frequency smoothing (axis=0) - based on cents window
            if smooth_cents > 0 and n_f0_cands > 1:
                # Convert cents window to approximate number of candidate bins
                # This assumes roughly constant cents resolution, which is true by design
                cents_per_bin = self.f0_candidate_resolution_cents
                freq_window_bins = max(1, int(round(smooth_cents / cents_per_bin)))
                # Apply smoothing along the frequency candidate axis (axis=0)
                ps_map_smoothed = uniform_filter1d(
                    ps_map_smoothed, size=freq_window_bins, axis=0, mode="nearest"
                )
                os_map_smoothed = uniform_filter1d(
                    os_map_smoothed, size=freq_window_bins, axis=0, mode="nearest"
                )
            logger.debug("Map smoothing applied.")
        # --- End Smoothing ---

        frame_times = self._times
        num_frames = len(multi_f0_per_frame)
        if len(frame_times) != num_frames:
            logger.warning(
                f"Mismatch frame times/F0 results: {len(frame_times)} vs {num_frames}. Truncating."
            )
            num_frames = min(len(frame_times), num_frames)
            if num_frames == 0:
                return []
            frame_times = frame_times[:num_frames]
            multi_f0_per_frame = multi_f0_per_frame[:num_frames]

        # --- Tracking Parameters ---
        hop_time = self.hop_length / self._sr
        max_pitch_gap_frames = max(
            1, int(np.ceil(self.tpcn_max_pitch_gap_sec / hop_time))
        )
        COST_INF = 1e6  # Cost for impossible assignments
        # [+] Weights for cost function
        W_PITCH = 1.0  # Weight for pitch difference cost
        W_SALIENCE = 0.5  # Weight for salience difference cost (Needs tuning)

        # --- Tracking State ---
        notes_list: List[Dict[str, Any]] = []
        active_notes: Dict[int, Dict[str, Any]] = {}
        track_id_counter = 0
        stats = defaultdict(int)  # Use defaultdict for cleaner stat tracking

        # --- Main Tracking Loop ---
        for frame_idx in range(num_frames):
            current_time = frame_times[frame_idx]
            current_f0s_hz = sorted([f for f in multi_f0_per_frame[frame_idx] if f > 0])
            num_current_f0s = len(current_f0s_hz)
            current_f0s_cents_np = hz_to_cents_numba(
                np.array(current_f0s_hz, dtype=np.float64)
            )

            # --- Prepare Active Tracks Data ---
            active_track_ids = list(active_notes.keys())
            num_active_tracks = len(active_track_ids)
            active_track_last_pitches_cents = np.zeros(
                num_active_tracks, dtype=np.float64
            )
            # [+] Store last known salience for cost calculation
            active_track_last_saliences = np.zeros(num_active_tracks, dtype=np.float64)
            if num_active_tracks > 0:
                for r, track_id in enumerate(active_track_ids):
                    pitch_hist_cents = active_notes[track_id].get("pitches_cents", [])
                    active_track_last_pitches_cents[r] = (
                        pitch_hist_cents[-1] if pitch_hist_cents else -np.inf
                    )
                    # [+] Get last salience
                    salience_hist = active_notes[track_id].get("saliences", [])
                    active_track_last_saliences[r] = (
                        salience_hist[-1] if salience_hist else 0.0
                    )

            # --- Cost Matrix Calculation (Pitch diff + Salience diff) ---
            cost_matrix = np.full(
                (num_active_tracks, num_current_f0s), COST_INF, dtype=np.float64
            )
            if num_active_tracks > 0 and num_current_f0s > 0:
                for r in range(num_active_tracks):
                    last_pitch_cents = active_track_last_pitches_cents[r]
                    last_salience = active_track_last_saliences[r]
                    if not np.isfinite(last_pitch_cents):
                        continue  # Skip inactive tracks

                    for c in range(num_current_f0s):
                        curr_pitch_cents = current_f0s_cents_np[c]
                        if not np.isfinite(curr_pitch_cents):
                            continue  # Skip invalid F0s

                        # --- Calculate Pitch Cost ---
                        pitch_diff = abs(curr_pitch_cents - last_pitch_cents)
                        pitch_cost = (
                            pitch_diff
                            if pitch_diff < self.tpcn_pitch_continuity_tolerance_cents
                            else COST_INF
                        )

                        # --- Calculate Salience Cost ---
                        salience_cost = COST_INF
                        if (
                            pitch_cost < COST_INF
                        ):  # Only calculate salience cost if pitch is plausible
                            # Find current salience value (needs f0_map_idx)
                            curr_f0_hz = current_f0s_hz[c]
                            diffs_hz = np.abs(f0_cand_list - curr_f0_hz)
                            curr_f0_map_idx = (
                                np.argmin(diffs_hz) if len(diffs_hz) > 0 else -1
                            )

                            current_salience = 0.0
                            if 0 <= curr_f0_map_idx < n_f0_cands:
                                current_salience = ps_map_smoothed[
                                    curr_f0_map_idx, frame_idx
                                ]

                            # Salience difference cost (absolute difference for now)
                            salience_diff = abs(current_salience - last_salience)
                            # Normalize or scale salience diff? Let's use absolute diff for now.
                            salience_cost = salience_diff
                        else:  # Pitch cost is infinite, make salience cost also infinite
                            salience_cost = COST_INF

                        # --- Combine Costs ---
                        # Ensure individual costs are finite before combining
                        if pitch_cost < COST_INF and salience_cost < COST_INF:
                            combined_cost = (
                                W_PITCH * pitch_cost + W_SALIENCE * salience_cost
                            )
                            cost_matrix[r, c] = combined_cost
                        # else: cost remains COST_INF

            # --- Solve Assignment Problem ---
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
                    assigned_rows = row_ind[valid_assignment_mask].astype(int)
                    assigned_cols = col_ind[valid_assignment_mask].astype(int)
                except ValueError as e:
                    logger.warning(f"Frame {frame_idx}: Hungarian failed: {e}")
                    assigned_rows, assigned_cols = np.array([]), np.array([])

                # --- Update Matched Tracks ---
                for r, c in zip(assigned_rows, assigned_cols):
                    track_id = active_track_ids[r]
                    f0_hz = current_f0s_hz[c]
                    f0_cents = current_f0s_cents_np[c]
                    note_info = active_notes[track_id]
                    note_info["pitches_hz"].append(f0_hz)
                    note_info["pitches_cents"].append(f0_cents)
                    note_info["last_update_frame"] = frame_idx
                    matched_track_indices.add(r)
                    matched_f0_indices.add(c)

            # --- Handle Unmatched Active Tracks (Potential Note Ends) ---
            ended_track_ids = []
            for r, track_id in enumerate(active_track_ids):
                if r not in matched_track_indices:
                    note_info = active_notes[track_id]
                    frames_since_update = frame_idx - note_info["last_update_frame"]

                    # --- Find index for last known pitch ---
                    last_pitch_hz = (
                        note_info["pitches_hz"][-1] if note_info["pitches_hz"] else 0.0
                    )
                    f0_map_idx = -1
                    if last_pitch_hz > 0:
                        diffs = np.abs(f0_cand_list - last_pitch_hz)
                        if len(diffs) > 0:
                            f0_map_idx = np.argmin(diffs)

                    # --- Check PS_lambda condition ---
                    ps_lambda_low = False
                    current_ps_lambda = 0.0
                    if 0 <= f0_map_idx < n_f0_cands:
                        current_ps_lambda = ps_map_smoothed[f0_map_idx, frame_idx]
                        ps_lambda_low = current_ps_lambda < self.tpcn_salience_threshold

                    # --- Determine End Condition ---
                    timed_out = frames_since_update >= max_pitch_gap_frames
                    should_end_note = timed_out or ps_lambda_low

                    if should_end_note:
                        ended_track_ids.append(track_id)
                        if timed_out:
                            stats["offset_timeout"] += 1
                        if ps_lambda_low:
                            stats["offset_salience_low"] += 1

                        # Finalize Note Data
                        start_time = note_info["start_time"]
                        last_update_idx = note_info["last_update_frame"]
                        offset_time = (
                            frame_times[last_update_idx] + hop_time
                            if 0 <= last_update_idx < len(frame_times)
                            else current_time
                        )
                        duration = offset_time - start_time
                        pitches_hz = note_info["pitches_hz"]

                        if duration >= self.tpcn_min_note_duration_sec and pitches_hz:
                            pitch_median = np.median(pitches_hz)
                            notes_list.append(
                                {
                                    "start_time": start_time,
                                    "offset_time": offset_time,
                                    "pitch_hz": pitch_median,
                                    "duration": duration,
                                    "pitches_hz": pitches_hz,
                                }
                            )

            # Remove ended tracks
            for track_id in ended_track_ids:
                if track_id in active_notes:
                    del active_notes[track_id]

            # --- Start New Tracks for Unmatched F0s ---
            for c, f0_hz in enumerate(current_f0s_hz):
                if c not in matched_f0_indices:
                    # Find corresponding F0 candidate index
                    f0_map_idx = -1
                    diffs = np.abs(f0_cand_list - f0_hz)
                    if len(diffs) > 0:
                        f0_map_idx = np.argmin(diffs)

                    # Check Onset Score Condition
                    onset_score_ok = False
                    current_os_tpcn = 0.0
                    if 0 <= f0_map_idx < n_f0_cands:
                        current_os_tpcn = os_map_smoothed[f0_map_idx, frame_idx]
                        onset_score_ok = (
                            current_os_tpcn >= self.tpcn_onset_score_threshold
                        )

                    # Check Salience Condition (already known from estimation)
                    salience_ok = False
                    current_ps_lambda = 0.0
                    if 0 <= f0_map_idx < n_f0_cands:
                        current_ps_lambda = ps_map_smoothed[f0_map_idx, frame_idx]
                        # We know f0_hz was selected, so PS_lambda was >= threshold during estimation
                        # Re-check just in case? Or trust the input F0 list? Let's trust input.
                        salience_ok = (
                            True  # Assumed True if f0_hz is in multi_f0_per_frame
                        )

                    # --- Start Note if Conditions Met ---
                    if onset_score_ok and salience_ok:
                        f0_cents = current_f0s_cents_np[c]
                        active_notes[track_id_counter] = {
                            "start_time": current_time,  # Start at beginning of current frame
                            "pitches_hz": [f0_hz],
                            "pitches_cents": [f0_cents],
                            "last_update_frame": frame_idx,
                        }
                        stats["onset_started"] += 1
                        track_id_counter += 1
                    else:
                        if not onset_score_ok:
                            stats["onset_suppressed_os_low"] += 1
                        # We don't log salience low suppression here as it was handled during F0 estimation

        # --- Process Remaining Active Notes at the End ---
        if frame_times is not None:
            num_frames_final = len(frame_times)
            for track_id, note_info in active_notes.items():
                start_time = note_info["start_time"]
                last_update_idx = note_info["last_update_frame"]
                offset_time = (
                    frame_times[last_update_idx] + hop_time
                    if 0 <= last_update_idx < num_frames_final
                    else frame_times[-1] + hop_time
                )
                duration = offset_time - start_time
                pitches_hz = note_info["pitches_hz"]
                if duration >= self.tpcn_min_note_duration_sec and pitches_hz:
                    pitch_median = np.median(pitches_hz)
                    notes_list.append(
                        {
                            "start_time": start_time,
                            "offset_time": offset_time,
                            "pitch_hz": pitch_median,
                            "duration": duration,
                            "pitches_hz": pitches_hz,
                        }
                    )

        logger.info(
            f"Note Tracking Stats: Onsets Started:{stats['onset_started']} | Onsets Suppressed [OS Low:{stats['onset_suppressed_os_low']}] | Offsets Advanced [Timeout:{stats['offset_timeout']}, Salience Low:{stats['offset_salience_low']}]"
        )
        return notes_list

    def _postprocess_notes(
        self, notes_list: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Post-processes raw notes: filter, merge, resolve overlaps, stability filter."""
        if not notes_list:
            return np.array([]).reshape(0, 2), np.array([])
        logger.debug(
            f"[_postprocess_notes_tpcn] Start: Received {len(notes_list)} raw notes."
        )

        # --- Initial Filtering & Data Structure ---
        notes = []
        for n in notes_list:
            if n.get("pitch_hz", 0.0) > 0 and n.get("duration", 0.0) > 1e-6:
                notes.append(
                    {
                        k: v
                        for k, v in n.items()
                        if k
                        in [
                            "start_time",
                            "offset_time",
                            "pitch_hz",
                            "duration",
                            "pitches_hz",
                        ]
                    }
                )
                notes[-1]["onset"] = notes[-1].pop(
                    "start_time"
                )  # Rename for consistency
                notes[-1]["offset"] = notes[-1].pop("offset_time")
                notes[-1]["pitch"] = notes[-1].pop("pitch_hz")

        logger.debug(
            f"[_postprocess_notes_tpcn] After initial filter: {len(notes)} notes."
        )
        if not notes:
            return np.array([]).reshape(0, 2), np.array([])
        notes.sort(key=lambda n: n["onset"])

        # --- Stage 1: Merge Consecutive Notes ---
        merged_notes: List[Dict[str, Any]] = []
        if notes:
            current_note = notes[0].copy()
            current_note_pitch_cents_arr = hz_to_cents_numba(
                np.array([current_note["pitch"]])
            )
            current_note_pitch_cents = (
                current_note_pitch_cents_arr[0]
                if len(current_note_pitch_cents_arr) > 0
                else -np.inf
            )

            for i in range(1, len(notes)):
                next_note = notes[i]
                gap = next_note["onset"] - current_note["offset"]
                next_note_pitch_cents_arr = hz_to_cents_numba(
                    np.array([next_note["pitch"]])
                )
                next_note_pitch_cents = (
                    next_note_pitch_cents_arr[0]
                    if len(next_note_pitch_cents_arr) > 0
                    else -np.inf
                )
                pitch_diff_cents = (
                    abs(current_note_pitch_cents - next_note_pitch_cents)
                    if np.isfinite(current_note_pitch_cents)
                    and np.isfinite(next_note_pitch_cents)
                    else np.inf
                )

                # Merge Condition: Small gap AND similar pitch
                if (
                    0 <= gap <= self.tpcn_min_silence_duration_sec
                    and pitch_diff_cents <= self.tpcn_note_merge_tolerance_cents
                ):
                    duration1 = current_note["duration"]
                    duration2 = next_note["duration"]
                    total_duration = duration1 + duration2
                    if total_duration > 1e-9:
                        current_note["pitch"] = (
                            current_note["pitch"] * duration1
                            + next_note["pitch"] * duration2
                        ) / total_duration
                    else:
                        current_note["pitch"] = (
                            current_note["pitch"] + next_note["pitch"]
                        ) / 2.0
                    current_note["offset"] = next_note["offset"]
                    current_note["duration"] = (
                        current_note["offset"] - current_note["onset"]
                    )
                    if "pitches_hz" not in current_note:
                        current_note["pitches_hz"] = []
                    current_note["pitches_hz"].extend(next_note.get("pitches_hz", []))
                    current_note_pitch_cents_arr = hz_to_cents_numba(
                        np.array([current_note["pitch"]])
                    )
                    current_note_pitch_cents = (
                        current_note_pitch_cents_arr[0]
                        if len(current_note_pitch_cents_arr) > 0
                        else -np.inf
                    )
                else:
                    merged_notes.append(current_note)
                    current_note = next_note.copy()
                    current_note_pitch_cents_arr = hz_to_cents_numba(
                        np.array([current_note["pitch"]])
                    )
                    current_note_pitch_cents = (
                        current_note_pitch_cents_arr[0]
                        if len(current_note_pitch_cents_arr) > 0
                        else -np.inf
                    )
            merged_notes.append(current_note)
            notes = merged_notes
            logger.debug(
                f"[_postprocess_notes_tpcn] After merging consecutive: {len(notes)} notes."
            )

        # --- Stage 2: Handle Overlaps (Merge similar pitch overlaps) ---
        processed_notes_overlap: List[Dict[str, Any]] = []
        if notes:
            notes.sort(key=lambda n: n["onset"])
            i = 0
            while i < len(notes):
                current_note = notes[i].copy()
                # Check subsequent notes for overlap
                j = i + 1
                while j < len(notes) and notes[j]["onset"] < current_note["offset"]:
                    other_note = notes[j]
                    pitch1_cents_arr = hz_to_cents_numba(
                        np.array([current_note["pitch"]])
                    )
                    pitch1_cents = (
                        pitch1_cents_arr[0] if len(pitch1_cents_arr) > 0 else -np.inf
                    )
                    pitch2_cents_arr = hz_to_cents_numba(
                        np.array([other_note["pitch"]])
                    )
                    pitch2_cents = (
                        pitch2_cents_arr[0] if len(pitch2_cents_arr) > 0 else -np.inf
                    )
                    pitch_diff_cents = (
                        abs(pitch1_cents - pitch2_cents)
                        if np.isfinite(pitch1_cents) and np.isfinite(pitch2_cents)
                        else np.inf
                    )

                    if pitch_diff_cents <= self.tpcn_note_merge_tolerance_cents:
                        dur1 = current_note["duration"]
                        dur2 = other_note["duration"]
                        total_dur = dur1 + dur2
                        if total_dur > 1e-9:
                            current_note["pitch"] = (
                                current_note["pitch"] * dur1
                                + other_note["pitch"] * dur2
                            ) / total_dur
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
                        if "pitches_hz" not in current_note:
                            current_note["pitches_hz"] = []
                        current_note["pitches_hz"].extend(
                            other_note.get("pitches_hz", [])
                        )
                        notes.pop(j)  # Remove merged note
                        continue  # Re-check overlaps with the expanded current_note
                    else:
                        j += (
                            1  # Different pitch overlap, move to next potential overlap
                        )
                processed_notes_overlap.append(current_note)
                i += 1
            notes = processed_notes_overlap
            logger.debug(
                f"[_postprocess_notes_tpcn] After overlap resolution: {len(notes)} notes."
            )

        # --- Stage 3: Pitch Stability Filter ---
        stable_notes: List[Dict[str, Any]] = []
        removed_unstable_count = 0
        for note in notes:
            pitches_hz = np.array(note.get("pitches_hz", []))
            if len(pitches_hz) >= 2:
                pitches_cents = hz_to_cents_numba(pitches_hz)
                valid_cents = pitches_cents[np.isfinite(pitches_cents)]
                if len(valid_cents) >= 2:
                    pitch_std_dev_cents = np.std(valid_cents)
                    if pitch_std_dev_cents <= self.tpcn_max_pitch_std_dev_cents:
                        stable_notes.append(note)
                    else:
                        removed_unstable_count += 1
                else:
                    stable_notes.append(note)
            else:
                stable_notes.append(note)
        if removed_unstable_count > 0:
            logger.debug(
                f"[_postprocess_notes_tpcn] Removed {removed_unstable_count} unstable notes."
            )
        notes = stable_notes
        logger.debug(
            f"[_postprocess_notes_tpcn] After pitch stability filter: {len(notes)} notes."
        )

        # --- Stage 4: Final Duration Filter ---
        final_notes: List[Dict[str, Any]] = []
        removed_short_count = 0
        for note in notes:
            note["duration"] = (
                note["offset"] - note["onset"]
            )  # Recalculate duration after merging
            if note["duration"] >= self.tpcn_min_note_duration_sec:
                pitches_hz_final = np.array(
                    note.get("pitches_hz", []), dtype=np.float64
                )
                if len(pitches_hz_final) > 0:
                    note["final_pitch_hz"] = np.median(pitches_hz_final)
                    final_notes.append(note)
                else:
                    removed_short_count += 1
            else:
                removed_short_count += 1
        if removed_short_count > 0:
            logger.debug(
                f"[_postprocess_notes_tpcn] Removed {removed_short_count} short notes in final filter."
            )
        logger.debug(
            f"[_postprocess_notes_tpcn] Final notes count: {len(final_notes)}."
        )

        # --- Format Output ---
        if not final_notes:
            return np.array([]).reshape(0, 2), np.array([])
        intervals = np.array([[n["onset"], n["offset"]] for n in final_notes])
        pitches = np.array([n["final_pitch_hz"] for n in final_notes])
        return intervals, pitches

    def detect(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Main detection pipeline for TPCN multi-pitch estimation.
        """
        start_process_time = time.time()
        logger.info(f"Starting detection with {self.__class__.__name__} (SR={sr})...")

        # --- Input Validation ---
        if (
            not isinstance(audio_data, np.ndarray)
            or audio_data.ndim != 1
            or audio_data.size == 0
        ):
            logger.error("Invalid or empty audio data.")
            return self._empty_result("Invalid or empty audio data")
        if sr <= 0:
            logger.error(f"Invalid sample rate: {sr}.")
            return self._empty_result(f"Invalid sample rate: {sr}")

        try:
            # === Stage 1: Initialization and Preprocessing ===
            logger.debug("Initializing run...")
            self._initialize_run(sr)

            logger.debug("Calculating STFT...")
            if (
                not self._calculate_stft(audio_data)
                or self._A_lin is None
                or self._times is None
            ):
                raise ValueError("STFT calculation failed or produced empty results.")
            n_bins, n_frames = self._A_lin.shape
            logger.debug(f"STFT calculated: {n_bins} bins, {n_frames} frames.")

            logger.debug("Calculating SPOD Map...")
            if not self._calculate_spod(audio_data) or self._spod_map is None:
                # Attempt SPOD calculation again (might be redundant if STFT includes phase)
                # If it fails, proceed without SPOD? TPCN relies heavily on it.
                logger.warning(
                    "SPOD map calculation failed. TPCN results might be inaccurate."
                )
                # Initialize SPOD map to zeros or ones? Zeros might be safer.
                self._spod_map = np.zeros_like(self._A_lin)

            # === Stage 2: TPCN Map Calculation ===
            logger.debug("Calculating TPCN Maps (PS_lambda, PC_diff, HME, OS_TPCN)...")
            # [+] Update call to receive HME_map
            if not self._calculate_tpcn_maps():  # Use the wrapper
                raise ValueError("TPCN map calculation failed.")
            logger.debug("TPCN maps calculated.")

            # === Stage 3: F0 Estimation (using new approach) ===
            logger.debug("Estimating frame-wise F0s from TPCN maps...")
            # [+] Pass required maps to the (yet to be implemented) new estimation function
            multi_f0_per_frame = (
                self._estimate_f0s_from_salience()
            )  # Rewritten function call
            num_f0_frames = sum(1 for f0s in multi_f0_per_frame if f0s)
            logger.debug(f"F0s estimated for {num_f0_frames} / {n_frames} frames.")

            # === Stage 4: Note Tracking and Segmentation ===
            logger.debug("Performing note tracking and segmentation...")
            # Pass F0s estimated by the new method
            raw_notes_list = self._frames_to_notes_tpcn(
                multi_f0_per_frame=multi_f0_per_frame
            )

            # === Stage 5: Post-processing Notes ===
            logger.debug("Post-processing detected notes...")
            intervals, note_pitches = self._postprocess_notes(raw_notes_list)
            final_note_count = len(note_pitches)

            # === Finalization ===
            detection_time = time.time() - start_process_time
            logger.info(
                f"Detection complete in {detection_time:.4f} seconds. Found {final_note_count} notes."
            )

            # Format Results
            frame_frequencies = np.zeros(n_frames, dtype=float)
            for idx, f0s in enumerate(multi_f0_per_frame):
                positive_f0s = [f for f in f0s if f > 0]
                frame_frequencies[idx] = min(positive_f0s) if positive_f0s else 0.0
            params = self.get_params()

            result = {
                "intervals": intervals,
                "note_pitches": note_pitches,
                "frame_times": self._times if self._times is not None else np.array([]),
                "frame_frequencies": frame_frequencies,
                "detector_name": self.__class__.__name__,
                "detection_time": detection_time,
                "additional_data": {
                    "note_count": final_note_count,
                    "parameters": params,
                },
            }
            return result

        except Exception as e:
            logger.error(f"Error during detection pipeline: {e}", exc_info=True)
            return self._empty_result(error_message=str(e))
        finally:
            self._cleanup_run_state()

    def _cleanup_run_state(self):
        """Clears internal state variables after a run."""
        self._sr = None
        self._freqs = None
        self._times = None
        self._f0_candidates_hz = None
        self._A_lin = None
        self._spod_map = None
        self._PS_lambda_map = None
        self._PC_diff_map = None
        self._HME_map = None
        self._OS_TPCN_map = None
        logger.debug("Internal state cleared.")

    def _empty_result(self, error_message: str = "Detection failed") -> Dict[str, Any]:
        """Returns a standardized empty result dictionary."""
        params = {}
        try:
            params = self.get_params()
        except Exception:
            pass
        return {
            "intervals": np.array([]).reshape(0, 2),
            "note_pitches": np.array([]),
            "frame_times": np.array([]),
            "frame_frequencies": np.array([]),
            "detector_name": self.__class__.__name__ + "_Error",
            "detection_time": 0.0,
            "additional_data": {
                "error": error_message,
                "note_count": 0,
                "parameters": params,
            },
        }


# --- End of TPCNDetector Class ---
