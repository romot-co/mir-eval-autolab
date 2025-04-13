# -*- coding: utf-8 -*-
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import librosa
import numba
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.optimize import linear_sum_assignment  # For Hungarian algorithm
from scipy.signal import find_peaks

from src.detectors.base_detector import BaseDetector
from src.detectors.pzstd_detector import (
    _calculate_harmonic_freqs_numba,  # May not be needed directly, but keep for now
)
from src.detectors.pzstd_detector import (  # Import helper functions
    _parabolic_interpolation_numba,
    cents_to_hz_numba,
    hz_to_cents_numba,
)

logger = logging.getLogger(__name__)

# --- Constants for Partial Tracking (Example) ---
COST_INF = 1e6  # Cost for infeasible assignments

# --- Unpredictability Based Detector Class ---


class UnpredictabilityBasedDetector(BaseDetector):
    """
    Multi-pitch and note detector based on integrated unpredictability measures.

    This detector utilizes the concept that musical signals are partially
    predictable in the short term. Deviations from predictions (unpredictability)
    across partials are aggregated to detect onsets and influence note tracking.
    A single meta-parameter 'beta' controls the sensitivity and time scale.
    """

    # --- Fixed Parameters (Examples, similar to PZSTD) ---
    N_FFT = 2048
    HOP_LENGTH = 1024
    WINDOW = "hann"
    MIN_FREQ_HZ = 30.0
    MAX_FREQ_HZ = 5000.0
    ENERGY_DB_THRESHOLD = -50.0  # For initial peak filtering
    PEAK_DISTANCE_BIN = 4  # For find_peaks

    # --- Default Tunable Meta-Parameter ---
    DEFAULT_BETA = 0.5  # Example: Controls sensitivity (0=low, 1=high)

    # --- Parameters related to partial tracking/prediction (Could be fixed or derived from beta) ---
    PARTIAL_MAX_FREQ_HZ = 5500.0  # Max freq for tracking partials
    PARTIAL_MAX_INACTIVE_FRAMES = (
        3  # Max frames a partial can be missing before termination
    )
    PARTIAL_MATCH_MAX_COST = (
        50.0  # Max cost (e.g., cents difference) for matching peaks to partials
    )

    def __init__(self, beta: float = DEFAULT_BETA, **kwargs):
        """
        Initializes the detector with the beta sensitivity parameter.
        """
        super().__init__(**kwargs)

        # --- Store Meta-Parameter ---
        self.beta = max(0.0, min(1.0, beta))  # Clip beta to [0, 1]

        # --- Calculate Beta-Dependent Internal Parameters ---
        # These functions will determine thresholds, smoothing based on beta
        self._calculate_internal_params()

        # --- Internal State Variables ---
        self._sr: Optional[int] = None
        self._times: Optional[np.ndarray] = None
        self._freqs: Optional[np.ndarray] = None
        self._freq_resolution_hz: Optional[float] = None
        self._hop_time: Optional[float] = None

        # --- State for Tracking ---
        self._active_partials: Dict[int, Dict[str, Any]] = (
            {}
        )  # track_id -> partial info
        self._partial_id_counter: int = 0

        logger.info(
            f"UnpredictabilityBasedDetector initialized with beta={self.beta:.3f}"
        )
        logger.info(f"  -> Onset Sensitivity: {self.onset_sensitivity_level}")
        logger.info(f"  -> Smoothing Frames: {self.smoothing_frames}")
        logger.info(
            f"  -> Relative Onset Threshold: {self.relative_onset_threshold:.3f}"
        )

    def _calculate_internal_params(self):
        """Calculates internal thresholds and smoothing based on beta."""
        # Example logic: Map beta (0-1) to different parameter ranges
        # Higher beta -> more sensitive -> shorter smoothing, lower thresholds

        # 1. Onset Smoothing Time (shorter for high beta)
        # Map beta=0 to long smoothing (e.g., 50ms), beta=1 to short (e.g., 10ms)
        min_smooth_sec = 0.010
        max_smooth_sec = 0.050
        smoothing_sec = max_smooth_sec - (max_smooth_sec - min_smooth_sec) * self.beta
        # Convert to frames (ensure sr and hop_length are available or use defaults for now)
        # Use default SR/Hop if not initialized yet, will be recalculated in _initialize_run
        sr_approx = 44100
        hop_approx = self.HOP_LENGTH
        if hasattr(self, "_sr") and self._sr is not None and self._sr > 0:
            sr_approx = self._sr
        if hasattr(self, "HOP_LENGTH") and self.HOP_LENGTH > 0:
            hop_approx = self.HOP_LENGTH

        self.smoothing_frames = max(
            1, int(np.ceil(smoothing_sec * sr_approx / hop_approx))
        )

        # 2. Relative Onset Threshold (lower for high beta)
        # Map beta=0 to high threshold (e.g., 0.3 relative), beta=1 to low (e.g., 0.05 relative)
        min_thresh_rel = 0.05
        max_thresh_rel = 0.30
        self.relative_onset_threshold = (
            max_thresh_rel - (max_thresh_rel - min_thresh_rel) * self.beta
        )

        # 3. Onset Sensitivity Level (descriptive)
        if self.beta > 0.75:
            self.onset_sensitivity_level = "High"
        elif self.beta > 0.4:
            self.onset_sensitivity_level = "Medium"
        else:
            self.onset_sensitivity_level = "Low"

        # Add more mappings as needed (e.g., for partial tracking costs, error thresholds)
        # Example: Partial matching cost threshold could be higher for lower beta
        self.partial_match_cost_threshold = self.PARTIAL_MATCH_MAX_COST * (
            1.5 - self.beta
        )

    def get_params(self) -> Dict[str, Any]:
        """Returns the detector's parameters, including the meta-parameter beta."""
        params = {
            # Meta-parameter
            "beta": self.beta,
            # Fixed parameters
            "n_fft": self.N_FFT,
            "hop_length": self.HOP_LENGTH,
            "window": self.WINDOW,
            "min_freq_hz": self.MIN_FREQ_HZ,
            "max_freq_hz": self.MAX_FREQ_HZ,
            "energy_db_threshold": self.ENERGY_DB_THRESHOLD,
            "peak_distance_bin": self.PEAK_DISTANCE_BIN,
            # Derived/Internal (for info)
            "smoothing_frames": self.smoothing_frames,
            "relative_onset_threshold": self.relative_onset_threshold,
            "partial_match_cost_threshold": self.partial_match_cost_threshold,
            "partial_max_inactive_frames": self.PARTIAL_MAX_INACTIVE_FRAMES,
        }
        return params

    def _initialize_run(self, sr: int):
        """Initializes internal state based on sample rate."""
        self._sr = sr
        self._times = None  # Will be set by STFT
        self._freqs = librosa.fft_frequencies(sr=self._sr, n_fft=self.N_FFT)
        self._freq_resolution_hz = (
            self._freqs[1] - self._freqs[0] if len(self._freqs) > 1 else sr / self.N_FFT
        )
        self._hop_time = self.HOP_LENGTH / self._sr if self._sr > 0 else None

        self._active_partials = {}
        self._partial_id_counter = 0

        # Recalculate beta-dependent params now that SR is known
        self._calculate_internal_params()

    # --- Core Processing Steps (Adapted from PZSTDDetector) ---

    def _calculate_stft(
        self, audio_data: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculates STFT. Returns S_amp, S_db, phases or Nones on error."""
        if self._sr is None:
            logger.error("SR not initialized.")
            return None, None, None
        try:
            S_complex = librosa.stft(
                audio_data,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                window=self.WINDOW,
                center=True,
            )
            S_amp = np.abs(S_complex)
            # Calculate dB relative to max amplitude
            ref_amp = np.max(S_amp) if S_amp.size > 0 else 1.0
            S_db = librosa.amplitude_to_db(
                S_amp, ref=ref_amp
            )  # Use calculated max as ref
            phases = np.angle(S_complex)
            self._times = librosa.frames_to_time(
                np.arange(S_complex.shape[1]), sr=self._sr, hop_length=self.HOP_LENGTH
            )
            return S_amp, S_db, phases
        except Exception as e:
            logger.error(f"Error calculating STFT: {e}", exc_info=True)
            self._times = np.array([])
            return None, None, None

    def _calculate_spod(self, phases: np.ndarray) -> Optional[np.ndarray]:
        """Calculates the SPOD map using Gaussian filter (fixed alpha for now)."""
        # TODO: Integrate SPOD alpha with beta? For now, keep fixed.
        SPOD_ALPHA_FIXED = 1.1  # From PZSTD
        SPOD_WINDOW_SEC_FIXED = 0.07  # From PZSTD

        n_bins, n_frames = phases.shape
        if (
            n_frames <= 1
            or self._sr is None
            or self._hop_time is None
            or self._hop_time <= 0
        ):
            logger.warning(
                "Cannot calculate SPOD: insufficient frames or invalid timebase."
            )
            return np.zeros_like(phases) if phases is not None else None

        spod_window_frames = max(
            1, int(np.ceil(SPOD_WINDOW_SEC_FIXED / self._hop_time))
        )

        phi_unwrapped = np.unwrap(phases, axis=1)
        # Pad last frame's diff to keep shape
        inst_freq = np.diff(phi_unwrapped, n=1, axis=1, append=phi_unwrapped[:, -1:])

        try:
            sigma = spod_window_frames / 4.0  # Gaussian sigma heuristic
            kwargs = {"sigma": sigma, "axis": 1, "mode": "nearest"}
            inst_freq_sq_mean = gaussian_filter1d(inst_freq**2, **kwargs)
            inst_freq_mean = gaussian_filter1d(inst_freq, **kwargs)
            inst_freq_var = np.maximum(0, inst_freq_sq_mean - inst_freq_mean**2)
        except Exception as e:
            logger.error(
                f"Error calculating moving variance for SPOD: {e}", exc_info=True
            )
            return None

        spod_map = np.exp(-SPOD_ALPHA_FIXED * inst_freq_var)
        return np.clip(spod_map, 0.0, 1.0)

    def _detect_peaks_per_frame(
        self, A_lin: np.ndarray, S_db: np.ndarray
    ) -> List[List[Tuple[float, float, float]]]:
        """
        Detects spectral peaks using Numba-optimized interpolation.
        Returns List[frame_idx] -> List[Tuple[freq_hz, amplitude, phase(placeholder)]]
        Phase is currently placeholder 0.0, SPOD will be added later.
        """
        if self._freqs is None:
            return [[] for _ in range(A_lin.shape[1])]
        n_bins, n_frames = A_lin.shape
        peaks_per_frame: List[List[Tuple[float, float, float]]] = [
            [] for _ in range(n_frames)
        ]

        min_bin_idx = np.searchsorted(self._freqs, self.MIN_FREQ_HZ, side="left")
        max_bin_idx = (
            np.searchsorted(self._freqs, self.PARTIAL_MAX_FREQ_HZ, side="right") - 1
        )  # Use partial max freq

        if min_bin_idx >= max_bin_idx:  # Use >= to handle edge case
            logger.warning(
                f"Invalid frequency range for peak detection: min_bin={min_bin_idx}, max_bin={max_bin_idx}"
            )
            return peaks_per_frame

        min_height_db = self.ENERGY_DB_THRESHOLD

        for t in range(n_frames):
            amp_t_full = A_lin[:, t]
            # Ensure slice indices are valid
            safe_min_bin = max(0, min_bin_idx)
            safe_max_bin = min(n_bins - 1, max_bin_idx)
            if safe_min_bin > safe_max_bin:
                continue  # Skip frame if range is invalid after safety check

            db_t_slice = S_db[safe_min_bin : safe_max_bin + 1, t]

            if db_t_slice.size == 0:
                continue

            # --- Peak Finding using dB for height threshold ---
            try:
                peak_indices_rel, properties = find_peaks(
                    db_t_slice,
                    distance=self.PEAK_DISTANCE_BIN,
                    height=min_height_db,
                    # TODO: Add prominence check? Could be beta-dependent.
                )
                peak_indices_abs = (
                    safe_min_bin + peak_indices_rel
                )  # Convert back to absolute indices

                frame_peaks_data: List[Tuple[float, float, float]] = []
                for p_idx_abs in peak_indices_abs:
                    # Ensure index is valid for interpolation
                    if p_idx_abs <= 0 or p_idx_abs >= n_bins - 1:
                        continue

                    # --- Parabolic Interpolation on Amplitude Spectrum ---
                    delta_bin, interp_amp = _parabolic_interpolation_numba(
                        amp_t_full, p_idx_abs
                    )
                    interp_bin = p_idx_abs + delta_bin
                    interp_freq = self._bin_to_freq(interp_bin)

                    if (
                        interp_freq is not None
                        and self.MIN_FREQ_HZ <= interp_freq <= self.PARTIAL_MAX_FREQ_HZ
                    ):
                        # Store: Freq, Interp Amplitude, Placeholder Phase=0.0
                        frame_peaks_data.append((interp_freq, interp_amp, 0.0))

                # Sort by frequency for consistency
                frame_peaks_data.sort(key=lambda x: x[0])
                peaks_per_frame[t] = frame_peaks_data

            except Exception as e:
                logger.error(
                    f"Frame {t}: Error during peak detection: {e}", exc_info=False
                )

        return peaks_per_frame

    def _bin_to_freq(self, bin_index: float) -> Optional[float]:
        """Converts a (potentially fractional) bin index to frequency (Hz)."""
        if (
            self._freqs is None
            or self._freq_resolution_hz is None
            or self._freq_resolution_hz <= 0
        ):
            return None
        # Clip result to the valid FFT frequency range
        min_fft_freq = self._freqs[0]
        max_fft_freq = self._freqs[-1]
        return np.clip(bin_index * self._freq_resolution_hz, min_fft_freq, max_fft_freq)

    # --- Partial Tracking (Basic Implementation) ---

    def _track_partials(
        self,
        peaks_per_frame: List[List[Tuple[float, float, float]]],
        phases_per_frame: Optional[np.ndarray],  # Full phase matrix [bins, frames]
        spod_map_per_frame: Optional[np.ndarray],  # Full SPOD matrix [bins, frames]
    ) -> List[Dict[int, Dict[str, Any]]]:
        """
        Tracks spectral peaks across frames to form partials using Hungarian algorithm.
        Returns a list where each element is the state of active_partials *at that frame*.
        """
        if self._freqs is None or self._hop_time is None or self._times is None:
            logger.error("Initialization incomplete for partial tracking.")
            return []

        num_frames = len(peaks_per_frame)
        all_frame_partials_state: List[Dict[int, Dict[str, Any]]] = []

        for frame_idx in range(num_frames):
            current_peaks = peaks_per_frame[frame_idx]
            num_current_peaks = len(current_peaks)

            active_track_ids = list(self._active_partials.keys())
            num_active_tracks = len(active_track_ids)

            cost_matrix = np.full(
                (num_active_tracks, num_current_peaks), COST_INF, dtype=np.float64
            )

            # --- Calculate Cost Matrix ---
            # Cost based primarily on frequency difference (in cents)
            if num_active_tracks > 0 and num_current_peaks > 0:
                # Prepare data for cost calculation
                active_last_freqs_hz = np.array(
                    [
                        self._active_partials[tid]["last_freq"]
                        for tid in active_track_ids
                    ]
                )
                # Handle cases where last_freq might be invalid
                valid_last_freqs = active_last_freqs_hz > 0
                if not np.all(valid_last_freqs):
                    logger.warning(
                        f"Frame {frame_idx}: Found invalid last frequencies in active partials."
                    )
                    # Handle or filter - for now, proceed but costs might be INF
                    active_last_freqs_hz = np.where(
                        valid_last_freqs, active_last_freqs_hz, 1.0
                    )  # Replace invalid with 1Hz to avoid log(0)

                active_last_freqs_cents = hz_to_cents_numba(active_last_freqs_hz)

                current_peak_freqs_hz = np.array([p[0] for p in current_peaks])
                # Ensure peaks are valid
                valid_peaks = current_peak_freqs_hz > 0
                if not np.all(valid_peaks):
                    logger.warning(
                        f"Frame {frame_idx}: Found invalid peak frequencies."
                    )
                    current_peak_freqs_hz = np.where(
                        valid_peaks, current_peak_freqs_hz, 1.0
                    )

                current_peak_freqs_cents = hz_to_cents_numba(current_peak_freqs_hz)

                # Calculate all-pairs cent differences
                # Ensure cents arrays are valid before calculating diffs
                valid_cents_mask = np.isfinite(
                    active_last_freqs_cents[:, np.newaxis]
                ) & np.isfinite(current_peak_freqs_cents[np.newaxis, :])
                cent_diffs = np.full(
                    (num_active_tracks, num_current_peaks), COST_INF, dtype=np.float64
                )
                if np.any(valid_cents_mask):
                    diffs = np.abs(
                        active_last_freqs_cents[:, np.newaxis]
                        - current_peak_freqs_cents[np.newaxis, :]
                    )
                    cent_diffs = np.where(valid_cents_mask, diffs, COST_INF)

                # TODO: Add amplitude difference, SPOD difference?
                # cost_matrix = cent_diffs # Simple cost for now

                # Apply cost only if difference is below a threshold (beta-dependent)
                valid_mask = cent_diffs < self.partial_match_cost_threshold
                cost_matrix = np.where(valid_mask, cent_diffs, COST_INF)

            # --- Solve Assignment Problem ---
            matched_track_indices: Set[int] = set()
            matched_peak_indices: Set[int] = set()
            if (
                num_active_tracks > 0
                and num_current_peaks > 0
                and np.any(cost_matrix < COST_INF)
            ):
                try:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    # Filter assignments above the cost threshold
                    valid_assignment_mask = cost_matrix[row_ind, col_ind] < COST_INF
                    assigned_rows = row_ind[valid_assignment_mask]
                    assigned_cols = col_ind[valid_assignment_mask]
                except ValueError as e:
                    logger.error(
                        f"Error in linear_sum_assignment at frame {frame_idx}: {e}"
                    )
                    assigned_rows, assigned_cols = np.array([]), np.array([])

                # --- Update Matched Partials ---
                for r, c in zip(assigned_rows.astype(int), assigned_cols.astype(int)):
                    track_id = active_track_ids[r]
                    peak_freq, peak_amp, _ = current_peaks[c]

                    # Get phase and SPOD for the matched peak's bin
                    peak_bin_approx = self._freq_to_bin_approx(peak_freq)
                    peak_bin_idx = (
                        int(round(peak_bin_approx))
                        if peak_bin_approx is not None
                        else -1
                    )

                    current_phase = 0.0
                    current_spod = 0.0
                    # Safely access phase/spod arrays
                    if (
                        phases_per_frame is not None
                        and spod_map_per_frame is not None
                        and peak_bin_idx >= 0
                        and peak_bin_idx < phases_per_frame.shape[0]
                        and frame_idx < phases_per_frame.shape[1]
                    ):
                        try:
                            current_phase = phases_per_frame[peak_bin_idx, frame_idx]
                            current_spod = spod_map_per_frame[peak_bin_idx, frame_idx]
                        except IndexError:
                            logger.warning(
                                f"Index error accessing phase/spod at frame {frame_idx}, bin {peak_bin_idx}"
                            )

                    partial = self._active_partials[track_id]
                    partial["history"].append(
                        {
                            "frame": frame_idx,
                            "time": self._times[frame_idx],
                            "freq": peak_freq,
                            "amp": peak_amp,
                            "phase": current_phase,
                            "spod": current_spod,
                        }
                    )
                    partial["last_freq"] = peak_freq
                    partial["last_update_frame"] = frame_idx
                    partial["inactive_frames"] = 0

                    matched_track_indices.add(r)
                    matched_peak_indices.add(c)

            # --- Handle Unmatched Partials (Potential Termination) ---
            terminated_track_ids: List[int] = []
            for r, track_id in enumerate(active_track_ids):
                if r not in matched_track_indices:
                    partial = self._active_partials[track_id]
                    partial["inactive_frames"] += 1
                    if partial["inactive_frames"] >= self.PARTIAL_MAX_INACTIVE_FRAMES:
                        terminated_track_ids.append(track_id)
                    else:
                        # Keep partial active but mark as inactive this frame
                        # Add a record indicating inactivity
                        partial["history"].append(
                            {
                                "frame": frame_idx,
                                "time": self._times[frame_idx],
                                "freq": partial["last_freq"],  # Carry over last freq
                                "amp": 0.0,  # Mark amplitude as zero
                                "phase": np.nan,  # Mark phase as undefined
                                "spod": 0.0,  # Mark SPOD as zero
                                "inactive": True,  # Add inactive flag
                            }
                        )

            # Remove terminated partials
            for track_id in terminated_track_ids:
                # TODO: Store terminated partials somewhere if needed later?
                if track_id in self._active_partials:  # Check existence before deleting
                    del self._active_partials[track_id]

            # --- Start New Partials for Unmatched Peaks ---
            for c, peak_data in enumerate(current_peaks):
                if c not in matched_peak_indices:
                    peak_freq, peak_amp, _ = peak_data
                    if peak_freq <= 0:
                        continue  # Skip invalid peaks

                    peak_bin_approx = self._freq_to_bin_approx(peak_freq)
                    peak_bin_idx = (
                        int(round(peak_bin_approx))
                        if peak_bin_approx is not None
                        else -1
                    )

                    current_phase = 0.0
                    current_spod = 0.0
                    # Safely access phase/spod arrays
                    if (
                        phases_per_frame is not None
                        and spod_map_per_frame is not None
                        and peak_bin_idx >= 0
                        and peak_bin_idx < phases_per_frame.shape[0]
                        and frame_idx < phases_per_frame.shape[1]
                    ):
                        try:
                            current_phase = phases_per_frame[peak_bin_idx, frame_idx]
                            current_spod = spod_map_per_frame[peak_bin_idx, frame_idx]
                        except IndexError:
                            logger.warning(
                                f"Index error accessing phase/spod for new partial at frame {frame_idx}, bin {peak_bin_idx}"
                            )

                    new_partial = {
                        "id": self._partial_id_counter,
                        "start_frame": frame_idx,
                        "start_time": self._times[frame_idx],
                        "history": [
                            {
                                "frame": frame_idx,
                                "time": self._times[frame_idx],
                                "freq": peak_freq,
                                "amp": peak_amp,
                                "phase": current_phase,
                                "spod": current_spod,
                            }
                        ],
                        "last_freq": peak_freq,
                        "last_update_frame": frame_idx,
                        "inactive_frames": 0,
                    }
                    self._active_partials[self._partial_id_counter] = new_partial
                    self._partial_id_counter += 1

            # Store the state of active partials *for this frame*
            # Need deep copy to avoid modification issues
            current_frame_state = {}
            for tid, p in self._active_partials.items():
                # Basic copy first
                p_copy = p.copy()
                # Deep copy the history list
                p_copy["history"] = list(p["history"])
                current_frame_state[tid] = p_copy

            all_frame_partials_state.append(current_frame_state)

        logger.info(
            f"Partial tracking completed. Max partial ID: {self._partial_id_counter}"
        )
        return all_frame_partials_state

    def _freq_to_bin_approx(self, freq_hz: float) -> Optional[float]:
        """Finds the approximate (fractional) bin index."""
        if (
            self._freqs is None
            or self._freq_resolution_hz is None
            or self._freq_resolution_hz <= 0
        ):
            return None
        # Check against actual FFT frequency range
        if freq_hz < self._freqs[0] or freq_hz > self._freqs[-1]:
            return None
        return freq_hz / self._freq_resolution_hz

    # --- Placeholder Methods for Next Stages ---

    def _calculate_unpredictability(
        self, partial_states_per_frame: List[Dict[int, Dict[str, Any]]]
    ) -> np.ndarray:
        """
        Calculates the frame-wise unpredictability U(t) based on partial prediction errors.
        Placeholder implementation.
        """
        num_frames = len(partial_states_per_frame)
        if num_frames == 0:
            return np.array([])

        unpredictability = np.zeros(num_frames)
        logger.warning("_calculate_unpredictability not fully implemented yet.")

        # TODO: Implement actual prediction and error calculation logic here
        # For each frame t > 0:
        #  - Get active partials P(t) and P(t-1)
        #  - For each partial p in P(t) that also existed in P(t-1):
        #      - Predict state of p at t based on state at t-1 (e.g., phase, amp)
        #      - Calculate error E_p(t) = |predicted_state - observed_state| + f(SPOD, amp_change)
        #  - Aggregate errors: U(t) = sum(E_p(t) for p in P(t)) or max(...)

        # Example: Simple placeholder just summing number of active partials
        for frame_idx, frame_state in enumerate(partial_states_per_frame):
            unpredictability[frame_idx] = len(frame_state)

        return unpredictability

    def _detect_onsets_from_unpredictability(
        self, unpredictability: np.ndarray
    ) -> Set[int]:
        """
        Detects onset frames by finding peaks in the smoothed unpredictability function.
        Uses beta-dependent smoothing and thresholding.
        Placeholder implementation.
        """
        if unpredictability.size == 0:
            return set()

        # 1. Smooth unpredictability using self.smoothing_frames
        if self.smoothing_frames > 1:
            smoothed_U = uniform_filter1d(
                unpredictability, size=self.smoothing_frames, mode="nearest"
            )
        else:
            smoothed_U = unpredictability

        # 2. Normalize smoothed_U (handle potential division by zero)
        max_U = np.max(smoothed_U)
        if max_U > 1e-9:
            norm_smoothed_U = smoothed_U / max_U
        else:
            norm_smoothed_U = (
                smoothed_U  # Avoid division by zero if signal is flat zero
            )

        # 3. Find peaks above relative threshold
        # Height threshold is relative to the max *before* normalization if using properties['peak_heights'] later
        # If using normalized height, threshold is absolute value from self.relative_onset_threshold
        peak_threshold = (
            self.relative_onset_threshold
        )  # Absolute threshold on 0-1 normalized signal

        try:
            peaks, properties = find_peaks(norm_smoothed_U, height=peak_threshold)
            logger.info(f"Detected {len(peaks)} onset peaks using unpredictability.")
            return set(peaks)
        except Exception as e:
            logger.error(
                f"Error detecting onset peaks from unpredictability: {e}", exc_info=True
            )
            return set()

    def _segment_notes(
        self,
        partial_states_per_frame: List[Dict[int, Dict[str, Any]]],
        onset_frames: Set[int],
    ) -> List[Dict[str, Any]]:
        """
        Segments notes based on tracked partials and detected onsets.
        Placeholder implementation.
        """
        logger.warning("_segment_notes not fully implemented yet.")
        # TODO: Implement note segmentation logic
        # - Iterate through frames
        # - If frame is an onset frame, potentially start new notes based on active partials.
        # - Track active notes based on continuous partial activity.
        # - Terminate notes when partials end or unpredictability increases significantly (offset detection).
        # - Estimate note pitch from constituent partials.

        # Very basic placeholder: treat every detected onset as a short note start
        notes_list = []
        if self._times is not None:
            for frame_idx in sorted(list(onset_frames)):
                if frame_idx < len(self._times):
                    start_time = self._times[frame_idx]
                    # Find first active partial at this frame as placeholder pitch
                    pitch_hz = 100.0  # Default placeholder
                    if frame_idx < len(partial_states_per_frame):
                        partials_now = partial_states_per_frame[frame_idx]
                        if partials_now:
                            first_partial_id = next(iter(partials_now))
                            history = partials_now[first_partial_id].get("history", [])
                            if history:
                                pitch_hz = history[-1].get("freq", 100.0)

                    notes_list.append(
                        {
                            "start_time": start_time,
                            "offset_time": start_time
                            + 0.1,  # Fixed 100ms duration placeholder
                            "pitch_hz": pitch_hz,
                            "duration": 0.1,
                        }
                    )
        return notes_list

    def _postprocess_notes(
        self, notes_list: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Post-processes notes (e.g., duration filtering, merging). Placeholder."""
        if not notes_list:
            return np.array([]).reshape(0, 2), np.array([])

        # Example: Filter by min_duration derived from beta?
        min_duration = 0.05 + 0.05 * (
            1 - self.beta
        )  # Example: 50ms (high beta) to 100ms (low beta)

        processed_notes = [
            n for n in notes_list if n.get("duration", 0) >= min_duration
        ]
        logger.info(
            f"Postprocessing: Kept {len(processed_notes)} notes after min duration filter ({min_duration:.3f}s)."
        )

        # Placeholder: Return notes after basic duration filter
        try:
            if not processed_notes:
                return np.array([]).reshape(0, 2), np.array([])
            intervals = np.array(
                [[n["start_time"], n["offset_time"]] for n in processed_notes]
            )
            pitches = np.array([n["pitch_hz"] for n in processed_notes])
            return intervals, pitches
        except KeyError as e:
            logger.error(f"Note list format incorrect during postprocessing: {e}")
            return np.array([]).reshape(0, 2), np.array([])

    # --- Main Detection Pipeline ---

    def detect(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Main detection method using the unpredictability-based approach.
        """
        start_process_time = time.time()
        logger.info(
            f"Starting detection ({self.__class__.__name__} beta={self.beta:.3f}) with SR={sr}."
        )

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
            # --- 1. Initialization & STFT ---
            self._initialize_run(sr)
            S_amp, S_db, phases = self._calculate_stft(audio_data)
            if S_amp is None or S_db is None or phases is None or self._times is None:
                raise ValueError("STFT failed.")

            # --- 2. SPOD Calculation ---
            logger.debug("Calculating SPOD Map...")
            spod_map = self._calculate_spod(phases)
            # Handle potential failure of SPOD calculation
            if spod_map is None:
                logger.warning(
                    "SPOD calculation failed. Proceeding without SPOD info for tracking."
                )
                spod_map_for_tracking = None
            elif spod_map.shape != S_amp.shape:
                logger.warning(
                    f"SPOD map shape {spod_map.shape} mismatch with STFT {S_amp.shape}. Proceeding without SPOD info."
                )
                spod_map_for_tracking = None
            else:
                spod_map_for_tracking = spod_map

            # --- 3. Peak Detection ---
            logger.debug("Detecting peaks per frame...")
            peaks_per_frame = self._detect_peaks_per_frame(S_amp, S_db)

            # --- 4. Partial Tracking ---
            logger.debug("Tracking partials across frames...")
            partial_states_per_frame = self._track_partials(
                peaks_per_frame, phases, spod_map_for_tracking
            )

            # --- 5. Unpredictability Calculation (Placeholder) ---
            logger.debug("Calculating frame-wise unpredictability...")
            unpredictability = self._calculate_unpredictability(
                partial_states_per_frame
            )

            # --- 6. Onset Detection (Using Placeholder Unpredictability) ---
            logger.debug("Detecting onsets from unpredictability...")
            onset_frames = self._detect_onsets_from_unpredictability(unpredictability)

            # --- 7. Note Segmentation (Placeholder) ---
            logger.debug("Segmenting notes...")
            notes_list = self._segment_notes(partial_states_per_frame, onset_frames)

            # --- 8. Post-processing (Placeholder) ---
            logger.debug("Post-processing notes...")
            intervals, note_pitches = self._postprocess_notes(notes_list)

            detection_time = time.time() - start_process_time
            logger.info(
                f"Detection completed in {detection_time:.4f} seconds. Found {len(note_pitches)} notes."
            )

            # --- Format Results ---
            # Placeholder for frame frequencies (e.g., from dominant partial)
            frame_frequencies = np.zeros_like(self._times)

            result = {
                "intervals": intervals,
                "note_pitches": note_pitches,
                "frame_times": self._times,
                "frame_frequencies": frame_frequencies,  # Placeholder
                "detector_name": self.__class__.__name__,
                "detection_time": detection_time,
                "additional_data": {
                    "note_count": len(note_pitches),
                    "parameters": self.get_params(),
                    "beta_derived_threshold": self.relative_onset_threshold,  # Example
                    "beta_derived_smoothing": self.smoothing_frames,  # Example
                },
            }
            return result

        except Exception as e:
            logger.error(f"Error during detection pipeline: {e}", exc_info=True)
            return self._empty_result(error_message=str(e))
        finally:
            # Clean up internal state (optional, depending on usage)
            # self._sr = None ... etc.
            pass

    def _empty_result(
        self, error_message: str = "Detection failed or invalid input"
    ) -> Dict[str, Any]:
        """Returns a standardized empty result dictionary."""
        return {
            "intervals": np.array([]).reshape(0, 2),
            "note_pitches": np.array([]),
            "frame_times": (
                self._times
                if hasattr(self, "_times") and self._times is not None
                else np.array([])
            ),
            "frame_frequencies": np.array([]),
            "detector_name": self.__class__.__name__,
            "detection_time": 0.0,
            "additional_data": {
                "error": error_message,
                "note_count": 0,
                "parameters": self.get_params(),
            },
        }


# Note: Numba helper functions like hz_to_cents_numba etc. are imported or can be defined here.
#       The core partial tracking, unpredictability, onset, and note segmentation logic
#       needs further implementation in subsequent steps.
