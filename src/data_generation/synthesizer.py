import numpy as np
import librosa
import soundfile as sf
import os
import csv
import logging
from typing import List, Tuple, Optional, Union
from scipy.signal import butter, filtfilt, fftconvolve
from src.utils.exception_utils import SynthesizerError, FileError, log_exception

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 基本パラメータ ---
SR = 44100  # サンプリングレート
DEFAULT_DURATION = 5.0  # デフォルトの持続時間 (秒)
AMP_MAX = 0.8  # 最大振幅 (正規化用)

# --- 出力ディレクトリ定義を修正 ---
# スクリプト自身の場所を取得
_THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# プロジェクトルートを想定 (このスクリプトが src/data_generation にあると仮定)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_SCRIPT_DIR, '..', '..'))
# プロジェクトルートからの相対パスで出力ディレクトリを定義
OUTPUT_BASE_DIR = os.path.join(_PROJECT_ROOT, "data", "synthesized")
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_BASE_DIR, "audio")
OUTPUT_LABEL_DIR = os.path.join(OUTPUT_BASE_DIR, "labels")

# Label type alias for clarity
Label = Tuple[float, float, float]

# --- ディレクトリ作成 --- (呼び出し側で実行した方が安全かも)
def ensure_output_dirs():
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    logging.info(f"出力ディレクトリを作成/確認しました: {OUTPUT_AUDIO_DIR}, {OUTPUT_LABEL_DIR}")

# --- ヘルパー関数 ---

def midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def generate_adsr_envelope(duration_samples: int, attack: float, decay: float, sustain_level: float, release: float, sr: int = SR) -> np.ndarray:
    assert attack >= 0, "Attack time must be non-negative"
    assert decay >= 0, "Decay time must be non-negative"
    assert 0.0 <= sustain_level <= 1.0, "Sustain level must be between 0.0 and 1.0"
    assert release >= 0, "Release time must be non-negative"
    assert duration_samples >= 0, "Duration must be non-negative"

    attack_samples = int(attack * sr)
    decay_samples = int(decay * sr)
    release_samples = int(release * sr)
    sustain_samples = max(0, duration_samples - attack_samples - decay_samples - release_samples)
    total_samples = attack_samples + decay_samples + sustain_samples + release_samples
    envelope = np.zeros(total_samples)
    if attack_samples > 0: envelope[:attack_samples] = np.linspace(0, 1, attack_samples, endpoint=False)
    if decay_samples > 0: envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples, endpoint=False)
    envelope[attack_samples + decay_samples : attack_samples + decay_samples + sustain_samples] = sustain_level
    if release_samples > 0: envelope[attack_samples + decay_samples + sustain_samples:] = np.linspace(sustain_level, 0, release_samples, endpoint=True)
    if total_samples < duration_samples: envelope = np.pad(envelope, (0, duration_samples - total_samples), 'constant', constant_values=(0,))
    elif total_samples > duration_samples: envelope = envelope[:duration_samples]
    return envelope

def generate_sine_wave(freq: float, duration_samples: int, amp: float = 1.0, sr: int = SR) -> np.ndarray:
    if freq <= 0:
        logging.debug(f"Frequency {freq:.2f} <= 0, returning zeros for {duration_samples} samples.")
        return np.zeros(duration_samples)
    assert duration_samples >= 0, "Duration must be non-negative"
    t = np.linspace(0., duration_samples / sr, duration_samples, endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)

def generate_harmonic_tone(base_freq: float, duration_samples: int, harmonics: List[Tuple[int, float]], amp: float = 1.0, sr: int = SR) -> np.ndarray:
    tone = np.zeros(duration_samples)
    if base_freq <= 0:
        logging.debug(f"Base frequency {base_freq:.2f} <= 0, returning zeros for {duration_samples} samples.")
        return tone
    assert duration_samples >= 0, "Duration must be non-negative"
    total_harmonic_amp = 0
    for harmonic_num, harmonic_amp in harmonics:
        assert harmonic_num > 0, f"Harmonic number must be positive: {harmonic_num}"
        assert harmonic_amp >= 0, f"Harmonic amplitude must be non-negative: {harmonic_amp}"
        freq = base_freq * harmonic_num
        tone += generate_sine_wave(freq, duration_samples, harmonic_amp, sr)
        total_harmonic_amp += harmonic_amp
    if total_harmonic_amp > 1.0: tone /= total_harmonic_amp # Normalize
    return amp * tone

def save_audio_and_label(filename_base: str, audio_data: np.ndarray, labels: List[Label], include_header: bool = False, sr: int = SR):
    # Check for NaN or Inf in audio data more robustly
    if not np.all(np.isfinite(audio_data)):
        logging.error(f"音声データに NaN または Inf が含まれています: {filename_base}。0に置換します。")
        audio_data = np.nan_to_num(audio_data)

    if audio_data.size == 0:
        logging.error(f"音声データが空です: {filename_base}。保存をスキップします。")
        return

    # Log basic stats (optional, keep at DEBUG level)
    # logging.debug(f"Saving {filename_base}: Audio shape={audio_data.shape}, Size={audio_data.size}, Min={np.min(audio_data):.4f}, Max={np.max(audio_data):.4f}, Mean={np.mean(audio_data):.4f}")
    # logging.debug(f"Saving {filename_base}: Labels={labels}")

    # --- Normalization --- #
    peak_amp = np.max(np.abs(audio_data))
    if peak_amp > 1e-9:
        normalized_audio = audio_data * (AMP_MAX / peak_amp)
    else:
        normalized_audio = audio_data
        logging.warning(f"オーディオ信号のピーク振幅が非常に小さいです: {filename_base}。正規化はスキップされました。")

    audio_path = os.path.join(OUTPUT_AUDIO_DIR, f"{filename_base}.wav")
    label_path = os.path.join(OUTPUT_LABEL_DIR, f"{filename_base}.csv")

    # --- Audio Saving --- #
    audio_saved_successfully = False
    try:
        ensure_output_dirs() # 保存前にディレクトリ確認/作成
        logging.debug(f"音声書き込み試行: {audio_path}")
        sf.write(audio_path, normalized_audio.astype(np.float32), sr)
        logging.info(f"音声を保存しました: {audio_path}")
        audio_saved_successfully = True
    except (IOError, PermissionError) as file_err:
        log_exception(logging.getLogger(__name__), file_err, f"音声ファイルの保存中にIO/Permissionエラー: {audio_path}", log_level=logging.ERROR)
        # FileError でラップして再発生させることも検討
        # raise FileError(f"Failed to save audio: {file_err}") from file_err
    except Exception as e:
        log_exception(logging.getLogger(__name__), e, f"音声ファイルの保存中に予期せぬエラー: {audio_path}", log_level=logging.ERROR)
        # 必要であれば FileError でラップ
        # raise FileError(f"Unexpected error saving audio: {e}") from e

    # --- Audio Save Verification --- #
    try:
        if audio_saved_successfully and os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                logging.error(f"音声ファイルは作成されましたが空です (0 バイト): {audio_path}")
            else:
                logging.debug(f"音声ファイル保存確認OK: {audio_path}, サイズ: {file_size} バイト")
        elif audio_saved_successfully:
             logging.error(f"音声保存関数は成功しましたが、ファイルが存在しません: {audio_path}")
    except (IOError, PermissionError) as verify_err:
        log_exception(logging.getLogger(__name__), verify_err, f"音声ファイルの存在/サイズ確認中にIO/Permissionエラー: {audio_path}", log_level=logging.ERROR)
    except Exception as ve:
        log_exception(logging.getLogger(__name__), ve, f"音声ファイルの存在確認中に予期せぬエラー: {audio_path}", log_level=logging.ERROR)

    # --- Label Saving --- #
    label_saved_successfully = False
    try:
        ensure_output_dirs() # 保存前にディレクトリ確認/作成
        logging.debug(f"ラベル書き込み試行: {label_path}")
        with open(label_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if include_header:
                writer.writerow(['onset', 'offset', 'frequency'])
            if not labels:
                 logging.warning(f"{filename_base} にラベルがありません。CSVは空またはヘッダーのみになります。")
            for onset, offset, freq_or_marker in labels:
                freq_str = f"{freq_or_marker:.3f}" if isinstance(freq_or_marker, (float, np.floating)) else str(freq_or_marker)
                writer.writerow([f"{onset:.6f}", f"{offset:.6f}", freq_str])
        logging.info(f"ラベルを保存しました: {label_path}")
        label_saved_successfully = True
    except (IOError, PermissionError) as file_err:
        log_exception(logging.getLogger(__name__), file_err, f"ラベルファイルの保存中にIO/Permissionエラー: {label_path}", log_level=logging.ERROR)
        # raise FileError(...) from file_err
    except csv.Error as csv_err: # csv固有のエラーも捕捉
         log_exception(logging.getLogger(__name__), csv_err, f"ラベルファイルのCSV書き込みエラー: {label_path}", log_level=logging.ERROR)
         # raise FileError(...) from csv_err
    except Exception as e:
        log_exception(logging.getLogger(__name__), e, f"ラベルファイルの保存中に予期せぬエラー: {label_path}", log_level=logging.ERROR)
        # raise FileError(...) from e

    # --- Label Save Verification --- #
    try:
        if label_saved_successfully and os.path.exists(label_path):
            file_size = os.path.getsize(label_path)
            if file_size == 0 and labels:
                 logging.error(f"ラベルファイルは作成されましたが空です (0 バイト): {label_path} (ラベル数: {len(labels)})")
            elif file_size == 0 and not labels:
                 logging.debug(f"ラベルファイルは空ですが、ラベルデータもありません: {label_path}")
            else:
                 logging.debug(f"ラベルファイル保存確認OK: {label_path}, サイズ: {file_size} バイト")
        elif label_saved_successfully:
            logging.error(f"ラベル保存関数は成功しましたが、ファイルが存在しません: {label_path}")
    except (IOError, PermissionError) as verify_err:
        log_exception(logging.getLogger(__name__), verify_err, f"ラベルファイルの存在/サイズ確認中にIO/Permissionエラー: {label_path}", log_level=logging.ERROR)
    except Exception as ve:
        log_exception(logging.getLogger(__name__), ve, f"ラベルファイルの存在確認中に予期せぬエラー: {label_path}", log_level=logging.ERROR)

def add_white_noise(audio_data: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(audio_data ** 2)
    if signal_power < 1e-9:
        logging.debug("Signal power is near zero, skipping noise addition.")
        return audio_data

    snr_linear = 10 ** (snr_db / 10.0)
    # Avoid division by zero or adding excessive noise for very low SNR
    if snr_linear < 1e-9: # Corresponds to SNR < -90dB, effectively infinite noise
        logging.warning(f"SNR {snr_db}dB is very low (linear {snr_linear:.2e}). Noise might dominate or cause issues. Returning original signal.")
        # Optionally return noise scaled to AMP_MAX? Or just return original? Returning original for safety.
        return audio_data

    noise_power = signal_power / snr_linear
    # Add check for potentially huge noise power if snr_linear is extremely small but non-zero
    if noise_power > 1e6 * signal_power: # Heuristic check: if noise power is million times signal power
        logging.warning(f"Calculated noise power ({noise_power:.2e}) is extremely high relative to signal power ({signal_power:.2e}) for SNR {snr_db}dB. Clipping noise or returning original signal might be needed.")
        # For now, proceed but this indicates potential issues.

    noise = np.random.randn(len(audio_data)) * np.sqrt(noise_power)
    return audio_data + noise

def generate_simple_ir(duration_sec: float, decay_rate: float, sr: int = SR) -> np.ndarray:
    samples = int(duration_sec * sr)
    t = np.arange(samples) / sr
    ir = np.random.randn(samples) * np.exp(-decay_rate * t)
    if np.max(np.abs(ir)) > 1e-9:
        ir /= np.max(np.abs(ir))
    return ir

def apply_reverb(audio_data: np.ndarray, ir: np.ndarray, mix_level: float = 0.3) -> np.ndarray:
    """
    音声データにインパルス応答を用いてリバーブを適用します。
    畳み込み結果の全長を保持します。

    Parameters
    ----------
    audio_data : np.ndarray
        ドライ音声データ
    ir : np.ndarray
        インパルス応答
    mix_level : float, optional
        ウェット信号のミックスレベル (0.0 ~ 1.0), by default 0.3

    Returns
    -------
    np.ndarray
        リバーブが付加された音声データ

    Raises
    ------
    SynthesizerError
        畳み込み処理中に予期せぬエラーが発生した場合
    """
    logger = logging.getLogger(__name__) # 関数スコープでロガーを取得

    if len(ir) == 0 or mix_level <= 0:
        logger.debug("IRが空またはミックスレベルが0以下なのでリバーブ適用をスキップします。")
        return audio_data
    if mix_level > 1.0:
        logger.warning(f"ミックスレベル {mix_level} が1.0を超えています。1.0にクリップします。")
        mix_level = 1.0

    # 畳み込みを実行し、リバーブ信号を生成
    try:
        # mode='full' で全長を取得
        reverb_signal = fftconvolve(audio_data, ir, mode='full')
    except ValueError as ve:
        # ValueError は不正な入力が原因の可能性が高い
        log_exception(logger, ve, f"fftconvolve中にValueErrorが発生しました。IR Length: {len(ir)}, Audio Length: {len(audio_data)}", log_level=logging.ERROR)
        # エラー時は元の音声を返すか、例外を発生させるか？ -> SynthesizerError を発生させる
        raise SynthesizerError(f"Convolution failed due to invalid input: {ve}") from ve
    except MemoryError as me:
        # MemoryError はリソース不足
        log_exception(logger, me, f"fftconvolve中にMemoryErrorが発生しました。IR Length: {len(ir)}, Audio Length: {len(audio_data)}", log_level=logging.CRITICAL)
        raise SynthesizerError(f"Convolution failed due to insufficient memory: {me}") from me
    except Exception as e:
        # その他の予期せぬエラー
        log_exception(logger, e, f"fftconvolve中に予期せぬエラーが発生しました。", log_level=logging.ERROR)
        raise SynthesizerError(f"Unexpected error during convolution: {e}") from e

    # 期待される出力長
    target_len = len(audio_data) + len(ir) - 1

    # 念のため長さを確認・調整（通常は不要だが数値誤差対策）
    if len(reverb_signal) != target_len:
        logger.warning(f"畳み込み結果の長さが期待値と異なります。期待値: {target_len}, 実際: {len(reverb_signal)}。調整します。")
        if len(reverb_signal) < target_len:
            reverb_signal = np.pad(reverb_signal, (0, target_len - len(reverb_signal)))
        else:
            reverb_signal = reverb_signal[:target_len]

    # ドライ信号をリバーブ信号の長さに合わせてパディング
    padded_dry = np.pad(audio_data, (0, target_len - len(audio_data)))

    # ドライ信号とウェット信号をミックス
    mixed_signal = padded_dry * (1.0 - mix_level) + reverb_signal * mix_level

    # ミックス後の信号のクリッピングも検討 (オプション)
    # max_val = np.max(np.abs(mixed_signal))
    # if max_val > 1.0:
    #     logger.warning(f"リバーブ適用後の信号がクリップしました (Max: {max_val:.2f})。クリッピングします。")
    #     mixed_signal = np.clip(mixed_signal, -1.0, 1.0)

    return mixed_signal

def generate_kick_sound(duration_sec: float = 0.3, freq: float = 60.0, decay_rate: float = 15.0, amp: float = 1.0, sr: int = SR) -> np.ndarray:
    duration_samples = int(duration_sec * sr)
    t = np.linspace(0., duration_sec, duration_samples, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    envelope = np.exp(-decay_rate * t)
    kick = wave * envelope
    return kick * amp

def generate_noise_perc_sound(duration_sec: float = 0.2, bandpass_freqs: Optional[Tuple[float, float]] = None, decay_rate: float = 25.0, amp: float = 1.0, sr: int = SR) -> np.ndarray:
    assert duration_sec > 0, "Duration must be positive"
    assert decay_rate >= 0, "Decay rate must be non-negative"
    duration_samples = int(duration_sec * sr)
    noise = np.random.randn(duration_samples)
    if bandpass_freqs:
        low, high = bandpass_freqs
        nyquist = 0.5 * sr
        assert 0 < low < high < nyquist, f"Invalid bandpass frequencies: {low}, {high} for sr={sr}"

        low_norm = low / nyquist
        high_norm = high / nyquist
        # Ensure norms are valid and low < high even after potential clipping (though assert should catch)
        low_norm = np.clip(low_norm, 1e-6, 1.0 - 1e-6)
        high_norm = np.clip(high_norm, 1e-6, 1.0 - 1e-6)
        if low_norm >= high_norm:
            high_norm = low_norm + 1e-6 # Ensure high > low
            logging.warning(f"Adjusted high norm frequency to be slightly above low norm: {low_norm=}, {high_norm=}")

        try:
            # Filter order (can be adjusted)
            order = 4
            b, a = butter(order, [low_norm, high_norm], btype='band')
            noise = filtfilt(b, a, noise)
        except Exception as e:
            logging.error(f"Bandpass filtering failed: {e}", exc_info=True)
            # Optionally raise SynthesizerError or return unfiltered noise
            # raise SynthesizerError(f"Bandpass filtering failed: {e}") from e
            pass # Continue with unfiltered noise for now

    t = np.linspace(0., duration_sec, duration_samples, endpoint=False) # Add this line
    envelope = np.exp(-decay_rate * t)
    perc = noise * envelope
    return perc * amp

def generate_formant_filter(formant_freqs: List[float], bandwidths: List[float], sr: int = SR):
    assert len(formant_freqs) == len(bandwidths), "Formant frequencies and bandwidths must have the same length"
    filter_response = np.zeros(sr // 2 + 1) # Frequency response up to Nyquist
    freqs = np.fft.rfftfreq(sr, d=1./sr)
    for f_formant, bw in zip(formant_freqs, bandwidths):
        assert f_formant > 0, f"Formant frequency must be positive: {f_formant}"
        assert bw > 0, f"Bandwidth must be positive: {bw}"
        # Simple resonance model (adjust gain as needed)
        gain = 1.0
        term = (freqs - f_formant) / (bw / 2.0)
        # Avoid division by zero if bw is extremely small? Add epsilon?
        resonance = gain / (1.0 + term**2 + 1e-12) # Added epsilon
        filter_response += resonance
    # Normalize filter response
    max_resp = np.max(filter_response)
    if max_resp > 1e-9:
        filter_response /= max_resp
    return filter_response

def apply_formant_filter(audio_data: np.ndarray, filter_response: np.ndarray, sr: int = SR) -> np.ndarray:
    """Applies a pre-computed formant filter response to audio data using FFT."""
    assert len(filter_response) == (sr // 2 + 1), "Filter response length mismatch"
    # Ensure audio_data is float
    audio_data_float = audio_data.astype(np.float32)
    audio_fft = np.fft.rfft(audio_data_float)
    # Ensure filter_response length matches audio_fft length
    if len(audio_fft) != len(filter_response):
        # This shouldn't happen if sr matches, but handle potential mismatch
        logging.warning(f"FFT length ({len(audio_fft)}) and filter response length ({len(filter_response)}) mismatch. Resizing filter.")
        # Simple zero-padding or truncation - more sophisticated resizing might be needed
        target_len = len(audio_fft)
        if target_len > len(filter_response):
            filter_response = np.pad(filter_response, (0, target_len - len(filter_response)))
    else:
            filter_response = filter_response[:target_len]

    filtered_fft = audio_fft * filter_response
    filtered_audio = np.fft.irfft(filtered_fft)
    # Ensure output length matches input length
    if len(filtered_audio) > len(audio_data):
        filtered_audio = filtered_audio[:len(audio_data)]
    elif len(filtered_audio) < len(audio_data):
        filtered_audio = np.pad(filtered_audio, (0, len(audio_data) - len(filtered_audio)))

    return filtered_audio

def add_click_noise(audio_data: np.ndarray, click_prob: float = 0.001, click_amp: float = 0.5, sr: int = SR) -> np.ndarray:
    assert 0.0 <= click_prob <= 1.0, "Click probability must be between 0.0 and 1.0"
    assert click_amp >= 0, "Click amplitude must be non-negative"
    num_samples = len(audio_data)
    click_mask = np.random.rand(num_samples) < click_prob
    click_noise = (np.random.rand(num_samples) * 2 - 1) * click_amp * click_mask
    return audio_data + click_noise

# --- 個別のデータ生成関数 --- (必要に応じて private 化 _generate_... も検討)

def generate_basic_sine_sequence(filename_base: str = "1_basic_sine", sr: int = SR):
    notes = [(60, 0.5), (62, 0.5), (64, 0.5), (65, 0.5), (67, 0.5), (69, 0.5), (71, 0.5), (72, 1.0)]
    adsr_params = {'attack': 0.02, 'decay': 0.1, 'sustain_level': 0.7, 'release': 0.1}
    total_duration_sec = sum(dur for _, dur in notes)
    total_duration_samples = int(total_duration_sec * sr)
    audio = np.zeros(total_duration_samples)
    labels: List[Label] = []
    current_time_samples = 0
    for midi_note, duration_sec in notes:
        freq = midi_to_hz(midi_note)
        note_duration_samples = int(duration_sec * sr)
        envelope_samples = int((duration_sec + adsr_params['release']) * sr)
        envelope = generate_adsr_envelope(envelope_samples, **adsr_params, sr=sr)
        wave = generate_sine_wave(freq, envelope_samples, amp=1.0, sr=sr)
        note_wave = wave * envelope
        end_sample = current_time_samples + note_duration_samples
        envelope_end_sample_in_context = current_time_samples + len(note_wave)
        write_end = min(total_duration_samples, envelope_end_sample_in_context)
        write_len = write_end - current_time_samples
        if write_len > 0: audio[current_time_samples:write_end] += note_wave[:write_len]
        # Offset calculation modified to include a portion of the release phase
        release_time_sec = adsr_params.get('release', 0.1)
        perceived_release_duration_sec = release_time_sec * 0.8
        offset_time_sec = end_sample / sr + perceived_release_duration_sec
        labels.append((current_time_samples / sr, offset_time_sec, freq))
        current_time_samples = end_sample
    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_harmonic_sequence(filename_base: str = "2_harmonic_tone", sr: int = SR):
    notes = [(60, 0.3), (64, 0.3), (67, 0.3), (72, 0.6), (71, 0.3), (67, 0.3), (64, 0.3), (60, 0.6)]
    harmonics = [(1, 1.0), (3, 0.5), (5, 0.25), (7, 0.1)]
    adsr_params = {'attack': 0.05, 'decay': 0.15, 'sustain_level': 0.6, 'release': 0.2}
    audio, labels = _generate_melody_audio_labels(notes, harmonics, adsr_params, sr=sr)
    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_dynamics_change(filename_base: str = "3_dynamics_change", midi_note: int = 67, sr: int = SR):
    duration_sec = 6.0
    duration_samples = int(duration_sec * sr)
    freq = midi_to_hz(midi_note)
    harmonics = [(1, 1.0), (2, 0.4), (3, 0.6), (4, 0.2)]
    adsr_params = {'attack': 0.1, 'decay': 0.1, 'sustain_level': 1.0, 'release': 0.3}
    envelope_samples = int((duration_sec + adsr_params['release']) * sr)
    envelope = generate_adsr_envelope(envelope_samples, **adsr_params, sr=sr)
    tone = generate_harmonic_tone(freq, envelope_samples, harmonics, amp=1.0, sr=sr)
    base_audio = tone * envelope
    base_audio = base_audio[:duration_samples]
    cres_samples = duration_samples // 2
    decres_samples = duration_samples - cres_samples
    dynamic_env = np.concatenate([np.linspace(0.1, 1.0, cres_samples), np.linspace(1.0, 0.1, decres_samples)])
    audio = base_audio * dynamic_env
    
    # Offset calculation modified to include a portion of the release phase
    release_time_sec = adsr_params.get('release', 0.3)
    perceived_release_duration_sec = release_time_sec * 0.8
    offset_time_sec = duration_sec + perceived_release_duration_sec
    
    labels: List[Label] = [(adsr_params['attack'], offset_time_sec, freq)]
    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_pitch_modulation(filename_base: str = "4_pitch_mod", midi_note: int = 69, sr: int = SR):
    duration_sec = 6.0
    duration_samples = int(duration_sec * sr)
    base_freq = midi_to_hz(midi_note)
    harmonics = [(1, 1.0), (3, 0.4)]
    adsr_params = {'attack': 0.05, 'decay': 0.1, 'sustain_level': 0.8, 'release': 0.2}
    t = np.linspace(0., duration_sec, duration_samples, endpoint=False)
    vib_samples = duration_samples // 2
    t_vib = t[:vib_samples]
    vib_rate = 5.0
    vib_depth_cents = 50
    vib_depth_hz = base_freq * (2**(vib_depth_cents / 1200) - 1)
    freq_mod_vib_segment = base_freq + vib_depth_hz * np.sin(2 * np.pi * vib_rate * t_vib)
    port_samples = duration_samples - vib_samples
    target_freq = midi_to_hz(midi_note + 5)
    portamento_start_freq = freq_mod_vib_segment[-1] if vib_samples > 0 else base_freq
    freq_mod_port_segment = np.linspace(portamento_start_freq, target_freq, port_samples)
    freq_mod_combined = np.concatenate((freq_mod_vib_segment, freq_mod_port_segment))
    if len(freq_mod_combined) != duration_samples:
        logging.warning(f"Pitch mod length mismatch: {len(freq_mod_combined)} vs {duration_samples}")
        if abs(len(freq_mod_combined) - duration_samples) < 5: freq_mod_combined = np.resize(freq_mod_combined, duration_samples)
        else: logging.error("Cannot generate pitch mod"); return
    phase = np.cumsum(2 * np.pi * freq_mod_combined / sr)
    modulated_tone = np.sin(phase)
    envelope_samples = int((duration_sec + adsr_params['release']) * sr)
    envelope = generate_adsr_envelope(envelope_samples, **adsr_params, sr=sr)
    audio = modulated_tone * envelope[:duration_samples]
    
    # Offset calculation modified to include a portion of the release phase
    release_time_sec = adsr_params.get('release', 0.2)
    perceived_release_duration_sec = release_time_sec * 0.8
    vib_offset_time_sec = vib_samples / sr + perceived_release_duration_sec
    final_offset_time_sec = duration_sec + perceived_release_duration_sec
    
    labels: List[Label] = [
        (adsr_params['attack'], vib_offset_time_sec, base_freq),
        (vib_samples / sr, final_offset_time_sec, (portamento_start_freq + target_freq) / 2) # Approx pitch for portamento
    ]
    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_noisy_sequence(filename_base_prefix: str = "5_noisy", sr: int = SR):
    base_audio, labels = _generate_harmonic_sequence_audio_labels(sr=sr)
    for snr_db in [30, 20, 10]:
        filename_base = f"{filename_base_prefix}_snr{snr_db}db"
        noisy_audio = add_white_noise(base_audio.copy(), snr_db=snr_db)
        save_audio_and_label(filename_base, noisy_audio, labels, sr=sr)

def generate_reverb_sequence(filename_base_prefix: str = "6_reverb", sr: int = SR):
    base_audio, labels = _generate_harmonic_sequence_audio_labels(sr=sr)
    reverb_settings = {"short": {"duration": 0.5, "decay": 10.0, "mix": 0.2}, "long": {"duration": 1.5, "decay": 3.0, "mix": 0.35}}
    for name, settings in reverb_settings.items():
        filename_base = f"{filename_base_prefix}_{name}"
        ir = generate_simple_ir(duration_sec=settings["duration"], decay_rate=settings["decay"], sr=sr)
        reverb_audio = apply_reverb(base_audio.copy(), ir, mix_level=settings["mix"])
        save_audio_and_label(filename_base, reverb_audio, labels, sr=sr)

def generate_chords(filename_base: str = "7_chords", sr: int = SR):
    chord_sequence = [([60, 64, 67], 1.5), ([65, 69, 72], 1.5), ([55, 59, 62], 2.0)]
    harmonics = [(1, 1.0), (2, 0.3), (3, 0.5)]
    adsr_params = {'attack': 0.03, 'decay': 0.2, 'sustain_level': 0.7, 'release': 0.3}
    total_duration_sec = sum(dur for _, dur in chord_sequence)
    total_duration_samples = int(total_duration_sec * sr)
    audio = np.zeros(total_duration_samples)
    labels: List[Label] = []
    current_time_samples = 0
    for midi_notes, duration_sec in chord_sequence:
        chord_duration_samples = int(duration_sec * sr)
        envelope_samples = int((duration_sec + adsr_params['release']) * sr)
        envelope = generate_adsr_envelope(envelope_samples, **adsr_params, sr=sr)
        chord_wave = np.zeros(envelope_samples)
        onset_time_sec = current_time_samples / sr
        
        # Offset calculation modified to include a portion of the release phase
        release_time_sec = adsr_params.get('release', 0.3)
        perceived_release_duration_sec = release_time_sec * 0.8
        offset_time_sec = (current_time_samples + chord_duration_samples) / sr + perceived_release_duration_sec
        
        for midi_note in midi_notes:
            freq = midi_to_hz(midi_note)
            tone = generate_harmonic_tone(freq, envelope_samples, harmonics, amp=1.0 / len(midi_notes), sr=sr)
            chord_wave += tone
            labels.append((onset_time_sec, offset_time_sec, freq))
        note_wave = chord_wave * envelope
        end_sample = current_time_samples + chord_duration_samples
        envelope_end_sample_in_context = current_time_samples + len(note_wave)
        write_end = min(total_duration_samples, envelope_end_sample_in_context)
        write_len = write_end - current_time_samples
        if write_len > 0: audio[current_time_samples:write_end] += note_wave[:write_len]
        current_time_samples = end_sample
    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_polyphony(filename_base: str = "8_polyphony", sr: int = SR):
    audio, labels = _generate_polyphony_audio_labels(sr=sr)
    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_percussion_only(filename_base: str = "9_10_percussion_only", sr: int = SR):
    duration_sec = 5.0
    duration_samples = int(duration_sec * sr)
    audio = np.zeros(duration_samples)
    labels: List[Label] = []
    beat_duration = 0.5
    kick_sound = generate_kick_sound(amp=0.9, sr=sr)
    kick_len = len(kick_sound)
    for i in range(int(duration_sec / (beat_duration * 2))):
        onset_time = i * beat_duration * 2
        onset_sample = int(onset_time * sr)
        end_sample = onset_sample + kick_len
        if end_sample <= duration_samples: audio[onset_sample:end_sample] += kick_sound; labels.append((onset_time, onset_time + 0.1, 0.0))
    snare_sound = generate_noise_perc_sound(bandpass_freqs=(200, 1000), amp=0.7, sr=sr)
    snare_len = len(snare_sound)
    for i in range(int(duration_sec / (beat_duration * 2))):
        onset_time = (i * 2 + 1) * beat_duration
        onset_sample = int(onset_time * sr)
        end_sample = onset_sample + snare_len
        if end_sample <= duration_samples: audio[onset_sample:end_sample] += snare_sound; labels.append((onset_time, onset_time + 0.1, 0.0))
    hh_sound = generate_noise_perc_sound(bandpass_freqs=(5000, 15000), decay_rate=50.0, amp=0.5, sr=sr)
    hh_len = len(hh_sound)
    for i in range(int(duration_sec / (beat_duration / 2))):
        onset_time = i * beat_duration / 2
        onset_sample = int(onset_time * sr)
        end_sample = onset_sample + hh_len
        if end_sample <= duration_samples: audio[onset_sample:end_sample] += hh_sound; labels.append((onset_time, onset_time + 0.05, 0.0))
    labels.sort()
    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_mixed_melody_percussion(filename_base: str = "11_mixed_melody_perc", sr: int = SR):
    melody_audio, melody_labels = _generate_harmonic_sequence_audio_labels(sr=sr)
    duration_samples = len(melody_audio)
    duration_sec = duration_samples / sr
    audio = melody_audio.copy()
    labels: List[Label] = melody_labels.copy()
    beat_duration = 0.6
    perc_amp_scale = 0.6
    kick_sound = generate_kick_sound(amp=0.9 * perc_amp_scale, sr=sr)
    kick_len = len(kick_sound)
    for i in range(int(duration_sec / (beat_duration * 2))):
        onset_time = i * beat_duration * 2
        onset_sample = int(onset_time * sr)
        end_sample = onset_sample + kick_len
        if end_sample <= duration_samples: audio[onset_sample:end_sample] += kick_sound; labels.append((onset_time, onset_time + 0.1, 0.0))
    snare_sound = generate_noise_perc_sound(bandpass_freqs=(200, 1000), amp=0.7 * perc_amp_scale, sr=sr)
    snare_len = len(snare_sound)
    for i in range(int(duration_sec / (beat_duration * 2))):
        onset_time = (i * 2 + 1) * beat_duration
        onset_sample = int(onset_time * sr)
        end_sample = onset_sample + snare_len
        if end_sample <= duration_samples: audio[onset_sample:end_sample] += snare_sound; labels.append((onset_time, onset_time + 0.1, 0.0))
    labels.sort()
    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_complex_mix(filename_base: str = "12_complex_mix", sr: int = SR):
    poly_audio, poly_labels = _generate_polyphony_audio_labels(sr=sr)
    duration_samples = len(poly_audio)
    duration_sec = duration_samples / sr
    audio = poly_audio.copy()
    labels: List[Label] = poly_labels.copy()
    beat_duration = 0.5
    perc_amp_scale = 0.5
    kick_sound = generate_kick_sound(amp=0.8 * perc_amp_scale, sr=sr)
    kick_len = len(kick_sound)
    for i in range(int(duration_sec / beat_duration)):
        if i % 2 == 0: # Kick on even beats
            onset_time = i * beat_duration
            onset_sample = int(onset_time * sr)
            end_sample = onset_sample + kick_len
            if end_sample <= duration_samples: audio[onset_sample:end_sample] += kick_sound; labels.append((onset_time, onset_time + 0.1, 0.0))
    hh_sound = generate_noise_perc_sound(bandpass_freqs=(6000, 16000), decay_rate=60.0, amp=0.4 * perc_amp_scale, sr=sr)
    hh_len = len(hh_sound)
    for i in range(int(duration_sec / (beat_duration / 2))):
        onset_time = i * beat_duration / 2
        onset_sample = int(onset_time * sr)
        end_sample = onset_sample + hh_len
        if end_sample <= duration_samples: audio[onset_sample:end_sample] += hh_sound; labels.append((onset_time, onset_time + 0.05, 0.0))
    audio = add_white_noise(audio, snr_db=25)
    ir = generate_simple_ir(duration_sec=0.7, decay_rate=8.0, sr=sr)
    audio = apply_reverb(audio, ir, mix_level=0.15)
    labels.sort()
    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_smooth_vibrato(filename_base: str = "13_smooth_vibrato", midi_note: int = 69, sr: int = SR):
    """滑らかなサイン波ビブラートを持つ単一ノートを生成"""
    duration_sec = 4.0
    duration_samples = int(duration_sec * sr)
    base_freq = midi_to_hz(midi_note) # A4
    harmonics = [(1, 1.0), (2, 0.5), (3, 0.3)]
    adsr_params = {'attack': 0.1, 'decay': 0.2, 'sustain_level': 0.7, 'release': 0.3}

    # Vibrato parameters
    vib_rate = 6.0  # Vibrato rate in Hz
    vib_depth_cents = 50 # Vibrato depth in cents
    vib_depth_hz = base_freq * (2**(vib_depth_cents / 1200) - 1)

    t = np.linspace(0., duration_sec, duration_samples, endpoint=False)
    # Frequency modulation: base_freq + sine wave modulation
    freq_mod = base_freq + vib_depth_hz * np.sin(2 * np.pi * vib_rate * t)

    # Generate audio with modulated frequency
    phase = np.cumsum(2 * np.pi * freq_mod / sr)
    modulated_tone = np.sin(phase)

    # Apply harmonics (simplified - apply to the modulated phase)
    harmonic_tone = np.zeros_like(modulated_tone)
    total_amp = 0
    for h_num, h_amp in harmonics:
        harmonic_tone += h_amp * np.sin(h_num * phase)
        total_amp += h_amp
    if total_amp > 0: harmonic_tone /= total_amp # Normalize harmonic amplitudes

    # Apply ADSR envelope
    envelope_samples = int((duration_sec + adsr_params['release']) * sr)
    envelope = generate_adsr_envelope(envelope_samples, **adsr_params, sr=sr)
    audio = harmonic_tone * envelope[:duration_samples]

    # Label: Use the average frequency for simplicity in this example
    # More sophisticated labeling might involve frame-based pitch
    labels: List[Label] = [(adsr_params['attack'], duration_sec, base_freq)]

    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_portamento(filename_base: str = "14_portamento", start_midi: int = 60, end_midi: int = 72, sr: int = SR):
    """あるピッチから別のピッチへ滑らかに移行する（ポルタメント）単一ノートを生成"""
    duration_sec = 3.0
    duration_samples = int(duration_sec * sr)
    start_freq = midi_to_hz(start_midi) # C4
    end_freq = midi_to_hz(end_midi)     # C5
    harmonics = [(1, 1.0), (3, 0.4), (5, 0.2)]
    adsr_params = {'attack': 0.2, 'decay': 0.1, 'sustain_level': 0.9, 'release': 0.4}

    t = np.linspace(0., duration_sec, duration_samples, endpoint=False)
    # Frequency modulation: linear change from start_freq to end_freq
    freq_mod = np.linspace(start_freq, end_freq, duration_samples)

    # Generate audio with modulated frequency
    phase = np.cumsum(2 * np.pi * freq_mod / sr)
    modulated_tone = np.sin(phase)

    # Apply harmonics (simplified)
    harmonic_tone = np.zeros_like(modulated_tone)
    total_amp = 0
    for h_num, h_amp in harmonics:
        harmonic_tone += h_amp * np.sin(h_num * phase)
        total_amp += h_amp
    if total_amp > 0: harmonic_tone /= total_amp

    # Apply ADSR envelope
    envelope_samples = int((duration_sec + adsr_params['release']) * sr)
    envelope = generate_adsr_envelope(envelope_samples, **adsr_params, sr=sr)
    audio = harmonic_tone * envelope[:duration_samples]

    # Label: Use the start and end frequency, split halfway for simplicity
    # Again, frame-based labels would be more accurate
    mid_time = duration_sec / 2.0
    labels: List[Label] = [
        (adsr_params['attack'], mid_time, (start_freq + end_freq) / 2) # Approximation
        # A single label representing the whole glide might be better depending on eval method
        # (adsr_params['attack'], duration_sec, (start_freq + end_freq) / 2) # Alternative label
    ]
    # For evaluation, a single label representing the entire glide might be more practical
    single_label: List[Label] = [(adsr_params['attack'], duration_sec, (start_freq + end_freq) / 2)]


    save_audio_and_label(filename_base, audio, single_label, sr=sr)

def generate_slow_attack(filename_base: str = "15_slow_attack", midi_note: int = 65, sr: int = SR):
    """非常に遅いアタックを持つ単一ノートを生成"""
    duration_sec = 5.0
    duration_samples = int(duration_sec * sr)
    freq = midi_to_hz(midi_note) # F4
    harmonics = [(1, 1.0), (2, 0.6), (4, 0.3)] # String-like harmonics
    # Slow attack ADSR
    adsr_params = {'attack': 1.5, 'decay': 0.5, 'sustain_level': 0.8, 'release': 1.0}

    envelope_samples = int((duration_sec + adsr_params['release']) * sr)
    envelope = generate_adsr_envelope(envelope_samples, **adsr_params, sr=sr)
    tone = generate_harmonic_tone(freq, envelope_samples, harmonics, amp=1.0, sr=sr)
    audio = tone * envelope
    audio = audio[:duration_samples] # Trim to exact duration before normalization in save

    # Label onset starts at the beginning of the attack phase
    labels: List[Label] = [(0.0, duration_sec - adsr_params['release'], freq)] # Offset before release starts
    # Alternative label considering attack time:
    # labels: List[Label] = [(adsr_params['attack'], duration_sec - adsr_params['release'], freq)]
    # Use 0.0 as onset for now, as per guideline (perceived start)

    save_audio_and_label(filename_base, audio, [(0.0, duration_sec, freq)], sr=sr)

def generate_staccato(filename_base: str = "16_staccato", sr: int = SR):
    """非常に短い（スタッカート）ノートのシーケンスを生成"""
    notes = [(60, 0.1), (64, 0.1), (67, 0.1), (60, 0.1), (72, 0.15), (71, 0.1), (69, 0.1)]
    harmonics = [(1, 1.0), (3, 0.5), (5, 0.2)]
    # Very short ADSR, quick release
    adsr_params = {'attack': 0.01, 'decay': 0.03, 'sustain_level': 0.5, 'release': 0.05}
    audio, labels = _generate_melody_audio_labels(notes, harmonics, adsr_params, sr=sr, gap_sec=0.05)

    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_pianissimo(filename_base: str = "17_pianissimo", sr: int = SR):
    """非常に小さい音量（ピアニッシモ）のノートシーケンスを生成"""
    notes = [(62, 0.8), (65, 0.8), (69, 1.0), (65, 0.8), (62, 1.0)]
    harmonics = [(1, 1.0), (2, 0.3), (3, 0.15)]
    adsr_params = {'attack': 0.08, 'decay': 0.3, 'sustain_level': 0.6, 'release': 0.4}
    # Generate audio with very low amplitude scale
    audio, labels = _generate_melody_audio_labels(notes, harmonics, adsr_params, amp_scale=0.05, sr=sr)

    # Save without normalization to keep it quiet
    # save_audio_and_label(filename_base, audio, labels, sr=sr) # This would normalize
    # Manually save without normalization
    audio_path = os.path.join(OUTPUT_AUDIO_DIR, f"{filename_base}.wav")
    label_path = os.path.join(OUTPUT_LABEL_DIR, f"{filename_base}.csv")
    try:
        # Ensure audio doesn't clip even at low volume, just in case
        clipped_audio = np.clip(audio, -1.0, 1.0)
        sf.write(audio_path, clipped_audio.astype(np.float32), sr)
        logging.info(f"音声を保存しました (低音量): {audio_path}")
    except Exception as e: logging.error(f"音声ファイルの保存に失敗しました {audio_path}: {e}")
    # Save labels as usual
    try:
        with open(label_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for onset, offset, freq in labels:
                writer.writerow([f"{onset:.6f}", f"{offset:.6f}", f"{freq:.3f}"])
        logging.info(f"ラベルを保存しました: {label_path}")
    except Exception as e: logging.error(f"ラベルファイルの保存に失敗しました {label_path}: {e}")

def generate_octave_errors(filename_base: str = "18_octave_errors", sr: int = SR):
    """オクターブ違いの音が隣接するシーケンスを生成"""
    # C4, C5, G4, G5, E4, E5, C4
    notes = [(60, 0.6), (72, 0.6), (67, 0.6), (79, 0.6), (64, 0.6), (76, 0.6), (60, 1.0)]
    harmonics = [(1, 1.0), (2, 0.4), (3, 0.2)] # Simple harmonics
    adsr_params = {'attack': 0.03, 'decay': 0.1, 'sustain_level': 0.8, 'release': 0.15}
    audio, labels = _generate_melody_audio_labels(notes, harmonics, adsr_params, sr=sr, gap_sec=0.02)

    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_inharmonicity(filename_base: str = "19_inharmonicity", midi_note: int = 60, sr: int = SR):
    """倍音がわずかにずれる（インハーモニシティ）単一ノートを生成 (ピアノ模倣)"""
    duration_sec = 4.0
    duration_samples = int(duration_sec * sr)
    base_freq = midi_to_hz(midi_note) # C4
    adsr_params = {'attack': 0.01, 'decay': 1.5, 'sustain_level': 0.3, 'release': 0.5} # Piano-like envelope

    # Inharmonicity: slightly stretch higher harmonics
    # B is the inharmonicity coefficient (higher means more stretch)
    B = 0.0005
    harmonics_inharmonic: List[Tuple[float, float]] = []
    for i in range(1, 9): # Generate first 8 harmonics
        harmonic_num = i
        # Stretched frequency based on f_n = f_0 * n * sqrt(1 + B*n^2)
        stretched_freq = base_freq * harmonic_num * np.sqrt(1 + B * (harmonic_num**2))
        # Amplitude decreases for higher harmonics
        harmonic_amp = 1.0 / (harmonic_num**1.2)
        harmonics_inharmonic.append((stretched_freq, harmonic_amp))

    # Generate tone using individual sine waves for stretched harmonics
    tone = np.zeros(duration_samples)
    total_amp = 0
    for freq, amp in harmonics_inharmonic:
        # Ensure frequency is positive
        if freq > 0:
           tone += generate_sine_wave(freq, duration_samples, amp, sr)
           total_amp += amp
    if total_amp > 0: tone /= total_amp # Normalize amplitudes

    # Apply ADSR envelope
    envelope_samples = int((duration_sec + adsr_params['release']) * sr)
    envelope = generate_adsr_envelope(envelope_samples, **adsr_params, sr=sr)
    audio = tone * envelope[:duration_samples]

    # Label uses the base frequency
    labels: List[Label] = [(adsr_params['attack'], duration_sec, base_freq)]

    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_legato(filename_base: str = "20_legato", sr: int = SR):
    """ノート間が滑らかに繋がる（レガート/クロスフェード）シーケンスを生成"""
    notes = [(60, 1.0), (62, 1.0), (64, 1.0), (65, 1.5), (64, 1.0), (62, 1.0), (60, 1.5)]
    harmonics = [(1, 1.0), (2, 0.5), (3, 0.3)]
    # Slightly longer release to encourage overlap
    adsr_params = {'attack': 0.05, 'decay': 0.2, 'sustain_level': 0.7, 'release': 0.3}

    # Generate with negative gap (overlap)
    # Note: _generate_melody_audio_labels' current implementation might clip releases if overlap is too large.
    # A small negative gap simulates crossfade.
    audio, labels = _generate_melody_audio_labels(notes, harmonics, adsr_params, sr=sr, gap_sec=-0.05)

    save_audio_and_label(filename_base, audio, labels, sr=sr)

def generate_vocal_imitation(filename_base: str = "21_vocal_imitation", sr: int = SR):
    logging.info(f"Generating vocal imitation: {filename_base}")
    notes = [
        (0.5, 1.5, 60), # C4
        (1.7, 2.5, 62), # D4
        (2.6, 3.8, 64), # E4
        (4.0, 4.6, 67), # G4
        (4.7, 5.5, 65), # F4
        (5.7, 6.5, 64), # E4
    ]
    # Basic harmonics for a slightly richer tone
    harmonics = [(1, 1.0), (2, 0.4), (3, 0.2)]
    total_duration_sec = 7.0
    total_samples = int(total_duration_sec * sr)
    audio_out = np.zeros(total_samples)
    labels = []

    # LFO for vibrato/pitch variation
    lfo_freq_pitch = 5.0 # Hz
    lfo_amp_cents = 30.0 # Cents range for pitch vibrato
    lfo_phase_offset_pitch = np.random.rand() * 2 * np.pi # Randomize start phase

    # LFO for amplitude variation (tremolo)
    lfo_freq_amp = 7.0 # Hz
    lfo_amp_level = 0.1 # Amplitude variation depth (0 to 1)
    lfo_phase_offset_amp = np.random.rand() * 2 * np.pi

    # Formant filter simulation (example values for a vowel-like sound)
    try:
        formant_freqs = [800, 1200, 2400]
        formant_bandwidths = [80, 100, 150]
        formant_filter = generate_formant_filter(formant_freqs, formant_bandwidths, sr)
        apply_filter = True
    except Exception as e:
        logging.warning(f"Failed to generate formant filter: {e}. Skipping filter.")
        apply_filter = False

    # ADSR for each note
    adsr_params = {'attack': 0.05, 'decay': 0.1, 'sustain_level': 0.7, 'release': 0.15}

    current_phase = 0.0

    try: # Add try block for the main generation loop
        for i, (onset, offset, midi_note) in enumerate(notes):
            assert 0 <= onset < offset <= total_duration_sec, f"Invalid note timing: {onset}, {offset}"
            assert midi_note > 0, f"Invalid MIDI note: {midi_note}"

            start_sample = int(onset * sr)
            end_sample = int(offset * sr)
            note_duration_samples = end_sample - start_sample
            assert note_duration_samples > 0, f"Note duration must be positive: {note_duration_samples}"

            base_freq = midi_to_hz(midi_note)
            labels.append((onset, offset, base_freq))

            # Time vector for this note
            t_note = np.arange(note_duration_samples) / sr
            t_global = (start_sample + np.arange(note_duration_samples)) / sr

            # Calculate frequency with pitch LFO
            pitch_mod_cents = lfo_amp_cents * np.sin(2 * np.pi * lfo_freq_pitch * t_global + lfo_phase_offset_pitch)
            current_freq = base_freq * (2.0 ** (pitch_mod_cents / 1200.0))
            # Ensure frequency is positive
            current_freq = np.maximum(current_freq, 1e-6)

            # Calculate phase delta and instantaneous phase
            phase_delta = 2 * np.pi * current_freq / sr
            instantaneous_phase = current_phase + np.cumsum(phase_delta)
            current_phase = instantaneous_phase[-1] # Store last phase for next note (smoother transitions?)

            # Generate base harmonic tone using instantaneous phase
            note_wave = np.zeros(note_duration_samples)
            total_harmonic_amp = sum(h_amp for _, h_amp in harmonics)
            if total_harmonic_amp < 1e-9: total_harmonic_amp = 1.0 # Avoid div by zero

            for harmonic_num, harmonic_amp in harmonics:
                # Use instantaneous phase for each harmonic
                note_wave += (harmonic_amp / total_harmonic_amp) * np.sin(harmonic_num * instantaneous_phase)

            # Apply ADSR envelope
            envelope = generate_adsr_envelope(note_duration_samples, **adsr_params, sr=sr)
            note_wave *= envelope

            # Apply amplitude LFO (tremolo)
            amp_mod = 1.0 - lfo_amp_level * (0.5 * (1 + np.sin(2 * np.pi * lfo_freq_amp * t_global + lfo_phase_offset_amp)))
            note_wave *= amp_mod

            # Write to output buffer, clipping indices just in case
            write_start = np.clip(start_sample, 0, total_samples)
            write_end = np.clip(end_sample, 0, total_samples)
            write_len = write_end - write_start
            if write_len > 0:
                if write_len == len(note_wave):
                    audio_out[write_start:write_end] += note_wave
                elif write_len < len(note_wave):
                    logging.warning(f"Note {i} wave truncated during write ({len(note_wave)} -> {write_len})")
                    audio_out[write_start:write_end] += note_wave[:write_len]
                else: # write_len > len(note_wave) - should not happen with clip
                    logging.warning(f"Note {i} wave shorter than write segment ({len(note_wave)} vs {write_len})")
                    audio_out[write_start : write_start + len(note_wave)] += note_wave
        else:
                logging.warning(f"Calculated zero write length for note {i} at {onset=}, {offset=}")

        # Apply formant filter if generated successfully
        if apply_filter:
            audio_out = apply_formant_filter(audio_out, formant_filter, sr)

        # Add some noise
        audio_out = add_white_noise(audio_out, snr_db=30)

        save_audio_and_label(filename_base, audio_out, labels, sr=sr)

    except AssertionError as ae:
        logging.error(f"Assertion failed during vocal imitation generation: {ae}", exc_info=True)
        raise SynthesizerError(f"Assertion failed: {ae}") from ae
    except Exception as e:
        logging.error(f"Error generating vocal imitation {filename_base}: {e}", exc_info=True)
        # Optionally save whatever was generated before the error?
        # save_audio_and_label(f"{filename_base}_partial", audio_out[:current_sample], labels, sr=sr)
        raise SynthesizerError(f"Vocal imitation generation failed: {e}") from e

def generate_clicks(filename_base: str = "22_clicks", sr: int = SR):
    """基本的なメロディにクリックノイズを付加"""
    # Use the basic harmonic sequence as the base
    base_audio, labels = _generate_harmonic_sequence_audio_labels(sr=sr)
    # Add clicks
    audio_with_clicks = add_click_noise(base_audio.copy(), click_prob=0.002, click_amp=0.6, sr=sr)

    save_audio_and_label(filename_base, audio_with_clicks, labels, sr=sr)

def generate_pitch_bend(filename_base: str = "14b_pitch_bend", midi_note: int = 60, bend_range_cents: float = 80.0, sr: int = SR):
    logging.info(f"Generating pitch bend: {filename_base}")
    assert midi_note > 0, "MIDI note must be positive"
    # Bend range can be positive or negative

    duration_sec = 3.0
    duration_samples = int(duration_sec * sr)
    audio_out = np.zeros(duration_samples)
    labels = []

    base_freq = midi_to_hz(midi_note)
    labels.append((0.0, duration_sec, base_freq)) # Label with the starting pitch

    # Bend curve: slow sine bend up and down
    t = np.linspace(0, duration_sec, duration_samples, endpoint=False)
    # Bend factor from 0 (no bend) to 1 (full bend range)
    bend_factor = 0.5 * (1 - np.cos(2 * np.pi * (1.0 / duration_sec) * t))
    current_bend_cents = bend_range_cents * bend_factor

    # Calculate frequency with bend
    current_freq = base_freq * (2.0 ** (current_bend_cents / 1200.0))
    # Ensure frequency is positive
    current_freq = np.maximum(current_freq, 1e-6)

    # Calculate phase delta and instantaneous phase
    phase_delta = 2 * np.pi * current_freq / sr
    # Check for extreme phase delta values that might indicate instability
    max_phase_delta = np.max(np.abs(phase_delta))
    if max_phase_delta > np.pi: # If instantaneous freq exceeds Nyquist, can cause aliasing/instability
        logging.warning(f"Maximum phase delta {max_phase_delta:.4f} exceeds pi. Potential instability/aliasing.")
        # Consider clipping freq or handling differently

    try:
        instantaneous_phase = np.cumsum(phase_delta)
    except Exception as e:
        logging.error(f"Error calculating cumulative sum for phase: {e}", exc_info=True)
        # Handle error, e.g., return zeros or raise
        raise SynthesizerError(f"Phase calculation failed: {e}") from e

    # Generate sine wave using instantaneous phase
    audio_out = np.sin(instantaneous_phase)

    # Apply envelope
    envelope = generate_adsr_envelope(duration_samples, attack=0.1, decay=0.2, sustain_level=0.8, release=0.5, sr=sr)
    # Ensure envelope length matches audio_out length
    if len(envelope) > len(audio_out):
        envelope = envelope[:len(audio_out)]
    elif len(envelope) < len(audio_out):
        envelope = np.pad(envelope, (0, len(audio_out) - len(envelope)))
    audio_out *= envelope

    # Add noise
    audio_out = add_white_noise(audio_out, snr_db=35)

    save_audio_and_label(filename_base, audio_out, labels, sr=sr)

# --- Helper functions to generate audio/labels without saving (used internally) ---

def _generate_melody_audio_labels(notes: List[Tuple[int, float]], harmonics: List[Tuple[int, float]], adsr_params: dict, amp_scale: float = 1.0, sr: int = SR, gap_sec: float = 0.0) -> Tuple[np.ndarray, List[Label]]:
    """メロディの音声データとラベルリストを生成して返す (ギャップ追加対応)"""
    # Calculate total duration including gaps
    total_note_duration_sec = sum(dur for _, dur in notes)
    num_gaps = max(0, len(notes) - 1)
    total_duration_sec = total_note_duration_sec + num_gaps * gap_sec
    total_duration_samples = int(total_duration_sec * sr)

    audio = np.zeros(total_duration_samples)
    labels: List[Label] = []
    current_time_samples = 0
    release_time_sec = adsr_params.get('release', 0.1) # Get release time for buffer calc

    # Calculate buffer needed for final release
    buffer_samples = int(release_time_sec * sr)
    # Adjust buffer based on gap - ensure enough space for release tail
    padded_audio_len = total_duration_samples + buffer_samples
    padded_audio = np.zeros(padded_audio_len)

    for i, (midi_note, duration_sec) in enumerate(notes):
        freq = midi_to_hz(midi_note)
        note_duration_samples = int(duration_sec * sr)
        envelope_samples = int((duration_sec + release_time_sec) * sr)
        envelope = generate_adsr_envelope(envelope_samples, **adsr_params, sr=sr)
        tone = generate_harmonic_tone(freq, envelope_samples, harmonics, amp=amp_scale, sr=sr)
        note_wave = tone * envelope

        onset_time_sec = current_time_samples / sr
        # Offset calculation modified to include a portion of the release phase
        release_time_sec = adsr_params.get('release', 0.1)
        perceived_release_duration_sec = release_time_sec * 0.8
        end_sample = current_time_samples + note_duration_samples
        offset_time_sec = end_sample / sr + perceived_release_duration_sec
        # Ensure offset doesn't exceed the total padded audio length
        offset_time_sec = min(offset_time_sec, padded_audio_len / sr)
        
        end_sample = current_time_samples + note_duration_samples # Theoretical end of note sustain
        envelope_end_sample_in_context = current_time_samples + len(note_wave)

        # Ensure write operation stays within bounds of padded_audio
        write_end = min(len(padded_audio), envelope_end_sample_in_context)
        write_len = write_end - current_time_samples
        if write_len > 0:
           padded_audio[current_time_samples:write_end] += note_wave[:write_len]

        labels.append((onset_time_sec, offset_time_sec, freq))

        # Move time marker to the start of the next note (including gap)
        current_time_samples += note_duration_samples # Move time based on sustain duration

    # Trim the final audio, keeping the buffer for the last note's release
    # Return audio potentially longer than total_duration_sec due to final release tail
    return padded_audio[:padded_audio_len], labels

def _generate_harmonic_sequence_audio_labels(sr: int = SR) -> Tuple[np.ndarray, List[Label]]:
    notes = [(60, 0.3), (64, 0.3), (67, 0.3), (72, 0.6), (71, 0.3), (67, 0.3), (64, 0.3), (60, 0.6)]
    harmonics = [(1, 1.0), (3, 0.5), (5, 0.25), (7, 0.1)]
    adsr_params = {'attack': 0.05, 'decay': 0.15, 'sustain_level': 0.6, 'release': 0.2}
    return _generate_melody_audio_labels(notes, harmonics, adsr_params, sr=sr)

def _generate_polyphony_audio_labels(sr: int = SR) -> Tuple[np.ndarray, List[Label]]:
    voice1_notes = [(72, 0.2), (71, 0.2), (69, 0.3), (71, 0.2), (72, 0.4), (67, 0.5), (69, 0.7)]
    voice1_harmonics = [(1, 1.0), (3, 0.6)]
    voice1_adsr = {'attack': 0.02, 'decay': 0.1, 'sustain_level': 0.7, 'release': 0.1}
    voice2_notes = [(60, 0.5), (64, 0.5), (55, 1.0), (59, 0.5), (60, 1.0)]
    voice2_harmonics = [(1, 1.0), (2, 0.4), (4, 0.2)]
    voice2_adsr = {'attack': 0.08, 'decay': 0.2, 'sustain_level': 0.6, 'release': 0.3}

    audio1, labels1 = _generate_melody_audio_labels(voice1_notes, voice1_harmonics, voice1_adsr, amp_scale=0.6, sr=sr)
    audio2, labels2 = _generate_melody_audio_labels(voice2_notes, voice2_harmonics, voice2_adsr, amp_scale=0.7, sr=sr)

    len1 = len(audio1)
    len2 = len(audio2)
    max_len = max(len1, len2)

    # Pad shorter audio to match length
    if len1 < max_len:
        audio1 = np.pad(audio1, (0, max_len - len1))
    if len2 < max_len:
        audio2 = np.pad(audio2, (0, max_len - len2))

    combined_audio = audio1 + audio2
    combined_labels = sorted(labels1 + labels2)
    return combined_audio, combined_labels 