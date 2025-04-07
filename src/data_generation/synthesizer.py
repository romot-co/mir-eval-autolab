import numpy as np
import librosa
import soundfile as sf
import os
import csv
import logging
from typing import List, Tuple, Optional, Union
from scipy.signal import butter, filtfilt, fftconvolve

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
    if freq <= 0: return np.zeros(duration_samples)
    t = np.linspace(0., duration_samples / sr, duration_samples, endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)

def generate_harmonic_tone(base_freq: float, duration_samples: int, harmonics: List[Tuple[int, float]], amp: float = 1.0, sr: int = SR) -> np.ndarray:
    tone = np.zeros(duration_samples)
    if base_freq <= 0: return tone
    total_harmonic_amp = 0
    for harmonic_num, harmonic_amp in harmonics:
        freq = base_freq * harmonic_num
        tone += generate_sine_wave(freq, duration_samples, harmonic_amp, sr)
        total_harmonic_amp += harmonic_amp
    if total_harmonic_amp > 1.0: tone /= total_harmonic_amp # Normalize
    return amp * tone

def save_audio_and_label(filename_base: str, audio_data: np.ndarray, labels: List[Label], include_header: bool = False, sr: int = SR):
    # --- Start Debugging ---
    try:
        # Check for NaN or Inf in audio data
        if not np.all(np.isfinite(audio_data)):
            logging.error(f"INVALID AUDIO DATA for {filename_base}: Contains NaN or Inf.")
            # Optionally return here or replace invalid values
            # return 
            audio_data = np.nan_to_num(audio_data) # Replace NaN with 0, Inf with large numbers

        if audio_data.size == 0:
            logging.error(f"EMPTY AUDIO DATA for {filename_base}. Skipping save.")
            return

        logging.debug(f"Saving {filename_base}: Audio shape={audio_data.shape}, Size={audio_data.size}, Min={np.min(audio_data):.4f}, Max={np.max(audio_data):.4f}, Mean={np.mean(audio_data):.4f}")
        logging.debug(f"Saving {filename_base}: Labels={labels}")
    except Exception as debug_e:
        logging.error(f"Error during pre-save debug logging for {filename_base}: {debug_e}")
    # --- End Debugging ---

    peak_amp = np.max(np.abs(audio_data))
    # Avoid division by zero or near-zero, ensure data is valid before division
    normalized_audio = audio_data * (AMP_MAX / peak_amp) if peak_amp > 1e-9 and np.all(np.isfinite(audio_data)) else audio_data

    audio_path = os.path.join(OUTPUT_AUDIO_DIR, f"{filename_base}.wav")
    label_path = os.path.join(OUTPUT_LABEL_DIR, f"{filename_base}.csv")

    # --- Audio Saving ---
    try:
        logging.debug(f"Attempting to write audio to: {audio_path}")
        sf.write(audio_path, normalized_audio.astype(np.float32), sr)
        logging.info(f"音声を保存しました: {audio_path}")
        # --- Start Debugging ---
        # Verify immediately after write attempt
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            logging.debug(f"VERIFIED: Audio file exists immediately after saving: {audio_path}, Size: {file_size} bytes")
            if file_size == 0:
                logging.warning(f"WARNING: Audio file {audio_path} was created but is empty (0 bytes).")
        else:
            logging.error(f"FAILED VERIFICATION: Audio file DOES NOT exist after saving attempt: {audio_path}")
        # --- End Debugging ---
    except Exception as e:
        logging.error(f"音声ファイルの保存に失敗しました {audio_path}: {e}")
        # --- Start Debugging ---
        logging.exception(f"Exception details during audio save for {filename_base}:") # Log stack trace
        # --- End Debugging ---

    # --- Label Saving ---
    try:
        logging.debug(f"Attempting to write labels to: {label_path}")
        with open(label_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if include_header:
                writer.writerow(['onset', 'offset', 'frequency'])
            # Ensure labels list is not empty before proceeding
            if not labels:
                 logging.warning(f"No labels provided for {filename_base}. CSV file will be empty or header-only.")
            for onset, offset, freq_or_marker in labels:
                # Format frequency carefully, handle potential non-float markers
                freq_str = f"{freq_or_marker:.3f}" if isinstance(freq_or_marker, (float, np.floating)) else str(freq_or_marker)
                writer.writerow([f"{onset:.6f}", f"{offset:.6f}", freq_str])
        logging.info(f"ラベルを保存しました: {label_path}")
        # --- Start Debugging ---
        # Verify immediately after write attempt
        if os.path.exists(label_path):
             file_size = os.path.getsize(label_path)
             logging.debug(f"VERIFIED: Label file exists immediately after saving: {label_path}, Size: {file_size} bytes")
             if file_size == 0 and labels: # Check if empty despite having labels
                 logging.warning(f"WARNING: Label file {label_path} was created but is empty (0 bytes), despite having labels.")
        else:
            logging.error(f"FAILED VERIFICATION: Label file DOES NOT exist after saving attempt: {label_path}")
        # --- End Debugging ---
    except Exception as e:
        logging.error(f"ラベルファイルの保存に失敗しました {label_path}: {e}")
        # --- Start Debugging ---
        logging.exception(f"Exception details during label save for {filename_base}:") # Log stack trace
        # --- End Debugging ---

def add_white_noise(audio_data: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(audio_data ** 2)
    if signal_power < 1e-9: return audio_data
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
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
    """
    if len(ir) == 0 or mix_level <= 0:
        # IRがない、またはミックスレベルが0なら元の音声を返す
        return audio_data
    if mix_level > 1.0:
        mix_level = 1.0

    # 畳み込みを実行し、リバーブ信号を生成
    # mode='full' で全長を取得
    try:
        reverb_signal = fftconvolve(audio_data, ir, mode='full')
    except ValueError as e:
        logging.error(f"fftconvolve中にエラーが発生しました: {e}. IR Length: {len(ir)}, Audio Length: {len(audio_data)}")
        return audio_data # エラー時は元の音声を返す

    # 期待される出力長
    target_len = len(audio_data) + len(ir) - 1

    # 念のため長さを確認・調整（通常は不要だが数値誤差対策）
    if len(reverb_signal) < target_len:
        reverb_signal = np.pad(reverb_signal, (0, target_len - len(reverb_signal)))
    elif len(reverb_signal) > target_len:
        reverb_signal = reverb_signal[:target_len]

    # ドライ信号をリバーブ信号の長さに合わせてパディング
    padded_dry = np.pad(audio_data, (0, target_len - len(audio_data)))

    # ドライ信号とウェット信号をミックス
    # Wet/Dry Mix: output = dry * (1 - mix) + wet * mix
    mixed_signal = padded_dry * (1.0 - mix_level) + reverb_signal * mix_level

    # 注意: ここではピークノーマライズは行わない。
    #       save_audio_and_label での最終的なノーマライズに任せる。
    # peak_amp = np.max(np.abs(mixed_signal))
    # if peak_amp > 1.0:
    #    mixed_signal /= peak_amp

    logging.debug(f"Reverb applied. Original len: {len(audio_data)}, IR len: {len(ir)}, Output len: {len(mixed_signal)}")
    return mixed_signal

def generate_kick_sound(duration_sec: float = 0.3, freq: float = 60.0, decay_rate: float = 15.0, amp: float = 1.0, sr: int = SR) -> np.ndarray:
    duration_samples = int(duration_sec * sr)
    t = np.linspace(0., duration_sec, duration_samples, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    envelope = np.exp(-decay_rate * t)
    kick = wave * envelope
    return kick * amp

def generate_noise_perc_sound(duration_sec: float = 0.2, bandpass_freqs: Optional[Tuple[float, float]] = None, decay_rate: float = 25.0, amp: float = 1.0, sr: int = SR) -> np.ndarray:
    duration_samples = int(duration_sec * sr)
    noise = np.random.randn(duration_samples)
    if bandpass_freqs:
        low, high = bandpass_freqs
        nyquist = 0.5 * sr
        low_norm = low / nyquist
        high_norm = high / nyquist
        if low_norm <= 0: low_norm = 0.01
        if high_norm >= 1: high_norm = 0.99
        if low_norm >= high_norm: high_norm = low_norm + 0.01
        try:
            b, a = butter(4, [low_norm, high_norm], btype='band')
            noise = filtfilt(b, a, noise)
        except ValueError as e:
            logging.warning(f"Butterworth filter failed ({bandpass_freqs}): {e}. Using unfiltered noise.")
    t = np.linspace(0., duration_sec, duration_samples, endpoint=False)
    envelope = np.exp(-decay_rate * t)
    perc = noise * envelope
    return perc * amp

# --- Helper functions ---

def generate_formant_filter(formant_freqs: List[float], bandwidths: List[float], sr: int = SR):
    """簡易的なフォルマントフィルタ（複数のバンドパスフィルタの合成）を生成"""
    # Note: This is a very simplified formant simulation
    num_samples = sr # Arbitrary length for impulse response generation
    impulse = np.zeros(num_samples)
    impulse[0] = 1.0
    filtered_impulse = np.zeros(num_samples)
    nyquist = 0.5 * sr

    for i, f in enumerate(formant_freqs):
        bw = bandwidths[i]
        f_low = f - bw / 2
        f_high = f + bw / 2
        # Normalize frequencies
        low_norm = f_low / nyquist
        high_norm = f_high / nyquist
        if low_norm <= 0: low_norm = 0.01
        if high_norm >= 1: high_norm = 0.99
        if low_norm >= high_norm: high_norm = low_norm + 0.01
        try:
            # Design a bandpass filter for this formant
            b, a = butter(2, [low_norm, high_norm], btype='band')
            # Apply filter to impulse and add to the total filtered impulse
            filtered_impulse += filtfilt(b, a, impulse)
        except ValueError as e:
            logging.warning(f"Butterworth filter failed for formant {f} Hz: {e}")
            # If filter fails, add the original impulse (less ideal)
            filtered_impulse += impulse * 0.1 # Add attenuated impulse

    # Normalize the combined impulse response
    if np.max(np.abs(filtered_impulse)) > 1e-9:
        filtered_impulse /= np.max(np.abs(filtered_impulse))

    # Trim the impulse response (heuristic)
    non_zero_indices = np.where(np.abs(filtered_impulse) > 1e-4)[0]
    if len(non_zero_indices) > 0:
       last_significant = non_zero_indices[-1]
       filtered_impulse = filtered_impulse[:last_significant + int(0.05*sr)] # Keep a small tail
    else:
       filtered_impulse = filtered_impulse[:int(0.1*sr)] # Default short length if all near zero

    return filtered_impulse

def add_click_noise(audio_data: np.ndarray, click_prob: float = 0.001, click_amp: float = 0.5, sr: int = SR) -> np.ndarray:
    """音声データにランダムなクリックノイズを追加"""
    clicks = np.zeros_like(audio_data)
    num_clicks = int(len(audio_data) * click_prob)
    click_indices = np.random.randint(0, len(audio_data), num_clicks)
    # Clicks are short impulses (1-2 samples)
    clicks[click_indices] = (np.random.rand(num_clicks) * 2 - 1) * click_amp
    # Optional: make clicks slightly wider
    if len(click_indices) > 0:
       click_indices_plus1 = np.clip(click_indices + 1, 0, len(audio_data) - 1)
       clicks[click_indices_plus1] += (np.random.rand(num_clicks) * 2 - 1) * click_amp * 0.5

    return audio_data + clicks

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
    """ピッチの揺らぎと簡易フォルマントを持つボーカル模倣音を生成"""
    # Simple melody
    notes = [(69, 0.8), (71, 0.6), (69, 0.8), (67, 1.0), (69, 0.6), (72, 0.8), (71, 1.2)] # A4 based melody
    adsr_params = {'attack': 0.08, 'decay': 0.2, 'sustain_level': 0.7, 'release': 0.3}

    # Vibrato parameters
    vibrato_rate_hz = 6.0
    vibrato_depth_cents = 15.0

    # --- Generate base audio with pitch fluctuation ---
    total_duration_sec = sum(dur for _, dur in notes)
    total_duration_samples = int(total_duration_sec * sr)
    labels: List[Label] = []
    current_time_samples = 0
    release_time_sec = adsr_params.get('release', 0.3) # Use the defined release time
    # Calculate buffer needed for final release tail of the *last* note
    buffer_samples = int(release_time_sec * sr)
    padded_audio_len = total_duration_samples + buffer_samples
    padded_audio = np.zeros(padded_audio_len) # Correctly initialize audio buffer

    # Add slight random pitch fluctuation and vibrato per note
    for i, (midi_note, duration_sec) in enumerate(notes):
        base_freq = midi_to_hz(midi_note)
        note_duration_samples = int(duration_sec * sr)
        # Total samples needed for this note's envelope including release
        envelope_samples_note = int((duration_sec + release_time_sec) * sr)

        t = np.linspace(0., duration_sec, note_duration_samples, endpoint=False)

        # Subtle random walk for pitch variation within a note (+/- 15 cents max)
        pitch_variation_cents = np.cumsum(np.random.randn(note_duration_samples) * 0.1) # Small steps
        pitch_variation_cents = np.clip(pitch_variation_cents, -15, 15)

        # Add sinusoidal vibrato
        vibrato_cents = vibrato_depth_cents * np.sin(2 * np.pi * vibrato_rate_hz * t)
        total_pitch_variation_cents = pitch_variation_cents + vibrato_cents

        # Apply total pitch variation
        freq_mod = base_freq * (2**(total_pitch_variation_cents / 1200))

        # Phase calculation using modulated frequency
        phase = np.cumsum(2 * np.pi * freq_mod / sr)

        # Basic harmonics for vocal-like sound (richer than pure sine)
        # Generate for sustain part only first
        harmonic_tone_sustain = np.sin(phase) + 0.4 * np.sin(2*phase) + 0.2 * np.sin(3*phase)
        harmonic_tone_sustain /= 1.6 # Normalize roughly

        # Apply ADSR envelope for the note segment
        envelope_note = generate_adsr_envelope(envelope_samples_note, **adsr_params, sr=sr)

        # Apply envelope to sustain part
        note_wave_sustain = harmonic_tone_sustain * envelope_note[:note_duration_samples]

        # Generate waveform for release part separately
        release_samples = envelope_samples_note - note_duration_samples
        note_wave_release = np.array([]) # Initialize release wave
        if release_samples > 0:
            # Estimate phase continuation for release (using last frequency)
            last_freq = freq_mod[-1] if len(freq_mod) > 0 else base_freq
            # Start phase for release is the end phase of sustain
            release_start_phase = phase[-1] if len(phase) > 0 else 0
            release_phase_increment = np.cumsum(2 * np.pi * last_freq / sr * np.ones(release_samples))
            release_phase = release_start_phase + release_phase_increment

            # Generate harmonics for release part
            harmonic_tone_release = np.sin(release_phase) + 0.4 * np.sin(2*release_phase) + 0.2 * np.sin(3*release_phase)
            harmonic_tone_release /= 1.6 # Normalize roughly

            # Apply envelope to release part (ensure lengths match)
            env_release_part = envelope_note[note_duration_samples:]
            min_len_release = min(len(harmonic_tone_release), len(env_release_part))
            # Use the sliced parts for calculation
            note_wave_release = harmonic_tone_release[:min_len_release] * env_release_part[:min_len_release]

        # --- BUG FIX: Write generated wave parts to the main audio buffer ---
        # Ensure indices are within bounds
        sustain_end_idx = current_time_samples + len(note_wave_sustain)
        if sustain_end_idx <= padded_audio_len:
            padded_audio[current_time_samples:sustain_end_idx] += note_wave_sustain
        else:
            # Handle boundary case if sustain part overflows
            safe_len = padded_audio_len - current_time_samples
            if safe_len > 0:
                padded_audio[current_time_samples:] += note_wave_sustain[:safe_len]

        release_start_idx = sustain_end_idx
        release_end_idx = release_start_idx + len(note_wave_release)
        if release_end_idx <= padded_audio_len:
             # Ensure release_start_idx is also valid before writing
             if release_start_idx < padded_audio_len:
                padded_audio[release_start_idx:release_end_idx] += note_wave_release
        else:
            # Handle boundary case if release part overflows
             if release_start_idx < padded_audio_len:
                safe_len = padded_audio_len - release_start_idx
                if safe_len > 0:
                    padded_audio[release_start_idx:] += note_wave_release[:safe_len]
        # --- End Bug Fix ---

        # Label generation
        onset_time_sec = current_time_samples / sr
        # Use the originally defined release time for offset calculation consistency
        # perceived_release_duration_sec = adsr_params['release'] * 0.8 # Use 0.3s
        # Let's use the _generate_melody_audio_labels calculation style for consistency
        perceived_release_duration_sec = adsr_params.get('release', 0.1) * 0.8 # Consistent with other funcs? Let's try 0.3 * 0.8
        perceived_release_duration_sec = 0.3 * 0.8
        offset_time_sec = (current_time_samples + note_duration_samples) / sr + perceived_release_duration_sec
        offset_time_sec = min(offset_time_sec, padded_audio_len / sr) # Ensure offset doesn't exceed audio length

        labels.append((onset_time_sec, offset_time_sec, base_freq))
        # Update current time based on note duration (not envelope duration)
        current_time_samples += note_duration_samples # Move time based on sustain duration

    # --- Apply simplified formant filter ---
    # Ensure audio is normalized before filtering to avoid clipping/scaling issues
    if np.max(np.abs(padded_audio)) > 0:
      padded_audio /= np.max(np.abs(padded_audio)) * 1.1 # Normalize with slight headroom

    formant_freqs = [700, 1200, 2500]
    bandwidths = [100, 150, 200]
    formant_ir = generate_formant_filter(formant_freqs, bandwidths, sr)
    # Apply formant filter via convolution
    # Use mode='same' to keep length consistent, might have slight edge effects
    audio_formant = fftconvolve(padded_audio, formant_ir, mode='same')

    # Final normalization after filtering
    if np.max(np.abs(audio_formant)) > 0:
        audio_formant /= np.max(np.abs(audio_formant)) # Normalize to [-1, 1]

    save_audio_and_label(filename_base, audio_formant, labels, sr=sr)

def generate_clicks(filename_base: str = "22_clicks", sr: int = SR):
    """基本的なメロディにクリックノイズを付加"""
    # Use the basic harmonic sequence as the base
    base_audio, labels = _generate_harmonic_sequence_audio_labels(sr=sr)
    # Add clicks
    audio_with_clicks = add_click_noise(base_audio.copy(), click_prob=0.002, click_amp=0.6, sr=sr)

    save_audio_and_label(filename_base, audio_with_clicks, labels, sr=sr)

def generate_pitch_bend(filename_base: str = "14b_pitch_bend", midi_note: int = 60, bend_range_cents: float = 80.0, sr: int = SR):
    """単一ノート内でピッチが滑らかに変化する（ピッチベンド）音声を生成"""
    duration_sec = 3.0
    duration_samples = int(duration_sec * sr)
    center_freq = midi_to_hz(midi_note)

    # セントから周波数比を計算
    # bend_range_cents は変化の全幅。中心からの変化幅はその半分
    half_bend_cents = bend_range_cents / 2.0
    ratio = 2.0 ** (half_bend_cents / 1200.0)
    start_freq = center_freq / ratio
    end_freq = center_freq * ratio

    harmonics = [(1, 1.0), (3, 0.4)] # シンプルな倍音
    adsr_params = {'attack': 0.05, 'decay': 0.1, 'sustain_level': 0.9, 'release': 0.2}
    envelope_samples = int((duration_sec + adsr_params['release']) * sr)
    envelope = generate_adsr_envelope(envelope_samples, **adsr_params, sr=sr)

    # 周波数の指数的遷移 (generate_portamentoと同様)
    bend_duration_samples = duration_samples # ここでは持続時間全体で変化
    log_start_freq = np.log(start_freq)
    log_end_freq = np.log(end_freq)
    t_bend = np.linspace(0., 1., bend_duration_samples)
    # 往復させる (例: 上がって下がる)
    # freq_trajectory_up = np.exp(log_start_freq + (log_end_freq - log_start_freq) * t_bend)
    # freq_trajectory_down = np.exp(log_end_freq + (log_start_freq - log_end_freq) * t_bend)
    # freq_trajectory = np.concatenate((freq_trajectory_up[:bend_duration_samples//2], freq_trajectory_down[bend_duration_samples//2:]))
    # 単純にstartからendへ遷移
    freq_trajectory = np.exp(log_start_freq + (log_end_freq - log_start_freq) * t_bend)

    # 位相計算
    phase = np.zeros(envelope_samples)
    # freq_trajectory を envelope_samples の長さに合わせる (末尾でパッド)
    instantaneous_freq = np.pad(freq_trajectory, (0, envelope_samples - len(freq_trajectory)), 'edge') / sr
    phase[1:] = 2 * np.pi * np.cumsum(instantaneous_freq[:-1])

    # 高調波も考慮した波形生成 (基本波の位相を共有)
    tone = np.zeros(envelope_samples)
    total_harmonic_amp = sum(amp for _, amp in harmonics)
    for h_num, h_amp in harmonics:
        tone += (h_amp / total_harmonic_amp) * np.sin(phase * h_num)

    audio = tone * envelope
    audio = audio[:duration_samples]

    # ラベル: 単一ノートとして、中心周波数を記録
    labels: List[Label] = [(adsr_params['attack'], duration_sec, center_freq)]

    save_audio_and_label(filename_base, audio, labels, sr=sr)

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
        current_time_samples = end_sample + int(gap_sec * sr)

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