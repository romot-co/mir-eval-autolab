import sys
import os
import logging

# Add src directory to Python path if run from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_generation import synthesizer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("合成データ生成を開始します...")
    # 出力ディレクトリを確認・作成
    synthesizer.ensure_output_dirs()

    # --- 既存のテストケース ---
    synthesizer.generate_basic_sine_sequence()
    synthesizer.generate_harmonic_sequence()
    synthesizer.generate_dynamics_change()
    synthesizer.generate_pitch_modulation() # Note: Existing one is step-like
    synthesizer.generate_noisy_sequence()
    synthesizer.generate_reverb_sequence()
    synthesizer.generate_chords()
    synthesizer.generate_polyphony()
    synthesizer.generate_percussion_only()
    synthesizer.generate_mixed_melody_percussion()
    synthesizer.generate_complex_mix()

    # --- 新しく追加したテストケース ---
    logging.info("新しいテストケースの生成を開始...")
    synthesizer.generate_smooth_vibrato()
    # synthesizer.generate_portamento() # Remove original portamento
    synthesizer.generate_pitch_bend(filename_base="14_pitch_bend") # Rename 14b to 14

    # --- Start Debugging: Check if 14_pitch_bend files exist immediately after generation call ---
    audio_check_path = os.path.join("data", "synthesized", "audio", "14_pitch_bend.wav")
    label_check_path = os.path.join("data", "synthesized", "labels", "14_pitch_bend.csv")
    if os.path.exists(audio_check_path):
        logging.info(f"DEBUG CHECK in generate_all: 14_pitch_bend.wav EXISTS at {audio_check_path}")
    else:
        logging.error(f"DEBUG CHECK in generate_all: 14_pitch_bend.wav DOES NOT EXIST at {audio_check_path}")
    if os.path.exists(label_check_path):
        logging.info(f"DEBUG CHECK in generate_all: 14_pitch_bend.csv EXISTS at {label_check_path}")
    else:
        logging.error(f"DEBUG CHECK in generate_all: 14_pitch_bend.csv DOES NOT EXIST at {label_check_path}")
    # --- End Debugging ---

    synthesizer.generate_slow_attack()
    synthesizer.generate_staccato()
    synthesizer.generate_pianissimo()
    synthesizer.generate_octave_errors()
    synthesizer.generate_inharmonicity()
    synthesizer.generate_legato()
    synthesizer.generate_vocal_imitation()
    synthesizer.generate_clicks()
    logging.info("新しいテストケースの生成を完了しました。")

    logging.info("すべての合成データ生成が完了しました。")

if __name__ == "__main__":
    main() 