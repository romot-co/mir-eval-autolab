"""
音声データと参照データの生成・読み込みモジュール

このモジュールは、音声データと参照データの生成および読み込みのためのユーティリティ関数を提供します。
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
from typing import List, Dict, Tuple, Any, Optional
from scipy import signal
import logging

logger = logging.getLogger(__name__)


def generate_sample_audio(sr: int = 44100, duration: float = 10.0, num_notes: int = 20,
                         start_silence: float = 0.0, end_silence: float = 0.0,
                         notes: Optional[List[Dict[str, float]]] = None,
                         noise_level: float = 0.01) -> Tuple[np.ndarray, List[Dict[str, float]], int]:
    """
    サンプル音声ファイルを生成する
    
    Parameters
    ----------
    sr : int, optional
        サンプリングレート, by default 44100
    duration : float, optional
        音声の長さ（秒）, by default 10.0
    num_notes : int, optional
        生成する音符の数, by default 20
    start_silence : float, optional
        開始前の無音時間（秒）, by default 0.0
    end_silence : float, optional
        終了後の無音時間（秒）, by default 0.0
    notes : Optional[List[Dict[str, float]]], optional
        生成する音符リスト（指定しない場合は自動生成）, by default None
    noise_level : float, optional
        追加するノイズのレベル, by default 0.01
    
    Returns
    -------
    Tuple[np.ndarray, List[Dict[str, float]], int]
        生成された音声データ、音符リスト、サンプリングレート
    """
    # 音符が指定されていない場合は自動生成
    if notes is None:
        # 可能なピッチの候補（ドレミファソラシド + オクターブ）
        possible_pitches = [
            262.0, 294.0, 330.0, 349.0, 392.0, 440.0, 494.0,  # C4-B4
            523.0, 587.0, 659.0, 698.0, 784.0, 880.0, 988.0   # C5-B5
        ]
        
        # 特殊な周波数も追加（低音と高音）
        special_pitches = [55.0, 110.0, 1760.0, 3520.0]  # 低音と高音
        
        # 音符の総数からいくつか特殊な周波数にするか決定
        special_count = max(1, num_notes // 10)  # 少なくとも1つは特殊な音符を入れる
        
        # 音符をランダムに生成
        notes = []
        attempts = 0
        
        while len(notes) < num_notes and attempts < 1000:  # 無限ループ防止
            # 音符の開始時刻と終了時刻（無音パディングを考慮）
            effective_duration = duration - start_silence - end_silence
            onset = start_silence + np.random.uniform(0, effective_duration * 0.9)
            note_length = np.random.uniform(0.2, 0.6)  # 音符の長さ
            offset = min(onset + note_length, start_silence + effective_duration)
            
            # 音符同士の重なりをチェック（許容値：10ms）
            overlap = False
            overlap_tolerance = 0.01  # 10ms
            
            for existing_note in notes:
                if (onset - overlap_tolerance <= existing_note['offset'] and 
                    offset + overlap_tolerance >= existing_note['onset']):
                    overlap = True
                    break
            
            if overlap:
                attempts += 1
                continue
            
            # ピッチの選択（特殊な周波数を一部含める）
            if len(notes) < special_count:
                freq = np.random.choice(special_pitches)
            else:
                freq = np.random.choice(possible_pitches)
            
            # 音量（ベロシティ）
            velocity = np.random.uniform(0.5, 1.0)
            
            # 音符の情報を追加
            notes.append({
                'onset': onset,
                'offset': offset,
                'pitch': freq,
                'velocity': velocity
            })
            
            attempts += 1
    
    # 音声データの初期化（無音パディングを含む全長）
    total_duration = duration + start_silence + end_silence
    num_samples = int(total_duration * sr)
    audio = np.zeros(num_samples)
    
    # 各音符を正弦波で生成し、音声データに加算
    for note in notes:
        onset_sample = int(note['onset'] * sr)
        offset_sample = int(note['offset'] * sr)
        freq = note['pitch']  # 周波数（Hz）
        velocity = note.get('velocity', 0.8)  # デフォルトのベロシティ
        
        # 範囲チェック
        if offset_sample > num_samples:
            offset_sample = num_samples
        if onset_sample >= offset_sample:
            continue
        
        # 音符の長さに応じた時間配列を作成
        t = np.arange(offset_sample - onset_sample) / sr
        
        # エンベロープ（ADSR: Attack, Decay, Sustain, Release）
        note_length = (offset_sample - onset_sample) / sr
        attack_time = min(0.02, note_length * 0.1)  # 20ms or 10% of note length
        decay_time = min(0.05, note_length * 0.2)   # 50ms or 20% of note length
        release_time = min(0.1, note_length * 0.3)  # 100ms or 30% of note length
        
        attack_samples = int(attack_time * sr)
        decay_samples = int(decay_time * sr)
        release_samples = int(release_time * sr)
        sustain_samples = (offset_sample - onset_sample) - attack_samples - decay_samples - release_samples
        
        # サステインが負にならないように調整
        if sustain_samples < 0:
            # 比率を保ってスケーリング
            total = attack_samples + decay_samples + release_samples
            ratio = (offset_sample - onset_sample) / total
            attack_samples = int(attack_samples * ratio)
            decay_samples = int(decay_samples * ratio)
            release_samples = (offset_sample - onset_sample) - attack_samples - decay_samples
            sustain_samples = 0
        
        # エンベロープ生成
        env = np.ones(offset_sample - onset_sample)
        
        # アタック部分
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # ディケイ部分（サステインレベルまで下がる）
        sustain_level = 0.7
        if decay_samples > 0:
            env[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain_level, decay_samples)
        
        # サステイン部分（一定レベルを保つ）
        if sustain_samples > 0:
            env[attack_samples+decay_samples:attack_samples+decay_samples+sustain_samples] = sustain_level
        
        # リリース部分
        if release_samples > 0:
            env[-release_samples:] = np.linspace(sustain_level if sustain_samples > 0 else 1, 0, release_samples)
        
        # 正弦波を生成し、エンベロープとベロシティを適用
        note_signal = np.sin(2 * np.pi * freq * t) * env * velocity
        
        # 高調波を追加（より自然な音色に）
        harmonics = [0.5, 0.25, 0.125]  # 第2, 第3, 第4倍音の振幅
        for i, amp in enumerate(harmonics, 2):
            # 倍音の周波数がナイキスト周波数を超えないことを確認
            if freq * i < sr / 2:
                harmonic_signal = amp * np.sin(2 * np.pi * freq * i * t) * env * velocity
                note_signal += harmonic_signal
        
        # 音声データに加算
        audio[onset_sample:offset_sample] += note_signal
    
    # クリッピング防止のためにスケーリング
    if np.max(np.abs(audio)) > 0:
        audio = 0.9 * audio / np.max(np.abs(audio))
    
    # 少量のノイズを追加（よりリアルな検出チャレンジのため）
    if noise_level > 0:
        noise = np.random.randn(num_samples) * noise_level
        audio = audio + noise
    
    # エイリアシング防止のためのローパスフィルタ
    nyquist = sr / 2
    cutoff = 0.9 * nyquist  # ナイキスト周波数の90%をカットオフに設定
    b, a = signal.butter(4, cutoff / nyquist, 'low')
    audio = signal.filtfilt(b, a, audio)
    
    # 最終的なスケーリング
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio, notes, sr


def save_reference_data(output_path: str, notes: List[Dict[str, float]]) -> None:
    """
    参照データ（音符リスト）をCSVファイルとして保存
    
    Parameters
    ----------
    output_path : str
        出力ファイルパス
    notes : List[Dict[str, float]]
        音符リスト
    """
    # 音符リストをDataFrameに変換
    notes_df = pd.DataFrame(notes)
    
    # 必要なカラムのみ抽出
    required_columns = ['onset', 'offset', 'pitch']
    for col in required_columns:
        if col not in notes_df.columns:
            notes_df[col] = 0.0  # デフォルト値
    
    # CSVに保存
    notes_df[required_columns].to_csv(output_path, index=False)
    print(f"参照データを保存しました: {output_path}")


def load_audio_and_reference(audio_path: str, ref_path: str) -> Tuple[np.ndarray, List[Dict[str, float]], int]:
    """
    音声ファイルと参照データを読み込む
    
    Parameters
    ----------
    audio_path : str
        音声ファイルのパス
    ref_path : str
        参照データのパス（CSVまたはMIDI）
    
    Returns
    -------
    Tuple[np.ndarray, List[Dict[str, float]], int]
        音声データ、音符リスト、サンプリングレート
    """
    # 音声ファイルの読み込み
    audio, sr = sf.read(audio_path)
    
    # ステレオの場合はモノラルに変換
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        print(f"ステレオ音声をモノラルに変換します: {audio_path}")
        audio = np.mean(audio, axis=1)
    
    # 参照データの読み込み
    if ref_path.endswith('.csv'):
        # CSV形式の参照データ
        notes_df = pd.read_csv(ref_path)
        notes = []
        
        for _, row in notes_df.iterrows():
            note = {
                'onset': row['onset'],
                'offset': row['offset'],
                'pitch': row['pitch']
            }
            # オプションのカラム
            if 'velocity' in row:
                note['velocity'] = row['velocity']
                
            notes.append(note)
            
    elif ref_path.endswith('.mid') or ref_path.endswith('.midi'):
        # MIDI形式の参照データ
        try:
            # MIDI機能は現在未使用
            # import pretty_midi
            # midi_data = pretty_midi.PrettyMIDI(ref_path)
            # notes = []
            # 
            # for instrument in midi_data.instruments:
            #     for note in instrument.notes:
            #         notes.append({
            #             'onset': note.start,
            #             'offset': note.end,
            #             'pitch': pretty_midi.note_number_to_hz(note.pitch),
            #             'velocity': note.velocity / 127.0  # MIDIのベロシティを0-1の範囲に正規化
            #         })
            # MIDI機能は無効化されています
            logger.warning("MIDI形式のファイルサポートは無効化されています")
            notes = []
        except ImportError:
            raise ImportError("MIDIファイルの読み込みには pretty_midi パッケージが必要です。")
    else:
        raise ValueError(f"サポートされていないファイル形式です: {ref_path}")
    
    return audio, notes, sr 