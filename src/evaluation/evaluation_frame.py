"""
フレーム単位の評価関連処理を集めたモジュール

このモジュールでは、フレーム単位のピッチ評価やその他の時系列評価に関連する機能を提供します。
"""

import logging
import numpy as np
import mir_eval
from typing import Dict, Any, Union, Tuple, List, Optional
from scipy.interpolate import interp1d

from src.utils.pitch_utils import hz_to_midi, midi_to_hz

def notes_to_frames(note_intervals: np.ndarray, 
                   note_pitches: np.ndarray, 
                   hop_time: float = 0.01, 
                   end_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    ノート単位のデータをフレーム単位のデータに変換します。

    Parameters
    ----------
    note_intervals : np.ndarray
        ノート区間の配列 (N, 2) 形式で各行は [onset, offset]
    note_pitches : np.ndarray
        ノートピッチの配列 (N,) 周波数(Hz)
    hop_time : float, optional
        フレーム間の時間間隔（秒）, by default 0.01
    end_time : Optional[float], optional
        データの終了時間。Noneの場合は最後のノートのオフセットを使用, by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        時間配列とピッチ配列のタプル (times, freqs)

    注意
    ----
    この関数は、フレームごとに単一のピッチ値を生成します。ポリフォニー（複数ノートの同時発音）の場合、
    同じ時間フレームに複数のノートが存在すると、後から処理されたノートのピッチで上書きされます。
    つまり、ポリフォニー情報は失われます。これは、フレームベース評価が単一ピッチの評価を目的としているためです。
    ポリフォニーの評価には、ノートベースの評価関数（mir_eval.transcription）を使用してください。
    """
    # 入力検証
    if len(note_intervals) == 0 or len(note_pitches) == 0:
        # 空の入力の場合の処理
        if end_time is None or end_time <= 0 or hop_time > end_time:
            # end_timeが無効な場合は空の配列を返す
            return np.array([]), np.array([])
        else:
            # end_timeが有効な場合はその長さに合わせたゼロ配列を返す
            n_frames = int(np.ceil(end_time / hop_time))
            times = np.arange(n_frames) * hop_time
            freqs = np.zeros(n_frames)
            return times, freqs
    
    # 終了時間が指定されていない場合は最後のノートのオフセットを使用
    if end_time is None:
        end_time = np.max(note_intervals[:, 1])
    elif end_time <= 0 or hop_time > end_time:
        # end_timeが無効な場合は空の配列を返す
        return np.array([]), np.array([])
    
    # 全体のフレーム数を計算
    n_frames = int(np.ceil(end_time / hop_time))
    
    # 時間配列を生成
    times = np.arange(n_frames) * hop_time
    
    # ピッチ配列を初期化（デフォルトは無声=0Hz）
    freqs = np.zeros(n_frames)
    
    # 各ノートについて処理
    for i, (interval, pitch) in enumerate(zip(note_intervals, note_pitches)):
        onset, offset = interval
        
        # ノート区間に含まれるフレームインデックスを計算
        # 開始フレーム: onset以上のフレームに割り当て（境界上は含む = inclusive）
        start_idx = int(np.floor(onset / hop_time))
        
        # 終了フレーム: offset未満のフレームまで含む（境界上は含まない = exclusive）
        end_idx = int(np.ceil(offset / hop_time)) - 1
        
        # インデックス範囲をチェック
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, n_frames - 1)
        
        # 有効なフレーム範囲がある場合のみ設定
        if start_idx <= end_idx:
            freqs[start_idx:end_idx+1] = pitch
    
    return times, freqs

def evaluate_frame_pitches(ref_times: np.ndarray,
                         ref_freqs: np.ndarray,
                         est_times: np.ndarray,
                         est_freqs: np.ndarray,
                         pitch_tolerance: float = 50.0,
                         use_pitch_chroma: bool = False) -> Dict[str, float]:
    """
    mir_eval.melodyを使用してフレーム単位のピッチ評価を行います。

    Parameters
    ----------
    ref_times : np.ndarray
        参照データの時刻配列
    ref_freqs : np.ndarray
        参照データの周波数配列 (Hz)
    est_times : np.ndarray
        推定データの時刻配列
    est_freqs : np.ndarray
        推定データの周波数配列 (Hz)
    pitch_tolerance : float, optional
        ピッチ評価の許容誤差 (セント), by default 50.0
    use_pitch_chroma : bool, optional
        ピッチクロマ（オクターブ違いを無視）を使用するか, by default False

    Returns
    -------
    Dict[str, float]
        評価結果の辞書。以下のキーを含みます：
        - voicing_recall: 有声フレームの再現率
        - voicing_false_alarm: 有声判定の誤報率
        - raw_pitch_accuracy: ピッチの正確度（有声フレームのみ）
        - raw_chroma_accuracy: クロマの正確度（有声フレームのみ）
        - overall_accuracy: 総合精度（有声/無声判定込み）
    """
    logger = logging.getLogger('evaluate')
    
    # 空データの扱いを統一: 参照と推定が両方空なら完全一致とみなす
    if (len(ref_times) == 0 and len(ref_freqs) == 0) and (len(est_times) == 0 and len(est_freqs) == 0):
        return {
            'voicing_recall': 1.0,
            'voicing_false_alarm': 0.0,
            'raw_pitch_accuracy': 1.0,
            'raw_chroma_accuracy': 1.0,
            'overall_accuracy': 1.0
        }
    
    # どちらか一方だけが空の場合に対処
    if len(ref_times) == 0 or len(ref_freqs) == 0:
        ref_times = np.array([0.0])
        ref_freqs = np.array([0.0])  # 無音
    
    if len(est_times) == 0 or len(est_freqs) == 0:
        est_times = np.array([0.0])
        est_freqs = np.array([0.0])  # 無音
    
    # データの検証
    if len(ref_times) != len(ref_freqs):
        logger.warning(f"参照データの時刻数とピッチ数が一致しません: {len(ref_times)} vs {len(ref_freqs)}")
        return {
            'voicing_recall': 0.0,
            'voicing_false_alarm': 1.0,
            'raw_pitch_accuracy': 0.0,
            'raw_chroma_accuracy': 0.0,
            'overall_accuracy': 0.0
        }
    
    if len(est_times) != len(est_freqs):
        logger.warning(f"推定データの時刻数とピッチ数が一致しません: {len(est_times)} vs {len(est_freqs)}")
        return {
            'voicing_recall': 0.0,
            'voicing_false_alarm': 1.0,
            'raw_pitch_accuracy': 0.0,
            'raw_chroma_accuracy': 0.0,
            'overall_accuracy': 0.0
        }
    
    try:
        # mir_eval.melodyを使用してフレーム評価を行う
        # まず参照と推定のデータをcent単位に変換し、voicing配列を取得
        ref_voiced, ref_cents, est_voiced, est_cents = mir_eval.melody.to_cent_voicing(
            ref_times, ref_freqs,
            est_times, est_freqs,
            base_frequency=10.0,  # デフォルト値
            hop=None,  # 自動で適切なホップサイズを計算
            kind='linear'  # 線形補間
        )
        
        # Voicing評価
        voicing_recall, voicing_false_alarm = mir_eval.melody.voicing_measures(
            ref_voiced, est_voiced
        )
        
        # ピッチ評価
        raw_pitch_accuracy = mir_eval.melody.raw_pitch_accuracy(
            ref_voiced, ref_cents,
            est_voiced, est_cents,
            cent_tolerance=pitch_tolerance
        )
        
        # クロマ評価
        raw_chroma_accuracy = mir_eval.melody.raw_chroma_accuracy(
            ref_voiced, ref_cents,
            est_voiced, est_cents,
            cent_tolerance=pitch_tolerance
        )
        
        # 総合精度
        overall_accuracy = mir_eval.melody.overall_accuracy(
            ref_voiced, ref_cents,
            est_voiced, est_cents,
            cent_tolerance=pitch_tolerance
        )
        
        # 結果を辞書にまとめて返す
        return {
            'voicing_recall': float(voicing_recall),
            'voicing_false_alarm': float(voicing_false_alarm),
            'raw_pitch_accuracy': float(raw_pitch_accuracy),
            'raw_chroma_accuracy': float(raw_chroma_accuracy),
            'overall_accuracy': float(overall_accuracy)
        }
    except Exception as e:
        logger.error(f"フレーム評価中にエラーが発生しました: {str(e)}")
        
        # エラー時のデフォルト値
        return {
            'voicing_recall': 0.0,
            'voicing_false_alarm': 1.0, # False alarm rate is 1 if all frames are incorrect/silent
            'raw_pitch_accuracy': 0.0,
            'raw_chroma_accuracy': 0.0,
            'overall_accuracy': 0.0
        } 