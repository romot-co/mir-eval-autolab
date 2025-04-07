"""
検出結果を表すデータクラス

このモジュールは、検出器からの出力を統一されたフォーマットで扱うためのデータクラスを提供します。
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

# 必須キー（検出結果に必ず含まれるべきキー）
ESSENTIAL_KEYS = ['intervals', 'note_pitches', 'frame_times', 'frame_frequencies']

# オプションキー（検出結果に含まれる可能性のあるキー）
OPTIONAL_KEYS = ['detection_time', 'detector_name']

@dataclass
class DetectionResult:
    """
    検出結果を表すデータクラス
    
    すべての検出器はこの形式で結果を返すことを期待しています。
    存在しないデータについては空の配列が設定されます。
    """
    # 必須キーとオプションキーをクラス属性として定義
    ESSENTIAL_KEYS = ESSENTIAL_KEYS
    OPTIONAL_KEYS = OPTIONAL_KEYS
    
    # ノート単位の情報（mir_evalのnote-based評価用）
    # [[onset1, offset1], [onset2, offset2], ...] の形式のノート区間 shape=(N, 2)
    intervals: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 2))
    
    # 各ノートのピッチ値の配列 (Hz) shape=(N,)
    note_pitches: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # フレーム単位の情報（mir_evalのframe-based評価用）
    # フレーム時刻の配列 (秒) shape=(M,)
    frame_times: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 各フレームのピッチ値の配列 (Hz) shape=(M,)
    frame_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 検出器の名前
    detector_name: str = "Unknown"
    
    # 検出にかかった時間 (秒)
    detection_time: float = 0.0
    
    # その他の検出データを保持する辞書
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, detection_dict: Dict[str, Any]) -> 'DetectionResult':
        """
        辞書から検出結果オブジェクトを作成する
        
        Parameters
        ----------
        detection_dict : Dict[str, Any]
            検出結果の辞書
            
        Returns
        -------
        DetectionResult
            検出結果オブジェクト
            
        Raises
        ------
            ValueError: 必要なキーが辞書にない場合
        """
        # デバッグ用に受け取った辞書のキーを出力
        logging.debug("==== 検出器から受け取った結果のキー ====")
        for key, value in detection_dict.items():
            if isinstance(value, np.ndarray):
                if value.size > 0:
                    logging.debug(f"キー '{key}': 形状={value.shape}, 長さ={len(value)}")
                    if len(value) < 10:
                        logging.debug(f"  内容: {value}")
                else:
                    logging.debug(f"キー '{key}': 形状={value.shape}, 長さ={value.size}")
                    logging.debug(f"  内容: 空の配列")
            else:
                logging.debug(f"キー '{key}': タイプ={type(value)}")
        
        # 必須キーが存在することを確認
        missing_keys = []
        for key in cls.ESSENTIAL_KEYS:
            if key not in detection_dict:
                missing_keys.append(key)
        
        # 必須キーがない場合はエラー
        if missing_keys:
            logging.error(f"検出結果に必須キーが不足しています: {missing_keys}")
            raise ValueError(f"検出結果に必須キーが不足しています: {missing_keys}")
        
        # 結果オブジェクトを初期化
        result = cls()
        
        # 各フィールドに値を設定
        # 必須キー
        for key in cls.ESSENTIAL_KEYS:
            value = detection_dict[key]
            setattr(result, key, np.asarray(value))
        
        # オプションキー
        for key in cls.OPTIONAL_KEYS:
            if key in detection_dict:
                setattr(result, key, np.asarray(detection_dict[key]) if isinstance(detection_dict[key], (list, np.ndarray)) else detection_dict[key])
        
        # データ完全性チェック
        if len(result.intervals) != len(result.note_pitches):
            logging.warning(f"検出結果の長さが一致しません: intervals={len(result.intervals)}, " +
                           f"note_pitches={len(result.note_pitches)}")
        
        # デバッグ用に変換後のデータを出力
        logging.debug("==== DetectionResultに変換後のデータ ====")
        logging.debug(f"intervals: 形状={result.intervals.shape}, 長さ={len(result.intervals)}")
        logging.debug(f"note_pitches: 形状={result.note_pitches.shape}, 長さ={len(result.note_pitches)}")
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        検出結果をdict形式に変換する
        
        Returns
        -------
        Dict[str, Any]
            検出結果の辞書
        """
        result = {
            'intervals': self.intervals,
            'note_pitches': self.note_pitches,
            'frame_times': self.frame_times,
            'frame_frequencies': self.frame_frequencies,
            'detector_name': self.detector_name,
            'detection_time': self.detection_time
        }
        
        # 追加データをマージ
        result.update(self.additional_data)
        
        return result
    
    def to_note_list(self) -> List[Dict[str, float]]:
        """
        検出結果をノートリスト形式に変換する
        
        Returns
        -------
        List[Dict[str, float]]
            ノートリスト形式の検出結果
            [{'onset': onset1, 'offset': offset1, 'pitch': pitch1}, ...]
        """
        notes = []
        
        if len(self.intervals) > 0:
            for i in range(len(self.intervals)):
                note = {
                    'onset': float(self.intervals[i, 0]),
                    'offset': float(self.intervals[i, 1]),
                }
                
                if i < len(self.note_pitches):
                    note['pitch'] = float(self.note_pitches[i])
                else:
                    note['pitch'] = 0.0
                
                notes.append(note)
        
        return notes

def extract_note_data(data: Union[Dict[str, Any], Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    検出結果または参照データから、間隔とピッチのデータを抽出します。

    Parameters
    ----------
    data : Union[Dict[str, Any], Any]
        検出結果または参照データ

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (intervals, pitches) のタプル
        - intervals: [[onset, offset], ...] の形式の2次元配列
        - pitches: [pitch1, pitch2, ...] の形式の1次元配列
    """
    # データがDictでない場合はそのまま返す
    if not isinstance(data, dict):
        if hasattr(data, 'get_note_data'):
            return data.get_note_data()
        return data
    
    # intervalsとピッチデータが含まれている場合
    if 'intervals' in data:
        pitch_data = None
        if 'note_pitches' in data:
            pitch_data = data['note_pitches']
        elif 'pitches' in data:  # 参照データは'pitches'キーを使用する場合がある
            pitch_data = data['pitches']
            
        if pitch_data is not None:
            return np.array(data['intervals']), np.array(pitch_data)
    
    # データが見つからない場合は空の配列を返す
    logger.warning("ノートデータが見つかりません")
    return np.array([]).reshape(0, 2), np.array([]) 