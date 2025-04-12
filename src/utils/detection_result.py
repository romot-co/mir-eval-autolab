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
    intervals: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    
    # 各ノートのピッチ値の配列 (Hz) shape=(N,)
    note_pitches: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    
    # フレーム単位の情報（mir_evalのframe-based評価用）
    # フレーム時刻の配列 (秒) shape=(M,)
    frame_times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    
    # 各フレームのピッチ値の配列 (Hz) shape=(M,)
    frame_frequencies: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    
    # 検出器の名前
    detector_name: str = "Unknown"
    
    # 検出にかかった時間 (秒)
    detection_time: float = 0.0
    
    # その他の検出データを保持する辞書
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初期化後にデータの整合性を検証する"""
        logger.debug("DetectionResult __post_init__ 開始")
        # --- intervals の検証 ---
        if self.intervals.size > 0:
            logger.debug("intervals 検証開始")
            # 負の値チェック
            if np.any(self.intervals < 0):
                logger.error("インターバル時間に負の値が含まれています: %s", self.intervals)
                raise ValueError("インターバル時間に負の値が含まれています")
            # 開始時刻 <= 終了時刻 チェック
            if np.any(self.intervals[:, 0] > self.intervals[:, 1]):
                invalid_indices = np.where(self.intervals[:, 0] > self.intervals[:, 1])[0]
                logger.error("インターバルの開始時刻が終了時刻より後になっています (indices: %s): %s", invalid_indices, self.intervals[invalid_indices])
                raise ValueError("インターバルの開始時刻が終了時刻より後になっています")
            logger.debug("intervals 検証終了")

        # --- note_pitches の検証 ---
        if self.note_pitches.size > 0:
            logger.debug("note_pitches 検証開始")
            # 負の値チェック (MIDIピッチとして)
            if np.any(self.note_pitches < 0):
                 # Note: 周波数(Hz)なら 0 以下でチェックだが、テストデータはMIDI値のようなので < 0 でチェック
                 logger.error("note_pitches に負の値が含まれています: %s", self.note_pitches)
                 raise ValueError("note_pitches に負の値が含まれています")
            logger.debug("note_pitches 検証終了")

        # --- frame_times の検証 ---
        if self.frame_times.size > 0:
            logger.debug("frame_times 検証開始")
            # 負の値チェック
            if np.any(self.frame_times < 0):
                logger.error("frame_times に負の値が含まれています: %s", self.frame_times)
                raise ValueError("frame_times に負の値が含まれています")
            # 単調増加チェック
            diffs = np.diff(self.frame_times)
            if np.any(diffs < 0):
                 invalid_indices = np.where(diffs < 0)[0]
                 logger.error("frame_times が単調増加ではありません (indices: %s): %s", invalid_indices, self.frame_times)
                 raise ValueError("frame_times が単調増加ではありません")
            logger.debug("frame_times 検証終了")

        # --- frame_frequencies の検証 ---
        if self.frame_frequencies.size > 0:
             logger.debug("frame_frequencies 検証開始")
             # 負の値チェック
             if np.any(self.frame_frequencies < 0):
                 logger.error("frame_frequencies に負の値が含まれています: %s", self.frame_frequencies)
                 raise ValueError("frame_frequencies に負の値が含まれています")
             logger.debug("frame_frequencies 検証終了")

        # --- 長さの一致検証 ---
        if len(self.intervals) != len(self.note_pitches):
            logger.error(f"intervals ({len(self.intervals)}) と note_pitches ({len(self.note_pitches)}) の長さが一致しません。")
            raise ValueError("intervals と note_pitches の長さが一致しません。")

        if len(self.frame_times) != len(self.frame_frequencies):
             # フレームデータはオプショナルなので警告に留めるか、厳密にするか？ -> ここでは警告とする
             # raise ValueError("frame_times と frame_frequencies の長さが一致しません。")
             logger.warning(f"frame_times ({len(self.frame_times)}) と frame_frequencies ({len(self.frame_frequencies)}) の長さが一致しません。")

        logger.debug("DetectionResult __post_init__ 終了")

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
            ValueError: 必要なキーが辞書にない場合、または初期化に失敗した場合
        """
        # デバッグ用に受け取った辞書のキーを出力
        logging.debug("==== 検出器から受け取った結果のキー (from_dict) ====")
        for key, value in detection_dict.items():
            # (ログ出力は省略)
            pass 
        
        # 必須キーが存在することを確認
        missing_keys = [key for key in cls.ESSENTIAL_KEYS if key not in detection_dict]
        if missing_keys:
            logging.error(f"検出結果に必須キーが不足しています: {missing_keys}")
            raise ValueError(f"検出結果に必須キーが不足しています: {missing_keys}")
        
        # コンストラクタへの引数を準備
        init_args = {}
        additional_data_collected = {}
        all_known_keys = set(cls.ESSENTIAL_KEYS) | set(cls.OPTIONAL_KEYS)

        # 必須キーとオプションキーを処理
        for key in all_known_keys:
            if key in detection_dict:
                value = detection_dict[key]
                # 数値配列フィールドは np.asarray で変換
                if key in ['intervals', 'note_pitches', 'frame_times', 'frame_frequencies']:
                    try:
                        # dtype=np.float64 を明示的に指定
                        init_args[key] = np.asarray(value, dtype=np.float64)
                    except Exception as e:
                        logger.warning(f"フィールド '{key}' の NumPy 配列(float64)への変換中にエラー: {e}。そのまま渡します。")
                        init_args[key] = value # そのまま渡し、__post_init__での検証に任せる
                else:
                    init_args[key] = value # 他の既知フィールドはそのままコピー
        
        # additional_data を処理 (入力辞書に直接指定されている場合も考慮)
        if 'additional_data' in detection_dict and isinstance(detection_dict['additional_data'], dict):
             additional_data_collected.update(detection_dict['additional_data'])
        
        # 既知のキー以外を additional_data に追加
        for key, value in detection_dict.items():
            if key not in all_known_keys and key != 'additional_data':
                additional_data_collected[key] = value
                
        if additional_data_collected: # additional_data が空でなければ引数に追加
             init_args['additional_data'] = additional_data_collected
        
        # 準備した引数でコンストラクタを呼び出す
        # これにより、__init__ -> __post_init__ が正しい値で実行される
        try:
            result = cls(**init_args)
        except TypeError as e:
            # キーの不一致などで TypeError が発生する可能性
            logger.error(f"DetectionResult の初期化中に TypeError: {e}. 引数キー: {list(init_args.keys())}")
            raise ValueError(f"DetectionResult の初期化に失敗しました: {e}") from e
        except ValueError as e:
             # __post_init__ で発生した ValueError をそのまま再raise
             logger.error(f"DetectionResult の __post_init__ で検証エラー: {e}")
             raise e
        except Exception as e:
             # その他の予期せぬエラー
             logger.error(f"DetectionResult の初期化中に予期せぬエラー: {e}")
             raise ValueError(f"DetectionResult の初期化中に予期せぬエラーが発生しました: {e}") from e

        # デバッグ用に生成されたオブジェクトの情報をログ出力（任意）
        logging.debug("==== DetectionResult オブジェクト生成完了 (from_dict) ====")
        # logging.debug(f"intervals shape: {result.intervals.shape}")
        # logging.debug(f"note_pitches shape: {result.note_pitches.shape}")
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """検出結果を辞書に変換する

        Returns
        -------
        Dict[str, Any]
            検出結果の辞書
        """
        result_dict = {}
        
        # 必須キー
        for key in self.ESSENTIAL_KEYS:
            result_dict[key] = getattr(self, key)
        
        # オプションキー
        for key in self.OPTIONAL_KEYS:
            value = getattr(self, key)
            if value is not None:
                result_dict[key] = value
        
        # additional_dataの内容を追加
        for key, value in self.additional_data.items():
            result_dict[key] = value
        
        return result_dict
    
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