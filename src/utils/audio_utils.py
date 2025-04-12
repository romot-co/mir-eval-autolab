"""
音声ファイルと参照データの読み込みを一元化するユーティリティモジュール
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import logging
from typing import Dict, Any, Tuple, Optional, Union
import json
import csv

from src.utils.exception_utils import log_exception

logger = logging.getLogger(__name__)


def load_audio_file(audio_path: str, logger: Optional[logging.Logger] = None) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    音声ファイルを読み込む共通関数

    Parameters
    ----------
    audio_path : str
        音声ファイルのパス
    logger : Optional[logging.Logger], optional
        ロガーオブジェクト, by default None

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[int]]
        (音声データ, サンプリングレート)のタプル、エラー時はNoneを含む
    """
    try:
        # サンプリングレートを44.1kHzに統一する
        audio, sr = librosa.load(audio_path, sr=44100, mono=True)
        
        if logger:
            logger.info(f"音声ファイルを読み込みました: {audio_path} (sr={sr}Hz, length={len(audio)/sr:.2f}秒)")
            
        return audio, sr
    except Exception as e:
        error_msg = f"音声ファイル '{audio_path}' の読み込み中にエラーが発生しました: {e}"
        if logger:
            log_exception(logger, e, error_msg)
        else:
            print(error_msg)
        return None, None


def load_reference_data(ref_path: str) -> Dict[str, np.ndarray]:
    """
    参照データファイルを読み込む関数
    
    Parameters
    ----------
    ref_path : str
        参照データファイルのパス
        
    Returns
    -------
    Dict[str, np.ndarray]
        参照データを含む辞書。エラー時は空の配列を返します。
        {
            'intervals': np.ndarray,  # shape=(N, 2), 各行は [onset, offset]
            'note_pitches': np.ndarray  # shape=(N,), 各ノートのピッチ値（Hz）
        }
    """
    logger.info(f"参照データファイルを読み込みます: {ref_path}")
    
    # 拡張子を取得
    ext = os.path.splitext(ref_path)[1].lower()
    
    try:
        if ext == '.csv':
            # CSVファイルの場合: まずヘッダーありで試す
            try:
                df = pd.read_csv(ref_path, header=0)
                # 期待される列名があるか確認
                expected_columns = ['onset', 'offset', 'pitch']
                has_expected_columns = all(col in df.columns for col in expected_columns)
            except Exception: # ヘッダーがないか、読み込みエラーの場合
                has_expected_columns = False

            # ヘッダーがない、または期待される列名がない場合、ヘッダーなしで再読み込み
            if not has_expected_columns:
                try:
                    df = pd.read_csv(ref_path, header=None)
                    if len(df.columns) >= 3:
                         # 位置に基づいて列名を割り当てる
                        df.columns = ['onset', 'offset', 'pitch'] + [f'col_{i}' for i in range(3, len(df.columns))]
                        logger.warning(f"列名が見つからないか不正なため、位置でデータを判断します: {ref_path}")
                    else:
                        # 列数が足りない場合はエラー
                        raise ValueError(f"ヘッダーなしで読み込んだが、列数が3未満です: {len(df.columns)}")
                except Exception as e_no_header:
                    logger.error(f"CSVファイルの読み込みに失敗しました (ヘッダー有無両方): {ref_path}, Error: {e_no_header}")
                    return {
                        'intervals': np.array([]).reshape(0, 2),
                        'note_pitches': np.array([])
                    }

            logger.debug(f"CSVファイルの列: {df.columns.tolist()}")
            logger.debug(f"CSVファイルの行数: {len(df)}")
            
            # データが空の場合
            if df.empty:
                logger.warning(f"参照データファイルが空です: {ref_path}")
                return {
                    'intervals': np.array([]).reshape(0, 2),
                    'note_pitches': np.array([])
                }
            
            # 必要な列があるか確認 (ヘッダーあり/なしで処理済みのはずだが念のため)
            if 'onset' in df.columns and 'offset' in df.columns and 'pitch' in df.columns:
                onsets = df['onset'].values
                offsets = df['offset'].values
                pitches = df['pitch'].values
            elif len(df.columns) >= 3:
                # 列名がなければ位置で判断 (ヘッダーなしの場合にここで割り当て済みのはず)
                logger.warning(f"列名が見つからないため、位置でデータを判断します: {df.columns.tolist()}")
                onsets = df.iloc[:, 0].values
                offsets = df.iloc[:, 1].values
                pitches = df.iloc[:, 2].values
            else:
                logger.error(f"CSVファイルのフォーマットが不正です (列数不足): {ref_path}")
                return {
                    'intervals': np.array([]).reshape(0, 2),
                    'note_pitches': np.array([])
                }
            
            # 無効なデータを除外（負の時間や無効なピッチ）
            valid_mask = (onsets >= 0) & (offsets > onsets) & np.isfinite(pitches)
            if not np.all(valid_mask):
                logger.warning(f"{np.sum(~valid_mask)}個の無効なノートを除外します")
                # 無効なデータが含まれる場合は空の配列を返す
                return {
                    'intervals': np.array([]).reshape(0, 2),
                    'note_pitches': np.array([])
                }
            
            # データの長さを確認して一致させる
            min_len = min(len(onsets), len(offsets), len(pitches))
            if min_len < len(onsets) or min_len < len(offsets) or min_len < len(pitches):
                logger.warning(f"データ長が一致しません (onsets: {len(onsets)}, offsets: {len(offsets)}, pitches: {len(pitches)})。最小値に合わせます: {min_len}")
                onsets = onsets[:min_len]
                offsets = offsets[:min_len]
                pitches = pitches[:min_len]
            
            # 区間配列の作成
            intervals = np.column_stack((onsets, offsets))
            
            logger.info(f"参照データを読み込みました: {len(intervals)}個のノート")
            return {
                'intervals': intervals,
                'note_pitches': pitches
            }
            
        elif ext == '.json':
            # JSONファイルの場合
            with open(ref_path, 'r') as f:
                data = json.load(f)
                
            # JSONデータをチェック
            if 'intervals' in data:
                intervals = np.array(data['intervals'])
                
                # ピッチデータの取得
                if 'note_pitches' in data:
                    pitches = np.array(data['note_pitches'])
                elif 'pitches' in data:
                    pitches = np.array(data['pitches'])
                else:
                    logger.warning(f"JSONファイルにピッチデータがありません: {ref_path}")
                    pitches = np.array([])
                
                logger.info(f"参照データを読み込みました: {len(intervals)}個のノート")
                
                return {
                    'intervals': intervals,
                    'note_pitches': pitches
                }
            else:
                logger.error(f"JSONファイルに必要なキーがありません: {ref_path}")
                return {
                    'intervals': np.array([]).reshape(0, 2),
                    'note_pitches': np.array([])
                }
        else:
            logger.error(f"未対応のファイル形式です: {ext}")
            return {
                'intervals': np.array([]).reshape(0, 2),
                'note_pitches': np.array([])
            }
            
    except Exception as e:
        logger.error(f"参照データの読み込み中にエラーが発生しました: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'intervals': np.array([]).reshape(0, 2),
            'note_pitches': np.array([])
        }


def load_audio_and_reference(audio_path: str, ref_path: str, logger: Optional[logging.Logger] = None) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Optional[int]]:
    """
    音声ファイルと参照データをまとめて読み込む。

    Parameters
    ----------
    audio_path : str
        音声ファイルのパス
    ref_path : str
        参照データファイルのパス
    logger : Optional[logging.Logger], optional
        ロガーオブジェクト, by default None

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Optional[int]]
        (音声データ, 参照データ, サンプリングレート)のタプル
        音声データと参照データはエラー時にNoneになる可能性がある
    """
    audio, sr = load_audio_file(audio_path, logger)
    ref_data = None
    
    if ref_path and os.path.exists(ref_path):
        ref_data = load_reference_data(ref_path)
    elif logger:
        logger.warning(f"参照ファイル {ref_path} が見つからないか指定されていません。")
    
    return audio, ref_data, sr


def make_output_path(output_dir: str, audio_file: str, detector_name: str, extension: str = "json") -> str:
    """
    出力ファイルのパスを生成する

    Parameters
    ----------
    output_dir : str
        出力ディレクトリ
    audio_file : str
        音声ファイルのパス
    detector_name : str
        検出器の名前
    extension : str, optional
        出力ファイルの拡張子, by default "json"

    Returns
    -------
    str
        出力ファイルのパス
    """
    # ディレクトリ作成はこの関数の責務ではないため削除
    # os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    
    # "_results" サフィックスがある場合は削除（ファイル名の末尾のみ）
    if base_name.endswith("_results"):
        base_name = base_name[:-8]  # "_results"の長さは8
        
    # 拡張子の先頭のドットを除去
    clean_extension = extension.lstrip('.')
    return os.path.join(output_dir, f"{base_name}_{detector_name}.{clean_extension}")