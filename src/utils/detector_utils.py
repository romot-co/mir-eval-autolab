#!/usr/bin/env python3
"""
検出器ユーティリティモジュール

このモジュールは、音楽検出アルゴリズムの使用を簡略化するためのユーティリティ関数を提供します。
検出器のロード、作成、検出結果の正規化などの機能を含みます。
"""

import os
import sys
import time
import glob
import importlib
import logging
import inspect
from typing import Dict, List, Any, Tuple, Optional, Type, Union, Callable
import numpy as np
import warnings
import traceback

from src.detectors.base_detector import BaseDetector
from src.utils.detection_result import DetectionResult
# 個別の検出器のインポートを削除
# 代わりにレジストリ関数をインポート
from src.detectors import get_registered_detector

logger = logging.getLogger(__name__)


def get_detector_class(detector_name: str) -> Type[BaseDetector]:
    """
    検出器名からクラスオブジェクトを取得します。

    Parameters
    ----------
    detector_name : str
        検出器名

    Returns
    -------
    Type[BaseDetector]
        検出器クラス

    Raises
    ------
    ImportError
        検出器がレジストリに見つからない場合
    """
    try:
        # レジストリから検出器クラスを取得
        detector_class = get_registered_detector(detector_name)
        if detector_class is None:
            raise ValueError(f"検出器 '{detector_name}' がレジストリに見つかりません")
            
        # BaseDetectorのサブクラスであることは登録時にチェック済みだが念のため
        try:
            if not issubclass(detector_class, BaseDetector):
                # レジストリが正しく管理されていれば、この場合は発生しないはず
                raise TypeError(f"'{detector_name}'に登録されたクラスがBaseDetectorを継承していません。")
        except TypeError as e:
            # issubclassが型チェックエラーを起こした場合（クラスオブジェクトでない可能性）
            raise ImportError(f"検出器 '{detector_name}' はBaseDetectorのサブクラスではありません: {e}")
            
        return detector_class
    except ValueError as e:
        # get_registered_detectorがValueErrorを発生させた場合
        raise ImportError(f"検出器 '{detector_name}' がレジストリに見つかりません: {e}")
    except Exception as e:
        # その他の予期せぬエラーを処理
        raise ImportError(f"検出器 '{detector_name}' の取得中にエラーが発生しました: {e}")


def normalize_detection_result(detection_result: Dict[str, Any], sr: int = None) -> Dict[str, Any]:
    """
    検出結果を標準形式に正規化します。

    Parameters
    ----------
    detection_result : Dict[str, Any]
        検出器からの生の検出結果
    sr : int, optional
        サンプリングレート, by default None

    Returns
    -------
    Dict[str, Any]
        正規化された検出結果。以下のキーを含みます：
        - intervals: np.ndarray, shape=(N, 2) - ノート区間 [onset, offset]
        - note_pitches: np.ndarray, shape=(N,) - ノートのピッチ値（Hz）
        - frame_times: np.ndarray, shape=(M,) - フレーム時刻
        - frame_frequencies: np.ndarray, shape=(M,) - フレーム単位ピッチ値（Hz）
        - detector_name: str - 検出器名
        - detection_time: float - 検出処理時間
        - additional_data: Dict - その他の追加データ
    """
    try:
        # 空の結果または無効な結果への対処
        if detection_result is None or not isinstance(detection_result, dict):
            logger.warning(f"無効な検出結果です。型: {type(detection_result)}。空の結果辞書を返します。")
            return {
                'intervals': np.array([]).reshape(0, 2),
                'note_pitches': np.array([]),
                'frame_times': np.array([]),
                'frame_frequencies': np.array([]),
                'detector_name': "Unknown",
                'detection_time': 0.0,
                'additional_data': {}
            }
        
        # 必須フィールドの定義
        required_fields = {
            'intervals': np.array([]).reshape(0, 2),
            'note_pitches': np.array([]),
            'frame_times': np.array([]),
            'frame_frequencies': np.array([]),
            'detector_name': "Unknown",
            'detection_time': 0.0,
            'additional_data': {}
        }
        
        # 結果辞書の初期化
        normalized_result = required_fields.copy()
        additional_data = {}
        
        # 必須フィールドの直接コピー
        for field in required_fields:
            if field in detection_result:
                try:
                    value = detection_result[field]
                    if isinstance(value, (list, np.ndarray)):
                        normalized_result[field] = np.asarray(value)
                    else:
                        normalized_result[field] = value
                except Exception as e:
                    logger.warning(f"フィールド {field} の変換中にエラー: {str(e)}")
        
        # intervalsとnote_pitchesの検証と修正
        if 'intervals' in detection_result and 'note_pitches' in detection_result:
            try:
                intervals = np.asarray(detection_result['intervals'], dtype=np.float64)
                pitches = np.asarray(detection_result['note_pitches'], dtype=np.float64)
                
                if len(intervals) > 0:
                    if intervals.shape[1] != 2:
                        logger.error(f"不正なインターバル形状: {intervals.shape}")
                    else:
                        # ピッチ情報の長さ調整
                        if len(pitches) != len(intervals):
                            logger.warning(f"ピッチ数 ({len(pitches)}) とインターバル数 ({len(intervals)}) が一致しません")
                            min_len = min(len(pitches), len(intervals))
                            normalized_result['note_pitches'] = pitches[:min_len]
                            normalized_result['intervals'] = intervals[:min_len]
                        else:
                            normalized_result['note_pitches'] = pitches
                            normalized_result['intervals'] = intervals
            except Exception as e:
                logger.error(f"インターバルとピッチの処理中にエラー: {str(e)}")
        
        # フレーム時間と周波数の情報を処理
        if 'frame_times' in detection_result and 'frame_frequencies' in detection_result:
            try:
                times = detection_result['frame_times']
                freqs = detection_result['frame_frequencies']
                if len(times) == len(freqs):
                    normalized_result['frame_times'] = np.asarray(times)
                    normalized_result['frame_frequencies'] = np.asarray(freqs)
                else:
                    logger.warning(f"フレーム時間 ({len(times)}) とフレーム周波数 ({len(freqs)}) の長さが一致しません")
            except Exception as e:
                logger.error(f"フレーム時間と周波数の処理中にエラー: {str(e)}")
        
        # 追加データを設定
        if 'additional_data' in detection_result:
            additional_data.update(detection_result['additional_data'])
        
        # 他のフィールドも追加データに保存
        for key, value in detection_result.items():
            if key not in required_fields:
                additional_data[key] = value
        
        normalized_result['additional_data'] = additional_data
        
        # 結果の整合性を確認
        if len(normalized_result['intervals']) != len(normalized_result['note_pitches']):
            logger.error("正規化後のインターバルとピッチの長さが一致しません")
            min_len = min(len(normalized_result['intervals']), len(normalized_result['note_pitches']))
            normalized_result['intervals'] = normalized_result['intervals'][:min_len]
            normalized_result['note_pitches'] = normalized_result['note_pitches'][:min_len]
        
        return normalized_result
        
    except Exception as e:
        logger.error(f"検出結果の正規化中にエラーが発生しました: {str(e)}")
        return {
            'intervals': np.array([]).reshape(0, 2),
            'note_pitches': np.array([]),
            'frame_times': np.array([]),
            'frame_frequencies': np.array([]),
            'detector_name': "Unknown",
            'detection_time': 0.0,
            'additional_data': {}
        }


def ensure_detector_output_format(detector_output):
    """
    検出器の出力フォーマットを検証し、標準化します。
    
    必須キー:
    - intervals: 音符の開始時間と終了時間を含む2次元配列
    - note_pitches: 各音符のピッチ値を含む1次元配列
    
    任意キー:
    - frame_times: フレームの時間情報を含む1次元配列
    - frame_frequencies: 各フレームの周波数情報を含む1次元配列
    - 追加フィールド: 他の任意のフィールドは保持されます
    
    Parameters
    ----------
    detector_output : dict
        検出器の出力辞書
        
    Returns
    -------
    dict
        検証済みで標準化された検出器出力
    """
    # detector_outputがNoneまたは辞書でない場合はNoneを返す（normalizeで行うため不要かも）
    if detector_output is None or not isinstance(detector_output, dict):
        logger.warning("ensure_detector_output_format に無効な入力が与えられました。空の辞書を返します。")
        return normalize_detection_result({}) # 空の結果を正規化して返す

    logger.debug(f"ensure_detector_output_format: 受信したキー: {list(detector_output.keys())}") # list()で囲む
    for key, value in detector_output.items():
        if hasattr(value, 'shape') and hasattr(value, 'dtype'): # NumPy配列かチェック
            logger.debug(f"  - {key}: 型={type(value).__name__}, 形状={value.shape}, dtype={value.dtype}")
        elif isinstance(value, (list, tuple)):
            logger.debug(f"  - {key}: 型={type(value).__name__}, 長さ={len(value)}")
        else:
            logger.debug(f"  - {key}: 型={type(value).__name__}")

    result_dict = {}
    
    # 必須キーの存在確認とNumPy配列への変換
    required_keys = ['intervals', 'note_pitches']
    for key in required_keys:
        if key not in detector_output:
            # warnings.warn(f"検出結果に必須キー '{key}' がありません。空の配列を使用します。") # loggerに変更
            logger.warning(f"検出結果に必須キー '{key}' がありません。空の配列を使用します。")
            if key == 'intervals':
                result_dict[key] = np.empty((0, 2), dtype=np.float64) # dtype指定
            else:
                result_dict[key] = np.empty(0, dtype=np.float64) # dtype指定
        else:
            try:
                # NumPy配列への変換を試みる
                converted_array = np.asarray(detector_output[key], dtype=np.float64)
                # intervals の形状チェック (N, 2)
                if key == 'intervals' and converted_array.ndim == 2 and converted_array.shape[1] != 2:
                    logger.error(f"キー '{key}' の形状が不正です ({converted_array.shape})。期待される形状: (N, 2)。空の配列を使用します。")
                    result_dict[key] = np.empty((0, 2), dtype=np.float64)
                elif key == 'intervals' and converted_array.ndim != 2:
                    logger.error(f"キー '{key}' の次元が不正です ({converted_array.ndim})。期待される次元: 2。空の配列を使用します。")
                    result_dict[key] = np.empty((0, 2), dtype=np.float64)
                # note_pitches の形状チェック (N,)
                elif key == 'note_pitches' and converted_array.ndim != 1:
                    logger.error(f"キー '{key}' の次元が不正です ({converted_array.ndim})。期待される次元: 1。空の配列を使用します。")
                    result_dict[key] = np.empty(0, dtype=np.float64)
                else:
                    result_dict[key] = converted_array
            except Exception as e:
                # 変換エラーの場合、ログを記録して空の配列を設定
                logger.error(f"キー '{key}' の値をNumPy配列に変換できませんでした: {e}。入力値: {detector_output[key]}。空の配列を使用します。")
                if key == 'intervals':
                    result_dict[key] = np.empty((0, 2), dtype=np.float64)
                else:
                    result_dict[key] = np.empty(0, dtype=np.float64)

    # 任意キーの処理 (NumPy配列に変換可能なものは変換)
    optional_keys = ['frame_times', 'frame_frequencies']
    for key in optional_keys:
        if key in detector_output:
            try:
                converted_array = np.asarray(detector_output[key], dtype=np.float64)
                # 形状チェック (M,)
                if converted_array.ndim != 1:
                    logger.error(f"キー '{key}' の次元が不正です ({converted_array.ndim})。期待される次元: 1。Noneを使用します。")
                    result_dict[key] = np.empty(0, dtype=np.float64)
                else:
                    result_dict[key] = converted_array
            except Exception as e:
                logger.error(f"キー '{key}' の値をNumPy配列に変換できませんでした: {e}。入力値: {detector_output[key]}。Noneを使用します。")
                result_dict[key] = np.empty(0, dtype=np.float64)
        else:
            result_dict[key] = np.empty(0, dtype=np.float64)

    # その他の追加キーをそのままコピー
    for key, value in detector_output.items():
        if key not in required_keys and key not in optional_keys:
            result_dict[key] = value

    logger.debug(f"変換後のキー: {list(result_dict.keys())}") # list()で囲む

    # intervals と note_pitches の長さチェックと調整
    intervals = result_dict.get('intervals')
    pitches = result_dict.get('note_pitches')

    # Noneチェックを追加
    if intervals is None or pitches is None:
        logger.error("intervals または note_pitches の変換に失敗したため、長さの比較ができません。")
        # この場合、両方を空にするか、エラーを伝播させるか検討。
        # ここでは両方を空にする
        result_dict['intervals'] = np.empty((0, 2), dtype=np.float64)
        result_dict['note_pitches'] = np.empty(0, dtype=np.float64)
        intervals = result_dict['intervals'] # 更新
        pitches = result_dict['note_pitches'] # 更新

    len_intervals = len(intervals)
    len_pitches = len(pitches)

    if len_intervals != len_pitches:
        # warnings.warn(f"intervalsの長さ ({len_intervals}) と note_pitchesの長さ ({len_pitches}) が一致しません。短い方に合わせます。") # loggerに変更
        logger.warning(f"intervalsの長さ ({len_intervals}) と note_pitchesの長さ ({len_pitches}) が一致しません。短い方に合わせます。")
        min_len = min(len_intervals, len_pitches)
        result_dict['intervals'] = intervals[:min_len]
        result_dict['note_pitches'] = pitches[:min_len]

    # --- ここに単調性チェックやピッチ範囲チェックを追加可能 ---
    # 例:
    # final_intervals = result_dict['intervals']
    # if len(final_intervals) > 1:
    #     onset_times = final_intervals[:, 0]
    #     if not np.all(np.diff(onset_times) >= 0):
    #         logger.warning("オンセット時刻が単調増加していません")
    #     invalid_intervals = final_intervals[:, 1] < final_intervals[:, 0]
    #     if np.any(invalid_intervals):
    #         logger.warning(f"{np.sum(invalid_intervals)}個の音符の開始時間が終了時間よりも後になっています。")
    #
    # final_pitches = result_dict['note_pitches']
    # invalid_pitch_mask = (final_pitches <= 0) | (final_pitches > 20000) # 例: 0Hz以下と20kHz超
    # if np.any(invalid_pitch_mask):
    #     logger.warning(f"{np.sum(invalid_pitch_mask)}個のピッチ値が不正な範囲です")
    # ----------------------------------------------------------

    # フレーム情報の長さチェック (任意キーなのでエラーではなく警告)
    frame_times = result_dict.get('frame_times')
    frame_frequencies = result_dict.get('frame_frequencies')

    if frame_times is not None and frame_frequencies is not None:
        if len(frame_times) != len(frame_frequencies):
             logger.warning(f"frame_times の長さ ({len(frame_times)}) と frame_frequencies の長さ ({len(frame_frequencies)}) が一致しません。そのまま保持します。")

    # 最終的な音符数とフレーム数をログに出力
    final_note_count = len(result_dict.get('intervals', np.empty((0, 2), dtype=np.float64)))
    # final_frame_count = len(result_dict.get('frame_times', [])) # ここでエラーが発生していた
    # frame_timesがNoneでないことを確認してからlen()を呼ぶ
    final_frame_count = len(result_dict.get('frame_times', np.empty(0, dtype=np.float64)))
    logger.debug(f"出力: 音符数={final_note_count}, フレーム数={final_frame_count}")

    # DetectionResultオブジェクトではなく辞書を返すように変更（関数名に合わせる）
    # return DetectionResult(
    #     intervals=result_dict.get('intervals', np.empty((0, 2))),\
    #     note_pitches=result_dict.get('note_pitches', np.empty(0)),\
    #     frame_times=result_dict.get('frame_times'),\
    #     frame_frequencies=result_dict.get('frame_frequencies'),\
    #     detector_name=result_dict.get('detector_name', 'Unknown'),\
    #     detection_time=result_dict.get('detection_time', 0.0),\
    #     additional_data={k: v for k, v in result_dict.items() if k not in required_keys and k not in optional_keys and k not in ['detector_name', 'detection_time']}\
    # )\n    return result_dict\n\ndef detection_result_to_dict(detection_result: DetectionResult) -> Dict[str, Any]:\n    \"\"\"\n
    return result_dict


def detection_result_to_dict(detection_result: DetectionResult) -> Dict[str, Any]:
    """
    DetectionResultオブジェクトを辞書に変換します。

    Parameters
    ----------
    detection_result : DetectionResult
        変換するDetectionResultオブジェクト

    Returns
    -------
    Dict[str, Any]
        変換された辞書
    """
    result_dict = {
        'intervals': detection_result.intervals,
        'note_pitches': detection_result.note_pitches,
        'frame_times': detection_result.frame_times,
        'frame_frequencies': detection_result.frame_frequencies,
        'detector_name': detection_result.detector_name,
        'detection_time': detection_result.detection_time
    }
    
    # 追加データがある場合は追加
    if detection_result.additional_data:
        result_dict.update(detection_result.additional_data)
    
    return result_dict


def create_detector(detector_name: str, detector_params: Dict[str, Any] = None) -> BaseDetector:
    """
    検出器のインスタンスを作成する

    Parameters
    ----------
    detector_name : str
        検出器の名前（例: 'BasicDetector'）
    detector_params : Dict[str, Any], optional
        検出器のパラメータ, by default None

    Returns
    -------
    BaseDetector
        検出器のインスタンス

    Raises
    ------
    ImportError
        検出器が見つからない場合
    """
    detector_class = get_detector_class(detector_name)
    return detector_class(**(detector_params or {})) 