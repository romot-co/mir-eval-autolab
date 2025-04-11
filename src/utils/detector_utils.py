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


def normalize_detection_result(detection_result: Dict[str, Any], sr: int = None) -> DetectionResult:
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
    DetectionResult
        正規化された検出結果オブジェクト。
    """
    try:
        # 空の結果または無効な結果への対処
        if detection_result is None or not isinstance(detection_result, dict):
            logger.warning(f"無効な検出結果です。型: {type(detection_result)}。空の結果オブジェクトを返します。")
            return DetectionResult() # 空のDetectionResultを返す

        # 必須フィールドの定義 (デフォルト値として使用)
        default_values = {
            'intervals': np.array([]).reshape(0, 2),
            'note_pitches': np.array([]),
            'frame_times': np.array([]),
            'frame_frequencies': np.array([]),
            'detector_name': "Unknown",
            'detection_time': 0.0,
            'additional_data': {}
        }

        # 結果辞書の初期化 (入力辞書をコピーし、デフォルト値で上書き)
        normalized_result = default_values.copy()
        additional_data = {} # 追加データ用

        # 入力辞書からフィールドをコピー (numpy配列化も行う)
        for field, default_val in default_values.items():
            if field in detection_result:
                try:
                    value = detection_result[field]
                    if isinstance(value, (list, np.ndarray)) and field not in ['detector_name', 'detection_time', 'additional_data']:
                         # Ensure correct dtype for numeric arrays, allow others
                        dtype = np.float64 if field in ['intervals', 'note_pitches', 'frame_times', 'frame_frequencies'] else None
                        normalized_result[field] = np.asarray(value, dtype=dtype)
                    elif field == 'additional_data' and isinstance(value, dict):
                         # additional_data should be a dict
                        normalized_result[field] = value # Use provided dict directly
                    elif field in ['detector_name', 'detection_time']:
                        normalized_result[field] = value # Copy directly
                    else:
                         # Handle cases where the type might not match expectation gracefully
                        logger.warning(f"フィールド '{field}' の型が期待値と異なります: 型={type(value)}。デフォルト値を使用します。")
                        # Keep the default value for this field
                except Exception as e:
                    logger.warning(f"フィールド {field} の変換中にエラー: {str(e)}。デフォルト値を使用します。")
                    # Keep the default value
            # else: field not in detection_result, default value is already set

        # intervalsとnote_pitchesの検証と修正
        # Ensure intervals is always 2D, even if empty
        if 'intervals' in normalized_result:
             intervals = normalized_result['intervals']
             if not isinstance(intervals, np.ndarray) or intervals.ndim != 2 or intervals.shape[1] != 2:
                 logger.warning(f"不正なインターバル形状: {intervals.shape if hasattr(intervals, 'shape') else type(intervals)}。空にします。")
                 normalized_result['intervals'] = np.empty((0, 2), dtype=np.float64)
                 normalized_result['note_pitches'] = np.empty((0,), dtype=np.float64) # Pitches must match intervals
             else:
                 # Ensure pitches match intervals length
                 pitches = normalized_result.get('note_pitches', np.array([]))
                 if not isinstance(pitches, np.ndarray):
                     pitches = np.asarray(pitches, dtype=np.float64) # Attempt conversion

                 if len(pitches) != len(intervals):
                     logger.warning(f"ピッチ数 ({len(pitches)}) とインターバル数 ({len(intervals)}) が一致しません。短い方に合わせます。")
                     min_len = min(len(pitches), len(intervals))
                     normalized_result['note_pitches'] = pitches[:min_len]
                     normalized_result['intervals'] = intervals[:min_len]
                 else:
                      normalized_result['note_pitches'] = pitches # Already ensured array type and matching length

        # フレーム時間と周波数の情報を処理
        if 'frame_times' in normalized_result and 'frame_frequencies' in normalized_result:
            times = normalized_result['frame_times']
            freqs = normalized_result['frame_frequencies']
            if not isinstance(times, np.ndarray): times = np.asarray(times, dtype=np.float64)
            if not isinstance(freqs, np.ndarray): freqs = np.asarray(freqs, dtype=np.float64)

            if len(times) != len(freqs):
                logger.warning(f"フレーム時間 ({len(times)}) とフレーム周波数 ({len(freqs)}) の長さが一致しません。短い方に合わせます。")
                min_len = min(len(times), len(freqs))
                normalized_result['frame_times'] = times[:min_len]
                normalized_result['frame_frequencies'] = freqs[:min_len]
            else:
                 normalized_result['frame_times'] = times
                 normalized_result['frame_frequencies'] = freqs

        # 追加データを設定 (入力辞書にのみ存在し、標準フィールドでないものを追加)
        input_additional = detection_result.get('additional_data', {})
        if isinstance(input_additional, dict):
             additional_data.update(input_additional) # Start with explicit additional_data

        for key, value in detection_result.items():
            if key not in default_values: # Add fields not part of the standard structure
                additional_data[key] = value

        normalized_result['additional_data'] = additional_data

        # 最終的な結果の整合性を再確認 (特に interval/pitch 長)
        # This check might be redundant due to earlier checks, but good for safety
        if len(normalized_result['intervals']) != len(normalized_result['note_pitches']):
            logger.error("最終正規化チェック: インターバルとピッチの長さが一致しません。短い方に合わせます。")
            min_len = min(len(normalized_result['intervals']), len(normalized_result['note_pitches']))
            normalized_result['intervals'] = normalized_result['intervals'][:min_len]
            normalized_result['note_pitches'] = normalized_result['note_pitches'][:min_len]

        # DetectionResult オブジェクトを作成して返す
        return DetectionResult.from_dict(normalized_result)

    except Exception as e:
        logger.error(f"検出結果の正規化中に予期せぬエラーが発生しました: {str(e)}\n{traceback.format_exc()}")
        return DetectionResult() # エラー時は空のDetectionResultを返す


def ensure_detector_output_format(detector_output: Any) -> DetectionResult:
    """
    検出器の出力フォーマットを検証し、標準化された DetectionResult オブジェクトを返します。
    内部で normalize_detection_result を呼び出します。

    Parameters
    ----------
    detector_output : Any
        検出器からの生の出力 (辞書を期待)

    Returns
    -------
    DetectionResult
        検証・正規化された検出結果オブジェクト
    """
    if not isinstance(detector_output, dict):
        logger.warning(f"ensure_detector_output_format に辞書でない入力が与えられました (型: {type(detector_output)})。空の結果として正規化します。")
        return normalize_detection_result({}) # 空の辞書を正規化

    logger.debug(f"ensure_detector_output_format: 入力辞書を正規化します: {list(detector_output.keys())}")

    # normalize_detection_result に処理を委譲
    return normalize_detection_result(detector_output)


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