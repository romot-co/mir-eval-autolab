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

from src.detectors.base_detector import BaseDetector
from src.utils.detection_result import DetectionResult
# 個別の検出器のインポートを削除
# 代わりにレジストリ関数をインポート
from src.detectors import get_registered_detector

logger = logging.getLogger(__name__)


def get_detector_class(detector_name: str) -> Type[BaseDetector]:
    """
    検出器クラスをレジストリから名前で取得する

    Parameters
    ----------
    detector_name : str
        検出器の名前（大文字小文字は区別しない）

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
        if not issubclass(detector_class, BaseDetector):
            # レジストリが正しく管理されていれば、この場合は発生しないはず
            raise TypeError(f"'{detector_name}'に登録されたクラスがBaseDetectorを継承していません。")
        return detector_class
    except ValueError as e: # get_registered_detectorはValueErrorを発生させる
        # 元の実装との整合性のためImportErrorとして再発生
        raise ImportError(f"検出器 '{detector_name}' がレジストリに見つかりません: {e}")
    except Exception as e:
        # 取得中のその他の潜在的なエラーを処理
        raise ImportError(f"検出器 '{detector_name}' の取得中に予期せぬエラーが発生しました: {e}")


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


def ensure_detector_output_format(detector_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    検出器の出力が評価に必要な全てのキーを持っていることを確認し、
    必要に応じて標準フォーマットに変換する。
    
    Parameters
    ----------
    detector_output : Dict[str, Any]
        検出器からの生の出力
        
    Returns
    -------
    Dict[str, Any]
        すべての必要なキーを持つ変換済み出力
    """
    try:
        # 標準フォーマットに正規化
        normalized = normalize_detection_result(detector_output)
        
        # 必須キーの定義
        required_keys = {
            'intervals': {'type': np.ndarray, 'shape': (None, 2), 'dtype': np.float64},
            'note_pitches': {'type': np.ndarray, 'shape': (None,), 'dtype': np.float64}
        }
        
        # 各必須キーの検証と修正
        for key, specs in required_keys.items():
            if key not in normalized:
                if specs['shape'][0] is None:
                    normalized[key] = np.array([], dtype=specs['dtype'])
                else:
                    normalized[key] = np.array([]).reshape(0, *specs['shape'][1:])
                logger.warning(f"必須キー '{key}' が存在しないため、空の配列を作成しました")
                continue
            
            value = normalized[key]
            
            # 型の検証と変換
            if not isinstance(value, specs['type']):
                try:
                    value = np.asarray(value, dtype=specs['dtype'])
                    normalized[key] = value
                    logger.debug(f"キー '{key}' を適切な型に変換しました")
                except Exception as e:
                    logger.error(f"キー '{key}' の型変換に失敗: {str(e)}")
                    if specs['shape'][0] is None:
                        normalized[key] = np.array([], dtype=specs['dtype'])
                    else:
                        normalized[key] = np.array([]).reshape(0, *specs['shape'][1:])
                    continue
            
            # 形状の検証
            if len(specs['shape']) != len(value.shape):
                logger.error(f"キー '{key}' の次元数が不正: 期待={len(specs['shape'])}, 実際={len(value.shape)}")
                if specs['shape'][0] is None:
                    normalized[key] = np.array([], dtype=specs['dtype'])
                else:
                    normalized[key] = np.array([]).reshape(0, *specs['shape'][1:])
                continue
            
            for i, (expected, actual) in enumerate(zip(specs['shape'][1:], value.shape[1:])):
                if expected is not None and expected != actual:
                    logger.error(f"キー '{key}' の形状が不正: 軸{i+1}で期待={expected}, 実際={actual}")
                    if specs['shape'][0] is None:
                        normalized[key] = np.array([], dtype=specs['dtype'])
                    else:
                        normalized[key] = np.array([]).reshape(0, *specs['shape'][1:])
                    break
        
        # データの整合性チェック
        if len(normalized['intervals']) != len(normalized['note_pitches']):
            min_len = min(len(normalized['intervals']), len(normalized['note_pitches']))
            logger.warning(f"インターバルとピッチの長さが一致しません。短い方 ({min_len}) に合わせます。")
            normalized['intervals'] = normalized['intervals'][:min_len]
            normalized['note_pitches'] = normalized['note_pitches'][:min_len]
        
        # 時間の単調性チェック
        if len(normalized['intervals']) > 0:
            # オンセット時刻の単調性
            if not np.all(np.diff(normalized['intervals'][:, 0]) >= 0):
                logger.warning("オンセット時刻が単調増加していません")
            
            # 各音符の区間の妥当性
            invalid_intervals = normalized['intervals'][:, 1] <= normalized['intervals'][:, 0]
            if np.any(invalid_intervals):
                logger.warning(f"{np.sum(invalid_intervals)}個の音符で終了時刻が開始時刻以前になっています")
        
        # ピッチ値の範囲チェック
        if len(normalized['note_pitches']) > 0:
            invalid_pitches = (normalized['note_pitches'] < 0) | (normalized['note_pitches'] > 20000)
            if np.any(invalid_pitches):
                logger.warning(f"{np.sum(invalid_pitches)}個のピッチ値が不正な範囲です")
        
        return normalized
        
    except Exception as e:
        logger.error(f"検出器出力の検証中に予期せぬエラーが発生: {str(e)}")
        return {
            'intervals': np.array([]).reshape(0, 2),
            'note_pitches': np.array([])
        }


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