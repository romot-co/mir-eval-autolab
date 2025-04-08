"""
評価ランナーモジュール

このモジュールは、音声検出アルゴリズムを評価するための中心的な機能を提供します。
実際の評価処理を行い、結果を計算します。

このモジュールでは、複数ファイルやディレクトリをまとめて評価する高レベルの実行管理や
評価ワークフロー全体の制御を行います。

## 主な関数

### 評価実行
- run_evaluation: 単一または複数のファイル、単一または複数の検出器による評価を実行

### 結果集計
- evaluate_detection_result: 検出結果を評価してメトリクスを計算
- _calculate_evaluation_summary: 評価結果のサマリー計算
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import datetime
import traceback
import tempfile
from pathlib import Path
from tqdm import tqdm

from src.utils.audio_utils import load_audio_file, load_reference_data, make_output_path
from src.utils.exception_utils import log_exception, create_error_result
from src.utils.json_utils import NumpyEncoder
from src.utils.detection_result import DetectionResult
from src.utils.detector_utils import get_detector_class, normalize_detection_result
from src.evaluation.evaluation_io import create_summary_dataframe, save_evaluation_result, print_summary_statistics

import mir_eval
import mir_eval.transcription

import logging

# プロット関連のインポート
plot_module_imported = False
try:
    from src.visualization.plots import plot_detection_results
    plot_module_imported = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

def evaluate_detection_result(detected_intervals, detected_pitches, 
                             reference_intervals, reference_pitches,
                             evaluator_config: Optional[Dict[str, Any]] = None,
                             detector_result: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
    """
    検出結果を評価し、各種評価指標を計算します。

    Parameters
    ----------
    detected_intervals : np.ndarray
        検出されたノート区間の配列 (N, 2) 形式で各行は [onset, offset]
    detected_pitches : np.ndarray
        検出されたノートピッチの配列 (N,) 周波数(Hz)
    reference_intervals : np.ndarray
        参照ノート区間の配列 (M, 2) 形式で各行は [onset, offset]
    reference_pitches : np.ndarray
        参照ノートピッチの配列 (M,) 周波数(Hz)
    evaluator_config : Optional[Dict[str, Any]], optional
        評価設定の辞書, by default None
        - onset_tolerance: オンセット許容誤差（秒）
        - pitch_tolerance: ピッチ許容誤差（セント）
        - offset_ratio: オフセット許容誤差の比率
        - offset_min_tolerance: オフセット許容誤差の最小値（秒）
        - use_pitch_chroma: ピッチクラスのみで評価するかどうか
    detector_result : Optional[Dict[str, Any]], optional
        検出器の出力結果の辞書, by default None
        - frame_times: フレーム時刻の配列
        - frame_frequencies: フレームごとの周波数の配列

    Returns
    -------
    Dict[str, Dict[str, float]]
        評価結果の辞書。以下のキーを含みます：
        - note: ノートベースの評価結果
            - precision: 適合率
            - recall: 再現率
            - f_measure: F値
        - pitch: ピッチベースの評価結果
            - precision: 適合率
            - recall: 再現率
            - f_measure: F値
        - frame_pitch: フレームベースの評価結果（フレームデータがある場合のみ）
            - precision: 適合率
            - recall: 再現率
            - f_measure: F値
            - accuracy: 正解率

    注意
    ----
    この関数は2種類の評価を行います：

    1. ノートベース評価（note, pitch）:
       - mir_eval.transcriptionを使用し、ポリフォニー（複数ノートの同時発音）を正しく評価します。
       - 参照ノートリストと推定ノートリスト間の最適なマッチングを行い、和音や独立した複数のメロディラインを
         適切に評価できます。
       - 主要な評価指標として使用してください。

    2. フレームベース評価（frame_pitch）:
       - 各時間フレームで単一のピッチのみを評価します。
       - ポリフォニーの場合、各フレームで最も顕著なピッチのみが評価され、他のピッチは無視されます。
       - 補助的な評価指標として使用してください。
       - フレームデータが提供されない場合は計算されません。
    """
    if evaluator_config is None:
        evaluator_config = {}
    
    # 検出結果と参照データのチェック
    detected_intervals = np.array(detected_intervals)
    detected_pitches = np.array(detected_pitches)
    reference_intervals = np.array(reference_intervals)
    reference_pitches = np.array(reference_pitches)
    
    logger = logging.getLogger('evaluate')
    
    # パラメータの取得
    onset_tolerance = evaluator_config.get('tolerance_onset', 0.05)  # デフォルト: 50ms
    offset_ratio = evaluator_config.get('offset_ratio', 0.2)  # デフォルト: 0.2
    offset_min_tolerance = evaluator_config.get('offset_min_tolerance', 0.05)  # デフォルト: 50ms
    pitch_tolerance = evaluator_config.get('tolerance_pitch', 50.0)  # デフォルト: 50セント
    use_pitch_chroma = evaluator_config.get('use_pitch_chroma', False)  # デフォルト: False
    use_offset = evaluator_config.get('use_strict_offset', True)  # オフセットを評価するかどうか
    
    # データの検証
    if len(detected_intervals) != len(detected_pitches):
        logger.warning(f"検出区間とピッチの数が一致しません: {len(detected_intervals)} vs {len(detected_pitches)}")
        return create_error_result("DetectionMismatch")
    
    if len(reference_intervals) != len(reference_pitches):
        logger.warning(f"参照区間とピッチの数が一致しません: {len(reference_intervals)} vs {len(reference_pitches)}")
        return create_error_result("ReferenceMismatch")
    
    # 0Hz以下の無効なピッチ値をチェック
    detected_pitch_mask = detected_pitches <= 0
    reference_pitch_mask = reference_pitches <= 0
    
    if np.any(detected_pitch_mask):
        logger.warning(f"検出結果に {np.sum(detected_pitch_mask)} 個の0Hz以下のピッチ値があります。評価用に1Hzに補正します。")
        # 複製を作成して元の配列を変更しない
        detected_pitches_for_eval = detected_pitches.copy()
        detected_pitches_for_eval[detected_pitch_mask] = 1.0  # 1Hzに設定（ほぼ無音）
    else:
        detected_pitches_for_eval = detected_pitches
        
    if np.any(reference_pitch_mask):
        logger.warning(f"参照データに {np.sum(reference_pitch_mask)} 個の0Hz以下のピッチ値があります。評価用に1Hzに補正します。")
        # 複製を作成して元の配列を変更しない
        reference_pitches_for_eval = reference_pitches.copy()
        reference_pitches_for_eval[reference_pitch_mask] = 1.0  # 1Hzに設定（ほぼ無音）
    else:
        reference_pitches_for_eval = reference_pitches
    
    # mir_eval.transcriptionを直接使って評価を実行
    
    # ==== オンセットのみの評価 ====
    onset_prec, onset_rec, onset_f = mir_eval.transcription.onset_precision_recall_f1(
        reference_intervals,
        detected_intervals,
        onset_tolerance=onset_tolerance,
        strict=False  # MIREX互換
    )
    onset_eval = {
        'precision': onset_prec,
        'recall': onset_rec,
        'f_measure': onset_f
    }
    
    # ==== オフセットのみの評価 ====
    offset_prec, offset_rec, offset_f = mir_eval.transcription.offset_precision_recall_f1(
        reference_intervals,
        detected_intervals,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
        strict=False  # MIREX互換
    )
    offset_eval = {
        'precision': offset_prec,
        'recall': offset_rec,
        'f_measure': offset_f
    }
    
    # ==== ノート評価（オンセット+ピッチ+オフセット） ====
    if use_offset:
        # オフセットも含めて評価
        prec_note, rec_note, f_note, avg_overlap_note = mir_eval.transcription.precision_recall_f1_overlap(
            reference_intervals,
            reference_pitches_for_eval,
            detected_intervals,
            detected_pitches_for_eval,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
            strict=False  # MIREX互換
        )
        note_eval = {
            'precision': prec_note,
            'recall': rec_note,
            'f_measure': f_note,
            'overlap_ratio': avg_overlap_note
        }
    else:
        # オンセット+ピッチのみで評価（オフセットを無視）
        prec_note, rec_note, f_note, _ = mir_eval.transcription.precision_recall_f1_overlap(
            reference_intervals,
            reference_pitches_for_eval,
            detected_intervals,
            detected_pitches_for_eval,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance,
            offset_ratio=None,  # オフセット無視
            strict=False  # MIREX互換
        )
        note_eval = {
            'precision': prec_note,
            'recall': rec_note,
            'f_measure': f_note
        }
    
    # ==== ピッチ評価（オンセット+ピッチのみの評価、オフセット無視） ====
    # オンセット+ピッチのみで評価（オフセットを無視）
    prec_pitch, rec_pitch, f_pitch, avg_overlap_pitch = mir_eval.transcription.precision_recall_f1_overlap(
        reference_intervals,
        reference_pitches_for_eval,
        detected_intervals,
        detected_pitches_for_eval,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=None,  # オフセット無視
        strict=False  # MIREX互換
    )
    pitch_eval = {
        'precision': prec_pitch,
        'recall': rec_pitch,
        'f_measure': f_pitch
    }
    
    # フレーム評価（必要な場合）
    frame_pitch_result = {
        'voicing_recall': 0.0,
        'voicing_false_alarm': 1.0,
        'raw_pitch_accuracy': 0.0,
        'raw_chroma_accuracy': 0.0,
        'overall_accuracy': 0.0
    }
    
    if detector_result is not None and 'frame_times' in detector_result and 'frame_frequencies' in detector_result:
        # 検出器からフレームデータを取得
        est_frame_times = detector_result.get('frame_times', [])
        est_frame_freqs = detector_result.get('frame_frequencies', [])
        
        # フレーム評価を実行
        try:
            from src.evaluation.evaluation_frame import evaluate_frame_pitches, notes_to_frames
            
            # 参照フレームデータの取得（detector_resultから取得するか、参照ノートデータから生成）
            ref_frame_times = []
            ref_frame_freqs = []
            
            # detector_resultから参照フレームデータを取得（存在する場合）
            if 'reference_frame_times' in detector_result and 'reference_frame_frequencies' in detector_result:
                ref_frame_times = detector_result.get('reference_frame_times', [])
                ref_frame_freqs = detector_result.get('reference_frame_frequencies', [])
            
            # 参照フレームデータが空の場合は、参照ノートデータからフレームデータを生成
            if len(ref_frame_times) == 0 or len(ref_frame_freqs) == 0:
                # 推定結果のフレームのホップサイズを推定
                if len(est_frame_times) >= 2:
                    hop_time = est_frame_times[1] - est_frame_times[0]
                else:
                    hop_time = 0.01  # デフォルト値
                
                # 終了時間を推定（推定結果と参照結果の両方の最大値）
                end_time = None
                if len(est_frame_times) > 0:
                    end_time = est_frame_times[-1]
                if len(reference_intervals) > 0:
                    ref_end_time = np.max(reference_intervals[:, 1])
                    if end_time is None or ref_end_time > end_time:
                        end_time = ref_end_time
                
                # ノート単位からフレーム単位に変換
                logger.info("参照データをノート単位からフレーム単位に変換しています...")
                ref_frame_times, ref_frame_freqs = notes_to_frames(
                    note_intervals=reference_intervals,
                    note_pitches=reference_pitches,
                    hop_time=hop_time,
                    end_time=end_time
                )
            
            # フレーム評価の実行
            frame_pitch_result = evaluate_frame_pitches(
                ref_times=ref_frame_times,
                ref_freqs=ref_frame_freqs,
                est_times=est_frame_times,
                est_freqs=est_frame_freqs,
                pitch_tolerance=pitch_tolerance,
                use_pitch_chroma=use_pitch_chroma
            )
            
            # 結果をログに出力
            logger.info(f"フレーム評価結果 - 有声再現率: {frame_pitch_result['voicing_recall']:.3f}, ピッチ精度: {frame_pitch_result['raw_pitch_accuracy']:.3f}, 総合精度: {frame_pitch_result['overall_accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"フレーム評価中にエラーが発生しました: {str(e)}")
            logger.error(traceback.format_exc())
    
    # 全体の評価結果をまとめる
    evaluation_result = {
        'onset': onset_eval,
        'note': note_eval, 
        'pitch': pitch_eval,
        'offset': offset_eval,
        'frame_pitch': frame_pitch_result
    }
    
    return evaluation_result

def create_error_result(error_msg: str) -> Dict[str, float]:
    """
    エラー時のデフォルト評価結果を生成します。

    Parameters
    ----------
    error_msg : str
        エラーメッセージ

    Returns
    -------
    Dict[str, float]
        デフォルトの評価結果
    """
    return {
        'voicing_recall': 0.0,
        'voicing_false_alarm': 1.0,
        'raw_pitch_accuracy': 0.0,
        'raw_chroma_accuracy': 0.0,
        'overall_accuracy': 0.0
    }

def run_evaluation(
    audio_paths: Union[str, List[str]],
    ref_paths: Union[str, List[str]],
    detector_names: Union[str, List[str]],
    detector_params: Optional[Dict[str, Dict[str, Any]]] = None,
    evaluator_config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    save_plots: bool = False,
    plot_format: str = 'png',
    save_results_json: bool = True,
    plot_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    音声検出アルゴリズムの評価を実行します。複数のファイルと複数の検出器に対応し、
    MIREX準拠のオンセット/オフセット/ピッチ/ノート評価を一括で行います。
    
    Parameters
    ----------
    audio_paths : Union[str, List[str]]
        評価する音声ファイルのパスまたはそのリスト
    ref_paths : Union[str, List[str]]
        参照データファイルのパスまたはそのリスト
    detector_names : Union[str, List[str]]
        使用する検出器の名前またはそのリスト
    detector_params : Optional[Dict[str, Dict[str, Any]]], optional
        検出器のパラメータ, by default None
        {detector_name: {param_name: param_value, ...}, ...}
    evaluator_config : Optional[Dict[str, Any]], optional
        評価設定, by default None
    output_dir : Optional[str], optional
        出力ディレクトリ, by default None
    save_plots : bool, optional
        結果のプロットを保存するかどうか, by default False
    plot_format : str, optional
        プロット形式, by default 'png'
    save_results_json : bool, optional
        JSON形式で結果を保存するかどうか, by default True
    plot_config : Optional[Dict[str, Any]], optional
        プロット設定, by default None
        {
            'show': bool,  # プロットを表示するかどうか
            'save_path': str,  # プロットの保存パス
            'figsize': Tuple[int, int],  # プロットのサイズ
            'dpi': int  # プロットの解像度
        }
        
    Returns
    -------
    Dict[str, Any]
        評価結果
        {
          "status": "success",
          "results": [ ...各ファイル×検出器の評価結果... ],
          "summary": { ... 全体集計 ... }
        }
    """
    # 入力パラメータの整理
    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]
    
    if isinstance(ref_paths, str):
        ref_paths = [ref_paths]
    
    if isinstance(detector_names, str):
        detector_names = [detector_names]
        
    if detector_params is None:
        detector_params = {}
    
    if evaluator_config is None:
        evaluator_config = {}
    
    # 出力ディレクトリの作成
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 結果格納用のデータ構造
    results = []
    errors = []
    
    # 検出器のインスタンス化
    detector_instances = {}
    for detector_name in detector_names:
        try:
            detector_class = get_detector_class(detector_name)
            # 検出器特有のパラメータがあれば適用
            params = detector_params.get(detector_name, {})
            detector_instances[detector_name] = detector_class(**params)
            logger.info(f"検出器 '{detector_name}' を初期化しました。パラメータ: {params}")
        except Exception as e:
            error_msg = f"検出器 '{detector_name}' の初期化中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    if not detector_instances:
        error_msg = "有効な検出器がありません"
        logger.error(error_msg)
        return {"status": "error", "error": error_msg}
    
    # ファイルと検出器の組み合わせをすべて評価
    for audio_idx, (audio_path, ref_path) in enumerate(zip(audio_paths, ref_paths)):
        file_basename = os.path.basename(audio_path)
        logger.info(f"ファイル {audio_idx+1}/{len(audio_paths)}: {file_basename} を評価します")
        
        # 音声ファイルの読み込み
        try:
            audio_data, sr = load_audio_file(audio_path)
            logger.info(f"音声ファイルを読み込みました: {audio_path}, サンプリングレート: {sr}Hz")
        except Exception as e:
            error_msg = f"音声ファイル '{audio_path}' の読み込み中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue
        
        # 参照データの読み込み
        try:
            reference_data = load_reference_data(ref_path)
            logger.info(f"参照データを読み込みました: {ref_path}")
        except Exception as e:
            error_msg = f"参照ファイル '{ref_path}' の読み込み中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue
        
        # 各検出器で処理
        for detector_name, detector in detector_instances.items():
            logger.info(f"検出器 '{detector_name}' で評価を実行します")
            
            # ファイル・検出器ごとの出力ディレクトリ
            if output_dir:
                detector_output_dir = os.path.join(output_dir, detector_name)
                os.makedirs(detector_output_dir, exist_ok=True)
                
                # ファイル名ベースのディレクトリ
                file_output_dir = os.path.join(detector_output_dir, os.path.splitext(file_basename)[0])
                os.makedirs(file_output_dir, exist_ok=True)
            else:
                file_output_dir = None
            
            # 検出実行
            try:
                start_time = time.time()
                detection_result = detector.detect(audio_data, sr)
                end_time = time.time()
                
                detection_time = end_time - start_time
                logger.info(f"検出処理を完了しました。処理時間: {detection_time:.3f}秒")
            except Exception as e:
                error_msg = f"検出器 '{detector_name}' の実行中にエラーが発生しました: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
            
            # 検出結果の正規化
            try:
                normalized_result = normalize_detection_result(detection_result, sr)
                logger.info("検出結果を正規化しました")
            except Exception as e:
                error_msg = f"検出結果の正規化中にエラーが発生しました: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
            
            # 評価の実行（オンセット・オフセット・ピッチ・ノート評価を常に実行）
            detected_intervals = np.array(normalized_result.get('intervals', []))
            detected_pitches = np.array(normalized_result.get('note_pitches', []))
            
            reference_intervals = np.array(reference_data.get('intervals', []))
            reference_pitches = np.array(reference_data.get('note_pitches', []))
            
            # 評価実行
            try:
                evaluation_result = evaluate_detection_result(
                    detected_intervals=detected_intervals,
                    detected_pitches=detected_pitches,
                    reference_intervals=reference_intervals,
                    reference_pitches=reference_pitches,
                    evaluator_config=evaluator_config,
                    detector_result=detection_result
                )
                logger.info(f"評価が完了しました: オンセットF値={evaluation_result['onset']['f_measure']:.3f}, ノートF値={evaluation_result['note']['f_measure']:.3f}")
            except Exception as e:
                error_msg = f"評価処理中にエラーが発生しました: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
            
            # フレーム評価結果のキー名を標準化
            if 'frame' in evaluation_result:
                # 古いキー名を削除（もし存在すれば）
                for old_key in ['precision', 'recall', 'f_measure', 'accuracy']:
                    if old_key in evaluation_result['frame']:
                        del evaluation_result['frame'][old_key]

            # 結果の整理
            result = {
                'file': audio_path,
                'reference_file': ref_path,
                'detector_name': detector_name,
                'detection_time': detection_time,
                'evaluation': evaluation_result,
                'detector_params': detector_params.get(detector_name, {}),
                'ref_note_count': len(reference_pitches),
                'est_note_count': len(detected_pitches)
            }
            
            # JSON形式で結果を保存
            if file_output_dir and save_results_json:
                # 評価結果の保存
                eval_result_path = os.path.join(file_output_dir, 'evaluation_result.json')
                with open(eval_result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, cls=NumpyEncoder)
                logger.info(f"評価結果をJSONに保存しました: {eval_result_path}")
                
                # 検出結果の保存
                detection_result_path = os.path.join(file_output_dir, 'detection_result.json')
                with open(detection_result_path, 'w', encoding='utf-8') as f:
                    json.dump(normalized_result, f, indent=2, cls=NumpyEncoder)
                logger.info(f"検出結果をJSONに保存しました: {detection_result_path}")
            
            # プロット設定の準備
            if plot_config is None:
                plot_config = {}
            
            # プロットの保存
            if save_plots and plot_module_imported:
                try:
                    plot_path = plot_config.get('save_path')
                    if not plot_path and file_output_dir:
                        base_name = os.path.splitext(file_basename)[0]
                        plot_path = os.path.join(file_output_dir, f'{detector_name}_{base_name}_plot.{plot_format}')
                    
                    plot_detection_results(
                        audio_data=audio_data,
                        sr=sr,
                        detection_result=normalized_result,
                        reference_data=reference_data,
                        title=f"{detector_name} - {file_basename}",
                        show=plot_config.get('show', False),
                        save_path=plot_path,
                        figsize=plot_config.get('figsize', (12, 8))
                    )
                    logger.info(f"プロットを保存しました: {plot_path}")
                except Exception as e:
                    error_msg = f"プロットの生成中にエラーが発生しました: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # 結果リストに追加
            results.append(result)
    
    # 評価結果のサマリーを計算
    summary = _calculate_evaluation_summary(results)
    
    # サマリーのJSON保存
    if output_dir and save_results_json:
        summary_path = os.path.join(output_dir, 'evaluation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        logger.info(f"評価サマリーを保存しました: {summary_path}")
    
    # 返却値の作成
    if errors:
        return {
            "status": "partial_success" if results else "error",
            "results": results,
            "summary": summary,
            "errors": errors
        }
    else:
        return {
            "status": "success",
            "results": results,
            "summary": summary
        }

def _calculate_evaluation_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    評価結果の要約を計算します。

    Parameters
    ----------
    results : List[Dict[str, Any]]
        評価結果のリスト

    Returns
    -------
    Dict[str, Any]
        評価結果の要約
    """
    # 検出器ごとのサマリを計算
    detector_summary = {}
    
    for detector_name in {result["detector_name"] for result in results}:
        # 対象の検出器の結果のみをフィルタ
        detector_results = [r for r in results if r["detector_name"] == detector_name]
        
        # 検出時間の平均を計算
        avg_detection_time = sum(r["detection_time"] for r in detector_results) / len(detector_results)
        
        # 評価結果の集計
        onset_precision = [r["evaluation"]["onset"]["precision"] for r in detector_results]
        onset_recall = [r["evaluation"]["onset"]["recall"] for r in detector_results]
        onset_f_measure = [r["evaluation"]["onset"]["f_measure"] for r in detector_results]
        
        offset_precision = [r["evaluation"]["offset"]["precision"] for r in detector_results]
        offset_recall = [r["evaluation"]["offset"]["recall"] for r in detector_results]
        offset_f_measure = [r["evaluation"]["offset"]["f_measure"] for r in detector_results]
        
        pitch_precision = [r["evaluation"]["pitch"]["precision"] for r in detector_results]
        pitch_recall = [r["evaluation"]["pitch"]["recall"] for r in detector_results]
        pitch_f_measure = [r["evaluation"]["pitch"]["f_measure"] for r in detector_results]
        
        note_precision = [r["evaluation"]["note"]["precision"] for r in detector_results]
        note_recall = [r["evaluation"]["note"]["recall"] for r in detector_results]
        note_f_measure = [r["evaluation"]["note"]["f_measure"] for r in detector_results]
        
        # オプショナルな指標（フレーム評価）
        frame_pitch_metrics = {
            "voicing_recall": [],
            "voicing_false_alarm": [],
            "raw_pitch_accuracy": [],
            "raw_chroma_accuracy": [],
            "overall_accuracy": [],
            "precision": [],
            "recall": [],
            "f_measure": [],
            "accuracy": []
        }
        
        for r in detector_results:
            if "frame_pitch" in r["evaluation"]:
                for metric in frame_pitch_metrics:
                    if metric in r["evaluation"]["frame_pitch"]:
                        frame_pitch_metrics[metric].append(r["evaluation"]["frame_pitch"][metric])
        
        # サマリの作成
        summary = {
            "files_count": len(detector_results),
            "avg_detection_time": avg_detection_time,
            "onset": {
                "precision": sum(onset_precision) / len(onset_precision),
                "recall": sum(onset_recall) / len(onset_recall),
                "f_measure": sum(onset_f_measure) / len(onset_f_measure)
            },
            "offset": {
                "precision": sum(offset_precision) / len(offset_precision),
                "recall": sum(offset_recall) / len(offset_recall),
                "f_measure": sum(offset_f_measure) / len(offset_f_measure)
            },
            "pitch": {
                "precision": sum(pitch_precision) / len(pitch_precision),
                "recall": sum(pitch_recall) / len(pitch_recall),
                "f_measure": sum(pitch_f_measure) / len(pitch_f_measure)
            },
            "note": {
                "precision": sum(note_precision) / len(note_precision),
                "recall": sum(note_recall) / len(note_recall),
                "f_measure": sum(note_f_measure) / len(note_f_measure)
            }
        }
        
        # フレーム評価の結果が存在すれば追加
        frame_pitch = {}
        for metric, values in frame_pitch_metrics.items():
            if values:
                frame_pitch[metric] = sum(values) / len(values)
        
        if frame_pitch:
            summary["frame_pitch"] = frame_pitch
        
        detector_summary[detector_name] = summary
    
    return detector_summary

def create_summary(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    複数の評価結果から要約を作成します。

    Parameters
    ----------
    results : List[Dict[str, Any]]
        評価結果のリスト

    Returns
    -------
    Dict[str, Dict[str, Any]]
        検出器ごとの要約
    """
    # 評価結果のフィルタリング（正常な結果のみを対象とする）
    valid_results = []
    for result in results:
        evaluation = result.get("evaluation", {})
        if evaluation and all(key in evaluation for key in ["onset", "note", "pitch"]):
            valid_results.append(result)
    
    if not valid_results:
        logger.warning("有効な評価結果がありません")
        return {}
    
    # 検出器ごとの要約を計算
    return _calculate_evaluation_summary(valid_results)

def save_evaluation_result(result: Dict[str, Any], output_path: str) -> None:
    """
    評価結果をJSONファイルに保存します。

    Parameters
    ----------
    result : Dict[str, Any]
        評価結果の辞書
    output_path : str
        出力ファイルのパス
    """
    # 出力ディレクトリがない場合は作成
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # JSONに保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, cls=NumpyEncoder, indent=2, ensure_ascii=False) 

def save_detection_plot(audio_data: np.ndarray, sr: int, detection_result: Dict[str, Any], 
                       reference: Dict[str, Any], output_path: str, plot_format: str = 'png'):
    """
    検出結果と参照データのプロットを保存します。

    Parameters
    ----------
    audio_data : np.ndarray
        音声データ
    sr : int
        サンプリングレート
    detection_result : Dict[str, Any]
        検出結果
    reference : Dict[str, Any]
        参照データ
    output_path : str
        出力ファイルパス
    plot_format : str, optional
        プロット形式, by default 'png'
    """
    # プロットモジュールがインポートされているか確認
    if not plot_module_imported:
        logger.warning("プロットモジュールが利用できないため、プロットを保存できません")
        return
    
    # プロット用のデータを準備
    plot_detection = {
        'intervals': detection_result.get('intervals', np.array([]).reshape(0, 2)),
        'note_pitches': detection_result.get('note_pitches', np.array([])),
    }
    
    # フレームデータがある場合は追加
    if 'frame_times' in detection_result:
        plot_detection['frame_times'] = detection_result['frame_times']
    if 'frame_frequencies' in detection_result:
        plot_detection['frame_frequencies'] = detection_result['frame_frequencies']
    
    # リファレンスデータを準備
    plot_reference = {
        'intervals': reference.get('intervals', np.array([]).reshape(0, 2)),
        'note_pitches': reference.get('note_pitches', np.array([])),
    }
    
    # リファレンスにフレームデータがある場合は追加
    if 'frame_times' in reference:
        plot_reference['frame_times'] = reference['frame_times']
    if 'frame_frequencies' in reference:
        plot_reference['frame_frequencies'] = reference['frame_frequencies']
    
    # 検出プロットを生成して保存
    try:
        from src.visualization.plots import plot_detection_results
        plot_detection_results(
            audio_data=audio_data,
            sr=sr,
            detection_result=plot_detection,
            reference_data=plot_reference,
            save_path=output_path
        )
        logger.info(f"検出結果のプロットを保存しました: {output_path}")
    except Exception as e:
        logger.error(f"プロット生成中にエラーが発生しました: {str(e)}")
        logger.error(traceback.format_exc())

def save_result_json(result, output_path):
    """
    評価結果をJSONファイルに保存します。

    Parameters
    ----------
    result : Dict[str, Any]
        評価結果の辞書
    output_path : str
        出力ファイルのパス
    """
    try:
        # 出力ディレクトリがない場合は作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # JSONに保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"評価結果の保存中にエラーが発生しました: {str(e)}") 

def evaluate_detector(detector_name, detector_params, audio_file, ref_file, output_dir):
    """
    検出器を使用して音声ファイルを評価し、結果を保存します。

    Parameters
    ----------
    detector_name : str
        検出器の名前
    detector_params : dict
        検出器のパラメータ
    audio_file : str
        評価する音声ファイルのパス
    ref_file : str
        参照データファイルのパス
    output_dir : str
        評価結果を保存するディレクトリ
    
    Returns
    -------
    dict
        評価結果
    """
    logger = logging.getLogger(__name__)
    
    # 検出器を初期化
    try:
        detector = get_detector_class(detector_name)(**detector_params)
        logger.info(f"検出器 '{detector_name}' を初期化しました。パラメータ: {detector_params}")
    except Exception as e:
        logger.error(f"検出器 '{detector_name}' の初期化中にエラーが発生しました: {str(e)}")
        return None
    
    # 音声ファイルを読み込み
    try:
        audio, sr = load_audio_file(audio_file)
        logger.info(f"音声ファイルを読み込みました: {audio_file}, サンプリングレート: {sr}Hz")
    except Exception as e:
        logger.error(f"音声ファイルの読み込み中にエラーが発生しました: {str(e)}")
        return None
    
    # 参照データを読み込み
    try:
        ref_intervals, ref_pitches = load_reference_data(ref_file)
        logger.info(f"参照データを読み込みました: {ref_file}")
    except Exception as e:
        logger.error(f"参照データの読み込み中にエラーが発生しました: {str(e)}")
        return None
    
    # 検出処理を実行
    try:
        logger.info(f"検出器 '{detector_name}' で評価を実行します")
        start_time = time.time()
        detection_result = detector.detect(audio, sr)
        end_time = time.time()
        detection_time = end_time - start_time
        logger.info(f"検出処理を完了しました。処理時間: {detection_time:.3f}秒")
    except Exception as e:
        logger.error(f"検出処理中にエラーが発生しました: {str(e)}")
        return None
    
    # 検出結果を正規化
    detection_result.detector_name = detector_name
    detection_result.detection_time = detection_time
    
    # フレーム形式の場合はノート形式に変換
    if detection_result.is_frame_based():
        try:
            # フレーム形式からノート形式に変換
            est_intervals, est_pitches = detection_result.to_notes()
            logger.info(f"フレーム形式からノート形式に変換しました: {len(est_intervals)}個のノート")
        except Exception as e:
            logger.error(f"ノート変換中にエラーが発生しました: {str(e)}")
            return None
    else:
        # 既にノート形式の場合はそのまま使用
        est_intervals, est_pitches = detection_result.intervals, detection_result.pitches
    
    # 非正のピッチ値を修正（mir_evalの要件）
    est_pitches = np.array(est_pitches)
    nonpositive_mask = (est_pitches <= 0) | np.isinf(est_pitches) | np.isnan(est_pitches)
    if np.any(nonpositive_mask):
        logger.warning(f"検出結果に {np.sum(nonpositive_mask)} 個の無効なピッチ値があります - 1Hzに修正します")
        est_pitches_copy = est_pitches.copy()
        est_pitches_copy[nonpositive_mask] = 1.0
        est_pitches = est_pitches_copy
    
    # 評価を実行
    try:
        logger.info("検出結果を正規化しました")
        evaluation_result = evaluate_notes_mirex(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches,
            onset_tolerance=0.05,
            pitch_tolerance=50.0,
            offset_ratio=0.2,
            offset_min_tolerance=0.05
        )
        
        # フレーム評価（オプション）
        if detection_result.is_frame_based():
            frame_eval_result = evaluate_frames(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                est_times=detection_result.frame_times,
                est_freqs=detection_result.frame_frequencies,
                sample_rate=sr
            )
            evaluation_result['frame'] = frame_eval_result
        
        # オンセットとノートのF値をログに出力
        onset_f = evaluation_result['onset']['f_measure']
        note_f = evaluation_result['note']['f_measure']
        logger.info(f"評価が完了しました: オンセットF値={onset_f:.3f}, ノートF値={note_f:.3f}")
        
        # 評価結果をJSONに保存
        os.makedirs(output_dir, exist_ok=True)
        
        # 評価結果をJSON形式で保存
        with open(os.path.join(output_dir, 'evaluation_result.json'), 'w') as f:
            result_dict = {
                'audio_file': audio_file,
                'reference_file': ref_file,
                'detector': detector_name,
                'parameters': detector_params,
                'metrics': evaluation_result,
                'reference_note_count': len(ref_intervals),
                'estimated_note_count': len(est_intervals)
            }
            json.dump(result_dict, f, indent=2)
        logger.info(f"評価結果をJSONに保存しました: {os.path.join(output_dir, 'evaluation_result.json')}")
        
        # 検出結果をJSON形式で保存
        detection_result_dict = detection_result.to_dict()
        with open(os.path.join(output_dir, 'detection_result.json'), 'w') as f:
            json.dump(detection_result_dict, f, indent=2)
        logger.info(f"検出結果をJSONに保存しました: {os.path.join(output_dir, 'detection_result.json')}")
        
        return evaluation_result
        
    except Exception as e:
        logger.error(f"評価中にエラーが発生しました: {str(e)}")
        traceback.print_exc()
        return None 

def frame_evaluate_wrapper(detection_result, reference_data):
    """
    フレーム評価のラッパー関数
    
    Parameters
    ----------
    detection_result : Dict[str, Any]
        検出結果
    reference_data : Dict[str, Any]
        参照データ
    
    Returns
    -------
    Dict[str, float]
        フレーム評価結果
    """
    # フレーム情報がない場合はデフォルト値を返す
    if ('frame_times' not in detection_result or 'frame_frequencies' not in detection_result or
        'frame_times' not in reference_data or 'frame_frequencies' not in reference_data):
        return {
            'voicing_recall': 0.0,
            'voicing_false_alarm': 1.0,
            'raw_pitch_accuracy': 0.0,
            'raw_chroma_accuracy': 0.0,
            'overall_accuracy': 0.0
        }
    
    try:
        from src.evaluation.evaluation_frame import evaluate_frame_pitches
        
        return evaluate_frame_pitches(
            ref_times=reference_data['frame_times'],
            ref_freqs=reference_data['frame_frequencies'],
            est_times=detection_result['frame_times'],
            est_freqs=detection_result['frame_frequencies']
        )
    except Exception as e:
        logger.error(f"フレーム評価中にエラーが発生しました: {str(e)}")
        return {
            'voicing_recall': 0.0,
            'voicing_false_alarm': 1.0,
            'raw_pitch_accuracy': 0.0,
            'raw_chroma_accuracy': 0.0,
            'overall_accuracy': 0.0
        } 