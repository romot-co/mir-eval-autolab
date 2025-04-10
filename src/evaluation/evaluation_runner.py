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
import pandas as pd
import multiprocessing
import traceback
import tempfile
from pathlib import Path
from tqdm import tqdm
import yaml
from functools import partial
from typing import Optional, Dict, Any, List, Tuple, Union
import datetime

# ★ 修正: exception_utils のインポートを元に戻す
from src.utils.exception_utils import log_exception, create_error_result, ConfigError, FileError

from src.utils.audio_utils import load_audio_file, load_reference_data, make_output_path
from src.utils.json_utils import NumpyEncoder
from src.utils.detection_result import DetectionResult
from src.utils.detector_utils import get_detector_class, normalize_detection_result
from src.utils.path_utils import ensure_dir
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

CONFIG = {}
def load_global_config():
    global CONFIG
    try:
        from src.utils.path_utils import get_project_root
        config_path = get_project_root() / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                CONFIG = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning(f"Config file not found at {config_path}")
            CONFIG = {}
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {e}")
        CONFIG = {}

load_global_config()

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

def _evaluate_file_for_detector(
    detector_name: str,
    params: Optional[Dict[str, Any]],
    audio_ref_tuple: Tuple[str, str],
    output_dir: Path, # output_base_dir ではなく、この実行専用の output_dir を受け取る
    eval_id: str, # ファイル固有のID (ファイル名など)
    evaluator_config: Optional[Dict[str, Any]],
    save_plots: bool,
    plot_format: str,
    plot_config: Optional[Dict[str, Any]],
    save_results_json: bool,
    # detector_cache: Optional[Dict[str, Any]] = None # キャッシュはプロセス内で有効か？
) -> Dict[str, Any]:
    """
    1つの音声ファイルに対して、1つの検出器で評価を実行します。
    run_evaluationから並列処理で呼び出されることを想定しています。

    Args:
        detector_name (str): 検出器名。
        params (Optional[Dict[str, Any]]): 検出器のパラメータ。
        audio_ref_tuple (Tuple[str, str]): (音声ファイルパス, 参照ファイルパス) のタプル。
        output_dir (Path): このファイル評価の結果を保存するディレクトリ。
                         (例: .../output/evaluation_xxx/<eval_id>)
                         run_evaluation から渡される。
        eval_id (str): 評価ID (通常はファイル名など)。ファイル名生成に使用。
        evaluator_config (Optional[Dict[str, Any]]): 評価設定。
        save_plots (bool): プロットを保存するかどうか。
        plot_format (str): プロットのフォーマット。
        plot_config (Optional[Dict[str, Any]]): プロット設定。
        save_results_json (bool): 結果をJSONで保存するかどうか。

    Returns:
        Dict[str, Any]: 評価結果の辞書。
    """
    audio_file, ref_file = audio_ref_tuple
    start_time = time.time()
    logger.info(f"[{eval_id}] Evaluating detector '{detector_name}' on {os.path.basename(audio_file)}")
    
    try:
        # evaluate_detector を呼び出す (output_dir をそのまま渡す)
        result = evaluate_detector(
            detector_name=detector_name,
            detector_params=params,
            audio_file=audio_file,
            ref_file=ref_file,
            output_dir=output_dir, # evaluate_detector にはこのファイル用の出力ディレクトリを渡す
            eval_id=eval_id,
            evaluator_config=evaluator_config,
            save_plots=save_plots,
            plot_format=plot_format,
            plot_config=plot_config,
            save_results_json=save_results_json
        )
        end_time = time.time()
        result['processing_time'] = end_time - start_time
        logger.info(f"[{eval_id}] Finished evaluating '{detector_name}' in {result['processing_time']:.2f}s")
        return result

    except Exception as e:
        end_time = time.time()
        logger.error(f"[{eval_id}] Error evaluating detector '{detector_name}' on {audio_file}: {e}", exc_info=True)
        # エラーが発生した場合、エラー情報を含む結果辞書を返す
        error_info = create_error_result(f"Evaluation failed: {e}")
        return {
            'detector_name': detector_name,
            'audio_file': os.path.basename(audio_file),
            'ref_file': os.path.basename(ref_file),
            'params': params,
            'eval_id': eval_id,
            'output_dir': str(output_dir),
            'status': 'error',
            'error_message': str(e),
            'metrics': error_info, # エラーを示すメトリクス
            'processing_time': end_time - start_time
        }

def run_evaluation(
    detector_names: Union[str, List[str]],
    output_dir: str, # 必須引数 - detector_names の直後に配置
    # --- 以下、デフォルト値あり引数 ---
    audio_paths: Optional[Union[str, List[str]]] = None,
    ref_paths: Optional[Union[str, List[str]]] = None,
    detector_params: Optional[Dict[str, Dict[str, Any]]] = None,
    evaluator_config: Optional[Dict[str, Any]] = None,
    save_plots: bool = False,
    plot_format: str = 'png',
    save_results_json: bool = True,
    plot_config: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    num_procs: Optional[int] = None
) -> Dict[str, Any]:
    """
    指定されたファイルまたはデータセットに対して、指定された検出器を用いて評価を実行します。

    Parameters
    ----------
    detector_names : Union[str, List[str]]
        評価する検出器の名前（単一またはリスト）。
    output_dir : str
        評価結果（JSON、プロット、サマリー）を保存するディレクトリ。
        MCPサーバーからジョブ固有のディレクトリパスが渡されることを想定しています。
        例: /path/to/output/evaluation_<job_id>
    audio_paths : Optional[Union[str, List[str]]], optional
        評価する音声ファイルのパス（単一またはリスト）。dataset_name指定時は無視されます。
    ref_paths : Optional[Union[str, List[str]]], optional
        参照ファイルのパス（単一またはリスト）。audio_paths と同じ順序である必要があります。
        dataset_name指定時は無視されます。
    detector_params : Optional[Dict[str, Dict[str, Any]]], optional
        検出器ごとのパラメータ設定 (例: {'DetectorA': {'param1': 10}}), by default None
    evaluator_config : Optional[Dict[str, Any]], optional
        mir_evalで使用する評価設定, by default None
    save_plots : bool, optional
        各ファイルの評価結果のプロットを保存するかどうか, by default False
    plot_format : str, optional
        プロットの保存形式 ('png', 'pdf', etc.), by default 'png'
    save_results_json : bool, optional
        各ファイルの評価結果をJSONファイルとして保存するかどうか, by default True
    plot_config : Optional[Dict[str, Any]], optional
        プロット生成のための追加設定, by default None
    dataset_name : Optional[str], optional
        config.yamlで定義されたデータセット名。指定された場合、audio_paths/ref_pathsは無視されます。
    num_procs : Optional[int], optional
        評価に使用するプロセス数。Noneの場合は利用可能なCPUコア数を使用, by default None

    Returns
    -------
    Dict[str, Any]
        評価全体のサマリー結果を含む辞書。

    Raises
    -------
    ValueError
        入力パスの数や形式が不正な場合。
    FileNotFoundError
        指定されたファイルやディレクトリが見つからない場合。
    ConfigError
        設定ファイルに問題がある場合。
    """
    start_time_overall = time.time()
    logger.info(f"Starting evaluation process...")
    logger.info(f"Output directory: {output_dir}")

    # --- 引数の検証と正規化 --- #
    if isinstance(detector_names, str):
        detector_names = [detector_names]
    if not detector_names:
        raise ValueError("At least one detector name must be provided.")

    if not output_dir:
        raise ValueError("Output directory (output_dir) must be provided.")

    # output_dir を Path オブジェクトに変換し、存在確認 (ensure_dir は呼び出し元で行う想定だが念のため)
    try:
        output_dir_path = Path(output_dir).resolve()
        ensure_dir(output_dir_path) # ここでディレクトリがなければ作成
    except Exception as e:
         raise ValueError(f"Invalid or inaccessible output directory: {output_dir}. Error: {e}") from e

    # evaluator_config のデフォルトを CONFIG から取得 (あれば)
    if evaluator_config is None:
        evaluator_config = CONFIG.get('evaluation', {}).get('mir_eval_options', {})
        logger.info(f"Using default evaluator config: {evaluator_config}")

    # --- ファイルリストの準備 --- #
    audio_ref_pairs: List[Tuple[str, str]] = []
    if dataset_name:
        logger.info(f"Loading file list from dataset: {dataset_name}")
        try:
            from src.utils.path_utils import get_dataset_paths
            # データセット設定からパスとパターンを取得
            audio_dir, ref_dir, audio_pattern, ref_pattern = get_dataset_paths(CONFIG, dataset_name)
            # ファイルペアを検索
            audio_files = sorted(list(audio_dir.glob(audio_pattern)))
            if not audio_files:
                 raise FileNotFoundError(f"No audio files found in {audio_dir} matching '{audio_pattern}'")
            for audio_path in audio_files:
                # 対応する参照ファイルを探す (拡張子を除いたファイル名が一致すると仮定)
                base_name = audio_path.stem
                expected_ref_name = ref_pattern.replace('*', base_name)
                ref_path = ref_dir / expected_ref_name
                if ref_path.exists():
                    audio_ref_pairs.append((str(audio_path), str(ref_path)))
                else:
                    logger.warning(f"Reference file not found for {audio_path}: {ref_path}. Skipping this pair.")
            if not audio_ref_pairs:
                 raise FileNotFoundError(f"No matching audio/reference pairs found for dataset '{dataset_name}'")
            logger.info(f"Found {len(audio_ref_pairs)} audio/reference pairs in dataset '{dataset_name}'.")
        except (KeyError, FileNotFoundError, ConfigError) as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {e}")
            raise
    elif audio_paths and ref_paths:
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        if isinstance(ref_paths, str):
            ref_paths = [ref_paths]
        if len(audio_paths) != len(ref_paths):
            raise ValueError("Number of audio paths and reference paths must match.")
        # パスの存在確認 (ここでは行わない、_evaluate_file_for_detector 内で行う)
        for audio_p, ref_p in zip(audio_paths, ref_paths):
             audio_ref_pairs.append((str(Path(audio_p).resolve()), str(Path(ref_p).resolve())))
        logger.info(f"Using {len(audio_ref_pairs)} explicitly provided audio/reference pairs.")
    else:
        raise ValueError("Either 'dataset_name' or both 'audio_paths' and 'ref_paths' must be provided.")

    # --- 評価タスクの準備 --- #
    tasks = []
    for detector_name in detector_names:
        params = detector_params.get(detector_name) if detector_params else None
        for i, (audio_file, ref_file) in enumerate(audio_ref_pairs):
            # ファイル固有の評価IDを生成 (例: 001_filename)
            # eval_id = f"{i:03d}_{Path(audio_file).stem}"
            eval_id = Path(audio_file).stem # シンプルにファイル名ステムを使用

            # このファイル評価用の出力ディレクトリパスを作成 (run_evaluation の output_dir の下に)
            # eval_output_dir = output_dir_path / eval_id # 以前はこの階層があった
            # 今回の変更: _evaluate_file_for_detector には run_evaluation の output_dir をそのまま渡す
            # evaluate_detector がファイル名を付けて保存する

            task_args = {
                'detector_name': detector_name,
                'params': params,
                'audio_ref_tuple': (audio_file, ref_file),
                'output_dir': output_dir_path, # run_evaluation の output_dir を渡す
                'eval_id': eval_id, # ファイル識別子として渡す
                'evaluator_config': evaluator_config,
                'save_plots': save_plots,
                'plot_format': plot_format,
                'plot_config': plot_config,
                'save_results_json': save_results_json
            }
            tasks.append(task_args)

    logger.info(f"Prepared {len(tasks)} evaluation tasks for {len(detector_names)} detectors across {len(audio_ref_pairs)} files.")

    # --- 並列評価の実行 --- #
    results: List[Dict[str, Any]] = []
    if not num_procs:
        num_procs = multiprocessing.cpu_count()
        logger.info(f"Using {num_procs} processes for evaluation.")

    try:
        if num_procs > 1 and len(tasks) > 1:
            logger.info(f"Running evaluations in parallel with {num_procs} processes...")
            with multiprocessing.Pool(processes=num_procs) as pool:
                # functools.partial を使って _evaluate_file_for_detector をラップし、引数を固定
                # map や imap_unordered は単一引数の関数を期待するため、タプルや辞書をそのまま渡せない
                # -> タプルのリストを作成し、それを展開するヘルパー関数を使うか、Pool.starmapを使う
                #    または、各タスク引数を辞書のまま渡し、ワーカー関数で受け取る

                # starmap を使う場合: _evaluate_file_for_detector が引数を直接受け取る必要がある
                # -> 現在の実装ではキーワード引数を受け取るので starmap は適さない

                # map/imap_unordered を使う場合: タスク引数を辞書のまま渡す
                # func_wrapper = partial(_evaluate_file_for_detector, **common_args) # これは間違い

                # tqdm を使って進捗を表示
                with tqdm(total=len(tasks), desc="Evaluating Files") as pbar:
                    # imap_unordered で結果を非同期に取得
                    # 各タスク引数 (辞書) をそのまま map/imap_unordered に渡すことはできない
                    # -> 各タスク引数を展開して関数に渡すラッパーが必要
                    def worker_wrapper(task_dict):
                         return _evaluate_file_for_detector(**task_dict)

                    for result in pool.imap_unordered(worker_wrapper, tasks):
                        results.append(result)
                        pbar.update(1)
        else:
            logger.info("Running evaluations sequentially...")
            for task_args in tqdm(tasks, desc="Evaluating Files"):
                results.append(_evaluate_file_for_detector(**task_args))

    except KeyboardInterrupt:
        logger.warning("Evaluation process interrupted by user.")
        # プールを適切に終了させる (必要であれば)
        # if 'pool' in locals(): pool.terminate()
        raise # 中断を上位に伝える

    except Exception as e:
         logger.error(f"An error occurred during parallel evaluation: {e}", exc_info=True)
         # if 'pool' in locals(): pool.terminate()
         # ここで部分的な結果を返すか、エラーを発生させるか
         raise RuntimeError(f"Parallel evaluation failed: {e}") from e

    logger.info(f"Finished evaluating all {len(tasks)} tasks.")

    # --- 結果の集計と保存 --- #
    if not results:
        logger.warning("No evaluation results were generated.")
        return {"summary": {}, "all_results": []}

    try:
        summary = _calculate_evaluation_summary(results)
        summary["evaluation_config"] = evaluator_config # Add config used
        summary["num_files_evaluated"] = len(results)
        summary["detector_names"] = detector_names
        summary["dataset_name"] = dataset_name or "Custom files"
        summary["timestamp"] = datetime.datetime.now().isoformat()
        summary["total_duration_seconds"] = time.time() - start_time_overall

        # サマリーをJSONファイルに保存
        summary_json_path = output_dir_path / "summary.json"
        save_result_json(summary, str(summary_json_path))
        logger.info(f"Overall evaluation summary saved to {summary_json_path}")

        # サマリーを表形式 (CSV) でも保存 (オプション)
        try:
             summary_df = create_summary_dataframe(results)
             summary_csv_path = output_dir_path / "summary.csv"
             summary_df.to_csv(summary_csv_path, index=False)
             logger.info(f"Evaluation summary table saved to {summary_csv_path}")
             # コンソールにも表示
             print("\n--- Evaluation Summary ---")
             print_summary_statistics(summary_df)
             print("------------------------")
        except ImportError:
             logger.warning("Pandas not installed. Cannot save summary table or print statistics.")
        except Exception as df_err:
             logger.error(f"Failed to create or save summary dataframe: {df_err}", exc_info=True)

        # 全結果もサマリーに含める（ただし大きなデータになる可能性あり）
        # summary['all_results'] = results # オプション：必要なら含める

        return {"summary": summary, "all_results": results} # all_results も返す

    except Exception as e:
        logger.error(f"Error calculating or saving evaluation summary: {e}", exc_info=True)
        # サマリー計算に失敗しても、個々の結果は返す
        return {"summary": {"error": f"Summary calculation failed: {e}"}, "all_results": results}

def _calculate_evaluation_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    複数の評価結果からサマリー統計を計算します。

    Parameters
    ----------
    results : List[Dict[str, Any]]
        評価結果のリスト

    Returns
    -------
    Dict[str, Any]
        サマリー統計を含む辞書
    """
    summary: Dict[str, Any] = {'detectors': {}}
    all_metrics: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        detector_name = result.get('detector_name', 'unknown')
        metrics = result.get('metrics') # 各評価カテゴリ (note, onset, ...) の辞書

        if detector_name not in summary['detectors']:
            summary['detectors'][detector_name] = {
                'count': 0,
                'metrics': {}
            }
            all_metrics[detector_name] = {}

        summary['detectors'][detector_name]['count'] += 1

        if metrics and isinstance(metrics, dict):
            for category, values in metrics.items(): # category: note, onset, offset, pitch, frame_pitch
                if isinstance(values, dict):
                    if category not in summary['detectors'][detector_name]['metrics']:
                        summary['detectors'][detector_name]['metrics'][category] = {
                            'precision': [], 'recall': [], 'f_measure': [], 'overlap_ratio': [], 'accuracy': []
                        }
                    if category not in all_metrics[detector_name]:
                         all_metrics[detector_name][category] = {
                             'precision': [], 'recall': [], 'f_measure': [], 'overlap_ratio': [], 'accuracy': []
                         }

                    for metric_name, value in values.items():
                        if isinstance(value, (float, int)) and np.isfinite(value):
                             # サマリー用リストに追加 (後で平均などを計算)
                             if metric_name in summary['detectors'][detector_name]['metrics'][category]:
                                 summary['detectors'][detector_name]['metrics'][category][metric_name].append(value)
                             # 全体集計用リストに追加
                             if metric_name in all_metrics[detector_name][category]:
                                 all_metrics[detector_name][category][metric_name].append(value)
                        elif value is not None:
                             logger.debug(f"Skipping non-finite metric value: {detector_name}/{category}/{metric_name} = {value}")

    # --- 平均、標準偏差などを計算 --- #
    overall_summary_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for detector_name, detector_data in summary['detectors'].items():
        detector_summary = {}
        for category, metrics_lists in detector_data['metrics'].items():
            cat_summary = {}
            for metric_name, values_list in metrics_lists.items():
                if values_list:
                     # nan を除外して計算
                     valid_values = [v for v in values_list if np.isfinite(v)]
                     if valid_values:
                         mean_val = float(np.mean(valid_values))
                         std_val = float(np.std(valid_values))
                         median_val = float(np.median(valid_values))
                         cat_summary[f'{metric_name}_mean'] = mean_val
                         cat_summary[f'{metric_name}_std'] = std_val
                         cat_summary[f'{metric_name}_median'] = median_val
                         cat_summary[f'{metric_name}_count'] = len(valid_values)
                     else:
                          # 有効な値がない場合は NaN または 0 を設定
                          cat_summary[f'{metric_name}_mean'] = np.nan
                          cat_summary[f'{metric_name}_std'] = np.nan
                          cat_summary[f'{metric_name}_median'] = np.nan
                          cat_summary[f'{metric_name}_count'] = 0
                else:
                    # リストが空の場合もNaNを設定
                    cat_summary[f'{metric_name}_mean'] = np.nan
                    cat_summary[f'{metric_name}_std'] = np.nan
                    cat_summary[f'{metric_name}_median'] = np.nan
                    cat_summary[f'{metric_name}_count'] = 0
            if cat_summary: # 何らかの指標が計算された場合のみ追加
                 detector_summary[category] = cat_summary
        # 計算したサマリー統計を元の 'metrics' キーに格納 (上書き)
        detector_data['metrics'] = detector_summary

        # 全体サマリー用にも別途格納 (mean のみ)
        overall_detector_summary = {}
        for category, cat_summary_data in detector_summary.items():
             overall_cat_summary = {}
             for key, val in cat_summary_data.items():
                 if key.endswith('_mean') and np.isfinite(val):
                     metric_name = key[:-5] # '_mean' を除去
                     overall_cat_summary[metric_name] = val
             if overall_cat_summary:
                 overall_detector_summary[category] = overall_cat_summary
        if overall_detector_summary:
            overall_summary_metrics[detector_name] = overall_detector_summary

    summary['overall_metrics'] = overall_summary_metrics

    # 全体 (全検出器) のサマリーも計算
    global_metrics: Dict[str, Dict[str, List[float]]] = {}
    total_count = 0
    for detector_name, detector_all_metrics in all_metrics.items():
        total_count += summary['detectors'][detector_name]['count']
        for category, metrics_lists in detector_all_metrics.items():
            if category not in global_metrics:
                 global_metrics[category] = {
                     'precision': [], 'recall': [], 'f_measure': [], 'overlap_ratio': [], 'accuracy': []
                 }
            for metric_name, values_list in metrics_lists.items():
                 if metric_name in global_metrics[category]:
                     global_metrics[category][metric_name].extend(values_list)

    global_summary_stats: Dict[str, Dict[str, float]] = {}
    for category, metrics_lists in global_metrics.items():
         cat_summary = {}
         for metric_name, values_list in metrics_lists.items():
             if values_list:
                 valid_values = [v for v in values_list if np.isfinite(v)]
                 if valid_values:
                     cat_summary[f'{metric_name}_mean'] = float(np.mean(valid_values))
                     cat_summary[f'{metric_name}_std'] = float(np.std(valid_values))
                     cat_summary[f'{metric_name}_median'] = float(np.median(valid_values))
                     cat_summary[f'{metric_name}_count'] = len(valid_values)
                 else:
                      cat_summary[f'{metric_name}_mean'] = np.nan
                      cat_summary[f'{metric_name}_std'] = np.nan
                      cat_summary[f'{metric_name}_median'] = np.nan
                      cat_summary[f'{metric_name}_count'] = 0
             else:
                 cat_summary[f'{metric_name}_mean'] = np.nan
                 cat_summary[f'{metric_name}_std'] = np.nan
                 cat_summary[f'{metric_name}_median'] = np.nan
                 cat_summary[f'{metric_name}_count'] = 0
         if cat_summary:
             global_summary_stats[category] = cat_summary

    summary['global_summary'] = {
        'metrics': global_summary_stats,
        'total_evaluations': total_count
    }

    return summary

def create_summary(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Calculate summary statistics from a list of evaluation results."""
    summary_data: Dict[str, Dict[str, List[float]]] = {}
    for result in results:
        detector = result.get('detector_name', 'unknown')
        metrics = result.get('metrics', {})
        if detector not in summary_data:
            summary_data[detector] = {}
        for category, values in metrics.items():
            if category not in summary_data[detector]:
                summary_data[detector][category] = {}
            for metric, value in values.items():
                if metric not in summary_data[detector][category]:
                    summary_data[detector][category][metric] = []
                if isinstance(value, (int, float)) and not np.isnan(value):
                    summary_data[detector][category][metric].append(value)

    final_summary: Dict[str, Dict[str, Any]] = {}
    for detector, categories in summary_data.items():
        final_summary[detector] = {}
        for category, metrics in categories.items():
            final_summary[detector][category] = {}
            for metric, values in metrics.items():
                if values:
                    final_summary[detector][category][f'{metric}_mean'] = float(np.mean(values))
                    final_summary[detector][category][f'{metric}_std'] = float(np.std(values))
                else:
                    final_summary[detector][category][f'{metric}_mean'] = 0.0
                    final_summary[detector][category][f'{metric}_std'] = 0.0
    return final_summary

def save_evaluation_result(result: Dict[str, Any], output_path: str) -> None:
    """Save evaluation result to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, cls=NumpyEncoder, indent=2)
        logger.debug(f"Evaluation result saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation result to {output_path}: {e}", exc_info=True)

def save_detection_plot(
    audio_data: np.ndarray,
    sr: int,
    detection_result: Dict[str, Any],
    reference: Dict[str, Any],
    output_path: str, # Required arg
    # --- Optional args follow ---
    plot_format: str = 'png',
    plot_config: Optional[Dict[str, Any]] = None
):
    """Save detection result plot."""
    if not plot_module_imported:
        logger.warning("Plotting module not found, skipping plot generation.")
        return

    try:
        logger.info(f"プロットを {output_path} に保存しています...")
        # Call plot_detection_results with the corrected keyword argument
        plot_detection_results(
            audio_data=audio_data,
            sr=sr,
            detection_result=detection_result, # Use 'detection_result'
            reference=reference,
            output_path=output_path,
            plot_config=plot_config,
            plot_format=plot_format
        )
        logger.info(f"プロットを {output_path} に保存しました。")
    except Exception as e:
        logger.error(f"Failed to save detection plot to {output_path}: {e}", exc_info=True)

def save_result_json(result, output_path):
    """Save result dictionary to a JSON file using NumpyEncoder."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, cls=NumpyEncoder, indent=2)
        logger.debug(f"Result saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {output_path}: {e}", exc_info=True)

def evaluate_detector(
    detector_name: str,
    detector_params: Optional[Dict[str, Any]],
    audio_file: str,
    ref_file: str,
    output_dir: Path, # Expect Path object
    eval_id: str,     # Moved required argument before optional ones
    evaluator_config: Optional[Dict[str, Any]] = None,
    save_plots: bool = False,
    plot_format: str = 'png',
    plot_config: Optional[Dict[str, Any]] = None,
    save_results_json: bool = True
) -> Dict[str, Any]:
    """
    単一の検出器と単一の音声ファイルで評価を実行します。

    Args:
        detector_name (str): 検出器名。
        detector_params (Optional[Dict[str, Any]]): 検出器パラメータ。
        audio_file (str): 音声ファイルパス。
        ref_file (str): 参照ファイルパス。
        output_dir (Path): この評価結果を保存するディレクトリ (例: /path/to/output/evaluation_xxx)。
                           ファイル名はこの関数内で生成されます。
        eval_id (str): 評価ID (通常はファイル名など)。ファイル名生成に使用。
        evaluator_config (Optional[Dict[str, Any]]): 評価設定。
        save_plots (bool): プロットを保存するかどうか。
        plot_format (str): プロット形式。
        plot_config (Optional[Dict[str, Any]]): プロット設定。
        save_results_json (bool): JSON結果を保存するかどうか。

    Returns:
        Dict[str, Any]: 評価結果の辞書。
    """
    start_time = time.time()
    eval_id = Path(audio_file).stem # ファイル名ステムを評価IDとして使用
    logger.info(f"Evaluating {detector_name} on {eval_id}...")

    try:
        # 1. 検出器クラスの取得とインスタンス化
        DetectorClass = get_detector_class(detector_name)
        if detector_params:
            detector = DetectorClass(**detector_params)
        else:
            detector = DetectorClass()

        # 2. 音声ファイルの読み込み
        audio_data, sr = load_audio_file(audio_file)

        # 3. 検出の実行
        detection_start_time = time.time()
        detection_result_raw = detector.detect(audio_data, sr)
        detection_duration = time.time() - detection_start_time

        # 4. 検出結果の正規化
        #    (detector_result に 'intervals', 'pitches', 'frame_times', 'frame_frequencies' を含む辞書にする)
        normalized_intervals, normalized_pitches = normalize_detection_result(
            detection_result_raw,
            detector_name=detector_name,
            audio_file=audio_file
        )

        # 5. 参照データの読み込み
        reference_data = load_reference_data(ref_file)
        if reference_data is None:
            raise FileNotFoundError(f"Failed to load reference data from {ref_file}")

        # 6. 評価の実行
        evaluation_metrics = evaluate_detection_result(
            detected_intervals=normalized_intervals,
            detected_pitches=normalized_pitches,
            reference_intervals=reference_data['intervals'],
            reference_pitches=reference_data['pitches'],
            detection_result_dict=detection_result_raw,
            reference_dict=reference_data,
            evaluator_config=evaluator_config
        )

        # 7. 結果辞書の作成
        result = {
            'eval_id': eval_id,
            'audio_file': os.path.basename(audio_file),
            'ref_file': os.path.basename(ref_file),
            'detector_name': detector_name,
            'detector_params': detector_params,
            'evaluation_metrics': evaluation_metrics,
            'detection_time': detection_duration,
            'valid': True, # Mark as valid if no exception occurred so far
            'error_message': None
        }

        # 8. 結果の保存 (JSON)
        if save_results_json:
            json_filename = f"{eval_id}_evaluation.json"
            json_output_path = output_dir / json_filename
            save_result_json(result, str(json_output_path))
            result['json_file'] = json_filename # Store relative path
            logger.debug(f"Saved evaluation results to {json_output_path}")

        # 9. プロットの保存
        if save_plots:
            plot_filename = f"{eval_id}_detection_plot.{plot_format}"
            plot_output_path = output_dir / plot_filename
            save_detection_plot(
                audio_data=audio_data,
                sr=sr,
                detection_result=detection_result_raw,
                reference=reference_data,
                output_path=str(plot_output_path),
                plot_format=plot_format,
                plot_config=plot_config
            )
            result['plot_file'] = plot_filename # Store relative path

        logger.info(f"Successfully evaluated {eval_id}")
        return result

    except Exception as e:
        # ... (error handling remains the same, ensuring 'valid' is False)
        result = {
            'eval_id': eval_id,
            'audio_file': Path(audio_file).name if audio_file else 'N/A',
            'ref_file': Path(ref_file).name if ref_file else 'N/A',
            'detector_name': detector_name,
            'valid': False,
            'error_message': str(e)
        }
        # Log the full error details
        log_exception(logger, e, f"Error during evaluation for {eval_id}")
        return result

def frame_evaluate_wrapper(detection_result, reference_data):
    """フレームベース評価を実行するラッパー"""
    if not detection_result or not reference_data: return {}
    ref_time = reference_data.get('frame_times')
    ref_freq = reference_data.get('frame_frequencies')
    est_time = detection_result.get('frame_times')
    est_freq = detection_result.get('frame_frequencies')

    if ref_time is None or ref_freq is None or est_time is None or est_freq is None:
        logger.debug("Frame data missing, skipping frame-based evaluation.")
        return {}

    # Ensure non-negative frequencies
    ref_freq = np.maximum(ref_freq, 0)
    est_freq = np.maximum(est_freq, 0)

    try:
        frame_metrics = mir_eval.melody.evaluate(ref_time=ref_time, ref_freq=ref_freq, 
                                                est_time=est_time, est_freq=est_freq)
        # Convert numpy types to standard python types for JSON serialization
        serializable_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                for k, v in frame_metrics.items()}
        return serializable_metrics
    except Exception as e:
        logger.error(f"Error during frame-based evaluation: {e}", exc_info=True)
        return create_error_result(f"FrameEvalError: {e}") 