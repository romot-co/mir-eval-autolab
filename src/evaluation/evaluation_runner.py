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

import datetime
import json
import logging
import multiprocessing
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mir_eval
import mir_eval.transcription
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.evaluation.evaluation_io import (
    create_summary_dataframe,
    print_summary_statistics,
)
from src.evaluation.evaluation_io import save_detection_plot as save_detection_plot_io
from src.evaluation.evaluation_io import save_evaluation_result
from src.utils.audio_utils import load_audio_file, load_reference_data, make_output_path
from src.utils.detection_result import DetectionResult
from src.utils.detector_utils import get_detector_class, normalize_detection_result

# ★ 修正: exception_utils のインポートを元に戻す
from src.utils.exception_utils import (
    ConfigError,
    EvaluationError,
    FileError,
    log_exception,
)
from src.utils.json_utils import NumpyEncoder
from src.utils.path_utils import ensure_dir, find_files, get_dataset_paths

logger = logging.getLogger(__name__)  # モジュールレベルでロガーを定義

# プロット関連のインポート (エラーハンドリング付き)
try:
    # from src.visualization.plots import plot_detection_results # コメントアウト
    plot_module_imported = False  # 直接 False に設定
except ImportError:
    plot_module_imported = False
    logger.warning(
        "matplotlib または関連モジュールが見つかりません。プロット機能は無効になります。"
    )

# グローバル設定のロード
CONFIG = {}


def load_global_config():
    global CONFIG
    try:
        from src.utils.path_utils import get_project_root

        config_path = get_project_root() / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                CONFIG = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning(f"Config file not found at {config_path}")
            CONFIG = {}
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {e}")
        CONFIG = {}


load_global_config()

from src.evaluation.evaluation_frame import evaluate_frame_pitches, notes_to_frames


def evaluate_detection_result(
    detected_intervals: np.ndarray,
    detected_pitches: np.ndarray,
    reference_intervals: np.ndarray,
    reference_pitches: np.ndarray,
    evaluator_config: Optional[Dict[str, Any]] = None,
    detector_result: Optional[
        Dict[str, Any]
    ] = None,  # 検出器の生の結果を含む辞書 (フレーム情報など)
    logger: logging.Logger = logging.getLogger(__name__),
) -> Dict[str, Dict[str, float]]:
    """
    検出結果を評価し、各種評価指標を計算します。

    mir_eval ライブラリの関数を利用して、ノートベースおよびフレームベースの指標を計算します。

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
        - tolerance_onset: オンセット許容誤差（秒）
        - tolerance_pitch: ピッチ許容誤差（セント）
        - offset_ratio: オフセット許容誤差の比率 (offset_min_toleranceも考慮)
        - offset_min_tolerance: オフセット許容誤差の最小値（秒）
        - use_pitch_chroma: ピッチクラスのみで評価するかどうか (現在未使用)
        - use_strict_offset: オフセットも評価に含めるかどうか (Trueの場合、`mir_eval.transcription.precision_recall_f1_overlap`を使用。Falseの場合、`mir_eval.transcription.precision_recall_f1`を使用)
    detector_result : Optional[Dict[str, Any]], optional
        検出器の出力結果の辞書, by default None
        - frame_times: フレーム時刻の配列
        - frame_frequencies: フレームごとの周波数の配列

    Returns
    -------
    Dict[str, Dict[str, float]]
        評価結果の辞書。以下のキーを含みます：
        - onset: オンセットのみの評価 (`mir_eval.transcription.onset_precision_recall_f1`)
            - precision, recall, f_measure
        - offset: オフセットのみの評価 (`mir_eval.transcription.offset_precision_recall_f1`)
            - precision, recall, f_measure
        - note: ノートベース評価（オンセット、ピッチ、オプションでオフセット）
                 (`mir_eval.transcription.precision_recall_f1_overlap` または `mir_eval.transcription.precision_recall_f1`)
            - precision, recall, f_measure, overlap_ratio (overlap使用時のみ)
        - pitch: ピッチのみの評価 (現在`note`と同じ関数で計算、将来的に分離可能性あり)
            - precision, recall, f_measure
        - frame_pitch: フレームベースの評価結果 (`mir_eval.melody.evaluate`) (フレームデータがある場合のみ)
            - raw_pitch_accuracy: 生のピッチ正解率
            - raw_chroma_accuracy: クロマ正解率
            - voicing_recall: 有声区間再現率
            - voicing_false_alarm: 有声区間誤検出率
            - overall_accuracy: 全体正解率

    注意
    ----
    この関数は主に2種類の評価を行います：

    1. ノートベース評価 (onset, offset, note, pitch):
       - `mir_eval.transcription` モジュールの関数を使用します。
       - `note` および `pitch` の指標は、`use_strict_offset` パラメータに応じて、オフセットを含めるか (`precision_recall_f1_overlap`)、含めないか (`precision_recall_f1`) を切り替えます。
       - ポリフォニックなデータ（和音など）を評価することを意図しています。

    2. フレームベース評価 (frame_pitch):
       - `mir_eval.melody.evaluate` 関数を使用します。
       - 各時間フレームにおける単一のピッチ（基本周波数 F0）を評価します。ポリフォニーは考慮されません。
       - `detector_result` に `frame_times` と `frame_frequencies` が含まれている場合にのみ計算されます。
       - ノートベース評価とは異なる側面を評価するため、**補助的な指標**として参照してください。
    """
    if evaluator_config is None:
        evaluator_config = {}

    # 検出結果と参照データのチェック
    detected_intervals = np.array(detected_intervals)
    detected_pitches = np.array(detected_pitches)
    reference_intervals = np.array(reference_intervals)
    reference_pitches = np.array(reference_pitches)

    logger = logging.getLogger("evaluate")

    # パラメータの取得
    onset_tolerance = evaluator_config.get("tolerance_onset", 0.05)  # デフォルト: 50ms
    offset_ratio = evaluator_config.get("offset_ratio", 0.2)  # デフォルト: 0.2
    offset_min_tolerance = evaluator_config.get(
        "offset_min_tolerance", 0.05
    )  # デフォルト: 50ms
    pitch_tolerance = evaluator_config.get(
        "tolerance_pitch", 50.0
    )  # デフォルト: 50セント
    use_pitch_chroma = evaluator_config.get(
        "use_pitch_chroma", False
    )  # デフォルト: False
    use_offset = evaluator_config.get(
        "use_strict_offset", True
    )  # オフセットを評価するかどうか

    # データの検証
    if len(detected_intervals) != len(detected_pitches):
        logger.warning(
            f"検出区間とピッチの数が一致しません: {len(detected_intervals)} vs {len(detected_pitches)}"
        )
        return create_error_result("DetectionMismatch")

    if len(reference_intervals) != len(reference_pitches):
        logger.warning(
            f"参照区間とピッチの数が一致しません: {len(reference_intervals)} vs {len(reference_pitches)}"
        )
        return create_error_result("ReferenceMismatch")

    # 0Hz以下の無効なピッチ値をチェック
    detected_pitch_mask = detected_pitches <= 0
    reference_pitch_mask = reference_pitches <= 0

    if np.any(detected_pitch_mask):
        logger.warning(
            f"検出結果に {np.sum(detected_pitch_mask)} 個の0Hz以下のピッチ値があります。評価用に1Hzに補正します。"
        )
        # 複製を作成して元の配列を変更しない
        detected_pitches_for_eval = detected_pitches.copy()
        detected_pitches_for_eval[detected_pitch_mask] = 1.0  # 1Hzに設定（ほぼ無音）
    else:
        detected_pitches_for_eval = detected_pitches

    if np.any(reference_pitch_mask):
        logger.warning(
            f"参照データに {np.sum(reference_pitch_mask)} 個の0Hz以下のピッチ値があります。評価用に1Hzに補正します。"
        )
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
        strict=False,  # MIREX互換
    )
    onset_eval = {"precision": onset_prec, "recall": onset_rec, "f_measure": onset_f}

    # ==== オフセットのみの評価 ====
    offset_prec, offset_rec, offset_f = (
        mir_eval.transcription.offset_precision_recall_f1(
            reference_intervals,
            detected_intervals,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
            strict=False,  # MIREX互換
        )
    )
    offset_eval = {
        "precision": offset_prec,
        "recall": offset_rec,
        "f_measure": offset_f,
    }

    # ==== ノート評価（オンセット+ピッチ+オフセット） ====
    if use_offset:
        # オフセットも含めて評価
        prec_note, rec_note, f_note, avg_overlap_note = (
            mir_eval.transcription.precision_recall_f1_overlap(
                reference_intervals,
                reference_pitches_for_eval,
                detected_intervals,
                detected_pitches_for_eval,
                onset_tolerance=onset_tolerance,
                pitch_tolerance=pitch_tolerance,
                offset_ratio=offset_ratio,
                offset_min_tolerance=offset_min_tolerance,
                strict=False,  # MIREX互換
            )
        )
        note_eval = {
            "precision": prec_note,
            "recall": rec_note,
            "f_measure": f_note,
            "overlap_ratio": avg_overlap_note,
        }
    else:
        # オンセット+ピッチのみで評価（オフセットを無視）
        prec_note, rec_note, f_note, _ = (
            mir_eval.transcription.precision_recall_f1_overlap(
                reference_intervals,
                reference_pitches_for_eval,
                detected_intervals,
                detected_pitches_for_eval,
                onset_tolerance=onset_tolerance,
                pitch_tolerance=pitch_tolerance,
                offset_ratio=None,  # オフセット無視
                strict=False,  # MIREX互換
            )
        )
        note_eval = {"precision": prec_note, "recall": rec_note, "f_measure": f_note}

    # ==== ピッチ評価（オンセット+ピッチのみの評価、オフセット無視） ====
    # オンセット+ピッチのみで評価（オフセットを無視）
    prec_pitch, rec_pitch, f_pitch, avg_overlap_pitch = (
        mir_eval.transcription.precision_recall_f1_overlap(
            reference_intervals,
            reference_pitches_for_eval,
            detected_intervals,
            detected_pitches_for_eval,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance,
            offset_ratio=None,  # オフセット無視
            strict=False,  # MIREX互換
        )
    )
    pitch_eval = {"precision": prec_pitch, "recall": rec_pitch, "f_measure": f_pitch}

    # フレーム評価（必要な場合）
    frame_pitch_result = {
        "voicing_recall": 0.0,
        "voicing_false_alarm": 1.0,
        "raw_pitch_accuracy": 0.0,
        "raw_chroma_accuracy": 0.0,
        "overall_accuracy": 0.0,
    }

    if (
        detector_result is not None
        and "frame_times" in detector_result
        and "frame_frequencies" in detector_result
    ):
        # 検出器からフレームデータを取得
        est_frame_times = detector_result.get("frame_times", [])
        est_frame_freqs = detector_result.get("frame_frequencies", [])

        # フレーム評価を実行
        try:
            from src.evaluation.evaluation_frame import (
                evaluate_frame_pitches,
                notes_to_frames,
            )

            # 参照フレームデータの取得（detector_resultから取得するか、参照ノートデータから生成）
            ref_frame_times = []
            ref_frame_freqs = []

            # detector_resultから参照フレームデータを取得（存在する場合）
            if (
                "reference_frame_times" in detector_result
                and "reference_frame_frequencies" in detector_result
            ):
                ref_frame_times = detector_result.get("reference_frame_times", [])
                ref_frame_freqs = detector_result.get("reference_frame_frequencies", [])

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
                    end_time=end_time,
                )

            # フレーム評価の実行
            frame_pitch_result = evaluate_frame_pitches(
                ref_times=ref_frame_times,
                ref_freqs=ref_frame_freqs,
                est_times=est_frame_times,
                est_freqs=est_frame_freqs,
                pitch_tolerance=pitch_tolerance,
                use_pitch_chroma=use_pitch_chroma,
            )

            # 結果をログに出力
            logger.info(
                f"フレーム評価結果 - 有声再現率: {frame_pitch_result['voicing_recall']:.3f}, ピッチ精度: {frame_pitch_result['raw_pitch_accuracy']:.3f}, 総合精度: {frame_pitch_result['overall_accuracy']:.3f}"
            )

        except Exception as e:
            logger.error(f"フレーム評価中にエラーが発生しました: {str(e)}")
            logger.error(traceback.format_exc())

    # 全体の評価結果をまとめる
    evaluation_result = {
        "onset": onset_eval,
        "note": note_eval,
        "pitch": pitch_eval,
        "offset": offset_eval,
        "frame_pitch": frame_pitch_result,
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
        "voicing_recall": 0.0,
        "voicing_false_alarm": 1.0,
        "raw_pitch_accuracy": 0.0,
        "raw_chroma_accuracy": 0.0,
        "overall_accuracy": 0.0,
    }


def _evaluate_file_for_detector(
    detector_name: str,
    params: Optional[Dict[str, Any]],
    audio_ref_tuple: Tuple[str, str],
    output_dir: Path,  # output_base_dir ではなく、この実行専用の output_dir を受け取る
    eval_id: str,  # ファイル固有のID (ファイル名など)
    evaluator_config: Optional[Dict[str, Any]],
    save_plots: bool,
    plot_format: str,
    plot_config: Optional[Dict[str, Any]],
    save_results_json: bool,
    logger: Optional[logging.Logger] = None,  # logger引数を追加
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
        logger: 使用するロガーインスタンス。

    Returns:
        Dict[str, Any]: 評価結果の辞書。
    """
    if logger is None:
        logger = logging.getLogger(
            f"_evaluate_file.{eval_id}"
        )  # プロセス固有のロガー名
        if not logger.handlers:
            # ハンドラがなければ基本的な設定を追加（メインプロセスで設定済みの想定）
            # ここで setup_logger を呼ぶと重複設定になる可能性があるので注意
            pass  # 必要に応じてファイルハンドラなどを追加

    audio_file, ref_file = audio_ref_tuple
    start_time = time.time()
    logger.info(
        f"[{eval_id}] Evaluating detector '{detector_name}' on {os.path.basename(audio_file)}"
    )

    try:
        # evaluate_detector を呼び出す (output_dir をそのまま渡す)
        result = evaluate_detector(
            detector_name=detector_name,
            detector_params=params,
            audio_file=audio_file,
            ref_file=ref_file,
            output_dir=output_dir,  # evaluate_detector にはこのファイル用の出力ディレクトリを渡す
            eval_id=eval_id,
            evaluator_config=evaluator_config,
            save_plots=save_plots,
            plot_format=plot_format,
            plot_config=plot_config,
            save_results_json=save_results_json,
            logger=logger,  # ★ 修正: logger を evaluate_detector に渡す
        )
        end_time = time.time()
        result["processing_time"] = end_time - start_time
        logger.info(
            f"[{eval_id}] Finished evaluating '{detector_name}' in {result['processing_time']:.2f}s"
        )
        return result

    except Exception as e:
        end_time = time.time()
        logger.error(
            f"[{eval_id}] Error evaluating detector '{detector_name}' on {audio_file}: {e}",
            exc_info=True,
        )
        # エラーが発生した場合、エラー情報を含む結果辞書を返す
        error_info = create_error_result(f"Evaluation failed: {e}")
        return {
            "detector_name": detector_name,
            "audio_file": os.path.basename(audio_file),
            "ref_file": os.path.basename(ref_file),
            "params": params,
            "eval_id": eval_id,
            "output_dir": str(output_dir),
            "status": "error",
            "error_message": str(e),
            "metrics": error_info,  # エラーを示すメトリクス
            "processing_time": end_time - start_time,
        }


def run_evaluation(
    detector_names: Union[str, List[str]],
    output_dir: str,
    # --- 以下、デフォルト値あり引数 ---
    audio_paths: Optional[Union[str, List[str]]] = None,
    ref_paths: Optional[Union[str, List[str]]] = None,
    detector_params: Optional[Dict[str, Dict[str, Any]]] = None,
    evaluator_config: Optional[Dict[str, Any]] = None,
    save_plots: bool = False,
    plot_format: str = "png",
    save_results_json: bool = True,
    plot_config: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    num_procs: Optional[int] = None,
    num_folds: Optional[int] = None,  # ★ 追加: クロスバリデーション用
    fold_index: Optional[int] = None,  # ★ 追加: クロスバリデーション用
) -> Dict[str, Any]:
    """
    指定された検出器とデータセット/ファイルリストで評価を実行します。

    検出器の出力を参照データと比較し、各種評価指標を計算します。
    結果は指定されたディレクトリにJSONファイルとして保存され、
    オプションでプロットも保存できます。
    クロスバリデーションのためのフォールド指定も可能です。

    Parameters
    ----------
    detector_names : Union[str, List[str]]
        評価する検出器クラス名のリストまたは単一の文字列。
    output_dir : str
        評価結果 (JSON, プロットなど) を保存するディレクトリ。
    audio_paths : Optional[Union[str, List[str]]], optional
        評価対象の音声ファイルパスのリストまたは単一の文字列。
        `dataset_name` が指定されていない場合に必須。
    ref_paths : Optional[Union[str, List[str]]], optional
        評価対象の参照ファイルパスのリストまたは単一の文字列。
        `dataset_name` が指定されていない場合に必須。
    detector_params : Optional[Dict[str, Dict[str, Any]]], optional
        検出器ごとのパラメータ設定を含む辞書。キーは検出器名。
    evaluator_config : Optional[Dict[str, Any]], optional
        `mir_eval` に渡す評価設定 (例: onset_tolerance)。
        省略した場合、config.yaml のデフォルト値を使用。
    save_plots : bool, optional
        評価結果のプロットを保存するかどうか。デフォルトは False。
    plot_format : str, optional
        保存するプロットの形式 (例: 'png', 'pdf')。デフォルトは 'png'。
    save_results_json : bool, optional
        個々のファイル評価結果をJSONで保存するかどうか。デフォルトは True。
    plot_config : Optional[Dict[str, Any]], optional
        プロット生成関数に渡す追加設定。
    dataset_name : Optional[str], optional
        config.yaml で定義されたデータセット名。
        指定された場合、`audio_paths` と `ref_paths` は無視されます。
    num_procs : Optional[int], optional
        評価を並列実行するプロセス数。
        None の場合は利用可能なCPUコア数を使用。
        1 の場合は逐次実行。
    num_folds : Optional[int], optional
        クロスバリデーションで使用するフォールド数。
        指定する場合、`fold_index` も必須。デフォルトは None (CVなし)。
    fold_index : Optional[int], optional
        クロスバリデーションで評価対象とするフォールドのインデックス (0始まり)。
        `num_folds` が指定された場合に必須。デフォルトは None。

    Returns
    -------
    Dict[str, Any]
        評価結果のサマリーと、全ファイルの詳細な評価結果を含む辞書。
        {'summary': dict, 'all_results': list}

    Raises
    ------
    ValueError
        引数が不正な場合 (例: 必須引数不足、パス不一致、不正なフォールド指定)。
    ConfigError
        設定ファイルの読み込みや解釈に失敗した場合。
    FileError
        ファイル/ディレクトリが見つからない場合。
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

    # ★ クロスバリデーション引数検証
    if num_folds is not None:
        if fold_index is None:
            raise ValueError("fold_index must be specified when num_folds is set.")
        if not isinstance(num_folds, int) or num_folds < 2:
            raise ValueError(f"num_folds must be an integer >= 2, but got {num_folds}.")
        if not isinstance(fold_index, int) or not (0 <= fold_index < num_folds):
            raise ValueError(
                f"fold_index must be an integer between 0 and {num_folds - 1}, but got {fold_index}."
            )
        logger.info(
            f"Cross-validation enabled: Evaluating Fold {fold_index + 1} of {num_folds}."
        )
    elif fold_index is not None:
        raise ValueError("num_folds must be specified when fold_index is set.")

    # output_dir を Path オブジェクトに変換し、存在確認
    try:
        output_dir_path = Path(output_dir).resolve()
        ensure_dir(output_dir_path)
    except Exception as e:
        raise ValueError(
            f"Invalid or inaccessible output directory: {output_dir}. Error: {e}"
        ) from e

    # evaluator_config のデフォルトを CONFIG から取得
    if evaluator_config is None:
        evaluator_config = CONFIG.get("evaluation", {}).get("mir_eval_options", {})
        logger.info(f"Using default evaluator config: {evaluator_config}")

    # --- ファイルリストの準備 --- #
    audio_ref_pairs_path: List[Tuple[Path, Path]] = []
    if dataset_name:
        logger.info(f"Loading file list from dataset: {dataset_name}")
        try:
            audio_dir, label_dir, file_pairs = get_dataset_paths(CONFIG, dataset_name)
            audio_ref_pairs_path = file_pairs
            if not audio_ref_pairs_path:
                raise FileNotFoundError(
                    f"No valid audio/reference file pairs found for dataset '{dataset_name}'."
                )
            logger.info(
                f"Found {len(audio_ref_pairs_path)} audio/reference pairs in dataset '{dataset_name}'."
            )
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
        for audio_p, ref_p in zip(audio_paths, ref_paths):
            audio_ref_pairs_path.append(
                (Path(audio_p).resolve(), Path(ref_p).resolve())
            )
        logger.info(
            f"Using {len(audio_ref_pairs_path)} explicitly provided audio/reference pairs."
        )
    else:
        raise ValueError(
            "Either 'dataset_name' or both 'audio_paths' and 'ref_paths' must be provided."
        )

    # ★ クロスバリデーション: ファイルリストのフィルタリング
    if num_folds is not None and fold_index is not None:
        n_files = len(audio_ref_pairs_path)
        if n_files < num_folds:
            # フォールド数よりファイル数が少ない場合のエラー処理
            raise ValueError(
                f"Number of files ({n_files}) is less than the number of folds ({num_folds}). Cannot perform cross-validation."
            )

        # ファイルリストをソートして再現性を確保 (例: オーディオファイルパスでソート)
        audio_ref_pairs_path.sort(key=lambda pair: pair[0])

        # フォールドに分割 (np.array_split を使うか、手動で分割)
        # 手動での分割例 (np.array_splitがなければこちらを使用)
        fold_size = n_files // num_folds
        remainder = n_files % num_folds
        start_index = fold_index * fold_size + min(fold_index, remainder)
        end_index = start_index + fold_size + (1 if fold_index < remainder else 0)

        selected_files = audio_ref_pairs_path[start_index:end_index]

        logger.info(
            f"Applying cross-validation: Using fold {fold_index + 1}/{num_folds}. Selected {len(selected_files)} out of {n_files} files."
        )
        audio_ref_pairs_path = selected_files  # 評価対象を絞り込んだリストで上書き

    # --- 評価タスクの準備 --- #
    tasks = []
    if not audio_ref_pairs_path:
        logger.warning(
            "No files selected for evaluation after filtering (or initially)."
        )
        # 空の結果を返す処理を追加することも検討
        # return {"summary": {"warning": "No files evaluated"}, "all_results": []}
        # ここでは後続の処理で空リストとして扱われるようにする

    for detector_name in detector_names:
        params = detector_params.get(detector_name) if detector_params else None
        for i, (audio_file_path, ref_file_path) in enumerate(audio_ref_pairs_path):
            eval_id = audio_file_path.stem
            task_args = {
                "detector_name": detector_name,
                "params": params,
                "audio_ref_tuple": (str(audio_file_path), str(ref_file_path)),
                "output_dir": output_dir_path,
                "eval_id": eval_id,
                "evaluator_config": evaluator_config,
                "save_plots": save_plots,
                "plot_format": plot_format,
                "plot_config": plot_config,
                "save_results_json": save_results_json,
            }
            tasks.append(task_args)

    logger.info(
        f"Prepared {len(tasks)} evaluation tasks for {len(detector_names)} detectors across {len(audio_ref_pairs_path)} files."
    )

    # --- 並列評価の実行 --- #
    results: List[Dict[str, Any]] = []
    if not num_procs:
        num_procs = multiprocessing.cpu_count()
        logger.info(f"Using {num_procs} processes for evaluation.")

    try:
        if num_procs > 1 and len(tasks) > 1:
            logger.info(
                f"Running evaluations in parallel with {num_procs} processes..."
            )
            with multiprocessing.Pool(processes=num_procs) as pool:

                def worker_wrapper(task_dict):
                    return _evaluate_file_for_detector(**task_dict)

                with tqdm(total=len(tasks), desc="Evaluating Files") as pbar:
                    for result in pool.imap_unordered(worker_wrapper, tasks):
                        results.append(result)
                        pbar.update(1)
        else:
            logger.info("Running evaluations sequentially...")
            for task_args in tqdm(tasks, desc="Evaluating Files"):
                results.append(_evaluate_file_for_detector(**task_args))

    except KeyboardInterrupt:
        logger.warning("Evaluation process interrupted by user.")
        raise

    except Exception as e:
        logger.error(
            f"An error occurred during parallel evaluation: {e}", exc_info=True
        )
        raise RuntimeError(f"Parallel evaluation failed: {e}") from e

    logger.info(f"Finished evaluating all {len(tasks)} tasks.")

    # --- 結果の集計と保存 --- #
    if not results:
        logger.warning("No evaluation results were generated.")
        return {"summary": {}, "all_results": []}

    try:
        # ★ 修正: results リスト内のファイル数を計算 (エラーも含む可能性あり)
        num_evaluated = len(
            set(
                (res.get("audio_file"), res.get("detector_name"))
                for res in results
                if res.get("audio_file")
            )
        )
        # 有効な結果のみでサマリーを計算することも検討可能
        # valid_results = [res for res in results if res.get('status') != 'error' and res.get('metrics')]
        # summary = _calculate_evaluation_summary(valid_results)

        summary = _calculate_evaluation_summary(
            results
        )  # 現状維持: エラー結果も含めて集計
        summary["evaluation_config"] = evaluator_config
        # summary["num_files_evaluated"] = len(results) # これはタスク数になる
        summary["num_files_processed"] = len(
            audio_ref_pairs_path
        )  # 処理対象ファイル数 (フィルタリング後)
        summary["num_tasks_executed"] = len(tasks)  # 実行タスク数
        summary["detector_names"] = detector_names
        summary["dataset_name"] = dataset_name or "Custom files"
        summary["timestamp"] = datetime.datetime.now().isoformat()
        summary["total_duration_seconds"] = time.time() - start_time_overall
        if num_folds is not None:
            summary["cross_validation"] = {
                "num_folds": num_folds,
                "fold_index": fold_index,
            }

        # サマリーをJSONファイルに保存
        summary_json_path = output_dir_path / "summary.json"
        save_evaluation_result(summary, str(summary_json_path))
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
            logger.warning(
                "Pandas not installed. Cannot save summary table or print statistics."
            )
        except Exception as df_err:
            logger.error(
                f"Failed to create or save summary dataframe: {df_err}", exc_info=True
            )

        return {"summary": summary, "all_results": results}

    except Exception as e:
        logger.error(
            f"Error calculating or saving evaluation summary: {e}", exc_info=True
        )
        return {
            "summary": {"error": f"Summary calculation failed: {e}"},
            "all_results": results,
        }


def _calculate_evaluation_summary(
    results: List[Dict[str, Any]], logger: logging.Logger = logging.getLogger(__name__)
) -> Dict[str, Any]:
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
    summary: Dict[str, Any] = {"detectors": {}}
    all_metrics: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        detector_name = result.get("detector_name", "unknown")
        metrics = result.get("metrics")  # 各評価カテゴリ (note, onset, ...) の辞書

        if detector_name not in summary["detectors"]:
            summary["detectors"][detector_name] = {"count": 0, "metrics": {}}
            all_metrics[detector_name] = {}

        summary["detectors"][detector_name]["count"] += 1

        if metrics and isinstance(metrics, dict):
            for (
                category,
                values,
            ) in metrics.items():  # category: note, onset, offset, pitch, frame_pitch
                if isinstance(values, dict):
                    if category not in summary["detectors"][detector_name]["metrics"]:
                        summary["detectors"][detector_name]["metrics"][category] = {
                            "precision": [],
                            "recall": [],
                            "f_measure": [],
                            "overlap_ratio": [],
                            "accuracy": [],
                        }
                    if category not in all_metrics[detector_name]:
                        all_metrics[detector_name][category] = {
                            "precision": [],
                            "recall": [],
                            "f_measure": [],
                            "overlap_ratio": [],
                            "accuracy": [],
                        }

                    for metric_name, value in values.items():
                        if isinstance(value, (float, int)) and np.isfinite(value):
                            # サマリー用リストに追加 (後で平均などを計算)
                            if (
                                metric_name
                                in summary["detectors"][detector_name]["metrics"][
                                    category
                                ]
                            ):
                                summary["detectors"][detector_name]["metrics"][
                                    category
                                ][metric_name].append(value)
                            # 全体集計用リストに追加
                            if metric_name in all_metrics[detector_name][category]:
                                all_metrics[detector_name][category][
                                    metric_name
                                ].append(value)
                        elif value is not None:
                            logger.debug(
                                f"Skipping non-finite metric value: {detector_name}/{category}/{metric_name} = {value}"
                            )

    # --- 平均、標準偏差などを計算 --- #
    overall_summary_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for detector_name, detector_data in summary["detectors"].items():
        detector_summary = {}
        for category, metrics_lists in detector_data["metrics"].items():
            cat_summary = {}
            for metric_name, values_list in metrics_lists.items():
                if values_list:
                    # nan を除外して計算
                    valid_values = [v for v in values_list if np.isfinite(v)]
                    if valid_values:
                        mean_val = float(np.mean(valid_values))
                        std_val = float(np.std(valid_values))
                        median_val = float(np.median(valid_values))
                        cat_summary[f"{metric_name}_mean"] = mean_val
                        cat_summary[f"{metric_name}_std"] = std_val
                        cat_summary[f"{metric_name}_median"] = median_val
                        cat_summary[f"{metric_name}_count"] = len(valid_values)
                    else:
                        # 有効な値がない場合は NaN または 0 を設定
                        cat_summary[f"{metric_name}_mean"] = np.nan
                        cat_summary[f"{metric_name}_std"] = np.nan
                        cat_summary[f"{metric_name}_median"] = np.nan
                        cat_summary[f"{metric_name}_count"] = 0
                else:
                    # リストが空の場合もNaNを設定
                    cat_summary[f"{metric_name}_mean"] = np.nan
                    cat_summary[f"{metric_name}_std"] = np.nan
                    cat_summary[f"{metric_name}_median"] = np.nan
                    cat_summary[f"{metric_name}_count"] = 0
            if cat_summary:  # 何らかの指標が計算された場合のみ追加
                detector_summary[category] = cat_summary
        # 計算したサマリー統計を元の 'metrics' キーに格納 (上書き)
        detector_data["metrics"] = detector_summary

        # 全体サマリー用にも別途格納 (mean のみ)
        overall_detector_summary = {}
        for category, cat_summary_data in detector_summary.items():
            overall_cat_summary = {}
            for key, val in cat_summary_data.items():
                if key.endswith("_mean") and np.isfinite(val):
                    metric_name = key[:-5]  # '_mean' を除去
                    overall_cat_summary[metric_name] = val
            if overall_cat_summary:
                overall_detector_summary[category] = overall_cat_summary
        if overall_detector_summary:
            overall_summary_metrics[detector_name] = overall_detector_summary

    summary["overall_metrics"] = overall_summary_metrics

    # 全体 (全検出器) のサマリーも計算
    global_metrics: Dict[str, Dict[str, List[float]]] = {}
    total_count = 0
    for detector_name, detector_all_metrics in all_metrics.items():
        total_count += summary["detectors"][detector_name]["count"]
        for category, metrics_lists in detector_all_metrics.items():
            if category not in global_metrics:
                global_metrics[category] = {
                    "precision": [],
                    "recall": [],
                    "f_measure": [],
                    "overlap_ratio": [],
                    "accuracy": [],
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
                    cat_summary[f"{metric_name}_mean"] = float(np.mean(valid_values))
                    cat_summary[f"{metric_name}_std"] = float(np.std(valid_values))
                    cat_summary[f"{metric_name}_median"] = float(
                        np.median(valid_values)
                    )
                    cat_summary[f"{metric_name}_count"] = len(valid_values)
                else:
                    cat_summary[f"{metric_name}_mean"] = np.nan
                    cat_summary[f"{metric_name}_std"] = np.nan
                    cat_summary[f"{metric_name}_median"] = np.nan
                    cat_summary[f"{metric_name}_count"] = 0
            else:
                cat_summary[f"{metric_name}_mean"] = np.nan
                cat_summary[f"{metric_name}_std"] = np.nan
                cat_summary[f"{metric_name}_median"] = np.nan
                cat_summary[f"{metric_name}_count"] = 0
        if cat_summary:
            global_summary_stats[category] = cat_summary

    summary["global_summary"] = {
        "metrics": global_summary_stats,
        "total_evaluations": total_count,
    }

    return summary


def create_summary(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Calculate summary statistics from a list of evaluation results."""
    summary_data: Dict[str, Dict[str, List[float]]] = {}
    for result in results:
        detector = result.get("detector_name", "unknown")
        metrics = result.get("metrics", {})
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
                    final_summary[detector][category][f"{metric}_mean"] = float(
                        np.mean(values)
                    )
                    final_summary[detector][category][f"{metric}_std"] = float(
                        np.std(values)
                    )
                else:
                    final_summary[detector][category][f"{metric}_mean"] = 0.0
                    final_summary[detector][category][f"{metric}_std"] = 0.0
    return final_summary


def save_evaluation_result(result: Dict[str, Any], output_path: str) -> None:
    """Save evaluation result to a JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, cls=NumpyEncoder, indent=2)
        logger.debug(f"Evaluation result saved to {output_path}")
    except Exception as e:
        logger.error(
            f"Failed to save evaluation result to {output_path}: {e}", exc_info=True
        )


def evaluate_detector(
    detector_name: str,
    detector_params: Optional[Dict[str, Any]],
    audio_file: str,
    ref_file: str,
    output_dir: Path,
    eval_id: str,
    evaluator_config: Dict[str, Any],
    save_results_json: bool = True,
    save_plots: bool = False,
    plot_format: str = "png",
    plot_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,  # logger引数を追加
) -> Dict[str, Any]:
    """特定の検出器と設定で単一のオーディオファイルを評価します。

    Args:
        detector_name: 使用する検出器の名前。
        detector_params: 検出器に渡すパラメータ。
        audio_file: 評価対象のオーディオファイルのパス。
        ref_file: 参照データのパス。
        output_dir: 結果を保存するディレクトリ。
        eval_id: 評価の識別子 (通常はファイル名など)。
        evaluator_config: 評価関数の設定。
        save_results_json: 結果をJSONファイルに保存するかどうか。
        save_plots: 検出結果のプロットを保存するかどうか。
        plot_format: プロットの保存形式 ('png', 'svg'など)。
        plot_config: プロット関数に渡す追加設定。
        logger: 使用するロガーインスタンス。

    Returns:
        評価結果とメタデータを含む辞書。
    """
    if logger is None:
        logger = logging.getLogger(__name__)  # デフォルトロガー

    logger.info(
        f"[{eval_id}] Evaluating detector '{detector_name}' on {os.path.basename(audio_file)}"
    )
    start_time = time.time()

    result_base = {
        "eval_id": eval_id,
        "audio_file": os.path.basename(audio_file),
        "ref_file": os.path.basename(ref_file),
        "detector_name": detector_name,
        "detector_params": detector_params,
        "metrics": {},
        "detection_time": 0.0,
        "valid": False,
        "error_message": None,
        "processing_time": 0.0,
        "json_file": None,
        "plot_file": None,
    }

    try:
        # 1. 検出器クラスを取得してインスタンス化
        try:
            DetectorClass = get_detector_class(detector_name)
            detector = DetectorClass(**(detector_params or {}))
        except (ImportError, ConfigError, TypeError) as e:
            error_msg = (
                f"Failed to get or instantiate detector class ({detector_name}): {e}"
            )
            logger.error(
                f"Error during evaluation for {eval_id}: {error_msg}", exc_info=True
            )
            result_base.update(
                {
                    "error_message": error_msg,
                    "processing_time": time.time() - start_time,
                }
            )
            return result_base

        # 2. 音声ファイルと参照データをロード
        try:
            logger.debug(f"Loading audio file: {audio_file}")
            audio_data, sr = load_audio_file(audio_file)
            if audio_data is None:
                raise FileError(f"Failed to load audio data (is None): {audio_file}")
        except (FileError, Exception) as e:
            error_msg = f"Failed to load audio file: {e}"
            logger.error(
                f"Error during evaluation for {eval_id}: {error_msg}", exc_info=True
            )
            result_base.update(
                {
                    "error_message": error_msg,
                    "processing_time": time.time() - start_time,
                }
            )
            return result_base

        try:
            logger.debug(f"Loading reference data: {ref_file}")
            reference_data = load_reference_data(ref_file)
            if (
                reference_data is None
                or not reference_data.get("intervals", np.array([])).size > 0
            ):
                raise FileNotFoundError(
                    f"Reference data not found or empty: {ref_file}"
                )
        except (FileNotFoundError, Exception) as e:
            error_msg = f"Failed to load reference data: {e}"
            logger.error(
                f"Error during evaluation for {eval_id}: {error_msg}", exc_info=True
            )
            result_base.update(
                {
                    "error_message": error_msg,
                    "processing_time": time.time() - start_time,
                }
            )
            return result_base

        # 3. 検出の実行
        detection_start_time = time.time()
        try:
            detection_result_raw = detector.detect(audio_data, sr)
            if detection_result_raw is None:  # detectがNoneを返す場合
                raise EvaluationError("Detector returned None.")
        except Exception as e:
            detection_duration = time.time() - detection_start_time
            error_msg = f"Error during detector execution ({detector_name}): {e}"
            logger.error(
                f"Error during evaluation for {eval_id}: {error_msg}", exc_info=True
            )
            result_base.update(
                {
                    "error_message": error_msg,
                    "detection_time": detection_duration,
                    "processing_time": time.time() - start_time,
                }
            )
            return result_base
        detection_duration = time.time() - detection_start_time
        result_base["detection_time"] = detection_duration

        # 4. 検出結果の正規化
        try:
            normalized_detection_result: DetectionResult = normalize_detection_result(
                detection_result_raw, sr=sr, detector_name=detector_name
            )
            if not isinstance(normalized_detection_result, DetectionResult):
                raise TypeError(
                    f"Normalization result is not a DetectionResult: {type(normalized_detection_result)}"
                )
        except (TypeError, ValueError, Exception) as e:
            error_msg = f"Error during detection result normalization: {e}"
            logger.error(
                f"Error during evaluation for {eval_id}: {error_msg}", exc_info=True
            )
            result_base.update(
                {
                    "error_message": error_msg,
                    "processing_time": time.time() - start_time,
                }
            )
            return result_base

        # 5. 評価の実行
        try:
            evaluation_metrics = evaluate_detection_result(
                detected_intervals=normalized_detection_result.intervals,
                detected_pitches=normalized_detection_result.note_pitches,
                reference_intervals=reference_data["intervals"],
                reference_pitches=reference_data["pitches"],
                evaluator_config=evaluator_config,
                detector_result=normalized_detection_result.to_dict(),  # 参照データを含まない検出結果辞書を渡す
                logger=logger,
            )
        except Exception as e:
            error_msg = f"Error during metric calculation: {e}"
            logger.error(
                f"Error during evaluation for {eval_id}: {error_msg}", exc_info=True
            )
            result_base.update(
                {
                    "error_message": error_msg,
                    "processing_time": time.time() - start_time,
                }
            )
            return result_base

        result_base["metrics"] = evaluation_metrics
        result_base["valid"] = True

        # 6. 結果の保存 (JSON)
        if save_results_json:
            json_filename = f"{eval_id}_{detector_name}_result.json"
            json_path = output_dir / json_filename
            try:
                save_evaluation_result(result_base, json_path)
                result_base["json_file"] = str(
                    json_path.relative_to(output_dir.parent)
                )  # 相対パスで保存
            except Exception as e:
                logger.warning(
                    f"Failed to save evaluation result JSON to {json_path}: {e}"
                )

        # 7. プロットの保存
        # ★ 修正: プロット保存処理を追加
        if save_plots and plot_module_imported:
            plot_filename = f"{eval_id}_{detector_name}_detection_plot.{plot_format}"
            plot_path = output_dir / plot_filename
            try:
                # reference_data を DetectionResult に変換（plot関数が期待する場合）
                # もしくは、plot関数が辞書を受け付けるならそのままでOK
                # ここでは reference_data (辞書) をそのまま渡す
                ref_for_plot = {
                    "intervals": reference_data["intervals"],
                    "note_pitches": reference_data["pitches"],
                    # 必要に応じて他のキーも追加
                }

                save_detection_plot_io(
                    detection_result=normalized_detection_result,
                    reference=ref_for_plot,
                    output_path=str(plot_path),
                    plot_format=plot_format,
                    plot_config=plot_config,  # plot_config を渡す
                    audio_path=audio_file,  # 音声ファイルパスも渡す
                )
                result_base["plot_file"] = str(
                    plot_path.relative_to(output_dir.parent)
                )  # 相対パスで保存
                logger.debug(f"Detection plot saved to {plot_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to save detection plot to {plot_path}: {e}", exc_info=True
                )
        elif save_plots and not plot_module_imported:
            logger.warning(
                "Plotting libraries not installed. Skipping plot generation."
            )

    except Exception as e:
        # 予期せぬエラー
        error_msg = f"Unexpected error during evaluation: {e}"
        logger.error(
            f"Error during evaluation for {eval_id}: {error_msg}", exc_info=True
        )
        result_base["valid"] = False
        result_base["error_message"] = error_msg

    result_base["processing_time"] = time.time() - start_time
    logger.info(
        f"[{eval_id}] Finished evaluating '{detector_name}' in {result_base['processing_time']:.2f}s"
    )

    return result_base
