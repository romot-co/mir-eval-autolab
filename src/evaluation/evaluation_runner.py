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
import yaml

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

def run_evaluation(
    detector_names: Union[str, List[str]],
    audio_paths: Optional[Union[str, List[str]]] = None,
    ref_paths: Optional[Union[str, List[str]]] = None,
    detector_params: Optional[Dict[str, Dict[str, Any]]] = None,
    evaluator_config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    save_plots: bool = False,
    plot_format: str = 'png',
    save_results_json: bool = True,
    plot_config: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    num_procs: Optional[int] = None
) -> Dict[str, Any]:
    """
    音声検出アルゴリズムの評価を実行します。

    ファイルパスリストまたはデータセット名を指定できます。

    Args:
        detector_names (Union[str, List[str]]): 評価する検出器の名前、または名前のリスト。
        audio_paths (Optional[Union[str, List[str]]]): 評価対象の音声ファイルパス、またはディレクトリパス。
                                                      省略した場合、dataset_name から取得します。
        ref_paths (Optional[Union[str, List[str]]]): 評価対象の参照ファイルパス、またはディレクトリパス。
                                                   省略した場合、dataset_name から取得します。
        detector_params (Optional[Dict[str, Dict[str, Any]]]): 検出器ごとのパラメータ設定。
        evaluator_config (Optional[Dict[str, Any]]): 評価設定 (mir_eval用)。
        output_dir (Optional[str]): 結果の出力先ディレクトリ。
        save_plots (bool): プロットを保存するかどうか。
        plot_format (str): プロットの保存形式 ('png', 'pdf' など)。
        save_results_json (bool): 評価結果をJSONファイルに保存するかどうか。
        plot_config (Optional[Dict[str, Any]]): プロット設定。
        dataset_name (Optional[str]): 使用するデータセット名 (config.yaml で定義)。
                                      audio_paths/ref_paths が省略された場合に必須。
        num_procs (Optional[int]): 並列処理に使用するプロセス数 (未実装)。

    Returns:
        Dict[str, Any]: 評価結果のサマリーを含む辞書。
    """
    start_time = time.time()
    logger.info(f"評価を開始します: 検出器={detector_names}, データセット={dataset_name or '指定なし'}")

    # --- パスリストの解決 ---
    final_audio_paths = []
    final_ref_paths = []

    if audio_paths and ref_paths:
        # 直接パスが指定された場合
        if isinstance(audio_paths, str):
            if Path(audio_paths).is_dir():
                # ディレクトリの場合はファイルを検索 (例: *.wav)
                final_audio_paths = sorted(list(Path(audio_paths).glob('*.wav')))
                # TODO: .flacなども考慮
            else:
                final_audio_paths = [Path(audio_paths)]
        elif isinstance(audio_paths, list):
            final_audio_paths = [Path(p) for p in audio_paths]
        else:
            raise ValueError("audio_pathsは文字列またはリストである必要があります")

        # ref_pathsも同様に処理
        if isinstance(ref_paths, str):
            if Path(ref_paths).is_dir():
                # ディレクトリの場合はファイルを検索 (拡張子は仮)
                final_ref_paths = sorted(list(Path(ref_paths).glob('*.csv')))
            else:
                final_ref_paths = [Path(ref_paths)]
        elif isinstance(ref_paths, list):
            final_ref_paths = [Path(p) for p in ref_paths]
        else:
            raise ValueError("ref_pathsは文字列またはリストである必要があります")
            
        # ファイル数のチェック
        if len(final_audio_paths) != len(final_ref_paths):
            logger.warning(f"音声ファイル数 ({len(final_audio_paths)}) と参照ファイル数 ({len(final_ref_paths)}) が一致しません。ファイル名のペアリングを試みます。")
            # ファイル名ベースでのペアリングロジックをここに追加 (例: stemが一致するもの)
            paired_paths = []
            ref_dict = {p.stem: p for p in final_ref_paths}
            temp_audio_paths = []
            temp_ref_paths = []
            for audio_p in final_audio_paths:
                if audio_p.stem in ref_dict:
                    temp_audio_paths.append(audio_p)
                    temp_ref_paths.append(ref_dict[audio_p.stem])
            if len(temp_audio_paths) == 0:
                 raise ValueError("音声ファイルと参照ファイルのペアが見つかりません。")
            final_audio_paths = temp_audio_paths
            final_ref_paths = temp_ref_paths
            logger.info(f"{len(final_audio_paths)} ペアのファイルが見つかりました。")
            
    elif dataset_name:
        # データセット名が指定された場合
        logger.info(f"データセット '{dataset_name}' のパスを取得します...")
        dataset_config = CONFIG.get('datasets', {}).get(dataset_name)
        if not dataset_config:
            raise ValueError(f"設定ファイルにデータセット '{dataset_name}' の定義が見つかりません。")

        audio_dir_path = Path(dataset_config.get('audio_dir'))
        label_dir_path = Path(dataset_config.get('label_dir'))
        label_pattern = dataset_config.get('label_pattern', '*.csv') # デフォルト *.csv
        audio_pattern = dataset_config.get('audio_pattern', '*.wav') # デフォルト *.wav

        if not audio_dir_path.is_dir():
            raise FileNotFoundError(f"データセットの音声ディレクトリが見つかりません: {audio_dir_path}")
        if not label_dir_path.is_dir():
            raise FileNotFoundError(f"データセットのラベルディレクトリが見つかりません: {label_dir_path}")

        # ファイルリストを取得してペアリング
        audio_files_found = sorted(list(audio_dir_path.glob(audio_pattern)))
        ref_files_dict = {p.stem: p for p in label_dir_path.glob(label_pattern)}

        if not audio_files_found:
            raise FileNotFoundError(f"音声ディレクトリに {audio_pattern} に一致するファイルが見つかりません: {audio_dir_path}")
        if not ref_files_dict:
             raise FileNotFoundError(f"ラベルディレクトリに {label_pattern} に一致するファイルが見つかりません: {label_dir_path}")

        for audio_p in audio_files_found:
            if audio_p.stem in ref_files_dict:
                final_audio_paths.append(audio_p)
                final_ref_paths.append(ref_files_dict[audio_p.stem])

        if not final_audio_paths:
            raise ValueError(f"データセット '{dataset_name}' で音声ファイルと参照ファイルのペアが見つかりませんでした。")
        logger.info(f"データセット '{dataset_name}' から {len(final_audio_paths)} ペアのファイルをロードしました。")

    else:
        raise ValueError("audio_paths/ref_paths または dataset_name のいずれかを指定する必要があります。")

    # --- 検出器リストの正規化 ---
    if isinstance(detector_names, str):
        detector_names = [detector_names]
    if detector_params is None:
        detector_params = {}

    # --- 出力ディレクトリの準備 ---
    if output_dir:
        output_base_dir = Path(output_dir)
    else:
        # デフォルトの出力先 (mcp_serverのワークスペース内を想定)
        try:
            from src.utils.path_utils import get_evaluation_results_dir
            output_base_dir = get_evaluation_results_dir()
        except ImportError:
            output_base_dir = Path("./evaluation_results")
            logger.warning("path_utils をインポートできません。出力先: ./evaluation_results")

    # 評価実行ID (一意なサブディレクトリ用)
    eval_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_per_detector: Dict[str, List[Dict[str, Any]]] = {name: [] for name in detector_names}

    # --- 評価ループ ---
    total_files = len(final_audio_paths)
    logger.info(f"合計 {total_files} ファイルの評価を開始します。検出器: {detector_names}")

    # tqdm を使用してプログレスバーを表示
    for i, (audio_file, ref_file) in enumerate(tqdm(zip(final_audio_paths, final_ref_paths), total=total_files, desc="Evaluating files")):
        logger.debug(f"Processing file {i+1}/{total_files}: {audio_file.name}")

        # 各検出器で処理
        for detector_name in detector_names:
            detector_specific_output_dir = output_base_dir / detector_name / eval_id
            detector_specific_output_dir.mkdir(parents=True, exist_ok=True)

            # evaluate_detector を呼び出す (引数を修正)
            file_result = evaluate_detector(
                detector_name=detector_name,
                detector_params=detector_params.get(detector_name),
                audio_file=str(audio_file),
                ref_file=str(ref_file),
                output_dir=str(detector_specific_output_dir), # 出力先を渡す
                evaluator_config=evaluator_config,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_config=plot_config,
                save_results_json=save_results_json # file ごとのjson保存も制御
            )
            file_result['audio_file'] = str(audio_file) # 結果にファイル名を追加
            file_result['ref_file'] = str(ref_file)
            results_per_detector[detector_name].append(file_result)

    # --- 結果集計 ---
    all_results = {
        "evaluation_config": evaluator_config or {},
        "detector_params": detector_params,
        "dataset_name": dataset_name or "Custom",
        "num_files": total_files,
        "detectors": {}
    }

    for detector_name, file_results in results_per_detector.items():
        summary = _calculate_evaluation_summary(file_results)
        all_results["detectors"][detector_name] = {
            "overall_metrics": summary,
            "file_metrics": {res["audio_file"]: res for res in file_results} # ファイル名をキーに
        }
        
        # 概要をログとコンソールに出力
        logger.info(f"--- 評価サマリー ({detector_name}) ---")
        print_summary_statistics(summary, detector_name)

    # 総合結果をJSONで保存
    if save_results_json:
        summary_output_dir = output_base_dir / "_summary" / eval_id
        summary_output_dir.mkdir(parents=True, exist_ok=True)
        summary_file_path = summary_output_dir / "evaluation_summary.json"
        try:
            with open(summary_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
            logger.info(f"評価サマリーを保存しました: {summary_file_path}")
        except Exception as e:
            logger.error(f"評価サマリーのJSON保存に失敗: {e}")

    elapsed_time = time.time() - start_time
    logger.info(f"評価が完了しました (所要時間: {elapsed_time:.2f}秒)")

    # MCPに返す結果（サマリー部分）
    # ここでは主要なメトリクスだけを返すなど、調整が必要かもしれない
    return all_results # とりあえず全結果を返す

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

def evaluate_detector(
    detector_name: str,
    detector_params: Optional[Dict[str, Any]],
    audio_file: str,
    ref_file: str,
    output_dir: str, # 具体的な出力ディレクトリを受け取る
    evaluator_config: Optional[Dict[str, Any]],
    save_plots: bool,
    plot_format: str,
    plot_config: Optional[Dict[str, Any]],
    save_results_json: bool
) -> Dict[str, Any]:
    """
    単一のファイルに対して単一の検出器を評価します。
    """
    output_path_base = Path(output_dir) / Path(audio_file).stem
    
    # ... (検出器取得、音声・参照データ読み込み)
    detection_time = 0.0 # 初期化
    try:
        # ... (検出実行)
        detector = get_detector_class(detector_name)
        detector_instance = detector(**(detector_params or {}))
        audio_data, sr = load_audio_file(audio_file)
        ref_intervals, ref_pitches = load_reference_data(ref_file)
        
        start_detect = time.time()
        detection_result_obj = detector_instance.detect(audio_data, sr)
        detection_time = time.time() - start_detect

        # 結果を正規化
        detected_intervals, detected_pitches = normalize_detection_result(detection_result_obj)
        
        # ... (評価実行: evaluate_detection_result)
        eval_metrics = evaluate_detection_result(
            detected_intervals, detected_pitches, 
            ref_intervals, ref_pitches, 
            evaluator_config,
            # detector_result=detection_result_obj # 必要に応じてフレームデータなどを渡す
        )

        # 結果辞書の作成
        result = {
            "audio_file": Path(audio_file).name,
            "ref_file": Path(ref_file).name,
            "detector_name": detector_name,
            "params": detector_params or {},
            "evaluation_metrics": eval_metrics,
            "detection_time": detection_time,
            "valid": True,
            "error": None
        }
        
        # 結果の保存 (JSON)
        if save_results_json:
            result_json_path = output_path_base.with_suffix(".evaluation.json")
            save_evaluation_result(result, str(result_json_path))

        # プロットの保存
        if save_plots and plot_module_imported:
             plot_output_path = output_path_base.with_suffix(f".{plot_format}")
             # save_detection_plot を呼び出す
             # TODO: save_detection_plot の引数を確認・調整
             try:
                save_detection_plot(
                    audio_data=audio_data,
                    sr=sr,
                    detection_result=detection_result_obj, # 元のDetectionResultオブジェクトを渡す
                    reference={'intervals': ref_intervals, 'pitches': ref_pitches},
                    output_path=str(plot_output_path),
                    plot_format=plot_format,
                    plot_config=plot_config
                )
             except Exception as plot_e:
                 logger.warning(f"プロットの保存に失敗しました ({plot_output_path}): {plot_e}")
        
        return result

    except Exception as e:
        logger.error(f"ファイル {Path(audio_file).name} の評価中にエラー: {e}")
        # エラー結果を作成
        error_result = create_error_result(str(e))
        error_result['audio_file'] = Path(audio_file).name
        error_result['ref_file'] = Path(ref_file).name
        error_result['detector_name'] = detector_name
        error_result['params'] = detector_params or {}
        error_result['detection_time'] = detection_time
        error_result['valid'] = False
        return error_result

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