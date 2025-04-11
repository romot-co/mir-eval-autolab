"""
評価結果の入出力関連処理を集めたモジュール

このモジュールでは、評価結果の保存やロードに関連する機能を提供します。
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple, Optional

from src.utils.json_utils import NumpyEncoder
from src.utils.exception_utils import FileError, log_exception
from src.utils.detection_result import DetectionResult

logger = logging.getLogger(__name__) # モジュールレベルのロガーを取得

# プロット関数をインポート
try:
    # from src.visualization.plots import plot_detection_results # コメントアウト
    plot_module_imported = False # インポート試行をやめたので False に設定
except ImportError:
    plot_module_imported = False
    logger.warning("matplotlib または関連モジュールが見つかりません。プロット機能は無効になります。")

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

def load_evaluation_result(input_path: str) -> Dict[str, Any]:
    """
    評価結果をJSONファイルからロードします。

    Parameters
    ----------
    input_path : str
        入力ファイルのパス

    Returns
    -------
    Dict[str, Any]
        評価結果の辞書
    """
    # JSONからロード
    with open(input_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    return result

def load_multiple_evaluation_results(input_dir: str, pattern: str = "*_evaluation.json") -> List[Dict[str, Any]]:
    """
    指定されたディレクトリから複数の評価結果ファイルをロードします。

    Parameters
    ----------
    input_dir : str
        入力ディレクトリのパス
    pattern : str, optional
        検索するファイルパターン, by default "*_evaluation.json"

    Returns
    -------
    List[Dict[str, Any]]
        評価結果の辞書のリスト
    """
    results = []

    # ディレクトリ内のファイルを検索
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logger.error(f"指定された入力パスはディレクトリではありません: {input_dir}")
        # FileError を発生させるか、空リストを返すか？ -> エラーとして処理
        raise FileError(f"入力パスがディレクトリではありません: {input_dir}")

    files = list(input_path.glob(pattern))
    logger.info(f"{len(files)} 件の評価結果ファイルを {input_dir} からロードします (パターン: {pattern})。")

    for file_path in files:
        try:
            result = load_evaluation_result(str(file_path))
            results.append(result)
        except FileNotFoundError:
            # ファイルが見つからない場合は警告ログ (ループは継続)
            logger.warning(f"評価結果ファイルが見つかりません: {file_path}")
        except PermissionError:
            # パーミッションエラーも警告ログ (ループは継続)
            logger.warning(f"ファイルへのアクセス権がありません: {file_path}")
        except json.JSONDecodeError as json_err:
            # JSONデコードエラーは警告ログ (ループは継続)
            logger.warning(f"JSONファイルの解析に失敗しました: {file_path} - {json_err}")
        except Exception as e:
            # その他の予期せぬエラーはエラーログを出し、カスタム例外でラップすることも検討
            log_exception(logger, e, f"ファイル {file_path} のロード中に予期せぬエラーが発生しました", log_level=logging.ERROR)
            # 必要に応じて raise FileError(...) from e するか、ループを継続するか判断
            # ここでは処理を継続し、ロードできたものだけを返す方針とする

    logger.info(f"{len(results)} 件の評価結果を正常にロードしました。")
    return results

def print_summary_statistics(results_summary: pd.DataFrame, logger: logging.Logger = None, 
                             metrics_list: List[str] = None) -> None:
    """
    評価結果のサマリーを標準出力に表示します。

    Parameters
    ----------
    results_summary : pd.DataFrame
        評価結果のサマリー
    logger : logging.Logger, optional
        使用するロガー（指定しない場合はデフォルトロガーを使用）
    metrics_list : List[str], optional
        表示する指標のリスト
    """
    # ロガーが指定されていない場合はデフォルトのロガーを使用
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if metrics_list is None:
        metrics_list = [
            "onset_precision", "onset_recall", "onset_f_measure",
            "offset_precision", "offset_recall", "offset_f_measure",
            "pitch_precision", "pitch_recall", "pitch_f_measure",
            "note_precision", "note_recall", "note_f_measure",
            "frame_pitch_voicing_recall", "frame_pitch_voicing_false_alarm",
            "frame_pitch_raw_pitch_accuracy", "frame_pitch_raw_chroma_accuracy", "frame_pitch_overall_accuracy",
            "frame_pitch_precision", "frame_pitch_recall", "frame_pitch_f_measure", "frame_pitch_accuracy"
        ]
    
    # 使用可能な指標を取得
    available_metrics = results_summary.columns
    
    # 表示する指標を絞り込む
    display_metrics = [m for m in metrics_list if m in available_metrics]
    
    if not display_metrics:
        logger.warning("表示する指標がありません")
        return
    
    # 評価結果の表示
    logger.info("\n評価結果のサマリー:")
    logger.info(f"検出器: {results_summary['detector_name'].iloc[0]}")
    logger.info(f"評価ファイル数: {results_summary['files_count'].iloc[0]}")
    logger.info(f"平均検出時間: {results_summary['detection_time_mean'].iloc[0]:.4f}秒")
    
    # 指標の表示
    for metric in display_metrics:
        if metric in available_metrics:
            value = results_summary[metric].iloc[0]
            logger.info(f"{metric}: {value:.4f}")
    
    logger.info("")

def create_summary_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    評価結果のリストからサマリDataFrameを作成します。

    Parameters
    ----------
    results : List[Dict[str, Any]]
        評価結果の辞書のリスト

    Returns
    -------
    pd.DataFrame
        サマリのDataFrame
    """
    # 成功した結果のみフィルタリング
    success_results = [r for r in results if r.get('status') == 'success']
    
    if not success_results:
        return pd.DataFrame()
    
    # 各ファイルのデータを収集
    data = []
    
    for result in success_results:
        # 基本情報
        row = {
            'audio_path': result.get('audio_path', ''),
            'detector_name': result.get('detector_name', ''),
            'detection_time': result.get('detection_time', 0.0)
        }
        
        # 評価結果がある場合は展開
        if 'evaluation' in result:
            evaluation = result['evaluation']
            
            # 各カテゴリと指標を展開
            for category, metrics in evaluation.items():
                if isinstance(metrics, dict) and category not in ['ref_note_count', 'est_note_count']:
                    # フレームピッチ評価の場合、互換性のために追加の指標を設定
                    if category == 'frame_pitch':
                        # 新しい標準指標から従来の指標を派生させる
                        if 'raw_pitch_accuracy' in metrics:
                            # 旧指標を追加（表示用）
                            row[f'{category}_precision'] = metrics.get('raw_pitch_accuracy', 0.0)
                            row[f'{category}_recall'] = metrics.get('raw_pitch_accuracy', 0.0)
                            row[f'{category}_f_measure'] = metrics.get('raw_pitch_accuracy', 0.0)
                            row[f'{category}_accuracy'] = metrics.get('raw_pitch_accuracy', 0.0)
                    
                    for metric, value in metrics.items():
                        row[f'{category}_{metric}'] = value
                elif category in ['ref_note_count', 'est_note_count']:
                    row[category] = metrics
        
        data.append(row)
    
    # DataFrameに変換
    df = pd.DataFrame(data)
    
    return df

def save_detection_plot(
    audio_data: np.ndarray,
    sr: int,
    detection_result: DetectionResult,
    reference: DetectionResult,
    output_path: str,
    plot_format: str = 'png',
    plot_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    検出結果と参照データをプロットして保存します。
    音声波形も一緒に表示します。

    Parameters
    ----------
    audio_data : np.ndarray
        音声データ
    sr : int
        サンプリングレート
    detection_result : DetectionResult
        検出結果オブジェクト
    reference : DetectionResult
        参照データオブジェクト
    output_path : str
        出力ファイルのパス
    plot_format : str, optional
        プロットの形式, by default 'png'
    plot_config : Optional[Dict[str, Any]], optional
        プロット設定, by default None
    """
    if not plot_module_imported:
        logger.warning("プロットモジュールがインポートされていないため、プロットをスキップします。")
        return

    # 出力ディレクトリがない場合は作成
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        logger.debug(f"プロットを {output_path} に保存中...")
        # plot_detection_results を呼び出す (引数を辞書に変換)
        # plot_detection_results(
        #     audio_data=audio_data,
        #     sr=sr,
        #     detection_result=detection_result.to_dict() if detection_result else None,
        #     reference_data=reference.to_dict() if reference else None,
        #     show=False,
        #     save_path=output_path,
        # )
        logger.info(f"プロットを {output_path} に保存しました。")

    except Exception as e:
        log_exception(logger, e, f"プロットの保存中にエラーが発生しました: {output_path}", log_level=logging.ERROR)

def save_detection_and_evaluation_results(detection_result: Dict[str, Any],
                                  evaluation_result: Dict[str, Any],
                                  output_dir: str,
                                  base_name: str,
                                  detector_name: str) -> Dict[str, str]:
    """
    検出結果と評価結果をJSONファイルとして保存します。
    
    Parameters
    ----------
    detection_result : Dict[str, Any]
        検出結果の辞書
    evaluation_result : Dict[str, Any]
        評価結果の辞書
    output_dir : str
        出力ディレクトリのパス
    base_name : str
        ベース名（通常は音声ファイル名のベース）
    detector_name : str
        検出器の名前
        
    Returns
    -------
    Dict[str, str]
        保存されたファイルのパスを含む辞書
    """
    import os
    import json
    from src.utils.json_utils import NumpyEncoder
    
    # 出力ディレクトリの作成
    file_output_dir = os.path.join(output_dir, base_name, detector_name)
    os.makedirs(file_output_dir, exist_ok=True)
    
    # 検出結果の保存
    detection_path = os.path.join(file_output_dir, "detection_result.json")
    with open(detection_path, 'w', encoding='utf-8') as f:
        json.dump(detection_result, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    
    # 評価結果の保存
    evaluation_path = os.path.join(file_output_dir, "evaluation_result.json")
    with open(evaluation_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    
    return {
        'detection_result': detection_path,
        'evaluation_result': evaluation_path
    } 