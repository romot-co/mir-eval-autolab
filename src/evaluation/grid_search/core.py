"""
グリッドサーチによるパラメータ最適化モジュール

このモジュールは、検出器のパラメータを最適化するためのグリッドサーチ機能を提供します。
複数のパラメータ組み合わせを試し、最も高いスコアを出す組み合わせを特定します。
"""

import os
import json
import logging
import numpy as np
import itertools
import time
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import datetime
import yaml
from tqdm import tqdm
import traceback

from src.utils.detector_utils import get_detector_class
from src.evaluation.evaluation_runner import run_evaluation
from src.evaluation.evaluation_io import save_evaluation_result, load_evaluation_result
from src.utils.logging_utils import get_logger
from src.utils.json_utils import NumpyEncoder


def create_grid_config(
    detector_name: str,
    audio_dir: str,
    reference_dir: str,
    output_path: str,
    param_grid: Dict[str, List[Any]],
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    グリッドサーチの設定ファイルを作成します。

    Parameters
    ----------
    detector_name : str
        検出器の名前
    audio_dir : str
        音声ファイルのディレクトリパス
    reference_dir : str
        参照ファイルのディレクトリパス
    output_path : str
        出力ファイルパス
    param_grid : Dict[str, List[Any]]
        パラメータグリッド（パラメータ名とその値のリストのマッピング）
    save_plots : bool, optional
        プロットを保存するかどうか, by default True

    Returns
    -------
    Dict[str, Any]
        グリッドサーチの設定辞書
    """
    config = {
        'detector': detector_name,
        'audio_dir': audio_dir,
        'reference_dir': reference_dir,
        'param_grid': param_grid,
        'save_plots': save_plots
    }
    
    # 設定をYAMLに保存
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    return config


def _save_results_to_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    グリッドサーチの結果をCSVファイルに保存します。
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        グリッドサーチの結果リスト
    output_path : str
        出力CSVファイルのパス
    """
    # 結果からデータを抽出
    data = []
    
    for result in results:
        # 基本的な指標
        row = {
            'run_id': result['run_id'],
            'execution_time': result['execution_time']
        }
        
        # 評価指標を抽出
        metrics = result.get('metrics', {})
        for category, category_metrics in metrics.items():
            # category_metricsが辞書の場合はキーと値のペアを処理
            if isinstance(category_metrics, dict):
                for metric_name, metric_value in category_metrics.items():
                    row[f'{category}.{metric_name}'] = metric_value
            # category_metricsが辞書でない場合（数値など）は直接値として扱う
            else:
                row[f'{category}'] = category_metrics
        
        # パラメータ値を抽出
        for param_name, param_value in result['params'].items():
            row[f'param_{param_name}'] = param_value
        
        data.append(row)
    
    # DataFrameを作成して保存
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def run_grid_search(
    config_path: str,
    grid_config_path: str,
    output_dir: str,
    best_metric: str = 'note.f_measure',
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    グリッドサーチを実行して最適なパラメータを見つけます。

    Parameters
    ----------
    config_path : str
        基本設定ファイルのパス
    grid_config_path : str
        グリッドサーチ設定ファイルのパス
    output_dir : str
        出力ディレクトリのパス
    best_metric : str, optional
        最適化する指標, by default 'note.f_measure'
        'category.metric'の形式で指定（例：'note.f_measure', 'onset.precision'など）
    logger : Optional[logging.Logger], optional
        ロガー, by default None

    Returns
    -------
    Dict[str, Any]
        最適なパラメータと評価結果
    """
    if logger is None:
        logger = get_logger('grid_search')
    
    # 設定ファイルを読み込む
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(grid_config_path, 'r') as f:
        grid_config = yaml.safe_load(f)
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 検出器の取得
    detector_name = grid_config['detector']
    detector_class = get_detector_class(detector_name)
    
    # 音声ファイルと参照ファイルのリストを取得
    try:
        audio_dir = config['audio_dir']
        reference_dir = config['reference_dir']
        # reference_pattern も必要に応じて config から読み込む（ここでは未使用だが念のため）
        # reference_pattern = config.get('reference_pattern', '*.csv') # デフォルト値付き
    except KeyError as e:
        logger.error(f"基本設定ファイル({config_path})に必要なキーが見つかりません: {e}")
        raise ValueError(f"基本設定ファイルに必要なキーがありません: {e}") from e
    
    # 音声ファイルと参照ファイルのリストを作成
    audio_files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                         if f.endswith(('.wav', '.mp3', '.flac'))])
    reference_files = []
    
    for audio_file in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        ref_file = None
        
        # 複数の拡張子をチェック
        for ext in ['.csv', '.json', '.txt']:
            possible_ref = os.path.join(reference_dir, f"{base_name}{ext}")
            if os.path.exists(possible_ref):
                ref_file = possible_ref
                break
        
        if ref_file:
            reference_files.append(ref_file)
        else:
            logger.warning(f"参照ファイルが見つかりません: {base_name}.*")
            audio_files.remove(audio_file)
    
    if len(audio_files) == 0:
        raise ValueError("評価対象の音声ファイルが見つかりません")
    
    # パラメータグリッドを作成
    param_grid = grid_config['param_grid']
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # すべてのパラメータ組み合わせを生成
    param_combinations = list(itertools.product(*param_values))
    
    logger.info(f"グリッドサーチを開始: {len(param_combinations)}個のパラメータ組み合わせを評価")
    
    # 評価設定を作成（通常評価と同じ設定を使用）
    evaluator_config = config.get('evaluation', {})
    if not evaluator_config:
        logger.warning("評価設定が見つかりません。デフォルト値を使用します。")
        evaluator_config = {
            'tolerance_onset': 0.05,
            'offset_ratio': 0.2,
            'offset_min_tolerance': 0.05,
            'tolerance_pitch': 50,
            'use_pitch_chroma': False
        }
    
    # 結果を保存するリスト
    results = []
    
    # 各パラメータ組み合わせで評価を実行
    for i, param_values in enumerate(param_combinations):
        params = dict(zip(param_names, param_values))
        run_id = f"run_{i+1}"
        
        logger.info(f"実行 {run_id}: パラメータ = {params}")
        
        # 評価を実行
        run_output_dir = os.path.join(output_dir, run_id)
        os.makedirs(run_output_dir, exist_ok=True)
        
        try:
            start_time = time.time()
            
            # 新しいrun_evaluation関数を使用
            evaluation_results = run_evaluation(
                audio_paths=audio_files,
                ref_paths=reference_files,
                detector_names=[detector_name],
                detector_params={detector_name: params},
                evaluator_config=evaluator_config,
                output_dir=run_output_dir,
                save_plots=grid_config.get('save_plots', True),
                save_results_json=True,
                plot_config={
                    'show': False,  # 非対話的な環境のため、表示はオフに
                    'save_path': os.path.join(run_output_dir, 'detection_plot.png')
                }
            )
            
            end_time = time.time()
            
            # 評価結果を取得（サマリー部分）
            if evaluation_results.get('status') in ['success', 'partial_success'] and 'summary' in evaluation_results:
                metrics = evaluation_results['summary'].get(detector_name, {})
                
                # 結果を保存
                result = {
                    'run_id': run_id,
                    'params': params,
                    'metrics': metrics,
                    'execution_time': end_time - start_time
                }
                
                results.append(result)
                
                # グリッドサーチ結果をJSONに保存
                grid_search_result_path = os.path.join(run_output_dir, 'grid_search_result.json')
                with open(grid_search_result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
                
            else:
                logger.error(f"実行 {run_id} の評価に失敗しました")
                
        except Exception as e:
            logger.error(f"実行 {run_id} でエラーが発生しました: {str(e)}")
            traceback.print_exc()
            continue
    
    if not results:
        logger.error("すべての評価が失敗しました")
        return None
    
    # 結果をCSVに保存
    csv_path = os.path.join(output_dir, 'grid_search_results.csv')
    _save_results_to_csv(results, csv_path)
    
    # 最適なパラメータを見つける
    category, metric = best_metric.split('.')
    
    def get_metric_value(result):
        return result.get('metrics', {}).get(category, {}).get(metric, float('-inf'))
    
    best_result = max(results, key=get_metric_value)
    
    # 最適なパラメータをJSONに保存
    best_params_path = os.path.join(output_dir, 'best_params.json')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_metric': best_metric,
            'best_value': get_metric_value(best_result),
            'params': best_result['params']
        }, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    
    # 全体のサマリーを作成
    combined_summary = {
        'grid_search_config': {
            'detector': detector_name,
            'best_metric': best_metric,
            'num_combinations': len(param_combinations),
            'num_completed': len(results)
        },
        'best_result': best_result,
        'all_results': results
    }
    
    # 全体サマリーを保存
    combined_summary_path = os.path.join(output_dir, 'combined_evaluation_summary.json')
    with open(combined_summary_path, 'w', encoding='utf-8') as f:
        json.dump(combined_summary, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    
    return best_result 